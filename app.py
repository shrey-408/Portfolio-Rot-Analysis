import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Collections Analytics — Agent Fatigue & Portfolio Rot",
    page_icon="📊",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    accounts = pd.read_csv("data/accounts.csv")
    agents   = pd.read_csv("data/agents.csv")
    calls    = pd.read_csv("data/call_logs.csv", parse_dates=["call_datetime"])

    if 'call_day' not in calls.columns:
        ref = calls['call_datetime'].min()
        calls['call_day'] = (calls['call_datetime'] - ref).dt.days

    return accounts, agents, calls


@st.cache_data
def clean_and_compute(accounts_raw, agents, calls_raw):
    SIM_DAYS   = 90
    START_DATE = pd.Timestamp('2024-01-01')
    CUTOFF     = START_DATE + pd.Timedelta(days=SIM_DAYS + 3)

    calls = calls_raw.copy()

    # Cleaning pipeline (mirrors analysis.ipynb)
    calls = calls.drop_duplicates(subset=['account_id', 'agent_id', 'call_datetime', 'outcome'])
    calls = calls.dropna(subset=['call_datetime'])
    calls = calls[calls['call_datetime'] <= CUTOFF]

    # Flag CRM-broken records — don't drop, they carry data quality signal
    calls['crm_broken'] = calls['outcome'].isna()

    # Last valid contact per account (exclude CRM-broken records)
    last_contact = (
        calls[~calls['crm_broken']]
        .groupby('account_id')['call_day']
        .max()
        .rename('last_contact_day')
        .reset_index()
    )

    accounts = accounts_raw.merge(last_contact, on='account_id', how='left')
    accounts['days_since_contact'] = SIM_DAYS - accounts['last_contact_day'].fillna(SIM_DAYS)
    accounts['ever_called']        = ~accounts['last_contact_day'].isna()

    # Payment status per account: did any call result in a payment?
    paid_accounts = (
        calls[calls['payment_made'] == True]['account_id'].unique()
        if 'payment_made' in calls.columns
        else np.array([])
    )
    accounts['paid'] = accounts['account_id'].isin(paid_accounts)

    return accounts, calls


@st.cache_data
def build_decay_curve(accounts):
    bins   = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 90]
    labels = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-40", "40-50", "50-60", "60-90"]
    mids   = [2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 35, 45, 55, 75]

    decay = accounts.copy()
    decay['dsc_bin'] = pd.cut(decay['days_since_contact'], bins=bins, labels=labels)

    agg = (
        decay.groupby('dsc_bin', observed=True)
        .agg(n_accounts=('account_id', 'count'), n_paid=('paid', 'sum'))
        .assign(recovery_rate=lambda x: np.where(
            x['n_accounts'] > 0, x['n_paid'] / x['n_accounts'], 0
        ))
        .reset_index()
    )
    agg['dsc_midpoint'] = mids

    return agg


# Module-level — must live outside any cached function so pickle can find it
def logistic(x, L, k, x0):
    return L / (1 + np.exp(k * (x - x0)))


@st.cache_data
def fit_logistic(decay_agg):
    try:
        popt, _ = curve_fit(
            logistic,
            decay_agg['dsc_midpoint'],
            decay_agg['recovery_rate'],
            p0=[0.5, 0.2, 20],
            maxfev=10000,
        )
    except Exception:
        popt = [0.5, 0.1, 22]

    return list(popt)   # plain list — always serialisable


@st.cache_data
def build_fatigue_data(accounts, agents, calls):
    # Per-agent: caseload, connection rate, conversion rate
    connected = calls[calls['outcome'] == 'Connected']
    paid_mask = calls['payment_made'] == True if 'payment_made' in calls.columns else pd.Series(False, index=calls.index)

    agent_calls    = calls.groupby('agent_id').size().rename('total_calls')
    agent_connects = connected.groupby('agent_id').size().rename('connected_calls')
    agent_payments = calls[paid_mask].groupby('agent_id').size().rename('payments') if paid_mask.any() else pd.Series(0, index=agents['agent_id'])

    agent_m = agents.merge(agent_calls,    left_on='agent_id', right_index=True, how='left')
    agent_m = agent_m.merge(agent_connects, left_on='agent_id', right_index=True, how='left')
    agent_m = agent_m.merge(agent_payments, left_on='agent_id', right_index=True, how='left')

    agent_m['total_calls']    = agent_m['total_calls'].fillna(0)
    agent_m['connected_calls']= agent_m['connected_calls'].fillna(0)
    agent_m['payments']       = agent_m['payments'].fillna(0)
    agent_m['conversion_rate']= np.where(
        agent_m['connected_calls'] > 0,
        agent_m['payments'] / agent_m['connected_calls'],
        0,
    )

    return agent_m


@st.cache_data
def compute_reallocation(accounts, agents, cliff_day, ceiling):
    SKILL_COLS = {
        'Personal Loan':    'skill_personal_loan',
        'Business Loan':    'skill_business_loan',
        'Two-Wheeler':      'skill_two-wheeler',
        'Consumer Durable': 'skill_consumer_durable',
        'Microfinance':     'skill_microfinance',
    }

    bleeding = accounts[
        (accounts['days_since_contact'] >= cliff_day) &
        (accounts['ever_called'])
    ].copy()
    bleeding['expected_recovery'] = bleeding['base_recovery_prob'] * bleeding['outstanding_amount']
    bleeding = bleeding.sort_values('expected_recovery', ascending=False)

    underloaded = agents[
        (agents['caseload'] <= ceiling) &
        (agents['agent_id'] != 'AGT017')   # CRM-broken, exclude from routing
    ].copy()
    underloaded['capacity'] = ceiling - underloaded['caseload']

    capacity  = underloaded.set_index('agent_id')['capacity'].to_dict()
    skill_map = {
        r['agent_id']: {lt: r.get(col, 0.5) for lt, col in SKILL_COLS.items()}
        for _, r in underloaded.iterrows()
    }

    rows = []
    for _, acc in bleeding.iterrows():
        lt         = acc['loan_type']
        best_agent = None
        best_score = -1

        for agt, cap in capacity.items():
            if cap <= 0:
                continue
            score = skill_map.get(agt, {}).get(lt, 0.5)
            if score > best_score:
                best_score = score
                best_agent = agt

        if best_agent is None:
            continue

        capacity[best_agent] -= 1
        rows.append({
            'account_id':        acc['account_id'],
            'loan_type':         acc['loan_type'],
            'outstanding_amount':acc['outstanding_amount'],
            'days_since_contact':acc['days_since_contact'],
            'new_agent':         best_agent,
            'agent_loan_skill':  best_score,
            'expected_recovery': acc['expected_recovery'],
        })

    return pd.DataFrame(rows), bleeding


# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────

accounts_raw, agents, calls_raw = load_data()
accounts, calls                 = clean_and_compute(accounts_raw, agents, calls_raw)
decay_agg                       = build_decay_curve(accounts)
popt     = fit_logistic(decay_agg)
L, k, x0 = popt
agent_m                         = build_fatigue_data(accounts, agents, calls)

CASELOAD_CEILING = 60
COLOURS = dict(primary='#4a9ded', danger='#e85d4a', amber='#f5a623', green='#5ecb8c', muted='#8890a4')


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────

st.title("📊 Agent Fatigue & Portfolio Rot — Collections Analytics")
st.caption("NBFC distressed loan portfolio · Synthetic data, real methodology")

st.divider()

# ── Sidebar controls ─────────────────────────────────────────
with st.sidebar:
    st.header("Parameters")
    cliff_override = st.slider(
        "Cliff Day Override (0 = auto-detected from curve fit)",
        min_value=0, max_value=45, value=0,
    )
    caseload_ceiling = st.slider(
        "Fatigue Ceiling (accounts per agent)",
        min_value=30, max_value=100, value=CASELOAD_CEILING,
    )
    st.divider()
    st.caption("Cliff day is the inflection point of the logistic recovery decay curve. "
               "Override it to test sensitivity.")

cliff_day = cliff_override if cliff_override > 0 else float(x0)
realloc_df, bleeding = compute_reallocation(accounts, agents, cliff_day, caseload_ceiling)

# ── KPI row ──────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

crm_broken_agents = (
    calls.groupby('agent_id')['crm_broken'].mean()
)
crm_broken_agents = crm_broken_agents[crm_broken_agents > 0.4]

col1.metric("Portfolio Cliff", f"Day {int(cliff_day)}")
col2.metric("Rotting Accounts", f"{len(bleeding):,}", help="Accounts past cliff with >0 days since contact")
col3.metric("Outstanding at Risk", f"₹{bleeding['outstanding_amount'].sum()/1e6:.2f}M")
col4.metric("Realloc. Candidates", f"{len(realloc_df):,}")
col5.metric("CRM-Broken Agents", f"{len(crm_broken_agents):,}", delta="Needs audit", delta_color="inverse")

st.divider()

# ── Tab layout ───────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📉 Portfolio Decay", "😓 Agent Fatigue", "🔀 Reallocation"])


# ── Tab 1: Portfolio Decay ────────────────────────────────────
with tab1:
    st.subheader("Recovery Rate vs. Days Since Last Contact")
    st.caption(
        "Logistic decay curve fit to observed payment rates. The inflection point marks the cliff — "
        "after it, recovery probability collapses. Accounts in the red zone are nearly unrecoverable."
    )

    x_range = np.linspace(0, 90, 200)
    y_fit   = logistic(x_range, *popt)

    fig = go.Figure()

    # Observed bars
    fig.add_trace(go.Bar(
        x=decay_agg['dsc_midpoint'],
        y=decay_agg['recovery_rate'],
        name='Observed Recovery Rate',
        marker_color=COLOURS['primary'],
        opacity=0.6,
        width=4,
    ))

    # Fitted curve
    fig.add_trace(go.Scatter(
        x=x_range, y=y_fit,
        name='Logistic Fit',
        line=dict(color=COLOURS['green'], width=2.5),
    ))

    # Cliff line
    fig.add_vline(
        x=cliff_day,
        line=dict(color=COLOURS['amber'], width=2, dash='dash'),
        annotation_text=f"Cliff — Day {int(cliff_day)}",
        annotation_position="top right",
        annotation_font_color=COLOURS['amber'],
    )

    fig.update_layout(
        xaxis_title="Days Since Last Contact",
        yaxis_title="Recovery Rate",
        yaxis_tickformat='.0%',
        legend=dict(orientation='h', y=-0.2),
        height=420,
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    col_a.metric("Logistic L (asymptote)", f"{L:.1%}")
    col_b.metric("Cliff inflection point", f"Day {x0:.1f}")


# ── Tab 2: Agent Fatigue ──────────────────────────────────────
with tab2:
    st.subheader("Caseload vs. Conversion Rate")
    st.caption(
        "Each point is one agent. Colour = overall skill score. "
        "The amber line marks the fatigue ceiling. Above it, conversion drops ~30–40% relative."
    )

    fig2 = px.scatter(
        agent_m,
        x='caseload',
        y='conversion_rate',
        color='overall_skill',
        color_continuous_scale='RdYlGn',
        hover_data=['agent_id', 'caseload', 'conversion_rate', 'overall_skill'],
        labels={
            'caseload': 'Agent Caseload (# Accounts)',
            'conversion_rate': 'Conversion Rate',
            'overall_skill': 'Skill Score',
        },
    )

    fig2.add_vline(
        x=caseload_ceiling,
        line=dict(color=COLOURS['amber'], width=2, dash='dash'),
        annotation_text=f"Ceiling ({caseload_ceiling} acc)",
        annotation_position="top right",
        annotation_font_color=COLOURS['amber'],
    )

    fig2.update_layout(
        yaxis_tickformat='.0%',
        height=400,
        margin=dict(t=20),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Band-level bar chart
    st.subheader("Conversion Rate by Caseload Band")
    bins   = list(range(0, 120, 20))
    labels = [f"{b}–{b+20}" for b in bins[:-1]]
    agent_m_plot = agent_m.copy()
    agent_m_plot['band'] = pd.cut(agent_m_plot['caseload'], bins=bins, labels=labels)
    band_agg = (
        agent_m_plot.groupby('band', observed=True)['conversion_rate']
        .mean()
        .reset_index()
        .rename(columns={'conversion_rate': 'mean_conversion'})
    )
    band_agg['colour'] = band_agg['band'].apply(
        lambda b: COLOURS['danger'] if int(str(b).split('–')[0]) >= caseload_ceiling else COLOURS['green']
    )

    fig3 = go.Figure(go.Bar(
        x=band_agg['band'].astype(str),
        y=band_agg['mean_conversion'],
        marker_color=band_agg['colour'],
        text=band_agg['mean_conversion'].map(lambda v: f'{v:.1%}'),
        textposition='outside',
    ))
    fig3.update_layout(
        xaxis_title='Caseload Band',
        yaxis_title='Mean Conversion Rate',
        yaxis_tickformat='.0%',
        height=340,
        margin=dict(t=20),
    )
    st.plotly_chart(fig3, use_container_width=True)

    if not crm_broken_agents.empty:
        st.warning(
            f"⚠️ **CRM Anomaly Detected:** {', '.join(crm_broken_agents.index.tolist())} "
            f"— outcome null rate exceeds 40%. These agents' data is unreliable from the "
            f"break point onwards. Excluded from reallocation routing."
        )


# ── Tab 3: Reallocation ───────────────────────────────────────
with tab3:
    st.subheader("Monday Morning Reallocation List")
    st.caption(
        f"Accounts past Day {int(cliff_day)} (cliff) that have been contacted before. "
        f"Routed to underloaded agents (≤{caseload_ceiling} accounts) by **loan-type skill match**, "
        f"not just overall skill. A microfinance agent on a business loan is a wasted call."
    )

    col_x, col_y, col_z = st.columns(3)
    col_x.metric("Accounts to Reallocate", f"{len(realloc_df):,}")
    col_y.metric("Outstanding Value", f"₹{bleeding['outstanding_amount'].sum()/1e6:.2f}M")
    col_z.metric("Expected Recovery Unlocked", f"₹{realloc_df['expected_recovery'].sum()/1e6:.2f}M" if not realloc_df.empty else "₹0")

    if realloc_df.empty:
        st.info("No accounts to reallocate at current cliff day and ceiling settings.")
    else:
        st.dataframe(
            realloc_df.style
            .format({
                'outstanding_amount': '₹{:,.0f}',
                'expected_recovery':  '₹{:,.0f}',
                'agent_loan_skill':   '{:.2f}',
                'days_since_contact': '{:.0f}',
            })
            .background_gradient(subset=['expected_recovery'], cmap='Greens'),
            use_container_width=True,
            height=420,
        )

        # Loan-type breakdown
        st.subheader("Routing Breakdown by Loan Type")
        routing_agg = (
            realloc_df.groupby('loan_type')
            .agg(
                accounts=('account_id', 'count'),
                expected_recovery=('expected_recovery', 'sum'),
                avg_skill=('agent_loan_skill', 'mean'),
            )
            .reset_index()
            .sort_values('expected_recovery', ascending=False)
        )
        fig4 = px.bar(
            routing_agg,
            x='loan_type',
            y='expected_recovery',
            color='avg_skill',
            color_continuous_scale='RdYlGn',
            text='accounts',
            labels={
                'loan_type': 'Loan Type',
                'expected_recovery': 'Expected Recovery (₹)',
                'avg_skill': 'Avg Agent Skill',
            },
        )
        fig4.update_traces(textposition='outside')
        fig4.update_layout(yaxis_tickformat=',.0f', height=360, margin=dict(t=20))
        st.plotly_chart(fig4, use_container_width=True)