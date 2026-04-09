import numpy as np
import pandas as pd

DEFAULT_CLIFF_DAY = 22
DEFAULT_FATIGUE_THRESHOLD = 60

DPD_OFFER_MAP = {
    '0-30': 'EMI_RESTRUCTURE',
    '31-60': 'EMI_RESTRUCTURE',
    '61-90': 'SETTLEMENT_50',
    '91-180': 'SETTLEMENT_30',
    '180+': 'SETTLEMENT_15',
}

OFFER_ECONOMICS = {
    'EMI_RESTRUCTURE': {'haircut': 0.00},
    'SETTLEMENT_50': {'haircut': 0.50},
    'SETTLEMENT_30': {'haircut': 0.70},
    'SETTLEMENT_15': {'haircut': 0.85},
}


# ─────────────────────────────────────────────────────────────
# URGENCY LOGIC
# ─────────────────────────────────────────────────────────────

def _urgency_multiplier(days_since_contact, cliff_day):
    if days_since_contact < 5:
        return 0.5
    elif days_since_contact < 15:
        return 1.0
    elif days_since_contact < cliff_day - 3:
        return 1.8
    else:
        return 3.0


def score_account(row, days_since_contact, cliff_day=DEFAULT_CLIFF_DAY):
    ev = row['base_recovery_prob'] * row['outstanding_amount']
    urgency = _urgency_multiplier(days_since_contact, cliff_day)
    return round(ev * urgency, 4)


# ─────────────────────────────────────────────────────────────
# OFFER LOGIC
# ─────────────────────────────────────────────────────────────

def decide_offer(row):
    return DPD_OFFER_MAP.get(row['dpd_bucket'], 'EMI_RESTRUCTURE')


# ─────────────────────────────────────────────────────────────
# AGENT SKILL
# ─────────────────────────────────────────────────────────────

def _agent_skill_for_loan(agent_row, loan_type):
    col_map = {
        'Personal Loan':    'skill_personal_loan',
        'Business Loan':    'skill_business_loan',
        'Two-Wheeler':      'skill_two-wheeler',
        'Consumer Durable': 'skill_consumer_durable',
        'Microfinance':     'skill_microfinance',
    }
    col = col_map.get(loan_type)
    if col and col in agent_row.index:
        return float(agent_row[col])
    return 0.5


# ─────────────────────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────────────────────

def build_daily_call_list(accounts_df, calls_df, agents_df,
                          today_day,
                          cliff_day=DEFAULT_CLIFF_DAY,
                          fatigue_threshold=DEFAULT_FATIGUE_THRESHOLD,
                          call_budget=200):

    # ── Ensure crm_broken column exists.
    # data_generator.py doesn't produce this column — it just leaves outcome as NaN.
    # The cleaning pipeline in analysis.ipynb creates it. We create it defensively
    # here so rule_engine works standalone without assuming upstream cleaning.
    if 'crm_broken' not in calls_df.columns:
        calls_df = calls_df.copy()
        calls_df['crm_broken'] = calls_df['outcome'].isna()

    # ── Last valid contact (exclude CRM-broken records — their outcomes are unreliable)
    last_call = (
        calls_df[
            (calls_df['call_day'] < today_day) &
            (~calls_df['crm_broken'])
        ]
        .groupby('account_id')['call_day']
        .max()
        .rename('last_call_day')
    )

    df = accounts_df.merge(last_call, on='account_id', how='left')
    df['days_since_contact'] = today_day - df['last_call_day'].fillna(today_day)

    # ── Remove unassigned accounts
    df = df[df['assigned_agent_id'].notna()].copy()

    # ── Remove already-paid accounts
    already_paid = calls_df[calls_df['payment_made'] == True]['account_id'].unique()
    df = df[~df['account_id'].isin(already_paid)].copy()

    # ── CRM anomaly detection: exclude agents whose outcomes are >40% null
    # (systematic pipeline failures, not random noise).
    # These agents' portfolios need a manual audit before being re-queued.
    crm_null_rate = (
        calls_df
        .groupby('agent_id')['crm_broken']
        .mean()
    )
    crm_issue_agents = crm_null_rate[crm_null_rate > 0.4].index
    df = df[~df['assigned_agent_id'].isin(crm_issue_agents)].copy()

    # ── Fatigue flag: accounts assigned to overloaded agents are marked
    # but NOT silently dropped. The calling strategy decides what to do with them —
    # either skip for today, or include at lower priority. Here we tag and let
    # the caller decide. The pl_simulator and app both respect the flag.
    agent_load = agents_df.set_index('agent_id')['caseload'].to_dict()
    df['agent_overloaded'] = df['assigned_agent_id'].map(
        lambda a: agent_load.get(a, 0) > fatigue_threshold
    )

    if df.empty:
        return pd.DataFrame()

    # ── Priority score
    df['priority_score'] = df.apply(
        lambda r: score_account(r, r['days_since_contact'], cliff_day),
        axis=1
    )

    # ── Offer assignment
    df['offer_type'] = df.apply(decide_offer, axis=1)

    # ── Agent skill mapping
    agents_idx = agents_df.set_index('agent_id')
    df['agent_skill'] = df.apply(
        lambda r: _agent_skill_for_loan(
            agents_idx.loc[r['assigned_agent_id']],
            r['loan_type']
        ),
        axis=1
    )

    # ── Expected net recovery (for display / reporting)
    df['expected_net_recovery'] = df.apply(
        lambda r: round(
            r['base_recovery_prob']
            * r['outstanding_amount']
            * (1 - OFFER_ECONOMICS[r['offer_type']]['haircut']),
            2
        ),
        axis=1
    )

    # ── Sort by priority; non-overloaded agents go first within same score tier
    df = df.sort_values(
        ['agent_overloaded', 'priority_score'],
        ascending=[True, False]
    ).head(call_budget)

    return df.reset_index(drop=True)
