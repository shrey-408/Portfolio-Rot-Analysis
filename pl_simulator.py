import numpy as np
import pandas as pd
from rule_engine import (
    build_daily_call_list,
    OFFER_ECONOMICS,
    DEFAULT_CLIFF_DAY,
    DEFAULT_FATIGUE_THRESHOLD,
)

# ── Economics constants ─────────────────────────────────────
COST_PER_CALL  = 12.0
CONNECT_RATE   = 0.40
WRITE_OFF_RATE = 0.95

STRATEGIES = ['current', 'dpd_only', 'ev_priority']

STRATEGY_LABELS = {
    'current':     'Current (no sort)',
    'dpd_only':    'DPD-Only Sort',
    'ev_priority': 'Rule Engine (EV × Urgency)',
}

# Used in notebook charts — defined here so a single import covers both
STRATEGY_COLOURS = {
    'current':     '#8890a4',   # neutral grey
    'dpd_only':    '#4a9ded',   # blue
    'ev_priority': '#5ecb8c',   # green
}


# ────────────────────────────────────────────────────────────
# PAYMENT SIMULATION
# ────────────────────────────────────────────────────────────

def _simulate_payment(account_row, offer_type, agent_skill, days_since_contact):
    # Fatigue drag: the longer an account has been untouched, the harder it is
    # to collect — up to a floor of 0.3 to avoid zeroing out recoverable accounts.
    fatigue_drag = max(0.3, 1 - 0.01 * days_since_contact)

    p_pay = account_row['base_recovery_prob'] * agent_skill * fatigue_drag

    if np.random.random() > p_pay:
        return False, 0.0, 0.0

    haircut = OFFER_ECONOMICS[offer_type]['haircut']
    gross   = account_row['outstanding_amount'] * np.random.uniform(0.05, 0.80)
    net     = gross * (1 - haircut)

    return True, round(gross, 2), round(net, 2)


# ────────────────────────────────────────────────────────────
# SORTING
# ────────────────────────────────────────────────────────────

def _sort_call_list(call_list, strategy):
    if strategy == 'current':
        return call_list.sort_values('account_id')

    elif strategy == 'dpd_only':
        dpd_order = {'180+': 0, '91-180': 1, '61-90': 2, '31-60': 3, '0-30': 4}
        call_list = call_list.copy()
        call_list['_dpd_rank'] = call_list['dpd_bucket'].map(dpd_order).fillna(5)
        return call_list.sort_values('_dpd_rank').drop(columns=['_dpd_rank'])

    elif strategy == 'ev_priority':
        return call_list.sort_values('priority_score', ascending=False)

    raise ValueError(f"Unknown strategy: {strategy}")


# ────────────────────────────────────────────────────────────
# MAIN SIMULATION
# ────────────────────────────────────────────────────────────

def simulate_strategy(
    accounts_df,
    calls_df,
    agents_df,
    strategy,
    cliff_day=DEFAULT_CLIFF_DAY,
    fatigue_threshold=DEFAULT_FATIGUE_THRESHOLD,
    call_budget=200,
    sim_days=90,
    random_seed=42,
):
    np.random.seed(random_seed)

    total_revenue   = 0.0
    total_gross     = 0.0
    total_haircut   = 0.0
    total_call_cost = 0.0
    total_calls     = 0
    total_payments  = 0

    contacted_accs = set()
    paid_accs      = set()

    daily_records = []

    # Build account lookup once — include all columns we need downstream
    account_lookup = accounts_df.set_index('account_id').to_dict('index')

    for day in range(sim_days):

        call_list = build_daily_call_list(
            accounts_df, calls_df, agents_df,
            today_day=day,
            cliff_day=cliff_day,
            fatigue_threshold=fatigue_threshold,
            call_budget=call_budget * 3,   # fetch wider, trim after sort
        )

        if call_list.empty:
            daily_records.append({'day': day, 'revenue': 0, 'cost': 0, 'payments': 0})
            continue

        call_list = call_list[~call_list['account_id'].isin(paid_accs)]
        call_list = _sort_call_list(call_list, strategy).head(call_budget)

        day_revenue = 0.0
        day_cost    = 0.0
        day_pmts    = 0

        for _, acc in call_list.iterrows():

            total_call_cost += COST_PER_CALL
            total_calls     += 1
            day_cost        += COST_PER_CALL

            acc_id = acc['account_id']
            contacted_accs.add(acc_id)

            # Connect probability
            if np.random.random() > CONNECT_RATE:
                continue

            account_row = account_lookup[acc_id]

            # days_since_contact comes from the call list (computed in rule_engine)
            dsc = float(acc.get('days_since_contact', 0))

            paid, gross, net = _simulate_payment(
                account_row,
                acc['offer_type'],
                acc['agent_skill'],
                dsc,
            )

            if paid:
                total_revenue   += net
                total_gross     += gross
                total_haircut   += (gross - net)
                total_payments  += 1
                day_revenue     += net
                day_pmts        += 1
                paid_accs.add(acc_id)

        daily_records.append({
            'day':      day,
            'revenue':  day_revenue,
            'cost':     day_cost,
            'payments': day_pmts,
        })

    # ── Write-offs: outstanding on accounts never contacted
    never_contacted = accounts_df[
        ~accounts_df['account_id'].isin(contacted_accs) &
        accounts_df['assigned_agent_id'].notna()
    ]
    write_off_value = never_contacted['outstanding_amount'].sum() * WRITE_OFF_RATE

    net_pl = total_revenue - total_call_cost - write_off_value

    return {
        'strategy':                 strategy,
        'label':                    STRATEGY_LABELS[strategy],
        'gross_revenue':            round(total_gross, 2),
        'net_revenue':              round(total_revenue, 2),
        'haircut_cost':             round(total_haircut, 2),
        'call_cost':                round(total_call_cost, 2),
        'write_offs':               round(write_off_value, 2),
        'net_pl':                   round(net_pl, 2),
        'total_calls':              total_calls,
        'total_payments':           total_payments,
        'accounts_paid':            len(paid_accs),
        'accounts_never_contacted': len(never_contacted),
        'daily_ledger':             pd.DataFrame(daily_records),
    }


# ────────────────────────────────────────────────────────────
# RUN ALL STRATEGIES
# ────────────────────────────────────────────────────────────

def run_all_strategies(
    accounts_df,
    calls_df,
    agents_df,
    cliff_day=DEFAULT_CLIFF_DAY,
    fatigue_threshold=DEFAULT_FATIGUE_THRESHOLD,
    call_budget=200,
    sim_days=90,
):
    results = []

    for strategy in STRATEGIES:
        print(f"Simulating: {STRATEGY_LABELS[strategy]}")
        r = simulate_strategy(
            accounts_df, calls_df, agents_df,
            strategy, cliff_day, fatigue_threshold, call_budget, sim_days,
        )
        results.append(r)

    summary_cols = [
        'strategy', 'label', 'gross_revenue', 'net_revenue',
        'haircut_cost', 'call_cost', 'write_offs', 'net_pl',
        'total_calls', 'total_payments', 'accounts_paid',
        'accounts_never_contacted',
    ]

    df = pd.DataFrame([{k: r[k] for k in summary_cols} for r in results])
    df = df.set_index('strategy')

    baseline              = df.loc['current', 'net_pl']
    df['lift_vs_current'] = df['net_pl'] - baseline
    df['lift_pct']        = (df['lift_vs_current'] / abs(baseline) * 100).round(1)

    # Attach daily ledgers as a dict so notebook charts can access them by strategy key.
    # Accessed as: pl_results._daily_ledgers['ev_priority']
    df._daily_ledgers = {r['strategy']: r['daily_ledger'] for r in results}

    return df
