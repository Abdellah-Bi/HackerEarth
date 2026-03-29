"""
PHASE 9 — ROI BREAKDOWN BY SIGNUP COHORT
=========================================
Bucketing users into monthly signup cohorts using 'created_at' (account timestamp).
For each cohort: tier distribution, revenue at risk, intervention cost, net savings,
ROI%, and mean survival probability.

Upstream: ROI Analysis — Retention Tiers (user_feature_table, p_retain, tier_labels,
          survival_features, ARPU_MONTHLY, LTV_PER_USER, CAC_PER_USER,
          INTERVENTION_COST, INTERVENTION_RETENTION_UPLIFT)
          User Retention Analysis (user_retention → created_at column)
Exports:  cohort_roi_summary (DataFrame)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Design tokens ─────────────────────────────────────────────────────────────
_BG       = "#1D1D20"
_TXT_PRI  = "#fbfbff"
_TXT_SEC  = "#909094"
_C_BLUE   = "#A1C9F4"
_C_ORANGE = "#FFB482"
_C_GREEN  = "#8DE5A1"
_C_CORAL  = "#FF9F9B"
_C_GOLD   = "#ffd400"
_C_LAV    = "#D0BBFF"
_GRID     = "#2e2e33"
_SEP      = "═" * 70

print(_SEP)
print("  PHASE 9 — COHORT ROI ANALYSIS BY SIGNUP DATE")
print(_SEP)

# ═══════════════════════════════════════════════════════════════════════════════
# [A] ASSEMBLE USER-LEVEL TABLE WITH COHORT, TIER & MODEL SCORES
# ═══════════════════════════════════════════════════════════════════════════════
# user_feature_table has distinct_id (index-aligned with p_retain / tier_labels)
# survival_features has user_id (same users, same order — confirmed upstream)
# user_retention has per-event rows with distinct_id + created_at (account ts)

# 1. Build per-user cohort from user_retention: earliest created_at per user
_user_cohort = (
    user_retention[["distinct_id", "created_at"]]
    .dropna(subset=["created_at"])
    .groupby("distinct_id")["created_at"]
    .min()
    .reset_index()
    .rename(columns={"created_at": "acct_created_at"})
)
_user_cohort["signup_month"] = _user_cohort["acct_created_at"].dt.to_period("M").astype(str)

print(f"\n[A] Cohort construction:")
print(f"    Users with account date  : {len(_user_cohort):,}")
print(f"    Signup month range       : {_user_cohort['signup_month'].min()} → {_user_cohort['signup_month'].max()}")
print(f"    Unique signup months     : {_user_cohort['signup_month'].nunique()}")

# 2. Attach tier_labels, p_retain, surv_prob_30d to user_feature_table rows
_user_df = user_feature_table.copy().reset_index(drop=True)
_user_df["tier"]         = tier_labels          # positional alignment confirmed upstream
_user_df["p_retain"]     = p_retain
_user_df["surv_prob_30d"] = survival_features["surv_prob_30d"].values

# 3. Merge with cohort
_user_df = _user_df.merge(_user_cohort, on="distinct_id", how="left")

_n_missing_cohort = _user_df["signup_month"].isna().sum()
print(f"    Users missing cohort date: {_n_missing_cohort} ({100*_n_missing_cohort/len(_user_df):.1f}%)")

# Drop users with no cohort date (can't assign to cohort)
_user_df = _user_df.dropna(subset=["signup_month"]).reset_index(drop=True)
print(f"    Users in cohort analysis : {len(_user_df):,}")

# ═══════════════════════════════════════════════════════════════════════════════
# [B] PER-COHORT × TIER ROI COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n[B] Computing per-cohort × tier ROI metrics…")

_tiers_ordered = ["High Risk", "At Risk", "Healthy"]

_cohort_rows = []
for _cohort, _cg in _user_df.groupby("signup_month"):
    _cohort_total = len(_cg)
    for _tier in _tiers_ordered:
        _tg = _cg[_cg["tier"] == _tier]
        _n  = len(_tg)
        if _n == 0:
            continue

        _mean_p   = float(_tg["p_retain"].mean())
        _mean_surv = float(_tg["surv_prob_30d"].mean())
        _upl      = INTERVENTION_RETENTION_UPLIFT[_tier]
        _c_per    = INTERVENTION_COST[_tier]

        _rev_at_risk   = _n * (1.0 - _mean_p) * LTV_PER_USER
        _users_retained = _n * _upl * (1.0 - _mean_p)
        _rev_saved     = _users_retained * LTV_PER_USER
        _total_cost    = _n * _c_per
        _net_savings   = _rev_saved - _total_cost
        _roi_pct       = (_net_savings / _total_cost * 100) if _total_cost > 0 else 0.0
        _payback       = (_total_cost / (_users_retained * ARPU_MONTHLY)
                          if _users_retained > 0 else float("inf"))

        _cohort_rows.append({
            "Cohort":              _cohort,
            "Tier":                _tier,
            "Users":               _n,
            "Cohort Total":        _cohort_total,
            "Tier %":              round(100.0 * _n / _cohort_total, 1),
            "Mean p_retain":       round(_mean_p, 4),
            "Mean Surv 30d":       round(_mean_surv, 4),
            "Revenue at Risk ($)": round(_rev_at_risk, 0),
            "Intervention Cost ($)": round(_total_cost, 0),
            "Net Savings ($)":     round(_net_savings, 0),
            "ROI (%)":             round(_roi_pct, 1),
            "Payback (mo)":        round(_payback, 1) if _payback != float("inf") else None,
        })

cohort_roi_summary = pd.DataFrame(_cohort_rows)
_cohorts = sorted(cohort_roi_summary["Cohort"].unique())
print(f"    Cohorts found            : {len(_cohorts)}  ({', '.join(_cohorts)})")
print(f"    Rows in summary table    : {len(cohort_roi_summary)}")

# ═══════════════════════════════════════════════════════════════════════════════
# [C] PRINT COHORT × TIER ROI TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n[C] COHORT × TIER ROI TABLE")
print(f"{'─'*100}")
_hdr = (f"  {'Cohort':<10} {'Tier':<12} {'N':>6} {'Tier%':>7} "
        f"{'p_retain':>9} {'Surv30d':>8} {'Rev@Risk':>11} "
        f"{'IntvCost':>10} {'NetSave':>10} {'ROI%':>7} {'Payback':>8}")
print(_hdr)
print(f"  {'─'*98}")

_prev_cohort = None
for _, _r in cohort_roi_summary.iterrows():
    _cohort_lbl = _r["Cohort"] if _r["Cohort"] != _prev_cohort else " " * 7
    _prev_cohort = _r["Cohort"]
    _pb = f"{_r['Payback (mo)']}m" if _r["Payback (mo)"] is not None else "∞"
    _tier_icon = {"High Risk": "🔴", "At Risk": "🟡", "Healthy": "🟢"}.get(_r["Tier"], "")
    print(
        f"  {_cohort_lbl:<10} {_tier_icon}{_r['Tier']:<11} "
        f"{int(_r['Users']):>6,} "
        f"{_r['Tier %']:>6.1f}% "
        f"{_r['Mean p_retain']:>9.4f} "
        f"{_r['Mean Surv 30d']:>8.4f} "
        f"${_r['Revenue at Risk ($)']:>9,.0f} "
        f"${_r['Intervention Cost ($)']:>8,.0f} "
        f"${_r['Net Savings ($)']:>8,.0f} "
        f"{_r['ROI (%)']:>6.1f}% "
        f"{_pb:>8}"
    )

print(f"  {'─'*98}")
# Portfolio row
_ptotal = cohort_roi_summary.groupby(level=None).agg(
    Users=("Users", "sum"),
    rev_risk=("Revenue at Risk ($)", "sum"),
    intv=("Intervention Cost ($)", "sum"),
    net=("Net Savings ($)", "sum"),
).iloc[0] if False else None

_total_users_c = cohort_roi_summary["Users"].sum()
_total_rev_c   = cohort_roi_summary["Revenue at Risk ($)"].sum()
_total_intv_c  = cohort_roi_summary["Intervention Cost ($)"].sum()
_total_net_c   = cohort_roi_summary["Net Savings ($)"].sum()
_portfolio_roi_c = (_total_net_c / _total_intv_c * 100) if _total_intv_c > 0 else 0.0
print(
    f"  {'TOTAL':<10} {'':<12} "
    f"{_total_users_c:>6,} "
    f"{'100.0':>6}% "
    f"{'—':>9} "
    f"{'—':>8} "
    f"${_total_rev_c:>9,.0f} "
    f"${_total_intv_c:>8,.0f} "
    f"${_total_net_c:>8,.0f} "
    f"{_portfolio_roi_c:>6.1f}% "
    f"{'—':>8}"
)
print(f"{'─'*100}")

# ═══════════════════════════════════════════════════════════════════════════════
# [D] CHART 1 — STACKED BAR: REVENUE AT RISK BY COHORT × TIER
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n[D] Rendering stacked bar chart: Revenue at Risk by Cohort × Tier…")

# Pivot revenue at risk
_rev_pivot = cohort_roi_summary.pivot_table(
    index="Cohort", columns="Tier", values="Revenue at Risk ($)", aggfunc="sum"
).fillna(0).reindex(columns=_tiers_ordered, fill_value=0)

_tier_colors = {"High Risk": _C_CORAL, "At Risk": _C_ORANGE, "Healthy": _C_GREEN}

cohort_rev_risk_chart = plt.figure(figsize=(12, 7))
cohort_rev_risk_chart.patch.set_facecolor(_BG)
_ax1 = cohort_rev_risk_chart.add_subplot(111)
_ax1.set_facecolor(_BG)

_x_pos = np.arange(len(_rev_pivot))
_bottoms = np.zeros(len(_rev_pivot))

for _tier in _tiers_ordered:
    _vals_k = _rev_pivot[_tier].values / 1000
    _ax1.bar(_x_pos, _vals_k, bottom=_bottoms / 1000, label=_tier,
             color=_tier_colors[_tier], alpha=0.9, edgecolor="none", width=0.65, zorder=3)
    _bottoms += _rev_pivot[_tier].values

# Total labels on top of each bar
_totals_k = _bottoms / 1000
for _xi, _tot in enumerate(_totals_k):
    _ax1.text(_xi, _tot + max(_totals_k) * 0.015, f"${_tot:.0f}K",
              ha="center", va="bottom", color=_TXT_PRI, fontsize=9, fontweight="bold")

_ax1.set_xticks(_x_pos)
_ax1.set_xticklabels(_rev_pivot.index, rotation=35, ha="right", color=_TXT_PRI, fontsize=10)
_ax1.set_ylabel("Revenue at Risk ($K)", color=_TXT_PRI, fontsize=11, labelpad=8)
_ax1.set_title(
    "Revenue at Risk by Signup Cohort × Tier\n(stacked by retention tier)",
    color=_TXT_PRI, fontsize=13, fontweight="bold", pad=14
)
_ax1.tick_params(colors=_TXT_PRI, labelsize=9)
for _sp in _ax1.spines.values():
    _sp.set_edgecolor(_GRID)
_ax1.yaxis.grid(True, color=_GRID, linewidth=0.6, alpha=0.5, zorder=0)
_ax1.set_axisbelow(True)
_ax1.set_ylim(0, max(_totals_k) * 1.25)

_ax1.legend(
    handles=[mpatches.Patch(color=_tier_colors[t], label=t) for t in _tiers_ordered],
    facecolor="#2a2a2e", edgecolor=_GRID, labelcolor=_TXT_PRI, fontsize=10,
    loc="upper right", framealpha=0.9
)

plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# [E] CHART 2 — LINE CHART: ROI% TREND BY COHORT (per tier + total)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n[E] Rendering ROI% trend line chart by cohort age…")

# Aggregate ROI% per cohort (weighted by intervention cost)
_cohort_agg = []
for _coh, _cg in cohort_roi_summary.groupby("Cohort"):
    _net   = _cg["Net Savings ($)"].sum()
    _intv  = _cg["Intervention Cost ($)"].sum()
    _roi   = (_net / _intv * 100) if _intv > 0 else 0.0
    _cohort_agg.append({"Cohort": _coh, "ROI_all": _roi})
    for _t in _tiers_ordered:
        _trow = _cg[_cg["Tier"] == _t]
        _t_net  = _trow["Net Savings ($)"].sum()
        _t_intv = _trow["Intervention Cost ($)"].sum()
        _cohort_agg[-1][f"ROI_{_t}"] = (_t_net / _t_intv * 100) if _t_intv > 0 else None

_roi_df = pd.DataFrame(_cohort_agg).sort_values("Cohort").reset_index(drop=True)

cohort_roi_trend_chart = plt.figure(figsize=(12, 7))
cohort_roi_trend_chart.patch.set_facecolor(_BG)
_ax2 = cohort_roi_trend_chart.add_subplot(111)
_ax2.set_facecolor(_BG)

_x_cohort = np.arange(len(_roi_df))
_line_styles = {
    "ROI_all":       (_C_GOLD,   "─── Portfolio", "--", 2.5),
    "ROI_High Risk": (_C_CORAL,  "🔴 High Risk",  "-",  2.0),
    "ROI_At Risk":   (_C_ORANGE, "🟡 At Risk",    "-",  2.0),
    "ROI_Healthy":   (_C_GREEN,  "🟢 Healthy",    "-",  2.0),
}

for _col, (_color, _label, _ls, _lw) in _line_styles.items():
    _vals = _roi_df[_col].tolist()
    _valid = [(i, v) for i, v in enumerate(_vals) if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if len(_valid) < 2:
        continue
    _xi_v, _yi_v = zip(*_valid)
    _ax2.plot(_xi_v, _yi_v, color=_color, linewidth=_lw,
              linestyle=_ls, marker="o", markersize=7,
              markerfacecolor=_color, markeredgecolor=_BG, label=_label, zorder=4)
    # Annotate last point
    _ax2.text(_xi_v[-1] + 0.05, _yi_v[-1], f"{_yi_v[-1]:.0f}%",
              va="center", ha="left", color=_color, fontsize=8.5, fontweight="bold")

_ax2.axhline(0, color=_GRID, linewidth=1, linestyle="--", alpha=0.7)

_ax2.set_xticks(_x_cohort)
_ax2.set_xticklabels(_roi_df["Cohort"], rotation=35, ha="right", color=_TXT_PRI, fontsize=10)
_ax2.set_ylabel("ROI (%)", color=_TXT_PRI, fontsize=11, labelpad=8)
_ax2.set_title(
    "ROI% Trend Across Signup Cohorts\n(Portfolio + per tier — shows cohort aging patterns)",
    color=_TXT_PRI, fontsize=13, fontweight="bold", pad=14
)
_ax2.tick_params(colors=_TXT_PRI, labelsize=9)
for _sp in _ax2.spines.values():
    _sp.set_edgecolor(_GRID)
_ax2.yaxis.grid(True, color=_GRID, linewidth=0.6, alpha=0.5, zorder=0)
_ax2.set_axisbelow(True)

_ax2.legend(
    facecolor="#2a2a2e", edgecolor=_GRID, labelcolor=_TXT_PRI, fontsize=10,
    loc="upper right", framealpha=0.9
)

plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# [F] SUMMARY PRINTOUT
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{_SEP}")
print("  PHASE 9 — COHORT SUMMARY")
print(_SEP)

# Best / worst cohort by ROI
_best_cohort  = _roi_df.loc[_roi_df["ROI_all"].idxmax(), "Cohort"]
_worst_cohort = _roi_df.loc[_roi_df["ROI_all"].idxmin(), "Cohort"]
_best_roi  = _roi_df["ROI_all"].max()
_worst_roi = _roi_df["ROI_all"].min()

print(f"  Cohorts analysed          : {len(_cohorts)}")
print(f"  Portfolio Rev at Risk     : ${_total_rev_c:>12,.0f}")
print(f"  Portfolio Net Savings     : ${_total_net_c:>12,.0f}")
print(f"  Portfolio ROI             : {_portfolio_roi_c:.1f}%")
print(f"  Best ROI cohort           : {_best_cohort}  ({_best_roi:.1f}%)")
print(f"  Worst ROI cohort          : {_worst_cohort}  ({_worst_roi:.1f}%)")
print(f"\n  Exported: cohort_roi_summary (DataFrame, {len(cohort_roi_summary)} rows)  ✅")
print(_SEP)
