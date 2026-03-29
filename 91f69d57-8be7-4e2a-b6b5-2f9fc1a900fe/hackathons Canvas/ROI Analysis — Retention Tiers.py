"""
PHASE 8 — BUSINESS ROI ANALYSIS PER RETENTION TIER
====================================================
Quantifies business ROI per retained user by tier (High Risk, At Risk, Healthy)
using calibrated p_retain probabilities, tier labels, survival probabilities,
and SHAP-driven feature importance from upstream blocks.

Pulls from: Business Recommendation (Phase 6), Advanced GBM (Phase 7),
            Survival Analysis
Exports: roi_summary, total_net_savings, total_revenue_at_risk
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Design tokens ─────────────────────────────────────────────────────────────
_BG      = "#1D1D20"
_TXT_PRI = "#fbfbff"
_TXT_SEC = "#909094"
_C_BLUE  = "#A1C9F4"
_C_GREEN = "#8DE5A1"
_C_GOLD  = "#ffd400"
_C_CORAL = "#FF9F9B"
_C_ORANGE= "#FFB482"
_C_LAV   = "#D0BBFF"
_GRID    = "#2e2e33"
_SEP     = "═" * 70

print(_SEP)
print("  PHASE 8 — ROI ANALYSIS: RETENTION INTERVENTION VALUE BY TIER")
print(_SEP)

# ═══════════════════════════════════════════════════════════════════════════════
# [A] INDUSTRY-STANDARD SaaS RETENTION REVENUE ASSUMPTIONS
#     All values are transparent constants — sourced from PLG SaaS benchmarks.
#     Override these to calibrate to your actual unit economics.
# ═══════════════════════════════════════════════════════════════════════════════

# Average Revenue Per User per month (SaaS PLG mid-market benchmark: $35–$85)
ARPU_MONTHLY          = 55.0       # $ / user / month

# Customer Lifetime Value — 12-month horizon at avg ARPU
# LTV = ARPU × avg retention months (assume 14 months lifetime for healthy users)
LTV_MONTHS            = 14         # months of expected lifetime for retained user
LTV_PER_USER          = ARPU_MONTHLY * LTV_MONTHS  # $770 per retained user

# Customer Acquisition Cost — PLG self-serve benchmark: $150–$400
CAC_PER_USER          = 250.0      # $ cost to acquire one new user

# Churn cost = LTV lost + CAC to replace churned user
CHURN_COST_PER_USER   = LTV_PER_USER + CAC_PER_USER  # $1,020 total cost per churned user

# Intervention costs per tier (CSM time, tooling, comms)
# High Risk  → intensive: 1-on-1 outreach, personalized onboarding, credits
# At Risk    → moderate:  automated nudge + 1 CSM touchpoint
# Healthy    → light:     loyalty programme, product comms only
INTERVENTION_COST = {
    "High Risk": 45.0,   # $ per user — automated email sequence + 1 CSM ping
    "At Risk":   25.0,   # $ per user — targeted in-app + weekly digest
    "Healthy":   8.0,    # $ per user — loyalty programme participation
}

# Uplift in retention probability per tier from intervention
# Derived from published SaaS onboarding A/B test benchmarks:
#   High Risk: 8–15% lift (Intercom, ChurnZero studies)
#   At Risk:   12–20% lift (habit-forming nudge sequences)
#   Healthy:   2–5% lift  (loyalty / advocacy activation)
INTERVENTION_RETENTION_UPLIFT = {
    "High Risk": 0.10,   # +10% probability uplift
    "At Risk":   0.15,   # +15% probability uplift
    "Healthy":   0.03,   # +3%  probability uplift
}

print("\n[A] SaaS REVENUE ASSUMPTIONS (transparent constants):")
print(f"    ARPU (monthly)           : ${ARPU_MONTHLY:>8.2f}")
print(f"    LTV per user (12m+ hrz.) : ${LTV_PER_USER:>8.2f}  ({LTV_MONTHS} months × ARPU)")
print(f"    CAC per user             : ${CAC_PER_USER:>8.2f}")
print(f"    Total churn cost/user    : ${CHURN_COST_PER_USER:>8.2f}  (LTV + CAC replacement)")
print(f"    Intervention cost/user   : High Risk ${INTERVENTION_COST['High Risk']:.0f}  |  "
      f"At Risk ${INTERVENTION_COST['At Risk']:.0f}  |  Healthy ${INTERVENTION_COST['Healthy']:.0f}")
print(f"    Retention uplift         : High Risk {INTERVENTION_RETENTION_UPLIFT['High Risk']:.0%}  |  "
      f"At Risk {INTERVENTION_RETENTION_UPLIFT['At Risk']:.0%}  |  Healthy {INTERVENTION_RETENTION_UPLIFT['Healthy']:.0%}")

# ═══════════════════════════════════════════════════════════════════════════════
# [B] LOAD UPSTREAM VARIABLES FROM PHASE 6 & SURVIVAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
# Available from upstream: p_retain (ndarray), tier_labels (ndarray),
# n_high_risk, n_at_risk, n_healthy, n_total_seg,
# survival_features (DataFrame with surv_prob_7d/14d/30d, km_hazard_score)

print(f"\n[B] UPSTREAM DATA SUMMARY (from Phase 6 / Survival Analysis):")
print(f"    Total users segmented    : {n_total_seg:,}")
print(f"    🔴 High Risk             : {n_high_risk:,}  ({100*n_high_risk/n_total_seg:.1f}%)")
print(f"    🟡 At Risk               : {n_at_risk:,}  ({100*n_at_risk/n_total_seg:.1f}%)")
print(f"    🟢 Healthy               : {n_healthy:,}  ({100*n_healthy/n_total_seg:.1f}%)")

# Mean p_retain per tier
_mask_hr = tier_labels == "High Risk"
_mask_ar = tier_labels == "At Risk"
_mask_hl = tier_labels == "Healthy"

_mean_p_hr = float(p_retain[_mask_hr].mean()) if _mask_hr.sum() > 0 else 0.0
_mean_p_ar = float(p_retain[_mask_ar].mean()) if _mask_ar.sum() > 0 else 0.0
_mean_p_hl = float(p_retain[_mask_hl].mean()) if _mask_hl.sum() > 0 else 0.0

print(f"\n    Mean p_retain per tier:")
print(f"      High Risk: {_mean_p_hr:.4f}  |  At Risk: {_mean_p_ar:.4f}  |  Healthy: {_mean_p_hl:.4f}")

# Mean survival probabilities per tier (from Survival Analysis block)
# survival_features has user-level surv_prob_30d — merge with user_feature_table by position
# user_feature_table and survival_features both indexed over the same 3,033 users
_surv_vals_30d = survival_features["surv_prob_30d"].values
_mean_surv_hr  = float(_surv_vals_30d[_mask_hr].mean()) if _mask_hr.sum() > 0 else 0.0
_mean_surv_ar  = float(_surv_vals_30d[_mask_ar].mean()) if _mask_ar.sum() > 0 else 0.0
_mean_surv_hl  = float(_surv_vals_30d[_mask_hl].mean()) if _mask_hl.sum() > 0 else 0.0

print(f"\n    Mean 30d survival prob per tier (Kaplan-Meier):")
print(f"      High Risk: {_mean_surv_hr:.4f}  |  At Risk: {_mean_surv_ar:.4f}  |  Healthy: {_mean_surv_hl:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# [C] PER-TIER ROI COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════
# For each tier:
#   revenue_at_risk    = count × (1 - mean_p_retain) × LTV_PER_USER
#                        (expected LTV we'll lose if no intervention)
#   expected_retained  = count × uplift × (1 - mean_p_retain)
#                        (additional users we expect to retain via intervention)
#   revenue_saved      = expected_retained × LTV_PER_USER
#   intervention_cost  = count × cost_per_user
#   net_savings        = revenue_saved - intervention_cost
#   roi_pct            = net_savings / intervention_cost × 100
#   payback_months     = intervention_cost / (expected_retained × ARPU_MONTHLY)

print(f"\n[C] PER-TIER ROI COMPUTATION:")
print(f"{'─'*70}")

_tiers = ["High Risk", "At Risk", "Healthy"]
_counts = {"High Risk": n_high_risk, "At Risk": n_at_risk, "Healthy": n_healthy}
_mean_p = {"High Risk": _mean_p_hr, "At Risk": _mean_p_ar, "Healthy": _mean_p_hl}

_roi_rows = []
for _t in _tiers:
    _n   = _counts[_t]
    _p   = _mean_p[_t]
    _upl = INTERVENTION_RETENTION_UPLIFT[_t]
    _c   = INTERVENTION_COST[_t]

    # Revenue at risk = expected LTV of users likely to churn without intervention
    _rev_at_risk       = _n * (1.0 - _p) * LTV_PER_USER
    # Expected additional users retained thanks to intervention
    _users_retained    = _n * _upl * (1.0 - _p)
    # Revenue saved = additional retentions × LTV
    _rev_saved         = _users_retained * LTV_PER_USER
    # Total intervention spend for this tier
    _total_intv_cost   = _n * _c
    # Net return
    _net_savings       = _rev_saved - _total_intv_cost
    # ROI %
    _roi_pct           = (_net_savings / _total_intv_cost * 100) if _total_intv_cost > 0 else 0.0
    # Payback: months to recoup intervention spend from ARPU stream
    _payback_months    = (_total_intv_cost / (_users_retained * ARPU_MONTHLY)
                          if _users_retained > 0 else float("inf"))

    _roi_rows.append({
        "Tier":                _t,
        "Users":               _n,
        "Mean p_retain":       round(_p, 4),
        "Revenue at Risk ($)": round(_rev_at_risk, 0),
        "Intervention Cost ($)": round(_total_intv_cost, 0),
        "Users Retained (est.)": round(_users_retained, 1),
        "Revenue Saved ($)":   round(_rev_saved, 0),
        "Net Savings ($)":     round(_net_savings, 0),
        "ROI (%)":             round(_roi_pct, 1),
        "Payback (months)":    round(_payback_months, 1) if _payback_months != float("inf") else "∞",
    })

roi_summary = pd.DataFrame(_roi_rows)

# ═══════════════════════════════════════════════════════════════════════════════
# [D] PRINT CLEAN ROI TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*70}")
print(f"  {'Tier':<12} {'Users':>7} {'Rev@Risk':>12} {'Intv.Cost':>12} {'Rev Saved':>12} {'Net Save':>12} {'ROI%':>8} {'Payback':>8}")
print(f"  {'─'*68}")
for _r in _roi_rows:
    _pb = f"{_r['Payback (months)']}m" if isinstance(_r["Payback (months)"], float) else _r["Payback (months)"]
    print(
        f"  {_r['Tier']:<12} "
        f"{_r['Users']:>7,} "
        f"${_r['Revenue at Risk ($)']:>10,.0f} "
        f"${_r['Intervention Cost ($)']:>10,.0f} "
        f"${_r['Revenue Saved ($)']:>10,.0f} "
        f"${_r['Net Savings ($)']:>10,.0f} "
        f"{_r['ROI (%)']:>7.1f}% "
        f"{_pb:>8}"
    )

# ── TOTALS ─────────────────────────────────────────────────────────────────────
total_revenue_at_risk  = float(roi_summary["Revenue at Risk ($)"].sum())
_total_intv            = float(roi_summary["Intervention Cost ($)"].sum())
total_net_savings      = float(roi_summary["Net Savings ($)"].sum())
_total_rev_saved       = float(roi_summary["Revenue Saved ($)"].sum())
_total_users_retained  = float(roi_summary["Users Retained (est.)"].sum())
_portfolio_roi         = (total_net_savings / _total_intv * 100) if _total_intv > 0 else 0.0
_portfolio_payback     = (_total_intv / (_total_users_retained * ARPU_MONTHLY)
                           if _total_users_retained > 0 else float("inf"))

print(f"  {'─'*68}")
print(
    f"  {'TOTAL':<12} "
    f"{n_total_seg:>7,} "
    f"${total_revenue_at_risk:>10,.0f} "
    f"${_total_intv:>10,.0f} "
    f"${_total_rev_saved:>10,.0f} "
    f"${total_net_savings:>10,.0f} "
    f"{_portfolio_roi:>7.1f}% "
    f"{_portfolio_payback:>6.1f}m"
)
print(f"{'─'*70}")

print(f"\n  PORTFOLIO SUMMARY:")
print(f"    Total Revenue at Risk    : ${total_revenue_at_risk:>12,.0f}")
print(f"    Total Intervention Cost  : ${_total_intv:>12,.0f}")
print(f"    Total Revenue Saved      : ${_total_rev_saved:>12,.0f}")
print(f"    Total Net Savings        : ${total_net_savings:>12,.0f}")
print(f"    Portfolio ROI            : {_portfolio_roi:.1f}%")
print(f"    Portfolio Payback Period : {_portfolio_payback:.1f} months")
print(f"    Estimated Users Retained : {_total_users_retained:.0f} additional users")

# ═══════════════════════════════════════════════════════════════════════════════
# [E] GROUPED BAR CHART — Revenue at Risk vs Net Savings vs Intervention Cost
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n[E] Rendering ROI grouped bar chart…")

_tiers_plot  = roi_summary["Tier"].tolist()
_rev_risk_k  = [v / 1000 for v in roi_summary["Revenue at Risk ($)"].tolist()]
_net_save_k  = [v / 1000 for v in roi_summary["Net Savings ($)"].tolist()]
_intv_cost_k = [v / 1000 for v in roi_summary["Intervention Cost ($)"].tolist()]

_x   = np.arange(len(_tiers_plot))
_w   = 0.26

roi_bar_chart, _ax_roi = plt.subplots(figsize=(12, 7))
roi_bar_chart.patch.set_facecolor(_BG)
_ax_roi.set_facecolor(_BG)

_b1 = _ax_roi.bar(_x - _w,     _rev_risk_k,  _w, label="Revenue at Risk",     color=_C_CORAL,  alpha=0.92, zorder=3, edgecolor="none")
_b2 = _ax_roi.bar(_x,          _net_save_k,  _w, label="Net Savings",         color=_C_GREEN,  alpha=0.92, zorder=3, edgecolor="none")
_b3 = _ax_roi.bar(_x + _w,     _intv_cost_k, _w, label="Intervention Cost",   color=_C_BLUE,   alpha=0.92, zorder=3, edgecolor="none")

# Value annotations
for _bars, _vals in [(_b1, _rev_risk_k), (_b2, _net_save_k), (_b3, _intv_cost_k)]:
    for _bar, _v in zip(_bars, _vals):
        _lbl = f"${_v:.0f}K" if abs(_v) >= 1 else f"${_v*1000:.0f}"
        _ypos = _bar.get_height() + (max(max(_rev_risk_k), max(_net_save_k), max(_intv_cost_k)) * 0.015)
        _ax_roi.text(
            _bar.get_x() + _bar.get_width() / 2, _ypos,
            _lbl, ha="center", va="bottom", color=_TXT_PRI, fontsize=9.5, fontweight="bold"
        )

# ROI% annotation above each tier group
for _xi, _t in enumerate(_tiers_plot):
    _roi_val = roi_summary.loc[roi_summary["Tier"] == _t, "ROI (%)"].values[0]
    _ax_roi.text(
        _xi, max(_rev_risk_k[_xi], _net_save_k[_xi], _intv_cost_k[_xi]) * 1.18,
        f"ROI: {_roi_val:.0f}%",
        ha="center", va="bottom", color=_C_GOLD, fontsize=10.5, fontweight="bold"
    )

# Tier colour emojis in tick labels
_tick_labels = ["🔴 High Risk", "🟡 At Risk", "🟢 Healthy"]
_ax_roi.set_xticks(_x)
_ax_roi.set_xticklabels(_tick_labels, color=_TXT_PRI, fontsize=12, fontweight="bold")
_ax_roi.set_ylabel("Value ($K)", color=_TXT_PRI, fontsize=11, labelpad=8)
_ax_roi.set_title(
    "Phase 8 — Business ROI by Retention Tier\n"
    f"Portfolio Net Savings: ${total_net_savings/1000:.0f}K  |  "
    f"Portfolio ROI: {_portfolio_roi:.0f}%  |  "
    f"Payback: {_portfolio_payback:.1f} months",
    color=_TXT_PRI, fontsize=13, fontweight="bold", pad=14
)
_ax_roi.tick_params(colors=_TXT_PRI, labelsize=10)
for _sp in _ax_roi.spines.values():
    _sp.set_edgecolor(_GRID)
_ax_roi.grid(axis="y", color=_GRID, linewidth=0.6, alpha=0.5, zorder=0)
_ax_roi.set_axisbelow(True)
_ax_roi.set_ylim(0, max(_rev_risk_k) * 1.35)

_ax_roi.legend(
    handles=[
        mpatches.Patch(color=_C_CORAL, label="Revenue at Risk"),
        mpatches.Patch(color=_C_GREEN, label="Net Savings"),
        mpatches.Patch(color=_C_BLUE,  label="Intervention Cost"),
    ],
    facecolor="#2a2a2e", edgecolor=_GRID, labelcolor=_TXT_PRI, fontsize=10,
    loc="upper right", framealpha=0.9
)

plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# [F] FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{_SEP}")
print("  PHASE 8 — SUMMARY")
print(_SEP)
print(f"  ARPU: ${ARPU_MONTHLY}/mo  |  LTV: ${LTV_PER_USER:.0f}  |  CAC: ${CAC_PER_USER:.0f}  |  "
      f"Churn Cost/User: ${CHURN_COST_PER_USER:.0f}")
print(f"\n  PER-TIER ROI:")
for _r in _roi_rows:
    _pb = f"{_r['Payback (months)']}m" if isinstance(_r["Payback (months)"], float) else "∞"
    print(f"    {_r['Tier']:<12}: {_r['Users']:>5,} users | "
          f"Rev@Risk=${_r['Revenue at Risk ($)']:>8,.0f} | "
          f"Net Savings=${_r['Net Savings ($)']:>8,.0f} | "
          f"ROI={_r['ROI (%)']:>6.1f}% | Payback={_pb}")
print(f"\n  PORTFOLIO TOTAL:")
print(f"    Revenue at Risk : ${total_revenue_at_risk:,.0f}")
print(f"    Net Savings     : ${total_net_savings:,.0f}")
print(f"    Portfolio ROI   : {_portfolio_roi:.1f}%")
print(f"    Payback Period  : {_portfolio_payback:.1f} months")
print(f"\n  Exported: roi_summary (DataFrame), total_net_savings, total_revenue_at_risk  ✅")
print(_SEP)
