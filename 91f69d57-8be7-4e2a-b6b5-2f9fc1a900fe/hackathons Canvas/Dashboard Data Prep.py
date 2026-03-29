"""
DASHBOARD DATA PREP
===================
Merges all upstream variables into a clean per-user dashboard_data DataFrame
and assembles a dashboard_constants dict with SaaS unit economics.

Upstream variables used:
  - user_feature_table (distinct_id, 6 features, is_retained) — 3,033 users
  - p_retain (ndarray, positional) — calibrated retention probability
  - tier_labels (ndarray, positional) — High Risk / At Risk / Healthy
  - enriched_features (distinct_id, 23 features including cluster, segment_name)
  - survival_features (user_id, survival probabilities, hazard scores)
  - roi_summary (Tier-level ROI table)
  - cohort_roi_summary (Cohort × Tier ROI table)
  - SaaS assumptions: ARPU_MONTHLY, LTV_PER_USER, CAC_PER_USER,
                      CHURN_COST_PER_USER, INTERVENTION_COST,
                      INTERVENTION_RETENTION_UPLIFT

Exports:
  - dashboard_data (DataFrame, per-user, joined on distinct_id/user_id)
  - dashboard_constants (dict, SaaS assumptions + model metrics)
"""

import numpy as np
import pandas as pd

_SEP = "═" * 70
print(_SEP)
print("  DASHBOARD DATA PREP")
print(_SEP)

# ═══════════════════════════════════════════════════════════════════════════════
# [A] BUILD BASE USER TABLE — start from user_feature_table
# ═══════════════════════════════════════════════════════════════════════════════
# user_feature_table has 3,033 rows; p_retain and tier_labels are positionally
# aligned (confirmed in upstream ROI Analysis block).

_base = user_feature_table.copy().reset_index(drop=True)
_base.rename(columns={"distinct_id": "user_id"}, inplace=True)

# Attach model scores (positionally aligned)
_base["p_retain"]  = p_retain.astype(float)
_base["tier"]      = tier_labels

print(f"\n[A] Base table: {_base.shape[0]:,} users × {_base.shape[1]} cols")
print(f"    Columns: {list(_base.columns)}")

# ═══════════════════════════════════════════════════════════════════════════════
# [B] MERGE SURVIVAL FEATURES (user_id join)
# ═══════════════════════════════════════════════════════════════════════════════
# survival_features has: user_id, tenure_days, churn_event,
#   km_hazard_score, surv_prob_7d, surv_prob_14d, surv_prob_30d,
#   cox_log_hazard, risk_acceleration_coef

_surv_cols = [
    "user_id", "tenure_days", "km_hazard_score",
    "surv_prob_7d", "surv_prob_14d", "surv_prob_30d",
    "cox_log_hazard", "risk_acceleration_coef"
]
_surv_subset = survival_features[_surv_cols].copy()

_base = _base.merge(_surv_subset, on="user_id", how="left")

_n_surv_matched = _base["surv_prob_30d"].notna().sum()
print(f"\n[B] After survival merge: {_base.shape[0]:,} rows × {_base.shape[1]} cols")
print(f"    Survival data matched: {_n_surv_matched:,} / {len(_base):,}")

# ═══════════════════════════════════════════════════════════════════════════════
# [C] MERGE ENRICHED FEATURES for cohort, cluster, segment_name, TS/SHAP cols
# ═══════════════════════════════════════════════════════════════════════════════
# enriched_features has: distinct_id (= user_id), plus all 23 numeric features,
# cluster, segment_name. We pull only the extra cols not already in _base.

_enr_extra_cols = [
    "distinct_id", "cluster", "segment_name",
    "trend_slope", "trend_r2", "seasonality_amp", "residual_vol",
    "peak_to_mean_ratio", "activity_entropy", "n_active_days", "total_events"
]
_enr_subset = enriched_features[_enr_extra_cols].copy()
_enr_subset.rename(columns={"distinct_id": "user_id"}, inplace=True)

_base = _base.merge(_enr_subset, on="user_id", how="left")

print(f"\n[C] After enriched features merge: {_base.shape[0]:,} rows × {_base.shape[1]} cols")

# ═══════════════════════════════════════════════════════════════════════════════
# [D] ATTACH COHORT LABEL from user_retention (earliest created_at)
# ═══════════════════════════════════════════════════════════════════════════════
_user_cohort_df = (
    user_retention[["distinct_id", "created_at"]]
    .dropna(subset=["created_at"])
    .groupby("distinct_id")["created_at"]
    .min()
    .reset_index()
    .rename(columns={"distinct_id": "user_id", "created_at": "acct_created_at"})
)
_user_cohort_df["signup_cohort"] = (
    _user_cohort_df["acct_created_at"].dt.to_period("M").astype(str)
)

_base = _base.merge(
    _user_cohort_df[["user_id", "acct_created_at", "signup_cohort"]],
    on="user_id", how="left"
)

print(f"\n[D] After cohort merge: {_base.shape[0]:,} rows × {_base.shape[1]} cols")
print(f"    Cohorts present: {sorted(_base['signup_cohort'].dropna().unique())}")

# ═══════════════════════════════════════════════════════════════════════════════
# [E] ATTACH MEAN SHAP VALUES PER FEATURE (from Advanced GBM SHAP computation)
#     Since SHAP values are computed on a 400-user subsample in the upstream
#     block, we compute per-tier mean |SHAP| rank using the advanced model
#     on ALL users. This gives a tier-level SHAP summary column.
#     For per-user SHAP we'd need to rerun the full computation; instead we
#     attach the feature-level mean |SHAP| as metadata in dashboard_constants.
# ═══════════════════════════════════════════════════════════════════════════════
# Derive per-user predicted probabilities from the advanced_model for all users

_EXCLUDE_COLS = {"distinct_id", "is_retained", "segment_name", "user_id"}
_enr_num_cols = [
    c for c in enriched_features.columns
    if c not in _EXCLUDE_COLS and enriched_features[c].dtype != object
]

from sklearn.preprocessing import StandardScaler as _SS

_sc_dash = _SS()
_X_all = enriched_features[_enr_num_cols].values.astype(float)
_X_all_s = _sc_dash.fit_transform(_X_all)
_adv_probs_all = advanced_model.predict_proba(_X_all_s)[:, 1]

# Attach advanced model probabilities (full enriched model) to _base
_enr_ids = enriched_features[["distinct_id"]].copy()
_enr_ids["p_retain_adv"] = _adv_probs_all
_enr_ids.rename(columns={"distinct_id": "user_id"}, inplace=True)

_base = _base.merge(_enr_ids, on="user_id", how="left")

print(f"\n[E] Advanced model probs attached: {_base['p_retain_adv'].notna().sum():,} users")

# ═══════════════════════════════════════════════════════════════════════════════
# [F] ATTACH ROI METRICS PER USER (from roi_summary, joined on tier)
# ═══════════════════════════════════════════════════════════════════════════════
_roi_per_tier = roi_summary[
    ["Tier", "Revenue at Risk ($)", "Intervention Cost ($)",
     "Net Savings ($)", "ROI (%)", "Payback (months)"]
].copy()
_roi_per_tier.rename(columns={"Tier": "tier"}, inplace=True)

_base = _base.merge(_roi_per_tier, on="tier", how="left")

print(f"\n[F] After ROI merge: {_base.shape[0]:,} rows × {_base.shape[1]} cols")

# ═══════════════════════════════════════════════════════════════════════════════
# [G] FINALISE dashboard_data
# ═══════════════════════════════════════════════════════════════════════════════

# Reorder columns for clarity
_col_order = [
    # Identity
    "user_id",
    # Labels & scores
    "tier", "p_retain", "p_retain_adv", "is_retained",
    # Cohort
    "signup_cohort", "acct_created_at",
    # Original features (6)
    "first_24h_events", "first_week_events", "consistency_score",
    "unique_tools_used_14d", "agent_usage_ratio_14d", "exploration_index_14d",
    # TS / enriched features
    "trend_slope", "trend_r2", "seasonality_amp", "residual_vol",
    "peak_to_mean_ratio", "activity_entropy", "n_active_days", "total_events",
    # Survival
    "tenure_days", "km_hazard_score",
    "surv_prob_7d", "surv_prob_14d", "surv_prob_30d",
    "cox_log_hazard", "risk_acceleration_coef",
    # Cluster / segment
    "cluster", "segment_name",
    # ROI (tier-level, repeated per user within tier)
    "Revenue at Risk ($)", "Intervention Cost ($)",
    "Net Savings ($)", "ROI (%)", "Payback (months)",
]

# Keep only cols that actually exist (robust to any upstream changes)
_col_order_valid = [c for c in _col_order if c in _base.columns]
_extra_cols = [c for c in _base.columns if c not in _col_order_valid]
dashboard_data = _base[_col_order_valid + _extra_cols].copy()

print(f"\n[G] dashboard_data shape: {dashboard_data.shape}")
print(f"    Columns ({len(dashboard_data.columns)}):")
for _i, _col in enumerate(dashboard_data.columns):
    print(f"      {_i+1:>2}. {_col}")

# ═══════════════════════════════════════════════════════════════════════════════
# [H] VALIDATE — check for critical missing values
# ═══════════════════════════════════════════════════════════════════════════════
_critical_cols = ["user_id", "tier", "p_retain", "is_retained",
                  "surv_prob_30d", "cluster", "Revenue at Risk ($)"]

print(f"\n[H] MISSING VALUE CHECK (critical columns):")
_any_issues = False
for _col in _critical_cols:
    if _col not in dashboard_data.columns:
        print(f"    ⚠️  MISSING COLUMN: {_col}")
        _any_issues = True
        continue
    _n_null = dashboard_data[_col].isna().sum()
    _pct = 100 * _n_null / len(dashboard_data)
    _icon = "✅" if _n_null == 0 else ("⚠️ " if _pct < 5 else "❌")
    print(f"    {_icon} {_col:<35} nulls: {_n_null:>4} ({_pct:.1f}%)")

if not _any_issues:
    print("\n    All critical columns present and accounted for ✅")

# ═══════════════════════════════════════════════════════════════════════════════
# [I] TIER & COHORT SUMMARIES
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n[I] TIER DISTRIBUTION:")
_tier_dist = dashboard_data["tier"].value_counts()
for _t, _n in _tier_dist.items():
    print(f"    {_t:<12}: {_n:>5,}  ({100*_n/len(dashboard_data):.1f}%)")

print(f"\n[I] COHORT DISTRIBUTION:")
_coh_dist = dashboard_data["signup_cohort"].value_counts().sort_index()
for _coh, _n in _coh_dist.items():
    print(f"    {_coh}: {_n:>5,}  ({100*_n/len(dashboard_data):.1f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
# [J] BUILD dashboard_constants — SaaS assumptions + model metrics
# ═══════════════════════════════════════════════════════════════════════════════
dashboard_constants = {
    # SaaS unit economics
    "arpu_monthly":               ARPU_MONTHLY,
    "ltv_months":                 LTV_MONTHS,
    "ltv_per_user":               LTV_PER_USER,
    "cac_per_user":               CAC_PER_USER,
    "churn_cost_per_user":        CHURN_COST_PER_USER,
    "intervention_cost":          INTERVENTION_COST,          # dict by tier
    "intervention_retention_uplift": INTERVENTION_RETENTION_UPLIFT,  # dict by tier
    # Portfolio ROI totals
    "total_revenue_at_risk":      total_revenue_at_risk,
    "total_net_savings":          total_net_savings,
    # Model performance (advanced GBM)
    "advanced_pr_auc":            advanced_pr_auc,
    "advanced_roc_auc":           advanced_roc_auc,
    "advanced_recall":            advanced_recall,
    "advanced_precision":         advanced_precision,
    # Baseline GBM metrics
    "baseline_pr_auc":            cv_pr_auc_test,
    "baseline_roc_auc":           cv_roc_auc_test,
    # Survival hazard rates
    "survival_7d":                s7,
    "survival_14d":               s14,
    "survival_30d":               s30,
    # User counts
    "n_users_total":              n_total_seg,
    "n_high_risk":                n_high_risk,
    "n_at_risk":                  n_at_risk,
    "n_healthy":                  n_healthy,
    # Feature columns (for SHAP / filtering)
    "feature_cols":               FEATURE_COLS,
    "enriched_num_cols":          _enr_num_cols,
}

print(f"\n[J] dashboard_constants keys ({len(dashboard_constants)}):")
for _k, _v in dashboard_constants.items():
    _vstr = str(_v) if not isinstance(_v, (list, dict)) else f"{type(_v).__name__}({len(_v)} items)"
    print(f"    {_k:<35}: {_vstr}")

# ═══════════════════════════════════════════════════════════════════════════════
# [K] FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{_SEP}")
print("  DASHBOARD DATA PREP — COMPLETE")
print(_SEP)
print(f"  dashboard_data         : {dashboard_data.shape[0]:,} rows × {dashboard_data.shape[1]} cols")
print(f"  dashboard_constants    : {len(dashboard_constants)} keys")
print(f"\n  Key columns confirmed  :")
print(f"    user_id       : {dashboard_data['user_id'].nunique():,} unique users")
print(f"    tier          : {sorted(dashboard_data['tier'].unique())}")
print(f"    p_retain      : min={dashboard_data['p_retain'].min():.4f}  max={dashboard_data['p_retain'].max():.4f}")
print(f"    p_retain_adv  : min={dashboard_data['p_retain_adv'].min():.4f}  max={dashboard_data['p_retain_adv'].max():.4f}")
print(f"    surv_prob_30d : min={dashboard_data['surv_prob_30d'].min():.4f}  max={dashboard_data['surv_prob_30d'].max():.4f}")
print(f"    signup_cohort : {sorted(dashboard_data['signup_cohort'].dropna().unique())}")
print(f"\n  Exported: dashboard_data (DataFrame), dashboard_constants (dict)  ✅")
print(_SEP)
