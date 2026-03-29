
import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 6 — ENRICHED FEATURE MATRIX
#  Merges ts_features, survival_features, and behavioral_features into the
#  existing feature matrix (user_feature_scaled / X_train / X_test).
#  Join key: distinct_id (base) ↔ user_id (ts/survival) / distinct_id (behavioral)
# ══════════════════════════════════════════════════════════════════════════════

SEP8 = "═" * 70

print(SEP8)
print("  PHASE 6 — ENRICHED FEATURE MATRIX CONSTRUCTION")
print(SEP8)

# ── [1] Baseline: user_feature_scaled as the anchor ──────────────────────────
_base = user_feature_scaled.copy()
_original_feat_count = _base.shape[1] - 2  # exclude distinct_id and is_retained
print(f"\n[1] Base feature matrix")
print(f"  Shape     : {_base.shape}  ({_base.shape[0]:,} users × {_base.shape[1]} cols)")
print(f"  Feature cols (original): {_original_feat_count}")
print(f"  Columns   : {list(_base.columns)}")

# ── [2] Normalise join keys ───────────────────────────────────────────────────
# ts_features      → 'user_id'  (rename to distinct_id)
# survival_features → 'user_id' (rename to distinct_id)
# behavioral_features → 'distinct_id' already
_ts   = ts_features.copy().rename(columns={"user_id": "distinct_id"})
_surv = survival_features.copy().rename(columns={"user_id": "distinct_id"})
_beh  = behavioral_features.copy()

print(f"\n[2] Source feature sets (before merge)")
print(f"  ts_features        : {ts_features.shape}  (user_id → distinct_id)")
print(f"  survival_features  : {survival_features.shape}  (user_id → distinct_id)")
print(f"  behavioral_features: {behavioral_features.shape}  (distinct_id)")

# ── [3] Drop columns that would duplicate base features ──────────────────────
_BASE_FEAT_COLS = [
    "first_24h_events", "first_week_events", "consistency_score",
    "unique_tools_used_14d", "agent_usage_ratio_14d", "exploration_index_14d",
]
# behavioral_features duplicates the 6 base features; keep only cluster/segment_name
_beh_extra = [c for c in _beh.columns if c not in _BASE_FEAT_COLS and c != "distinct_id"]
_beh_join  = _beh[["distinct_id"] + _beh_extra].copy()

# ts_features: drop 'used_stl' (binary flag, not a modelling feature)
_ts_feat_cols = [c for c in _ts.columns if c not in ("distinct_id", "used_stl")]
_ts_join      = _ts[["distinct_id"] + _ts_feat_cols].copy()

# survival_features: drop 'churn_event' (leaky — it IS the label)
_surv_feat_cols = [c for c in _surv.columns if c not in ("distinct_id", "churn_event")]
_surv_join      = _surv[["distinct_id"] + _surv_feat_cols].copy()

print(f"\n[3] Columns to merge (after de-duplication / leakage removal)")
print(f"  ts_features       : {_ts_feat_cols}  ({len(_ts_feat_cols)} cols)")
print(f"  survival_features : {_surv_feat_cols}  ({len(_surv_feat_cols)} cols)")
print(f"  behavioral_extras : {_beh_extra}  ({len(_beh_extra)} cols)")
print(f"  Total new cols    : {len(_ts_feat_cols) + len(_surv_feat_cols) + len(_beh_extra)}")

# ── [4] Left-join onto base (preserve all base users) ────────────────────────
_merged = _base.copy()
_merged = _merged.merge(_ts_join,   on="distinct_id", how="left")
_merged = _merged.merge(_surv_join, on="distinct_id", how="left")
_merged = _merged.merge(_beh_join,  on="distinct_id", how="left")

print(f"\n[4] Post-merge shape: {_merged.shape}  ({_merged.shape[0]:,} users × {_merged.shape[1]} cols)")

# ── [5] Handle missing values ─────────────────────────────────────────────────
_pre_null     = _merged.isnull().sum()
_cols_w_nulls = _pre_null[_pre_null > 0]

print(f"\n[5] Missing value audit (before fill)")
if len(_cols_w_nulls) == 0:
    print("  ✅ No missing values — all users matched across all feature sets")
else:
    print(f"  {len(_cols_w_nulls)} columns with nulls (likely ts/survival coverage gaps):")
    for _col, _cnt in _cols_w_nulls.items():
        print(f"    {_col:<40}  {_cnt:>5} nulls  ({_cnt/_merged.shape[0]*100:.1f}%)")

# Fill strategy:
#   - Numeric new cols → column median (robust to outliers)
#   - String cols (segment_name) → 'Unknown'
#   - cluster (int) → -1 (unmatched sentinel)
_numeric_new = _ts_feat_cols + _surv_feat_cols
_cat_new     = [c for c in _beh_extra if _merged[c].dtype == object]

for _col in _numeric_new:
    if _col in _merged.columns:
        _merged[_col] = _merged[_col].fillna(_merged[_col].median())

for _col in _cat_new:
    if _col in _merged.columns:
        _merged[_col] = _merged[_col].fillna("Unknown")

if "cluster" in _merged.columns:
    _merged["cluster"] = _merged["cluster"].fillna(-1).astype(int)

_post_null = _merged.isnull().sum().sum()
assert _post_null == 0, f"Still have {_post_null} nulls after fill!"
print(f"\n  ✅ All nulls resolved — total nulls remaining: 0")

# ── [6] Export enriched_features ─────────────────────────────────────────────
enriched_features = _merged.reset_index(drop=True)

_new_feat_cols  = [c for c in enriched_features.columns
                   if c not in ("distinct_id", "is_retained")]
_new_feat_count = len(_new_feat_cols)

print(f"\n[6] Enriched feature matrix exported")
print(f"  Shape       : {enriched_features.shape}")
print(f"  Users       : {enriched_features.shape[0]:,}")
print(f"  Feature cols: {_new_feat_count}  (was {_original_feat_count}, +{_new_feat_count - _original_feat_count} new)")

print(f"\n  Column inventory:")
for _col in _new_feat_cols:
    _dtype  = str(enriched_features[_col].dtype)
    _n_null = enriched_features[_col].isnull().sum()
    print(f"    {_col:<40}  {_dtype:<12}  nulls={_n_null}")

# ── [7] Verify feature count increase vs original ────────────────────────────
assert _new_feat_count > _original_feat_count, (
    f"Expected feature count > {_original_feat_count}, got {_new_feat_count}"
)
print(f"\n  ✅ Feature count check: {_new_feat_count} > {_original_feat_count} (original) — PASS")
print(f"  ✅ Net new features: +{_new_feat_count - _original_feat_count}")

# ── [8] Sample of enriched matrix ────────────────────────────────────────────
print(f"\n[8] Sample enriched_features (first 5 rows, selected columns)")
_sample_cols = (
    ["distinct_id"]
    + _BASE_FEAT_COLS[:3]
    + _ts_feat_cols[:3]
    + _surv_feat_cols[:3]
    + ["cluster", "segment_name"]
    + ["is_retained"]
)
_sample_cols = [c for c in _sample_cols if c in enriched_features.columns]
print(enriched_features[_sample_cols].head(5).to_string(index=False))

# ── [9] Summary ───────────────────────────────────────────────────────────────
print(f"\n{SEP8}")
print(f"  SUMMARY")
print(SEP8)
print(f"  Base users             : {_base.shape[0]:,}")
print(f"  ts_features users      : {_ts.shape[0]:,}")
print(f"  survival_features users: {_surv.shape[0]:,}")
print(f"  behavioral_features    : {_beh.shape[0]:,}")
print(f"  enriched_features      : {enriched_features.shape[0]:,} users × {enriched_features.shape[1]} cols")
print(f"  Original feature count : {_original_feat_count}")
print(f"  Enriched feature count : {_new_feat_count}")
print(f"  Net new features       : +{_new_feat_count - _original_feat_count}")
print(f"  Nulls in output        : 0 ✅")
print(f"  exported variable      : enriched_features ✅")
