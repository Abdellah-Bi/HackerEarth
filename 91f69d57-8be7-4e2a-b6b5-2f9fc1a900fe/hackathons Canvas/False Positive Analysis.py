
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

# ── Reproduce exact pipeline to get FP distinct_ids ──────────────────────────
# Constants matching upstream blocks
_USER_COL    = "distinct_id"
_TS_COL      = "timestamp"
_CREATED_COL = "created_at"
_TOOL_COL    = "prop_tool_name"
_CODER_AGENT = "Coder Agent"
_THRESHOLD   = 70.0
_WEEK_THRESH = 3
_FEATURE_COLS = [
    "log_total_events",
    "log_days_since_first_event",
    "agent_usage_ratio",
    "unique_tools_used",
    "first_week_events",
]
_LABEL_COL = "is_retained"

# ── Step 1: Rebuild raw → feature table (same logic as User Feature Table) ───
_raw = user_retention.copy()

# Drop >70% missing cols
_missing_pct = _raw.isnull().sum() / len(_raw) * 100
_cols_to_drop = _missing_pct[_missing_pct > _THRESHOLD].index.tolist()

# is_retained
_iso_week_s = _raw[_TS_COL].dt.isocalendar().apply(
    lambda r: f"{int(r['year'])}-W{int(r['week']):02d}", axis=1
)
_raw_w = _raw.copy()
_raw_w["_iso_week"] = _iso_week_s
_distinct_weeks = _raw_w.groupby(_USER_COL)["_iso_week"].nunique()
_is_retained    = (_distinct_weeks >= _WEEK_THRESH).astype(int).rename("is_retained")

# total_events (raw count for unscaled use)
_feat_total_events_raw = _raw.groupby(_USER_COL).size().rename("total_events_raw").astype(int)

# total_events log1p
_feat_total_events = np.log1p(_feat_total_events_raw).rename("total_events")

# agent_usage_ratio
_tool_only = _raw[_raw[_TOOL_COL].notna()]
_coder_n   = (_tool_only[_tool_only[_TOOL_COL] == _CODER_AGENT]
              .groupby(_USER_COL).size().rename("_coder_n"))
_tool_n    = _tool_only.groupby(_USER_COL).size().rename("_tool_n")
_feat_agent_usage = (
    _coder_n.to_frame()
    .join(_tool_n, how="outer")
    .fillna(0)
    .assign(agent_usage_ratio=lambda d: d["_coder_n"] / d["_tool_n"].replace(0, np.nan))
    ["agent_usage_ratio"]
    .fillna(0.0)
)

# days_since_first_event log1p
_max_ts = _raw[_TS_COL].max()
_feat_days_since = (
    np.log1p(
        _raw.groupby(_USER_COL)[_TS_COL]
        .min()
        .apply(lambda t: (_max_ts - t).days)
        .astype(int)
    )
    .rename("days_since_first_event")
)

# unique_tools_used (raw/unscaled)
_feat_unique_tools = (
    _raw.dropna(subset=[_TOOL_COL])
    .groupby(_USER_COL)[_TOOL_COL]
    .nunique()
    .rename("unique_tools_used")
    .astype(int)
)

# first_week_events
_first_signup = _raw.groupby(_USER_COL)[_CREATED_COL].min().rename("_first_signup_ts")
_raw_ws = _raw[[_USER_COL, _TS_COL]].join(_first_signup, on=_USER_COL)
_within_7d = (
    (_raw_ws[_TS_COL] >= _raw_ws["_first_signup_ts"]) &
    (_raw_ws[_TS_COL] <  _raw_ws["_first_signup_ts"] + pd.Timedelta(days=7))
)
_feat_first_week_events = (
    _raw_ws[_within_7d]
    .groupby(_USER_COL)
    .size()
    .rename("first_week_events")
    .astype(int)
)

# Build feature table (unscaled)
_uft = (
    _feat_total_events.to_frame()
    .join(_feat_agent_usage,       how="left")
    .join(_feat_days_since,        how="left")
    .join(_feat_unique_tools,      how="left")
    .join(_feat_first_week_events, how="left")
    .join(_is_retained,            how="left")
    .join(_feat_total_events_raw,  how="left")
    .reset_index()
)
_uft["unique_tools_used"]  = _uft["unique_tools_used"].fillna(0).astype(int)
_uft["agent_usage_ratio"]  = _uft["agent_usage_ratio"].fillna(0.0)
_uft["first_week_events"]  = _uft["first_week_events"].fillna(0).astype(int)
_uft["is_retained"]        = _uft["is_retained"].astype(int)

# ── Step 2: Scale exactly as Feature Scaling block ───────────────────────────
_RENAME_MAP = {
    "total_events":           "log_total_events",
    "days_since_first_event": "log_days_since_first_event",
}
_ft = _uft.rename(columns=_RENAME_MAP)
_scale_cols = [_RENAME_MAP.get(c, c) for c in [
    "total_events", "days_since_first_event",
    "agent_usage_ratio", "unique_tools_used", "first_week_events",
]]

_scaler = StandardScaler()
_scaled_vals = _scaler.fit_transform(_ft[_scale_cols])
_scaled_df   = pd.DataFrame(_scaled_vals, columns=_scale_cols, index=_ft.index)

_fp_ufs = pd.concat([
    _ft[[_USER_COL]].reset_index(drop=True),
    _scaled_df.reset_index(drop=True),
    _ft[[_LABEL_COL]].reset_index(drop=True),
], axis=1)

# ── Step 3: Train/test split — identical random_state, stratify ──────────────
_X = _fp_ufs[_FEATURE_COLS].values
_y = _fp_ufs[_LABEL_COL].values

_X_train, _X_test, _y_train, _y_test, _idx_train, _idx_test = train_test_split(
    _X, _y, np.arange(len(_y)),
    test_size=0.2,
    random_state=42,
    stratify=_y,
)

# Also split the full feature table rows to map back to distinct_id
_fp_test_meta = _fp_ufs.iloc[_idx_test].reset_index(drop=True)

# ── Step 4: Train same GBM model ─────────────────────────────────────────────
_sample_weights = compute_sample_weight(class_weight={0: 1, 1: 38}, y=_y_train)

_gbm = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    max_features=0.8,
    random_state=42,
    validation_fraction=0.1,
    n_iter_no_change=20,
    tol=1e-4,
    verbose=0,
)
_gbm.fit(_X_train, _y_train, sample_weight=_sample_weights)

# ── Step 5: Predict on test set ───────────────────────────────────────────────
_preds = _gbm.predict(_X_test)

# ── Step 6: Identify False Positives (pred=1, actual=0) ──────────────────────
_fp_mask = (_preds == 1) & (_y_test == 0)
_fp_test = _fp_test_meta[_fp_mask].copy()
_fp_distinct_ids = set(_fp_test[_USER_COL].tolist())

print("=" * 65)
print("FALSE POSITIVE ANALYSIS  (predicted retained=1, actual=0)")
print("=" * 65)
print(f"Total false positives in test set : {len(_fp_distinct_ids)}")
print()

# ── Q1: Average unique_tools_used (unscaled) ──────────────────────────────────
# Merge back to original unscaled feature table to get raw unique_tools_used
_uft_indexed = _uft.set_index(_USER_COL)
_fp_unscaled = _uft_indexed.loc[list(_fp_distinct_ids)]
_avg_unique_tools = _fp_unscaled["unique_tools_used"].mean()

print("=" * 65)
print("Q1 — AVERAGE unique_tools_used (unscaled) for False Positives")
print("=" * 65)
print(f"  N false positives         : {len(_fp_unscaled)}")
print(f"  Mean unique_tools_used    : {_avg_unique_tools:.4f}")
print(f"  Median unique_tools_used  : {_fp_unscaled['unique_tools_used'].median():.1f}")
print(f"  Min / Max                 : {_fp_unscaled['unique_tools_used'].min()} / {_fp_unscaled['unique_tools_used'].max()}")
print()

# ── Q2: How many FP users reached 3+ total_events (raw count)? ───────────────
# "total_events" in uft is log1p of raw count → expm1 to recover raw count
_fp_unscaled = _fp_unscaled.copy()
_fp_unscaled["total_events_raw"] = _fp_unscaled["total_events_raw"].astype(int)
_fp_reached_3 = _fp_unscaled[_fp_unscaled["total_events_raw"] >= 3]
_n_reached_3  = len(_fp_reached_3)
_pct_reached_3 = _n_reached_3 / len(_fp_unscaled) * 100

print("=" * 65)
print("Q2 — FALSE POSITIVES WHO REACHED 3+ total_events (raw count)")
print("=" * 65)
print(f"  Total FP users                   : {len(_fp_unscaled)}")
print(f"  FP users with total_events >= 3  : {_n_reached_3}")
print(f"  Percentage                       : {_pct_reached_3:.1f}%")
print()

# ── Q3: Top 5 prop_tool_name for FP users who hit 3+ events ──────────────────
_fp_3plus_ids = set(_fp_reached_3.index.tolist())

# Join back to raw event-level data
_raw_events = user_retention[
    (user_retention[_USER_COL].isin(_fp_3plus_ids)) &
    (user_retention[_TOOL_COL].notna())
][[_USER_COL, _TOOL_COL]]

# Frequency table
_tool_freq = _raw_events[_TOOL_COL].value_counts()
_tool_total = _tool_freq.sum()
_top5_tools = _tool_freq.head(5)

_top5_df = pd.DataFrame({
    "prop_tool_name": _top5_tools.index,
    "count":          _top5_tools.values,
    "pct_of_tool_events": (_top5_tools.values / _tool_total * 100).round(2),
}).reset_index(drop=True)
_top5_df.index = range(1, len(_top5_df) + 1)
_top5_df.index.name = "Rank"

print("=" * 65)
print("Q3 — TOP 5 prop_tool_name FOR FP USERS WHO HIT 3+ TOTAL EVENTS")
print(f"     (based on {len(_fp_3plus_ids)} FP user(s), {_tool_total} tool-tagged events)")
print("=" * 65)
print(_top5_df.to_string())
print()
print(f"* Percentages are of all {_tool_total} tool-tagged events from these {len(_fp_3plus_ids)} FP users")
print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  False positives identified : {len(_fp_distinct_ids)}")
print(f"  Avg unique_tools_used      : {_avg_unique_tools:.4f}")
print(f"  Reached 3+ events          : {_n_reached_3} / {len(_fp_unscaled)} ({_pct_reached_3:.1f}%)")
print(f"  Top tool (for 3+ FPs)      : {_top5_df['prop_tool_name'].iloc[0] if len(_top5_df) > 0 else 'N/A'}")
