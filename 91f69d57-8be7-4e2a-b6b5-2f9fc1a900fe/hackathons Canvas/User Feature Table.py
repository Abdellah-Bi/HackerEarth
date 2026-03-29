
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 4 — USER FEATURE TABLE (leak-free)
#
#  PHASE 4 FIX: retention_label has an int64 RangeIndex due to Zerve
#  serialization — join to string-indexed feature users always produces
#  all-NaN and thus 0 retained. Fixed by computing is_retained directly
#  from label_events, identically to the upstream labeling logic.
#
#  All 6 features computed EXCLUSIVELY from feature_events (days 1–14).
#  Output columns (8 total):
#    distinct_id, first_24h_events, first_week_events, consistency_score,
#    unique_tools_used_14d, agent_usage_ratio_14d, exploration_index_14d,
#    is_retained
# ══════════════════════════════════════════════════════════════════════════════

_label_event_row_count = len(label_events)
assert isinstance(label_events, pd.DataFrame), "label_events must be a DataFrame"
assert isinstance(feature_events, pd.DataFrame), "feature_events must be a DataFrame"
print(f"✅ label_events ({_label_event_row_count:,} rows) — used ONLY for label, not for features.")

USER_COL    = "distinct_id"
TS_COL      = "timestamp"
ACCT_COL    = "created_at"
TOOL_COL    = "prop_tool_name"
CODER_AGENT = "Coder Agent"
WEEK_THRESH = 3  # minimum distinct ISO weeks in label window to count as retained

# ── 1. Work exclusively on feature_events (days 1–14 after signup) ────────────
_fe = feature_events.copy()
print(f"\n[1] feature_events: {_fe.shape[0]:,} rows × {_fe.shape[1]} cols | users: {_fe[USER_COL].nunique():,}")

_signup = _fe.groupby(USER_COL)[ACCT_COL].min().rename("_signup_ts")
_fe = _fe.join(_signup, on=USER_COL)

# ── 2. first_24h_events ───────────────────────────────────────────────────────
_mask_24h = (
    (_fe[TS_COL] >= _fe["_signup_ts"]) &
    (_fe[TS_COL] <  _fe["_signup_ts"] + pd.Timedelta(hours=24))
)
_feat_first_24h = _fe[_mask_24h].groupby(USER_COL).size().rename("first_24h_events").astype(int)
print(f"[2] first_24h_events — users with ≥1 event: {len(_feat_first_24h):,}")

# ── 3. first_week_events ──────────────────────────────────────────────────────
_mask_7d = (
    (_fe[TS_COL] >= _fe["_signup_ts"]) &
    (_fe[TS_COL] <  _fe["_signup_ts"] + pd.Timedelta(days=7))
)
_feat_first_week = _fe[_mask_7d].groupby(USER_COL).size().rename("first_week_events").astype(int)
print(f"[3] first_week_events — users with ≥1 event: {len(_feat_first_week):,}")

# ── 4. consistency_score ──────────────────────────────────────────────────────
_feat_consistency = (
    _fe.groupby(USER_COL)[TS_COL]
    .apply(lambda x: x.dt.date.nunique())
    .rename("consistency_score")
)
print(f"[4] consistency_score — {len(_feat_consistency):,} users")

# ── 5. unique_tools_used_14d ──────────────────────────────────────────────────
_feat_unique_tools = (
    _fe.dropna(subset=[TOOL_COL])
    .groupby(USER_COL)[TOOL_COL]
    .nunique()
    .rename("unique_tools_used_14d")
    .astype(int)
)
print(f"[5] unique_tools_used_14d — {len(_feat_unique_tools):,} users")

# ── 6. agent_usage_ratio_14d ──────────────────────────────────────────────────
_tool_fe    = _fe.dropna(subset=[TOOL_COL])
_coder_cnt  = _tool_fe[_tool_fe[TOOL_COL] == CODER_AGENT].groupby(USER_COL).size().rename("_coder_n")
_tool_cnt   = _tool_fe.groupby(USER_COL).size().rename("_tool_n")
_feat_agent_ratio = (
    _coder_cnt.to_frame()
    .join(_tool_cnt, how="outer")
    .fillna(0)
    .assign(agent_usage_ratio_14d=lambda d: d["_coder_n"] / d["_tool_n"].replace(0, np.nan))
    ["agent_usage_ratio_14d"]
    .fillna(0.0)
)
print(f"[6] agent_usage_ratio_14d — {len(_feat_agent_ratio):,} users")

# ── 7. exploration_index_14d ──────────────────────────────────────────────────
_raw_evts = _fe.groupby(USER_COL).size().rename("_raw_events_14d")
_feat_exploration = (
    _feat_unique_tools.to_frame()
    .join(_raw_evts, how="left")
    .fillna(0)
    .assign(exploration_index_14d=lambda d: d["unique_tools_used_14d"] / (d["_raw_events_14d"] + 1))
    ["exploration_index_14d"]
)
print(f"[7] exploration_index_14d — {len(_feat_exploration):,} users")

# ── 8. is_retained — FIXED: compute directly from label_events ────────────────
# Computes: users w/ ≥ WEEK_THRESH distinct ISO calendar weeks in days 15–90
# This matches the upstream User Retention Labeling block logic exactly,
# but avoids the Zerve RangeIndex serialization issue with retention_label Series.
_le = label_events[[USER_COL, TS_COL]].copy()
_le["_isoweek"] = (
    _le[TS_COL].dt.isocalendar()
    .apply(lambda r: f"{int(r['year'])}-W{int(r['week']):02d}", axis=1)
)
_weeks_per_user = _le.groupby(USER_COL)["_isoweek"].nunique()  # str index (distinct_id)
# Index the feature-window users
_all_feature_users = pd.Index(_fe[USER_COL].unique(), name=USER_COL)
_feat_label = (
    _weeks_per_user
    .reindex(_all_feature_users, fill_value=0)
    .ge(WEEK_THRESH)
    .astype(int)
    .rename("is_retained")
)
print(f"\n[8] Retention label (≥{WEEK_THRESH} label-window weeks)")
print(f"    Retained: {int((_feat_label == 1).sum())} of {len(_feat_label):,} feature-window users ({100*_feat_label.mean():.2f}%)")

# ── 9. Assemble user-level table ───────────────────────────────────────────────
user_feature_table = (
    pd.DataFrame(index=_all_feature_users)
    .join(_feat_first_24h,    how="left")
    .join(_feat_first_week,   how="left")
    .join(_feat_consistency,  how="left")
    .join(_feat_unique_tools, how="left")
    .join(_feat_agent_ratio,  how="left")
    .join(_feat_exploration,  how="left")
    .join(_feat_label,        how="left")
    .reset_index()
)

# Fill nulls for users with zero activity in sub-windows
user_feature_table["first_24h_events"]      = user_feature_table["first_24h_events"].fillna(0).astype(int)
user_feature_table["first_week_events"]     = user_feature_table["first_week_events"].fillna(0).astype(int)
user_feature_table["consistency_score"]     = user_feature_table["consistency_score"].fillna(0.0).astype(int)
user_feature_table["unique_tools_used_14d"] = user_feature_table["unique_tools_used_14d"].fillna(0).astype(int)
user_feature_table["agent_usage_ratio_14d"] = user_feature_table["agent_usage_ratio_14d"].fillna(0.0)
user_feature_table["exploration_index_14d"] = user_feature_table["exploration_index_14d"].fillna(0.0)
user_feature_table["is_retained"]           = user_feature_table["is_retained"].fillna(0).astype(int)

# ── 10. Validation assertions ─────────────────────────────────────────────────
EXPECTED_FEATURE_COLS = [
    "first_24h_events", "first_week_events", "consistency_score",
    "unique_tools_used_14d", "agent_usage_ratio_14d", "exploration_index_14d",
]
EXPECTED_ALL_COLS = ["distinct_id"] + EXPECTED_FEATURE_COLS + ["is_retained"]
LEAKY_FEATURES    = ["total_events", "days_since_first_event"]
FEATURE_COLS      = EXPECTED_FEATURE_COLS

assert list(user_feature_table.columns) == EXPECTED_ALL_COLS, (
    f"Column mismatch!\n  Expected: {EXPECTED_ALL_COLS}\n  Got: {list(user_feature_table.columns)}"
)
for _lf in LEAKY_FEATURES:
    assert _lf not in user_feature_table.columns, f"LEAKY FEATURE FOUND: {_lf}"
assert len(label_events) == _label_event_row_count, \
    f"label_events was mutated! {len(label_events)} != {_label_event_row_count}"
assert user_feature_table[USER_COL].nunique() == len(user_feature_table), "Duplicate user rows!"
_null_counts = user_feature_table.isnull().sum()
assert not _null_counts.any(), f"Nulls found:\n{_null_counts[_null_counts > 0]}"
assert set(user_feature_table["is_retained"].unique()).issubset({0, 1})

print("\n✅ ALL ASSERTIONS PASSED — zero leakage, exactly 8 columns")

# ── 11. Summary ───────────────────────────────────────────────────────────────
print("\n" + "═" * 65)
print("PHASE 4 — USER FEATURE TABLE (leak-free, feature window only)")
print("═" * 65)
_n_retained = int(user_feature_table["is_retained"].sum())
_n_total    = len(user_feature_table)
print(f"\nShape : {user_feature_table.shape}  (8 cols = distinct_id + 6 features + is_retained)")
print(f"Retention rate: {_n_retained}/{_n_total} = {_n_retained/_n_total*100:.2f}%")
print(f"\n✅ Phase 4 fix: is_retained computed from label_events directly")
print(f"   (avoids Zerve int64 index deserialization bug in retention_label)")
print(f"\nColumns: {list(user_feature_table.columns)}")
print(f"\nDescriptive statistics:")
print(user_feature_table[EXPECTED_FEATURE_COLS].describe().round(4).to_string())
print(f"\n✗ Excluded leaky features: {LEAKY_FEATURES}")
print(f"\nSample — 5 rows:")
print(user_feature_table.head(5).to_string(index=False))
