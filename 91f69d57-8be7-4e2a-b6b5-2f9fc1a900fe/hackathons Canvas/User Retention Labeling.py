
import pandas as pd

# ══════════════════════════════════════════════════════════════════
#  BLOCK 3 — COHORT DEFINITION + WINDOWED EVENT PARTITIONING
#
#  Step 1: Parse each user's signup date from `created_at`
#  Step 2: Hard-partition ALL events into two disjoint windows:
#          • feature_events  → days  1 – 14 after signup (inclusive)
#          • label_events    → days 15 – 90 after signup (inclusive)
#  Step 3: Compute is_retained exclusively from label_events
#          (≥ WEEK_THRESH distinct ISO-weeks active → retained)
#  Step 4: Merge label back to produce `retention_labeled`
#  Step 5: Assert zero-overlap and print cohort statistics
# ══════════════════════════════════════════════════════════════════

# ── Column references ──────────────────────────────────────────────
USER_COL   = "distinct_id"
TS_COL     = "timestamp"
ACCT_COL   = "created_at"   # per-row account creation timestamp

# ── Window constants ──────────────────────────────────────────────
FEATURE_WINDOW_DAYS   = 14   # days 1–14 after signup = feature window
LABEL_WINDOW_START    = 15   # first day of label window
LABEL_WINDOW_END      = 90   # last  day of label window (inclusive)

# ── Retention threshold (unchanged) ──────────────────────────────
WEEK_THRESH = 3   # ≥ 3 distinct ISO-weeks in label window → retained

SEP  = "═" * 64
SEP2 = "─" * 64

print(SEP)
print("  BLOCK 3 │ COHORT DEFINITION & WINDOWED EVENT PARTITIONING")
print(SEP)
print(f"\n  FEATURE_WINDOW_DAYS : days  1 – {FEATURE_WINDOW_DAYS}  after signup")
print(f"  LABEL_WINDOW        : days {LABEL_WINDOW_START} – {LABEL_WINDOW_END} after signup")
print(f"  WEEK_THRESH         : {WEEK_THRESH} distinct ISO-weeks (label window only)")
print()

# ─────────────────────────────────────────────────────────────────
# [1] PARSE SIGNUP DATE — per user, take the minimum created_at
# ─────────────────────────────────────────────────────────────────
print("[1] Parsing user signup dates from `created_at`...")
print(SEP2)

_base = eda_df.copy()   # 409k rows, 107 cols from block 2

# Each row already has the user's created_at; use the minimum per user
# to guard against any per-row variation (should be identical per user)
_signup_dates = (
    _base.groupby(USER_COL)[ACCT_COL]
    .min()
    .rename("_signup_ts")
    .reset_index()
)

print(f"  Unique users with signup dates: {len(_signup_dates):,}")
print(f"  Earliest signup : {_signup_dates['_signup_ts'].min()}")
print(f"  Latest signup   : {_signup_dates['_signup_ts'].max()}")

# Attach signup date to every event row
_df = _base.merge(_signup_dates, on=USER_COL, how="left")

# Compute days elapsed since signup for every event
_df["_days_since_signup"] = (
    (_df[TS_COL] - _df["_signup_ts"]).dt.total_seconds() / 86_400
)

print(f"\n  Events with valid days_since_signup: {_df['_days_since_signup'].notna().sum():,}")
print(f"  Days range: [{_df['_days_since_signup'].min():.2f}, {_df['_days_since_signup'].max():.2f}]")

# ─────────────────────────────────────────────────────────────────
# [2] HARD-PARTITION INTO DISJOINT WINDOWS
#     feature_events : 0 < days_since_signup <= FEATURE_WINDOW_DAYS
#     label_events   : LABEL_WINDOW_START <= days_since_signup <= LABEL_WINDOW_END
#     (events outside both windows are discarded from analysis)
#
#     NOTE: days_since_signup uses fractional days from signup timestamp.
#     We treat day boundaries as >0 and <=N to capture "day N" inclusively.
# ─────────────────────────────────────────────────────────────────
print(f"\n[2] Partitioning events into disjoint windows...")
print(SEP2)

_feature_mask = (
    (_df["_days_since_signup"] > 0) &
    (_df["_days_since_signup"] <= FEATURE_WINDOW_DAYS)
)
_label_mask = (
    (_df["_days_since_signup"] >= LABEL_WINDOW_START) &
    (_df["_days_since_signup"] <= LABEL_WINDOW_END)
)

feature_events = _df[_feature_mask].copy()
label_events   = _df[_label_mask].copy()

# Drop internal helper columns from output frames
feature_events = feature_events.drop(columns=["_signup_ts", "_days_since_signup"])
label_events   = label_events.drop(columns=["_signup_ts", "_days_since_signup"])

print(f"  feature_events  (days  1–{FEATURE_WINDOW_DAYS})  : {len(feature_events):>8,} rows")
print(f"  label_events    (days {LABEL_WINDOW_START}–{LABEL_WINDOW_END}) : {len(label_events):>8,} rows")
print(f"  Events outside both windows     : {len(_df) - len(feature_events) - len(label_events):>8,} rows (discarded)")
print(f"\n  Users in feature window: {feature_events[USER_COL].nunique():,}")
print(f"  Users in label window  : {label_events[USER_COL].nunique():,}")

# ─────────────────────────────────────────────────────────────────
# [3] ASSERT ZERO OVERLAP — the two windows must be perfectly disjoint
# ─────────────────────────────────────────────────────────────────
print(f"\n[3] Zero-overlap assertion...")
print(SEP2)

# Overlap check via UUID (each event has a unique uuid)
_feat_uuids  = set(feature_events["uuid"].values)
_label_uuids = set(label_events["uuid"].values)
_overlap_uuids = _feat_uuids & _label_uuids

print(f"  Overlap (shared event UUIDs) : {len(_overlap_uuids)}")
assert len(_overlap_uuids) == 0, (
    f"CRITICAL: {len(_overlap_uuids)} events appear in BOTH feature and label windows!"
)
print("  ✅ ZERO overlap confirmed — windows are fully disjoint")

# ─────────────────────────────────────────────────────────────────
# [4] COMPUTE is_retained EXCLUSIVELY FROM LABEL WINDOW
# ─────────────────────────────────────────────────────────────────
print(f"\n[4] Computing is_retained from label_events only (days {LABEL_WINDOW_START}–{LABEL_WINDOW_END})...")
print(SEP2)

# ISO year-week strings for label window events only
_label_week = label_events[TS_COL].dt.isocalendar().apply(
    lambda r: f"{int(r['year'])}-W{int(r['week']):02d}", axis=1
)
_label_with_week = label_events.copy()
_label_with_week["_iso_week"] = _label_week

_distinct_label_weeks = (
    _label_with_week
    .groupby(USER_COL)["_iso_week"]
    .nunique()
    .rename("_label_weeks")
)

# All users in the cohort (appear in either window)
_all_cohort_users = pd.Index(
    set(feature_events[USER_COL].unique()) | set(label_events[USER_COL].unique()),
    name=USER_COL
)

# Build retention label: users not in label window get 0 (no activity → churned)
retention_label = (
    _distinct_label_weeks
    .reindex(_all_cohort_users, fill_value=0)
    .pipe(lambda s: (s >= WEEK_THRESH).astype(int))
    .rename("is_retained")
)

_n_cohort   = len(retention_label)
_n_retained = int((retention_label == 1).sum())
_n_churned  = int((retention_label == 0).sum())
_retention_rate = _n_retained / _n_cohort * 100

print(f"  Total cohort users  : {_n_cohort:,}")
print(f"  Retained (label=1)  : {_n_retained:,}  ({_retention_rate:.2f}%)")
print(f"  Churned  (label=0)  : {_n_churned:,}  ({100 - _retention_rate:.2f}%)")

assert set(retention_label.unique()).issubset({0, 1}), "is_retained must only contain 0/1!"
print("\n  ✅ is_retained is binary (0/1 only)")

# ─────────────────────────────────────────────────────────────────
# [5] MERGE LABEL BACK — produce retention_labeled (feature window events)
#
#     retention_labeled  : feature_events + is_retained column
#                          (the feature engineering base for downstream)
#     retention_clean    : alias of retention_labeled (preserves contract
#                          with downstream User Retention Labeling / Feature
#                          Table blocks that reference `retention_clean`)
# ─────────────────────────────────────────────────────────────────
print(f"\n[5] Merging is_retained onto feature_events...")
print(SEP2)

retention_labeled = feature_events.merge(
    retention_label.reset_index(),   # distinct_id | is_retained
    on=USER_COL,
    how="left",
)

# Users with events in feature window but zero label-window activity → 0
retention_labeled["is_retained"] = retention_labeled["is_retained"].fillna(0).astype(int)

assert "is_retained" in retention_labeled.columns, "Merge failed — is_retained missing!"
assert set(retention_labeled["is_retained"].unique()).issubset({0, 1}), \
    "Non-binary values in is_retained!"
assert len(retention_labeled) == len(feature_events), \
    "Row count changed after label merge!"

print(f"  retention_labeled shape: {retention_labeled.shape[0]:,} rows × {retention_labeled.shape[1]} cols")

# Keep only the 19-col slim schema (drop >70% sparse cols) for `retention_clean`
# to preserve backward compatibility with downstream blocks
_THRESHOLD    = 70.0
_missing_pct  = retention_labeled.isnull().sum() / len(retention_labeled) * 100
_cols_to_drop = _missing_pct[_missing_pct > _THRESHOLD].index.tolist()

retention_clean = retention_labeled.drop(columns=_cols_to_drop)
print(f"  retention_clean shape  : {retention_clean.shape[0]:,} rows × {retention_clean.shape[1]} cols")
print(f"  (dropped {len(_cols_to_drop)} cols with >{_THRESHOLD:.0f}% missing)")

# ─────────────────────────────────────────────────────────────────
# [6] SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print("  BLOCK 3 — COHORT & WINDOW SUMMARY")
print(f"{'='*64}")
print(f"\n  Window Definitions")
print(f"  {'─'*40}")
print(f"  Feature window  : days  1 – {FEATURE_WINDOW_DAYS} after signup")
print(f"  Label window    : days {LABEL_WINDOW_START} – {LABEL_WINDOW_END} after signup")
print(f"  Retention rule  : ≥ {WEEK_THRESH} distinct ISO-weeks active in label window")
print(f"\n  Cohort Sizes")
print(f"  {'─'*40}")
print(f"  Users in feature window  : {feature_events[USER_COL].nunique():,}")
print(f"  Users in label window    : {label_events[USER_COL].nunique():,}")
print(f"  Total cohort size        : {_n_cohort:,}")
print(f"\n  Retention Distribution")
print(f"  {'─'*40}")
print(f"  {'Label':<20}  {'Users':>8}  {'%':>8}")
print(f"  {'─'*20}  {'─'*8}  {'─'*8}")
print(f"  {'0 (Churned)':<20}  {_n_churned:>8,}  {100-_retention_rate:>7.2f}%")
print(f"  {'1 (Retained)':<20}  {_n_retained:>8,}  {_retention_rate:>7.2f}%")
print(f"  {'─'*20}  {'─'*8}  {'─'*8}")
print(f"  {'TOTAL':<20}  {_n_cohort:>8,}  {'100.00%':>8}")
print(f"\n  Data Integrity")
print(f"  {'─'*40}")
print(f"  Overlap between windows  : {len(_overlap_uuids)} events  ← must be 0")
print(f"  is_retained binary check : PASSED")
print(f"\n{'='*64}")
print("  BLOCK 3 — COHORT DEFINITION COMPLETE ✔")
print(f"{'='*64}")
