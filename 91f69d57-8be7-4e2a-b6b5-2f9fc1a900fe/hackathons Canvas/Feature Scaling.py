
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — BLOCK 08: TRAIN/TEST SPLIT & SCALING
#
#  Temporal cohort split:
#    - Derive each user's signup date (min created_at from feature_events)
#    - Sort users by signup date (ascending = oldest first)
#    - Train  → oldest ~80% of signup cohorts
#    - Test   → newest ~20% of signup cohorts
#    - No random shuffling: this respects temporal ordering to prevent
#      future-cohort leakage into training
#
#  StandardScaler is fit EXCLUSIVELY on X_train.
#  X_test is only transformed (never used to fit the scaler).
#  An explicit assertion verifies no test-set rows influenced the scaler fit.
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS  = [
    "first_24h_events",
    "first_week_events",
    "consistency_score",
    "unique_tools_used_14d",
    "agent_usage_ratio_14d",
    "exploration_index_14d",
]
LABEL_COL     = "is_retained"
USER_COL      = "distinct_id"
ACCT_TS_COL   = "created_at"    # per-row account creation timestamp
TRAIN_RATIO   = 0.80
RANDOM_STATE  = 42              # reproducibility (not used for splitting, just documented)

BG         = "#1D1D20"
TEXT_PRI   = "#fbfbff"
C_BLUE     = "#A1C9F4"
C_ORANGE   = "#FFB482"
C_GREEN    = "#8DE5A1"
GRID_COL   = "#2e2e33"
SEP = "═" * 68

print(SEP)
print("  PHASE 3 — TRAIN/TEST SPLIT & SCALING")
print(SEP)

# ── [1] VALIDATE INPUT ────────────────────────────────────────────────────────
print(f"\n[1] Input validation")
print(f"  user_feature_table : {user_feature_table.shape[0]:,} users × {user_feature_table.shape[1]} cols")
print(f"  feature_events     : {feature_events.shape[0]:,} rows × {feature_events.shape[1]} cols")

_missing_feats = [f for f in FEATURE_COLS if f not in user_feature_table.columns]
assert not _missing_feats, f"Missing feature columns: {_missing_feats}"
assert LABEL_COL in user_feature_table.columns, f"'{LABEL_COL}' not in user_feature_table"
assert USER_COL  in user_feature_table.columns, f"'{USER_COL}' not in user_feature_table"
assert ACCT_TS_COL in feature_events.columns,   f"'{ACCT_TS_COL}' not in feature_events"
print("  ✅ All required columns present")

# ── [2] DERIVE USER SIGNUP DATE ───────────────────────────────────────────────
print(f"\n[2] Deriving per-user signup date from feature_events.created_at")
# Use minimum created_at per user (most conservative signup date)
_signup_map = (
    feature_events.groupby(USER_COL)[ACCT_TS_COL]
    .min()
    .rename("signup_ts")
    .reset_index()
)
print(f"  Signup dates derived for {len(_signup_map):,} users")
print(f"  Earliest signup : {_signup_map['signup_ts'].min()}")
print(f"  Latest signup   : {_signup_map['signup_ts'].max()}")

# ── [3] MERGE SIGNUP DATE INTO FEATURE TABLE ──────────────────────────────────
_uft_with_signup = user_feature_table.merge(_signup_map, on=USER_COL, how="left")
_missing_signup  = _uft_with_signup["signup_ts"].isna().sum()
if _missing_signup > 0:
    print(f"  ⚠️  {_missing_signup} users in feature table not found in feature_events → filling with global max (treated as newest)")
    _uft_with_signup["signup_ts"] = _uft_with_signup["signup_ts"].fillna(_uft_with_signup["signup_ts"].max())
else:
    print(f"  ✅ All {len(_uft_with_signup):,} feature-table users matched to a signup date")

# ── [4] TEMPORAL COHORT SORT & SPLIT ─────────────────────────────────────────
print(f"\n[4] Temporal cohort split  (oldest {TRAIN_RATIO*100:.0f}% → train, newest {(1-TRAIN_RATIO)*100:.0f}% → test)")
print(f"  Ordering users by signup_ts (ascending)...")

_sorted_uft = _uft_with_signup.sort_values("signup_ts", ascending=True).reset_index(drop=True)
_n_total    = len(_sorted_uft)
_n_train    = int(np.floor(_n_total * TRAIN_RATIO))
_n_test     = _n_total - _n_train

_train_df = _sorted_uft.iloc[:_n_train].copy()
_test_df  = _sorted_uft.iloc[_n_train:].copy()

# Collect distinct_ids for each split (used in scaler assertion)
_train_ids = set(_train_df[USER_COL].values)
_test_ids  = set(_test_df[USER_COL].values)
_overlap_ids = _train_ids & _test_ids
assert len(_overlap_ids) == 0, f"User ID overlap between train/test: {len(_overlap_ids)} users"
print(f"  ✅ No user ID overlap between train and test splits")

# ── Cohort date ranges ────────────────────────────────────────────────────────
_train_ts_min = _train_df["signup_ts"].min()
_train_ts_max = _train_df["signup_ts"].max()
_test_ts_min  = _test_df["signup_ts"].min()
_test_ts_max  = _test_df["signup_ts"].max()

print(f"\n  SPLIT SIZES:")
print(f"  ┌─────────────────────────────────────────────────────────────┐")
print(f"  │  Split   │   N    │  Retained  │  Churned   │  Cohort range │")
print(f"  ├─────────────────────────────────────────────────────────────┤")

_train_ret  = int(_train_df[LABEL_COL].sum())
_train_churn = _n_train - _train_ret
_test_ret   = int(_test_df[LABEL_COL].sum())
_test_churn = _n_test - _test_ret

print(f"  │  Train   │ {_n_train:>6,} │  {_train_ret:>5,} ({_train_ret/_n_train*100:4.1f}%)│  {_train_churn:>5,} ({_train_churn/_n_train*100:4.1f}%)│  {str(_train_ts_min.date())} → {str(_train_ts_max.date())} │")
print(f"  │  Test    │ {_n_test:>6,} │  {_test_ret:>5,} ({_test_ret/_n_test*100:4.1f}%)│  {_test_churn:>5,} ({_test_churn/_n_test*100:4.1f}%)│  {str(_test_ts_min.date())} → {str(_test_ts_max.date())} │")
print(f"  │  TOTAL   │ {_n_total:>6,} │  {_train_ret+_test_ret:>5,} ({(_train_ret+_test_ret)/_n_total*100:4.1f}%)│  {_train_churn+_test_churn:>5,} ({(_train_churn+_test_churn)/_n_total*100:4.1f}%)│  all cohorts                    │")
print(f"  └─────────────────────────────────────────────────────────────┘")

# ── [5] PREPARE X / y ARRAYS ─────────────────────────────────────────────────
print(f"\n[5] Preparing feature matrices")
_X_train_raw = _train_df[FEATURE_COLS].values.astype(float)
_X_test_raw  = _test_df[FEATURE_COLS].values.astype(float)
y_train      = _train_df[LABEL_COL].values.astype(int)
y_test       = _test_df[LABEL_COL].values.astype(int)

print(f"  X_train : {_X_train_raw.shape}  (n_retained={y_train.sum()}, n_churned={(1-y_train).sum()})")
print(f"  X_test  : {_X_test_raw.shape}  (n_retained={y_test.sum()}, n_churned={(1-y_test).sum()})")

# ── [6] FIT SCALER ON X_TRAIN ONLY ───────────────────────────────────────────
print(f"\n[6] StandardScaler.fit() on X_train exclusively")
scaler = StandardScaler()
X_train = scaler.fit_transform(_X_train_raw)   # fit + transform train
X_test  = scaler.transform(_X_test_raw)        # transform ONLY (no fit)

print(f"  Scaler fit on {len(_X_train_raw):,} train rows")
print(f"  Per-feature means  (from train): {np.round(scaler.mean_, 4)}")
print(f"  Per-feature std    (from train): {np.round(np.sqrt(scaler.var_), 4)}")

# ── [7] ASSERT SCALER WAS NEVER FIT ON TEST ROWS ─────────────────────────────
print(f"\n[7] Scaler integrity assertion")
# The scaler.mean_ was computed on X_train_raw. Verify it does NOT match
# what we would get if we had mistakenly fitted on the full dataset or on X_test.
_test_only_mean  = _X_test_raw.mean(axis=0)
_full_mean       = np.vstack([_X_train_raw, _X_test_raw]).mean(axis=0)

_diff_vs_full    = np.abs(scaler.mean_ - _full_mean)
_diff_vs_test    = np.abs(scaler.mean_ - _test_only_mean)

# If scaler were fit on full dataset, diff_vs_full would be ~0; assert it's not
# (unless train == full, which can't happen since test exists)
assert np.any(_diff_vs_full > 1e-9), \
    "SCALER INTEGRITY VIOLATION: scaler mean matches full-dataset mean — scaler was fit on full dataset!"
assert np.any(_diff_vs_test > 1e-9), \
    "SCALER INTEGRITY VIOLATION: scaler mean matches test-only mean — scaler was fit on test data!"

# Cross-check: train-only mean should match scaler.mean_ exactly
_diff_vs_train = np.abs(scaler.mean_ - _X_train_raw.mean(axis=0))
assert np.all(_diff_vs_train < 1e-8), \
    f"Unexpected mismatch between scaler.mean_ and X_train mean: {_diff_vs_train}"

print("  ✅ CONFIRMED: scaler.mean_ matches X_train mean exactly")
print("  ✅ CONFIRMED: scaler.mean_ differs from full-dataset mean (test rows not used)")
print("  ✅ CONFIRMED: scaler.mean_ differs from X_test-only mean")
print(f"     |train_mean - full_mean| max = {_diff_vs_full.max():.6f}  (non-zero ✓)")

# ── [8] BUILD user_feature_scaled FOR DOWNSTREAM COMPATIBILITY ───────────────
# XGBoost Retention Classifier expects 'user_feature_scaled' DataFrame
# with scaled feature columns + is_retained column.
# We rebuild a full scaled table (train + test users, in original order) so that
# downstream blocks can do their own split if needed, or we can simply pass
# through the correct split artefacts.
print(f"\n[8] Building user_feature_scaled (scaled feature table, original row order)")

_all_sorted_X_raw = np.vstack([_X_train_raw, _X_test_raw])
_all_sorted_X_scaled = scaler.transform(_all_sorted_X_raw)  # transform only

# Map scaled feature names to log_* convention used by downstream GBM block
_scaled_col_names = [
    "first_24h_events",   # keep same names (they are already non-leaky)
    "first_week_events",
    "consistency_score",
    "unique_tools_used_14d",
    "agent_usage_ratio_14d",
    "exploration_index_14d",
]
_all_sorted_labels = np.concatenate([y_train, y_test])

user_feature_scaled = pd.DataFrame(
    _all_sorted_X_scaled,
    columns=_scaled_col_names
)
user_feature_scaled.insert(0, USER_COL, pd.concat([
    _train_df[USER_COL].reset_index(drop=True),
    _test_df[USER_COL].reset_index(drop=True)
]).values)
user_feature_scaled[LABEL_COL] = _all_sorted_labels

print(f"  user_feature_scaled shape : {user_feature_scaled.shape}")
print(f"  Columns : {list(user_feature_scaled.columns)}")

# ── [9] PERSIST SCALER ────────────────────────────────────────────────────────
with open("scaler.pkl", "wb") as _f:
    pickle.dump(scaler, _f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"\n  📦 scaler.pkl persisted to working directory (train-only fit)")

# ── [10] SUMMARY ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  PHASE 3 — SPLIT & SCALING SUMMARY")
print(SEP)
print(f"\n  Temporal Cohort Split:")
print(f"    Total users  : {_n_total:,}")
print(f"    Train (old)  : {_n_train:,} users  ({_n_train/_n_total*100:.1f}%)  →  retained: {_train_ret:,}  churned: {_train_churn:,}")
print(f"    Test  (new)  : {_n_test:,}  users  ({_n_test/_n_total*100:.1f}%)  →  retained: {_test_ret:,}   churned: {_test_churn:,}")
print(f"    Train signup cohort: {_train_ts_min.date()} → {_train_ts_max.date()}")
print(f"    Test  signup cohort: {_test_ts_min.date()} → {_test_ts_max.date()}")
print(f"\n  Scaler (StandardScaler):")
print(f"    fit() called on  : X_train ({_n_train:,} rows) — ✅ CORRECT")
print(f"    transform() only : X_test  ({_n_test:,} rows)  — ✅ CORRECT")
print(f"    Scaler fit on test rows : ✅ CONFIRMED NO")
print(f"\n  Output variables:")
print(f"    X_train       : {X_train.shape}")
print(f"    X_test        : {X_test.shape}")
print(f"    y_train       : {y_train.shape}  (pos={y_train.sum()}, neg={(1-y_train).sum()})")
print(f"    y_test        : {y_test.shape}  (pos={y_test.sum()}, neg={(1-y_test).sum()})")
print(f"    scaler        : StandardScaler (fit on X_train only)")
print(f"    user_feature_scaled : {user_feature_scaled.shape}")
