import pandas as pd

# ══════════════════════════════════════════════════════════════════
#  BLOCK 1 — RAW DATA LOAD
#  Loads user_retention.parquet and asserts schema, dtypes,
#  row counts, and event date ranges.
# ══════════════════════════════════════════════════════════════════

EXPECTED_ROWS     = 409_287
EXPECTED_COLS     = 107
USER_ID_COL       = "distinct_id"
TIMESTAMP_COL     = "timestamp"
EVENT_COL         = "event"
ACCOUNT_TS_COL    = "created_at"

# Required columns and their expected pandas dtype kinds
# (kind: 'O'=object/str, 'M'=datetime64, 'i'/'u'=int, 'f'=float)
REQUIRED_SCHEMA = {
    "distinct_id":  "O",
    "person_id":    "O",
    "event":        "O",
    "timestamp":    "M",
    "created_at":   "M",
    "uuid":         "O",
}

SEP = "═" * 62

# ── Load ──────────────────────────────────────────────────────────
user_retention = pd.read_parquet("user_retention.parquet")

print(SEP)
print("  BLOCK 1 │ RAW DATA LOAD — STRUCTURED SUMMARY")
print(SEP)

# ── 1. Row & Column Count Assertions ─────────────────────────────
actual_rows, actual_cols = user_retention.shape
print(f"\n[1] SHAPE CHECK")
print(f"    Expected : {EXPECTED_ROWS:>10,} rows  ×  {EXPECTED_COLS} cols")
print(f"    Actual   : {actual_rows:>10,} rows  ×  {actual_cols} cols")

assert actual_rows == EXPECTED_ROWS, (
    f"Row count mismatch: expected {EXPECTED_ROWS}, got {actual_rows}"
)
assert actual_cols == EXPECTED_COLS, (
    f"Column count mismatch: expected {EXPECTED_COLS}, got {actual_cols}"
)
print("    ✔ Shape assertion passed")

# ── 2. Required Column Presence ───────────────────────────────────
print(f"\n[2] REQUIRED COLUMN PRESENCE")
missing_required = [c for c in REQUIRED_SCHEMA if c not in user_retention.columns]
if missing_required:
    raise AssertionError(f"Missing required columns: {missing_required}")
print(f"    All {len(REQUIRED_SCHEMA)} required columns present  ✔")

# ── 3. Dtype Assertions ───────────────────────────────────────────
print(f"\n[3] DTYPE ASSERTIONS")
print(f"    {'Column':<40}  {'Expected Kind':<14}  {'Actual Dtype':<20}  Status")
print(f"    {'─'*40}  {'─'*14}  {'─'*20}  ──────")
dtype_failures = []
for col_name, expected_kind in REQUIRED_SCHEMA.items():
    actual_kind = user_retention[col_name].dtype.kind
    actual_dtype = str(user_retention[col_name].dtype)
    status = "✔" if actual_kind == expected_kind else "✘ FAIL"
    if actual_kind != expected_kind:
        dtype_failures.append((col_name, expected_kind, actual_kind))
    print(f"    {col_name:<40}  {expected_kind!r:<14}  {actual_dtype:<20}  {status}")

if dtype_failures:
    for col_name, exp, got in dtype_failures:
        print(f"    ⚠  {col_name}: expected kind '{exp}', got '{got}'")
    raise AssertionError(f"Dtype assertion failed for: {[c[0] for c in dtype_failures]}")
print("    ✔ All dtype assertions passed")

# ── 4. Event Date Range ───────────────────────────────────────────
print(f"\n[4] EVENT DATE RANGE  (column: '{TIMESTAMP_COL}')")
ts_min = user_retention[TIMESTAMP_COL].min()
ts_max = user_retention[TIMESTAMP_COL].max()
ts_span_days = (ts_max - ts_min).days
print(f"    Earliest event : {ts_min}")
print(f"    Latest event   : {ts_max}")
print(f"    Span           : {ts_span_days} days  (~{ts_span_days / 7:.1f} weeks)")
assert pd.notna(ts_min) and pd.notna(ts_max), "Timestamp column contains all-null values!"
assert ts_min < ts_max, "Timestamp min ≥ max — data ordering issue!"
print("    ✔ Date range assertion passed")

# ── 5. Account Creation Date Range ───────────────────────────────
print(f"\n[5] ACCOUNT CREATION DATE RANGE  (column: '{ACCOUNT_TS_COL}')")
ca_min = user_retention[ACCOUNT_TS_COL].min()
ca_max = user_retention[ACCOUNT_TS_COL].max()
print(f"    Earliest account created : {ca_min}")
print(f"    Latest account created   : {ca_max}")

# ── 6. Unique User & Event Counts ────────────────────────────────
print(f"\n[6] DISTINCT COUNTS")
n_users  = user_retention[USER_ID_COL].nunique()
n_events = user_retention[EVENT_COL].nunique()
print(f"    Distinct users  ({USER_ID_COL})  : {n_users:>7,}")
print(f"    Distinct event types ({EVENT_COL}) : {n_events:>7,}")

# ── 7. All Columns & Dtypes ───────────────────────────────────────
print(f"\n[7] ALL COLUMNS ({actual_cols} total)")
print(f"    {'#':<4}  {'Column':<60}  Dtype")
print(f"    {'─'*4}  {'─'*60}  ─────────────")
for idx, (c, dt) in enumerate(user_retention.dtypes.items(), 1):
    print(f"    {idx:<4}  {c:<60}  {dt}")

print(f"\n{SEP}")
print("  BLOCK 1 — LOAD COMPLETE ✔")
print(SEP)
