import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ══════════════════════════════════════════════════════════════════
#  BLOCK 2 — EXPLORATORY DATA ANALYSIS
#  Covers:
#    1. Missing value audit (all 107 columns)
#    2. Event type distribution (top 20)
#    3. User tenure histogram (account age in weeks)
#    4. Class imbalance ratio (retained vs churned users)
# ══════════════════════════════════════════════════════════════════

# ── Style constants ───────────────────────────────────────────────
BG          = "#1D1D20"
TEXT_PRI    = "#fbfbff"
TEXT_SEC    = "#909094"
C_BLUE      = "#A1C9F4"
C_ORANGE    = "#FFB482"
C_CORAL     = "#FF9F9B"
C_GREEN     = "#8DE5A1"
C_GOLD      = "#ffd400"
GRID_COL    = "#2e2e33"

SEP   = "═" * 64
SEP2  = "─" * 64

# Config mirrors downstream labeling logic
USER_COL   = "distinct_id"
TS_COL     = "timestamp"
ACCT_COL   = "created_at"
EVENT_COL  = "event"
WEEK_THRESH = 3       # ≥ 3 distinct ISO-weeks = retained

print(SEP)
print("  BLOCK 2 │ EDA — STRUCTURED SUMMARY")
print(SEP)

# ─────────────────────────────────────────────────────────────────
# [1] MISSING VALUE AUDIT
# ─────────────────────────────────────────────────────────────────
print("\n[1] MISSING VALUE AUDIT")
print(SEP2)

eda_df = user_retention.copy()
n_rows = len(eda_df)

_miss = eda_df.isnull().sum()
_miss_pct = (_miss / n_rows * 100).round(2)
_miss_summary = pd.DataFrame({
    "null_count": _miss,
    "null_pct":   _miss_pct
}).sort_values("null_pct", ascending=False)

# Tier breakdown
tier_zero    = (_miss_pct == 0).sum()
tier_low     = ((_miss_pct > 0) & (_miss_pct <= 20)).sum()
tier_medium  = ((_miss_pct > 20) & (_miss_pct <= 70)).sum()
tier_high    = (_miss_pct > 70).sum()

print(f"    Total columns          : {len(_miss_summary)}")
print(f"    Columns with 0% null   : {tier_zero}")
print(f"    Columns with 1–20% null: {tier_low}")
print(f"    Columns with 21–70%    : {tier_medium}")
print(f"    Columns with >70%      : {tier_high}  ← candidates for dropping")
print()
print(f"    {'Column':<60}  {'Nulls':>8}  {'%':>7}")
print(f"    {'─'*60}  {'─'*8}  {'─'*7}")
for col_name, row_data in _miss_summary.iterrows():
    if row_data["null_count"] > 0:
        print(f"    {col_name:<60}  {int(row_data['null_count']):>8,}  {row_data['null_pct']:>6.2f}%")

print(f"\n    ✔ {tier_zero} columns fully populated; {tier_high} columns >70% sparse")

# ─────────────────────────────────────────────────────────────────
# [2] EVENT TYPE DISTRIBUTION (Top 20)
# ─────────────────────────────────────────────────────────────────
print(f"\n[2] EVENT TYPE DISTRIBUTION")
print(SEP2)

_event_counts = eda_df[EVENT_COL].value_counts()
_total_events = len(eda_df)
_n_event_types = len(_event_counts)
_top20_events = _event_counts.head(20)

print(f"    Total events           : {_total_events:>10,}")
print(f"    Distinct event types   : {_n_event_types:>10,}")
print()
print(f"    {'Rank':<5}  {'Event Type':<45}  {'Count':>9}  {'%':>7}")
print(f"    {'─'*5}  {'─'*45}  {'─'*9}  {'─'*7}")
for _rank, (_ev, _ct) in enumerate(_top20_events.items(), 1):
    _pct = _ct / _total_events * 100
    print(f"    {_rank:<5}  {str(_ev):<45}  {_ct:>9,}  {_pct:>6.2f}%")
print(f"    ... ({_n_event_types - 20} more event types not shown)")

# Chart — Event Distribution
fig_events, ax_events = plt.subplots(figsize=(12, 7), facecolor=BG)
ax_events.set_facecolor(BG)

_labels_ev = [str(e)[:35] + "…" if len(str(e)) > 35 else str(e) for e in _top20_events.index]
_pcts_ev   = (_top20_events.values / _total_events * 100)
_colors_ev = [C_BLUE if i % 2 == 0 else C_ORANGE for i in range(len(_labels_ev))]

_bars = ax_events.barh(range(len(_labels_ev)), _pcts_ev, color=_colors_ev, height=0.7)
ax_events.set_yticks(range(len(_labels_ev)))
ax_events.set_yticklabels(_labels_ev, color=TEXT_PRI, fontsize=9)
ax_events.invert_yaxis()
ax_events.set_xlabel("% of Total Events", color=TEXT_SEC, fontsize=10)
ax_events.set_title("Top 20 Event Types by Frequency", color=TEXT_PRI, fontsize=13, pad=14)
ax_events.tick_params(colors=TEXT_SEC, labelcolor=TEXT_PRI)
for spine in ax_events.spines.values():
    spine.set_edgecolor(GRID_COL)
ax_events.xaxis.grid(True, color=GRID_COL, linestyle="--", alpha=0.5)
ax_events.set_axisbelow(True)

# Add value labels on bars
for _bar_obj, _pct_val in zip(_bars, _pcts_ev):
    _bw = _bar_obj.get_width()
    ax_events.text(_bw + 0.1, _bar_obj.get_y() + _bar_obj.get_height() / 2,
                   f"{_pct_val:.1f}%", va="center", ha="left",
                   color=TEXT_PRI, fontsize=8)

plt.tight_layout()
plt.savefig("event_distribution.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()

# ─────────────────────────────────────────────────────────────────
# [3] USER TENURE HISTOGRAM
#     Tenure = days between account created_at and last event
# ─────────────────────────────────────────────────────────────────
print(f"\n[3] USER TENURE (Account Age at Last Event)")
print(SEP2)

_user_first_created = eda_df.groupby(USER_COL)[ACCT_COL].min()
_user_last_event    = eda_df.groupby(USER_COL)[TS_COL].max()
_user_tenure_days   = (_user_last_event - _user_first_created).dt.days.clip(lower=0)
_user_tenure_weeks  = (_user_tenure_days / 7).round(1)

_p25, _p50, _p75 = np.percentile(_user_tenure_days.dropna(), [25, 50, 75])
print(f"    Users with tenure data : {_user_tenure_days.notna().sum():,}")
print(f"    Min tenure             : {_user_tenure_days.min():.0f} days")
print(f"    P25 tenure             : {_p25:.0f} days")
print(f"    Median tenure          : {_p50:.0f} days")
print(f"    P75 tenure             : {_p75:.0f} days")
print(f"    Max tenure             : {_user_tenure_days.max():.0f} days")
print(f"    Mean tenure            : {_user_tenure_days.mean():.1f} days")

# Chart — User Tenure Histogram
fig_tenure, ax_tenure = plt.subplots(figsize=(11, 6), facecolor=BG)
ax_tenure.set_facecolor(BG)

_tenure_vals = _user_tenure_days.dropna().values
ax_tenure.hist(_tenure_vals, bins=40, color=C_BLUE, edgecolor=BG, alpha=0.9)

# Percentile lines
for _pv, _plabel, _pcol in [(_p25, "P25", C_ORANGE), (_p50, "Median", C_GOLD), (_p75, "P75", C_CORAL)]:
    ax_tenure.axvline(_pv, color=_pcol, linewidth=1.6, linestyle="--", label=f"{_plabel}: {_pv:.0f}d")

ax_tenure.set_xlabel("Account Age in Days (at last event)", color=TEXT_SEC, fontsize=10)
ax_tenure.set_ylabel("Number of Users", color=TEXT_SEC, fontsize=10)
ax_tenure.set_title("User Tenure Distribution", color=TEXT_PRI, fontsize=13, pad=14)
ax_tenure.tick_params(colors=TEXT_SEC, labelcolor=TEXT_PRI)
for spine in ax_tenure.spines.values():
    spine.set_edgecolor(GRID_COL)
ax_tenure.yaxis.grid(True, color=GRID_COL, linestyle="--", alpha=0.5)
ax_tenure.set_axisbelow(True)
_leg = ax_tenure.legend(facecolor="#2a2a2e", edgecolor=GRID_COL, labelcolor=TEXT_PRI, fontsize=9)

plt.tight_layout()
plt.savefig("user_tenure_histogram.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()

# ─────────────────────────────────────────────────────────────────
# [4] CLASS IMBALANCE RATIO
#     Mirrors downstream labeling: ≥ WEEK_THRESH distinct ISO-weeks = retained
# ─────────────────────────────────────────────────────────────────
print(f"\n[4] CLASS IMBALANCE RATIO  (label threshold: ≥{WEEK_THRESH} distinct ISO-weeks)")
print(SEP2)

_week_str = eda_df[TS_COL].dt.isocalendar().apply(
    lambda r: f"{int(r['year'])}-W{int(r['week']):02d}", axis=1
)
_weeks_per_user = (
    eda_df.assign(_iso_week=_week_str)
    .groupby(USER_COL)["_iso_week"]
    .nunique()
)
_label_series = (_weeks_per_user >= WEEK_THRESH).astype(int)

_n_total   = len(_label_series)
_n_retained = int((_label_series == 1).sum())
_n_churned  = int((_label_series == 0).sum())
_pct_retained = _n_retained / _n_total * 100
_pct_churned  = _n_churned  / _n_total * 100
_imbalance_ratio = _n_churned / max(_n_retained, 1)

assert _n_total == _n_retained + _n_churned, "Label counts don't sum to total!"

print(f"\n    Labeling rule  : ≥ {WEEK_THRESH} distinct ISO-weeks of activity → is_retained = 1")
print(f"    Total users    : {_n_total:>7,}")
print()
print(f"    {'Label':<20}  {'Users':>8}  {'%':>8}  {'Imbalance Ratio':>16}")
print(f"    {'─'*20}  {'─'*8}  {'─'*8}  {'─'*16}")
print(f"    {'0 (Churned)':<20}  {_n_churned:>8,}  {_pct_churned:>7.2f}%  {'—':>16}")
print(f"    {'1 (Retained)':<20}  {_n_retained:>8,}  {_pct_retained:>7.2f}%  {_imbalance_ratio:>14.1f}:1")
print(f"    {'─'*20}  {'─'*8}  {'─'*8}  {'─'*16}")
print(f"    {'TOTAL':<20}  {_n_total:>8,}  {'100.00%':>8}")
print()
print(f"    ⚠  CLASS IMBALANCE RATIO  →  {_imbalance_ratio:.1f}:1  (churned : retained)")
print(f"       {_pct_churned:.2f}% of users are churned  |  only {_pct_retained:.2f}% are retained")
print(f"       → Recommend class_weight='balanced' or SMOTE in downstream modelling")

print(f"\n{SEP}")
print("  BLOCK 2 — EDA COMPLETE ✔")
print(SEP)
