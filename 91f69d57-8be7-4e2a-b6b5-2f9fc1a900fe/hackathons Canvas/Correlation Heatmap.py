
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import chi2_contingency, fisher_exact

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — TOOL USAGE EDA  (feature window only, days 1–14)
#
#  Source : feature_events  (days 1–14 after signup — NO LEAKAGE)
#  Label  : retention_label (from label_events days 15–90 — no leakage)
#
#  Outputs:
#   1. Grouped bar chart — tool event frequency by retention group
#   2. Statistical significance per tool — chi-squared or Fisher's exact
#      (Fisher's exact used when any expected cell count < 5)
# ══════════════════════════════════════════════════════════════════════════════

TOOL_COL  = "prop_tool_name"
USER_COL  = "distinct_id"
LABEL_COL = "is_retained"
TOP_N     = 8   # top N tools by total event count

# Zerve design system
BG_COLOR        = "#1D1D20"
TEXT_PRIMARY    = "#fbfbff"
TEXT_SEC        = "#909094"
COLOR_RETAINED  = "#8DE5A1"
COLOR_CHURNED   = "#FF9F9B"
GRID_COLOR      = "#2e2e34"

# ── 1. Attach retention label to feature_events ────────────────────────────────
_fe_with_label = feature_events.copy()
_fe_with_label[LABEL_COL] = _fe_with_label[USER_COL].map(retention_label).fillna(0).astype(int)

print(f"feature_events shape       : {feature_events.shape[0]:,} rows")
print(f"Users in feature window    : {_fe_with_label[USER_COL].nunique():,}")
print(f"Retained users (label=1)   : {int((retention_label == 1).sum())}")
print(f"Churned users  (label=0)   : {int((retention_label == 0).sum())}")

# ── 2. Filter to tool events only ─────────────────────────────────────────────
_tool_fe = _fe_with_label.dropna(subset=[TOOL_COL])
print(f"\nTool events in feature window: {len(_tool_fe):,}  "
      f"({len(_tool_fe)/len(_fe_with_label)*100:.1f}% of feature events)")

# ── 3. Aggregate: event counts per (retention_group, tool) ────────────────────
_tool_counts = (
    _tool_fe
    .groupby([LABEL_COL, TOOL_COL])
    .size()
    .rename("event_count")
    .reset_index()
)

_total_by_tool = _tool_counts.groupby(TOOL_COL)["event_count"].sum()
_top_tools     = _total_by_tool.nlargest(TOP_N).index.tolist()
_top_counts    = _tool_counts[_tool_counts[TOOL_COL].isin(_top_tools)]

_pivot = (
    _top_counts
    .pivot_table(index=TOOL_COL, columns=LABEL_COL, values="event_count", aggfunc="sum")
    .fillna(0)
    .astype(int)
    .rename(columns={0: "Churned", 1: "Retained"})
)
for _col in ["Churned", "Retained"]:
    if _col not in _pivot.columns:
        _pivot[_col] = 0

_pivot = (_pivot
    .assign(_total=_pivot.sum(axis=1))
    .sort_values("_total", ascending=False)
    .drop(columns="_total")
)

print("\n" + "═" * 65)
print(f"PHASE 2 — TOP {TOP_N} TOOL FREQUENCIES  (feature window, days 1–14)")
print("═" * 65)
print(_pivot.to_string())

# ── 4. Statistical significance per tool ──────────────────────────────────────
# User-level 2×2 contingency: used_tool × retained
# • χ² (with Yates) when expected counts ≥ 5 in all cells
# • Fisher's exact when any expected count < 5 (handles sparse class imbalance)

_user_tool_set = (
    _tool_fe
    .groupby(TOOL_COL)[USER_COL]
    .apply(set)
    .to_dict()
)

_all_users = (
    _fe_with_label[[USER_COL, LABEL_COL]]
    .drop_duplicates(subset=[USER_COL])
    .set_index(USER_COL)
)

print("\n" + "─" * 70)
print(f"  {'Tool':<30}  {'Test':<8}  {'Stat':>8}  {'p-value':>10}  Sig?")
print("─" * 70)

_chi2_results = []
for _tool in _pivot.index:
    _users_for_tool = _user_tool_set.get(_tool, set())
    _used_flag  = _all_users.index.isin(_users_for_tool).astype(int)
    _label_vals = _all_users[LABEL_COL].values

    # Build 2×2 contingency manually
    _used_arr   = _used_flag.values if hasattr(_used_flag, 'values') else np.array(_used_flag)
    _a = int(np.sum((_used_arr == 1) & (_label_vals == 1)))   # used & retained
    _b = int(np.sum((_used_arr == 1) & (_label_vals == 0)))   # used & churned
    _c = int(np.sum((_used_arr == 0) & (_label_vals == 1)))   # not used & retained
    _d = int(np.sum((_used_arr == 0) & (_label_vals == 0)))   # not used & churned

    _ct = np.array([[_a, _b], [_c, _d]])
    _n  = _a + _b + _c + _d

    # Expected counts for cell (0,0): (_a+_b)*(_a+_c) / n
    _exp_min = min(
        (_a + _b) * (_a + _c) / _n,
        (_a + _b) * (_b + _d) / _n,
        (_c + _d) * (_a + _c) / _n,
        (_c + _d) * (_b + _d) / _n,
    ) if _n > 0 else 0

    if _exp_min < 5 or _b == 0 or _c == 0:
        # Use Fisher's exact test for sparse tables
        _odds, _p_val = fisher_exact(_ct, alternative="two-sided")
        _stat         = _odds
        _test_name    = "Fisher"
    else:
        _chi2_val, _p_val, _, _ = chi2_contingency(_ct, correction=True)
        _stat                   = _chi2_val
        _test_name              = "χ²"

    _sig = "✅" if _p_val < 0.05 else "  "
    _chi2_results.append({
        "tool": _tool, "test": _test_name, "statistic": _stat,
        "p_value": _p_val, "significant": _p_val < 0.05,
        "n_used_retained": _a, "n_used_churned": _b,
    })
    print(f"  {_tool:<30}  {_test_name:<8}  {_stat:>8.3f}  {_p_val:>10.4f}  {_sig}")

print("─" * 70)
print("  χ²: Yates' correction | Fisher: two-sided | α = 0.05")
_n_sig = sum(r["significant"] for r in _chi2_results)
print(f"  {_n_sig}/{TOP_N} tools show significant association with retention\n")

chi2_results_df = pd.DataFrame(_chi2_results).sort_values("p_value")
print("Sorted by p-value:")
print(chi2_results_df[["tool", "test", "p_value", "significant"]].to_string(index=False))

# ── 5. Grouped bar chart ───────────────────────────────────────────────────────
_tools   = list(_pivot.index)
_n_tools = len(_tools)
_x       = np.arange(_n_tools)
_bar_w   = 0.38

fig_tool_eda, ax_tool = plt.subplots(figsize=(13, 6.5))
fig_tool_eda.patch.set_facecolor(BG_COLOR)
ax_tool.set_facecolor(BG_COLOR)

_bars_retained = ax_tool.bar(
    _x - _bar_w / 2, _pivot["Retained"].values,
    width=_bar_w, label="Retained (≥3 active weeks in label window)",
    color=COLOR_RETAINED, zorder=3,
)
_bars_churned = ax_tool.bar(
    _x + _bar_w / 2, _pivot["Churned"].values,
    width=_bar_w, label="Churned (<3 active weeks in label window)",
    color=COLOR_CHURNED, zorder=3,
)

_max_val = max(_pivot["Retained"].max(), _pivot["Churned"].max())
_y_off   = _max_val * 0.01

for _bars, _vals, _col in zip(
    [_bars_retained, _bars_churned],
    [_pivot["Retained"].values, _pivot["Churned"].values],
    [COLOR_RETAINED, COLOR_CHURNED],
):
    for _bar, _val in zip(_bars, _vals):
        if _val > 0:
            ax_tool.text(
                _bar.get_x() + _bar.get_width() / 2, _val + _y_off,
                f"{int(_val):,}", ha="center", va="bottom",
                color=_col, fontsize=8.5, fontweight="bold",
            )

# X-tick labels with p-value annotations
_p_lookup = {r["tool"]: r["p_value"] for r in _chi2_results}
_xtick_labels = []
for _tname in _tools:
    _pv  = _p_lookup.get(_tname, 1.0)
    _star = "★" if _pv < 0.05 else ""
    _xtick_labels.append(f"{_tname}\n{_star}p={_pv:.4f}")

ax_tool.set_xticks(_x)
ax_tool.set_xticklabels(_xtick_labels, rotation=20, ha="right", color=TEXT_PRIMARY, fontsize=8.5)
ax_tool.tick_params(axis="y", colors=TEXT_SEC, labelsize=9)
ax_tool.tick_params(axis="x", colors=TEXT_PRIMARY)

ax_tool.set_xlabel("Tool (prop_tool_name) — ★ = significant at α=0.05", color=TEXT_SEC, fontsize=10, labelpad=12)
ax_tool.set_ylabel("Event Count (days 1–14 feature window)", color=TEXT_SEC, fontsize=10, labelpad=10)
ax_tool.set_title(
    f"Phase 2 — Top {TOP_N} Tool Frequency by Retention Group (Feature Window Only)",
    color=TEXT_PRIMARY, fontsize=13, fontweight="bold", pad=16,
)

ax_tool.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
ax_tool.set_ylim(0, _max_val * 1.22)
ax_tool.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
ax_tool.set_axisbelow(True)
for _sp in ax_tool.spines.values():
    _sp.set_visible(False)

ax_tool.legend(
    frameon=True, framealpha=0.15, facecolor="#2e2e34",
    edgecolor="#444", labelcolor=TEXT_PRIMARY, fontsize=9.5, loc="upper right",
)

plt.tight_layout()
plt.savefig("tool_retention_grouped_bar.png", dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
plt.show()
print("\nChart saved → tool_retention_grouped_bar.png")
