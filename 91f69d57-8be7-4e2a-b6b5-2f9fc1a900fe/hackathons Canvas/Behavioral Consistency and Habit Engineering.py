
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — BLOCK 07: FEATURE VALIDATION & CORRELATION
#
#  Computes Pearson + Spearman correlation matrices on the 6 model features,
#  flags any feature pair with |r| > 0.75 as multicollinear, and computes
#  Variance Inflation Factor (VIF) scores for all 6 features via OLS R²
#  (VIF_j = 1 / (1 - R²_j) where R²_j is from regressing feature j on
#  all other features).
# ══════════════════════════════════════════════════════════════════════════════

# ── Constants ─────────────────────────────────────────────────────────────────
MULTICOLLINEAR_THRESHOLD = 0.75
VIF_HIGH_THRESHOLD       = 5.0
VIF_SEVERE_THRESHOLD     = 10.0
FEATURE_COLS = [
    "first_24h_events",
    "first_week_events",
    "consistency_score",
    "unique_tools_used_14d",
    "agent_usage_ratio_14d",
    "exploration_index_14d",
]

BG         = "#1D1D20"
TEXT_PRI   = "#fbfbff"
TEXT_SEC   = "#909094"
C_BLUE     = "#A1C9F4"
C_ORANGE   = "#FFB482"
C_CORAL    = "#FF9F9B"
GRID_COL   = "#2e2e33"
SEP = "═" * 68

# ── [1] VALIDATE FEATURES ─────────────────────────────────────────────────────
print(SEP)
print("  PHASE 3 — FEATURE VALIDATION & CORRELATION ANALYSIS")
print(SEP)
print(f"\n[1] Feature presence check on user_feature_table {user_feature_table.shape}")
_missing_cols = [f for f in FEATURE_COLS if f not in user_feature_table.columns]
assert not _missing_cols, f"Missing feature columns: {_missing_cols}"
print(f"  ✅ All 6 features present")

_feat_df = user_feature_table[FEATURE_COLS].copy()
_null_counts = _feat_df.isnull().sum()
assert not _null_counts.any(), f"NaNs found:\n{_null_counts[_null_counts > 0]}"
print(f"  ✅ Zero NaNs in all 6 feature columns")
print(f"  Dataset: {len(_feat_df):,} users")

# ── [2] PEARSON CORRELATION MATRIX ───────────────────────────────────────────
print(f"\n[2] Pearson Correlation Matrix (linear, n={len(_feat_df):,})")
print("─" * 68)
pearson_corr = _feat_df.corr(method="pearson")
print(pearson_corr.round(4).to_string())

# ── [3] SPEARMAN CORRELATION MATRIX ──────────────────────────────────────────
print(f"\n[3] Spearman Correlation Matrix (rank-based, robust to outliers)")
print("─" * 68)
spearman_corr = _feat_df.corr(method="spearman")
print(spearman_corr.round(4).to_string())

# ── [4] MULTICOLLINEARITY FLAGS ───────────────────────────────────────────────
print(f"\n[4] Multicollinearity Flags  (|Pearson r| > {MULTICOLLINEAR_THRESHOLD})")
print("─" * 68)
_high_pearson = []
for _i, fi in enumerate(FEATURE_COLS):
    for _j, fj in enumerate(FEATURE_COLS):
        if _j <= _i:
            continue
        r_val = pearson_corr.loc[fi, fj]
        if abs(r_val) > MULTICOLLINEAR_THRESHOLD:
            _high_pearson.append((fi, fj, r_val))
            print(f"  ⚠️  MULTICOLLINEAR  {fi} ↔ {fj}   Pearson r = {r_val:.4f}")

if not _high_pearson:
    print(f"  ✅ No Pearson pair exceeds |r| = {MULTICOLLINEAR_THRESHOLD}")
else:
    print(f"\n  {len(_high_pearson)} multicollinear Pearson pair(s) flagged")

print(f"\n  Spearman |ρ| > {MULTICOLLINEAR_THRESHOLD}:")
_high_spearman = []
for _i, fi in enumerate(FEATURE_COLS):
    for _j, fj in enumerate(FEATURE_COLS):
        if _j <= _i:
            continue
        rho_val = spearman_corr.loc[fi, fj]
        if abs(rho_val) > MULTICOLLINEAR_THRESHOLD:
            _high_spearman.append((fi, fj, rho_val))
            print(f"  ⚠️  MULTICOLLINEAR  {fi} ↔ {fj}   Spearman ρ = {rho_val:.4f}")

if not _high_spearman:
    print(f"  ✅ No Spearman pair exceeds |ρ| = {MULTICOLLINEAR_THRESHOLD}")
else:
    print(f"\n  {len(_high_spearman)} multicollinear Spearman pair(s) flagged")

# ── [5] VIF SCORES — computed via OLS R² ─────────────────────────────────────
print(f"\n[5] Variance Inflation Factor (VIF) — all 6 features")
print("─" * 68)
print("  VIF interpretation:")
print("  VIF = 1        → no collinearity with other features")
print("  VIF 1–5        → acceptable (moderate correlation OK)")
print("  VIF > 5        → ⚠️  warning: high multicollinearity")
print("  VIF > 10       → ❌ severe: consider dropping/combining")
print()

_X_mat = _feat_df.values.astype(float)
_vif_records = []

for _idx, _col in enumerate(FEATURE_COLS):
    _y_vif = _X_mat[:, _idx]
    _X_others = np.delete(_X_mat, _idx, axis=1)
    _X_with_int = np.column_stack([np.ones(len(_y_vif)), _X_others])
    _coefs, _, _, _ = np.linalg.lstsq(_X_with_int, _y_vif, rcond=None)
    _y_hat = _X_with_int @ _coefs
    _ss_res = np.sum((_y_vif - _y_hat) ** 2)
    _ss_tot = np.sum((_y_vif - _y_vif.mean()) ** 2)
    _r2 = 1.0 - _ss_res / _ss_tot if _ss_tot > 0 else 0.0
    _r2 = min(_r2, 0.9999999)
    _vif_val = 1.0 / (1.0 - _r2)
    _flag = "❌ SEVERE" if _vif_val > VIF_SEVERE_THRESHOLD else (
        "⚠️  HIGH" if _vif_val > VIF_HIGH_THRESHOLD else "✅  OK"
    )
    _vif_records.append({"feature": _col, "VIF": round(_vif_val, 4), "R2": round(_r2, 4), "status": _flag})

vif_df = pd.DataFrame(_vif_records)
print(vif_df.to_string(index=False))

_max_vif   = vif_df["VIF"].max()
_max_feat  = vif_df.loc[vif_df["VIF"].idxmax(), "feature"]
print(f"\n  Highest VIF : {_max_feat} = {_max_vif:.4f}")
print(f"  Average VIF : {vif_df['VIF'].mean():.4f}")

# ── [6] DUAL CORRELATION HEATMAP ─────────────────────────────────────────────
fig_feat_corr, (ax_p, ax_s) = plt.subplots(1, 2, figsize=(16, 7))
fig_feat_corr.patch.set_facecolor(BG)

_axis_labels = [
    "24h Events",
    "Week 1\nEvents",
    "Consistency",
    "Unique\nTools",
    "Agent\nRatio",
    "Exploration",
]

for ax, _cmat, _method in [(ax_p, pearson_corr, "Pearson r"), (ax_s, spearman_corr, "Spearman ρ")]:
    ax.set_facecolor(BG)
    _annot_arr = np.empty_like(_cmat.values, dtype=object)
    for _i in range(len(FEATURE_COLS)):
        for _j in range(len(FEATURE_COLS)):
            _val = _cmat.values[_i, _j]
            _txt = f"{_val:.2f}"
            if _i != _j and abs(_val) > MULTICOLLINEAR_THRESHOLD:
                _txt += "\n⚠️"
            _annot_arr[_i, _j] = _txt

    sns.heatmap(
        _cmat,
        ax=ax,
        annot=_annot_arr,
        fmt="",
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        linecolor=BG,
        annot_kws={"size": 9, "color": TEXT_PRI, "fontweight": "bold"},
        cbar_kws={"shrink": 0.75, "pad": 0.02},
        xticklabels=_axis_labels,
        yticklabels=_axis_labels,
    )
    _cbar = ax.collections[0].colorbar
    _cbar.ax.yaxis.set_tick_params(color=TEXT_SEC, labelcolor=TEXT_SEC)
    _cbar.outline.set_edgecolor(GRID_COL)
    _cbar.set_label(_method, color=TEXT_SEC, fontsize=10)
    ax.tick_params(axis="both", colors=TEXT_PRI, labelsize=9, length=0)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", color=TEXT_PRI, fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", color=TEXT_PRI, fontsize=9)
    for _sp in ax.spines.values():
        _sp.set_visible(False)
    ax.set_title(
        f"PHASE 3 — {_method} Correlation Matrix\n(⚠️  = |r| > {MULTICOLLINEAR_THRESHOLD})",
        color=TEXT_PRI, fontsize=12, fontweight="bold", pad=14,
    )

plt.tight_layout()
plt.show()

# ── [7] VIF BAR CHART ────────────────────────────────────────────────────────
fig_vif, ax_vif = plt.subplots(figsize=(9, 5))
fig_vif.patch.set_facecolor(BG)
ax_vif.set_facecolor(BG)

_vif_colors = [
    "#f04438" if v > VIF_SEVERE_THRESHOLD else ("FFB482" if v > VIF_HIGH_THRESHOLD else C_BLUE)
    for v in vif_df["VIF"]
]
_bars_vif = ax_vif.barh(
    vif_df["feature"], vif_df["VIF"],
    color=_vif_colors, edgecolor="none", height=0.6,
)
for _bar, _vif_val in zip(_bars_vif, vif_df["VIF"]):
    ax_vif.text(
        _vif_val + 0.08,
        _bar.get_y() + _bar.get_height() / 2,
        f"{_vif_val:.2f}",
        va="center", ha="left", color=TEXT_PRI, fontsize=10,
    )
ax_vif.axvline(x=5,  color=C_ORANGE, linestyle="--", lw=1.2, alpha=0.8, label="VIF = 5 (warn)")
ax_vif.axvline(x=10, color="#f04438", linestyle="--", lw=1.2, alpha=0.8, label="VIF = 10 (severe)")
ax_vif.set_xlabel("VIF Score", color=TEXT_PRI, fontsize=11, labelpad=8)
ax_vif.set_title(
    "PHASE 3 — Variance Inflation Factor (VIF) — 6 Model Features",
    color=TEXT_PRI, fontsize=13, fontweight="bold", pad=14,
)
ax_vif.tick_params(colors=TEXT_PRI, labelsize=10)
for _sp in ax_vif.spines.values():
    _sp.set_edgecolor(GRID_COL)
ax_vif.set_xlim(0, max(vif_df["VIF"].max() * 1.25, 12))
ax_vif.grid(axis="x", color=GRID_COL, lw=0.7, alpha=0.6)
ax_vif.legend(facecolor=BG, edgecolor=GRID_COL, labelcolor=TEXT_PRI, fontsize=9)
plt.tight_layout()
plt.show()

# ── [8] SUMMARY ──────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  PHASE 3 — VALIDATION SUMMARY")
print(SEP)
print(f"\n  user_feature_table shape     : {user_feature_table.shape}")
print(f"  Features validated           : {len(FEATURE_COLS)}")
print(f"\n  Multicollinear pairs (Pearson |r| > {MULTICOLLINEAR_THRESHOLD})  : {len(_high_pearson)}")
print(f"  Multicollinear pairs (Spearman|ρ| > {MULTICOLLINEAR_THRESHOLD}) : {len(_high_spearman)}")
print(f"\n  VIF scores — all 6 features:")
for _, _vif_row in vif_df.iterrows():
    print(f"    {_vif_row['feature']:<30}  VIF = {_vif_row['VIF']:>8.4f}   {_vif_row['status']}")
print(f"\n  Max VIF  : {_max_feat} = {_max_vif:.4f}")
print(f"  Avg VIF  : {vif_df['VIF'].mean():.4f}")
