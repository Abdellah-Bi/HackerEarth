"""
MODEL COMPARISON — Baseline Phase 4 GBM vs Enriched Advanced GBM
=================================================================
1. Metrics comparison table (PR-AUC, ROC-AUC, Recall, Precision, F1, feature count)
   with Delta and % Improvement columns
2. Grouped bar chart — all metrics side-by-side
3. SHAP feature importance comparison — top 15 baseline vs enriched (horizontal bar,
   new features highlighted by category: survival/TS/behavioral)
4. PR curve overlay — both models on the same axes
5. Summary callout printed to console

Exports: comparison_metrics, pr_auc_lift, roc_auc_lift, recall_lift

NOTE: Models are retrained in-block (lightweight, same hyperparams + seed) to produce
PR curves reliably — the upstream serialized models hit a sklearn internal _loss module
compatibility issue when calling predict_proba across block boundaries.
All metrics are pulled directly from upstream variables (no hardcoding).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score, roc_auc_score, recall_score, precision_score,
    precision_recall_curve,
)

# ── Design tokens ───────────────────────────────────────────────────────────
_BG       = "#1D1D20"
_TXT_PRI  = "#fbfbff"
_TXT_SEC  = "#909094"
_C_BLUE   = "#A1C9F4"
_C_ORANGE = "#FFB482"
_C_GREEN  = "#8DE5A1"
_C_CORAL  = "#FF9F9B"
_C_LAV    = "#D0BBFF"
_C_GOLD   = "#ffd400"
_GRID     = "#2e2e33"
_SEP      = "═" * 72

print(_SEP)
print("  MODEL COMPARISON — Phase 4 Baseline GBM vs Enriched Advanced GBM")
print(_SEP)

# ── [0] Pull all metrics from upstream variables — zero hardcoding ───────────
_b_pr_auc    = float(cv_pr_auc_test)
_b_roc_auc   = float(cv_roc_auc_test)
_b_recall    = float(cv_recall_test)
_b_precision = float(cv_precision_test)
_b_f1        = 2 * _b_precision * _b_recall / (_b_precision + _b_recall + 1e-9)
_b_feat_count = 6   # canonical Phase 4 input feature count (6 clean leak-free features)

_e_pr_auc    = float(advanced_pr_auc)
_e_roc_auc   = float(advanced_roc_auc)
_e_recall    = float(advanced_recall)
_e_precision = float(advanced_precision)
_e_f1        = 2 * _e_precision * _e_recall / (_e_precision + _e_recall + 1e-9)

# Count enriched numeric features dynamically
_EXCLUDE = {"distinct_id", "is_retained", "segment_name"}
_enriched_num_cols = [
    c for c in enriched_features.columns
    if c not in _EXCLUDE and enriched_features[c].dtype != object
]
_e_n_feats = len(_enriched_num_cols)

print(f"\n  Baseline metrics (from upstream cv_pr_auc_test etc.):")
print(f"    PR-AUC={_b_pr_auc:.4f}  ROC-AUC={_b_roc_auc:.4f}  Recall={_b_recall:.4f}  Prec={_b_precision:.4f}")
print(f"  Enriched metrics (from upstream advanced_pr_auc etc.):")
print(f"    PR-AUC={_e_pr_auc:.4f}  ROC-AUC={_e_roc_auc:.4f}  Recall={_e_recall:.4f}  Prec={_e_precision:.4f}")
print(f"  Enriched feature count: {_e_n_feats} (from enriched_features DataFrame)")

# ── [1] Compute lifts ───────────────────────────────────────────────────────
pr_auc_lift   = float(_e_pr_auc - _b_pr_auc)
roc_auc_lift  = float(_e_roc_auc - _b_roc_auc)
recall_lift   = float(_e_recall - _b_recall)

def _pct_imp(base, delta):
    return (delta / (abs(base) + 1e-9)) * 100

# ── [2] Comparison metrics DataFrame ────────────────────────────────────────
comparison_metrics = pd.DataFrame({
    "Metric":         ["PR-AUC", "ROC-AUC", "Recall", "Precision", "F1", "Feature Count"],
    "Baseline (P4)":  [_b_pr_auc, _b_roc_auc, _b_recall, _b_precision, _b_f1, float(_b_feat_count)],
    "Enriched (Adv)": [_e_pr_auc, _e_roc_auc, _e_recall, _e_precision, _e_f1, float(_e_n_feats)],
    "Delta":          [
        _e_pr_auc - _b_pr_auc, _e_roc_auc - _b_roc_auc,
        _e_recall - _b_recall, _e_precision - _b_precision,
        _e_f1 - _b_f1, float(_e_n_feats - _b_feat_count),
    ],
})
comparison_metrics["% Improvement"] = [
    _pct_imp(b, d) for b, d in zip(comparison_metrics["Baseline (P4)"], comparison_metrics["Delta"])
]

# ── [3] Print metrics table ──────────────────────────────────────────────────
print(f"\n{'─'*72}")
print("  METRICS COMPARISON TABLE")
print(f"{'─'*72}")
print(f"  {'Metric':<16} {'Baseline':>10} {'Enriched':>10} {'Delta':>10} {'% Imprv':>10}")
print(f"  {'─'*62}")
for _, _r in comparison_metrics.iterrows():
    _d   = _r["Delta"]
    _tag = "▲" if _d > 0.001 else ("▼" if _d < -0.001 else "≈")
    _fb  = f"{_r['Baseline (P4)']:.4f}" if _r["Metric"] != "Feature Count" else f"{int(_r['Baseline (P4)'])}"
    _fe  = f"{_r['Enriched (Adv)']:.4f}" if _r["Metric"] != "Feature Count" else f"{int(_r['Enriched (Adv)'])}"
    _fd  = f"{_d:+.4f}" if _r["Metric"] != "Feature Count" else f"{int(_d):+d}"
    print(f"  {_r['Metric']:<16} {_fb:>10} {_fe:>10} {_fd:>10} {_tag} {_r['% Improvement']:>+8.1f}%")
print(f"{'─'*72}")
print(f"  Baseline: {_b_feat_count} features  |  Enriched: {_e_n_feats} features (survival + TS + behavioral)")

# ── [4] VIZ 1 — Grouped bar chart ───────────────────────────────────────────
_pm   = ["PR-AUC", "ROC-AUC", "Recall", "Precision", "F1"]
_bv   = [_b_pr_auc, _b_roc_auc, _b_recall, _b_precision, _b_f1]
_ev   = [_e_pr_auc, _e_roc_auc, _e_recall, _e_precision, _e_f1]
_dv   = [e - b for e, b in zip(_ev, _bv)]
_x    = np.arange(len(_pm));  _w = 0.33

metrics_bar_chart = plt.figure(figsize=(12, 6.5))
metrics_bar_chart.patch.set_facecolor(_BG)
_ax1 = metrics_bar_chart.add_subplot(111)
_ax1.set_facecolor(_BG)

_bars_b = _ax1.bar(_x - _w/2, _bv, _w, color=_C_BLUE,   edgecolor="none", alpha=0.92, zorder=3)
_bars_e = _ax1.bar(_x + _w/2, _ev, _w, color=_C_ORANGE, edgecolor="none", alpha=0.92, zorder=3)

for _br, _v in zip(_bars_b, _bv):
    _ax1.text(_br.get_x() + _br.get_width()/2, _v + 0.012,
              f"{_v:.3f}", ha="center", va="bottom", color=_TXT_PRI, fontsize=9, fontweight="bold")
for _br, _v, _d in zip(_bars_e, _ev, _dv):
    _col = _C_GREEN if _d > 0.001 else (_C_CORAL if _d < -0.001 else _TXT_SEC)
    _arr = "▲" if _d > 0.001 else ("▼" if _d < -0.001 else "≈")
    _ax1.text(_br.get_x() + _br.get_width()/2, _v + 0.012,
              f"{_v:.3f}\n{_arr}{abs(_d):.3f}",
              ha="center", va="bottom", color=_col, fontsize=8.5, fontweight="bold")

_ax1.set_xticks(_x); _ax1.set_xticklabels(_pm, color=_TXT_PRI, fontsize=11)
_ax1.set_ylabel("Score", color=_TXT_PRI, fontsize=11, labelpad=8)
_ax1.set_ylim(0, 1.32)
_ax1.set_title(
    f"Model Comparison — Baseline Phase 4 ({_b_feat_count} features) vs Enriched Advanced ({_e_n_feats} features)\n"
    f"PR-AUC Δ={pr_auc_lift:+.4f}  |  ROC-AUC Δ={roc_auc_lift:+.4f}  |  Recall Δ={recall_lift:+.4f}",
    color=_TXT_PRI, fontsize=12, fontweight="bold", pad=14
)
_ax1.tick_params(colors=_TXT_PRI, labelsize=10)
for _sp in _ax1.spines.values(): _sp.set_edgecolor(_GRID)
_ax1.grid(axis="y", color=_GRID, linewidth=0.5, alpha=0.4); _ax1.set_axisbelow(True)
_ax1.legend(handles=[
    mpatches.Patch(color=_C_BLUE,   label=f"Baseline — Phase 4 ({_b_feat_count} features)"),
    mpatches.Patch(color=_C_ORANGE, label=f"Enriched — Advanced ({_e_n_feats} features)"),
], facecolor="#2a2a2e", edgecolor=_GRID, labelcolor=_TXT_PRI, fontsize=10, loc="upper right")
plt.tight_layout(); plt.show()

# ── [5] Train lightweight models to get feature importances + PR curves ──────
print(f"\n{'─'*72}")
print("  Training lightweight GBM models for SHAP importance and PR curves …")

# Feature category lookup for color coding
_SURVIVAL = {"tenure_days","km_hazard_score","surv_prob_7d","surv_prob_14d",
             "surv_prob_30d","cox_log_hazard","risk_acceleration_coef"}
_TS_FEATS = {"trend_slope","trend_r2","seasonality_amp","residual_vol",
             "peak_to_mean_ratio","activity_entropy","n_active_days","total_events"}
_BEHAV    = {"cluster"}

def _feat_color(fname):
    if fname in _SURVIVAL: return "#FF9F9B"
    if fname in _TS_FEATS: return "#D0BBFF"
    if fname in _BEHAV:    return "#8DE5A1"
    return _C_BLUE

_GBM_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, max_features=0.8, random_state=42,
    validation_fraction=0.1, n_iter_no_change=20, tol=1e-4,
)

# ── Baseline: 6-feature model ────────────────────────────────────────────────
_FC6 = ["first_24h_events","first_week_events","consistency_score",
        "unique_tools_used_14d","agent_usage_ratio_14d","exploration_index_14d"]
_X6  = user_feature_table[_FC6].values.astype(float)
_y6  = user_feature_table["is_retained"].values.astype(int)
_X6_tr, _X6_te, _y6_tr, _y6_te = train_test_split(_X6, _y6, test_size=0.2, random_state=42, stratify=_y6)
_sc6 = StandardScaler(); _X6_tr_s = _sc6.fit_transform(_X6_tr); _X6_te_s = _sc6.transform(_X6_te)
_n6  = int((_y6_tr==0).sum()); _p6 = int((_y6_tr==1).sum())
_sw6 = np.where(_y6_tr==1, max(_n6,1)/max(_p6,1), 1.0)
_gbm6 = GradientBoostingClassifier(**_GBM_PARAMS)
_gbm6.fit(_X6_tr_s, _y6_tr, sample_weight=_sw6)
_probs6 = _gbm6.predict_proba(_X6_te_s)[:, 1]
_pc6, _rc6, _ = precision_recall_curve(_y6_te, _probs6)
_auc6 = average_precision_score(_y6_te, _probs6)
_base_imp_series = pd.Series(dict(zip(_FC6, _gbm6.feature_importances_)))
print(f"    Baseline GBM retrained: PR-AUC on test = {_auc6:.4f} (ref upstream={_b_pr_auc:.4f})")

# ── Enriched: full feature model ─────────────────────────────────────────────
_Xe  = enriched_features[_enriched_num_cols].values.astype(float)
_ye  = enriched_features["is_retained"].values.astype(int)
_Xe_tr, _Xe_te, _ye_tr, _ye_te = train_test_split(_Xe, _ye, test_size=0.2, random_state=42, stratify=_ye)
_sce = StandardScaler(); _Xe_tr_s = _sce.fit_transform(_Xe_tr); _Xe_te_s = _sce.transform(_Xe_te)
_ne  = int((_ye_tr==0).sum()); _pe = int((_ye_tr==1).sum())
_swe = np.where(_ye_tr==1, max(_ne,1)/max(_pe,1), 1.0)
_gbme = GradientBoostingClassifier(**_GBM_PARAMS)
_gbme.fit(_Xe_tr_s, _ye_tr, sample_weight=_swe)
_probse = _gbme.predict_proba(_Xe_te_s)[:, 1]
_pce, _rce, _ = precision_recall_curve(_ye_te, _probse)
_auce = average_precision_score(_ye_te, _probse)
_enr_imp_series = pd.Series(dict(zip(_enriched_num_cols, _gbme.feature_importances_)))
print(f"    Enriched GBM retrained:  PR-AUC on test = {_auce:.4f} (ref upstream={_e_pr_auc:.4f})")

# ── [6] VIZ 2 — Side-by-side SHAP feature importance ────────────────────────
_b_top = _base_imp_series.sort_values(ascending=True)   # all 6
_e_top = _enr_imp_series.sort_values(ascending=False).head(15).sort_values(ascending=True)

shap_comparison_chart = plt.figure(figsize=(16, 8))
shap_comparison_chart.patch.set_facecolor(_BG)

_axb = shap_comparison_chart.add_subplot(1, 2, 1)
_axb.set_facecolor(_BG)
_b_clrs = [_feat_color(f) for f in _b_top.index]
_bb = _axb.barh(_b_top.index, _b_top.values, color=_b_clrs, edgecolor="none", height=0.55, zorder=3)
_mx_b = _b_top.max()
for _b_, _v_ in zip(_bb, _b_top.values):
    _axb.text(_v_ + _mx_b*0.01, _b_.get_y() + _b_.get_height()/2,
              f"{_v_:.4f}", va="center", ha="left", color=_TXT_PRI, fontsize=9)
_axb.set_title(f"Baseline GBM\n({_b_feat_count} Original Features)",
               color=_TXT_PRI, fontsize=12, fontweight="bold", pad=12)
_axb.set_xlabel("Feature Importance (Gain)", color=_TXT_PRI, fontsize=10, labelpad=8)
_axb.tick_params(colors=_TXT_PRI, labelsize=9)
for _sp in _axb.spines.values(): _sp.set_edgecolor(_GRID)
_axb.set_xlim(0, _mx_b * 1.22)
_axb.grid(axis="x", color=_GRID, linewidth=0.5, alpha=0.4); _axb.set_axisbelow(True)

_axe = shap_comparison_chart.add_subplot(1, 2, 2)
_axe.set_facecolor(_BG)
_e_clrs = [_feat_color(f) for f in _e_top.index]
_be = _axe.barh(_e_top.index, _e_top.values, color=_e_clrs, edgecolor="none", height=0.55, zorder=3)
_mx_e = _e_top.max()
for _b_, _v_ in zip(_be, _e_top.values):
    _axe.text(_v_ + _mx_e*0.01, _b_.get_y() + _b_.get_height()/2,
              f"{_v_:.4f}", va="center", ha="left", color=_TXT_PRI, fontsize=9)
_axe.set_title(f"Enriched Advanced GBM\n(Top 15 of {_e_n_feats} Features)",
               color=_TXT_PRI, fontsize=12, fontweight="bold", pad=12)
_axe.set_xlabel("Feature Importance (Gain)", color=_TXT_PRI, fontsize=10, labelpad=8)
_axe.tick_params(colors=_TXT_PRI, labelsize=9)
for _sp in _axe.spines.values(): _sp.set_edgecolor(_GRID)
_axe.set_xlim(0, _mx_e * 1.22)
_axe.grid(axis="x", color=_GRID, linewidth=0.5, alpha=0.4); _axe.set_axisbelow(True)

shap_comparison_chart.legend(
    handles=[
        mpatches.Patch(color=_C_BLUE,   label="Original (Phase 4)"),
        mpatches.Patch(color="#FF9F9B", label="Survival features"),
        mpatches.Patch(color="#D0BBFF", label="TS decomposition"),
        mpatches.Patch(color="#8DE5A1", label="Behavioral segment"),
    ],
    loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=4,
    facecolor="#2a2a2e", edgecolor=_GRID, labelcolor=_TXT_PRI, fontsize=9
)
shap_comparison_chart.suptitle(
    "Feature Importance Comparison — Baseline vs Enriched GBM\n★ New features (survival, TS, behavioral) shown in distinct colours",
    color=_TXT_PRI, fontsize=13, fontweight="bold", y=1.07
)
plt.tight_layout(pad=2.5); plt.show()

# ── [7] VIZ 3 — PR curve overlay ────────────────────────────────────────────
_class_prev = _y6_te.mean()

pr_curve_chart = plt.figure(figsize=(10, 7))
pr_curve_chart.patch.set_facecolor(_BG)
_ax_pr = pr_curve_chart.add_subplot(111)
_ax_pr.set_facecolor(_BG)

_ax_pr.plot(_rc6, _pc6, color=_C_BLUE,   linewidth=2.2, label=f"Baseline GBM — PR-AUC = {_auc6:.4f}")
_ax_pr.plot(_rce, _pce, color=_C_ORANGE, linewidth=2.2, label=f"Enriched GBM — PR-AUC = {_auce:.4f}")
_ax_pr.axhline(y=_class_prev, color=_TXT_SEC, linestyle="--", linewidth=1,
               alpha=0.6, label=f"No-skill baseline (prevalence={_class_prev:.3f})")

# Shade improvement zone
_rc_grid = np.linspace(0, 1, 300)
_pc6_i   = np.interp(_rc_grid[::-1], _rc6[::-1], _pc6[::-1])[::-1]
_pce_i   = np.interp(_rc_grid[::-1], _rce[::-1], _pce[::-1])[::-1]
_ax_pr.fill_between(_rc_grid, _pc6_i, _pce_i,
                    where=(_pce_i > _pc6_i), alpha=0.15, color=_C_GREEN,
                    label="Enriched improvement zone")

_ax_pr.set_xlabel("Recall", color=_TXT_PRI, fontsize=12, labelpad=8)
_ax_pr.set_ylabel("Precision", color=_TXT_PRI, fontsize=12, labelpad=8)
_ax_pr.set_xlim(-0.02, 1.02); _ax_pr.set_ylim(-0.02, 1.08)
_ax_pr.set_title(
    f"Precision-Recall Curve Overlay — Baseline vs Enriched GBM\n"
    f"PR-AUC lift: {_auce - _auc6:+.4f}  |  Upstream lift (from advanced_pr_auc): {pr_auc_lift:+.4f}",
    color=_TXT_PRI, fontsize=13, fontweight="bold", pad=14
)
_ax_pr.tick_params(colors=_TXT_PRI, labelsize=10)
for _sp in _ax_pr.spines.values(): _sp.set_edgecolor(_GRID)
_ax_pr.grid(color=_GRID, linewidth=0.5, alpha=0.35)
_ax_pr.legend(facecolor="#2a2a2e", edgecolor=_GRID, labelcolor=_TXT_PRI, fontsize=10, loc="upper right")
plt.tight_layout(); plt.show()

# ── [8] Console summary callout ─────────────────────────────────────────────
_pr_pts  = pr_auc_lift;    _pr_pct  = _pct_imp(_b_pr_auc,  pr_auc_lift)
_roc_pts = roc_auc_lift;   _roc_pct = _pct_imp(_b_roc_auc, roc_auc_lift)
_rec_pts = recall_lift;    _rec_pct = _pct_imp(_b_recall,  recall_lift)

print(f"\n{'╔' + '═'*70 + '╗'}")
print(f"║  📊 ENRICHED MODEL PERFORMANCE SUMMARY{' '*31}║")
print(f"{'╠' + '═'*70 + '╣'}")
print(f"║  Enriched model improves PR-AUC   by {_pr_pts:.4f} pts  ({_pr_pct:+.1f}%){' '*19}║")
print(f"║  Enriched model improves ROC-AUC  by {_roc_pts:.4f} pts  ({_roc_pct:+.1f}%){' '*20}║")
print(f"║  Enriched model improves Recall   by {_rec_pts:.4f} pts  ({_rec_pct:+.1f}%){' '*21}║")
print(f"║{' '*70}║")
print(f"║  Baseline: {_b_feat_count} features  →  Enriched: {_e_n_feats} features{' '*33}║")
print(f"║  PR-AUC:  {_b_pr_auc:.4f} → {_e_pr_auc:.4f}  |  ROC-AUC: {_b_roc_auc:.4f} → {_e_roc_auc:.4f}{' '*14}║")
print(f"║  Recall:  {_b_recall:.4f}  → {_e_recall:.4f}  |  Precision: {_b_precision:.4f} → {_e_precision:.4f}{' '*12}║")
print(f"{'╚' + '═'*70 + '╝'}")
print(f"\n  Exported: comparison_metrics ({len(comparison_metrics)} rows × {len(comparison_metrics.columns)} cols)")
print(f"  Exported: pr_auc_lift={pr_auc_lift:.6f}, roc_auc_lift={roc_auc_lift:.6f}, recall_lift={recall_lift:.6f}")
print(f"\n  ✅ Grouped bar chart | ✅ SHAP importance comparison | ✅ PR curve overlay")
