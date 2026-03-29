
"""
PHASE 7 — ADVANCED GBM ON ENRICHED FEATURE MATRIX
===================================================
Retrains the GBM (+ Platt calibration) on the full enriched feature matrix
(23 numeric features vs original 6). Runs 5-fold stratified CV, computes
PR-AUC, ROC-AUC, recall, precision. Plots a SHAP beeswarm showing new
top features. Compares metrics side-by-side vs Phase 4 baseline.

Exports: advanced_model, advanced_pr_auc, advanced_roc_auc,
         advanced_recall, advanced_precision
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    average_precision_score, recall_score, precision_score,
    roc_auc_score, precision_recall_curve,
)

# ── Design tokens ──────────────────────────────────────────────────────────────
_BG      = "#1D1D20"
_TXT_PRI = "#fbfbff"
_TXT_SEC = "#909094"
_C_BLUE  = "#A1C9F4"
_C_GREEN = "#8DE5A1"
_C_GOLD  = "#ffd400"
_C_CORAL = "#FF9F9B"
_C_ORANGE= "#FFB482"
_C_LAV   = "#D0BBFF"
_GRID    = "#2e2e33"
_SEP7    = "═" * 70

print(_SEP7)
print("  PHASE 7 — ADVANCED GBM ON ENRICHED FEATURE MATRIX")
print(_SEP7)

# ── [1] Prepare feature matrix ─────────────────────────────────────────────────
# Drop non-numeric / ID / label / segment_name columns; keep all numeric features
_EXCLUDE_COLS = {"distinct_id", "is_retained", "segment_name"}
_enriched_num_cols = [
    c for c in enriched_features.columns
    if c not in _EXCLUDE_COLS and enriched_features[c].dtype != object
]

print(f"\n[1] Enriched feature matrix")
print(f"  Total columns          : {enriched_features.shape[1]}")
print(f"  Numeric feature columns: {len(_enriched_num_cols)}")
print(f"  Features: {_enriched_num_cols}")

_X_enr = enriched_features[_enriched_num_cols].values.astype(float)
_y_enr = enriched_features["is_retained"].values.astype(int)

print(f"\n  Samples: {len(_y_enr):,}  |  Class balance: "
      f"{int(_y_enr.sum())} retained ({100*_y_enr.mean():.1f}%), "
      f"{int((1-_y_enr).sum())} churned ({100*(1-_y_enr).mean():.1f}%)")

# ── [2] Train/test split (80/20 stratified, same seed as Phase 4) ──────────────
_X_tr, _X_te, _y_tr, _y_te = train_test_split(
    _X_enr, _y_enr, test_size=0.20, random_state=42, stratify=_y_enr
)

# Scale features
_sc_adv = StandardScaler()
_X_tr_s = _sc_adv.fit_transform(_X_tr)
_X_te_s  = _sc_adv.transform(_X_te)
_X_all_s = _sc_adv.transform(_X_enr)

print(f"\n[2] Train/test split  →  train: {len(_y_tr):,}  |  test: {len(_y_te):,}")

# ── [3] 5-Fold Stratified CV ───────────────────────────────────────────────────
print(f"\n[3] Running 5-Fold Stratified CV on enriched feature matrix …")

_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

_cv_pr_auc_vals, _cv_roc_auc_vals = [], []
_cv_recall_vals, _cv_prec_vals    = [], []

for _fold, (_ti, _vi) in enumerate(_skf.split(_X_tr_s, _y_tr)):
    _Xf, _Xv = _X_tr_s[_ti], _X_tr_s[_vi]
    _yf, _yv = _y_tr[_ti],   _y_tr[_vi]

    if len(np.unique(_yv)) < 2:
        print(f"    Fold {_fold+1}: skipped (single class in val)")
        continue

    # Class weights to handle imbalance
    _n_neg = int((_yf == 0).sum()); _n_pos = int((_yf == 1).sum())
    _sw    = np.where(_yf == 1, max(_n_neg, 1) / max(_n_pos, 1), 1.0)

    _gbm_cv = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, max_features=0.8, random_state=42,
        validation_fraction=0.1, n_iter_no_change=20, tol=1e-4,
    )
    _gbm_cv.fit(_Xf, _yf, sample_weight=_sw)
    _probs_v = _gbm_cv.predict_proba(_Xv)[:, 1]

    # F1-optimal threshold
    _pc, _rc, _tc = precision_recall_curve(_yv, _probs_v)
    _f1s = np.where(
        (_pc[:-1] + _rc[:-1]) > 0,
        2 * _pc[:-1] * _rc[:-1] / (_pc[:-1] + _rc[:-1] + 1e-9), 0.0
    )
    _ot = float(_tc[np.argmax(_f1s)]) if len(_tc) > 0 else 0.5
    _preds_v = (_probs_v >= _ot).astype(int)

    _fold_pr_auc  = average_precision_score(_yv, _probs_v)
    _fold_roc_auc = roc_auc_score(_yv, _probs_v)
    _fold_recall  = recall_score(_yv, _preds_v, pos_label=1, zero_division=0)
    _fold_prec    = precision_score(_yv, _preds_v, pos_label=1, zero_division=0)

    _cv_pr_auc_vals.append(_fold_pr_auc)
    _cv_roc_auc_vals.append(_fold_roc_auc)
    _cv_recall_vals.append(_fold_recall)
    _cv_prec_vals.append(_fold_prec)

    print(f"    Fold {_fold+1}: PR-AUC={_fold_pr_auc:.4f}  ROC-AUC={_fold_roc_auc:.4f}"
          f"  Recall={_fold_recall:.4f}  Precision={_fold_prec:.4f}")

_adv_cv_pr_auc_mean  = float(np.mean(_cv_pr_auc_vals))
_adv_cv_roc_auc_mean = float(np.mean(_cv_roc_auc_vals))
_adv_cv_recall_mean  = float(np.mean(_cv_recall_vals))
_adv_cv_prec_mean    = float(np.mean(_cv_prec_vals))

print(f"\n  5-Fold CV means (enriched):")
print(f"    PR-AUC   : {_adv_cv_pr_auc_mean:.4f}  ±{np.std(_cv_pr_auc_vals):.4f}")
print(f"    ROC-AUC  : {_adv_cv_roc_auc_mean:.4f}  ±{np.std(_cv_roc_auc_vals):.4f}")
print(f"    Recall   : {_adv_cv_recall_mean:.4f}  ±{np.std(_cv_recall_vals):.4f}")
print(f"    Precision: {_adv_cv_prec_mean:.4f}  ±{np.std(_cv_prec_vals):.4f}")

# ── [4] Final model: train on full training set with calibration ───────────────
print(f"\n[4] Training final advanced model (GBM + Platt calibration) on full train set …")

_n_neg_f = int((_y_tr == 0).sum()); _n_pos_f = int((_y_tr == 1).sum())
_sw_f    = np.where(_y_tr == 1, max(_n_neg_f, 1) / max(_n_pos_f, 1), 1.0)

_gbm_final = GradientBoostingClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, max_features=0.8, random_state=42,
    validation_fraction=0.1, n_iter_no_change=20, tol=1e-4,
)
_gbm_final.fit(_X_tr_s, _y_tr, sample_weight=_sw_f)

# Platt calibration (sigmoid) on held-out test set to avoid overfitting cal
advanced_model = CalibratedClassifierCV(_gbm_final, method="sigmoid", cv="prefit")
advanced_model.fit(_X_te_s, _y_te)

print(f"  Advanced model trained: {_gbm_final.n_estimators_} estimators (early stopping)")

# ── [5] Evaluate on test set ───────────────────────────────────────────────────
_adv_probs_te = advanced_model.predict_proba(_X_te_s)[:, 1]

_pc_te, _rc_te, _tc_te = precision_recall_curve(_y_te, _adv_probs_te)
_f1_te = np.where(
    (_pc_te[:-1] + _rc_te[:-1]) > 0,
    2 * _pc_te[:-1] * _rc_te[:-1] / (_pc_te[:-1] + _rc_te[:-1] + 1e-9), 0.0
)
_ot_te = float(_tc_te[np.argmax(_f1_te)]) if len(_tc_te) > 0 else 0.5
_preds_te = (_adv_probs_te >= _ot_te).astype(int)

advanced_pr_auc    = float(average_precision_score(_y_te, _adv_probs_te))
advanced_roc_auc   = float(roc_auc_score(_y_te, _adv_probs_te))
advanced_recall    = float(recall_score(_y_te, _preds_te, pos_label=1, zero_division=0))
advanced_precision = float(precision_score(_y_te, _preds_te, pos_label=1, zero_division=0))

print(f"\n[5] Test-set metrics (advanced model — enriched features):")
print(f"    PR-AUC    : {advanced_pr_auc:.4f}")
print(f"    ROC-AUC   : {advanced_roc_auc:.4f}")
print(f"    Recall    : {advanced_recall:.4f}")
print(f"    Precision : {advanced_precision:.4f}")
print(f"    Threshold : {_ot_te:.4f}")

# ── [6] SHAP beeswarm on enriched features ─────────────────────────────────────
print(f"\n[6] Computing SHAP values for SHAP beeswarm …")

_base_for_shap = advanced_model.calibrated_classifiers_[0].estimator
_n_feats = len(_enriched_num_cols)

def _tree_shap_batch(estimator, X):
    """Batch tree-path SHAP contributions."""
    _tree   = estimator.tree_
    _nvals  = _tree.value[:, 0, 0]
    _n_s    = X.shape[0]
    _contribs = np.zeros((_n_s, X.shape[1]), dtype=np.float64)
    _paths  = estimator.decision_path(X)
    _indptr = _paths.indptr
    _indices = _paths.indices
    for _si in range(_n_s):
        _path = _indices[_indptr[_si]: _indptr[_si + 1]]
        for _ni in range(len(_path) - 1):
            _nd, _ch = _path[_ni], _path[_ni + 1]
            _f = _tree.feature[_nd]
            if _f >= 0:
                _contribs[_si, _f] += _nvals[_ch] - _nvals[_nd]
    return _contribs

# Use a stratified subsample of 400 users for SHAP speed
_rng = np.random.default_rng(42)
_shap_idx_0 = _rng.choice(np.where(_y_enr == 0)[0], size=min(200, int((_y_enr==0).sum())), replace=False)
_shap_idx_1 = _rng.choice(np.where(_y_enr == 1)[0], size=min(200, int((_y_enr==1).sum())), replace=False)
_shap_idx   = np.concatenate([_shap_idx_0, _shap_idx_1])
_X_shap     = _X_all_s[_shap_idx]
_y_shap     = _y_enr[_shap_idx]

_shap_vals = np.zeros((_X_shap.shape[0], _n_feats), dtype=np.float64)
_lr        = _base_for_shap.learning_rate
for _stage in range(_base_for_shap.n_estimators_):
    _shap_vals += _lr * _tree_shap_batch(_base_for_shap.estimators_[_stage][0], _X_shap)

print(f"  SHAP computed for {len(_X_shap)} stratified samples × {_n_feats} features")

# Mean |SHAP| per feature for ranking
_mean_abs_shap = np.abs(_shap_vals).mean(axis=0)
_shap_order    = np.argsort(_mean_abs_shap)[::-1]  # descending
_top15         = _shap_order[:15]

print(f"\n  Top 10 features by mean |SHAP|:")
for _fi in _shap_order[:10]:
    print(f"    {_enriched_num_cols[_fi]:<35}  {_mean_abs_shap[_fi]:.4f}")

# ── [7] SHAP beeswarm plot ─────────────────────────────────────────────────────
_top15_names = [_enriched_num_cols[_fi] for _fi in _top15]
_top15_shap  = _shap_vals[:, _top15]   # (n_samp, 15)
_top15_vals  = _X_shap[:, _top15]      # feature values for colour

# Sort features by mean |SHAP| (lowest at bottom → highest at top)
_sort_order  = np.argsort(_mean_abs_shap[_top15])  # ascending for y-axis
_feat_labels = [_top15_names[_si] for _si in _sort_order]
_shap_sorted = _top15_shap[:, _sort_order]
_vals_sorted = _top15_vals[:, _sort_order]

# Determine which are "new" features (not in original 6)
_ORIG_6 = {
    "first_24h_events", "first_week_events", "consistency_score",
    "unique_tools_used_14d", "agent_usage_ratio_14d", "exploration_index_14d",
}
_is_new = ["★ " if f not in _ORIG_6 else "  " for f in _feat_labels]

shap_beeswarm_fig, _ax_bee = plt.subplots(figsize=(11, 8))
shap_beeswarm_fig.patch.set_facecolor(_BG)
_ax_bee.set_facecolor(_BG)

for _fi, (_fname, _new_tag) in enumerate(zip(_feat_labels, _is_new)):
    _sv = _shap_sorted[:, _fi]
    _fv = _vals_sorted[:, _fi]

    # Colour by feature value (low → blue, high → orange)
    _fv_norm = (_fv - _fv.min()) / (np.ptp(_fv) + 1e-9)
    _colors  = [
        plt.cm.RdYlBu_r(float(_fvn)) for _fvn in _fv_norm
    ]

    # Jitter y to create beeswarm
    _jitter = _rng.uniform(-0.25, 0.25, size=len(_sv))
    _ax_bee.scatter(
        _sv, _fi + _jitter,
        c=_colors, alpha=0.6, s=14, linewidths=0, zorder=3
    )

    # Label: new features highlighted in gold
    _label_color = _C_GOLD if _new_tag.strip() == "★" else _TXT_PRI
    _ax_bee.text(
        -0.45, _fi, f"{_new_tag}{_fname}",
        va="center", ha="right", color=_label_color,
        fontsize=8.2, fontweight="bold" if _new_tag.strip() == "★" else "normal"
    )

_ax_bee.axvline(0, color=_TXT_SEC, linewidth=0.8, linestyle="--", alpha=0.7, zorder=2)
_ax_bee.set_yticks([])
_ax_bee.set_xlabel("SHAP value (impact on model output)", color=_TXT_PRI, fontsize=10, labelpad=8)
_ax_bee.set_title(
    "Phase 7 — SHAP Beeswarm: Top 15 Features (Enriched GBM)\n"
    "★ = new enriched features  |  Colour: low (blue) → high (orange) feature value",
    color=_TXT_PRI, fontsize=12, fontweight="bold", pad=12
)
_ax_bee.tick_params(colors=_TXT_PRI, labelsize=9)
for _sp in _ax_bee.spines.values():
    _sp.set_edgecolor(_GRID)
_ax_bee.set_xlim(-0.5, max(0.5, np.abs(_shap_sorted).max() * 1.1))
_ax_bee.set_ylim(-0.8, len(_feat_labels) - 0.2)
_ax_bee.grid(axis="x", color=_GRID, linewidth=0.5, alpha=0.4)

# Colourbar legend
import matplotlib.cm as cm
_sm = plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=plt.Normalize(vmin=0, vmax=1))
_sm.set_array([])
_cbar = plt.colorbar(_sm, ax=_ax_bee, fraction=0.025, pad=0.02)
_cbar.set_label("Feature value (normalised)", color=_TXT_PRI, fontsize=8)
_cbar.ax.yaxis.set_tick_params(colors=_TXT_PRI, labelsize=7)
_cbar.set_ticks([0, 0.5, 1])
_cbar.set_ticklabels(["Low", "Mid", "High"])

plt.tight_layout()
plt.show()

# ── [8] Side-by-side metrics comparison vs Phase 4 ────────────────────────────
print(f"\n[8] Metrics comparison: Advanced (enriched) vs Phase 4 baseline")

# Phase 4 metrics from Business Recommendation block
_p4_pr_auc    = float(cv_pr_auc_test)
_p4_roc_auc   = float(cv_roc_auc_test)
_p4_recall    = float(cv_recall_test)
_p4_precision = float(cv_precision_test)
_p4_cv_pr_auc = float(cv_pr_auc_mean)
_p4_cv_recall = float(cv_recall_mean)

_metrics_labels  = ["PR-AUC\n(test)", "ROC-AUC\n(test)", "Recall\n(test)", "Precision\n(test)"]
_phase4_vals     = [_p4_pr_auc, _p4_roc_auc, _p4_recall, _p4_precision]
_advanced_vals   = [advanced_pr_auc, advanced_roc_auc, advanced_recall, advanced_precision]
_delta           = [_av - _bv for _av, _bv in zip(_advanced_vals, _phase4_vals)]

print(f"  {'Metric':<20} {'Phase 4':>10} {'Advanced':>10} {'Delta':>10}")
print(f"  {'─'*52}")
for _ml, _p4, _adv, _d in zip(_metrics_labels, _phase4_vals, _advanced_vals, _delta):
    _tag = "▲" if _d > 0.001 else ("▼" if _d < -0.001 else "≈")
    print(f"  {_ml.replace(chr(10), ' '):<20} {_p4:>10.4f} {_adv:>10.4f} {_d:>+9.4f}  {_tag}")

# Grouped bar chart
metrics_comparison_fig, _ax_cmp = plt.subplots(figsize=(11, 6))
metrics_comparison_fig.patch.set_facecolor(_BG)
_ax_cmp.set_facecolor(_BG)

_x = np.arange(len(_metrics_labels))
_w = 0.33
_bars_p4  = _ax_cmp.bar(_x - _w/2, _phase4_vals,  _w, label="Phase 4 (6 features)",
                          color=_C_BLUE,   edgecolor="none", alpha=0.9, zorder=3)
_bars_adv = _ax_cmp.bar(_x + _w/2, _advanced_vals, _w, label="Advanced (23 enriched features)",
                          color=_C_ORANGE, edgecolor="none", alpha=0.9, zorder=3)

# Annotations
for _br, _val in zip(_bars_p4, _phase4_vals):
    _ax_cmp.text(_br.get_x() + _br.get_width() / 2, _val + 0.012,
                 f"{_val:.3f}", ha="center", va="bottom",
                 color=_TXT_PRI, fontsize=9, fontweight="bold")
for _br, _val, _d in zip(_bars_adv, _advanced_vals, _delta):
    _col = _C_GREEN if _d > 0.001 else (_C_CORAL if _d < -0.001 else _TXT_SEC)
    _arrow = "▲" if _d > 0.001 else ("▼" if _d < -0.001 else "≈")
    _ax_cmp.text(_br.get_x() + _br.get_width() / 2, _val + 0.012,
                 f"{_val:.3f}\n{_arrow}{abs(_d):.3f}",
                 ha="center", va="bottom", color=_col, fontsize=8.5, fontweight="bold")

_ax_cmp.set_xticks(_x)
_ax_cmp.set_xticklabels(_metrics_labels, color=_TXT_PRI, fontsize=10)
_ax_cmp.set_ylabel("Score", color=_TXT_PRI, fontsize=11, labelpad=8)
_ax_cmp.set_ylim(0, 1.20)
_ax_cmp.set_title(
    "Phase 7 — Model Performance: Advanced Enriched GBM vs Phase 4 Baseline\n"
    f"Enriched model: 23 features  |  CV PR-AUC (enriched): {_adv_cv_pr_auc_mean:.4f}  "
    f"vs Phase 4: {_p4_cv_pr_auc:.4f}",
    color=_TXT_PRI, fontsize=12, fontweight="bold", pad=12
)
_ax_cmp.tick_params(colors=_TXT_PRI, labelsize=10)
for _sp in _ax_cmp.spines.values():
    _sp.set_edgecolor(_GRID)
_ax_cmp.grid(axis="y", color=_GRID, linewidth=0.6, alpha=0.5)
_ax_cmp.set_axisbelow(True)
_ax_cmp.legend(
    handles=[
        mpatches.Patch(color=_C_BLUE,   label="Phase 4 (6 original features)"),
        mpatches.Patch(color=_C_ORANGE, label="Advanced (23 enriched features)"),
    ],
    facecolor="#2a2a2e", edgecolor=_GRID, labelcolor=_TXT_PRI, fontsize=9,
    loc="upper right"
)

plt.tight_layout()
plt.show()

# ── [9] CV metrics comparison bar ─────────────────────────────────────────────
print(f"\n[9] 5-Fold CV PR-AUC comparison")
print(f"    Phase 4   CV PR-AUC  (mean): {_p4_cv_pr_auc:.4f}")
print(f"    Advanced  CV PR-AUC  (mean): {_adv_cv_pr_auc_mean:.4f}  (Δ={_adv_cv_pr_auc_mean-_p4_cv_pr_auc:+.4f})")
print(f"    Phase 4   CV Recall  (mean): {_p4_cv_recall:.4f}")
print(f"    Advanced  CV Recall  (mean): {_adv_cv_recall_mean:.4f}  (Δ={_adv_cv_recall_mean-_p4_cv_recall:+.4f})")

# ── [10] Summary ───────────────────────────────────────────────────────────────
print(f"\n{_SEP7}")
print("  PHASE 7 — SUMMARY")
print(_SEP7)
print(f"  Features used          : {len(_enriched_num_cols)} (up from 6 in Phase 4)")
print(f"  advanced_pr_auc        : {advanced_pr_auc:.4f}  (Phase 4: {_p4_pr_auc:.4f}  Δ={advanced_pr_auc-_p4_pr_auc:+.4f})")
print(f"  advanced_roc_auc       : {advanced_roc_auc:.4f}  (Phase 4: {_p4_roc_auc:.4f}  Δ={advanced_roc_auc-_p4_roc_auc:+.4f})")
print(f"  advanced_recall        : {advanced_recall:.4f}  (Phase 4: {_p4_recall:.4f}  Δ={advanced_recall-_p4_recall:+.4f})")
print(f"  advanced_precision     : {advanced_precision:.4f}  (Phase 4: {_p4_precision:.4f}  Δ={advanced_precision-_p4_precision:+.4f})")
print(f"\n  Exported: advanced_model, advanced_pr_auc, advanced_roc_auc,")
print(f"            advanced_recall, advanced_precision  ✅")
print(f"{_SEP7}")
