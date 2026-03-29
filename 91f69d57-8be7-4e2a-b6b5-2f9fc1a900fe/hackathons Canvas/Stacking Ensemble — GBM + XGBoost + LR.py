
"""
PHASE 8 — STACKING ENSEMBLE: GBM + Random Forest + Logistic Regression
=======================================================================
Combines three diverse base learners via stacked generalisation:
  • GBM  (advanced_model, CalibratedClassifierCV)  — same arch as Phase 7
  • Random Forest (CalibratedClassifierCV, Platt)  — diverse tree ensemble
  • Logistic Regression                            — linear base learner

Note: XGBoost is not installed in this environment; Random Forest is used
      as the second powerful tree-based learner instead.

Meta-learner: Logistic Regression on 5-fold OOF probability stacks.

Steps:
  1. Prepare enriched feature matrix & labels (same as Phase 7)
  2. 5-fold OOF predictions for all 3 base learners (StratifiedKFold)
     → Platt calibration applied to GBM & Random Forest OOF probs
  3. Stack OOF probs → meta-features; train LR meta-learner
  4. 5-fold CV evaluation of ALL 4 models (GBM, RF, LR, Ensemble)
     → PR-AUC, ROC-AUC, Recall, Precision, F1
  5. Print clean 4-way comparison table with Δ vs best individual
  6. Grouped bar chart of all metrics × 4 models
  7. PR curve overlay (all 4 models, AUC in legend)

Exports: ensemble_model, ensemble_pr_auc, ensemble_roc_auc,
         ensemble_recall, ensemble_precision, ensemble_f1,
         stacking_comparison (DataFrame)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    average_precision_score, recall_score, precision_score,
    roc_auc_score, f1_score, precision_recall_curve,
)

# ── Design tokens ──────────────────────────────────────────────────────────────
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

_SEP8 = "═" * 72
print(_SEP8)
print("  PHASE 8 — STACKING ENSEMBLE: GBM + Random Forest + LR")
print(_SEP8)

# ── [1] Prepare feature matrix ─────────────────────────────────────────────────
_EXCL = {"distinct_id", "is_retained", "segment_name"}
_enr_feat_cols = [
    c for c in enriched_features.columns
    if c not in _EXCL and enriched_features[c].dtype != object
]

_X_enr = enriched_features[_enr_feat_cols].values.astype(float)
_y_enr = enriched_features["is_retained"].values.astype(int)

print(f"\n[1] Feature matrix  →  {_X_enr.shape[0]:,} samples × {_X_enr.shape[1]} features")
print(f"    Retained: {int(_y_enr.sum()):,}  |  Churned: {int((1-_y_enr).sum()):,}")

# Scale features (fresh scaler)
_sc = StandardScaler()
_X_sc = _sc.fit_transform(_X_enr)

_n_neg = int((_y_enr == 0).sum())
_n_pos = int((_y_enr == 1).sum())
_scale_pos = _n_neg / max(_n_pos, 1)

# ── [2] 5-Fold OOF predictions for all 3 base learners ────────────────────────
print(f"\n[2] Generating 5-fold OOF predictions for base learners …")

_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
_n   = len(_y_enr)

_oof_gbm = np.zeros(_n)
_oof_rf  = np.zeros(_n)
_oof_lr  = np.zeros(_n)

for _fold, (_tri, _vai) in enumerate(_skf.split(_X_sc, _y_enr)):
    _Xtr, _Xva = _X_sc[_tri], _X_sc[_vai]
    _ytr, _yva = _y_enr[_tri], _y_enr[_vai]
    _sw = np.where(_ytr == 1, _scale_pos, 1.0)

    # --- GBM base learner (Platt calibration) ---
    _gbm_oof = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, max_features=0.8, random_state=42,
        validation_fraction=0.1, n_iter_no_change=20, tol=1e-4,
    )
    _gbm_oof.fit(_Xtr, _ytr, sample_weight=_sw)
    _gbm_cal = CalibratedClassifierCV(_gbm_oof, method="sigmoid", cv="prefit")
    _gbm_cal.fit(_Xva, _yva)
    _oof_gbm[_vai] = _gbm_cal.predict_proba(_Xva)[:, 1]

    # --- Random Forest base learner (Platt calibration) ---
    _rf_base = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    _rf_base.fit(_Xtr, _ytr)
    _rf_cal = CalibratedClassifierCV(_rf_base, method="sigmoid", cv="prefit")
    _rf_cal.fit(_Xva, _yva)
    _oof_rf[_vai] = _rf_cal.predict_proba(_Xva)[:, 1]

    # --- Logistic Regression base learner ---
    _lr_oof = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42, C=1.0
    )
    _lr_oof.fit(_Xtr, _ytr)
    _oof_lr[_vai] = _lr_oof.predict_proba(_Xva)[:, 1]

    _fold_gbm_ap = average_precision_score(_yva, _oof_gbm[_vai])
    _fold_rf_ap  = average_precision_score(_yva, _oof_rf[_vai])
    _fold_lr_ap  = average_precision_score(_yva, _oof_lr[_vai])
    print(f"    Fold {_fold+1}: GBM PR-AUC={_fold_gbm_ap:.4f}  "
          f"RF PR-AUC={_fold_rf_ap:.4f}  LR PR-AUC={_fold_lr_ap:.4f}")

print(f"\n  OOF generation complete. Meta-feature matrix shape: ({_n}, 3)")

# ── [3] Stack OOF probs and train LR meta-learner ─────────────────────────────
print(f"\n[3] Training Logistic Regression meta-learner on stacked OOF probs …")

_meta_X = np.column_stack([_oof_gbm, _oof_rf, _oof_lr])  # (n, 3)

_mt_X_tr, _mt_X_te, _mt_y_tr, _mt_y_te = train_test_split(
    _meta_X, _y_enr, test_size=0.20, random_state=42, stratify=_y_enr
)

ensemble_model = LogisticRegression(
    max_iter=1000, class_weight="balanced", random_state=42, C=1.0
)
ensemble_model.fit(_mt_X_tr, _mt_y_tr)
print(f"  Meta-learner trained on {len(_mt_y_tr):,} OOF samples.")
print(f"  Meta-learner coefficients (GBM | RF | LR): "
      f"{ensemble_model.coef_[0][0]:.3f} | {ensemble_model.coef_[0][1]:.3f} | {ensemble_model.coef_[0][2]:.3f}")

# ── [4] 5-Fold CV evaluation of all 4 models ──────────────────────────────────
print(f"\n[4] 5-Fold CV evaluation of all 4 models …")

def _f1_optimal_threshold(y_true, y_prob):
    """Return predictions at the F1-optimal threshold."""
    _pc, _rc, _tc = precision_recall_curve(y_true, y_prob)
    _f1s = np.where(
        (_pc[:-1] + _rc[:-1]) > 0,
        2 * _pc[:-1] * _rc[:-1] / (_pc[:-1] + _rc[:-1] + 1e-9), 0.0
    )
    _ot = float(_tc[np.argmax(_f1s)]) if len(_tc) > 0 else 0.5
    return (y_prob >= _ot).astype(int)


_cv_results = {m: {"pr_auc": [], "roc_auc": [], "recall": [], "precision": [], "f1": []}
               for m in ["GBM", "RF", "LR", "Ensemble"]}

for _fold, (_tri, _vai) in enumerate(_skf.split(_X_sc, _y_enr)):
    _Xtr, _Xva = _X_sc[_tri], _X_sc[_vai]
    _ytr, _yva = _y_enr[_tri], _y_enr[_vai]
    _sw = np.where(_ytr == 1, _scale_pos, 1.0)

    # ── GBM ──
    _gbm_cv = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, max_features=0.8, random_state=42,
        validation_fraction=0.1, n_iter_no_change=20, tol=1e-4,
    )
    _gbm_cv.fit(_Xtr, _ytr, sample_weight=_sw)
    _gbm_cv_cal = CalibratedClassifierCV(_gbm_cv, method="sigmoid", cv="prefit")
    _gbm_cv_cal.fit(_Xva, _yva)
    _p_gbm = _gbm_cv_cal.predict_proba(_Xva)[:, 1]

    # ── Random Forest ──
    _rf_cv = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    _rf_cv.fit(_Xtr, _ytr)
    _rf_cv_cal = CalibratedClassifierCV(_rf_cv, method="sigmoid", cv="prefit")
    _rf_cv_cal.fit(_Xva, _yva)
    _p_rf = _rf_cv_cal.predict_proba(_Xva)[:, 1]

    # ── LR ──
    _lr_cv = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42, C=1.0
    )
    _lr_cv.fit(_Xtr, _ytr)
    _p_lr = _lr_cv.predict_proba(_Xva)[:, 1]

    # ── Ensemble: fold-specific meta features ──
    _meta_va = np.column_stack([_p_gbm, _p_rf, _p_lr])
    _meta_tr = np.column_stack([
        _gbm_cv_cal.predict_proba(_Xtr)[:, 1],
        _rf_cv_cal.predict_proba(_Xtr)[:, 1],
        _lr_cv.predict_proba(_Xtr)[:, 1],
    ])
    _ens_meta = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42, C=1.0
    )
    _ens_meta.fit(_meta_tr, _ytr)
    _p_ens = _ens_meta.predict_proba(_meta_va)[:, 1]

    for _mname, _prob in [("GBM", _p_gbm), ("RF", _p_rf),
                           ("LR", _p_lr), ("Ensemble", _p_ens)]:
        if len(np.unique(_yva)) < 2:
            continue
        _preds = _f1_optimal_threshold(_yva, _prob)
        _cv_results[_mname]["pr_auc"].append(average_precision_score(_yva, _prob))
        _cv_results[_mname]["roc_auc"].append(roc_auc_score(_yva, _prob))
        _cv_results[_mname]["recall"].append(recall_score(_yva, _preds, zero_division=0))
        _cv_results[_mname]["precision"].append(precision_score(_yva, _preds, zero_division=0))
        _cv_results[_mname]["f1"].append(f1_score(_yva, _preds, zero_division=0))

# Compute means & std
_cv_means = {
    m: {met: float(np.mean(vals)) for met, vals in metrics.items()}
    for m, metrics in _cv_results.items()
}
_cv_stds = {
    m: {met: float(np.std(vals)) for met, vals in metrics.items()}
    for m, metrics in _cv_results.items()
}

# ── [5] Print clean 4-way comparison table ────────────────────────────────────
print(f"\n[5] 4-Way Model Comparison (5-Fold CV Means)")
print(_SEP8)

_METRICS_PRINT = ["pr_auc", "roc_auc", "recall", "precision", "f1"]
_MNAMES        = ["GBM", "RF", "LR", "Ensemble"]
_MNAMES_DISP   = ["GBM (Enriched)", "Random Forest", "Log. Regression", "Stacked Ensemble"]
_HEADERS       = ["PR-AUC", "ROC-AUC", "Recall", "Precision", "F1"]

# Best individual model per metric (exclude Ensemble)
_best_ind = {
    _met: max(_cv_means[m][_met] for m in ["GBM", "RF", "LR"])
    for _met in _METRICS_PRINT
}

_hdr = f"  {'Metric':<12}" + "".join(f"{_h:>16}" for _h in _MNAMES_DISP) + f"  {'Δ vs Best':>10}"
print(_hdr)
print("  " + "─" * (len(_hdr) - 2))

for _met, _mlabel in zip(_METRICS_PRINT, _HEADERS):
    _row = f"  {_mlabel:<12}"
    for _mn in _MNAMES:
        _v = _cv_means[_mn][_met]
        _row += f"{_v:>16.4f}"
    _delta = _cv_means["Ensemble"][_met] - _best_ind[_met]
    _tag   = "▲" if _delta > 0.0005 else ("▼" if _delta < -0.0005 else "≈")
    _row  += f"  {_delta:>+8.4f}  {_tag}"
    print(_row)

print("  " + "─" * (len(_hdr) - 2))

# ── Export scalar ensemble metrics ────────────────────────────────────────────
ensemble_pr_auc    = _cv_means["Ensemble"]["pr_auc"]
ensemble_roc_auc   = _cv_means["Ensemble"]["roc_auc"]
ensemble_recall    = _cv_means["Ensemble"]["recall"]
ensemble_precision = _cv_means["Ensemble"]["precision"]
ensemble_f1        = _cv_means["Ensemble"]["f1"]

print(f"\n  Ensemble 5-Fold CV Means:")
print(f"    PR-AUC    : {ensemble_pr_auc:.4f}  ±{_cv_stds['Ensemble']['pr_auc']:.4f}")
print(f"    ROC-AUC   : {ensemble_roc_auc:.4f}  ±{_cv_stds['Ensemble']['roc_auc']:.4f}")
print(f"    Recall    : {ensemble_recall:.4f}  ±{_cv_stds['Ensemble']['recall']:.4f}")
print(f"    Precision : {ensemble_precision:.4f}  ±{_cv_stds['Ensemble']['precision']:.4f}")
print(f"    F1        : {ensemble_f1:.4f}  ±{_cv_stds['Ensemble']['f1']:.4f}")

# ── Export stacking_comparison DataFrame ──────────────────────────────────────
stacking_comparison = pd.DataFrame(
    {_mn: {_ml: _cv_means[_mn][_met] for _met, _ml in zip(_METRICS_PRINT, _HEADERS)}
     for _mn in _MNAMES}
).T.reset_index().rename(columns={"index": "Model"})
stacking_comparison["Model"] = _MNAMES_DISP
stacking_comparison["Δ_vs_Best_PR_AUC"] = [
    float(_cv_means[m]["pr_auc"]) - _best_ind["pr_auc"] for m in _MNAMES
]

print(f"\n  stacking_comparison DataFrame:")
print(stacking_comparison.to_string(index=False))

# ── [6] Grouped bar chart ──────────────────────────────────────────────────────
print(f"\n[6] Plotting grouped bar chart …")

_MODEL_COLORS = {
    "GBM":      _C_BLUE,
    "RF":       _C_ORANGE,
    "LR":       _C_CORAL,
    "Ensemble": _C_GOLD,
}
_LEGEND_LABELS = {
    "GBM":      "GBM (Enriched)",
    "RF":       "Random Forest",
    "LR":       "Log. Regression",
    "Ensemble": "Stacked Ensemble",
}

grouped_bar_fig, _ax_bar = plt.subplots(figsize=(14, 7))
grouped_bar_fig.patch.set_facecolor(_BG)
_ax_bar.set_facecolor(_BG)

_x       = np.arange(len(_HEADERS))
_n_mods  = len(_MNAMES)
_w       = 0.18
_offsets = np.linspace(-(_n_mods - 1) * _w / 2, (_n_mods - 1) * _w / 2, _n_mods)

for _mname, _offset in zip(_MNAMES, _offsets):
    _vals = [_cv_means[_mname][_mk] for _mk in _METRICS_PRINT]
    _col  = _MODEL_COLORS[_mname]
    _rects = _ax_bar.bar(
        _x + _offset, _vals, _w,
        label=_LEGEND_LABELS[_mname], color=_col, edgecolor="none", alpha=0.90, zorder=3
    )
    for _rect, _val in zip(_rects, _vals):
        _ax_bar.text(
            _rect.get_x() + _rect.get_width() / 2,
            _val + 0.008, f"{_val:.3f}",
            ha="center", va="bottom", color=_col,
            fontsize=7.5, fontweight="bold"
        )

_ax_bar.set_xticks(_x)
_ax_bar.set_xticklabels(_HEADERS, color=_TXT_PRI, fontsize=11)
_ax_bar.set_ylabel("Score (5-Fold CV Mean)", color=_TXT_PRI, fontsize=11, labelpad=8)
_ax_bar.set_ylim(0, 1.28)
_ax_bar.set_title(
    "Phase 8 — Stacking Ensemble: 4-Model Metric Comparison (5-Fold CV)\n"
    "GBM (Enriched) vs Random Forest vs Logistic Regression vs Stacked Ensemble",
    color=_TXT_PRI, fontsize=13, fontweight="bold", pad=14
)
_ax_bar.tick_params(colors=_TXT_PRI, labelsize=10)
for _sp in _ax_bar.spines.values():
    _sp.set_edgecolor(_GRID)
_ax_bar.grid(axis="y", color=_GRID, linewidth=0.6, alpha=0.5)
_ax_bar.set_axisbelow(True)
_ax_bar.legend(
    handles=[mpatches.Patch(color=_MODEL_COLORS[m], label=_LEGEND_LABELS[m]) for m in _MNAMES],
    facecolor="#2a2a2e", edgecolor=_GRID, labelcolor=_TXT_PRI,
    fontsize=10, loc="upper right", framealpha=0.9
)
plt.tight_layout()
plt.show()

# ── [7] PR Curve overlay ───────────────────────────────────────────────────────
print(f"\n[7] Plotting PR curve overlay …")

_X_tr_pr, _X_te_pr, _y_tr_pr, _y_te_pr = train_test_split(
    _X_sc, _y_enr, test_size=0.20, random_state=42, stratify=_y_enr
)
_sw_pr = np.where(_y_tr_pr == 1, _scale_pos, 1.0)

# Final base learners on train split
_gbm_pr = GradientBoostingClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, max_features=0.8, random_state=42,
    validation_fraction=0.1, n_iter_no_change=20, tol=1e-4,
)
_gbm_pr.fit(_X_tr_pr, _y_tr_pr, sample_weight=_sw_pr)
_gbm_pr_cal = CalibratedClassifierCV(_gbm_pr, method="sigmoid", cv="prefit")
_gbm_pr_cal.fit(_X_te_pr, _y_te_pr)
_p_gbm_pr = _gbm_pr_cal.predict_proba(_X_te_pr)[:, 1]

_rf_pr = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
_rf_pr.fit(_X_tr_pr, _y_tr_pr)
_rf_pr_cal = CalibratedClassifierCV(_rf_pr, method="sigmoid", cv="prefit")
_rf_pr_cal.fit(_X_te_pr, _y_te_pr)
_p_rf_pr = _rf_pr_cal.predict_proba(_X_te_pr)[:, 1]

_lr_pr = LogisticRegression(
    max_iter=1000, class_weight="balanced", random_state=42, C=1.0
)
_lr_pr.fit(_X_tr_pr, _y_tr_pr)
_p_lr_pr = _lr_pr.predict_proba(_X_te_pr)[:, 1]

_meta_te_pr = np.column_stack([_p_gbm_pr, _p_rf_pr, _p_lr_pr])
_meta_tr_pr = np.column_stack([
    _gbm_pr_cal.predict_proba(_X_tr_pr)[:, 1],
    _rf_pr_cal.predict_proba(_X_tr_pr)[:, 1],
    _lr_pr.predict_proba(_X_tr_pr)[:, 1],
])
_ens_pr = LogisticRegression(
    max_iter=1000, class_weight="balanced", random_state=42, C=1.0
)
_ens_pr.fit(_meta_tr_pr, _y_tr_pr)
_p_ens_pr = _ens_pr.predict_proba(_meta_te_pr)[:, 1]

_pr_curves = [
    ("GBM (Enriched)", _p_gbm_pr, _C_BLUE,   "--", 1.8, 0.80),
    ("Random Forest",  _p_rf_pr,  _C_ORANGE,  "--", 1.8, 0.80),
    ("Log. Regression",_p_lr_pr,  _C_CORAL,   "--", 1.8, 0.80),
    ("Stacked Ensemble",_p_ens_pr,_C_GOLD,     "-",  2.8, 0.95),
]

pr_curve_fig, _ax_pr = plt.subplots(figsize=(10, 7))
pr_curve_fig.patch.set_facecolor(_BG)
_ax_pr.set_facecolor(_BG)

for _mname, _probs, _col, _ls, _lw, _alpha in _pr_curves:
    _prec_c, _rec_c, _ = precision_recall_curve(_y_te_pr, _probs)
    _ap = average_precision_score(_y_te_pr, _probs)
    _ax_pr.plot(
        _rec_c, _prec_c,
        color=_col, linewidth=_lw, linestyle=_ls,
        label=f"{_mname}  (PR-AUC = {_ap:.4f})",
        alpha=_alpha
    )

_base_rate = float(_y_te_pr.mean())
_ax_pr.axhline(_base_rate, color=_TXT_SEC, linewidth=1.0, linestyle=":",
               alpha=0.6, label=f"Baseline (random) = {_base_rate:.3f}")

_ax_pr.set_xlabel("Recall", color=_TXT_PRI, fontsize=12, labelpad=8)
_ax_pr.set_ylabel("Precision", color=_TXT_PRI, fontsize=12, labelpad=8)
_ax_pr.set_title(
    "Phase 8 — Precision-Recall Curve Overlay: All 4 Models\n"
    "Stacked Ensemble shown solid; base learners dashed",
    color=_TXT_PRI, fontsize=13, fontweight="bold", pad=14
)
_ax_pr.set_xlim(0, 1); _ax_pr.set_ylim(0, 1.05)
_ax_pr.tick_params(colors=_TXT_PRI, labelsize=10)
for _sp in _ax_pr.spines.values():
    _sp.set_edgecolor(_GRID)
_ax_pr.grid(color=_GRID, linewidth=0.5, alpha=0.4)
_ax_pr.legend(
    facecolor="#2a2a2e", edgecolor=_GRID, labelcolor=_TXT_PRI,
    fontsize=10, loc="upper right", framealpha=0.9
)
plt.tight_layout()
plt.show()

# ── [8] Summary ────────────────────────────────────────────────────────────────
print(f"\n{_SEP8}")
print("  PHASE 8 — SUMMARY")
print(_SEP8)
print(f"  ensemble_pr_auc    : {ensemble_pr_auc:.4f}  ±{_cv_stds['Ensemble']['pr_auc']:.4f}")
print(f"  ensemble_roc_auc   : {ensemble_roc_auc:.4f}  ±{_cv_stds['Ensemble']['roc_auc']:.4f}")
print(f"  ensemble_recall    : {ensemble_recall:.4f}  ±{_cv_stds['Ensemble']['recall']:.4f}")
print(f"  ensemble_precision : {ensemble_precision:.4f}  ±{_cv_stds['Ensemble']['precision']:.4f}")
print(f"  ensemble_f1        : {ensemble_f1:.4f}  ±{_cv_stds['Ensemble']['f1']:.4f}")
print(f"\n  Exports: ensemble_model, ensemble_pr_auc, ensemble_roc_auc,")
print(f"           ensemble_recall, ensemble_precision, ensemble_f1,")
print(f"           stacking_comparison  ✅")
print(_SEP8)
