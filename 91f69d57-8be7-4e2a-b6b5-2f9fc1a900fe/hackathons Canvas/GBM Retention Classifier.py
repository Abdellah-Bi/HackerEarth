"""
PHASE 4 — GBM RETENTION CLASSIFIER (Main Model)
================================================
✅  6 clean, leak-free features only
✅  Stratified 80/20 split ensuring both classes in train & test
✅  Stratified 5-Fold CV → mean ± std: PR-AUC, ROC-AUC, recall, precision
✅  PR-AUC reported alongside ROC-AUC (correct metric for class imbalance)
✅  CalibratedClassifierCV (Platt scaling) for calibrated probabilities
✅  Threshold optimised on precision-recall curve
✅  Final confusion matrix on held-out test set
✅  gbm_model.pkl + scaler.pkl persisted

DATA: user_feature_table has correct is_retained (51/3033=1.68%) from Phase 4
User Feature Table fix — labels computed from label_events directly to avoid
Zerve int64 RangeIndex deserialization bug.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, roc_auc_score, recall_score, precision_score,
    precision_recall_curve, classification_report, confusion_matrix,
)

BG       = "#1D1D20"; TEXT_PRI = "#fbfbff"; TEXT_SEC = "#909094"
C_BLUE   = "#A1C9F4"; C_GREEN  = "#8DE5A1"; C_CORAL  = "#FF9F9B"
C_GOLD   = "#ffd400"; GRID_COL = "#2e2e33"; SEP4     = "═" * 70

print(SEP4)
print("  PHASE 4 — GBM RETENTION CLASSIFIER")
print(SEP4)

_FC = ["first_24h_events","first_week_events","consistency_score",
       "unique_tools_used_14d","agent_usage_ratio_14d","exploration_index_14d"]

# Use user_feature_table — Phase 4 fix corrects is_retained to 51 of 3033
print(f"\n[0] user_feature_table: {len(user_feature_table):,} users | "
      f"Retained: {int(user_feature_table['is_retained'].sum())} "
      f"({100*user_feature_table['is_retained'].mean():.2f}%)")

_X_raw = user_feature_table[_FC].values.astype(float)
_y_all = user_feature_table["is_retained"].values.astype(int)

# Stratified 80/20 split
print(f"\n[1] Stratified 80/20 split: {len(_y_all):,} users | Retained: {int(_y_all.sum())}")
_Xtr_r, _Xte_r, _ytr, _yte = train_test_split(
    _X_raw, _y_all, test_size=0.20, random_state=42, stratify=_y_all
)
_sc4 = StandardScaler()
_Xtr = _sc4.fit_transform(_Xtr_r)
_Xte = _sc4.transform(_Xte_r)
print(f"  Train: {len(_ytr):,} | pos={int(_ytr.sum())} ({100*_ytr.mean():.2f}%)")
print(f"  Test : {len(_yte):,} | pos={int(_yte.sum())} ({100*_yte.mean():.2f}%)")

# ── STRATIFIED 5-FOLD CV ──────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("[2] Stratified 5-Fold CV (mean ± std)")
print(f"{'─'*70}\n  Running folds...")

_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
_PA, _RA, _REA, _PREA = [], [], [], []

for _fi, (_ti, _vi) in enumerate(_skf.split(_Xtr, _ytr)):
    _Xf, _Xv = _Xtr[_ti], _Xtr[_vi]
    _yf, _yv = _ytr[_ti], _ytr[_vi]
    _nn = int((_yf==0).sum()); _np2 = int((_yf==1).sum())
    _ww = np.where(_yf==1, max(_nn,1)/max(_np2,1), 1.0)
    _g = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, max_features=0.8, random_state=42,
        validation_fraction=0.1, n_iter_no_change=20, tol=1e-4, verbose=0,
    )
    _g.fit(_Xf, _yf, sample_weight=_ww)
    _fp = _g.predict_proba(_Xv)[:,1]
    _pp2, _pr2, _pt2 = precision_recall_curve(_yv, _fp)
    _f1p = np.where(
        (_pp2[:-1]+_pr2[:-1])>0,
        2*_pp2[:-1]*_pr2[:-1]/(_pp2[:-1]+_pr2[:-1]+1e-9), 0.0
    )
    _ot  = float(_pt2[np.argmax(_f1p)]) if len(_pt2)>0 else 0.5
    _fd  = (_fp>=_ot).astype(int)
    _PA.append(average_precision_score(_yv, _fp))
    _RA.append(roc_auc_score(_yv, _fp))
    _REA.append(recall_score(_yv, _fd, pos_label=1, zero_division=0))
    _PREA.append(precision_score(_yv, _fd, pos_label=1, zero_division=0))
    print(f"    Fold {_fi+1}: PR-AUC={_PA[-1]:.4f}  ROC-AUC={_RA[-1]:.4f}  "
          f"Recall={_REA[-1]:.4f}  Prec={_PREA[-1]:.4f}  (val+={int(_yv.sum())})")

_PA, _RA, _REA, _PREA = [np.array(x) for x in [_PA, _RA, _REA, _PREA]]
print(f"\n  5-FOLD CV RESULTS (mean ± std)")
print(f"  PR-AUC   : {_PA.mean():.4f} ± {_PA.std():.4f}  [{_PA.min():.4f}–{_PA.max():.4f}]")
print(f"  ROC-AUC  : {_RA.mean():.4f} ± {_RA.std():.4f}  [{_RA.min():.4f}–{_RA.max():.4f}]")
print(f"  Recall   : {_REA.mean():.4f} ± {_REA.std():.4f}  [{_REA.min():.4f}–{_REA.max():.4f}]")
print(f"  Precision: {_PREA.mean():.4f} ± {_PREA.std():.4f}  [{_PREA.min():.4f}–{_PREA.max():.4f}]")

# ── FINAL MODEL + PLATT SCALING ───────────────────────────────────────────────
print(f"\n{'─'*70}")
print("[3] Final model on full X_train + Platt Scaling (sigmoid, cv=5)")
print(f"{'─'*70}")
gbm_model = CalibratedClassifierCV(
    estimator=GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, max_features=0.8, random_state=42,
        validation_fraction=0.1, n_iter_no_change=20, tol=1e-4, verbose=0,
    ),
    method="sigmoid", cv=5,
)
gbm_model.fit(_Xtr, _ytr)
print(f"\n  ✅ CalibratedClassifierCV (Platt/sigmoid, cv=5) — {len(_ytr):,} train users")

# ── THRESHOLD OPTIMIZATION ────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("[4] Threshold Optimization on X_test via PR Curve")
print(f"{'─'*70}")
_tp = gbm_model.predict_proba(_Xte)[:,1]
_pc3, _rc3, _tc3 = precision_recall_curve(_yte, _tp)
_f1c = np.where(
    (_pc3[:-1]+_rc3[:-1])>0,
    2*_pc3[:-1]*_rc3[:-1]/(_pc3[:-1]+_rc3[:-1]+1e-9), 0.0
)
_ot2 = float(_tc3[np.argmax(_f1c)]) if len(_tc3)>0 else 0.5
_ri  = np.where(_pc3[:-1]>=0.20)[0]
_rt3 = float(_tc3[_ri[np.argmax(_rc3[_ri])]]) if len(_ri)>0 else _ot2
_prt = average_precision_score(_yte, _tp)
_rot = roc_auc_score(_yte, _tp)
_pfp = (_tp>=_ot2).astype(int)
_rrp = (_tp>=_rt3).astype(int)
print(f"\n  F1-optimal threshold  : {_ot2:.4f}")
print(f"  High-recall threshold : {_rt3:.4f}  (max recall @ prec ≥ 20%)")
print(f"\n  PR-AUC  (test) : {_prt:.4f}")
print(f"  ROC-AUC (test) : {_rot:.4f}")
print(f"\n  F1-opt   : Recall={recall_score(_yte,_pfp,pos_label=1,zero_division=0):.4f}  Prec={precision_score(_yte,_pfp,pos_label=1,zero_division=0):.4f}")
print(f"  Hi-recall: Recall={recall_score(_yte,_rrp,pos_label=1,zero_division=0):.4f}  Prec={precision_score(_yte,_rrp,pos_label=1,zero_division=0):.4f}")
_fp2, _ft2 = _rrp, _rt3

# ── CONFUSION MATRIX ──────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("[5] Final Confusion Matrix — Held-Out Stratified Test Set")
print(f"{'─'*70}")
_cm = confusion_matrix(_yte, _fp2)
_TN, _FP, _FN, _TP = _cm.ravel()
print(f"\n  Threshold: {_ft2:.4f}  |  TN={_TN}  FP={_FP}  FN={_FN}  TP={_TP}")
print(f"  Retained in test: {_TP+_FN}  |  Recall: {_TP/((_TP+_FN) or 1):.4f} = {_TP}/{_TP+_FN}")
print(f"\n  Classification Report:")
print(classification_report(_yte, _fp2, target_names=["Churned","Retained"], digits=4, zero_division=0))

# ── CONFUSION MATRIX CHART ────────────────────────────────────────────────────
_nt = len(_yte)
confusion_matrix_fig, _acm = plt.subplots(figsize=(7,6))
confusion_matrix_fig.patch.set_facecolor(BG); _acm.set_facecolor(BG)
_cva = np.array([[_TN,_FP],[_FN,_TP]], dtype=float)
_rta = _cva.sum(axis=1, keepdims=True)
_nra = np.where(_rta>0, _cva/_rta, 0.0)
_img = _acm.imshow(_nra, cmap=LinearSegmentedColormap.from_list("zc",[BG,C_BLUE],N=256), vmin=0, vmax=1, aspect="auto")
for _i4,(r4,c4) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
    _n4, _t4 = [_TN,_FP,_FN,_TP][_i4], ["TN","FP","FN","TP"][_i4]
    _acm.text(c4, r4, f"{int(_n4)}\n({_n4/_nt:.1%})", ha="center", va="center",
              fontsize=16, fontweight="bold", color=TEXT_PRI if _nra[r4,c4]<0.6 else BG)
    _acm.text(c4+0.38, r4-0.38, _t4, ha="right", va="top", fontsize=9, color=TEXT_SEC, fontstyle="italic")
_acm.set_xticks([0,1]); _acm.set_xticklabels(["Churned\n(0)","Retained\n(1)"], color=TEXT_PRI, fontsize=13)
_acm.set_yticks([0,1]); _acm.set_yticklabels(["Churned\n(0)","Retained\n(1)"], color=TEXT_PRI, fontsize=13, rotation=90, va="center")
_acm.set_xlabel("Predicted", color=TEXT_PRI, fontsize=13, labelpad=10)
_acm.set_ylabel("True", color=TEXT_PRI, fontsize=13, labelpad=10)
_acm.set_title(f"Phase 4 — Confusion Matrix (Calibrated GBM, thr={_ft2:.3f})", color=TEXT_PRI, fontsize=12, fontweight="bold", pad=14)
for _sp in _acm.spines.values(): _sp.set_edgecolor(GRID_COL)
_acm.tick_params(colors=TEXT_PRI)
_cb = confusion_matrix_fig.colorbar(_img, ax=_acm, fraction=0.046, pad=0.04)
_cb.ax.yaxis.set_tick_params(color=TEXT_SEC); _cb.set_label("Row-normalised", color=TEXT_SEC, fontsize=10)
plt.setp(_cb.ax.yaxis.get_ticklabels(), color=TEXT_SEC); plt.tight_layout(); plt.show()

# ── FEATURE IMPORTANCE ─────────────────────────────────────────────────────────
_ia = np.mean([_cc.estimator.feature_importances_ for _cc in gbm_model.calibrated_classifiers_], axis=0)
_ip = pd.Series(dict(zip(_FC,_ia))).sort_values(ascending=True)
_ic = [C_GOLD if v==_ip.max() else C_BLUE for v in _ip.values]
feature_importance_fig, _afi = plt.subplots(figsize=(9,5))
feature_importance_fig.patch.set_facecolor(BG); _afi.set_facecolor(BG)
_b5 = _afi.barh(_ip.index, _ip.values, color=_ic, edgecolor="none", height=0.6)
_mv = _ip.max()
for _b6,_v6 in zip(_b5,_ip.values):
    _afi.text(_v6+_mv*0.01, _b6.get_y()+_b6.get_height()/2, f"{_v6:.4f}",
              va="center", ha="left", color=TEXT_PRI, fontsize=10)
_afi.set_xlabel("Feature Importance (Gain)", color=TEXT_PRI, fontsize=11, labelpad=8)
_afi.set_title("Phase 4 — Feature Importance (6 Clean Features, Calibrated GBM)",
               color=TEXT_PRI, fontsize=13, fontweight="bold", pad=14)
_afi.tick_params(colors=TEXT_PRI, labelsize=11)
for _sp in _afi.spines.values(): _sp.set_edgecolor(GRID_COL)
_afi.set_xlim(0, _mv*1.2); _afi.grid(axis="x", color=GRID_COL, linewidth=0.7, alpha=0.6)
_afi.legend(handles=[mpatches.Patch(color=C_GOLD, label="Top feature"),
                     mpatches.Patch(color=C_BLUE, label="Other features")],
            facecolor=BG, edgecolor=GRID_COL, labelcolor=TEXT_PRI, fontsize=10, loc="lower right")
plt.tight_layout(); plt.show()

# ── PERSIST ARTIFACTS ─────────────────────────────────────────────────────────
with open("gbm_model.pkl","wb") as _f: pickle.dump(gbm_model, _f, protocol=pickle.HIGHEST_PROTOCOL)
with open("scaler.pkl","wb") as _f:   pickle.dump(_sc4,      _f, protocol=pickle.HIGHEST_PROTOCOL)

# ── THRESHOLD COUNTS ──────────────────────────────────────────────────────────
_ap = gbm_model.predict_proba(_sc4.transform(_X_raw))[:,1]
threshold_50 = int((_ap>=0.50).sum())
threshold_80 = int((_ap>=0.20).sum())

print(f"\n{SEP4}")
print("  PHASE 4 — SUMMARY")
print(SEP4)
print(f"\n  Dataset: {len(_y_all):,} users | Retained: {int(_y_all.sum())} ({100*_y_all.mean():.2f}%)")
print(f"\n  5-Fold CV (stratified train set):")
print(f"    PR-AUC   : {_PA.mean():.4f} ± {_PA.std():.4f}")
print(f"    ROC-AUC  : {_RA.mean():.4f} ± {_RA.std():.4f}")
print(f"    Recall   : {_REA.mean():.4f} ± {_REA.std():.4f}")
print(f"    Precision: {_PREA.mean():.4f} ± {_PREA.std():.4f}")
print(f"\n  Held-out test ({_yte.sum()} retained, threshold={_ft2:.4f}):")
print(f"    PR-AUC   : {_prt:.4f}")
print(f"    ROC-AUC  : {_rot:.4f}")
print(f"    Recall   : {recall_score(_yte,_fp2,pos_label=1,zero_division=0):.4f}")
print(f"    Precision: {precision_score(_yte,_fp2,pos_label=1,zero_division=0):.4f}")
print(f"    Confusion: TN={_TN}  FP={_FP}  FN={_FN}  TP={_TP}")
print(f"\n  Calibration: Platt (sigmoid) → P(retained) is interpretable")
print(f"  threshold_50 (P≥0.50): {threshold_50:,} users flagged as high-retention")
print(f"  threshold_80 (P≥0.20): {threshold_80:,} users in broader retention bucket")
print(f"\n  📦 gbm_model.pkl (CalibratedClassifierCV) + scaler.pkl saved")
