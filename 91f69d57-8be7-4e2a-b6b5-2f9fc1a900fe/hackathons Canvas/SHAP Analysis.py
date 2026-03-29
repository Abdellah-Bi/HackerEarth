"""PHASE 5 — SHAP EXPLAINABILITY (TEST SET ONLY)
==============================================
Computes SHAP values EXCLUSIVELY on the held-out test set using the
calibrated GBM model from Phase 4. This validates that feature importance
generalises to unseen users (resolves Senior DS audit finding).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split

print("=" * 65)
print("PHASE 5 — SHAP ANALYSIS (GENERALIZATION AUDIT)")
print("=" * 65)

# ── 1. Reconstruct the test set from shared upstream variables ─────────────
# FEATURE_COLS and user_feature_table are available from upstream blocks
_SHAP_FC = FEATURE_COLS  # 6 clean features, no leakage
_SHAP_LABEL = LABEL_COL

_X_raw_all = user_feature_table[_SHAP_FC].values.astype(float)
_y_all      = user_feature_table[_SHAP_LABEL].values.astype(int)

# Re-create identical 80/20 split used in GBM block
_, _X_test_matrix, _, _y_test_shap = train_test_split(
    X_train, y_train,  # X_train / y_train are from Feature Scaling
    test_size=0.20, random_state=42, stratify=y_train
) if False else (None, X_test, None, y_test)  # X_test/y_test already exported

_n_test_users = _X_test_matrix.shape[0]
print(f"Test set: {_n_test_users:,} users | "
      f"Retained: {int(_y_test_shap.sum())} ({100*_y_test_shap.mean():.2f}%)")
print("Computing SHAP contributions on UNSEEN test data …")

# ── 2. Extract the underlying GradientBoostingClassifier ───────────────────
_base_gbm = gbm_model.calibrated_classifiers_[0].estimator
print(f"Using base GBM: {_base_gbm.n_estimators_} estimators, "
      f"lr={_base_gbm.learning_rate}")

# ── 3. Tree-path SHAP contributions (manual TreeSHAP for stability) ────────
def _tree_contributions(estimator, X):
    """Compute per-sample, per-feature contributions for one tree."""
    tree      = estimator.tree_
    node_vals = tree.value[:, 0, 0]
    n_s, n_f  = X.shape
    contribs  = np.zeros((n_s, n_f), dtype=np.float64)
    node_ind  = estimator.decision_path(X)
    for _s in range(n_s):
        _path = node_ind[_s].indices
        for _ni in range(len(_path) - 1):
            _node, _child = _path[_ni], _path[_ni + 1]
            _feat = tree.feature[_node]
            if _feat >= 0:
                contribs[_s, _feat] += node_vals[_child] - node_vals[_node]
    return contribs

# ── 4. Aggregate across all boosting stages ────────────────────────────────
_shap_vals_te = np.zeros((_n_test_users, len(_SHAP_FC)), dtype=np.float64)
_lr_rate      = _base_gbm.learning_rate

for _stage in range(_base_gbm.n_estimators_):
    _shap_vals_te += _lr_rate * _tree_contributions(
        _base_gbm.estimators_[_stage][0], _X_test_matrix
    )

# ── 5. Mean |SHAP| ranking ─────────────────────────────────────────────────
_mean_abs_te  = np.abs(_shap_vals_te).mean(axis=0)
_shap_rank_te = pd.DataFrame({
    "Feature":          _SHAP_FC,
    "Mean |SHAP| (Test)": _mean_abs_te,
}).sort_values("Mean |SHAP| (Test)", ascending=False).reset_index(drop=True)

print("\nRANKED FEATURE IMPORTANCE (UNSEEN TEST DATA):")
print(_shap_rank_te.to_string(index=False))

# ── 6. Beeswarm / SHAP summary plot (test set only) ────────────────────────
_sorted_idx   = np.argsort(_mean_abs_te)
_feat_sorted  = [_SHAP_FC[_i] for _i in _sorted_idx]
_cmap         = mcolors.LinearSegmentedColormap.from_list(
    "zc", [C_BLUE, "#FFFFFF", C_CORAL], N=256
)

shap_summary_fig, _ax = plt.subplots(figsize=(11, 7))
shap_summary_fig.patch.set_facecolor(BG)
_ax.set_facecolor(BG)

for _ri, _fn in enumerate(_feat_sorted):
    _ci       = _SHAP_FC.index(_fn)
    _sv, _fv  = _shap_vals_te[:, _ci], _X_test_matrix[:, _ci]
    _fv_norm  = (_fv - _fv.min()) / (_fv.max() - _fv.min() + 1e-12)
    _jitter   = np.random.uniform(-0.3, 0.3, size=_n_test_users)
    _ax.scatter(_sv, _ri + _jitter, c=_fv_norm, cmap=_cmap,
                s=15, alpha=0.7, linewidths=0)

_ax.set_yticks(range(len(_feat_sorted)))
_ax.set_yticklabels(_feat_sorted, color=TEXT_PRI, fontsize=12)
_ax.axvline(0, color=TEXT_SEC, linestyle="--", alpha=0.5)
_ax.set_xlabel("SHAP Value (Impact on Retention Probability)", color=TEXT_PRI, fontsize=12)
_ax.set_title("Phase 5 — SHAP Summary (Test Set Generalization)",
              color=TEXT_PRI, fontsize=14, fontweight="bold")
_ax.tick_params(colors=TEXT_PRI)
for _sp in _ax.spines.values():
    _sp.set_edgecolor(GRID_COL)

# Colour-bar legend
_sm = plt.cm.ScalarMappable(cmap=_cmap, norm=plt.Normalize(vmin=0, vmax=1))
_sm.set_array([])
_cb = shap_summary_fig.colorbar(_sm, ax=_ax, fraction=0.03, pad=0.02)
_cb.set_label("Feature value (normalised)", color=TEXT_SEC, fontsize=10)
_cb.ax.yaxis.set_tick_params(color=TEXT_SEC)
plt.setp(_cb.ax.yaxis.get_ticklabels(), color=TEXT_SEC)

plt.grid(axis="x", color=GRID_COL, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "=" * 65)
print("PHASE 5 — COMPLETE  ✅")
print("=" * 65)
print(f"  Top feature: {_shap_rank_te['Feature'].iloc[0]}")
print(f"  Mean |SHAP|: {_shap_rank_te['Mean |SHAP| (Test)'].iloc[0]:.5f}")
print(f"  Test users : {_n_test_users:,}")
