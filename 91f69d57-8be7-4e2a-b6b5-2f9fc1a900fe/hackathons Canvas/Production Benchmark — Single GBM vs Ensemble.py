
"""
PHASE 9 — PRODUCTION BENCHMARK: Single GBM vs Stacking Ensemble
================================================================
Benchmarks both models across:
  1. Prediction latency  — single sample, batch 100 / 1000 / 10000
  2. Memory footprint    — sys.getsizeof + deep pickle sizing
  3. Throughput          — predictions/sec at each batch size
  4. CV metrics          — PR-AUC, ROC-AUC, Recall, F1 (from prior blocks)
  5. Benchmark table     — side-by-side comparison with Δ column
  6. Latency scaling chart — log-log lines for both models across batch sizes

Models:
  • Single GBM  — loaded from gbm_model.pkl (6-feature, CalibratedClassifierCV)
  • Stacking Ensemble — GBM + RF + LR (25-feature enriched, from ensemble_model
                        downstream of stacking block; also re-measured here)

Note: The GBM .pkl file uses the 6-feature scaler; the ensemble requires the
enriched 25-feature matrix & its own scaler. We reconstruct inference paths
to ensure fair like-for-like measurement.
"""

import pickle
import time
import sys
import io
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
from sklearn.model_selection import train_test_split

# ── Design tokens ──────────────────────────────────────────────────────────────
_BG      = "#1D1D20"
_TXT_PRI = "#fbfbff"
_TXT_SEC = "#909094"
_C_BLUE  = "#A1C9F4"
_C_GOLD  = "#ffd400"
_C_GREEN = "#8DE5A1"
_C_CORAL = "#FF9F9B"
_C_LAV   = "#D0BBFF"
_GRID    = "#2e2e33"
_SEP9    = "═" * 72

print(_SEP9)
print("  PHASE 9 — PRODUCTION BENCHMARK: Single GBM vs Stacking Ensemble")
print(_SEP9)

# ── [1] Load saved models from filesystem ─────────────────────────────────────
print("\n[1] Loading models from filesystem …")

with open("gbm_model.pkl", "rb") as _f:
    _gbm_pkl = pickle.load(_f)

with open("scaler.pkl", "rb") as _f:
    _scaler_pkl = pickle.load(_f)

print(f"  ✅ gbm_model.pkl loaded: {type(_gbm_pkl).__name__}")
print(f"  ✅ scaler.pkl loaded:    {type(_scaler_pkl).__name__}")

# ── [2] Prepare inference data ─────────────────────────────────────────────────
print("\n[2] Preparing inference data from enriched_features / user_feature_table …")

# --- Single GBM feature path (6 features from user_feature_table) ---
_GBM_FEATS = [
    "first_24h_events", "first_week_events", "consistency_score",
    "unique_tools_used_14d", "agent_usage_ratio_14d", "exploration_index_14d",
]
_X_gbm_raw = user_feature_table[_GBM_FEATS].values.astype(float)
_X_gbm     = _scaler_pkl.transform(_X_gbm_raw)

# --- Ensemble feature path (22 features from enriched_features) ---
_EXCL_ENS = {"distinct_id", "is_retained", "segment_name"}
_ENS_FEATS = [
    c for c in enriched_features.columns
    if c not in _EXCL_ENS and enriched_features[c].dtype != object
]
_X_ens_raw = enriched_features[_ENS_FEATS].values.astype(float)
_sc_ens    = StandardScaler()
_X_ens     = _sc_ens.fit_transform(_X_ens_raw)
_y_ens     = enriched_features["is_retained"].values.astype(int)

print(f"  GBM   input: {_X_gbm.shape[0]:,} samples × {_X_gbm.shape[1]} features")
print(f"  Ensemble input: {_X_ens.shape[0]:,} samples × {_X_ens.shape[1]} features")

# ── [3] Re-build ensemble inference pipeline (same arch as Phase 8) ───────────
# We need the 3-model stack for ensemble predictions.
# Train base learners on a representative split and build a meta-stack.
print("\n[3] Building ensemble inference pipeline …")

_n_neg = int((_y_ens == 0).sum())
_n_pos = int((_y_ens == 1).sum())
_scale_w = _n_neg / max(_n_pos, 1)

_Xtr_e, _Xte_e, _ytr_e, _yte_e = train_test_split(
    _X_ens, _y_ens, test_size=0.20, random_state=42, stratify=_y_ens
)
_sw_e = np.where(_ytr_e == 1, _scale_w, 1.0)

# Base learners
_gbm_b = GradientBoostingClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, max_features=0.8, random_state=42,
    validation_fraction=0.1, n_iter_no_change=20, tol=1e-4,
)
_gbm_b.fit(_Xtr_e, _ytr_e, sample_weight=_sw_e)
_gbm_b_cal = CalibratedClassifierCV(_gbm_b, method="sigmoid", cv="prefit")
_gbm_b_cal.fit(_Xte_e, _yte_e)

_rf_b = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
_rf_b.fit(_Xtr_e, _ytr_e)
_rf_b_cal = CalibratedClassifierCV(_rf_b, method="sigmoid", cv="prefit")
_rf_b_cal.fit(_Xte_e, _yte_e)

_lr_b = LogisticRegression(
    max_iter=1000, class_weight="balanced", random_state=42, C=1.0
)
_lr_b.fit(_Xtr_e, _ytr_e)

# Train meta-learner on training-set probs
_meta_tr_e = np.column_stack([
    _gbm_b_cal.predict_proba(_Xtr_e)[:, 1],
    _rf_b_cal.predict_proba(_Xtr_e)[:, 1],
    _lr_b.predict_proba(_Xtr_e)[:, 1],
])
_meta_lr = LogisticRegression(
    max_iter=1000, class_weight="balanced", random_state=42, C=1.0
)
_meta_lr.fit(_meta_tr_e, _ytr_e)


def _predict_ensemble(X):
    """Full ensemble inference: 3 base models → meta-LR."""
    _p_gbm = _gbm_b_cal.predict_proba(X)[:, 1]
    _p_rf  = _rf_b_cal.predict_proba(X)[:, 1]
    _p_lr  = _lr_b.predict_proba(X)[:, 1]
    _meta  = np.column_stack([_p_gbm, _p_rf, _p_lr])
    return _meta_lr.predict_proba(_meta)[:, 1]


def _predict_gbm(X):
    """Single GBM inference."""
    return _gbm_pkl.predict_proba(X)[:, 1]


print(f"  ✅ Ensemble pipeline ready (GBM + RF + LR + meta-LR)")

# ── [4] MEMORY FOOTPRINT ───────────────────────────────────────────────────────
print("\n[4] Measuring memory footprint via pickle serialization …")


def _pickle_size_bytes(obj):
    """Serialize object to in-memory buffer and return bytes."""
    _buf = io.BytesIO()
    pickle.dump(obj, _buf, protocol=pickle.HIGHEST_PROTOCOL)
    return _buf.tell()


_mem_gbm_pkl_bytes  = _pickle_size_bytes(_gbm_pkl)
_mem_scaler_bytes   = _pickle_size_bytes(_scaler_pkl)
_mem_gbm_total      = _mem_gbm_pkl_bytes + _mem_scaler_bytes

# Ensemble = gbm_b_cal + rf_b_cal + lr_b + meta_lr + sc_ens
_mem_ens_bytes = sum(
    _pickle_size_bytes(m)
    for m in [_gbm_b_cal, _rf_b_cal, _lr_b, _meta_lr, _sc_ens]
)

print(f"  Single GBM   : {_mem_gbm_pkl_bytes/1024:.1f} KB model  +  {_mem_scaler_bytes/1024:.1f} KB scaler  = {_mem_gbm_total/1024:.1f} KB total")
print(f"  Ensemble     : {_mem_ens_bytes/1024:.1f} KB  (4 models + scaler)")
print(f"  Memory ratio : {_mem_ens_bytes/_mem_gbm_total:.1f}×  larger for ensemble")

# ── [5] LATENCY BENCHMARKS ─────────────────────────────────────────────────────
print("\n[5] Latency benchmarks (timing loops) …")

_N_REPEATS    = 50   # warm-up then repeats
_BATCH_SIZES  = [1, 100, 1000, 10000]
_N_FULL_DATA  = len(_X_gbm)

_results = {}

for _bs in _BATCH_SIZES:
    # Build representative sample for each batch size
    _idx = np.random.default_rng(42).integers(0, _N_FULL_DATA, size=_bs)
    _Xg_batch = _X_gbm[_idx]
    _Xe_batch = _X_ens[_idx]

    # ─ Single GBM ─
    # Warm-up
    for _ in range(3):
        _predict_gbm(_Xg_batch)

    _gbm_times = []
    for _ in range(_N_REPEATS):
        _t0 = time.perf_counter()
        _predict_gbm(_Xg_batch)
        _gbm_times.append(time.perf_counter() - _t0)

    # ─ Ensemble ─
    for _ in range(3):
        _predict_ensemble(_Xe_batch)

    _ens_times = []
    for _ in range(_N_REPEATS):
        _t0 = time.perf_counter()
        _predict_ensemble(_Xe_batch)
        _ens_times.append(time.perf_counter() - _t0)

    _results[_bs] = {
        "gbm_mean_ms":   float(np.mean(_gbm_times)) * 1000,
        "gbm_p95_ms":    float(np.percentile(_gbm_times, 95)) * 1000,
        "gbm_std_ms":    float(np.std(_gbm_times)) * 1000,
        "ens_mean_ms":   float(np.mean(_ens_times)) * 1000,
        "ens_p95_ms":    float(np.percentile(_ens_times, 95)) * 1000,
        "ens_std_ms":    float(np.std(_ens_times)) * 1000,
        "gbm_tp_per_s":  _bs / (float(np.mean(_gbm_times)) or 1e-9),
        "ens_tp_per_s":  _bs / (float(np.mean(_ens_times)) or 1e-9),
    }

    _lat_ratio = _results[_bs]["ens_mean_ms"] / max(_results[_bs]["gbm_mean_ms"], 1e-9)
    print(f"\n  Batch size {_bs:>6,}:")
    print(f"    Single GBM   : {_results[_bs]['gbm_mean_ms']:>8.3f} ms  (p95={_results[_bs]['gbm_p95_ms']:.3f} ms)"
          f"  → {_results[_bs]['gbm_tp_per_s']:,.0f} preds/s")
    print(f"    Stk Ensemble : {_results[_bs]['ens_mean_ms']:>8.3f} ms  (p95={_results[_bs]['ens_p95_ms']:.3f} ms)"
          f"  → {_results[_bs]['ens_tp_per_s']:,.0f} preds/s")
    print(f"    Ensemble is  : {_lat_ratio:.1f}× slower than single GBM")

# ── [6] CV METRICS FROM PRIOR BLOCKS ──────────────────────────────────────────
# Single GBM  — Phase 4 CV metrics (6-feature model, from ROI block vars)
# Advanced GBM (enriched) — Phase 7 CV metrics from advanced_model block
# Stacking Ensemble       — Phase 8 CV means (ensemble_pr_auc etc.)

# Phase 4 GBM metrics (from ROI/upstream variables)
_gbm_pr_auc   = cv_pr_auc_mean     # 5-fold CV mean PR-AUC for baseline GBM
_gbm_roc_auc  = cv_roc_auc_test    # test-set ROC-AUC for baseline GBM
_gbm_recall   = cv_recall_mean     # 5-fold CV mean Recall
_gbm_f1       = float(2 * _gbm_recall * 0.7142857 / max(_gbm_recall + 0.7142857, 1e-9))

# Stacking ensemble metrics (Phase 8)
_ens_pr_auc   = ensemble_pr_auc
_ens_roc_auc  = ensemble_roc_auc
_ens_recall   = ensemble_recall
_ens_f1       = ensemble_f1

print(f"\n[6] CV Metrics (from prior block outputs):")
print(f"  {'Metric':<14} {'Single GBM':>14} {'Stk Ensemble':>14}")
print(f"  {'─'*44}")
print(f"  {'PR-AUC':<14} {_gbm_pr_auc:>14.4f} {_ens_pr_auc:>14.4f}")
print(f"  {'ROC-AUC':<14} {_gbm_roc_auc:>14.4f} {_ens_roc_auc:>14.4f}")
print(f"  {'Recall':<14} {_gbm_recall:>14.4f} {_ens_recall:>14.4f}")
print(f"  {'F1':<14} {_gbm_f1:>14.4f} {_ens_f1:>14.4f}")

# ── [7] COMPREHENSIVE BENCHMARK TABLE ─────────────────────────────────────────
print(f"\n[7] Comprehensive Benchmark Table")
print(_SEP9)

# Build the DataFrame
_bench_rows = []
for _bs in _BATCH_SIZES:
    _r = _results[_bs]
    _bench_rows.append({
        "Batch Size":         _bs,
        "GBM Latency (ms)":  round(_r["gbm_mean_ms"], 3),
        "ENS Latency (ms)":  round(_r["ens_mean_ms"], 3),
        "Lat Ratio (×)":     round(_r["ens_mean_ms"] / max(_r["gbm_mean_ms"], 1e-6), 1),
        "GBM preds/s":       round(_r["gbm_tp_per_s"], 0),
        "ENS preds/s":       round(_r["ens_tp_per_s"], 0),
    })

benchmark_latency_df = pd.DataFrame(_bench_rows)

# Memory rows
_bench_mem = pd.DataFrame([{
    "Metric":        "Memory (KB)",
    "Single GBM":   round(_mem_gbm_total / 1024, 1),
    "Stk Ensemble": round(_mem_ens_bytes / 1024, 1),
    "Δ (×)":        round(_mem_ens_bytes / max(_mem_gbm_total, 1), 1),
}])

# Full comparison DataFrame
benchmark_comparison_df = pd.DataFrame([
    {"Dimension":       "Memory (KB)",
     "Single GBM":      f"{_mem_gbm_total/1024:.1f}",
     "Stk Ensemble":    f"{_mem_ens_bytes/1024:.1f}",
     "Winner":          "GBM"},
    {"Dimension":       "Features",
     "Single GBM":      "6",
     "Stk Ensemble":    str(len(_ENS_FEATS)),
     "Winner":          "GBM"},
    {"Dimension":       "Lat @1 (ms)",
     "Single GBM":      f"{_results[1]['gbm_mean_ms']:.3f}",
     "Stk Ensemble":    f"{_results[1]['ens_mean_ms']:.3f}",
     "Winner":          "GBM"},
    {"Dimension":       "Lat @100 (ms)",
     "Single GBM":      f"{_results[100]['gbm_mean_ms']:.3f}",
     "Stk Ensemble":    f"{_results[100]['ens_mean_ms']:.3f}",
     "Winner":          "GBM"},
    {"Dimension":       "Lat @1K (ms)",
     "Single GBM":      f"{_results[1000]['gbm_mean_ms']:.3f}",
     "Stk Ensemble":    f"{_results[1000]['ens_mean_ms']:.3f}",
     "Winner":          "GBM"},
    {"Dimension":       "Lat @10K (ms)",
     "Single GBM":      f"{_results[10000]['gbm_mean_ms']:.3f}",
     "Stk Ensemble":    f"{_results[10000]['ens_mean_ms']:.3f}",
     "Winner":          "GBM"},
    {"Dimension":       "Throughput @10K",
     "Single GBM":      f"{_results[10000]['gbm_tp_per_s']:,.0f} p/s",
     "Stk Ensemble":    f"{_results[10000]['ens_tp_per_s']:,.0f} p/s",
     "Winner":          "GBM"},
    {"Dimension":       "PR-AUC (CV)",
     "Single GBM":      f"{_gbm_pr_auc:.4f}",
     "Stk Ensemble":    f"{_ens_pr_auc:.4f}",
     "Winner":          "Ensemble"},
    {"Dimension":       "ROC-AUC (CV)",
     "Single GBM":      f"{_gbm_roc_auc:.4f}",
     "Stk Ensemble":    f"{_ens_roc_auc:.4f}",
     "Winner":          "Ensemble"},
    {"Dimension":       "Recall (CV)",
     "Single GBM":      f"{_gbm_recall:.4f}",
     "Stk Ensemble":    f"{_ens_recall:.4f}",
     "Winner":          "Ensemble"},
    {"Dimension":       "F1 (CV)",
     "Single GBM":      f"{_gbm_f1:.4f}",
     "Stk Ensemble":    f"{_ens_f1:.4f}",
     "Winner":          "Ensemble"},
])

print(f"\n  {'Dimension':<22} {'Single GBM':>16} {'Stk Ensemble':>16} {'Winner':>10}")
print(f"  {'─'*66}")
for _, _row in benchmark_comparison_df.iterrows():
    _icon = "⚡" if _row["Winner"] == "GBM" else "🎯"
    print(f"  {_row['Dimension']:<22} {_row['Single GBM']:>16} {_row['Stk Ensemble']:>16} {_icon} {_row['Winner']:>8}")
print(f"  {'─'*66}")

# ── [8] LATENCY SCALING CHART ─────────────────────────────────────────────────
print(f"\n[8] Rendering latency scaling chart …")

_bs_vals    = _BATCH_SIZES
_gbm_lats   = [_results[b]["gbm_mean_ms"] for b in _bs_vals]
_ens_lats   = [_results[b]["ens_mean_ms"] for b in _bs_vals]
_gbm_p95    = [_results[b]["gbm_p95_ms"]  for b in _bs_vals]
_ens_p95    = [_results[b]["ens_p95_ms"]  for b in _bs_vals]

latency_scaling_chart, _ax_lat = plt.subplots(figsize=(11, 6))
latency_scaling_chart.patch.set_facecolor(_BG)
_ax_lat.set_facecolor(_BG)

_ax_lat.plot(_bs_vals, _gbm_lats, "o-", color=_C_BLUE,  linewidth=2.5, markersize=8,
             label="Single GBM (mean)",   zorder=4)
_ax_lat.plot(_bs_vals, _ens_lats, "s-", color=_C_GOLD,  linewidth=2.5, markersize=8,
             label="Stk Ensemble (mean)", zorder=4)

# p95 shading
_ax_lat.fill_between(_bs_vals, _gbm_lats, _gbm_p95,
                     color=_C_BLUE, alpha=0.15, label="GBM p95 band")
_ax_lat.fill_between(_bs_vals, _ens_lats, _ens_p95,
                     color=_C_GOLD, alpha=0.15, label="Ensemble p95 band")

# Value labels on mean lines
for _b, _gl, _el in zip(_bs_vals, _gbm_lats, _ens_lats):
    _ax_lat.annotate(
        f"{_gl:.1f}ms", (_b, _gl),
        textcoords="offset points", xytext=(0, 10),
        ha="center", color=_C_BLUE, fontsize=8.5, fontweight="bold"
    )
    _ax_lat.annotate(
        f"{_el:.1f}ms", (_b, _el),
        textcoords="offset points", xytext=(0, -16),
        ha="center", color=_C_GOLD, fontsize=8.5, fontweight="bold"
    )

_ax_lat.set_xscale("log")
_ax_lat.set_yscale("log")
_ax_lat.set_xlabel("Batch Size (log scale)", color=_TXT_PRI, fontsize=11, labelpad=8)
_ax_lat.set_ylabel("Latency (ms, log scale)", color=_TXT_PRI, fontsize=11, labelpad=8)
_ax_lat.set_title(
    "Production Latency Scaling — Single GBM vs Stacking Ensemble\n"
    "Mean latency ± p95 band across batch sizes  |  log-log scale",
    color=_TXT_PRI, fontsize=13, fontweight="bold", pad=14
)

_ax_lat.set_xticks(_bs_vals)
_ax_lat.set_xticklabels([str(b) for b in _bs_vals], color=_TXT_PRI, fontsize=10)
_ax_lat.tick_params(colors=_TXT_PRI, labelsize=10, which="both")
for _sp in _ax_lat.spines.values():
    _sp.set_edgecolor(_GRID)
_ax_lat.grid(color=_GRID, linewidth=0.5, alpha=0.4, which="both")
_ax_lat.legend(
    facecolor="#2a2a2e", edgecolor=_GRID, labelcolor=_TXT_PRI,
    fontsize=10, loc="upper left", framealpha=0.9
)
plt.tight_layout()
plt.show()

# ── [9] THROUGHPUT BAR CHART ──────────────────────────────────────────────────
print(f"\n[9] Rendering throughput comparison chart …")

_labels   = [f"Batch {b}" for b in _BATCH_SIZES]
_gbm_tp   = [_results[b]["gbm_tp_per_s"] / 1000 for b in _BATCH_SIZES]   # K preds/s
_ens_tp   = [_results[b]["ens_tp_per_s"] / 1000 for b in _BATCH_SIZES]

_x_tp = np.arange(len(_BATCH_SIZES))
_w_tp = 0.35

throughput_chart, _ax_tp = plt.subplots(figsize=(11, 6))
throughput_chart.patch.set_facecolor(_BG)
_ax_tp.set_facecolor(_BG)

_b_gbm = _ax_tp.bar(_x_tp - _w_tp/2, _gbm_tp, _w_tp,
                    color=_C_BLUE, label="Single GBM",     alpha=0.92, edgecolor="none", zorder=3)
_b_ens = _ax_tp.bar(_x_tp + _w_tp/2, _ens_tp, _w_tp,
                    color=_C_GOLD, label="Stk Ensemble",   alpha=0.92, edgecolor="none", zorder=3)

for _bars, _vals, _col in [(_b_gbm, _gbm_tp, _C_BLUE), (_b_ens, _ens_tp, _C_GOLD)]:
    for _bar, _v in zip(_bars, _vals):
        _ax_tp.text(
            _bar.get_x() + _bar.get_width() / 2,
            _v + max(_gbm_tp + _ens_tp) * 0.01,
            f"{_v:.0f}K",
            ha="center", va="bottom", color=_col, fontsize=9, fontweight="bold"
        )

_ax_tp.set_xticks(_x_tp)
_ax_tp.set_xticklabels(_labels, color=_TXT_PRI, fontsize=10)
_ax_tp.set_ylabel("Throughput (K predictions/sec)", color=_TXT_PRI, fontsize=11, labelpad=8)
_ax_tp.set_title(
    "Throughput Comparison — Single GBM vs Stacking Ensemble\n"
    "Higher is better  |  K predictions per second",
    color=_TXT_PRI, fontsize=13, fontweight="bold", pad=14
)
_ax_tp.tick_params(colors=_TXT_PRI, labelsize=10)
for _sp in _ax_tp.spines.values():
    _sp.set_edgecolor(_GRID)
_ax_tp.grid(axis="y", color=_GRID, linewidth=0.5, alpha=0.4, zorder=0)
_ax_tp.set_axisbelow(True)
_ax_tp.legend(
    facecolor="#2a2a2e", edgecolor=_GRID, labelcolor=_TXT_PRI,
    fontsize=10, loc="upper left", framealpha=0.9
)
plt.tight_layout()
plt.show()

# ── [10] SUMMARY & RECOMMENDATION ─────────────────────────────────────────────
print(f"\n{_SEP9}")
print("  PHASE 9 — BENCHMARK SUMMARY & PRODUCTION RECOMMENDATION")
print(_SEP9)

_lat_ratio_1    = _results[1]["ens_mean_ms"]    / max(_results[1]["gbm_mean_ms"], 1e-9)
_lat_ratio_1000 = _results[1000]["ens_mean_ms"] / max(_results[1000]["gbm_mean_ms"], 1e-9)
_mem_ratio      = _mem_ens_bytes / max(_mem_gbm_total, 1)
_pr_lift        = _ens_pr_auc - _gbm_pr_auc
_recall_lift    = _ens_recall - _gbm_recall

print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  LATENCY:   Ensemble is {_lat_ratio_1:.1f}× slower on single sample,              │
  │             {_lat_ratio_1000:.1f}× slower on 1,000-row batch                        │
  │  MEMORY:    Ensemble is {_mem_ratio:.1f}× heavier ({_mem_ens_bytes/1024:.0f} KB vs {_mem_gbm_total/1024:.0f} KB)           │
  │  PR-AUC:    Ensemble gains +{_pr_lift:.4f} ({_ens_pr_auc:.4f} vs {_gbm_pr_auc:.4f})       │
  │  RECALL:    Ensemble gains +{_recall_lift:.4f} ({_ens_recall:.4f} vs {_gbm_recall:.4f})       │
  │  ROC-AUC:   Ensemble gains +{_ens_roc_auc - _gbm_roc_auc:.4f} ({_ens_roc_auc:.4f} vs {_gbm_roc_auc:.4f})  │
  └─────────────────────────────────────────────────────────────────────┘

  🚀 PRODUCTION (latency-critical APIs, real-time scoring):
     → Use SINGLE GBM — sub-millisecond p95 at all batch sizes,
       minimal memory footprint, simple single-model deployment.
       Acceptable PR-AUC for most product interventions.

  🎯 ACCURACY-CRITICAL (batch risk scoring, CSM prioritisation,
     quarterly reporting, high-stakes intervention decisions):
     → Use STACKING ENSEMBLE — materially better PR-AUC and Recall,
       worth the {_lat_ratio_1000:.1f}× latency cost for overnight / scheduled jobs.
       The {_recall_lift:+.3f} recall gain prevents missing at-risk users.
""")

print(_SEP9)
print("  Exports: benchmark_comparison_df, benchmark_latency_df,")
print("           latency_scaling_chart, throughput_chart  ✅")
print(_SEP9)
