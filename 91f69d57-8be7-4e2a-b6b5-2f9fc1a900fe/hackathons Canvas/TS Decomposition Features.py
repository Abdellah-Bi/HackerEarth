"""
TIME-SERIES DECOMPOSITION FEATURES
====================================
Engineers per-user time-series decomposition features from event activity:
  1. trend_slope       — linear trend in daily activity (OLS slope via rolling regression)
  2. seasonality_amp   — amplitude of weekly seasonality (max - min of day-of-week means)
  3. residual_vol      — residual volatility (std of detrended, deseasonalized signal)
  4. trend_r2          — goodness-of-fit of linear trend
  5. peak_to_mean_ratio — max daily events / mean daily events (burst behavior)
  6. activity_entropy  — Shannon entropy of day-of-week event distribution

Uses STL decomposition (statsmodels) for users with ≥14 observations, falling back
to rolling regression + DOW means for users with fewer data points.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ── Column references (consistent with upstream blocks) ──────────────────────
_USER_COL  = "distinct_id"
_TS_COL    = "timestamp"

# ── Zerve design system colors ───────────────────────────────────────────────
_BG        = "#1D1D20"
_TEXT_PRI  = "#fbfbff"
_TEXT_SEC  = "#909094"
_C_BLUE    = "#A1C9F4"
_C_ORANGE  = "#FFB482"
_C_GREEN   = "#8DE5A1"
_C_CORAL   = "#FF9F9B"
_C_GOLD    = "#ffd400"
_GRID_COL  = "#2e2e33"

_SEP = "═" * 70

print(_SEP)
print("  TIME-SERIES DECOMPOSITION FEATURE ENGINEERING")
print(_SEP)

# ── Step 1: Build daily event counts per user ─────────────────────────────────
print("\n[1] Aggregating daily event counts per user...")

_df = user_retention[[_USER_COL, _TS_COL]].copy()
_df[_TS_COL] = pd.to_datetime(_df[_TS_COL], utc=True)
_df["_date"] = _df[_TS_COL].dt.normalize()

# Daily counts per user
_daily = (
    _df.groupby([_USER_COL, "_date"])
    .size()
    .reset_index(name="_events")
)

_all_users = _daily[_USER_COL].unique()
_n_users_total = len(_all_users)
print(f"    Total users: {_n_users_total:,}")
print(f"    Total (user, day) pairs: {len(_daily):,}")

# Global date range for reindexing
_date_min = _daily["_date"].min()
_date_max = _daily["_date"].max()
_all_dates = pd.date_range(_date_min, _date_max, freq="D")
_n_days = len(_all_dates)
print(f"    Date range: {_date_min.date()} → {_date_max.date()} ({_n_days} days)")

# ── Step 2: Per-user decomposition ────────────────────────────────────────────
print("\n[2] Computing decomposition features per user (STL / rolling regression)...")

# Try to import STL
_has_stl = False
try:
    from statsmodels.tsa.seasonal import STL
    _has_stl = True
    print("    STL decomposition available ✓")
except ImportError:
    print("    STL not available — using rolling regression + DOW means")

# Minimum observations for STL (must be > 2 * seasonal_period)
_STL_MIN_OBS = 14
_SEASONAL_PERIOD = 7  # weekly
_t_arr = np.arange(_n_days, dtype=float)

_records = []

for _uid, _grp in _daily.groupby(_USER_COL):
    # Reindex to full date range, fill missing days with 0
    _grp_idx = _grp.set_index("_date")["_events"]
    _ts = _grp_idx.reindex(_all_dates, fill_value=0).values.astype(float)
    _n_obs = len(_ts)
    _n_active = int((_ts > 0).sum())  # days with at least one event

    # ── Trend via OLS (linear regression on time index) ───────────────────────
    _slope, _intercept, _r_val, _p_val, _se = stats.linregress(_t_arr, _ts)
    _trend_fitted = _intercept + _slope * _t_arr
    _trend_slope = float(_slope)
    _trend_r2 = float(_r_val ** 2)

    # ── Seasonality amplitude (day-of-week means) ─────────────────────────────
    _dow = np.arange(_n_obs) % 7  # 0=Mon offset (arbitrary, relative DOW)
    _dow_means = np.array([_ts[_dow == d].mean() if (_dow == d).sum() > 0 else 0.0
                           for d in range(7)])
    _seasonality_amp = float(_dow_means.max() - _dow_means.min())

    # ── Residual volatility ───────────────────────────────────────────────────
    if _has_stl and _n_active >= _STL_MIN_OBS:
        # STL on the full series; use robust=True to handle zero-heavy series
        _stl_res = STL(_ts, period=_SEASONAL_PERIOD, robust=True).fit()
        _residuals = _stl_res.resid
        # Override trend slope with actual STL trend slope
        _stl_trend = _stl_res.trend
        _stl_slope, _, _stl_r, _, _ = stats.linregress(_t_arr, _stl_trend)
        _trend_slope = float(_stl_slope)
        _trend_r2 = float(_stl_r ** 2)
        # Seasonality amplitude from STL seasonal component
        _stl_seasonal = _stl_res.seasonal
        _seasonality_amp = float(_stl_seasonal.max() - _stl_seasonal.min())
    else:
        # Detrend by subtracting OLS fit; deseasonalize by subtracting DOW means
        _detrended = _ts - _trend_fitted
        _deseasonalized = _detrended - np.array([_dow_means[d] for d in _dow])
        _residuals = _deseasonalized

    _residual_vol = float(np.std(_residuals, ddof=1)) if len(_residuals) > 1 else 0.0

    # ── Peak-to-mean ratio ────────────────────────────────────────────────────
    _ts_mean = float(_ts.mean())
    _ts_max = float(_ts.max())
    _peak_to_mean = float(_ts_max / _ts_mean) if _ts_mean > 0 else 0.0

    # ── Activity entropy (DOW distribution) ───────────────────────────────────
    _dow_probs = _dow_means / (_dow_means.sum() + 1e-12)
    _dow_probs_safe = _dow_probs[_dow_probs > 0]
    _activity_entropy = float(-np.sum(_dow_probs_safe * np.log2(_dow_probs_safe))) if len(_dow_probs_safe) > 0 else 0.0

    _records.append({
        "user_id":           _uid,
        "trend_slope":       _trend_slope,
        "trend_r2":          _trend_r2,
        "seasonality_amp":   _seasonality_amp,
        "residual_vol":      _residual_vol,
        "peak_to_mean_ratio": _peak_to_mean,
        "activity_entropy":  _activity_entropy,
        "n_active_days":     _n_active,
        "total_events":      int(_ts.sum()),
        "used_stl":          int(_has_stl and _n_active >= _STL_MIN_OBS),
    })

ts_features = pd.DataFrame(_records)

print(f"\n    ✅ Decomposition complete: {len(ts_features):,} users processed")
print(f"    STL applied to: {ts_features['used_stl'].sum():,} users ({ts_features['used_stl'].mean()*100:.1f}%)")
print(f"    Fallback (OLS+DOW): {(ts_features['used_stl']==0).sum():,} users")

# ── Step 3: Summary statistics ────────────────────────────────────────────────
print(f"\n[3] Feature Summary Statistics")
print("─" * 70)
_feat_cols = ["trend_slope", "seasonality_amp", "residual_vol", "trend_r2",
              "peak_to_mean_ratio", "activity_entropy"]
for _fc in _feat_cols:
    _v = ts_features[_fc]
    print(f"    {_fc:<22} | mean={_v.mean():>8.4f}  std={_v.std():>8.4f}  "
          f"min={_v.min():>8.4f}  max={_v.max():>8.4f}")

print(f"\n    Null values per column:")
for _fc in ts_features.columns:
    _null_n = ts_features[_fc].isnull().sum()
    if _null_n > 0:
        print(f"      {_fc}: {_null_n}")
    else:
        print(f"      {_fc}: 0 ✓")

# ── Step 4: Visualization ────────────────────────────────────────────────────
print("\n[4] Generating feature distribution charts...")

fig_ts_decomp, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=_BG)
fig_ts_decomp.suptitle(
    "Time-Series Decomposition Features — Per-User Distributions",
    color=_TEXT_PRI, fontsize=14, fontweight="bold", y=0.98
)

_feat_config = [
    ("trend_slope",        "Trend Slope\n(events/day)",        _C_BLUE),
    ("seasonality_amp",    "Seasonality Amplitude\n(events)",  _C_ORANGE),
    ("residual_vol",       "Residual Volatility\n(events/day std)", _C_CORAL),
    ("trend_r2",           "Trend R²\n(linearity of activity)", _C_GREEN),
    ("peak_to_mean_ratio", "Peak-to-Mean Ratio\n(burst factor)", _C_GOLD),
    ("activity_entropy",   "Activity Entropy\n(DOW diversity bits)", "#D0BBFF"),
]

for _ax, (_col, _label, _color) in zip(axes.flat, _feat_config):
    _ax.set_facecolor(_BG)
    _vals = ts_features[_col].replace([np.inf, -np.inf], np.nan).dropna()

    # Clip extreme outliers for display (keep 1st–99th percentile)
    _p1, _p99 = np.percentile(_vals, [1, 99])
    _vals_clipped = _vals.clip(_p1, _p99)

    _ax.hist(_vals_clipped, bins=40, color=_color, alpha=0.85, edgecolor=_BG)
    _ax.axvline(_vals.median(), color=_TEXT_PRI, linewidth=1.5,
                linestyle="--", label=f"Median: {_vals.median():.3f}")

    _ax.set_title(_label, color=_TEXT_PRI, fontsize=10, pad=8)
    _ax.set_xlabel("Value", color=_TEXT_SEC, fontsize=8)
    _ax.set_ylabel("Users", color=_TEXT_SEC, fontsize=8)
    _ax.tick_params(colors=_TEXT_SEC, labelcolor=_TEXT_PRI, labelsize=7)
    for _sp in _ax.spines.values():
        _sp.set_edgecolor(_GRID_COL)
    _ax.yaxis.grid(True, color=_GRID_COL, linestyle="--", alpha=0.4)
    _ax.set_axisbelow(True)
    _ax.legend(facecolor="#2a2a2e", edgecolor=_GRID_COL,
                labelcolor=_TEXT_PRI, fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# ── Step 5: Final schema validation ──────────────────────────────────────────
print(f"\n{_SEP}")
print("  ts_features SCHEMA VALIDATION")
print(_SEP)

_required_cols = ["user_id", "trend_slope", "seasonality_amp", "residual_vol"]
_missing_cols = [c for c in _required_cols if c not in ts_features.columns]
assert len(_missing_cols) == 0, f"Missing required columns: {_missing_cols}"

print(f"\n    Shape      : {ts_features.shape}")
print(f"    Columns    : {list(ts_features.columns)}")
print(f"    Users      : {ts_features['user_id'].nunique():,}")
print(f"    Nulls      : {ts_features.isnull().sum().sum()}")
print(f"\n    Sample (first 5 rows):")
print(ts_features[["user_id", "trend_slope", "seasonality_amp", "residual_vol",
                    "trend_r2", "peak_to_mean_ratio", "activity_entropy"]].head().to_string(index=False))

print(f"\n{_SEP}")
print("  ✅ ts_features EXPORT COMPLETE")
print(_SEP)
