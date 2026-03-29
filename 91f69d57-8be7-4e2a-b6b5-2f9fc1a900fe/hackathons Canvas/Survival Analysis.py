"""
PHASE 7 — SURVIVAL ANALYSIS
============================
Kaplan-Meier + Cox Proportional Hazards on user tenure & churn events.
Implemented from scratch (numpy/scipy) — no external survival libraries needed.

Exports: survival_features (user_id + survival-derived columns)
  - tenure_days             : days from signup to last observed event
  - churn_event             : 1=churned, 0=censored
  - km_hazard_score         : 1 - S(T) per user from KM curve
  - surv_prob_7d/14d/30d    : KM population survival at day checkpoints
  - cox_log_hazard          : Cox partial hazard (log scale) per user
  - risk_acceleration_coef  : per-user rate of hazard acceleration

KM curve plotted for overall cohort.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize

# ── Design tokens ─────────────────────────────────────────────────────────────
_BG        = "#1D1D20"
_TEXT_PRI  = "#fbfbff"
_TEXT_SEC  = "#909094"
_C_BLUE    = "#A1C9F4"
_C_ORANGE  = "#FFB482"
_C_GREEN   = "#8DE5A1"
_C_CORAL   = "#FF9F9B"
_GRID_COL  = "#2a2a2e"

SEP7 = "═" * 70
print(SEP7)
print("  PHASE 7 — SURVIVAL ANALYSIS")
print(SEP7)

# ── Step 1: Build per-user survival table ─────────────────────────────────────
_sa = user_retention[["distinct_id", "created_at", "timestamp"]].copy()
_sa["created_at"] = pd.to_datetime(_sa["created_at"], errors="coerce", utc=False)
_sa["timestamp"]  = pd.to_datetime(_sa["timestamp"],  errors="coerce", utc=False)

# Strip timezone for arithmetic
if hasattr(_sa["created_at"].dt, "tz") and _sa["created_at"].dt.tz is not None:
    _sa["created_at"] = _sa["created_at"].dt.tz_localize(None)
if hasattr(_sa["timestamp"].dt, "tz") and _sa["timestamp"].dt.tz is not None:
    _sa["timestamp"] = _sa["timestamp"].dt.tz_localize(None)

_user_obs = (
    _sa.groupby("distinct_id")
    .agg(acct_created=("created_at", "min"), last_event=("timestamp", "max"))
    .reset_index()
)
_tenure_raw = (_user_obs["last_event"] - _user_obs["acct_created"]).dt.total_seconds() / 86400
_user_obs["tenure_days"] = _tenure_raw.clip(lower=0.5)  # floor: avoid degenerate zero durations

_labels   = user_feature_table[["distinct_id", "is_retained"]].copy()
_user_obs = _user_obs.merge(_labels, on="distinct_id", how="inner")
_user_obs["churn_event"] = (1 - _user_obs["is_retained"]).astype(int)

n_sa      = len(_user_obs)
n_churned = int(_user_obs["churn_event"].sum())
print(f"\n[Survival Table] {n_sa:,} users | churned={n_churned} | censored={n_sa - n_churned}")
print(f"[Tenure] min={_user_obs['tenure_days'].min():.1f}d  "
      f"median={_user_obs['tenure_days'].median():.1f}d  "
      f"max={_user_obs['tenure_days'].max():.1f}d")

# ── Step 2: Kaplan-Meier estimator (from scratch) ─────────────────────────────
def kaplan_meier(durations, events):
    """Return (times, S(t)) step function arrays."""
    _df = pd.DataFrame({"t": durations, "e": events}).sort_values("t").reset_index(drop=True)
    _t_list, _s_list = [0.0], [1.0]
    _s = 1.0
    for _t_val, _grp in _df.groupby("t"):
        _n_at_risk = int((_df["t"] >= _t_val).sum())
        _d = int(_grp["e"].sum())
        if _n_at_risk > 0 and _d > 0:
            _s *= (1.0 - _d / _n_at_risk)
        _t_list.append(_t_val)
        _s_list.append(_s)
    return np.array(_t_list), np.array(_s_list)

_km_times, _km_surv = kaplan_meier(
    _user_obs["tenure_days"].values, _user_obs["churn_event"].values
)

def _km_at(t):
    """S(t) from step function."""
    _idx = int(np.searchsorted(_km_times, t, side="right")) - 1
    _idx = max(0, min(_idx, len(_km_surv) - 1))
    return float(_km_surv[_idx])

s7  = _km_at(7)
s14 = _km_at(14)
s30 = _km_at(30)
print(f"\n[KM] S(7d)={s7:.4f}  S(14d)={s14:.4f}  S(30d)={s30:.4f}")

def _km_vec(t_arr):
    """Vectorised KM lookup."""
    _idx = np.searchsorted(_km_times, t_arr, side="right") - 1
    _idx = np.clip(_idx, 0, len(_km_surv) - 1)
    return _km_surv[_idx]

_user_obs["km_surv_at_tenure"] = _km_vec(_user_obs["tenure_days"].values)
_user_obs["km_hazard_score"]   = 1.0 - _user_obs["km_surv_at_tenure"]
_user_obs["surv_prob_7d"]      = s7
_user_obs["surv_prob_14d"]     = s14
_user_obs["surv_prob_30d"]     = s30

# ── Step 3: Cox PH — penalized partial likelihood ─────────────────────────────
_cox_cols = [
    "first_24h_events", "first_week_events", "consistency_score",
    "unique_tools_used_14d", "agent_usage_ratio_14d", "exploration_index_14d",
]
_cox_df = _user_obs.merge(
    user_feature_table[["distinct_id"] + _cox_cols], on="distinct_id", how="left"
)
_cox_df[_cox_cols] = _cox_df[_cox_cols].fillna(0)

_T  = _cox_df["tenure_days"].values
_E  = _cox_df["churn_event"].values
_X  = _cox_df[_cox_cols].values.astype(float)

# Standardise for numeric stability
_X_mean = _X.mean(axis=0)
_X_std  = np.where(_X.std(axis=0) > 1e-8, _X.std(axis=0), 1.0)
_Xs     = (_X - _X_mean) / _X_std

PENALIZER = 0.1

def _neg_cox_pll(beta):
    """Negative penalized partial log-likelihood."""
    xb  = _Xs @ beta
    pll = 0.0
    for _ii in range(len(_T)):
        if _E[_ii] == 1:
            _rs = xb[_T >= _T[_ii]]
            _mx = _rs.max()
            pll += xb[_ii] - (_mx + np.log(np.sum(np.exp(_rs - _mx))))
    return -(pll - (PENALIZER / 2) * float(beta @ beta))

_result   = minimize(_neg_cox_pll, np.zeros(_Xs.shape[1]),
                     method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-9})
_beta_hat = _result.x

print("\n[CoxPH] Converged:", _result.success)
print(f"  {'Feature':<30}  {'Beta':>10}  {'HR (exp(B))':>12}")
print(f"  {'─'*30}  {'─'*10}  {'─'*12}")
for _nm, _b in zip(_cox_cols, _beta_hat):
    print(f"  {_nm:<30}  {_b:>10.4f}  {np.exp(_b):>12.4f}")

_user_obs["cox_log_hazard"] = _Xs @ _beta_hat

# ── Step 4: Risk Acceleration Coefficient ─────────────────────────────────────
_H7  = float(np.maximum(-np.log(max(s7,  1e-9)), 1e-9))
_H14 = float(np.maximum(-np.log(max(s14, 1e-9)), 1e-9))
_H30 = float(np.maximum(-np.log(max(s30, 1e-9)), 1e-9))

_pts     = np.array([7.0, 14.0, 30.0])
_log_H   = np.array([np.log(_H7), np.log(_H14), np.log(_H30)])
_b_coef  = float(np.polyfit(_pts, _log_H, 1)[0])   # slope = risk acceleration
print(f"\n[Risk Acceleration] population log-H slope = {_b_coef:.6f} per day")

_cx_mean = float(_user_obs["cox_log_hazard"].mean())
_cx_std  = float(max(_user_obs["cox_log_hazard"].std(), 1e-6))
_user_obs["risk_acceleration_coef"] = _b_coef * np.exp(
    (_user_obs["cox_log_hazard"] - _cx_mean) / _cx_std
)

# ── Step 5: Export survival_features ─────────────────────────────────────────
survival_features = (
    _user_obs[[
        "distinct_id", "tenure_days", "churn_event",
        "km_hazard_score", "surv_prob_7d", "surv_prob_14d", "surv_prob_30d",
        "cox_log_hazard", "risk_acceleration_coef",
    ]]
    .rename(columns={"distinct_id": "user_id"})
    .copy()
)

print(f"\n{SEP7}")
print(f"  survival_features  shape: {survival_features.shape}")
print(SEP7)
print(survival_features.describe().round(4).to_string())
print(f"\n  ✅ survival_features exported: {len(survival_features):,} users, "
      f"{survival_features.shape[1]} columns")
print(f"  Columns: {list(survival_features.columns)}")

# ── Step 6: KM Curve Plot ─────────────────────────────────────────────────────
matplotlib.rcParams["font.family"] = "DejaVu Sans"
fig_km, ax_km = plt.subplots(figsize=(10, 6), facecolor=_BG)
ax_km.set_facecolor(_BG)

# Step curve
ax_km.step(_km_times, _km_surv, where="post",
           color=_C_BLUE, linewidth=2.5)

# Greenwood CI
_df_gw   = pd.DataFrame({"t": _user_obs["tenure_days"].values,
                          "e": _user_obs["churn_event"].values}).sort_values("t")
_gw_sum  = 0.0
_ci_lo_l, _ci_hi_l, _ci_t_l = [], [], []
for _t_val, _grp in _df_gw.groupby("t"):
    _n = int((_df_gw["t"] >= _t_val).sum())
    _d = int(_grp["e"].sum())
    if _n > _d and _d > 0:
        _gw_sum += _d / (_n * (_n - _d))
    _s_val = _km_at(_t_val)
    if _s_val > 0 and _s_val < 1:
        _lls  = np.log(-np.log(_s_val))
        _se   = 1.96 * np.sqrt(_gw_sum) / abs(np.log(_s_val) + 1e-12)
        _ci_lo_l.append(np.exp(-np.exp(_lls + _se)))
        _ci_hi_l.append(np.exp(-np.exp(_lls - _se)))
        _ci_t_l.append(_t_val)

if _ci_t_l:
    ax_km.fill_between(_ci_t_l, _ci_lo_l, _ci_hi_l, step="post",
                       alpha=0.18, color=_C_BLUE)

# Checkpoint crosshairs
for _t, _s, _col, _lbl in [
    (7,  s7,  _C_ORANGE, f"S(7d) = {s7:.3f}"),
    (14, s14, _C_GREEN,  f"S(14d) = {s14:.3f}"),
    (30, s30, _C_CORAL,  f"S(30d) = {s30:.3f}"),
]:
    ax_km.vlines(_t, 0, _s, colors=_col, linestyles="--", linewidth=1.3, alpha=0.85)
    ax_km.hlines(_s, 0, _t, colors=_col, linestyles="--", linewidth=1.3, alpha=0.85)
    ax_km.scatter([_t], [_s], color=_col, s=75, zorder=5)
    ax_km.text(_t + 0.8, _s + 0.015, _lbl, color=_col, fontsize=9.5)

ax_km.set_xlim(left=0)
ax_km.set_ylim(0, 1.05)
ax_km.set_xlabel("Days since account creation", color=_TEXT_PRI, fontsize=11)
ax_km.set_ylabel("Survival probability  S(t)", color=_TEXT_PRI, fontsize=11)
ax_km.set_title("Kaplan-Meier Survival Curve — User Churn",
                color=_TEXT_PRI, fontsize=14, pad=14, fontweight="bold")
ax_km.tick_params(colors=_TEXT_SEC, labelsize=9)
for _sp in ax_km.spines.values():
    _sp.set_edgecolor(_GRID_COL)
ax_km.yaxis.grid(True, color=_GRID_COL, linewidth=0.7)
ax_km.set_axisbelow(True)

ax_km.text(0.98, 0.97,
           f"n = {n_sa:,} users\nChurned: {n_churned} | Censored: {n_sa - n_churned}",
           transform=ax_km.transAxes, ha="right", va="top", color=_TEXT_SEC, fontsize=9,
           bbox=dict(boxstyle="round,pad=0.4", facecolor=_BG, edgecolor=_GRID_COL))

_p1 = mpatches.Patch(color=_C_BLUE, label="KM Survival Curve")
_p2 = mpatches.Patch(color=_C_BLUE, alpha=0.25, label="95% CI (Greenwood)")
ax_km.legend(handles=[_p1, _p2], facecolor=_BG, edgecolor=_GRID_COL,
             labelcolor=_TEXT_PRI, fontsize=9.5)

plt.tight_layout()
plt.show()
print("\n✅ KM curve plotted successfully.")
