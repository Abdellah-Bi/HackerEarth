
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ── Design system ─────────────────────────────────────────────────
BG_CLUST      = "#1D1D20"
TXT_PRI       = "#fbfbff"
TXT_SEC       = "#909094"
C_BLUE        = "#A1C9F4"
C_ORANGE      = "#FFB482"
C_GREEN       = "#8DE5A1"
C_CORAL       = "#FF9F9B"
C_LAV         = "#D0BBFF"
C_GOLD        = "#ffd400"
PALETTE       = [C_BLUE, C_ORANGE, C_GREEN, C_CORAL, C_LAV]
DIVIDER_C     = "=" * 70

# ── Feature columns used for clustering ───────────────────────────
CLUST_FEATURE_COLS = [
    "first_24h_events",
    "first_week_events",
    "consistency_score",
    "unique_tools_used_14d",
    "agent_usage_ratio_14d",
    "exploration_index_14d",
]

# ── Prepare feature matrix from user_feature_table ────────────────
clust_df = user_feature_table[["distinct_id"] + CLUST_FEATURE_COLS].copy().dropna()

X_raw = clust_df[CLUST_FEATURE_COLS].values
scaler_clust = StandardScaler()
X_scaled = scaler_clust.fit_transform(X_raw)

print(f"Clustering on {X_scaled.shape[0]:,} users × {X_scaled.shape[1]} features")
print(f"Features: {CLUST_FEATURE_COLS}\n")

# ── Optimal k via Silhouette Score (k = 2 … 6) ────────────────────
K_RANGE = range(2, 7)
sil_scores = {}

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = km.fit_predict(X_scaled)
    sil_scores[k] = silhouette_score(X_scaled, labels_k, sample_size=min(2000, len(labels_k)))

print("Silhouette Scores:")
for k, s in sil_scores.items():
    bar = "█" * int(s * 40)
    print(f"  k={k}  {s:.4f}  {bar}")

optimal_k = max(sil_scores, key=sil_scores.get)
print(f"\n✓ Optimal k = {optimal_k}  (silhouette = {sil_scores[optimal_k]:.4f})")

# ── Fit final KMeans with optimal k ───────────────────────────────
km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = km_final.fit_predict(X_scaled)

clust_df = clust_df.copy()
clust_df["cluster"] = cluster_labels

# ── Cluster Profiling ──────────────────────────────────────────────
cluster_profile = (
    clust_df.groupby("cluster")[CLUST_FEATURE_COLS]
    .mean()
    .round(3)
)
cluster_sizes = clust_df["cluster"].value_counts().sort_index().rename("n_users")
cluster_profile = cluster_profile.join(cluster_sizes)

# ── Descriptive Segment Names ──────────────────────────────────────
grand_mean = clust_df[CLUST_FEATURE_COLS].mean()
grand_std  = clust_df[CLUST_FEATURE_COLS].std().replace(0, 1)

def _segment_name(row_means):
    """Assign a human-readable segment name from feature means."""
    z = (row_means - grand_mean) / grand_std

    consistency  = z["consistency_score"]
    early_events = z["first_24h_events"]
    tools        = z["unique_tools_used_14d"]
    agent        = z["agent_usage_ratio_14d"]
    exploration  = z["exploration_index_14d"]
    weekly       = z["first_week_events"]

    if consistency > 0.5 and weekly > 0.5 and tools > 0.5:
        return "🚀 Power Users"
    if agent > 0.8:
        return "🤖 AI-Native Users"
    if exploration > 0.5 and tools > 0.3:
        return "🔭 Tool Explorers"
    if early_events > 0.0 and consistency < 0.0:
        return "☕ Casual Users"
    if early_events < -0.3 and consistency < -0.3:
        return "⚠️ At-Risk / Passive"
    if consistency > 0.3 and tools < 0.0:
        return "🔧 Focused Builders"
    return "📌 Mixed / Moderate"

segment_names = {}
for c_idx in range(optimal_k):
    row_means = cluster_profile.loc[c_idx, CLUST_FEATURE_COLS]
    segment_names[c_idx] = _segment_name(row_means)

cluster_profile["segment_name"] = pd.Series(segment_names)

# ── Print Cluster Profiles ─────────────────────────────────────────
print(f"\n{DIVIDER_C}")
print("BEHAVIORAL SEGMENTATION — CLUSTER PROFILES")
print(DIVIDER_C)

for c_idx in range(optimal_k):
    _seg  = segment_names[c_idx]
    _n    = int(cluster_profile.loc[c_idx, "n_users"])
    _pct  = _n / len(clust_df) * 100
    _clust_row = cluster_profile.loc[c_idx, CLUST_FEATURE_COLS]
    print(f"\n  Cluster {c_idx}  │  {_seg}  │  {_n:,} users ({_pct:.1f}%)")
    print(f"  {'─'*60}")
    for _feat in CLUST_FEATURE_COLS:
        _val   = _clust_row[_feat]
        _z_val = (_val - grand_mean[_feat]) / grand_std[_feat]
        _tag   = "▲▲" if _z_val > 1 else ("▲" if _z_val > 0.3 else ("▼" if _z_val < -0.3 else " "))
        print(f"    {_feat:<30}  {_val:>8.3f}  {_tag}")

print(f"\n{DIVIDER_C}")

# ── Export behavioral_features dataframe ──────────────────────────
behavioral_features = (
    clust_df[["distinct_id", "cluster"] + CLUST_FEATURE_COLS]
    .copy()
)
behavioral_features["segment_name"] = behavioral_features["cluster"].map(segment_names)

print(f"\n✓ behavioral_features exported — shape: {behavioral_features.shape}")
print(behavioral_features.head(8).to_string(index=False))

# ── Visualisation: Silhouette Scores ──────────────────────────────
fig_sil, ax_sil = plt.subplots(figsize=(8, 4))
fig_sil.patch.set_facecolor(BG_CLUST)
ax_sil.set_facecolor(BG_CLUST)

k_vals  = list(sil_scores.keys())
s_vals  = list(sil_scores.values())
bar_colors = [C_GOLD if k == optimal_k else C_BLUE for k in k_vals]

bars = ax_sil.bar(k_vals, s_vals, color=bar_colors, width=0.55, zorder=3)
for _bar, _bar_val in zip(bars, s_vals):
    ax_sil.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.003,
                f"{_bar_val:.3f}", ha="center", va="bottom",
                color=TXT_PRI, fontsize=10, fontweight="bold")

ax_sil.set_title("Silhouette Score by Number of Clusters",
                 color=TXT_PRI, fontsize=13, fontweight="bold", pad=14)
ax_sil.set_xlabel("k (Number of Clusters)", color=TXT_SEC, fontsize=11)
ax_sil.set_ylabel("Silhouette Score", color=TXT_SEC, fontsize=11)
ax_sil.tick_params(colors=TXT_PRI)
for spine in ax_sil.spines.values():
    spine.set_edgecolor("#333340")
ax_sil.set_xticks(k_vals)
ax_sil.annotate(f"Optimal k={optimal_k}", xy=(optimal_k, sil_scores[optimal_k]),
                xytext=(optimal_k + 0.35, sil_scores[optimal_k] - 0.012),
                color=C_GOLD, fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_GOLD, lw=1.4))

plt.tight_layout()
plt.savefig("silhouette_scores.png", facecolor=BG_CLUST, dpi=150, bbox_inches="tight")

# ── Visualisation: Cluster Size Distribution ──────────────────────
seg_labels = [f"C{c}: {segment_names[c]}" for c in range(optimal_k)]
seg_counts  = [int(cluster_profile.loc[c, "n_users"]) for c in range(optimal_k)]

fig_seg, ax_seg = plt.subplots(figsize=(9, 4.5))
fig_seg.patch.set_facecolor(BG_CLUST)
ax_seg.set_facecolor(BG_CLUST)

bar_colors_seg = PALETTE[:optimal_k]
b_seg = ax_seg.barh(seg_labels, seg_counts, color=bar_colors_seg, height=0.55, zorder=3)
for _seg_bar, _cnt in zip(b_seg, seg_counts):
    ax_seg.text(_seg_bar.get_width() + 8, _seg_bar.get_y() + _seg_bar.get_height() / 2,
                f"{_cnt:,}  ({_cnt/len(clust_df)*100:.1f}%)",
                va="center", color=TXT_PRI, fontsize=10)

ax_seg.set_title("Behavioral Segment Size Distribution",
                 color=TXT_PRI, fontsize=13, fontweight="bold", pad=14)
ax_seg.set_xlabel("Number of Users", color=TXT_SEC, fontsize=11)
ax_seg.tick_params(colors=TXT_PRI, labelsize=10)
for spine in ax_seg.spines.values():
    spine.set_edgecolor("#333340")
ax_seg.invert_yaxis()
plt.tight_layout()
plt.savefig("segment_distribution.png", facecolor=BG_CLUST, dpi=150, bbox_inches="tight")

# ── Visualisation: Feature Heatmap by Segment ─────────────────────
heat_data = cluster_profile[CLUST_FEATURE_COLS].copy()
heat_z = (heat_data - heat_data.mean()) / heat_data.std().replace(0, 1)

fig_heat, ax_heat = plt.subplots(figsize=(10, 1.2 * optimal_k + 2))
fig_heat.patch.set_facecolor(BG_CLUST)
ax_heat.set_facecolor(BG_CLUST)

im = ax_heat.imshow(heat_z.values, aspect="auto", cmap="RdYlGn", vmin=-2, vmax=2)

ax_heat.set_xticks(range(len(CLUST_FEATURE_COLS)))
ax_heat.set_xticklabels(CLUST_FEATURE_COLS, rotation=35, ha="right", color=TXT_PRI, fontsize=10)
ax_heat.set_yticks(range(optimal_k))
ax_heat.set_yticklabels(seg_labels, color=TXT_PRI, fontsize=10)

for r in range(optimal_k):
    for c_col in range(len(CLUST_FEATURE_COLS)):
        val_z = heat_z.values[r, c_col]
        txt_color = "black" if abs(val_z) < 1.2 else TXT_PRI
        ax_heat.text(c_col, r, f"{heat_data.values[r, c_col]:.2f}",
                     ha="center", va="center", fontsize=9, color=txt_color)

cbar = plt.colorbar(im, ax=ax_heat, pad=0.02)
cbar.ax.tick_params(colors=TXT_PRI)
cbar.set_label("Z-score (relative)", color=TXT_SEC, fontsize=10)
ax_heat.set_title("Cluster Feature Profiles (Mean Values, Z-score Shading)",
                  color=TXT_PRI, fontsize=13, fontweight="bold", pad=14)
for spine in ax_heat.spines.values():
    spine.set_edgecolor("#333340")

plt.tight_layout()
plt.savefig("cluster_feature_heatmap.png", facecolor=BG_CLUST, dpi=150, bbox_inches="tight")

print(f"\n✓ All charts saved. Behavioral segmentation complete.")
print(f"  Optimal k = {optimal_k} | Silhouette = {sil_scores[optimal_k]:.4f}")
