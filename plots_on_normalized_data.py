"""
VFT Semantic Cluster IRT Visualizer
====================================
Visualizes within-cluster vs between-cluster IRTs across domains.

Requires:
    - vft_data_normalized.csv  (same folder)
    - cluster_map.py           (same folder)

Usage:
    python irt_cluster_viz.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import warnings
import numpy.ma as ma
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

# # ── import your cluster map ──────────────────────────────────────────────────
# from cluster_map import cluster_map, domain_overrides, get_cluster

from semantic_clusters import cluster_map

# handle cross-domain words (e.g. "orange" = fruit in foods, colour in colours)
domain_overrides = {
    ("foods", "orange"): "fruits",
}

def get_cluster(row):
    key = (row["domain"], row["word_normalized"])
    if key in domain_overrides:
        return domain_overrides[key]
    return cluster_map.get(row["word_normalized"], None)

# ── load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("vft_data_normalized.csv")
df["cluster"] = df.apply(get_cluster, axis=1)
df = df.dropna(subset=["cluster"])          # drop any unmapped words
df = df[df["word_order"] > 1]               # word_order 1 has no meaningful IRT


# ── helpers ──────────────────────────────────────────────────────────────────
def classify_transitions(df_in):
    """
    Labels each row's IRT as 'within' (same cluster as previous word)
    or 'between' (different cluster). Manual loop preserves all columns.
    """
    rows = []
    for (pid, domain), grp in df_in.groupby(["participant_id", "domain"]):
        grp = grp.sort_values("word_order").reset_index(drop=True)
        for i in range(len(grp)):
            row = grp.loc[i].to_dict()
            row["participant_id"] = pid
            row["domain"] = domain
            row["transition"] = (
                None if i == 0
                else ("within" if grp.loc[i, "cluster"] == grp.loc[i - 1, "cluster"]
                      else "between")
            )
            rows.append(row)
    return pd.DataFrame(rows)


# apply across every participant × domain
df = classify_transitions(df)
df = df.dropna(subset=["transition"])

# ── colour palette ────────────────────────────────────────────────────────────
WITHIN_COLOR  = "#1f77b4"   # blue
BETWEEN_COLOR = "#ff7f0e"   # orange
DOMAIN_COLORS = {
    "animals":    "#A8DADC",
    "foods":      "#F4A261",
    "colours":    "#C77DFF",
    "body-parts": "#74C69D",
}
# BG      = "#0F0F14"
# PANEL   = "#1A1A24"
# GRID    = "#2A2A3A"
# TEXT    = "#E8E8F0"
# ACCENT  = "#FFD166"
BG      = "#FFFFFF"   # figure background
PANEL   = "#FFFFFF"   # panel background
GRID    = "#D9D9D9"   # light grid
TEXT    = "#222222"   # dark text
ACCENT  = "#444444"

domains = ["animals", "foods", "colours", "body-parts"]
within  = df[df["transition"] == "within"]
between = df[df["transition"] == "between"]


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Within vs Between IRT per domain  (violin + strip)
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(1, 4, figsize=(18, 7))
fig1.patch.set_facecolor(BG)
fig1.suptitle("Within-Cluster vs Between-Cluster IRTs by Domain",
              fontsize=17, fontweight="bold", color=TEXT, y=1.01)

for ax, domain in zip(axes, domains):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)

    d_within  = within[within["domain"] == domain]["IRT"].clip(upper=30)
    d_between = between[between["domain"] == domain]["IRT"].clip(upper=30)

    data = [d_within, d_between]
    positions = [1, 2]
    colors    = [WITHIN_COLOR, BETWEEN_COLOR]

    # violin
    vp = ax.violinplot(data, positions=positions,
                       showmedians=False, showextrema=False, widths=0.7)
    for body, col in zip(vp["bodies"], colors):
        body.set_facecolor(col)
        body.set_alpha(0.7)
        body.set_edgecolor(col)

    # box + median
    for d, pos, col in zip(data, positions, colors):
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        iqr = q3 - q1
        ax.plot([pos - 0.12, pos + 0.12], [med, med],
                color=col, lw=2.5, zorder=5)
        ax.add_patch(mpatches.FancyBboxPatch(
            (pos - 0.12, q1), 0.24, iqr,
            boxstyle="round,pad=0.01",
            linewidth=1.2, edgecolor=col, facecolor=col, alpha=0.25, zorder=4))
        # jitter
        jitter = np.random.uniform(-0.1, 0.1, size=len(d))
        ax.scatter(pos + jitter, d, s=10, color=col, alpha=0.35, zorder=3)
        # mean marker
        ax.scatter(pos, d.mean(), s=60, color=ACCENT, zorder=6,
                   marker="D", linewidths=0)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Within\nCluster", "Between\nCluster"],
                       color=TEXT, fontsize=10)
    ax.set_title(domain.upper(), color=DOMAIN_COLORS[domain],
                 fontsize=13, fontweight="bold", pad=8)
    ax.set_ylabel("IRT (seconds)" if domain == "animals" else "",
                  color=TEXT, fontsize=10)
    ax.tick_params(colors=TEXT)
    ax.yaxis.grid(True, color=GRID, lw=0.7, linestyle="--")
    ax.set_axisbelow(True)

    # annotate means
    for d, pos, col in zip(data, positions, colors):
        ax.text(pos, ax.get_ylim()[1] * 0.95,
                f"μ={d.mean():.2f}s", ha="center", va="top",
                color=col, fontsize=8.5, fontweight="bold")

legend_elems = [
    mpatches.Patch(facecolor=WITHIN_COLOR,  label="Within Cluster"),
    mpatches.Patch(facecolor=BETWEEN_COLOR, label="Between Cluster"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=ACCENT,
           markersize=7, label="Mean"),
]
fig1.legend(handles=legend_elems, loc="upper center",
            ncol=3, frameon=False, labelcolor=TEXT, fontsize=11,
            bbox_to_anchor=(0.5, -0.01))

plt.tight_layout()
plt.savefig("fig1_within_between_violin.png", dpi=150,
            bbox_inches="tight")
print("Saved fig1_within_between_violin.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Cluster-switch timeline per participant (animals only, sample 12)
# ══════════════════════════════════════════════════════════════════════════════
DOMAIN_VIZ  = "animals"
sample_pids = df[df["domain"] == DOMAIN_VIZ]["participant_id"].unique()[:12]

fig2, axes = plt.subplots(3, 4, figsize=(20, 10))
fig2.patch.set_facecolor(BG)
fig2.suptitle(f"Cluster Retrieval Timeline — {DOMAIN_VIZ.upper()}\n"
              "(bar height = IRT, colour = cluster, ↑ tall = cluster switch)",
              fontsize=15, fontweight="bold", color=TEXT, y=1.02)

axes_flat = axes.flatten()

# assign a distinct colour per cluster
all_clusters = sorted(df[df["domain"] == DOMAIN_VIZ]["cluster"].unique())
cmap_base    = plt.cm.get_cmap("tab20", len(all_clusters))
cluster_color = {c: cmap_base(i) for i, c in enumerate(all_clusters)}

for ax, pid in zip(axes_flat, sample_pids):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)

    sub = (df[(df["participant_id"] == pid) & (df["domain"] == DOMAIN_VIZ)]
           .sort_values("word_order").reset_index(drop=True))

    for _, row in sub.iterrows():
        col  = cluster_color.get(row["cluster"], "#888")
        edge = BETWEEN_COLOR if row["transition"] == "between" else col
        lw   = 2.0           if row["transition"] == "between" else 0.4
        ax.bar(row["word_order"], row["IRT"],
               color=col, edgecolor=edge, linewidth=lw,
               width=0.75, alpha=0.85, zorder=3)
        ax.text(row["word_order"], row["IRT"] + 0.15,
                row["word_normalized"][:4],
                ha="center", va="bottom", fontsize=5.5,
                color=TEXT, rotation=45)

    ax.set_title(f"P-{str(pid)[-4:]}", color=TEXT, fontsize=10)
    ax.tick_params(colors=TEXT, labelsize=7)
    ax.yaxis.grid(True, color=GRID, lw=0.5, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_xlabel("Word Order", color=TEXT, fontsize=8)
    ax.set_ylabel("IRT (s)", color=TEXT, fontsize=8)

# cluster legend
legend_handles = [mpatches.Patch(facecolor=cluster_color[c], label=c)
                  for c in all_clusters]
fig2.legend(handles=legend_handles, loc="lower center",
            ncol=len(all_clusters), frameon=False,
            labelcolor=TEXT, fontsize=8,
            bbox_to_anchor=(0.5, -0.03))

plt.tight_layout()
plt.savefig("fig2_cluster_timeline_animals.png", dpi=150,
            bbox_inches="tight")
print("Saved fig2_cluster_timeline_animals.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Summary bar: mean within vs between IRT, all domains
# + cluster-level mean IRT heatmap
# ══════════════════════════════════════════════════════════════════════════════
fig3 = plt.figure(figsize=(18, 7))
fig3.patch.set_facecolor("white")
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2.2], wspace=0.35)

# ── LEFT: grouped bar ────────────────────────────────────────────────────────
ax_bar = fig3.add_subplot(gs[0])
ax_bar.set_facecolor(PANEL)
for spine in ax_bar.spines.values():
    spine.set_color(GRID)

x      = np.arange(len(domains))
width  = 0.32
w_means = [within[within["domain"]==d]["IRT"].mean() for d in domains]
b_means = [between[between["domain"]==d]["IRT"].mean() for d in domains]
w_sems  = [within[within["domain"]==d]["IRT"].sem() for d in domains]
b_sems  = [between[between["domain"]==d]["IRT"].sem() for d in domains]

bars_w = ax_bar.bar(x - width/2, w_means, width, color=WITHIN_COLOR,
                    alpha=0.85, label="Within", zorder=3)
bars_b = ax_bar.bar(x + width/2, b_means, width, color=BETWEEN_COLOR,
                    alpha=0.85, label="Between", zorder=3)
ax_bar.errorbar(x - width/2, w_means, yerr=w_sems,
                fmt="none", color="white", capsize=4, lw=1.5, zorder=5)
ax_bar.errorbar(x + width/2, b_means, yerr=b_sems,
                fmt="none", color="white", capsize=4, lw=1.5, zorder=5)

# difference annotation
for i, (wm, bm) in enumerate(zip(w_means, b_means)):
    diff = bm - wm
    sign = "+" if diff > 0 else ""
    ax_bar.text(i, max(wm, bm) + 0.4, f"{sign}{diff:.2f}s",
                ha="center", color=ACCENT, fontsize=9, fontweight="bold")

ax_bar.set_xticks(x)
ax_bar.set_xticklabels([d.upper() for d in domains],
                       color=TEXT, fontsize=9, rotation=12)
ax_bar.tick_params(colors=TEXT)
ax_bar.yaxis.grid(True, color=GRID, lw=0.7, linestyle="--")
ax_bar.set_axisbelow(True)
ax_bar.set_ylabel("Mean IRT (seconds)", color=TEXT, fontsize=11)
ax_bar.set_title("Within vs Between\nMean IRT (±SEM)",
                 color=TEXT, fontsize=12, fontweight="bold")
ax_bar.legend(frameon=False, labelcolor=TEXT, fontsize=10)

# ── RIGHT: heatmap — mean IRT per cluster per domain ─────────────────────────
ax_heat = fig3.add_subplot(gs[1])
ax_heat.grid(False)
ax_heat.set_facecolor(PANEL)
for spine in ax_heat.spines.values():
    spine.set_color(GRID)

cluster_irt = (df.groupby(["domain", "cluster"])["IRT"]
               .mean().unstack(level=0))
cluster_irt = cluster_irt.dropna(how="all")

im = ax_heat.imshow(cluster_irt.values, aspect="auto",
                    cmap="YlOrRd", interpolation="nearest")

ax_heat.set_xticks(range(len(cluster_irt.columns)))
ax_heat.set_xticklabels([c.upper() for c in cluster_irt.columns],
                        color=TEXT, fontsize=10)
ax_heat.set_yticks(range(len(cluster_irt.index)))
ax_heat.set_yticklabels(cluster_irt.index, color=TEXT, fontsize=9)
ax_heat.tick_params(colors=TEXT)

# annotate cells
for i in range(cluster_irt.shape[0]):
    for j in range(cluster_irt.shape[1]):
        val = cluster_irt.values[i, j]
        if not np.isnan(val):
            ax_heat.text(j, i, f"{val:.1f}", ha="center", va="center",
                         fontsize=7.5, color="black" if val < 10 else "white")

cbar = fig3.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.03)
cbar.ax.tick_params(colors=TEXT, labelsize=8)
cbar.set_label("Mean IRT (s)", color=TEXT, fontsize=9)
ax_heat.set_title("Mean IRT Heatmap\nby Cluster × Domain",
                  color=TEXT, fontsize=12, fontweight="bold")

plt.savefig("fig3_summary_heatmap.png", dpi=150,
            bbox_inches="tight", facecolor="white")
print("Saved fig3_summary_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — IRT position curve: within vs between, per domain
# ══════════════════════════════════════════════════════════════════════════════
fig4, axes = plt.subplots(2, 2, figsize=(16, 9))
fig4.patch.set_facecolor(BG)
fig4.suptitle("IRT Across Word Position: Within vs Between-Cluster Transitions",
              fontsize=15, fontweight="bold", color=TEXT, y=1.01)

axes_flat = axes.flatten()
MAX_POS = 20

for ax, domain in zip(axes_flat, domains):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)

    for trans, col, lbl in [("within", WITHIN_COLOR, "Within"),
                             ("between", BETWEEN_COLOR, "Between")]:
        d = df[(df["domain"] == domain) & (df["transition"] == trans)
               & (df["word_order"] <= MAX_POS)]
        means = d.groupby("word_order")["IRT"].mean()
        sems  = d.groupby("word_order")["IRT"].sem()

        ax.plot(means.index, means.values, color=col, lw=2.2,
                label=lbl, marker="o", markersize=4, zorder=4)
        ax.fill_between(means.index,
                        means.values - sems.values,
                        means.values + sems.values,
                        color=col, alpha=0.18, zorder=3)

    ax.set_title(domain.upper(), color=DOMAIN_COLORS[domain],
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Word Production Order", color=TEXT, fontsize=10)
    ax.set_ylabel("Mean IRT (s)", color=TEXT, fontsize=10)
    ax.tick_params(colors=TEXT)
    ax.yaxis.grid(True, color=GRID, lw=0.6, linestyle="--")
    ax.xaxis.grid(True, color=GRID, lw=0.4, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=10)

plt.tight_layout()
plt.savefig("fig4_irt_position_curve.png", dpi=150,
            bbox_inches="tight")
print("Saved fig4_irt_position_curve.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Per-participant: cluster switch count vs mean between-IRT
# (scatter — who switches more? does switching cost time?)
# ══════════════════════════════════════════════════════════════════════════════
fig5, axes = plt.subplots(1, 4, figsize=(18, 5))
fig5.patch.set_facecolor(BG)
fig5.suptitle("Cluster Switching Frequency vs Mean Between-Cluster IRT\n"
              "(each dot = one participant)",
              fontsize=14, fontweight="bold", color=TEXT, y=1.02)

for ax, domain in zip(axes, domains):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)

    d = df[df["domain"] == domain]
    stats = d.groupby("participant_id").apply(
        lambda g: pd.Series({
            "n_switches": (g["transition"] == "between").sum(),
            "mean_between_irt": g.loc[g["transition"] == "between", "IRT"].mean(),
            "total_words": len(g),
        })
    ).reset_index()
    stats = stats.dropna()
    stats["switch_rate"] = stats["n_switches"] / stats["total_words"]

    ax.scatter(stats["switch_rate"], stats["mean_between_irt"],
               color=DOMAIN_COLORS[domain], s=80, alpha=0.8,
               edgecolors="white", linewidths=0.5, zorder=4)

    # trend line
    if len(stats) > 3:
        z  = np.polyfit(stats["switch_rate"], stats["mean_between_irt"], 1)
        xr = np.linspace(stats["switch_rate"].min(), stats["switch_rate"].max(), 60)
        ax.plot(xr, np.poly1d(z)(xr), color=ACCENT, lw=1.8,
                linestyle="--", zorder=5)
        r = stats["switch_rate"].corr(stats["mean_between_irt"])
        ax.text(0.97, 0.96, f"r = {r:.2f}", transform=ax.transAxes,
                ha="right", va="top", color=ACCENT, fontsize=10,
                fontweight="bold")

    ax.set_title(domain.upper(), color=DOMAIN_COLORS[domain],
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Switch Rate\n(between-transitions / total)", color=TEXT, fontsize=9)
    ax.set_ylabel("Mean Between-IRT (s)" if domain == "animals" else "",
                  color=TEXT, fontsize=9)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.6, linestyle="--")
    ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("fig5_switch_vs_irt_scatter.png", dpi=150,
            bbox_inches="tight")
print("Saved fig5_switch_vs_irt_scatter.png")


# ── console summary ───────────────────────────────────────────────────────────
print("\n── Quick Stats ──────────────────────────────────────────────────")
for domain in domains:
    w = within[within["domain"]  == domain]["IRT"]
    b = between[between["domain"] == domain]["IRT"]
    diff  = b.mean() - w.mean()
    pct   = (diff / w.mean()) * 100
    print(f"{domain:<12} Within μ={w.mean():.2f}s  Between μ={b.mean():.2f}s  "
          f"Δ={diff:+.2f}s  ({pct:+.1f}%)")

plt.show()
print("\nAll figures saved.")