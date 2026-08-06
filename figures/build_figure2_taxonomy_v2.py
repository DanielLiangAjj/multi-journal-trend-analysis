"""Figure 2 v2: hierarchical agglomerative taxonomy of K=100 clusters.

Replaces the sparse GPT-derived parent-child taxonomy (10/33 edges, 60+ orphans)
with a complete Ward-linkage dendrogram on cluster centroids, so every cluster
is a connected leaf and the multi-level theme structure is visible.
"""
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parent.parent
MAIN_MIRROR = Path("/Users/danielliang/Library/CloudStorage/Dropbox/multi_journal_trend_analysis")
N_SUPER_BY_DOMAIN = {"methodology": 6, "health": 7}
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#17becf"]


def load_domain(domain: str):
    """Aggregate the 100 cluster centroids by their dominant topic name,
    weighted by cluster keyword count. Returns one centroid per unique topic."""
    cent = pickle.load(open(ROOT / f"data/clusters_k100/{domain}_centroids.pkl", "rb"))
    topics_raw = json.load(open(ROOT / f"data/topics_k100/{domain}_topic_names.json"))
    sizes = pd.read_csv(ROOT / f"data/clusters_k100/{domain}_cluster_summary.csv").set_index("cluster_id")["size"]

    cluster_ctr = defaultdict(Counter)
    for key, name in topics_raw.items():
        cid = int(key.split("-")[0])
        cluster_ctr[cid][name] += 1
    dom_topic = {cid: ctr.most_common(1)[0][0] for cid, ctr in cluster_ctr.items()}

    # Aggregate centroids and sizes by topic name (weighted mean)
    topic_vec = defaultdict(lambda: np.zeros(cent.shape[1]))
    topic_w = defaultdict(int)
    for cid, name in dom_topic.items():
        w = int(sizes.loc[cid])
        topic_vec[name] += w * cent[cid]
        topic_w[name] += w
    topics = sorted(topic_vec.keys())
    centroids = np.array([topic_vec[n] / max(1, topic_w[n]) for n in topics], dtype=float)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    sizes_arr = np.array([topic_w[n] for n in topics])
    return topics, centroids, topics, sizes_arr


def super_cluster_label(member_topics, member_sizes):
    """Pick a short label for a super-cluster from its biggest constituent topic."""
    weight = defaultdict(int)
    for name, sz in zip(member_topics, member_sizes):
        weight[name] += sz
    return max(weight.items(), key=lambda x: x[1])[0]


def build(domain: str, out_path: Path):
    cids, centroids, labels, sizes_arr = load_domain(domain)
    n = len(cids)
    n_super = N_SUPER_BY_DOMAIN[domain]

    Z = linkage(centroids, method="ward", metric="euclidean")
    super_ids = fcluster(Z, t=n_super, criterion="maxclust")

    # Dendrogram leaf order
    ddata = dendrogram(Z, no_plot=True, color_threshold=-1)
    leaf_order = ddata["leaves"]

    # Super-cluster colour per leaf, then per link
    leaf_super = super_ids[leaf_order]
    leaf_colour = [PALETTE[(int(s) - 1) % len(PALETTE)] for s in leaf_super]

    # Build link-colour list using super-cluster of leftmost descendant
    n_internal = Z.shape[0]
    leaf_set_per_node = {i: {i} for i in range(n)}
    for k, (a, b, _, _) in enumerate(Z):
        leaf_set_per_node[n + k] = leaf_set_per_node[int(a)] | leaf_set_per_node[int(b)]
    link_colour = {}
    for k in range(n_internal):
        members = leaf_set_per_node[n + k]
        # super-cluster vote among members
        votes = Counter(super_ids[m] for m in members)
        winner, _ = votes.most_common(1)[0]
        all_same = len(votes) == 1
        link_colour[n + k] = PALETTE[(int(winner) - 1) % len(PALETTE)] if all_same else "#888888"

    def _link_color(node_id):
        return link_colour.get(node_id, "#888888")

    # Figure layout: horizontal dendrogram on the left, leaf labels on the right.
    # Reserve top for title and bottom for legend so neither collides with axes.
    row_h = 0.32
    height = max(11, row_h * n + 4.5)
    fig = plt.figure(figsize=(15, height))
    title_band = 0.9 / height
    legend_band = (1.4 + 0.25 * ((len(set(super_ids)) + 1) // 2)) / height
    body_top = 1.0 - title_band
    body_bot = legend_band
    body_h = body_top - body_bot
    ax = fig.add_axes([0.05, body_bot, 0.32, body_h])
    ax_lab = fig.add_axes([0.39, body_bot, 0.58, body_h])

    dendrogram(
        Z,
        orientation="left",
        labels=None,
        no_labels=True,
        link_color_func=_link_color,
        ax=ax,
        above_threshold_color="#888888",
    )
    ax.invert_yaxis()
    ax.set_xlabel("Ward linkage distance", fontsize=10)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

    # Right panel: leaf labels colour-coded by super-cluster
    ax_lab.set_xlim(0, 1)
    ax_lab.set_ylim(0, n)
    ax_lab.invert_yaxis()
    ax_lab.axis("off")

    # Display labels in dendrogram leaf order (top to bottom)
    for i, leaf_idx in enumerate(leaf_order):
        name = labels[leaf_idx]
        size = sizes_arr[leaf_idx]
        colour = leaf_colour[i]
        y = i + 0.5
        # colour swatch
        ax_lab.add_patch(mpatches.Rectangle((0.005, y - 0.35), 0.018, 0.7,
                                            color=colour, linewidth=0))
        ax_lab.text(0.060, y, f"{name}", fontsize=9.5, va="center", color="#222")
        ax_lab.text(0.985, y, f"{size:,}", fontsize=8.5, va="center",
                    ha="right", color="#666", family="monospace")

    # ---- Super-cluster bracket annotations on the FAR-RIGHT of the labels ----
    # Place them after the n_keywords column so nothing overlaps.
    bracket_x_left = 1.005
    bracket_x_right = 1.060
    label_x = 1.080
    ax_lab.set_xlim(0, 1.20)  # widen so brackets fit
    i = 0
    while i < n:
        sid = leaf_super[i]
        j = i
        while j < n and leaf_super[j] == sid:
            j += 1
        # group covers leaves [i, j-1]
        y_top = i + 0.10
        y_bot = (j - 1) + 0.90
        col = PALETTE[(int(sid) - 1) % len(PALETTE)]
        members = [labels[leaf_order[k]] for k in range(i, j)]
        member_sizes = [sizes_arr[leaf_order[k]] for k in range(i, j)]
        super_lab = super_cluster_label(members, member_sizes)
        # Filled coloured bracket
        ax_lab.add_patch(mpatches.Rectangle(
            (bracket_x_left, y_top),
            bracket_x_right - bracket_x_left, y_bot - y_top,
            facecolor=col, alpha=0.55, edgecolor=col, lw=1.4,
            clip_on=False, zorder=4))
        # Super-cluster label, rotated, white-on-colour for contrast
        if (j - i) >= 2:
            ax_lab.text((bracket_x_left + bracket_x_right) / 2,
                        (y_top + y_bot) / 2, super_lab,
                        rotation=90, fontsize=9, ha="center", va="center",
                        color="white", fontweight="bold",
                        clip_on=False, zorder=5)
        else:
            # single-topic super-cluster: place a small label to the FAR RIGHT
            # (past the bracket) so it doesn't crowd the topic-name row
            ax_lab.text(bracket_x_right + 0.02, (y_top + y_bot) / 2,
                        f"  ← {super_lab}",
                        fontsize=7, ha="left", va="center", color=col,
                        style="italic", clip_on=False, zorder=5)
        i = j

    # Header sits inside the labels axes, slightly above the first leaf
    ax_lab.text(0.035, -0.9, "Topic", fontsize=10.5, fontweight="bold", color="#333")
    ax_lab.text(0.98, -0.9, "n keywords", fontsize=10.5, fontweight="bold",
                color="#333", ha="right")

    # Super-cluster legend
    super_groups = defaultdict(list)
    super_size = defaultdict(int)
    for i, sid in enumerate(super_ids):
        super_groups[sid].append(labels[i])
        super_size[sid] += int(sizes_arr[i])
    legend_items = []
    for sid in sorted(super_groups):
        super_label = super_cluster_label(super_groups[sid],
                                          [sizes_arr[i] for i in range(n) if super_ids[i] == sid])
        n_topics = len(super_groups[sid])
        legend_items.append((sid, super_label, n_topics, super_size[sid]))
    legend_items.sort(key=lambda x: -x[3])

    legend_handles = [
        mpatches.Patch(color=PALETTE[(int(sid) - 1) % len(PALETTE)],
                       label=f"{lbl}  ({n_t} topics, {sz:,} keywords)")
        for sid, lbl, n_t, sz in legend_items
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, 0.012), frameon=False,
               fontsize=10, title=f"{n_super} super-clusters (Ward cut)",
               title_fontsize=10.5)

    domain_label = "Methodology" if domain == "methodology" else "Health"
    fig_no = "8" if domain == "methodology" else "9"
    fig.text(0.5, 1.0 - 0.4 * title_band,
             f"Figure {fig_no}. {domain_label} topic taxonomy (K=100, Ward agglomerative clustering)",
             fontsize=14, fontweight="bold", ha="center", va="top")
    fig.text(0.5, 1.0 - 0.85 * title_band,
             f"{n} unique topics  •  {n_super} super-clusters at the Ward cut  •  link colour = super-cluster of leftmost descendant",
             fontsize=10.5, ha="center", va="top", color="#444")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {out_path}  ({n} unique topic leaves)")

    # Mirror to MAIN Dropbox folder so the user's Dropbox UI shows the new file.
    mirror = MAIN_MIRROR / "figures" / out_path.name
    if mirror.parent.exists():
        import shutil
        shutil.copy2(out_path, mirror)
        print(f"  mirrored to {mirror}")


def main():
    out_dir = ROOT / "figures"
    for d in ("methodology", "health"):
        out = out_dir / f"figure2_{d}_taxonomy.png"
        build(d, out)


if __name__ == "__main__":
    main()
