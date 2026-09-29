"""Figure 14 — journal similarity network with Louvain communities and the JBI bridge.

Replaces data/visualizations_k100/n6_journal_similarity_network.png, in which every
node was the same shade of blue. The caption and §4.2.4 both claim the layout shows
sub-communities and that JBI is the structural bridge; neither was visible.

Network construction is shared with build_figure_betweenness.py so the modularity
and betweenness values quoted in the text come from exactly one graph.
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _mirror import mirror_to_main
from build_figure_betweenness import ABBR, build_network

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_com
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "figure14_journal_similarity_network.png"
CACHE = ROOT / "data" / "journal_similarity_graph.pkl"

JBI = "Journal of Biomedical Informatics (JBI)"

# CVD-safe categorical pair (Okabe-Ito), same family as Figure 2's wave palette.
# ---- community naming (round 23): Louvain on the check-tag-cleaned network yields four weakly separated
# communities; name/colour them by membership so the assignment is stable across reruns ----
COMM_RULES = [("Bioinformatics", "Computational biology", "#0072B2"),
              ("Journal of Medical Internet Research (JMIR)", "Clinical informatics and digital health", "#D55E00"),
              ("Journal of Biomedical Informatics (JBI)", "AI methods and bridging venues", "#009E73"),
              ("Nature Medicine", "General and digital medicine", "#CC79A7")]
def name_communities(communities):
    """return (ordered communities, names, colours) using COMM_RULES; leftovers get grey."""
    ordered, names, cols = [], [], []
    used = set()
    for anchor, nm, col in COMM_RULES:
        for ci, c in enumerate(communities):
            if ci in used: continue
            if anchor in c:
                ordered.append(c); names.append(nm); cols.append(col); used.add(ci); break
    for ci, c in enumerate(communities):
        if ci not in used:
            ordered.append(c); names.append("Other"); cols.append("#8c8c8c")
    return ordered, names, cols

INK, MUTED = "#1a1a1a", "#5c5c5c"


def get_graph():
    if CACHE.exists():
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)
    G = build_network()
    with open(CACHE, "wb") as fh:
        pickle.dump(G, fh)
    return G


def main():
    G = get_graph()
    communities = nx_com.louvain_communities(G, weight="weight", seed=42)
    modularity = nx_com.modularity(G, communities, weight="weight")
    bc = nx.betweenness_centrality(G, weight=lambda u, v, d: 1 - d["weight"])

    # Order communities so the one holding Bioinformatics is drawn first, keeping
    # the colour assignment stable across reruns of the seeded Louvain call.
    communities, COMM_NAME, COMM_COLOUR = name_communities(communities)
    comm_of = {j: ci for ci, c in enumerate(communities) for j in c}

    pos = nx.kamada_kawai_layout(G, weight="weight")

    fig, ax = plt.subplots(figsize=(14, 11))

    # Edges: opacity and width track topic-overlap similarity.
    w = np.array([d["weight"] for *_, d in G.edges(data=True)])
    lo, hi = w.min(), w.max()
    norm = (w - lo) / max(hi - lo, 1e-12)
    for (u, v, d), t in zip(G.edges(data=True), norm):
        same = comm_of[u] == comm_of[v]
        ax.plot(*zip(pos[u], pos[v]), lw=0.35 + 2.0 * t,
                color=COMM_COLOUR[comm_of[u]] if same else "#8c8c8c",
                alpha=0.10 + 0.30 * t, zorder=1, solid_capstyle="round")

    sizes = np.array([G.nodes[n]["size"] for n in G.nodes()])
    node_px = 90 + 2400 * (sizes / sizes.max()) ** 0.62

    for n, s in zip(G.nodes(), node_px):
        is_jbi = n == JBI
        ax.scatter(*pos[n], s=s, color=COMM_COLOUR[comm_of[n]],
                   alpha=0.92 if is_jbi else 0.78,
                   edgecolor="#111111" if is_jbi else "white",
                   linewidth=2.6 if is_jbi else 1.0,
                   zorder=4 if is_jbi else 3)
    # Halo marking the highest-betweenness node.
    ax.scatter(*pos[JBI], s=node_px[list(G.nodes()).index(JBI)] * 3.1,
               facecolor="none", edgecolor="#111111", linewidth=1.5,
               linestyle=(0, (3, 2)), zorder=5)

    for n in G.nodes():
        is_jbi = n == JBI
        ax.annotate(
            ABBR.get(n, n[:22]), pos[n],
            textcoords="offset points", xytext=(0, 13 if is_jbi else 11),
            ha="center", va="bottom",
            fontsize=10.5 if is_jbi else 8.6,
            fontweight="bold" if is_jbi else "normal",
            color=INK, zorder=6,
            bbox=dict(boxstyle="round,pad=0.22", fc="white",
                      ec="#111111" if is_jbi else "none",
                      lw=1.1 if is_jbi else 0, alpha=0.88),
        )

    ax.annotate(
        f"JBI: highest betweenness centrality ({bc[JBI]:.3f})",
        pos[JBI], xytext=(0.79, 0.47), textcoords=ax.transAxes,
        ha="center", va="center", fontsize=9.5, style="italic", color=INK, zorder=7,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#111111", lw=0.9, alpha=0.95),
        arrowprops=dict(arrowstyle="-", color="#111111", lw=0.9,
                        connectionstyle="arc3,rad=0.15"),
    )

    handles = [mpatches.Patch(color=COMM_COLOUR[i], alpha=0.8,
                              label=f"{COMM_NAME[i]}  ({len(communities[i])} journals)")
               for i in range(len(communities))]
    handles += [
        plt.Line2D([], [], marker="o", ls="", markerfacecolor="#bbbbbb",
                   markeredgecolor="white", markersize=ms,
                   label=f"{lab} articles")
        for ms, lab in ((5, "500"), (10, "3,000"), (16, "8,000"))
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=True,
              framealpha=0.94, labelspacing=0.8, borderpad=0.8)

    # in-image figure title removed (caption carries it)
    print(f"  edges={G.number_of_edges()} modularity={modularity:.3f}")
    # modularity footnote removed (round 23, reviewer request); the caption/text carry it
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUT, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {OUT}")
    print(f"  communities={len(communities)} sizes={[len(c) for c in communities]} "
          f"modularity={modularity:.4f} edges={G.number_of_edges()}")
    print(f"  JBI betweenness={bc[JBI]:.4f}  rank="
          f"{sorted(bc.values(), reverse=True).index(bc[JBI]) + 1}")
    for j, b in sorted(bc.items(), key=lambda x: -x[1])[:6]:
        print(f"     {ABBR.get(j, j)[:28]:28s} {b:.4f}  comm{comm_of[j]}")
    mirror_to_main(OUT)


if __name__ == "__main__":
    main()
