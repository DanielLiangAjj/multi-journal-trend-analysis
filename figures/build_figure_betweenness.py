"""Quick bridge-journal betweenness centrality bar chart.

Adds quantitative substrate to the §4.2.4 sub-community identification
which was previously visual-only.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _mirror import mirror_to_main

import json
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "figure_supp_bridge_centrality.png"

ABBR = {
    "Journal of Biomedical Informatics (JBI)": "JBI",
    "JMIR Medical Informatics (JMI)": "JMIR Med Inform",
    "Briefings in Bioinformatics": "Brief Bioinform",
    "International Journal of Medical Informatics (IJMI)": "IJMI",
    "Health Informatics Journal": "Health Inform J",
    "PLOS Digital Health": "PLOS Digit Health",
    "BMC Medical Informatics and Decision Making": "BMC MIDM",
    "Methods of Information in Medicine": "Methods Inf Med",
    "Artificial Intelligence in Medicine": "Artif Intell Med",
    "Journal of the American Medical Informatics Association (JAMIA)": "JAMIA",
    "IEEE Journal of Biomedical and Health Informatics (J-BHI)": "IEEE J-BHI",
    "Journal of Medical Internet Research (JMIR)": "JMIR",
    "Bioinformatics": "Bioinformatics",
    "Computers in Biology and Medicine": "CBM",
    "Nature Methods": "Nat Methods",
    "Nature Medicine": "Nat Med",
    "npj Digital Medicine": "npj Digit Med",
    "Database: The Journal of Biological Databases and Curation": "Database",
    "Journal of Medical Systems": "J Med Syst",
    "JMIR mHealth and uHealth": "JMIR mHealth",
    "Applied Clinical Informatics": "Appl Clin Inform",
    "JAMIA Open": "JAMIA Open",
    "Bioinformatics Advances": "Bioinform Adv",
    "Frontiers in Digital Health": "Front Digit Health",
    "The Lancet Digital Health": "Lancet Digit Health",
    "BMJ Health & Care Informatics": "BMJ HCI",
    "Digital Biomarkers": "Digit Biomark",
    "Journal of Innovation in Health Informatics": "JIHI",
    "Journal of Clinical and Translational Science": "J Clin Transl Sci",
}


def build_network():
    """Reuse the same network construction as Figure 18."""
    M = defaultdict(lambda: defaultdict(int))
    journal_total = Counter()
    topics_combined = {}
    for dom in ["methodology", "health"]:
        topics = json.load(open(ROOT / f"data/hierarchy_k100/{dom}_final_topics.json"))
        kw_to_topics = defaultdict(set)
        for tid, dat in topics.items():
            for kw in dat.get("keywords", []):
                kw_to_topics[kw.strip().lower()].add(f"{dom[0].upper()}:{dat['name']}")
        topics_combined[dom] = kw_to_topics

    for chunk in pd.read_csv(
        ROOT / "data/pubmed_all_journals_2011_2025_w_keywords_patched.csv",
        usecols=["journal", "year", "keywords"], chunksize=50_000,
    ):
        chunk = chunk.dropna(subset=["keywords", "year", "journal"])
        chunk = chunk[chunk["year"].between(2011, 2025)]
        for j, kws in zip(chunk["journal"], chunk["keywords"].astype(str)):
            tset = set()
            for kw in kws.split(";"):
                k = kw.strip().lower()
                for dom in ["methodology", "health"]:
                    if k in topics_combined[dom]:
                        tset |= topics_combined[dom][k]
            journal_total[j] += 1
            for t in tset:
                M[j][t] += 1

    journals = sorted(journal_total.keys())
    all_topics = sorted({t for ctr in M.values() for t in ctr.keys()})
    mat = np.zeros((len(journals), len(all_topics)))
    for i, jn in enumerate(journals):
        denom = max(1, journal_total[jn])
        for k, t in enumerate(all_topics):
            mat[i, k] = M[jn].get(t, 0) / denom
    n = len(journals)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                S[i, j] = np.dot(mat[i], mat[j]) / max(
                    np.linalg.norm(mat[i]) * np.linalg.norm(mat[j]), 1e-12
                )
    thresh = np.percentile(S[np.triu_indices(n, k=1)], 50)
    G = nx.Graph()
    for jn in journals:
        G.add_node(jn, size=journal_total[jn])
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] >= thresh:
                G.add_edge(journals[i], journals[j], weight=float(S[i, j]))
    return G


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

def main():
    G = build_network()
    bc = nx.betweenness_centrality(G, weight=lambda u, v, d: 1 - d["weight"])
    bc_sorted = sorted(bc.items(), key=lambda x: -x[1])

    # Top 12 journals by betweenness centrality
    top = bc_sorted[:12]
    labels = [ABBR.get(j, j[:25]) for j, _ in top]
    values = [b for _, b in top]

    # Color by Louvain community
    import networkx.algorithms.community as nx_com
    communities = nx_com.louvain_communities(G, weight="weight", seed=42)
    communities, COMM_NAME, COMM_COLOUR = name_communities(communities)
    j_to_comm = {j: ci for ci, comm in enumerate(communities) for j in comm}
    colours = [COMM_COLOUR[j_to_comm[j]] for j, _ in top]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    y_pos = np.arange(len(top))
    bars = ax.barh(y_pos, values, color=colours, alpha=0.82,
                   edgecolor="white", linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Betweenness centrality\n(higher = more often on shortest paths between other journal pairs)",
                  fontsize=10)
    ax.set_title(
        "Bridge journals by betweenness centrality\n"
        "Top 12 journals on the topic-overlap similarity network (29 nodes, 203 edges, 50%-percentile edge cut). "
        f"Louvain community detection finds {len(communities)} communities (colour); "
        f"modularity = {nx_com.modularity(G, communities, weight='weight'):.3f}.",
        fontsize=11, fontweight="bold", pad=10,
    )
    # Annotate values
    for i, v in enumerate(values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9, color="#333")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.25, lw=0.5)

    # Legend for Louvain communities
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=COMM_COLOUR[i],
                              label=f"{COMM_NAME[i]}  ({len(communities[i])} journals)")
               for i in range(len(communities))]
    # Legend sits inside the empty lower-right of the plot; anchoring it below the
    # axes made it overprint the two-line x-axis label.
    ax.legend(handles=handles, loc="lower right", fontsize=9, frameon=True,
              framealpha=0.94, borderpad=0.7)

    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(OUT, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {OUT}")
    mirror_to_main(OUT)


if __name__ == "__main__":
    main()
