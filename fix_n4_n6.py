"""
Fix N4 (topic emergence) and N6 (journal similarity network) visualizations.

N4 fix: redefine "emergence" as the first year a topic became established
        (>= 50 articles/year), rather than the first year it had any article.
        This produces a meaningful timeline showing which topics became
        established early vs. recently.

N6 fix: use clean journal abbreviations, larger figure, kamada-kawai layout
        with improved spacing for readable labels.
"""

import csv
import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = os.path.expanduser("~/Dropbox/multi_journal_trend_analysis/data")
VIZ_DIR = os.path.join(DATA_DIR, "visualizations")
HIER_DIR = os.path.join(DATA_DIR, "hierarchy")
ARTICLES_CSV = os.path.join(DATA_DIR, "pubmed_all_journals_2011_2025_w_keywords.csv")

YEARS = list(range(2011, 2026))

# Clean journal abbreviations
JOURNAL_ABBR = {
    "Journal of the American Medical Informatics Association (JAMIA)": "JAMIA",
    "Journal of Medical Internet Research (JMIR)": "JMIR",
    "IEEE Journal of Biomedical and Health Informatics (J-BHI)": "IEEE J-BHI",
    "Journal of Biomedical Informatics (JBI)": "JBI",
    "International Journal of Medical Informatics (IJMI)": "IJMI",
    "JMIR Medical Informatics (JMI)": "JMIR Med Inform",
    "npj Digital Medicine": "npj Digit Med",
    "The Lancet Digital Health": "Lancet Digit Health",
    "Briefings in Bioinformatics": "Brief Bioinform",
    "Bioinformatics": "Bioinformatics",
    "JAMIA Open": "JAMIA Open",
    "BMC Medical Informatics and Decision Making": "BMC Med Inform",
    "PLOS Digital Health": "PLOS Digit Health",
    "Journal of Medical Systems": "J Med Syst",
    "Methods of Information in Medicine": "Methods Inf Med",
    "BMJ Health & Care Informatics": "BMJ Health Care",
    "Health Informatics Journal": "Health Inform J",
    "Applied Clinical Informatics": "Appl Clin Inform",
    "JMIR mHealth and uHealth": "JMIR mHealth",
    "Computers in Biology and Medicine": "Comput Biol Med",
    "Artificial Intelligence in Medicine": "Artif Intell Med",
    "Journal of Clinical and Translational Science": "J Clin Transl Sci",
    "Database: The Journal of Biological Databases and Curation": "Database",
    "Bioinformatics Advances": "Bioinform Adv",
    "Journal of Innovation in Health Informatics": "J Innov Health Inform",
    "Digital Biomarkers": "Digit Biomark",
    "Frontiers in Digital Health": "Front Digit Health",
    "Nature Medicine": "Nat Med",
    "Nature Methods": "Nat Methods",
}


# ---------------------------------------------------------------------------
# N4 FIX: Topic Emergence (defined as first year reaching threshold)
# ---------------------------------------------------------------------------

def plot_emergence_fixed(domain, threshold=50, top_n=30):
    """Plot when each top topic became 'established' (>= threshold articles/year)."""
    csv_path = os.path.join(VIZ_DIR, f"{domain}_topic_year_counts.csv")
    df = pd.read_csv(csv_path)

    # For each topic, find the first year it reached the threshold
    rows = []
    for _, row in df.iterrows():
        topic = row["topic"]
        total = row["total"]
        # Find first year >= threshold
        emergence_year = None
        for y in YEARS:
            if row[str(y)] >= threshold:
                emergence_year = y
                break
        if emergence_year is None:
            continue
        rows.append({
            "topic": topic,
            "total": total,
            "emergence_year": emergence_year,
        })

    if not rows:
        print(f"  No topics found for {domain}")
        return

    emerge_df = pd.DataFrame(rows)
    # Take top N by total
    emerge_df = emerge_df.nlargest(top_n, "total")
    # Sort by emergence year (newest first at top of plot)
    emerge_df = emerge_df.sort_values("emergence_year", ascending=False)

    # Color by emergence year
    norm = (emerge_df["emergence_year"] - YEARS[0]) / (YEARS[-1] - YEARS[0])
    colors = plt.cm.viridis(norm)

    fig, ax = plt.subplots(figsize=(14, max(8, top_n * 0.35)))
    ax.barh(
        emerge_df["topic"],
        emerge_df["emergence_year"] - YEARS[0],
        left=YEARS[0],
        color=colors,
        edgecolor="white",
    )
    for i, row in enumerate(emerge_df.itertuples()):
        ax.text(
            row.emergence_year + 0.15,
            i,
            f"{row.emergence_year} ({int(row.total):,} articles total)",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel(
        f"First Year Topic Reached >= {threshold} Articles/Year",
        fontsize=13,
    )
    ax.set_title(
        f"{domain.title()} Topic Establishment Timeline\n"
        f"(when each topic became established at >= {threshold} articles/year)",
        fontsize=15,
    )
    ax.set_xlim(YEARS[0] - 0.5, YEARS[-1] + 3)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()

    out_path = os.path.join(VIZ_DIR, f"n4_{domain}_topic_emergence.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# N6 FIX: Journal Similarity Network with readable labels
# ---------------------------------------------------------------------------

def plot_similarity_network_fixed():
    """Re-plot the journal similarity network with clean abbreviations and layout."""
    import networkx as nx
    from sklearn.metrics.pairwise import cosine_similarity

    # Load the topic mapping data we need
    print("  Loading articles CSV...")
    df = pd.read_csv(ARTICLES_CSV, quoting=csv.QUOTE_ALL)

    print("  Loading methodology topics...")
    with open(os.path.join(HIER_DIR, "methodology_final_topics.json"), "r") as f:
        method_topics = json.load(f)

    # Build keyword -> topic name map
    kw_to_topic = defaultdict(set)
    for tid, data in method_topics.items():
        for kw in data.get("keywords", []):
            kw_to_topic[kw.strip().lower()].add(data["name"])

    # For each journal, count articles per topic
    print("  Mapping articles to topics...")
    journal_topic = defaultdict(Counter)
    journal_total = Counter()

    for _, row in df.iterrows():
        kw_str = row.get("keywords", "")
        if pd.isna(kw_str) or not kw_str:
            continue
        journal = row.get("journal", "unknown")
        journal_total[journal] += 1
        topics_in_article = set()
        for kw in str(kw_str).split(";"):
            kw = kw.strip().lower()
            if kw in kw_to_topic:
                topics_in_article.update(kw_to_topic[kw])
        for t in topics_in_article:
            journal_topic[journal][t] += 1

    # Build topic-distribution vectors per journal
    all_topics = sorted({t for j in journal_topic.values() for t in j})
    journals = sorted(journal_topic.keys())

    vectors = []
    for j in journals:
        total = journal_total[j]
        vec = np.array([journal_topic[j].get(t, 0) / max(total, 1) for t in all_topics])
        vectors.append(vec)
    vectors = np.array(vectors)

    sim = cosine_similarity(vectors)

    # Build network
    G = nx.Graph()
    for j in journals:
        abbr = JOURNAL_ABBR.get(j, j[:20])
        G.add_node(abbr, size=journal_total[j])

    # Connect journals with similarity above 70th percentile
    upper = sim[np.triu_indices_from(sim, k=1)]
    threshold = np.percentile(upper, 70)
    for i in range(len(journals)):
        for j in range(i + 1, len(journals)):
            if sim[i, j] > threshold:
                a = JOURNAL_ABBR.get(journals[i], journals[i][:20])
                b = JOURNAL_ABBR.get(journals[j], journals[j][:20])
                G.add_edge(a, b, weight=float(sim[i, j]))

    print(f"  Network: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # Use kamada-kawai layout for better spacing
    pos = nx.kamada_kawai_layout(G)

    fig, ax = plt.subplots(figsize=(20, 16))

    # Node sizes proportional to article count
    sizes = [G.nodes[n].get("size", 100) / 4 for n in G.nodes]
    weights = [G[u][v]["weight"] * 4 for u, v in G.edges]

    nx.draw_networkx_edges(
        G, pos, width=weights, alpha=0.4, edge_color="#7f8c8d", ax=ax,
    )
    nx.draw_networkx_nodes(
        G, pos,
        node_color="#5dade2",
        node_size=sizes,
        edgecolors="#1f618d",
        linewidths=2,
        alpha=0.9,
        ax=ax,
    )

    # Draw labels with white background for readability
    label_pos = {n: (p[0], p[1] + 0.04) for n, p in pos.items()}
    for node, (x, y) in label_pos.items():
        ax.text(
            x, y, node,
            ha="center", va="center",
            fontsize=11, fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor="#1f618d",
                alpha=0.9,
            ),
        )

    ax.set_title(
        "Journal Similarity Network\n"
        "(based on methodology topic distributions; edge thickness = similarity)",
        fontsize=18,
    )
    ax.axis("off")
    ax.margins(0.15)
    plt.tight_layout()

    out_path = os.path.join(VIZ_DIR, "n6_journal_similarity_network.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Fixing N4: Topic Emergence (using >= 50 articles/year threshold)...")
    plot_emergence_fixed("methodology", threshold=50, top_n=30)
    plot_emergence_fixed("health", threshold=50, top_n=30)

    print("\nFixing N6: Journal Similarity Network (clean abbreviations)...")
    plot_similarity_network_fixed()

    print("\nDone.")
