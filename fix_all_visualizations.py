"""
Comprehensive fix for all visualization issues identified in review.

Fixes:
  N4: Redesign as "year of steepest growth" (not meaningless emergence)
  N5: Use percentage-share slope (not absolute) so "declining" = negative share
  N6: Lower threshold, add size legend, label all nodes, remove isolates
  Q2: Clarify "Other" label in composition chart
  Q5: Fix text overlap in scatter, annotate more points
  Q6: Fix submission landscape annotations
  Q7 heatmap: Remove confusing vertical event lines
  Q7 event-annotated: Fix text overlap, curate events per topic panel
"""

import csv
import json
import math
import os
import textwrap
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

DATA_DIR = os.path.expanduser("~/Dropbox/multi_journal_trend_analysis/data")
VIZ_DIR = os.path.join(DATA_DIR, "visualizations")
HIER_DIR = os.path.join(DATA_DIR, "hierarchy")
ARTICLES_CSV = os.path.join(DATA_DIR, "pubmed_all_journals_2011_2025_w_keywords.csv")

YEARS = list(range(2011, 2026))

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

# Curated event-topic mapping for Q7 (only show relevant events per topic)
TOPIC_EVENTS = {
    "Machine learning methods": [
        (2012, "AlexNet"), (2018, "BERT"), (2022, "ChatGPT"), (2023, "GPT-4"),
    ],
    "Digital research methods": [
        (2018, "BERT"), (2020, "COVID-19"), (2022, "ChatGPT"),
    ],
    "Digital health": [
        (2013, "HITECH MU2"), (2016, "Cures Act"), (2020, "COVID-19"),
    ],
    "Human-computer interaction": [
        (2015, "ResNet"), (2017, "Transformer"), (2020, "COVID-19"),
    ],
    "Deep learning architectures": [
        (2012, "AlexNet"), (2015, "ResNet"), (2017, "Transformer"), (2022, "ChatGPT"),
    ],
    "recommender algorithm": [
        (2018, "BERT"), (2020, "COVID-19"),
    ],
    "Healthcare modeling": [
        (2018, "BERT"), (2020, "COVID-19"), (2021, "FDA AI/ML"),
    ],
    "Natural language processing": [
        (2017, "Transformer"), (2018, "BERT"), (2022, "ChatGPT"), (2023, "GPT-4"),
    ],
    # Health topics
    "Computational biology": [
        (2015, "Precision Med"), (2018, "BERT"),
    ],
    "Healthcare semantic interoperability": [
        (2013, "HITECH MU2"), (2016, "Cures Act"),
    ],
    "Mental health": [
        (2020, "COVID-19"),
    ],
    "Gene expression regulation": [
        (2015, "Precision Med"),
    ],
    "Healthcare outcomes": [
        (2016, "Cures Act"), (2020, "COVID-19"),
    ],
    "Primary care": [
        (2020, "COVID-19"),
    ],
    "Clinical prognosis": [
        (2018, "BERT"), (2021, "FDA AI/ML"),
    ],
}

EVENT_COLORS_MAP = {
    "AlexNet": "#3498db", "ResNet": "#3498db", "Transformer": "#3498db",
    "BERT": "#3498db", "ChatGPT": "#3498db", "GPT-4": "#3498db",
    "HITECH MU2": "#e67e22", "Cures Act": "#e67e22", "Precision Med": "#e67e22",
    "GDPR": "#e67e22", "FDA AI/ML": "#e67e22", "EU AI Act": "#e67e22",
    "COVID-19": "#e74c3c",
}


def abbr(j):
    return JOURNAL_ABBR.get(j, j[:25])


def load_year_counts(domain):
    path = os.path.join(VIZ_DIR, f"{domain}_topic_year_counts.csv")
    return pd.read_csv(path)


def load_topics_and_map(domain):
    path = os.path.join(HIER_DIR, f"{domain}_final_topics.json")
    with open(path) as f:
        topics = json.load(f)
    kw_map = defaultdict(set)
    for data in topics.values():
        for kw in data.get("keywords", []):
            kw_map[kw.strip().lower()].add(data["name"])
    return topics, kw_map


def load_articles():
    print("Loading articles...")
    df = pd.read_csv(ARTICLES_CSV, quoting=csv.QUOTE_ALL, dtype={"pmid": str})
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    return df


# ---------------------------------------------------------------------------
# N3 check: which journal lacks Machine Learning
# ---------------------------------------------------------------------------

def check_n3_missing_ml(df, kw_m):
    print("\n=== N3: Which journal lacks Machine Learning? ===")
    ml_journals = set()
    for _, row in df.iterrows():
        kw_str = row.get("keywords", "")
        if pd.isna(kw_str):
            continue
        topics = set()
        for kw in str(kw_str).split(";"):
            kw = kw.strip().lower()
            if kw in kw_m:
                topics.update(kw_m[kw])
        if "Machine learning methods" in topics:
            ml_journals.add(row["journal"])

    all_journals = set(df["journal"].unique())
    missing = all_journals - ml_journals
    print(f"  Journals WITHOUT Machine Learning: {missing if missing else 'None — all journals have ML'}")
    for j in missing:
        count = len(df[df["journal"] == j])
        print(f"    {j}: {count} total articles")


# ---------------------------------------------------------------------------
# N4 FIX: Year of steepest growth (always meaningful)
# ---------------------------------------------------------------------------

def fix_n4(domain, top_n=30):
    print(f"\n=== Fixing N4 ({domain}): Year of steepest growth ===")
    df = load_year_counts(domain)

    rows = []
    for _, row in df.iterrows():
        topic = row["topic"]
        total = int(row["total"])
        yearly = [int(row[str(y)]) for y in YEARS]
        growth = np.diff(yearly)

        if len(growth) == 0 or total < 20:
            continue

        max_idx = int(np.argmax(growth))
        breakthrough_year = YEARS[max_idx + 1]
        max_growth = int(growth[max_idx])

        rows.append({
            "topic": topic, "total": total,
            "breakthrough_year": breakthrough_year,
            "max_yoy_growth": max_growth,
        })

    rdf = pd.DataFrame(rows).nlargest(top_n, "total")
    rdf = rdf.sort_values(["breakthrough_year", "total"], ascending=[False, True])

    fig, ax = plt.subplots(figsize=(15, max(10, top_n * 0.38)))
    y_pos = np.arange(len(rdf))
    norm = (rdf["breakthrough_year"].values - YEARS[0]) / (YEARS[-1] - YEARS[0])
    colors = plt.cm.viridis(norm)
    sizes = (rdf["max_yoy_growth"] / rdf["max_yoy_growth"].max()) * 500 + 50

    ax.scatter(rdf["breakthrough_year"], y_pos, s=sizes, c=colors,
               edgecolors="black", linewidths=1, zorder=3, alpha=0.85)
    ax.hlines(y_pos, YEARS[0], rdf["breakthrough_year"],
              colors="#bdc3c7", linewidth=0.8, alpha=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(rdf["topic"], fontsize=9)

    for i, row in enumerate(rdf.itertuples()):
        ax.text(YEARS[-1] + 0.5, i,
                f"{row.breakthrough_year}  (+{row.max_yoy_growth} articles YoY)",
                va="center", fontsize=8, color="#2c3e50")

    ax.set_xlabel("Year of Steepest Year-over-Year Growth", fontsize=12)
    ax.set_title(
        f"N4: {domain.title()} — Year of Maximum Growth Acceleration\n"
        f"(dot size = magnitude of YoY increase; "
        f"shows WHEN each topic grew fastest)",
        fontsize=14)
    ax.set_xlim(YEARS[0] - 0.5, YEARS[-1] + 8)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f"n4_{domain}_topic_emergence.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved n4_{domain}_topic_emergence.png")


# ---------------------------------------------------------------------------
# N5 FIX: Use percentage-share slope (truly declining = losing market share)
# ---------------------------------------------------------------------------

def fix_n5(domain, top_n=10):
    print(f"\n=== Fixing N5 ({domain}): Share-based rising/declining ===")
    df = load_year_counts(domain)
    total_per_year = {y: df[str(y)].sum() for y in YEARS}

    rows = []
    for _, row in df.iterrows():
        topic = row["topic"]
        total = int(row["total"])
        if total < 50:
            continue
        shares = [(row[str(y)] / max(total_per_year[y], 1)) * 100 for y in YEARS]
        slope_share = np.polyfit(np.array(YEARS, dtype=float), shares, 1)[0]
        slope_abs = np.polyfit(np.array(YEARS, dtype=float),
                               [row[str(y)] for y in YEARS], 1)[0]
        rows.append({
            "topic": topic, "total": total,
            "share_slope": slope_share,
            "abs_slope": slope_abs,
        })

    rdf = pd.DataFrame(rows)

    rising = rdf.nlargest(top_n, "share_slope")
    # Only show truly declining (negative share slope)
    truly_declining = rdf[rdf["share_slope"] < 0].nsmallest(top_n, "share_slope")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    ax1.barh(rising["topic"], rising["share_slope"], color="#2ecc71", edgecolor="white")
    for i, (_, r) in enumerate(rising.iterrows()):
        ax1.text(r["share_slope"] + 0.005, i,
                 f'+{r["share_slope"]:.3f}%/yr ({r["total"]:,} articles)',
                 va="center", fontsize=9)
    ax1.set_xlabel("Share Slope (%-points per year)", fontsize=12)
    ax1.set_title(f"Top {top_n} Rising {domain.title()} Topics\n"
                  f"(gaining share of total publications)", fontsize=13)
    ax1.grid(True, alpha=0.3, axis="x")

    if len(truly_declining) > 0:
        ax2.barh(truly_declining["topic"], truly_declining["share_slope"],
                 color="#e74c3c", edgecolor="white")
        for i, (_, r) in enumerate(truly_declining.iterrows()):
            ax2.text(r["share_slope"] - 0.005, i,
                     f'{r["share_slope"]:.3f}%/yr ({r["total"]:,} articles)',
                     va="center", fontsize=9, ha="right")
        ax2.set_xlabel("Share Slope (%-points per year)", fontsize=12)
        n_shown = len(truly_declining)
        ax2.set_title(f"Declining {domain.title()} Topics (n={n_shown})\n"
                      f"(losing share — ALL have negative share trend)", fontsize=13)
    else:
        ax2.text(0.5, 0.5, "No topics with negative\nshare slope found",
                 ha="center", va="center", fontsize=14, transform=ax2.transAxes)
        ax2.set_title(f"Declining {domain.title()} Topics", fontsize=13)
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f"n5_{domain}_rising_declining.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved n5_{domain}_rising_declining.png")


# ---------------------------------------------------------------------------
# N6 FIX: Lower threshold, add legend, label all nodes
# ---------------------------------------------------------------------------

def fix_n6(df, kw_m):
    print("\n=== Fixing N6: Journal similarity network ===")
    import networkx as nx
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    with open(os.path.join(HIER_DIR, "methodology_final_topics.json")) as f:
        topics = json.load(f)
    kw_to_topic = defaultdict(set)
    for data in topics.values():
        for kw in data.get("keywords", []):
            kw_to_topic[kw.strip().lower()].add(data["name"])

    journal_topic = defaultdict(Counter)
    journal_total = Counter()
    for _, row in df.iterrows():
        kw_str = row.get("keywords", "")
        if pd.isna(kw_str):
            continue
        j = row["journal"]
        journal_total[j] += 1
        for kw in str(kw_str).split(";"):
            kw = kw.strip().lower()
            if kw in kw_to_topic:
                for t in kw_to_topic[kw]:
                    journal_topic[j][t] += 1

    all_topics = sorted({t for jc in journal_topic.values() for t in jc})
    journals = sorted(journal_topic.keys())
    vectors = []
    for j in journals:
        total = journal_total[j]
        vectors.append([journal_topic[j].get(t, 0) / max(total, 1) for t in all_topics])
    vectors = np.array(vectors)
    sim = cos_sim(vectors)

    G = nx.Graph()
    for j in journals:
        G.add_node(abbr(j), size=journal_total[j])

    # Use 50th percentile threshold (lower = more connections)
    upper = sim[np.triu_indices_from(sim, k=1)]
    threshold = np.percentile(upper, 50)

    for i in range(len(journals)):
        for j in range(i + 1, len(journals)):
            if sim[i, j] > threshold:
                G.add_edge(abbr(journals[i]), abbr(journals[j]),
                           weight=float(sim[i, j]))

    # Remove truly isolated nodes (no edges)
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    pos = nx.kamada_kawai_layout(G)
    fig, ax = plt.subplots(figsize=(20, 16))

    sizes = np.array([G.nodes[n].get("size", 100) for n in G.nodes])
    node_sizes = (sizes / sizes.max()) * 2000 + 100
    weights = [G[u][v]["weight"] * 5 for u, v in G.edges]

    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.35,
                           edge_color="#7f8c8d", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color="#5dade2", node_size=node_sizes,
                           edgecolors="#1f618d", linewidths=2, alpha=0.85, ax=ax)

    # Labels with white background
    for node, (x, y) in pos.items():
        ax.text(x, y + 0.04, node, ha="center", va="center",
                fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#1f618d", alpha=0.92))

    # Size legend
    legend_sizes = [500, 3000, 8000]
    legend_handles = []
    for s in legend_sizes:
        scaled = (s / sizes.max()) * 2000 + 100
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#5dade2", markeredgecolor="#1f618d",
                   markersize=np.sqrt(scaled) / 3,
                   label=f"{s:,} articles"))
    ax.legend(handles=legend_handles, loc="lower right", fontsize=11,
              title="Journal Size (articles)", title_fontsize=12,
              framealpha=0.95)

    if isolates:
        ax.text(0.01, 0.01,
                f"Not shown (too dissimilar): {', '.join(isolates)}",
                transform=ax.transAxes, fontsize=9, alpha=0.6)

    ax.set_title("Journal Similarity Network\n"
                 "(edges = above-median cosine similarity of topic distributions; "
                 "node size = total articles)", fontsize=16)
    ax.axis("off")
    ax.margins(0.12)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "n6_journal_similarity_network.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved n6_journal_similarity_network.png")


# ---------------------------------------------------------------------------
# Q2 FIX: Clarify composition labels
# ---------------------------------------------------------------------------

def fix_q2_composition():
    print("\n=== Fixing Q2: Composition chart labels ===")
    stats = pd.read_csv(os.path.join(VIZ_DIR, "pi_q2_authorship_stats.csv"))

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.stackplot(stats["year"],
                 stats["pct_interdisciplinary"], stats["pct_clinical_only"],
                 stats["pct_computational_only"], stats["pct_neither"],
                 labels=[
                     "Interdisciplinary\n(both clinical + computational affiliations)",
                     "Clinical affiliations only\n(hospitals, medical schools)",
                     "Computational affiliations only\n(CS, informatics, engineering, stats)",
                     "No clinical/computational affiliation\ndetected in metadata",
                 ],
                 colors=["#9b59b6", "#e74c3c", "#3498db", "#bdc3c7"], alpha=0.85)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("% of Articles", fontsize=12)
    ax.set_title("Q2: Author Affiliation Composition Over Time\n"
                 "(based on regex classification of PubMed affiliation strings)",
                 fontsize=14)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "pi_q2_composition_over_time.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved pi_q2_composition_over_time.png")


# ---------------------------------------------------------------------------
# Q5 FIX: Fix scatter — annotate top topics, no text overlap
# ---------------------------------------------------------------------------

def fix_q5_scatter(domain):
    print(f"\n=== Fixing Q5 scatter ({domain}): Annotate top topics ===")
    df = pd.read_csv(os.path.join(VIZ_DIR, f"pi_q5_{domain}_specialization.csv"))

    fig, ax = plt.subplots(figsize=(14, 9))
    sizes = np.sqrt(df["total"]) * 4
    scatter = ax.scatter(df["total"], df["norm_entropy"], s=sizes,
                         c=df["norm_entropy"], cmap="RdYlGn",
                         edgecolors="black", linewidths=0.4, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Total Articles (log scale)", fontsize=12)
    ax.set_ylabel("Normalized Shannon Entropy\n"
                  "(0 = all articles in 1 journal, "
                  "1 = evenly spread across all 29 journals)", fontsize=11)
    ax.set_title(f"Q5: {domain.title()} Topic Specialization vs Sharing\n"
                 f"(dot size = article count; green = broadly shared, "
                 f"red = concentrated in few journals)", fontsize=13)
    ax.axhline(df["norm_entropy"].median(), color="gray", linestyle="--",
               alpha=0.5, label=f"Median = {df['norm_entropy'].median():.2f}")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate top 20 topics by total (enough labels to be informative)
    to_annotate = df.nlargest(20, "total")
    for _, r in to_annotate.iterrows():
        label = textwrap.shorten(r["topic"], width=30, placeholder="...")
        # Offset labels above/below alternating to reduce overlap
        offset_y = 8 if hash(r["topic"]) % 2 == 0 else -12
        ax.annotate(label, (r["total"], r["norm_entropy"]),
                    fontsize=7, alpha=0.85,
                    xytext=(8, offset_y), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", alpha=0.3, lw=0.5))

    plt.colorbar(scatter, ax=ax, label="Normalized Entropy", shrink=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f"pi_q5_{domain}_specialization_scatter.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved pi_q5_{domain}_specialization_scatter.png")


# ---------------------------------------------------------------------------
# Q6 FIX: Submission landscape — annotate all significant topics
# ---------------------------------------------------------------------------

def fix_q6_landscape(domain, records, key):
    print(f"\n=== Fixing Q6 submission landscape ({domain}) ===")
    tj = defaultdict(Counter)
    topic_total = Counter()
    for r in records:
        for t in r[key]:
            tj[t][r["journal"]] += 1
            topic_total[t] += 1

    rows = []
    for t, total in topic_total.items():
        n_j = len(tj[t])
        rows.append({"topic": t, "total": total, "n_journals": n_j})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.scatter(df["n_journals"], df["total"], s=80, c="#3498db",
               edgecolors="black", linewidths=0.4, alpha=0.7)
    ax.set_xlabel("# Journals Publishing This Topic\n(more = more submission options)",
                  fontsize=12)
    ax.set_ylabel("Total Articles\n(more = more competition)", fontsize=12)
    ax.set_title(f"Q6: Submission Landscape — {domain.title()} Topics\n"
                 f"(bottom-right = less competitive; top-left = more competitive)",
                 fontsize=14)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Annotate ALL topics with >= 200 articles
    to_annotate = df[df["total"] >= 200]
    for _, r in to_annotate.iterrows():
        label = textwrap.shorten(r["topic"], 28, placeholder="...")
        offset_y = 8 if hash(r["topic"]) % 2 == 0 else -12
        ax.annotate(label, (r["n_journals"], r["total"]),
                    fontsize=7, alpha=0.85,
                    xytext=(6, offset_y), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", alpha=0.3, lw=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f"pi_q6_{domain}_submission_landscape.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved pi_q6_{domain}_submission_landscape.png")


# ---------------------------------------------------------------------------
# Q7 FIX: Event-annotated trends (curated events, no text overlap)
# ---------------------------------------------------------------------------

def fix_q7_event_annotated(domain):
    print(f"\n=== Fixing Q7 event-annotated ({domain}) ===")
    df = load_year_counts(domain)

    topic_totals = Counter()
    for _, row in df.iterrows():
        topic_totals[row["topic"]] = int(row["total"])
    top8 = [t for t, _ in Counter(topic_totals).most_common(8)]

    fig, axes = plt.subplots(4, 2, figsize=(20, 24))
    axes = axes.flatten()

    for idx in range(len(top8), len(axes)):
        axes[idx].set_visible(False)

    for idx, topic in enumerate(top8):
        ax = axes[idx]
        row = df[df["topic"] == topic].iloc[0]
        counts = [int(row[str(y)]) for y in YEARS]

        ax.plot(YEARS, counts, "o-", color="#2c3e50", linewidth=2.5,
                markersize=6, zorder=3)
        ax.fill_between(YEARS, 0, counts, alpha=0.12, color="#3498db")

        # Show only RELEVANT events for this topic (curated)
        events = TOPIC_EVENTS.get(topic, [])
        used_y_positions = []
        for ev_year, ev_name in events:
            if ev_year < YEARS[0] or ev_year > YEARS[-1]:
                continue
            color = EVENT_COLORS_MAP.get(ev_name, "#7f8c8d")
            ax.axvline(ev_year, color=color, linestyle="--", alpha=0.7, linewidth=1.5)
            # Stagger text position to avoid overlap
            y_pos = ax.get_ylim()[1] * 0.92
            for used_y in used_y_positions:
                if abs(ev_year - used_y) <= 1:
                    y_pos *= 0.78
            used_y_positions.append(ev_year)
            ax.text(ev_year + 0.2, y_pos, ev_name, fontsize=9,
                    color=color, fontweight="bold", rotation=0,
                    va="top", ha="left",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor=color,
                              boxstyle="round,pad=0.2"))

        title = "\n".join(textwrap.wrap(topic, 35))
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Articles/Year", fontsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        ax.set_xlim(YEARS[0] - 0.5, YEARS[-1] + 0.5)

    legend_handles = [
        mpatches.Patch(color="#3498db", label="AI Milestone"),
        mpatches.Patch(color="#e67e22", label="Policy"),
        mpatches.Patch(color="#e74c3c", label="Pandemic"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=12,
               title="Event Type", title_fontsize=13)
    fig.suptitle(f"Q7: {domain.title()} Topic Trends with Historical Events\n"
                 f"(only showing events most relevant to each topic)",
                 fontsize=18, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f"pi_q7_{domain}_event_annotated.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved pi_q7_{domain}_event_annotated.png")


# ---------------------------------------------------------------------------
# Q7 FIX: Heatmap — remove confusing vertical lines
# ---------------------------------------------------------------------------

def fix_q7_heatmap(domain):
    print(f"\n=== Fixing Q7 heatmap ({domain}): Remove vertical lines ===")
    df = load_year_counts(domain)

    topic_totals = Counter()
    for _, row in df.iterrows():
        topic_totals[row["topic"]] = int(row["total"])
    top20 = [t for t, _ in Counter(topic_totals).most_common(20)]

    mat = []
    for t in top20:
        row = df[df["topic"] == t].iloc[0]
        counts = [int(row[str(y)]) for y in YEARS]
        growth = [0] + list(np.diff(counts))
        mat.append(growth)
    mat = np.array(mat)

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(mat, xticklabels=YEARS, yticklabels=top20,
                cmap="RdBu_r", center=0, ax=ax,
                linewidths=0.5, linecolor="white", annot=False)
    ax.set_title(f"Q7: Year-over-Year Growth — Top 20 {domain.title()} Topics\n"
                 f"(red = surge in articles, blue = decline)",
                 fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    # NO vertical event lines — they were confusing
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f"pi_q7_{domain}_changepoint_heatmap.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved pi_q7_{domain}_changepoint_heatmap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_articles()

    _, kw_m = load_topics_and_map("methodology")
    _, kw_h = load_topics_and_map("health")

    # Build records for Q6 landscape
    records = []
    for _, row in df.iterrows():
        kw_str = row.get("keywords", "")
        mt, ht = set(), set()
        if not pd.isna(kw_str) and kw_str:
            for kw in str(kw_str).split(";"):
                kw = kw.strip().lower()
                if kw in kw_m: mt.update(kw_m[kw])
                if kw in kw_h: ht.update(kw_h[kw])
        records.append({"pmid": row["pmid"], "year": int(row["year"]),
                        "journal": row.get("journal", ""), "method_topics": mt,
                        "health_topics": ht})

    # Run all fixes
    check_n3_missing_ml(df, kw_m)

    fix_n4("methodology")
    fix_n4("health")

    fix_n5("methodology")
    fix_n5("health")

    fix_n6(df, kw_m)

    fix_q2_composition()

    fix_q5_scatter("methodology")
    fix_q5_scatter("health")

    fix_q6_landscape("methodology", records, "method_topics")
    fix_q6_landscape("health", records, "health_topics")

    fix_q7_event_annotated("methodology")
    fix_q7_event_annotated("health")

    fix_q7_heatmap("methodology")
    fix_q7_heatmap("health")

    print("\n=== ALL FIXES COMPLETE ===")
