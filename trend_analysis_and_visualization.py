"""
Step 6: Trend Analysis and Visualization

Replicates the paper's analyses (RQ1-RQ4) and adds new multi-journal insights.

Paper's analyses:
  RQ1. Publication volume trends over time
  RQ2. Topic popularity trends (count + percentage, rising/declining detection)
  RQ3. Methodology x Health topic co-occurrence
  RQ4. Top topics by time period (stacked bar charts)

New multi-journal analyses:
  N1. Per-journal publication volume comparison
  N2. Journal specialization heatmap (journal x topic)
  N3. Cross-journal topic sharing (shared vs unique topics)
  N4. Topic emergence timeline across journals
  N5. Rising/declining topic detection with trend slopes
  N6. Journal similarity network (based on topic distributions)
  N7. Keyword composition within top topics (stacked area charts)

Usage:
    python trend_analysis_and_visualization.py \\
        --articles-csv data/pubmed_all_journals_2011_2025_w_keywords.csv \\
        --topics-dir data/hierarchy \\
        --output-dir data/visualizations \\
        [--start-year 2011] [--end-year 2025] \\
        [--top-n 20]

Requirements:
    pip install matplotlib seaborn pandas numpy scikit-learn networkx
"""

import argparse
import csv
import json
import logging
import os
import sys
import textwrap
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 6: Trend analysis and visualization."
    )
    parser.add_argument(
        "--articles-csv", required=True,
        help="Original articles CSV from data collection",
    )
    parser.add_argument(
        "--topics-dir", default="data/hierarchy",
        help="Directory with Step 4 output (final topics JSON)",
    )
    parser.add_argument(
        "--output-dir", default="data/visualizations",
        help="Output directory for plots and data",
    )
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="Number of top topics to show in detailed plots",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data Loading and Topic Mapping
# ---------------------------------------------------------------------------

def load_articles(csv_path):
    """Load article data."""
    logger.info(f"Loading articles from {csv_path}")
    df = pd.read_csv(csv_path, quoting=csv.QUOTE_ALL)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    logger.info(f"  Loaded {len(df)} articles")
    return df


def load_final_topics(topics_dir, domain):
    """Load deduplicated topics from Step 4."""
    path = os.path.join(topics_dir, f"{domain}_final_topics.json")
    with open(path, "r", encoding="utf-8") as f:
        topics = json.load(f)
    logger.info(f"  Loaded {len(topics)} {domain} topics")
    return topics


def build_keyword_to_topic_map(topics):
    """Build reverse mapping: keyword -> set of topic names."""
    kw_to_topics = defaultdict(set)
    for tid, data in topics.items():
        name = data["name"]
        for kw in data.get("keywords", []):
            kw_to_topics[kw.strip().lower()].add(name)
    return kw_to_topics


def map_articles_to_topics(df, kw_to_method, kw_to_health):
    """
    For each article, determine which methodology and health topics it belongs to.
    Returns lists of (year, journal, method_topics, health_topics) tuples.
    """
    records = []
    for _, row in df.iterrows():
        kw_str = row.get("keywords", "")
        if pd.isna(kw_str) or not kw_str:
            continue

        year = int(row["year"])
        journal = row.get("journal", "unknown")

        method_topics = set()
        health_topics = set()
        for kw in str(kw_str).split(";"):
            kw = kw.strip().lower()
            if kw in kw_to_method:
                method_topics.update(kw_to_method[kw])
            if kw in kw_to_health:
                health_topics.update(kw_to_health[kw])

        records.append({
            "year": year,
            "journal": journal,
            "method_topics": method_topics,
            "health_topics": health_topics,
        })
    return records


def count_topic_year(records, domain_key):
    """Count articles per topic per year."""
    topic_year_count = Counter()
    for r in records:
        for topic in r[domain_key]:
            topic_year_count[(topic, r["year"])] += 1
    return topic_year_count


def count_topic_journal_year(records, domain_key):
    """Count articles per topic per journal per year."""
    topic_journal_year = Counter()
    for r in records:
        for topic in r[domain_key]:
            topic_journal_year[(topic, r["journal"], r["year"])] += 1
    return topic_journal_year


# ---------------------------------------------------------------------------
# RQ1: Publication Volume Trends
# ---------------------------------------------------------------------------

def plot_total_publication_volume(df, years, output_dir):
    """Plot total publication volume across all journals over time."""
    year_counts = df[df["year"].between(years[0], years[-1])]["year"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(year_counts.index, year_counts.values, color="#4169e1", alpha=0.8)
    ax.plot(year_counts.index, year_counts.values, "o-", color="#000080", linewidth=2)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Number of Articles", fontsize=14)
    ax.set_title("Total Publication Volume (All 30 Journals)", fontsize=16)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rq1_total_volume.png"), dpi=300,
                bbox_inches="tight")
    plt.close()
    logger.info("  Saved rq1_total_volume.png")


# ---------------------------------------------------------------------------
# N1: Per-Journal Publication Volume
# ---------------------------------------------------------------------------

def plot_journal_volume_comparison(df, years, output_dir, top_n=15):
    """Compare publication volumes across journals."""
    filtered = df[df["year"].between(years[0], years[-1])]

    # Total per journal
    journal_totals = filtered["journal"].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(12, 8))
    journal_totals.sort_values().plot(kind="barh", ax=ax, color="#4169e1", alpha=0.8)
    ax.set_xlabel("Total Articles", fontsize=14)
    ax.set_title(f"Top {top_n} Journals by Publication Volume ({years[0]}-{years[-1]})",
                 fontsize=16)
    for i, v in enumerate(journal_totals.sort_values().values):
        ax.text(v + 20, i, str(v), va="center", fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "n1_journal_volume_comparison.png"), dpi=300,
                bbox_inches="tight")
    plt.close()
    logger.info("  Saved n1_journal_volume_comparison.png")

    # Per-journal trend lines
    top_journals = journal_totals.index.tolist()[:10]
    fig, ax = plt.subplots(figsize=(14, 7))
    for journal in top_journals:
        j_data = filtered[filtered["journal"] == journal]
        y_counts = j_data["year"].value_counts().sort_index()
        short_name = journal.split("(")[0].strip()[:30]
        ax.plot(y_counts.index, y_counts.values, "o-", label=short_name,
                linewidth=2, markersize=4)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Articles", fontsize=14)
    ax.set_title("Publication Trends by Journal (Top 10)", fontsize=16)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "n1_journal_trends.png"), dpi=300,
                bbox_inches="tight")
    plt.close()
    logger.info("  Saved n1_journal_trends.png")


# ---------------------------------------------------------------------------
# RQ2: Topic Popularity Trends
# ---------------------------------------------------------------------------

def plot_topic_trends(topic_year_count, total_per_year, domain, years,
                      output_dir, top_n=16):
    """
    Plot topic trends with dual axes (count + percentage).
    Paper: plot_multiple_topics_trend_multi_rows_cols.
    """
    topic_totals = Counter()
    for (topic, year), count in topic_year_count.items():
        topic_totals[topic] += count
    top_topics = [t for t, _ in topic_totals.most_common(top_n)]

    num_cols = 4
    num_rows = (len(top_topics) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(28, 6 * num_rows))
    axes = axes.flatten()

    color_count = "#38b775"
    color_pct = "#31688e"

    for idx, topic in enumerate(top_topics):
        ax1 = axes[idx]
        ax2 = ax1.twinx()

        counts = [topic_year_count.get((topic, y), 0) for y in years]
        pcts = [
            (topic_year_count.get((topic, y), 0) / total_per_year.get(y, 1)) * 100
            for y in years
        ]

        ax2.plot(years, counts, marker="o", linestyle="--", color=color_count,
                 linewidth=2, markersize=6, label="Count", zorder=1)
        ax1.plot(years, pcts, marker="s", color=color_pct, linewidth=2,
                 markersize=6, label="% of Articles", zorder=2)

        wrapped = "\n".join(textwrap.wrap(topic, width=35))
        ax1.set_title(wrapped, fontsize=13, fontweight="bold")
        ax1.set_ylabel("% of Articles", color=color_pct, fontsize=11)
        ax2.set_ylabel("Count", color=color_count, fontsize=11)
        ax1.tick_params(axis="y", labelcolor=color_pct)
        ax2.tick_params(axis="y", labelcolor=color_count)
        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)
        ax1.grid(True, alpha=0.3)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.setp(ax1.get_xticklabels(), rotation=45, fontsize=9)

    for idx in range(len(top_topics), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"Top {top_n} {domain.title()} Topic Trends", fontsize=20, y=1.01)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"rq2_{domain}_topic_trends.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close()
    logger.info(f"  Saved rq2_{domain}_topic_trends.png")


# ---------------------------------------------------------------------------
# RQ4: Top Topics by Time Period (Stacked Bar Charts)
# ---------------------------------------------------------------------------

def plot_top_topics_by_period(topic_year_count, domain, years, output_dir,
                              top_n=25):
    """
    Stacked horizontal bar chart of top topics by time period.
    Paper: draw_top_topics_bar_plot.
    """
    periods = []
    step = 5
    for start in range(years[0], years[-1] + 1, step):
        end = min(start + step - 1, years[-1])
        periods.append((start, end))

    rows = []
    for (topic, year), count in topic_year_count.items():
        for p_start, p_end in periods:
            if p_start <= year <= p_end:
                rows.append({
                    "topic": topic,
                    "period": f"{p_start}-{p_end}",
                    "count": count,
                })
                break

    df_plot = pd.DataFrame(rows)
    if df_plot.empty:
        return

    pivot = df_plot.pivot_table(
        index="topic", columns="period", values="count", aggfunc="sum",
    ).fillna(0)
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=True).tail(top_n)
    pivot = pivot.drop(columns=["total"])

    colors = ["#add8e6", "#6495ed", "#4169e1", "#000080"]
    colors = colors[:len(pivot.columns)]

    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))
    pivot.plot(kind="barh", stacked=True, ax=ax, color=colors)
    totals = pivot.sum(axis=1)
    for i, total in enumerate(totals):
        ax.text(total + 5, i, str(int(total)), va="center", fontsize=9)
    ax.set_xlabel("Number of Articles", fontsize=14)
    ax.set_title(f"Top {top_n} {domain.title()} Topics by Time Period", fontsize=16)
    ax.legend(title="Period", fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"rq4_{domain}_topics_by_period.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close()
    logger.info(f"  Saved rq4_{domain}_topics_by_period.png")


# ---------------------------------------------------------------------------
# RQ3: Methodology x Health Co-occurrence
# ---------------------------------------------------------------------------

def compute_cooccurrence(records, top_n_method=15, top_n_health=15):
    """
    Compute co-occurrence matrix: methodology topic x health topic.
    Each cell = number of articles containing keywords from both topics.
    """
    # Count topic totals
    method_totals = Counter()
    health_totals = Counter()
    cooccurrence = Counter()

    for r in records:
        for mt in r["method_topics"]:
            method_totals[mt] += 1
            for ht in r["health_topics"]:
                cooccurrence[(mt, ht)] += 1
        for ht in r["health_topics"]:
            health_totals[ht] += 1

    top_methods = [t for t, _ in method_totals.most_common(top_n_method)]
    top_healths = [t for t, _ in health_totals.most_common(top_n_health)]

    matrix = pd.DataFrame(0, index=top_methods, columns=top_healths)
    for mt in top_methods:
        for ht in top_healths:
            matrix.loc[mt, ht] = cooccurrence.get((mt, ht), 0)

    return matrix


def plot_cooccurrence_heatmap(matrix, output_dir):
    """Plot methodology x health co-occurrence heatmap."""
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
        linewidths=0.5, linecolor="white",
        annot_kws={"fontsize": 8},
    )
    ax.set_title("Methodology × Health Domain Co-occurrence", fontsize=16)
    ax.set_xlabel("Health Topics", fontsize=14)
    ax.set_ylabel("Methodology Topics", fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "rq3_cooccurrence_heatmap.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close()
    logger.info("  Saved rq3_cooccurrence_heatmap.png")


# ---------------------------------------------------------------------------
# N2: Journal Specialization Heatmap
# ---------------------------------------------------------------------------

def plot_journal_topic_heatmap(records, domain_key, domain, output_dir,
                               top_n_journals=15, top_n_topics=20):
    """
    Heatmap showing which topics each journal focuses on.
    Reveals journal specialization patterns.
    """
    journal_topic = Counter()
    journal_totals = Counter()
    for r in records:
        journal_totals[r["journal"]] += 1
        for topic in r[domain_key]:
            journal_topic[(r["journal"], topic)] += 1

    top_journals = [j for j, _ in journal_totals.most_common(top_n_journals)]

    topic_totals = Counter()
    for (j, t), c in journal_topic.items():
        if j in top_journals:
            topic_totals[t] += c
    top_topics = [t for t, _ in topic_totals.most_common(top_n_topics)]

    # Build matrix: proportion of each journal's articles in each topic
    matrix = pd.DataFrame(0.0, index=top_journals, columns=top_topics)
    for j in top_journals:
        j_total = journal_totals[j]
        for t in top_topics:
            count = journal_topic.get((j, t), 0)
            matrix.loc[j, t] = (count / j_total * 100) if j_total > 0 else 0

    # Shorten journal names for readability
    short_names = []
    for j in matrix.index:
        short = j.split("(")[0].strip()
        if len(short) > 40:
            short = short[:37] + "..."
        short_names.append(short)
    matrix.index = short_names

    # Wrap topic names
    matrix.columns = ["\n".join(textwrap.wrap(t, 20)) for t in matrix.columns]

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(
        matrix, annot=True, fmt=".1f", cmap="Blues", ax=ax,
        linewidths=0.5, linecolor="white", annot_kws={"fontsize": 7},
    )
    ax.set_title(
        f"Journal Specialization — {domain.title()} Topics (% of journal articles)",
        fontsize=16,
    )
    ax.set_xlabel(f"{domain.title()} Topics", fontsize=13)
    ax.set_ylabel("Journals", fontsize=13)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"n2_{domain}_journal_specialization.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close()
    logger.info(f"  Saved n2_{domain}_journal_specialization.png")


# ---------------------------------------------------------------------------
# N3: Cross-Journal Topic Sharing
# ---------------------------------------------------------------------------

def plot_topic_sharing(records, domain_key, domain, output_dir, top_n=25):
    """
    Show how many journals each topic appears in.
    Topics shared across many journals = broad themes.
    Topics in few journals = specialized niches.
    """
    topic_journals = defaultdict(set)
    topic_totals = Counter()
    for r in records:
        for topic in r[domain_key]:
            topic_journals[topic].add(r["journal"])
            topic_totals[topic] += 1

    top_topics = [t for t, _ in topic_totals.most_common(top_n)]

    data = []
    for t in top_topics:
        data.append({
            "topic": t,
            "n_journals": len(topic_journals[t]),
            "total_articles": topic_totals[t],
        })
    df_share = pd.DataFrame(data).sort_values("n_journals", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.35)))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(df_share)))
    bars = ax.barh(
        df_share["topic"], df_share["n_journals"], color=colors, edgecolor="white",
    )
    for i, (nj, na) in enumerate(
        zip(df_share["n_journals"], df_share["total_articles"])
    ):
        ax.text(nj + 0.3, i, f"{nj} journals ({na} articles)", va="center",
                fontsize=9)
    ax.set_xlabel("Number of Journals", fontsize=14)
    ax.set_title(
        f"Cross-Journal {domain.title()} Topic Sharing (Top {top_n})",
        fontsize=16,
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"n3_{domain}_topic_sharing.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close()
    logger.info(f"  Saved n3_{domain}_topic_sharing.png")


# ---------------------------------------------------------------------------
# N4: Topic Emergence Timeline
# ---------------------------------------------------------------------------

def plot_topic_emergence(topic_year_count, domain, years, output_dir, top_n=30):
    """
    Show when each topic first appeared (first year with articles).
    Reveals waves of topic emergence over time.
    """
    topic_first_year = {}
    topic_totals = Counter()
    for (topic, year), count in topic_year_count.items():
        if topic not in topic_first_year or year < topic_first_year[topic]:
            topic_first_year[topic] = year
        topic_totals[topic] += count

    top_topics = [t for t, _ in topic_totals.most_common(top_n)]

    data = []
    for t in top_topics:
        data.append({
            "topic": t,
            "first_year": topic_first_year[t],
            "total": topic_totals[t],
        })
    df_emerge = pd.DataFrame(data).sort_values("first_year")

    fig, ax = plt.subplots(figsize=(14, max(8, top_n * 0.35)))
    colors = plt.cm.viridis(
        np.linspace(0, 1, len(df_emerge)),
    )
    ax.barh(df_emerge["topic"], df_emerge["first_year"] - years[0],
            left=years[0], color=colors, edgecolor="white")
    for i, row in df_emerge.iterrows():
        ax.text(row["first_year"] + 0.2, list(df_emerge["topic"]).index(row["topic"]),
                f'{row["first_year"]} ({row["total"]} articles)',
                va="center", fontsize=8)
    ax.set_xlabel("Year of First Appearance", fontsize=14)
    ax.set_title(f"{domain.title()} Topic Emergence Timeline", fontsize=16)
    ax.set_xlim(years[0] - 0.5, years[-1] + 2)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"n4_{domain}_topic_emergence.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close()
    logger.info(f"  Saved n4_{domain}_topic_emergence.png")


# ---------------------------------------------------------------------------
# N5: Rising/Declining Topic Detection
# ---------------------------------------------------------------------------

def compute_topic_trends(topic_year_count, years, min_articles=10):
    """
    Compute linear trend slopes for each topic.
    Positive slope = rising, negative = declining.
    """
    topic_totals = Counter()
    for (topic, year), count in topic_year_count.items():
        topic_totals[topic] += count

    trends = []
    year_arr = np.array(years, dtype=float)

    for topic, total in topic_totals.items():
        if total < min_articles:
            continue
        counts = np.array([topic_year_count.get((topic, y), 0) for y in years],
                          dtype=float)
        # Linear regression
        if counts.sum() == 0:
            continue
        slope, intercept = np.polyfit(year_arr, counts, 1)
        # Relative slope (normalized by mean)
        mean_count = counts.mean()
        rel_slope = slope / mean_count if mean_count > 0 else 0

        trends.append({
            "topic": topic,
            "total": total,
            "slope": slope,
            "relative_slope": rel_slope,
            "mean_count": mean_count,
            "recent_count": counts[-3:].mean(),  # last 3 years avg
            "early_count": counts[:3].mean(),     # first 3 years avg
        })

    return pd.DataFrame(trends)


def plot_rising_declining(trend_df, domain, output_dir, top_n=10):
    """Plot the top rising and declining topics."""
    if trend_df.empty:
        return

    rising = trend_df.nlargest(top_n, "slope")
    declining = trend_df.nsmallest(top_n, "slope")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Rising
    ax1.barh(rising["topic"], rising["slope"], color="#2ecc71", edgecolor="white")
    ax1.set_xlabel("Trend Slope (articles/year)", fontsize=12)
    ax1.set_title(f"Top {top_n} Rising {domain.title()} Topics", fontsize=14)
    ax1.grid(True, alpha=0.3, axis="x")

    # Declining
    ax2.barh(declining["topic"], declining["slope"], color="#e74c3c", edgecolor="white")
    ax2.set_xlabel("Trend Slope (articles/year)", fontsize=12)
    ax2.set_title(f"Top {top_n} Declining {domain.title()} Topics", fontsize=14)
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"n5_{domain}_rising_declining.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close()
    logger.info(f"  Saved n5_{domain}_rising_declining.png")


# ---------------------------------------------------------------------------
# N6: Journal Similarity Network
# ---------------------------------------------------------------------------

def plot_journal_similarity_network(records, domain_key, output_dir,
                                    top_n_journals=20):
    """
    Network graph showing journal similarity based on topic distributions.
    Journals with similar topic profiles are connected.
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("  networkx not installed, skipping journal similarity network")
        return

    journal_topic_vec = defaultdict(Counter)
    for r in records:
        for topic in r[domain_key]:
            journal_topic_vec[r["journal"]][topic] += 1

    # Get top journals by article count
    journal_totals = Counter()
    for r in records:
        journal_totals[r["journal"]] += 1
    top_journals = [j for j, _ in journal_totals.most_common(top_n_journals)]

    # Build topic vectors
    all_topics = set()
    for j in top_journals:
        all_topics.update(journal_topic_vec[j].keys())
    all_topics = sorted(all_topics)

    vectors = {}
    for j in top_journals:
        total = sum(journal_topic_vec[j].values())
        vec = np.array([journal_topic_vec[j].get(t, 0) / max(total, 1)
                        for t in all_topics])
        vectors[j] = vec

    # Compute pairwise cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    names = list(vectors.keys())
    mat = np.array([vectors[n] for n in names])
    sim = cos_sim(mat)

    # Build network with edges above threshold
    G = nx.Graph()
    short_names = {}
    for j in names:
        short = j.split("(")[0].strip()
        if len(short) > 25:
            short = short[:22] + "..."
        short_names[j] = short
        G.add_node(short, size=journal_totals[j])

    threshold = np.percentile(sim[np.triu_indices_from(sim, k=1)], 70)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if sim[i, j] > threshold:
                G.add_edge(short_names[names[i]], short_names[names[j]],
                           weight=float(sim[i, j]))

    fig, ax = plt.subplots(figsize=(14, 14))
    pos = nx.spring_layout(G, seed=42, k=2)
    sizes = [G.nodes[n].get("size", 100) / 5 for n in G.nodes]
    weights = [G[u][v]["weight"] * 3 for u, v in G.edges]

    nx.draw_networkx_nodes(G, pos, node_color="#85c1e9", node_size=sizes,
                           edgecolors="#5dade2", linewidths=1.5, ax=ax)
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5,
                           edge_color="grey", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)
    ax.set_title("Journal Similarity Network (based on topic distributions)",
                 fontsize=16)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "n6_journal_similarity_network.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close()
    logger.info("  Saved n6_journal_similarity_network.png")


# ---------------------------------------------------------------------------
# N7: Keyword Composition within Topics (Stacked Area)
# ---------------------------------------------------------------------------

def plot_keyword_composition(df, kw_to_topics, domain, years, output_dir,
                             top_n_topics=6, top_k_keywords=5):
    """
    Stacked area chart showing keyword composition within top topics over time.
    Paper: plot_keyword_distribution_multi_topics_proportion.
    """
    # Build keyword-year counts per topic
    topic_kw_year = defaultdict(lambda: defaultdict(Counter))

    for _, row in df.iterrows():
        kw_str = row.get("keywords", "")
        if pd.isna(kw_str) or not kw_str:
            continue
        year = int(row["year"])
        if year < years[0] or year > years[-1]:
            continue

        for kw in str(kw_str).split(";"):
            kw = kw.strip().lower()
            if kw in kw_to_topics:
                for topic in kw_to_topics[kw]:
                    topic_kw_year[topic][year][kw] += 1

    # Get top topics by total article count
    topic_totals = Counter()
    for topic, year_data in topic_kw_year.items():
        for year, kw_counts in year_data.items():
            topic_totals[topic] += sum(kw_counts.values())
    top_topics = [t for t, _ in topic_totals.most_common(top_n_topics)]

    if not top_topics:
        return

    num_cols = 3
    num_rows = (len(top_topics) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, 6 * num_rows))
    axes = axes.flatten() if num_rows > 1 else [axes] if num_rows == 1 and num_cols == 1 else axes.flatten()

    for idx, topic in enumerate(top_topics):
        ax = axes[idx]

        # Build DataFrame for this topic
        rows = []
        for year in years:
            kw_counts = topic_kw_year[topic].get(year, {})
            for kw, count in kw_counts.items():
                rows.append({"Year": year, "Keyword": kw, "Count": count})

        if not rows:
            ax.set_visible(False)
            continue

        df_kw = pd.DataFrame(rows)
        pivot = df_kw.pivot_table(
            index="Year", columns="Keyword", values="Count", aggfunc="sum",
        ).fillna(0)

        # Top-k keywords + "others"
        top_kws = pivot.sum().sort_values(ascending=False).head(top_k_keywords).index.tolist()
        other_cols = [c for c in pivot.columns if c not in top_kws]
        if other_cols:
            pivot["others"] = pivot[other_cols].sum(axis=1)
            pivot = pivot[top_kws + ["others"]]
        else:
            pivot = pivot[top_kws]

        # Normalize to proportions
        row_sums = pivot.sum(axis=1)
        pivot = pivot.div(row_sums, axis=0).fillna(0)

        # Colors
        n_colors = len(pivot.columns)
        cmap = cm.get_cmap("terrain", n_colors + 1)
        colors = [mcolors.to_hex(cmap(i)) for i in range(n_colors)]

        pivot.plot.area(ax=ax, color=colors, legend=False, alpha=0.8)
        ax.set_title("\n".join(textwrap.wrap(topic, 30)), fontsize=12,
                     fontweight="bold")
        ax.set_ylabel("Proportion", fontsize=10)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        short_labels = [l[:25] + "..." if len(l) > 25 else l for l in labels]
        ax.legend(handles, short_labels, fontsize=7, loc="upper left",
                  bbox_to_anchor=(1.01, 1))

    for idx in range(len(top_topics), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Keyword Composition in Top {domain.title()} Topics Over Time",
        fontsize=18, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"n7_{domain}_keyword_composition.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close()
    logger.info(f"  Saved n7_{domain}_keyword_composition.png")


# ---------------------------------------------------------------------------
# Save data tables
# ---------------------------------------------------------------------------

def save_topic_counts_csv(topic_year_count, total_per_year, domain, years,
                          output_dir):
    """Save topic counts and percentages as CSV."""
    topic_totals = Counter()
    for (topic, year), count in topic_year_count.items():
        topic_totals[topic] += count

    csv_path = os.path.join(output_dir, f"{domain}_topic_year_counts.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["topic", "total"] + [str(y) for y in years])
        for topic, total in topic_totals.most_common():
            row = [topic, total]
            for y in years:
                row.append(topic_year_count.get((topic, y), 0))
            writer.writerow(row)
    logger.info(f"  Saved {csv_path}")


def save_cooccurrence_csv(matrix, output_dir):
    """Save co-occurrence matrix as CSV."""
    csv_path = os.path.join(output_dir, "cooccurrence_matrix.csv")
    matrix.to_csv(csv_path)
    logger.info(f"  Saved {csv_path}")


def save_trend_csv(trend_df, domain, output_dir):
    """Save trend analysis results."""
    csv_path = os.path.join(output_dir, f"{domain}_trend_analysis.csv")
    trend_df.sort_values("slope", ascending=False).to_csv(csv_path, index=False)
    logger.info(f"  Saved {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Step 6: Trend Analysis and Visualization")
    logger.info("=" * 70)

    years = list(range(args.start_year, args.end_year + 1))

    # ---- Load data ----
    df = load_articles(args.articles_csv)

    logger.info("Loading final topics...")
    method_topics = load_final_topics(args.topics_dir, "methodology")
    health_topics = load_final_topics(args.topics_dir, "health")

    kw_to_method = build_keyword_to_topic_map(method_topics)
    kw_to_health = build_keyword_to_topic_map(health_topics)
    logger.info(
        f"  Keyword-to-topic maps: {len(kw_to_method)} methodology keywords, "
        f"{len(kw_to_health)} health keywords"
    )

    # ---- Map articles to topics ----
    logger.info("Mapping articles to topics...")
    records = map_articles_to_topics(df, kw_to_method, kw_to_health)
    logger.info(f"  Mapped {len(records)} articles")

    # Count topic occurrences
    method_year_count = count_topic_year(records, "method_topics")
    health_year_count = count_topic_year(records, "health_topics")

    total_per_year = df[df["year"].between(years[0], years[-1])]["year"].value_counts().to_dict()

    # ---- RQ1: Total publication volume ----
    logger.info("\n--- RQ1: Publication Volume ---")
    plot_total_publication_volume(df, years, args.output_dir)

    # ---- N1: Per-journal comparison ----
    logger.info("\n--- N1: Journal Volume Comparison ---")
    plot_journal_volume_comparison(df, years, args.output_dir)

    # ---- RQ2: Topic popularity trends ----
    logger.info("\n--- RQ2: Topic Popularity Trends ---")
    for domain, tyc in [("methodology", method_year_count),
                        ("health", health_year_count)]:
        plot_topic_trends(tyc, total_per_year, domain, years,
                          args.output_dir, top_n=min(args.top_n, 16))

    # ---- RQ4: Top topics by time period ----
    logger.info("\n--- RQ4: Topics by Time Period ---")
    for domain, tyc in [("methodology", method_year_count),
                        ("health", health_year_count)]:
        plot_top_topics_by_period(tyc, domain, years, args.output_dir,
                                 top_n=args.top_n)

    # ---- RQ3: Co-occurrence ----
    logger.info("\n--- RQ3: Methodology x Health Co-occurrence ---")
    cooc_matrix = compute_cooccurrence(records)
    plot_cooccurrence_heatmap(cooc_matrix, args.output_dir)
    save_cooccurrence_csv(cooc_matrix, args.output_dir)

    # ---- N2: Journal specialization ----
    logger.info("\n--- N2: Journal Specialization ---")
    for domain, dk in [("methodology", "method_topics"),
                       ("health", "health_topics")]:
        plot_journal_topic_heatmap(records, dk, domain, args.output_dir)

    # ---- N3: Cross-journal topic sharing ----
    logger.info("\n--- N3: Cross-Journal Topic Sharing ---")
    for domain, dk in [("methodology", "method_topics"),
                       ("health", "health_topics")]:
        plot_topic_sharing(records, dk, domain, args.output_dir)

    # ---- N4: Topic emergence ----
    logger.info("\n--- N4: Topic Emergence Timeline ---")
    for domain, tyc in [("methodology", method_year_count),
                        ("health", health_year_count)]:
        plot_topic_emergence(tyc, domain, years, args.output_dir)

    # ---- N5: Rising/declining topics ----
    logger.info("\n--- N5: Rising/Declining Topics ---")
    for domain, tyc in [("methodology", method_year_count),
                        ("health", health_year_count)]:
        trend_df = compute_topic_trends(tyc, years)
        plot_rising_declining(trend_df, domain, args.output_dir)
        save_trend_csv(trend_df, domain, args.output_dir)

    # ---- N6: Journal similarity network ----
    logger.info("\n--- N6: Journal Similarity Network ---")
    plot_journal_similarity_network(records, "method_topics", args.output_dir)

    # ---- N7: Keyword composition ----
    logger.info("\n--- N7: Keyword Composition ---")
    for domain, kw_map in [("methodology", kw_to_method),
                           ("health", kw_to_health)]:
        plot_keyword_composition(df, kw_map, domain, years, args.output_dir)

    # ---- Save data tables ----
    logger.info("\n--- Saving Data Tables ---")
    for domain, tyc in [("methodology", method_year_count),
                        ("health", health_year_count)]:
        save_topic_counts_csv(tyc, total_per_year, domain, years, args.output_dir)

    # ---- Summary ----
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6 COMPLETE - TREND ANALYSIS & VISUALIZATION")
    logger.info("=" * 70)
    logger.info(f"\nAll outputs saved to {args.output_dir}/")
    logger.info("\nPlots generated:")
    logger.info("  rq1_total_volume.png                   Overall publication trends")
    logger.info("  rq2_[domain]_topic_trends.png          Top topic trends (count + %)")
    logger.info("  rq3_cooccurrence_heatmap.png           Method x Health co-occurrence")
    logger.info("  rq4_[domain]_topics_by_period.png      Topics by 5-year periods")
    logger.info("  n1_journal_volume_comparison.png       Journal volume ranking")
    logger.info("  n1_journal_trends.png                  Journal trend lines")
    logger.info("  n2_[domain]_journal_specialization.png Journal x Topic heatmap")
    logger.info("  n3_[domain]_topic_sharing.png          Cross-journal topic sharing")
    logger.info("  n4_[domain]_topic_emergence.png        Topic emergence timeline")
    logger.info("  n5_[domain]_rising_declining.png       Rising & declining topics")
    logger.info("  n6_journal_similarity_network.png      Journal similarity network")
    logger.info("  n7_[domain]_keyword_composition.png    Keyword composition areas")
    logger.info("\nData tables:")
    logger.info("  [domain]_topic_year_counts.csv         Topic counts per year")
    logger.info("  [domain]_trend_analysis.csv            Trend slopes per topic")
    logger.info("  cooccurrence_matrix.csv                Co-occurrence matrix")


if __name__ == "__main__":
    main()
