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
    n_journals = df["journal"].nunique()
    ax.set_title(f"Total Publication Volume (All {n_journals} Journals)", fontsize=16)
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
        ax.plot(y_counts.index, y_counts.values, "o-", label=_abbr(journal),
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

def compute_cooccurrence(records, top_n_method=30, top_n_health=30):
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
    """Plot methodology x health co-occurrence heatmap.

    Suppresses annotations for low-count cells (<5% of max) so dense regions
    stay readable while genuine peaks remain labeled.
    """
    fig, ax = plt.subplots(figsize=(22, 15))
    threshold = max(5, matrix.values.max() * 0.05)
    annot_labels = matrix.applymap(
        lambda v: str(int(v)) if v >= threshold else ""
    )
    sns.heatmap(
        matrix, annot=annot_labels, fmt="", cmap="YlOrRd", ax=ax,
        linewidths=0.5, linecolor="white",
        annot_kws={"fontsize": 8},
    )
    ax.set_title(
        f"Methodology × Health Domain Co-occurrence\n"
        f"(values shown only when ≥{int(threshold)} co-occurring articles)",
        fontsize=16,
    )
    ax.set_xlabel("Health Topics", fontsize=14)
    ax.set_ylabel("Methodology Topics", fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right",
             rotation_mode="anchor", fontsize=10)
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

    matrix.index = [_abbr(j) for j in matrix.index]

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
    Show the year of steepest year-over-year growth for each top topic.
    "First appearance" is not meaningful since most journals existed in 2011;
    instead, highlight WHEN each topic's growth accelerated most.
    """
    topic_totals = Counter()
    for (topic, year), count in topic_year_count.items():
        topic_totals[topic] += count

    rows = []
    for topic, total in topic_totals.items():
        if total < 20:
            continue
        yearly = [topic_year_count.get((topic, y), 0) for y in years]
        growth = np.diff(yearly)
        if len(growth) == 0:
            continue
        max_idx = int(np.argmax(growth))
        rows.append({
            "topic": topic,
            "total": total,
            "breakthrough_year": years[max_idx + 1],
            "max_yoy_growth": int(growth[max_idx]),
        })

    if not rows:
        logger.warning(f"  No topics with >=20 articles for n4_{domain}")
        return
    rdf = pd.DataFrame(rows).nlargest(top_n, "total")
    rdf = rdf.sort_values(["breakthrough_year", "total"], ascending=[False, True])

    fig, ax = plt.subplots(figsize=(15, max(10, top_n * 0.38)))
    y_pos = np.arange(len(rdf))
    norm = (rdf["breakthrough_year"].values - years[0]) / max(years[-1] - years[0], 1)
    colors = plt.cm.viridis(norm)
    sizes = (rdf["max_yoy_growth"] / max(rdf["max_yoy_growth"].max(), 1)) * 500 + 50

    ax.scatter(rdf["breakthrough_year"], y_pos, s=sizes, c=colors,
               edgecolors="black", linewidths=1, zorder=3, alpha=0.85)
    ax.hlines(y_pos, years[0], rdf["breakthrough_year"],
              colors="#bdc3c7", linewidth=0.8, alpha=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(rdf["topic"], fontsize=9)

    for i, row in enumerate(rdf.itertuples()):
        ax.text(years[-1] + 0.5, i,
                f"{row.breakthrough_year} (+{row.max_yoy_growth} YoY)",
                va="center", fontsize=8, color="#2c3e50")

    ax.set_xlabel("Year of Steepest Year-over-Year Growth", fontsize=12)
    ax.set_title(
        f"N4: {domain.title()} — Year of Maximum Growth Acceleration\n"
        f"(dot size = magnitude of YoY increase; shows WHEN each topic grew fastest)",
        fontsize=14,
    )
    ax.set_xlim(years[0] - 0.5, years[-1] + 4.5)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
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
    Compute linear trend slopes for each topic with significance tests.
    Positive slope = rising, negative = declining.
    Uses scipy.stats.linregress for two-sided test that slope != 0;
    also reports Holm-Bonferroni adjusted p-values across topics.
    """
    from scipy import stats as sstats

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
        if counts.sum() == 0:
            continue
        reg = sstats.linregress(year_arr, counts)
        mean_count = counts.mean()
        rel_slope = reg.slope / mean_count if mean_count > 0 else 0

        trends.append({
            "topic": topic,
            "total": total,
            "slope": reg.slope,
            "relative_slope": rel_slope,
            "mean_count": mean_count,
            "recent_count": counts[-3:].mean(),
            "early_count": counts[:3].mean(),
            "r_squared": reg.rvalue ** 2,
            "p_value": reg.pvalue,
        })

    df = pd.DataFrame(trends)
    if not df.empty:
        # Holm-Bonferroni multiple-testing correction across topics in domain.
        order = np.argsort(df["p_value"].values)
        sorted_p = df["p_value"].values[order]
        n = len(sorted_p)
        adj_sorted = np.maximum.accumulate(
            np.minimum((n - np.arange(n)) * sorted_p, 1.0)
        )
        adj = np.empty(n)
        adj[order] = adj_sorted
        df["p_value_holm"] = adj
        df["significant_holm"] = df["p_value_holm"] < 0.05
    return df


def plot_rising_declining(trend_df, domain, output_dir, top_n=10,
                          topic_year_count=None, years=None):
    """
    Plot rising/declining topics by percentage-share slope.
    Absolute slope is misleading because the field is growing overall; a topic
    with a small positive absolute slope can still be losing market share.
    Share slope (% of year's publications) captures true decline.
    """
    if trend_df.empty or topic_year_count is None or years is None:
        return

    total_per_year = {
        y: sum(c for (_, yy), c in topic_year_count.items() if yy == y)
        for y in years
    }
    year_arr = np.array(years, dtype=float)

    rows = []
    for _, row in trend_df.iterrows():
        topic = row["topic"]
        total = int(row["total"])
        if total < 50:
            continue
        shares = [(topic_year_count.get((topic, y), 0) /
                   max(total_per_year[y], 1)) * 100 for y in years]
        share_slope = np.polyfit(year_arr, shares, 1)[0]
        rows.append({"topic": topic, "total": total, "share_slope": share_slope})
    rdf = pd.DataFrame(rows)
    if rdf.empty:
        return

    rising = rdf.nlargest(top_n, "share_slope")
    truly_declining = rdf[rdf["share_slope"] < 0].nsmallest(top_n, "share_slope")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    ax1.barh(rising["topic"], rising["share_slope"],
             color="#2ecc71", edgecolor="white")
    for i, (_, r) in enumerate(rising.iterrows()):
        ax1.text(r["share_slope"] + 0.005, i,
                 f'+{r["share_slope"]:.3f}%/yr ({r["total"]:,} articles)',
                 va="center", fontsize=9)
    ax1.set_xlabel("Share Slope (%-points per year)", fontsize=12)
    ax1.set_title(
        f"Top {top_n} Rising {domain.title()} Topics\n"
        f"(gaining share of total publications)", fontsize=13)
    ax1.grid(True, alpha=0.3, axis="x")

    if len(truly_declining) > 0:
        ax2.barh(truly_declining["topic"], truly_declining["share_slope"],
                 color="#e74c3c", edgecolor="white")
        # Place labels on the right (positive) side of the y-axis so they
        # never overlap the bars themselves.
        max_abs = float(truly_declining["share_slope"].abs().max())
        for i, (_, r) in enumerate(truly_declining.iterrows()):
            ax2.text(max_abs * 0.05, i,
                     f'{r["share_slope"]:.3f}%/yr ({r["total"]:,} articles)',
                     va="center", fontsize=9, ha="left")
        ax2.set_xlabel("Share Slope (%-points per year)", fontsize=12)
        ax2.set_xlim(left=-max_abs * 1.1, right=max_abs * 0.85)
        n_shown = len(truly_declining)
        ax2.set_title(
            f"Declining {domain.title()} Topics (n={n_shown})\n"
            f"(losing share — ALL have negative share trend)", fontsize=13)
    else:
        ax2.text(0.5, 0.5, "No topics with negative\nshare slope found",
                 ha="center", va="center", fontsize=14, transform=ax2.transAxes)
        ax2.set_title(f"Declining {domain.title()} Topics", fontsize=13)
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


def _abbr(j):
    return JOURNAL_ABBR.get(j, j[:25])


def plot_journal_similarity_network(records, domain_key, output_dir):
    """
    Network graph showing journal similarity based on topic distributions.
    Uses Kamada-Kawai layout, short journal abbreviations, and a size legend.
    Edges are drawn above the 50th percentile of cosine similarity.
    """
    try:
        import networkx as nx
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        from matplotlib.lines import Line2D
    except ImportError:
        logger.warning("  networkx/sklearn not installed, skipping N6")
        return

    journal_topic_vec = defaultdict(Counter)
    journal_totals = Counter()
    for r in records:
        journal_totals[r["journal"]] += 1
        for topic in r[domain_key]:
            journal_topic_vec[r["journal"]][topic] += 1

    journals = sorted(journal_topic_vec.keys())
    all_topics = sorted({t for jc in journal_topic_vec.values() for t in jc})

    vectors = []
    for j in journals:
        total = journal_totals[j]
        vectors.append([journal_topic_vec[j].get(t, 0) / max(total, 1)
                        for t in all_topics])
    mat = np.array(vectors)
    sim = cos_sim(mat)

    G = nx.Graph()
    for j in journals:
        G.add_node(_abbr(j), size=journal_totals[j])

    upper = sim[np.triu_indices_from(sim, k=1)]
    threshold = np.percentile(upper, 50)
    for i in range(len(journals)):
        for j in range(i + 1, len(journals)):
            if sim[i, j] > threshold:
                G.add_edge(_abbr(journals[i]), _abbr(journals[j]),
                           weight=float(sim[i, j]))

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
    for node, (x, y) in pos.items():
        ax.text(x, y + 0.04, node, ha="center", va="center",
                fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#1f618d", alpha=0.92))

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
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(28, 6 * num_rows))
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

        # Legend — wrap long keywords across lines instead of truncating
        handles, labels = ax.get_legend_handles_labels()
        wrapped_labels = ["\n".join(textwrap.wrap(l, 28)) for l in labels]
        ax.legend(handles, wrapped_labels, fontsize=8, loc="upper left",
                  bbox_to_anchor=(1.02, 1), borderaxespad=0,
                  handlelength=1.5)

    for idx in range(len(top_topics), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Keyword Composition in Top {domain.title()} Topics Over Time",
        fontsize=18, y=1.01,
    )
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.55)
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
    """Save trend analysis results.
    Topics with fewer than compute_topic_trends.min_articles (default 10)
    are excluded here — row count may be less than the total topic count.
    """
    csv_path = os.path.join(output_dir, f"{domain}_trend_analysis.csv")
    trend_df.sort_values("slope", ascending=False).to_csv(csv_path, index=False)
    logger.info(f"  Saved {csv_path} ({len(trend_df)} topics with >=10 articles)")


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
        plot_rising_declining(trend_df, domain, args.output_dir,
                              topic_year_count=tyc, years=years)
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
