"""
analyze_pi_questions.py

Answers PI's research questions with visualizations:

  Q1: When and where (in which journal) did each topic first emerge?
  Q2: How are authorship compositions characterized (n_authors, n_affiliations,
      MD vs PhD vs MD+PhD) and how have they evolved over time?
  Q3: Which informatics journals have more team science publications?
  Q4: How often are emerging topics from team science vs solo PhD?
  Q5: Which topics are specialized vs shared across journals?

Usage:
    python analyze_pi_questions.py \\
        --articles-csv data/pubmed_all_journals_2011_2025_w_keywords.csv \\
        --authors-csv data/articles_with_authors.csv \\
        --topics-dir data/hierarchy \\
        --output-dir data/visualizations
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import textwrap
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# Clean journal abbreviations for plot labels
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze PI's research questions.",
    )
    parser.add_argument("--articles-csv", required=True)
    parser.add_argument("--authors-csv", required=True)
    parser.add_argument("--topics-dir", default="data/hierarchy")
    parser.add_argument("--output-dir", default="data/visualizations")
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-year", type=int, default=2025)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_articles(csv_path):
    logger.info(f"Loading articles from {csv_path}")
    df = pd.read_csv(csv_path, quoting=csv.QUOTE_ALL, dtype={"pmid": str})
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    logger.info(f"  Loaded {len(df)} articles")
    return df


def load_authors(csv_path):
    logger.info(f"Loading author data from {csv_path}")
    df = pd.read_csv(csv_path, dtype={"pmid": str})
    logger.info(f"  Loaded {len(df)} author records")
    return df


def load_topics(topics_dir, domain):
    path = os.path.join(topics_dir, f"{domain}_final_topics.json")
    with open(path, "r", encoding="utf-8") as f:
        topics = json.load(f)
    logger.info(f"  Loaded {len(topics)} {domain} topics")
    return topics


def build_keyword_to_topic_map(topics):
    kw_to_topics = defaultdict(set)
    for tid, data in topics.items():
        name = data["name"]
        for kw in data.get("keywords", []):
            kw_to_topics[kw.strip().lower()].add(name)
    return kw_to_topics


def map_articles_to_topics(df, kw_to_method, kw_to_health):
    """Returns a list of dicts with article topic assignments."""
    records = []
    for _, row in df.iterrows():
        kw_str = row.get("keywords", "")
        if pd.isna(kw_str) or not kw_str:
            method_topics = set()
            health_topics = set()
        else:
            method_topics = set()
            health_topics = set()
            for kw in str(kw_str).split(";"):
                kw = kw.strip().lower()
                if kw in kw_to_method:
                    method_topics.update(kw_to_method[kw])
                if kw in kw_to_health:
                    health_topics.update(kw_to_health[kw])
        records.append({
            "pmid": row["pmid"],
            "year": int(row["year"]),
            "journal": row.get("journal", "unknown"),
            "method_topics": method_topics,
            "health_topics": health_topics,
        })
    return records


# ---------------------------------------------------------------------------
# Q1: When and where did each topic first emerge?
# ---------------------------------------------------------------------------

def q1_first_emergence(records, output_dir):
    """For each topic, find the first article and first journal."""
    logger.info("\n=== Q1: First emergence of each topic ===")

    sorted_records = sorted(records, key=lambda x: (x["year"], x["pmid"]))

    method_first = {}
    health_first = {}
    for r in sorted_records:
        for topic in r["method_topics"]:
            if topic not in method_first:
                method_first[topic] = {
                    "year": r["year"],
                    "journal": r["journal"],
                    "pmid": r["pmid"],
                }
        for topic in r["health_topics"]:
            if topic not in health_first:
                health_first[topic] = {
                    "year": r["year"],
                    "journal": r["journal"],
                    "pmid": r["pmid"],
                }

    # Save data tables
    for domain, first_dict in [("methodology", method_first), ("health", health_first)]:
        rows = []
        for topic, info in first_dict.items():
            rows.append({
                "topic": topic,
                "first_year": info["year"],
                "first_journal": info["journal"],
                "first_pmid": info["pmid"],
            })
        df_first = pd.DataFrame(rows).sort_values(["first_year", "topic"])
        csv_path = os.path.join(output_dir, f"pi_q1_{domain}_first_emergence.csv")
        df_first.to_csv(csv_path, index=False)
        logger.info(f"  Saved {csv_path}")

    # Visualization 1: Pioneer journal frequency (which journals first published topics)
    for domain, first_dict in [("methodology", method_first), ("health", health_first)]:
        pioneer_counts = Counter(info["journal"] for info in first_dict.values())
        # Use abbreviated names
        abbr_counts = Counter(
            {JOURNAL_ABBR.get(j, j[:25]): c for j, c in pioneer_counts.items()}
        )
        top = dict(abbr_counts.most_common(15))

        fig, ax = plt.subplots(figsize=(11, 7))
        names = list(top.keys())
        counts = list(top.values())
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
        ax.barh(names[::-1], counts[::-1], color=colors[::-1], edgecolor="white")
        for i, c in enumerate(counts[::-1]):
            ax.text(c + 0.3, i, str(c), va="center", fontsize=10)
        ax.set_xlabel(f"Number of {domain} topics first published in this journal",
                      fontsize=12)
        ax.set_title(
            f"Q1: Pioneer Journals - {domain.title()} Topics\n"
            f"(which journals were FIRST to publish each topic)",
            fontsize=14,
        )
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        out = os.path.join(output_dir, f"pi_q1_{domain}_pioneer_journals.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved {out}")

    # Visualization 2: First emergence year distribution
    for domain, first_dict in [("methodology", method_first), ("health", health_first)]:
        years = [info["year"] for info in first_dict.values()]
        year_counts = Counter(years)

        fig, ax = plt.subplots(figsize=(11, 5))
        sorted_years = sorted(year_counts.keys())
        ax.bar(sorted_years, [year_counts[y] for y in sorted_years],
               color="#3498db", edgecolor="white")
        for y in sorted_years:
            ax.text(y, year_counts[y] + 0.5, str(year_counts[y]),
                    ha="center", fontsize=9)
        ax.set_xlabel("First Year Topic Appeared", fontsize=12)
        ax.set_ylabel("Number of Topics", fontsize=12)
        ax.set_title(
            f"Q1: When did each {domain} topic first appear?\n"
            f"(distribution of first-appearance years across {len(first_dict)} topics)",
            fontsize=13,
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        out = os.path.join(output_dir, f"pi_q1_{domain}_emergence_year_dist.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Q2: Authorship characteristics over time
# ---------------------------------------------------------------------------

def q2_authorship_over_time(merged_df, years, output_dir):
    """Track median/max/min authors and affiliation patterns over time."""
    logger.info("\n=== Q2: Authorship characteristics over time ===")

    # Filter to articles with author data
    valid = merged_df.dropna(subset=["n_authors"]).copy()
    valid["n_authors"] = valid["n_authors"].astype(int)
    valid["n_unique_affiliations"] = valid["n_unique_affiliations"].fillna(0).astype(int)
    valid["has_clinical"] = valid["has_clinical"].fillna(0).astype(int)
    valid["has_computational"] = valid["has_computational"].fillna(0).astype(int)
    valid["is_interdisciplinary"] = valid["is_interdisciplinary"].fillna(0).astype(int)

    # Compute per-year statistics
    yearly_stats = []
    for y in years:
        year_data = valid[valid["year"] == y]
        if len(year_data) == 0:
            continue
        n_auths = year_data["n_authors"]
        n_affs = year_data["n_unique_affiliations"]
        yearly_stats.append({
            "year": y,
            "n_articles": len(year_data),
            "median_authors": n_auths.median(),
            "mean_authors": n_auths.mean(),
            "max_authors": n_auths.max(),
            "min_authors": n_auths.min(),
            "p25_authors": n_auths.quantile(0.25),
            "p75_authors": n_auths.quantile(0.75),
            "median_affs": n_affs.median(),
            "mean_affs": n_affs.mean(),
            "p75_affs": n_affs.quantile(0.75),
            "pct_clinical_only": (
                (year_data["has_clinical"] == 1) & (year_data["has_computational"] == 0)
            ).mean() * 100,
            "pct_computational_only": (
                (year_data["has_clinical"] == 0) & (year_data["has_computational"] == 1)
            ).mean() * 100,
            "pct_interdisciplinary": (
                year_data["is_interdisciplinary"] == 1
            ).mean() * 100,
            "pct_neither": (
                (year_data["has_clinical"] == 0) & (year_data["has_computational"] == 0)
            ).mean() * 100,
        })

    stats_df = pd.DataFrame(yearly_stats)
    csv_path = os.path.join(output_dir, "pi_q2_authorship_over_time.csv")
    stats_df.to_csv(csv_path, index=False)
    logger.info(f"  Saved {csv_path}")

    # Visualization 1: Author count over time (median + percentile band)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(
        stats_df["year"],
        stats_df["p25_authors"],
        stats_df["p75_authors"],
        alpha=0.3, color="#3498db", label="25-75 percentile",
    )
    ax.plot(stats_df["year"], stats_df["median_authors"],
            "o-", color="#2c3e50", linewidth=2.5, markersize=7, label="Median")
    ax.plot(stats_df["year"], stats_df["mean_authors"],
            "s--", color="#e74c3c", linewidth=2, markersize=6, label="Mean")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Authors per Article", fontsize=12)
    ax.set_title(
        "Q2: Co-Author Count per Article Over Time\n"
        "(Multi-Journal Biomedical Informatics)",
        fontsize=14,
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, "pi_q2_authors_over_time.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {out}")

    # Visualization 2: Affiliation count over time
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stats_df["year"], stats_df["median_affs"],
            "o-", color="#16a085", linewidth=2.5, markersize=7, label="Median")
    ax.plot(stats_df["year"], stats_df["mean_affs"],
            "s--", color="#e67e22", linewidth=2, markersize=6, label="Mean")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Distinct Affiliations per Article", fontsize=12)
    ax.set_title(
        "Q2: Distinct Author Affiliations per Article Over Time\n"
        "(proxy for inter-institutional collaboration)",
        fontsize=14,
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, "pi_q2_affiliations_over_time.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {out}")

    # Visualization 3: Author composition over time (stacked area)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(
        stats_df["year"],
        stats_df["pct_interdisciplinary"],
        stats_df["pct_clinical_only"],
        stats_df["pct_computational_only"],
        stats_df["pct_neither"],
        labels=[
            "Interdisciplinary (Clinical + Computational)",
            "Clinical only (medical/hospital affiliations)",
            "Computational only (CS/informatics/engineering)",
            "Other / unclassified",
        ],
        colors=["#9b59b6", "#e74c3c", "#3498db", "#bdc3c7"],
        alpha=0.85,
    )
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Percentage of Articles (%)", fontsize=12)
    ax.set_title(
        "Q2: Author Composition Over Time\n"
        "(based on author affiliations: clinical vs computational vs interdisciplinary)",
        fontsize=14,
    )
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = os.path.join(output_dir, "pi_q2_composition_over_time.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Q3: Which journals have more team science?
# ---------------------------------------------------------------------------

def q3_team_science_by_journal(merged_df, output_dir):
    """Rank journals by team science indicators."""
    logger.info("\n=== Q3: Team science by journal ===")

    valid = merged_df.dropna(subset=["n_authors"]).copy()
    valid["n_authors"] = valid["n_authors"].astype(int)
    valid["n_unique_affiliations"] = valid["n_unique_affiliations"].fillna(0).astype(int)
    valid["is_interdisciplinary"] = valid["is_interdisciplinary"].fillna(0).astype(int)

    journal_stats = valid.groupby("journal").agg(
        n_articles=("pmid", "count"),
        median_authors=("n_authors", "median"),
        mean_authors=("n_authors", "mean"),
        max_authors=("n_authors", "max"),
        median_affs=("n_unique_affiliations", "median"),
        mean_affs=("n_unique_affiliations", "mean"),
        pct_interdisciplinary=("is_interdisciplinary", lambda x: x.mean() * 100),
    ).reset_index()

    # Filter to journals with at least 100 articles
    journal_stats = journal_stats[journal_stats["n_articles"] >= 100]

    # Add abbreviations
    journal_stats["abbr"] = journal_stats["journal"].map(
        lambda j: JOURNAL_ABBR.get(j, j[:25])
    )

    # Save table
    csv_path = os.path.join(output_dir, "pi_q3_team_science_by_journal.csv")
    journal_stats.sort_values("mean_authors", ascending=False).to_csv(
        csv_path, index=False,
    )
    logger.info(f"  Saved {csv_path}")

    # Visualization: 3-panel bar chart
    fig, axes = plt.subplots(1, 3, figsize=(20, 9))

    # Panel 1: Mean authors per article
    sorted_auth = journal_stats.sort_values("mean_authors", ascending=True).tail(20)
    axes[0].barh(
        sorted_auth["abbr"],
        sorted_auth["mean_authors"],
        color="#3498db",
        edgecolor="white",
    )
    for i, (_, row) in enumerate(sorted_auth.iterrows()):
        axes[0].text(
            row["mean_authors"] + 0.1, i,
            f'{row["mean_authors"]:.1f}',
            va="center", fontsize=9,
        )
    axes[0].set_xlabel("Mean Authors per Article", fontsize=12)
    axes[0].set_title("Mean Authors per Article", fontsize=13)
    axes[0].grid(True, alpha=0.3, axis="x")

    # Panel 2: Mean unique affiliations
    sorted_aff = journal_stats.sort_values("mean_affs", ascending=True).tail(20)
    axes[1].barh(
        sorted_aff["abbr"],
        sorted_aff["mean_affs"],
        color="#16a085",
        edgecolor="white",
    )
    for i, (_, row) in enumerate(sorted_aff.iterrows()):
        axes[1].text(
            row["mean_affs"] + 0.05, i,
            f'{row["mean_affs"]:.1f}',
            va="center", fontsize=9,
        )
    axes[1].set_xlabel("Mean Distinct Affiliations per Article", fontsize=12)
    axes[1].set_title("Multi-Institutional Collaboration", fontsize=13)
    axes[1].grid(True, alpha=0.3, axis="x")

    # Panel 3: % interdisciplinary
    sorted_inter = journal_stats.sort_values("pct_interdisciplinary", ascending=True).tail(20)
    axes[2].barh(
        sorted_inter["abbr"],
        sorted_inter["pct_interdisciplinary"],
        color="#9b59b6",
        edgecolor="white",
    )
    for i, (_, row) in enumerate(sorted_inter.iterrows()):
        axes[2].text(
            row["pct_interdisciplinary"] + 0.5, i,
            f'{row["pct_interdisciplinary"]:.1f}%',
            va="center", fontsize=9,
        )
    axes[2].set_xlabel("% Interdisciplinary Articles", fontsize=12)
    axes[2].set_title(
        "% Articles with Both Clinical\n& Computational Authors",
        fontsize=13,
    )
    axes[2].grid(True, alpha=0.3, axis="x")

    fig.suptitle(
        "Q3: Team Science Indicators by Journal "
        "(top 20 by each metric, journals with >=100 articles)",
        fontsize=15, y=1.02,
    )
    plt.tight_layout()
    out = os.path.join(output_dir, "pi_q3_team_science_by_journal.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Q4: How often are emerging topics from team science vs solo PhD?
# ---------------------------------------------------------------------------

def q4_emerging_topics_team_science(merged_df, records, output_dir,
                                     emerging_year_cutoff=2018):
    """Compare team science composition between emerging and established topics."""
    logger.info("\n=== Q4: Team science for emerging topics ===")

    # Determine emerging topics: those whose first year reaching 25% of peak
    # is >= emerging_year_cutoff
    method_year = defaultdict(list)
    for r in records:
        for t in r["method_topics"]:
            method_year[t].append(r["year"])

    emerging_method_topics = set()
    established_method_topics = set()
    for topic, years_list in method_year.items():
        year_counts = Counter(years_list)
        peak = max(year_counts.values())
        threshold = peak * 0.25
        for y in sorted(year_counts.keys()):
            if year_counts[y] >= threshold:
                if y >= emerging_year_cutoff:
                    emerging_method_topics.add(topic)
                else:
                    established_method_topics.add(topic)
                break

    logger.info(
        f"  Emerging methodology topics (takeoff >= {emerging_year_cutoff}): "
        f"{len(emerging_method_topics)}"
    )
    logger.info(
        f"  Established methodology topics (takeoff < {emerging_year_cutoff}): "
        f"{len(established_method_topics)}"
    )

    # For each article, determine if it has emerging or established topics
    pmid_categories = {}  # pmid -> "emerging" / "established" / "both"
    for r in records:
        has_emerging = bool(r["method_topics"] & emerging_method_topics)
        has_established = bool(r["method_topics"] & established_method_topics)
        if has_emerging and not has_established:
            pmid_categories[r["pmid"]] = "emerging"
        elif has_established and not has_emerging:
            pmid_categories[r["pmid"]] = "established"
        elif has_emerging and has_established:
            pmid_categories[r["pmid"]] = "both"

    # Merge with author data
    merged_df["category"] = merged_df["pmid"].map(pmid_categories)
    valid = merged_df.dropna(subset=["n_authors", "category"]).copy()
    valid["n_authors"] = valid["n_authors"].astype(int)

    # Define team types
    def team_type(row):
        n = row["n_authors"]
        inter = int(row.get("is_interdisciplinary", 0))
        if n <= 1:
            return "Solo author"
        if n <= 3:
            return "Small team (2-3)"
        if n <= 10:
            return "Medium team (4-10)"
        return "Large team (11+)"

    valid["team_type"] = valid.apply(team_type, axis=1)

    # Compute proportions for emerging vs established
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, category, title in [
        (axes[0], "emerging", f"Emerging Topics (took off >= {emerging_year_cutoff})"),
        (axes[1], "established", f"Established Topics (active before {emerging_year_cutoff})"),
    ]:
        cat_data = valid[valid["category"] == category]
        if len(cat_data) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            continue

        team_counts = cat_data["team_type"].value_counts()
        order = ["Solo author", "Small team (2-3)", "Medium team (4-10)", "Large team (11+)"]
        team_counts = team_counts.reindex(order, fill_value=0)
        proportions = team_counts / team_counts.sum() * 100

        colors = ["#e74c3c", "#f39c12", "#3498db", "#27ae60"]
        bars = ax.bar(order, proportions, color=colors, edgecolor="white")
        for bar, p in zip(bars, proportions):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{p:.1f}%",
                ha="center", fontsize=10,
            )

        # Add interdisciplinary rate
        inter_rate = cat_data["is_interdisciplinary"].mean() * 100
        ax.text(
            0.98, 0.95,
            f"Interdisciplinary: {inter_rate:.1f}%\n"
            f"n={len(cat_data):,} articles",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9),
        )

        ax.set_ylabel("% of Articles", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, max(proportions.max() + 10, 50))
        ax.grid(True, alpha=0.3, axis="y")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=10)

    fig.suptitle(
        "Q4: Team Science in Emerging vs Established Topics",
        fontsize=15, y=1.02,
    )
    plt.tight_layout()
    out = os.path.join(output_dir, "pi_q4_emerging_team_science.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {out}")

    # Save data
    summary = []
    for category in ["emerging", "established", "both"]:
        cat_data = valid[valid["category"] == category]
        if len(cat_data) == 0:
            continue
        summary.append({
            "category": category,
            "n_articles": len(cat_data),
            "median_authors": cat_data["n_authors"].median(),
            "mean_authors": cat_data["n_authors"].mean(),
            "pct_solo": (cat_data["n_authors"] == 1).mean() * 100,
            "pct_large_team": (cat_data["n_authors"] >= 11).mean() * 100,
            "pct_interdisciplinary": cat_data["is_interdisciplinary"].mean() * 100,
        })
    csv_path = os.path.join(output_dir, "pi_q4_emerging_team_science.csv")
    pd.DataFrame(summary).to_csv(csv_path, index=False)
    logger.info(f"  Saved {csv_path}")


# ---------------------------------------------------------------------------
# Q5: Specialized vs shared topics
# ---------------------------------------------------------------------------

def q5_specialized_vs_shared(records, output_dir):
    """Compute journal-distribution entropy for each topic."""
    logger.info("\n=== Q5: Specialized vs shared topics ===")

    n_journals = len({r["journal"] for r in records})
    log_n = math.log(n_journals)

    for domain, key in [("methodology", "method_topics"), ("health", "health_topics")]:
        topic_journals = defaultdict(Counter)
        topic_total = Counter()
        for r in records:
            for topic in r[key]:
                topic_journals[topic][r["journal"]] += 1
                topic_total[topic] += 1

        # Compute entropy and normalized entropy
        rows = []
        for topic, j_counts in topic_journals.items():
            total = sum(j_counts.values())
            if total == 0:
                continue
            probs = [c / total for c in j_counts.values() if c > 0]
            H = -sum(p * math.log(p) for p in probs)
            H_norm = H / log_n
            rows.append({
                "topic": topic,
                "total_articles": total,
                "n_journals": len(j_counts),
                "entropy": H,
                "normalized_entropy": H_norm,
                "max_journal_share": max(j_counts.values()) / total,
            })

        df = pd.DataFrame(rows).sort_values("normalized_entropy", ascending=False)
        csv_path = os.path.join(output_dir, f"pi_q5_{domain}_specialization.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"  Saved {csv_path}")

        # Visualization 1: Scatter plot (entropy vs total articles)
        fig, ax = plt.subplots(figsize=(13, 8))
        sizes = np.sqrt(df["total_articles"]) * 5
        scatter = ax.scatter(
            df["total_articles"],
            df["normalized_entropy"],
            s=sizes,
            c=df["normalized_entropy"],
            cmap="RdYlGn",
            edgecolors="black",
            linewidths=0.5,
            alpha=0.7,
        )
        ax.set_xscale("log")
        ax.set_xlabel("Total Articles (log scale)", fontsize=12)
        ax.set_ylabel("Normalized Entropy (0=specialized, 1=evenly shared)",
                      fontsize=12)
        ax.set_title(
            f"Q5: {domain.title()} Topic Specialization vs Sharing\n"
            f"(higher = topic spread evenly across journals; "
            f"lower = topic concentrated in few journals)",
            fontsize=13,
        )
        ax.axhline(y=df["normalized_entropy"].median(), color="gray",
                   linestyle="--", alpha=0.5,
                   label=f"Median entropy = {df['normalized_entropy'].median():.2f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Annotate top 5 most specialized and top 5 most shared
        most_specialized = df.nsmallest(5, "normalized_entropy")
        most_shared = df.nlargest(5, "normalized_entropy")
        for _, row in most_specialized.iterrows():
            label = textwrap.shorten(row["topic"], width=22, placeholder="...")
            ax.annotate(
                label, (row["total_articles"], row["normalized_entropy"]),
                fontsize=8, alpha=0.8,
                xytext=(5, -8), textcoords="offset points",
            )
        for _, row in most_shared.iterrows():
            label = textwrap.shorten(row["topic"], width=22, placeholder="...")
            ax.annotate(
                label, (row["total_articles"], row["normalized_entropy"]),
                fontsize=8, alpha=0.8,
                xytext=(5, 5), textcoords="offset points",
            )

        plt.tight_layout()
        out = os.path.join(output_dir, f"pi_q5_{domain}_specialization_scatter.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved {out}")

        # Visualization 2: Top 15 specialized + top 15 shared bar charts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Filter to topics with at least 50 articles to avoid noise
        df_filt = df[df["total_articles"] >= 50]
        most_special = df_filt.nsmallest(15, "normalized_entropy").iloc[::-1]
        most_shared = df_filt.nlargest(15, "normalized_entropy").iloc[::-1]

        ax1.barh(
            most_special["topic"],
            most_special["normalized_entropy"],
            color="#e74c3c",
            edgecolor="white",
        )
        for i, row in enumerate(most_special.itertuples()):
            ax1.text(
                row.normalized_entropy + 0.005,
                i,
                f"{row.normalized_entropy:.2f} ({row.n_journals} journals, "
                f"{row.total_articles:,} arts)",
                va="center", fontsize=8,
            )
        ax1.set_xlabel("Normalized Entropy", fontsize=11)
        ax1.set_title(f"Most SPECIALIZED {domain} topics\n(concentrated in few journals)",
                      fontsize=13)
        ax1.set_xlim(0, 1)
        ax1.grid(True, alpha=0.3, axis="x")

        ax2.barh(
            most_shared["topic"],
            most_shared["normalized_entropy"],
            color="#27ae60",
            edgecolor="white",
        )
        for i, row in enumerate(most_shared.itertuples()):
            ax2.text(
                row.normalized_entropy + 0.005,
                i,
                f"{row.normalized_entropy:.2f} ({row.n_journals} journals, "
                f"{row.total_articles:,} arts)",
                va="center", fontsize=8,
            )
        ax2.set_xlabel("Normalized Entropy", fontsize=11)
        ax2.set_title(f"Most SHARED {domain} topics\n(spread across many journals)",
                      fontsize=13)
        ax2.set_xlim(0, 1)
        ax2.grid(True, alpha=0.3, axis="x")

        fig.suptitle(
            f"Q5: {domain.title()} Topics — Specialized vs Shared",
            fontsize=15, y=1.02,
        )
        plt.tight_layout()
        out = os.path.join(output_dir, f"pi_q5_{domain}_specialization_ranking.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Analyzing PI's Research Questions")
    logger.info("=" * 70)

    # Load data
    articles_df = load_articles(args.articles_csv)
    authors_df = load_authors(args.authors_csv)
    method_topics = load_topics(args.topics_dir, "methodology")
    health_topics = load_topics(args.topics_dir, "health")

    kw_to_method = build_keyword_to_topic_map(method_topics)
    kw_to_health = build_keyword_to_topic_map(health_topics)

    logger.info("Mapping articles to topics...")
    records = map_articles_to_topics(articles_df, kw_to_method, kw_to_health)

    # Merge articles with author data
    merged_df = articles_df.merge(authors_df, on="pmid", how="left")
    logger.info(
        f"Merged: {len(merged_df)} articles, "
        f"{merged_df['n_authors'].notna().sum()} with author data"
    )

    years = list(range(args.start_year, args.end_year + 1))

    # Run all analyses
    q1_first_emergence(records, args.output_dir)
    q2_authorship_over_time(merged_df, years, args.output_dir)
    q3_team_science_by_journal(merged_df, args.output_dir)
    q4_emerging_topics_team_science(merged_df, records, args.output_dir)
    q5_specialized_vs_shared(records, args.output_dir)

    logger.info("\n" + "=" * 70)
    logger.info("PI ANALYSIS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nAll outputs in {args.output_dir}/ with prefix 'pi_q'")


if __name__ == "__main__":
    main()
