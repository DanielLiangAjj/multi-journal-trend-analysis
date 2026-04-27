"""
analyze_all_questions.py

Comprehensive analysis answering all PI research questions + new analyses.

PI Questions:
  Q1: When and where (in which journal) did each topic first emerge?
  Q2: Authorship characteristics over time (requires author data)
  Q3: Which journals have more team science? (requires author data)
  Q4: How often are emerging topics from team science? (requires author data)
  Q5: Which topics are specialized vs shared across journals?

New Analyses:
  Q6: Journal selection guidance for researchers
  Q7: Historical event correlation with topic trends

Usage:
    # Without author data (Q1, Q5, Q6, Q7 only):
    python analyze_all_questions.py \\
        --articles-csv data/pubmed_all_journals_2011_2025_w_keywords.csv \\
        --topics-dir data/hierarchy \\
        --output-dir data/visualizations

    # With author data (all questions):
    python analyze_all_questions.py \\
        --articles-csv data/pubmed_all_journals_2011_2025_w_keywords.csv \\
        --authors-csv data/articles_with_authors.csv \\
        --topics-dir data/hierarchy \\
        --output-dir data/visualizations

Requirements:
    pip install matplotlib seaborn pandas numpy scikit-learn
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
import matplotlib.patches as mpatches
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

# Historical events for Q7
HISTORICAL_EVENTS = [
    (2012, "AlexNet / ImageNet", "ai", "Deep Learning begins"),
    (2013, "HITECH MU Stage 2", "policy", "EHR adoption push"),
    (2015, "Precision Med Initiative", "policy", "Genomics/personalized medicine"),
    (2015, "ResNet", "ai", "Deep CNN architectures"),
    (2016, "21st Century Cures Act", "policy", "Interoperability mandate"),
    (2017, "Transformer paper", "ai", "Attention Is All You Need"),
    (2018, "BERT published", "ai", "NLP revolution"),
    (2018, "GDPR takes effect", "policy", "Privacy/data protection"),
    (2020, "COVID-19 pandemic", "pandemic", "Digital health & surveillance surge"),
    (2021, "FDA AI/ML Action Plan", "policy", "AI medical device regulation"),
    (2022, "ChatGPT released", "ai", "LLM public awareness"),
    (2023, "GPT-4 released", "ai", "LLM in healthcare applications"),
    (2023, "EU AI Act proposed", "policy", "Explainable AI requirements"),
]

EVENT_COLORS = {"ai": "#3498db", "policy": "#e67e22", "pandemic": "#e74c3c"}


# ---------------------------------------------------------------------------
# Args and data loading
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive analysis: Q1-Q7."
    )
    parser.add_argument("--articles-csv", required=True)
    parser.add_argument("--authors-csv", default=None,
                        help="Author data CSV (optional; Q2-Q4 skipped if missing)")
    parser.add_argument("--topics-dir", default="data/hierarchy")
    parser.add_argument("--output-dir", default="data/visualizations")
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-year", type=int, default=2025)
    return parser.parse_args()


def abbr(journal):
    return JOURNAL_ABBR.get(journal, journal[:25])


def load_articles(path):
    logger.info(f"Loading articles from {path}")
    df = pd.read_csv(path, quoting=csv.QUOTE_ALL, dtype={"pmid": str})
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    logger.info(f"  {len(df)} articles loaded")
    return df


def load_topics(topics_dir, domain):
    path = os.path.join(topics_dir, f"{domain}_final_topics.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_kw_map(topics):
    m = defaultdict(set)
    for data in topics.values():
        for kw in data.get("keywords", []):
            m[kw.strip().lower()].add(data["name"])
    return m


def map_articles(df, kw_m, kw_h):
    records = []
    for _, row in df.iterrows():
        kw_str = row.get("keywords", "")
        mt, ht = set(), set()
        if not pd.isna(kw_str) and kw_str:
            for kw in str(kw_str).split(";"):
                kw = kw.strip().lower()
                if kw in kw_m:
                    mt.update(kw_m[kw])
                if kw in kw_h:
                    ht.update(kw_h[kw])
        records.append({
            "pmid": row["pmid"], "year": int(row["year"]),
            "journal": row.get("journal", "unknown"),
            "method_topics": mt, "health_topics": ht,
        })
    return records


def build_counts(records, years):
    """Build all core count structures."""
    topic_year = {"methodology": Counter(), "health": Counter()}
    topic_journal = {"methodology": Counter(), "health": Counter()}
    journal_total = Counter()
    year_total = Counter()

    for r in records:
        journal_total[r["journal"]] += 1
        year_total[r["year"]] += 1
        for t in r["method_topics"]:
            topic_year["methodology"][(t, r["year"])] += 1
            topic_journal["methodology"][(t, r["journal"])] += 1
        for t in r["health_topics"]:
            topic_year["health"][(t, r["year"])] += 1
            topic_journal["health"][(t, r["journal"])] += 1

    return topic_year, topic_journal, journal_total, year_total


# ---------------------------------------------------------------------------
# Q1: First emergence — when and where
# ---------------------------------------------------------------------------

def q1_first_emergence(records, output_dir):
    logger.info("\n=== Q1: When and where did each topic first emerge? ===")

    sorted_recs = sorted(records, key=lambda x: (x["year"], x["pmid"]))

    for domain, key in [("methodology", "method_topics"),
                        ("health", "health_topics")]:
        first = {}
        for r in sorted_recs:
            for t in r[key]:
                if t not in first:
                    first[t] = {"year": r["year"], "journal": r["journal"]}

        # Save CSV
        rows = [{"topic": t, "first_year": v["year"],
                 "first_journal": v["journal"]} for t, v in first.items()]
        df = pd.DataFrame(rows).sort_values(["first_year", "topic"])
        df.to_csv(os.path.join(output_dir,
                  f"pi_q1_{domain}_first_emergence.csv"), index=False)

        # Pioneer journal bar chart
        pioneer = Counter(v["journal"] for v in first.values())
        top = dict(Counter({abbr(j): c for j, c in pioneer.items()}).most_common(15))

        fig, ax = plt.subplots(figsize=(11, 7))
        names = list(top.keys())[::-1]
        counts = list(top.values())[::-1]
        ax.barh(names, counts, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(names))),
                edgecolor="white")
        for i, c in enumerate(counts):
            ax.text(c + 0.3, i, str(c), va="center", fontsize=10)
        ax.set_xlabel(f"# of {domain} topics first published", fontsize=12)
        ax.set_title(f"Q1: Pioneer Journals — {domain.title()} Topics", fontsize=14)
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                    f"pi_q1_{domain}_pioneer_journals.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved pi_q1_{domain}_pioneer_journals.png")

        # First emergence year histogram
        year_counts = Counter(v["year"] for v in first.values())
        fig, ax = plt.subplots(figsize=(11, 5))
        yrs = sorted(year_counts.keys())
        ax.bar(yrs, [year_counts[y] for y in yrs], color="#3498db", edgecolor="white")
        for y in yrs:
            ax.text(y, year_counts[y] + 0.5, str(year_counts[y]),
                    ha="center", fontsize=9)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("# Topics First Appearing", fontsize=12)
        ax.set_title(f"Q1: {domain.title()} Topic First Appearance Distribution",
                     fontsize=13)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                    f"pi_q1_{domain}_emergence_year_dist.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved pi_q1_{domain}_emergence_year_dist.png")


# ---------------------------------------------------------------------------
# Q2: Authorship characteristics over time
# ---------------------------------------------------------------------------

def q2_authorship_over_time(merged_df, years, output_dir):
    logger.info("\n=== Q2: Authorship characteristics over time ===")

    v = merged_df.dropna(subset=["n_authors"]).copy()
    v["n_authors"] = v["n_authors"].astype(int)
    v["n_unique_affiliations"] = v["n_unique_affiliations"].fillna(0).astype(int)
    v["is_interdisciplinary"] = v["is_interdisciplinary"].fillna(0).astype(int)
    v["has_clinical"] = v["has_clinical"].fillna(0).astype(int)
    v["has_computational"] = v["has_computational"].fillna(0).astype(int)

    # Exclude journals with <10% PubMed-affiliation coverage from affiliation-
    # based analyses (e.g. IEEE J-BHI submits author lists without affiliations).
    # Author counts remain valid; only clinical/computational classification is
    # unreliable for these journals.
    aff_coverage = v.groupby("journal")["n_unique_affiliations"].apply(
        lambda x: (x > 0).mean() * 100
    )
    excluded = aff_coverage[aff_coverage < 10].index.tolist()
    if excluded:
        logger.warning(
            f"  Excluding {len(excluded)} journals from Q2 affiliation composition "
            f"(metadata gap): {excluded}"
        )
    v_aff = v[~v["journal"].isin(excluded)]

    stats = []
    for y in years:
        yd = v[v["year"] == y]
        yd_aff = v_aff[v_aff["year"] == y]
        if len(yd) == 0:
            continue
        stats.append({
            "year": y, "n": len(yd),
            "median_authors": yd["n_authors"].median(),
            "mean_authors": yd["n_authors"].mean(),
            "p25_authors": yd["n_authors"].quantile(0.25),
            "p75_authors": yd["n_authors"].quantile(0.75),
            "max_authors": yd["n_authors"].max(),
            "median_affs": yd_aff["n_unique_affiliations"].median() if len(yd_aff) else np.nan,
            "mean_affs": yd_aff["n_unique_affiliations"].mean() if len(yd_aff) else np.nan,
            "pct_clinical_only": (((yd_aff["has_clinical"] == 1) & (yd_aff["has_computational"] == 0)).mean() * 100) if len(yd_aff) else np.nan,
            "pct_computational_only": (((yd_aff["has_clinical"] == 0) & (yd_aff["has_computational"] == 1)).mean() * 100) if len(yd_aff) else np.nan,
            "pct_interdisciplinary": ((yd_aff["is_interdisciplinary"] == 1).mean() * 100) if len(yd_aff) else np.nan,
            "pct_neither": (((yd_aff["has_clinical"] == 0) & (yd_aff["has_computational"] == 0)).mean() * 100) if len(yd_aff) else np.nan,
        })
    sdf = pd.DataFrame(stats)
    sdf.to_csv(os.path.join(output_dir, "pi_q2_authorship_stats.csv"), index=False)

    # Plot 1: Author count over time
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(sdf["year"], sdf["p25_authors"], sdf["p75_authors"],
                    alpha=0.3, color="#3498db", label="25-75 percentile")
    ax.plot(sdf["year"], sdf["median_authors"], "o-", color="#2c3e50",
            linewidth=2.5, markersize=7, label="Median")
    ax.plot(sdf["year"], sdf["mean_authors"], "s--", color="#e74c3c",
            linewidth=2, markersize=6, label="Mean")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Authors per Article", fontsize=12)
    ax.set_title("Q2: Co-Author Count Over Time", fontsize=14)
    ax.legend(fontsize=11)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pi_q2_authors_over_time.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Affiliations over time
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sdf["year"], sdf["median_affs"], "o-", color="#16a085",
            linewidth=2.5, markersize=7, label="Median")
    ax.plot(sdf["year"], sdf["mean_affs"], "s--", color="#e67e22",
            linewidth=2, markersize=6, label="Mean")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Distinct Affiliations per Article", fontsize=12)
    ax.set_title("Q2: Inter-Institutional Collaboration Over Time", fontsize=14)
    ax.legend(fontsize=11)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pi_q2_affiliations_over_time.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 3: Composition stacked area
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(sdf["year"],
                 sdf["pct_interdisciplinary"], sdf["pct_clinical_only"],
                 sdf["pct_computational_only"], sdf["pct_neither"],
                 labels=["Interdisciplinary (Clinical+Computational)",
                         "Clinical only", "Computational only", "Other/unclassified"],
                 colors=["#9b59b6", "#e74c3c", "#3498db", "#bdc3c7"], alpha=0.85)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("% of Articles", fontsize=12)
    ax.set_title("Q2: Author Affiliation Composition Over Time", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pi_q2_composition_over_time.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("  Saved Q2 plots (3)")


# ---------------------------------------------------------------------------
# Q3: Team science by journal
# ---------------------------------------------------------------------------

def q3_team_science_journals(merged_df, output_dir):
    logger.info("\n=== Q3: Team science by journal ===")

    v = merged_df.dropna(subset=["n_authors"]).copy()
    v["n_authors"] = v["n_authors"].astype(int)
    v["n_unique_affiliations"] = v["n_unique_affiliations"].fillna(0).astype(int)
    v["is_interdisciplinary"] = v["is_interdisciplinary"].fillna(0).astype(int)

    js = v.groupby("journal").agg(
        n=("pmid", "count"),
        mean_auth=("n_authors", "mean"),
        median_auth=("n_authors", "median"),
        max_auth=("n_authors", "max"),
        mean_aff=("n_unique_affiliations", "mean"),
        pct_aff_coverage=("n_unique_affiliations",
                          lambda x: (x > 0).mean() * 100),
        pct_inter=("is_interdisciplinary", lambda x: x.mean() * 100),
    ).reset_index()
    js = js[js["n"] >= 100]
    # Flag journals where PubMed lacks affiliation strings (e.g. IEEE J-BHI).
    # Author counts are still meaningful; affiliation-derived metrics are not.
    AFF_COVERAGE_THRESHOLD = 10.0
    poor_aff = js["pct_aff_coverage"] < AFF_COVERAGE_THRESHOLD
    if poor_aff.any():
        flagged = js.loc[poor_aff, "journal"].tolist()
        logger.warning(
            f"  Flagging {len(flagged)} journals with <{AFF_COVERAGE_THRESHOLD}% "
            f"affiliation coverage (metadata gap, not zero collaboration): "
            f"{flagged}"
        )
        js.loc[poor_aff, "mean_aff"] = np.nan
        js.loc[poor_aff, "pct_inter"] = np.nan
    js["abbr"] = js["journal"].map(abbr)
    js.to_csv(os.path.join(output_dir, "pi_q3_team_science.csv"), index=False)

    flagged_abbrs = js.loc[poor_aff, "abbr"].tolist() if poor_aff.any() else []

    fig, axes = plt.subplots(1, 3, figsize=(20, 9))
    for ax, col, label, color in [
        (axes[0], "mean_auth", "Mean Authors/Article", "#3498db"),
        (axes[1], "mean_aff", "Mean Distinct Affiliations/Article", "#16a085"),
        (axes[2], "pct_inter", "% Interdisciplinary Articles", "#9b59b6"),
    ]:
        s = js.dropna(subset=[col]).sort_values(col, ascending=True).tail(20)
        ax.barh(s["abbr"], s[col], color=color, edgecolor="white")
        for i, val in enumerate(s[col]):
            fmt = f"{val:.1f}%" if "pct" in col else f"{val:.1f}"
            ax.text(val + 0.1 * s[col].max() / 20, i, fmt, va="center", fontsize=9)
        ax.set_xlabel(label, fontsize=12)
        ax.set_title(label, fontsize=13)
        ax.grid(True, alpha=0.3, axis="x")
        # Affiliation-derived panels (mean_aff, pct_inter) drop journals with
        # <10% PubMed affiliation coverage — flag them so readers don't read
        # absence as zero collaboration.
        if col in ("mean_aff", "pct_inter") and flagged_abbrs:
            ax.text(0.98, 0.02,
                    "n/a (low PubMed affiliation coverage):\n"
                    + ", ".join(flagged_abbrs),
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=9, style="italic", color="#7f8c8d",
                    bbox=dict(facecolor="white", edgecolor="#bdc3c7",
                              alpha=0.9, boxstyle="round,pad=0.3"))

    fig.suptitle("Q3: Team Science Indicators by Journal (≥100 articles)",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pi_q3_team_science_journals.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("  Saved pi_q3_team_science_journals.png")


# ---------------------------------------------------------------------------
# Q4: Emerging topics — team science vs solo
# ---------------------------------------------------------------------------

def q4_emerging_team(merged_df, records, years, output_dir, cutoff=2018):
    logger.info(f"\n=== Q4: Team science for emerging (>={cutoff}) vs established topics ===")

    # Determine topic takeoff years (both methodology AND health)
    topic_years = defaultdict(Counter)
    for r in records:
        for t in r["method_topics"]:
            topic_years[t][r["year"]] += 1
        for t in r["health_topics"]:
            topic_years[t][r["year"]] += 1

    emerging, established = set(), set()
    for t, yc in topic_years.items():
        peak = max(yc.values())
        for y in sorted(yc.keys()):
            if yc[y] >= peak * 0.25:
                (emerging if y >= cutoff else established).add(t)
                break

    # Classify each article based on both methodology and health topics
    pmid_cat = {}
    for r in records:
        all_topics = r["method_topics"] | r["health_topics"]
        has_e = bool(all_topics & emerging)
        has_s = bool(all_topics & established)
        if has_e and not has_s:
            pmid_cat[r["pmid"]] = "emerging"
        elif has_s and not has_e:
            pmid_cat[r["pmid"]] = "established"

    v = merged_df.copy()
    v["category"] = v["pmid"].map(pmid_cat)
    v = v.dropna(subset=["n_authors", "category"]).copy()
    v["n_authors"] = v["n_authors"].astype(int)

    def team_type(n):
        if n <= 1: return "Solo author"
        if n <= 3: return "Small team (2-3)"
        if n <= 10: return "Medium team (4-10)"
        return "Large team (11+)"

    v["team"] = v["n_authors"].apply(team_type)
    order = ["Solo author", "Small team (2-3)", "Medium team (4-10)", "Large team (11+)"]
    colors = ["#e74c3c", "#f39c12", "#3498db", "#27ae60"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, cat, title in [
        (axes[0], "emerging", f"Emerging Topics (took off ≥{cutoff})"),
        (axes[1], "established", f"Established Topics (took off <{cutoff})"),
    ]:
        cd = v[v["category"] == cat]
        if len(cd) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        tc = cd["team"].value_counts().reindex(order, fill_value=0)
        pcts = tc / tc.sum() * 100
        bars = ax.bar(order, pcts, color=colors, edgecolor="white")
        for b, p in zip(bars, pcts):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                    f"{p:.1f}%", ha="center", fontsize=10)
        inter = cd["is_interdisciplinary"].mean() * 100
        ax.text(0.98, 0.95, f"Interdisciplinary: {inter:.1f}%\nn={len(cd):,}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9))
        ax.set_ylabel("% of Articles", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, max(pcts.max() + 10, 50))
        ax.grid(True, alpha=0.3, axis="y")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    fig.suptitle("Q4: Team Science — Emerging vs Established Topics", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pi_q4_emerging_team.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("  Saved pi_q4_emerging_team.png")


# ---------------------------------------------------------------------------
# Q5: Specialized vs shared topics
# ---------------------------------------------------------------------------

def q5_specialization(records, output_dir):
    logger.info("\n=== Q5: Specialized vs shared topics ===")

    n_journals = len({r["journal"] for r in records})
    log_n = math.log(max(n_journals, 2))  # guard against log(1)=0 or log(0)

    for domain, key in [("methodology", "method_topics"),
                        ("health", "health_topics")]:
        tj = defaultdict(Counter)
        for r in records:
            for t in r[key]:
                tj[t][r["journal"]] += 1

        rows = []
        for t, jc in tj.items():
            total = sum(jc.values())
            probs = [c / total for c in jc.values()]
            H = -sum(p * math.log(p) for p in probs)
            rows.append({
                "topic": t, "total": total, "n_journals": len(jc),
                "entropy": H, "norm_entropy": H / log_n,
                "max_journal": max(jc, key=jc.get),
                "max_share": max(jc.values()) / total * 100,
            })
        df = pd.DataFrame(rows).sort_values("norm_entropy", ascending=False)
        df.to_csv(os.path.join(output_dir,
                  f"pi_q5_{domain}_specialization.csv"), index=False)

        # Scatter: entropy vs total articles
        fig, ax = plt.subplots(figsize=(13, 8))
        ax.scatter(df["total"], df["norm_entropy"],
                   s=np.sqrt(df["total"]) * 5, c=df["norm_entropy"],
                   cmap="RdYlGn", edgecolors="black", linewidths=0.5, alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("Total Articles (log)", fontsize=12)
        ax.set_ylabel("Normalized Entropy (0=specialized, 1=shared)", fontsize=12)
        ax.set_title(f"Q5: {domain.title()} Topic Specialization vs Sharing",
                     fontsize=14)
        ax.axhline(df["norm_entropy"].median(), color="gray", linestyle="--",
                   alpha=0.5, label=f"Median={df['norm_entropy'].median():.2f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Annotate extremes
        for _, r in df.nsmallest(5, "norm_entropy").iterrows():
            ax.annotate(textwrap.shorten(r["topic"], 22, placeholder="..."),
                        (r["total"], r["norm_entropy"]), fontsize=7, alpha=0.8,
                        xytext=(5, -8), textcoords="offset points")
        for _, r in df.nlargest(5, "norm_entropy").iterrows():
            ax.annotate(textwrap.shorten(r["topic"], 22, placeholder="..."),
                        (r["total"], r["norm_entropy"]), fontsize=7, alpha=0.8,
                        xytext=(5, 5), textcoords="offset points")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                    f"pi_q5_{domain}_specialization_scatter.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

        # Ranking: top 15 specialized + top 15 shared
        filt = df[df["total"] >= 50]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        for ax, subset, color, title_word in [
            (ax1, filt.nsmallest(15, "norm_entropy").iloc[::-1], "#e74c3c", "SPECIALIZED"),
            (ax2, filt.nlargest(15, "norm_entropy").iloc[::-1], "#27ae60", "SHARED"),
        ]:
            ax.barh(subset["topic"], subset["norm_entropy"], color=color, edgecolor="white")
            for i, r in enumerate(subset.itertuples()):
                ax.text(r.norm_entropy + 0.005, i,
                        f"{r.norm_entropy:.2f} ({r.n_journals}j, {r.total:,}a)",
                        va="center", fontsize=8)
            ax.set_xlabel("Normalized Entropy", fontsize=11)
            ax.set_title(f"Most {title_word} {domain} topics", fontsize=13)
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3, axis="x")
        fig.suptitle(f"Q5: {domain.title()} — Specialized vs Shared (≥50 articles)",
                     fontsize=15, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                    f"pi_q5_{domain}_specialization_ranking.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved Q5 {domain} plots (2)")


# ---------------------------------------------------------------------------
# Q6: Journal selection guidance
# ---------------------------------------------------------------------------

def q6_journal_guidance(records, topic_year, topic_journal, journal_total,
                        years, output_dir):
    logger.info("\n=== Q6: Journal selection guidance ===")

    for domain, key in [("methodology", "method_topics"),
                        ("health", "health_topics")]:
        tyc = topic_year[domain]
        tjc = topic_journal[domain]

        # Get top 20 topics by total articles
        topic_totals = Counter()
        for (t, y), c in tyc.items():
            topic_totals[t] += c
        top_topics = [t for t, _ in topic_totals.most_common(20)]

        # Get top 15 journals
        top_journals_full = [j for j, _ in journal_total.most_common(15)]
        top_journals = [abbr(j) for j in top_journals_full]

        # Build fit matrix: % of journal articles in each topic
        fit = pd.DataFrame(0.0, index=top_journals, columns=top_topics)
        for j_full, j_abbr in zip(top_journals_full, top_journals):
            jt = journal_total[j_full]
            for t in top_topics:
                fit.loc[j_abbr, t] = tjc.get((t, j_full), 0) / max(jt, 1) * 100

        # Wrap topic names — narrower wrap fits within heatmap column width
        fit.columns = ["\n".join(textwrap.wrap(t, 14)) for t in fit.columns]

        fig, ax = plt.subplots(figsize=(26, 13))
        sns.heatmap(fit, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax,
                    linewidths=0.5, linecolor="white", annot_kws={"fontsize": 8})
        ax.set_title(f"Q6: Journal-Topic Fit — {domain.title()}\n"
                     f"(% of each journal's articles in each topic — "
                     f"higher = better fit for your paper)", fontsize=14)
        ax.set_ylabel("Journal", fontsize=12)
        ax.set_xlabel(f"{domain.title()} Topic", fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=10)
        plt.setp(ax.get_yticklabels(), fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                    f"pi_q6_{domain}_journal_fit.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved pi_q6_{domain}_journal_fit.png")

    # Journal innovation index: which journals first published emerging topics
    logger.info("  Computing journal innovation index...")
    sorted_recs = sorted(records, key=lambda x: (x["year"], x["pmid"]))

    for domain, key in [("methodology", "method_topics"),
                        ("health", "health_topics")]:
        tyc = topic_year[domain]
        # Find topics that took off after 2015 (genuine emerging topics)
        topic_totals = Counter()
        topic_takeoff = {}
        for (t, y), c in tyc.items():
            topic_totals[t] += c

        for t in topic_totals:
            yearly = {y: tyc.get((t, y), 0) for y in years}
            peak = max(yearly.values()) if yearly else 0
            if peak == 0:
                continue
            for y in sorted(yearly.keys()):
                if yearly[y] >= peak * 0.25:
                    topic_takeoff[t] = y
                    break

        emerging_topics = {t for t, y in topic_takeoff.items() if y >= 2015}

        # For each emerging topic, find the pioneer journal
        first_journal = {}
        for r in sorted_recs:
            for t in r[key]:
                if t in emerging_topics and t not in first_journal:
                    first_journal[t] = r["journal"]

        innovation = Counter(abbr(j) for j in first_journal.values())

        if innovation:
            fig, ax = plt.subplots(figsize=(11, 7))
            top = dict(innovation.most_common(15))
            names = list(top.keys())[::-1]
            counts = list(top.values())[::-1]
            ax.barh(names, counts,
                    color=plt.cm.plasma(np.linspace(0.2, 0.8, len(names))),
                    edgecolor="white")
            for i, c in enumerate(counts):
                ax.text(c + 0.2, i, str(c), va="center", fontsize=10)
            ax.set_xlabel("# Emerging Topics First Published", fontsize=12)
            ax.set_title(f"Q6: Journal Innovation Index — {domain.title()}\n"
                         f"(which journals pioneered topics that took off ≥2015)",
                         fontsize=13)
            ax.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir,
                        f"pi_q6_{domain}_innovation_index.png"),
                        dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"  Saved pi_q6_{domain}_innovation_index.png")

    # Submission landscape: competition vs options per topic
    for domain, key in [("methodology", "method_topics"),
                        ("health", "health_topics")]:
        tjc = topic_journal[domain]
        tyc = topic_year[domain]

        topic_totals = Counter()
        for (t, y), c in tyc.items():
            topic_totals[t] += c

        rows = []
        for t, total in topic_totals.items():
            n_journals = len({j for (tp, j) in tjc if tp == t and tjc[(tp, j)] > 0})
            rows.append({"topic": t, "total": total, "n_journals": n_journals})
        df = pd.DataFrame(rows)
        if df.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(df["n_journals"], df["total"],
                   s=80, c="#3498db", edgecolors="black", linewidths=0.5, alpha=0.7)
        ax.set_xlabel("# Journals Publishing This Topic (more = more options)", fontsize=12)
        ax.set_ylabel("Total Articles (more = more competition)", fontsize=12)
        ax.set_title(f"Q6: Submission Landscape — {domain.title()} Topics\n"
                     f"(bottom-right = easy; top-left = competitive)", fontsize=13)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        # Annotate corners
        for _, r in df.nlargest(3, "total").iterrows():
            ax.annotate(textwrap.shorten(r["topic"], 25, placeholder="..."),
                        (r["n_journals"], r["total"]), fontsize=8,
                        xytext=(5, 5), textcoords="offset points")
        for _, r in df.nsmallest(3, "n_journals").iterrows():
            ax.annotate(textwrap.shorten(r["topic"], 25, placeholder="..."),
                        (r["n_journals"], r["total"]), fontsize=8,
                        xytext=(5, -10), textcoords="offset points")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                    f"pi_q6_{domain}_submission_landscape.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved pi_q6_{domain}_submission_landscape.png")


# ---------------------------------------------------------------------------
# Q7: Historical event correlation
# ---------------------------------------------------------------------------

def q7_historical_events(topic_year, year_total, years, output_dir):
    logger.info("\n=== Q7: Historical event correlation ===")

    for domain in ["methodology", "health"]:
        tyc = topic_year[domain]

        # Get top 8 topics by total
        topic_totals = Counter()
        for (t, y), c in tyc.items():
            topic_totals[t] += c
        top8 = [t for t, _ in topic_totals.most_common(8)]

        # Plot: event-annotated trend lines
        fig, axes = plt.subplots(4, 2, figsize=(20, 24))
        axes = axes.flatten()

        for idx in range(len(top8), len(axes)):
            axes[idx].set_visible(False)

        for idx, topic in enumerate(top8):
            ax = axes[idx]
            counts = [tyc.get((topic, y), 0) for y in years]
            ax.plot(years, counts, "o-", color="#2c3e50", linewidth=2.5,
                    markersize=6, zorder=3)
            ax.fill_between(years, 0, counts, alpha=0.15, color="#3498db")

            # Add event lines
            for ev_year, ev_name, ev_type, ev_desc in HISTORICAL_EVENTS:
                if ev_year < years[0] or ev_year > years[-1]:
                    continue
                color = EVENT_COLORS.get(ev_type, "gray")
                ax.axvline(ev_year, color=color, linestyle="--", alpha=0.6,
                           linewidth=1.5)
                ax.text(ev_year + 0.15, ax.get_ylim()[1] * 0.95,
                        ev_name, rotation=90, va="top", fontsize=7,
                        color=color, alpha=0.8)

            ax.set_title("\n".join(textwrap.wrap(topic, 35)), fontsize=12,
                         fontweight="bold")
            ax.set_ylabel("Articles/Year", fontsize=10)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, alpha=0.3)
            ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)

        # Legend for event types
        handles = [mpatches.Patch(color=c, label=t.title())
                   for t, c in EVENT_COLORS.items()]
        fig.legend(handles=handles, loc="upper right", fontsize=11,
                   title="Event Type", title_fontsize=12)
        fig.suptitle(
            f"Q7: {domain.title()} Topic Trends with Historical Events",
            fontsize=18, y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                    f"pi_q7_{domain}_event_annotated.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved pi_q7_{domain}_event_annotated.png")

    # COVID impact analysis (both domains combined)
    logger.info("  Computing COVID impact...")
    covid_rows = []
    for domain in ["methodology", "health"]:
        tyc = topic_year[domain]
        topic_totals = Counter()
        for (t, y), c in tyc.items():
            topic_totals[t] += c

        # Match the plot's >=100 article filter so the CSV doesn't carry
        # rows the figure already excludes. Also require pre-COVID avg >=3
        # so 3->6 doesn't show as "100% growth" — same noise that produced
        # the original 4900% / 34900% artifacts at lower thresholds.
        from scipy import stats as sstats
        MIN_TOTAL = 100
        MIN_PRE = 3.0
        for t in topic_totals:
            if topic_totals[t] < MIN_TOTAL:
                continue
            pre_yrs = [tyc.get((t, y), 0) for y in [2018, 2019]]
            during_yrs = [tyc.get((t, y), 0) for y in [2020, 2021]]
            post_yrs = [tyc.get((t, y), 0) for y in [2022, 2023, 2024, 2025]]
            pre = np.mean(pre_yrs)
            during = np.mean(during_yrs)
            post = np.mean(post_yrs)
            if pre < MIN_PRE:
                pct_during = np.nan
                pct_post = np.nan
                p_during = np.nan
                p_post = np.nan
            else:
                pct_during = (during - pre) / pre * 100
                pct_post = (post - pre) / pre * 100
                # Poisson rate test: H0 is "during/post yearly rate matches
                # pre-COVID rate". Treat each year's count as a Poisson draw;
                # chi-square goodness-of-fit on summed counts has much more
                # power than Mann-Whitney with n=2 vs n=2 yearly observations.
                pre_rate = sum(pre_yrs) / len(pre_yrs)
                for label, group, p_var in (
                    ("during", during_yrs, "p_during"),
                    ("post", post_yrs, "p_post"),
                ):
                    obs = sum(group)
                    exp = pre_rate * len(group)
                    if exp <= 0:
                        pval = np.nan
                    else:
                        chi2 = (obs - exp) ** 2 / exp
                        pval = float(sstats.chi2.sf(chi2, df=1))
                    if p_var == "p_during":
                        p_during = pval
                    else:
                        p_post = pval
            covid_rows.append({
                "domain": domain, "topic": t, "total": topic_totals[t],
                "pre_covid_avg": pre, "during_covid_avg": during,
                "post_covid_avg": post,
                "pct_change_during": pct_during,
                "pct_change_post": pct_post,
                "p_value_during": p_during,
                "p_value_post": p_post,
            })

    covid_df = pd.DataFrame(covid_rows)
    covid_df.to_csv(os.path.join(output_dir, "pi_q7_covid_impact.csv"), index=False)

    # COVID impact waterfall: top gainers and losers
    for domain in ["methodology", "health"]:
        cd = covid_df[(covid_df["domain"] == domain) &
                      (covid_df["total"] >= 100) &
                      covid_df["pct_change_during"].notna()]
        cd = cd.sort_values("pct_change_during")

        top_gain = cd.nlargest(10, "pct_change_during")
        top_loss = cd.nsmallest(10, "pct_change_during")
        combined = pd.concat([top_loss, top_gain]).drop_duplicates(subset="topic")

        fig, ax = plt.subplots(figsize=(14, 8))
        colors = ["#e74c3c" if v < 0 else "#27ae60"
                  for v in combined["pct_change_during"]]
        ax.barh(combined["topic"], combined["pct_change_during"],
                color=colors, edgecolor="white")
        for i, (_, r) in enumerate(combined.iterrows()):
            side = 5 if r["pct_change_during"] >= 0 else -5
            ax.text(r["pct_change_during"] + side, i,
                    f'{r["pct_change_during"]:.0f}%', va="center", fontsize=9,
                    ha="left" if side > 0 else "right")
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("% Change: Pre-COVID (2018-19) → During COVID (2020-21)",
                      fontsize=12)
        ax.set_title(f"Q7: COVID-19 Impact on {domain.title()} Topics\n"
                     f"(top 10 gainers and losers, ≥100 articles)", fontsize=14)
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                    f"pi_q7_{domain}_covid_impact.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved pi_q7_{domain}_covid_impact.png")

    # Change-point detection: year of max growth acceleration per topic
    logger.info("  Computing change points...")
    cp_rows = []
    for domain in ["methodology", "health"]:
        tyc = topic_year[domain]
        topic_totals = Counter()
        for (t, y), c in tyc.items():
            topic_totals[t] += c

        for t in [tp for tp, _ in topic_totals.most_common(30)]:
            counts = np.array([tyc.get((t, y), 0) for y in years], dtype=float)
            if counts.sum() < 50:
                continue
            # Year-over-year growth rate
            growth = np.diff(counts)
            # Year of max acceleration (largest jump)
            max_idx = np.argmax(growth)
            cp_year = years[max_idx + 1]
            cp_rows.append({
                "domain": domain, "topic": t, "change_point_year": cp_year,
                "max_growth": growth[max_idx],
                "total": topic_totals[t],
            })

    cp_df = pd.DataFrame(cp_rows)
    cp_df.to_csv(os.path.join(output_dir, "pi_q7_change_points.csv"), index=False)

    # Change-point heatmap
    for domain in ["methodology", "health"]:
        if cp_df.empty or len(cp_df[cp_df["domain"] == domain]) == 0:
            logger.info(f"  Skipping Q7 change-point heatmap for {domain} (no data)")
            continue
        cd = cp_df[cp_df["domain"] == domain].nlargest(20, "total")
        if len(cd) == 0:
            continue

        fig, ax = plt.subplots(figsize=(14, 8))
        # Build growth-rate matrix for heatmap
        tyc = topic_year[domain]
        topics = cd["topic"].tolist()
        mat = []
        for t in topics:
            counts = [tyc.get((t, y), 0) for y in years]
            growth = [0] + list(np.diff(counts))
            mat.append(growth)
        mat = np.array(mat)

        sns.heatmap(mat, xticklabels=years, yticklabels=topics,
                    cmap="RdBu_r", center=0, ax=ax,
                    linewidths=0.5, linecolor="white", annot=False)
        ax.set_title(f"Q7: Year-over-Year Growth Rate — Top 20 {domain.title()} Topics\n"
                     f"(red = surge, blue = decline; find change points visually)",
                     fontsize=13)
        ax.set_xlabel("Year", fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=45, fontsize=9)
        plt.setp(ax.get_yticklabels(), fontsize=9)

        # Mark event years on x-axis
        for ev_year, ev_name, _, _ in HISTORICAL_EVENTS:
            if ev_year in years:
                idx = years.index(ev_year)
                ax.axvline(idx + 0.5, color="black", linewidth=1.5,
                           alpha=0.4, linestyle="--")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                    f"pi_q7_{domain}_changepoint_heatmap.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved pi_q7_{domain}_changepoint_heatmap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Comprehensive Analysis: All PI Questions (Q1-Q7)")
    logger.info("=" * 70)

    # Load core data
    articles_df = load_articles(args.articles_csv)
    method_topics = load_topics(args.topics_dir, "methodology")
    health_topics = load_topics(args.topics_dir, "health")

    kw_m = build_kw_map(method_topics)
    kw_h = build_kw_map(health_topics)

    logger.info("Mapping articles to topics...")
    records = map_articles(articles_df, kw_m, kw_h)
    years = list(range(args.start_year, args.end_year + 1))
    topic_year, topic_journal, journal_total, year_total = build_counts(records, years)

    # Load author data (optional)
    has_authors = False
    merged_df = None
    if args.authors_csv and os.path.exists(args.authors_csv):
        authors_df = pd.read_csv(args.authors_csv, dtype={"pmid": str})
        merged_df = articles_df.merge(authors_df, on="pmid", how="left")
        n_with = merged_df["n_authors"].notna().sum()
        logger.info(f"Author data loaded: {n_with}/{len(merged_df)} articles with data")
        has_authors = True
    else:
        logger.warning("No author data — Q2, Q3, Q4 will be skipped. "
                       "Run collect_author_data.py first to enable them.")

    # --- Run analyses ---

    q1_first_emergence(records, args.output_dir)

    if has_authors:
        q2_authorship_over_time(merged_df, years, args.output_dir)
        q3_team_science_journals(merged_df, args.output_dir)
        q4_emerging_team(merged_df, records, years, args.output_dir)
    else:
        logger.info("\n=== Q2/Q3/Q4: SKIPPED (no author data) ===")

    q5_specialization(records, args.output_dir)
    q6_journal_guidance(records, topic_year, topic_journal, journal_total,
                        years, args.output_dir)
    q7_historical_events(topic_year, year_total, years, args.output_dir)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("ALL ANALYSES COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nOutputs in {args.output_dir}/ with prefix 'pi_q':")
    logger.info("  Q1: pi_q1_*_pioneer_journals.png, pi_q1_*_emergence_year_dist.png")
    if has_authors:
        logger.info("  Q2: pi_q2_authors_over_time.png, pi_q2_affiliations_over_time.png, "
                     "pi_q2_composition_over_time.png")
        logger.info("  Q3: pi_q3_team_science_journals.png")
        logger.info("  Q4: pi_q4_emerging_team.png")
    logger.info("  Q5: pi_q5_*_specialization_scatter.png, pi_q5_*_specialization_ranking.png")
    logger.info("  Q6: pi_q6_*_journal_fit.png, pi_q6_*_innovation_index.png, "
                 "pi_q6_*_submission_landscape.png")
    logger.info("  Q7: pi_q7_*_event_annotated.png, pi_q7_*_covid_impact.png, "
                 "pi_q7_*_changepoint_heatmap.png")


if __name__ == "__main__":
    main()
