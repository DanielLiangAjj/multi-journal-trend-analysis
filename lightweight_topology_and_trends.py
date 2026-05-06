"""
Lightweight topology + trend pipeline for the K=1,050 / K=1,100 sensitivity
analysis (B7).  No GPT-based topic naming, no hierarchy construction --
cluster IDs serve directly as topic labels.

Steps:
  2. Topology stats (keywords/topic, articles/topic distributions).
  3. Article -> cluster mapping via keyword-overlap (chunked CSV read).
  4. Linear-trend testing per cluster with Holm-Bonferroni adjustment.

Produces:
  data/k_comparison/topology_table_k1050_1100.csv
  data/k_comparison/topology_table_k1050_1100.md
  data/visualizations_k1050_1100/{domain}_topic_year_counts.csv
  data/visualizations_k1050_1100/{domain}_trend_analysis.csv
  data/visualizations_k1050_1100/topology_summary.json
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import stats as sstats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DOMAINS = ("methodology", "health")
START_YEAR = 2011
END_YEAR = 2025
MIN_ARTICLES_FOR_TREND = 10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clusters-dir", required=True,
                   help="Directory with {domain}_keyword_clusters.csv")
    p.add_argument("--corpus", required=True,
                   help="Patched articles CSV")
    p.add_argument("--vis-dir", required=True,
                   help="Output directory for trend / topic_year_counts CSVs")
    p.add_argument("--k-comparison-dir", required=True,
                   help="Directory for topology_table_*.csv/md outputs")
    p.add_argument("--k-methodology", type=int, required=True)
    p.add_argument("--k-health", type=int, required=True)
    p.add_argument("--chunksize", type=int, default=20000)
    p.add_argument("--skip-mapping", action="store_true",
                   help="Skip Step 3 if topic_year_counts already exist")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 2 / Step 3 helpers
# ---------------------------------------------------------------------------

def load_keyword_clusters(clusters_dir, domain):
    """Return dict: keyword (lowercase) -> cluster_id (int)."""
    path = os.path.join(clusters_dir, f"{domain}_keyword_clusters.csv")
    df = pd.read_csv(path, quoting=csv.QUOTE_ALL)
    df["keyword"] = df["keyword"].astype(str).str.strip().str.lower()
    df["cluster_id"] = df["cluster_id"].astype(int)
    kw_to_cid = dict(zip(df["keyword"], df["cluster_id"]))
    logger.info(f"  {domain}: {len(kw_to_cid)} keyword->cluster mappings "
                f"(unique clusters: {df['cluster_id'].nunique()})")
    return kw_to_cid, df


def map_articles_chunked(corpus_path, kw_to_cid_meth, kw_to_cid_health,
                         chunksize=20000, start_year=START_YEAR,
                         end_year=END_YEAR):
    """
    Walk the corpus in chunks; for each article assign it to the union of
    clusters whose keyword set it intersects, in each domain.

    Returns:
        meth_topic_year: dict (cluster_id, year) -> count
        health_topic_year: same
        meth_topic_total: dict cluster_id -> count
        health_topic_total: same
        n_articles_seen: int
        n_articles_with_year: int
    """
    meth_topic_year = Counter()
    health_topic_year = Counter()

    n_total = 0
    n_with_year = 0
    n_with_kws = 0

    t0 = time.time()
    for chunk in pd.read_csv(corpus_path, quoting=csv.QUOTE_ALL,
                             chunksize=chunksize, dtype=str):
        n_total += len(chunk)

        # Year coercion
        years = pd.to_numeric(chunk["year"], errors="coerce")
        chunk = chunk[years.between(start_year, end_year)].copy()
        chunk["year"] = years[years.between(start_year, end_year)].astype(int)
        n_with_year += len(chunk)

        for _, row in chunk.iterrows():
            kw_str = row.get("keywords", "")
            if not isinstance(kw_str, str) or not kw_str:
                continue
            year = int(row["year"])

            meth_clusters = set()
            health_clusters = set()
            for kw in kw_str.split(";"):
                kw = kw.strip().lower()
                if not kw:
                    continue
                cid = kw_to_cid_meth.get(kw)
                if cid is not None:
                    meth_clusters.add(cid)
                cid = kw_to_cid_health.get(kw)
                if cid is not None:
                    health_clusters.add(cid)

            if meth_clusters or health_clusters:
                n_with_kws += 1

            for cid in meth_clusters:
                meth_topic_year[(cid, year)] += 1
            for cid in health_clusters:
                health_topic_year[(cid, year)] += 1

        elapsed = time.time() - t0
        logger.info(f"  Processed {n_total} rows ({elapsed:.0f}s elapsed)")

    logger.info(f"Article totals: seen={n_total}, with valid year in "
                f"[{start_year},{end_year}]={n_with_year}, "
                f"mapped to >=1 cluster={n_with_kws}")
    return meth_topic_year, health_topic_year, n_total, n_with_year, n_with_kws


def save_topic_year_counts(topic_year, k, domain, vis_dir,
                           start_year=START_YEAR, end_year=END_YEAR):
    """Save (cluster_id, total, year_counts...) wide-format CSV."""
    years = list(range(start_year, end_year + 1))
    rows = []
    cluster_ids = sorted({cid for (cid, _) in topic_year.keys()})
    for cid in cluster_ids:
        year_counts = [topic_year.get((cid, y), 0) for y in years]
        total = sum(year_counts)
        rows.append([cid, total] + year_counts)
    out = os.path.join(vis_dir, f"{domain}_topic_year_counts.csv")
    df = pd.DataFrame(rows, columns=["topic", "total"] + [str(y) for y in years])
    df.to_csv(out, index=False, quoting=csv.QUOTE_ALL)
    logger.info(f"  Saved {len(df)} cluster rows to {out}")
    # Also report empty clusters (clusters in K but with zero articles)
    n_with_articles = len(cluster_ids)
    n_empty = k - n_with_articles
    logger.info(f"  {domain}: {n_with_articles}/{k} clusters have >=1 article "
                f"({n_empty} empty after keyword-overlap mapping)")
    return df, n_empty


# ---------------------------------------------------------------------------
# Step 4: linear-trend testing with Holm-Bonferroni
# ---------------------------------------------------------------------------

def compute_topic_trends(topic_year, years, min_articles=MIN_ARTICLES_FOR_TREND):
    """
    Replicates trend_analysis_and_visualization.py:compute_topic_trends
    (lines 651-679), keyed on cluster_id rather than topic name.
    """
    topic_totals = Counter()
    for (topic, year), count in topic_year.items():
        topic_totals[topic] += count

    trends = []
    year_arr = np.array(years, dtype=float)

    for topic, total in topic_totals.items():
        if total < min_articles:
            continue
        counts = np.array([topic_year.get((topic, y), 0) for y in years],
                          dtype=float)
        if counts.sum() == 0:
            continue
        reg = sstats.linregress(year_arr, counts)
        mean_count = counts.mean()
        rel_slope = reg.slope / mean_count if mean_count > 0 else 0

        trends.append({
            "topic_id": topic,
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


# ---------------------------------------------------------------------------
# Step 2: topology summary
# ---------------------------------------------------------------------------

def topology_stats_for_domain(cluster_df, topic_year_counts_df, k):
    """
    Compute keyword-per-topic and articles-per-topic distributions.
    cluster_df: from {domain}_keyword_clusters.csv
    topic_year_counts_df: per-cluster annual counts (rows = cluster_ids w/>=1 article)
    """
    # Keywords per topic (over ALL k clusters; empty clusters count as 0)
    sizes_per_cluster = cluster_df.groupby("cluster_id").size()
    # ensure we have entries for all 0..k-1 (fill 0 for fully empty cluster IDs)
    full = pd.Series(0, index=range(k), dtype=int)
    full.update(sizes_per_cluster)
    kw_sizes = full.values

    # Articles per topic (also include zero-article clusters)
    articles_full = pd.Series(0, index=range(k), dtype=int)
    if "topic" in topic_year_counts_df.columns and "total" in topic_year_counts_df.columns:
        for cid, tot in zip(topic_year_counts_df["topic"], topic_year_counts_df["total"]):
            articles_full.loc[int(cid)] = int(tot)
    article_sizes = articles_full.values

    def stats(arr):
        arr = np.asarray(arr)
        return {
            "median": float(np.median(arr)),
            "q1": float(np.percentile(arr, 25)),
            "q3": float(np.percentile(arr, 75)),
            "min": int(arr.min()),
            "max": int(arr.max()),
            "mean": float(arr.mean()),
        }

    return {
        "n_topics": k,
        "keywords_per_topic": stats(kw_sizes),
        "articles_per_topic": stats(article_sizes),
    }


def write_topology_csv(stats_by_domain, out_path):
    rows = []
    for domain, stats in stats_by_domain.items():
        rows.append([domain, "n_topics", stats["n_topics"]])
        for prefix in ("keywords_per_topic", "articles_per_topic"):
            for stat_name, val in stats[prefix].items():
                rows.append([domain, f"{prefix}_{stat_name}", val])
    pd.DataFrame(rows, columns=["domain", "statistic", "value"]).to_csv(
        out_path, index=False
    )
    logger.info(f"Wrote {out_path}")


def write_topology_md(stats_by_domain, k_by_domain, n_significant_by_domain,
                      n_testable_by_domain, n_empty_by_domain, out_path):
    """Mirror topology_table_k100.md style."""
    lines = []
    lines.append("# Topology Table — K = 1,050 (methodology) / K = 1,100 (health)")
    lines.append("")
    lines.append("Lightweight (cluster-ID-only, no GPT topic names).")
    lines.append("")
    lines.append("| Statistic | Methodological Innovation (K=1,050) | Health Domain (K=1,100) |")
    lines.append("| --- | --- | --- |")

    def cell(stats_dict, key):
        d = stats_dict[key]
        return (f"Median {d['median']:.0f} [Q1 {d['q1']:.0f}, "
                f"Q3 {d['q3']:.0f}]; Min {d['min']}; Max {d['max']}")

    m = stats_by_domain["methodology"]
    h = stats_by_domain["health"]
    lines.append(f"| Number of topics (K) | {m['n_topics']} | {h['n_topics']} |")
    lines.append(f"| Topics with >=1 article | "
                 f"{m['n_topics'] - n_empty_by_domain['methodology']} | "
                 f"{h['n_topics'] - n_empty_by_domain['health']} |")
    lines.append(f"| Empty topics (zero articles after mapping) | "
                 f"{n_empty_by_domain['methodology']} | "
                 f"{n_empty_by_domain['health']} |")
    lines.append(f"| Keywords/topic | "
                 f"{cell(m, 'keywords_per_topic')} | "
                 f"{cell(h, 'keywords_per_topic')} |")
    lines.append(f"| Articles/topic | "
                 f"{cell(m, 'articles_per_topic')} | "
                 f"{cell(h, 'articles_per_topic')} |")
    lines.append(f"| Holm-significant trends | "
                 f"{n_significant_by_domain['methodology']}/"
                 f"{n_testable_by_domain['methodology']} testable | "
                 f"{n_significant_by_domain['health']}/"
                 f"{n_testable_by_domain['health']} testable |")
    lines.append("")
    lines.append("*Note: \"testable\" = clusters with >=10 total articles 2011-2025. "
                 "Holm-Bonferroni adjustment applied within each domain.*")
    lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.vis_dir, exist_ok=True)
    os.makedirs(args.k_comparison_dir, exist_ok=True)
    k_by_domain = {"methodology": args.k_methodology, "health": args.k_health}

    # ----- load cluster maps -----
    logger.info("Loading keyword->cluster maps")
    kw_to_cid_meth, meth_cluster_df = load_keyword_clusters(
        args.clusters_dir, "methodology")
    kw_to_cid_health, health_cluster_df = load_keyword_clusters(
        args.clusters_dir, "health")

    # ----- Step 3: article -> cluster mapping (or load existing) -----
    meth_tyc_path = os.path.join(args.vis_dir, "methodology_topic_year_counts.csv")
    health_tyc_path = os.path.join(args.vis_dir, "health_topic_year_counts.csv")
    skip_mapping = args.skip_mapping or (
        os.path.exists(meth_tyc_path) and os.path.exists(health_tyc_path)
    )

    if skip_mapping:
        logger.info("Topic-year-counts already exist; skipping mapping")
        meth_tyc_df = pd.read_csv(meth_tyc_path, quoting=csv.QUOTE_ALL)
        health_tyc_df = pd.read_csv(health_tyc_path, quoting=csv.QUOTE_ALL)
        years = [c for c in meth_tyc_df.columns if c not in ("topic", "total")]
        # Reconstruct topic_year dicts for trend computation
        meth_topic_year = Counter()
        for _, row in meth_tyc_df.iterrows():
            cid = int(row["topic"])
            for y in years:
                meth_topic_year[(cid, int(y))] = int(row[y])
        health_topic_year = Counter()
        for _, row in health_tyc_df.iterrows():
            cid = int(row["topic"])
            for y in years:
                health_topic_year[(cid, int(y))] = int(row[y])
    else:
        logger.info("Mapping articles to clusters via keyword overlap")
        (meth_topic_year, health_topic_year,
         n_total, n_with_year, n_with_kws) = map_articles_chunked(
            args.corpus, kw_to_cid_meth, kw_to_cid_health,
            chunksize=args.chunksize)

        meth_tyc_df, _ = save_topic_year_counts(
            meth_topic_year, args.k_methodology, "methodology", args.vis_dir)
        health_tyc_df, _ = save_topic_year_counts(
            health_topic_year, args.k_health, "health", args.vis_dir)

    # Recompute n_empty (cluster IDs absent from topic_year_counts)
    n_empty_by_domain = {
        "methodology": args.k_methodology - meth_tyc_df["topic"].nunique(),
        "health": args.k_health - health_tyc_df["topic"].nunique(),
    }

    # ----- Step 4: trend testing with Holm-Bonferroni -----
    years = list(range(START_YEAR, END_YEAR + 1))
    n_significant = {}
    n_testable = {}
    for domain, k, ty in [
        ("methodology", args.k_methodology, meth_topic_year),
        ("health",      args.k_health,      health_topic_year),
    ]:
        logger.info(f"Running linregress + Holm-Bonferroni for {domain} K={k}")
        tdf = compute_topic_trends(ty, years, min_articles=MIN_ARTICLES_FOR_TREND)
        n_testable[domain] = len(tdf)
        if len(tdf) > 0:
            n_sig = int(tdf["significant_holm"].sum())
        else:
            n_sig = 0
        n_significant[domain] = n_sig

        # Save trend CSV in the requested column subset
        cols_out = ["topic_id", "total", "slope", "relative_slope",
                    "p_value", "p_value_holm", "significant_holm"]
        if not tdf.empty:
            out_df = tdf[cols_out].copy()
        else:
            out_df = pd.DataFrame(columns=cols_out)
        out_path = os.path.join(args.vis_dir, f"{domain}_trend_analysis.csv")
        out_df.to_csv(out_path, index=False)
        logger.info(f"  {domain}: {n_sig}/{len(tdf)} clusters with Holm-significant trends -> {out_path}")

    # ----- Step 2: topology stats -----
    stats_by_domain = {
        "methodology": topology_stats_for_domain(meth_cluster_df, meth_tyc_df,
                                                 args.k_methodology),
        "health": topology_stats_for_domain(health_cluster_df, health_tyc_df,
                                            args.k_health),
    }

    write_topology_csv(stats_by_domain,
                       os.path.join(args.k_comparison_dir,
                                    "topology_table_k1050_1100.csv"))
    write_topology_md(stats_by_domain, k_by_domain,
                      n_significant, n_testable, n_empty_by_domain,
                      os.path.join(args.k_comparison_dir,
                                   "topology_table_k1050_1100.md"))

    # Also dump a JSON summary for the final report assembly
    summary = {
        "k": k_by_domain,
        "n_significant_holm": n_significant,
        "n_testable": n_testable,
        "n_empty_clusters": n_empty_by_domain,
        "topology": {
            d: {
                "n_topics": s["n_topics"],
                "keywords_per_topic": s["keywords_per_topic"],
                "articles_per_topic": s["articles_per_topic"],
            } for d, s in stats_by_domain.items()
        },
    }
    with open(os.path.join(args.vis_dir, "topology_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Done.")


if __name__ == "__main__":
    main()
