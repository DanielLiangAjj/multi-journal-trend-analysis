"""
Step 2: Keyword Clustering using K-means on BiomedBERT Embeddings

Follows the paper's methodology (taxonomy_util.py):
  1. Load categorized keywords from Step 1 output
  2. Load BiomedBERT CLS-token embeddings (cached from Step 1)
  3. Split keywords into two pools:
     - methodology pool = "methodology" + "both" keywords
     - health pool = "health" + "both" keywords
     - "neither" keywords are EXCLUDED from clustering
  4. Run K-means on each pool separately (default K=750, faithful to paper)
  5. [Optional] Run elbow + silhouette scan to evaluate K choices (--scan-k)
  6. Save cluster assignments, summaries, and diagnostic plots

Data integrity guarantees:
  - "both" keywords are included in BOTH pools (not dropped)
  - All keywords with valid labels and embeddings are clustered
  - Validation checks at every stage with detailed logging
  - Deterministic results via fixed random_state=42
  - Output row counts verified against input

Usage:
    python cluster_keywords.py \\
        --categorized data/keywords_categorized.csv \\
        --embedding-cache data/keyword_embeddings.pkl \\
        --output-dir data/clusters \\
        [--k 750] \\
        [--k-methodology 750] [--k-health 750] \\
        [--scan-k] [--scan-k-min 50] [--scan-k-max 1000] [--scan-k-step 50] \\
        [--silhouette-sample 5000]

Requirements:
    pip install scikit-learn numpy pandas matplotlib torch transformers tqdm
"""

import argparse
import csv
import json
import logging
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server
import matplotlib.pyplot as plt
from tqdm import tqdm

BIOMEDBERT_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

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
        description="Step 2: Cluster keywords using K-means on BiomedBERT embeddings."
    )
    parser.add_argument(
        "--categorized", required=True,
        help="Path to keywords_categorized.csv from Step 1",
    )
    parser.add_argument(
        "--embedding-cache", default="data/keyword_embeddings.pkl",
        help="Path to BiomedBERT embedding cache from Step 1",
    )
    parser.add_argument(
        "--output-dir", default="data/clusters",
        help="Output directory for cluster results",
    )

    # K specification
    parser.add_argument(
        "--k", type=int, default=750,
        help="Number of clusters K (default: 750, matching the paper). "
             "Applied to both domains unless --k-methodology / --k-health are set.",
    )
    parser.add_argument(
        "--k-methodology", type=int, default=None,
        help="Override K specifically for methodology domain",
    )
    parser.add_argument(
        "--k-health", type=int, default=None,
        help="Override K specifically for health domain",
    )

    # Optional K scan
    parser.add_argument(
        "--scan-k", action="store_true",
        help="Run elbow + silhouette analysis to help choose K. "
             "Saves diagnostic plots and CSV. Does NOT change the K used for "
             "final clustering (re-run with --k to use a different value).",
    )
    parser.add_argument("--scan-k-min", type=int, default=50,
                        help="Minimum K for scan (default: 50)")
    parser.add_argument("--scan-k-max", type=int, default=1000,
                        help="Maximum K for scan (default: 1000)")
    parser.add_argument("--scan-k-step", type=int, default=50,
                        help="Step size for K scan (default: 50)")
    parser.add_argument(
        "--silhouette-sample", type=int, default=5000,
        help="Sample size for silhouette score (default: 5000). "
             "Larger = more accurate but slower.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Load and validate data
# ---------------------------------------------------------------------------

def load_categorized_keywords(csv_path):
    """
    Load categorized keywords from Step 1 output.
    Normalizes keywords to lowercase and deduplicates.
    """
    logger.info(f"Loading categorized keywords from {csv_path}")
    df = pd.read_csv(csv_path, quoting=csv.QUOTE_ALL)

    # Validate required columns
    required_cols = {"keyword", "label", "count"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in {csv_path}: {missing_cols}")

    initial_count = len(df)

    # Normalize keywords to lowercase and strip whitespace
    df["keyword"] = df["keyword"].astype(str).str.strip().str.lower()

    # Drop rows with empty keywords
    df = df[df["keyword"].str.len() > 0].copy()

    # Normalize labels
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # Validate labels
    valid_labels = {"methodology", "health", "both", "neither"}
    invalid_mask = ~df["label"].isin(valid_labels)
    if invalid_mask.sum() > 0:
        logger.warning(
            f"Found {invalid_mask.sum()} keywords with invalid labels "
            f"(e.g., {df.loc[invalid_mask, 'label'].unique()[:5]}), treating as 'neither'"
        )
        df.loc[invalid_mask, "label"] = "neither"

    # Ensure count is integer
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)

    # Deduplicate: if the same keyword appears with different labels after
    # normalization, merge them intelligently rather than dropping data:
    #   methodology + health -> both
    #   any + neither -> keep the informative one
    #   same label duplicates -> keep one, sum counts
    before_dedup = len(df)
    if df["keyword"].duplicated().any():
        merged_rows = []
        for kw, group in df.groupby("keyword", sort=False):
            labels_set = set(group["label"])
            total_count = group["count"].sum()

            if len(labels_set) == 1:
                final_label = labels_set.pop()
            else:
                # Remove "neither" — keep informative labels
                labels_set.discard("neither")
                if not labels_set:
                    final_label = "neither"
                elif labels_set == {"both"} or len(labels_set) > 1:
                    # methodology+health, methodology+both, health+both, etc. -> both
                    final_label = "both"
                else:
                    final_label = labels_set.pop()

            merged_rows.append({
                "keyword": kw,
                "label": final_label,
                "count": total_count,
            })
        df = pd.DataFrame(merged_rows)
        dedup_removed = before_dedup - len(df)
        if dedup_removed > 0:
            logger.warning(
                f"Merged {dedup_removed} duplicate keyword rows after normalization "
                f"(labels combined: methodology+health -> both)"
            )

    # Log label distribution
    label_counts = df["label"].value_counts()
    logger.info(f"Loaded {len(df)} categorized keywords:")
    for label in ["methodology", "health", "both", "neither"]:
        count = label_counts.get(label, 0)
        pct = count / len(df) * 100
        logger.info(f"  {label}: {count} ({pct:.1f}%)")

    return df


def load_embedding_cache(cache_path):
    """
    Load BiomedBERT embedding cache from Step 1.
    Returns (keyword_list, embedding_matrix, keyword_to_index_dict).
    """
    logger.info(f"Loading embedding cache from {cache_path}")

    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Embedding cache not found: {cache_path}\n"
            f"This file should have been created during Step 1 (categorize_keywords.py).\n"
            f"Please verify Step 1 completed successfully."
        )

    with open(cache_path, "rb") as f:
        cached = pickle.load(f)

    cached_keywords = cached["labels"]
    cached_embeddings = cached["embeddings"]

    logger.info(
        f"Loaded {len(cached_keywords)} keyword embeddings, "
        f"shape: {cached_embeddings.shape}"
    )

    # Build case-insensitive index: keyword (lowercase) -> row index
    kw_to_idx = {}
    for i, kw in enumerate(cached_keywords):
        kw_lower = str(kw).strip().lower()
        if kw_lower not in kw_to_idx:
            kw_to_idx[kw_lower] = i

    return cached_keywords, cached_embeddings, kw_to_idx


def compute_missing_embeddings(missing_keywords, cache_path, cached_keywords, cached_embeddings):
    """
    Compute BiomedBERT embeddings for keywords missing from the cache.
    Updates the cache file on disk.
    Returns updated (keywords_list, embeddings_matrix).
    """
    logger.info(f"Computing BiomedBERT embeddings for {len(missing_keywords)} missing keywords...")

    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(BIOMEDBERT_MODEL)
    model = AutoModel.from_pretrained(BIOMEDBERT_MODEL)
    model.eval()

    new_embeddings = []
    batch_size = 64
    for i in tqdm(range(0, len(missing_keywords), batch_size), desc="Embedding missing keywords"):
        batch = missing_keywords[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True,
            padding=True, max_length=512,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :].numpy()
        new_embeddings.append(cls_emb)

    new_embeddings = np.vstack(new_embeddings)

    # Merge with existing cache
    updated_keywords = list(cached_keywords) + list(missing_keywords)
    updated_embeddings = np.vstack([cached_embeddings, new_embeddings])

    # Save updated cache
    with open(cache_path, "wb") as f:
        pickle.dump({"labels": updated_keywords, "embeddings": updated_embeddings}, f)

    logger.info(f"Updated embedding cache: {len(updated_keywords)} keywords total")

    return updated_keywords, updated_embeddings


# ---------------------------------------------------------------------------
# Build domain pools
# ---------------------------------------------------------------------------

def build_domain_pools(cat_df, kw_to_idx):
    """
    Split keywords into methodology and health pools.

    CRITICAL: "both" keywords go into BOTH pools — this is essential for
    accurate research analysis, since these cross-cutting keywords (e.g.,
    "allergen detection") contain both a method and a health concept.

    "neither" keywords are excluded from clustering.
    Keywords without embeddings are logged and excluded (should be 0).
    """
    meth_kws = []
    health_kws = []
    excluded_neither = 0
    excluded_no_emb = 0
    both_count = 0

    for _, row in cat_df.iterrows():
        kw = row["keyword"]
        label = row["label"]

        if label == "neither":
            excluded_neither += 1
            continue

        if kw not in kw_to_idx:
            excluded_no_emb += 1
            continue

        if label == "methodology":
            meth_kws.append(kw)
        elif label == "health":
            health_kws.append(kw)
        elif label == "both":
            meth_kws.append(kw)
            health_kws.append(kw)
            both_count += 1

    logger.info("Domain pool construction:")
    logger.info(f"  Methodology pool: {len(meth_kws)} keywords "
                f"({len(meth_kws) - both_count} methodology-only + {both_count} both)")
    logger.info(f"  Health pool:      {len(health_kws)} keywords "
                f"({len(health_kws) - both_count} health-only + {both_count} both)")
    logger.info(f"  Excluded (neither): {excluded_neither}")
    if excluded_no_emb > 0:
        logger.warning(f"  Excluded (no embedding): {excluded_no_emb}")

    # Validation: every non-'neither' keyword with an embedding should be in at least one pool
    expected_included = len(cat_df[cat_df["label"] != "neither"]) - excluded_no_emb
    actual_included = len(set(meth_kws) | set(health_kws))
    if actual_included != expected_included:
        logger.error(
            f"DATA INTEGRITY ERROR: Expected {expected_included} keywords in pools, "
            f"but found {actual_included}. Investigate before trusting results!"
        )
    else:
        logger.info(f"  Data integrity check PASSED: "
                    f"all {actual_included} eligible keywords included in pools")

    return meth_kws, health_kws


# ---------------------------------------------------------------------------
# K-value scanning (optional)
# ---------------------------------------------------------------------------

def scan_k_values(embeddings, k_min, k_max, k_step, silhouette_sample,
                  domain_name, output_dir):
    """
    Scan K values using MiniBatchKMeans (fast) + silhouette analysis.
    Saves diagnostic plots and a CSV of results.
    Returns the K with the highest silhouette score.
    """
    # Ensure K range is valid for the pool size
    max_valid_k = len(embeddings) - 1
    k_values = [k for k in range(k_min, min(k_max + 1, max_valid_k + 1), k_step)]

    if not k_values:
        logger.warning(
            f"No valid K values in range [{k_min}, {k_max}] for "
            f"{len(embeddings)} keywords. Skipping scan."
        )
        return k_min

    inertias = []
    sil_scores = []

    logger.info(
        f"Scanning K from {k_values[0]} to {k_values[-1]} (step {k_step}) "
        f"for {domain_name}..."
    )
    logger.info(f"  Pool: {len(embeddings)} keywords, dim: {embeddings.shape[1]}")

    for k in tqdm(k_values, desc=f"K-scan ({domain_name})"):
        kmeans = MiniBatchKMeans(
            n_clusters=k, random_state=42, batch_size=1024,
            n_init=3, max_iter=300,
        )
        labels = kmeans.fit_predict(embeddings)
        inertias.append(kmeans.inertia_)

        # Silhouette score (sampled for speed on large datasets)
        sample_size = min(silhouette_sample, len(embeddings))
        sil = silhouette_score(
            embeddings, labels, sample_size=sample_size, random_state=42,
        )
        sil_scores.append(sil)

        logger.info(f"  K={k}: inertia={kmeans.inertia_:.0f}, silhouette={sil:.4f}")

    # Find best K by silhouette
    best_idx = int(np.argmax(sil_scores))
    best_k = k_values[best_idx]
    best_sil = sil_scores[best_idx]

    logger.info(f"Best K for {domain_name}: {best_k} (silhouette={best_sil:.4f})")

    # Save scan results as CSV
    scan_csv = os.path.join(output_dir, f"{domain_name}_k_scan.csv")
    with open(scan_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "inertia", "silhouette_score"])
        for j, k in enumerate(k_values):
            writer.writerow([k, f"{inertias[j]:.2f}", f"{sil_scores[j]:.6f}"])
    logger.info(f"K-scan results saved to {scan_csv}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(k_values, inertias, "b-o", markersize=3)
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia")
    ax1.set_title(f"Elbow Method - {domain_name.title()}")
    ax1.axvline(x=best_k, color="r", linestyle="--", alpha=0.7,
                label=f"Best K={best_k}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(k_values, sil_scores, "g-o", markersize=3)
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title(f"Silhouette Analysis - {domain_name.title()}")
    ax2.axvline(x=best_k, color="r", linestyle="--", alpha=0.7,
                label=f"Best K={best_k}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{domain_name}_elbow_silhouette.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Diagnostic plot saved to {plot_path}")

    return best_k


# ---------------------------------------------------------------------------
# K-means clustering
# ---------------------------------------------------------------------------

def run_final_kmeans(embeddings, k, domain_name):
    """
    Run final K-means clustering with the chosen K.
    Uses full KMeans (not MiniBatch) with n_init=10 for best quality.
    Faithful to paper: KMeans(n_clusters=K, random_state=42).
    """
    # Guard: K must be < number of samples, and we need at least 2 samples
    if len(embeddings) < 2:
        logger.error(
            f"Pool '{domain_name}' has only {len(embeddings)} keyword(s) — "
            f"too few to cluster. Skipping."
        )
        return (
            np.zeros(len(embeddings), dtype=int),
            np.zeros((1, embeddings.shape[1])),
            np.zeros(len(embeddings)),
            0.0,
            1,
        )
    if k >= len(embeddings):
        old_k = k
        k = max(2, min(len(embeddings) - 1, len(embeddings) // 2))
        logger.warning(
            f"K={old_k} >= pool size {len(embeddings)}, reducing to K={k}"
        )

    logger.info(
        f"Running K-means for {domain_name}: K={k}, "
        f"n={len(embeddings)}, dim={embeddings.shape[1]}"
    )
    start_time = time.time()

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(embeddings)

    elapsed = time.time() - start_time
    logger.info(f"K-means completed in {elapsed:.1f}s")

    # Compute distance from each keyword to its assigned centroid
    distances = np.linalg.norm(
        embeddings - kmeans.cluster_centers_[labels], axis=1,
    )

    # Cluster size statistics
    unique_labels, cluster_sizes = np.unique(labels, return_counts=True)
    n_non_empty = len(unique_labels)
    n_empty = k - n_non_empty

    logger.info(f"  Non-empty clusters: {n_non_empty} / {k}")
    if n_empty > 0:
        logger.warning(f"  Empty clusters: {n_empty}")
    logger.info(
        f"  Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
        f"mean={cluster_sizes.mean():.1f}, median={np.median(cluster_sizes):.0f}"
    )

    # Verify every input got a label
    assert len(labels) == len(embeddings), \
        f"K-means label count {len(labels)} != input count {len(embeddings)}"

    return labels, kmeans.cluster_centers_, distances, kmeans.inertia_, k


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_cluster_assignments(keywords, labels, distances, counts_dict,
                             domain, output_dir):
    """
    Save per-keyword cluster assignments to CSV.
    Validates output row count matches input.
    """
    output_path = os.path.join(output_dir, f"{domain}_keyword_clusters.csv")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["keyword", "cluster_id", "distance_to_centroid", "count"])
        for i, kw in enumerate(keywords):
            writer.writerow([
                kw,
                int(labels[i]),
                f"{distances[i]:.6f}",
                counts_dict.get(kw, 0),
            ])

    # Validate output file
    with open(output_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        output_count = sum(1 for _ in reader)

    if output_count != len(keywords):
        logger.error(
            f"OUTPUT VALIDATION FAILED: wrote {output_count} rows "
            f"but expected {len(keywords)} for {domain}"
        )
    else:
        logger.info(
            f"Saved {len(keywords)} keyword-cluster assignments to {output_path} "
            f"(validated)"
        )

    return output_path


def save_cluster_summary(keywords, labels, distances, k, domain, output_dir,
                         top_n=20):
    """
    Save per-cluster summary with top representative keywords
    (sorted by proximity to centroid).
    """
    output_path = os.path.join(output_dir, f"{domain}_cluster_summary.csv")

    keywords_arr = np.array(keywords)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["cluster_id", "size", "top_keywords"])

        for cluster_id in range(k):
            mask = labels == cluster_id
            cluster_size = int(mask.sum())

            if cluster_size == 0:
                writer.writerow([cluster_id, 0, ""])
                continue

            cluster_kws = keywords_arr[mask]
            cluster_dists = distances[mask]

            # Sort by distance to centroid — closest keywords are most representative
            sorted_idx = np.argsort(cluster_dists)
            top_kws = cluster_kws[sorted_idx][:top_n]

            writer.writerow([
                cluster_id,
                cluster_size,
                "; ".join(top_kws),
            ])

    logger.info(f"Saved cluster summary ({k} clusters) to {output_path}")
    return output_path


def save_cluster_dict_json(keywords, labels, k, domain, output_dir):
    """
    Save cluster assignments as JSON dict: {cluster_id: [keywords...]}.
    Compatible with the paper's extract_clusters() post-processing format.
    Validates that total keywords in JSON == total input keywords.
    """
    output_path = os.path.join(output_dir, f"{domain}_clusters.json")

    cluster_dict = {}
    for cluster_id in range(k):
        mask = labels == cluster_id
        cluster_kws = [keywords[i] for i in range(len(keywords)) if mask[i]]
        if cluster_kws:
            cluster_dict[str(cluster_id)] = cluster_kws

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cluster_dict, f, indent=2, ensure_ascii=False)

    # Validate: total keywords across all clusters must equal input
    total_in_json = sum(len(kws) for kws in cluster_dict.values())
    if total_in_json != len(keywords):
        logger.error(
            f"JSON VALIDATION FAILED: {total_in_json} keywords in JSON "
            f"vs {len(keywords)} input for {domain}"
        )
    else:
        logger.info(
            f"Saved cluster dict ({len(cluster_dict)} non-empty clusters) "
            f"to {output_path} (validated)"
        )

    return output_path


def save_centroids(centroids, domain, output_dir):
    """Save cluster centroid vectors for Step 3 (topic naming)."""
    output_path = os.path.join(output_dir, f"{domain}_centroids.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(centroids, f)
    logger.info(f"Saved {len(centroids)} centroids to {output_path}")


def save_cluster_size_plot(labels, k, domain, output_dir):
    """Save histogram of cluster size distribution."""
    _, cluster_sizes = np.unique(labels, return_counts=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(cluster_sizes, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Cluster Size (number of keywords)")
    ax.set_ylabel("Number of Clusters")
    ax.set_title(f"Cluster Size Distribution - {domain.title()} (K={k})")
    ax.axvline(
        np.median(cluster_sizes), color="r", linestyle="--",
        label=f"Median={np.median(cluster_sizes):.0f}",
    )
    ax.axvline(
        cluster_sizes.mean(), color="orange", linestyle="--",
        label=f"Mean={cluster_sizes.mean():.0f}",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    hist_path = os.path.join(output_dir, f"{domain}_cluster_sizes.png")
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Cluster size distribution plot saved to {hist_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Step 2: Keyword Clustering using K-means on BiomedBERT Embeddings")
    logger.info("=" * 70)

    # ---- 1. Load categorized keywords from Step 1 ----
    cat_df = load_categorized_keywords(args.categorized)

    # ---- 2. Load embedding cache from Step 1 ----
    cached_keywords, cached_embeddings, kw_to_idx = load_embedding_cache(
        args.embedding_cache,
    )

    # ---- 3. Validate embedding coverage for non-'neither' keywords ----
    non_neither_kws = cat_df.loc[cat_df["label"] != "neither", "keyword"].tolist()
    missing_kws = [kw for kw in non_neither_kws if kw not in kw_to_idx]

    if missing_kws:
        logger.warning(
            f"{len(missing_kws)} non-'neither' keywords missing from embedding cache"
        )
        logger.warning(f"First 10 missing: {missing_kws[:10]}")

        # Attempt to recompute missing embeddings
        cached_keywords, cached_embeddings = compute_missing_embeddings(
            missing_kws, args.embedding_cache, cached_keywords, cached_embeddings,
        )
        # Rebuild index after adding new embeddings
        kw_to_idx = {}
        for i, kw in enumerate(cached_keywords):
            kw_lower = str(kw).strip().lower()
            if kw_lower not in kw_to_idx:
                kw_to_idx[kw_lower] = i

        # Verify all are now covered
        still_missing = [kw for kw in non_neither_kws if kw not in kw_to_idx]
        if still_missing:
            logger.error(
                f"Still {len(still_missing)} keywords without embeddings "
                f"after recomputation! These will be excluded from clustering."
            )
    else:
        logger.info(
            f"All {len(non_neither_kws)} non-'neither' keywords found in "
            f"embedding cache"
        )

    # ---- 4. Build domain pools ----
    meth_kws, health_kws = build_domain_pools(cat_df, kw_to_idx)

    # Build counts dict for output
    counts_dict = dict(zip(cat_df["keyword"], cat_df["count"]))

    # ---- 5. Optional K scan ----
    if args.scan_k:
        logger.info("\n" + "=" * 50)
        logger.info("K-value scan (elbow + silhouette analysis)")
        logger.info("=" * 50)

        for domain, pool_kws in [("methodology", meth_kws), ("health", health_kws)]:
            indices = [kw_to_idx[kw] for kw in pool_kws]
            pool_embs = cached_embeddings[indices]

            recommended_k = scan_k_values(
                pool_embs,
                args.scan_k_min, args.scan_k_max, args.scan_k_step,
                args.silhouette_sample, domain, args.output_dir,
            )
            logger.info(
                f"Recommended K for {domain}: {recommended_k} "
                f"(to use: re-run with --k-{domain} {recommended_k})"
            )

    # ---- 6. Final K-means clustering ----
    logger.info("\n" + "=" * 50)
    logger.info("Final K-means clustering")
    logger.info("=" * 50)

    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": {k: str(v) for k, v in vars(args).items()},
    }

    for domain, pool_kws in [("methodology", meth_kws), ("health", health_kws)]:
        logger.info(f"\n--- {domain.upper()} DOMAIN ---")

        # Determine K for this domain
        if domain == "methodology" and args.k_methodology is not None:
            k = args.k_methodology
        elif domain == "health" and args.k_health is not None:
            k = args.k_health
        else:
            k = args.k

        logger.info(f"K={k}, pool size={len(pool_kws)}")
        logger.info(
            f"Expected avg cluster size: {len(pool_kws) / k:.1f} keywords "
            f"(paper used K=750 on ~12K keywords = ~16 avg)"
        )

        # Extract embeddings for this pool
        indices = [kw_to_idx[kw] for kw in pool_kws]
        pool_embs = cached_embeddings[indices]

        # Verify embedding shape
        assert pool_embs.shape[0] == len(pool_kws), \
            f"Embedding row count {pool_embs.shape[0]} != keyword count {len(pool_kws)}"

        # Run K-means
        labels, centroids, distances, inertia, actual_k = run_final_kmeans(
            pool_embs, k, domain,
        )

        # Save all results
        save_cluster_assignments(
            pool_kws, labels, distances, counts_dict, domain, args.output_dir,
        )
        save_cluster_summary(
            pool_kws, labels, distances, actual_k, domain, args.output_dir,
        )
        save_cluster_dict_json(
            pool_kws, labels, actual_k, domain, args.output_dir,
        )
        save_centroids(centroids, domain, args.output_dir)
        save_cluster_size_plot(labels, actual_k, domain, args.output_dir)

        # Record metadata
        unique_labels, cluster_sizes = np.unique(labels, return_counts=True)
        metadata[domain] = {
            "k": actual_k,
            "n_keywords": len(pool_kws),
            "inertia": float(inertia),
            "n_non_empty_clusters": int(len(unique_labels)),
            "cluster_size_min": int(cluster_sizes.min()),
            "cluster_size_max": int(cluster_sizes.max()),
            "cluster_size_mean": round(float(cluster_sizes.mean()), 1),
            "cluster_size_median": round(float(np.median(cluster_sizes)), 0),
        }

    # ---- 7. Save metadata ----
    meta_path = os.path.join(args.output_dir, "clustering_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"\nMetadata saved to {meta_path}")

    # ---- 8. Final summary ----
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2 COMPLETE - CLUSTERING SUMMARY")
    logger.info("=" * 70)

    for domain in ["methodology", "health"]:
        m = metadata[domain]
        logger.info(f"\n{domain.upper()}:")
        logger.info(f"  Keywords clustered: {m['n_keywords']}")
        logger.info(f"  K (clusters):       {m['k']}")
        logger.info(f"  Non-empty clusters: {m['n_non_empty_clusters']}")
        logger.info(
            f"  Cluster sizes: min={m['cluster_size_min']}, "
            f"max={m['cluster_size_max']}, "
            f"mean={m['cluster_size_mean']}, "
            f"median={m['cluster_size_median']}"
        )

    both_count = len(cat_df[cat_df["label"] == "both"])
    neither_count = len(cat_df[cat_df["label"] == "neither"])
    total_unique = len(set(meth_kws) | set(health_kws))

    logger.info(f"\nData accounting:")
    logger.info(f"  Total keywords from Step 1:   {len(cat_df)}")
    logger.info(f"  Excluded ('neither'):          {neither_count}")
    logger.info(f"  Unique keywords in pools:      {total_unique}")
    logger.info(f"  'both' keywords (in 2 pools):  {both_count}")
    logger.info(
        f"  Total assignments:             "
        f"{metadata['methodology']['n_keywords']} + "
        f"{metadata['health']['n_keywords']} = "
        f"{metadata['methodology']['n_keywords'] + metadata['health']['n_keywords']}"
    )

    logger.info(f"\nOutput files in {args.output_dir}/:")
    logger.info("  [domain]_keyword_clusters.csv   per-keyword cluster assignments")
    logger.info("  [domain]_cluster_summary.csv    per-cluster top representative keywords")
    logger.info("  [domain]_clusters.json           cluster -> keyword list (for Step 3)")
    logger.info("  [domain]_centroids.pkl           cluster centroid vectors")
    logger.info("  [domain]_cluster_sizes.png       cluster size distribution")
    logger.info("  clustering_metadata.json          run metadata and statistics")
    if args.scan_k:
        logger.info("  [domain]_k_scan.csv              K scan results")
        logger.info("  [domain]_elbow_silhouette.png    elbow + silhouette plots")


if __name__ == "__main__":
    main()
