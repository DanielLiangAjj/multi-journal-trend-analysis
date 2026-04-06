"""
Step 3: Cluster Post-processing and Topic Naming

Following the paper's methodology (Cluster postprocessing notebooks):

Phase 1 - Cluster Post-processing:
  1. Load K-means cluster assignments from Step 2
  2. Remove abbreviation-heavy clusters (avg keyword length < 10 chars)
  3. Check cluster coherence using GPT; split non-coherent clusters into sub-topics
  4. Post-process: recover keywords dropped by GPT, remove hallucinated keywords

Phase 2 - Topic Naming:
  5. Count keyword occurrences from original article data
  6. Compute keyword frequency distributions per topic
  7. Name each topic using GPT (<=3 word phrase based on keyword distribution)
  8. Deduplicate topic names using SentenceTransformer embedding similarity

Usage:
    python postprocess_and_name_topics.py \\
        --cluster-dir data/clusters \\
        --articles-csv data/pubmed_all_journals_2011_2025_w_keywords.csv \\
        --embedding-cache data/keyword_embeddings.pkl \\
        --output-dir data/topics \\
        --openai-api-key sk-... \\
        [--model gpt-4o] \\
        [--abbr-threshold 10]

Requirements:
    pip install openai pydantic pandas numpy scikit-learn tqdm sentence-transformers
"""

import argparse
import csv
import json
import logging
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from openai import OpenAI
from pydantic import BaseModel, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models for structured GPT output
# ---------------------------------------------------------------------------

class ClusterCoherenceResponse(BaseModel):
    """GPT response for cluster coherence checking."""
    label: str  # "yes" or "no"
    subtopics: List[List[str]]  # empty if coherent, filled with sub-groups if not

    @field_validator("label")
    @classmethod
    def validate_label(cls, v):
        v = v.strip().lower()
        if v not in {"yes", "no"}:
            raise ValueError("label must be 'yes' or 'no'")
        return v


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 3: Post-process clusters and name topics."
    )
    parser.add_argument(
        "--cluster-dir", default="data/clusters",
        help="Directory with Step 2 output (clusters JSON files)",
    )
    parser.add_argument(
        "--articles-csv", required=True,
        help="Original articles CSV from data collection step",
    )
    parser.add_argument(
        "--embedding-cache", default="data/keyword_embeddings.pkl",
        help="BiomedBERT embedding cache from Step 1",
    )
    parser.add_argument(
        "--output-dir", default="data/topics",
        help="Output directory for topic results",
    )
    parser.add_argument(
        "--openai-api-key", required=True,
        help="OpenAI API key",
    )
    parser.add_argument(
        "--model", default="gpt-4o",
        help="OpenAI model for GPT calls (default: gpt-4o)",
    )
    parser.add_argument(
        "--abbr-threshold", type=int, default=10,
        help="Remove clusters with avg keyword length < this (default: 10)",
    )
    parser.add_argument(
        "--name-similarity-k", type=int, default=3,
        help="Number of similar existing names to check for reuse (default: 3)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# GPT helper
# ---------------------------------------------------------------------------

def gpt_text_response(client, model, prompt, max_retries=3):
    """Simple text response from GPT via Responses API."""
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                input=[{"role": "user", "content": prompt}],
            )
            return response.output_text.strip()
        except Exception as e:
            logger.warning(f"GPT call failed (attempt {attempt + 1}): {e}")
            time.sleep(2 * (attempt + 1))
    return ""


# ---------------------------------------------------------------------------
# Phase 1: Cluster Post-processing
# ---------------------------------------------------------------------------

def load_clusters(cluster_dir, domain):
    """Load cluster JSON from Step 2."""
    json_path = os.path.join(cluster_dir, f"{domain}_clusters.json")
    logger.info(f"Loading clusters from {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        cluster_dict = json.load(f)

    total_kws = sum(len(v) for v in cluster_dict.values())
    logger.info(f"  Loaded {len(cluster_dict)} clusters, {total_kws} total keywords")
    return cluster_dict


def check_abbreviation(cluster_dict, threshold=10):
    """
    Identify abbreviation-heavy clusters (avg keyword length < threshold).
    Paper: check_abbreviation function.
    """
    abbr_ids = []
    for cid, keywords in cluster_dict.items():
        avg_len = sum(len(kw) for kw in keywords) / max(len(keywords), 1)
        if avg_len < threshold:
            abbr_ids.append(cid)

    if abbr_ids:
        logger.info(f"  Found {len(abbr_ids)} abbreviation-heavy clusters to remove")
        for cid in abbr_ids[:5]:
            preview = cluster_dict[cid][:5]
            logger.info(f"    Cluster {cid}: {preview}...")
    else:
        logger.info("  No abbreviation-heavy clusters detected")

    return set(abbr_ids)


def break_clusters(cluster_dict, client, model, progress_path):
    """
    Check cluster coherence using GPT. Split non-coherent clusters into sub-topics.
    Paper: break_cluster function.

    Returns: {cluster_id: [[subtopic1_keywords], [subtopic2_keywords], ...]}
    Coherent clusters: {cid: [[all_keywords]]}
    """
    # Load progress
    if os.path.exists(progress_path):
        logger.info(f"  Resuming from {progress_path}")
        with open(progress_path, "r") as f:
            result = json.load(f)
    else:
        result = {}

    remaining = [cid for cid in sorted(cluster_dict.keys(), key=int)
                 if cid not in result]
    logger.info(
        f"  Breaking clusters: {len(remaining)} remaining, "
        f"{len(result)} already done"
    )

    for cid in tqdm(remaining, desc="Checking cluster coherence"):
        keywords = cluster_dict[cid]

        prompt = f"""{keywords}
Please help determine if these terms focus on a coherent topic or not. Answer strictly in the following JSON format:

If the terms form a coherent topic:
{{
  "label": "yes",
  "subtopics": []
}}

If not:
{{
  "label": "no",
  "subtopics": [
    [list of keywords for topic 1],
    [list of keywords for topic 2],
    ...
  ]
}}
**Important**: If the answer is no, group the terms by subtopics. Do not add new terms or remove existing terms from the provided list of terms."""

        for attempt in range(3):
            try:
                response = client.responses.parse(
                    model=model,
                    input=[{"role": "user", "content": prompt}],
                    text_format=ClusterCoherenceResponse,
                )
                parsed = response.output_parsed

                if parsed.label == "yes":
                    result[cid] = [keywords]
                else:
                    result[cid] = parsed.subtopics if parsed.subtopics else [keywords]
                break
            except Exception as e:
                logger.warning(f"  Cluster {cid} attempt {attempt + 1} failed: {e}")
                time.sleep(2 * (attempt + 1))
        else:
            # All retries failed — keep cluster as-is
            logger.warning(f"  Cluster {cid}: all retries failed, keeping as-is")
            result[cid] = [keywords]

        # Save progress every 10 clusters
        if len(result) % 10 == 0:
            with open(progress_path, "w") as f:
                json.dump(result, f)

    # Final save
    with open(progress_path, "w") as f:
        json.dump(result, f)

    # Stats
    total_subtopics = sum(len(v) for v in result.values())
    coherent = sum(1 for v in result.values() if len(v) == 1)
    split = sum(1 for v in result.values() if len(v) > 1)
    logger.info(
        f"  Coherence check done: {coherent} coherent, {split} split "
        f"-> {total_subtopics} total sub-topics"
    )

    return result


def postprocess_subclusters(original_clusters, subclusters, kw_embeddings, kw_to_idx):
    """
    Post-process GPT's sub-cluster assignments:
      1. Remove duplicate sub-clusters
      2. Remove hallucinated keywords (not in original cluster)
      3. Recover dropped keywords using BiomedBERT embedding similarity
      4. Deduplicate keywords within sub-clusters

    Paper: break_cluster_postprocess function.
    """
    logger.info("  Post-processing sub-clusters...")
    cleaned = {}
    total_recovered = 0
    total_removed_hallucinated = 0

    for cid in tqdm(sorted(original_clusters.keys(), key=int), desc="Post-processing"):
        original_kws = set(original_clusters[cid])
        subs = subclusters.get(cid, [list(original_kws)])

        # 1. Remove duplicate sub-clusters
        seen = set()
        unique_subs = []
        for sub in subs:
            norm = tuple(sorted(sub))
            if norm not in seen:
                seen.add(norm)
                unique_subs.append(sub)

        # 2. Remove hallucinated keywords (not in original)
        filtered_subs = []
        for sub in unique_subs:
            filtered = [kw for kw in sub if kw in original_kws]
            removed = len(sub) - len(filtered)
            total_removed_hallucinated += removed
            if filtered:
                filtered_subs.append(filtered)

        if not filtered_subs:
            filtered_subs = [list(original_kws)]

        # 3. Find dropped keywords
        subclustered_kws = set()
        for sub in filtered_subs:
            subclustered_kws.update(sub)

        dropped = [kw for kw in original_kws if kw not in subclustered_kws]

        if dropped and len(filtered_subs) > 0:
            # Assign each dropped keyword to the nearest sub-cluster
            # using BiomedBERT embeddings (already cached)
            for kw in dropped:
                if kw not in kw_to_idx:
                    # Can't embed — add to largest sub-cluster
                    largest_sub = max(filtered_subs, key=len)
                    largest_sub.append(kw)
                    total_recovered += 1
                    continue

                kw_emb = kw_embeddings[kw_to_idx[kw]].reshape(1, -1)

                # Compute centroid of each sub-cluster
                best_sub_idx = 0
                best_sim = -1
                for si, sub in enumerate(filtered_subs):
                    sub_indices = [kw_to_idx[s] for s in sub if s in kw_to_idx]
                    if not sub_indices:
                        continue
                    sub_centroid = kw_embeddings[sub_indices].mean(axis=0).reshape(1, -1)
                    sim = cosine_similarity(kw_emb, sub_centroid)[0, 0]
                    if sim > best_sim:
                        best_sim = sim
                        best_sub_idx = si

                filtered_subs[best_sub_idx].append(kw)
                total_recovered += 1

        # 4. Deduplicate within sub-clusters
        deduped_subs = [list(set(sub)) for sub in filtered_subs]
        deduped_subs = [sub for sub in deduped_subs if sub]  # remove empty

        cleaned[cid] = deduped_subs

    logger.info(
        f"  Post-processing done: recovered {total_recovered} dropped keywords, "
        f"removed {total_removed_hallucinated} hallucinated keywords"
    )

    # Validate: total keywords should match
    original_total = sum(len(v) for v in original_clusters.values())
    cleaned_total = sum(len(kw) for subs in cleaned.values() for kw in subs)
    if cleaned_total != original_total:
        logger.warning(
            f"  Keyword count mismatch after post-processing: "
            f"original={original_total}, cleaned={cleaned_total}"
        )
    else:
        logger.info(f"  Keyword count validated: {cleaned_total} (matches original)")

    return cleaned


# ---------------------------------------------------------------------------
# Phase 2: Topic Naming
# ---------------------------------------------------------------------------

def count_global_keyword_occurrences(articles_csv):
    """
    Count how many articles contain each keyword (globally).
    Paper: count_papers_per_keyword — counts per keyword across all articles.
    """
    logger.info(f"Counting keyword occurrences from {articles_csv}")
    df = pd.read_csv(articles_csv, quoting=csv.QUOTE_ALL)

    keyword_counts = Counter()
    for _, row in df.iterrows():
        kw_str = row.get("keywords", "")
        if pd.isna(kw_str) or not kw_str:
            continue
        for kw in str(kw_str).split(";"):
            kw = kw.strip().lower()
            if kw:
                keyword_counts[kw] += 1

    logger.info(f"  Counted {len(keyword_counts)} unique keywords across articles")
    return keyword_counts


def build_topic_dict(subclusters):
    """
    Convert sub-cluster dict to flat topic dict.
    Topic ID format: "cluster_id-subcluster_index" (e.g., "42-0", "42-1").
    Paper convention matching extract_clusters → count_papers_per_keyword.
    """
    topic_dict = {}
    for cid, subs in subclusters.items():
        for idx, kws in enumerate(subs):
            tid = f"{cid}-{idx}"
            topic_dict[tid] = kws
    return topic_dict


def build_topic_keyword_counts(topic_dict, global_counts):
    """
    For each topic, get the global count of each keyword.
    Paper: count_papers_per_keyword + convert_counts_to_proportions.
    """
    topic_kw_counts = {}
    for tid, keywords in topic_dict.items():
        kw_counts = {}
        for kw in keywords:
            if kw in global_counts:
                kw_counts[kw] = global_counts[kw]
        topic_kw_counts[tid] = kw_counts
    return topic_kw_counts


def convert_to_proportions(topic_kw_counts):
    """Convert keyword counts to proportions within each topic."""
    topic_proportions = {}
    for tid, kw_counts in topic_kw_counts.items():
        total = sum(kw_counts.values())
        if total == 0:
            topic_proportions[tid] = {kw: 0.0 for kw in kw_counts}
        else:
            topic_proportions[tid] = {
                kw: count / total for kw, count in kw_counts.items()
            }
    return topic_proportions


def generate_name(keyword_distribution, domain, client, model):
    """
    Generate a <=3 word topic name from keyword distribution.
    Paper: generate_name function.
    """
    if len(keyword_distribution) == 1:
        return list(keyword_distribution.keys())[0]

    if domain == "methodology":
        prompt = f"""Please use 1 precise and succinct phrase in no more than 3 words to summarize the main topic of the following list of keywords in the methodology domain based on their distribution:
{keyword_distribution}"""
    else:
        prompt = f"""Please use 1 precise and succinct phrase in no more than 3 words to summarize the main topic of the following list of keywords in the health domain based on their distribution:
{keyword_distribution}"""

    name = gpt_text_response(client, model, prompt)
    name = name.strip().rstrip(".")
    if not name:
        # Fallback: use the most frequent keyword
        name = max(keyword_distribution, key=keyword_distribution.get)
    return name


def check_existing_names_suitability(name, similar_names, keyword_distribution,
                                     domain, client, model, max_retries=3):
    """
    Check if any existing topic name can be reused for this topic.
    Paper: check_existing_names_suitability function.
    """
    phrase_str = ""
    for idx, s_name in enumerate(similar_names):
        phrase_str += f"{idx}. {s_name}\n"

    if domain == "methodology":
        prompt = f"""Please check if any of these phrases can summarize the main topic of the following list of keywords in the methodology domain based on their distribution:
{keyword_distribution}
Here is the list of phrases:
{phrase_str}

Return the ID only of the top 1 suitable phrase without providing the rationale. If none, return none only."""
    else:
        prompt = f"""Please check if any of these phrases can summarize the main topic of the following list of keywords in the health domain based on their distribution:
{keyword_distribution}
Here is the list of phrases:
{phrase_str}

Return the ID only of the top 1 suitable phrase without providing the rationale. If none, return none only."""

    for attempt in range(max_retries):
        response = gpt_text_response(client, model, prompt)
        response = response.strip().rstrip(".").lower()

        if not response:
            continue
        if response[:4] == "none":
            return name
        if response[0].isdigit():
            try:
                idx = int(response[0])
                if 0 <= idx < len(similar_names):
                    return similar_names[idx]
            except ValueError:
                pass

        logger.debug(f"  Invalid suitability response '{response}', retrying...")

    return name


def name_all_topics(topic_proportions, domain, client, model, embedder,
                    name_k, progress_path):
    """
    Name all topics using GPT, with deduplication via SentenceTransformer.
    Paper: topic_naming function.
    """
    # Load progress
    if os.path.exists(progress_path):
        logger.info(f"  Resuming naming from {progress_path}")
        with open(progress_path, "r") as f:
            saved = json.load(f)
        topic_names = saved.get("topic_names", {})
        existing_names = saved.get("existing_names", [])
        existing_embeddings = np.array(saved.get("existing_embeddings", []))
        if len(existing_embeddings) == 0:
            existing_embeddings = np.empty((0, 768))
    else:
        topic_names = {}
        existing_names = []
        existing_embeddings = np.empty((0, 768))

    remaining_tids = [tid for tid in topic_proportions if tid not in topic_names]
    logger.info(
        f"  Naming topics: {len(remaining_tids)} remaining, "
        f"{len(topic_names)} already done"
    )

    for i, tid in enumerate(tqdm(remaining_tids, desc=f"Naming topics ({domain})")):
        keyword_dist = topic_proportions[tid]
        if not keyword_dist:
            topic_names[tid] = f"unknown_topic_{tid}"
            continue

        # 1. Generate name
        name = generate_name(keyword_dist, domain, client, model)

        # 2. Check if name already exists (exact match)
        if name in existing_names:
            topic_names[tid] = name
            continue

        # 3. Embed the generated name
        name_emb = embedder.encode([name], convert_to_numpy=True)[0]

        # 4. If we have existing names, check similarity
        if len(existing_names) > 0:
            sims = cosine_similarity(
                name_emb.reshape(1, -1), existing_embeddings,
            )[0]
            top_k_idx = np.argsort(sims)[::-1][:name_k]
            top_k_names = [existing_names[j] for j in top_k_idx]

            # 5. Ask GPT if any existing name is suitable
            selected = check_existing_names_suitability(
                name, top_k_names, keyword_dist, domain, client, model,
            )

            if selected not in existing_names:
                existing_names.append(selected)
                existing_embeddings = np.vstack([
                    existing_embeddings,
                    embedder.encode([selected], convert_to_numpy=True),
                ])
            topic_names[tid] = selected
        else:
            # First topic — just add it
            existing_names.append(name)
            existing_embeddings = np.vstack([
                existing_embeddings, name_emb.reshape(1, -1),
            ])
            topic_names[tid] = name

        # Save progress periodically
        if (i + 1) % 50 == 0:
            _save_naming_progress(
                progress_path, topic_names, existing_names, existing_embeddings,
            )

    # Final save
    _save_naming_progress(
        progress_path, topic_names, existing_names, existing_embeddings,
    )

    unique_names = len(set(topic_names.values()))
    logger.info(
        f"  Naming done: {len(topic_names)} topics, "
        f"{unique_names} unique names"
    )
    return topic_names


def _save_naming_progress(path, topic_names, existing_names, existing_embeddings):
    """Save naming progress to JSON."""
    with open(path, "w") as f:
        json.dump({
            "topic_names": topic_names,
            "existing_names": existing_names,
            "existing_embeddings": existing_embeddings.tolist(),
        }, f)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(topic_dict, topic_names, topic_kw_counts, domain, output_dir):
    """Save all topic results."""
    # 1. Full topic data JSON: {tid: {name, keywords, keyword_counts}}
    full_data = {}
    for tid, keywords in topic_dict.items():
        full_data[tid] = {
            "name": topic_names.get(tid, "unnamed"),
            "keywords": keywords,
            "keyword_counts": topic_kw_counts.get(tid, {}),
            "size": len(keywords),
        }

    json_path = os.path.join(output_dir, f"{domain}_topics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_data, f, indent=2, ensure_ascii=False)
    logger.info(f"  Saved full topic data to {json_path}")

    # 2. Topic names JSON: {tid: name}
    names_path = os.path.join(output_dir, f"{domain}_topic_names.json")
    with open(names_path, "w", encoding="utf-8") as f:
        json.dump(topic_names, f, indent=2, ensure_ascii=False)
    logger.info(f"  Saved topic names to {names_path}")

    # 3. Summary CSV
    csv_path = os.path.join(output_dir, f"{domain}_topic_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["topic_id", "name", "size", "top_keywords"])

        for tid in sorted(full_data.keys(), key=lambda x: (int(x.split("-")[0]), int(x.split("-")[1]))):
            data = full_data[tid]
            # Top keywords by count
            sorted_kws = sorted(
                data["keyword_counts"].items(),
                key=lambda x: -x[1],
            )[:15]
            top_str = "; ".join(f"{kw} ({c})" for kw, c in sorted_kws)
            writer.writerow([tid, data["name"], data["size"], top_str])

    logger.info(f"  Saved topic summary to {csv_path}")

    # Validate
    total_kws = sum(d["size"] for d in full_data.values())
    named = sum(1 for n in topic_names.values() if n and n != "unnamed")
    logger.info(
        f"  {domain}: {len(full_data)} topics, {total_kws} total keywords, "
        f"{named} named"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Step 3: Cluster Post-processing and Topic Naming")
    logger.info("=" * 70)

    # Initialize OpenAI client
    client = OpenAI(api_key=args.openai_api_key)

    # Load BiomedBERT embeddings (for post-processing dropped keyword recovery)
    logger.info(f"Loading BiomedBERT embedding cache from {args.embedding_cache}")
    with open(args.embedding_cache, "rb") as f:
        cached = pickle.load(f)
    kw_embeddings = cached["embeddings"]
    kw_to_idx = {}
    for i, kw in enumerate(cached["labels"]):
        kw_lower = str(kw).strip().lower()
        if kw_lower not in kw_to_idx:
            kw_to_idx[kw_lower] = i
    logger.info(f"  Loaded {len(kw_to_idx)} keyword embeddings")

    # Load SentenceTransformer for topic name deduplication
    logger.info("Loading SentenceTransformer (all-mpnet-base-v2) for name dedup...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-mpnet-base-v2")
    logger.info("  SentenceTransformer loaded")

    # Count global keyword occurrences from article data
    global_kw_counts = count_global_keyword_occurrences(args.articles_csv)

    # Process each domain
    for domain in ["methodology", "health"]:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Processing {domain.upper()} domain")
        logger.info(f"{'=' * 50}")

        # ---- Phase 1: Post-processing ----
        logger.info("\n--- Phase 1: Cluster Post-processing ---")

        # 1. Load clusters
        clusters = load_clusters(args.cluster_dir, domain)

        # 2. Remove abbreviation clusters
        abbr_ids = check_abbreviation(clusters, args.abbr_threshold)
        if abbr_ids:
            removed_kws = sum(len(clusters[cid]) for cid in abbr_ids)
            logger.info(
                f"  Removing {len(abbr_ids)} abbreviation clusters "
                f"({removed_kws} keywords)"
            )
            # Save removed clusters for review
            removed_path = os.path.join(
                args.output_dir, f"{domain}_removed_abbreviation_clusters.json",
            )
            removed_data = {cid: clusters[cid] for cid in abbr_ids}
            with open(removed_path, "w") as f:
                json.dump(removed_data, f, indent=2)
            logger.info(f"  Saved removed clusters to {removed_path}")

            clusters = {k: v for k, v in clusters.items() if k not in abbr_ids}

        # 3. Break non-coherent clusters
        break_progress = os.path.join(
            args.output_dir, f"{domain}_break_progress.json",
        )
        subclusters = break_clusters(
            clusters, client, args.model, break_progress,
        )

        # 4. Post-process sub-clusters
        subclusters = postprocess_subclusters(
            clusters, subclusters, kw_embeddings, kw_to_idx,
        )

        # Build flat topic dict
        topic_dict = build_topic_dict(subclusters)
        logger.info(f"  Final topic count: {len(topic_dict)}")

        # ---- Phase 2: Topic Naming ----
        logger.info("\n--- Phase 2: Topic Naming ---")

        # 5. Build keyword counts per topic
        topic_kw_counts = build_topic_keyword_counts(topic_dict, global_kw_counts)

        # 6. Convert to proportions
        topic_proportions = convert_to_proportions(topic_kw_counts)

        # 7. Name topics
        naming_progress = os.path.join(
            args.output_dir, f"{domain}_naming_progress.json",
        )
        topic_names = name_all_topics(
            topic_proportions, domain, client, args.model, embedder,
            args.name_similarity_k, naming_progress,
        )

        # ---- Save results ----
        logger.info(f"\n--- Saving {domain} results ---")
        save_results(topic_dict, topic_names, topic_kw_counts, domain, args.output_dir)

    # ---- Final summary ----
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3 COMPLETE - TOPIC NAMING SUMMARY")
    logger.info("=" * 70)

    for domain in ["methodology", "health"]:
        json_path = os.path.join(args.output_dir, f"{domain}_topics.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
            total_kws = sum(d["size"] for d in data.values())
            unique_names = len(set(d["name"] for d in data.values()))
            logger.info(f"\n{domain.upper()}:")
            logger.info(f"  Topics: {len(data)}")
            logger.info(f"  Unique names: {unique_names}")
            logger.info(f"  Total keywords: {total_kws}")

    logger.info(f"\nOutput files in {args.output_dir}/:")
    logger.info("  [domain]_topics.json         full topic data (name, keywords, counts)")
    logger.info("  [domain]_topic_names.json    topic_id -> name mapping")
    logger.info("  [domain]_topic_summary.csv   summary with top keywords per topic")


if __name__ == "__main__":
    main()
