"""
Step 4: Topic Deduplication and Hierarchy Construction

Following the paper's methodology:

Phase 1 - Stemming Merge (local, no GPT):
  1. Merge topics with the same Porter-stemmed name
  2. Union their keywords when merging

Phase 2 - Embedding Similarity (local, no GPT):
  3. Embed unique topic names with SentenceTransformer (all-mpnet-base-v2)
  4. Compute pairwise cosine similarity matrix

Phase 3 - Threshold Discovery (GPT):
  5. For each topic, binary search for the similarity percentile threshold
     where topics transition from "related" to "not related"

Phase 4 - Relationship Classification (GPT):
  6. For topic pairs above threshold, classify: Superset/Subset/Equal/NoOverlap

Phase 5 - Merge Equals & Build Hierarchy:
  7. Merge "Equal" topics (deduplication)
  8. Extract subset/superset edges (hierarchy)
  9. Double-check subsumption relations iteratively (GPT verification)
  10. Remove cycles from directed graph
  11. Transitive reduction (keep only immediate parent-child edges)
  12. Write final hierarchy tree

Usage:
    python dedup_and_hierarchy.py \\
        --topics-dir data/topics \\
        --output-dir data/hierarchy \\
        --openai-api-key sk-... \\
        [--model gpt-5-nano] \\
        [--threshold-precision 0.005] \\
        [--verification-rounds 4]

Requirements:
    pip install openai pandas numpy scikit-learn tqdm sentence-transformers nltk
"""

import argparse
import csv
import json
import logging
import os
import pickle
import sys
import time
from collections import Counter, defaultdict, deque

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from openai import OpenAI

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
        description="Step 4: Topic deduplication and hierarchy construction."
    )
    parser.add_argument(
        "--topics-dir", default="data/topics",
        help="Directory with Step 3 output (topic JSON files)",
    )
    parser.add_argument(
        "--output-dir", default="data/hierarchy",
        help="Output directory for hierarchy results",
    )
    parser.add_argument(
        "--openai-api-key", required=True,
        help="OpenAI API key",
    )
    parser.add_argument(
        "--model", default="gpt-5-nano",
        help="OpenAI model for GPT calls (default: gpt-5-nano)",
    )
    parser.add_argument(
        "--threshold-precision", type=float, default=0.005,
        help="Binary search precision for threshold discovery (default: 0.005)",
    )
    parser.add_argument(
        "--verification-rounds", type=int, default=4,
        help="Number of iterative edge verification rounds (default: 4)",
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
# Phase 1: Stemming Merge
# ---------------------------------------------------------------------------

def load_topics(topics_dir, domain):
    """Load topic data from Step 3."""
    json_path = os.path.join(topics_dir, f"{domain}_topics.json")
    logger.info(f"Loading topics from {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        topics = json.load(f)

    names_path = os.path.join(topics_dir, f"{domain}_topic_names.json")
    with open(names_path, "r", encoding="utf-8") as f:
        topic_names = json.load(f)

    logger.info(f"  Loaded {len(topics)} topics, {len(set(topic_names.values()))} unique names")
    return topics, topic_names


def stemming_merge(topic_names):
    """
    Merge topics with the same Porter-stemmed name.
    Paper: combine_topic_w_same_name function.

    Returns:
      stemmed_groups: {stemmed_name: [list of topic_ids]}
      stem_to_original: {stemmed_name: original_name (first occurrence)}
    """
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()

    # Group topic IDs by their stemmed name
    stemmed_groups = defaultdict(list)
    for tid, name in topic_names.items():
        stemmed = " ".join(
            stemmer.stem(word.lower()) for word in name.split()
        )
        stemmed_groups[stemmed].append(tid)

    # Map stemmed name -> first original name (canonical form)
    stem_to_original = {}
    for stemmed, tids in stemmed_groups.items():
        # Use the first topic's name as the canonical name
        stem_to_original[stemmed] = topic_names[tids[0]]

    n_before = len(set(topic_names.values()))
    n_after = len(stemmed_groups)
    logger.info(
        f"  Stemming merge: {n_before} unique names -> "
        f"{n_after} unique stemmed names"
    )

    return stemmed_groups, stem_to_original


def apply_stemming_merge(topics_data, topic_names, stemmed_groups, stem_to_original):
    """
    Merge topics that share the same stemmed name.
    Returns new topic dict and name mapping.
    """
    merged_topics = {}
    merged_names = {}

    for stemmed, tids in stemmed_groups.items():
        # Use the ORIGINAL canonical name as the key (not stemmed),
        # so it matches the names used in similarity/relation phases
        canonical_name = stem_to_original[stemmed]

        # Union all keywords from constituent topics
        all_keywords = []
        all_counts = {}
        for tid in tids:
            if tid in topics_data:
                kws = topics_data[tid].get("keywords", [])
                all_keywords.extend(kws)
                kw_counts = topics_data[tid].get("keyword_counts", {})
                for kw, c in kw_counts.items():
                    all_counts[kw] = max(all_counts.get(kw, 0), c)

        if canonical_name in merged_topics:
            # Name collision from different stemmed groups — merge into existing
            merged_topics[canonical_name]["keywords"].extend(all_keywords)
            for kw, c in all_counts.items():
                existing = merged_topics[canonical_name]["keyword_counts"]
                existing[kw] = max(existing.get(kw, 0), c)
            merged_topics[canonical_name]["constituent_topic_ids"].extend(tids)
        else:
            merged_topics[canonical_name] = {
                "name": canonical_name,
                "keywords": all_keywords,
                "keyword_counts": all_counts,
                "constituent_topic_ids": tids,
            }
        merged_names[canonical_name] = canonical_name

    # Deduplicate keywords within each merged topic
    for tid, data in merged_topics.items():
        data["keywords"] = list(set(data["keywords"]))
        data["size"] = len(data["keywords"])

    logger.info(f"  After stemming merge: {len(merged_topics)} topics")
    return merged_topics, merged_names


# ---------------------------------------------------------------------------
# Phase 2: Embedding Similarity
# ---------------------------------------------------------------------------

def compute_pairwise_topic_similarity(topic_names_list, embedder):
    """
    Compute pairwise cosine similarity between all unique topic names.
    Paper: compute_pairwise_cosine_similarity function.
    """
    logger.info(f"  Computing embeddings for {len(topic_names_list)} topic names...")
    embeddings = embedder.encode(topic_names_list, convert_to_numpy=True,
                                 batch_size=256)

    logger.info("  Computing pairwise cosine similarity...")
    sim_matrix = cosine_similarity(embeddings)

    # Build similarity dict (upper triangle only, i < j)
    similarity_dict = {}
    n = len(topic_names_list)
    for i in range(n):
        for j in range(i + 1, n):
            similarity_dict[(topic_names_list[i], topic_names_list[j])] = float(
                sim_matrix[i, j]
            )

    logger.info(f"  Computed {len(similarity_dict)} pairwise similarities")
    return similarity_dict, embeddings


# ---------------------------------------------------------------------------
# Phase 3: Threshold Discovery
# ---------------------------------------------------------------------------

def have_relationship(topic1, topic2, client, model):
    """
    Ask GPT if two topics are related (any relation other than NoOverlap).
    Paper: have_relationship function.
    """
    prompt = f"""Determine the relationship between Topic A and Topic B. Use the following relationship labels:
Superset: Topic B is a subcategory of Topic A, but Topic A is not a subcategory of Topic B.
Subset: Topic A is a subcategory of Topic B, but Topic B is not a subcategory of Topic A.
Equal: Topic A is a Topic B and Topic B is a Topic A.
NoOverlap: The two topics are conceptually distinct.

Here are the topics:
Topic A: {topic1}
Topic B: {topic2}

Return the label only."""

    response = gpt_text_response(client, model, prompt)
    return response.strip(".").lower() != "nooverlap"


def find_threshold_per_topic(similarity_dict, client, model, precision=0.005,
                             progress_path=None):
    """
    For each topic, use binary search to find the similarity percentile
    threshold where topics transition from "related" to "not related".
    Paper: find_threshold_by_pct function.
    """
    # Load progress
    thresholds = {}
    if progress_path and os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            thresholds = json.load(f)
        logger.info(f"  Resuming threshold discovery: {len(thresholds)} done")

    # Group by topic1 only (upper triangle — faithful to paper)
    topic_groups = defaultdict(list)
    for (t1, t2), score in similarity_dict.items():
        topic_groups[t1].append((t2, score))

    remaining = [t for t in topic_groups if t not in thresholds]
    logger.info(
        f"  Finding thresholds for {len(remaining)} topics "
        f"(precision={precision})..."
    )

    for topic1 in tqdm(remaining, desc="Threshold discovery"):
        sorted_pairs = sorted(topic_groups[topic1], key=lambda x: x[1])
        sorted_names = [t2 for t2, _ in sorted_pairs]
        n = len(sorted_names)

        low, high = 0.0, 1.0
        best_pct = 0.0
        prev_topic2 = None

        while high - low > precision:
            mid_pct = (low + high) / 2
            k = max(1, int(mid_pct * n))
            selected_topic2 = sorted_names[k - 1]

            related = have_relationship(topic1, selected_topic2, client, model)
            if related:
                best_pct = mid_pct
                high = mid_pct
            else:
                low = mid_pct

            if selected_topic2 == prev_topic2:
                break
            prev_topic2 = selected_topic2

        thresholds[topic1] = best_pct

        # Save progress periodically
        if progress_path and len(thresholds) % 10 == 0:
            with open(progress_path, "w") as f:
                json.dump(thresholds, f)

    if progress_path:
        with open(progress_path, "w") as f:
            json.dump(thresholds, f)

    # Stats
    nonzero = sum(1 for v in thresholds.values() if v > 0)
    logger.info(
        f"  Threshold discovery done: {nonzero} topics have relationships, "
        f"{len(thresholds) - nonzero} isolated"
    )
    return thresholds


# ---------------------------------------------------------------------------
# Phase 4: Relationship Classification
# ---------------------------------------------------------------------------

def check_relationship(topic1, topic2, client, model):
    """
    Classify the relationship between two topics.
    Paper: check_relationship function.
    Returns: "superset", "subset", "equal", or "nooverlap"
    """
    prompt = f"""Determine the relationship between Topic A and Topic B. Use the following relationship labels:
Superset: Topic B is a subcategory of Topic A, but Topic A is not a subcategory of Topic B.
Subset: Topic A is a subcategory of Topic B, but Topic B is not a subcategory of Topic A.
Equal: Topic A is a Topic B and Topic B is a Topic A.
NoOverlap: The two topics are conceptually distinct.

Here are the topics:
Topic A: {topic1}
Topic B: {topic2}

Return the label only."""

    response = gpt_text_response(client, model, prompt)
    return response.strip(".").lower()


def classify_topic_relations(similarity_dict, thresholds, client, model,
                             progress_path=None):
    """
    For topic pairs above the similarity threshold, classify their relationship.
    Paper: check_topic_relation function.
    """
    # Load progress
    relations = {}
    if progress_path and os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            saved = json.load(f)
        # Convert string keys back to tuples
        relations = {tuple(k.split("|||")): v for k, v in saved.items()}
        logger.info(f"  Resuming relation classification: {len(relations)} done")

    # Group by topic1 only (upper triangle — faithful to paper)
    topic_groups = defaultdict(list)
    for (t1, t2), score in similarity_dict.items():
        topic_groups[t1].append((t2, score))

    # Find pairs to check
    pairs_to_check = []
    for topic1, pairs in topic_groups.items():
        if thresholds.get(topic1, 0) == 0:
            continue
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        n = len(sorted_pairs)
        k = max(1, int((1 - thresholds[topic1]) * n))
        selected = sorted_pairs[:k]
        for topic2, sim in selected:
            pair_key = tuple(sorted([topic1, topic2]))
            if pair_key not in relations:
                pairs_to_check.append((topic1, topic2))

    # Deduplicate
    seen = set()
    unique_pairs = []
    for t1, t2 in pairs_to_check:
        key = tuple(sorted([t1, t2]))
        if key not in seen and key not in relations:
            seen.add(key)
            unique_pairs.append((t1, t2))

    logger.info(f"  Classifying {len(unique_pairs)} topic pairs...")

    for i, (t1, t2) in enumerate(tqdm(unique_pairs, desc="Classifying relations")):
        relation = check_relationship(t1, t2, client, model)
        relations[(t1, t2)] = relation

        # Save progress periodically
        if progress_path and (i + 1) % 20 == 0:
            _save_relations(progress_path, relations)

    if progress_path:
        _save_relations(progress_path, relations)

    # Stats
    rel_counts = Counter(relations.values())
    logger.info(f"  Relation counts: {dict(rel_counts)}")
    return relations


def _save_relations(path, relations):
    """Save relations dict with string keys for JSON compatibility."""
    serializable = {"|||".join(k): v for k, v in relations.items()}
    with open(path, "w") as f:
        json.dump(serializable, f)


# ---------------------------------------------------------------------------
# Phase 5: Merge Equals & Build Hierarchy
# ---------------------------------------------------------------------------

def construct_equal_topic_map(relations):
    """
    Build a mapping that merges "Equal" topics to a canonical name.
    Paper: construct_equal_topic_map function.
    """
    topic_name_map = {}
    equal_pairs = sorted([
        tuple(sorted(pair, reverse=True))
        for pair, rel in relations.items() if rel == "equal"
    ])

    for t1, t2 in equal_pairs:
        if t2 in topic_name_map:
            topic_name_map[t1] = topic_name_map[t2]
        else:
            topic_name_map[t1] = t2

    logger.info(f"  Found {len(equal_pairs)} equal pairs -> {len(topic_name_map)} merges")
    return topic_name_map


def merge_equal_topics_in_data(merged_topics, equal_map):
    """
    Merge topics classified as "Equal" — combine their keywords.
    """
    if not equal_map:
        return merged_topics

    # Build reverse map: canonical -> list of topics to merge into it
    canonical_to_sources = defaultdict(list)
    for source, target in equal_map.items():
        canonical_to_sources[target].append(source)

    final_topics = {}
    merged_away = set(equal_map.keys())

    for tid, data in merged_topics.items():
        if tid in merged_away:
            continue

        # Start with this topic's data
        all_keywords = list(data.get("keywords", []))
        all_counts = dict(data.get("keyword_counts", {}))
        constituent_ids = list(data.get("constituent_topic_ids", [tid]))

        # Merge in any topics mapped to this one
        for source_tid in canonical_to_sources.get(tid, []):
            if source_tid in merged_topics:
                src = merged_topics[source_tid]
                all_keywords.extend(src.get("keywords", []))
                for kw, c in src.get("keyword_counts", {}).items():
                    all_counts[kw] = max(all_counts.get(kw, 0), c)
                constituent_ids.extend(src.get("constituent_topic_ids", [source_tid]))

        final_topics[tid] = {
            "name": data["name"],
            "keywords": list(set(all_keywords)),
            "keyword_counts": all_counts,
            "size": len(set(all_keywords)),
            "constituent_topic_ids": constituent_ids,
        }

    logger.info(
        f"  After equal merge: {len(final_topics)} topics "
        f"(merged {len(merged_away)} duplicates)"
    )
    return final_topics


def merge_equal_in_relations(relations, equal_map):
    """
    Update relation keys to use canonical topic names after equal merging.
    Paper: merge_equal_topics function.
    """
    new_relations = {}
    for (t1, t2), rel in relations.items():
        t1 = equal_map.get(t1, t1)
        t2 = equal_map.get(t2, t2)
        if t1 != t2:
            new_relations[(t1, t2)] = rel
    return new_relations


def extract_subset_edges(relations):
    """
    Extract parent-child edges from subset/superset relations.
    Paper: extract_subset_relation function.
    Returns list of (child, parent) tuples.
    """
    edges = []
    for (t1, t2), rel in relations.items():
        if rel == "superset":
            # t1 is superset of t2 -> t2 is child of t1
            edges.append((t2, t1))
        elif rel == "subset":
            # t1 is subset of t2 -> t1 is child of t2
            edges.append((t1, t2))
    logger.info(f"  Extracted {len(edges)} subset edges")
    return edges


def double_check_edges(edges, client, model, rounds=4, progress_path=None):
    """
    Iteratively verify subset edges by asking GPT in alternating directions.
    Paper: double_check_subsumption_relation + select_consistent_edge_subset.
    """
    current_edges = list(edges)
    logger.info(f"  Starting edge verification: {len(current_edges)} edges, {rounds} rounds")

    for round_num in range(rounds):
        reverse = (round_num % 2 == 0)
        direction = "reverse" if reverse else "forward"
        logger.info(f"  Round {round_num + 1}/{rounds} ({direction}): {len(current_edges)} edges")

        verified = []
        rel_counts = Counter()

        # Load progress for this round
        round_progress_path = f"{progress_path}_round{round_num + 1}" if progress_path else None
        done = {}
        if round_progress_path and os.path.exists(round_progress_path):
            with open(round_progress_path, "r") as f:
                done = json.load(f)

        for i, (child, parent) in enumerate(tqdm(
            current_edges, desc=f"Verify round {round_num + 1}"
        )):
            edge_key = f"{child}|||{parent}"
            if edge_key in done:
                relation = done[edge_key]
            else:
                if reverse:
                    relation = check_relationship(parent, child, client, model)
                else:
                    relation = check_relationship(child, parent, client, model)
                done[edge_key] = relation

                if round_progress_path and (i + 1) % 20 == 0:
                    with open(round_progress_path, "w") as f:
                        json.dump(done, f)

            rel_counts[relation] += 1
            expected = "superset" if reverse else "subset"
            if relation == expected:
                verified.append((child, parent))

        if round_progress_path:
            with open(round_progress_path, "w") as f:
                json.dump(done, f)

        logger.info(f"    Relations: {dict(rel_counts)}")
        logger.info(
            f"    Kept {len(verified)}/{len(current_edges)} edges "
            f"(removed {len(current_edges) - len(verified)})"
        )
        current_edges = verified

        if len(current_edges) == 0:
            break

    logger.info(f"  Edge verification done: {len(current_edges)} verified edges")
    return current_edges


# ---------------------------------------------------------------------------
# Cycle removal and transitive reduction
# ---------------------------------------------------------------------------

def find_first_cycle(edges):
    """
    Find the first cycle in a directed graph.
    Paper: find_first_cycle function.
    """
    import random
    graph = defaultdict(list)
    nodes = set()
    for child, parent in edges:
        graph[parent].append(child)
        nodes.add(child)
        nodes.add(parent)

    if not edges:
        return None

    random.seed(42)
    visited = set()
    rec_stack = []

    def dfs(node):
        visited.add(node)
        rec_stack.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                result = dfs(neighbor)
                if result:
                    return result
            elif neighbor in rec_stack:
                cycle_start = rec_stack.index(neighbor)
                return rec_stack[cycle_start:] + [neighbor]
        rec_stack.pop()
        return None

    for node in nodes:
        if node not in visited:
            cycle = dfs(node)
            if cycle:
                return cycle
    return None


def remove_cycles(edges):
    """
    Iteratively remove cycles by removing the weakest edge in each cycle.
    """
    edge_set = set(edges)
    removed = 0

    while True:
        cycle = find_first_cycle(list(edge_set))
        if cycle is None:
            break

        # Remove the last edge in the cycle (arbitrary but deterministic)
        for i in range(len(cycle) - 1):
            edge = (cycle[i + 1], cycle[i])  # (child, parent) format
            if edge in edge_set:
                edge_set.remove(edge)
                removed += 1
                break

    logger.info(f"  Removed {removed} edges to break cycles, {len(edge_set)} remain")
    return list(edge_set)


def get_immediate_edges(edges):
    """
    Transitive reduction: remove edges implied by longer paths.
    Paper: get_immediate_edges function.
    """
    graph = defaultdict(set)
    for child, parent in edges:
        graph[parent].add(child)

    def has_indirect_path(start, end, direct_edge):
        """Check if there's a path from start to end without using direct_edge."""
        queue = deque([start])
        visited = set()
        while queue:
            node = queue.popleft()
            if node == end:
                return True
            for neighbor in graph[node]:
                if (node, neighbor) == direct_edge:
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    immediate = []
    for child, parent in edges:
        # Check if there's an indirect path from parent to child
        if not has_indirect_path(parent, child, (parent, child)):
            immediate.append((child, parent))

    removed = len(edges) - len(immediate)
    logger.info(
        f"  Transitive reduction: removed {removed} redundant edges, "
        f"{len(immediate)} immediate edges remain"
    )
    return immediate


# ---------------------------------------------------------------------------
# Write hierarchy
# ---------------------------------------------------------------------------

def write_hierarchy_tree(edges, all_topic_names, domain, output_dir):
    """
    Write the topic hierarchy as an indented tree file.
    Paper: write_hierarchy_to_file function.
    """
    # Build parent -> children map
    tree = defaultdict(list)
    all_children = set()
    for child, parent in edges:
        tree[parent].append(child)
        all_children.add(child)

    # Root nodes: appear as parents but never as children
    all_parents = set(tree.keys())
    roots = sorted(all_parents - all_children)

    # Isolated nodes: not in any edge
    all_in_edges = all_children | all_parents
    all_topics = set(all_topic_names.keys()) if isinstance(all_topic_names, dict) else set(all_topic_names)
    isolated = sorted(all_topics - all_in_edges)

    # Write tree
    tree_path = os.path.join(output_dir, f"{domain}_hierarchy.txt")

    def write_subtree(f, node, level=0):
        name = all_topic_names.get(node, node) if isinstance(all_topic_names, dict) else node
        f.write("\t" * level + f"- {name}\n")
        for child in sorted(tree.get(node, [])):
            write_subtree(f, child, level + 1)

    with open(tree_path, "w", encoding="utf-8") as f:
        for root in roots:
            write_subtree(f, root)
        # Add isolated nodes at the top level
        for node in isolated:
            name = all_topic_names.get(node, node) if isinstance(all_topic_names, dict) else node
            f.write(f"- {name}\n")

    logger.info(
        f"  Hierarchy written to {tree_path}: "
        f"{len(roots)} root nodes, {len(isolated)} isolated nodes"
    )

    # Also save edges as JSON
    edges_path = os.path.join(output_dir, f"{domain}_edges.json")
    with open(edges_path, "w") as f:
        json.dump([(c, p) for c, p in edges], f, indent=2)
    logger.info(f"  Edges saved to {edges_path}")

    return tree_path


def find_root_nodes(edges):
    """Find nodes that are never children."""
    children = {child for child, parent in edges}
    parents = {parent for child, parent in edges}
    return sorted(parents - children)


def get_max_depth(edges):
    """Compute the maximum depth of the hierarchy tree."""
    graph = defaultdict(list)
    children = set()
    parents = set()
    for child, parent in edges:
        graph[parent].append(child)
        children.add(child)
        parents.add(parent)

    roots = parents - children
    if not roots:
        return 0

    def dfs(node, depth):
        if node not in graph:
            return depth
        return max(dfs(child, depth + 1) for child in graph[node])

    return max(dfs(root, 1) for root in roots)


# ---------------------------------------------------------------------------
# Save final results
# ---------------------------------------------------------------------------

def save_final_topics(final_topics, domain, output_dir):
    """Save the deduplicated topic data."""
    # Full JSON
    json_path = os.path.join(output_dir, f"{domain}_final_topics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_topics, f, indent=2, ensure_ascii=False)
    logger.info(f"  Saved {len(final_topics)} final topics to {json_path}")

    # Summary CSV
    csv_path = os.path.join(output_dir, f"{domain}_final_topic_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["topic_name", "size", "n_constituent_topics", "top_keywords"])
        for tid, data in sorted(final_topics.items(), key=lambda x: -x[1]["size"]):
            sorted_kws = sorted(
                data.get("keyword_counts", {}).items(),
                key=lambda x: -x[1],
            )[:10]
            top_str = "; ".join(f"{kw} ({c})" for kw, c in sorted_kws)
            writer.writerow([
                data["name"],
                data["size"],
                len(data.get("constituent_topic_ids", [])),
                top_str,
            ])
    logger.info(f"  Saved topic summary to {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Step 4: Topic Deduplication and Hierarchy Construction")
    logger.info("=" * 70)

    client = OpenAI(api_key=args.openai_api_key)

    # Load SentenceTransformer
    logger.info("Loading SentenceTransformer (all-mpnet-base-v2)...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-mpnet-base-v2")
    logger.info("  Loaded")

    for domain in ["methodology", "health"]:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Processing {domain.upper()} domain")
        logger.info(f"{'=' * 50}")

        # ---- Phase 1: Stemming Merge ----
        logger.info("\n--- Phase 1: Stemming Merge ---")
        topics_data, topic_names = load_topics(args.topics_dir, domain)
        stemmed_groups, stem_to_original = stemming_merge(topic_names)
        merged_topics, merged_names = apply_stemming_merge(
            topics_data, topic_names, stemmed_groups, stem_to_original,
        )

        # ---- Phase 2: Embedding Similarity ----
        logger.info("\n--- Phase 2: Embedding Similarity ---")
        unique_names = sorted(set(merged_names.values()))
        logger.info(f"  {len(unique_names)} unique topic names to compare")

        sim_dict, embeddings = compute_pairwise_topic_similarity(
            unique_names, embedder,
        )

        # ---- Phase 3: Threshold Discovery ----
        logger.info("\n--- Phase 3: Threshold Discovery (GPT) ---")
        threshold_progress = os.path.join(
            args.output_dir, f"{domain}_threshold_progress.json",
        )
        thresholds = find_threshold_per_topic(
            sim_dict, client, args.model, args.threshold_precision,
            threshold_progress,
        )

        # ---- Phase 4: Relationship Classification ----
        logger.info("\n--- Phase 4: Relationship Classification (GPT) ---")
        relation_progress = os.path.join(
            args.output_dir, f"{domain}_relation_progress.json",
        )
        relations = classify_topic_relations(
            sim_dict, thresholds, client, args.model, relation_progress,
        )

        # ---- Phase 5: Merge Equals ----
        logger.info("\n--- Phase 5: Merge Equals ---")
        equal_map = construct_equal_topic_map(relations)
        final_topics = merge_equal_topics_in_data(merged_topics, equal_map)

        # Save deduplicated topics
        save_final_topics(final_topics, domain, args.output_dir)

        # ---- Phase 6: Build Hierarchy ----
        logger.info("\n--- Phase 6: Build Hierarchy ---")
        merged_relations = merge_equal_in_relations(relations, equal_map)
        subset_edges = extract_subset_edges(merged_relations)

        if subset_edges:
            # Double-check edges
            edge_progress = os.path.join(
                args.output_dir, f"{domain}_edge_verification",
            )
            verified_edges = double_check_edges(
                subset_edges, client, args.model,
                rounds=args.verification_rounds,
                progress_path=edge_progress,
            )

            # Remove cycles
            if verified_edges:
                dag_edges = remove_cycles(verified_edges)
                # Transitive reduction
                immediate_edges = get_immediate_edges(dag_edges)
            else:
                immediate_edges = []
        else:
            immediate_edges = []
            logger.info("  No subset edges found — flat topic structure")

        # Write hierarchy
        name_lookup = {tid: d["name"] for tid, d in final_topics.items()}
        write_hierarchy_tree(immediate_edges, name_lookup, domain, args.output_dir)

        # Stats
        if immediate_edges:
            depth = get_max_depth(immediate_edges)
            roots = find_root_nodes(immediate_edges)
            logger.info(f"  Hierarchy depth: {depth}, root nodes: {len(roots)}")

    # ---- Final Summary ----
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4 COMPLETE - DEDUPLICATION & HIERARCHY SUMMARY")
    logger.info("=" * 70)

    for domain in ["methodology", "health"]:
        json_path = os.path.join(args.output_dir, f"{domain}_final_topics.json")
        edges_path = os.path.join(args.output_dir, f"{domain}_edges.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
            total_kws = sum(d["size"] for d in data.values())
            logger.info(f"\n{domain.upper()}:")
            logger.info(f"  Final topics: {len(data)}")
            logger.info(f"  Total keywords: {total_kws}")
        if os.path.exists(edges_path):
            with open(edges_path, "r") as f:
                edges = json.load(f)
            logger.info(f"  Hierarchy edges: {len(edges)}")

    logger.info(f"\nOutput files in {args.output_dir}/:")
    logger.info("  [domain]_final_topics.json        deduplicated topic data")
    logger.info("  [domain]_final_topic_summary.csv   sorted topic summary")
    logger.info("  [domain]_hierarchy.txt             indented hierarchy tree")
    logger.info("  [domain]_edges.json                parent-child edge list")


if __name__ == "__main__":
    main()
