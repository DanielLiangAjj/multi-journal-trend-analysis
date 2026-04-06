"""
Step 1: Keyword Categorization into Methodology vs Health Domain

Follows the paper's methodology:
  1. Download MeSH terms from the top two levels of the MeSH hierarchy
  2. Annotate MeSH terms as 'methodology' or 'health' using GPT (paper used 2 experts)
  3. Embed both annotated MeSH terms and author keywords with BiomedBERT
  4. For each keyword, find the most similar annotated MeSH terms by cosine similarity
  5. Use GPT-5 Nano with few-shot prompting to classify each keyword

Usage:
    python categorize_keywords.py \
        --input data/pubmed_all_journals_2011_2025_w_keywords.csv \
        --output data/keywords_categorized.csv \
        --openai-api-key sk-... \
        [--mesh-cache data/mesh_terms_annotated.json] \
        [--embedding-cache data/keyword_embeddings.pkl] \
        [--batch-size 20] \
        [--model gpt-5-nano]

Requirements:
    pip install openai torch transformers scikit-learn pandas tqdm
"""

import argparse
import csv
import json
import logging
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from openai import OpenAI
from pydantic import BaseModel
from typing import List

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BIOMEDBERT_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

# MeSH tree top-level branches and their typical domain assignments.
# The paper: "Two experts independently annotated the terms with labels,
# methodologies or health domains." We replicate this with GPT annotation
# of the full set, but use these branch-level priors as guidance.
MESH_TOP_BRANCHES = {
    "A": "Anatomy",
    "B": "Organisms",
    "C": "Diseases",
    "D": "Chemicals and Drugs",
    "E": "Analytical, Diagnostic and Therapeutic Techniques, and Equipment",
    "F": "Psychiatry and Psychology",
    "G": "Phenomena and Processes",
    "H": "Disciplines and Occupations",
    "I": "Anthropology, Education, Sociology, and Social Phenomena",
    "J": "Technology, Industry, and Agriculture",
    "K": "Humanities",
    "L": "Information Science",
    "M": "Named Groups",
    "N": "Health Care",
    "V": "Publication Characteristics",
    "Z": "Geographicals",
}

MESH_DOWNLOAD_URL = "https://nlmpubs.nlm.nih.gov/projects/mesh/2024/meshtrees/mtrees2024.bin"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models for structured GPT output
# ---------------------------------------------------------------------------

class MeSHAnnotation(BaseModel):
    term: str
    label: str  # "methodology", "health", or "both"


class KeywordClassification(BaseModel):
    keyword: str
    label: str  # "methodology", "health", "both", or "neither"


class BatchKeywordClassification(BaseModel):
    classifications: List[KeywordClassification]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Categorize keywords into methodology vs health domain."
    )
    parser.add_argument(
        "--input", required=True,
        help="Input CSV (output of collect_pubmed_data.py)",
    )
    parser.add_argument(
        "--output", default="data/keywords_categorized.csv",
        help="Output CSV with keyword categorizations",
    )
    parser.add_argument(
        "--openai-api-key", required=True,
        help="OpenAI API key",
    )
    parser.add_argument(
        "--model", default="gpt-5-nano",
        help="OpenAI model to use (default: gpt-5-nano)",
    )
    parser.add_argument(
        "--mesh-cache", default="data/mesh_terms_annotated.json",
        help="Path to cache annotated MeSH terms",
    )
    parser.add_argument(
        "--embedding-cache", default="data/keyword_embeddings.pkl",
        help="Path to cache BiomedBERT embeddings",
    )
    parser.add_argument(
        "--batch-size", type=int, default=20,
        help="Number of keywords to classify per GPT call",
    )
    parser.add_argument(
        "--k-examples", type=int, default=5,
        help="Number of most-similar MeSH exemplars for few-shot prompting",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Step 1a: Download and parse MeSH tree (top 2 levels)
# ---------------------------------------------------------------------------

def download_mesh_tree(cache_path="data/mesh_tree_raw.bin"):
    """Download MeSH tree file from NLM if not cached."""
    if os.path.exists(cache_path):
        logger.info(f"Loading cached MeSH tree from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    logger.info(f"Downloading MeSH tree from {MESH_DOWNLOAD_URL}")
    import requests
    resp = requests.get(MESH_DOWNLOAD_URL, timeout=60)
    resp.raise_for_status()

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(resp.text)
    return resp.text


def parse_mesh_top2_levels(mesh_text):
    """
    Parse MeSH tree file and extract terms from the top 2 levels.
    MeSH tree format: TermName;TreeNumber (e.g., "Body Regions;A01",
    "Anatomic Landmarks;A01.111", "Algorithms;L01.224.050")
    Level 1 = 0 dots (e.g., "A01"), Level 2 = 1 dot (e.g., "A01.111").
    """
    terms = {}  # term_name -> set of tree_numbers

    for line in mesh_text.strip().split("\n"):
        line = line.strip()
        if not line or ";" not in line:
            continue
        parts = line.split(";")
        term_name = parts[0].strip()
        tree_number = parts[1].strip()

        # Count depth by number of dots in tree number
        # Level 1: 0 dots (e.g., "A01"), Level 2: 1 dot (e.g., "A01.111")
        dot_count = tree_number.count(".")
        if dot_count <= 1:
            if term_name not in terms:
                terms[term_name] = set()
            terms[term_name].add(tree_number)

    logger.info(f"Extracted {len(terms)} MeSH terms from top 2 levels")
    return terms


# ---------------------------------------------------------------------------
# Step 1b: Annotate MeSH terms as methodology/health using GPT
# ---------------------------------------------------------------------------

def annotate_mesh_terms(mesh_terms, client, model, cache_path):
    """
    Annotate each MeSH term as 'methodology' or 'health' using GPT.
    Paper: "Two experts independently annotated the terms."
    We use GPT as a scalable proxy with clear instructions.
    """
    if os.path.exists(cache_path):
        logger.info(f"Loading cached MeSH annotations from {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)

    logger.info(f"Annotating {len(mesh_terms)} MeSH terms with GPT...")
    term_list = sorted(mesh_terms.keys())
    annotations = {}

    # Process in batches of 50
    batch_size = 50
    for i in tqdm(range(0, len(term_list), batch_size), desc="Annotating MeSH terms"):
        batch = term_list[i:i + batch_size]
        batch_str = "\n".join(f"- {t}" for t in batch)

        prompt = f"""You are an expert in biomedical informatics. Classify each of the following MeSH (Medical Subject Headings) terms into one of two categories:

- "methodology": The term describes a computational method, algorithm, technique, tool, data structure, information system, or analytical approach. Examples: "Algorithms", "Natural Language Processing", "Machine Learning", "Information Storage and Retrieval", "Decision Support Systems", "Software", "Databases".

- "health": The term describes a disease, condition, anatomical structure, organism, drug, clinical concept, patient population, or health-related domain. Examples: "Neurodegenerative Diseases", "Diabetes Mellitus", "Heart", "Neoplasms", "Pharmaceutical Preparations", "Mental Health".

- "both": If the term genuinely belongs to both categories (rare).

For each term, return ONLY the label. Here are the terms:

{batch_str}

Return your answer as a JSON array of objects with "term" and "label" fields."""

        for attempt in range(3):
            try:
                response = client.responses.create(
                    model=model,
                    input=[{"role": "user", "content": prompt}],
                )
                text = response.output_text.strip()
                # Extract JSON from response
                if "```" in text:
                    text = text.split("```json")[-1].split("```")[0].strip()
                    if not text:
                        text = response.output_text.split("```")[1].strip()

                parsed = json.loads(text)
                for item in parsed:
                    term = item["term"]
                    label = item["label"].lower().strip()
                    if label in ("methodology", "health", "both"):
                        annotations[term] = label
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for MeSH batch: {e}")
                time.sleep(2)

    # Save cache
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(annotations, f, indent=2)

    method_count = sum(1 for v in annotations.values() if v == "methodology")
    health_count = sum(1 for v in annotations.values() if v == "health")
    both_count = sum(1 for v in annotations.values() if v == "both")
    logger.info(
        f"MeSH annotation complete: {method_count} methodology, "
        f"{health_count} health, {both_count} both"
    )
    return annotations


# ---------------------------------------------------------------------------
# Step 1c: BiomedBERT embeddings
# ---------------------------------------------------------------------------

def load_biomedbert():
    """Load BiomedBERT tokenizer and model."""
    logger.info(f"Loading BiomedBERT model: {BIOMEDBERT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BIOMEDBERT_MODEL)
    model = AutoModel.from_pretrained(BIOMEDBERT_MODEL)
    model.eval()
    return tokenizer, model


def embed_texts(texts, tokenizer, model, batch_size=64):
    """
    Generate BiomedBERT CLS embeddings for a list of texts.
    Faithful to the author's approach in taxonomy_util.py.
    """
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True,
            padding=True, max_length=512,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        # CLS token embedding (same as author's approach)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        all_embeddings.append(cls_embeddings)

    return np.vstack(all_embeddings)


def load_or_compute_embeddings(texts, text_labels, tokenizer, model, cache_path):
    """Load cached embeddings or compute fresh ones."""
    if os.path.exists(cache_path):
        logger.info(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        # Check if cache matches current texts
        if cached.get("labels") == text_labels:
            return cached["embeddings"]
        logger.info("Cache mismatch, recomputing embeddings...")

    embeddings = embed_texts(texts, tokenizer, model)

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"labels": text_labels, "embeddings": embeddings}, f)

    return embeddings


# ---------------------------------------------------------------------------
# Step 1d: Few-shot keyword classification with GPT
# ---------------------------------------------------------------------------

def find_similar_mesh_exemplars(keyword_embedding, mesh_embeddings, mesh_terms_list,
                                 mesh_annotations, k=5):
    """
    For a keyword, find the k most similar annotated MeSH terms.
    Returns list of (term, label, similarity_score).
    """
    sims = cosine_similarity(keyword_embedding.reshape(1, -1), mesh_embeddings)[0]
    top_k_idx = np.argsort(sims)[::-1][:k]

    exemplars = []
    for idx in top_k_idx:
        term = mesh_terms_list[idx]
        label = mesh_annotations.get(term, "health")
        score = sims[idx]
        exemplars.append((term, label, float(score)))

    return exemplars


def classify_keywords_batch(keywords, keyword_embeddings, mesh_embeddings,
                            mesh_terms_list, mesh_annotations, client, model,
                            k_examples=5, batch_size=20):
    """
    Classify a batch of keywords using few-shot prompting with similar MeSH exemplars.
    Faithful to the paper: "keywords were systematically classified into these two
    categories using GPT-4o with few-shot prompting, leveraging the most similar
    annotated terms as examples."
    """
    results = {}

    for i in tqdm(range(0, len(keywords), batch_size), desc="Classifying keywords"):
        batch_kws = keywords[i:i + batch_size]
        batch_embs = keyword_embeddings[i:i + batch_size]

        # Build few-shot examples for each keyword in the batch
        keyword_examples = []
        for j, kw in enumerate(batch_kws):
            exemplars = find_similar_mesh_exemplars(
                batch_embs[j], mesh_embeddings, mesh_terms_list,
                mesh_annotations, k=k_examples,
            )
            examples_str = ", ".join(
                f'"{term}" -> {label}' for term, label, _ in exemplars
            )
            keyword_examples.append((kw, examples_str))

        # Build the prompt
        examples_block = "\n".join(
            f'Keyword: "{kw}"\nSimilar MeSH terms and their labels: {ex}'
            for kw, ex in keyword_examples
        )

        prompt = f"""You are an expert in biomedical informatics. Classify each keyword into one of four categories based on the similar MeSH term examples provided:

- "methodology": Indicates a computational method, algorithm, technique, tool, software, data structure, model architecture, evaluation metric, or analytical approach.
- "health": Indicates a disease, condition, anatomical structure, organism, drug, clinical concept, patient population, health system, or health-related domain.
- "both": The keyword genuinely belongs to both methodology and health domains (e.g., "allergen detection" combines a method with a health concept).
- "neither": The keyword does not fit either category (e.g., geographic names, generic terms like "California").

Here are the keywords to classify, each with their most similar annotated MeSH terms as reference:

{examples_block}

Classify each keyword. Return a JSON object with a "classifications" array, where each entry has "keyword" and "label" fields. Example:
{{"classifications": [{{"keyword": "deep learning", "label": "methodology"}}, {{"keyword": "lung cancer", "label": "health"}}]}}"""

        for attempt in range(3):
            try:
                response = client.responses.parse(
                    model=model,
                    input=[{"role": "user", "content": prompt}],
                    text_format=BatchKeywordClassification,
                )
                parsed = response.output_parsed
                for item in parsed.classifications:
                    kw = item.keyword.strip()
                    label = item.label.strip().lower()
                    if label in ("methodology", "health", "both", "neither"):
                        results[kw] = label
                    else:
                        results[kw] = "neither"
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for batch at index {i}: {e}")
                time.sleep(2)

        # Mark any keywords that weren't in the response
        for kw in batch_kws:
            if kw not in results:
                results[kw] = "neither"

    return results


# ---------------------------------------------------------------------------
# Step 1e: Extract unique keywords from collected data
# ---------------------------------------------------------------------------

def extract_unique_keywords(input_csv):
    """
    Extract all unique keywords from the collected data.
    Paper: "We preprocessed the keywords by removing capitalization,
    converting them into a comma-separated list, and removing duplicates."
    """
    logger.info(f"Loading keywords from {input_csv}")
    df = pd.read_csv(input_csv, quoting=csv.QUOTE_ALL)

    all_keywords = []
    for _, row in df.iterrows():
        kw_str = row.get("keywords", "")
        if pd.isna(kw_str) or not kw_str:
            continue
        # Split by semicolon (our CSV format) or comma
        for kw in str(kw_str).split(";"):
            kw = kw.strip().lower()
            if kw:
                all_keywords.append(kw)

    keyword_counts = Counter(all_keywords)
    unique_keywords = sorted(keyword_counts.keys())

    logger.info(
        f"Extracted {len(unique_keywords)} unique keywords "
        f"from {len(all_keywords)} total keyword occurrences"
    )
    return unique_keywords, keyword_counts


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def load_classification_progress(progress_path):
    """Load partially completed classifications."""
    if os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            return json.load(f)
    return {}


def save_classification_progress(progress_path, classifications):
    """Save classification progress."""
    with open(progress_path, "w") as f:
        json.dump(classifications, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Initialize OpenAI client
    client = OpenAI(api_key=args.openai_api_key)

    # --- Step 1a: Download and parse MeSH terms ---
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    mesh_text = download_mesh_tree(cache_path="data/mesh_tree_raw.bin")
    mesh_terms = parse_mesh_top2_levels(mesh_text)

    # --- Step 1b: Annotate MeSH terms ---
    mesh_annotations = annotate_mesh_terms(
        mesh_terms, client, args.model, args.mesh_cache,
    )

    # Filter to only annotated terms
    annotated_mesh_terms = {
        t: mesh_annotations[t]
        for t in mesh_annotations
        if t in mesh_terms
    }
    mesh_terms_list = sorted(annotated_mesh_terms.keys())
    logger.info(f"Using {len(mesh_terms_list)} annotated MeSH terms as exemplars")

    # --- Step 1c: Extract unique keywords ---
    unique_keywords, keyword_counts = extract_unique_keywords(args.input)

    # --- Step 1d: Compute BiomedBERT embeddings ---
    tokenizer, model = load_biomedbert()

    # Embed MeSH terms
    mesh_embeddings = load_or_compute_embeddings(
        mesh_terms_list, mesh_terms_list, tokenizer, model,
        cache_path="data/mesh_embeddings.pkl",
    )

    # Embed keywords
    keyword_embeddings = load_or_compute_embeddings(
        unique_keywords, unique_keywords, tokenizer, model,
        cache_path=args.embedding_cache,
    )

    # --- Step 1e: Classify keywords with few-shot prompting ---
    progress_path = args.output + ".progress.json"
    existing_classifications = load_classification_progress(progress_path)

    # Filter out already-classified keywords
    remaining_keywords = [kw for kw in unique_keywords if kw not in existing_classifications]
    remaining_indices = [unique_keywords.index(kw) for kw in remaining_keywords]
    remaining_embeddings = keyword_embeddings[remaining_indices] if remaining_indices else np.array([])

    logger.info(
        f"Classifying {len(remaining_keywords)} keywords "
        f"({len(existing_classifications)} already done)"
    )

    if len(remaining_keywords) > 0:
        new_classifications = classify_keywords_batch(
            remaining_keywords, remaining_embeddings,
            mesh_embeddings, mesh_terms_list, mesh_annotations,
            client, args.model,
            k_examples=args.k_examples, batch_size=args.batch_size,
        )
        existing_classifications.update(new_classifications)
        save_classification_progress(progress_path, existing_classifications)

    # --- Step 1f: Write output ---
    logger.info(f"Writing categorized keywords to {args.output}")

    # Count statistics
    label_counts = Counter(existing_classifications.values())
    logger.info(
        f"Classification results: "
        f"{label_counts.get('methodology', 0)} methodology, "
        f"{label_counts.get('health', 0)} health, "
        f"{label_counts.get('both', 0)} both, "
        f"{label_counts.get('neither', 0)} neither"
    )

    # Write CSV
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["keyword", "label", "count"])
        for kw in sorted(existing_classifications.keys()):
            writer.writerow([kw, existing_classifications[kw], keyword_counts.get(kw, 0)])

    logger.info(f"Done! Output: {args.output}")

    # Summary
    logger.info("\n--- Summary ---")
    logger.info(f"Total unique keywords: {len(existing_classifications)}")
    for label in ["methodology", "health", "both", "neither"]:
        count = label_counts.get(label, 0)
        pct = count / len(existing_classifications) * 100 if existing_classifications else 0
        logger.info(f"  {label}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
