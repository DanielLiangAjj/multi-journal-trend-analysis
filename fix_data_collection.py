"""
fix_data_collection.py

Fixes Step 0 data collection issues, then the user re-runs Steps 1-6.

Fixes:
  1. Re-collects ~900 missing Bioinformatics Advances articles from PubMed
  2. For articles without PubMed keywords or MeSH (like Bioinformatics Advances),
     uses GPT-5-nano to extract author-style keywords from abstracts as a
     4th-pass fallback (consistent with the LLM-based pipeline)
  3. Adds keyword_source column for transparency

After running this script, re-run Steps 1-6 on the patched CSV for a
clean, consistent pipeline.

Usage:
    python fix_data_collection.py \\
        --articles-csv data/pubmed_all_journals_2011_2025_w_keywords.csv \\
        --output data/pubmed_all_journals_2011_2025_w_keywords_patched.csv \\
        --email YOUR_EMAIL \\
        --openai-api-key YOUR_KEY \\
        [--api-key YOUR_NCBI_KEY] \\
        [--model gpt-5-nano]
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from xml.etree import ElementTree as ET

import requests
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BATCH_SIZE = 200
MAX_RETRIES = 5

MISSING_JOURNALS = {
    "Bioinformatics Advances": "Bioinform Adv",
}

MESH_HEAVY_JOURNALS = {
    "Bioinformatics",
    "The Lancet Digital Health",
    "Database: The Journal of Biological Databases and Curation",
    "Applied Clinical Informatics",
    "Methods of Information in Medicine",
    "Journal of Medical Systems",
}

MESH_INDICATORS = {
    "humans", "male", "female", "adult", "aged", "middle aged",
    "animals", "child", "infant", "adolescent", "young adult",
    "aged, 80 and over", "child, preschool", "infant, newborn",
    "prospective studies", "retrospective studies", "cross-sectional studies",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fix Step 0 data collection issues.",
    )
    parser.add_argument("--articles-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--api-key", default=None, help="NCBI API key")
    parser.add_argument("--openai-api-key", required=True, help="OpenAI API key")
    parser.add_argument("--model", default="gpt-5-nano", help="OpenAI model")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# PubMed fetching
# ---------------------------------------------------------------------------

def rate_limit(has_key):
    time.sleep(0.11 if has_key else 0.34)


def request_retry(url, params, has_key):
    for attempt in range(MAX_RETRIES):
        try:
            rate_limit(has_key)
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 429:
                time.sleep(2 * (2 ** attempt))
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            logger.warning(f"  Request error: {e}. Retrying ({attempt + 1})...")
            time.sleep(2 * (2 ** attempt))
    return None


def _get_element_text(elem):
    return "".join(elem.itertext()).strip()


def fetch_journal_articles(journal_ta, email, api_key, start_year=2011, end_year=2025):
    """Fetch all articles for a journal from PubMed."""
    query = f'"{journal_ta}"[ta] AND {start_year}:{end_year}[dp] AND hasabstract'
    params = {
        "db": "pubmed", "term": query, "retmax": 0,
        "retmode": "json", "email": email,
    }
    if api_key:
        params["api_key"] = api_key

    resp = request_retry(ESEARCH_URL, params, bool(api_key))
    if resp is None:
        return []

    data = json.loads(resp.text.replace("\n", " "))
    total = int(data["esearchresult"]["count"])
    logger.info(f"  Found {total} articles for '{journal_ta}'")

    all_pmids = []
    for retstart in range(0, total, 10000):
        params["retmax"] = min(10000, total - retstart)
        params["retstart"] = retstart
        resp = request_retry(ESEARCH_URL, params, bool(api_key))
        if resp:
            data = json.loads(resp.text.replace("\n", " "))
            all_pmids.extend(data["esearchresult"]["idlist"])

    logger.info(f"  Retrieved {len(all_pmids)} PMIDs")

    articles = []
    total_batches = (len(all_pmids) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        batch = all_pmids[start:start + BATCH_SIZE]

        params = {
            "db": "pubmed", "id": ",".join(batch),
            "rettype": "xml", "retmode": "xml", "email": email,
        }
        if api_key:
            params["api_key"] = api_key

        resp = request_retry(EFETCH_URL, params, bool(api_key))
        if resp is None:
            continue

        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError:
            continue

        for article_elem in root.findall(".//PubmedArticle"):
            pmid_elem = article_elem.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""

            title_elem = article_elem.find(".//ArticleTitle")
            title = _get_element_text(title_elem) if title_elem is not None else ""

            abstract_parts = []
            abstract_elem = article_elem.find(".//Abstract")
            if abstract_elem is not None:
                for abs_text in abstract_elem.findall("AbstractText"):
                    label = abs_text.get("Label", "")
                    text = _get_element_text(abs_text)
                    if text:
                        abstract_parts.append(f"{label}: {text}" if label else text)
            abstract = " ".join(abstract_parts)

            year = ""
            for date_path in [".//JournalIssue/PubDate", ".//Article/ArticleDate"]:
                date_elem = article_elem.find(date_path)
                if date_elem is not None:
                    year_elem = date_elem.find("Year")
                    if year_elem is not None and year_elem.text:
                        year = year_elem.text.strip()
                        break
                    medline = date_elem.find("MedlineDate")
                    if medline is not None and medline.text:
                        year = medline.text.strip()[:4]
                        break

            mesh_terms = []
            for mh in article_elem.findall(".//MeshHeadingList/MeshHeading"):
                desc = mh.find("DescriptorName")
                if desc is not None and desc.text:
                    mesh_terms.append(desc.text.strip())

            keywords = []
            for kw_list in article_elem.findall(".//KeywordList"):
                for kw in kw_list.findall("Keyword"):
                    text = _get_element_text(kw)
                    if text:
                        keywords.append(text.strip())

            articles.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "year": year,
                "mesh_terms": "; ".join(mesh_terms),
                "keywords": "; ".join(keywords),
            })

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            logger.info(f"  Fetched batch {batch_idx + 1}/{total_batches} "
                        f"({len(articles)} articles)")

    return articles


# ---------------------------------------------------------------------------
# GPT-based keyword extraction (4th pass fallback)
# ---------------------------------------------------------------------------

def extract_keywords_gpt(articles, client, model, top_n=10, batch_size=5):
    """
    Extract author-style keywords from abstracts using GPT.
    Produces keywords that look like real author keywords (e.g.,
    "deep learning", "protein structure prediction"), keeping the
    pipeline LLM-based end-to-end.
    """
    import time
    from tqdm import tqdm

    extracted = 0
    to_process = [a for a in articles if not a.get("keywords")
                  and a.get("abstract", "") and len(a.get("abstract", "")) >= 50]

    logger.info(f"  Extracting keywords for {len(to_process)} articles via GPT...")

    for i in tqdm(range(0, len(to_process), batch_size),
                  desc="GPT keyword extraction"):
        batch = to_process[i:i + batch_size]

        for article in batch:
            abstract = article["abstract"]
            # Truncate very long abstracts to save tokens
            if len(abstract) > 2000:
                abstract = abstract[:2000]

            prompt = f"""Given the following scientific abstract, extract exactly {top_n} author-style keywords or keyphrases that a researcher would use to describe this paper when submitting to a journal. Return ONLY the keywords, separated by semicolons. Do not include numbering or explanations.

Abstract:
{abstract}

Keywords:"""

            for attempt in range(3):
                try:
                    response = client.responses.create(
                        model=model,
                        input=[{"role": "user", "content": prompt}],
                    )
                    text = response.output_text.strip()

                    # Parse keywords from response
                    # Handle both semicolon and comma separators
                    if ";" in text:
                        kws = [kw.strip() for kw in text.split(";")]
                    else:
                        kws = [kw.strip() for kw in text.split(",")]

                    # Clean up
                    kws = [kw.strip("- .0123456789)") for kw in kws]
                    kws = [kw for kw in kws if len(kw) > 2 and len(kw) < 100]

                    if kws:
                        article["keywords"] = "; ".join(kws[:top_n])
                        extracted += 1
                    break
                except Exception as e:
                    logger.warning(
                        f"  GPT failed for PMID {article.get('pmid')} "
                        f"(attempt {attempt + 1}): {e}"
                    )
                    time.sleep(2 * (attempt + 1))

    logger.info(f"  Extracted keywords for {extracted}/{len(to_process)} "
                f"articles via GPT")
    return extracted


# ---------------------------------------------------------------------------
# Keyword source classification
# ---------------------------------------------------------------------------

def classify_keyword_source(keywords_str, mesh_terms_str):
    if not keywords_str:
        return "none"
    kw_set = {k.strip().lower() for k in keywords_str.split(";")}
    mesh_set = {m.strip().lower() for m in mesh_terms_str.split(";")} if mesh_terms_str else set()
    if mesh_set and kw_set:
        overlap = len(kw_set & mesh_set) / max(len(kw_set), 1)
        if overlap > 0.7:
            return "mesh"
    if kw_set & MESH_INDICATORS:
        return "mesh"
    return "author_or_doi"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    logger.info("=" * 70)
    logger.info("Step 0 Data Collection Fix")
    logger.info("=" * 70)

    # Load existing articles
    logger.info(f"Loading existing articles from {args.articles_csv}")
    existing_articles = []
    existing_pmids = set()
    fieldnames = None
    with open(args.articles_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            existing_articles.append(row)
            existing_pmids.add(row["pmid"])
    logger.info(f"  Loaded {len(existing_articles)} existing articles")

    # --- FIX 1: Re-collect missing journals ---
    logger.info("\n--- Fix 1: Re-collecting missing journal articles ---")
    new_articles = []
    for journal_name, journal_ta in MISSING_JOURNALS.items():
        logger.info(f"Fetching {journal_name} ('{journal_ta}')...")
        articles = fetch_journal_articles(journal_ta, args.email, args.api_key)

        fresh = [a for a in articles if a["pmid"] not in existing_pmids]
        logger.info(f"  {len(fresh)} new articles (after dedup)")
        for a in fresh:
            a["journal"] = journal_name
        new_articles.extend(fresh)

    # 4th pass: use GPT to extract author-style keywords from abstracts
    if new_articles:
        n_no_kw = sum(1 for a in new_articles if not a["keywords"])
        logger.info(f"\n  {n_no_kw}/{len(new_articles)} articles have no keywords — "
                     f"running GPT 4th-pass keyword extraction...")
        client = OpenAI(api_key=args.openai_api_key)
        extract_keywords_gpt(new_articles, client, args.model)

        # MeSH fallback for any still missing after GPT
        mesh_filled = 0
        for a in new_articles:
            if not a["keywords"] and a["mesh_terms"]:
                a["keywords"] = a["mesh_terms"]
                mesh_filled += 1
        if mesh_filled:
            logger.info(f"  {mesh_filled} additional articles filled via MeSH fallback")

        # Drop articles still without keywords
        before = len(new_articles)
        new_articles = [a for a in new_articles if a.get("keywords")]
        logger.info(f"  Final: {len(new_articles)} articles with keywords "
                     f"(dropped {before - len(new_articles)} without any)")

    # --- FIX 3: Add keyword_source column ---
    logger.info("\n--- Fix 3: Classifying keyword sources ---")
    source_counts = Counter()
    for a in existing_articles:
        src = classify_keyword_source(a.get("keywords", ""), a.get("mesh_terms", ""))
        a["keyword_source"] = src
        source_counts[src] += 1
    for a in new_articles:
        a["keyword_source"] = "abstract_gpt"
        source_counts["abstract_gpt"] += 1

    for src, count in source_counts.most_common():
        total = len(existing_articles) + len(new_articles)
        logger.info(f"  {src}: {count} ({count/total*100:.1f}%)")

    # --- Save patched CSV ---
    all_articles = existing_articles + new_articles
    output_fields = list(fieldnames)
    if "keyword_source" not in output_fields:
        output_fields.append("keyword_source")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields,
                                quoting=csv.QUOTE_ALL, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_articles)

    logger.info(f"\nSaved: {args.output}")
    logger.info(f"  Original: {len(existing_articles)} articles")
    logger.info(f"  Added: {len(new_articles)} new (Bioinformatics Advances)")
    logger.info(f"  Total: {len(all_articles)} articles")

    # --- Per-journal summary ---
    logger.info("\n" + "=" * 70)
    logger.info("PER-JOURNAL SUMMARY")
    logger.info("=" * 70)
    journal_counts = Counter(a.get("journal", "?") for a in all_articles)
    for j, c in journal_counts.most_common():
        flag = " [NEW]" if j in MISSING_JOURNALS else ""
        flag += " [MESH-HEAVY]" if j in MESH_HEAVY_JOURNALS else ""
        logger.info(f"  {c:>6}  {j}{flag}")
    logger.info(f"  {'':>6}  ---")
    logger.info(f"  {len(all_articles):>6}  TOTAL")

    logger.info(f"\n{'='*70}")
    logger.info("NEXT STEPS: Re-run the full pipeline on the patched CSV:")
    logger.info(f"  Step 1: python categorize_keywords.py --input {args.output} ...")
    logger.info(f"  Step 2: python cluster_keywords.py ...")
    logger.info(f"  Step 3: python postprocess_and_name_topics.py ...")
    logger.info(f"  Step 4: python dedup_and_hierarchy.py ...")
    logger.info(f"  Step 6: python trend_analysis_and_visualization.py --articles-csv {args.output} ...")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
