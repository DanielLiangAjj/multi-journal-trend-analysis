"""
collect_author_data.py

Re-fetches author and affiliation data from PubMed for all PMIDs in the existing
articles CSV. Computes derived authorship features:

  - n_authors: number of authors per article
  - n_unique_affiliations: number of unique institutional affiliations
  - has_clinical_affiliation: any author at hospital/medical school/clinical dept
  - has_computational_affiliation: any author at CS/informatics/engineering/stats
  - is_interdisciplinary: both clinical AND computational affiliations present
                          (proxy for MD+PhD or clinician + computational scientist
                          collaboration)

Resumable: saves progress every 50 batches.

Usage:
    python collect_author_data.py \\
        --articles-csv data/pubmed_all_journals_2011_2025_w_keywords.csv \\
        --output data/articles_with_authors.csv \\
        --email your_email@example.com \\
        [--api-key YOUR_NCBI_API_KEY]
"""

import argparse
import csv
import logging
import os
import re
import sys
import time
from xml.etree import ElementTree as ET

import requests

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BATCH_SIZE = 200
MAX_RETRIES = 5
RETRY_BACKOFF = 2

# ---------------------------------------------------------------------------
# Affiliation classification heuristics
# ---------------------------------------------------------------------------

CLINICAL_PATTERNS = [
    r"\bschool of medicine\b",
    r"\bmedical school\b",
    r"\bfaculty of medicine\b",
    r"\bcollege of medicine\b",
    r"\bhospital\b",
    r"\bmedical center\b",
    r"\bmedical centre\b",
    r"\bcancer center\b",
    r"\bclinic\b",
    r"\bdepartment of (medicine|surgery|pediatrics|cardiology|neurology|"
    r"oncology|radiology|pathology|psychiatry|anesthesiology|internal medicine|"
    r"family medicine|emergency medicine|pulmonology|gastroenterology|"
    r"endocrinology|rheumatology|dermatology|ophthalmology|otolaryngology|"
    r"urology|obstetrics|gynecology|pharmacology|nursing|critical care|"
    r"orthopaedics|orthopedics|hematology|nephrology|infectious diseases|"
    r"public health|preventive medicine|geriatrics|family practice)\b",
    r"\bclinical (research|sciences|trials|epidemiology|medicine|pharmacology)\b",
    r"\bhealth (system|service|services)\b",
    r"\bnhs\b",
]

COMPUTATIONAL_PATTERNS = [
    r"\bcomputer science\b",
    r"\bcomputer engineering\b",
    r"\bcomputational\b",
    r"\binformatics\b",
    r"\bbioinformatics\b",
    r"\bmedical informatics\b",
    r"\bbiomedical informatics\b",
    r"\bhealth informatics\b",
    r"\bcomputing\b",
    r"\binformation (systems|technology|science)\b",
    r"\b(biomedical|electrical|chemical|software|computer|systems) engineering\b",
    r"\bstatistics\b",
    r"\bbiostatistics\b",
    r"\bmathematics\b",
    r"\bdata science\b",
    r"\bartificial intelligence\b",
    r"\bmachine learning\b",
    r"\bschool of (engineering|computing|informatics)\b",
]

CLINICAL_REGEX = re.compile("|".join(CLINICAL_PATTERNS), re.IGNORECASE)
COMPUTATIONAL_REGEX = re.compile("|".join(COMPUTATIONAL_PATTERNS), re.IGNORECASE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect author/affiliation data from PubMed.",
    )
    parser.add_argument("--articles-csv", required=True)
    parser.add_argument("--output", default="data/articles_with_authors.csv")
    parser.add_argument("--email", required=True)
    parser.add_argument("--api-key", default=None)
    return parser.parse_args()


def load_existing_pmids(csv_path):
    """Load PMIDs from the existing articles CSV."""
    logger.info(f"Loading PMIDs from {csv_path}")
    pmids = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmid = row.get("pmid", "").strip()
            if pmid:
                pmids.append(pmid)
    logger.info(f"  Loaded {len(pmids)} PMIDs")
    return pmids


def load_existing_output(output_path):
    """Load already-processed PMIDs from output file (for resume)."""
    if not os.path.exists(output_path):
        return set()
    done = set()
    with open(output_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmid = row.get("pmid", "").strip()
            if pmid:
                done.add(pmid)
    logger.info(f"  Found {len(done)} already-processed PMIDs in output")
    return done


def normalize_affiliation(aff):
    """Normalize affiliation string for deduplication."""
    if not aff:
        return ""
    # Strip email and electronic address
    aff = re.sub(r"electronic address.*$", "", aff, flags=re.IGNORECASE)
    aff = re.sub(r"\bemail.*$", "", aff, flags=re.IGNORECASE)
    aff = re.sub(r"\S+@\S+", "", aff)
    # Lowercase + collapse whitespace + strip punctuation
    aff = re.sub(r"[^\w\s]", " ", aff.lower())
    aff = " ".join(aff.split())
    # Use first 100 chars as identifier
    return aff[:100]


def parse_pubmed_xml_authors(xml_text):
    """Parse PubMed XML and extract author/affiliation features per PMID."""
    results = {}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return results

    for article_elem in root.findall(".//PubmedArticle"):
        pmid_elem = article_elem.find(".//PMID")
        if pmid_elem is None or pmid_elem.text is None:
            continue
        pmid = pmid_elem.text.strip()

        n_authors = 0
        all_affiliations = []

        for author_elem in article_elem.findall(".//AuthorList/Author"):
            last = author_elem.findtext("LastName")
            collective = author_elem.findtext("CollectiveName")
            if not (last or collective):
                continue
            n_authors += 1

            for aff_elem in author_elem.findall(".//AffiliationInfo/Affiliation"):
                if aff_elem.text:
                    all_affiliations.append(aff_elem.text.strip())

        # Compute unique affiliations
        unique_affs = set()
        for aff in all_affiliations:
            norm = normalize_affiliation(aff)
            if norm:
                unique_affs.add(norm)
        n_unique_affiliations = len(unique_affs)

        # Classify affiliations
        all_aff_text = " ".join(all_affiliations).lower()
        has_clinical = bool(CLINICAL_REGEX.search(all_aff_text))
        has_computational = bool(COMPUTATIONAL_REGEX.search(all_aff_text))
        is_interdisciplinary = has_clinical and has_computational

        results[pmid] = {
            "n_authors": n_authors,
            "n_unique_affiliations": n_unique_affiliations,
            "has_clinical": int(has_clinical),
            "has_computational": int(has_computational),
            "is_interdisciplinary": int(is_interdisciplinary),
        }

    return results


def fetch_batch(pmids, email, api_key):
    """Fetch a batch of PMIDs from PubMed efetch."""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "xml",
        "retmode": "xml",
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key

    delay = 0.11 if api_key else 0.34
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(delay)
            resp = requests.get(EFETCH_URL, params=params, timeout=60)
            if resp.status_code == 429:
                wait = RETRY_BACKOFF * (2 ** attempt)
                logger.warning(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.text
        except requests.exceptions.RequestException as e:
            wait = RETRY_BACKOFF * (2 ** attempt)
            logger.warning(f"  Request failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    return None


def main():
    args = parse_args()

    logger.info("=" * 70)
    logger.info("Author Data Collection from PubMed")
    logger.info("=" * 70)

    pmids = load_existing_pmids(args.articles_csv)
    done_pmids = load_existing_output(args.output)
    remaining = [p for p in pmids if p not in done_pmids]
    logger.info(f"Need to process {len(remaining)} of {len(pmids)} PMIDs")

    if not remaining:
        logger.info("All PMIDs already processed. Done.")
        return

    fieldnames = [
        "pmid", "n_authors", "n_unique_affiliations",
        "has_clinical", "has_computational", "is_interdisciplinary",
    ]

    write_header = not os.path.exists(args.output)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
    processed = 0
    start_time = time.time()

    with open(args.output, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for batch_idx in range(total_batches):
            start = batch_idx * BATCH_SIZE
            end = start + BATCH_SIZE
            batch = remaining[start:end]

            xml = fetch_batch(batch, args.email, args.api_key)
            if xml is None:
                logger.error(f"  Batch {batch_idx + 1}/{total_batches} failed")
                continue

            results = parse_pubmed_xml_authors(xml)

            for pmid in batch:
                row = results.get(pmid, {
                    "n_authors": 0,
                    "n_unique_affiliations": 0,
                    "has_clinical": 0,
                    "has_computational": 0,
                    "is_interdisciplinary": 0,
                })
                row["pmid"] = pmid
                writer.writerow(row)
                processed += 1

            f.flush()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                elapsed = time.time() - start_time
                rate = processed / max(elapsed, 1)
                eta = (len(remaining) - processed) / max(rate, 0.01)
                logger.info(
                    f"  Batch {batch_idx + 1}/{total_batches}: "
                    f"{processed}/{len(remaining)} processed "
                    f"({rate:.0f}/s, ETA {eta/60:.1f} min)"
                )

    logger.info(f"\nDone! Processed {processed} PMIDs in {(time.time()-start_time)/60:.1f} min")
    logger.info(f"Output: {args.output}")


if __name__ == "__main__":
    main()
