"""
Collect PubMed publication data (PMID, title, abstract, year, MeSH terms, keywords)
for a curated list of biomedical informatics journals.

Two-pass approach:
  Pass 1: PubMed E-utilities (esearch + efetch) — gets keywords for journals that
           submit them to PubMed (~60% of journals).
  Pass 2: DOI-based fallback — for articles where PubMed has no keywords, fetches
           author keywords from publisher pages (IEEE Xplore, Nature, etc.).

Usage:
    python collect_pubmed_data.py \
        --journals biomedical_informatics_journals.csv \
        --output data/pubmed_all_journals_2011_2025_w_keywords.csv \
        --email your_email@example.com \
        [--api-key YOUR_NCBI_API_KEY] \
        [--start-year 2011] \
        [--end-year 2025]
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from xml.etree import ElementTree as ET

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

BATCH_SIZE = 200  # PMIDs per efetch request
MAX_RETRIES = 5
RETRY_BACKOFF = 2  # seconds, doubles each retry
DOI_FETCH_DELAY = 1.5  # seconds between publisher page requests (be polite)

# PubMed journal name -> NLM Title Abbreviation mapping
JOURNAL_PUBMED_NAMES = {
    "Journal of the American Medical Informatics Association (JAMIA)": "J Am Med Inform Assoc",
    "Journal of Medical Internet Research (JMIR)": "J Med Internet Res",
    "IEEE Journal of Biomedical and Health Informatics (J-BHI)": "IEEE J Biomed Health Inform",
    "Journal of Biomedical Informatics (JBI)": "J Biomed Inform",
    "International Journal of Medical Informatics (IJMI)": "Int J Med Inform",
    "JMIR Medical Informatics (JMI)": "JMIR Med Inform",
    "npj Digital Medicine": "NPJ Digit Med",
    "The Lancet Digital Health": "Lancet Digit Health",
    "Briefings in Bioinformatics": "Brief Bioinform",
    "Bioinformatics": "Bioinformatics",
    "JAMIA Open": "JAMIA Open",
    "BMC Medical Informatics and Decision Making": "BMC Med Inform Decis Mak",
    "PLOS Digital Health": "PLOS Digit Health",
    "Journal of Medical Systems": "J Med Syst",
    "Methods of Information in Medicine": "Methods Inf Med",
    "BMJ Health & Care Informatics": "BMJ Health Care Inform",
    "Health Informatics Journal": "Health Informatics J",
    "Applied Clinical Informatics": "Appl Clin Inform",
    "JMIR mHealth and uHealth": "JMIR Mhealth Uhealth",
    "Computers in Biology and Medicine": "Comput Biol Med",
    "Artificial Intelligence in Medicine": "Artif Intell Med",
    "Journal of Clinical and Translational Science": "J Clin Transl Sci",
    "Database: The Journal of Biological Databases and Curation": "Database (Oxford)",
    "Bioinformatics Advances": "Bioinform Adv",
    "Journal of Innovation in Health Informatics": "J Innov Health Inform",
    "Digital Biomarkers": "Digit Biomark",
    "Frontiers in Digital Health": "Front Digit Health",
    "Nature Medicine": "Nat Med",
    "Nature Methods": "Nat Methods",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect PubMed data for biomedical informatics journals."
    )
    parser.add_argument(
        "--journals", required=True,
        help="Path to biomedical_informatics_journals.csv",
    )
    parser.add_argument(
        "--output", default="data/pubmed_all_journals_2011_2025_w_keywords.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--email", required=True,
        help="Email for NCBI E-utilities (required by NCBI policy)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="NCBI API key for higher rate limit (10 req/s vs 3 req/s)",
    )
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-year", type=int, default=2025)
    return parser.parse_args()


def load_journal_list(csv_path):
    """Read journal names from the curated CSV."""
    journals = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Journal Name"].strip()
            if name:
                journals.append(name)
    return journals


def get_pubmed_name(journal_name):
    """Map CSV journal name to the PubMed search term."""
    return JOURNAL_PUBMED_NAMES.get(journal_name, journal_name)


def rate_limit_delay(has_api_key):
    """Respect NCBI rate limits."""
    time.sleep(0.11 if has_api_key else 0.34)


def request_with_retry(url, params, has_api_key):
    """Make a request with exponential backoff retry."""
    for attempt in range(MAX_RETRIES):
        try:
            rate_limit_delay(has_api_key)
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 429:
                wait = RETRY_BACKOFF * (2 ** attempt)
                logger.warning(f"Rate limited (429). Retrying in {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            wait = RETRY_BACKOFF * (2 ** attempt)
            logger.warning(f"Request error: {e}. Retrying in {wait}s... (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
    logger.error(f"Failed after {MAX_RETRIES} retries for {url}")
    return None


# ---------------------------------------------------------------------------
# PubMed API
# ---------------------------------------------------------------------------

def esearch_journal(journal_ta, email, api_key, start_year, end_year):
    """Search PubMed for all articles in a journal within the date range."""
    query = f'"{journal_ta}"[ta] AND {start_year}:{end_year}[dp] AND hasabstract'

    params = {
        "db": "pubmed",
        "term": query,
        "retmax": 0,
        "retmode": "json",
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key

    resp = request_with_retry(ESEARCH_URL, params, bool(api_key))
    if resp is None:
        return []

    try:
        data = json.loads(resp.text.replace('\n', ' ').replace('\r', ''))
        total = int(data["esearchresult"]["count"])
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"  Failed to parse esearch count response: {e}")
        return []
    logger.info(f"  Found {total} articles with abstracts for '{journal_ta}'")

    if total == 0:
        return []

    all_pmids = []
    for retstart in range(0, total, 10000):
        params["retmax"] = min(10000, total - retstart)
        params["retstart"] = retstart
        resp = request_with_retry(ESEARCH_URL, params, bool(api_key))
        if resp is None:
            continue
        try:
            data = json.loads(resp.text.replace('\n', ' ').replace('\r', ''))
            all_pmids.extend(data["esearchresult"]["idlist"])
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"  Failed to parse esearch response at retstart={retstart}: {e}")
            continue

    logger.info(f"  Retrieved {len(all_pmids)} PMIDs")
    return all_pmids


def efetch_articles(pmids, email, api_key):
    """
    Fetch article details for a list of PMIDs.
    Returns ALL articles (with and without keywords).
    Articles without PubMed keywords will have keywords="" and doi populated
    for the second-pass DOI fallback.
    """
    articles = []
    total_batches = (len(pmids) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = start + BATCH_SIZE
        batch_pmids = pmids[start:end]

        params = {
            "db": "pubmed",
            "id": ",".join(batch_pmids),
            "rettype": "xml",
            "retmode": "xml",
            "email": email,
        }
        if api_key:
            params["api_key"] = api_key

        resp = request_with_retry(EFETCH_URL, params, bool(api_key))
        if resp is None:
            logger.error(f"  Batch {batch_idx + 1}/{total_batches} failed, skipping")
            continue

        batch_articles = parse_pubmed_xml(resp.text)
        articles.extend(batch_articles)

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == total_batches:
            with_kw = sum(1 for a in articles if a["keywords"])
            logger.info(
                f"  Fetched batch {batch_idx + 1}/{total_batches} "
                f"({len(articles)} articles, {with_kw} with PubMed keywords)"
            )

    return articles


def parse_pubmed_xml(xml_text):
    """
    Parse PubMed XML response. Returns ALL articles with their metadata.
    Articles without keywords get keywords="" and are candidates for DOI fallback.
    """
    articles = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning(f"  Malformed XML response, skipping batch: {e}")
        return articles

    for article_elem in root.findall(".//PubmedArticle"):
        # --- Keywords (from all KeywordList elements) ---
        keywords = []
        for kw_list in article_elem.findall(".//KeywordList"):
            for kw in kw_list.findall("Keyword"):
                text = _get_element_text(kw)
                if text:
                    keywords.append(text.strip())

        # --- PMID ---
        pmid_elem = article_elem.find(".//PMID")
        pmid = pmid_elem.text if pmid_elem is not None else ""

        # --- DOI ---
        doi_elem = article_elem.find('.//ELocationID[@EIdType="doi"]')
        doi = doi_elem.text.strip() if doi_elem is not None and doi_elem.text else ""

        # --- Title ---
        title_elem = article_elem.find(".//ArticleTitle")
        title = _get_element_text(title_elem) if title_elem is not None else ""

        # --- Abstract ---
        abstract_parts = []
        abstract_elem = article_elem.find(".//Abstract")
        if abstract_elem is not None:
            for abs_text in abstract_elem.findall("AbstractText"):
                label = abs_text.get("Label", "")
                text = _get_element_text(abs_text)
                if text:
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

        # --- Year ---
        year = ""
        # Prefer PubDate (journal issue date) over ArticleDate (e-pub date)
        for date_path in [
            ".//JournalIssue/PubDate",
            ".//Article/ArticleDate",
        ]:
            date_elem = article_elem.find(date_path)
            if date_elem is not None:
                year_elem = date_elem.find("Year")
                if year_elem is not None and year_elem.text:
                    year = year_elem.text.strip()
                    break
                medline_date = date_elem.find("MedlineDate")
                if medline_date is not None and medline_date.text:
                    year = medline_date.text.strip()[:4]
                    break

        # --- MeSH Terms ---
        mesh_terms = []
        for mesh_heading in article_elem.findall(".//MeshHeadingList/MeshHeading"):
            descriptor = mesh_heading.find("DescriptorName")
            if descriptor is not None and descriptor.text:
                mesh_terms.append(descriptor.text.strip())

        articles.append({
            "pmid": pmid,
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "year": year,
            "mesh_terms": "; ".join(mesh_terms),
            "keywords": "; ".join(keywords),
        })

    return articles


def _get_element_text(elem):
    """Get all text from an XML element, including child elements."""
    return "".join(elem.itertext()).strip()


# ---------------------------------------------------------------------------
# Pass 2: DOI-based keyword fallback for publishers that don't submit to PubMed
# ---------------------------------------------------------------------------

_session = None

def _get_session():
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })
    return _session


def fetch_keywords_from_doi(doi):
    """
    Attempt to extract author keywords from a publisher page via DOI.
    Supports: IEEE Xplore, Nature, Elsevier/ScienceDirect.
    Returns a list of keyword strings, or [] if extraction fails.
    """
    if not doi:
        return []

    session = _get_session()

    try:
        resp = session.get(f"https://doi.org/{doi}", timeout=20, allow_redirects=True)
    except requests.exceptions.RequestException:
        return []

    if resp.status_code != 200:
        return []

    url = resp.url
    text = resp.text

    # --- IEEE Xplore ---
    if "ieeexplore.ieee.org" in url:
        return _extract_ieee_keywords(text)

    # --- Nature ---
    if "nature.com" in url:
        return _extract_nature_keywords(text)

    # --- Elsevier / ScienceDirect ---
    if "sciencedirect.com" in url or "elsevier" in url:
        return _extract_elsevier_keywords(text)

    # --- Springer / BMC ---
    if "springer.com" in url or "biomedcentral.com" in url:
        return _extract_springer_keywords(text)

    # --- Generic: try common meta tags ---
    return _extract_generic_keywords(text)


def _extract_ieee_keywords(text):
    """Extract Author Keywords from IEEE Xplore page."""
    groups = re.findall(
        r'\{"type":"Author Keywords","kwd":\[([^\]]+)\]\}', text
    )
    if groups:
        return [k.strip().strip('"') for k in groups[0].split('","')]
    return []


def _extract_nature_keywords(text):
    """Extract keywords from Nature page meta tags."""
    kws = re.findall(
        r'<meta name="(?:keywords|citation_keywords|dc\.subject)" content="([^"]+)"',
        text, re.IGNORECASE,
    )
    if kws:
        # Nature often puts multiple keywords in a single meta tag, comma-separated
        all_kws = []
        for entry in kws:
            all_kws.extend([k.strip() for k in entry.split(",") if k.strip()])
        return all_kws

    # Fallback: JSON-LD
    jsonld_blocks = re.findall(
        r'<script type="application/ld\+json">(.*?)</script>', text, re.DOTALL
    )
    for block in jsonld_blocks:
        try:
            d = json.loads(block)
            if isinstance(d, dict) and "keywords" in d:
                kw_val = d["keywords"]
                if isinstance(kw_val, list):
                    return kw_val
                if isinstance(kw_val, str):
                    return [k.strip() for k in kw_val.split(",") if k.strip()]
        except (json.JSONDecodeError, TypeError):
            pass
    return []


def _extract_elsevier_keywords(text):
    """Extract keywords from Elsevier/ScienceDirect pages."""
    # ScienceDirect renders keywords in div.keyword spans
    kws = re.findall(
        r'class="keyword[^"]*"[^>]*><span[^>]*>([^<]+)</span>', text
    )
    if kws:
        return [k.strip() for k in kws if k.strip()]

    # Fallback: meta tags
    kws = re.findall(
        r'<meta name="citation_keywords" content="([^"]+)"', text, re.IGNORECASE
    )
    if kws:
        all_kws = []
        for entry in kws:
            all_kws.extend([k.strip() for k in entry.split(",") if k.strip()])
        return all_kws
    return []


def _extract_springer_keywords(text):
    """Extract keywords from Springer/BMC pages."""
    # Springer uses c-article-subject-list or keyword groups
    kws = re.findall(r'<span class="Keyword"[^>]*>([^<]+)</span>', text)
    if kws:
        return [k.strip() for k in kws if k.strip()]

    # BMC uses <meta name="citation_keywords">
    kws = re.findall(
        r'<meta name="citation_keywords" content="([^"]+)"', text, re.IGNORECASE
    )
    if kws:
        all_kws = []
        for entry in kws:
            all_kws.extend([k.strip() for k in entry.split(",") if k.strip()])
        return all_kws
    return []


def _extract_generic_keywords(text):
    """Generic keyword extraction from common HTML meta patterns."""
    # Try citation_keywords meta tags
    kws = re.findall(
        r'<meta name="(?:citation_keywords|keywords|dc\.subject)" content="([^"]+)"',
        text, re.IGNORECASE,
    )
    if kws:
        all_kws = []
        for entry in kws:
            all_kws.extend([k.strip() for k in entry.split(",") if k.strip()])
        return all_kws
    return []


def fill_missing_keywords(articles):
    """
    Pass 2: For articles without PubMed keywords, attempt to fetch
    author keywords from publisher pages via DOI.
    """
    missing = [a for a in articles if not a["keywords"] and a["doi"]]
    if not missing:
        return 0

    logger.info(f"  Pass 2: Fetching keywords via DOI for {len(missing)} articles without PubMed keywords...")
    filled = 0

    for i, article in enumerate(missing):
        kws = fetch_keywords_from_doi(article["doi"])
        if kws:
            article["keywords"] = "; ".join(kws)
            filled += 1

        if (i + 1) % 50 == 0:
            logger.info(f"    DOI progress: {i + 1}/{len(missing)} checked, {filled} filled")

        time.sleep(DOI_FETCH_DELAY)

    logger.info(f"  Pass 2 complete: filled {filled}/{len(missing)} articles via DOI")
    return filled


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def load_progress(progress_path):
    """Load set of already-completed journal names."""
    if os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            return set(json.load(f))
    return set()


def save_progress(progress_path, completed):
    """Save set of completed journal names."""
    with open(progress_path, "w") as f:
        json.dump(sorted(completed), f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    journals = load_journal_list(args.journals)
    logger.info(f"Loaded {len(journals)} journals from {args.journals}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    progress_path = args.output + ".progress.json"
    completed = load_progress(progress_path)

    fieldnames = ["pmid", "title", "abstract", "year", "mesh_terms", "keywords", "journal"]

    write_header = not os.path.exists(args.output)

    # Load existing PMIDs to avoid duplicates on partial re-runs
    existing_pmids = set()
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_pmids.add(row["pmid"])
        logger.info(f"Loaded {len(existing_pmids)} existing PMIDs from output file")

    total_collected = 0
    total_skipped = 0

    for idx, journal_name in enumerate(journals):
        logger.info(f"\n[{idx + 1}/{len(journals)}] Processing: {journal_name}")

        if journal_name in completed:
            logger.info(f"  Already completed, skipping.")
            total_skipped += 1
            continue

        journal_ta = get_pubmed_name(journal_name)
        logger.info(f"  PubMed search term: '{journal_ta}'")

        # Step 1: Search for PMIDs
        pmids = esearch_journal(
            journal_ta, args.email, args.api_key,
            args.start_year, args.end_year,
        )

        if not pmids:
            logger.warning(f"  No articles found. Check if the journal name mapping is correct.")
            completed.add(journal_name)
            save_progress(progress_path, completed)
            continue

        # Step 2: Fetch ALL article details from PubMed (with and without keywords)
        articles = efetch_articles(pmids, args.email, args.api_key)
        pubmed_kw_count = sum(1 for a in articles if a["keywords"])
        logger.info(f"  Pass 1 (PubMed): {pubmed_kw_count}/{len(articles)} articles have keywords")

        # Step 3: DOI fallback for articles missing keywords
        doi_filled = fill_missing_keywords(articles)

        # Step 4: MeSH fallback — for articles still without keywords
        # (e.g., Bioinformatics, Lancet DH don't publish author keywords)
        mesh_filled = 0
        for article in articles:
            if not article["keywords"] and article["mesh_terms"]:
                article["keywords"] = article["mesh_terms"]
                mesh_filled += 1
        if mesh_filled:
            logger.info(f"  Pass 3 (MeSH fallback): filled {mesh_filled} articles using MeSH terms")

        # Step 5: Filter to only articles WITH keywords and deduplicate
        new_articles = []
        for article in articles:
            if not article["keywords"]:
                continue
            if article["pmid"] in existing_pmids:
                continue
            article["journal"] = journal_name
            # Remove doi from output (not in original schema)
            article.pop("doi", None)
            new_articles.append(article)
            existing_pmids.add(article["pmid"])

        # Step 5: Append to output CSV
        with open(args.output, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerows(new_articles)

        total_collected += len(new_articles)
        logger.info(
            f"  Saved {len(new_articles)} articles with keywords "
            f"(PubMed: {pubmed_kw_count}, DOI fallback: {doi_filled}, "
            f"MeSH fallback: {mesh_filled}, total so far: {total_collected})"
        )

        completed.add(journal_name)
        save_progress(progress_path, completed)

    logger.info(f"\nDone! Collected {total_collected} articles across {len(journals) - total_skipped} journals.")
    logger.info(f"Output: {args.output}")

    # Print summary
    logger.info("\n--- Summary ---")
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            journal_counts = {}
            for row in reader:
                j = row.get("journal", "unknown")
                journal_counts[j] = journal_counts.get(j, 0) + 1
        for j, count in sorted(journal_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {j}: {count} articles")
        logger.info(f"  TOTAL: {sum(journal_counts.values())} articles")


if __name__ == "__main__":
    main()
