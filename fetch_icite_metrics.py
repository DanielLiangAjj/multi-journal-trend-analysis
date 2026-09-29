#!/usr/bin/env python3
"""Fetch NIH iCite citation metrics for every PMID in the corpus (public API, no key).

Outputs (data/):
  icite_metrics.csv        pmid, year, citation_count, relative_citation_ratio, expected_citations_per_year,
                           field_citation_rate, is_research_article, n_references, n_cited_by
  icite_links.json         {pmid: {"cited_by": [...], "references": [...]}}  (PMID lists, for the within-corpus citation network)
  icite_fetch_log.json     batches, failures, fetch date
Idempotent: PMIDs already present in icite_links.json are skipped.
"""
import csv, json, os, sys, time
import requests
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(ROOT, "data", "pubmed_all_journals_2011_2025_w_keywords_patched.csv")
OUT_CSV = os.path.join(ROOT, "data", "icite_metrics.csv")
OUT_LINKS = os.path.join(ROOT, "data", "icite_links.json")
OUT_LOG = os.path.join(ROOT, "data", "icite_fetch_log.json")
API = "https://icite.od.nih.gov/api/pubs"
FIELDS = "pmid,year,citation_count,relative_citation_ratio,expected_citations_per_year,field_citation_rate,is_research_article,cited_by,references"
BATCH = 300   # 1,000 PMIDs per GET exceeded the server URL limit (HTTP 413)


def main():
    df = pd.read_csv(CORPUS, usecols=["pmid", "year"], dtype={"pmid": str})
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    pmids = [p for p in df.loc[df.year.between(2011, 2025), "pmid"].astype(str) if p.isdigit()]
    print(f"{len(pmids)} PMIDs (2011–2025)")
    links = json.load(open(OUT_LINKS)) if os.path.exists(OUT_LINKS) else {}
    rows = {r["pmid"]: r for r in csv.DictReader(open(OUT_CSV))} if os.path.exists(OUT_CSV) else {}
    todo = [p for p in pmids if p not in links]
    print(f"{len(todo)} to fetch")
    log = {"date": time.strftime("%Y-%m-%d"), "batches": 0, "failed_batches": [], "missing_pmids": []}
    sess = requests.Session()
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        for attempt in range(4):
            try:
                r = sess.get(API, params={"pmids": ",".join(chunk), "fl": FIELDS}, timeout=120)
                r.raise_for_status()
                data = r.json().get("data", [])
                break
            except Exception as e:
                print(f"batch {i // BATCH}: attempt {attempt} failed: {e}", flush=True)
                time.sleep(5 * (attempt + 1))
                data = None
        if data is None:
            log["failed_batches"].append(i // BATCH)
            continue
        got = set()
        for rec in data:
            p = str(rec.get("pmid"))
            got.add(p)
            links[p] = {"cited_by": [int(x) for x in (rec.get("cited_by") or [])],
                        "references": [int(x) for x in (rec.get("references") or [])]}
            rows[p] = {"pmid": p, "year": rec.get("year"), "citation_count": rec.get("citation_count"),
                       "relative_citation_ratio": rec.get("relative_citation_ratio"),
                       "expected_citations_per_year": rec.get("expected_citations_per_year"),
                       "field_citation_rate": rec.get("field_citation_rate"),
                       "is_research_article": rec.get("is_research_article"),
                       "n_references": len(rec.get("references") or []), "n_cited_by": len(rec.get("cited_by") or [])}
        for p in chunk:
            if p not in got:
                log["missing_pmids"].append(p)
                links[p] = {"cited_by": [], "references": [], "missing": True}
        log["batches"] += 1
        if log["batches"] % 10 == 0:
            print(f"  {log['batches']} batches, {len(rows)} records", flush=True)
            json.dump(links, open(OUT_LINKS, "w"))
        time.sleep(0.5)
    json.dump(links, open(OUT_LINKS, "w"))
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pmid", "year", "citation_count", "relative_citation_ratio", "expected_citations_per_year", "field_citation_rate", "is_research_article", "n_references", "n_cited_by"])
        w.writeheader()
        for p in pmids:
            if p in rows: w.writerow(rows[p])
    json.dump(log, open(OUT_LOG, "w"), indent=1)
    print(f"done: {len(rows)} records with metrics; {len(log['missing_pmids'])} PMIDs not in iCite; failed batches {log['failed_batches']}")


if __name__ == "__main__":
    main()
