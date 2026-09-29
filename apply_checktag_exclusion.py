#!/usr/bin/env python3
"""Round-23 data-cleaning step (2026-09-12): remove MeSH check tags (population/organism tags) from the
keywords of MeSH-substituted records (keyword_source == 'mesh') in the patched corpus.

Why: Pass 3 substituted ALL MeSH descriptors, including check tags such as "Humans", "Female", "Middle Aged",
which describe the study population rather than research content. They were clustered into topics
("humans" -> clinical study designs + mixed medical imaging; "female" -> infectious diseases; "middle aged" ->
older people; ...), inflating those topics and creating spurious co-occurrence pairs. Author-provided keywords
are left untouched (an author keyword "pregnancy" is content).

The original file is backed up once as *.bak_prechecktag_20260912.csv; the script is idempotent.
"""
import csv, json, os, re, shutil, sys
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "data", "pubmed_all_journals_2011_2025_w_keywords_patched.csv")
BAK = SRC.replace(".csv", ".bak_prechecktag_20260912.csv")
LOG = os.path.join(ROOT, "data", "checktag_exclusion_log.json")

CHECK_TAGS = {
    # the MeSH check tags (population and organism tags that indexers add to almost every record)
    "humans", "animals", "male", "female", "adolescent", "adult", "aged", "aged, 80 and over", "child",
    "child, preschool", "infant", "infant, newborn", "middle aged", "young adult", "pregnancy",
    "mice", "rats", "cats", "cattle", "dogs", "guinea pigs", "hamsters", "rabbits", "chick embryo",
}
# inbred-strain qualifiers of the organism check tags (e.g. "mice, inbred c57bl", "rats, sprague-dawley")
ORGANISM_PREFIX = re.compile(r"^(mice|rats|dogs|cats|cattle|rabbits|hamsters|guinea pigs),\s")


def excluded(kw: str) -> bool:
    k = kw.strip().lower()
    return k in CHECK_TAGS or bool(ORGANISM_PREFIX.match(k))


def main():
    if not os.path.exists(BAK):
        shutil.copy2(SRC, BAK)
        print(f"backed up original corpus to {BAK}")
    df = pd.read_csv(BAK, quoting=csv.QUOTE_ALL, dtype=str, keep_default_na=False)
    removed = {}
    n_rec_changed = 0
    n_kw_removed = 0
    new_keywords = []
    for src, kws in zip(df["keyword_source"], df["keywords"]):
        if src != "mesh" or not kws:
            new_keywords.append(kws)
            continue
        parts = [k.strip() for k in kws.split(";") if k.strip()]
        keep = [k for k in parts if not excluded(k)]
        drop = [k for k in parts if excluded(k)]
        if drop:
            n_rec_changed += 1
            n_kw_removed += len(drop)
            for k in drop:
                removed[k.lower()] = removed.get(k.lower(), 0) + 1
        new_keywords.append("; ".join(keep))
    df["keywords"] = new_keywords
    df.to_csv(SRC, index=False, quoting=csv.QUOTE_ALL)
    log = {
        "date": "2026-09-12",
        "rule": "keyword_source == 'mesh' only: drop the MeSH check tags (population/organism tags) and their inbred-strain qualifiers",
        "records_total": int(len(df)),
        "mesh_records": int((df["keyword_source"] == "mesh").sum()),
        "records_changed": n_rec_changed,
        "keyword_occurrences_removed": n_kw_removed,
        "distinct_strings_removed": len(removed),
        "removed_strings_by_count": dict(sorted(removed.items(), key=lambda x: -x[1])),
        "records_left_with_no_keywords": int(((df["keyword_source"] == "mesh") & (df["keywords"] == "")).sum()),
    }
    json.dump(log, open(LOG, "w"), indent=1)
    print(json.dumps({k: v for k, v in log.items() if k != "removed_strings_by_count"}, indent=1))
    print("top removed:", list(log["removed_strings_by_count"].items())[:15])


if __name__ == "__main__":
    main()
