# Multi-Journal Biomedical Informatics Trend Analysis

A data-driven, AI-augmented pipeline for large-scale research trend analysis across **29 biomedical informatics journals** spanning **2011–2025**. This project extends the methodology of Liang et al.'s single-journal (Journal of Biomedical Informatics) analysis to a multi-journal corpus, enabling cross-journal comparison, journal-level specialization analysis, and field-wide trend detection.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Statistics](#key-statistics)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Methodology Details](#methodology-details)
5. [Repository Structure](#repository-structure)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Output Files](#output-files)
9. [Key Findings](#key-findings)
10. [Comparison with Original Paper](#comparison-with-original-paper)
11. [Data Sources](#data-sources)
12. [Reference](#reference)

---

## Project Overview

This project answers the following research questions about biomedical informatics research trends:

**Original questions (from the paper):**
- **RQ1:** How has overall publication volume changed over time?
- **RQ2:** Which methodology and health topics are most popular, and how have their trends evolved?
- **RQ3:** Which methodology topics co-occur with which health domains?
- **RQ4:** How do top topics distribute across different time periods?

**New multi-journal questions (added in this expansion):**
- **N1:** How do the 29 journals compare in publication volume and growth?
- **N2:** What topic specializations does each journal exhibit?
- **N3:** Which topics are universal across journals vs. concentrated in a few?
- **N4:** When did each topic first emerge in the literature?
- **N5:** Which topics are rising fastest and which are declining (trend slopes)?
- **N6:** Which journals have similar research profiles (similarity network)?
- **N7:** How has the internal keyword composition within top topics shifted over time?

---

## Key Statistics

| Metric | Value |
|---|---|
| Journals analyzed | 29 |
| Time range | 2011–2025 |
| Articles collected | 77,568 |
| Unique keywords | 90,386 |
| Methodology keywords | 31,257 (34.6%) |
| Health keywords | 47,694 (52.8%) |
| Both (cross-cutting) | 10,316 (11.4%) |
| Excluded (neither) | 1,119 (1.2%) |
| K-means clusters per domain | 100 (chosen via silhouette analysis) |
| Final methodology topics | 116 |
| Final health topics | 95 |
| Methodology hierarchy edges | 39 |
| Health hierarchy edges | 33 (max depth 3) |
| Visualizations generated | 18 plots + 5 data tables |

---

## Pipeline Architecture

The pipeline consists of 6 sequential stages, each producing intermediate outputs that feed into the next:

```
[Step 0] Data Collection (PubMed)
    │
    ▼  77,568 articles + 90,386 keywords
[Step 1] Keyword Categorization (BiomedBERT + GPT-5-nano)
    │
    ▼  Methodology / Health / Both / Neither labels
[Step 2] Keyword Clustering (K-means on BiomedBERT embeddings)
    │
    ▼  100 clusters per domain
[Step 3] Cluster Post-processing & Topic Naming (GPT-5-nano)
    │
    ▼  921 methodology + 602 health sub-topics → 116 + 95 unique names
[Step 4] Deduplication & Hierarchy (Stemming + GPT relationships)
    │
    ▼  116 methodology + 95 health final topics with hierarchy
[Step 6] Trend Analysis & Visualization
    │
    ▼  18 plots + 5 CSVs answering RQ1–RQ4 + N1–N7
```

Each stage is independent, fully resumable (progress tracking), and faithfully implements the methodology from the original paper while adding multi-journal extensions.

---

## Methodology Details

### Step 0: Data Collection (`collect_pubmed_data.py`)

Collects PubMed publication metadata for 29 curated journals using a three-pass strategy to maximize keyword coverage:

1. **Pass 1 — PubMed E-utilities:** `esearch` + `efetch` to retrieve PMID, title, abstract, year, MeSH terms, and author keywords for each journal/year combination.
2. **Pass 2 — DOI publisher scraping fallback:** For articles without PubMed-supplied keywords, scrape author keywords from publisher pages (IEEE Xplore, Nature, Springer, Elsevier) via the article's DOI.
3. **Pass 3 — MeSH fallback:** For articles still missing keywords (e.g., journals like Bioinformatics that don't submit author keywords), use the article's MeSH descriptor terms as a substitute.

Features: PMID-based deduplication, per-journal progress tracking (resumable), incremental CSV writing, NCBI rate limit handling with API key support.

### Step 1: Keyword Categorization (`categorize_keywords.py`)

Classifies each unique keyword as one of: `methodology`, `health`, `both`, or `neither`, using a hybrid embedding + LLM approach:

1. **Build a reference taxonomy:** Download MeSH 2024 tree, extract terms from the top 2 hierarchy levels (~2,078 terms).
2. **Annotate MeSH terms:** Use GPT-5-nano to label each MeSH term as methodology, health, or both — providing a ground-truth-like reference set.
3. **Embed everything:** Compute BiomedBERT (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`) CLS-token embeddings for all MeSH terms and all keywords.
4. **Few-shot classification:** For each keyword, find the 5 most similar annotated MeSH terms via cosine similarity, then ask GPT-5-nano to classify the keyword using these as in-context examples.
5. **Batch processing:** 20 keywords per GPT call with structured Pydantic output, full progress tracking.

### Step 2: Keyword Clustering (`cluster_keywords.py`)

Groups semantically similar keywords into candidate topics using K-means on BiomedBERT embeddings:

1. **Domain pool construction:** Methodology and health keywords are placed into separate clustering pools. Crucially, "both" keywords are placed into BOTH pools (preventing data loss for cross-cutting concepts).
2. **K selection:** Optional silhouette analysis scans K=100–1500. K=100 was selected based on best silhouette score for both domains.
3. **K-means clustering:** Run separately on each domain pool with `n_init=10` and fixed `random_state=42` for reproducibility.
4. **Output:** Per-keyword cluster assignments, per-cluster summaries with top representative keywords (sorted by distance to centroid), centroid vectors, and diagnostic plots.

Quality safeguards: data integrity checks at every stage, deduplication-aware merging for keywords with conflicting labels, automatic recomputation of missing embeddings.

### Step 3: Cluster Post-processing & Topic Naming (`postprocess_and_name_topics.py`)

Refines clusters and generates human-readable topic names:

1. **Abbreviation cluster removal:** Clusters where the average keyword length is below 10 characters (e.g., clusters dominated by acronyms like CNN, RNN, SVM) are flagged and removed as low-information.
2. **Cluster coherence checking:** Each remaining cluster is sent to GPT with the question: "Do these terms focus on a coherent topic?" If not, GPT splits the cluster into coherent sub-topics using structured Pydantic output.
3. **Post-processing safeguards:** GPT sometimes drops or hallucinates keywords. We:
   - Remove any keywords GPT invented (not in the original cluster)
   - Recover dropped keywords by assigning them to the nearest sub-cluster using BiomedBERT embedding similarity
   - Validate that the keyword count after equals the keyword count before (zero data loss)
4. **Topic naming:** For each (sub-)cluster, count the global frequency of each keyword, then ask GPT to generate a precise 3-word phrase summarizing the topic based on the keyword frequency distribution.
5. **Name deduplication:** Before accepting a new topic name, embed it with SentenceTransformer (`all-mpnet-base-v2`), find the most similar existing names, and ask GPT whether any existing name fits — preventing duplicate names like "Heart Disease" vs. "Cardiac Disease".

### Step 4: Deduplication & Hierarchy Construction (`dedup_and_hierarchy.py`)

Merges semantically equivalent topics and builds parent-child relationships:

**Phase 1 — Stemming Merge** (no GPT): Apply Porter stemmer to all topic names; topics with identical stemmed forms are merged (e.g., "Clinical Decision" and "Clinical Decisions" → same stem → merge).

**Phase 2 — Embedding Similarity** (no GPT): Compute pairwise cosine similarity between all unique topic names using SentenceTransformer embeddings.

**Phase 3 — Threshold Discovery** (GPT): For each topic, use binary search with `have_relationship` GPT calls to find the similarity percentile threshold where topics transition from "related" to "not related".

**Phase 4 — Relationship Classification** (GPT): For topic pairs above the threshold, ask GPT to classify the relationship as one of:
- **Equal** — synonymous topics (merge)
- **Superset** — topic A is broader than topic B
- **Subset** — topic A is narrower than topic B
- **NoOverlap** — conceptually distinct (no edge)

**Phase 5 — Merge Equals & Build Hierarchy:**
- Equal topics merged via canonical name mapping
- Subset/Superset relations extracted as parent-child edges
- **Iterative edge verification:** 4 rounds of GPT verification, alternating direction (forward/reverse). Each round filters out edges where GPT no longer confirms the relationship, increasing precision.
- **Cycle removal:** DFS-based cycle detection; cycles broken by removing the lowest-confidence edge.
- **Transitive reduction:** Remove edges implied by longer paths, keeping only immediate parent-child relationships.
- Output: indented hierarchy text file + JSON edge list.

### Step 6: Trend Analysis & Visualization (`trend_analysis_and_visualization.py`)

Maps each article's keywords to topics, then aggregates by year/journal/topic to produce 18 visualizations and 5 data tables. No GPT calls — pure local computation.

**Paper's analyses replicated:**
- **RQ1:** Total publication volume bar/line chart
- **RQ2:** Top 16 topic dual-axis trend lines (count + percentage)
- **RQ3:** Methodology × Health co-occurrence heatmap
- **RQ4:** Top 20 topics by 5-year time period (stacked bar charts)

**New multi-journal analyses added:**
- **N1:** Per-journal volume comparison + per-journal trend lines
- **N2:** Journal × Topic specialization heatmap (% of journal's articles in each topic)
- **N3:** Cross-journal topic sharing (how many journals each topic appears in)
- **N4:** Topic emergence timeline (first year each topic appeared)
- **N5:** Rising/declining topic detection via linear regression slopes
- **N6:** Journal similarity network using cosine similarity of topic distributions
- **N7:** Stacked area charts showing keyword composition evolution within top topics

---

## Repository Structure

```
multi_journal_trend_analysis/
├── README.md                                  # This file
├── .gitignore
│
├── collect_pubmed_data.py                     # Step 0: PubMed data collection
├── categorize_keywords.py                     # Step 1: Keyword categorization
├── cluster_keywords.py                        # Step 2: K-means clustering
├── postprocess_and_name_topics.py             # Step 3: Cluster post-processing & naming
├── dedup_and_hierarchy.py                     # Step 4: Dedup & hierarchy
├── trend_analysis_and_visualization.py        # Step 6: Analysis & visualization
│
├── biomedical_informatics_journals.csv        # Curated 29-journal list
│
└── data/
    ├── mesh_terms_annotated.json              # Step 1: GPT-annotated MeSH reference
    │
    ├── clusters/                              # Step 2 outputs
    │   ├── methodology_clusters.json
    │   ├── health_clusters.json
    │   ├── methodology_keyword_clusters.csv
    │   ├── health_keyword_clusters.csv
    │   ├── methodology_cluster_summary.csv
    │   ├── health_cluster_summary.csv
    │   ├── methodology_centroids.pkl
    │   ├── health_centroids.pkl
    │   ├── methodology_k_scan.csv             # K-value scan results
    │   ├── health_k_scan.csv
    │   ├── methodology_elbow_silhouette.png   # K selection diagnostic plots
    │   ├── health_elbow_silhouette.png
    │   ├── methodology_cluster_sizes.png
    │   ├── health_cluster_sizes.png
    │   └── clustering_metadata.json
    │
    ├── topics/                                # Step 3 outputs
    │   ├── methodology_topics.json
    │   ├── health_topics.json
    │   ├── methodology_topic_names.json
    │   ├── health_topic_names.json
    │   ├── methodology_topic_summary.csv
    │   ├── health_topic_summary.csv
    │   └── *_removed_abbreviation_clusters.json
    │
    ├── hierarchy/                             # Step 4 outputs
    │   ├── methodology_final_topics.json      # Deduplicated topics
    │   ├── health_final_topics.json
    │   ├── methodology_final_topic_summary.csv
    │   ├── health_final_topic_summary.csv
    │   ├── methodology_hierarchy.txt          # Indented hierarchy tree
    │   ├── health_hierarchy.txt
    │   ├── methodology_edges.json             # Parent-child edge list
    │   └── health_edges.json
    │
    └── visualizations/                        # Step 6 outputs (18 plots + 5 CSVs)
        ├── rq1_total_volume.png
        ├── rq2_methodology_topic_trends.png
        ├── rq2_health_topic_trends.png
        ├── rq3_cooccurrence_heatmap.png
        ├── rq4_methodology_topics_by_period.png
        ├── rq4_health_topics_by_period.png
        ├── n1_journal_volume_comparison.png
        ├── n1_journal_trends.png
        ├── n2_methodology_journal_specialization.png
        ├── n2_health_journal_specialization.png
        ├── n3_methodology_topic_sharing.png
        ├── n3_health_topic_sharing.png
        ├── n4_methodology_topic_emergence.png
        ├── n4_health_topic_emergence.png
        ├── n5_methodology_rising_declining.png
        ├── n5_health_rising_declining.png
        ├── n6_journal_similarity_network.png
        ├── n7_methodology_keyword_composition.png
        ├── n7_health_keyword_composition.png
        ├── methodology_topic_year_counts.csv
        ├── health_topic_year_counts.csv
        ├── methodology_trend_analysis.csv
        ├── health_trend_analysis.csv
        └── cooccurrence_matrix.csv
```

**Excluded from repo (regenerable from pipeline scripts):**
- `data/keyword_embeddings.pkl` (267 MB) — BiomedBERT embeddings, regenerable via Step 1
- `data/pubmed_all_journals_2011_2025_w_keywords.csv` (157 MB) — raw articles, regenerable via Step 0
- `data/mesh_embeddings.pkl` (6 MB) — MeSH embeddings, regenerable
- `data/mesh_tree_raw.bin` — MeSH 2024 raw download
- All `*.progress.json` and `*_progress` intermediate state files
- All `*.log`, `*.txt`, `nohup.out` logs

---

## Installation

**Requirements:** Python 3.10+, ~5 GB disk for embeddings, GPU recommended (but not required) for BiomedBERT inference.

```bash
# Clone the repo
git clone https://github.com/<your-username>/multi-journal-trend-analysis.git
cd multi-journal-trend-analysis

# Install dependencies
pip install openai pydantic torch transformers sentence-transformers \
            scikit-learn pandas numpy matplotlib seaborn networkx tqdm \
            requests nltk

# Download NLTK data (for Porter stemmer)
python -c "import nltk; nltk.download('punkt')"
```

You will need:
- An **OpenAI API key** (for GPT-based steps 1, 3, 4) — gpt-5-nano is recommended for cost efficiency
- An **NCBI email address** (required by NCBI policy for PubMed E-utilities)
- An optional **NCBI API key** for higher rate limits (10 req/s vs 3 req/s)

---

## Usage

Each script is independent and supports `--help` for parameter documentation.

### Full Pipeline Run

```bash
# Step 0: Collect PubMed data (~6-12 hours depending on rate limits)
python collect_pubmed_data.py \
    --journals biomedical_informatics_journals.csv \
    --output data/pubmed_all_journals_2011_2025_w_keywords.csv \
    --email your_email@example.com \
    --api-key YOUR_NCBI_API_KEY

# Step 1: Categorize keywords (~3-6 hours, GPT-heavy)
python categorize_keywords.py \
    --input data/pubmed_all_journals_2011_2025_w_keywords.csv \
    --output data/keywords_categorized.csv \
    --openai-api-key sk-... \
    --model gpt-5-nano

# Step 2: Cluster keywords (~5-10 minutes)
python cluster_keywords.py \
    --categorized data/keywords_categorized.csv \
    --embedding-cache data/keyword_embeddings.pkl \
    --output-dir data/clusters \
    --k 100

# Step 3: Cluster post-processing & naming (~1-3 hours, GPT-heavy)
python postprocess_and_name_topics.py \
    --cluster-dir data/clusters \
    --articles-csv data/pubmed_all_journals_2011_2025_w_keywords.csv \
    --embedding-cache data/keyword_embeddings.pkl \
    --output-dir data/topics \
    --openai-api-key sk-... \
    --model gpt-5-nano

# Step 4: Deduplication & hierarchy (~30-60 minutes, GPT-heavy)
python dedup_and_hierarchy.py \
    --topics-dir data/topics \
    --output-dir data/hierarchy \
    --openai-api-key sk-... \
    --model gpt-5-nano

# Step 6: Trend analysis & visualization (~2-5 minutes, no GPT)
python trend_analysis_and_visualization.py \
    --articles-csv data/pubmed_all_journals_2011_2025_w_keywords.csv \
    --topics-dir data/hierarchy \
    --output-dir data/visualizations
```

### Resumability

All long-running steps (1, 3, 4) save progress incrementally and can be safely interrupted/resumed by re-running the same command. Step 2 is fast enough that resumability isn't needed.

### Server Execution (recommended)

Use `nohup` to run scripts in the background:

```bash
nohup python3 step_script.py [args] > step_log.log 2>&1 &
tail -f step_log.log    # monitor progress
```

---

## Output Files

### Step 4 final outputs (most important for interpretation)

- **`data/hierarchy/methodology_final_topics.json`** — 116 methodology topics with name, keyword list, keyword counts, and constituent sub-topic IDs
- **`data/hierarchy/health_final_topics.json`** — 95 health topics
- **`data/hierarchy/methodology_hierarchy.txt`** — Indented tree of methodology topic hierarchy (root → leaves)
- **`data/hierarchy/health_hierarchy.txt`** — Health topic hierarchy
- **`data/hierarchy/*_edges.json`** — Parent-child edge lists for programmatic use

### Step 6 visualizations (18 plots)

| File | Description |
|---|---|
| `rq1_total_volume.png` | Total publication volume across all 29 journals over time |
| `rq2_methodology_topic_trends.png` | Top 16 methodology topics — dual-axis count + percentage trends |
| `rq2_health_topic_trends.png` | Top 16 health topics — dual-axis trends |
| `rq3_cooccurrence_heatmap.png` | 15×15 methodology × health co-occurrence heatmap |
| `rq4_methodology_topics_by_period.png` | Top 20 methodology topics by 5-year periods (stacked bars) |
| `rq4_health_topics_by_period.png` | Top 20 health topics by 5-year periods |
| `n1_journal_volume_comparison.png` | Top 15 journals ranked by total publication volume |
| `n1_journal_trends.png` | Per-journal trend lines (top 10) |
| `n2_methodology_journal_specialization.png` | Journal × methodology topic heatmap (% of journal's articles) |
| `n2_health_journal_specialization.png` | Journal × health topic heatmap |
| `n3_methodology_topic_sharing.png` | How many journals each top methodology topic appears in |
| `n3_health_topic_sharing.png` | Cross-journal sharing for health topics |
| `n4_methodology_topic_emergence.png` | First-appearance timeline for methodology topics |
| `n4_health_topic_emergence.png` | First-appearance timeline for health topics |
| `n5_methodology_rising_declining.png` | Top 10 rising and declining methodology topics (by trend slope) |
| `n5_health_rising_declining.png` | Top 10 rising and declining health topics |
| `n6_journal_similarity_network.png` | Network graph of journals connected by topic-distribution similarity |
| `n7_methodology_keyword_composition.png` | Stacked area charts of keyword composition within top methodology topics |
| `n7_health_keyword_composition.png` | Stacked area charts for top health topics |

### Step 6 data tables (5 CSVs)

| File | Description |
|---|---|
| `methodology_topic_year_counts.csv` | Per-topic per-year article counts (matrix form) |
| `health_topic_year_counts.csv` | Same for health topics |
| `methodology_trend_analysis.csv` | Per-topic linear regression results (slope, relative slope, mean count, recent vs. early counts) |
| `health_trend_analysis.csv` | Same for health topics |
| `cooccurrence_matrix.csv` | Methodology × health co-occurrence counts |

---

## Key Findings

### Field-Wide Trends

- **Publication volume grew ~8x** from ~1,200 articles/year (2011) to ~10,500/year (2025), with sharp acceleration post-2018 driven by the AI/deep learning boom and the COVID-19 pandemic.
- **Three innovation waves visible:** genomics era (2011–2015), machine learning surge (2016–2020), LLM/explainable AI era (2021–2025).
- **Large Language Models** are visibly rising in the keyword composition of the Deep Learning topic since 2023.

### Top Methodology Topics

Machine Learning Methods (17,706 articles, +217.7/year), Digital Research Methods (15,022), Digital Health (14,215), Human–Computer Interaction (13,064), Deep Learning Architectures (8,913, +107/year), Recommender Algorithm (8,242), Healthcare Modeling (7,993), Natural Language Processing (6,928).

**Fastest growth (relative):** Contrastive Learning (+24.4%/year), Attention Mechanisms (+19.8%/year) — both reflecting cutting-edge AI architecture trends.

### Top Health Topics

Digital Health (44,501 articles, +310/year — overwhelmingly dominant, 3x the next topic), Computational Biology (14,363), Healthcare Semantic Interoperability (8,799), Mental Health (8,143, +80/year), Gene Expression Regulation (8,036), Healthcare Outcomes (6,179), Primary Care (5,781), Clinical Prognosis (5,421).

**Notable risers:** Explainable Medical AI (+47/year), Privacy-preserving Healthcare Data (+54/year), COVID-19 Pandemic (emerged 2020).

### Methodology × Health Co-occurrence

- **Strongest pairing:** Human–Computer Interaction × Digital Health (12,170 articles)
- **Machine Learning is broadly applied across all health domains** — not siloed
- **Sequence Alignment remains highly concentrated within Computational Biology** — methodologically isolated from clinical informatics

### Multi-Journal Insights (New)

- **Three distinct journal clusters emerged:**
  1. Clinical informatics (JAMIA, BMC MedInform, IJMI, JBI, JMIR MedInform)
  2. Digital/consumer health (JMIR, JMIR mHealth, npj Digital Medicine, Frontiers in Digital Health)
  3. Computational biology (Bioinformatics, Briefings in Bioinformatics, Nature Methods, Database)
- **IEEE J-BHI bridges** between clinical and computational clusters
- **Journals show extreme specialization:** Bioinformatics allocates 40–48% of content to Sequence Alignment/Computational Biology; JMIR allocates 40–63% to Digital Health
- **Despite specialization, most methodology topics appear in 27–28 of 29 journals**, indicating a shared methodological vocabulary across the field

---

## Comparison with Original Paper

| Dimension | Original Paper (JBI only) | This Project (29 Journals) |
|---|---|---|
| Journals | 1 | 29 |
| Time range | 2004–2024 | 2011–2025 |
| Articles | 2,427 | 77,568 |
| Initial K | 750 | 100 (silhouette-selected) |
| Methodology topics | 2,276 | 116 |
| Health topics | 1,687 | 95 |
| Median keywords/topic | 6 | ~348 (methodology), ~571 (health) |

The smaller number of final topics in the multi-journal version reflects the lower K value (chosen via silhouette analysis), which produces broader, higher-level research themes suitable for cross-journal comparison rather than fine-grained sub-topic identification.

**Key contributions of the multi-journal expansion:**

1. **Cross-journal generalizability:** The AI transformation observed in JBI is field-wide. Machine Learning appears in 28 of 29 journals.
2. **Journal ecosystem structure:** Distinct sub-communities (clinical informatics, digital/consumer health, computational biology) emerge that are invisible in single-journal analysis.
3. **Scale-driven topic granularity:** Captures broader themes at higher abstraction levels for cross-journal comparability.

---

## Data Sources

- **PubMed E-utilities** (`esearch`, `efetch`) — article metadata
- **MeSH 2024** tree from NLM — methodology vs. health reference exemplars
- **microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext** — keyword embeddings (768-dim)
- **sentence-transformers/all-mpnet-base-v2** — topic name similarity embeddings
- **OpenAI gpt-5-nano** — keyword classification, cluster coherence checking, topic naming, relationship extraction
- **DOI publisher pages** (IEEE Xplore, Nature, Springer, Elsevier) — keyword fallback scraping

---

## Reference

Methodology based on:

> Liang et al., *Generative AI–Driven Analysis of Research Trends in Biomedical Informatics* (single-journal JBI analysis).

This project extends the original methodology to a 29-journal multi-journal corpus and adds 7 new analyses (N1–N7) specifically designed for cross-journal comparison.

---

## License

Research use only. Please cite the original paper if using this methodology in your own work.
