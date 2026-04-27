# Next-steps commands

All commands assume:
- Local repo: `~/Dropbox/multi_journal_trend_analysis/`
- Server: `ly2680@aqua.dbmi.columbia.edu`
- Server repo: `/public/multi_journal_trend_analysis/`
- Patched articles CSV already on server:
  `data/pubmed_all_journals_2011_2025_w_keywords_patched.csv`
- Keyword categorization already done:
  `data/keywords_categorized.csv`

## 1. Push the updated scripts to the server

```bash
# Run these locally
cd ~/Dropbox/multi_journal_trend_analysis
scp trend_analysis_and_visualization.py analyze_all_questions.py \
    ly2680@aqua.dbmi.columbia.edu:/public/multi_journal_trend_analysis/
```

## 2. Re-run PI Q1–Q7 on the patched K=750 dataset

```bash
# SSH onto the server first
ssh ly2680@aqua.dbmi.columbia.edu
cd /public/multi_journal_trend_analysis

# Re-run PI questions using K=750 topic assignments
nohup python analyze_all_questions.py \
    --articles-csv data/pubmed_all_journals_2011_2025_w_keywords_patched.csv \
    --authors-csv data/articles_with_authors.csv \
    --topics-dir data/hierarchy_k750 \
    --output-dir data/visualizations_k750 \
    > logs/analyze_all_k750.log 2>&1 &
```

When it finishes, download the new PI outputs:

```bash
# Locally
cd ~/Dropbox/multi_journal_trend_analysis
scp "ly2680@aqua.dbmi.columbia.edu:/public/multi_journal_trend_analysis/data/visualizations_k750/pi_q*" \
    data/visualizations_k750/
scp ly2680@aqua.dbmi.columbia.edu:/public/multi_journal_trend_analysis/logs/analyze_all_k750.log \
    logs/
```

## 3. Re-run Step 6 (trend visualizations) for K=100 and K=750 with N4/N5 fixes

The pipeline script now generates correct N4/N5 plots, so re-running Step 6
against existing K=100 and K=750 topic assignments is enough — no re-clustering.

```bash
# On server
cd /public/multi_journal_trend_analysis

# K=750
nohup python trend_analysis_and_visualization.py \
    --articles-csv data/pubmed_all_journals_2011_2025_w_keywords_patched.csv \
    --topics-dir data/hierarchy_k750 \
    --output-dir data/visualizations_k750 \
    > logs/step6_k750_rerun.log 2>&1 &

# K=100
nohup python trend_analysis_and_visualization.py \
    --articles-csv data/pubmed_all_journals_2011_2025_w_keywords_patched.csv \
    --topics-dir data/hierarchy_k100 \
    --output-dir data/visualizations_k100 \
    > logs/step6_k100_rerun.log 2>&1 &
```

Download the updated N4/N5 plots:

```bash
# Locally
scp "ly2680@aqua.dbmi.columbia.edu:/public/multi_journal_trend_analysis/data/visualizations_k750/n{4,5}_*" \
    data/visualizations_k750/
scp "ly2680@aqua.dbmi.columbia.edu:/public/multi_journal_trend_analysis/data/visualizations_k100/n{4,5}_*" \
    data/visualizations_k100/
```

## 4. Run the K=300 pipeline (intermediate granularity)

```bash
# On server
cd /public/multi_journal_trend_analysis

# Step 2: cluster at K=300
nohup python cluster_keywords.py \
    --keywords-csv data/keywords_categorized.csv \
    --embeddings-cache data/keyword_embeddings.pkl \
    --output-dir data/clusters_k300 \
    --method-k 300 --health-k 300 \
    > logs/step2_k300.log 2>&1 &

# When step 2 finishes: Step 3 (post-process + name topics)
nohup python postprocess_and_name_topics.py \
    --clusters-dir data/clusters_k300 \
    --output-dir data/topics_k300 \
    --openai-api-key "$OPENAI_API_KEY" \
    > logs/step3_k300.log 2>&1 &

# When step 3 finishes: Step 4 (dedup + hierarchy)
nohup python dedup_and_hierarchy.py \
    --topics-dir data/topics_k300 \
    --output-dir data/hierarchy_k300 \
    --openai-api-key "$OPENAI_API_KEY" \
    > logs/step4_k300.log 2>&1 &

# When step 4 finishes: Step 6 (trend viz)
nohup python trend_analysis_and_visualization.py \
    --articles-csv data/pubmed_all_journals_2011_2025_w_keywords_patched.csv \
    --topics-dir data/hierarchy_k300 \
    --output-dir data/visualizations_k300 \
    > logs/step6_k300.log 2>&1 &

# PI Q1–Q7
nohup python analyze_all_questions.py \
    --articles-csv data/pubmed_all_journals_2011_2025_w_keywords_patched.csv \
    --authors-csv data/articles_with_authors.csv \
    --topics-dir data/hierarchy_k300 \
    --output-dir data/visualizations_k300 \
    > logs/analyze_all_k300.log 2>&1 &
```

Download when done:

```bash
# Locally
mkdir -p data/visualizations_k300 data/hierarchy_k300 data/topics_k300 data/clusters_k300
scp -r ly2680@aqua.dbmi.columbia.edu:/public/multi_journal_trend_analysis/data/visualizations_k300/ data/
scp -r ly2680@aqua.dbmi.columbia.edu:/public/multi_journal_trend_analysis/data/hierarchy_k300/    data/
scp "ly2680@aqua.dbmi.columbia.edu:/public/multi_journal_trend_analysis/logs/*k300*"              logs/
```

## Notes

- **IEEE J-BHI affiliations**: confirmed the PubMed XML for ~99% of J-BHI records has
  `<Author>` with no `<AffiliationInfo>`. This is an IEEE metadata-submission gap,
  not a parser bug. `analyze_all_questions.py` now flags any journal with
  <10% affiliation coverage as NaN in `mean_aff` / `pct_inter` so J-BHI no longer
  appears as "0 collaboration." Author-count metrics remain valid.
- **Q7 COVID filter**: `pct_change_during`/`pct_change_post` are now NaN for
  topics with pre-COVID avg < 1 article/year, and topics with total < 50 are
  dropped entirely. That removes the 4900% / 34900% artifacts.
- **N4/N5 propagation**: fixes now live in the pipeline script itself, so every
  future run produces correct plots without needing `fix_all_visualizations.py`.
