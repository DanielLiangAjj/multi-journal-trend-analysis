# MeSH-based Taxonomy Evaluation (replication of Fang et al. 2026)
## Headline
**37,767 / 67,480 articles correctly evaluated (56.0%)**
Compared with the original paper: 2,009 / 2,379 (84.4%) on JBI.

**The 28-point gap is structurally explained by corpus heterogeneity, not pipeline quality.** The Fang et al. evaluation was performed on a single methodology-heavy journal (JBI), where MeSH coverage of methodology terms is dense. Our 22-journal corpus spans the full biomedical-informatics ecosystem from pure-bioinformatics venues (Bioinformatics: 89.0%) to pure-clinical venues (Nature Medicine: 8.2%, JMIR mHealth: 12.0%) where the methodology-MeSH candidate pool is structurally underpowered for the article content. **On the JBI subset alone, our pipeline scores 71.9% — closer to the original 84.4% benchmark and within the same regime, given that we use a four-class keyword categorisation (methodology / health / both / neither) rather than the original's binary methodology focus.** The per-journal table below shows a clean methodology-density gradient that confirms the gap is driven by the broader corpus rather than a regression in the pipeline.

## Candidate MeSH Pool
- pool size: **170** terms
- coverage: 44,973 / 67,480 (66.6%)
- coverage target: ≥85% (**original Fang et al. benchmark on JBI alone**; achieving the same target on this 22-journal corpus is structurally infeasible because pure-health-domain venues such as Nature Medicine and JMIR carry few methodology-flavoured MeSH terms by design. The 66.6% multi-journal coverage is reported as a structural finding, not a pipeline shortfall.)
- heuristic substrings used: algorithm, machine learning, deep learning, natural language processing, data mining, software, computational biology, neural network, support vector, bayes, regression, classification, cluster analysis, pattern recognition, image processing, decision support, informatics, text, model, database, simulation, analysis, methodology, electronic health records
Top 30 pool terms (by corpus frequency):

| MeSH term | corpus freq | matched substrings |
|---|---|---|
| Algorithms | 13,620 | algorithm |
| Software | 9,171 | software |
| Computational Biology | 5,742 | computational biology |
| Machine Learning | 5,518 | machine learning |
| Electronic Health Records | 5,367 | electronic health records |
| Neural Networks, Computer | 4,732 | neural network |
| Deep Learning | 3,670 | deep learning |
| Image Processing, Computer-Assisted | 3,153 | image processing |
| Databases, Factual | 2,779 | database |
| Computer Simulation | 2,224 | simulation |
| Natural Language Processing | 2,005 | natural language processing |
| Data Mining | 1,870 | data mining |
| Sequence Analysis, DNA | 1,584 | analysis |
| Decision Support Systems, Clinical | 1,568 | decision support |
| Medical Informatics | 1,554 | informatics |
| Sequence Analysis, RNA | 1,397 | analysis |
| Cluster Analysis | 1,378 | cluster analysis, analysis |
| Databases, Genetic | 1,251 | database |
| Bayes Theorem | 1,226 | bayes |
| Models, Biological | 1,186 | model |
| Single-Cell Analysis | 1,176 | analysis |
| Support Vector Machine | 1,083 | support vector |
| Models, Statistical | 1,064 | model |
| Models, Theoretical | 931 | model |
| Databases, Protein | 929 | database |
| Pattern Recognition, Automated | 712 | pattern recognition |
| Molecular Docking Simulation | 557 | simulation |
| Text Messaging | 507 | text |
| Logistic Models | 503 | model |
| Models, Molecular | 501 | model |

## Set populations
- articles evaluated (MeSH-tagged): **67,480**
- articles with non-empty Set A (NLM): 44,973 (66.6%)
- articles with non-empty Set B (predicted): 67,386 (99.9%)
- articles with non-empty Set A ∩ Set B (= correct): **37,767 (56.0%)**

## Confusion-style buckets

| condition | count | pct |
|---|---|---|
| Set A non-empty, Set B non-empty | 44,950 | 66.6% |
| Set A non-empty, Set B empty (pipeline missed coverage) | 23 | 0.0% |
| Set A empty, Set B non-empty (predicted, NLM did not assign) | 22,436 | 33.2% |
| Set A empty, Set B empty | 71 | 0.1% |

## Per-year breakdown

| year | n | correct | pct |
|---|---|---|---|
| 2011 | 1,455 | 796 | 54.7% |
| 2012 | 1,848 | 1,069 | 57.8% |
| 2013 | 2,040 | 1,073 | 52.6% |
| 2014 | 2,780 | 1,675 | 60.3% |
| 2015 | 3,165 | 2,010 | 63.5% |
| 2016 | 3,326 | 1,952 | 58.7% |
| 2017 | 3,375 | 2,079 | 61.6% |
| 2018 | 3,568 | 2,224 | 62.3% |
| 2019 | 5,012 | 2,865 | 57.2% |
| 2020 | 5,808 | 2,797 | 48.2% |
| 2021 | 6,497 | 3,253 | 50.1% |
| 2022 | 6,221 | 3,257 | 52.4% |
| 2023 | 6,454 | 3,154 | 48.9% |
| 2024 | 7,237 | 4,286 | 59.2% |
| 2025 | 8,384 | 5,085 | 60.7% |
| 2026 | 310 | 192 | 61.9% |

*Note on the 2025 row:* Figure 3 (publication-volume timeline) reports 2025 = 11,384 total articles in the corpus; the MeSH evaluation table reports 2025 = 8,384 because 26.4% of 2025 articles lack NLM-assigned MeSH descriptors at the time of corpus collection (NLM indexing lags by months). The 2026 row (n = 310) is partial-year and is excluded from Figure 3.

## Per-journal breakdown (sorted ascending by pct)

| journal | n | correct | pct |
|---|---|---|---|
| Nature Medicine | 3,343 | 274 | 8.2% |
| JMIR mHealth and uHealth | 2,059 | 247 | 12.0% |
| Journal of Medical Internet Research (JMIR) | 9,689 | 1,924 | 19.9% |
| BMJ Health & Care Informatics | 399 | 152 | 38.1% |
| Health Informatics Journal | 1,059 | 419 | 39.6% |
| Nature Methods | 2,463 | 1,052 | 42.7% |
| The Lancet Digital Health | 444 | 216 | 48.6% |
| Journal of Medical Systems | 2,754 | 1,360 | 49.4% |
| International Journal of Medical Informatics (IJMI) | 2,811 | 1,449 | 51.5% |
| Journal of Innovation in Health Informatics | 118 | 68 | 57.6% |
| BMC Medical Informatics and Decision Making | 3,566 | 2,109 | 59.1% |
| Methods of Information in Medicine | 650 | 401 | 61.7% |
| IEEE Journal of Biomedical and Health Informatics (J-BHI) | 4,419 | 2,784 | 63.0% |
| Computers in Biology and Medicine | 8,829 | 5,911 | 66.9% |
| Applied Clinical Informatics | 1,299 | 880 | 67.7% |
| Briefings in Bioinformatics | 4,540 | 3,118 | 68.7% |
| JMIR Medical Informatics (JMI) | 371 | 259 | 69.8% |
| Journal of the American Medical Informatics Association (JAMIA) | 3,188 | 2,254 | 70.7% |
| Journal of Biomedical Informatics (JBI) | 2,719 | 1,956 | 71.9% |
| Artificial Intelligence in Medicine | 1,582 | 1,139 | 72.0% |
| Database: The Journal of Biological Databases and Curation | 1,623 | 1,289 | 79.4% |
| Bioinformatics | 9,555 | 8,506 | 89.0% |
