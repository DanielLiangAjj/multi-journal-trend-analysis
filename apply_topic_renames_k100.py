"""
Apply K=100 topic renames discovered during the multi-agent audit.

Rename map for the K=100 methodology and health hierarchies, mirroring
apply_topic_renames.py for K=750.
"""
import json
import shutil
from pathlib import Path

DATA = Path("/Users/danielliang/Dropbox/multi_journal_trend_analysis/data")
HIER = DATA / "hierarchy_k100"

METHODOLOGY_RENAMES = [
    ("Web-based bioinformatics tools", "Genomics and sequence bioinformatics"),
    ("Wearable sensors", "Mobile health and social media"),
    ("Digital health dashboards", "eHealth and mobile applications"),
    ("Multimodal information fusion", "Mixed deep learning applications"),
    ("Neuropsychological assessment", "Neuroscience and cognitive psychology"),
    ("Clinical outcome assessment", "Drug discovery and evaluation"),
    ("Causal inference methods", "Bayesian and statistical modeling"),
    ("differential privacy", "Privacy and de-identification"),
    ("Data warehouse", "Big data infrastructure and interoperability"),
    ("Data normalization", "Mass spectrometry and ROC analysis"),
    ("Corpus construction", "Generative AI and corpus methods"),
    ("Meta learning", "Multi-task and meta learning"),
    ("Design science research", "Software and usability research"),
    ("Information quality assessment", "Data governance and standards"),
    ("Recruitment methods", "Qualitative research methods"),
    ("Process optimization", "Mixed automation and optimization methods"),
    ("Acoustic propagation", "Ultrasound and echocardiography"),
    ("Environmental scanning", "Clinical monitoring and surveillance"),
    ("Robust domain generalization", "Transfer learning variants"),
    ("Inverse problems", "Mixed mathematical and geometric methods"),
    ("Graph-based clustering", "Graph and network methods"),
    ("Explainable ML", "Mixed reasoning and modeling methods"),
    ("Stochastic diffusion dynamics", "Diffusion generative models"),
    ("Measurement theory", "Mixed psychological and measurement theories"),
    ("Feminist narrative inquiry", "Mixed narrative and language studies"),
    ("Data analysis", "Clinical study designs"),
    ("Computational fluid dynamics", "Mixed engineering and imaging acronyms"),
    ("Clinical phenotyping methods", "Mixed clinical NLP applications"),
    ("Generative modeling", "Predictive and Markov modeling"),
    ("Patient participation", "Research ethics and patient involvement"),
    ("Pilot studies", "Pilot studies and synthetic data"),
]

HEALTH_RENAMES = [
    ("CRISPR-Cas systems", "Repetitive DNA sequences"),
    ("Survival analysis", "Prognosis and risk prediction"),
    ("Health data bias", "Mixed health informatics safety topics"),
    ("Clinical pathology testing", "Cancer types and screening"),
    ("Computational physiology", "Physiological signals and AI methods"),
    ("University health research", "Translational and clinical research"),
    ("Prostate cancer", "Cancer treatment modalities"),
    ("Ovarian health", "Maternal and reproductive health"),
    ("Lymph node metastasis", "Tumor immunology and microenvironment"),
    ("Cerebrovascular disease", "Cardiovascular diseases"),
    ("Cardiac remodeling patterns", "Cardiovascular hemodynamics"),
    ("Bio-inspired optimization", "Unclassified keyword cluster"),
    ("Microbial phylogenomics", "Microbial and fungal genomics"),
    ("TCR-pMHC binding", "Protein-peptide binding"),
    ("Patient-reported outcomes", "Health and functional status assessment"),
    ("Medicinal plants", "Natural products and traditional medicine"),
    ("One Health", "Environmental biota and species"),
    ("Health equity", "Countries and demographics"),
    ("Angiogenesis focus", "Cell signaling and oncogenic pathways"),
    ("Cell type analysis", "Mixed terminology and histopathology"),
    ("Immune repertoire sequencing", "Sequence and structure prediction"),
    ("Chronic disease management", "Primary care delivery"),
    ("Medical conditions", "Mixed clinical conditions and demographics"),
    ("aaa growth", "Unclassified keyword cluster"),
    ("Ophthalmic image analysis", "Mixed medical imaging"),
    ("Quality of care", "Caregivers and care quality"),
    ("Biomedical graph learning", "Protein and gene networks"),
    ("Attention mechanisms", "Neuroscience and cognition"),
    ("Eye diseases", "Eye, ENT, and oral diseases"),
    ("Protein phosphorylation", "Post-translational modifications"),
]


def apply_renames(domain, renames):
    topics_path = HIER / f"{domain}_final_topics.json"
    edges_path = HIER / f"{domain}_edges.json"

    bk_topics = topics_path.with_suffix(".json.bak_prerename")
    bk_edges = edges_path.with_suffix(".json.bak_prerename")
    if not bk_topics.exists():
        shutil.copy2(topics_path, bk_topics)
        shutil.copy2(edges_path, bk_edges)
        print(f"  Backed up {domain} JSONs")

    with open(topics_path) as f:
        topics = json.load(f)
    with open(edges_path) as f:
        edges = json.load(f)

    rename_map = dict(renames)
    unseen, applied = [], 0
    new_topics = {}
    for old_name, entry in topics.items():
        new_name = rename_map.get(old_name, old_name)
        if new_name != old_name:
            applied += 1
        if isinstance(entry, dict) and "name" in entry:
            entry["name"] = new_name
        if new_name in new_topics:
            print(f"  COLLISION: {old_name} -> {new_name} already exists. Skipping.")
            continue
        new_topics[new_name] = entry

    for old_name in rename_map:
        if old_name not in topics:
            unseen.append(old_name)

    new_edges = [
        [rename_map.get(a, a), rename_map.get(b, b)] for a, b in edges
    ]
    new_edges = [e for e in new_edges if e[0] != e[1]]

    with open(topics_path, "w") as f:
        json.dump(new_topics, f, indent=2)
    with open(edges_path, "w") as f:
        json.dump(new_edges, f, indent=2)

    print(f"  {domain}: renamed {applied}/{len(rename_map)} topics, "
          f"{len(new_topics)} total, {len(new_edges)} edges")
    if unseen:
        print(f"  {domain} rename keys not found: {unseen}")


if __name__ == "__main__":
    apply_renames("methodology", METHODOLOGY_RENAMES)
    apply_renames("health", HEALTH_RENAMES)
    print("\nDone. Original JSONs backed up with .bak_prerename suffix.")
