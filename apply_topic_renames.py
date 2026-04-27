"""
Apply curated topic renames to hierarchy_k750 JSON files.

The audit found 64 topics where GPT-5-nano's assigned name did not match the
underlying keyword cluster (e.g., "Large language models" cluster contained
keywords about natural human languages, not neural LLMs).

This script updates both the dict keys and the "name" field in
{methodology,health}_final_topics.json and the edge tuples in
{methodology,health}_edges.json so downstream Step 6 and PI analyses pick up
the corrected names.
"""
import json
import os
import shutil
from pathlib import Path

DATA = Path("/Users/danielliang/Dropbox/multi_journal_trend_analysis/data")
HIER = DATA / "hierarchy_k750"

# (old, new) pairs. Order matters only in that each old name must be unique.
METHODOLOGY_RENAMES = [
    ("Haptic technology", "Mixed compute hardware and acquisition"),
    ("bci painting system", "Web APIs and applications"),
    ("audio-based aal applications", "Audio and multi-agent systems"),
    ("Lightweight networks", "Hardware accelerators (GPU/FPGA/ASIC)"),
    ("Trustworthy AI", "Attention and adversarial training methods"),
    ("Discriminant analysis", "Researcher profiling and data sharing"),
    ("information criteria", "Information-theoretic measures"),
    ("Distributed algorithms", "Secure and embedded computation"),
    ("Sensor-based instrumentation", "Sensors and wireless networks"),
    ("Sequence alignment", "Sequencing and assembly"),
    ("Large language models", "Multilingual text and distributional semantics"),
    ("Transformer-based methods", "Swin transformer with metaheuristic optimizers"),
    ("human error assessment and reduction technique (heart)",
     "User testing and qualitative evaluation"),
    ("Ensemble learning", "Mixed machine learning methods"),
    ("Multiple kernel learning", "Diverse supervised learning methods"),
    ("Chaotic maps", "Nonlinear dynamical systems"),
    ("Self-Organizing Maps", "Clustering methods"),
    ("Cough classification", "Medical image classification"),
    ("brute-force attack", "Digital attacks"),
    ("Liquid-liquid phase separation", "Signal and source separation"),
    ("Cross-domain generalization", "Domain adaptation and transfer learning"),
    ("Prompt learning", "Vision-language and Q&A methods"),
    ("Slime mould algorithm", "CNN architecture components"),
    ("equally weighted variables", "Unclassified keyword cluster"),
    ("Sample entropy", "Technical replicates and QC"),
    ("Observability", "Mixed CS and application keywords"),
]

HEALTH_RENAMES = [
    ("Neoantigen vaccines", "Mixed immunology and therapeutics"),
    ("N-of-1 trials", "Bayesian methods"),
    ("Middle East health", "Middle East region and language"),
    ("Osteoimmunology", "Head and neck anatomy"),
    ("poroelasticity", "Tissue biophysics"),
    ("Myelinated nerve fibers", "Movement and motor control"),
    ("cytoplasm", "Subcellular localization"),
    ("One Health", "Ecology and wildlife"),
    ("Raspberry Pi health", "Raspberry Pi hardware"),
    ("Pathogenicity islands", "Host-pathogen interactions"),
    ("Parasite detection", "Invertebrates and parasites"),
    ("Swallowing health", "ENT and upper airway diseases"),
    ("Syndrome differentiation", "Diagnosis and misdiagnosis"),
    ("Pain catastrophizing", "Pain types"),
    ("Celiac disease", "Gastrointestinal diseases"),
    ("multisensory stimulation", "Mixed physiology and interaction"),
    ("Health archetypes", "Frailty and life-space indices"),
    ("Phosphatase types", "Protein families"),
    ("Intrinsically disordered proteins", "Proteins and protein domains"),
    ("Smart contracts", "Blockchain and finance"),
    ("Domain transfer", "Genomic TADs and multimodal fusion"),
    ("Echinodermata health", "Unclassified keyword cluster"),
    ("Transformers in health", "Aerospace and transportation"),
    ("Space medicine", "Unclassified keyword cluster"),
    ("Digitalization Across Sectors", "Cross-sector digitalization"),
    ("ACE2 distribution", "Angiotensin-converting enzymes"),
    ("Growth hormone", "Mixed endocrine signals"),
    ("Symmetry in health", "Symmetry and asymmetry methods"),
    ("Bacterial infections", "Hand hygiene and infection prevention"),
    ("Target identification", "Drug targeting and transport"),
    ("Biosignal channels", "Cellular neurophysiology"),
    ("Nonverbal health cues", "Vital signs and physiological signals"),
    ("Fine-grained health data", "Fine-grained annotation methods"),
    ("panoptosis", "Cell death and stress"),
    ("Dating apps", "Mobile app types"),
    ("Infusion pump focus", "Cellular biomaterials"),
    ("Biomedical data science", "Fractional calculus and tensor methods"),
]


def apply_renames(domain, renames):
    """Rename keys in {domain}_final_topics.json and update edge tuples."""
    topics_path = HIER / f"{domain}_final_topics.json"
    edges_path = HIER / f"{domain}_edges.json"

    # Backup once
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
    unseen = []
    applied = 0

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
          f"{len(new_topics)} total topics remain, {len(new_edges)} edges")
    if unseen:
        print(f"  {domain} rename keys not found: {unseen}")


if __name__ == "__main__":
    apply_renames("methodology", METHODOLOGY_RENAMES)
    apply_renames("health", HEALTH_RENAMES)
    print("\nDone. Original JSONs backed up with .bak_prerename suffix.")
