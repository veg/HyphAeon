# RhizAeon Documentation Suite

Welcome to the documentation and tutorial suite for **RhizAeon: Tree-Free Reticulate Evolution and Recombination Inference**.

RhizAeon bypasses the computational bottlenecks of classical phylogenetic trees and exhaustive triplet scans by embedding pairwise genetic distance matrices into a continuous metric space (the **Phylogenetic GPS**). Recombination is detected as continuous sequence trajectories migrating between parental clade manifolds in linear time ($\mathcal{O}(N^2 L)$).

---

## The Four Pedagogical Tracks

Following modern cognitive instructional design (**Diátaxis Framework**, **Cognitive Load Theory**, and **Minimalist Instruction**), the documentation is organized into **four synchronized, equivalent tracks** tailored to different cognitive modes, user proficiencies, and analytical objectives:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                RHIZAEON TUTORIAL SUITE                                 │
└────────────────────────────────────────────────────────────────────────────────────────┘
          │                                                    │
          ▼                                                    ▼
┌───────────────────────────────────┐        ┌───────────────────────────────────┐
│ Track 1: Visual Quickstart        │        │ Track 2: Practitioner How-To      │
│ (Show-First / Worked-Example)     │        │ (Diagnostic & Forensic Workflow)  │
│                                   │        │                                   │
│ • Audience: Evaluators, Biologists│        │ • Audience: Genomic Epidemiologists│
│ • TTFI: < 2 minutes               │        │ • Focus: Dirty data, QC, triage   │
│ • Link: track1_visual_quickstart  │        │ • Link: track2_practitioner_guide │
└───────────────────────────────────┘        └───────────────────────────────────┘
          │                                                    │
          ▼                                                    ▼
┌───────────────────────────────────┐        ┌───────────────────────────────────┐
│ Track 3: Algorithmic Derivation   │        │ Track 4: Conceptual Monograph     │
│ (Axiomatic Mathematical Build-Up) │        │ (Architectural Phase Space)       │
│                                   │        │                                   │
│ • Audience: Methodologists, Devs  │        │ • Audience: Power Users, Systems  │
│ • Focus: Eigenspaces, proofs, SVD │        │ • Focus: Phase boundaries, Sieve  │
│ • Link: track3_algorithmic_derivation│     │ • Link: track4_conceptual_monograph│
└───────────────────────────────────┘        └───────────────────────────────────┘
```

---

### Which Track Should You Read?

| If you are... | And your primary question is... | Start Here |
| :--- | :--- | :--- |
| **A busy biologist or evaluator** | *"What does this tool actually do, what does the output look like, and can I run it on standard data in five minutes?"* | **[Track 1: Fast-Track Worked-Example](track1_visual_quickstart.md)** |
| **An active analyst with real data** | *"How do I configure parameters, filter rate-heterogeneity false positives, diagnose messy signals, and disassemble my alignment?"* | **[Track 2: Diagnostic / Forensic Workflow](track2_practitioner_guide.md)** |
| **A methodologist or bioinformatician** | *"Where are the formal mathematical proofs, spectral bounds, algorithmic pseudocode, and asymptotic complexity guarantees?"* | **[Track 3: Deductive / Algorithmic Derivation](track3_algorithmic_derivation.md)** |
| **An evolutionary theorist or power user** | *"Why do naive approaches fail, what happens when frame of reference collapses, and how does reticulation link to selection?"* | **[Track 4: Conceptual Deep-Dive & Monograph](track4_conceptual_monograph.md)** |

---

## Biological Scope of RhizAeon

RhizAeon is organism-agnostic. It analyzes any aligned homologous genetic sequence dataset ($X \in \{A, C, G, T, -\}^{N \times L}$) experiencing reticulate evolution:
* **Viruses:** RNA viruses (HIV-1, SARS-CoV-2, Hepatitis C), segmented chimeras (Influenza A), DNA viruses (Herpesviruses, Geminiviruses).
* **Bacteria & Archaea:** Naturally transformable pathogens (*Streptococcus pneumoniae*, *Neisseria*), ICE elements, micro-conversions.
* **Eukaryotic Parasites & Fungi:** *Plasmodium falciparum* drug-resistance cassettes, pathogenic fungal chimeras (*Candida*, *Cryptococcus*), *Saccharomyces* hybrids.
* **Organellar Genomes:** Recombining plant chloroplasts and fungal mitochondria.
* **Targeted Eukaryotic Loci:** Host gene families undergoing meiotic crossing-over or non-allelic gene conversion (Major Histocompatibility Complex [MHC/HLA], opsin clusters, immunoglobulins).

---

## Core Command Quick Reference

```bash
# 1. Ultra-fast linear manifold screening
rhizaeon scan alignment.fasta --window 25

# 2. Recursive Partitioning FDA with Single-Base ML Polishing & Alignment Disassembly
rhizaeon rp-fda alignment.fasta \
    --min-len 200 \
    --polish-ml \
    --export-partitions ./partitions/ \
    --export-nexus partitioned.nex \
    --export-hyphy-json partitions.json \
    --html dashboard.html

# 3. Standalone Interactive Visualizer
rhizaeon visualize alignment.fasta --output dashboard.html
```
