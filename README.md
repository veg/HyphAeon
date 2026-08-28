<div align="center">

<img src="figures/hyphaeon_logo.png" alt="HyphAeon Logo" width="280"/>

# HyphAeon
### A Deep-Time Phylogenetic Foundation Model for Multi-Scale Evolutionary, Structural, and Clinical Genomics

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

</div>

---

**HyphAeon** is a deep-time phylogenetic foundation model designed to bridge computational phylogenetics, structural biology, and foundation AI. Built upon a 2D axial transformer backbone (**`PhyloAxialTransformer`**) with patristic distance-decay attention and classical multidimensional scaling (MDS) tree embeddings, HyphAeon ingests multi-species codon alignments and explicit evolutionary trees spanning 200 million years of deep time.

By replacing computationally prohibitive numerical likelihood optimizations with structured representation learning, HyphAeon unifies the gold-standard HyPhy ecosystem into a single, high-throughput foundation engine.

---

## 🌟 The Five Foundational Pillars of HyphAeon

```
                           ┌────────────────────────────────────────────────┐
                           │            HyphAeon Backbone                   │
                           │   PhyloAxialTransformer (Sequence x Tree)      │
                           └───────┬──────┬──────┬──────┬──────┬───────────┘
                                   │      │      │      │      │
       ┌───────────────────────────┼──────┼──────┼──────┼──────┴───────────────────────────┐
       ▼                           ▼      ▼      ▼      ▼                                  ▼
┌──────────────┐            ┌───────────┐ ┌───────────┐ ┌──────────────┐            ┌──────────────┐
│ 1. MEME Head │            │ 2. Co-Evo │ │ 3. Digital│ │ 4. BUSTED+S  │            │ 5. aBSREL    │
│ (Site Level) │            │ Epistasis │ │    DMS    │ │ Head (Gene)  │            │ Head (Branch)│
├──────────────┤            ├───────────┤ ├───────────┤ ├──────────────┤            ├──────────────┤
│ • Identifies │            │ • 3D pair │ │ • Zero-shot│ • Alignment-  │            │ • Episodic   │
│   per-codon  │            │   contact │ │   fitness │   wide omnibus │            │   selection  │
│   diversify- │            │ • CPD     │ │   landscape│  LRT & p-val  │            │   per branch │
│   ing sites  │            │   partners│ │ • Grantham│ • Synonymous   │            │ • Branch p_b │
│ • p_site     │            │   (s_comp)│ │   deltas  │   variance(SRV)│            │   in 5 ms    │
└──────────────┘            └───────────┘ └───────────┘ └──────────────┘            └──────────────┘
```

1. **Pillar 1: Site-Level Diversifying Selection (MEME \& FEL)**:
   * Accurately infers codon-specific selection rates and LRT statistics in under 5 milliseconds per alignment ($>10,000\times$ faster than classical numerical optimization).
2. **Pillar 2: Epistatic Co-Evolution \& 3D Structural Packing**:
   * Symmetrized column-attention maps recover true tertiary protein contact maps ($C_\beta - C_\beta < 8\text{\AA}$) with top-$L/5$ precision exceeding $78\%$.
3. **Pillar 3: Digital Deep Mutational Scanning (DMS) \& Clinical Epistasis**:
   * Resolves the *Deleterious Mutation Paradox* by mapping **9,871 Compensated Pathogenic Deviations (CPDs)** across 1,713 human Mendelian disease genes in ClinVar. De novo predicts exact 3D compensatory partners ($s_{\text{comp}}$).
4. **Pillar 4: Alignment-Wide Selection (BUSTED \& BUSTED+S)**:
   * Multi-query cross-attention pooling head evaluates whole-gene episodic diversification and filters Synonymous Rate Variation (SRV) false positives.
5. **Pillar 5: Branch-Site Episodic Selection (aBSREL)**:
   * Predicts lineage-specific episodic bursts across all branches of a 120-species phylogeny simultaneously in $<10\text{ ms}$.

---

## 🛠️ Installation & Dependencies

```bash
git clone https://github.com/veg/TOGA_MEME.git
cd TOGA_MEME
pip install -r requirements.txt
```

---

## 🚀 Quickstart: Running HyphAeon Inference

Run site-level and gene-level selection inference using the pre-trained `hyphaeon_core.pt` checkpoint:

```bash
python3 scripts/predict_regression_nexus.py \
  --model hyphaeon_core.pt \
  --alignment path/to/alignment.nex \
  --output predictions.csv
```

---

## 📄 Documentation & Manuscripts

* **Core Foundation Model Manuscript**: [`hyphaeon_paper/main.pdf`](hyphaeon_paper/main.pdf) / [`paper/hyphaeon_core_foundation_model.pdf`](paper/hyphaeon_core_foundation_model.pdf)
* **Architecture Specification**: [`docs/axomeme_architecture_spec.pdf`](docs/axomeme_architecture_spec.pdf)
* **Digital DMS & Clinical CPD Atlas**: [`scratch/compensated_pathogenic_deviations_atlas.csv`](scratch/compensated_pathogenic_deviations_atlas.csv)
* **Evolutionary Predictability Atlas**: [`scratch/dimension5_replay_of_life_atlas.csv`](scratch/dimension5_replay_of_life_atlas.csv)

---

## 📜 Citation

If you use **HyphAeon** in your research, please cite:

```bibtex
@article{hyphaeon2026,
  title={HyphAeon: A Deep-Time Phylogenetic Foundation Model for Multi-Scale Evolutionary, Structural, and Clinical Genomics},
  author={Kosakovsky Pond, Sergei L. and team},
  journal={Nature Methods / Nature Biotechnology (in submission)},
  year={2026}
}
```
