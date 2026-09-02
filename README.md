<div align="center">

<img src="assets/hyphaeon_logo.png" alt="HyphAeon Logo" width="280"/>

# HyphAeon
### A Deep-Time Phylogenetic Foundation Model for Multi-Scale Evolutionary, Structural, and Clinical Genomics

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

</div>

---

**HyphAeon** is a deep-time phylogenetic foundation model designed to bridge computational phylogenetics, structural biology, and foundation AI. Built upon a 2D axial transformer backbone (**`PhyloAxialTransformer`**) with patristic distance-decay attention and classical multidimensional scaling (MDS) tree embeddings, HyphAeon ingests multi-species codon alignments and explicit evolutionary trees spanning 200 million years of deep time.

---

## 🚀 Key Capabilities & Unified Commands

HyphAeon integrates five complementary phylogenetic deep learning and geometric
projection engines, plus a pooled MEME concordance workflow:

1. **`hyphaeon meme` (Site-Level Diversifying Selection)**:
   Neural episodic positive selection inference ($>100\times$ faster than standard numerical MLE and codon-MCMC models like HyPhy MEME/FEL) using Tree-RoPE 4D geometric branch embeddings and axial tree attention.
2. **`hyphaeon epistasis` (3D Co-Evolution & Epistatic Sectors)**:
   Multi-scale epistatic sector mining implementing phylogenetic branch attribution, exact tree hypergeometric tests, Jaccard overlap suppression, and contact map recovery ($C_\beta - C_\beta < 8\text{\AA}$).
3. **`hyphaeon dms` (Digital Deep Mutational Scanning & CPDs)**:
   In silico Selection Deep Mutational Scanning. Performs high-throughput sweeps of all 19 alternative amino acids across every codon position in seconds, calculating the **Epistatic Selection Sensitivity Matrix (ESSM)**, Intrinsic Mutational Plasticity ($\mathbf{E}_{i,i}$), and de novo predicting compensatory partners ($s_{\text{comp}}$) that rescue human disease mutations (Compensated Pathogenic Deviations).
4. **`hyphaeon busted` (Alignment-Wide Omnibus Selection)**:
   Multi-query cross-attention pooling head that evaluates whole-gene episodic selection and filters Synonymous Rate Variation (SRV) false positives in milliseconds.
5. **`hyphaeon phenotype` (PhyloWAS)**:
   Directional phenotype-genotype association mapping on the unit hypersphere $\mathbb{S}^{M-1}$. Computes spectral trait energies ($\Psi_{\text{Spectral}}$), exact sequenced-taxa null scaling $p$-values, Benjamini-Hochberg FDR $q$-values, and **Phenotype-Associated Residue Signatures (PARS)**.

---

## 📦 Installation

```bash
git clone https://github.com/veg/hyphaeon.git
cd hyphaeon
pip install -e .
```

---

## 📂 Included Benchmark Datasets

All example alignments and phylogenetic trees required to reproduce these analyses are bundled directly in the `examples/` directory:

| Dataset | Alignment File | Tree File | Taxa ($N$) | Codons ($L$) | Description & Biological Domain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HIV-1 RT** | [`examples/HIV1_RT.fasta`](examples/HIV1_RT.fasta) | [`examples/HIV1_RT.nwk`](examples/HIV1_RT.nwk) | 476 | 335 | Retroviral Reverse Transcriptase polymerase domain (drug resistance & epistasis). |
| **Rhodopsin** | [`examples/RHO.fasta`](examples/RHO.fasta) | Embedded / Auto | 710 | 349 | Mammalian Rhodopsin visual pigments (deep-sea diving sensory adaptation). |
| **Smc6** | [`examples/Smc6.fasta`](examples/Smc6.fasta) | [`examples/Smc6.nwk`](examples/Smc6.nwk) | 20 | 1,097 | Primate Smc6 structural maintenance of chromosomes (antiviral host restriction). |
| **Bat OAS1** | [`examples/bat_oas1.fasta`](examples/bat_oas1.fasta) | [`examples/bat_oas1.nwk`](examples/bat_oas1.nwk) | 18 | 351 | Chiropteran OAS1 2'-5'-oligoadenylate synthetase (innate immunity escape). |
| **Camelid VHH** | [`examples/camelid.fasta`](examples/camelid.fasta) | [`examples/camelid.nwk`](examples/camelid.nwk) | 212 | 96 | Camelid single-domain antibody heavy-chain variable domain (antigenic diversity). |

---

## 🔬 Reproducible Benchmark Examples

### Example 1: Inter-Site Epistasis & Branch Co-Selection in HIV-1 Reverse Transcriptase

```bash
# Run branch co-selection, sector mining, and export co-selection network
hyphaeon epistasis -a examples/HIV1_RT.fasta -t examples/HIV1_RT.nwk -o examples/HIV1_RT_epistasis.json -c examples/HIV1_RT_edges.csv --graphml examples/HIV1_RT_coselection.graphml
```

#### Key Biological Discoveries:
1. **Unsupervised Discovery of Multi-Drug Catalytic Complexes (Q151M MDR Complex)**:
   * HyphAeon places the co-evolution of residue 116 with residue 151 at **#1 overall** across all candidate pairs:
     $$\text{F116} \longleftrightarrow \text{Q151} \quad (\text{Co-Sel} = 0.8660, \; p_{\text{hyper}} = 7.02 \times 10^{-9}, \; \text{FDR } q = 1.17 \times 10^{-7})$$
2. **Autonomous Dissection of Mutually Exclusive Pathways (TAM-1 vs. TAM-2)**:
   * HyphAeon's branch co-selection metric autonomously isolates the **TAM-1 triad** (`M41L + L210W + T215Y`, $q < 10^{-7}$) from the mutually antagonistic **TAM-2 cluster** (`D67N + K70R + K219Q`, $q < 10^{-3}$).

---

### Example 2: In Silico Selection Deep Mutational Scanning (Digital DMS / ESSM)

```bash
# Run digital DMS sweep on HIV-1 RT
hyphaeon dms -a examples/HIV1_RT.fasta -t examples/HIV1_RT.nwk -o examples/HIV1_RT_dms.json -c examples/HIV1_RT_dms.csv
```

---

### Example 3: Convergent Sensory Adaptation & Spectral Tuning in Rhodopsin

```bash
# Run PhyloWAS for marine diving mammal visual adaptation
hyphaeon phenotype -a examples/RHO.fasta -fg "turTru,balMus,balPhys,orcOrc,delDelp,phyCat,phoVit,halGryp,mirLeo,zalCali,odoRos" -o examples/RHO_marine_phenotype.json -c examples/RHO_marine_sites.csv
```

---

### Example 4: Ultra-Fast Episodic Positive Selection (`predict`), Feature Attribution (`--attribute`), & Alignment Error Filtering (`--filter`)

```bash
# Standard per-codon episodic selection inference
hyphaeon meme -a examples/Smc6.fasta -t examples/Smc6.nwk -o examples/Smc6_results.json -c examples/Smc6_results.csv

# Enable mechanistic feature attribution (identifies driving species & evolutionary timing)
hyphaeon meme -a examples/Smc6.fasta -t examples/Smc6.nwk --attribute --attribution-min-lrt 3.84 -o examples/Smc6_attributed.json

# Run inference with automated dual-stage alignment error filtering & export cleaned alignment
hyphaeon meme -a examples/Smc6.fasta -t examples/Smc6.nwk --filter --filter-out-aln examples/Smc6_cleaned.fasta -c examples/Smc6_clean.csv
```

#### 1. Mechanistic Feature Attribution (`--attribute`):
* **Single-Taxon Counterfactual Perturbation ($\Delta\text{LRT}$)**: In silico mutates each non-consensus species back to ancestral state to rank driving taxa by marginal selection evidence explained ($\%\text{ Signal Explained}$).
* **Evolutionary Epoch Decomposition**: Classifies selection timing by weighted root patristic depth into **Recent Terminal / Tip Sweep** ($\ge 0.60$), **Intermediate Subclade Burst** ($0.35\text{--}0.60$), and **Deep Ancestral / Basal Divergence** ($< 0.35$), separating **Recurrent Multi-Lineage Adaptation** from single-lineage sweeps.

#### 2. Automated Alignment Error Screening (`--filter`):
* **Dual-Stage Algorithm**: Detects 1D selective clusters via exact upper-tail hypergeometric scan ($p_{\text{local}} \le 0.01$), then evaluates the Outlier Contamination Index ($\text{OCI} \ge 0.25$) to flag private frameshifts ($\ge 3$ contiguous radical mutations in an isolated leaf against conserved species).
* **Surgical In-Place Masking**: Automatically masks only the guilty taxon's anomalous span with `NNN` and re-evaluates the cleaned alignment in milliseconds, eliminating false positives while preserving legitimate multi-species selection.

---

## 🛠️ Retraining & Fine-Tuning HyphAeon

### 1. Build per-gene training tensors

Prepare one alignment and one official HyPhy MEME JSON result per gene. Trees may be supplied as matching Newick files or embedded in the alignments:

```bash
python scripts/build_training_npz.py \
  --alignment_dir /path/to/training_alignments/ \
  --tree_dir /path/to/trees/ \
  --meme_dir /path/to/meme_results/ \
  --output_dir /path/to/training_npz/
```

### 2. Fine-tune the foundation model

```bash
python train.py \
  --data_dir /path/to/training_npz/ \
  --epochs 30 \
  --batch_size 1 \
  --lr 3e-4 \
  --embed_dim 384 \
  --layers 6 \
  --heads 12 \
  --fp16 \
  --output_dir /path/to/run_weights/
```

---

## ⚡ CLI Reference Summary

| Command | Action | Description |
| :--- | :--- | :--- |
| `hyphaeon meme` | Site-Level Selection | Fast per-codon LRT & selection rate prediction ($>10,000\times$ faster than MLE). |
| `hyphaeon epistasis` | 3D Epistatic Sectors | Co-selection networks, hypergeometric tree overlaps, and 3D contact recovery. |
| `hyphaeon dms` | Digital DMS | 19-AA in silico perturbation sweeps and Compensated Pathogenic Deviation mapping. |
| `hyphaeon busted` | Alignment Omnibus | Alignment-wide episodic selection testing and SRV false-positive filtering. |
| `hyphaeon phenotype`| Directional PhyloWAS | Directional trait mapping on the unit hypersphere across convergent clades. |

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
