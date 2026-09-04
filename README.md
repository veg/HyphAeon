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
   Multi-scale epistatic sector mining implementing phylogenetic branch attribution, exact tree hypergeometric tests, Jaccard overlap suppression, contact map recovery ($C_\beta - C_\beta < 8\text{\AA}$), and vectorized Monte Carlo permutation significance testing (`--n-permutations`, `--max-perm-p`).
3. **`hyphaeon dms` (Digital Deep Mutational Scanning & CPDs)**:
   In silico Selection Deep Mutational Scanning. Performs high-throughput sweeps of all 19 alternative amino acids across every codon position in seconds, calculating the **Epistatic Selection Sensitivity Matrix (ESSM)**, Intrinsic Mutational Plasticity ($\mathbf{E}_{i,i}$), and de novo predicting compensatory partners ($s_{\text{comp}}$) that rescue human disease mutations (Compensated Pathogenic Deviations).
4. **`hyphaeon busted` (Alignment-Wide Omnibus Selection)**:
   Multi-query cross-attention pooling head that evaluates whole-gene episodic selection and filters Synonymous Rate Variation (SRV) false positives in milliseconds.
5. **`hyphaeon phenotype` (PhyloWAS)**:
   Directional phenotype-genotype association mapping on the unit hypersphere $\mathbb{S}^{M-1}$. Computes spectral trait energies ($\Psi_{\text{Spectral}}$), exact sequenced-taxa null scaling $p$-values, Benjamini-Hochberg FDR $q$-values, **Phenotype-Associated Residue Signatures (PARS)**, macromolecular trait sector permutation testing (`--n-permutations`, `--max-perm-p`), and gene-level Brownian motion liability permulations (`--permulations`).
6. **`hyphaeon evaluate` (HyPhy MEME Concordance)**:
   Dataset-level evaluation of HyphAeon site predictions against matched HyPhy
   MEME results, with site pooling across genes and machine-readable metrics.

---

## 📦 Installation

HyphAeon requires Python ≥ 3.8 and PyTorch ≥ 2.0. At runtime it auto-selects
the best available device (CUDA → Apple MPS → CPU), so no manual configuration
is needed regardless of which install path you choose.

| Method | Command | Torch | GPU? |
| :--- | :--- | :--- | :--- |
| **pip** (default) | `pip install hyphaeon` | CUDA-bundled wheel (~550 MB) | NVIDIA GPU if driver matches; else CPU |
| **pip** (CPU-only) | `pip install torch --index-url https://download.pytorch.org/whl/cpu` then `pip install hyphaeon` | CPU-only wheel (~200 MB) | CPU |
| **Bioconda** | `conda install -c bioconda hyphaeon` | CPU-only `pytorch` from conda-forge | CPU by default; swap in `pytorch-gpu` for GPU |
| **NVIDIA Jetson** | See [issue #31](https://github.com/veg/HyphAeon/issues/31) | JetPack-native wheel (cp38 only) | Jetson GPU |

To use a GPU with Bioconda, install conda-forge's GPU PyTorch variant first:

```bash
conda create -n hyphaeon-gpu -c conda-forge pytorch-gpu
conda activate hyphaeon-gpu
conda install -c bioconda hyphaeon
```

You can always install a specific PyTorch build before installing HyphAeon if
none of the above defaults suit your system (e.g. a particular CUDA version,
a custom wheel, or a CPU-only build on a server without GPU).

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
# Run branch co-selection, sector mining, and export co-selection network with Monte Carlo permutation testing
hyphaeon epistasis \
  -a examples/HIV1_RT.fasta \
  -t examples/HIV1_RT.nwk \
  --n-permutations 10000 \
  --max-perm-p 0.05 \
  -o examples/HIV1_RT_epistasis.json \
  -c examples/HIV1_RT_edges.csv \
  --graphml examples/HIV1_RT_coselection.graphml
```

#### Key Biological Discoveries:
1. **Unsupervised Discovery of Multi-Drug Catalytic Complexes (Q151M MDR Complex)**:
   * HyphAeon places the co-evolution of residue 116 with residue 151 at **#1 overall** across all candidate pairs:
     $$\text{F116} \longleftrightarrow \text{Q151} \quad (\text{Co-Sel} = 0.8660, \; p_{\text{hyper}} = 7.02 \times 10^{-9}, \; \text{FDR } q = 1.17 \times 10^{-7})$$
2. **Autonomous Dissection of Mutually Exclusive Pathways (TAM-1 vs. TAM-2)**:
   * HyphAeon's branch co-selection metric autonomously isolates the **TAM-1 triad** (`M41L + L210W + T215Y`, $q < 10^{-7}$) from the mutually antagonistic **TAM-2 cluster** (`D67N + K70R + K219Q`, $q < 10^{-3}$).

#### Monte Carlo Permutation Testing for Epistatic Sectors:
To distinguish authentic structural/functional sectors from stochastic subsets of variable sites, HyphAeon tests the spectral coherence of candidate sectors against an empirical null distribution:
* **Vectorized Permutation Engine (`--n-permutations <int>`, default: `10000`)**: For a discovered sector $\mathcal{S}$ of size $K$, samples $B$ random $K$-site subgraphs uniformly without replacement from active candidate sites. Coherence is computed across null batches via tensor contraction and Hermitian eigenvalue decomposition:
  $$C(\mathcal{S}) = \frac{\lambda_1(\mathbf{A}[\mathcal{S}, :] \mathbf{A}[\mathcal{S}, :]^\top)}{\text{Tr}(\mathbf{A}[\mathcal{S}, :] \mathbf{A}[\mathcal{S}, :]^\top)}$$
* **Output Metrics**: Each sector reports empirical one-sided $p_{\text{perm}} = \frac{1}{B} \sum_{b=1}^B \mathbb{I}(C(\mathcal{S}^{(b)}) \ge C(\mathcal{S}))$, null mean $\mathbb{E}[C_{\text{null}}]$, standard deviation, 95th percentile cutoff $C_{95}$, and theoretical isotropic baseline $1/K$. Set `--n-permutations 0` to disable permutation testing.
* **Empirical Filtering (`--max-perm-p <float>`, default: `None`)**: Retains only sectors whose spectral coherence satisfies $p_{\text{perm}} \le \text{threshold}$ (e.g., `--max-perm-p 0.05`).

---

### Example 2: In Silico Selection Deep Mutational Scanning (Digital DMS / ESSM)

```bash
# Run digital DMS sweep on HIV-1 RT
hyphaeon dms -a examples/HIV1_RT.fasta -t examples/HIV1_RT.nwk -o examples/HIV1_RT_dms.json -c examples/HIV1_RT_dms.csv
```

---

### Example 3: Convergent Sensory Adaptation & Spectral Tuning in Rhodopsin

```bash
# Run PhyloWAS with trait sector permutation testing and gene-level phylogenetic permulations
hyphaeon phenotype \
  -a examples/RHO.fasta \
  -fg "turTru,balMus,balPhys,orcOrc,delDelp,phyCat,phoVit,halGryp,mirLeo,zalCali,odoRos" \
  --n-permutations 10000 \
  --max-perm-p 0.05 \
  --permulations 1000 \
  -o examples/RHO_marine_phenotype.json \
  -c examples/RHO_marine_sites.csv
```

#### Multi-Scale Permutation & Null Testing in PhyloWAS:
HyphAeon implements two complementary null testing layers addressing distinct evolutionary hypotheses:
1. **Macromolecular Trait Sector Permutations (`--n-permutations <int>`, default: `10000`; `--max-perm-p <float>`, default: `None`)**:
   * Following single-site phenotype association ($\text{FDR } q \le \alpha$), HyphAeon extracts coherent epistatic sectors among trait-associated residues.
   * Tests whether trait sector coherence $C(\mathcal{S})$ significantly exceeds random $K$-site subgraphs sampled across the alignment ($p_{\text{perm}} \le \text{max\_perm\_p}$), confirming that convergent phenotype adaptation drives coordinated macromolecular re-organization rather than unlinked mutations.
2. **Gene-Level Brownian Motion Liability Permulations (`--permulations <int>`, default: `0` / parametric)**:
   * Simulates neutral continuous phenotype evolution along the phylogenetic tree using Brownian motion (Saputra et al. 2021 / RERconverge null model).
   * Computes empirical gene-level $p$-values ($p_{\text{gene}}$) testing whether the length-normalized spectral energy ($\bar{\Psi}$) or maximum site association ($\rho_{\max}$) exceeds neutral phylogenetic drift.

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

### Example 5: Evaluate predictions against HyPhy MEME

`hyphaeon evaluate` compares the site-level output of `hyphaeon meme` with
HyPhy MEME used as the reference. Here, "true" means concordant with MEME; it
does not imply independently established biological ground truth.

#### Evaluate folders of genes

Prediction and MEME files are paired by their exact gene-name stem:
`Gene1.csv` matches `Gene1.MEME.json`. All matched sites from all genes are
pooled before calculating metrics—metrics are not calculated per gene and then
averaged.

```bash
hyphaeon evaluate \
  --predictions-dir /path/to/hyphaeon_predictions/ \
  --meme-dir /path/to/meme_results/ \
  --output pooled_metrics.json
```

#### Evaluate one gene

Pass a matched pair directly with `--prediction` and `--meme-result`:

```bash
hyphaeon evaluate \
  --prediction /path/to/Gene1.csv \
  --meme-result /path/to/Gene1.MEME.json \
  --output Gene1_metrics.json
```

The filename stems must match. Directory flags and direct-file flags cannot be
mixed in the same invocation.

#### Metrics and classification rules

| Output | Definition |
| :--- | :--- |
| Total sites | Number of site IDs shared by the matched prediction/MEME pairs. |
| Pearson $r$ | Pearson correlation between HyphAeon `hyphaeon_lrt` and MEME LRT over all pooled evaluated sites. |
| Spearman $\rho$ | Spearman rank correlation between the same pooled LRT values. |
| ROC-AUC at $\alpha$ | MEME `p-value <= alpha` supplies the binary reference label; continuous HyphAeon `hyphaeon_lrt` supplies the ranking score. |
| PPV at $\alpha$ | $TP/(TP+FP)$, where MEME and HyphAeon calls both use `p_value <= alpha`. |
| FPR at $\alpha$ | $FP/(FP+TN)$, where MEME and HyphAeon calls both use `p_value <= alpha`. |

A true positive is a site called significant by both MEME and HyphAeon. A true
negative is a site called non-significant by both. The JSON report includes the
full TP, FP, TN, and FN counts used for PPV and FPR.

All matched sites are evaluated by default. Use `--variable-only` to exclude
HyphAeon rows marked `is_invariable`; total-site counts still include those
rows. Negative MEME LRT numerical artifacts are clamped to zero and reported
as a warning.

#### Output and input validation

The default standard output is a compact report (illustrative values shown):

```text
Matched genes: 2
Total sites: 450
Evaluated sites: 450 (all matched sites)
Pearson r (LRT): 0.412345
Spearman rho (LRT): 0.501234

Metric                 p <= 0.05    p <= 0.10
ROC-AUC                  0.731000      0.749000
PPV                      0.420000      0.465000
FPR                      0.083000      0.121000
```

Use `--format json` for JSON on standard output or `--output FILE.json` to
write the detailed report. The JSON includes input paths, aggregate counts,
correlations, threshold metrics, both confusion matrices, per-gene counts, and
warnings. Undefined metrics—for example, ROC-AUC when MEME has only one class—
are represented as JSON `null`.

By default, unmatched genes or unequal site sets stop evaluation to prevent
silent misalignment. Folder mode supports `--allow-unmatched` to ignore genes
without a counterpart. Both modes support `--allow-site-mismatch` to use the
site intersection and report dropped counts. Custom filename conventions can
be supplied with `--prediction-suffix` and `--meme-suffix`.

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
| `hyphaeon evaluate` | MEME Concordance | Pooled ROC-AUC, LRT correlations, PPV, and FPR for folders or a single matched gene. |
| `hyphaeon epistasis` | 3D Epistatic Sectors | Co-selection networks, hypergeometric tree overlaps, and Monte Carlo sector permutations. |
| `hyphaeon dms` | Digital DMS | 19-AA in silico perturbation sweeps and Compensated Pathogenic Deviation mapping. |
| `hyphaeon busted` | Alignment Omnibus | Alignment-wide episodic selection testing and SRV false-positive filtering. |
| `hyphaeon phenotype`| Directional PhyloWAS | Directional trait mapping on the unit hypersphere, trait sector permutations, and liability permulations. |

### Key Permutation Testing Arguments:

#### `hyphaeon epistasis`
| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--n-permutations` | `int` | `10000` | Number of random $K$-site subset Monte Carlo permutations for sector significance testing (set `0` to disable). |
| `--max-perm-p` | `float` | `None` | Maximum empirical permutation $p$-value threshold to retain sectors (default retains all $C(\mathcal{S}) \ge \text{min\_coherence}$). |
| `--min-coherence` | `float` | `0.50` | Minimum spectral coherence ratio $C(\mathcal{S}) = \lambda_1 / \text{Tr}$ for candidate sectors. |
| `--min-clique-size` | `int` | `3` | Minimum clique seed size for epistatic sectors. |
| `--max-overlap` | `float` | `0.50` | Maximum Jaccard overlap allowed between discovered sectors. |
| `--no-tree` / `--use-tn93` | `flag` | `False` | Estimate pairwise evolutionary distances directly from alignment via TN93 (skips tree). Requires the optional `tn93` package (`pip install hyphaeon[tn93]`) or the `tn93` binary on PATH. |

#### `hyphaeon phenotype`
| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--n-permutations` | `int` | `10000` | Number of random $K$-site subset Monte Carlo permutations for trait sector significance testing (set `0` to disable). |
| `--max-perm-p` | `float` | `None` | Maximum permutation $p$-value threshold to retain trait sectors (default retains all $C(\mathcal{S}) \ge 0.45$). |
| `--permulations` | `int` | `0` | Number of Brownian motion phylogenetic permulations for gene-level empirical $p$-values (RERconverge null model; default `0` / parametric). |
| `--alpha` | `float` | `0.05` | Benjamini-Hochberg FDR significance threshold for trait-associated sites. |
| `--continuous` | `flag` | `False` | Treat trait values as continuous phylogenetic contrasts rather than discrete foreground/background. |
| `--min-taxa` | `int` | `4` | Minimum sequenced taxa required per site. |

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
