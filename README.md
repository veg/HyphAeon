<div align="center">

<img src="assets/hyphaeon_logo.png" alt="HyphAeon Logo" width="280"/>

# HyphAeon
### Attention on Evolution Across Deep Time Transforms Comparative Genomics

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-2026.09.06.749597-b31b1b.svg)](https://www.biorxiv.org/content/10.64898/2026.09.06.749597v1)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Interactive%20Presentation-00e5ff.svg)](https://veg.github.io/HyphAeon/)

</div>

---

**HyphAeon** is a deep-time phylogenetic foundation model designed to bridge computational phylogenetics, structural biology, and foundation AI. Built upon a 2D axial transformer backbone (**`PhyloAxialTransformer`**) with patristic distance-decay attention and classical multidimensional scaling (MDS) tree embeddings, HyphAeon ingests multi-species codon alignments and explicit evolutionary trees spanning 200 million years of deep time.

---

> [!TIP]
> **Migrating from HyPhy?** See our comprehensive [**HyPhy to HyphAeon Migration Guide**](MIGRATION_GUIDE.md) for direct method-by-method translations (`hyphy meme` → `hyphaeon meme`, `contrast-fel` → `hyphaeon phenotype`, `prime` → `hyphaeon dms`) and biological recipes categorized by empirical data regime.

## 🚀 Key Capabilities & Unified Commands

HyphAeon integrates six complementary phylogenetic deep learning and geometric
projection engines:

1. **`hyphaeon meme` (Site-Level Diversifying Selection)**:
   Neural episodic positive selection inference (100×–1,100× faster than standard numerical MLE and codon-MCMC models like HyPhy MEME/FEL; see ARCHITECTURE.md for detailed benchmarks) using Tree-RoPE 4D geometric branch embeddings and axial tree attention.
2. **`hyphaeon epistasis` (3D Co-Evolution & Epistatic Sectors)**:
   Multi-scale epistatic sector mining implementing phylogenetic branch attribution, exact tree hypergeometric tests, Jaccard overlap suppression, contact map recovery (C<sub>β</sub>–C<sub>β</sub> < 8 Å), and vectorized Monte Carlo permutation significance testing (`--n-permutations`, `--max-perm-p`).
3. **`hyphaeon dms` (Digital Deep Mutational Scanning & CPDs)**:
   In silico Selection Deep Mutational Scanning. Performs high-throughput sweeps of all 19 alternative amino acids across every codon position in seconds, calculating the **Epistatic Selection Sensitivity Matrix (ESSM)**, Intrinsic Mutational Plasticity (E<sub>i,i</sub>), and de novo predicting compensatory partners (s<sub>comp</sub>) that rescue human disease mutations (Compensated Pathogenic Deviations).
4. **`hyphaeon phenotype` (PhyloWAS)**:
   Directional phenotype-genotype association mapping on the unit hypersphere S<sup>M-1</sup>. Computes spectral trait energies (Ψ<sub>Spectral</sub>), exact sequenced-taxa null scaling p-values, Benjamini-Hochberg FDR q-values, **Phenotype-Associated Residue Signatures (PARS)**, macromolecular trait sector permutation testing (`--n-permutations`, `--max-perm-p`), and gene-level Brownian motion liability permulations (`--permulations`).
5. **`hyphaeon temporal` (Continuous Surveillance Dynamics & Sweep Velocity)**:
   Time-resolved episodic selection tracking using continuous logistic trajectory regression, positive sweep velocity v<sub>s</sub>(t) = max(0, d/dt â<sub>s</sub>(t)), Dynamic Time Warping (DTW) wave decomposition, and temporal SVD factor loadings. See the [**Temporal Analysis Operational Guide**](TEMPORAL_ANALYSIS_GUIDE.md).
6. **`hyphaeon splits` (Spectral Graph Bisection & Tree-Free Clade Discovery)**:
   Recovers well-supported phylogenetic macro-clades and deep hierarchical bipartitions by fusing pairwise continuous 4D MDS geometry with discrete cross-taxa attention maps. Delivers up to 28× speedups over traditional ML tree search without requiring pre-computed phylogenies. See the [**Spectral Splits & Benchmarking Report**](SPECTRAL_SPLITS_BENCHMARK.md).
9. **`hyphaeon dating` (Molecular Clock Calibration & t_MRCA Dating)**:
   Heterochronous molecular clock calibration, ancestor dating, automated adaptive ridge regularization (`--tune-ridge` via fast spectral PRESS LOOCV), and non-linear clock model adjudication (Restricted Cubic Splines). Implements centered root-to-tip OLS (TempEst emulation), time-decay weighted consensus rooting, and HyphAeon Attention-Derived PGLS ($\boldsymbol{\Sigma} = \mathbf{A}_{\text{fused}} + \lambda\mathbf{I}$) resolving phylogenetic pseudoreplication. Replicates landmark studies such as Bette Korber et al. (Science 2000) dating the ancestor of HIV-1 group M to ~1931 in seconds. See the [**Molecular Clock & Dating Guide**](DATING_GUIDE.md).
10. **`hyphaeon geo` (Discrete Phylogeography & Spatial Transmission Network Inference)**:
   Ultra-fast discrete phylogeography, directed migration flux matrices ($M_{jk} \ne M_{kj}$), vectorized permutation BSSVS Bayes Factors ($\text{BF} \ge 3.0$), and spatial PGLS continuous root epicenter estimation. Replicates landmark studies such as Philippe Lemey et al. (PLoS Comput Biol 2009) avian influenza H5N1 dispersal across 7 Chinese provinces in < 1 second.

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

> [!NOTE]
> **Model weights** are downloaded automatically from [Hugging Face](https://huggingface.co/datamonkey/hyphaeon)
> on first use (cached in `~/.cache/hyphaeon/`). No authentication or token is
> required. Use `--model-variant viral` to select the viral-tuned variant, or
> `--weights /path/to/checkpoint` to use a local file.

---

## 📂 Included Benchmark Datasets

All example alignments and phylogenetic trees required to reproduce these analyses are bundled
directly in `examples/`, including the canonical historical and pandemic outbreak benchmarks:

| Dataset | Alignment | Tree / Coordinates | Taxa | Sites | Scientific Significance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HIV-1 RT** | [`examples/HIV1_RT.fasta`](examples/HIV1_RT.fasta) | [`examples/HIV1_RT.nwk`](examples/HIV1_RT.nwk) | 476 | 335 | Retroviral Reverse Transcriptase polymerase domain (drug resistance & epistasis). |
| **Rhodopsin** | [`examples/RHO.fasta`](examples/RHO.fasta) | Auto (TN93) | 710 | 349 | Mammalian Rhodopsin visual pigments (deep-sea diving sensory adaptation). No tree file provided; uses TN93 distance estimation. |
| **Smc6** | [`examples/Smc6.fasta`](examples/Smc6.fasta) | [`examples/Smc6.nwk`](examples/Smc6.nwk) | 20 | 1,097 | Primate Smc6 structural maintenance of chromosomes (antiviral host restriction). |
| **Bat OAS1** | [`examples/bat_oas1.fasta`](examples/bat_oas1.fasta) | [`examples/bat_oas1.nwk`](examples/bat_oas1.nwk) | 18 | 351 | Chiropteran OAS1 2'-5'-oligoadenylate synthetase (innate immunity escape). |
| **Camelid VHH** | [`examples/camelid.fasta`](examples/camelid.fasta) | [`examples/camelid.nwk`](examples/camelid.nwk) | 212 | 96 | Camelid single-domain antibody heavy-chain variable domain (antigenic diversity). Used for integration testing; no dedicated example section. |
| **HIV-1 gp160 (Korber 2000)** | [`examples/korber_env_gp160.fasta`](examples/korber_env_gp160.fasta) | Tree-Free / Consensus | 143 | 981 | Bette Korber et al. (Science 2000) landmark molecular clock dataset (1959–1997 HIV-1 group M). |
| **Avian Flu H5N1 (Lemey 2009)** | [`examples/H5N1_HA_geo.fasta`](examples/H5N1_HA_geo.fasta) | [`examples/H5N1_HA.nwk`](examples/H5N1_HA.nwk) | 98 | 566 | Lemey et al. (PLoS Comput Biol 2009) benchmark discrete phylogeography across 7 Chinese provinces. |
| **Pandemic H1N1 (Fraser 2009)** | [`examples/H1N1_2009_pandemic.fasta`](examples/H1N1_2009_pandemic.fasta) | [`examples/H1N1_2009_pandemic.nwk`](examples/H1N1_2009_pandemic.nwk) | 100 | 1701 | Fraser et al. (Science 2009) landmark phylodynamics and early growth rate benchmark ($R_0$ estimation). |

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
     > **F116 ⟷ Q151** (Co-Sel = 0.8660, p<sub>hyper</sub> = 7.02 × 10⁻⁹, FDR q = 1.17 × 10⁻⁷)
2. **Autonomous Dissection of Mutually Exclusive Pathways (TAM-1 vs. TAM-2)**:
   * HyphAeon's branch co-selection metric autonomously isolates the **TAM-1 triad** (`M41L + L210W + T215Y`, q < 10⁻⁷) from the mutually antagonistic **TAM-2 cluster** (`D67N + K70R + K219Q`, q < 10⁻³).

#### Monte Carlo Permutation Testing for Epistatic Sectors:
To distinguish authentic structural/functional sectors from stochastic subsets of variable sites, HyphAeon tests the spectral coherence of candidate sectors against an empirical null distribution:
* **Vectorized Permutation Engine (`--n-permutations <int>`, default: `10000`)**: For a discovered sector S of size K, samples B random K-site subgraphs uniformly without replacement from active candidate sites. Coherence is computed across null batches via tensor contraction and Hermitian eigenvalue decomposition:
  ```text
  C(S) = λ₁(A[S, :] A[S, :]ᵀ) / Tr(A[S, :] A[S, :]ᵀ)
  ```
* **Output Metrics**: Each sector reports empirical one-sided permutation p-value:
  ```text
  p_perm = (1/B) Σ I(C(S^(b)) ≥ C(S))
  ```
  along with null mean E[C<sub>null</sub>], standard deviation, 95th percentile cutoff C<sub>95</sub>, and theoretical isotropic baseline 1/K. Set `--n-permutations 0` to disable permutation testing.
* **Empirical Filtering (`--max-perm-p <float>`, default: `None`)**: Retains only sectors whose spectral coherence satisfies `p_perm ≤ threshold` (e.g., `--max-perm-p 0.05`).

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
   * Following single-site phenotype association (FDR q ≤ α), HyphAeon extracts coherent epistatic sectors among trait-associated residues.
   * Tests whether trait sector coherence C(S) significantly exceeds random K-site subgraphs sampled across the alignment (p<sub>perm</sub> ≤ max_perm_p), confirming that convergent phenotype adaptation drives coordinated macromolecular re-organization rather than unlinked mutations.
2. **Gene-Level Brownian Motion Liability Permulations (`--permulations <int>`, default: `0` / parametric)**:
   * Simulates neutral continuous phenotype evolution along the phylogenetic tree using Brownian motion (Saputra et al. 2021 / RERconverge null model).
   * Computes empirical gene-level p-values (p<sub>gene</sub>) testing whether the length-normalized spectral energy (Ψ̄) or maximum site association (ρ<sub>max</sub>) exceeds neutral phylogenetic drift.

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
* **Single-Taxon Counterfactual Perturbation (ΔLRT)**: In silico mutates each non-consensus species back to ancestral state to rank driving taxa by marginal selection evidence explained (% Signal Explained).
* **Evolutionary Epoch Decomposition**: Classifies selection timing by weighted root patristic depth into **Recent Terminal / Tip Sweep** (≥ 0.60), **Intermediate Subclade Burst** (0.35–0.60), and **Deep Ancestral / Basal Divergence** (< 0.35), separating **Recurrent Multi-Lineage Adaptation** from single-lineage sweeps.

#### 2. Automated Alignment Error Screening (`--filter`):
* **Dual-Stage Algorithm**: Detects 1D selective clusters via exact upper-tail hypergeometric scan (p<sub>local</sub> ≤ 0.01), then evaluates the Outlier Contamination Index (OCI ≥ 0.25) to flag private frameshifts (≥ 3 contiguous radical mutations in an isolated leaf against conserved species).
* **Surgical In-Place Masking**: Automatically masks only the guilty taxon's anomalous span with `NNN` and re-evaluates the cleaned alignment in milliseconds, eliminating false positives while preserving legitimate multi-species selection.

---

### Example 5: Spectral Graph Bisection & Tree-Free Phylogenetic Splits (`hyphaeon splits`)

```bash
# Basic Tree-Free Macro-Split Discovery (Outputs Newick Tree & Clade CSV)
hyphaeon splits \
  -a examples/bat_oas1.fasta \
  --no-tree \
  -o examples/bat_oas1_spectral_tree.nwk \
  -c examples/bat_oas1_clades.csv \
  --cpu
```

#### Spectral Bisection Architecture:
* **Multi-Modal Affinity Fusion**: Combines cross-taxa attention matrices ($\bar{\mathbf{A}}$) from the axial transformer, continuous 4D metric space from Multidimensional Scaling (MDS) on pairwise distances, and sequence-level latent representations into a fused affinity matrix $\mathbf{A}_{\text{fused}} = \mathbf{S}_{\text{attn}} \odot \mathbf{K}_{\text{MDS}} \odot \mathbf{K}_{\text{emb}}$.
* **Normalized Graph Laplacian & Fiedler Vector**: Partitions taxa along the Fiedler vector $\mathbf{v}_2$ of $\mathbf{L}_{\text{sym}} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}_{\text{fused}} \mathbf{D}^{-1/2}$, quantifying macro-clade split stability via the spectral eigengap $\Delta\lambda = \lambda_3 - \lambda_2$.
* **Comprehensive Benchmarks**: See [`SPECTRAL_SPLITS_BENCHMARK.md`](SPECTRAL_SPLITS_BENCHMARK.md) for full benchmarks against IQ-TREE 2, RAxML-NG, FastTree, and Neighbor-Joining across empirical datasets.

---

### Example 7: Heterochronous Molecular Clock Calibration & MRCA Dating (`hyphaeon dating`)

Replicating the landmark study of **Bette Korber et al. (Science 2000)** dating the origin of HIV-1 group M to ~1931:

```bash
# Full Heterochronous Dating: Centered OLS + Attention PGLS + Latent Manifold Collapse
hyphaeon dating \
  -a examples/korber_env_gp160.fasta \
  --root-taxon CONSENSUS \
  --no-tree \
  --method all \
  -o examples/korber_dating_results.json \
  -c examples/korber_dating_taxa.csv \
  --plot-path examples/korber_clock_diagnostic.png
```

#### Output Summary:
```text
=========================================================================================================
Method / Estimator                   Estimated t_MRCA     95% Confidence Interval    Rate (μ / year)    R^2   
---------------------------------------------------------------------------------------------------------
1. Standard OLS (TempEst RTT)        1930.82            [1866.5, 1945.8]              0.001874      0.472
2. HyphAeon Attention PGLS           1927.57            [1916.4, 1938.7]              0.001875      0.518
3. Latent Manifold Collapse          1975.96            [Non-Parametric Coalescent]    0.017032 [Var/yr] 0.429
---------------------------------------------------------------------------------------------------------

[*] Flagged Temporal Outliers (|Z| >= 2.5):
    • Z59ZR.ZHU: Sampling Date=1959.5, Predicted Date=1933.4 (Discrepancy: -26.09 yr, Z=-5.40)
```

* **Accurate Ancestor Dating**: Recovers $t_{\text{MRCA}} = 1930.8$ (OLS) and $1927.6$ (Attention PGLS), closely reproducing Korber et al.'s supercomputer maximum-likelihood estimate of **1931.4 [1914.5, 1944.0]** and Thorne's MCMC relaxed clock (**1922–1929 [1889–1952]**) in seconds.
* **Resolving Pseudoreplication**: Cross-taxa attention covariance $\boldsymbol{\Sigma} = \mathbf{A}_{\text{fused}} + \lambda\mathbf{I}$ whitens shared phylogenetic history, preventing false statistical precision without requiring tree inference.
* **Historical Validation**: Accurately isolates the 1959 Léopoldville archival isolate `Z59ZR.ZHU` as a temporal outlier relative to the contemporary 1983–1997 cohort.
* **Comprehensive Guide**: See [`DATING_GUIDE.md`](DATING_GUIDE.md) for full mathematical formulation, intra-host clinical applications (e.g. CD4+ T cell latent reservoir integration timing in CAP286), and CLI documentation.

---

### Example 8: Discrete Phylogeography & Spatial Transmission Networks (`hyphaeon geo`)

Replicating the landmark discrete phylogeography study of **Philippe Lemey et al. (PLoS Comput Biol 2009)** reconstructing the epicentral origin and dispersal corridors of Avian Influenza A (H5N1) across 7 Chinese provinces:

```bash
# Run the built-in worked example with a single command
hyphaeon geo --example --no-neural
```

Or execute directly on custom alignments and metadata:
```bash
hyphaeon geo \
  -a examples/H5N1_HA_geo.fasta \
  -g examples/H5N1_HA_metadata.csv \
  -t examples/H5N1_HA.nwk \
  --no-neural \
  --n-perms 1000 \
  --min-bf 3.0 \
  --geojson examples/H5N1_HA_geo.geojson \
  -o examples/H5N1_HA_geo_results.json \
  -c examples/H5N1_HA_routes.csv \
  --plot-path examples/H5N1_HA_geo_diagnostic.png
```

#### Output Summary:
```text
=========================================================================================================
Rank   Geographic Region        Posterior P(Root)      Isolates     Role / Dynamics         
---------------------------------------------------------------------------------------------------------
1      Guangdong                  1.0000                15         Source / Exporter    ★ EPICENTER
2      Fujian                     0.0000                 8         Source / Exporter   
3      Guangxi                    0.0000                27         Source / Exporter   
4      Hebei                      0.0000                 3         Sink / Importer     
5      Henan                      0.0000                 8         Source / Exporter   
6      HongKong                   0.0000                28         Sink / Importer     
7      Hunan                      0.0000                 9         Sink / Importer     
---------------------------------------------------------------------------------------------------------

[*] Statistically Supported Transmission Routes (BF >= 3.0 or FDR <= 0.10):
Source           Target (Sink)    Flux         Z-Score    p-value    FDR q      Bayes Factor   Support         
---------------------------------------------------------------------------------------------------------
Guangdong        Fujian            0.36515       3.63     0.0060    0.2517      247.5     Decisive (BF >= 100)
Henan            Hebei             0.20412       3.92     0.0559    1.0000       43.0     Strong (10 <= BF < 100)
Guangdong        Guangxi           0.24845       1.33     0.1578    1.0000       13.4     Strong (10 <= BF < 100)
Fujian           Hebei             0.20412       1.69     0.1948    1.0000       10.4     Strong (10 <= BF < 100)
Henan            Hunan             0.11785       0.76     0.3986    1.0000        3.8     Substantial (3 <= BF < 10)
Fujian           Henan             0.12500       0.67     0.4226    1.0000        3.4     Substantial (3 <= BF < 10)
Guangdong        HongKong          0.14639       0.40     0.4426    1.0000        3.2     Substantial (3 <= BF < 10)
```

#### Key Innovations over BEAST (Lemey et al. 2009):
* **Ultra-Fast Speed (< 1 Second vs. Hours)**: Replaces tens of millions of MCMC iterations over $2^{K(K-1)/2}$ graph configurations with closed-form ancestral state reconstruction and vectorized matrix permutations.
* **Naturally Asymmetric Directed Migration**: Unlike BEAST's reversible rate matrix ($\mathbf{\Lambda} = \mu \mathbf{S} \mathbf{P}$, which enforces $s_{jk} = s_{kj}$), HyphAeon measures true directional transmission ($M_{jk} \ne M_{kj}$), capturing directional source-sink dynamics.
* **Vectorized Permutation BSSVS**: Generates exact empirical Bayes Factors ($\text{BF} \ge 3.0$) and Benjamini-Hochberg FDR $q$-values from 1,000 null permutations in $< 0.1$ seconds.
* **Spatial PGLS Epicenter**: Infers the continuous geographic epicenter coordinates ($28.10^\circ\text{N}, 111.83^\circ\text{E}$) with analytical 95% geographic confidence radii.
* **Modern Web GIS Export**: Generates standard GeoJSON feature collections (`.geojson`) compatible with Kepler.gl and Nextstrain/Auspice.

---

## 🛠️ Retraining & Fine-Tuning HyphAeon

### 1. Build per-gene training tensors

Prepare one alignment and one official HyPhy MEME JSON result per gene. Trees may be supplied as matching Newick files or embedded in the alignments:

```bash
python training/build_training_npz.py \
  --alignment_dir /path/to/training_alignments/ \
  --tree_dir /path/to/trees/ \
  --meme_dir /path/to/meme_results/ \
  --output_dir /path/to/training_npz/
```

### 2. Fine-tune the foundation model

```bash
python training/train.py \
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
| `hyphaeon meme` | Site-Level Selection | Fast per-codon LRT & selection rate prediction (100×–1,100× faster than MLE). |
| `hyphaeon epistasis` | 3D Epistatic Sectors | Co-selection networks, hypergeometric tree overlaps, and Monte Carlo sector permutations. |
| `hyphaeon dms` | Digital DMS | 19-AA in silico perturbation sweeps and Compensated Pathogenic Deviation mapping. |
| `hyphaeon phenotype`| Directional PhyloWAS | Directional trait mapping on the unit hypersphere, trait sector permutations, and liability permulations. |
| `hyphaeon temporal` | Dynamic Surveillance | Continuous logistic trajectory regression, sweep velocity, DTW waves, and temporal SVD. |
| `hyphaeon splits` | Spectral Bisection | Tree-free phylogenetic macro-splits via cross-taxa attention and MDS graph Laplacian. |
| `hyphaeon disease` | Pathogenicity Prediction | Predict disease variant effects and pathogenicity using HyphAeon attention attributions. |
| `hyphaeon filter` | Alignment QC | Automated alignment error detection and surgical masking of anomalous regions. |
| `hyphaeon dating` | Molecular Clock & MRCA | Heterochronous root-to-tip OLS, Attention PGLS, and latent manifold variance collapse. |

### Key Command Arguments:

#### `hyphaeon dating`
| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `-a` / `--alignment` | `path` | Required | Path to in-frame codon FASTA or NEXUS alignment ($L_{\text{nt}} \pmod 3 == 0$). |
| `-t` / `--tree` | `path` | `None` | Optional Newick/NEXUS phylogenetic tree (optional if embedded, or if `--no-tree`/`--use-tn93` is set). |
| `--no-tree` / `--use-tn93` | `flag` | `False` | Skip phylogenetic tree and estimate pairwise evolutionary distances directly from alignment via TN93. |
| `-d` / `--dates` | `path` | `None` | Path to Nextstrain Auspice JSON, metadata CSV/TSV, or omitted to auto-extract timestamps from headers. |
| `--root-taxon` | `str` | `None` | Anchor/root taxon name (e.g. `'CONSENSUS'`, earliest taxon, or outgroup). |
| `--method` | `str` | `all` | Dating estimator(s) to run: `all`, `ols`, or `pgls`. |
| `--clock-model` | `str` | `auto` | Clock model: `auto` (spline vs linear adjudication), `linear`, `spline`, or `power`. |
| `--ridge` | `float/str` | `0.05` | Regularization parameter for PGLS cross-taxa attention covariance (or `'auto'`). |
| `--tune-ridge` | `flag` | `False` | Automatically tune ridge parameter $\lambda^*$ via fast spectral PRESS LOOCV. |
| `--bootstrap` | `int` | `1000` | Number of non-parametric bootstrap resamples for empirical confidence intervals. |
| `--plot` | `flag` | `False` | Generate publication-grade diagnostic PDF and PNG figures. |
| `--plot-path` | `path` | `None` | Custom output path for diagnostic plot (e.g. `mrca_clock.png`). |
| `-o` / `--output` | `path` | `None` | Optional path to export JSON summary results. |
| `-c` / `--csv` | `path` | `None` | Optional path to export per-taxon diagnostic table (`.csv`). |

#### `hyphaeon epistasis`
| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--n-permutations` | `int` | `10000` | Number of random K-site subset Monte Carlo permutations for sector significance testing (set `0` to disable). |
| `--max-perm-p` | `float` | `None` | Maximum empirical permutation p-value threshold to retain sectors (default retains all C(S) ≥ min_coherence). |
| `--min-coherence` | `float` | `0.50` | Minimum spectral coherence ratio C(S) = λ₁ / Tr for candidate sectors. |
| `--min-clique-size` | `int` | `3` | Minimum clique seed size for epistatic sectors. |
| `--max-overlap` | `float` | `0.50` | Maximum Jaccard overlap allowed between discovered sectors. |
| `--no-tree` / `--use-tn93` | `flag` | `False` | Estimate pairwise evolutionary distances directly from alignment via TN93 (skips tree). Requires the optional `tn93` package (`pip install hyphaeon[tn93]`) or the `tn93` binary on PATH. |

#### `hyphaeon phenotype`
| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--n-permutations` | `int` | `10000` | Number of random K-site subset Monte Carlo permutations for trait sector significance testing (set `0` to disable). |
| `--max-perm-p` | `float` | `None` | Maximum permutation p-value threshold to retain trait sectors (default retains all C(S) ≥ 0.45). |
| `--permulations` | `int` | `0` | Number of Brownian motion phylogenetic permulations for gene-level empirical p-values (RERconverge null model; default `0` / parametric). |
| `--alpha` | `float` | `0.05` | Benjamini-Hochberg FDR significance threshold for trait-associated sites. |
| `--continuous` | `flag` | `False` | Treat trait values as continuous phylogenetic contrasts rather than discrete foreground/background. |
| `--min-taxa` | `int` | `4` | Minimum sequenced taxa required per site. |

#### `hyphaeon splits`
| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `-a` / `--alignment` | `path` | Required | Path to in-frame codon FASTA or NEXUS alignment. |
| `-t` / `--tree` | `path` | `None` | Optional Newick/NEXUS phylogenetic tree (optional if embedded, or if `--no-tree`/`--use-tn93` is set). |
| `--no-tree` / `--use-tn93` | `flag` | `False` | Skip phylogenetic tree and estimate pairwise evolutionary distances directly from alignment via TN93. Requires `tn93` (`pip install hyphaeon[tn93]`) or the `tn93` binary on PATH. |
| `--min-clade-size` | `int` | `2` | Minimum clade size floor to terminate recursive bisection. |
| `--max-depth` | `int` | `10` | Maximum tree hierarchy recursion depth. |
| `-o` / `--output` | `path` | `None` | Optional path to export derived hierarchical Newick tree (`.nwk`). |
| `-c` / `--csv` | `path` | `None` | Optional path to export split clade membership assignments (`.csv`). |
| `-w` / `--weights` | `path` | `None` | Path to local model weights file (overrides HF download). |
| `--cpu` | `flag` | `False` | Force CPU execution. |

#### `hyphaeon geo`
| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--example` / `--run-example` | `flag` | `False` | Execute the built-in worked benchmark example (Avian Flu H5N1 across 7 Chinese provinces from Lemey et al. 2009). |
| `-a` / `--alignment` | `path` | `None` | Path to FASTA alignment (required unless `--example` is set). |
| `-g` / `--metadata` | `path` | `None` | Path to metadata CSV/TSV or Auspice JSON containing discrete location labels (required unless `--example` is set). |
| `-t` / `--tree` | `path` | `None` | Optional Newick tree (auto-built via FastTree if omitted). |
| `--location-col` | `str` | `None` | Column name for discrete location / region in metadata (auto-detected if omitted). |
| `--strain-col` | `str` | `None` | Column name for taxon identifier in metadata. |
| `--date-col` | `str` | `None` | Column name for sample collection date in metadata (used for temporal rooting). |
| `--lat-col` / `--lon-col` | `str` | `None` | Column names for latitude / longitude coordinates (used for continuous spatial PGLS epicenter). |
| `--n-perms` | `int` | `1000` | Number of label permutations for null flux distribution and Bayes Factor calculation. |
| `--min-bf` | `float` | `3.0` | Bayes Factor threshold for significant transmission routes ($\text{BF} \ge 3.0$). |
| `--fdr` | `float` | `0.10` | Benjamini-Hochberg FDR threshold for significant transmission routes. |
| `--no-neural` | `flag` | `False` | Disable neural attention backbone; use phylogenetic tree branch transitions. |
| `--plot` | `flag` | `False` | Generate publication-grade diagnostic PDF and PNG figures. |
| `--plot-path` | `path` | `None` | Custom output path for diagnostic plot (e.g. `h5n1_geo_diagnostic.png`). |
| `--geojson` | `path` | `None` | Path to export standard GeoJSON feature collection for Kepler.gl / Auspice GIS visualization. |
| `-o` / `--output` | `path` | `None` | Optional path to export JSON summary results. |
| `-c` / `--csv` | `path` | `None` | Optional path to export transmission routes table (`.csv`). |

#### `hyphaeon r0`
| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--example` / `--run-example` | `flag` | `False` | Execute the built-in worked benchmark example (2009 Pandemic H1N1 Origin from Fraser et al. 2009). |
| `-a` / `--alignment` | `path` | `None` | Path to FASTA alignment with dates in headers or metadata. |
| `-t` / `--tree` | `path` | `None` | Path to Newick tree with calibrated or substitution branch lengths. |
| `-g` / `--metadata` | `path` | `None` | Path to metadata CSV/TSV containing collection dates. |
| `--pathogen` | `str` | `None` | Pathogen preset (`h1n1`, `ebola`, `sars-cov-2`, `measles`, `hiv_early`). |
| `--generation-time` | `float` | `None` | Mean clinical generation interval / serial interval $T_g$ (default from pathogen preset or 5.0 days). |
| `--generation-sd` | `float` | `None` | Standard deviation of generation interval $\sigma_g$ (for gamma renewal model). |
| `--latent-time` | `float` | `None` | Optional latent period in days for SEIR renewal model. |
| `--units` | `str` | `days` | Time units for generation interval parameters (`days` or `years`). |
| `--date-col` | `str` | `None` | Metadata CSV column name for sample collection date. |
| `--strain-col` | `str` | `None` | Metadata CSV column name for taxon identifier. |
| `--mu-prior` | `float` | `None` | Clock rate prior ($\text{sub/site/yr}$) to convert substitution trees to calendar time via LSD. |
| `--window-size` | `float` | `0.25` | Temporal window span in years for dynamic $R(t)$ skyline (default: 0.25 years / 3 months). |
| `--step-size` | `float` | `0.05` | Sliding window step size in years for dynamic $R(t)$ skyline (default: 0.05 years). |
| `--plot` | `flag` | `False` | Generate publication-grade 3-panel diagnostic figures (LTT dynamics, Profile Likelihood, Dynamic $R_t$). |
| `--plot-path` | `path` | `None` | Custom output path for diagnostic plot (e.g. `r0_diagnostics.png`). |
| `-o` / `--output` | `path` | `None` | Optional path to export JSON summary results. |
| `-c` / `--csv` | `path` | `None` | Optional path to export dynamic $R(t)$ skyline table (`.csv`). |

---

## 📜 Citation

If you use **HyphAeon** in your research, please cite:

> Sergei L. Kosakovsky Pond, Steven Weaver, Danielle Callan, Jordan D. Zehr, Alexander G. Lucaci, Hannah Verdonk, Avery Selberg, Gallean Brown, Maria Chikina, Nathan L. Clark, Kateryna D. Makova, Darren P. Martin, and Anton Nekrutenko.  
> **HyphAeon: Attention on Evolution Across Deep Time Transforms Comparative Genomics**.  
> *bioRxiv* 2026.09.06.749597; doi: [https://doi.org/10.64898/2026.09.06.749597](https://www.biorxiv.org/content/10.64898/2026.09.06.749597v1)

```bibtex
@article{kosakovskypond2026hyphaeon,
  title={HyphAeon: Attention on Evolution Across Deep Time Transforms Comparative Genomics},
  author={Kosakovsky Pond, Sergei L. and Weaver, Steven and Callan, Danielle and Zehr, Jordan D. and Lucaci, Alexander G. and Verdonk, Hannah and Selberg, Avery and Brown, Gallean and Chikina, Maria and Clark, Nathan L. and Makova, Kateryna D. and Martin, Darren P. and Nekrutenko, Anton},
  journal={bioRxiv},
  pages={2026.09.06.749597},
  year={2026},
  doi={10.64898/2026.09.06.749597},
  url={https://www.biorxiv.org/content/10.64898/2026.09.06.749597v1}
}
```

---

## JavaScript library (@veg/hyphaeon-js)

`js/` is a **library** of pure functions that mirror `hyphaeon/*.py`: alignment and tree parsing,
tokenisation, patristic distances and MDS, tensor assembly, the statistics (MEME mixture p-values,
Benjamini-Hochberg, ACAT/Simes), filtering, attribution, epistasis and sector mining, DMS,
phenotype association and permulations, evaluation, diagnostics. Nothing runs here: no
onnxruntime, no file I/O beyond parsing strings, no workers, no server, no MCP. It exists so that
the port of a method is reviewed in the same pull request as the method, replays the same fixtures
(`fixtures/`, generated by `scripts/gen_fixtures.py`) in the same CI run, and ships from the same
commit.

**One repository, one tag.** A tag `vX.Y.Z` publishes `hyphaeon==X.Y.Z` to PyPI and
`@veg/hyphaeon-js@X.Y.Z` to npm from the same commit (`.github/workflows/release.yml`), with
`models/manifest.json` and `MDS_SIGN.md` attached to the GitHub release. `js/package.json` and
`pyproject.toml` must both carry the tag's version; the release workflow refuses otherwise. Both
registries publish over OIDC trusted publishing, so there are no tokens in this repository.

```bash
npm install @veg/hyphaeon-js
```

**What CI checks.** `js.yml` runs three gates on every change to `js/`, `fixtures/` or `models/`:
`npm test` (vitest, including the fixture replays — a wrong value), `npm run typecheck` (a wrong
shape), and `node js/scripts/fixture-coverage.mjs`, which fails when a fixture file is read by no
test — 48 of 48 today, so a new fixture from `scripts/gen_fixtures.py` fails the build until
something replays it. `parity.yml` runs the **reference surface only**: the Python CLI on the
bundled examples, on the tree path and — for the examples whose tree carries no branch lengths —
on the `--use-tn93` path (D22), plus `scripts/parity.py`'s self-check that p and q recompute from
each file's own LRTs. The Node, browser and MCP surfaces are produced by the application
repository's runners into `parity/<surface>/` (and `parity/<surface>-tn93/` for a tree-free run)
and compared there with `scripts/parity.py` from this repository at the engine tag the app pins —
the contract is written out in the header of `.github/workflows/parity.yml`.

```bash
cd js && npm ci && npm test && npm run typecheck && node scripts/fixture-coverage.mjs
```

- Parity between the Python reference and the JavaScript surfaces, the tolerance classes, and
  how to run the harness: [`PARITY.md`](PARITY.md) and `scripts/parity.py`.
- The MDS eigenvector sign convention (`--mds-sign`, default `canonical`), what it changed and by
  how much: [`MDS_SIGN.md`](MDS_SIGN.md).
- **Reference bugs and quirks the port found and reproduced rather than fixed**, with the line,
  the impact, the suggested fix and whether fixing it moves the fixtures:
  [`UPSTREAM.md`](UPSTREAM.md). Port discipline is: fix the Python, regenerate `fixtures/`, then
  fix the port — never the other way round.
- Everything that runs the library, the web application, the Node runtime with the ONNX sessions,
  and the MCP server (`@veg/hyphaeon-mcp`), lives in
  [`veg/hyphaeon-app`](https://github.com/veg/hyphaeon-app), which pins this package at an exact
  version.
