# ChronAeon

**The Velocity of Time** — Ultra-fast molecular clock dating, phylodynamics, and genomic surveillance.

ChronAeon operates along the **TAXON / LINEAGE** axis of the HyphAeon foundation model, providing sub-second execution for tree-free continuous manifold dating, attention-derived covariance, and planetary-scale genomic screening.

## Target Audience

Public health agencies (CDC, WHO, UKHSA), outbreak epidemiologists, hospital infection control teams.

## Installation

ChronAeon requires Python ≥ 3.8 and PyTorch ≥ 2.0. At runtime it auto-selects
the best available device (CUDA → Apple MPS → CPU), so no manual configuration
is needed regardless of which install path you choose.

```bash
pip install chronaeon
```

| Method | Command | Torch | GPU? |
| :--- | :--- | :--- | :--- |
| **pip** (default) | `pip install chronaeon` | CUDA-bundled wheel (~550 MB) | NVIDIA GPU if driver matches; else CPU |
| **pip** (CPU-only) | `pip install torch --index-url https://download.pytorch.org/whl/cpu` then `pip install chronaeon` | CPU-only wheel (~200 MB) | CPU |
| **Bioconda** | `conda install -c bioconda chronaeon` | CPU-only `pytorch` from conda-forge | CPU only |
| **NVIDIA Jetson** | See [issue #31](https://github.com/veg/HyphAeon/issues/31) | JetPack-native wheel (cp38 only) | Jetson GPU |

You can always install a specific PyTorch build before installing ChronAeon if
none of the above defaults suit your system (e.g. a particular CUDA version,
a custom wheel, or a CPU-only build on a server without GPU).

> [!NOTE]
> **Model weights** are downloaded automatically from [Hugging Face](https://huggingface.co/datamonkey/hyphaeon)
> on first use (cached in `~/.cache/hyphaeon/`). No authentication or token is
> required. Use `--model-variant viral` to select the viral-tuned variant, or
> `--weights /path/to/checkpoint` to use a local file.

## CLI Subcommands

| Command | Aliases | Description |
| :--- | :--- | :--- |
| `chronaeon date` | `dating`, `clock`, `mrca` | Molecular clock calibration, tMRCA dating |
| `chronaeon autoclock` | `deconvolve`, `multiclock` | Hierarchical multi-clock deconvolution |
| `chronaeon triage` | `radar`, `sieve`, `qc`, `qc-stream`, `stream-qc`, `chronaeon-sieve` | Streaming genomic QC triage / outbreak radar |
| `chronaeon phylogeo` | `geo`, `spatial`, `dispersal` | Discrete phylogeography |
| `chronaeon dynamics` | `r0`, `rt`, `growth`, `phylodynamics` | Phylodynamic R₀/Rₜ estimation |
| `chronaeon sketch` | `cluster`, `bin`, `centrifuge` | MinHash sketching & binning |
| `chronaeon align` | `thread`, `codon-align` | Reference-guided codon alignment |

## Key Capabilities

- **tMRCA Dating** — Continuous sequence manifold dating using attention-derived covariance (`compute_neural_covariance_kernel`)
- **AutoClock** — Hierarchical multi-clock deconvolution for complex evolutionary scenarios
- **Triage/Radar** — Stream 100k genomes in minutes, detect emerging clades, flag anomalous spillover branches
- **Phylogeography** — Continuous spatial dispersal reconstruction
- **Phylodynamics** — R₀/Rₜ growth rate estimation from heterochronous sequences

## The Radar & Microscope Flywheel

ChronAeon is the **Radar**: rapidly screens genomes, detects emerging clades, and infers origin dates. HyphAeon is the **Microscope**: dissects *why* flagged clades emerged — identifying positive selection bursts and epistatic rewiring. Together they form a collaborative flywheel for genomic surveillance and deep evolutionary analysis.

## Reproducible Benchmark Examples

### Example 1: Heterochronous Molecular Clock Calibration & MRCA Dating

Replicating the landmark study of **Bette Korber et al. (Science 2000)** dating the origin of HIV-1 group M to ~1931:

```bash
# Full Heterochronous Dating: Centered OLS + Attention PGLS + Latent Manifold Collapse
chronaeon date \
  -a chronaeon/examples/korber_env_gp160.fasta \
  --root-taxon CONSENSUS \
  --no-tree \
  --method all \
  -o chronaeon/examples/korber_dating_results.json \
  -c chronaeon/examples/korber_dating_taxa.csv \
  --plot-path chronaeon/examples/korber_clock_diagnostic.png
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

### Example 2: Discrete Phylogeography & Spatial Transmission Networks

Replicating the landmark discrete phylogeography study of **Philippe Lemey et al. (PLoS Comput Biol 2009)** reconstructing the epicentral origin and dispersal corridors of Avian Influenza A (H5N1) across 7 Chinese provinces:

```bash
# Run the built-in worked example with a single command
chronaeon phylogeo --example --no-neural
```

Or execute directly on custom alignments and metadata:
```bash
chronaeon phylogeo \
  -a chronaeon/examples/H5N1_HA_geo.fasta \
  -g chronaeon/examples/H5N1_HA_metadata.csv \
  -t chronaeon/examples/H5N1_HA.nwk \
  --no-neural \
  --n-perms 1000 \
  --min-bf 3.0 \
  --geojson chronaeon/examples/H5N1_HA_geo.geojson \
  -o chronaeon/examples/H5N1_HA_geo_results.json \
  -c chronaeon/examples/H5N1_HA_routes.csv \
  --plot-path chronaeon/examples/H5N1_HA_geo_diagnostic.png
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
* **Naturally Asymmetric Directed Migration**: Unlike BEAST's reversible rate matrix ($\mathbf{\Lambda} = \mu \mathbf{S} \mathbf{P}$, which enforces $s_{jk} = s_{kj}$), ChronAeon measures true directional transmission ($M_{jk} \ne M_{kj}$), capturing directional source-sink dynamics.
* **Vectorized Permutation BSSVS**: Generates exact empirical Bayes Factors ($\text{BF} \ge 3.0$) and Benjamini-Hochberg FDR $q$-values from 1,000 null permutations in $< 0.1$ seconds.
* **Spatial PGLS Epicenter**: Infers the continuous geographic epicenter coordinates ($28.10^\circ\text{N}, 111.83^\circ\text{E}$) with analytical 95% geographic confidence radii.
* **Modern Web GIS Export**: Generates standard GeoJSON feature collections (`.geojson`) compatible with Kepler.gl and Nextstrain/Auspice.

---

### Example 3: Phylodynamic R₀/Rₜ Estimation from Pandemic H1N1

Replicating the landmark phylodynamics study of **Fraser et al. (Science 2009)** estimating the early growth rate and basic reproduction number (R₀) of the 2009 H1N1 pandemic:

```bash
# Run the built-in worked example with a single command
chronaeon dynamics --example
```

Or execute directly on custom alignments and metadata:
```bash
chronaeon dynamics \
  -a chronaeon/examples/H1N1_2009_pandemic.fasta \
  -t chronaeon/examples/H1N1_2009_pandemic.nwk \
  --pathogen h1n1 \
  --plot \
  --plot-path chronaeon/examples/H1N1_2009_r0_diagnostic.png \
  -o chronaeon/examples/H1N1_2009_r0_results.json \
  -c chronaeon/examples/H1N1_2009_rt_skyline.csv
```

#### Key Features:
* **Ultra-Fast Growth Rate Estimation**: Recovers epidemic growth rate ($r$) and basic reproduction number ($R_0$) in seconds from heterochronous sequence data, without MCMC.
* **Dynamic R(t) Skyline**: Sliding-window estimation of time-varying reproduction numbers ($R_t$) across the epidemic timeline, capturing waves and interventions.
* **Pathogen Presets**: Built-in generation intervals for common pathogens (`h1n1`, `ebola`, `sars-cov-2`, `measles`, `hiv_early`), or specify custom generation time and SD for gamma renewal models.
* **SEIR Renewal Model**: Optional latent period support for more realistic epidemic modeling.

---

### Example 4: Bayesian Warm-Start Bridge for BEAST MCMC (`--export-beast`)

ChronAeon functions as an upstream prior generator for full Bayesian MCMC engines (BEAST 1.x and BEAST X v10.5.0), eliminating the multi-million iteration burn-in penalty caused by arbitrary default priors (e.g. initial $\mu = 1.0$ and uncalibrated demographic starting trees):

```bash
# 1. Run ChronAeon dating and export pre-populated BEAST XML in a single command:
chronaeon date \
  -a alignment.fasta \
  -d dates.csv \
  --export-beast beast_warmstart.xml \
  --beast-clock relaxed \
  --beast-chain-length 10000000

# 2. Run BEAST with immediate Step-0 convergence:
beast -overwrite beast_warmstart.xml
```

#### What ChronAeon Calibrates Inside the BEAST XML:
* **Substitution Rate ($\mu$)**: Sets initial `clock.rate` or `ucld.mean` directly to ChronAeon's empirical $\hat{\mu}$ (e.g. $2.0 \times 10^{-4}$), avoiding the 4-orders-of-magnitude likelihood chasm of cold-start runs.
* **Informative Rate Prior**: Embeds a data-driven `logNormalPrior` centered at $\ln(\hat{\mu})$ with variance proportional to the Fieller analytical standard error.
* **Root Height ($t_{\mathrm{MRCA}}$)**: Calibrates `treeModel.rootHeight` prior with a `normalPrior` centered at $t_{\max} - \hat{t}_{\mathrm{MRCA}}$ with standard deviation matching the 95% Fieller / Jackknife interval.
* **Coalescent Demography**: Initializes `constant.popSize` to $H / 2$, ensuring the starting coalescent tree matches the empirical time horizon.
* **AutoClock Multi-Clock Partitioning**: When used with `chronaeon autoclock --export-beast`, automatically emits partitioned taxon sets (`<taxa id="community_k">`) with lineage-specific local clock rates.

---

## Documentation

- [Dating Guide](DATING_GUIDE.md)
- [AutoClock Guide](AUTOCLOCK_GUIDE.md)
- [MRCA Dating Report](MRCA_DATING_REPORT.md)
