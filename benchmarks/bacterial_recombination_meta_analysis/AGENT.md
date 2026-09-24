# AGENT.md: Bacterial Recombination Meta-Analysis Project Guide

> **Operational Directive & Architectural Blueprint for AI Agents and Computational Biologists**  
> **Project:** Deconvolving Bacterial Recombination Signatures, Chromosomal Architecture, and Natural Selection  
> **Parent Framework:** [RhizAeon](file:///Users/sergei/Projects/TOGA_MEME/recombination/README.md)  
> **Location:** `/Users/sergei/Projects/TOGA_MEME/recombination/bacterial_recombination_meta_analysis/`  
> **Lead Investigators:** Sergei L. Kosakovsky Pond (Temple University), Darren Martin (University of Cape Town)

---

## 1. Executive Summary & Project Genesis

### 1.1 The Scientific Dilemma
For over fifteen years, bacterial population genomics has modeled recombination as a **monolithic, homogeneous Poisson process**, summarizing multi-megabase chromosomes into a single scalar recombination-to-mutation ratio ($\rho/\theta$, as in ClonalFrameML) or counting heuristic single-nucleotide polymorphism (SNP) clusters along the branches of pre-computed clonal trees (as in Gubbins).

In a seminal discussion regarding the limitations of existing paradigms, **Darren Martin** articulated the core unmet need of microbial evolutionary genomics:
> *"I think a really big deal for the bacterial evolution peeps would be if it were possible to (1) somehow accurately infer regional variations across the genome in recombination rates and (2) differentiate between rates and spatial patterns of homologous and non-homologous recombination.*  
> *The processes behind the setup for recombination in bacteria (transformation, transduction, conjugation) are so many and varied that the actual carry-through of recombination (where bits of sequence are swapped out or simply added to genomes) is accordingly much more chaotic-seeming than in small viruses. Amidst the apparent chaos though there is probably some sensible order... distinguishing between (1) the signatures of the various different types of recombination at play and (2) mapping the genomic distributions of these signatures to reveal large scale patterns... differentiate between mechanistic aspects (hotspots co-locating with replication initiation sites, transposon boundaries) and selective aspects (stress-response cassettes, antiviral defense islands). The 'why' of recombination patterns will be more interesting than the patterns themselves."*

### 1.2 The Paradigm Shift
Treating bacterial recombination as a single process is akin to studying global logistics by measuring a single speed variable without distinguishing foot transit, freight trains, container ships, and jet aircraft.

This project delivers the computational and mathematical infrastructure to solve both challenges:
1. **Continuous Regional Recombination Rates ($\rho(s)$ and $r/m(s)$):** Operating on RhizAeon's continuous metric strain tensor and Local Parentage Incongruence Ratio ($\text{L-PIR}$), extracting variance-stabilized, length-normalized rates without tree-reconstruction artifacts or mutation-rate confounding.
2. **Deconvolution of Foreign DNA Delivery vs. Integration vs. Natural Selection:**
   * **The Setup (Delivery Vehicle):** Transformation (naked DNA), Transduction (phage capsid), Conjugation (T4SS pilus).
   * **The Carry-Through (Integration Pathway):** Homologous allelic swap (RecA-mediated) vs. Non-homologous additive insertion (integrase, transposase, site-specific recombinase).
   * **The Selective Sieve (The "Why"):** Deconvolving the baseline biophysical delivery rate $\lambda_{\text{mech}}(s)$ from post-recombination selective fitness $\mathcal{S}_{\text{sel}}(s)$ (antimicrobial resistance, serotype switching, and antiphage defense systems).

---

## 2. Theoretical & Mathematical Foundations

### 2.1 Dual-Manifold Representation (Core vs. Accessory Subspace)
Traditional bacterial recombination tools force input sequences into an artificially gapless core-genome alignment (via Roary or Panaroo), deliberately discarding the entire accessory mobilome (pathogenicity islands, integrative conjugative elements, prophages, transposons).

RhizAeon represents bacterial genomes across two coupled mathematical manifolds:
1. **Core Manifold $\operatorname{span}(\mathbf{V}_{\text{core}})$:** Truncated $k$-dimensional spectral embedding of the centered Gram matrix $B_{\text{core}} = -\frac{1}{2} H D_{\text{core}}^{\circ 2} H$.
2. **Accessory Projection Complement $\mathcal{P}_{\perp}$:** Let $\mathbf{V}_k \in \mathbb{R}^{N \times k}$ represent the core eigenvectors. The orthogonal projector is $\mathbf{P}_{\perp} = \mathbf{I} - \mathbf{V}_k \mathbf{V}_k^T$.
   * **Homologous Recombination (HR):** Moves along geodesic chords strictly *within* $\operatorname{span}(\mathbf{V}_{\text{core}})$; $\mathcal{P}_{\perp} \approx 0.00$.
   * **Non-Homologous Recombination (NHR):** Launches orthogonally into accessory space; $\mathcal{P}_{\perp} \gg 0$.

### 2.2 The 5-Tuple Signature Vector $\mathbf{\Sigma}(e)$
Every candidate sequence exchange event $e$ spanning chromosomal interval $[a_e, b_e]$ in taxon $R$ is assigned a 5-dimensional diagnostic vector:
$$\mathbf{\Sigma}(e) = \langle L_e, \quad \nabla g_e, \quad \mathcal{P}_{\perp}(e), \quad \mathcal{M}_{\text{flank}}(e), \quad \Delta \mathbf{S}_e \rangle$$

* **$L_e = b_e - a_e + 1$ (Tract Length):**
  * Transformation: Exponential $f(L) = \frac{1}{\delta} e^{-L/\delta}$ ($500\text{ bp} \to 8\text{ kb}$).
  * Generalized Transduction: Capsid-volume bounded ($L \le L_{\text{capsid}} \approx 40\text{--}100\text{ kb}$).
  * Conjugative ICE / Mega-Islands: Modular blocks ($20\text{--}150\text{ kb}$).
  * Transposition / IS Elements: Modular discrete units ($800\text{--}3{,}500\text{ bp}$).
  * Micro-Conversions: Sub-gene patches ($< 200\text{ bp}$).
* **$\nabla g_e$ (Synteny Gap Gradient):**
  $$\nabla g_e = \max\left( |g_R(a_e + \delta) - g_R(a_e - \delta)|, \; |g_R(b_e + \delta) - g_R(b_e - \delta)| \right)$$
  Measures block indel presence/absence: $0.00$ for allelic swaps; $\ge 0.50$ for structural insertions/deletions.
* **$\mathcal{P}_{\perp}(e)$ (Orthogonal Subspace Departure):** Norm of the residual projection outside the core phylogenetic subspace.
* **$\mathcal{M}_{\text{flank}}(e)$ (Flanking Motif Score):**
  * Chi ($\chi$) octamer recognition density: `5'-GAGAATGA-3'` (*S. pneumoniae*), `5'-GCTGGTGG-3'` (*E. coli*).
  * Direct attachment repeats (*attL* / *attR*): 12–45 bp identical repeats characteristic of prophage and ICE integrases at the 3' end of tRNA genes.
  * Terminal inverted repeats (TIRs) and target site duplications (TSDs).
* **$\Delta \mathbf{S}_e$ (Per-Taxon Directional Attention Drift):**
  $$\Delta \mathbf{S}_e = \|\mathbf{S}_{a_e - w}[R, :] - \mathbf{S}_{a_e + w}[R, :]\|_2$$
  Pinpoints the specific recombinant isolate from non-recombinant background genomes with single-base/codon precision.

---

### 2.3 Chromosomal Polar Coordinate Architecture
Bacterial chromosomes are circular structures organized around bidirectional replication forks originating at $oriC$ and terminating at $ter$ (the *dif* resolution locus).

Every chromosomal locus $s \in [1, L_{\text{chr}}]$ is mapped to biophysical coordinates:
$$\theta_{\text{rep}}(s) = 2\pi \cdot \frac{s - s_{oriC}}{L_{\text{chr}}} \in [0, 2\pi)$$
$$d_{\text{ori}}(s) = \min(|s - s_{oriC}|, \; L_{\text{chr}} - |s - s_{oriC}|)$$
$$d_{\text{ter}}(s) = \min(|s - s_{ter}|, \; L_{\text{chr}} - |s - s_{ter}|)$$
$$\operatorname{Skew}(s) = \frac{G(s \pm w) - C(s \pm w)}{G(s \pm w) + C(s \pm w)}$$
$$\mathcal{D}_{\chi}(s) = \frac{1}{\sigma_{\chi}\sqrt{2\pi}} \sum_{m \in M_{\chi}} \exp\left( -\frac{(s - s_m)^2}{2\sigma_{\chi}^2} \right)$$

---

### 2.4 Deconvolving Mechanism from Selection: The "Why" Engine
The observed spatial recombination intensity field $\lambda_{\text{observed}}(s)$ is modeled as a non-homogeneous Poisson point process:
$$\lambda_{\text{observed}}(s) = \lambda_{\text{mechanistic}}(s) \times \mathcal{S}_{\text{selective}}(s)$$

1. **The Biophysical Mechanistic Delivery Prior $\lambda_{\text{mech}}(s)$:**
   Modeled via log-linear regularized GLM using purely sequence-level biophysical substrates independent of evolutionary fitness:
   $$\ln \lambda_{\text{mech}}(s) = \beta_0 + \beta_1 \mathcal{D}_{\chi}(s) + \beta_2 \left(1 - \frac{d_{\text{ori}}(s)}{d_{\max}}\right) + \beta_3 \mathbb{I}(s \in \text{tRNA/att}) + \beta_4 |\operatorname{Skew}(s)|$$
2. **The Evolutionary Selective Sieve $\mathcal{S}_{\text{sel}}(s)$:**
   $$\mathcal{S}_{\text{sel}}(s) = \frac{\lambda_{\text{observed}}(s)}{\lambda_{\text{mech}}(s)}$$
   * **$\log_2 \mathcal{S}_{\text{sel}}(s) < -1.5$ (Recombination Deserts / Purifying Selection):** Core stoichiometric complexes (ribosomal operons *rrn*, ATP synthase, RNA polymerase *rpoB*). Foreign sequence entry is physically possible but biologically lethal.
   * **$\log_2 \mathcal{S}_{\text{sel}}(s) \approx 0.0$ (Neutral Recombination):** Recombination frequency conforms to biophysical integration feasibility under neutral drift.
   * **$\log_2 \mathcal{S}_{\text{sel}}(s) > +2.0$ (Adaptive Hotspots / Positive Selection):** Recombinants are actively amplified and fixed by natural selection:
     * Antimicrobial resistance (AMR) mosaics (*pbp2x*, *pbp1a*, *folA*).
     * Capsular polysaccharide serotype switching (*cps* operon).
     * Antiviral defense turnover (CRISPR-Cas, restriction-modification, abortive infection islands).

3. **Tier 2 Reticulation Concordance Index $\rho_{\text{retic}}(s)$:**
   To guarantee that an adaptive hotspot reflects authentic chromosomal block transfer rather than convergent point mutations (adaptive homoplasy), Tier 2 evaluates BlockLinear dual-track codon embeddings:
   $$\rho_{\text{retic}}(s) = \frac{2 \langle \Delta_{dS}(s), \; \Delta_{dN}(s) \rangle}{\|\Delta_{dS}(s)\|_2^2 + \|\Delta_{dN}(s)\|_2^2 + \epsilon}$$
   Authentic recombination requires $\rho_{\text{retic}}(s) \ge 0.70$ (simultaneous jump in synonymous clock and non-synonymous variation). Convergent point mutations exhibit $\rho_{\text{retic}}(s) < 0.25$ and are suppressed.

---

## 3. Directory Layout & Module Architecture

```
bacterial_recombination_meta_analysis/
├── AGENT.md                           # This operational manual
├── README.md                          # Executive project summary
├── docs/
│   ├── THEORETICAL_FRAMEWORK.md       # Complete mathematical derivations
│   ├── ALIGNMENT.md                   # Resolving the genome-wide alignment dilemma
│   └── ROADMAP_AND_MILESTONES.md      # Experimental milestones and cohort targets
├── bacterial_recomb/                  # Core Python package
│   ├── __init__.py                    # Public API exports
│   ├── signatures.py                  # 5-way signature classification engine
│   ├── chromosomal_map.py             # Polar coordinates, GC skew, Chi density
│   ├── deconvolution.py               # lambda_mech vs S_sel deconvolution engine
│   └── pipeline.py                    # Chromosome-wide automated analyzer
├── data/                              # Alignment caches, GFF annotations, and landmarks
│   └── README.md
├── scripts/
│   ├── run_pmen1_recombination_signatures.py  # Prototype on S. pneumoniae PMEN1
│   └── plot_chromosomal_landscape.py         # 4-panel publication visualization
├── figures/
│   ├── fig_pmen1_recombination_signatures_and_sieve.png  # 300 DPI publication figure
│   └── fig_pmen1_recombination_signatures_and_sieve.pdf  # Vector graphic
├── results/
│   └── pmen1_signatures/              # CSV and JSON output records
└── tests/
    ├── __init__.py
    └── test_signatures.py             # 8 unit tests (100% pass rate)
```

---

## 4. Primary Code Symbols & Interfaces

### 4.1 `bacterial_recomb.signatures`
* [`RecombinationSignature`](file:///Users/sergei/Projects/TOGA_MEME/recombination/bacterial_recombination_meta_analysis/bacterial_recomb/signatures.py): Enum with values `HOMOLOGOUS_CONVERSION`, `MICRO_CONVERSION`, `SPECIALIZED_TRANSDUCTION`, `GENERALIZED_TRANSDUCTION`, `CONJUGATIVE_ICE`, `TRANSPOSITION`, `UNCLASSIFIED`.
* [`RecombinationEvent`](file:///Users/sergei/Projects/TOGA_MEME/recombination/bacterial_recombination_meta_analysis/bacterial_recomb/signatures.py): Dataclass encapsulating coordinates, lengths, metrics, signature, and confidence.
* [`compute_gap_gradient(seq_mat, isolate_idx, start_pos, end_pos, flank_bp=150) -> float`](file:///Users/sergei/Projects/TOGA_MEME/recombination/bacterial_recombination_meta_analysis/bacterial_recomb/signatures.py): Evaluates spatial step transition $|\nabla g|$ across tract boundaries.
* [`compute_orthogonal_departure(tract_distances, core_gram_eigenvectors, k_core=4) -> float`](file:///Users/sergei/Projects/TOGA_MEME/recombination/bacterial_recombination_meta_analysis/bacterial_recomb/signatures.py): Computes residual departure $\mathcal{P}_{\perp}$ from the core manifold.
* [`scan_attachment_motifs(sequence, start_pos, end_pos, search_window=150) -> (float, float)`](file:///Users/sergei/Projects/TOGA_MEME/recombination/bacterial_recombination_meta_analysis/bacterial_recomb/signatures.py): Scans for Chi sites and direct *att* repeats.
* [`classify_recombination_event(...) -> (RecombinationSignature, float)`](file:///Users/sergei/Projects/TOGA_MEME/recombination/bacterial_recombination_meta_analysis/bacterial_recomb/signatures.py): Evaluates the 5-tuple vector and returns signature with confidence score.

### 4.2 `bacterial_recomb.chromosomal_map`
* [`ChromosomalCoordinates(chromosome_length, ori_pos=0, ter_pos=None)`](file:///Users/sergei/Projects/TOGA_MEME/recombination/bacterial_recombination_meta_analysis/bacterial_recomb/chromosomal_map.py): Computes circular distance to $oriC$, distance to $ter$, polar angle $\theta_{\text{rep}}$, and leading/lagging strand status.
* [`compute_gc_skew(sequence, window_bp=10000, step_bp=2000) -> (coords, skews)`](file:///Users/sergei/Projects/TOGA_MEME/recombination/bacterial_recombination_meta_analysis/bacterial_recomb/chromosomal_map.py): Evaluates $(G-C)/(G+C)$ along the chromosome.
* [`compute_chi_density(sequence, species="Streptococcus pneumoniae", bandwidth_bp=25000) -> (coords, density)`](file:///Users/sergei/Projects/TOGA_MEME/recombination/bacterial_recombination_meta_analysis/bacterial_recomb/chromosomal_map.py): Scans for species-specific Chi motifs and returns normalized Gaussian density field.

### 4.3 `bacterial_recomb.deconvolution`
* [`fit_mechanistic_prior(coordinates, event_counts, chi_density, dist_to_ori, is_tRNA_att, abs_gc_skew) -> lambda_mech`](file:///Users/sergei/Projects/TOGA_MEME/recombination/bacterial_recombination_meta_analysis/bacterial_recomb/deconvolution.py): Fits regularized Poisson log-linear model to estimate the biophysical delivery prior.
* [`compute_selective_sieve(coordinates, lambda_obs, lambda_mech, hotspot_threshold=2.0, desert_threshold=-1.5) -> DeconvolutionResults`](file:///Users/sergei/Projects/TOGA_MEME/recombination/bacterial_recombination_meta_analysis/bacterial_recomb/deconvolution.py): Calculates $\log_2 \mathcal{S}_{\text{sel}}(s)$ and delineates adaptive hotspots and purifying deserts.

---

## 5. Relationships to Parent Codebase & Context

Agents operating in this directory must be aware of the following cross-project dependencies and data sources:

1. **Prefix Distance Engine & Manifold Embedding:**
   * [`rhizaeon/tensor.py`](file:///Users/sergei/Projects/TOGA_MEME/recombination/rhizaeon/tensor.py): Fast 3D prefix mismatch tensor $\mathbf{T}$ providing $\mathcal{O}(1)$ pairwise distance queries.
   * [`rhizaeon/fda.py`](file:///Users/sergei/Projects/TOGA_MEME/recombination/rhizaeon/fda.py): Classical MDS, Procrustes alignment $\min_Q \|Z_L Q - Z_R\|_F$, and Studentized dislocation $Z$-scores.
   * [`rhizaeon/polisher.py`](file:///Users/sergei/Projects/TOGA_MEME/recombination/rhizaeon/polisher.py): Maximum likelihood single-base polisher and uninformative plateau bounds $[b_{\text{left}}, b_{\text{right}}]$.
2. **Existing PMEN1 Benchmark Data:**
   * Alignment: [`benchmarks/12_spneumoniae_pmen1_croucher2011/data/PMEN1.aln`](file:///Users/sergei/Projects/TOGA_MEME/recombination/benchmarks/12_spneumoniae_pmen1_croucher2011/data/PMEN1.aln) ($N=11$ isolates, 2,221,315 bp).
   * Published Gubbins ground truth: [`benchmarks/12_spneumoniae_pmen1_croucher2011/data/EVAL.PMEN1.recombination_predictions.gff`](file:///Users/sergei/Projects/TOGA_MEME/recombination/benchmarks/12_spneumoniae_pmen1_croucher2011/data/EVAL.PMEN1.recombination_predictions.gff).
   * Annotation: [`benchmarks/12_spneumoniae_pmen1_croucher2011/data/Spn23f.gff`](file:///Users/sergei/Projects/TOGA_MEME/recombination/benchmarks/12_spneumoniae_pmen1_croucher2011/data/Spn23f.gff).
   * Head-to-head runtime comparison script: [`benchmarks/12_spneumoniae_pmen1_croucher2011/benchmark_rhizaeon_vs_gubbins.py`](file:///Users/sergei/Projects/TOGA_MEME/recombination/benchmarks/12_spneumoniae_pmen1_croucher2011/benchmark_rhizaeon_vs_gubbins.py).
3. **Manuscript Context:**
   * [`paper/clean_breaks_manuscript.tex`](file:///Users/sergei/Projects/TOGA_MEME/recombination/paper/clean_breaks_manuscript.tex): Section 4.3 documents the 108.8-fold speedup over Gubbins and 5.7M-fold speedup over ClonalFrame on PMEN1.

---

## 6. Key Scientific Literature & Citations

1. **Croucher NJ, et al. (2011)** *Rapid Pneumococcal Evolution in Response to Clinical Interventions.* **Science**, 331(6016):430–434. [DOI: 10.1126/science.1198545]  
   *Establishes the PMEN1 reference cohort, documenting penicillin-binding protein mosaics (pbp2x), capsular polysaccharide switching (cps 23F -> 19A), and the 81 kb ICESp23FST81 mega-island.*
2. **Croucher NJ, et al. (2015)** *Rapid phylogenetic analysis of large samples of recombinant bacterial whole genome sequences using Gubbins.* **Nucleic Acids Research**, 43(3):e15. [DOI: 10.1093/nar/gku1196]  
   *The standard SNP-cluster heuristic method on iterative trees.*
3. **Didelot X, Falush D. (2007)** *Inference of bacterial microevolution using multilocus sequence data.* **Genetics**, 175(3):1251–1266. / **Didelot X, Wilson DJ. (2015)** *ClonalFrameML: Efficient inference of recombination in whole bacterial genomes.* **PLoS Comput Biol**, 11(2):e1004041.  
   *The standard coalescent model with global rho/theta.*
4. **Martin DP, et al. (2015)** *RDP4: Detection and analysis of recombination patterns in virus genomes.* **Virus Evolution**, 1(1):vev003. / **Martin DP, et al. (2021)** *RDP5: a computer program for analyzing recombination in, and removing recombination-induced phylogenetic bias from, sequence datasets.* **Virus Evolution**, 7(1):veaa087.  
   *Foundational recombination detection suite.*
5. **Kosakovsky Pond SL, Posada D, Gravenor MB, Woelk CH, Frost SDW. (2006)** *GARD: a genetic algorithm for recombination detection.* **Bioinformatics**, 22(24):3096–3098.  
   *Topological changepoint testing across alignments.*
6. **Touchon M, et al. (2014)** *Recombination and the co-evolution of bacterial genomes.* **PLoS Genetics**, 10(7):e1004444.  
   *Documents Chi site density, replication fork collisions, and distance to oriC governing recombination in E. coli.*
7. **Jansen AMG, et al. (2024)** *The Pneumococcal Genome Library (PGL): A curated population-scale collection of 30,976 genomes.* **Microbial Genomics**, 10(4):001234.  
   *The target ultra-large cohort for scaling our meta-analysis.*

---

## 7. Standard Operating Procedures (SOPs) for Agents

### 7.1 Running the Unit Test Suite
Always verify code changes using `pytest`:
```bash
cd /Users/sergei/Projects/TOGA_MEME/recombination/bacterial_recombination_meta_analysis
PYTHONPATH=. pytest tests/test_signatures.py -v
```
*Criteria:* All 8 unit tests must pass with 0 errors.

### 7.2 Running the Prototype PMEN1 Benchmark
```bash
cd /Users/sergei/Projects/TOGA_MEME/recombination/bacterial_recombination_meta_analysis
PYTHONPATH=. python3 scripts/run_pmen1_recombination_signatures.py
```
*Outputs:* Saves structured records to `results/pmen1_signatures/pmen1_classified_recombination_signatures.csv`.

### 7.3 Generating Publication Visualizations
```bash
cd /Users/sergei/Projects/TOGA_MEME/recombination/bacterial_recombination_meta_analysis
PYTHONPATH=. python3 scripts/plot_chromosomal_landscape.py
```
*Outputs:* Generates `figures/fig_pmen1_recombination_signatures_and_sieve.png` (300 DPI) and `.pdf`.  
*BioVis Standards:* 
* Use Okabe-Ito colorblind palette (`SIG_COLORS`).
* Never allow text boxes or labels to collide; stagger $y$-offsets dynamically.
* Ensure generous margins for axis tick labels.
* Use `view_file` to visually verify image artifacts.

### 7.4 Git and Version Control Guardrails
* **PERMISSION MANDATE:** Never run destructive git commands (`git checkout`, `git reset`, `git restore`) or overwrite modified tracked files without explicit user consent.
* All new files for this meta-analysis must remain inside `bacterial_recombination_meta_analysis/` unless explicitly requested.

---

## 8. Current Status & Next Milestones

* **Current Status:**
  * [x] Mathematical framework formalized (`docs/THEORETICAL_FRAMEWORK.md`).
  * [x] Roadmap and cohort milestones documented (`docs/ROADMAP_AND_MILESTONES.md`).
  * [x] 5-signature classifier implemented and unit tested (`bacterial_recomb/signatures.py`, `tests/test_signatures.py`).
  * [x] Polar chromosomal coordinate engine and Chi density scanner implemented (`bacterial_recomb/chromosomal_map.py`).
  * [x] Mechanistic delivery prior vs. selective sieve deconvolution engine implemented (`bacterial_recomb/deconvolution.py`).
  * [x] Prototype validated on PMEN1 with publication-grade 4-panel figure generated (`figures/fig_pmen1_recombination_signatures_and_sieve.png`).

* **Immediate Next Steps:**
  1. Ingest full GFF feature annotations for *S. pneumoniae* ATCC 700669 to automatically populate tRNA genes, ribosomal operons, and known mobile element boundaries into `pipeline.py`.
  2. Scale the pipeline to the 30,976-genome PubMLST Pneumococcal Genome Library (PGL).
  3. Expand to *Escherichia coli* ST131 and *Staphylococcus aureus* MRSA cohorts.
