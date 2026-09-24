# Dataset Provenance: Operational Boundaries and Frame Collapse Simulation Benchmark

## Canonical Citations & Conceptual Framework

> **Kosakovsky Pond, S. L., et al. (2026).**  
> *Clean Breaks: Tree-Free Discovery of Mosaic Genomes on Continuous Sequence Manifolds.*  
> In preparation.

> **Martin, D. P., et al. (2021).**  
> *RDP5: A computer program for analysing recombination in, and removing recombination from, multiple sequence alignments.*  
> **Molecular Biology and Evolution**, 38(6), 2690–2697.  
> DOI: [10.1093/molbev/msab063](https://doi.org/10.1093/molbev/msab063)

> **Didelot, X., & Wilson, D. J. (2015).**  
> *ClonalFrameML: Efficient inference of recombination in whole bacterial genomes.*  
> **PLoS Computational Biology**, 11(2), e1004041.  
> DOI: [10.1371/journal.pcbi.1004041](https://doi.org/10.1371/journal.pcbi.1004041)

> **Condat, L. (2013).**  
> *A direct algorithm for 1-D total variation denoising.*  
> **IEEE Signal Processing Letters**, 20(11), 1054–1057.  
> DOI: [10.1109/LSP.2013.2278363](https://doi.org/10.1109/LSP.2013.2278363)

---

## The Scientific Challenge: Darren Martin's Dilemma

During peer review and technical dialogue regarding tree-free manifold approaches for mosaic genome inference, Dr. Darren P. Martin formulated the central operational challenge for the field:

> *"How it works in practice when shit starts getting really complex and there is no longer a meaningfully/accurately fixable frame of reference."*

This benchmark was designed, executed, and analyzed to provide a definitive, mathematically grounded, and empirically validated answer to this exact challenge. Rather than treating frame collapse as a fatal bug or an unquantifiable edge case, this benchmark maps the phase space of sequence manifold geometry across 1,050 full synthetic genomes ($L = 3{,}000$ nt).

---

## Generative Experimental Matrix (Exactly 1,050 Alignments)

All synthetic alignments ($L = 3{,}000$ nt) were simulated under continuous-time Markov substitution processes (HKY85, transition/transversion ratio $\kappa = 2.5$, stationary nucleotide frequencies $\boldsymbol{\pi} = [0.30, 0.20, 0.20, 0.30]$) with continuous site-to-site Gamma distributed rate variation ($\alpha = 0.5$ for recombinant regimes, $\alpha = 0.25$ for high-heterogeneity clonal controls):

### 1. Experiment A: Mutational Information Floor $\mathcal{I}_{\text{mut}} = L_{\text{tract}} \cdot \Delta d$ (450 Alignments)
- Evaluates the lower physical boundary of distance-based and manifold derivative signal extraction:
  - 5 tract lengths: $L_{\text{tract}} \in \{50, 100, 250, 500, 1000\}$ bp.
  - 6 parental divergence levels: $\Delta d \in \{0.002, 0.005, 0.01, 0.03, 0.10, 0.25\}$.
  - 30 parameter cells $\times 15$ independent replicates = **450 alignments** ($N = 20$ taxa).
  - Mutational payload spans 3 orders of magnitude: from $\mathcal{I}_{\text{mut}} = 50 \times 0.002 = 0.1$ expected SNPs up to $1000 \times 0.25 = 250$ expected SNPs.

### 2. Experiment B: Panmictic Frame Collapse $\rho / \theta$ (240 Alignments)
- Evaluates the breakdown of a single global Euclidean coordinate frame under extensive reticulation:
  - Recombination-to-mutation ratio: $\rho / \theta \in \{0.0, 0.05, 0.2, 0.5, 1.0, 2.5, 5.0, 10.0\}$.
  - Taxonomic cohort size: $N \in \{20, 50\}$ taxa.
  - 16 parameter cells $\times 15$ independent replicates = **240 alignments**.
  - At $\rho / \theta = 0.0$, sequences evolve along a strictly clonal tree. At $\rho / \theta \ge 2.5$, chromosomes become multi-fragment mosaic patchworks assembled from dozens of ancestral donor lineages.
  - Tracks the Frame Rigidity Index: $\mathcal{F}_{\text{frame}} = (\lambda_1 + \lambda_2 + \lambda_3) / \sum_{i=1}^N \lambda_i$.

### 3. Experiment C: Distant Parents & Mutational Saturation (150 Alignments)
- Evaluates deep divergence, homoplasy, and multiple-hit ceiling saturation:
  - Divergence levels: $\Delta d(P_1, P_2) \in \{0.05, 0.15, 0.30, 0.45, 0.60\}$.
  - 2 tract lengths: $L_{\text{tract}} \in \{250, 1000\}$ bp.
  - 10 parameter cells $\times 15$ independent replicates = **150 alignments** ($N = 20$ taxa).

### 4. Experiment D: Clade Torque & Negative Clonal Controls (210 Alignments)
- Evaluates whole-clade reassortment, Procrustes rotational torque, and false positive calibration:
  - Part 1 (Clade Torque): Mosaic fraction $f_{\text{recomb}} \in \{1/N, 0.10, 0.25, 0.50\}$ across $\Delta d \in \{0.02, 0.05, 0.15\}$.
    12 parameter cells $\times 15$ replicates = **180 alignments**.
  - Part 2 (Negative Clonal Controls): **30 independent clonal alignments** under spatial Gamma rate variation ($\alpha = 0.25$) with zero recombination.
  - Total Experiment D: **210 alignments**.

**Total Dataset Count:** $450 + 240 + 150 + 210 = \mathbf{1{,}050}$ alignments.

---

## Standardized RhizAeon Execution Workflow

All alignments were processed through the standardized multi-engine RhizAeon pipeline:
```python
# 1. Prefix Distance Engine with continuous Markov transition acceleration
engine = PrefixDistanceEngine(mat, codon_aligned=False, compute_transitions=True)

# 2. Global Frame Rigidity Evaluation
D_glob = engine.query_distance_matrix(0, L)
F_frame = compute_frame_rigidity(D_glob)

# 3. Recursive Partitioning Functional Data Analysis (RP-FDA)
bps = run_recursive_partition_fda_screen(
    engine,
    taxa_names=taxa_names,
    min_z=1.8,
    min_pir=0.06,
    crossover_validation=True,
    crossover_p_threshold=0.005,
    min_informative_sites=3,
    polish_ml=True
)

# 4. Normalized Graph Laplacian Spectral Bipartitioning (Fiedler Vector Phase Shift)
v2_L = compute_fiedler_vector(D_left)
v2_R = compute_fiedler_vector(D_right)
d_Fiedler = 1.0 - np.dot(v2_L, v2_R)

# 5. Distribution-Free Conformal Prediction Set (Coverage 95%)
p0 = (1.0 + np.sum(calib_scores >= S_test)) / (n_calib + 1.0)
```

---

## Primary Artifact Registry

| Artifact Filename | Format | Description |
| :--- | :--- | :--- |
| `generator_envelope.py` | Python Script | Parametric continuous-time Markov evolver generating 1,050 alignments. |
| `run_envelope_benchmark.py` | Python Script | Parallel benchmark execution harness with metrics, Laplacian, and conformal prediction. |
| `plot_envelope_benchmark.py` | Python Script | BioVis-compliant publication plotting script producing Figure 4-panel phase diagram. |
| `data/manifest.json` | JSON | Complete metadata manifest for all 1,050 synthetic alignments. |
| `envelope_raw_results.csv` | CSV Table | Per-alignment raw results (1,050 rows, 27 metric columns). |
| `envelope_summary.csv` | CSV Table | Aggregated summary metrics across all 69 parameter cells. |
| `fig_operational_boundaries_phase_diagram.pdf` | Vector PDF | Publication-quality 4-panel phase diagram (300 DPI, vector typography). |
| `fig_operational_boundaries_phase_diagram.png` | Raster PNG | High-resolution raster rendering for digital inspection. |
| `OPERATIONAL_ENVELOPE_REPORT.md` | Markdown Report | Exhaustive technical report with Darren Martin decision matrix. |
