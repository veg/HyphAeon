# RhizAeon

**The Reticulation Engine** — Tree-Free Reticulate Evolution, Continuous Sequence Manifolds, and Recombination Detection.

RhizAeon operates along the **SEQUENCE / LOCUS** axis of the HyphAeon foundation model ecosystem, providing sub-second execution for continuous manifold tracking, prefix distance querying, Haar wavelet modulus maxima, Total Variation 1D ($L_1$ trend filtering) segmentation, maximum-likelihood breakpoint polishing, and distribution-free conformal prediction guarantees.

---

## 🎯 Target Audience

Evolutionary biologists, viral genomic surveillance teams (CDC, WHO, UKHSA), microbial epidemiologists, and researchers studying mosaic pathogen emergence (HIV-1, SARS-CoV-2, Influenza, Enteroviruses, Pneumococcus).

---

## 📦 Installation

RhizAeon requires Python ≥ 3.8 and PyTorch ≥ 2.0. At runtime it auto-selects the best available accelerator (CUDA → Apple MPS → CPU).

```bash
pip install rhizaeon
```

Or install in editable mode alongside sibling packages in the HyphAeon monorepo:

```bash
pip install -e aeon-core/ -e rhizaeon/[dev]
```

### Optional Dependencies
* `dev`: `pytest>=7.0` for running the unit and integration test suite.

---

## 🚀 CLI Subcommands

| Command | Description |
| :--- | :--- |
| `rhizaeon scan` | Sliding-window sequence manifold scan with Two-Tier prefix distance engines |
| `rhizaeon rp-fda` | Recursive Partitioning Functional Data Analysis (RP-FDA) with TV-1D & ML Polisher |
| `rhizaeon alluvial` | Render Alluvial Genome River plot visualizing reticulate phylogenetic flow |

---

## 🔬 Core Methodological Pillars

### 1. Continuous Sequence Manifolds
RhizAeon dispenses with discrete tree searches across partitions. Instead, it embeds windowed genetic distance matrices into low-dimensional Euclidean manifolds via **Classical Multidimensional Scaling (MDS)** or **Normalized Laplacian Eigenmaps**:
$$\mathbf{K}_{\text{norm}} = \mathbf{D}_{\text{deg}}^{-1/2} \mathbf{W} \mathbf{D}_{\text{deg}}^{-1/2}$$
Recombination events manifest as continuous kinetic inflections and trajectory deflections across manifold coordinates.

### 2. $O(1)$ Prefix Distance Tensor Engines
By precomputing 3D cumulative prefix difference tensors:
$$\mathbf{P}[u, i, j] = \sum_{k=1}^u \mathbf{d}(s_{i,k}, s_{j,k})$$
any sub-alignment distance matrix over interval $[u_1, u_2)$ is queried in $O(1)$ time:
$$\mathbf{D}_{[u_1, u_2)} = \mathbf{P}[u_2] - \mathbf{P}[u_1]$$
For multi-megabase bacterial genomes, RhizAeon provides the **`SNPCompressedPrefixEngine`**, compressing invariant sites to achieve up to 220× memory reduction.

### 3. Local Phylogenetic Incongruence Ratio (L-PIR) & Ghost Node $Z$-Scores
To distinguish authentic mosaic reticulation from rate heterogeneity (SRV) and homoplasy, RhizAeon pairs the L-PIR statistic:
$$\mathcal{L}_{\text{PIR}} = \frac{d(R, P_1) + d(R, P_2) - d(P_1, P_2)}{d(R, P_1) + d(R, P_2) + \epsilon}$$
with **Ghost Node $Z$-scores** derived from Procrustes residual tracking, proving that the recombinant lineage transitions between distinct parental basins of attraction.

### 4. Recursive Partitioning FDA (RP-FDA) & ML Breakpoint Polisher
For sub-nucleotide breakpoint precision, RP-FDA applies **Total Variation 1D ($L_1$ trend filtering)** to isolate step discontinuities in sequence divergence trajectories, followed by a **Maximum-Likelihood Polisher** that evaluates the exact binomial likelihood plateau:
$$\Delta \ln \mathcal{L}(c) = \ln \mathcal{L}(c \mid P_1, P_2) - \ln \mathcal{L}_0$$
returning finite uncertainty intervals $[\text{CI}_{\text{left}}, \text{CI}_{\text{right}}]$ bounded by informative flanking SNPs.

### 5. Two-Tier Adaptive Foundation Transformer
RhizAeon natively bridges scalar distance sieves with the **HyphAeon 2D Axial Transformer**. Tier 1 screens candidate recombination regions using rapid prefix lookups; Tier 2 dispatches full 384-dimensional latent token embeddings or contextual cross-taxa attention hidden states to verify disputed or ambiguous breakpoints.

---

## 🧬 Reproducible Walkthrough: HIV-1 KAL153 Mosaic

To analyze the canonical circulating recombinant form CRF01_AE / Subtype B mosaic isolate (`KAL153`):

```bash
# 1. Run Two-Tier sliding manifold scan
rhizaeon scan rhizaeon/examples/hiv1_kal153.fasta \
  --codon \
  --window 25 \
  --min-tract 35 \
  --json kal153_results.json \
  --alluvial kal153_river.png

# 2. Run high-resolution RP-FDA with ML Breakpoint Polisher
rhizaeon rp-fda rhizaeon/examples/hiv1_kal153.fasta \
  --min-len 40 \
  --min-z 2.0 \
  --json kal153_rpfda.json \
  --export-nexus kal153_partitions.nex
```

### Example JSON Output:
```json
{
  "alignment": "rhizaeon/examples/hiv1_kal153.fasta",
  "num_taxa": 11,
  "length": 8943,
  "runtime_sec": 0.042,
  "num_breakpoints": 4,
  "breakpoints": [
    {
      "breakpoint_nt": 2248,
      "ci_left": 2210,
      "ci_right": 2265,
      "plateau_width": 55,
      "log_likelihood_gain": 34.2,
      "recombinant": "KAL153",
      "parent_1": "CRF01_AE",
      "parent_2": "Subtype_B",
      "kinetic_z": 4.12,
      "l_pir": 0.384
    }
  ]
}
```

---

## 🧪 Testing

Run the full RhizAeon test suite with pytest:

```bash
pytest rhizaeon/tests/ -v
```

All unit tests execute in $< 5$ seconds on CPU.
