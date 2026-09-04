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
   * Symmetrized column-attention maps recover true tertiary protein contact maps ($C_\beta - C_\beta < 8\text{\AA}$) with top-$L/5$ precision exceeding $78\%$. Vectorized Monte Carlo permutation testing (`--n-permutations`, `--max-perm-p`) validates multi-residue epistatic sectors against random $K$-site graph nulls.
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
pip install -e axomeme_repo
```

---

## 🚀 CLI Reference: Prediction, Attribution, and Error Detection

HyphAeon provides a unified command-line tool (`hyphaeon` or `python3 -m axomeme.cli`) with full support for selection inference, mechanistic feature attribution, and automated alignment error detection.

```bash
hyphaeon predict -a <alignment.fa> [-t <tree.nwk>] [--attribute] [--filter] [options]
```

### 1. Basic Site-Level Selection Prediction

Run fast neural inference to compute sitewise Likelihood Ratio Test ($\text{LRT}$) statistics, asymptotic $p$-values, and Benjamini-Hochberg False Discovery Rate ($q$-values):

```bash
# Standard selection scan (auto-detects embedded trees or fits branch lengths if unscaled)
hyphaeon predict \
  -a examples/HIV1_RT.fasta \
  -t examples/HIV1_RT.nwk \
  -c results_selection.csv \
  -o results_selection.json
```

**Key Arguments**:
* `-a, --alignment`: Path to in-frame codon FASTA or NEXUS alignment (required).
* `-t, --tree`: Path to Newick or NEXUS phylogenetic tree (optional; if omitted, HyphAeon extracts the embedded tree or estimates branch lengths via HyPhy).
* `-c, --csv`: Path to export sitewise selection statistics as CSV.
* `-o, --output`: Path to export complete structured JSON results.
* `--cpu`: Force CPU inference (defaults to CUDA or Apple Silicon MPS acceleration).

---

### 2. Mechanistic Feature Attribution (`--attribute`)

Standard phylogenetic selection methods report a single $p$-value per site without explaining **which species drive the signal** or **when in evolutionary time selection occurred**. The `--attribute` engine resolves this using single-taxon counterfactual sensitivity analysis.

```bash
# Enable mechanistic feature attribution for all candidate selected sites
hyphaeon predict \
  -a examples/HIV1_RT.fasta \
  -t examples/HIV1_RT.nwk \
  --attribute \
  --attribution-min-lrt 3.84 \
  -o results_attributed.json
```

#### How Feature Attribution Works:
1. **Single-Taxon Counterfactual Perturbation ($\Delta\text{LRT}$)**:
   * For each selected site ($\text{LRT} \ge \text{min\_lrt}$), HyphAeon in silico mutates each non-consensus taxon back to the ancestral/consensus state:
     $$\Delta\text{LRT}_k = \text{LRT}_{\text{observed}} - \text{LRT}_{\text{taxon } k \to \text{consensus}}$$
   * Taxa are ranked by their marginal contribution to the selection evidence ($\%\text{ Signal Explained}$).
2. **Evolutionary Epoch Decomposition**:
   * Computes the weighted patristic depth of driving mutations relative to the tree root, classifying the evolutionary timing into:
     * **Recent Terminal / Tip Sweep** ($\text{depth ratio} \ge 0.60$): Lineage-specific adaptation at outer branches.
     * **Intermediate Subclade Burst** ($0.35 \le \text{depth ratio} < 0.60$): Episodic burst fixed across a sub-family clade.
     * **Deep Ancestral / Basal Divergence** ($\text{depth ratio} < 0.35$): Ancient functional divergence near the root.
3. **Adaptation Mode**:
   * Categorizes the site as **Recurrent / Multi-Lineage Adaptation** (convergent evolution across $\ge 2$ independent clades) vs. **Single-Lineage Clade Sweep**.

#### Structured JSON Output for Attributed Sites:
```json
{
  "site_1indexed": 103,
  "predicted_lrt": 14.82,
  "consensus_aa": "K",
  "num_mutated_taxa": 12,
  "when_selection_occurred": {
    "evolutionary_epoch": "Intermediate Subclade Burst",
    "mode_of_adaptation": "Recurrent / Multi-Lineage Adaptation",
    "weighted_patristic_depth": 0.421,
    "tree_depth_ratio": 0.485
  },
  "driving_species": [
    {
      "taxon": "HIV1_Subtype_B_isolate_14",
      "observed_aa": "N",
      "consensus_aa": "K",
      "delta_lrt": 6.32,
      "pct_signal_explained": 42.6,
      "mean_patristic_depth": 0.452
    }
  ]
}
```

---

### 3. Automated Alignment Error & Artifact Filtering (`--filter`)

Sequencing errors, assembly insertions, pseudogene indels, and misaligned exon boundaries frequently create artificial runs of radical amino acid mutations in a single species, which numerical selection tools erroneously flag as "extreme positive selection." 

HyphAeon incorporates an automated **Dual-Stage Error Detection and Surgical Masking Engine**:

```bash
# Run selection inference with automated alignment error cleaning & export cleaned alignment
hyphaeon predict \
  -a examples/Smc6.fasta \
  -t examples/Smc6.nwk \
  --filter \
  --filter-out-aln cleaned_Smc6.fasta \
  --filter-p-thresh 0.01 \
  --min-patch-consec 3 \
  -c Smc6_clean_selection.csv
```

#### Dual-Stage Error Detection Architecture:
1. **Stage 1: Hypergeometric Local Patch Detection**:
   * Scans the alignment with a sliding window to detect spatial clusters of candidate selected sites that deviate from a Poisson spatial process ($p_{\text{local}} < 0.01$).
2. **Stage 2: Outlier Contamination Index (OCI) & Surgical Masking**:
   * Evaluates the distribution of non-synonymous mutations across taxa within each patch.
   * If a single taxon harbors $\ge 3$ consecutive non-synonymous mutations accounting for $\ge 25\%$ of the total patch mutations ($\text{OCI} \ge 0.25$), it is flagged as an **assembly/frameshift artifact**.
   * HyphAeon **surgically masks only the guilty taxon's anomalous span** with `NNN`, leaving valid sequence data across all other species intact.
3. **Automated Re-Evaluation**:
   * The neural model instantaneously re-evaluates the cleaned alignment, eliminating artifactual false positives while preserving legitimate multi-species selection.

---

## 🐍 Python API Reference

You can also integrate HyphAeon directly into Python pipelines for custom genomic workflows:

```python
import torch
from axomeme.dataset import load_alignment_and_tree
from axomeme.model import PhyloAxialTransformer
from axomeme.weights import load_weights, load_arch_config
from axomeme.attribution import attribute_selection

# 1. Load Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = load_arch_config()
model = PhyloAxialTransformer(
    embed_dim=config['embed_dim'],
    num_layers=config['num_layers'],
    num_heads=config['num_heads'],
    window_size=config['window_size']
).to(device)

state_dict = load_weights(map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

# 2. Ingest Alignment and Tree
c, a, d, z, inv, taxa, L = load_alignment_and_tree(
    "examples/HIV1_RT.fasta",
    "examples/HIV1_RT.nwk"
)

# 3. Precompute Tree Attention & Infer Sitewise LRTs
tree_cache = model.precompute_tree_cache(d.to(device), z.to(device))
with torch.no_grad():
    y_soft, _ = model.forward_cached(c.to(device), a.to(device), tree_cache)
    lrts = torch.clamp(y_soft.squeeze(-1), min=0.0).cpu().numpy().flatten()
    lrts[inv] = 0.0

# 4. Run Mechanistic Feature Attribution on Selected Sites
attributions = attribute_selection(
    model=model,
    c=c,
    a=a,
    d=d,
    z=z,
    inv=inv,
    taxa=taxa,
    min_lrt=3.84,
    base_lrts=lrts,
    cache=tree_cache
)

for site_idx, attr in attributions.items():
    print(f"Codon {attr['site_1indexed']}: LRT = {attr['predicted_lrt']:.2f} ({attr['when_selection_occurred']['evolutionary_epoch']})")
    for driver in attr['driving_species'][:3]:
        print(f"  -> {driver['taxon']}: {driver['observed_aa']} (Explains {driver['pct_signal_explained']:.1f}% of signal)")
```

---

## 🔬 Additional Unified Modules

* **Epistatic Co-Evolution & 3D Contact Mining (`hyphaeon epistasis`)**:
  ```bash
  # Run epistatic sector mining with 10,000 Monte Carlo graph permutations and p-value filtering
  hyphaeon epistasis \
    -a examples/HIV1_RT.fasta \
    -t examples/HIV1_RT.nwk \
    --n-permutations 10000 \
    --max-perm-p 0.05 \
    --min-coherence 0.50 \
    -o epistasis.json \
    --graphml network.graphml
  ```
  * Evaluates spectral coherence $C(\mathcal{S}) = \lambda_1 / \operatorname{Tr}$ against $B=10,000$ random $K$-site subgraphs drawn uniformly without replacement from active alignment sites.
  * Outputs empirical one-sided $p_{\text{perm}}$, null mean $\mathbb{E}[C_{\text{null}}]$, standard deviation, 95th percentile cutoff $C_{95}$, and theoretical isotropic baseline $1/K$. Set `--n-permutations 0` to disable permutation testing.

* **Digital Deep Mutational Scanning (DMS / ESSM) (`hyphaeon dms`)**:
  ```bash
  hyphaeon dms -a examples/HIV1_RT.fasta -t examples/HIV1_RT.nwk -o dms_landscape.json
  ```

* **PhyloWAS Phenotype-Genotype Association (`hyphaeon phenotype`)**:
  ```bash
  # Run directional PhyloWAS with trait sector permutations and phylogenetic permulations
  hyphaeon phenotype \
    -a examples/RHO.fasta \
    -fg "turTru,balMus,balPhys,orcOrc,delDelp,phyCat,phoVit,halGryp,mirLeo,zalCali,odoRos" \
    --n-permutations 10000 \
    --max-perm-p 0.05 \
    --permulations 1000 \
    -o trait_results.json \
    -c trait_sites.csv
  ```
  * **Trait Sector Permutations (`--n-permutations`, `--max-perm-p`)**: Vectorized Monte Carlo permutations testing whether spectral coherence among trait-associated sites ($\text{FDR } q \le \alpha$) significantly exceeds random $K$-site subgraphs.
  * **Brownian Motion Liability Permulations (`--permulations`)**: Simulates continuous neutral trait evolution along the phylogeny (RERconverge null model) to compute empirical gene-level $p$-values testing alignment-wide spectral energy ($\bar{\Psi}$) and peak site association ($\rho_{\max}$).

* **Alignment-Wide BUSTED Omnibus Selection (`hyphaeon busted`)**:
  ```bash
  hyphaeon busted -a examples/HIV1_RT.fasta -t examples/HIV1_RT.nwk
  ```

---

## 📄 Documentation & Manuscripts

* **Core Foundation Model Manuscript**: [`hyphaeon_paper/main.pdf`](hyphaeon_paper/main.pdf) / [`paper/hyphaeon_core_foundation_model.pdf`](paper/hyphaeon_core_foundation_model.pdf)
* **Architecture Specification**: [`docs/axomeme_architecture_spec.pdf`](docs/axomeme_architecture_spec.pdf)
* **Compensated Pathogenic Deviations (CPD) Atlas**: [`scratch/compensated_pathogenic_deviations_atlas.csv`](scratch/compensated_pathogenic_deviations_atlas.csv)

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
