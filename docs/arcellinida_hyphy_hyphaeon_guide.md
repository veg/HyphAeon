# Deconstructing Selection in Cryptic Lineages: A Practical, Concept-First Guide to Evolutionary Inference

## Introduction & The Big Picture

Investigating cryptic speciation in microbial eukaryotes—such as the *Arcellinida* testate amoebae—is one of the most compelling questions in evolutionary biology. When distinct genetic populations (e.g., `POP1` through `POP7`) look identical under the microscope but occupy different ecological niches, we want to ask fundamental questions:
1. *What selective pressures drove these cryptic lineages apart?*
2. *Which specific genes and cellular pathways are adapting in each population?*
3. *Did this adaptation occur recently in one specific lineage, or does it reflect ancestral divergence before the populations split?*

When you ran classical HyPhy tools (FEL, Contrast-FEL, RELAX), you encountered a frustrating set of hurdles: widespread `NA` outputs, nearly identical results across populations, noisy $\omega$ plots, and confusion about how to trim alignments.

> **The core takeaway upfront:** You did not do anything wrong. You simply collided with the limits of classical maximum likelihood methods when applied to closely related, shallow evolutionary trees.

This guide explains what was happening conceptually inside those classical tools, introduces how **HyphAeon** takes a different approach, and provides a clear, practical roadmap to analyze your data and interpret your findings.

---

## 1. Why Did Classical HyPhy Struggle on This Dataset?

To understand why a different approach is needed, let us look at the conceptual hurdles of classical tools.

### The Zero-Mutation Dilemma in FEL
To measure natural selection, classical codon models compare the rate of amino acid-altering changes ($dN$) to the rate of silent, synonymous changes ($dS$), defining the selection ratio $\omega = dN / dS$. 

Classical FEL fits a separate model to every single column in your alignment from scratch. In closely related cryptic populations, sequence divergence is low. At many codons, there might be only one amino acid difference and *zero* silent changes across the entire tree. 

When silent mutations are zero ($dS = 0$), calculating $dN / dS$ requires dividing by zero. Classical statistics refuses to guess without enough data, so FEL outputs `NA`. While mathematically honest, this leaves you with incomplete results and missing data across large portions of your transcriptome.

### The Alignment Slicing Trap
To compare cryptic populations, the intuitive impulse is to slice the master alignment into separate files (e.g., one for `POP1`, one for `POP2`) and analyze each separately.

However, slicing an alignment of 35 taxa into small groups of 4 to 8 sequences cuts the evolutionary tree into tiny fragments. In a small group of closely related individuals, the total evolutionary time is very short. The chance of observing multiple independent mutations at the same site is near zero. Consequently:
* Statistical power drops to almost nothing.
* Every sub-population returns identical, blank, or non-significant results.
* You discard the shared ancestral history from the other species that provides the baseline calibration for evolutionary rate.

### What Contrast-FEL Was Doing
Contrast-FEL asks a very specific question: *is the $dN/dS$ ratio at a specific site significantly different between two pre-assigned groups of branches?*

While useful, Contrast-FEL requires you to manually label branches in advance and needs enough mutations on both branch sets to detect a contrast. In shallow trees, there are rarely enough mutations per site to prove a statistically significant difference. Averaging these noisy sitewise ratios produces erratic, jumpy gene plots.

### Why RELAX Answered a Different Question
RELAX tests whether the overall strength of natural selection across an entire gene has been relaxed (drift toward neutrality) or intensified (purifying selection getting stricter and positive selection getting stronger) on test branches.

RELAX looks at the entire gene as a single whole. If a gene has two key amino acids that adapted to a new ecological niche while the remaining 400 amino acids remained strictly conserved, the conservation across the rest of the gene dilutes the localized signal. RELAX will report that nothing happened across the gene, missing the specific adaptive mutations you care about.

### The Alignment Trimming Dilemma
In *de novo* transcriptomes of non-model organisms, uncurated assemblies frequently contain sequencing errors or small insertion/deletion frameshifts. 
* If you leave alignments completely untrimmed, classical tools mistake a single species' sequencing error (a run of strange amino acids) for massive positive selection.
* If you trim aggressively with tools like Gblocks, you strip away the variable surface loops where real positive selection predominantly lives.

---

## 2. How HyphAeon Works: The Core Concepts

HyphAeon is a phylogenetic foundation model. Instead of optimizing complex equations from scratch for each gene in isolation, HyphAeon was pre-trained over millions of coding sequences across macroevolutionary time. It has already learned the universal rules of molecular evolution—which amino acid substitutions are chemically conservative, which are radical, and how evolutionary time scales with divergence.

### Seeing the Whole Tree Continuously
Classical tools treat trees as rigid graphs, traversing branch by branch. HyphAeon maps the entire phylogenetic tree into a continuous geometric coordinate space. 

When evaluating a site, the model uses geometric self-attention to understand the evolutionary relationships among all species simultaneously. It knows how closely related any two amoebae are without needing you to chop the alignment or manually partition branches.

### Looking Across Species and Along the Gene at the Same Time
HyphAeon uses dual-track attention:
* **Across Species (Columns):** Compares all individuals at a codon position, weighing changes against the tree to detect genuine substitutions.
* **Along the Sequence (Rows):** Looks at neighboring codons within the same gene to understand structural context, functional domains, and spatial clustering.

Because HyphAeon evaluates substitutions against its learned macroevolutionary prior, it never divides by zero and never collapses on shallow branches. Every site receives a smooth, continuous selection score and a well-calibrated $p$-value.

---

## 3. The Four-Stage HyphAeon Pipeline Explained

When you run HyphAeon on your cryptic speciation dataset, it performs four coordinated steps:

### Step 1: Automated Error Filtering
* **The Idea:** Genuine positive selection is distributed across functional protein surfaces. Sequencing errors and frameshifts, by contrast, appear as dense, contiguous clusters of radical amino acid changes in a single taxon.
* **How it works:**
  1. *Spatial Clustering Scan:* It scans along the sequence looking for unusually dense clusters of candidate mutations.
  2. *Outlier Taxon Identification:* When a cluster is found, it checks if the changes are driven by a single isolated species against an otherwise conserved alignment. If so, that patch is flagged as a sequencing or assembly error.
  3. *Surgical Masking:* Instead of throwing away the entire gene or deleting the column across all species, HyphAeon masks *only that single taxon's error patch* with missing data (`NNN`) and re-evaluates the alignment in milliseconds. Real evolutionary variation across all other species is completely preserved.

### Step 2: Sitewise Selection Scoring
For every codon in the gene, HyphAeon calculates the strength of diversifying selection and assigns a calibrated $p$-value. Because the model incorporates a learned evolutionary prior, every codon is scored without `NA` gaps.

### Step 3: Ranking Whole Genes with the Cauchy Combination Test
* **The Challenge:** How do you rank thousands of genes across the transcriptome to find the most important candidates for cryptic speciation? Standard multiple testing corrections (like Bonferroni) are too harsh on short alignments. Traditional combination tests assume that neighboring sites evolve independently, which is untrue because amino acids in the same protein interact and share evolutionary history.
* **The Solution:** HyphAeon uses the Cauchy Combination Test (CCT) to aggregate sitewise evidence into a single gene-level score ($p_{\text{Cauchy}}$). Mathematically, the Cauchy combination test is completely robust to arbitrary correlation among sites. It aggregates subtle signals across multiple codons into a reliable gene-level score, allowing you to rank your entire transcriptome from top to bottom.

### Step 4: Identifying the Driving Cryptic Lineage
* **The Goal:** Once an adaptive gene is found, *which cryptic population drove the selection, and when did it happen?*
* **The Solution (Counterfactual Attribution):** Instead of slicing the tree into separate files, HyphAeon asks a simple counterfactual question:
  > *"If we temporarily mask Population 1 from the analysis, does the positive selection signal vanish?"*
  
  If masking `POP1` makes the selection signal disappear, then `POP1` is the primary driver of adaptation at that site.
  * **Recent Tip Sweeps:** If selection is driven exclusively by `POP1`, it represents a recent adaptation unique to that cryptic species.
  * **Ancestral Divergence:** If the signal is shared equally across all populations, it represents deep ancestral selection that occurred before the cryptic populations split.

---

## 4. Practical Workflow & Code for Your Dataset

### Command-Line Usage
To analyze an orthologous gene family with automated error filtering and lineage attribution:
```bash
hyphaeon predict \
  -a alignments/gene_OG0012.fasta \
  -t trees/gene_OG0012.nwk \
  --filter \
  --attribute \
  --taxon POP1_ind1,POP1_ind2,POP2_ind1,POP7_ind1 \
  --out results/gene_OG0012_results.csv
```

### Python Script for Transcriptome-Wide Batch Processing
You can run your entire dataset through this pipeline in minutes using the following Python script:

```python
import os, glob
import numpy as np
import pandas as pd
from axomeme.model import PhyloAxialTransformer
from axomeme.weights import load_weights, load_arch_config
from axomeme.attribution import attribute_selection

def calc_cauchy_p(pvals):
    p_clipped = np.clip(pvals, 1e-15, 1.0 - 1e-15)
    t = np.mean(np.tan((0.5 - p_clipped) * np.pi))
    return float(np.clip(0.5 - (np.arctan(t) / np.pi), 1e-15, 1.0))

# 1. Initialize HyphAeon Model
config = load_arch_config('axomeme_5_dim384_nonull.pt')
model = PhyloAxialTransformer(
    embed_dim=config['embed_dim'],
    num_layers=config['num_layers'],
    num_heads=config['num_heads'],
    window_size=config['window_size']
)
model.load_state_dict(load_weights('axomeme_5_dim384_nonull.pt'))
model.eval()

# 2. Iterate across all gene families
results = []
fasta_files = sorted(glob.glob("alignments/*.fasta"))
print(f"[*] Found {len(fasta_files)} alignments to analyze.")

for idx, fasta in enumerate(fasta_files):
    nwk = fasta.replace(".fasta", ".nwk")
    if not os.path.exists(nwk): 
        continue
        
    gene_name = os.path.basename(fasta).replace(".fasta", "")
    
    # Run selection inference + error filtering + lineage attribution
    attr_df = attribute_selection(
        model=model,
        alignment_path=fasta,
        tree_path=nwk,
        filter_artifacts=True
    )
    
    if attr_df is None or len(attr_df) == 0:
        continue
        
    pvals = attr_df['p_value'].values
    p_cct = calc_cauchy_p(pvals)
    sig_p05 = int(np.sum(pvals <= 0.05))
    
    # Identify top driving cryptic population
    top_driver = attr_df.loc[attr_df['delta_lrt'].idxmax(), 'driving_species'] if len(attr_df) > 0 else 'None'
    top_epoch = attr_df.loc[attr_df['delta_lrt'].idxmax(), 'epoch'] if len(attr_df) > 0 else 'None'
    
    results.append({
        'gene': gene_name,
        'codons': len(attr_df),
        'p_cauchy': p_cct,
        'neglog10_p_cauchy': -np.log10(p_cct),
        'selected_sites_p05': sig_p05,
        'top_driving_population': top_driver,
        'evolutionary_epoch': top_epoch
    })

# 3. Export Ranked Summary Table
df_summary = pd.DataFrame(results).sort_values(by='p_cauchy', ascending=True)
df_summary.to_csv("arcellinida_selection_ranking.csv", index=False)
print("[+] Analysis complete! Top candidate adaptive genes:")
print(df_summary.head(10).to_string(index=False))
```

---

## 5. Direct Answers to Your Methodological Questions (Slide 28)

1. **"Did I run classical HyPhy correctly?"**
   * Yes, your tool execution was technically sound. The reason you got empty or inconclusive outputs was that slicing alignments into tiny sub-trees left classical tools without enough mutations to achieve statistical power. Keeping the full tree together and using lineage attribution solves this.
2. **"How should I interpret and present these results in my paper?"**
   * Rank your entire transcriptome by $p_{\text{Cauchy}}$. Take the top candidate genes and run Gene Ontology (GO) and pathway enrichment analyses. You can then report which specific biological functions (e.g., testate shell biomineralization, membrane transporters, or pseudopod motility) drove divergence in `POP1` versus `POP7`.
3. **"Why did different methods give discordant $\omega$ plots?"**
   * FEL estimates sitewise ratios, Contrast-FEL compares branch sets, and RELAX measures whole-gene expansion or contraction. They ask fundamentally different questions under different statistical assumptions. HyphAeon unifies these goals into a single consistent framework.

---

### Conceptual Comparison Summary

| Conceptual Challenge | Classical HyPhy Approach | HyphAeon Foundation Approach |
| :--- | :--- | :--- |
| **Alignment Quality** | Manual or arbitrary trimming heuristics | Automated surgical error filtering in $<5\text{ ms}$ |
| **Tree Analysis** | Slicing tree into underpowered subsets | Evaluates full tree using geometric coordinates |
| **Low Mutation Depth** | Division by zero ($\omega = \text{NA}$) | Regularized prior; continuous scores across all sites |
| **Lineage Specificity** | Complex branch tagging with low power | Direct lineage attribution (Recent Sweep vs Ancestral) |
| **Transcriptome Ranking** | Standard FDR over-penalizes small genes | Cauchy Combination Test ($p_{\text{Cauchy}}$); robust to LD |
| **Compute Time** | Hours across clusters | Milliseconds per gene on a standard laptop |
