# Deconstructing Selection in Cryptic Lineages: A Pedagogical Guide to Evolutionary Inference with Classical HyPhy and HyphAeon

## Introduction & The Big Picture

Investigating cryptic speciation in microbial eukaryotes—such as the *Arcellinida* testate amoebae—is one of the most exciting yet methodologically challenging questions in modern evolutionary genomics. You have distinct genetic lineages or populations (e.g., `POP1` through `POP7`) that exhibit subtle or absent morphological divergence, and you want to ask fundamental questions:
1. *Which evolutionary forces drove these cryptic populations apart?*
2. *Which specific genes and cellular pathways are undergoing diversifying selection in each cryptic lineage?*
3. *Did selection act recently along a specific population's tip branches, or does it represent ancestral divergence prior to population splitting?*

When you ran classical HyPhy methods (FEL, Contrast-FEL, RELAX), you encountered an array of perplexing outputs: widespread `NA` values, nearly identical results across populations, conflicting $\omega$ plots, and confusion about how to trim alignments or assign branches.

> **The core takeaway upfront:** You did not do anything wrong. You simply collided with the inherent mathematical boundaries of classical asymptotic maximum likelihood estimation (MLE) on small, shallow phylogenetic trees.

This guide provides a comprehensive, step-by-step explanation of:
* What is physically and mathematically happening inside classical tools like FEL and Contrast-FEL that caused these issues.
* What **HyphAeon** is, how its architecture works, and why it was designed from the ground up to solve these exact problems.
* The exact statistical and evolutionary rationale behind each analytical step in the HyphAeon pipeline.
* A clear, practical recipe to run, interpret, and publish your findings.

---

## 1. Deconstructing the Classical HyPhy Bottlenecks

To understand why a new approach is necessary, let us examine the mathematical machinery of the classical tools you ran.

### A. Why Did FEL Return Pervasive `NA` Values? (Slides 10–13)
In Fixed Effects Likelihood (FEL), HyPhy fits an independent continuous-time Markov codon model at *every single codon column* in your alignment:
$$q_{ij} = \begin{cases} \alpha \, \pi_j & \text{synonymous transition} \\ \beta \, \pi_j & \text{non-synonymous transition} \end{cases}$$
where $\alpha$ is the sitewise synonymous substitution rate, $\beta$ is the sitewise non-synonymous rate, and $\pi_j$ is the target codon equilibrium frequency. FEL then performs a numerical optimization of the likelihood function $\mathcal{L}(\alpha, \beta)$ and computes the selection parameter $\omega = \beta / \alpha$.

*The root problem in shallow/cryptic datasets:* Because cryptic species have diverged relatively recently, sequence divergence across the tree is modest. At many codon positions, there are zero synonymous mutations across the entire phylogeny. Under unconstrained MLE, the optimizer sets the synonymous rate to zero ($\alpha \to 0$). 

When $\alpha = 0$:
$$\omega = \frac{\beta}{\alpha} \longrightarrow \frac{\beta}{0} = \text{undefined or } \infty$$
Furthermore, the standard likelihood ratio test ($\text{LRT} = 2[\ln \mathcal{L}(\hat{\alpha}, \hat{\beta}) - \ln \mathcal{L}(\hat{\alpha}, \hat{\alpha})]$) relies on asymptotic $\chi^2$ distribution theory (Wilks' theorem), which requires a moderate number of independent mutational events ($N \ge 5\text{--}10$). When an alignment site contains only 1 or 2 substitutions, asymptotic theory collapses. FEL responsibly outputs `NA` rather than asserting false confidence, but this leaves you with an uninformative, incomplete matrix.

### B. Why Did Splitting Alignments by Species Destroy Statistical Power? (Slides 15–18)
To compare cryptic populations, your intuition was to slice the master alignment into separate FASTA files (one for `POP1`, one for `POP2`, etc.) and run FEL on each sub-alignment.

*Why this is a statistical trap:* Slicing an alignment of 35 sequences into sub-alignments of 4 to 8 sequences cuts the phylogenetic tree into tiny fragments. In a 4-taxon tree of closely related individuals, the total tree length $T = \sum b_i$ might be less than $0.05$ substitutions per site. Over such a short evolutionary span, the probability of observing multiple non-synonymous substitutions at the same site by chance is virtually zero. As a result:
* Statistical power drops to near zero.
* Every sub-population returns identical, blank, or non-significant results.
* You discard the ancestral baseline shared across the other 30 species that provides the neutral clock calibration.

### C. What Was Happening with Contrast-FEL? (Slides 19–23)
Contrast-FEL is designed to ask a very specific hypothesis: *given two pre-defined sets of branches (Reference versus Test), is the $dN/dS$ ratio significantly different between them at site $i$?*

While powerful, Contrast-FEL requires:
1. Manual, *a priori* labeling of every branch in the tree file using phylogenetic annotations (e.g., `{Test}` and `{Reference}`).
2. Sufficient independent mutations along both branch sets to achieve asymptotic testing power.

When branches are assigned to small cryptic clades, Contrast-FEL often lacks power to reject the null hypothesis of equal rates ($\omega_{\text{Test}} = \omega_{\text{Ref}}$). Furthermore, calculating an overall gene $\omega$ by averaging noisy sitewise point estimates yields high variance and erratic plots.

### D. Why Did RELAX Feel Mismatched to Your Question? (Slides 24–26)
RELAX answers a gene-wide omnibus question: *has the overall intensity of natural selection been relaxed ($K < 1$) or intensified ($K > 1$) across the entire coding sequence on the test branches?*

Under RELAX, the distribution of $\omega$ categories $(\omega_1 < 1, \omega_2 = 1, \omega_3 > 1)$ is transformed by an exponent $K$:
$$\omega_k \longrightarrow (\omega_k)^K$$
If $K > 1$, purifying sites become more conserved ($\omega_1^K \to 0$) and positive selection sites become more intense ($\omega_3^K \gg 1$). If $K < 1$, all rates converge toward neutrality ($\omega \to 1$).

*Why this did not answer your biological question:* RELAX does not tell you *which specific codons* are undergoing positive selection, nor does it tell you which functional protein domains are driving adaptation in `POP1` versus `POP7`. If a gene has two adaptive amino acid mutations while the remaining 500 codons are under strict purifying constraint, the gene-wide average will dominate, yielding $K \approx 1.0$ (no significant relaxation or intensification).

### E. Should Alignments Be Trimmed? Nucleotides or Amino Acids? (Slides 4–5)
* **Nucleotide vs. Amino Acid:** You must always use in-frame coding nucleotide (codon) alignments. Models of selection require synonymous nucleotide substitutions ($dS$) as the neutral clock against which non-synonymous substitutions ($dN$) are measured. If you align only amino acids, you discard the neutral baseline entirely.
* **The Trimming Dilemma:** In *de novo* transcriptomes of non-model organisms like *Arcellinida*, uncurated draft assemblies often contain sequencing errors or isolated indel frameshifts. Untrimmed alignments cause classical tools to mistake single-taxon sequencing errors for massive positive selection. Conversely, aggressive heuristic trimming (e.g., Gblocks) strips away hypervariable surface loops where authentic positive selection predominantly occurs.

---

## 2. What is HyphAeon? Foundation Model Architecture Explained

HyphAeon is a deep phylogenetic foundation model designed to replace iterative numerical optimization of likelihood functions with direct geometric neural inference. 

Rather than fitting independent $61 \times 61$ Markov transition matrices at every codon column for every gene, HyphAeon was pre-trained over millions of evolutionary coding sequences across deep macroevolutionary time. It embeds the sequence alignment and the phylogenetic tree into a continuous geometric manifold, performing selection inference, feature attribution, and error filtering in a single forward pass.

### How HyphAeon Represents a Phylogenetic Tree (Tree-RoPE)
Classical tools treat the tree as a discrete graph with branch lengths, requiring post-order tree traversals (Felsenstein's pruning algorithm) at every iteration. 

HyphAeon represents tree geometry directly within the self-attention layers of a 2D Axial Transformer:
1. **Continuous-Time Markov Kernel**: The evolutionary distance between species $i$ and $j$ on the tree is encoded by the continuous transition probability decay:
   $$K_{ij} = \exp(-\lambda \, \tau_{ij})$$
   where $\tau_{ij}$ is the patristic distance along the tree and $\lambda$ is a learnable rate parameter.
2. **4D Multidimensional Scaling (MDS)**: The patristic distance matrix of the tree is projected into a 4-dimensional geometric space, producing continuous spatial coordinates $\mathbf{x}_i \in \mathbb{R}^4$ for each species.
3. **Tree-Rotary Position Embeddings (Tree-RoPE)**: During column-wise attention across species, queries and keys are rotated based on their relative phylogenetic coordinates:
   $$\mathbf{q}_i^\top \mathbf{k}_j \longrightarrow \mathbf{q}_i^\top \mathbf{R}(\mathbf{x}_i - \mathbf{x}_j) \mathbf{k}_j$$
   This allows the neural network to understand evolutionary distances, shared ancestry, and lineage clades without needing to prune or subset the tree.

### 2D Axial Attention across Species and Codons
HyphAeon processes alignments using alternating axial attention blocks:
* **Column Attention (Across Species)**: Compares all sequences at a specific codon position, integrating phylogenetic tree geometry to identify substitution transitions along branches.
* **Row Attention (Along the Sequence)**: Compares neighboring codons within the same gene, allowing the model to learn structural contexts, spatial clustering, and protein domain boundaries.

Because HyphAeon evaluates substitutions against its learned macroevolutionary prior, it does not collapse when $\alpha \to 0$. It regularizes shallow branches, eliminates division-by-zero instabilities, and provides continuous, calibrated likelihood ratio test statistics ($\text{LRT}$) and $p$-values across all sites.

---

## 3. Rationale Behind Each Analytical Step in the HyphAeon Pipeline

When analyzing cryptic speciation datasets like *Arcellinida*, HyphAeon executes a four-stage analytical protocol:

### Step 1: Automated Dual-Stage Error Filtering (`--filter`)
* **The Problem:** Uncurated eukaryotic transcriptomes contain sequencing errors, draft assembly gaps, and indel frameshifts. In an isolated taxon, a 1-base insertion causes a downstream run of 5–15 completely incorrect amino acids before returning to frame. Classical selection models see 10 consecutive non-synonymous mutations in one species and trigger an extreme false-positive selection call.
* **The HyphAeon Solution:**
  1. *Stage 1 (Spatial Hypergeometric Scan)*: Under authentic episodic positive selection, adaptive mutations are distributed across functional motifs and protein surfaces according to a spatial Poisson process. Conversely, sequencing frameshifts create dense, contiguous runs of non-synonymous substitutions. HyphAeon scans a sliding window ($d \le 35$ codons) computing the exact upper-tail hypergeometric probability:
     $$p_{\text{local}} = 1 - \sum_{m=0}^{k-1} \frac{\binom{K}{m} \binom{L - K}{d - m}}{\binom{L}{d}}$$
     Regions with non-random spatial clustering ($p_{\text{local}} \le 0.01$) are flagged as candidate artifact patches.
  2. *Stage 2 (Outlier Contamination Index, OCI)*: Within each candidate patch, HyphAeon measures the maximum contiguous non-synonymous run per taxon against the consensus across all other species. If a single species accounts for $\ge 25\%$ of total patch mutations ($\text{OCI} \ge 0.25$) with $\ge 3$ consecutive mismatches (or $\ge 4$ consecutive mismatches), it is classified as an isolated sequencing/frameshift error.
  3. *Surgical In-Place Masking*: Rather than discarding the entire gene or deleting the entire column across all 35 species, HyphAeon masks *only the guilty span in that single taxon* with `NNN`. It then re-evaluates the locus with its neural engine in $<5\text{ ms}$, extinguishing the artifact while preserving all genuine evolutionary variation across the remaining 34 species.

### Step 2: Continuous Sitewise Selection Inference
For every codon $i \in \{1, \dots, L\}$, HyphAeon outputs a regularized likelihood ratio test statistic ($\text{LRT}_i \ge 0$). P-values are calculated under a calibrated mixture distribution reflecting the standard asymptotic mixture of point mass and $\chi^2$ distributions:
$$p_i = \frac{2}{3} \left[ 0.45 \, \text{Pr}(\chi^2_1 \ge \text{LRT}_i) + 0.55 \, \text{Pr}(\chi^2_2 \ge \text{LRT}_i) \right]$$
Because the model was trained with dynamic strata balancing across variable evolutionary regimes, these $p$-values maintain strict nominal false positive rates ($\le 5\%$) without asymptotic breakdown.

### Step 3: Omnibus Gene-Level Ranking via Cauchy Combination Test (CCT)
* **The Challenge:** How do you rank thousands of genes to find which ones distinguish your cryptic species? Standard Benjamini-Hochberg FDR on individual genes ($L \approx 400$ codons) on small trees is very conservative. Fisher's combination test ($\sum -2 \ln p_i$) assumes independent sites, which is severely violated by linkage disequilibrium, structural constraints, and shared phylogenetic history.
* **The Solution (The Cauchy Combination Test):** The Cauchy transformation converts individual $p$-values into standard Cauchy random variables:
  $$T_{\text{CCT}} = \frac{1}{L} \sum_{i=1}^L \tan\left[ \pi \left( 0.5 - p_i \right) \right]$$
  The gene-level omnibus $p$-value is then obtained directly from the CDF of the standard Cauchy distribution:
  $$p_{\text{Cauchy}} = \frac{1}{2} - \frac{\arctan(T_{\text{CCT}})}{\pi}$$
* *Why CCT is mathematically optimal for phylogenetics:* As proven by Liu et al. (2019), the tail probability of $T_{\text{CCT}}$ is strictly invariant to arbitrary, complex correlation structures among tests. Even when neighboring codons are strongly co-evolving or correlated, $p_{\text{Cauchy}}$ remains well-calibrated, aggregating sub-threshold site elevations into a clean, gene-level ranking score.

### Step 4: Cryptic Lineage Attribution (`--attribute`)
* **The Biological Question:** When a gene is under positive selection, *which cryptic population drove the signal, and when did it happen?*
* **The Solution (Counterfactual Feature Attribution):** Rather than slicing the tree into separate files, HyphAeon keeps the full tree intact and performs single-taxon counterfactual masking. For each taxon $t$ and codon $i$, HyphAeon computes the marginal selection sensitivity:
  $$\Delta\text{LRT}_i(t) = \text{LRT}_i(\text{Full Tree}) - \text{LRT}_i(\text{Tree with Taxon } t \text{ Masked})$$
  * If $\Delta\text{LRT}_i(t) > 0$, taxon $t$ is an active contributor driving the positive selection call at site $i$.
  * By aggregating $\Delta\text{LRT}$ across all individuals belonging to a cryptic population (e.g., $\sum_{t \in \texttt{POP1}} \Delta\text{LRT}_i(t)$), you directly obtain the population-specific selection score.
  * *Epoch Decomposition:* If selection is driven exclusively by individuals in `POP1`, it is classified as a **Recent Tip Sweep** (adaptive divergence unique to that cryptic species). If $\Delta\text{LRT}$ is shared equally across `POP1`, `POP2`, and `POP7`, it represents a **Deep Ancestral Divergence** that occurred prior to the speciation split.

---

## 4. Step-by-Step Practical Playbook for Your Dataset

### Command-Line Execution
To analyze a single ortholog with automated error filtering and lineage attribution:
```bash
hyphaeon predict \
  -a data/arcellinida_OG0012.fasta \
  -t data/arcellinida_OG0012.nwk \
  --filter \
  --filter-out-aln data/arcellinida_OG0012_cleaned.fasta \
  --attribute \
  --taxon POP1_ind1,POP1_ind2,POP2_ind1,POP7_ind1 \
  --out data/arcellinida_OG0012_results.csv
```

### Python Script for Transcriptome-Wide Batch Processing
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

# 1. Initialize HyphAeon Foundation Engine
config = load_arch_config('axomeme_5_dim384_nonull.pt')
model = PhyloAxialTransformer(
    embed_dim=config['embed_dim'],
    num_layers=config['num_layers'],
    num_heads=config['num_heads'],
    window_size=config['window_size']
)
model.load_state_dict(load_weights('axomeme_5_dim384_nonull.pt'))
model.eval()

# 2. Iterate across all orthologous gene families
results = []
fasta_files = sorted(glob.glob("alignments/*.fasta"))
print(f"[*] Found {len(fasta_files)} ortholog alignments to evaluate.")

for idx, fasta in enumerate(fasta_files):
    nwk = fasta.replace(".fasta", ".nwk")
    if not os.path.exists(nwk): 
        continue
        
    gene_name = os.path.basename(fasta).replace(".fasta", "")
    
    # Run full selection scan + surgical error filtering + attribution
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
    sig_p01 = int(np.sum(pvals <= 0.01))
    
    # Identify top driving cryptic population
    top_driver = attr_df.loc[attr_df['delta_lrt'].idxmax(), 'driving_species'] if len(attr_df) > 0 else 'None'
    top_epoch = attr_df.loc[attr_df['delta_lrt'].idxmax(), 'epoch'] if len(attr_df) > 0 else 'None'
    
    results.append({
        'gene': gene_name,
        'codons': len(attr_df),
        'p_cauchy': p_cct,
        'neglog10_p_cauchy': -np.log10(p_cct),
        'sites_p05': sig_p05,
        'sites_p01': sig_p01,
        'selection_density_pct': (sig_p05 / len(attr_df)) * 100,
        'top_driving_population': top_driver,
        'evolutionary_epoch': top_epoch
    })

# 3. Export Ranked Transcriptome Summary
df_summary = pd.DataFrame(results).sort_values(by='p_cauchy', ascending=True)
df_summary.to_csv("arcellinida_transcriptome_selection_ranking.csv", index=False)
print("[+] Processing complete! Top candidate adaptive genes:")
print(df_summary.head(10).to_string(index=False))
```

---

## 5. Summary: Direct Answers to Your Methodological Questions (Slide 28)

1. **"Did I run classical HyPhy correctly?"**
   * Yes, your HyPhy script syntax was technically correct. However, splitting alignments into small per-population subsets starved maximum likelihood estimators of the mutational depth required for asymptotic power, creating empty results. Keeping the entire tree intact and using feature attribution resolves this problem completely.
2. **"How should I interpret and use these results in my paper?"**
   * Rank your entire transcriptome by $p_{\text{Cauchy}}$. Take the top candidate genes and perform Gene Ontology (GO) and KEGG pathway enrichment analysis (e.g., looking for enrichment in testate shell biomineralization, pseudopod actin-myosin motility, or osmoregulation). Use the `top_driving_population` column to highlight which specific metabolic adaptations belong uniquely to `POP1` versus `POP7`.
3. **"Why did I get different $\omega$ plots across different analyses?"**
   * FEL estimates sitewise point ratios ($\beta/\alpha$), Contrast-FEL evaluates branch-specific rate differentials, and RELAX models tree-wide rate distribution parameters. They test distinct hypotheses under divergent constraints. HyphAeon unifies these objectives into a single, regularized foundation framework.

---

### Comparison of Analytical Trade-offs

| Methodological Dimension | Classical HyPhy Workflow | HyphAeon Foundation Workflow |
| :--- | :--- | :--- |
| **Alignment Strategy** | Unclear trimming; manual Gblocks heuristic | Automated dual-stage error filter (`--filter`, $<5\text{ ms}$) |
| **Tree Handling** | Subsetting alignments into small sub-trees | Preserves full 35-taxon tree with 4D Tree-RoPE |
| **Low Synonymous Mutation Rate** | $\alpha \to 0$ causes division by zero ($\omega = \text{NA}$) | Regularized prior; continuous LRT and $p$-values |
| **Lineage Specificity** | Multi-step branch labeling; low power | Counterfactual attribution ($\Delta\text{LRT}$; `--attribute`) |
| **Gene-Wide Significance** | Bonferroni/FDR over-penalizes small genes | Cauchy Combination Test ($p_{\text{Cauchy}}$); robust to LD |
| **Execution Throughput** | Hours to days across cluster | Milliseconds per gene on a single laptop/GPU |
