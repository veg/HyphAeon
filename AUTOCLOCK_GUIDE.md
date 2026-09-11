# ChronAeon Hierarchical AutoClock & Pan-Viral Surveillance Suite

**Platform:** `HyphAeon` / `ChronAeon`  
**Module Location:** [`hyphaeon/autoclock.py`](hyphaeon/autoclock.py), [`hyphaeon/sketch.py`](hyphaeon/sketch.py), [`hyphaeon/alignment.py`](hyphaeon/alignment.py), [`hyphaeon/dating.py`](hyphaeon/dating.py)  
**Version:** 2.1.0  
**Authors:** Sergei L. Kosakovsky Pond & DeepMind Antigravity Pair Programmer  

---

## 1. Overview & Motivation

Real-world molecular epidemiology and pathogen genomic surveillance feeds (such as BV-BRC, GISAID, and NCBI Virus) receive massive, continuous streams of unaligned, variable-length, and frequently contaminated sequencing reads. Traditional phylodynamic pipelines fail on these streams because:
1. **Quadratic Alignment Bottleneck:** Multiple sequence alignment (MSA) of tens of thousands of genomes with insertions, deletions, and strand inversions scales as $\mathcal{O}(N^2 L^2)$ or $\mathcal{O}(N \log N \cdot L^2)$.
2. **The Single Molecular Clock Fallacy:** Fitting a single strict or relaxed clock across mixed serotypes, multi-decadal epochs, or zoonotic reservoirs introduces massive deep-divergence leverage that flattens slopes to non-positive rates ($\mu \le 0$) and breaks divergence-time estimation.
3. **Discrete Tree Over-Partitioning:** Standard binary tree algorithms force rapid star radiations into arbitrary bifurcations and lack principled stopping criteria to prevent over-clustering continuous epidemic waves.

`ChronAeon` solves these challenges via a modular **3-Tier Streaming Architecture**:
- **Tier 0 (Alignment-Free Topological Centrifuge):** Linear-time MinHash sketching ($\mathcal{O}(N \cdot s)$) that sifts contaminants and bins sequences to canonical reference profiles at $>400-700\text{ seqs/s}$.
- **Tier 1 (Reference-Guided Codon-Aware Threader):** Linear-time coordinate normalization ($\mathcal{O}(N \cdot L)$) producing frame-locked codon grids ($[L_{\text{ref}}, N, 3]$) with zero frameshifts at $100-540\text{ seqs/s}$.
- **Tier 2 (Hierarchical AutoClock Deconvolution):** Recursive manifold deconvolution ($\mathcal{O}(N \cdot m \cdot L)$) that isolates coherent evolutionary communities, filters out-of-sample ghost nodes, and infers emergence horizons ($t_{\text{MRCA}}$).

---

## 2. Architecture Specification

```
Raw Unaligned FASTA Feeds (Contaminants, Mixed Subtypes, Inverted Strands)
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Tier 0: Alignment-Free MinHash Centrifuge (hyphaeon.sketch)            │
│ • Canonical 15-mers, dual CRC32 hashing, bottom-s sketch (s=1024)      │
│ • Jaccard Sieve: J < 0.015 (Quarantine Contaminants), J >= 0.10 (Bin) │
│ • Throughput: 400 - 700 genomes/second                                │
└────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Tier 1: Reference-Guided Codon-Aware Threader (hyphaeon.alignment)     │
│ • 6-frame stop-codon audit for strand & reading frame detection       │
│ • C-accelerated affine translation alignment (Bio.Align.Pairwise)      │
│ • Coordinate projection: pads deletions '---', clips insertions        │
│ • Dense Frame-Locked MSA Grid: [L_ref, N, 3], zero frameshifts         │
│ • Throughput: 100 - 540 genomes/second                                │
└────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Tier 2: Hierarchical AutoClock Deconvolution (hyphaeon.autoclock)      │
│ • Adaptive Nyström Landmark Spectral Graph Embedding (m <= 2,500)      │
│ • Recursive Manifold Partitioning governed by 5 Invariant Criteria:    │
│   1. Sample Size Floor (N >= 2 * N_min)                                │
│   2. Timespan Floor (Delta_t >= 1.0 year)                              │
│   3. AICc Parsimony (Delta_AICc >= 15.0)                               │
│   4. Topological Cheeger Bottleneck (Laplacian eigengap)               │
│   5. Phylodynamic Rate Homogeneity (Spread < 15%)                      │
│ • Instantaneous LOOCV Ghost Node Triage (Sherman-Morrison rank-1)      │
│ • Closed-Form Fieller Confidence Intervals for t_MRCA                 │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Algorithmic Mechanics

### 3.1 Tier 0: MinHash Topological Centrifuge (`hyphaeon.sketch.AlignmentFreeCentrifuge`)

1. **Canonical $k$-mers:** For every $k$-mer window $w_i = S[i : i+k]$:
   $$k_c(w_i) = \min(w_i, \text{ReverseComplement}(w_i))$$
   Default $k = 15$ nucleotides.
2. **Dual-Seed CRC32 64-bit Hashing:**
   $$h(k_c) = (\text{CRC32}(k_c, 0\text{x}12345678) \ll 32) \mid \text{CRC32}(k_c, 0\text{x}87654321)$$
3. **Bottom-$s$ Sketching:**
   $$\Omega(S) = \text{bottom}_s \{ h(k_c(w_i)) \mid i = 1, \dots, |S| - k + 1 \}$$
   Default $s = 1{,}024$ integer hashes.
4. **Jaccard Distance & Sieve Audit:**
   $$\hat{J}(A, B) = \frac{|\Omega(A) \cap \Omega(B)|}{|\Omega(A) \cup \Omega(B)|}, \quad D_{\text{Mash}}(A, B) = -\frac{1}{k}\ln\left(\frac{2\hat{J}}{1+\hat{J}}\right)$$
   - $\hat{J} \ge 0.10$: Assigned to candidate reference profile.
   - $\hat{J} < 0.015$: Automatically quarantined as non-target contaminant (e.g., Neuraminidase spiked into Hemagglutinin feeds).

### 3.2 Tier 1: Reference-Guided Codon-Aware Threader (`hyphaeon.alignment.ReferenceCodonAligner`)

1. **Automated Frame Detection:**
   For sequence $S \in \{S, \text{RC}(S)\}$ and frame $f \in \{0, 1, 2\}$, selects $(f^*, S^*)$ minimizing internal stop codons:
   $$f^* = \arg\min_{f, S} \sum_{j=0}^{\lfloor (|S|-f)/3 \rfloor - 1} \mathbb{I}[\text{Translate}(S[f+3j:f+3j+3]) == \text{'*'}' ]$$
2. **C-Accelerated Affine Translation Alignment:**
   Pairs translated protein $P^*$ with reference amino acid sequence $P_{\text{ref}}$ using `Bio.Align.PairwiseAligner` with affine gap penalties ($\text{match}=+2, \text{mismatch}=-1, \text{gap}_{\text{open}}=-10, \text{gap}_{\text{extend}}=-1$).
3. **Reverse-Codon Projection:**
   Maps aligned amino acids back to nucleotide triplets:
   - Matches: Extracted from $S^*$.
   - Deletions relative to reference: Padded with `---`.
   - Insertions relative to reference: Clipped to maintain invariant coordinate dimensionality ($L_{\text{ref}}$).

### 3.3 Tier 2: Hierarchical AutoClock (`hyphaeon.autoclock.HierarchicalAutoClock`)

#### The Five Invariant Stopping Criteria
Recursion down the hierarchy tree at node $\mathcal{C}$ terminates when any of the following criteria is met:

1. **Criterion 1 (Sample Size Floor):**
   If $|\mathcal{C}| < 2 N_{\min}$ (default $N_{\min} = 25$), degrees of freedom are insufficient for reliable multi-split dating.
2. **Criterion 2 (Timespan Floor):**
   If $\Delta t(\mathcal{C}) = \max(t_i) - \min(t_i) < 1.0\text{ year}$, further splitting risks over-partitioning synchronous clonal transmission waves.
3. **Criterion 3 (AICc Parsimony):**
   Evaluates candidate community counts $K \in \{1, \dots, K_{\max}\}$:
   $$\text{AICc}(K) = 2 p_K + N \ln\left(\frac{\text{RSS}_K}{N}\right) + \frac{2 p_K (p_K + 1)}{N - p_K - 1}$$
   where $p_K = 3K - 1$. Splitting is accepted only if:
   $$\Delta \text{AICc} = \text{AICc}(K=1) - \min_{K \ge 2} \text{AICc}(K) \ge 15.0$$
4. **Criterion 4 (Cheeger Spectral Bottleneck):**
   Evaluates normalized Laplacian eigengap $\Delta\lambda_k = \lambda_{k+1} - \lambda_k$. If $\max_k \Delta\lambda_k < \varepsilon_{\lambda}$ (adaptively $5 \times 10^{-4}$ for Nyström mode) and $\Delta\text{AICc} < 100.0$, the manifold lacks modular bottlenecks.
5. **Criterion 5 (Phylodynamic Rate Homogeneity):**
   If the parent node exhibits a valid clock ($R^2 \ge 0.20, \mu > 0$), candidate subcommunities are evaluated for evolutionary rate spread:
   $$\frac{\max_k(\mu_k) - \min_k(\mu_k)}{\frac{1}{K}\sum_k \mu_k} < 0.15$$
   If rate spread is $<15\%$, sublineages evolve at indistinguishable molecular paces; recursion halts to preserve unified single-clock dynamics.

#### Out-of-Sample Ghost Node Triage
Within any leaf community, sequences are evaluated via closed-form Leave-One-Out Cross-Validation (LOOCV) using Sherman-Morrison rank-1 down-dating in $\mathcal{O}(N)$ time:
$$\hat{t}_{(-i)} = \frac{d_i - \hat{\beta}_{0, (-i)}}{\hat{\mu}_{(-i)}}$$
Sequences with $|z_i| = \frac{|\hat{t}_{(-i)} - t_i|}{\hat{\sigma}_{(-i)}} > 3.0$ are triaged as **Ghost Nodes** (e.g., hypermutated proviral DNA, sequencing assembly errors, or severe within-host immune escape) and isolated without distorting the primary clock regression.

#### Fieller Analytical Ratio Inversion for $t_{\text{MRCA}}$
The ancestral origin $t_{\text{MRCA}} = -\beta_0 / \mu$ is the ratio of two correlated Gaussian estimators. Rather than using fragile first-order Taylor approximations (Delta method) or slow MCMC sampling, `ChronAeon` derives the exact Fieller confidence intervals:
$$g = \frac{z_{\alpha/2}^2 \cdot \mathrm{Var}(\hat{\mu})}{\hat{\mu}^2}$$
When $g < 1$ (statistically significant positive rate), the exact confidence bounds are:
$$t_{\mathrm{MRCA}} \in \frac{\hat{t}_{\mathrm{MRCA}} - \frac{g \cdot \mathrm{Cov}(\hat{\beta}_0, \hat{\mu})}{\hat{\mu} \cdot \mathrm{Var}(\hat{\mu})} \pm \frac{z_{\alpha/2}}{\hat{\mu}} \sqrt{\mathrm{Var}(\hat{\beta}_0) - 2 \hat{t}_{\mathrm{MRCA}}\mathrm{Cov}(\hat{\beta}_0, \hat{\mu}) + \hat{t}_{\mathrm{MRCA}}^2 \mathrm{Var}(\hat{\mu}) - g \left(\mathrm{Var}(\hat{\beta}_0) - \frac{\mathrm{Cov}^2}{\mathrm{Var}(\hat{\mu})}\right)}}{1 - g}$$

---

## 4. Python API Usage

### 4.1 Tier 0: Sifting and Binning with MinHash Centrifuge

```python
from hyphaeon.sketch import AlignmentFreeCentrifuge

# Initialize centrifuge with canonical k=15, bottom-s=1024
centrifuge = AlignmentFreeCentrifuge(k=15, sketch_size=1024)

# Register canonical reference profiles
centrifuge.add_reference("H5_clade2344b", "ATGGAGAAAATAGTGCTTCTT...")
centrifuge.add_reference("H1N1_pdm09",    "ATGAAGGCAATACTAGTAGTT...")

# Sieve an incoming stream of unaligned, uncurated FASTA sequences
results = centrifuge.classify_fasta("raw_surveillance_feed.fasta", min_jaccard=0.10, quarantine_threshold=0.015)

print(f"Assigned to H5: {len(results['H5_clade2344b'])}")
print(f"Quarantined Contaminants: {len(results['quarantined'])}")
```

### 4.2 Tier 1: Codon-Aware Threading

```python
from hyphaeon.alignment import ReferenceCodonAligner

# Initialize threader with reference nucleotide CDS
aligner = ReferenceCodonAligner(ref_seq="ATGGAGAAAATAGTGCTTCTT...", ref_name="H5_reference")

# Thread unaligned nucleotide sequences into dense, frame-locked grid
result = aligner.align_batch(
    seq_dict={"seq_001": "ATGGAGAAAATAGTG...", "seq_002": "..."},
    max_workers=4
)
aligned_records = result["aligned_seqs"]
```

### 4.3 Tier 2: Hierarchical AutoClock Deconvolution

```python
from hyphaeon.autoclock import HierarchicalAutoClock

# Configure hierarchical engine
autoclock = HierarchicalAutoClock(
    max_depth=3,
    min_cluster_size=25,
    min_timespan=1.0,
    aic_delta_threshold=15.0,
    rate_homogeneity_threshold=0.15,
    n_landmarks=1024
)

# Fit deconvolution tree on frame-locked MSA with collection dates
tree = autoclock.fit(
    alignment_file="frame_locked_grid.fasta",
    dates={"seq_001": 2024.15, "seq_002": 2023.82, ...}
)

# Export interactive hierarchy and leaf rate summary
summary = autoclock.get_summary_table()
print(summary[["community_id", "n_taxa", "rate", "r2", "t_mrca", "stop_reason"]])
```

---

## 5. Command-Line Interface (CLI)

`HyphAeon` provides unified CLI entry points for pipeline integration:

```bash
# 1. Run MinHash Centrifuge on raw unaligned FASTA
hyphaeon sketch centrifuge \
    --input raw_stream.fasta \
    --references ref_profiles.fasta \
    --k 15 \
    --sketch-size 1024 \
    --output-dir ./centrifuge_bins/

# 2. Run Reference-Guided Codon Threader
hyphaeon align thread \
    --input ./centrifuge_bins/H5_clade2344b.fasta \
    --reference-protein ref_h5_protein.faa \
    --output-alignment ./h5_aligned_frame_locked.fasta

# 3. Run Hierarchical AutoClock Deconvolution
hyphaeon autoclock run \
    --alignment ./h5_aligned_frame_locked.fasta \
    --metadata ./surveillance_meta.csv \
    --date-col collection_date \
    --max-depth 3 \
    --output-dir ./autoclock_results/
```

---

## 6. Empirical Validation Across 128,000 Genomes

In full-scale benchmark testing across 7 viral pathogen systems from BV-BRC (127,998 total raw genomes), the 3-tier suite demonstrated:
- **Throughput:** Processed all 128k genomes (Tier 0 sketching, Tier 1 frame-locking, and Tier 2 deconvolution into 266 leaf communities) in **18.2 minutes** total wall-clock time on a single workstation.
- **Contamination Filtering:** Sifted 100% of non-target contaminants ($J < 0.015$; e.g. 40 NA reads spiked into HA feeds) with zero false positives.
- **Dating Accuracy:** Pinpointed the H5N1 clade 2.3.4.4b dairy cattle spillover horizon to $t_{\text{MRCA}} = 2023.79$ (late 2023), matching epidemiological records of the introduction into Texas herds.
