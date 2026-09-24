# RhizAeon Track 2: Diagnostic / Forensic Workflow
*Practitioner How-To & Case Study*

---

## 1. Scenario & Input Specification

RhizAeon analyzes aligned genetic sequence cohorts to locate recombination breakpoints, identify mosaic taxa, and trace parental lineages. It accepts a standard Multiple Sequence Alignment (MSA) in FASTA format:
$$X \in \{A, C, G, T, -\}^{N \times L}$$

### Input Data Regimes & Biological Scope
RhizAeon operates across a wide spectrum of biological systems, provided sequences can be aligned into homologous blocks:

| System | Example Datasets | Mean Divergence ($\bar{d}$) | Typical Window ($W$) | Expected Signature |
| :--- | :--- | :--- | :--- | :--- |
| **RNA Viruses** | HIV-1, Coronaviruses, HCV | $5\%\text{--}15\%$ | $200\text{--}300\text{ bp}$ | Macro-recombinant cassettes ($1\text{--}5\text{ kb}$) |
| **Bacteria (Core)** | *S. pneumoniae*, *N. meningitidis* | $0.5\%\text{--}2.0\%$ | $30\text{--}50\text{ SNPs}$ | Micro-conversions ($200\text{--}1{,}500\text{ bp}$) |
| **Eukaryotic Parasites** | *Plasmodium falciparum* (*pfcrt*) | $1.0\%\text{--}4.0\%$ | $150\text{--}300\text{ bp}$ | Meiotic crossovers & gene conversion |
| **Fungi & Yeasts** | *Saccharomyces* hybrids | $2.0\%\text{--}8.0\%$ | $200\text{--}500\text{ bp}$ | Chromosomal arm exchanges & chimeras |
| **Multi-Gene Arrays** | Human MHC / HLA, opsin genes | $3.0\%\text{--}10.0\%$ | $100\text{--}200\text{ bp}$ | Non-allelic interlocus gene conversion |

### Pre-Flight Quality Control Checklist
Before initiating RhizAeon, ensure your alignment satisfies these quality gates:
1. **Collinear Homology:** Sequences must share homologous, collinear coordinates. Alignments containing large unaligned structural inversions or non-homologous insertions must be pre-trimmed into collinear syntenic blocks.
2. **Gap Thresholding:** Sites where $>50\%$ of taxa contain gap characters (`-`) or ambiguous characters (`N`) should be stripped if alignment quality is suspect. RhizAeon treats gaps as informative mismatch states or projects over segregating positions.
3. **Cohort Size:** Works from $N = 4$ taxa up to $N = 30{,}000+$ taxa. For cohorts with $N > 300$, enable `--engine randomized` or RP-FDA.

---

## 2. Parameter Selection & Trade-Off Matrix

Configuring RhizAeon involves matching window resolutions and statistical filters to the evolutionary rate of your organism:

| Parameter | Recommended Default | High-Divergence Regime ($\bar{d} > 5\%$) | Low-Divergence Regime ($\bar{d} < 1\%$) | Operational Trade-Off |
| :--- | :--- | :--- | :--- | :--- |
| **`--min-len`** | `200` (nt) | `100` | `400` | Shorter lengths catch micro-conversions but increase sensitivity to stochastic noise. |
| **`--min-z`** | `2.5` | `3.0` | `2.0` | Peak prominence threshold. Higher values eliminate false positives; lower values recover subtle events. |
| **`--min-pir`** | `0.35` | `0.40` | `0.30` | Topological exchange ratio. Filters out single-lineage rate acceleration spikes. |
| **`--window`** | `25` (units) | `15` | `50` | Sliding window width $W$. Broad windows give stable distances; narrow windows give sharper boundaries. |
| **`--codon`** | Flag (`False`) | Optional | Optional | Enforces triplet coordinate framing for protein-coding genes (*pol*, *spike*, *pbp2x*). |
| **`--polish-ml`** | Flag (`True`) | Always Enable | Always Enable | Refines coarse window changepoints to single-base profile likelihood plateaus. |

---

## 3. Step-by-Step Diagnostic Pipeline

### Phase A: Screening & Fast Triage
For initial cohort surveillance, run the ultra-fast linear manifold scan to establish whether any mosaic sequences exist:

```bash
rhizaeon scan alignment.fasta \
    --window 25 \
    --min-tract 20 \
    --export-hyphy-json initial_screen.json
```

**Diagnostic Gate:**
* If the report returns `Alignment is CLONAL / NON-RECOMBINANT (0 breakpoints detected)`, stop here. The alignment can be safely modeled with a single bifurcating phylogenetic tree.
* If candidate breakpoints are flagged, proceed immediately to Phase B.

---

### Phase B: Execution & Checkpoint Validation (RP-FDA)
Run Recursive Partitioning Functional Data Analysis (RP-FDA) with single-base maximum likelihood polishing and automated alignment disassembly:

```bash
rhizaeon rp-fda alignment.fasta \
    --min-len 200 \
    --min-z 2.5 \
    --min-pir 0.35 \
    --polish-ml \
    --export-partitions ./nonrecombinant_partitions/ \
    --export-nexus partitioned_alignment.nex \
    --export-hyphy-json hyphy_partitions.json \
    --html report.html
```

#### Monitoring Output Diagnostics:
1. **Runtime:** Should scale linearly with alignment length ($<1\text{ s}$ for viral genomes; $<15\text{ s}$ for bacterial core genomes).
2. **Log-Likelihood Gain:** For each polished breakpoint, verify that `LL Gain > +5.0`. Gains below $+2.0$ indicate uninformative regions where parentage cannot be distinguished.
3. **Plateau Width ($\Delta$):** Note the interval between flanking informative mutations. A wide plateau ($\Delta > 100\text{ bp}$) reflects regional sequence identity between parents, not algorithmic uncertainty.

---

### Phase C: Progressive Dataset Disassembly & Tree Verification
RhizAeon does not require you to write custom slicing scripts. The `--export-partitions` flag automatically disassembles the alignment into independent non-recombinant FASTA files:
* `./nonrecombinant_partitions/partition_1.fasta` (e.g., nt 1–2824)
* `./nonrecombinant_partitions/partition_2.fasta` (e.g., nt 2825–8847)
* `./nonrecombinant_partitions/partition_3.fasta` (e.g., nt 8848–9953)

You can now dispatch independent maximum-likelihood tree estimation on each non-recombinant partition to verify topological discordance:

```bash
# Infer independent trees on each partition using IQ-TREE or FastTree
iqtree2 -s nonrecombinant_partitions/partition_1.fasta -m GTR+G -B 1000 --prefix tree_part1
iqtree2 -s nonrecombinant_partitions/partition_2.fasta -m GTR+G -B 1000 --prefix tree_part2
```

Verify that the recombinant taxon switches topological positions between `tree_part1` and `tree_part2`.

---

## 4. The Forensic Decision Matrix (Troubleshooting Triage)

When analyzing empirical data from clinical outbreaks or diverse global cohorts, use this triage matrix to diagnose ambiguous signatures:

| Symptom / Visual Signature | Underlying Numerical / Physical Cause | Corrective Diagnostic Action |
| :--- | :--- | :--- |
| **High kinetic peak ($Z > 5.0$) but L-PIR collapses ($\text{L-PIR} < 0.15$)** | **Spatial Rate Heterogeneity / Gene Boundary:** A hypervariable domain (e.g., HIV-1 *env* V3 loop or bacterial surface antigen) causes a sudden surge in distance without changing phylogenetic tree topology. | **Trust L-PIR filter:** The event is automatically discarded as a non-recombinant rate artifact. No parameter adjustments needed. |
| **Erratic Brownian jitter across entire chromosome (many low-$Z$ peaks)** | **Window Starvation (Too Few SNPs):** Window width $W$ is too small for the local mutation density, collapsing the metric projection into random Poisson noise. | **Increase window width:** Double `--min-len` or switch to SNP-compressed units (ensure $\ge 15\text{--}20$ polymorphic sites per window). |
| **Breakpoint detected, but Recombinant shows equal distance to all clades** | **Unsampled Ghost Lineage:** The true parental lineage is missing from the dataset; the recombinant was donated by an unsampled outgroup ($Z_{\text{ghost}} > 3.0$). | **Inspect Ghost Z-score:** Check `ghost_z` in the JSON output. If elevated, classify the event as a horizontal import from an unrepresented clade. |
| **Known micro-conversion ($<200\text{ bp}$) is missed in output** | **Window Oversmoothing:** A broad window ($W = 500\text{ bp}$) diluted the short recombinant signal with $80\%$ clonal flanking sequence. | **Lower `--min-len` to 50–100 bp:** Re-run with RP-FDA, which dynamically shrinks interval half-widths down to $\delta = 20\text{ bp}$. |
| **Recombinant taxon alternates repeatedly between two parents ($A-B-A-B$)** | **Complex Mosaicism / Gene Conversion:** Sequence underwent multiple consecutive template switches or secondary recombination. | **Inspect Partition Disassembly:** Check the partitioned FASTA outputs. Verify that each independent block produces consistent parental clustering. |

---

## 5. Reporting & Sanity Checklist

Before publishing or archiving recombination events inferred by RhizAeon, confirm that all five non-negotiable verification gates are satisfied:

- [ ] **1. Single-Base Likelihood Gain:** Every reported breakpoint exhibits an ML log-likelihood gain $\Delta \ln L \ge +5.0$ over a single-partition model.
- [ ] **2. Flanking Informative Mutations:** The reported ML plateau $[\text{CI}_{\text{left}}, \text{CI}_{\text{right}}]$ is strictly bounded by at least one informative SNP distinguishing Parent 1 from Parent 2 on each flank.
- [ ] **3. L-PIR Validation:** All reported events have an $\text{L-PIR} \ge 0.35$, confirming that the recombinant taxon switches nearest-neighbor phylogenetic clades across the junction.
- [ ] **4. Clonal Flank Stability:** Outside the breakpoint transition zone, the recombinant taxon maintains a flat, stable trajectory within its parental cluster ($Z < 2.0$).
- [ ] **5. Independent Tree Verification:** Reconstructing independent ML trees from the `--export-partitions` FASTA files confirms statistically significant topological incongruence for the recombinant lineage.
