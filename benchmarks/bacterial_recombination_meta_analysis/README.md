# Bacterial Recombination Meta-Analysis: Mechanistic Signatures & Selective Landscapes

A computational genomics meta-analysis framework built on **RhizAeon's geometric manifold and dual-tier attention architecture** to resolve the fundamental "why" of bacterial recombination.

---

## 1. Executive Scientific Vision

For decades, bacterial population genomics has modeled recombination as a **monolithic, homogeneous Poisson process**, summarizing chromosomes into a single scalar recombination-to-mutation ratio ($\rho/\theta$) or counting heuristic SNP clusters on pre-computed clonal trees (e.g., in Gubbins or ClonalFrameML).

This framework breaks this paradigm to address two foundational questions articulated by Darren Martin:
1. **Accurately inferring continuous regional variations in recombination rates ($\rho(s)$ and $r/m(s)$)** without tree-reconstruction bias or mutation-rate confounding.
2. **Differentiating between the rates, structural types, and spatial genomic patterns of homologous versus non-homologous recombination.**

Crucially, foreign DNA entering a bacterial cell is delivered by distinct biophysical vehicles (**transformation**, **transduction**, **conjugation**) and integrated through distinct enzymatic pathways (**RecA-mediated allelic replacement**, **site-specific recombinase integration**, **transposition**). By decoupling the biophysical delivery vehicle from the evolutionary selective sieve, we deconvolve the observed recombination landscape into:
* **The "How" (Mechanistic Architecture):** Where does foreign DNA physically enter and integrate along the chromosome (Chi site density, replication fork collisions, replication polar coordinates $oriC \to ter$, tRNA integration anchors)?
* **The "Why" (Selective Sieve):** Which recombinant cassettes survive and fix across clinical lineages (antimicrobial resistance islands, capsular polysaccharide switches, antiviral defense systems like CRISPR-Cas and R-M)?

---

## 2. The 5-Signature Bacterial Recombination Taxonomy

Each recombination event $e$ is characterized by a 5-tuple diagnostic vector:
$$\mathbf{\Sigma}(e) = \langle L_e, \quad \nabla g_e, \quad \mathcal{P}_{\perp}(e), \quad \mathcal{M}_{\text{flank}}(e), \quad \Delta \mathbf{S}_e \rangle$$

| Signature | Biophysical Setup | Carry-Through Integration | Alignment & Manifold Signature | Typical Tract Length | Spatial Chromosomal Distribution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Homologous Conversion** | Natural transformation (competence machinery / naked DNA) | RecA-dependent allelic replacement (swapped out) | Gapless synteny ($\nabla g = 0$), stays in core manifold $\operatorname{span}(\mathbf{V}_{\text{core}})$ | Exponential: $500\text{ bp} \to 8\text{ kb}$ | Distributed across core genes; enriched at Chi ($\chi$) octamer peaks |
| **2. Micro-Conversion** | Abortive transformation / mismatch repair | Ultra-short allelic patch | Poisson noise extinguishes scalar derivative; resolved via per-taxon attention drift $\|\Delta \mathbf{S}_i\|$ | Single codons to $<200\text{ bp}$ | Localized within individual protein domains |
| **3. Specialized Transduction** | Temperate bacteriophage tail injection | Integrase-mediated site-specific insertion (*attP* $\times$ *attB*) | Indel step $|\nabla g| \to 1.0$, orthogonal departure $\mathcal{P}_{\perp} \gg 0$, direct repeats | Modular: $30\text{--}45\text{ kb}$ | Strictly anchored at specific tRNA gene 3' ends |
| **4. Generalized Transduction** | Pseudo-virion headful packaging (*pac* / *cos*) | RecA-dependent homologous allelic swap | Gapless synteny ($\nabla g = 0$), core manifold flight | Strictly bounded by capsid volume: $\le 45\text{--}100\text{ kb}$ | Bounded by phage packaging capacity |
| **5. Conjugative Mega-Island (ICE)** | Type IV Secretion System (T4SS) pilus | Site-specific recombinase / autonomous excision & integration | Mega-indel step ($|\nabla g| \to 1.0$), orthogonal subspace departure $\mathcal{P}_{\perp} \gg 0$ | Massive blocks: $20\text{--}150\text{ kb}$ | Clustered near chromosomal terminus $ter$ / *dif* and ribosomal anchors |

---

## 3. The Mechanistic vs. Selective Deconvolution Formulation

The observed recombination rate field along the circular chromosome ($s \in [1, L]$) is modeled as the product of biophysical integration feasibility and post-recombination evolutionary fitness:

$$\lambda_{\text{observed}}(s) = \lambda_{\text{mechanistic}}(s) \times \mathcal{S}_{\text{selective}}(s)$$

* **$\lambda_{\text{mechanistic}}(s)$ (The Biophysical Prior):** Modeled from chromosomal features independent of lineage history:
  $$\ln \lambda_{\text{mech}}(s) = \beta_0 + \beta_1 \mathcal{D}_{\chi}(s) + \beta_2 \cos(\theta_{\text{rep}}(s)) + \beta_3 \mathbb{I}(s \in \text{tRNA/att}) + \beta_4 |\text{Skew}(s)|$$
* **$\mathcal{S}_{\text{selective}}(s)$ (The Selective Sieve):** 
  $$\mathcal{S}_{\text{selective}}(s) = \frac{\lambda_{\text{observed}}(s)}{\lambda_{\text{mechanistic}}(s)}$$
  * **$\mathcal{S}_{\text{selective}} \ll 1$ (Purifying Selection / Recombination Deserts):** Tightly coupled stoichiometric core complexes (ribosomal operons, ATP synthase, RNA polymerase).
  * **$\mathcal{S}_{\text{selective}} \gg 1$ (Adaptive Hotspots):** Antimicrobial resistance (AMR), surface antigen diversification (*cps*, *pspA*), and antiphage defense islands.
  * **Tier 2 $dN/dS$ Decoupling:** BlockLinear dual-track embedding computes $\rho_{\text{retic}}(s)$, confirming genuine chromosomal block transfer ($\rho_{\text{retic}} \ge 0.70$) versus convergent positive selection ($\rho_{\text{retic}} < 0.25$).

---

## 4. Directory Structure

```
bacterial_recombination_meta_analysis/
├── README.md                      # This project overview and blueprint
├── docs/
│   ├── THEORETICAL_FRAMEWORK.md   # Rigorous mathematical foundations
│   ├── ALIGNMENT.md               # Resolving the genome-wide alignment dilemma
│   └── ROADMAP_AND_MILESTONES.md  # Experimental phases and targets
├── bacterial_recomb/              # Core Python module
│   ├── __init__.py
│   ├── signatures.py              # 5-way signature classification engine
│   ├── chromosomal_map.py         # Polar coordinates, GC skew, Chi scanners
│   ├── deconvolution.py           # lambda_mech vs S_sel deconvolution
│   └── pipeline.py                # End-to-end chromosome-wide analyzer
├── data/                          # Reference genomes, landmarks, GFFs
│   └── README.md
├── scripts/                       # Execution and plotting scripts
│   ├── run_pmen1_recombination_signatures.py
│   └── plot_chromosomal_landscape.py
└── tests/                         # Unit tests and synthetic benchmarks
    ├── __init__.py
    └── test_signatures.py
```

---

## 5. Target Bacterial Cohorts

1. **Streptococcus pneumoniae:**
   * Canonical PMEN1 ($N=11$, 2.22 Mb, Spain$^{23\mathrm{F}}$-1 multidrug-resistant reference).
   * PubMLST Pneumococcal Genome Library (PGL, $N=30,976$ curated genomes).
2. **Escherichia coli / Shigella:**
   * Global pandemic multidrug-resistant ST131 lineage.
   * Pan-genome cohort testing chromosome-wide Chi site distribution and pathogenicity island (PAI) integration.
3. **Mycobacterium tuberculosis:**
   * Clonal control cohort (recombination-deficient reference baseline, testing 0% false-positive rate under low diversity).
4. **Staphylococcus aureus:**
   * Methicillin-resistant (MRSA) SCC*mec* island integration and pathogenicity islands (SaPIs).
5. **Pseudomonas aeruginosa:**
   * High-recombination opportunistic pathogen with large, plastic accessory mobilomes.
