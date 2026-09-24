# Project Roadmap & Experimental Milestones: Bacterial Recombination Meta-Analysis

This document tracks the experimental phases, milestones, and deliverables for the bacterial recombination meta-analysis project.

---

## Phase 1: The 5-Signature Recombination Classifier (Weeks 1–3)
**Goal:** Build and validate the core module that classifies candidate recombination events into the 5 biological signatures based on the 5-tuple diagnostic vector $\mathbf{\Sigma}(e) = \langle L_e, \nabla g_e, \mathcal{P}_{\perp}(e), \mathcal{M}_{\text{flank}}(e), \Delta \mathbf{S}_e \rangle$.

* [x] **Milestone 1.1:** Mathematical formulation of the 5-tuple signature vector (`bacterial_recomb/signatures.py`).
* [x] **Milestone 1.2:** Unit testing suite verifying synthetic benchmarks (`tests/test_signatures.py`).
* [ ] **Milestone 1.3:** Prototype validation on *Streptococcus pneumoniae* PMEN1 (11 isolates, 2.22 Mb chromosome), proving separation of *pbp2x* / *cps* allelic swaps from ICESp23FST81 (81 kb) and prophage insertions.
* [ ] **Milestone 1.4:** Sensitivity and specificity benchmark against published Gubbins GFF records on PMEN1.

---

## Phase 2: Chromosomal Polar Mapping & Biophysical Features (Weeks 4–6)
**Goal:** Map classified recombination signatures onto the physical architecture of the circular bacterial chromosome.

* [x] **Milestone 2.1:** Implement replication polar coordinate engine ($\theta_{\text{rep}}$, $d_{\text{ori}}$, $d_{\text{ter}}$, GC skew polarity) in `bacterial_recomb/chromosomal_map.py`.
* [x] **Milestone 2.2:** Implement species-specific Chi ($\chi$) octamer scanner and Gaussian kernel density field $\mathcal{D}_{\chi}(s)$ (*S. pneumoniae* `5'-GAGAATGA-3'`, *E. coli* `5'-GCTGGTGG-3'`).
* [ ] **Milestone 2.3:** Map tRNA 3' integration anchors and *dif* terminus resolution sites from GFF annotations.
* [ ] **Milestone 2.4:** Circular chromosomal visualizer (`scripts/plot_chromosomal_landscape.py`) displaying signature distributions across the replication axis.

---

## Phase 3: Mechanistic vs. Selective Deconvolution Engine (Weeks 7–9)
**Goal:** Deconvolve the observed recombination rate field into the biophysical mechanistic delivery rate $\lambda_{\text{mech}}(s)$ and the evolutionary selective sieve $\mathcal{S}_{\text{sel}}(s)$.

* [x] **Milestone 3.1:** Implement Poisson GLM / GAM fitting for the biophysical prior $\lambda_{\text{mech}}(s)$ (`bacterial_recomb/deconvolution.py`).
* [x] **Milestone 3.2:** Compute the Selective Sieve $\mathcal{S}_{\text{sel}}(s) = \lambda_{\text{obs}}(s) / \lambda_{\text{mech}}(s)$, identifying recombination deserts ($\mathcal{S}_{\text{sel}} \ll 1$) and adaptive hotspots ($\mathcal{S}_{\text{sel}} \gg 1$).
* [ ] **Milestone 3.3:** Integrate Tier 2 BlockLinear dual-track $dN/dS$ Concordance Index ($\rho_{\text{retic}}$) to verify adaptive chromosomal blocks against point mutation homoplasy.
* [ ] **Milestone 3.4:** Functional enrichment testing of adaptive hotspots against AMR databases (CARD), capsule serotypes, and antiphage defense systems (DefenseFinder / CRISPRCasFinder).

---

## Phase 4: Cross-Species Pan-Genomic Meta-Analysis (Weeks 10–14)
**Goal:** Apply the validated pipeline to large-scale, real-world bacterial cohorts across contrasting evolutionary lifestyles.

* [ ] **Cohort 1 (*Streptococcus pneumoniae*):** PubMLST Pneumococcal Genome Library (PGL, 30,976 genomes). Testing natural transformation landscapes and capsule-switch hotspots across serotypes.
* [ ] **Cohort 2 (*Escherichia coli* / *Shigella*):** Global ST131 multidrug-resistant clade ($N > 2{,}000$). Testing conjugative plasmid integration, PAI islands, and Chi-site modulation along the 4.6 Mb chromosome.
* [ ] **Cohort 3 (*Staphylococcus aureus*):** Hospital-acquired MRSA ($N > 1{,}500$). Testing SCC*mec* island non-homologous insertion vs. core homologous micro-conversions.
* [ ] **Cohort 4 (*Mycobacterium tuberculosis*):** Strict clonal control cohort ($N > 1{,}000$). Verifying 0.00% false-positive rate on an essentially non-recombining pathogen.
* [ ] **Cohort 5 (*Pseudomonas aeruginosa*):** High-recombination CF lung isolates ($N > 800$). Testing mobilome plastic exchange vs. essential core constraints.

---

## Phase 5: Manuscript Preparation & Synthesis (Weeks 15–18)
**Goal:** Author and submit the flagship bacterial recombination meta-analysis manuscript with Darren Martin and collaborators.

* [ ] **Figure 1:** Conceptual overview of the 5-signature taxonomy and biophysical delivery vs. integration pathways.
* [ ] **Figure 2:** Chromosomal polar landscapes: Distance to $oriC$, GC skew, Chi density, and tRNA anchors dictating $\lambda_{\text{mech}}(s)$.
* [ ] **Figure 3:** The Selective Sieve $\mathcal{S}_{\text{sel}}(s)$: Recombination deserts in essential operons vs. massive adaptive peaks in AMR and capsule loci.
* [ ] **Figure 4:** Cross-species comparison: Contrasting transformation-dominated (*S. pneumoniae*), conjugation-dominated (*E. coli*), and clonal (*M. tuberculosis*) architectures.
