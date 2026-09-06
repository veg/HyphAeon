# Overlapping reading frame (dual-coding) detection

**Status: DRAFT / proposal for discussion — not production.** This adds a fast screen for
overlapping reading frames to HyphAeon and sketches the path to a model-native detector.

## Motivation

In an overlapping (dual-coding) region one nucleotide belongs to two codons at once, so a
substitution synonymous in frame A may be non-synonymous — and constrained — in frame B.
Single-frame selection models (MEME/FEL, and HyphAeon as trained) cannot represent this
coupling. Classic examples: CDKN2A p16INK4a / p14ARF (Szklarczyk et al. 2007, PNAS), the
ARFome (Chung et al. 2007, PLoS Comp Biol), and most viral genomes.

## Key idea

HyphAeon already ingests codon alignments. **An overlapping frame is just the same
alignment re-tokenized at a 1-nt offset** — no re-alignment. So overlap detection is a
lightweight extension of HyphAeon's existing per-column codon processing: read frames +1,
+2 (and optionally the antisense strand) and test whether the alternate frame carries a
genuine dual-coding signal.

## What this PR adds

- `hyphaeon/overlap.py` — self-contained detection:
  - discover stop-free alternate-frame ORFs (conservation-scored);
  - a model-free **double-constraint signature** per candidate: substitution-density
    suppression, position-3 (wobble) suppression, `syn_in_B` (fraction of substitutions
    synonymous in frame B), `synB_at_nonsynA` (geometry-robust frame-B constraint at
    frame-A non-synonymous sites), and an `inflation_flag` for the codon-usage confounder.
- `hyphaeon overlap` CLI subcommand (aliases `olg`, `dual-coding`).
- `tests/test_overlap.py`.

## Example

```
hyphaeon overlap --alignment CDKN2A.fasta --reference hg38
  frame   nt_region  codons  syn_in_B  synB@nsA  pos3  dens   call
    +1     251-448      66     0.70      0.81    0.34  0.45   OVERLAP    <- p14ARF
    +2     258-449      64     0.13      0.15    0.34  0.48   screen-only
```

Per-frame HyPhy FEL on the +1 call: ω_B ≈ 0.0 with purifying sites — i.e. the alternate
frame is a real, constrained coding frame (reproduces the CDKN2A asymmetry).

## Honest scope / calibration

- The signature is a **screen**, calibrated on CDKN2A/p14ARF (PASS) vs passenger ORFs.
  Confounders (extreme codon-usage constraint, e.g. KRAS; frame-A conservation
  bleed-through, e.g. hyperconserved small proteins) are the main false positives — the
  `inflation_flag` and `synB_at_nonsynA` axes mitigate but do not eliminate them.
- **Per-frame dN/dS (HyPhy FEL/MEME on each frame) is the adjudicator** for any candidate.
- Antisense detection is supported but more false-positive-prone; off by default.

## Roadmap (the reason for this draft)

The natural next step is a **dual-frame HyphAeon head**: predict per-frame selection
(LRT_A, LRT_B) plus a frame-B stop-avoidance term in one forward pass, trained on
overlap-labelled alignments (per-frame MEME/dN/dS). That turns the fast screen into a
fast *reliable* detector — the same speed, learned discrimination of real dual-coding vs
the confounders above. This PR is the detection scaffolding and CLI surface to build that
on; feedback on the interface and where it should live is welcome.
