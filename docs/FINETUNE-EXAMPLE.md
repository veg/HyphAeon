# Fine-Tuning HyphAeon: A Worked Example (Viral OOD Repair)

## What and why

HyphAeon's site-level selection model (`hyphaeon predict`) is trained on deep-time
**mammalian** codon alignments (TOGA/VGP, ~200 My of divergence). On fast-evolving,
**shallow viral** alignments the base model transfers poorly — an out-of-distribution
(OOD) gap. Fine-tuning adapts the mammal-trained weights to real viral data **without
discarding** the deep-time signal the base model already captures.

This example uses our completed **Phase-2 datamonkey viral fine-tune**. It is a
worked reference, not a run-this recipe — the numbers below are the validated results
of that run.

## Setup

- **Base model:** the mammal-trained HyphAeon checkpoint (TOGA/VGP codon MSAs).
- **Fine-tune data:** ~**9,300** real datamonkey MEME jobs on viral alignments, each
  providing per-site MEME LRT labels (the same ground truth `predict` approximates).
- **Held-out unseen families:** **influenza (Orthomyxoviridae)** and
  **coronavirus (Coronaviridae)** are *excluded* from fine-tuning and used only for
  evaluation — a true cross-family OOD generalization test.
- **Mammal guardrail:** a held-out TOGA mammal validation set is scored every epoch to
  watch for catastrophic forgetting.

## The path at a high level

1. **Extract** per-site training tensors from the datamonkey MEME jobs (codon MSA +
   tree + MEME LRT labels) into the per-gene `.npz` archive format
   (`hyphaeon/training_data.py`; cf. `scripts/build_training_npz.py`).
2. **Fine-tune** from the base checkpoint with the standard trainer (`train.py`),
   starting from the mammal weights and a gentle learning rate, holding out the
   flu/corona families.
3. **Evaluate** each epoch on three splits — in-distribution viral, unseen viral
   families (flu+corona), and the TOGA-mammal guardrail — reporting Spearman rho
   against MEME LRT.

_(Fine-tuning reuses the same model, loss, and dataset code as base training; it
differs only in starting from a checkpoint, a lower LR, and the held-out families.)_

## Results

Fine-tuned on a **single NVIDIA A100**, **~40 min/epoch**, **3 epochs**
(converged by epoch 1–2). Metric = Spearman rho vs HyPhy MEME LRT.

| Split | Base (mammal) | Fine-tuned | Change |
|---|---|---|---|
| In-distribution viral | 0.534 | 0.606 | +0.072 |
| **Unseen families (flu + corona)** | **0.403** | **0.526** | **+0.123** |
| TOGA mammal (guardrail) | 0.595 | 0.574 | -0.021 |

**Takeaway.** Fine-tuning on real datamonkey viral data lifts the unseen-family
Spearman rho **0.403 → 0.526 (+0.123)** — generalization to viral families *never seen
during fine-tuning* — while mammal performance is essentially preserved
(**0.595 → 0.574**, a controlled ~0.02 drop). The viral OOD gap is largely closed at a
one-time cost of roughly **2 GPU-hours** (~40 min/epoch × 3 epochs on one A100), with
convergence by epoch 1–2.
