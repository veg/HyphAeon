"""
Composition bias calibration: does HyphAeon stay calibrated under non-uniform
nucleotide composition?

All other calibration tests use seq-gen with -f0.25,0.25,0.25,0.25 (uniform
ATCG). Real genomes are never like this:
  - Plasmodium falciparum: ~80% AT
  - Mycobacterium tuberculosis: ~65% GC
  - Pseudomonas aeruginosa: ~33% GC

If the model was trained on uniform-composition simulations or on a narrow
range of real genomes, it may miscalibrate on extremes. This test checks FPR
under three biologically realistic composition regimes.

Requires seq-gen on PATH.
"""
import pytest
import numpy as np

from _harness import evaluate_alignment, fpr_at
from _sim import simulate_neutral_alignment

# Composition profiles matching real genomes.
# seq-gen -f takes A,C,G,T frequencies (note the order: A,C,G,T, not A,T,C,G).
_COMPOSITION_PROFILES = {
    "uniform": (0.25, 0.25, 0.25, 0.25),
    "at_rich": (0.40, 0.10, 0.10, 0.40),  # ~80% AT (Plasmodium-like)
    "gc_rich": (0.15, 0.35, 0.35, 0.15),  # ~70% GC (Mycobacterium-like)
}

_SEEDS = [0, 1, 2]


@pytest.mark.parametrize("comp_name", ["uniform", "at_rich", "gc_rich"])
class TestHyphAeonCompositionBias:
    """FPR should be stable across nucleotide composition regimes.

    Pools p-values across multiple seeds per composition profile before
    asserting, so the FPR estimate is over ~150+ sites rather than ~50.
    Asserting per-seed at ~50 sites has a non-trivial false-alarm rate
    under a correctly calibrated model, which makes the suite flake.

    Uses a moderate tree (50 taxa, depth 0.2) where the model is known to
    be calibrated under uniform composition. If FPR inflates under AT-rich
    or GC-rich composition, the model is overfit to uniform composition.

    Threshold: pooled FPR at alpha=0.05 <= 15% (slightly more lenient than
    the uniform test's 10%, to allow for some composition-driven variance).
    """

    def test_composition_fpr(self, model, seqgen_available, comp_name,
                             artifacts_dir):
        freqs = _COMPOSITION_PROFILES[comp_name]
        all_pvals = []
        per_seed = []
        for seed in _SEEDS:
            fa, nwk = simulate_neutral_alignment(
                n_taxa=50, n_codons=100, tree_depth=0.2,
                freqs=freqs, seed=seed, scale=1.0)
            res = evaluate_alignment(model, fa, nwk)
            tested = res["tested"]
            p_tested = res["pval"][tested]
            fpr_05 = fpr_at(res["pval"], tested)
            per_seed.append((seed, int(tested.sum()), fpr_05))
            if tested.sum() > 0:
                all_pvals.append(p_tested)

        if not all_pvals:
            pytest.skip(f"No variable sites for {comp_name}")
        pooled = np.concatenate(all_pvals)
        fpr_05 = float(np.mean(pooled <= 0.05))

        # Measure actual AT content of one representative simulation
        from Bio import SeqIO
        fa, _ = simulate_neutral_alignment(
            n_taxa=50, n_codons=100, tree_depth=0.2,
            freqs=freqs, seed=0, scale=1.0)
        target_at = freqs[0] + freqs[3]
        at_count = 0
        total = 0
        for rec in SeqIO.parse(fa, "fasta"):
            s = str(rec.seq).upper()
            at_count += s.count("A") + s.count("T")
            total += len(s)
        actual_at = at_count / total if total > 0 else 0.0

        print(f"\n[{comp_name}] pooled variable sites: {len(pooled)}")
        for seed, n_var, s_fpr in per_seed:
            print(f"  seed={seed}: {n_var} sites, FPR@0.05={s_fpr:.1%}")
        print(f"  Target AT: {target_at:.0%}, actual: {actual_at:.1%}")
        print(f"  pooled FPR@0.05: {fpr_05:.1%} (ideal: 5%)")

        assert fpr_05 <= 0.15, (
            f"HyphAeon FPR at alpha=0.05 is {fpr_05:.1%} under {comp_name} "
            f"composition (pooled over {len(_SEEDS)} seeds, {len(pooled)} "
            f"sites, AT={actual_at:.0%}). Threshold: <=15%. The model "
            f"miscalibrates under non-uniform nucleotide composition — "
            f"likely overfit to uniform ATCG."
        )
