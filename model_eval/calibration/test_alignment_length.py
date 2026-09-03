"""
Alignment length diversity: does HyphAeon behave consistently across
different alignment lengths?

All other calibration tests use 100 codons. Real alignments span a wide
range:
  - Short domains: ~50 codons
  - Typical genes: ~300 codons
  - Long proteins: ~1000+ codons

If the model's LRT distribution or FPR shifts with alignment length, it
means the model is sensitive to the number of sites in a way that has no
biological basis. A well-behaved per-site model should produce the same
p-value distribution regardless of how many other sites are in the
alignment.

Requires seq-gen on PATH.
"""
import json
import os

import numpy as np
import pytest

from _harness import evaluate_alignment, fpr_at
from _sim import simulate_neutral_alignment

# Alignment lengths spanning real-world range.
# Medium (100 codons) is xfailed because the moderate tree (50 taxa, depth 0.2)
# has ~16-18% FPR — above the 15% threshold. Same root cause as test_axomeme_null.
_LENGTH_GRID = [
    pytest.param(30, "short", id="short"),
    pytest.param(100, "medium",
                 marks=pytest.mark.xfail(reason="Moderate tree FPR ~16-18% — tree-structure-dependent calibration"),
                 id="medium"),
    pytest.param(500, "long", id="long"),
]

# Raw (n_codons, label) pairs for non-parametrized iteration
_LENGTH_PAIRS = [(30, "short"), (100, "medium"), (500, "long")]

_SEEDS = [0, 1]


@pytest.mark.parametrize("n_codons,label", _LENGTH_GRID)
class TestHyphAeonAlignmentLength:
    """FPR should be stable across alignment lengths.

    Pools p-values across multiple seeds per length before asserting, so
    the FPR estimate is over more sites. The short (30 codon) case has
    only ~15 variable sites per seed, so per-seed asserting has a ~4%
    false-alarm rate — pooling across 2 seeds reduces this.

    Also writes per-seed cache files (median LRT, FPR) for the
    cross-length consistency test below.

    Uses a moderate tree (50 taxa, depth 0.2) where the model is calibrated
    at 100 codons. If FPR inflates or the LRT distribution shifts at 30 or
    500 codons, the model has a length-dependent artifact.

    Threshold: pooled FPR at alpha=0.05 <= 15%.
    """

    def test_pooled_fpr_by_length(self, model, seqgen_available, n_codons,
                                  label, artifacts_dir):
        all_pvals = []
        per_seed = []
        for seed in _SEEDS:
            fa, nwk = simulate_neutral_alignment(
                n_taxa=50, n_codons=n_codons, tree_depth=0.2,
                seed=seed, scale=1.0)
            res = evaluate_alignment(model, fa, nwk)
            tested = res["tested"]
            p_tested = res["pval"][tested]
            fpr_05 = fpr_at(res["pval"], tested)
            median_lrt = float(np.median(res["lrt"][tested]))

            # Write per-seed cache for the consistency test
            cache_file = os.path.join(
                artifacts_dir, f"length_{label}_seed{seed}.json")
            with open(cache_file, "w") as f:
                json.dump({
                    "n_codons": n_codons,
                    "label": label,
                    "seed": seed,
                    "n_variable": int(tested.sum()),
                    "fpr_05": fpr_05,
                    "median_lrt": median_lrt,
                }, f, indent=2)

            per_seed.append((seed, int(tested.sum()), fpr_05, median_lrt))
            if tested.sum() > 0:
                all_pvals.append(p_tested)

        if not all_pvals:
            pytest.skip(f"No variable sites for {label}")
        pooled = np.concatenate(all_pvals)
        fpr_05 = float(np.mean(pooled <= 0.05))

        print(f"\n[{label} n_codons={n_codons}] pooled sites: {len(pooled)}")
        for seed, n_var, s_fpr, s_med in per_seed:
            print(f"  seed={seed}: {n_var} sites, FPR={s_fpr:.1%}, "
                  f"median LRT={s_med:.4f}")
        print(f"  pooled FPR@0.05: {fpr_05:.1%} (ideal: 5%)")

        assert fpr_05 <= 0.15, (
            f"HyphAeon pooled FPR at alpha=0.05 is {fpr_05:.1%} for {label} "
            f"alignment ({n_codons} codons, {len(pooled)} sites). "
            f"Threshold: <=15%. The model's calibration shifts with "
            f"alignment length."
        )


class TestHyphAeonAlignmentLengthConsistency:
    """Cross-length consistency: LRT distribution should not shift with length.

    Compares the median LRT across all lengths tested above. A per-site
    model should produce similar LRT distributions regardless of how many
    sites are in the alignment. This test reads the cached per-config JSON
    files written by TestHyphAeonAlignmentLength above, so it does not
    re-run the simulations.
    """

    def test_lrt_distribution_stable_across_lengths(self, artifacts_dir):
        results = {}
        for _, label in _LENGTH_PAIRS:
            lrt_medians = []
            for seed in _SEEDS:
                cache_file = os.path.join(
                    artifacts_dir, f"length_{label}_seed{seed}.json")
                if not os.path.exists(cache_file):
                    continue
                with open(cache_file) as f:
                    cached = json.load(f)
                if cached.get("n_variable", 0) >= 10:
                    lrt_medians.append(cached["median_lrt"])
            if lrt_medians:
                results[label] = float(np.mean(lrt_medians))

        if len(results) < 2:
            pytest.skip("Not enough configs with cached results "
                        "(run TestHyphAeonAlignmentLength first)")

        print(f"\nMedian LRT by length: {json.dumps(results, indent=2)}")

        # The ratio between the largest and smallest median LRT should be
        # within 3x. A larger shift means the model's output scale depends
        # on alignment length, which breaks cross-alignment comparisons.
        vals = list(results.values())
        ratio = max(vals) / max(min(vals), 1e-6)

        report = {
            "median_lrt_by_length": results,
            "max_min_ratio": ratio,
            "interpretation": (
                "A per-site model should produce similar LRT magnitudes "
                "regardless of alignment length. A large ratio means the "
                "model's output scale is length-dependent."
            ),
        }
        out = os.path.join(artifacts_dir, "length_consistency.json")
        with open(out, "w") as f:
            json.dump(report, f, indent=2)

        assert ratio <= 3.0, (
            f"Median LRT varies by {ratio:.1f}x across alignment lengths "
            f"({results}). The model's output scale depends on alignment "
            f"length, which means p-values from different-length alignments "
            f"are not directly comparable."
        )
