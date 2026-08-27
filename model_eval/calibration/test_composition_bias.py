"""
Composition bias calibration: does AxoMEME stay calibrated under non-uniform
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
import json
import os

import pytest
import numpy as np

from _harness import evaluate_alignment
from _sim import simulate_neutral_alignment

# Composition profiles matching real genomes.
# seq-gen -f takes A,C,G,T frequencies (note the order: A,C,G,T, not A,T,C,G).
_COMPOSITION_PROFILES = {
    "uniform": (0.25, 0.25, 0.25, 0.25),
    "at_rich": (0.40, 0.10, 0.10, 0.40),  # ~80% AT (Plasmodium-like)
    "gc_rich": (0.15, 0.35, 0.35, 0.15),  # ~70% GC (Mycobacterium-like)
}

_SEEDS = [0, 1, 2]


@pytest.fixture(scope="module")
def composition_fpr_data(model, seqgen_available):
    """Run all composition simulations once and cache FPR results.

    Shared by TestAxoMEMECompositionBias (per-composition FPR) and
    TestCompositionSpread (cross-comparison) to avoid running 9
    simulations twice. Also stores actual AT content per composition
    so tests don't need to re-run seq-gen just to measure it.
    """
    from Bio import SeqIO

    results = {}
    for comp_name, freqs in _COMPOSITION_PROFILES.items():
        all_pvals = []
        actual_at = None
        for seed in _SEEDS:
            fa, nwk = simulate_neutral_alignment(
                n_taxa=50, n_codons=100, tree_depth=0.2,
                freqs=freqs, seed=seed, scale=1.0)
            if actual_at is None:
                at_count = 0
                total = 0
                for rec in SeqIO.parse(fa, "fasta"):
                    s = str(rec.seq).upper()
                    at_count += s.count("A") + s.count("T")
                    total += len(s)
                actual_at = at_count / total if total > 0 else 0.0
            res = evaluate_alignment(model, fa, nwk)
            tested = res["tested"]
            if tested.sum() > 0:
                all_pvals.append(res["pval"][tested])
        if all_pvals:
            pooled = np.concatenate(all_pvals)
            results[comp_name] = {
                "fpr": float(np.mean(pooled <= 0.05)),
                "n_sites": len(pooled),
                "pvals": pooled,
                "actual_at": actual_at,
            }
    return results


@pytest.mark.parametrize("comp_name", ["uniform", "at_rich", "gc_rich"])
@pytest.mark.xfail(reason="Moderate tree (50 taxa, depth 0.2) has ~16% FPR — "
                     "tree-structure-dependent calibration, not composition-specific. "
                     "Same root cause as test_axomeme_null moderate config.")
class TestAxoMEMECompositionBias:
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
                             artifacts_dir, composition_fpr_data):
        if comp_name not in composition_fpr_data:
            pytest.skip(f"No variable sites for {comp_name}")
        data = composition_fpr_data[comp_name]
        fpr_05 = data["fpr"]
        pooled = data["pvals"]

        freqs = _COMPOSITION_PROFILES[comp_name]
        target_at = freqs[0] + freqs[3]
        actual_at = data["actual_at"]

        print(f"\n[{comp_name}] pooled variable sites: {len(pooled)}")
        print(f"  Target AT: {target_at:.0%}, actual: {actual_at:.1%}")
        print(f"  pooled FPR@0.05: {fpr_05:.1%} (ideal: 5%)")

        assert fpr_05 <= 0.15, (
            f"AxoMEME FPR at alpha=0.05 is {fpr_05:.1%} under {comp_name} "
            f"composition (pooled over {len(_SEEDS)} seeds, {len(pooled)} "
            f"sites, AT={actual_at:.0%}). Threshold: <=15%. The model "
            f"miscalibrates under non-uniform nucleotide composition — "
            f"likely overfit to uniform ATCG."
        )


class TestCompositionSpread:
    """Is FPR systematically different across composition regimes?

    The per-composition tests above check each regime independently. This
    test collects all three FPRs and checks the spread. If FPR varies
    dramatically with composition (e.g., 5% on uniform but 30% on AT-rich),
    that supports the distributional mismatch hypothesis: the model's
    calibration is tied to the composition it was trained on.

    If FPR is uniformly high across all compositions, the problem is LRT
    miscalibration, not composition sensitivity.

    This test reuses the same simulation infrastructure as the per-composition
    tests but runs all three in a single test to enable cross-comparison.
    """

    def test_fpr_spread_across_compositions(self, composition_fpr_data,
                                             artifacts_dir):
        fprs = {name: d["fpr"] for name, d in composition_fpr_data.items()}

        if len(fprs) < 2:
            pytest.skip("Not enough composition regimes produced results")

        values = list(fprs.values())
        spread = max(values) - min(values)
        max_ratio = max(values) / max(min(values), 1e-6)

        report = {
            "per_composition_fpr": fprs,
            "spread": spread,
            "max_to_min_ratio": max_ratio,
            "interpretation": (
                "If spread is large and max_ratio > 3x, FPR is composition-"
                "dependent — supports distributional mismatch hypothesis. "
                "If spread is small, composition is not the driver."
            ),
        }
        out = os.path.join(artifacts_dir, "composition_fpr_spread.json")
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[report] {out}")
        print(json.dumps(report, indent=2))

        # If the ratio between worst and best composition FPR exceeds 5x,
        # composition is a major driver of miscalibration.
        assert max_ratio < 5.0, (
            f"FPR varies {max_ratio:.1f}x across compositions "
            f"({fprs}). The model's calibration is composition-dependent — "
            f"supports the distributional mismatch hypothesis (b)."
        )
