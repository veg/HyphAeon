"""
Power/sensitivity test: does HyphAeon detect simulated positive selection?

The calibration tests check the false positive rate (FPR) on neutral data.
This test checks the true positive rate (TPR) — does the model detect
selection when it's actually there?

We simulate neutral alignments with seq-gen, then inject positive selection
at a subset of sites by replacing codons on a subset of branches with
non-synonymous substitutions. This creates a controlled ground truth: we
know exactly which sites have selection and which don't.

This is a conservative test because the injected substitutions are "perfect"
selection signals (radical AA changes on multiple branches), which is easier
to detect than real episodic selection. If the model can't detect these, it
certainly can't detect real selection.

Requires seq-gen on PATH.
"""
import json

import numpy as np
import pytest

from _harness import evaluate_alignment
from _sim import simulate_neutral_alignment, inject_selection as _inject_selection


@pytest.mark.parametrize("n_taxa,depth,label", [
    (20, 0.1, "small_shallow"),
    (50, 0.2, "moderate"),
    (100, 0.5, "large_deep"),
])
@pytest.mark.parametrize("sim_seed", [42, 43])
class TestHyphAeonPower:
    """Does HyphAeon detect injected positive selection?

    We inject selection at 10% of sites (radical AA changes on 20% of taxa)
    and check whether HyphAeon's p-values at those sites are lower than at
    neutral sites.

    Parametrized over 2 simulation seeds to assess variance in power
    estimates. A single seed can't distinguish "model has no power" from
    "unlucky seed."

    Threshold: median p-value at selected sites < median at neutral sites,
    and TPR at alpha=0.05 >= 20%. The TPR threshold is low because injected
    selection is not identical to real selection — but a model that can't
    even detect this has no power.
    """

    def test_detects_injected_selection(self, model, seqgen_available,
                                        n_taxa, depth, label, sim_seed,
                                        artifacts_dir):
        n_codons = 100
        n_selected = 10  # 10% of sites
        n_branches = max(4, n_taxa // 5)  # 20% of taxa

        # Simulate neutral baseline
        fa, nwk = simulate_neutral_alignment(
            n_taxa=n_taxa, n_codons=n_codons, tree_depth=depth,
            seed=sim_seed, scale=1.0)

        # Inject selection
        fa_sel, selected_sites, n_selected_taxa_actual = _inject_selection(
            fa, nwk, n_taxa, n_codons, n_selected, n_branches, seed=99)

        # Run HyphAeon on the modified alignment
        res = evaluate_alignment(model, fa_sel, nwk)
        pvals, tested = res["pval"], res["tested"]

        # Compare p-values at selected vs neutral sites
        selected_mask = np.array([
            (i in selected_sites) and tested[i] for i in range(len(pvals))
        ], dtype=bool)
        neutral_mask = np.array([
            (i not in selected_sites) and tested[i] for i in range(len(pvals))
        ], dtype=bool)

        if selected_mask.sum() == 0:
            pytest.skip("No selected sites are variable")

        p_selected = pvals[selected_mask]
        p_neutral = pvals[neutral_mask]

        median_p_sel = float(np.median(p_selected))
        median_p_neut = float(np.median(p_neutral))
        tpr_05 = float(np.mean(p_selected <= 0.05))
        fpr_05 = float(np.mean(p_neutral <= 0.05))

        report = {
            "config": label,
            "sim_seed": sim_seed,
            "n_taxa": n_taxa,
            "n_selected_sites": int(selected_mask.sum()),
            "n_neutral_sites": int(neutral_mask.sum()),
            "n_selected_taxa_actual": n_selected_taxa_actual,
            "n_selected_taxa_requested": n_branches,
            "median_p_selected": median_p_sel,
            "median_p_neutral": median_p_neut,
            "tpr_005": tpr_05,
            "fpr_005": fpr_05,
            "power_ratio": tpr_05 / max(fpr_05, 0.001),
        }
        print(f"\n[{label} seed={sim_seed}] {json.dumps(report, indent=2)}")

        # The model should at least have lower p-values at selected sites
        assert median_p_sel < median_p_neut, (
            f"HyphAeon does not distinguish injected selection from neutral "
            f"sites ({label}, seed={sim_seed}). median p at selected="
            f"{median_p_sel:.3f} vs neutral={median_p_neut:.3f}. The model "
            f"has no power to detect even radical AA changes on multiple "
            f"branches."
        )

        # And should flag at least 20% of selected sites at alpha=0.05
        assert tpr_05 >= 0.20, (
            f"HyphAeon TPR at alpha=0.05 is {tpr_05:.1%} on injected selection "
            f"({label}, seed={sim_seed}). Threshold: >=20%. The model misses "
            f"the majority of sites with clear selection signal."
        )

        # TPR must exceed FPR — otherwise the model is just calling everything
        # significant, not detecting selection specifically.
        assert tpr_05 > fpr_05, (
            f"HyphAeon TPR ({tpr_05:.1%}) does not exceed FPR ({fpr_05:.1%}) "
            f"on {label} (seed={sim_seed}). The model flags selected and "
            f"neutral sites at similar rates — it is not distinguishing "
            f"selection from noise."
        )
