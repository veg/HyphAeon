"""
Mode I vs Mode II phenotype (PhyloWAS) comparison tests.

Mode I (raw binary substitution counting + Poisson test) is phylogeny-blind
by construction. Mode II (neural transformer attributions) is supposed to be
phylogeny-aware via axial attention + MDS tree coordinates.

These tests directly compare the two modes on the same data to assess
whether Mode II's neural machinery actually improves over raw counting,
particularly under phylogenetic confounding.

Key question: If Mode II's FPR ≈ Mode I's FPR on confounded data, the
neural model's phylogeny insensitivity is of significant practical impact —
it adds complexity without separating signal from phylogenetic noise.

Mode I requires no weights and runs on any alignment. Mode II requires
weights and will skip if unavailable.
"""
import numpy as np
import pytest

from _harness import get_taxa, fg_string, phylowas_pvals
from _mode_i_baseline import mode_i_phylowas_pvals


def _collect_neutral_pvals(sim_alignments, runner, pval_extractor, use_tree=True, **runner_kw):
    """Run runner on first 2 sim alignments with 1/3 foreground taxa.

    Returns array of p-values for all variable sites across datasets.
    Factored out of test_mode_i_neutral_fpr, test_mode_ii_neutral_fpr,
    and test_mode_ii_neutral_fpr_not_worse_than_mode_i.

    use_tree=False for Mode I (no tree parameter).
    """
    if not sim_alignments:
        return np.array([])

    all_pvals = []
    for ds in sim_alignments[:2]:
        fa, nwk = ds["fa"], ds["nwk"]
        taxa = ds["taxa"]
        n_fg = max(2, len(taxa) // 3)
        fg = ",".join(taxa[:n_fg])

        if use_tree:
            result = runner(fa, tree_path=nwk, foreground=fg, **runner_kw)
        else:
            result = runner(fa, foreground=fg, **runner_kw)
        all_pvals.extend(pval_extractor(result).tolist())

    return np.array(all_pvals)


class TestSisterTaxaConfounding:
    """When sister taxa (adjacent in tree) are assigned as foreground, both
    modes should produce false positives from shared ancestral substitutions.
    Mode II should produce fewer than Mode I if it's truly phylogeny-aware.
    """

    @pytest.fixture(scope="class")
    def sister_taxa_pvals(self, mode_i_phylowas_runner,
                          smc6_paths, sister_taxa_7_result):
        """Run Mode I on Smc6 with 7 sister-taxa foreground; reuse
        session-scoped Mode II result from sister_taxa_7_result.
        Shared across all 3 tests in this class."""
        fa, nwk = smc6_paths
        taxa = get_taxa(fa, nwk)
        fg = fg_string(taxa, 7)
        result_i = mode_i_phylowas_runner(fa, foreground=fg, min_taxa_per_site=2)
        return mode_i_phylowas_pvals(result_i), phylowas_pvals(sister_taxa_7_result)

    def test_mode_i_sister_taxa_fpr(self, sister_taxa_pvals):
        """Mode I baseline: sister-taxa foreground should produce false positives.

        With 7 foreground taxa (out of 20), Mode I's Poisson test flags ~6%
        of sites as significant. These are false positives — the first 7 taxa
        in the Smc6 tree are closely related, so shared substitutions reflect
        common descent, not phenotype association. Mode I is phylogeny-blind
        and cannot distinguish these cases.
        """
        pvals_i, _ = sister_taxa_pvals
        if len(pvals_i) < 5:
            pytest.skip("Not enough variable sites")

        fpr = float(np.mean(pvals_i <= 0.05))
        # Mode I should produce some false positives with 7 sister foreground taxa.
        # With fewer fg taxa (3), the Poisson test is too conservative to trigger.
        assert fpr > 0.0, (
            f"Mode I FPR {fpr:.2%} with 7 sister-taxa foreground — "
            f"expected some false positives from phylogenetic confounding"
        )

    def test_mode_ii_sister_taxa_fpr(self, sister_taxa_pvals):
        """Mode II: sister-taxa foreground should not produce excessive FPR.

        Uses Smc6 (shallow tree, 20 taxa) where the model has ~5.6% FPR on
        neutral data. With 7 sister-taxa foreground, phylogenetic confounding
        may inflate FPR, but it should stay below 20% on this shallow tree.
        """
        _, pvals_ii = sister_taxa_pvals
        if len(pvals_ii) < 5:
            pytest.skip("Not enough variable sites")

        fpr = float(np.mean(pvals_ii <= 0.05))
        assert fpr < 0.20, (
            f"Mode II FPR {fpr:.2%} with sister-taxa foreground — "
            f"phylogenetic confounding inflates FPR beyond 20% on Smc6"
        )

    @pytest.mark.xfail(reason="Mode II FPR (9.3%) > Mode I FPR (6.2%) on sister-taxa "
                         "confounded data — the neural model's phylogeny insensitivity "
                         "(AGENTS.md) means it adds noise without phylogenetic benefit")
    def test_mode_ii_fpr_not_worse_than_mode_i(self, sister_taxa_pvals):
        """Mode II should not have a higher FPR than Mode I on sister-taxa
        confounded data. If it does, the neural model is adding noise without
        phylogenetic benefit — strong evidence that phylogeny insensitivity
        is of significant impact.

        Uses 7 foreground taxa (out of 20) — enough for Mode I's Poisson
        test to produce false positives from phylogenetic confounding.
        """
        pvals_i, pvals_ii = sister_taxa_pvals
        if len(pvals_i) < 5 or len(pvals_ii) < 5:
            pytest.skip("Not enough variable sites for comparison")

        fpr_i = float(np.mean(pvals_i <= 0.05))
        fpr_ii = float(np.mean(pvals_ii <= 0.05))

        assert fpr_ii <= fpr_i, (
            f"Mode II FPR ({fpr_ii:.2%}) > Mode I FPR ({fpr_i:.2%}) on "
            f"sister-taxa confounded data — the neural model is not "
            f"phylogeny-aware enough to improve over raw counting."
        )


class TestNeutralCalibrationComparison:
    """On neutral simulated data, both modes should have controlled FPR.
    Mode I's Poisson test should be roughly calibrated (it tests a simple
    null). Mode II's ACAT combination of LRT + association may not be.
    """

    @pytest.fixture(scope="class")
    def neutral_pvals(self, mode_i_phylowas_runner, phylowas_runner,
                      sim_alignments):
        """Compute Mode I and Mode II neutral p-values once, shared across
        all 3 tests in this class."""
        pvals_i = _collect_neutral_pvals(
            sim_alignments, mode_i_phylowas_runner, mode_i_phylowas_pvals,
            use_tree=False, min_taxa_per_site=2)
        pvals_ii = _collect_neutral_pvals(
            sim_alignments, phylowas_runner, phylowas_pvals)
        return pvals_i, pvals_ii

    def test_mode_i_neutral_fpr(self, neutral_pvals):
        """Mode I on neutral data: Poisson test should be roughly calibrated."""
        pvals_i, _ = neutral_pvals
        if len(pvals_i) < 10:
            pytest.skip("Not enough variable sites for FPR test")

        fpr = float(np.mean(pvals_i <= 0.05))
        # Mode I's Poisson test should be roughly calibrated under neutrality.
        # Allow up to 20% (the test is approximate due to limited sample sizes).
        assert fpr < 0.20, (
            f"Mode I FPR {fpr:.2%} on neutral data — "
            f"Poisson test is not calibrated under the null"
        )

    @pytest.mark.xfail(reason="Mode II FPR ~94% on neutral data — assertion fpr < 0.20 fails "
                         "because ACAT combination is massively anti-conservative")
    def test_mode_ii_neutral_fpr(self, neutral_pvals):
        """Mode II on neutral data: ACAT combination should be roughly calibrated.

        Mode II's FPR is known to be elevated on seq-gen neutral data (see
        TestTreeDepthSpread — tree-structure mismatch is the primary driver).
        We assert FPR < 20% — the same threshold as Mode I's test.
        """
        _, pvals_ii = neutral_pvals
        if len(pvals_ii) < 10:
            pytest.skip("Not enough variable sites for FPR test")

        fpr = float(np.mean(pvals_ii <= 0.05))
        assert fpr < 0.20, (
            f"Mode II FPR {fpr:.2%} on neutral data — "
            f"ACAT combination is not calibrated under the null"
        )

    @pytest.mark.xfail(reason="Mode II FPR ~94% vs Mode I ~0.5% on neutral data — "
                         "known model limitation: neural LRT is massively miscalibrated")
    def test_mode_ii_neutral_fpr_not_worse_than_mode_i(self, neutral_pvals):
        """Mode II should not have a dramatically higher FPR than Mode I on
        neutral data. If it does, the neural model's LRT is producing false
        positives that the ACAT combination amplifies.

        NOTE: The core model's known 36% FPR on sim_100_deep means Mode II
        may fail this. That failure is meaningful — it means the neural
        model is worse than raw counting at controlling false positives.

        RESOLVED (R6): Three diagnostic tests now distinguish the hypotheses.
        TestMEMENegativeFPR (PASSED) shows the LRT is calibrated on real data.
        TestCompositionSpread (PASSED) shows composition is not a driver.
        TestTreeDepthSpread (XFAIL, 12.9x spread) shows tree-structure mismatch
        is the primary driver — the model breaks down on deep trees.
        """
        pvals_i, pvals_ii = neutral_pvals
        if len(pvals_i) < 10 or len(pvals_ii) < 10:
            pytest.skip("Not enough variable sites for comparison")

        fpr_i = float(np.mean(pvals_i <= 0.05))
        fpr_ii = float(np.mean(pvals_ii <= 0.05))

        # Allow 30% absolute margin — Mode II's LRT is known to be noisy.
        # If it exceeds this, the neural model is substantially worse than
        # raw binary counting at null calibration.
        assert fpr_ii <= fpr_i + 0.30, (
            f"Mode II FPR ({fpr_ii:.2%}) is dramatically worse than "
            f"Mode I FPR ({fpr_i:.2%}) on neutral data — "
            f"the neural model's LRT is producing false positives that "
            f"raw binary counting does not"
        )


class TestSignalPreservation:
    """Both modes should rank truly associated sites above noise. On real
    data with known selection (Smc6), both modes should identify some
    significant sites. The ranking correlation between modes tells us
    whether they're finding the same signal or different noise.
    """

    @pytest.fixture(scope="class")
    def signal_results(self, mode_i_phylowas_runner, phylowas_runner,
                       smc6_paths):
        """Run both modes once on Smc6 with fg_string(taxa, 7), shared
        across all 3 tests in this class."""
        fa, nwk = smc6_paths
        taxa = get_taxa(fa, nwk)
        fg = fg_string(taxa, 7)
        result_i = mode_i_phylowas_runner(fa, foreground=fg, min_taxa_per_site=2)
        result_ii = phylowas_runner(fa, tree_path=nwk, foreground=fg,
                                     min_taxa_per_site=2)
        return result_i, result_ii

    def test_both_modes_produce_results(self, signal_results):
        """Both modes should produce site-level results on Smc6."""
        result_i, result_ii = signal_results
        assert len(result_i["sites"]) > 0, "Mode I produced no sites"
        assert len(result_ii["sites"]) > 0, "Mode II produced no sites"

    def test_site_overlap(self, signal_results):
        """Both modes should test overlapping sites. The intersection of
        tested sites tells us whether they're looking at the same data.
        """
        result_i, result_ii = signal_results
        sites_i = {s["site"] for s in result_i["sites"]}
        sites_ii = {s["site"] for s in result_ii["sites"]}

        overlap = sites_i & sites_ii
        # Both modes should test at least some of the same sites
        assert len(overlap) > 0, (
            f"No overlapping sites between Mode I ({len(sites_i)} sites) "
            f"and Mode II ({len(sites_ii)} sites)"
        )

    def test_rank_correlation_on_shared_sites(self, signal_results):
        """On shared sites, the p-value rankings should be positively correlated.
        If they disagree on what's significant, one of them is finding noise.

        With only one dataset, this is a weak test. But strong anti-correlation
        would be alarming — it would mean the modes are finding opposite signals.
        """
        result_i, result_ii = signal_results
        sites_i = {s["site"]: s["p_value"] for s in result_i["sites"]}
        sites_ii = {s["site"]: s["p_value"] for s in result_ii["sites"]}

        shared = sorted(set(sites_i) & set(sites_ii))
        if len(shared) < 5:
            pytest.skip("Not enough shared sites for correlation")

        from scipy.stats import spearmanr
        pvals_i = [sites_i[s] for s in shared]
        pvals_ii = [sites_ii[s] for s in shared]
        rho, _ = spearmanr(pvals_i, pvals_ii)

        # We expect non-negative correlation — both modes should broadly
        # agree on which sites are more vs less significant.
        assert rho >= -0.3, (
            f"Spearman rho={rho:.3f} between Mode I and Mode II p-values — "
            f"modes are anti-correlated, one is finding noise"
        )


class TestTruePositiveDetection:
    """Both modes should detect sites with known injected selection signal.

    Uses inject_selection() to create alignments with unambiguous positive
    selection on a clade of taxa. The foreground assignment matches the
    selected clade, so both modes should detect the injected sites.

    This complements the FPR tests above: FPR tells us about false
    positives, TPR tells us about true positives. A method that controls
    FPR but has zero TPR is useless; a method with high TPR but 94% FPR
    is also useless. Both dimensions matter.
    """

    @pytest.fixture(scope="class")
    def injected_results(self, mode_i_phylowas_runner, phylowas_runner,
                         injected_dataset):
        """Run both modes once on injected dataset, shared across all 3 tests."""
        ds = injected_dataset
        fg = ",".join(ds["selected_taxa"])
        selected_1idx = ds["selected_1idx"]

        result_i = mode_i_phylowas_runner(ds["fa"], foreground=fg,
                                           min_taxa_per_site=2)
        sites_i = {s["site"]: s["p_value"] for s in result_i["sites"]}

        result_ii = phylowas_runner(ds["fa"], tree_path=ds["nwk"],
                                     foreground=fg, min_taxa_per_site=2)
        sites_ii = {s["site"]: s["p_value"] for s in result_ii["sites"]}

        return selected_1idx, sites_i, sites_ii

    def test_mode_i_detects_injected_sites(self, injected_results):
        """Mode I should detect at least some injected selection sites.

        Mode I counts shared substitutions between foreground taxa. The
        injected selection creates radical AA changes on a clade, so
        Mode I should flag those sites as having excess shared mutations.
        """
        selected_1idx, sites_i, _ = injected_results

        detected = sum(1 for s in selected_1idx
                       if s in sites_i and sites_i[s] <= 0.05)
        total_tested = sum(1 for s in selected_1idx if s in sites_i)

        if total_tested == 0:
            pytest.skip("No injected sites in Mode I tested set")

        tpr = detected / total_tested
        # Mode I should detect at least 1 injected site. With 10 injected
        # radical changes on 15 taxa, the Poisson test should fire.
        assert tpr > 0.0, (
            f"Mode I TPR {tpr:.0%} ({detected}/{total_tested}) — "
            f"detected none of the injected selection sites"
        )

    @pytest.mark.xfail(reason="Mode II flags ~94% of all sites as significant — "
                         "precision (injected/significant) is very low because "
                         "high FPR drowns out true signal")
    def test_mode_ii_detects_injected_sites(self, injected_results):
        """Mode II should detect injected selection sites with meaningful precision.

        With ~94% FPR, Mode II flags nearly everything as significant. We test
        precision (injected/significant) instead of TPR — if precision is very
        low, the model can't distinguish true signal from noise.
        """
        selected_1idx, _, sites_ii = injected_results

        detected = sum(1 for s in selected_1idx
                       if s in sites_ii and sites_ii[s] <= 0.05)
        total_tested = sum(1 for s in selected_1idx if s in sites_ii)

        if total_tested == 0:
            pytest.skip("No injected sites in Mode II tested set")

        # Precision = detected_injected / total_significant. With ~94% FPR,
        # Mode II flags nearly everything as significant, so precision is
        # very low. This is the meaningful metric when FPR is high.
        total_significant = sum(1 for s in sites_ii.values() if s <= 0.05)
        precision = detected / total_significant if total_significant > 0 else 0.0
        assert precision > 0.15, (
            f"Mode II precision {precision:.0%} ({detected}/{total_significant}) — "
            f"injected sites are not enriched among significant calls"
        )

    @pytest.mark.xfail(reason="Mode II flags ~94% of all sites as significant — "
                         "precision is much lower than Mode I because high FPR drowns "
                         "out true signal")
    def test_mode_ii_tpr_not_worse_than_mode_i(self, injected_results):
        """Mode II should not have worse precision than Mode I on injected signal.

        With ~94% FPR, Mode II's TPR is trivially high. We compare precision
        (TP/(TP+FP)) instead — if Mode II's precision is lower than Mode I's,
        the neural model's high FPR drowns out true signal.
        """
        selected_1idx, sites_i, sites_ii = injected_results

        tested_i = sum(1 for s in selected_1idx if s in sites_i)
        tested_ii = sum(1 for s in selected_1idx if s in sites_ii)
        if tested_i == 0 or tested_ii == 0:
            pytest.skip("Not enough injected sites tested by both modes")

        det_i = sum(1 for s in selected_1idx
                    if s in sites_i and sites_i[s] <= 0.05)
        det_ii = sum(1 for s in selected_1idx
                     if s in sites_ii and sites_ii[s] <= 0.05)

        # Compare precision instead of TPR — with 94% FPR, Mode II's TPR
        # is trivially high. Precision (TP/(TP+FP)) is the meaningful metric.
        total_sig_i = sum(1 for s in sites_i.values() if s <= 0.05)
        total_sig_ii = sum(1 for s in sites_ii.values() if s <= 0.05)
        prec_i = det_i / total_sig_i if total_sig_i > 0 else 0.0
        prec_ii = det_ii / total_sig_ii if total_sig_ii > 0 else 0.0

        assert prec_ii >= prec_i, (
            f"Mode II precision ({prec_ii:.0%}, {det_ii}/{total_sig_ii}) < "
            f"Mode I precision ({prec_i:.0%}, {det_i}/{total_sig_i}) — "
            f"neural model's high FPR drowns out true signal"
        )
