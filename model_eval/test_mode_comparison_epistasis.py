"""
Mode I vs Mode II epistasis (ESSM) comparison tests.

Mode I epistasis: cosine similarity on binary substitution profiles +
Poisson test. Mode II epistasis: cosine similarity on neural attribution
vectors + t-test (BH FDR filter).

These tests compare the two modes on co-selection network behavior:
edge overlap, neutral FPR calibration, and true positive detection
on injected selection signal.
"""
import numpy as np
import pytest


class TestEpistasisComparison:
    """Compare Mode I and Mode II epistasis (co-selection network) behavior.

    Mode I epistasis: cosine similarity on binary substitution profiles +
    Poisson test (max_p_pair filter). Mode II epistasis: cosine similarity
    on neural attribution vectors + t-test (BH FDR filter).

    WHY ALL EDGES ARE SIGNIFICANT: Both modes filter at the p-value level
    before reporting edges. Mode I's max_p_pair=0.05 keeps only p ≤ 0.05.
    Mode II's max_fdr=0.05 applies BH FDR, which by construction keeps only
    p < 0.05. So "all edges significant" is expected — the filter IS the
    significance test. The interesting question is what fraction of ALL
    candidate pairs are significant before filtering, which is what the
    FPR tests below measure.
    """

    @pytest.fixture(scope="class")
    def epistasis_results(self, mode_i_essm_runner, essm_smc6_result,
                          smc6_paths):
        """Run Mode I on Smc6; reuse session-scoped Mode II result."""
        fa, _ = smc6_paths
        result_i = mode_i_essm_runner(fa)
        return result_i, essm_smc6_result

    def test_both_modes_produce_edges(self, epistasis_results):
        """Both modes should produce co-selection edges on Smc6."""
        result_i, result_ii = epistasis_results
        assert "edges" in result_i, "Mode I epistasis produced no edges key"
        assert "edges" in result_ii, "Mode II epistasis produced no edges key"
        assert len(result_i["edges"]) > 0, "Mode I produced no epistasis edges"
        assert len(result_ii["edges"]) > 0, "Mode II produced no epistasis edges"

    def test_edge_site_pairs_overlap(self, epistasis_results):
        """Both modes should identify some overlapping site pairs.

        Complete non-overlap would mean the modes are looking at entirely
        different signals — suspicious even for very different methods.
        """
        result_i, result_ii = epistasis_results
        pairs_i = {(e["site_u"], e["site_v"]) for e in result_i["edges"]}
        pairs_ii = {(e["site_u"], e["site_v"]) for e in result_ii["edges"]}

        overlap = pairs_i & pairs_ii
        if len(pairs_i) == 0 or len(pairs_ii) == 0:
            pytest.skip("One mode produced no edges")

        assert len(overlap) > 0, (
            f"No overlapping site pairs between Mode I ({len(pairs_i)} edges) "
            f"and Mode II ({len(pairs_ii)} edges)"
        )

    def test_mode_ii_sim_range_wider(self, epistasis_results):
        """Mode II's continuous attribution vectors should produce a wider
        similarity range than Mode I's binary substitution profiles.

        Binary cosine similarity is inherently polarized (sites either share
        substitutions or don't), while neural attribution vectors capture
        graded relationships. This is a structural property, not a quality
        judgment.
        """
        result_i, result_ii = epistasis_results
        if not result_i["edges"] or not result_ii["edges"]:
            pytest.skip("One mode produced no edges")

        sim_range_i = (max(e["similarity"] for e in result_i["edges"]) -
                       min(e["similarity"] for e in result_i["edges"]))
        sim_range_ii = (max(e["similarity"] for e in result_ii["edges"]) -
                        min(e["similarity"] for e in result_ii["edges"]))

        assert sim_range_ii > sim_range_i, (
            f"Mode II sim range ({sim_range_ii:.3f}) should be wider than "
            f"Mode I ({sim_range_i:.3f}) — continuous attributions capture "
            f"graded relationships that binary profiles cannot"
        )


class TestEpistasisNeutralCalibration:
    """Compare Mode I and Mode II epistasis FPR on neutral simulated data.

    Runs both modes on neutral alignments with no filters (max_p_pair=1.0,
    max_fdr=1.0) to see the full p-value distribution before any filtering.
    The fraction of candidate pairs with p < 0.05 should be ~5% if the test
    is calibrated.
    """

    @pytest.fixture(scope="class")
    def neutral_pvals(self, mode_i_essm_runner, essm_runner, sim_alignments):
        """Compute Mode I and Mode II edge p-values once, shared across
        all 3 tests in this class."""
        if not sim_alignments:
            return None, None

        pvals_i = []
        pvals_ii = []
        for ds in sim_alignments[:2]:
            result_i = mode_i_essm_runner(ds["fa"], max_p_pair=1.0, min_sim=0.0)
            pvals_i.extend(e["p_value"] for e in result_i["edges"])

            result_ii = essm_runner(ds["fa"], tree_path=ds["nwk"],
                                    max_fdr=1.0, min_sim=0.0, min_lrt=0.0,
                                    min_shared=1, min_cesi=0.0)
            pvals_ii.extend(e["p_val"] for e in result_ii["edges"])

        return np.array(pvals_i), np.array(pvals_ii)

    def test_mode_i_edge_fpr_neutral(self, neutral_pvals):
        """Mode I's Poisson test should produce ~5% false positives on
        neutral data (before any FDR or p-value filtering)."""
        pvals_i, _ = neutral_pvals
        if pvals_i is None or len(pvals_i) < 20:
            pytest.skip("Not enough candidate pairs for FPR estimation")

        fpr = float(np.mean(pvals_i <= 0.05))
        # Mode I's Poisson test should be roughly calibrated. Allow up to 30%
        # (the test is approximate with small sample sizes).
        assert fpr < 0.30, (
            f"Mode I epistasis FPR {fpr:.1%} on neutral data — "
            f"Poisson test is not calibrated under the null"
        )

    @pytest.mark.xfail(reason="Mode II t-test produces ~84% false positives on "
                         "neutral data — same anti-conservatism as phenotype LRT")
    def test_mode_ii_edge_fpr_neutral(self, neutral_pvals):
        """Mode II's t-test should produce ~5% false positives on neutral
        data (before FDR filtering).

        XFAIL: Mode II's t-test on cosine similarity of neural attribution
        vectors is massively anti-conservative. On Smc6, 84% of candidate
        pairs have p < 0.05 before filtering. This is the same pattern as
        the phenotype LRT — the neural model's attribution vectors are too
        similar across sites, producing inflated significance.
        """
        _, pvals_ii = neutral_pvals
        if pvals_ii is None or len(pvals_ii) < 20:
            pytest.skip("Not enough candidate pairs for FPR estimation")

        fpr = float(np.mean(pvals_ii <= 0.05))
        assert fpr < 0.30, (
            f"Mode II epistasis FPR {fpr:.1%} on neutral data — "
            f"t-test is not calibrated under the null"
        )

    def test_mode_i_fpr_better_than_mode_ii(self, neutral_pvals):
        """Mode I's Poisson test should have lower FPR than Mode II's t-test
        on neutral data. This is the epistasis analogue of the phenotype
        FPR comparison."""
        pvals_i, pvals_ii = neutral_pvals
        if pvals_i is None or len(pvals_i) < 20 or len(pvals_ii) < 20:
            pytest.skip("Not enough candidate pairs for FPR comparison")

        fpr_i = float(np.mean(pvals_i <= 0.05))
        fpr_ii = float(np.mean(pvals_ii <= 0.05))

        assert fpr_i < fpr_ii, (
            f"Mode I FPR ({fpr_i:.1%}) >= Mode II FPR ({fpr_ii:.1%}) — "
            f"Poisson test is not better calibrated than t-test on neutral data"
        )


class TestEpistasisTruePositiveDetection:
    """Both modes should detect co-selection edges among injected selection
    sites on data with known selection signal.

    Uses inject_selection() to create alignments with selection on a clade.
    Sites with injected selection should co-occur in epistasis edges more
    than background sites, because the selected taxa share radical AA
    changes at those sites.
    """

    def _edges_among_injected(self, result, injected_sites):
        """Count edges where both endpoints are injected selection sites."""
        return sum(1 for e in result["edges"]
                   if e["site_u"] in injected_sites
                   and e["site_v"] in injected_sites)

    @pytest.fixture(scope="class")
    def injected_edge_results(self, mode_i_essm_runner, essm_runner,
                              injected_dataset):
        """Run both modes once on injected dataset, shared across all 3 tests."""
        ds = injected_dataset
        result_i = mode_i_essm_runner(ds["fa"], min_sim=0.0, max_p_pair=0.05)
        result_ii = essm_runner(ds["fa"], tree_path=ds["nwk"],
                                min_sim=0.0, min_lrt=0.0, min_shared=1,
                                max_fdr=0.05, min_cesi=0.0)
        return ds, result_i, result_ii

    def test_mode_i_detects_injected_edges(self, injected_edge_results):
        """Mode I should find at least some edges among injected selection
        sites. The radical AA changes create shared substitution patterns
        that binary cosine similarity should detect."""
        ds, result_i, _ = injected_edge_results

        injected_edges = self._edges_among_injected(result_i, ds["selected_1idx"])
        total_edges = len(result_i["edges"])

        if total_edges == 0:
            pytest.skip("Mode I produced no edges")

        # With 10 injected sites, there are 45 possible pairs.
        # Mode I should find at least 1 edge among them.
        assert injected_edges > 0, (
            f"Mode I found 0 edges among {len(ds['selected_1idx'])} injected "
            f"sites (out of {total_edges} total edges) — binary cosine "
            f"similarity missed the co-selection signal"
        )

    @pytest.mark.xfail(reason="Mode II t-test has ~87% FPR on neutral data — "
                         "precision (injected_edges/total_edges) is very low because "
                         "high FPR drowns out true co-selection signal")
    def test_mode_ii_detects_injected_edges(self, injected_edge_results):
        """Mode II should find at least some edges among injected selection
        sites."""
        ds, _, result_ii = injected_edge_results

        injected_edges = self._edges_among_injected(result_ii, ds["selected_1idx"])
        total_edges = len(result_ii["edges"])

        if total_edges == 0:
            pytest.skip("Mode II produced no edges")

        # Precision = injected_edges / total_edges. With ~87% FPR, Mode II
        # produces many edges regardless of signal, so precision is low.
        precision = injected_edges / total_edges if total_edges > 0 else 0.0
        assert precision > 0.10, (
            f"Mode II precision {precision:.0%} ({injected_edges}/{total_edges}) — "
            f"injected edges are not enriched among all significant edges"
        )

    @pytest.mark.xfail(reason="Mode II t-test has ~87% FPR on neutral data — "
                         "precision is much lower than Mode I because high FPR drowns "
                         "out true co-selection signal")
    def test_mode_ii_tpr_not_worse_than_mode_i(self, injected_edge_results):
        """Mode II should not have worse precision than Mode I on injected signal.

        With ~87% FPR, Mode II's TPR is trivially high. We compare precision
        (injected_edges/total_edges) instead — if Mode II's precision is lower
        than Mode I's, the neural model's high FPR drowns out true co-selection
        signal.
        """
        ds, result_i, result_ii = injected_edge_results
        n_injected = len(ds["selected_1idx"])
        max_possible_pairs = n_injected * (n_injected - 1) // 2

        edges_i = self._edges_among_injected(result_i, ds["selected_1idx"])
        edges_ii = self._edges_among_injected(result_ii, ds["selected_1idx"])

        if max_possible_pairs == 0:
            pytest.skip("No injected sites to form pairs")

        # Compare precision instead of TPR — with 87% FPR, Mode II's TPR
        # is trivially high. Precision is the meaningful metric.
        total_i = len(result_i["edges"])
        total_ii = len(result_ii["edges"])
        prec_i = edges_i / total_i if total_i > 0 else 0.0
        prec_ii = edges_ii / total_ii if total_ii > 0 else 0.0

        assert prec_ii >= prec_i, (
            f"Mode II precision ({prec_ii:.0%}, {edges_ii}/{total_ii}) < "
            f"Mode I precision ({prec_i:.0%}, {edges_i}/{total_i}) — "
            f"neural model's high FPR drowns out true co-selection signal"
        )
