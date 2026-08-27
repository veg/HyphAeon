"""
Output contract tests for hyphaeon BUSTED omnibus inference.

Tests the alignment-wide BUSTED selection test: ACAT/Simes p-value
combination, neural BUSTED head outputs, rate distribution structure,
and gene-level verdict logic.

Uses the Smc6 example dataset (20 taxa, shallow tree).
"""
import numpy as np
import pytest


class TestBustedOutputFields:
    """BUSTED output should have all required fields with valid values."""

    def test_output_has_required_fields(self, busted_smc6_result):
        assert isinstance(busted_smc6_result, list)
        assert len(busted_smc6_result) == 1
        record = busted_smc6_result[0]
        assert "alignment" in record
        assert "gene" in record
        assert "taxa" in record
        assert "sites" in record
        assert "p_value_acat" in record
        assert "p_value_simes" in record
        assert "omnibus_lrt" in record
        assert "predicted_gene_lrt" in record
        assert "selection_probability" in record
        assert "synonymous_rate_variation" in record
        assert "total_selection_energy" in record
        assert "sig_sites_p05" in record
        assert "sig_sites_p10" in record
        assert "rate_distributions" in record
        assert "positive_selection_detected" in record
        assert "elapsed_seconds" in record

    def test_taxa_and_sites_match_input(self, busted_smc6_result):
        record = busted_smc6_result[0]
        assert record["taxa"] == 20
        assert record["sites"] > 0


class TestBustedPValues:
    """ACAT and Simes p-values should be valid and sensible."""

    def test_p_values_in_unit_interval(self, busted_smc6_result):
        record = busted_smc6_result[0]
        assert 0.0 <= record["p_value_acat"] <= 1.0
        assert 0.0 <= record["p_value_simes"] <= 1.0

    def test_p_values_are_finite(self, busted_smc6_result):
        record = busted_smc6_result[0]
        assert np.isfinite(record["p_value_acat"])
        assert np.isfinite(record["p_value_simes"])

    def test_simes_ge_acat(self, busted_smc6_result):
        """Simes is more conservative than ACAT (Cauchy combination).
        ACAT downweights large p-values via tan transform; Simes takes
        min(L/rank * p_(i)), which is dominated by the smallest p-values
        without the Cauchy tail amplification. So p_simes >= p_acat.

        NOTE: This is an empirical regularity on typical p-value configurations,
        not a mathematical guarantee for all possible inputs. If it fails on
        a new dataset, it may indicate an unusual p-value distribution rather
        than a bug.
        """
        record = busted_smc6_result[0]
        assert record["p_value_simes"] >= record["p_value_acat"] - 1e-10


class TestBustedNeuralHead:
    """Neural BUSTED head outputs should be valid."""

    def test_selection_probability_in_unit_interval(self, busted_smc6_result):
        record = busted_smc6_result[0]
        assert 0.0 <= record["selection_probability"] <= 1.0

    def test_predicted_gene_lrt_non_negative(self, busted_smc6_result):
        record = busted_smc6_result[0]
        assert record["predicted_gene_lrt"] >= 0.0

    def test_synonymous_rate_variation_non_negative(self, busted_smc6_result):
        record = busted_smc6_result[0]
        assert record["synonymous_rate_variation"] >= 0.0


class TestBustedRateDistributions:
    """Rate distribution (omega) structure should have 3 components."""

    def test_rate_distributions_has_three_components(self, busted_smc6_result):
        rd = busted_smc6_result[0]["rate_distributions"]
        for i in [1, 2, 3]:
            assert f"omega_{i}" in rd, f"Missing omega_{i}"
            assert f"proportion_{i}" in rd, f"Missing proportion_{i}"

    def test_proportions_sum_to_one(self, busted_smc6_result):
        rd = busted_smc6_result[0]["rate_distributions"]
        total = rd["proportion_1"] + rd["proportion_2"] + rd["proportion_3"]
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_proportions_non_negative(self, busted_smc6_result):
        rd = busted_smc6_result[0]["rate_distributions"]
        for i in [1, 2, 3]:
            assert rd[f"proportion_{i}"] >= 0.0

    def test_omega_values_non_negative(self, busted_smc6_result):
        rd = busted_smc6_result[0]["rate_distributions"]
        for i in [1, 2, 3]:
            assert rd[f"omega_{i}"] >= 0.0

    def test_omega_1_and_2_are_fixed(self, busted_smc6_result):
        """omega_1=0.10 (purifying) and omega_2=1.00 (neutral) are hardcoded
        in cli.py:352. Only omega_3 (positive selection) is data-dependent."""
        rd = busted_smc6_result[0]["rate_distributions"]
        assert rd["omega_1"] == pytest.approx(0.10, abs=1e-10)
        assert rd["omega_2"] == pytest.approx(1.00, abs=1e-10)


class TestBustedVerdict:
    """Gene-level positive selection verdict logic."""

    def test_verdict_is_boolean(self, busted_smc6_result):
        assert isinstance(busted_smc6_result[0]["positive_selection_detected"], bool)

    def test_verdict_logic(self, busted_smc6_result):
        """positive_selection_detected should be True if p_acat < 0.05 or prob > 0.50.

        Mirrors cli.py:379 — update both together if the verdict logic changes.
        """
        record = busted_smc6_result[0]
        expected = bool(record["p_value_acat"] < 0.05 or record["selection_probability"] > 0.50)
        assert record["positive_selection_detected"] == expected


class TestBustedSigSites:
    """Significant site count fields."""

    def test_sig_sites_non_negative(self, busted_smc6_result):
        record = busted_smc6_result[0]
        assert record["sig_sites_p05"] >= 0
        assert record["sig_sites_p10"] >= 0

    def test_sig_sites_p10_ge_p05(self, busted_smc6_result):
        """More sites significant at 0.10 than at 0.05."""
        record = busted_smc6_result[0]
        assert record["sig_sites_p10"] >= record["sig_sites_p05"]

    def test_sig_sites_le_total_sites(self, busted_smc6_result):
        record = busted_smc6_result[0]
        assert record["sig_sites_p05"] <= record["sites"]
        assert record["sig_sites_p10"] <= record["sites"]


class TestBustedDerivedFields:
    """Derived fields: omnibus_lrt, total_selection_energy, elapsed_seconds."""

    def test_omnibus_lrt_non_negative(self, busted_smc6_result):
        """omnibus_lrt = sum(max(0, lrt_i - 3.841)) — thresholded sum of
        per-site LRTs above the chi^2(1) 0.05 critical value (3.841)."""
        assert busted_smc6_result[0]["omnibus_lrt"] >= 0.0

    def test_omnibus_lrt_le_total_selection_energy(self, busted_smc6_result):
        """omnibus_lrt = sum(max(0, lrt_i - 3.841)) <= sum(lrt_i) = total_selection_energy
        for non-negative LRTs, since max(0, x - 3.841) <= x when x >= 0."""
        record = busted_smc6_result[0]
        assert record["omnibus_lrt"] <= record["total_selection_energy"] + 1e-10

    def test_total_selection_energy_non_negative(self, busted_smc6_result):
        """total_selection_energy = sum(lrt_i) — sum of all per-site LRTs."""
        assert busted_smc6_result[0]["total_selection_energy"] >= 0.0

    def test_elapsed_seconds_positive(self, busted_smc6_result):
        assert busted_smc6_result[0]["elapsed_seconds"] > 0.0


class TestBustedMultipleDatasets:
    """BUSTED should work across different example datasets and produce
    valid output structure, not just run without crashing.
    """

    def _check_output_contract(self, result, expected_taxa):
        assert len(result) == 1
        record = result[0]
        assert record["taxa"] == expected_taxa
        assert record["sites"] > 0
        # P-values in valid range
        assert 0.0 <= record["p_value_acat"] <= 1.0
        assert 0.0 <= record["p_value_simes"] <= 1.0
        assert record["p_value_simes"] >= record["p_value_acat"] - 1e-10
        # Rate distributions have required structure
        rd = record["rate_distributions"]
        for i in [1, 2, 3]:
            assert f"omega_{i}" in rd, f"Missing omega_{i}"
            assert f"proportion_{i}" in rd, f"Missing proportion_{i}"
            assert rd[f"proportion_{i}"] >= 0.0
        total = rd["proportion_1"] + rd["proportion_2"] + rd["proportion_3"]
        assert total == pytest.approx(1.0, abs=1e-4)
        # Verdict is boolean and matches logic
        assert isinstance(record["positive_selection_detected"], bool)
        expected_verdict = bool(
            record["p_value_acat"] < 0.05 or
            record["selection_probability"] > 0.50
        )
        assert record["positive_selection_detected"] == expected_verdict

    def test_bat_oas1_runs(self, busted_cross_dataset):
        result = busted_cross_dataset["bat_oas1"]
        self._check_output_contract(result, 18)

    def test_camelid_runs(self, busted_cross_dataset):
        result = busted_cross_dataset["camelid"]
        self._check_output_contract(result, 212)


class TestBustedErrorHandling:
    """BUSTED should handle missing or malformed tree gracefully."""

    def test_missing_tree_file(self, busted_runner, smc6_paths, tmp_path):
        """tree_path pointing at a nonexistent file should raise an error,
        not silently produce results."""
        fa, _ = smc6_paths
        bad_tree = str(tmp_path / "nonexistent.nwk")
        with pytest.raises((FileNotFoundError, ValueError, IndexError)):
            busted_runner(fa, tree_path=bad_tree)

    def test_malformed_tree(self, busted_runner, smc6_paths, tmp_path):
        """A malformed Newick string should raise an error."""
        fa, _ = smc6_paths
        bad_tree = tmp_path / "malformed.nwk"
        bad_tree.write_text("not_a_tree(((((")
        with pytest.raises((ValueError, IndexError)):
            busted_runner(fa, tree_path=str(bad_tree))
