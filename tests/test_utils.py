"""Unit tests for shared utilities in hyphaeon.utils."""

import numpy as np
import pytest
import torch

from hyphaeon.utils import (
    REV_AA_MAP,
    select_device,
    find_taxon_index,
    lrt_to_pvals,
    benjamini_hochberg,
)
from hyphaeon.dataset import AA_MAP


class TestRevAaMap:
    def test_is_inverse_of_aa_map(self):
        for aa, tok in AA_MAP.items():
            assert REV_AA_MAP[tok] == aa

    def test_gap_token_not_in_map(self):
        assert 20 not in REV_AA_MAP


class TestSelectDevice:
    def test_cpu_flag_forces_cpu(self):
        dev = select_device(cpu=True)
        assert dev.type == "cpu"

    def test_returns_torch_device(self):
        dev = select_device()
        assert isinstance(dev, torch.device)


class TestFindTaxonIndex:
    def test_exact_match(self):
        taxa = ["homSap", "musMus", "danRer"]
        assert find_taxon_index(taxa, "musMus") == 1

    def test_case_insensitive(self):
        taxa = ["homSap", "musMus", "danRer"]
        assert find_taxon_index(taxa, "MUSMUS") == 1

    def test_substring_match(self):
        taxa = ["homo_sapiens", "mus_musculus", "danio_rerio"]
        assert find_taxon_index(taxa, "musculus") == 1

    def test_no_match_returns_none(self):
        taxa = ["homSap", "musMus"]
        assert find_taxon_index(taxa, "felCat") is None

    def test_empty_taxa_returns_none(self):
        assert find_taxon_index([], "anything") is None

    def test_first_match_wins(self):
        taxa = ["abc_bat", "def_bat", "ghi_bat"]
        assert find_taxon_index(taxa, "bat") == 0


class TestLrtToPvals:
    def test_zero_lrt_gives_p_one(self):
        lrts = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        pvals = lrt_to_pvals(lrts)
        assert np.allclose(pvals, 1.0)

    def test_positive_lrt_gives_p_lt_half(self):
        lrts = np.array([5.0], dtype=np.float32)
        pvals = lrt_to_pvals(lrts)
        assert 0 < pvals[0] < 0.5

    def test_negative_lrt_gives_p_one(self):
        lrts = np.array([-3.0], dtype=np.float32)
        pvals = lrt_to_pvals(lrts)
        assert pvals[0] == 1.0

    def test_mixed(self):
        lrts = np.array([0.0, 5.0, -1.0, 10.0], dtype=np.float32)
        pvals = lrt_to_pvals(lrts)
        assert pvals[0] == 1.0
        assert pvals[1] < 0.5
        assert pvals[2] == 1.0
        assert pvals[3] < pvals[1]

    def test_preserves_shape(self):
        lrts = np.zeros((3, 4), dtype=np.float32)
        pvals = lrt_to_pvals(lrts)
        assert pvals.shape == (3, 4)

    def test_preserves_dtype(self):
        lrts = np.array([5.0, 0.0, -1.0], dtype=np.float64)
        pvals = lrt_to_pvals(lrts)
        assert pvals.dtype == np.float64

    def test_preserves_dtype_float32(self):
        lrts = np.array([5.0, 0.0, -1.0], dtype=np.float32)
        pvals = lrt_to_pvals(lrts)
        assert pvals.dtype == np.float32

    def test_matches_manual_chi2(self):
        from scipy.stats import chi2
        lrt = 4.0
        expected = 0.5 * chi2.sf(lrt, df=1)
        actual = lrt_to_pvals(np.array([lrt], dtype=np.float32))[0]
        assert np.isclose(actual, expected, rtol=1e-5)

    def test_float64_precision(self):
        """float64 input should preserve full precision (cmd_busted use case)."""
        from scipy.stats import chi2
        lrt = 10.0
        expected = 0.5 * chi2.sf(lrt, df=1)
        actual = lrt_to_pvals(np.array([lrt], dtype=np.float64))[0]
        assert np.isclose(actual, expected, rtol=1e-10)


class TestBenjaminiHochberg:
    def test_empty_input(self):
        q = benjamini_hochberg(np.array([], dtype=np.float32))
        assert len(q) == 0

    def test_single_pvalue(self):
        pvals = np.array([0.01], dtype=np.float32)
        q = benjamini_hochberg(pvals)
        assert np.isclose(q[0], 0.01)

    def test_monotonic_q_values(self):
        pvals = np.array([0.001, 0.01, 0.05, 0.1], dtype=np.float32)
        q = benjamini_hochberg(pvals)
        order = np.argsort(pvals)
        q_sorted = q[order]
        assert np.all(np.diff(q_sorted) >= -1e-7)  # non-decreasing along sorted p-values

    def test_clipped_to_one(self):
        pvals = np.array([0.9, 0.95, 1.0], dtype=np.float32)
        q = benjamini_hochberg(pvals)
        assert np.all(q <= 1.0)

    def test_known_values(self):
        pvals = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float32)
        q = benjamini_hochberg(pvals)
        m = 5
        expected = np.array([0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
        assert np.allclose(q, expected, atol=1e-6)

    def test_preserves_order(self):
        pvals = np.array([0.05, 0.01, 0.03], dtype=np.float32)
        q = benjamini_hochberg(pvals)
        assert q[1] <= q[0]
        assert q[1] <= q[2]

    def test_returns_float32(self):
        pvals = np.array([0.01, 0.02], dtype=np.float64)
        q = benjamini_hochberg(pvals)
        assert q.dtype == np.float32

    def test_matches_manual_bh(self):
        """Verify against a manual BH computation matching the original inline code."""
        pvals = np.array([0.001, 0.01, 0.05, 0.1, 0.5], dtype=np.float32)
        n = len(pvals)
        order = np.argsort(pvals)
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(1, n + 1)
        raw_q = pvals * (n / ranks)
        sorted_q = raw_q[order]
        for i in range(n - 2, -1, -1):
            sorted_q[i] = min(sorted_q[i], sorted_q[i + 1])
        raw_q[order] = sorted_q
        expected = np.clip(raw_q, 0.0, 1.0).astype(np.float32)
        actual = benjamini_hochberg(pvals)
        assert np.allclose(actual, expected, atol=1e-7)
