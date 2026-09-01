"""
Determinism tests: same input + same weights → same output.

These tests verify that HyphAeon's predictions don't change with batch size
(floating-point summation order) or repeated runs. A model dev who retrains
and pushes new weights needs to know whether observed differences are real
model changes or floating-point noise.

Also tests fundamental model output properties:
  - LRT clamping: all LRTs >= 0 (enforced by torch.clamp in predict)
  - Invariable-site gate: invariable sites get LRT=0
  - No significance flips near the decision boundary across batch sizes
"""
import numpy as np
import pytest

from _harness import load_tensors, predict, pvals_from_lrt


class TestBatchSizeDeterminism:
    """Changing the batch size should not change predictions (beyond
    floating-point epsilon).

    The model processes variable sites in batches. Different batch sizes
    produce different summation orders in attention, which can cause
    small floating-point differences. These should be negligible (< 1e-3).

    Also checks that significance calls (p <= 0.05) don't flip between
    batch sizes — a small LRT difference near the decision boundary
    (LRT ≈ 2.71) could flip a p-value from 0.049 to 0.051.

    Parametrized over both Smc6 (20 taxa, shallow) and camelid (212 taxa,
    100% variable) to cover small and large datasets.
    """

    @pytest.mark.parametrize("dataset_name", ["smc6", "camelid"])
    @pytest.mark.parametrize("batch_a,batch_b", [(1, 64), (8, 128), (1, 256)])
    def test_batch_size_invariance(self, model, smc6_base, camelid_base,
                                   dataset_name, batch_a, batch_b):
        if dataset_name == "camelid":
            base = camelid_base
            if base is None:
                pytest.skip("camelid failed to load")
        else:
            base = smc6_base

        lrt_a = predict(model, base["c"], base["a"], base["d"], base["z"],
                        base["inv"], batch=batch_a)
        lrt_b = predict(model, base["c"], base["a"], base["d"], base["z"],
                        base["inv"], batch=batch_b)

        tested = base["tested"]
        max_diff = float(np.abs(lrt_a - lrt_b)[tested].max())

        assert max_diff < 1e-3, (
            f"[{dataset_name}] Batch size {batch_a} vs {batch_b}: "
            f"max|ΔLRT|={max_diff:.2e} (threshold <1e-3). "
            f"Predictions are not deterministic across batch sizes."
        )

        # Check no significance flips at alpha=0.05
        # LRT threshold for p=0.05 under 0.5*chi2.sf: 0.5*chi2.sf(x,1)=0.05
        # → chi2.sf(x,1)=0.1 → x = chi2.isf(0.1, 1) ≈ 2.706
        from scipy.stats import chi2
        lrt_threshold = float(chi2.isf(0.10, df=1))  # ≈ 2.706
        sig_a = lrt_a[tested] >= lrt_threshold
        sig_b = lrt_b[tested] >= lrt_threshold
        n_flips = int(np.sum(sig_a != sig_b))
        assert n_flips == 0, (
            f"[{dataset_name}] Batch size {batch_a} vs {batch_b}: "
            f"{n_flips} significance calls flipped at alpha=0.05 "
            f"(LRT threshold={lrt_threshold:.3f}). "
            f"Floating-point differences are crossing the decision boundary."
        )


class TestRunRepeatability:
    """Running the model twice with the same inputs should produce identical
    output (model is in eval mode, no dropout).
    """

    def test_repeated_runs_identical(self, model, smc6_base):
        base = smc6_base
        lrt_1 = predict(model, base["c"], base["a"], base["d"], base["z"],
                        base["inv"])
        lrt_2 = predict(model, base["c"], base["a"], base["d"], base["z"],
                        base["inv"])

        tested = base["tested"]
        max_diff = float(np.abs(lrt_1 - lrt_2)[tested].max())
        assert max_diff == 0.0, (
            f"Repeated runs differ by max|ΔLRT|={max_diff:.2e}. "
            f"Model should be deterministic in eval mode."
        )


class TestLRTProperties:
    """Fundamental properties of the model's LRT output that must always hold.

    These are not calibration tests (they don't check FPR) — they verify
    that the model's output satisfies basic structural constraints:
      - LRTs are non-negative (enforced by torch.clamp in predict)
      - Invariable sites get LRT=0 (the predict function skips them)
    """

    def test_lrts_non_negative(self, model, smc6_base):
        """All LRTs must be >= 0 (enforced by torch.clamp(min=0.0) in predict)."""
        base = smc6_base
        tested = base["tested"]
        lrt = base["lrt"]
        assert (lrt[tested] >= 0).all(), (
            f"Found {(lrt[tested] < 0).sum()} negative LRTs out of "
            f"{tested.sum()} tested sites. The clamp(min=0.0) in predict "
            f"is not working."
        )

    def test_invariable_sites_have_zero_lrt(self, model, smc6_base):
        """Invariable sites must get LRT=0 (predict skips them)."""
        base = smc6_base
        inv = base["inv"]
        lrt = base["lrt"]
        if inv.sum() == 0:
            pytest.skip("No invariable sites in Smc6")
        inv_lrts = lrt[inv]
        assert (inv_lrts == 0).all(), (
            f"Found {(inv_lrts != 0).sum()} non-zero LRTs out of "
            f"{inv.sum()} invariable sites. The invariable-site gate "
            f"(predict skips inv sites) is not working."
        )
        assert inv_lrts.std() == 0.0, (
            f"Invariable site LRTs have non-zero std ({inv_lrts.std():.2e}). "
            f"All invariable sites should produce LRT=0."
        )

    def test_invariable_sites_have_zero_lrt_camelid(self, model, camelid_base):
        """Same check on camelid (100% variable — should skip cleanly)."""
        if camelid_base is None:
            pytest.skip("camelid failed to load")
        inv = camelid_base["inv"]
        if inv.sum() == 0:
            pytest.skip("No invariable sites in camelid (100% variable)")
        inv_lrts = camelid_base["lrt"][inv]
        assert (inv_lrts == 0).all(), (
            f"Camelid: {(inv_lrts != 0).sum()} non-zero LRTs out of "
            f"{inv.sum()} invariable sites."
        )
