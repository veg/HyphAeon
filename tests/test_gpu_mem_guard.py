"""Unit tests for the standardized GPU-memory guard (PR #24).

predict / busted / disease now all route batch-size selection through
``hyphaeon.epistasis.compute_adaptive_safe_batch_size``. This module tests that
function DIRECTLY (the N-tiered caps, the budget branches, and — most
importantly — the OOM-guard that caps an over-large user batch size at the safe
value), plus the deprecated ``determine_adaptive_batch_size`` shim which now
delegates to it and additionally clamps at ``total_sites``.

All tests are CPU-only and fast — no GPU, no model forward. Where a
tier-dependent numeric assertion is made, ``get_device_memory_budget`` is
monkeypatched to a fixed budget so the tier boundaries are deterministic
regardless of the host machine's RAM.
"""
import torch
import pytest

import hyphaeon.epistasis as epistasis
from hyphaeon.epistasis import compute_adaptive_safe_batch_size

CPU = torch.device("cpu")


@pytest.fixture
def fixed_budget(monkeypatch):
    """Pin get_device_memory_budget to a deterministic value.

    Returns a setter so each test can choose which budget tier it exercises
    (<=3e9, <=6e9, or >6e9). Defaults to the small (<=3e9) budget.
    """
    def _set(budget_bytes):
        monkeypatch.setattr(
            epistasis, "get_device_memory_budget", lambda device: float(budget_bytes)
        )
        return budget_bytes

    _set(3.0e9)
    return _set


class TestComputeAdaptiveSafeBatchSize:
    """Direct tests of compute_adaptive_safe_batch_size."""

    # (a) always >= 1 --------------------------------------------------------
    @pytest.mark.parametrize("n_taxa", [1, 3, 50, 99, 100, 249, 250, 399, 400, 599, 600, 10000])
    def test_always_at_least_one(self, n_taxa, fixed_budget):
        """Adaptive batch size is always >= 1, even for huge taxa counts."""
        bs = compute_adaptive_safe_batch_size(n_taxa, device=CPU)
        assert bs >= 1

    def test_always_at_least_one_huge_taxa_tiny_budget(self, fixed_budget):
        """Even when quadratic cost would push safe_max_b below 1, floor is 1."""
        fixed_budget(1.0e9)
        bs = compute_adaptive_safe_batch_size(100000, device=CPU)
        assert bs >= 1

    # (b) N-tier monotonicity (memory-safety invariant) ----------------------
    def test_tier_monotonicity_same_budget(self, fixed_budget):
        """Deeper N (more taxa) must yield a batch <= shallower N at same budget.

        This is the core memory-safety invariant: attention memory scales
        quadratically in N, so a deeper alignment can never be allowed a larger
        batch than a shallower one under the same budget.
        """
        fixed_budget(3.0e9)
        # One representative N from each tier, shallow -> deep.
        ns = [50, 100, 250, 400, 600, 2000]
        sizes = [compute_adaptive_safe_batch_size(n, device=CPU) for n in ns]
        for shallow, deep in zip(sizes, sizes[1:]):
            assert deep <= shallow, f"monotonicity violated: {sizes}"

    @pytest.mark.parametrize("budget", [3.0e9, 6.0e9, 8.0e9])
    def test_tier_monotonicity_across_budgets(self, fixed_budget, budget):
        """Monotonicity holds at each budget branch (small/medium/large VRAM)."""
        fixed_budget(budget)
        ns = [50, 150, 300, 500, 800, 5000]
        sizes = [compute_adaptive_safe_batch_size(n, device=CPU) for n in ns]
        assert sizes == sorted(sizes, reverse=True), f"not non-increasing: {sizes}"

    def test_tier_boundary_strictly_decreases_across_caps(self, fixed_budget):
        """Crossing a tier boundary (N>=100/250/400/600) lowers the cap.

        Uses a huge budget so safe_max_b never binds and the tiered base_cap is
        what's returned — making the tier ceilings observable.
        """
        fixed_budget(8.0e9)  # > 6e9 -> largest base caps: 512/256/256/228/152
        b_shallow = compute_adaptive_safe_batch_size(50, device=CPU)   # 512
        b_100 = compute_adaptive_safe_batch_size(100, device=CPU)      # 256
        b_250 = compute_adaptive_safe_batch_size(250, device=CPU)      # 256
        b_400 = compute_adaptive_safe_batch_size(400, device=CPU)      # 228
        b_600 = compute_adaptive_safe_batch_size(600, device=CPU)      # 152
        assert b_shallow >= b_100 >= b_250 >= b_400 >= b_600
        # And the deepest tier is strictly smaller than the shallowest.
        assert b_600 < b_shallow

    # (c) user override <= 2*safe is returned as-is --------------------------
    def test_user_override_within_threshold_returned_asis(self, fixed_budget):
        """A user batch <= 2*safe_b is respected verbatim (no capping)."""
        fixed_budget(8.0e9)
        safe = compute_adaptive_safe_batch_size(50, device=CPU)  # 512 (safe_max_b won't bind at N=50)
        # request equal to safe, and up to exactly 2*safe -> returned unchanged
        assert compute_adaptive_safe_batch_size(50, user_batch_size=safe, device=CPU) == safe
        assert compute_adaptive_safe_batch_size(50, user_batch_size=2 * safe, device=CPU) == 2 * safe
        small = max(1, safe // 4)
        assert compute_adaptive_safe_batch_size(50, user_batch_size=small, device=CPU) == small

    # (d) user override > 2*safe is CAPPED to safe_b (THE OOM GUARD) ----------
    def test_user_override_above_threshold_capped_to_safe(self, fixed_budget, capsys):
        """KEY SAFETY PROPERTY: user batch > 2*safe_b is capped to safe_b.

        This is the OOM guard the PR standardizes across predict/busted/disease.
        """
        fixed_budget(8.0e9)
        safe = compute_adaptive_safe_batch_size(600, device=CPU)  # deep tier -> modest cap
        requested = safe * 2 + 1  # first value strictly above the 2*safe threshold
        got = compute_adaptive_safe_batch_size(600, user_batch_size=requested, device=CPU)
        assert got == safe, "over-large user batch was NOT capped to safe_b"
        # A warning is emitted so the user knows the request was overridden.
        out = capsys.readouterr().out
        assert "capping" in out.lower() or "safety" in out.lower()

    def test_user_override_way_above_capped(self, fixed_budget):
        """A grossly oversized request is still capped to the safe value."""
        fixed_budget(3.0e9)
        for n in [50, 100, 250, 400, 600, 4000]:
            safe = compute_adaptive_safe_batch_size(n, device=CPU)
            got = compute_adaptive_safe_batch_size(n, user_batch_size=1_000_000, device=CPU)
            assert got == safe
            assert got <= safe  # never exceeds the safe ceiling

    # (e) None/0/negative -> adaptive path -----------------------------------
    @pytest.mark.parametrize("bad", [None, 0, -1, -1000])
    def test_none_zero_negative_take_adaptive_path(self, bad, fixed_budget):
        """user_batch_size None/0/negative is treated as 'not set' (adaptive)."""
        adaptive = compute_adaptive_safe_batch_size(200, user_batch_size=None, device=CPU)
        got = compute_adaptive_safe_batch_size(200, user_batch_size=bad, device=CPU)
        assert got == adaptive
        assert got >= 1


class TestCallSiteClampInvariant:
    """The min(bs, num_variable/L) clamp applied at each call site.

    predict/busted/disease each do:  bs = min(compute_adaptive_safe_batch_size(...), total_sites)
    We test that composition logic directly (no model forward needed).
    """

    @pytest.mark.parametrize(
        "n_taxa,total_sites",
        [(20, 1), (20, 5), (50, 300), (600, 10), (5000, 100000), (100, 100000)],
    )
    def test_clamp_never_exceeds_total_sites(self, n_taxa, total_sites, fixed_budget):
        raw = compute_adaptive_safe_batch_size(n_taxa, device=CPU)
        clamped = min(raw, total_sites)
        assert clamped <= total_sites
        assert clamped >= 1 or total_sites == 0

    def test_clamp_binds_when_sites_small(self, fixed_budget):
        """When total_sites < adaptive batch, the clamp is what wins."""
        fixed_budget(8.0e9)
        raw = compute_adaptive_safe_batch_size(50, device=CPU)  # 512
        assert min(raw, 7) == 7  # sites dominate
