"""Unit tests for determine_adaptive_batch_size in hyphaeon.cli."""
import torch
import pytest

from hyphaeon.cli import determine_adaptive_batch_size


class TestDetermineAdaptiveBatchSize:
    def test_user_override_respected(self):
        """When user_batch_size is provided and positive, it is used (capped at total_sites)."""
        bs = determine_adaptive_batch_size(20, 1000, torch.device("cpu"), user_batch_size=64)
        assert bs == 64

    def test_user_override_capped_at_total_sites(self):
        """User batch size larger than total sites is capped."""
        bs = determine_adaptive_batch_size(20, 50, torch.device("cpu"), user_batch_size=1000)
        assert bs == 50

    def test_user_override_zero_falls_through_to_adaptive(self):
        """user_batch_size=0 is treated as 'not set' — adaptive path runs."""
        bs = determine_adaptive_batch_size(20, 1000, torch.device("cpu"), user_batch_size=0)
        assert bs >= 1
        assert bs <= 1000

    def test_user_override_negative_falls_through_to_adaptive(self):
        """Negative user_batch_size is treated as 'not set'."""
        bs = determine_adaptive_batch_size(20, 1000, torch.device("cpu"), user_batch_size=-1)
        assert bs >= 1
        assert bs <= 1000

    def test_auto_at_least_one(self):
        """Adaptive batch size is always >= 1, even for huge taxa counts."""
        bs = determine_adaptive_batch_size(10000, 100, torch.device("cpu"))
        assert bs >= 1

    def test_auto_capped_at_total_sites(self):
        """Adaptive batch size never exceeds total_sites."""
        bs = determine_adaptive_batch_size(3, 5, torch.device("cpu"))
        assert bs <= 5

    def test_auto_more_taxa_means_smaller_batch(self):
        """With the same memory budget, more taxa (quadratic cost) → smaller batch."""
        bs_small = determine_adaptive_batch_size(10, 10000, torch.device("cpu"))
        bs_large = determine_adaptive_batch_size(500, 10000, torch.device("cpu"))
        assert bs_large < bs_small

    def test_auto_fewer_sites_means_batch_equals_sites(self):
        """When total_sites is small, batch size equals total_sites (no chunking needed)."""
        bs = determine_adaptive_batch_size(20, 5, torch.device("cpu"))
        assert bs == 5
