"""
aeon-core/tests/test_splits.py
-----------------------------
Unit tests for aeon_core.splits: compute_fused_affinity_matrix.
"""

import numpy as np
import pytest

from aeon_core.splits import compute_fused_affinity_matrix


class TestFusedAffinityMatrix:
    """Unit tests for the multiplicative affinity tensor fusion."""

    def test_symmetry_and_diagonal_zeroing(self):
        n = 8
        rng = np.random.RandomState(42)
        attn = rng.uniform(0.01, 0.5, size=(n, n))
        mds = rng.normal(0, 1, size=(n, 4))
        taxa_repr = rng.normal(0, 1, size=(n, 16))

        A = compute_fused_affinity_matrix(attn, mds, taxa_repr)

        assert A.shape == (n, n)
        assert np.allclose(A, A.T, atol=1e-6)
        assert np.all(np.diag(A) == 0.0)
        assert np.all(A >= 0.0)
        assert np.all(np.isfinite(A))

    def test_without_taxa_repr(self):
        n = 6
        rng = np.random.RandomState(42)
        attn = rng.uniform(0.01, 0.5, size=(n, n))
        mds = rng.normal(0, 1, size=(n, 4))

        A = compute_fused_affinity_matrix(attn, mds, taxa_repr=None)
        assert A.shape == (n, n)
        assert np.allclose(A, A.T, atol=1e-6)
        assert np.all(np.diag(A) == 0.0)
        assert np.all(A >= 0.0)

    def test_degenerate_zero_distance_mds(self):
        n = 5
        attn = np.ones((n, n), dtype=np.float64)
        mds = np.zeros((n, 4), dtype=np.float64)

        A = compute_fused_affinity_matrix(attn, mds)
        assert A.shape == (n, n)
        assert np.all(np.isfinite(A))
        assert np.all(np.diag(A) == 0.0)


class TestSafetyChunkingExtraction:
    """Unit tests for safety chunking and streaming macro-batching in extract_cross_taxa_attentions_and_embeddings."""

    def test_macro_batching_and_chunking(self):
        import torch
        from aeon_core.model import PhyloAxialTransformer
        from aeon_core.splits import extract_cross_taxa_attentions_and_embeddings

        embed_dim = 16
        num_heads = 2
        num_species = 6
        num_codons = 80  # Exceeds default macro_batch_size = 64 to exercise streaming

        model = PhyloAxialTransformer(
            embed_dim=embed_dim, num_layers=2, num_heads=num_heads, window_size=1
        )
        model.eval()

        # Dummy inputs: (num_codons, num_species, 1)
        msa_codons = torch.randint(0, 64, (num_codons, num_species, 1), dtype=torch.long)
        msa_aas = torch.randint(0, 20, (num_codons, num_species, 1), dtype=torch.long)

        dist_matrix = np.random.uniform(0.01, 0.5, size=(num_species, num_species))
        dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
        np.fill_diagonal(dist_matrix, 0.0)
        mds_coords = np.random.normal(0, 1, size=(num_species, 4))

        tree_cache = model.precompute_tree_cache(dist_matrix, mds_coords)

        mean_cross_attn, mean_taxa_repr = extract_cross_taxa_attentions_and_embeddings(
            model, msa_codons, msa_aas, tree_cache, device="cpu"
        )

        assert mean_cross_attn.shape == (num_species, num_species)
        assert mean_taxa_repr.shape == (num_species, embed_dim)
        assert np.all(np.isfinite(mean_cross_attn))
        assert np.all(np.isfinite(mean_taxa_repr))
        assert np.all(mean_cross_attn >= 0.0)

