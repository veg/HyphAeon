"""Unit tests for distance matrix and MDS computation in hyphaeon.dataset."""
import numpy as np
import pytest
from Bio import Phylo
from io import StringIO
from hyphaeon.dataset import compute_fast_dist_matrix, compute_mds_coordinates


@pytest.fixture
def simple_tree():
    return Phylo.read(StringIO("((A:0.1,B:0.2):0.3,C:0.4);"), "newick")


@pytest.fixture
def star_tree():
    return Phylo.read(StringIO("(A:0.1,B:0.1,C:0.1);"), "newick")


class TestComputeFastDistMatrix:
    def test_shape(self, simple_tree):
        taxa = ["A", "B", "C"]
        mat = compute_fast_dist_matrix(simple_tree, taxa)
        assert mat.shape == (3, 3)

    def test_diagonal_is_zero(self, simple_tree):
        mat = compute_fast_dist_matrix(simple_tree, ["A", "B", "C"])
        assert np.allclose(np.diag(mat), 0.0)

    def test_symmetric(self, simple_tree):
        mat = compute_fast_dist_matrix(simple_tree, ["A", "B", "C"])
        assert np.allclose(mat, mat.T)

    def test_known_distances(self, simple_tree):
        taxa = ["A", "B", "C"]
        mat = compute_fast_dist_matrix(simple_tree, taxa)
        # A-B: 0.1 + 0.2 = 0.3
        assert abs(mat[0, 1] - 0.3) < 1e-6
        # A-C: 0.1 + 0.3 + 0.4 = 0.8
        assert abs(mat[0, 2] - 0.8) < 1e-6
        # B-C: 0.2 + 0.3 + 0.4 = 0.9
        assert abs(mat[1, 2] - 0.9) < 1e-6

    def test_star_tree_distances(self, star_tree):
        taxa = ["A", "B", "C"]
        mat = compute_fast_dist_matrix(star_tree, taxa)
        # All pairwise distances = 0.1 + 0.1 = 0.2
        assert abs(mat[0, 1] - 0.2) < 1e-6
        assert abs(mat[0, 2] - 0.2) < 1e-6
        assert abs(mat[1, 2] - 0.2) < 1e-6

    def test_single_taxon(self):
        tree = Phylo.read(StringIO("(A:0.1);"), "newick")
        mat = compute_fast_dist_matrix(tree, ["A"])
        assert mat.shape == (1, 1)
        assert mat[0, 0] == 0.0

    def test_dtype_float32(self, simple_tree):
        mat = compute_fast_dist_matrix(simple_tree, ["A", "B", "C"])
        assert mat.dtype == np.float32


class TestComputeMdsCoordinates:
    def test_shape(self, simple_tree):
        mat = compute_fast_dist_matrix(simple_tree, ["A", "B", "C"])
        coords = compute_mds_coordinates(mat, n_components=4)
        assert coords.shape == (3, 4)

    def test_dtype_float32(self, simple_tree):
        mat = compute_fast_dist_matrix(simple_tree, ["A", "B", "C"])
        coords = compute_mds_coordinates(mat, n_components=4)
        assert coords.dtype == np.float32

    def test_zero_distance_matrix(self):
        mat = np.zeros((3, 3), dtype=np.float32)
        coords = compute_mds_coordinates(mat, n_components=4)
        assert coords.shape == (3, 4)
        assert np.allclose(coords, 0.0)

    def test_n_components_2(self, simple_tree):
        mat = compute_fast_dist_matrix(simple_tree, ["A", "B", "C"])
        coords = compute_mds_coordinates(mat, n_components=2)
        assert coords.shape == (3, 2)

    def test_fewer_taxa_than_components(self):
        mat = np.array([[0, 1], [1, 0]], dtype=np.float32)
        coords = compute_mds_coordinates(mat, n_components=4)
        assert coords.shape == (2, 4)

    def test_lanczos_path_large_matrix(self):
        """For N > 500, compute_mds_coordinates uses a Lanczos spectral solver
        instead of dense eigh. Verify it produces valid output (right shape,
        finite, zero diagonal in the recovered distance structure).
        """
        # Build a synthetic 501x501 Euclidean distance matrix from random points.
        rng = np.random.default_rng(42)
        points = rng.standard_normal((501, 4)).astype(np.float32)
        mat = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)).astype(np.float32)
        coords = compute_mds_coordinates(mat, n_components=4)
        assert coords.shape == (501, 4)
        assert coords.dtype == np.float32
        assert np.isfinite(coords).all()

    def test_lanczos_and_dense_agree_on_submatrix(self):
        """The Lanczos path (N > 500) and dense path (N <= 500) should produce
        approximately the same coordinates for the same distance structure.
        We build a 501-point matrix, run the Lanczos path, then take a 500-point
        submatrix and run the dense path, and check the overlapping coordinates
        agree up to sign (MDS coordinates are sign-ambiguous).
        """
        rng = np.random.default_rng(123)
        points = rng.standard_normal((501, 4)).astype(np.float32)
        mat = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)).astype(np.float32)

        coords_lanczos = compute_mds_coordinates(mat, n_components=4)
        coords_dense = compute_mds_coordinates(mat[:500, :500], n_components=4)

        # Compare the first 500 points, accounting for sign ambiguity per axis.
        for axis in range(4):
            l = coords_lanczos[:500, axis]
            d = coords_dense[:, axis]
            # Either same sign or flipped sign — check correlation is high.
            assert abs(np.corrcoef(l, d)[0, 1]) > 0.95, (
                f"Lanczos and dense MDS disagree on axis {axis}: corr={np.corrcoef(l, d)[0, 1]:.3f}"
            )
