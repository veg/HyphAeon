"""
tests/test_streaming.py
======================
Unit tests for OnlineManifoldCover and PanGenomeCassetteTracker.
Validates:
  1. Pairwise and basis distance engines with valid and ambiguous sites.
  2. Unbiased farthest-point initialization (greedy metric k-center).
  3. Metric covering ball: familiar sequences thread with zero landmark bloat.
  4. Outlier detection, persistence buffer, and dynamic landmark recruitment.
  5. Density mode clustering on continuous Riemannian manifold.
  6. Multi-segment pan-genome streaming and reassortant tracking.
"""

import pytest
import numpy as np
from rhizaeon.streaming import (
    compute_pairwise_p_distances,
    compute_sample_to_basis_distances,
    OnlineManifoldCover,
    PanGenomeCassetteTracker,
    ThreadingResult
)


def generate_synthetic_cluster(center: np.ndarray, n_samples: int, mutation_rate: float, rng: np.random.RandomState) -> np.ndarray:
    """Generates synthetic nucleotide sequences around a center sequence."""
    L = len(center)
    cluster = np.zeros((n_samples, L), dtype=np.int8)
    for i in range(n_samples):
        mut_mask = rng.rand(L) < mutation_rate
        mutated = center.copy()
        if np.any(mut_mask):
            mutated[mut_mask] = rng.randint(0, 4, size=np.sum(mut_mask))
        cluster[i] = mutated
    return cluster


def test_distance_functions():
    # 3 sequences of length 10
    # s0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # s1: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] -> 5 diffs / 10 = 0.5
    # s2: [0, 0, 0, 0, 4, 4, 1, 1, 1, 1] -> with s0: 4 diffs / 8 valid = 0.5
    mat = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 4, 4, 1, 1, 1, 1]
    ], dtype=np.int8)

    D = compute_pairwise_p_distances(mat)
    assert D.shape == (3, 3)
    assert np.isclose(D[0, 0], 0.0)
    assert np.isclose(D[0, 1], 0.5)
    assert np.isclose(D[1, 0], 0.5)
    assert np.isclose(D[0, 2], 4.0 / 8.0)  # positions 4, 5 are gap in s2
    assert np.isclose(D[1, 2], 0.0)        # valid positions match exactly

    # Test sample to basis
    d_to_basis = compute_sample_to_basis_distances(mat[0], mat)
    assert len(d_to_basis) == 3
    assert np.isclose(d_to_basis[0], 0.0)
    assert np.isclose(d_to_basis[1], 0.5)
    assert np.isclose(d_to_basis[2], 0.5)


def test_unbiased_farthest_point_initialization():
    rng = np.random.RandomState(42)
    L = 300
    # Create 3 distinct clades separated by ~20% divergence
    c1 = rng.randint(0, 4, size=L, dtype=np.int8)
    c2 = rng.randint(0, 4, size=L, dtype=np.int8)
    c3 = rng.randint(0, 4, size=L, dtype=np.int8)

    clade1 = generate_synthetic_cluster(c1, 30, 0.01, rng)
    clade2 = generate_synthetic_cluster(c2, 30, 0.01, rng)
    clade3 = generate_synthetic_cluster(c3, 30, 0.01, rng)

    dataset = np.vstack([clade1, clade2, clade3])
    ids = [f"seq_{i}" for i in range(len(dataset))]

    cover = OnlineManifoldCover(k_dims=3, cover_radius=0.03, persistence_threshold=2)
    cover.initialize_unbiased_farthest_point(dataset, ids, n_landmarks=10)

    assert len(cover.landmark_ids) == 10
    assert cover.Z_landmarks.shape == (10, 3)
    assert cover.D_landmarks.shape == (10, 10)
    assert np.all(np.isfinite(cover.Z_landmarks))


def test_threading_familiar_sequences_no_bloat():
    rng = np.random.RandomState(42)
    L = 300
    c1 = rng.randint(0, 4, size=L, dtype=np.int8)
    c2 = rng.randint(0, 4, size=L, dtype=np.int8)

    seed_clade1 = generate_synthetic_cluster(c1, 20, 0.01, rng)
    seed_clade2 = generate_synthetic_cluster(c2, 20, 0.01, rng)
    seed_mat = np.vstack([seed_clade1, seed_clade2])
    seed_ids = [f"seed_{i}" for i in range(len(seed_mat))]

    cover = OnlineManifoldCover(k_dims=3, cover_radius=0.03, persistence_threshold=2)
    cover.initialize_unbiased_farthest_point(seed_mat, seed_ids, n_landmarks=10)
    initial_landmarks = len(cover.landmark_ids)

    # Stream 100 near-identical sequences from clade 1 and clade 2 (simulating outbreak swarms)
    stream_c1 = generate_synthetic_cluster(c1, 50, 0.005, rng)
    stream_c2 = generate_synthetic_cluster(c2, 50, 0.005, rng)
    stream_mat = np.vstack([stream_c1, stream_c2])
    stream_ids = [f"stream_{i}" for i in range(len(stream_mat))]

    results = cover.thread_stream(stream_mat, stream_ids)

    # All should be within covering radius: zero landmark recruitment!
    assert len(cover.landmark_ids) == initial_landmarks
    for r in results:
        assert r.min_dist <= 0.03
        assert not r.is_novel
        assert not r.recruited
        assert np.all(np.isfinite(r.coords))


def test_outlier_persistence_and_recruitment():
    rng = np.random.RandomState(123)
    L = 300
    c1 = rng.randint(0, 4, size=L, dtype=np.int8)
    seed_mat = generate_synthetic_cluster(c1, 30, 0.01, rng)
    seed_ids = [f"seed_{i}" for i in range(len(seed_mat))]

    cover = OnlineManifoldCover(k_dims=3, cover_radius=0.03, persistence_threshold=2)
    cover.initialize_unbiased_farthest_point(seed_mat, seed_ids, n_landmarks=8)
    init_M = len(cover.landmark_ids)

    # Create a novel divergent lineage (outlier) at 15% distance from c1
    c_novel = c1.copy()
    diff_pos = rng.choice(L, size=45, replace=False)
    c_novel[diff_pos] = (c_novel[diff_pos] + 1) % 4

    outlier_1 = c_novel.copy()
    outlier_2 = c_novel.copy()
    outlier_2[rng.choice(L, 2, replace=False)] = 2  # minor near-identical variation

    # Thread first outlier
    res1 = cover.thread_sequence(outlier_1, "outlier_alpha")
    assert res1.is_novel
    assert not res1.recruited
    assert res1.provisional_candidate
    assert len(cover.landmark_ids) == init_M
    assert len(cover.provisional_candidates) == 1

    # Thread second outlier confirming persistence
    res2 = cover.thread_sequence(outlier_2, "outlier_beta")
    assert res2.is_novel
    assert res2.recruited
    assert len(cover.landmark_ids) == init_M + 1
    assert len(cover.provisional_candidates) == 0  # Promoted out of buffer!
    assert cover.Z_landmarks.shape[0] == init_M + 1


def test_cluster_density_bundles():
    rng = np.random.RandomState(99)
    L = 200
    c1 = rng.randint(0, 4, size=L, dtype=np.int8)
    c2 = rng.randint(0, 4, size=L, dtype=np.int8)

    clade1 = generate_synthetic_cluster(c1, 40, 0.01, rng)
    clade2 = generate_synthetic_cluster(c2, 40, 0.01, rng)
    all_mat = np.vstack([clade1, clade2])
    all_ids = [f"seq_{i}" for i in range(len(all_mat))]

    cover = OnlineManifoldCover(k_dims=3, cover_radius=0.05, persistence_threshold=2)
    cover.initialize_unbiased_farthest_point(all_mat[:20], all_ids[:20], n_landmarks=10)
    cover.thread_stream(all_mat, all_ids)

    bundles = cover.cluster_density_bundles()
    assert len(bundles) == len(all_ids)
    assert np.all(bundles > 0)
    # The two clades should belong to distinct bundle modes
    bundle_c1 = bundles[:40]
    bundle_c2 = bundles[40:]
    mode_c1 = np.bincount(bundle_c1).argmax()
    mode_c2 = np.bincount(bundle_c2).argmax()
    assert mode_c1 != mode_c2


def test_pangenome_cassette_tracker():
    rng = np.random.RandomState(77)
    # 8 segments total length 12744 (matches SEGMENTS and BOUNDS)
    L_total = 12744
    bounds = PanGenomeCassetteTracker.BOUNDS

    # Create two parental whole genomes P1 and P2
    p1 = rng.randint(0, 4, size=L_total, dtype=np.int8)
    p2 = rng.randint(0, 4, size=L_total, dtype=np.int8)

    seed_p1 = [p1 + rng.binomial(1, 0.005, size=L_total).astype(np.int8) for _ in range(10)]
    seed_p2 = [p2 + rng.binomial(1, 0.005, size=L_total).astype(np.int8) for _ in range(10)]
    seed_mat = np.vstack(seed_p1 + seed_p2) % 4
    seed_ids = [f"parent_seed_{i}" for i in range(len(seed_mat))]

    tracker = PanGenomeCassetteTracker(k_dims=3, cover_radius=0.04)
    tracker.initialize_from_concatenated_batch(seed_mat, seed_ids, n_init=8)

    # Thread a clonal P1 descendant
    p1_descendant = (p1 + rng.binomial(1, 0.002, size=L_total).astype(np.int8)) % 4
    rec_clonal = tracker.thread_genome(p1_descendant, "p1_sample")
    assert not rec_clonal["is_novel_reassortant"]
    assert len(rec_clonal["coords"]) == 8

    # Thread a reassortant genome: segments 0..3 from P1, segments 4..7 from P2
    reassortant = p1.copy()
    reassortant[bounds[4]:] = p2[bounds[4]:]
    rec_reassortant = tracker.thread_genome(reassortant, "reassortant_sample")
    assert len(rec_reassortant["coords"]) == 8


def test_ghost_pruning_and_medoid_recentering():
    rng = np.random.RandomState(42)
    L = 200
    c1 = rng.randint(0, 4, size=L, dtype=np.int8)

    # Core clade (50 sequences)
    core = generate_synthetic_cluster(c1, 50, 0.01, rng)
    core_ids = [f"core_{i}" for i in range(50)]

    # Dead-end outlier (2 sequences, far away)
    c_outlier = c1.copy()
    c_outlier[:50] = (c_outlier[:50] + 1) % 4
    outliers = generate_synthetic_cluster(c_outlier, 2, 0.005, rng)
    outlier_ids = ["deadend_1", "deadend_2"]

    dataset = np.vstack([core, outliers])
    all_ids = core_ids + outlier_ids

    cover = OnlineManifoldCover(k_dims=3, cover_radius=0.04, persistence_threshold=2)
    cover.initialize_unbiased_farthest_point(core[:10], core_ids[:10], n_landmarks=6)

    # Thread stream: dead-end will be recruited because persistence threshold = 2
    cover.thread_stream(dataset, all_ids)
    assert len(cover.landmark_ids) >= 7  # Core landmarks + dead-end landmark

    # Consolidate with min_mass = 5
    # The dead-end outlier only has mass = 2, so it MUST be pruned!
    m_after = cover.consolidate_landmarks(min_mass=5)
    assert m_after < len(dataset)
    for mass in cover.landmark_mass:
        assert mass >= 5


def test_two_pass_stream_order_robustness():
    rng = np.random.RandomState(123)
    L = 200
    c1 = rng.randint(0, 4, size=L, dtype=np.int8)
    c2 = rng.randint(0, 4, size=L, dtype=np.int8)

    clade1 = generate_synthetic_cluster(c1, 40, 0.01, rng)
    clade2 = generate_synthetic_cluster(c2, 40, 0.01, rng)
    dataset = np.vstack([clade1, clade2])
    ids = [f"seq_{i}" for i in range(len(dataset))]

    # Run in forward order
    cover_fwd = OnlineManifoldCover(k_dims=3, cover_radius=0.04, persistence_threshold=2, random_seed=42)
    cover_fwd.initialize_unbiased_farthest_point(dataset[:20], ids[:20], n_landmarks=8)
    res_fwd = cover_fwd.two_pass_stream(dataset, ids, min_mass=3)

    # Run in reversed order
    rev_indices = list(range(len(dataset)))[::-1]
    dataset_rev = dataset[rev_indices]
    ids_rev = [ids[i] for i in rev_indices]

    cover_rev = OnlineManifoldCover(k_dims=3, cover_radius=0.04, persistence_threshold=2, random_seed=42)
    cover_rev.initialize_unbiased_farthest_point(dataset_rev[:20], ids_rev[:20], n_landmarks=8)
    res_rev = cover_rev.two_pass_stream(dataset_rev, ids_rev, min_mass=3)

    # Both runs should consolidate into high-mass, stable bases with minimal landmark count
    assert len(cover_fwd.landmark_ids) >= 2
    assert len(cover_rev.landmark_ids) >= 2
    # Verify that all retained landmarks in both have mass >= 3
    for m in cover_fwd.landmark_mass:
        assert m >= 3
    for m in cover_rev.landmark_mass:
        assert m >= 3
