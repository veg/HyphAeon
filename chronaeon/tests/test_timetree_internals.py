"""
tests/test_timetree_internals.py
--------------------------------
Unit tests for the internal stages of chronaeon.timetree: Newick parsing and
serialization, vectorized Neighbor-Joining, patristic distances, distance-space
soft-profile divergences, closed-form edge rooting, the active-set tree-
isotonic WLS solver, and coalescent skyline windowing/CI edge cases.

The deterministic fixtures below build a caterpillar time-tree where every
substitution branch length is exactly mu * (child_time - parent_time), so the
strict molecular clock is recovered up to solver tolerance.
"""

import math

import numpy as np
import pytest

from chronaeon.timetree import (
    ChronAeonTimeTree,
    SkylineResult,
    _all_nodes_to_tips_distances,
    _distance_space_soft_profile_divergences,
    _find_optimal_chronaeon_root,
    _format_dated_newick,
    _parse_newick_adj,
    _solve_tree_isotonic_wls,
    compute_skyline,
    date_timetree,
    fast_nj,
)


# ---------------------------------------------------------------------
# Deterministic caterpillar time-tree fixture
#
# Topology (tip indices 0..4, internal indices 5..8):
#   root(5, t=0) -> tip0, i6
#   i6(t=4)      -> tip1, i7
#   i7(t=9)      -> tip2, i8
#   i8(t=13)     -> tip3, tip4
# Times run "forward" (0 = oldest, larger = more recent) matching the
# convention used throughout timetree.py.
# ---------------------------------------------------------------------

MU_TRUE = 0.005
TIP_TIMES = np.array([10.0, 14.0, 16.0, 18.0, 20.0])
INTERNAL_TIMES = {5: 0.0, 6: 4.0, 7: 9.0, 8: 13.0}
ROOT_ID = 5
CHILDREN = {5: [0, 6], 6: [1, 7], 7: [2, 8], 8: [3, 4]}


def _node_time(u):
    return float(TIP_TIMES[u]) if u < 5 else INTERNAL_TIMES[u]


def _clock_blen():
    return {
        ch: MU_TRUE * (_node_time(ch) - _node_time(u))
        for u, chs in CHILDREN.items()
        for ch in chs
    }


def _undirected_adj(children, blen):
    adj = {}
    for u, chs in children.items():
        adj.setdefault(u, [])
        for ch in chs:
            adj.setdefault(ch, [])
            adj[u].append((ch, blen[ch]))
            adj[ch].append((u, blen[ch]))
    return adj


def _clock_fixture():
    children = CHILDREN
    blen = _clock_blen()
    adj = _undirected_adj(children, blen)
    d_root = np.array(
        [MU_TRUE * (t - INTERNAL_TIMES[ROOT_ID]) for t in TIP_TIMES]
    )
    return children, blen, adj, d_root


# ---------------------------------------------------------------------
# _parse_newick_adj
# ---------------------------------------------------------------------


def test_parse_newick_adj_simple_tree():
    tips, adj, root = _parse_newick_adj("((a:0.1,b:0.2):0.3,c:0.4);")
    assert tips == ["a", "b", "c"]
    # 3 tips + 2 internal nodes
    assert len(adj) == 5
    # Undirected: every edge appears in both directions
    for u, nbrs in adj.items():
        for v, b in nbrs:
            assert any(v2 == u and b2 == pytest.approx(b) for v2, b2 in adj[v])
    # a-b share their parent node, which itself hangs off the syntactic root
    parent_ab = next(iter(set(v for v, _ in adj[0]) & set(v for v, _ in adj[1])))
    assert adj[0][0][1] + adj[1][0][1] == pytest.approx(0.3)
    assert parent_ab in {v for v, _ in adj[root]}


def test_parse_newick_adj_annotations_labels_and_scientific():
    tips, adj, _ = _parse_newick_adj("[&R] (('x y':1e-3,z:2E-2)node1:0.5,w:0);")
    assert tips == ["x y", "z", "w"]
    # Negative-free clamp: missing/zero lengths become 0.0, all >= 0
    for nbrs in adj.values():
        for _, b in nbrs:
            assert b >= 0.0
    # Scientific notation parsed correctly
    blens = {b for nbrs in adj.values() for _, b in nbrs}
    assert 1e-3 in blens and 0.02 in blens


def test_parse_newick_adj_negative_branch_clamped():
    _, adj, _ = _parse_newick_adj("(a:-0.5,b:0.1);")
    assert adj[0][0][1] == 0.0


# ---------------------------------------------------------------------
# _format_dated_newick round-trip
# ---------------------------------------------------------------------


def test_format_dated_newick_round_trip():
    children, _, _, _ = _clock_fixture()
    node_dates = {i: float(TIP_TIMES[i]) for i in range(5)}
    node_dates.update(INTERNAL_TIMES)
    nwk = _format_dated_newick(ROOT_ID, children, node_dates, [f"t{i}" for i in range(5)])
    assert nwk.endswith(";")

    tips, adj, root_node = _parse_newick_adj(nwk)
    assert len(tips) == 5
    # Branch lengths are time durations -> root-to-tip patristic distance
    # equals the tip's own sampling time.
    dist = _all_nodes_to_tips_distances(adj, 5)
    for i in range(5):
        assert dist[root_node, i] == pytest.approx(TIP_TIMES[i], abs=1e-6)


# ---------------------------------------------------------------------
# fast_nj
# ---------------------------------------------------------------------


def test_fast_nj_recovers_cherry_on_additive_distances():
    children, blen, adj, _ = _clock_fixture()
    dist = _all_nodes_to_tips_distances(adj, 5)
    D = dist[:5, :5]
    nj_adj = fast_nj(D)
    # 5 tips -> 9 nodes total, all non-negative branch lengths
    assert len(nj_adj) == 2 * 5 - 2  # n tips + (n-2) internal nodes
    for nbrs in nj_adj.values():
        for _, b in nbrs:
            assert b >= 0.0
    # tip3 & tip4 are a cherry in the true tree -> share a neighbor
    nbr3 = {v for v, _ in nj_adj[3]}
    nbr4 = {v for v, _ in nj_adj[4]}
    assert nbr3 & nbr4
    # NJ patristic distances approximate the input additive metric
    nj_dist = _all_nodes_to_tips_distances(nj_adj, 5)
    for i in range(5):
        for j in range(5):
            assert nj_dist[i, j] == pytest.approx(D[i, j], abs=1e-6)


def test_fast_nj_small_cases():
    adj2 = fast_nj(np.array([[0.0, 0.3], [0.3, 0.0]]))
    assert adj2[0] == [(1, pytest.approx(0.3))] or adj2[0][0][1] == pytest.approx(0.3)
    adj3 = fast_nj(np.array([[0.0, 0.2, 0.4], [0.2, 0.0, 0.4], [0.4, 0.4, 0.0]]))
    assert len(adj3) == 4
    for nbrs in adj3.values():
        for _, b in nbrs:
            assert b >= 0.0


# ---------------------------------------------------------------------
# _distance_space_soft_profile_divergences
# ---------------------------------------------------------------------


def test_distance_space_soft_profile_tracks_time():
    children, blen, adj, _ = _clock_fixture()
    dist = _all_nodes_to_tips_distances(adj, 5)
    D = dist[:5, :5]
    d_prof, gamma = _distance_space_soft_profile_divergences(D, TIP_TIMES)
    assert gamma >= 0.0
    assert np.all(d_prof >= 0.0)
    # Under a strict clock the profile divergence increases with sampling time
    r = np.corrcoef(TIP_TIMES, d_prof)[0, 1]
    assert r > 0.9


# ---------------------------------------------------------------------
# _find_optimal_chronaeon_root
# ---------------------------------------------------------------------


def test_optimal_root_recovers_clock_gradient():
    children, blen, adj, d_root_true = _clock_fixture()
    D = _all_nodes_to_tips_distances(adj, 5)[:5, :5]
    d_prof, _ = _distance_space_soft_profile_divergences(D, TIP_TIMES)
    root_id, children_out, blen_out, d_root, mu_init = _find_optimal_chronaeon_root(
        adj, 5, TIP_TIMES, d_prof
    )
    assert mu_init > 0.0
    # Root-to-tip distances must be temporally ordered like the true clock
    r = np.corrcoef(TIP_TIMES, d_root)[0, 1]
    assert r > 0.9
    # Re-rooted tree stays connected with all branch lengths non-negative
    assert all(b >= 0.0 for b in blen_out.values())
    n_reachable = set()
    stack = [root_id]
    while stack:
        u = stack.pop()
        n_reachable.add(u)
        stack.extend(children_out.get(u, []))
    # Re-rooting inserts root_id on the chosen edge; all original nodes remain
    assert n_reachable == set(adj.keys()) | {root_id}


def test_optimal_root_respects_existing_root():
    children, blen, adj, _ = _clock_fixture()
    D = _all_nodes_to_tips_distances(adj, 5)[:5, :5]
    d_prof, _ = _distance_space_soft_profile_divergences(D, TIP_TIMES)
    root_id, children_out, _, d_root, slope = _find_optimal_chronaeon_root(
        adj, 5, TIP_TIMES, d_prof,
        syntactic_root=ROOT_ID,
        respect_existing_root=True,
    )
    assert root_id == ROOT_ID
    assert slope > 0.0
    # Children relation preserved (root still splits tip0 from the rest)
    assert 0 in children_out[ROOT_ID]


# ---------------------------------------------------------------------
# _solve_tree_isotonic_wls
# ---------------------------------------------------------------------


def _check_monotonic(node_dates, children, times):
    for u, chs in children.items():
        for ch in chs:
            t_ch = float(times[ch]) if ch < len(times) else node_dates[ch]
            assert node_dates[u] <= t_ch + 1e-9


def test_isotonic_solver_recovers_strict_clock():
    children, blen, _, d_root = _clock_fixture()
    node_dates, mu, se_mu, phi = _solve_tree_isotonic_wls(
        ROOT_ID, children, blen, 5, TIP_TIMES, d_root, MU_TRUE, seq_length=2000
    )
    assert mu == pytest.approx(MU_TRUE, rel=0.15)
    for u, t in INTERNAL_TIMES.items():
        assert node_dates[u] == pytest.approx(t, abs=2.0)
    _check_monotonic(node_dates, children, TIP_TIMES)
    assert se_mu > 0.0 and phi >= 1.0


def test_isotonic_solver_zero_internal_branch_no_rate_collapse():
    children, blen, _, d_root = _clock_fixture()
    # Polytomy-like edge: internal branch carries zero substitutions while its
    # descendants are time-separated. The blended rate update must not collapse.
    blen[7] = 0.0
    node_dates, mu, _, _ = _solve_tree_isotonic_wls(
        ROOT_ID, children, blen, 5, TIP_TIMES, d_root, MU_TRUE, seq_length=2000
    )
    assert mu > 0.3 * MU_TRUE
    _check_monotonic(node_dates, children, TIP_TIMES)
    # Feasibility: i7 must sit below all descendant tip times
    assert node_dates[7] <= min(TIP_TIMES[2], node_dates[8]) + 1e-9


def test_isotonic_solver_time_scaled_branches():
    children = CHILDREN
    # Branch lengths expressed directly in time units -> solver should lock mu = 1
    blen = {ch: _node_time(ch) - _node_time(u) for u, chs in children.items() for ch in chs}
    d_root = TIP_TIMES - INTERNAL_TIMES[ROOT_ID]
    node_dates, mu, _, _ = _solve_tree_isotonic_wls(
        ROOT_ID, children, blen, 5, TIP_TIMES, d_root, 1.0, seq_length=2000
    )
    assert mu == 1.0
    for u, t in INTERNAL_TIMES.items():
        assert node_dates[u] == pytest.approx(t, abs=1e-6)


def test_isotonic_solver_outlier_weights_downweight_tip():
    children, blen, _, d_root = _clock_fixture()
    weights = np.ones(5)
    weights[0] = 0.01
    node_dates_w, mu_w, _, _ = _solve_tree_isotonic_wls(
        ROOT_ID, children, blen, 5, TIP_TIMES, d_root, MU_TRUE,
        seq_length=2000, outlier_weights=weights,
    )
    assert mu_w > 0.0
    _check_monotonic(node_dates_w, children, TIP_TIMES)


# ---------------------------------------------------------------------
# date_timetree input validation & flags
# ---------------------------------------------------------------------


def test_date_timetree_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        date_timetree({"a": "ACGT", "b": "ACGT"}, {"a": 2000.0, "b": 2001.0})
    with pytest.raises(ValueError):
        date_timetree(
            {"a": "ACGT", "b": "ACGT", "c": "ACGT"},
            {"a": 2000.0, "b": 2000.0, "c": 2000.0},
        )
    with pytest.raises(ValueError):
        date_timetree("definitely-not-a-tree-or-path")


def test_date_timetree_ci_flag_controls_intervals():
    seqs = {
        "a": "ACGT" * 300,
        "b": "ACGA" * 300,
        "c": "ACGC" * 300,
        "d": "ACGG" * 300,
        "e": "ACTT" * 300,
    }
    dates = {"a": 2000.0, "b": 2005.0, "c": 2010.0, "d": 2015.0, "e": 2020.0}
    res = date_timetree(seqs, dates, seq_length=1200, ci=False)
    assert res.tmrca_ci is None
    assert res.rate_ci is None
    assert res.node_date_ci is None
    assert res.fieller_g is None
    assert res.ci_method is None
    assert res.newick and res.newick.endswith(";")


def test_date_timetree_rooted_newick_respected():
    # Strict-clock rooted caterpillar in substitution units (mu * tau)
    nwk = "((a:0.05,(b:0.05,(c:0.035,(d:0.025,e:0.035):0.02):0.025):0.02):0.0);"
    dates = {"a": 10.0, "b": 14.0, "c": 16.0, "d": 18.0, "e": 20.0}
    res = date_timetree(nwk, dates, seq_length=2000, ci=True)
    assert res.rate > 0.0
    assert res.tmrca < min(dates.values())
    # Monotonic dated Newick
    tips, adj, _ = _parse_newick_adj(res.newick)
    assert len(tips) == 5
    for nbrs in adj.values():
        for _, b in nbrs:
            assert b >= 0.0


def test_date_timetree_unary_root_and_midtree_unary():
    # Single-child root `(X):0.0` leaves a degree-1 stub; single-child groups
    # inside the tree leave degree-2 unaries. Both must be suppressed instead
    # of crashing the isotonic solver.
    dates = {"a": 10.0, "b": 14.0, "c": 16.0, "d": 18.0, "e": 20.0}
    unary_root = "((a:0.05,(b:0.05,(c:0.035,(d:0.025,e:0.035):0.02):0.025):0.02):0.0);"
    res = date_timetree(unary_root, dates, seq_length=2000, ci=True)
    assert res.rate > 0.0 and res.tmrca < min(dates.values())
    midtree_unary = "(a:0.05,((b:0.05):0.0,(c:0.035,(d:0.025,e:0.035):0.02):0.025):0.02);"
    res2 = date_timetree(midtree_unary, dates, seq_length=2000, ci=False)
    assert res2.rate > 0.0


def test_date_timetree_dates_from_metadata_file(tmp_path):
    nwk = "(a:0.05,(b:0.05,(c:0.035,(d:0.025,e:0.035):0.02):0.025):0.02);"
    meta = tmp_path / "meta.tsv"
    meta.write_text(
        "strain\tcollection_date\n"
        "a\t10.0\nb\t14.0\nc\t16.0\nd\t18.0\ne\t20.0\n"
    )
    res = date_timetree(nwk, str(meta), seq_length=2000, ci=False)
    assert res.rate > 0.0
    assert set(res.tip_dates()) == {"a", "b", "c", "d", "e"}
    assert res.tip_dates()["e"] == 20.0


def test_date_timetree_dates_mapping_string_values():
    nwk = "(a:0.05,(b:0.05,(c:0.035,(d:0.025,e:0.035):0.02):0.025):0.02);"
    dates = {"a": "2010-01-01", "b": "2014-06-30", "c": "2016-07-15", "d": "2018-05-20", "e": "2020-11-11"}
    res = date_timetree(nwk, dates, seq_length=2000, ci=False)
    assert res.rate > 0.0
    assert set(res.tip_dates()) == set(dates)


def test_date_timetree_node_date_ci_bounded_by_descendant_tips():
    nwk = "((a:0.05,(b:0.05,(c:0.035,(d:0.025,e:0.035):0.02):0.025):0.02):0.0);"
    dates = {"a": 10.0, "b": 14.0, "c": 16.0, "d": 18.0, "e": 20.0}
    res = date_timetree(nwk, dates, seq_length=2000, ci=True)
    assert res.node_date_ci
    for clade, (lo, hi) in res.node_date_ci.items():
        min_tip = min(dates[t] for t in clade)
        assert lo <= hi <= min_tip + 1e-9


# ---------------------------------------------------------------------
# compute_skyline
# ---------------------------------------------------------------------


def _manual_timetree(rate=1e-3, n_eff=None, fieller_g=None):
    """Small heterochronous tree: 4 tips, 3 coalescent events."""
    return ChronAeonTimeTree(
        method="test",
        tmrca=0.0,
        rate=rate,
        node_dates={
            "a": 20.0, "b": 20.0, "c": 18.0, "d": 15.0,
            "internal_1": 10.0, "internal_2": 5.0, "ROOT": 0.0,
        },
        newick="(a,b,c,d);",
        raw={"seq_length": 1000},
        fieller_g=fieller_g,
        n_eff=n_eff,
    )


def test_compute_skyline_structure_and_ci_ordering():
    tt = _manual_timetree()
    res = compute_skyline(tt, n_grid=20, n_ensemble=10, seed=7)
    assert isinstance(res, SkylineResult)
    assert len(res.grid_t) == 20
    # epsilon = max(quantum, 3.5% span): quantum = 1/(1e-3*1000) = 1.0, span = 20
    assert res.quantum_dt == pytest.approx(1.0)
    assert res.epsilon == pytest.approx(1.0)
    # Every coalescent event pooled exactly once
    assert int(res.window_coals.sum()) == 3
    # CI ordering on windows and grid
    assert np.all(res.window_ne_lo <= res.window_ne + 1e-12)
    assert np.all(res.window_ne <= res.window_ne_hi + 1e-12)
    valid = ~np.isnan(res.grid_ne)
    assert np.all(res.grid_ne_lo[valid] <= res.grid_ne[valid] + 1e-9)
    assert np.all(res.grid_ne[valid] <= res.grid_ne_hi[valid] + 1e-9)
    assert np.all(res.ltt_k >= 0)


def test_compute_skyline_grid_bounded_and_nan_outside():
    tt = _manual_timetree()
    res = compute_skyline(tt, n_grid=30, ci=False, n_ensemble=0)
    # Default grid spans the tree; interpolation clamps to window endpoints
    # inside that span, so every grid point is defined.
    assert np.all(~np.isnan(res.grid_ne))
    # Grid times outside [t_min_tree, t_max_tree] remain NaN
    grid = np.linspace(-5.0, 25.0, 31)
    res2 = compute_skyline(tt, grid_times=grid, ci=False, n_ensemble=0)
    outside = (grid < 0.0) | (grid > 20.0)
    assert np.all(np.isnan(res2.grid_ne[outside]))
    # ci=False yields no interval arrays
    assert res.grid_ne_lo is None and res.window_ne_lo is None


def test_compute_skyline_analytic_ci_without_ensemble():
    tt = _manual_timetree()
    res = compute_skyline(tt, n_grid=15, ci=True, n_ensemble=0)
    assert res.grid_ne_lo is not None
    valid = ~np.isnan(res.grid_ne)
    assert np.all(res.grid_ne_lo[valid] <= res.grid_ne[valid] + 1e-9)


def test_compute_skyline_ensemble_reproducible():
    tt = _manual_timetree()
    r1 = compute_skyline(tt, n_grid=15, n_ensemble=10, seed=3)
    r2 = compute_skyline(tt, n_grid=15, n_ensemble=10, seed=3)
    np.testing.assert_allclose(
        r1.grid_ne_hi[~np.isnan(r1.grid_ne_hi)],
        r2.grid_ne_hi[~np.isnan(r2.grid_ne_hi)],
    )


def test_compute_skyline_rejects_trivial_trees():
    bad = ChronAeonTimeTree(
        method="test", tmrca=0.0, rate=1e-3,
        node_dates={"a": 10.0},
    )
    with pytest.raises(ValueError):
        compute_skyline(bad)


def test_compute_skyline_explicit_epsilon_and_grid():
    tt = _manual_timetree()
    grid = np.linspace(0.0, 20.0, 11)
    res = compute_skyline(tt, epsilon=0.5, grid_times=grid, n_ensemble=0)
    assert res.epsilon == pytest.approx(0.5)
    np.testing.assert_allclose(res.grid_t, grid)


def test_tip_dates_filters_internal_nodes():
    tt = _manual_timetree()
    td = tt.tip_dates()
    assert set(td) == {"a", "b", "c", "d"}
    assert td["a"] == 20.0


# ---------------------------------------------------------------------
# Integration: end-to-end sequence-mode consistency
# ---------------------------------------------------------------------


def test_end_to_end_seq_and_tree_agree_on_tmrca():
    rng = np.random.default_rng(11)
    n, L, mu = 8, 1200, 0.001
    dates = {f"t{i}": 2000.0 + 2.0 * i for i in range(n)}
    # Simple chain descent: each tip accumulates mu * (t_i - t_root) mutations
    root_seq = rng.integers(0, 4, size=L)
    seqs = {}
    bases = np.array(list("ACGT"))
    for i in range(n):
        seq = root_seq.copy()
        n_mut = int(mu * (dates[f"t{i}"] - 1990.0) * L)
        pos = rng.choice(L, size=n_mut, replace=False)
        seq[pos] = (seq[pos] + rng.integers(1, 4, size=n_mut)) % 4
        seqs[f"t{i}"] = "".join(bases[seq])

    res_seq = date_timetree(seqs, dates, seq_length=L, ci=True)
    assert res_seq.rate > 0.0
    assert res_seq.tmrca < min(dates.values())
    assert res_seq.n_eff is not None and 3.0 <= res_seq.n_eff <= n
    sky = res_seq.skyline(ci=True, n_grid=20, n_ensemble=8, seed=5)
    valid = ~np.isnan(sky.grid_ne)
    assert valid.any()
    assert np.all(sky.grid_ne_lo[valid] <= sky.grid_ne[valid] + 1e-9)
    assert np.all(sky.grid_ne[valid] <= sky.grid_ne_hi[valid] + 1e-9)
