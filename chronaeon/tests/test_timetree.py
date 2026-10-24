"""
tests/test_timetree.py
----------------------
Unit and integration tests for ChronAeon TimeTree (`chronaeon.timetree`).
"""

import numpy as np
import pytest

from chronaeon.timetree import (
    ChronAeonTimeTree,
    SkylineResult,
    compute_skyline,
    date_timetree,
    _parse_newick_adj,
)


def _simulate_sequences_and_tree(n_tips=20, mu=0.001, L=900, seed=42):
    rng = np.random.default_rng(seed)
    labels = [f"t{i}" for i in range(n_tips)]

    def rand_topo(labs):
        if len(labs) == 1:
            return labs[0]
        rng.shuffle(labs)
        k = int(rng.integers(1, len(labs)))
        return (rand_topo(labs[:k]), rand_topo(labs[k:]))

    topo = rand_topo(list(labels))
    dates = {}
    bases = np.array(["A", "C", "G", "T"])
    root_seq = rng.integers(0, 4, size=L)
    seqs = {}

    def emit(node, my_time, parent_seq):
        if isinstance(node, str):
            dates[node] = my_time
            seqs[node] = "".join(bases[parent_seq])
            return node
        parts = []
        for child in node:
            tau = float(rng.exponential(12.0))
            ct = my_time + tau
            n_mut = int(rng.poisson(mu * tau * L))
            ch_seq = parent_seq.copy()
            for _ in range(n_mut):
                pos = int(rng.integers(0, L))
                ch_seq[pos] = (ch_seq[pos] + int(rng.integers(1, 4))) % 4
            b = n_mut / float(L)
            sub = emit(child, ct, ch_seq)
            parts.append(f"{sub}:{b:.8f}")
        return "(" + ",".join(parts) + ")"

    nwk = emit(topo, 0.0, root_seq) + ";"
    return nwk, seqs, dates, 0.0, mu


def test_timetree_from_sequences_and_tree():
    nwk, seqs, dates, true_tmrca, true_mu = _simulate_sequences_and_tree(n_tips=25, mu=0.001, L=1200, seed=101)

    # 1. From raw sequences (tree-free input)
    res_seq = date_timetree(seqs, dates, seq_length=1200, ci=True, autoclock=True)
    assert isinstance(res_seq, ChronAeonTimeTree)
    assert abs(res_seq.tmrca - true_tmrca) < 8.0
    assert abs(res_seq.rate - true_mu) < 0.0005
    assert res_seq.tmrca_ci is not None
    assert res_seq.tmrca_ci[0] <= res_seq.tmrca <= res_seq.tmrca_ci[1]
    assert res_seq.newick is not None and res_seq.newick.endswith(";")

    # Verify temporal monotonicity: all branch lengths in dated Newick >= 0
    tips, adj, _ = _parse_newick_adj(res_seq.newick)
    assert len(tips) == 25
    for u, nbrs in adj.items():
        for _v, bl in nbrs:
            assert bl >= 0.0

    # 2. From substitution tree
    res_tree = date_timetree(nwk, dates, seq_length=1200, ci=True)
    assert abs(res_tree.tmrca - true_tmrca) < 5.0
    assert abs(res_tree.rate - true_mu) < 0.0004
    assert res_tree.node_date_ci is not None and len(res_tree.node_date_ci) > 0

    # 3. Coalescent skyline + 95% confidence intervals
    sky_seq = res_seq.skyline(ci=True, n_grid=50)
    assert isinstance(sky_seq, SkylineResult)
    assert len(sky_seq.grid_t) == 50
    assert sky_seq.grid_ne_lo is not None and sky_seq.grid_ne_hi is not None
    valid = ~np.isnan(sky_seq.grid_ne)
    assert np.all(sky_seq.grid_ne_lo[valid] <= sky_seq.grid_ne[valid] + 1e-9)
    assert np.all(sky_seq.grid_ne[valid] <= sky_seq.grid_ne_hi[valid] + 1e-9)
    assert np.all(sky_seq.window_ne_lo <= sky_seq.window_ne)
    assert np.all(sky_seq.window_ne <= sky_seq.window_ne_hi)


def test_timetree_cli(tmp_path):
    import subprocess
    import sys

    _nwk, seqs, dates, _true_t, _true_mu = _simulate_sequences_and_tree(n_tips=15, mu=0.001, L=600, seed=7)
    fa_path = tmp_path / "sim.fasta"
    with open(fa_path, "w") as f:
        for k, v in seqs.items():
            f.write(f">{k}|{dates[k]:.3f}\n{v}\n")

    out_nwk = tmp_path / "dated.nwk"
    out_json = tmp_path / "dated.json"
    cmd = [
        sys.executable, "-m", "chronaeon.cli", "timetree",
        "-a", str(fa_path),
        "-o", str(out_nwk),
        "-j", str(out_json),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert out_nwk.exists()
    assert out_json.exists()
