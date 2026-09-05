"""
WHY THIS FILE EXISTS

Reference values for js/src/epistasis.js beyond what fixtures/epistasis/compute_branch_coselection_network.json
pins. That fixture covers the network on one planted 40 x 12 float32 matrix at four filter settings plus the
V < 2 early return; it does not cover compute_transformer_attributions at all (the fixture generator has no
model-free entry to it), the Student-t negative branch (negative cosine), an exactly collinear pair (the
1e-9 guard under 1 - sim^2), df = 1 (N <= 3), a wide matrix (N = 300, where the float32 reduction order of
the norms is visible), or the float32 promotion of the Python thresholds (`sim_arr >= min_sim` compares a
float32 array against a weak Python float, so 0.35 is compared as float32(0.35) = 0.34999999404).

compute_transformer_attributions (hyphaeon/epistasis.py:51-114 at cf838ab) is driven here with a FAKE
torch model: an object whose `eval()` is a no-op and whose `forward_cached(c, a, cache, return_attentions=True)`
returns pre-drawn y_soft rows (some negative, to pin the clamp) and pre-drawn attention rows for the chunk it
is handed, keyed by the chunk's position in the sequential loop. The consensus / delta / p-value / product
lines are therefore the reference's own, on inputs this file records. It is NOT a fixture in the
fixtures/README.md sense (scripts/gen_fixtures.py is not edited by the port); it lives with the tests and
js/test/epistasis.test.js reads its output.

Regenerate from js/ with the reference environment (repository root on sys.path, torch installed):

    python test/data/epistasis/gen.py

Output: transformer_attributions_fake_model.json and coselection_network_cases.json in this directory.
NaN and infinities are written as the strings "NaN", "Infinity", "-Infinity" (the fixtures/README.md
convention); float32 arrays are written as their exact float64 values.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]  # HyphAeon/
sys.path.insert(0, str(ROOT))

from hyphaeon.epistasis import compute_transformer_attributions, compute_branch_coselection_network  # noqa: E402

SEED = 20260905


def jsonable(v):
    if isinstance(v, (np.floating, float)):
        f = float(v)
        if math.isnan(f):
            return "NaN"
        if math.isinf(f):
            return "Infinity" if f > 0 else "-Infinity"
        return f
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return [jsonable(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    return v


def write(name, payload):
    (HERE / name).write_text(json.dumps(jsonable(payload), indent=1) + "\n")


def graph_to_json(G):
    return {
        "nodes": [[n, dict(d)] for n, d in G.nodes(data=True)],
        "edges": [[u, v, dict(d)] for u, v, d in G.edges(data=True)],
    }


class FakeModel:
    """Plays back pre-drawn y_soft / root attention rows chunk by chunk (the loop is sequential)."""

    def __init__(self, y_soft, attn):
        self.y = torch.tensor(y_soft, dtype=torch.float32)
        self.attn = torch.tensor(attn, dtype=torch.float32)
        self.cursor = 0
        self.chunks = []

    def eval(self):
        return self

    def forward_cached(self, c_chunk, a_chunk, tree_cache, return_attentions=False):
        b = c_chunk.shape[0]
        s, e = self.cursor, self.cursor + b
        self.cursor = e
        self.chunks.append(b)
        return self.y[s:e], None, self.attn[s:e]


def gen_attributions():
    rng = np.random.default_rng(SEED)
    cases = []

    def run(name, a_np, y_soft, attn, notes):
        L, N = a_np.shape
        c = torch.zeros((L, N, 1), dtype=torch.long)
        a = torch.tensor(a_np, dtype=torch.long).unsqueeze(-1)
        model = FakeModel(y_soft, attn)
        leaf, lrts, pvals, cons = compute_transformer_attributions(
            model, c, a, {}, [f"t{i}" for i in range(N)], torch.device("cpu"), batch_size=7, progress=False
        )
        assert model.cursor == L
        cases.append({
            "name": name,
            "inputs": {"aa_tokens": a_np, "y_soft": y_soft, "mean_root_attns": attn, "L": L, "N": N},
            "outputs": {
                "leaf_attributions": leaf, "lrts": lrts, "pvals": pvals, "consensus_aas": cons,
                "leaf_dtype": str(leaf.dtype), "lrts_dtype": str(lrts.dtype), "pvals_dtype": str(pvals.dtype),
            },
            "notes": notes,
        })

    # 1. random tokens with 8% unknown (20), ties possible, some negative y_soft
    L, N = 30, 12
    a_np = rng.integers(0, 20, size=(L, N)).astype(np.int64)
    a_np[rng.random((L, N)) < 0.08] = 20
    a_np[3, :] = 20                       # an all-invalid site -> '-' and zero delta
    a_np[4, :6] = 7; a_np[4, 6:] = 2      # exact tie 6 vs 6 between I(7) and D(2): argmax -> lowest token, D
    a_np[5, :] = 11                       # fully conserved -> delta all zero
    y = (rng.standard_normal(L) * 3).astype(np.float32)
    y[0] = -2.5; y[1] = 0.0               # clamp pins
    attn = rng.random((L, N), dtype=np.float32)
    attn = attn / attn.sum(axis=1, keepdims=True)
    run("random_tokens_with_unknowns", a_np, y, attn,
        "a_np: 8% tokens 20; site 3 all 20 (consensus '-', delta 0); site 4 a 6/6 tie between tokens 7 and 2 "
        "(np.argmax -> lowest); site 5 conserved; y_soft[0] negative, y_soft[1] zero (clamp -> 0, Self-Liang p 1.0). "
        "batch_size=7 so the chunk loop runs 5 times (7,7,7,7,2).")

    # 2. token 20 everywhere except one taxon per site -> majority is that single taxon, delta zero
    L, N = 6, 5
    a_np = np.full((L, N), 20, dtype=np.int64)
    for s in range(L):
        a_np[s, s % N] = s % 20
    y = np.linspace(-1, 4, L).astype(np.float32)
    attn = np.full((L, N), 0.2, dtype=np.float32)
    run("single_valid_taxon_per_site", a_np, y, attn,
        "One valid token per site: the consensus is that residue and delta is all zero, so every leaf attribution is 0.")

    # 3. wide N (per-row attention sums to 1 in float32; products with 0/1 delta are exact)
    L, N = 9, 300
    a_np = rng.integers(0, 20, size=(L, N)).astype(np.int64)
    a_np[rng.random((L, N)) < 0.3] = 20
    y = (rng.standard_normal(L) * 4 + 2).astype(np.float32)
    attn = rng.random((L, N), dtype=np.float32) * 0.01
    run("wide_n300", a_np, y, attn, "N=300, 30% unknown tokens; batch_size=7 -> chunks (7,2).")

    write("transformer_attributions_fake_model.json", cases)


def gen_network():
    rng = np.random.default_rng(SEED + 1)
    cases = []

    def run(name, attr, lrts, aas, kw, notes):
        attr = np.asarray(attr, dtype=np.float32)
        lrts = np.asarray(lrts, dtype=np.float32)
        L, N = attr.shape
        pairs, G = compute_branch_coselection_network(attr, lrts, [f"b{i}" for i in range(N)], list(aas), **kw)
        cases.append({
            "name": name,
            "inputs": {"attributions": attr, "lrts": lrts, "consensus_aas": list(aas), "L": L, "N": N, **kw},
            "outputs": {"sig_pairs": pairs, "graph": graph_to_json(G)},
            "notes": notes,
        })

    loose = dict(min_sim=-1.0, min_shared=0, max_fdr=1.0, min_lrt=0.0, min_cesi=-1.0)
    aas20 = "ACDEFGHIKLMNPQRSTVWY"

    # 1. every active pair passes: full check of sim/t/p/q/cesi over all V(V-1)/2 pairs, N=300
    L, N = 24, 300
    attr = rng.random((L, N), dtype=np.float32) * (rng.random((L, N)) < 0.35)
    attr[7, :] = 0.0
    attr[8, :] = attr[2, :] * np.float32(2.0)       # exactly collinear with site 2 -> sim == 1 (or 1 ulp under)
    lrts = (rng.random(L) * 9).astype(np.float32)
    aas = [aas20[i % 20] for i in range(L)]
    run("all_pairs_loose_n300", attr, lrts, aas, loose,
        "Every active pair is an edge (min_sim=-1, max_fdr=1, min_cesi=-1): all M_total statistics visible. Site 7 (0-indexed) "
        "inactive; site 8 = 2 * site 2 so their cosine hits the 1 - sim^2 <= 1e-9 guard. N=300 -> the float32 pairwise norm "
        "order is visible; the sgemm dot is the reference's BLAS.")

    # 2. signed attributions: negative cosines -> negative t -> p > 0.5 (the t.sf(t<0) branch); df = 1 via N = 3
    L, N = 10, 3
    attr = rng.standard_normal((L, N)).astype(np.float32)
    attr[0, :] = 0.0
    lrts = (rng.random(L) * 5).astype(np.float32)
    run("signed_n3_df1", attr, lrts, aas20[:L], loose,
        "N=3 -> df = max(1, 1) = 1; standard normal attributions give negative cosines (t.sf of a negative t). "
        "A_bin = attr > 0 so shared counts only positive coordinates.")

    # 3. N = 2 -> df = max(1, 0) = 1; N = 1 -> df = 1 and every active pair has |sim| == 1
    attr = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]], np.float32)
    run("n2_df_floor", attr, [1, 2, 3, 4], "ACDE", loose, "N=2 -> df=max(1,0)=1; orthogonal (sim 0, t 0, p 0.5), 45-degree and collinear pairs.")
    attr = np.array([[1.0], [2.0], [0.0], [0.5]], np.float32)
    run("n1_all_collinear", attr, [1, 2, 3, 4], "ACDE", loose, "N=1: every active pair is collinear (sim exactly 1).")

    # 4. defaults on the planted-like matrix but with an lrt exactly 0.1 and thresholds hit exactly
    L, N = 12, 10
    attr = rng.random((L, N), dtype=np.float32) * (rng.random((L, N)) < 0.5)
    lrts = (rng.random(L) * 6).astype(np.float32)
    lrts[0] = 0.1; lrts[1] = 1.0; lrts[2] = 0.0
    run("defaults_small", attr, lrts, aas20[:L], {}, "Function defaults (min_sim 0.35, min_shared 2, max_fdr 0.05, min_lrt 1.0, min_cesi 2.0); lrt 0.1 / 1.0 / 0.0 present.")
    run("cli_min_sim_030", attr, lrts, aas20[:L], dict(min_sim=0.30), "CLI default min_sim=0.30 with the other function defaults.")

    # 5. float32 promotion of the thresholds: sim exactly float32(0.35) vs min_sim 0.35 (Python compares in float32)
    b = math.sqrt(1.0 - 0.35 * 0.35)
    attr = np.array([[1.0, 0.0], [0.35, b]], np.float32)
    n1 = np.linalg.norm(attr, axis=1)
    sim = np.float32((attr @ attr.T)[0, 1]) / np.float32(n1[0] * n1[1])
    run("threshold_float32_promotion", attr, [3.0, 3.0], "AC",
        dict(min_sim=0.35, min_shared=0, max_fdr=1.0, min_lrt=0.0, min_cesi=0.0),
        f"sim_arr is float32 and `sim_arr >= min_sim` compares against float32(0.35) = {float(np.float32(0.35))!r} < 0.35; "
        f"sim here = {float(sim)!r}, np.float32(0.35) = {float(np.float32(0.35))!r}; edge present iff sim >= float32(0.35).")
    # the same for min_lrt: max(lrt_u, lrt_v) is float32 and min_lrt 1.1 becomes float32(1.1) = 1.10000002384 > 1.1
    attr = np.array([[1.0, 0.5], [1.0, 0.5]], np.float32)
    run("min_lrt_float32_promotion", attr, [1.1, 0.0], "AC",
        dict(min_sim=0.0, min_shared=0, max_fdr=1.0, min_lrt=1.1, min_cesi=0.0),
        f"max(lrts) = float32(1.1) = {float(np.float32(1.1))!r} compared with min_lrt 1.1 promoted to float32: equal, so the pair passes; "
        "a float64 comparison (1.1000000238 >= 1.1) also passes. Recorded for completeness.")
    run("max_fdr_float32_promotion", np.array([[1.0, 0.0], [0.0, 1.0]], np.float32), [2.0, 2.0], "AC",
        dict(min_sim=-1.0, min_shared=0, max_fdr=0.5, min_lrt=0.0, min_cesi=-1.0),
        "Orthogonal pair: sim 0 -> t 0 -> p 0.5 -> q 0.5 (one test); q is float32(0.5) = 0.5 and `q <= max_fdr` passes.")

    write("coselection_network_cases.json", cases)


if __name__ == "__main__":
    gen_attributions()
    gen_network()
    print("wrote", [p.name for p in HERE.glob("*.json")])
