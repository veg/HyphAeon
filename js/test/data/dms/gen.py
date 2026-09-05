"""
WHY THIS FILE EXISTS

Reference values for js/src/dms.js beyond what fixtures/dms/run_insilico_selection_dms.json can
pin. That fixture is two bat_oas1 sweeps through the real network: every number in it is a
forward pass, so a JS replay can only play the recorded outputs back and check the arithmetic
around them (at the 1e-5 class, and only up to float32 reconstruction of the mutant LRTs). What
it cannot exercise, because bat_oas1's focal taxa carry a residue at every target site, are the
branches of hyphaeon/epistasis.py:446-628 (at cf838ab) that decide WHAT is swept:

  - the wild-type fallback when the focal token is >= 20 (gap, in-frame stop, ambiguity):
    majority residue by np.bincount + argmax (the LOWEST token among tied modes), or 'A' when no
    taxon carries a residue                                                    epistasis.py:522-527
  - focal taxon by case-insensitive substring, FIRST match, index 0 when nothing matches or no
    name is given                                                              epistasis.py:481-487
  - target_sites: ints, dropped when outside [0, L), de-duplicated, sorted   epistasis.py:467-470
  - every site swept whether variable or not (inv_mask is not an argument)
  - the chunking: baseline in chunks of max(1, min(64, safe_batch)); mutants in chunks of
    19 * max(1, safe_batch // 19) elements                                    epistasis.py:477-478, 492
  - the float32 arithmetic: delta = float32(mutant) - float32(baseline); np.mean over the 19
    float32 deltas (numpy pairwise sum, float32 divide)                       epistasis.py:546-556
  - run_digital_dms_analysis's result dict, including `focal_taxon` reporting the caller's
    string rather than the resolved taxon                                     epistasis.py:761-770

So this script runs the real run_insilico_selection_dms and run_digital_dms_analysis with a FAKE
MODEL whose output is a deterministic function of the token tensors:

    y = float32( sum_{i < N} ((a_i + 1) * 0.113 + (c_i + 1) * 0.0071) * (1 + 0.05 * i)  -  10.0 )

accumulated left to right in float64 and rounded to float32 once (torch.tensor(dtype=float32)),
so a JavaScript loop in the same order reproduces it bit for bit and the outputs below are
comparable EXACTLY, not at a tolerance. Sites whose raw value is negative exercise the clamp
(baseline 0 -> p = 1.0). The fake model records the [batch, N] shape of every call so the
chunking can be pinned too.

The alignment is synthetic (6 taxa, 8 codons) and built so that the focal taxon (index 0) has a
gap at sites 1-3, an in-frame stop at site 4 and an ambiguity at site 6, with clear majorities,
ties and an all-gap column among the other taxa. Distances and MDS come from the real
load_alignment_and_tree; the fake model ignores them (they are the same on both sides anyway,
pinned by fixtures/dataset).

It is NOT a fixture in the fixtures/README.md sense (scripts/gen_fixtures.py is not edited by
the port); it lives with the tests. Regenerate from js/ with the reference environment
(repository root on sys.path; torch, numpy, Biopython):

    python test/data/dms/gen.py

Output: run_insilico_selection_dms_fake_model.json in this directory.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]  # HyphAeon/
sys.path.insert(0, str(ROOT))

import hyphaeon.epistasis as E  # noqa: E402
from hyphaeon.dataset import load_alignment_and_tree  # noqa: E402

K_OFFSET = 10.0

# 6 taxa x 8 codons. Column design (focal = M_lyra, index 0):
#   0  all ATG                      invariable; raw value slightly negative -> clamp -> baseline 0
#   1  focal ---, A A A C C         gap at focal -> majority A (token 0)
#   2  focal ---, C C D D F         gap at focal -> tie C(1)/D(2) -> lowest token -> C
#   3  all ---                      no residue anywhere -> wt token 0 -> 'A'
#   4  focal TAA, K K P P P         in-frame stop at focal -> majority P
#   5  focal GTT, V L I F W         a residue at focal, mixed column
#   6  focal NNN, G G S S W         ambiguity at focal -> tie G(5)/S(15) -> G
#   7  R by six codons              codon-variable, amino-acid-invariable
CODONS = {
    "M_lyra":  ["ATG", "---", "---", "---", "TAA", "GTT", "NNN", "CGT"],
    "Beta":    ["ATG", "GCT", "TGT", "---", "AAA", "GTC", "GGG", "CGC"],
    "Gamma":   ["ATG", "GCT", "TGT", "---", "AAA", "CTG", "GGG", "AGA"],
    "delta_X": ["ATG", "GCT", "GAT", "---", "CCC", "ATT", "AGT", "AGG"],
    "Epsilon": ["ATG", "TGT", "GAT", "---", "CCC", "TTT", "AGC", "CGG"],
    "Zeta":    ["ATG", "TGT", "TTT", "---", "CCC", "TGG", "TGG", "CGA"],
}
ALIGNMENT = "".join(f">{name}\n{''.join(cods)}\n" for name, cods in CODONS.items())
TREE = "((M_lyra:0.1,Beta:0.2):0.05,(Gamma:0.15,(delta_X:0.1,Epsilon:0.2):0.1):0.05,Zeta:0.3);\n"


def fake_lrt(c_row, a_row) -> float:
    acc = 0.0
    for i in range(len(a_row)):
        ai = int(a_row[i])
        ci = int(c_row[i])
        acc = acc + ((ai + 1) * 0.113 + (ci + 1) * 0.0071) * (1.0 + 0.05 * i)
    return acc - K_OFFSET


class FakeModel:
    """Quacks like PhyloAxialTransformer for run_insilico_selection_dms / run_digital_dms_analysis."""

    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def eval(self) -> "FakeModel":
        return self

    def precompute_tree_cache(self, dist_matrix, mds_coords):
        return None

    def forward_cached(self, msa_codons, msa_aas, tree_cache):
        c = msa_codons.squeeze(-1).cpu().numpy()
        a = msa_aas.squeeze(-1).cpu().numpy()
        self.calls.append([int(c.shape[0]), int(c.shape[1])])
        vals = [fake_lrt(c[b], a[b]) for b in range(c.shape[0])]
        return torch.tensor(vals, dtype=torch.float32).unsqueeze(-1), None


def focal_of(taxa: list[str], focal_taxon):
    """epistasis.py:481-487 verbatim, for the expected index/name."""
    idx, name = 0, (taxa[0] if taxa else "consensus")
    if focal_taxon:
        for i, t in enumerate(taxa):
            if focal_taxon.lower() in t.lower():
                idx, name = i, t
                break
    return idx, name


def main() -> None:
    tmp = tempfile.TemporaryDirectory()
    fa = Path(tmp.name) / "synthetic.fasta"
    nwk = Path(tmp.name) / "synthetic.nwk"
    fa.write_text(ALIGNMENT)
    nwk.write_text(TREE)
    c, a, d, z, inv, taxa, L = load_alignment_and_tree(str(fa), str(nwk), prune_duplicates=True)
    device = torch.device("cpu")
    a_np = a.squeeze(-1).numpy()
    assert list(taxa) == list(CODONS), taxa
    assert L == 8, L

    cases = []

    def run(name, focal_taxon, target_sites, batch_size, notes):
        """One run_insilico_selection_dms call, with the shapes of every forward pass."""
        model = FakeModel()
        res = E.run_insilico_selection_dms(
            model, c, a, None, taxa, device,
            focal_taxon=focal_taxon, batch_size=batch_size, progress=False,
            target_sites=target_sites,
        )
        idx, fname = focal_of(list(taxa), focal_taxon)
        safe = E.compute_adaptive_safe_batch_size(len(taxa), batch_size, device=device)
        cases.append({
            "name": name,
            "inputs": {
                "focal_taxon": focal_taxon,
                "target_sites": target_sites,
                "batch_size": batch_size,
                "safe_batch_size": int(safe),
                "sites_per_chunk": max(1, int(safe) // 19),
                "baseline_chunk": max(1, min(64, int(safe))),
            },
            "outputs": {
                "focal_index": idx,
                "focal_name": fname,
                "plasticity": res,
                "call_shapes": model.calls,
            },
            "tolerance": "exact",
            "notes": notes,
        })

    run("all_sites_focal_default", None, None, 64,
        "target_sites=None sweeps every one of the 8 codons, invariable included. Focal is index 0 "
        "(M_lyra) with a gap at sites 1-3, an in-frame stop at 4 and an ambiguity at 6, so the "
        "wt fallback (bincount argmax, lowest token on ties; 'A' when no taxon carries a residue) "
        "runs at five of the eight sites. safe_batch=64 -> 3 sites (57 mutants) per forward pass, "
        "baseline in chunks of 64 (one pass for 8 sites).")
    run("focal_delta_x_substring", "DELTA_x", None, 64,
        "focal taxon by CASE-INSENSITIVE SUBSTRING on the taxon name: 'DELTA_x' matches 'delta_X' "
        "at index 3. The whole sweep moves to that taxon's residues.")
    run("focal_no_match_falls_back_to_index_0", "no_such_taxon", [0, 5], 64,
        "a focal_taxon that matches nothing leaves focal_idx 0 / focal_name taxa[0] (epistasis.py "
        "sets the defaults before the loop and never reports the miss).")
    run("focal_empty_string_is_falsy", "", [5], 64,
        "`if focal_taxon:` is false for the empty string, so the substring search never runs.")
    run("target_sites_dedup_sort_and_range", None, [7, 7, 2, -1, 99, 0, 2], 64,
        "target_sites are int()-cast, dropped outside [0, L), de-duplicated and sorted: "
        "[7,7,2,-1,99,0,2] -> [0, 2, 7].")
    run("target_sites_empty_returns_no_records", None, [], 64,
        "an empty target list returns [] BEFORE the baseline sweep, so the model is never called "
        "(call_shapes is empty) — the early return is above the baseline loop.")
    run("batch_size_19_one_site_per_pass", None, None, 19,
        "safe_batch=19 -> sites_per_chunk 1 (19 mutants per pass) and baseline chunks of 19. The "
        "records must be identical to all_sites_focal_default: batching changes call shapes only.")
    run("batch_size_1_minimum_chunks", None, [3, 4], 1,
        "safe_batch=1 -> sites_per_chunk max(1, 0) = 1 and baseline chunks of 1 (8 single-site "
        "passes). Site 3 is the all-gap column ('A' fallback), site 4 the in-frame stop.")

    # run_digital_dms_analysis (epistasis.py:724-770) with the same fake model, via monkeypatched
    # device/model loading — the reference's steps 1-3 are the app runtime's in the port.
    model = FakeModel()
    orig_load_model, orig_get_device = E.load_model, E.get_device
    E.load_model = lambda **kw: model
    E.get_device = lambda cpu=False: device
    try:
        full = E.run_digital_dms_analysis(
            alignment_path=str(fa), tree_path=str(nwk), weights_path=None, variant=None,
            focal_taxon="beta", cpu=True, batch_size=64, progress=False, use_tn93=False,
        )
    finally:
        E.load_model, E.get_device = orig_load_model, orig_get_device
    full["alignment"] = fa.name
    full["tree"] = nwk.name
    digital = {
        "name": "run_digital_dms_analysis_focal_beta",
        "inputs": {"alignment": fa.name, "tree": nwk.name, "focal_taxon": "beta", "batch_size": 64},
        "outputs": {
            "result": full,
            "key_order": list(full.keys()),
            "plasticity_is_selection_dms_plasticity": full["plasticity"] is full["selection_dms_plasticity"],
            "call_shapes": model.calls,
        },
        "tolerance": "exact",
        "notes": ("run_digital_dms_analysis(alignment_path, tree_path, ...) -> the result dict of "
                  "epistasis.py:761-770. `focal_taxon` reports the CALLER'S string ('beta'), not the "
                  "resolved taxon name ('Beta'); total_mutations is 19 * codon_count (every site is "
                  "swept, target_sites is not passed); plasticity and selection_dms_plasticity are "
                  "the same list object. Steps 1-3 (device, alignment load, model load) are the app "
                  "runtime's in the port, so the JS takes the loaded bundle and a predict callback."),
    }

    payload = {
        "generator": "js/test/data/dms/gen.py",
        "alignment": ALIGNMENT,
        "tree": TREE,
        "taxa": list(taxa),
        "L": int(L),
        "k_offset": K_OFFSET,
        "fake_model": ("y = float32(sum_i ((a_i + 1) * 0.113 + (c_i + 1) * 0.0071) * (1 + 0.05 * i) "
                       "- 10.0), accumulated left to right in float64 over the N taxa of one site, "
                       "rounded to float32 once. Deterministic in the tokens alone."),
        "codon_tokens": [[int(v) for v in row] for row in c.squeeze(-1).numpy()],
        "aa_tokens": [[int(v) for v in row] for row in a_np],
        "invariable": [bool(v) for v in inv],
        "cases": cases,
        "digital": digital,
    }
    out = HERE / "run_insilico_selection_dms_fake_model.json"
    out.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    print(f"wrote {out} ({out.stat().st_size} bytes): {len(cases)} sweep cases + the digital record")
    tmp.cleanup()


if __name__ == "__main__":
    main()
