"""
WHY THIS FILE EXISTS

`fixtures/e2e/phenotype_RHO_marine_n_permutations_0.json` is the only end-to-end reference for
`run_phenotype_association`, and it records the CLI's JSON and nothing else: 655 taxa x 349
codons of attention and LRT went into it and none of them are in the file. A JavaScript replay of
that fixture can therefore only re-derive the fields that are functions of the recorded columns
(score, ACAT, BH, the EVD block, CESI, the pair p-values, the PARS bracket) — it cannot check the
half of phenotype.py:409-497 that turns the model's two outputs into `association_rho`,
`attribution_norm` and the group means, and it cannot reach the permulation branch at all.

So this script runs the REAL `run_phenotype_association` — the driver, `compute_transformer_
attributions`, `extract_epistatic_sectors_tse`, `generate_permulations`, all of it — against a
FAKE MODEL whose two outputs are a deterministic function of the token tensors, on a synthetic
alignment small enough to record in full. The JSON it writes carries the model outputs, so
`js/test/phenotype.test.js` plays them back through `runPhenotypeAssociation` and compares the
whole report: every site record, every co-selection pair, every sector, and the gene-level block.
It is the same device `test/data/dms/gen.py` uses for the DMS sweep, and it exists for the same
reason: the fixture proves the arithmetic, this proves the plumbing.

THE FAKE MODEL. `forward_cached(c, a, cache, return_attentions=True)` returns

    y_soft[b]   = float32( sum_i ((a_bi + 1) * 0.113 + (c_bi + 1) * 0.0071) * (1 + 0.05 i) - K )
    root_attns[b, i] = float32( 0.002 + 0.03 * (i / (N - 1))**3 + 0.00002 * a_bi )

accumulated left to right in float64 and rounded to float32 once. The attention rises with the
TAXON INDEX, and the synthetic foreground is the four highest-indexed taxa, so `association_rho`
is strongly positive at the variable sites and the pillar reaches its co-selection and sector
branches on ten codons. Sites whose raw y is negative exercise `torch.clamp(y_soft, min=0)` and
the `norm_a > 0` / `min_taxa_per_site` gates.

WHAT EACH CASE IS FOR
  base                     permulations 0, n_permutations 0: every field deterministic, compared
                           exactly (float32 where the reference holds float32).
  min_taxa_per_site_8      raises the per-site gate so sites with few residues drop out.
  alpha_strict             alpha 0.001 shrinks the significant set and with it the co-selection
                           block — the `len(trait_site_indices) >= 2` early exit.
  continuous_trait         a continuous y from the CSV branch: no minimum-foreground check, a
                           z-scored trait vector, and negative rho at some sites.
  permulations_200         permulations 200 with the tree: the Monte Carlo fields cannot match
                           (numpy MT19937 vs Xoshiro256, PLAN.md §5.4), so the test checks their
                           SHAPE — p_assoc == p_assoc_perm, both a multiple of 1/(P+1), and
                           permulations_count == 200 — and every deterministic field exactly.

Regenerate from js/ with the reference environment (repository root on sys.path; torch, numpy,
pandas, Biopython — no weights are loaded, the model is fake):

    python test/data/phenotype/gen_fake_model.py

Output: run_phenotype_association_fake_model.json in this directory.
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

import hyphaeon.phenotype as P  # noqa: E402
from hyphaeon.dataset import load_alignment_and_tree  # noqa: E402

K_OFFSET = 14.0

# 10 taxa x 14 codons. Taxa are ordered so that the four foreground names (fg_*) are the four
# highest indices, which is where the fake attention is largest.
CODONS = {
    "bg_alpha":  ["ATG", "GCT", "TGT", "GAT", "AAA", "GTT", "GGG", "CGT", "TTT", "CCC", "ACT", "TGG", "CAT", "AAA"],
    "bg_beta":   ["ATG", "GCT", "TGT", "GAT", "AAA", "GTC", "GGG", "CGC", "TTT", "CCC", "ACT", "TGG", "CAT", "AAA"],
    "bg_gamma":  ["ATG", "GCT", "TGT", "GAT", "AAA", "CTG", "GGG", "AGA", "TTT", "CCC", "ACT", "TGG", "CAT", "AAA"],
    "bg_delta":  ["ATG", "GCT", "GAT", "GAT", "CCC", "ATT", "AGT", "AGG", "TTT", "---", "ACT", "TGG", "CAT", "AAA"],
    "bg_eps":    ["ATG", "TGT", "GAT", "GAT", "CCC", "TTT", "AGC", "CGG", "TTT", "---", "ACT", "TGG", "TAA", "AAA"],
    "bg_zeta":   ["ATG", "TGT", "TTT", "GAT", "CCC", "TGG", "TGG", "CGA", "TTT", "---", "ACT", "TGG", "NNN", "AAA"],
    "fg_eta":    ["ATG", "TGT", "TTT", "AAT", "CCC", "TGG", "TGG", "CGA", "TAT", "GGG", "GCT", "TGG", "CAT", "AAG"],
    "fg_theta":  ["ATG", "TGT", "TTT", "AAT", "CCC", "TGG", "TGG", "CGA", "TAT", "GGG", "GCT", "TGC", "CAT", "AAG"],
    "fg_iota":   ["ATG", "TGT", "TTT", "AAT", "CCT", "TGG", "TGG", "CGA", "TAT", "GGG", "GCT", "TGG", "CAT", "AAG"],
    "fg_kappa":  ["ATG", "TGT", "TTT", "AAT", "CCT", "TGG", "TGG", "CGA", "TAT", "GGG", "GCT", "TAC", "CAT", "AAG"],
}
ALIGNMENT = "".join(f">{name}\n{''.join(cods)}\n" for name, cods in CODONS.items())
TREE = (
    "((((bg_alpha:0.05,bg_beta:0.05):0.04,bg_gamma:0.09):0.03,"
    "(bg_delta:0.07,(bg_eps:0.04,bg_zeta:0.04):0.03):0.05):0.06,"
    "((fg_eta:0.03,fg_theta:0.03):0.05,(fg_iota:0.04,fg_kappa:0.04):0.04):0.10);\n"
)
TRAIT_CSV = (
    "species,mass\n"
    "bg_alpha,1.5\nbg_beta,2.0\nbg_gamma,2.5\nbg_delta,3.0\nbg_eps,3.5\nbg_zeta,4.0\n"
    "fg_eta,18.0\nfg_theta,19.5\nfg_iota,21.0\nfg_kappa,24.0\n"
)


def fake_site(c_row, a_row):
    """(y_soft, root_attns) for one site, in the accumulation order a JS loop reproduces."""
    acc = 0.0
    for i in range(len(a_row)):
        acc = acc + ((int(a_row[i]) + 1) * 0.113 + (int(c_row[i]) + 1) * 0.0071) * (1.0 + 0.05 * i)
    n = len(a_row)
    attn = [np.float32(0.002 + 0.03 * (i / (n - 1)) ** 3 + 0.00002 * int(a_row[i])) for i in range(n)]
    return acc - K_OFFSET, attn


class FakeModel:
    """Quacks like PhyloAxialTransformer for compute_transformer_attributions."""

    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def eval(self) -> "FakeModel":
        return self

    def precompute_tree_cache(self, dist_matrix, mds_coords):
        return None

    def forward_cached(self, msa_codons, msa_aas, tree_cache, return_attentions=False):
        c = msa_codons.squeeze(-1).cpu().numpy()
        a = msa_aas.squeeze(-1).cpu().numpy()
        self.calls.append([int(c.shape[0]), int(c.shape[1])])
        ys, attns = [], []
        for b in range(c.shape[0]):
            y, at = fake_site(c[b], a[b])
            ys.append(y)
            attns.append(at)
        y_t = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)
        a_t = torch.tensor(np.array(attns, dtype=np.float32), dtype=torch.float32)
        if return_attentions:
            return y_t, None, a_t
        return y_t, None


def main() -> None:
    tmp = tempfile.TemporaryDirectory()
    fa = Path(tmp.name) / "synthetic.fasta"
    nwk = Path(tmp.name) / "synthetic.nwk"
    csv = Path(tmp.name) / "traits.csv"
    fa.write_text(ALIGNMENT)
    nwk.write_text(TREE)
    csv.write_text(TRAIT_CSV)

    c, a, d, z, inv, taxa, L = load_alignment_and_tree(str(fa), str(nwk), prune_duplicates=True)
    assert list(taxa) == list(CODONS), taxa
    assert L == 14, L

    # The model outputs, recorded so the JS replays exactly what the reference saw.
    c_np = c.squeeze(-1).numpy()
    a_np = a.squeeze(-1).numpy()
    lrt_raw, attention = [], []
    for s in range(L):
        y, at = fake_site(c_np[s], a_np[s])
        lrt_raw.append(float(np.float32(y)))
        attention.append([float(v) for v in at])

    P.load_model = lambda weights=None, variant=None, device=None: FakeModel()  # noqa: E731

    cases = []

    def run(name, kwargs, notes):
        res = P.run_phenotype_association(
            alignment_path=str(fa), tree_path=str(nwk), weights_path=None,
            progress=False, cpu=True, **kwargs,
        )
        res["alignment"] = fa.name
        res["tree"] = nwk.name
        cases.append({"name": name, "inputs": kwargs, "outputs": res, "notes": notes})

    run(
        "base",
        {"foreground": "fg_", "permulations": 0, "n_permutations": 0, "seed": 42},
        "permulations 0 and n_permutations 0: no Monte Carlo anywhere, so every field is a "
        "deterministic function of the recorded model outputs and is compared exactly.",
    )
    run(
        "min_taxa_per_site_8",
        {"foreground": "fg_", "permulations": 0, "n_permutations": 0, "min_taxa_per_site": 8, "seed": 42},
        "min_taxa_per_site 8 drops the sites where fewer than 8 taxa carry a residue (site 10 has "
        "three gaps, site 13 a stop and an ambiguity), shortening `sites` and moving BH with it.",
    )
    run(
        "alpha_strict",
        {"foreground": "fg_", "permulations": 0, "n_permutations": 0, "alpha": 0.001, "seed": 42},
        "alpha 0.001 shrinks `significant_sites_count`, which is what feeds the co-selection "
        "block; the `sites` table and the gene-level block are unchanged from `base`.",
    )
    run(
        "continuous_trait",
        {"phenotype_file": str(csv), "continuous": True, "permulations": 0, "n_permutations": 0, "seed": 42},
        "the continuous branch: y is the z-scored body mass, the minimum-foreground check is "
        "skipped, `is_fg` is `y > 0` (the four fg_ taxa, whose mass is above the mean), and "
        "phenotype_meta keeps foreground_count / background_count at 0.",
    )
    run(
        "permulations_200",
        {"foreground": "fg_", "permulations": 200, "n_permutations": 0, "seed": 42},
        "the permulation branch. p_assoc_perm, p_assoc, p_value, q_value, gene_p_value_perm and "
        "everything downstream of them (the sort order, significant_sites_count, the pairs) come "
        "from numpy's MT19937 stream and CANNOT match the port (PLAN.md §5.4); the test checks "
        "their shape and the fields that do not depend on the draws.",
    )

    payload = {
        "generated_by": "js/test/data/phenotype/gen_fake_model.py",
        "numpy": np.__version__,
        "torch": torch.__version__,
        "k_offset": K_OFFSET,
        "alignment": ALIGNMENT,
        "newick": TREE,
        "trait_csv": TRAIT_CSV,
        "trait_csv_name": csv.name,
        "taxa": list(taxa),
        "L": int(L),
        "N": int(len(taxa)),
        "c": [[int(v) for v in row] for row in c_np],
        "a": [[int(v) for v in row] for row in a_np],
        "model_outputs": {"lrt_raw": lrt_raw, "attention": attention},
        "cases": cases,
    }
    out = HERE / "run_phenotype_association_fake_model.json"
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=False)
        fh.write("\n")
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    for case in cases:
        o = case["outputs"]
        print(
            f"  {case['name']:22s} sites={len(o['sites']):3d} sig={o['significant_sites_count']:3d} "
            f"pairs={o['coselection_pairs_count']:3d} sectors={o['trait_sectors_count']} "
            f"perm={o['permulations_count']} pars={o['compact_pars_signature'][:40]}"
        )


if __name__ == "__main__":
    main()
