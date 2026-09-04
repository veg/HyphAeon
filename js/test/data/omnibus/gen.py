"""
WHY THIS FILE EXISTS

Reference values for js/src/omnibus.js beyond what fixtures/e2e/busted_*.json can pin: the e2e
fixtures carry only two alignments' worth of aggregate statistics (and no per-site LRTs of their
own — the replay in js/test/omnibus.test.js takes those from fixtures/e2e/meme_*.json, which the
same model produced), so the edge cases of the statistical bridge — no variable sites, a single
site, LRTs exactly at 3.841, huge LRTs, NaN-free float32 rounding at sums near 2^9 — are computed
here with the exact lines of hyphaeon/cli.py cmd_busted (cli.py:469-484 at 267f5cf) on seeded
random inputs. It is NOT a fixture in the fixtures/README.md sense (scripts/gen_fixtures.py is not
edited by the port); it lives with the tests.

Regenerate from js/ with the reference environment (repository root on sys.path):

    python test/data/omnibus/gen.py

Output: busted_stats.json in this directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]  # HyphAeon/
sys.path.insert(0, str(ROOT))

from hyphaeon.stats import cauchy_combination_p, pvals_from_lrt_self_liang  # noqa: E402

SEED = 20260904


def busted_stats(lrts: np.ndarray, inv: np.ndarray) -> dict:
    """cli.py:433, 469-484 verbatim."""
    L = len(lrts)
    variable_indices = np.where(~inv)[0]
    num_variable = len(variable_indices)
    pvals = pvals_from_lrt_self_liang(lrts)
    var_p = pvals[variable_indices] if num_variable > 0 else pvals
    p_acat = cauchy_combination_p(var_p)
    sorted_p = np.sort(pvals)
    ranks = np.arange(1, L + 1)
    p_simes = float(np.min((L / ranks) * sorted_p))
    p_simes = max(1e-15, min(1.0, p_simes))
    sig_sites_05 = int(np.sum(pvals < 0.05))
    sig_sites_10 = int(np.sum(pvals < 0.10))
    total_selection_energy = float(np.sum(lrts))
    omnibus_lrt = float(np.sum(np.maximum(0.0, lrts - 3.841)))
    return {
        "num_variable": int(num_variable),
        "pvals": [float(p) for p in pvals],
        "p_value_acat": p_acat,
        "p_value_simes": p_simes,
        "omnibus_lrt": omnibus_lrt,
        "total_selection_energy": total_selection_energy,
        "sig_sites_p05": sig_sites_05,
        "sig_sites_p10": sig_sites_10,
    }


def case(name: str, lrts: np.ndarray, inv: np.ndarray, notes: str) -> dict:
    lrts = lrts.astype(np.float32)
    inv = inv.astype(bool)
    # cmd_busted only writes variable sites; invariable sites stay at the float32 zero.
    lrts[inv] = 0.0
    return {
        "name": name,
        "inputs": {"lrts": [float(x) for x in lrts], "invariable": [bool(b) for b in inv]},
        "outputs": busted_stats(lrts, inv),
        "notes": notes,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    cases = []

    # Exponential-ish LRTs like the model's, with ~10 % invariable sites.
    for L in (1, 7, 9, 130, 335, 1097, 5000):
        lr = np.where(rng.random(L) < 0.6, 0.0, rng.exponential(2.5, L)).astype(np.float32)
        inv = rng.random(L) < 0.1
        cases.append(case(f"random_{L}", lr, inv, f"seeded exponential LRTs, L={L}, ~10% invariable"))

    L = 200
    cases.append(case("all_invariable", np.zeros(L), np.ones(L), "num_variable == 0: ACAT runs over ALL p (= 1.0)"))
    cases.append(case("all_variable_zero_lrt", np.zeros(L), np.zeros(L), "every LRT 0: p = 1 everywhere"))
    lr = np.full(L, 3.841)
    cases.append(case("exactly_threshold", lr, np.zeros(L), "LRT == 3.841 everywhere: float32(3.841) - float32(3.841) = 0 excess"))
    lr = np.full(L, 3.8410001)
    cases.append(case("just_above_threshold", lr, np.zeros(L), "one float32 ulp above 3.841"))
    lr = rng.exponential(80.0, 300)
    cases.append(case("huge_lrts", lr, rng.random(300) < 0.3, "large LRTs, p underflow toward 0, Simes at the 1e-15 floor"))
    lr = np.concatenate([np.full(64, 1.0), np.full(64, 0.5), np.full(64, 0.25), np.full(64, 3.0)])
    cases.append(case("ties", lr, np.zeros(256), "tied p-values; Simes over ties"))
    lr = np.array([1e-45, 1e-30, 1e-6, 1e-3, 0.1, 1.0, 10.0, 100.0, 1000.0], dtype=np.float32)
    cases.append(case("magnitudes", lr, np.zeros(9), "nine magnitudes incl. float32 denormal"))

    out = {
        "generator": "js/test/data/omnibus/gen.py",
        "reference": "hyphaeon/cli.py cmd_busted lines 433, 469-484 at 267f5cf",
        "numpy": np.__version__,
        "seed": SEED,
        "cases": cases,
    }
    (HERE / "busted_stats.json").write_text(json.dumps(out, indent=None, separators=(",", ":")) + "\n")
    print(f"wrote {len(cases)} cases to {HERE / 'busted_stats.json'}")


if __name__ == "__main__":
    main()
