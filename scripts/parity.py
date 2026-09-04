#!/usr/bin/env python
"""
parity.py -- run the Python reference on the bundled examples and compare every other surface
against it, class by class.

WHY THIS FILE EXISTS

`hyphaeon/*.py` is the reference implementation of the methods; `js/` is a second implementation
of the same methods in TypeScript, consumed by three runtime surfaces that live in the app
repository (veg/hyphaeon-app): the library under Node, the library in headless Chromium, and the
MCP tool. A second implementation drifts. Fixture replay (`js/test`) catches drift per function;
this script catches it end to end, on real alignments, at the level a user sees: the per-site
table, the gene-level omnibus record, the co-selection network.

The script has two halves and they are deliberately decoupled:

  1. It RUNS the Python CLI (`python -m hyphaeon.cli <analysis> ...`) on each requested example
     and writes `parity/python/<example>.<analysis>.json` -- the CLI's own `-o` JSON, unmodified.
  2. It READS `parity/<surface>/<example>.<analysis>.json` for every other requested surface
     and compares. It does not know how those files are produced; the runner that produces them
     lives in the app repository, because nothing in this repository runs a model outside torch
     (js/ is a library: no onnxruntime, no I/O). A surface whose directory is absent is reported
     as missing, never as a failure, unless `--strict-missing` is given.

Tolerance classes are PLAN.md 5.4 in veg/hyphaeon-app, restated in PARITY.md next to this
script. The constants below are cited there; the non-obvious ones:

  - p_value / q_value are checked at 1e-9 "given equal LRT": the reference p-value function
    (`hyphaeon.stats.pvals_from_lrt_meme`) is evaluated on the SURFACE's own LRT, and the
    surface's q is checked against `benjamini_hochberg` of the SURFACE's own p. A 1e-6 drift in
    the LRT (the graph class) would otherwise propagate into p and fail the 1e-9 class on every
    site, telling us nothing new. Evaluating the special function at the surface's input isolates
    the special function, which is what the 1e-9 class is for.
  - p and q are rounded to float32 on both sides before that check because the reference CLI
    writes them as float32 (`cli.py`, cmd_meme: `pvals_from_lrt_meme(lrts).astype(np.float32)`
    and `benjamini_hochberg(pvals).astype(np.float32)`). Comparing a float64 port to a float32
    file at 1e-9 would fail by construction (float32 spacing near 0.67 is ~6e-8).
  - `omnibus_lrt` and `total_selection_energy` are sums over L sites, each within 1e-6, so their
    tolerance is L * 1e-6, not 1e-6.
  - `p_value_acat` and `p_value_simes` are downstream of per-site LRTs that may drift by 1e-6,
    so they are checked at 1e-6, not 1e-9; the combination functions themselves get the 1e-9
    class in fixture replay, where the inputs are identical by construction.
  - `spectral_coherence` is an eigenvalue ratio, so it takes the eigendecomposition tolerance
    (1e-5, the MDS class), not the graph tolerance.
  - The statistical class (p_perm) compares two Monte Carlo estimates at B permutations each,
    with their own seeds: |dp| <= 3 * sqrt(p(1-p)/B), p floored at 1/B so p = 0 keeps a width.
    Null moments (null_coherence_mean / _std / _95) within 2% relative.

Epistasis is only run when the CLI exposes `--seed` (PLAN.md section 7, item 7). Without it the
Python side is reproducible only through the library default (`rng_seed=42` in epistasis.py),
which the CLI does not surface, so the run is skipped with a note rather than producing a file
whose provenance cannot be recorded.

Exit codes: 0 no violations; 1 at least one violation (or a missing surface under
--strict-missing); 2 a reference run failed or the arguments were unusable.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO / "examples"
DEFAULT_OUT = REPO / "parity"

REFERENCE = "python"
KNOWN_SURFACES = ("python", "node", "web", "mcp")
ANALYSES = ("meme", "busted", "epistasis")

# --- tolerance classes (PLAN.md 5.4; PARITY.md) ---------------------------------------------
TOL_GRAPH = 1e-6          # LRT and anything else that comes straight out of the network
TOL_EIGEN = 1e-5          # MDS / eigendecomposition-derived quantities
TOL_SPECIAL = 1e-9        # chi2 / hypergeometric / normal p-values, BH, on identical inputs
TOL_DERIVED = 1e-6        # p-values downstream of graph-class inputs (ACAT, Simes, DMS p)
STAT_SIGMAS = 3.0         # statistical class: |dp| <= 3 * sqrt(p(1-p)/B)
STAT_NULL_REL = 0.02      # statistical class: null moments within 2% relative
DEFAULT_B = 10000         # permutations per side for the statistical class
DEFAULT_SEED = 42         # D17: xoshiro256** on the JS side, default seed 42; --seed here


# --- reference functions ------------------------------------------------------------------
def _reference_stats():
    try:
        from hyphaeon.stats import benjamini_hochberg, pvals_from_lrt_meme
    except ImportError as exc:  # pragma: no cover - environment problem, not a parity result
        print(f"[parity] cannot import hyphaeon.stats ({exc}); install the package first", file=sys.stderr)
        sys.exit(2)
    return pvals_from_lrt_meme, benjamini_hochberg


def f32(x):
    """Round to float32 the way the reference CLI does before writing p and q."""
    return float(np.float32(x))


def rel(path: Path) -> str:
    """Repo-relative for display when inside the repo; absolute otherwise (--out may point anywhere)."""
    path = Path(path)
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


# --- examples ------------------------------------------------------------------------------
def discover_examples() -> dict[str, dict]:
    """Every alignment under examples/ with its optional Newick tree.

    RHO.fasta carries its tree embedded (README: 'Embedded / Auto'), so a missing .nwk means
    'let the CLI extract it', not 'skip'.
    """
    out = {}
    for fasta in sorted(EXAMPLES_DIR.glob("*.fasta")):
        name = fasta.stem
        nwk = EXAMPLES_DIR / f"{name}.nwk"
        out[name] = {"alignment": fasta, "tree": nwk if nwk.exists() else None}
    return out


# --- running the reference -----------------------------------------------------------------
def cli_has_flag(analysis: str, flag: str) -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "hyphaeon.cli", analysis, "--help"],
        cwd=REPO, capture_output=True, text=True,
    )
    # Match the option token itself; 'seed' also appears in prose ("clique seed size").
    return re.search(rf"(^|\s){re.escape(flag)}(\b|[=\s])", proc.stdout) is not None


def run_reference(example: str, spec: dict, analysis: str, out_dir: Path, args) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{example}.{analysis}.json"
    log = out_dir / f"{example}.{analysis}.log"
    record = {"surface": REFERENCE, "example": example, "analysis": analysis,
              "path": rel(out_json), "status": "ok", "note": None, "seconds": None}

    if args.no_run:
        if out_json.exists():
            record["note"] = "reused existing file (--no-run)"
        else:
            record["status"] = "missing"
            record["note"] = "no existing file and --no-run given"
        return record

    cmd = [sys.executable, "-m", "hyphaeon.cli", analysis,
           "-a", str(spec["alignment"].relative_to(REPO)), "--cpu", "-o", str(out_json)]
    if spec["tree"] is not None:
        cmd += ["-t", str(spec["tree"].relative_to(REPO))]
    if args.weights:
        cmd += ["-w", args.weights]
    if analysis == "epistasis":
        cmd += ["--n-permutations", str(args.n_permutations), "--seed", str(args.seed)]
    record["command"] = " ".join(cmd)

    t0 = time.time()
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, text=True)
    record["seconds"] = round(time.time() - t0, 3)
    if proc.returncode != 0 or not out_json.exists():
        record["status"] = "failed"
        record["note"] = f"exit {proc.returncode}; see {rel(log)}"
    return record


# --- comparison primitives -----------------------------------------------------------------
class Check:
    """One field (or one keyed collection of a field) compared under one class."""

    def __init__(self, field: str, cls: str, tol: float | None = None):
        self.field = field
        self.cls = cls
        self.tol = tol
        self.n = 0
        self.max_abs_diff = 0.0
        self.violations: list[dict] = []
        self.note: str | None = None

    def add(self, key, ref, got, tol: float | None = None):
        self.n += 1
        tol = self.tol if tol is None else tol
        if self.cls == "exact":
            if ref != got:
                self.violations.append({"key": key, "ref": ref, "got": got})
            return
        if ref is None or got is None:
            self.violations.append({"key": key, "ref": ref, "got": got, "reason": "null"})
            return
        try:
            diff = abs(float(ref) - float(got))
        except (TypeError, ValueError):
            self.violations.append({"key": key, "ref": ref, "got": got, "reason": "not numeric"})
            return
        if math.isnan(diff):
            same_nan = isinstance(ref, float) and isinstance(got, float) and math.isnan(ref) and math.isnan(got)
            if not same_nan:
                self.violations.append({"key": key, "ref": ref, "got": got, "reason": "nan"})
            return
        self.max_abs_diff = max(self.max_abs_diff, diff)
        if diff > tol:
            self.violations.append({"key": key, "ref": ref, "got": got, "abs_diff": diff, "tol": tol})

    def to_json(self) -> dict:
        d = {"field": self.field, "class": self.cls, "n": self.n,
             "violations": len(self.violations)}
        if self.cls != "exact":
            d["tol"] = self.tol
            d["max_abs_diff"] = self.max_abs_diff
        if self.note:
            d["note"] = self.note
        if self.violations:
            d["first_violations"] = self.violations[:20]
        return d


def stat_tol(p_ref: float, B: int) -> float:
    p = min(max(float(p_ref), 1.0 / B), 1.0 - 1.0 / B)
    return STAT_SIGMAS * math.sqrt(p * (1.0 - p) / B)


def _get(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def scalar_checks(ref: dict, got: dict, spec: list[tuple[str, str, float | None]], scale: float = 1.0) -> list[Check]:
    checks = []
    for field, cls, tol in spec:
        c = Check(field, cls, None if tol is None else tol * (scale if cls == "sum" else 1.0))
        if cls == "sum":
            c.cls = "tolerance"
        r, g = _get(ref, field), _get(got, field)
        if r is None and g is None:
            c.note = "absent on both sides"
            checks.append(c)
            continue
        c.add(field, r, g)
        checks.append(c)
    return checks


# --- meme ----------------------------------------------------------------------------------
MEME_TOP = [
    ("taxa_count", "exact", None),
    ("codon_count", "exact", None),
]


def compare_meme(ref: dict, got: dict) -> tuple[list[Check], list[str]]:
    pvals_from_lrt_meme, benjamini_hochberg = _reference_stats()
    notes: list[str] = []
    checks = scalar_checks(ref, got, MEME_TOP)

    rs, gs = ref.get("sites") or [], got.get("sites") or []
    order = Check("site order", "exact")
    r_ids, g_ids = [s.get("site") for s in rs], [s.get("site") for s in gs]
    order.n = max(len(r_ids), len(g_ids))
    if r_ids != g_ids:
        first = next((i for i, (a, b) in enumerate(zip(r_ids, g_ids)) if a != b), min(len(r_ids), len(g_ids)))
        order.violations.append({"key": f"index {first}", "ref": r_ids[first] if first < len(r_ids) else None,
                                 "got": g_ids[first] if first < len(g_ids) else None,
                                 "ref_len": len(r_ids), "got_len": len(g_ids)})
        checks.append(order)
        notes.append("site order differs; per-site fields not compared")
        return checks, notes
    checks.append(order)

    inv = Check("is_invariable", "exact")
    lrt = Check("hyphaeon_lrt", "tolerance", TOL_GRAPH)
    for r, g in zip(rs, gs):
        inv.add(r["site"], r.get("is_invariable"), g.get("is_invariable"))
        lrt.add(r["site"], r.get("hyphaeon_lrt"), g.get("hyphaeon_lrt"))
    checks += [inv, lrt]
    checks += _pq_given_lrt(gs, pvals_from_lrt_meme, benjamini_hochberg)
    return checks, notes


def _pq_given_lrt(sites: list[dict], pvals_from_lrt_meme, benjamini_hochberg) -> list[Check]:
    """p and q at 1e-9 given the surface's own LRT (see the header)."""
    p = Check("p_value", "special", TOL_SPECIAL)
    q = Check("q_value", "special", TOL_SPECIAL)
    p.note = "expected = float32(pvals_from_lrt_meme(surface lrt)); surface p rounded to float32"
    q.note = "expected = float32(benjamini_hochberg(surface p)); surface q rounded to float32"
    if not sites:
        return [p, q]
    lrts = np.array([float(s.get("hyphaeon_lrt", 0.0)) for s in sites], dtype=np.float64)
    got_p = np.array([float(s.get("p_value", np.nan)) for s in sites], dtype=np.float64)
    got_q = np.array([float(s.get("q_value", np.nan)) for s in sites], dtype=np.float64)
    exp_p = pvals_from_lrt_meme(lrts)
    # The reference computes BH on the float32-cast p, so feed BH the float32-rounded surface p.
    exp_q = benjamini_hochberg(got_p.astype(np.float32))
    for s, gp, ep, gq, eq in zip(sites, got_p, exp_p, got_q, exp_q):
        p.add(s["site"], f32(ep), f32(gp))
        q.add(s["site"], f32(eq), f32(gq))
    return [p, q]


# --- busted --------------------------------------------------------------------------------
BUSTED_SPEC = [
    ("taxa", "exact", None),
    ("sites", "exact", None),
    ("sig_sites_p05", "exact", None),
    ("sig_sites_p10", "exact", None),
    ("positive_selection_detected", "exact", None),
    ("predicted_gene_lrt", "tolerance", TOL_GRAPH),
    ("selection_probability", "tolerance", TOL_GRAPH),
    ("synonymous_rate_variation", "tolerance", TOL_GRAPH),
    ("rate_distributions.omega_1", "tolerance", TOL_GRAPH),
    ("rate_distributions.omega_2", "tolerance", TOL_GRAPH),
    ("rate_distributions.omega_3", "tolerance", TOL_GRAPH),
    ("rate_distributions.proportion_1", "tolerance", TOL_GRAPH),
    ("rate_distributions.proportion_2", "tolerance", TOL_GRAPH),
    ("rate_distributions.proportion_3", "tolerance", TOL_GRAPH),
    ("omnibus_lrt", "sum", TOL_GRAPH),               # scaled by L (sum of L graph-class values)
    ("total_selection_energy", "sum", TOL_GRAPH),    # scaled by L
    ("p_value_acat", "derived", TOL_DERIVED),
    ("p_value_simes", "derived", TOL_DERIVED),
]


def compare_busted(ref: dict, got: dict) -> tuple[list[Check], list[str]]:
    # The CLI writes a list for batch runs and a dict for a single alignment; parity runs one.
    if isinstance(ref, list):
        ref = ref[0]
    if isinstance(got, list):
        got = got[0]
    L = int(ref.get("sites") or 1)
    return scalar_checks(ref, got, BUSTED_SPEC, scale=L), []


# --- epistasis -----------------------------------------------------------------------------
EPI_TOP = [
    ("taxa_count", "exact", None),
    ("codon_count", "exact", None),
    ("branch_count", "exact", None),
    ("coevolution_edges_count", "exact", None),
    ("sectors_discovered", "exact", None),
]
EDGE_EXACT = ("ref_u", "ref_v", "shared_branches", "branches_u", "branches_v")
EDGE_GRAPH = ("lrt_u", "lrt_v", "similarity", "cesi")
EDGE_SPECIAL = ("p_hyper", "fdr_q")     # hypergeometric on integer counts, BH over those
SECTOR_EXACT = ("size", "sites", "shared_branches", "pars_signature")
SECTOR_STAT_MOMENTS = ("null_coherence_mean", "null_coherence_std", "null_coherence_95")
PLAST_EXACT = ("wt_aa", "focal_taxon")
PLAST_GRAPH = ("baseline_lrt", "intrinsic_plasticity", "max_shift")


def _keyed_collection(items: list[dict], key_fields: tuple[str, ...]) -> tuple[list, dict]:
    keys = [tuple(it.get(k) for k in key_fields) for it in items]
    return keys, dict(zip(keys, items))


def compare_epistasis(ref: dict, got: dict, B: int) -> tuple[list[Check], list[str]]:
    notes: list[str] = []
    checks = scalar_checks(ref, got, EPI_TOP)

    # edges: the set and order are exact (graph edges), then per-edge fields
    r_keys, r_edges = _keyed_collection(ref.get("edges") or [], ("site_u", "site_v"))
    g_keys, g_edges = _keyed_collection(got.get("edges") or [], ("site_u", "site_v"))
    eorder = Check("edges (site_u, site_v) order", "exact")
    eorder.n = max(len(r_keys), len(g_keys))
    if r_keys != g_keys:
        missing = sorted(set(r_keys) - set(g_keys))[:10]
        extra = sorted(set(g_keys) - set(r_keys))[:10]
        eorder.violations.append({"key": "edges", "ref_len": len(r_keys), "got_len": len(g_keys),
                                  "missing_in_surface": missing, "extra_in_surface": extra})
    checks.append(eorder)
    for name in EDGE_EXACT:
        checks.append(_keyed_check(name, "exact", None, r_keys, r_edges, g_edges))
    for name in EDGE_GRAPH:
        checks.append(_keyed_check(name, "tolerance", TOL_GRAPH, r_keys, r_edges, g_edges))
    for name in EDGE_SPECIAL:
        checks.append(_keyed_check(name, "special", TOL_SPECIAL, r_keys, r_edges, g_edges))

    # sectors
    r_keys, r_sec = _keyed_collection(ref.get("sectors") or [], ("sector_id",))
    g_keys, g_sec = _keyed_collection(got.get("sectors") or [], ("sector_id",))
    sorder = Check("sectors sector_id order", "exact")
    sorder.n = max(len(r_keys), len(g_keys))
    if r_keys != g_keys:
        sorder.violations.append({"key": "sectors", "ref": r_keys, "got": g_keys})
    checks.append(sorder)
    for name in SECTOR_EXACT:
        checks.append(_keyed_check(name, "exact", None, r_keys, r_sec, g_sec))
    checks.append(_keyed_check("mean_lrt", "tolerance", TOL_GRAPH, r_keys, r_sec, g_sec))
    checks.append(_keyed_check("spectral_coherence", "tolerance", TOL_EIGEN, r_keys, r_sec, g_sec))

    have_ref = any("p_perm" in r_sec[k] for k in r_keys)
    have_got = any("p_perm" in g_sec.get(k, {}) for k in r_keys)
    if r_keys and have_ref and have_got:
        pp = Check("p_perm", "statistical", None)
        pp.note = f"|dp| <= {STAT_SIGMAS:g}*sqrt(p(1-p)/B), B={B}, p floored at 1/B"
        for k in r_keys:
            r, g = r_sec[k], g_sec.get(k)
            if g is None or "p_perm" not in r or "p_perm" not in g:
                continue
            pp.add(k[0], r["p_perm"], g["p_perm"], tol=stat_tol(r["p_perm"], B))
        checks.append(pp)
        for name in SECTOR_STAT_MOMENTS:
            mc = Check(name, "statistical", None)
            mc.note = f"relative difference <= {STAT_NULL_REL:g}"
            for k in r_keys:
                r, g = r_sec[k], g_sec.get(k)
                if g is None or name not in r or name not in g:
                    continue
                mc.add(k[0], r[name], g[name], tol=STAT_NULL_REL * max(abs(float(r[name])), 1e-12))
            checks.append(mc)
    else:
        why = "no sectors" if not r_keys else ("absent in reference" if not have_ref else "absent in surface")
        notes.append(f"statistical class (p_perm, null_coherence_*) skipped: {why}")

    # plasticity (DMS per site), present unless --no-dms
    r_keys, r_pl = _keyed_collection(ref.get("plasticity") or [], ("site",))
    g_keys, g_pl = _keyed_collection(got.get("plasticity") or [], ("site",))
    if r_keys or g_keys:
        porder = Check("plasticity site order", "exact")
        porder.n = max(len(r_keys), len(g_keys))
        if r_keys != g_keys:
            porder.violations.append({"key": "plasticity", "ref_len": len(r_keys), "got_len": len(g_keys)})
        checks.append(porder)
        for name in PLAST_EXACT:
            checks.append(_keyed_check(name, "exact", None, r_keys, r_pl, g_pl))
        for name in PLAST_GRAPH:
            checks.append(_keyed_check(name, "tolerance", TOL_GRAPH, r_keys, r_pl, g_pl))
        checks.append(_keyed_check("p_value", "derived", TOL_DERIVED, r_keys, r_pl, g_pl))
    else:
        notes.append("plasticity absent on both sides (--no-dms?)")
    return checks, notes


def _keyed_check(name: str, cls: str, tol, keys: list, ref: dict, got: dict) -> Check:
    c = Check(name, cls, tol)
    present = 0
    for k in keys:
        r, g = ref.get(k), got.get(k)
        if g is None:
            continue  # the order check already reports the missing item
        if name not in r and name not in g:
            continue
        present += 1
        c.add(k if len(k) > 1 else k[0], r.get(name), g.get(name))
    if keys and present == 0:
        c.note = "absent on both sides"
    return c


# --- driver --------------------------------------------------------------------------------
def load_json(path: Path):
    with open(path) as fh:
        return json.load(fh)


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def parse_list(value: str, universe: tuple[str, ...] | None, name: str) -> list[str]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    if universe is not None:
        bad = [v for v in items if v not in universe]
        if bad:
            print(f"[parity] unknown {name}: {', '.join(bad)} (choose from {', '.join(universe)})", file=sys.stderr)
            sys.exit(2)
    return items


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", default="all",
                    help="comma-separated example names from examples/ (stem of the .fasta), or 'all' (default)")
    ap.add_argument("--surfaces", default="python,node",
                    help=f"comma-separated surfaces to compare; '{REFERENCE}' is the reference and is always run "
                         f"(default: python,node; known: {', '.join(KNOWN_SURFACES)})")
    ap.add_argument("--analyses", default=",".join(ANALYSES),
                    help=f"comma-separated analyses (default: {','.join(ANALYSES)})")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="parity directory (default: <repo>/parity)")
    ap.add_argument("--weights", default=os.environ.get("HYPHAEON_WEIGHTS") or None,
                    help="passed to the CLI as -w (default: $HYPHAEON_WEIGHTS; unset means the CLI's own resolution)")
    ap.add_argument("--n-permutations", type=int, default=DEFAULT_B,
                    help=f"B for the statistical class and for the epistasis run (default {DEFAULT_B})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"seed passed to epistasis --seed (default {DEFAULT_SEED})")
    ap.add_argument("--no-run", action="store_true", help="do not run the Python CLI; reuse parity/python/*.json")
    ap.add_argument("--strict-missing", action="store_true",
                    help="treat a missing surface file as a violation (off: reported, exit 0)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out).resolve()
    examples = discover_examples()
    if args.examples == "all":
        wanted = list(examples)
    else:
        wanted = parse_list(args.examples, tuple(examples), "example")
    surfaces = parse_list(args.surfaces, KNOWN_SURFACES, "surface")
    if REFERENCE not in surfaces:
        surfaces.insert(0, REFERENCE)
    others = [s for s in surfaces if s != REFERENCE]
    analyses = parse_list(args.analyses, ANALYSES, "analysis")

    report = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": git_commit(),
        "reference": REFERENCE,
        "surfaces_requested": surfaces,
        "surfaces_present": [],
        "surfaces_missing": [],
        "examples": wanted,
        "analyses": analyses,
        "weights": args.weights,
        "n_permutations": args.n_permutations,
        "seed": args.seed,
        "tolerances": {
            "exact": "equality after canonicalisation",
            "tolerance_graph": TOL_GRAPH, "tolerance_eigen": TOL_EIGEN,
            "special": TOL_SPECIAL, "derived": TOL_DERIVED,
            "statistical": {"sigmas": STAT_SIGMAS, "null_moment_rel": STAT_NULL_REL},
        },
        "runs": [],
        "self_checks": [],
        "comparisons": [],
        "notes": [],
        "summary": {},
    }

    # epistasis needs a CLI seed to be reproducible on the Python side
    if "epistasis" in analyses and not args.no_run and not cli_has_flag("epistasis", "--seed"):
        report["notes"].append("epistasis skipped: the CLI has no --seed flag yet (PLAN.md section 7, item 7); "
                               "the library default rng_seed=42 is not surfaced, so the run would not be reproducible")
        analyses = [a for a in analyses if a != "epistasis"]
        report["analyses"] = analyses

    # 1. reference runs
    ref_failed = 0
    for ex in wanted:
        for an in analyses:
            rec = run_reference(ex, examples[ex], an, out_dir / REFERENCE, args)
            report["runs"].append(rec)
            status = rec["status"]
            print(f"[parity] {REFERENCE:7s} {ex:10s} {an:9s} {status}" + (f" ({rec['seconds']}s)" if rec.get("seconds") else "")
                  + (f" -- {rec['note']}" if rec.get("note") else ""))
            if status == "failed":
                ref_failed += 1

    # reference self-consistency: the harness's p/q recomputation must reproduce the CLI's own file
    for ex in wanted:
        if "meme" not in analyses:
            break
        path = out_dir / REFERENCE / f"{ex}.meme.json"
        if not path.exists():
            continue
        pv, bh = _reference_stats()
        checks = _pq_given_lrt(load_json(path).get("sites") or [], pv, bh)
        report["self_checks"].append({"surface": REFERENCE, "example": ex, "analysis": "meme",
                                      "checks": [c.to_json() for c in checks]})

    # 2. other surfaces
    violations = 0
    missing = 0
    for surface in others:
        sdir = out_dir / surface
        present_any = False
        for ex in wanted:
            for an in analyses:
                ref_path = out_dir / REFERENCE / f"{ex}.{an}.json"
                got_path = sdir / f"{ex}.{an}.json"
                entry = {"surface": surface, "example": ex, "analysis": an,
                         "path": rel(got_path),
                         "status": None, "checks": [], "notes": []}
                if not ref_path.exists():
                    entry["status"] = "no-reference"
                    report["comparisons"].append(entry)
                    continue
                if not got_path.exists():
                    entry["status"] = "missing"
                    missing += 1
                    report["comparisons"].append(entry)
                    continue
                present_any = True
                ref, got = load_json(ref_path), load_json(got_path)
                if an == "meme":
                    checks, notes = compare_meme(ref, got)
                elif an == "busted":
                    checks, notes = compare_busted(ref, got)
                else:
                    checks, notes = compare_epistasis(ref, got, args.n_permutations)
                n_viol = sum(len(c.violations) for c in checks)
                violations += n_viol
                entry["status"] = "pass" if n_viol == 0 else "fail"
                entry["violations"] = n_viol
                entry["checks"] = [c.to_json() for c in checks]
                entry["notes"] = notes
                report["comparisons"].append(entry)
                print(f"[parity] {surface:7s} {ex:10s} {an:9s} {entry['status']} ({n_viol} violations)")
        if present_any:
            report["surfaces_present"].append(surface)
        else:
            report["surfaces_missing"].append(surface)
            report["notes"].append(f"surface '{surface}' has no files under {rel(sdir)}; "
                                   "nothing compared (the runner lives in veg/hyphaeon-app; see PARITY.md)")

    self_viol = sum(len(c.get("first_violations", [])) for sc in report["self_checks"] for c in sc["checks"])
    report["summary"] = {
        "reference_runs": len(report["runs"]),
        "reference_failed": ref_failed,
        "self_check_violations": self_viol,
        "comparisons": sum(1 for c in report["comparisons"] if c["status"] in ("pass", "fail")),
        "missing": missing,
        "violations": violations,
        "strict_missing": args.strict_missing,
    }
    ok = ref_failed == 0 and violations == 0 and self_viol == 0 and (missing == 0 or not args.strict_missing)
    report["summary"]["passed"] = ok

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print()
    for note in report["notes"]:
        print(f"[parity] note: {note}")
    s = report["summary"]
    print(f"[parity] reference runs: {s['reference_runs']} ({s['reference_failed']} failed); "
          f"self-check violations: {s['self_check_violations']}; comparisons: {s['comparisons']}; "
          f"missing: {s['missing']}; violations: {s['violations']}")
    print(f"[parity] report: {report_path}")
    print(f"[parity] {'PASS' if ok else 'FAIL'}")
    if ref_failed:
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
