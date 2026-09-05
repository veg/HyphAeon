#!/usr/bin/env python
"""
parity.py -- run the Python reference on the bundled examples and compare every other surface
against it, class by class.

WHY THIS FILE EXISTS

`hyphaeon/*.py` is the reference implementation of the methods; `js/` is a second implementation
of the same methods, consumed by runtime surfaces that live in the app repository
(veg/hyphaeon-app): the library under Node, the library in headless Chromium, and the MCP tool.
A second implementation drifts. Fixture replay (`js/test`) catches drift per function; this
script catches it end to end, on real alignments, at the level a user sees: the per-site table,
the gene-level omnibus record, the co-selection network, the DMS heatmap, the trait association.

The script has two halves and they are deliberately decoupled:

  1. It RUNS the Python CLI (`python -m hyphaeon.cli <analysis> ...`) on each requested example
     and writes `parity/python/<example>.<analysis>.json` -- the CLI's own `-o` JSON, unmodified.
     For the tree-free examples (PLAN.md D22: a tree with no branch lengths, or no tree at all,
     takes the TN93 path) it runs the same commands with `--use-tn93` into `parity/python-tn93/`.
  2. It READS `parity/<surface>/<example>.<analysis>.json` for every other requested surface
     and compares. It does not know how those files are produced; the runner that produces them
     lives in the app repository, because nothing in this repository runs a model outside torch
     (js/ is a library: no onnxruntime, no I/O). A surface whose directory is absent is reported
     as missing, never as a failure, unless `--strict-missing` is given.

SURFACES. `python` is the reference. `node`, `web` (alias `browser`: the e2e writes
`parity/browser/`, the integrator copies to `parity/web/`; either directory is read) and `mcp`
are the tree-based surfaces, compared against `parity/python/`. Each has a tree-free sibling
`<surface>-tn93` (`node-tn93`, `web-tn93`, `mcp-tn93`), compared against `parity/python-tn93/`.
A tree-based surface has no file for a tree-free example by construction (the runner writes it to
the sibling directory), so that pair is reported as `redirected`, not as missing.

TOLERANCE CLASSES are PLAN.md 5.4 (app repository) as refined by the phase documents
(PHASE0.md, PHASE1A.md, PHASE2A.md, PHASE3A.md here; PHASE3.md there), restated in PARITY.md.
The constants below are cited there; the non-obvious decisions:

  - graph class: 1e-5 * max(1, |lrt|) for anything that came out of the network. PLAN 5.4 measured
    in Phase 0 that fp32 torch paths themselves differ by 6.7e-6, so the 1e-6 of the first harness
    was unreachable by construction and PHASE3.md re-evaluated every comparison by hand. The class
    is applied to the per-site LRT, the edge/sector/plasticity LRT fields, `cesi` (a function of
    the LRTs) and the phenotype score tracks. Sums over L such values (`omnibus_lrt`,
    `total_selection_energy`) take L * 1e-5. A DMS delta is `mutant_lrt - baseline_lrt`
    (epistasis.py:560), a difference of two graph-class values, so `mutant_deltas`, the
    mean/max/min delta and `intrinsic_plasticity` (mean |delta|) take the sum of both operands'
    budgets, 1e-5 * (max(1, |baseline|) + max(1, |mutant|)); a delta of 0.03 between two LRTs of
    5 is otherwise held to 1e-5 while its inputs may each move 5e-5 (measured up to 4.1x the
    single-value bound on RHO before this rule).
  - p_value / q_value (meme) are checked at 1e-9 "given equal LRT": the reference p-value function
    is evaluated on the SURFACE's own LRT, and the surface's q against `benjamini_hochberg` of the
    SURFACE's own p, both rounded to float32 because the reference CLI writes them as float32
    (`cli.py`, cmd_meme). Evaluating the special function at the surface's input isolates the
    special function, which is what the 1e-9 class is for; a graph-class drift in the LRT would
    otherwise fail every site for a reason already reported under `hyphaeon_lrt`.
  - edge `similarity`, `p_val`, `hyper_p`, `fdr_q` (and the phenotype pair `p_value`/`q_value`)
    are 1e-6 absolute: the cosine is float32 arithmetic on float32 attributions whose BLAS
    accumulation order nobody reproduces (PHASE2A.md gap 4), and the hypergeometric/BH values
    downstream inherit that.
  - statistical class (`p_perm`, `gene_p_value_perm`, `p_assoc_perm`): two Monte Carlo estimates
    with their own seeds and their OWN B. The bound is 3 sigma of the difference,
    3 * sqrt(p(1-p) * (1/B_ref + 1/B_surface)), p floored at 1/min(B) so p = 0 keeps a width.
    B_surface is read from the file (`provenance.options.permutations`), else from the runner's
    `summary.json` beside it, else assumed equal to `--n-permutations` (noted). The browser
    surface runs B = 1,000 (its default) against a reference at B = 10,000.
  - null moments (`null_coherence_mean/_std/_95`) are within 2% relative only when BOTH sides
    ran B >= 10,000: PHASE2A.md measured the standard error of the std estimate at B = 1,000 as
    about 2.2% relative, so at that B a 3-5% excursion is the estimator, not a defect. Below
    10,000 the moments are reported as `informational` and never count as violations.
  - BUSTED neural-head fields (`predicted_gene_lrt`, `selection_probability`,
    `synonymous_rate_variation`, `rate_distributions.omega_3`, `proportion_1..3`,
    `positive_selection_detected`) are SKIPPED with a note: `model.safetensors` lacks 11
    `BustedMultiTaskHead` parameters and `cmd_busted` loads the head unseeded (PHASE0.md,
    PHASE1A.md gap 4), so the reference draws them anew on every run; `busted_head.onnx` is one
    seeded draw. Nothing can be at parity with an unseeded reference.
  - a surface file whose taxon count differs from the reference (a taxon cap: the browser's default
    is 256, the fixture and the CLI run all 655 RHO taxa) analysed a different taxon set; nothing
    below the counts can be aligned, so it is `incomparable` with the counts in the note, never
    silently compared. `--strict-missing` makes missing and incomparable fail the run.
  - epistasis `plasticity` absent from a surface (the browser runs its DMS as a separate analysis
    with a work budget) is a note, not a violation; the DMS comparator covers those fields.

Exit codes: 0 when every check is within its class (missing, redirected and incomparable are
reported, not failures, unless --strict-missing); 1 otherwise; 2 a reference run failed or the
arguments were unusable.
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
TN93 = "-tn93"                      # suffix of a tree-free surface directory (D22)
BASE_SURFACES = ("python", "node", "web", "mcp")
SURFACE_ALIASES = {"browser": "web"}   # the e2e writes parity/browser/; the surface is `web`
ANALYSES = ("meme", "busted", "epistasis", "dms", "phenotype")

# --- tolerance classes (PLAN.md 5.4; PARITY.md) ---------------------------------------------
TOL_GRAPH_REL = 1e-5      # graph class: 1e-5 * max(1, |value|) for anything out of the network
TOL_EIGEN = 1e-5          # MDS / eigendecomposition-derived quantities (spectral_coherence)
TOL_SPECIAL = 1e-9        # chi2 / BH on identical inputs (p, q given the surface's own LRT)
TOL_DERIVED = 1e-6        # float32 cosine networks and everything downstream of them; p-values
                          # downstream of graph-class inputs (ACAT, Simes, DMS p, phenotype site p)
STAT_SIGMAS = 3.0         # statistical class: |dp| <= 3 * sqrt(p(1-p) (1/B_ref + 1/B_got))
STAT_NULL_REL = 0.02      # statistical class: null moments within 2% relative ...
STAT_NULL_MIN_B = 10000   # ... only when both sides ran at least this many permutations
DEFAULT_B = 10000         # permutations per side for the statistical class
DEFAULT_SEED = 42         # D17: xoshiro256** on the JS side, default seed 42; --seed here

BUSTED_NEURAL = ("predicted_gene_lrt", "selection_probability", "synonymous_rate_variation",
                 "rate_distributions.omega_3", "rate_distributions.proportion_1",
                 "rate_distributions.proportion_2", "rate_distributions.proportion_3",
                 "positive_selection_detected")
BUSTED_NEURAL_NOTE = ("skipped: BustedMultiTaskHead output; model.safetensors lacks 11 head parameters and "
                      "cmd_busted loads it unseeded (PHASE1A.md gap 4), so the reference redraws this on every run")

# README Example 3 marine foreground for examples/RHO.fasta -- the argv the phenotype fixture and
# the app runner reproduce (fixtures/e2e/phenotype_RHO_marine_n_permutations_0.json).
RHO_MARINE_FOREGROUND = "turTru,balMus,balPhys,orcOrc,delDelp,phyCat,phoVit,halGryp,mirLeo,zalCali,odoRos"


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
def tree_has_branch_lengths(nwk: Path) -> bool:
    """D22's predicate: a usable tree carries at least one `:<number>` branch length."""
    return re.search(r":\s*-?\d", nwk.read_text()) is not None


def discover_examples() -> dict[str, dict]:
    """Every alignment under examples/ with its optional Newick tree.

    RHO.fasta carries its tree embedded (README: 'Embedded / Auto'), so a missing .nwk means
    'let the CLI extract it', not 'skip'. camelid.nwk and HIV1_RT.nwk carry a topology and no
    branch lengths, so under D22 every runtime surface runs them tree-free; the reference's
    tree-based run is still made (it is what the CLI does with `-t`), but the pair the surfaces
    are compared against is the `--use-tn93` run.
    """
    out = {}
    for fasta in sorted(EXAMPLES_DIR.glob("*.fasta")):
        name = fasta.stem
        nwk = EXAMPLES_DIR / f"{name}.nwk"
        tree = nwk if nwk.exists() else None
        out[name] = {"alignment": fasta, "tree": tree,
                     "tree_free": tree is not None and not tree_has_branch_lengths(tree)}
    return out


def path_without_tn93_binary() -> str:
    """PATH with every directory holding a `tn93` executable removed.

    `dataset.compute_tn93_distance_matrix` prefers the compiled binary; the fixtures and the JS
    port pin the Python `tn93` package path (gen_fixtures.py; the two measured identical), so the
    reference's tree-free runs are made the same way.
    """
    kept = []
    for d in os.environ.get("PATH", "").split(os.pathsep):
        cand = os.path.join(d, "tn93") if d else "tn93"
        if d and os.path.isfile(cand) and os.access(cand, os.X_OK):
            continue
        kept.append(d)
    return os.pathsep.join(kept)


# --- running the reference -----------------------------------------------------------------
def cli_has_flag(analysis: str, flag: str) -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "hyphaeon.cli", analysis, "--help"],
        cwd=REPO, capture_output=True, text=True,
    )
    # Match the option token itself; 'seed' also appears in prose ("clique seed size").
    return re.search(rf"(^|\s){re.escape(flag)}(\b|[=\s])", proc.stdout) is not None


def reference_command(example: str, spec: dict, analysis: str, out_json: Path, args, tn93: bool) -> list[str]:
    cmd = [sys.executable, "-m", "hyphaeon.cli", analysis,
           "-a", str(spec["alignment"].relative_to(REPO)), "--cpu", "-o", str(out_json)]
    if tn93:
        cmd.append("--use-tn93")
    elif spec["tree"] is not None:
        cmd += ["-t", str(spec["tree"].relative_to(REPO))]
    if args.weights:
        cmd += ["-w", args.weights]
    if analysis == "epistasis":
        cmd += ["--n-permutations", str(args.n_permutations), "--seed", str(args.seed)]
    if analysis == "phenotype":
        cmd += ["-fg", args.phenotype_fg, "--n-permutations", str(args.n_permutations),
                "--permulations", str(args.permulations), "--seed", str(args.seed)]
    return cmd


def run_reference(example: str, spec: dict, analysis: str, out_dir: Path, args, tn93: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{example}.{analysis}.json"
    log = out_dir / f"{example}.{analysis}.log"
    surface = REFERENCE + (TN93 if tn93 else "")
    record = {"surface": surface, "example": example, "analysis": analysis,
              "path": rel(out_json), "status": "ok", "note": None, "seconds": None}

    if args.no_run or (args.reuse and out_json.exists()):
        if out_json.exists():
            record["note"] = "reused existing file (--no-run)" if args.no_run else "reused existing file (--reuse)"
        else:
            record["status"] = "missing"
            record["note"] = "no existing file and --no-run given"
        return record

    cmd = reference_command(example, spec, analysis, out_json, args, tn93)
    record["command"] = " ".join(cmd)
    env = dict(os.environ)
    if tn93:
        env["PATH"] = path_without_tn93_binary()
        record["note"] = "tn93 binary hidden from PATH: the Python tn93 package computes the distances, as the fixtures and the port do"

    t0 = time.time()
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, text=True, env=env)
    record["seconds"] = round(time.time() - t0, 3)
    if proc.returncode != 0 or not out_json.exists():
        record["status"] = "failed"
        record["note"] = f"exit {proc.returncode}; see {rel(log)}"
    return record


# --- comparison primitives -----------------------------------------------------------------
class Check:
    """One field (or one keyed collection of a field) compared under one class.

    Classes: exact | graph (1e-5 * max(1, |ref|)) | graph-delta (per-item tol from delta_tol) |
    tolerance (absolute `tol`) | special | statistical | informational (reported, never a
    violation) | skipped (not compared, see note).
    """

    def __init__(self, field: str, cls: str, tol: float | None = None, note: str | None = None):
        self.field = field
        self.cls = cls
        self.tol = tol
        self.n = 0
        self.max_abs_diff = 0.0
        self.max_ratio = 0.0          # worst |diff| / tol, the distance to the class boundary
        self.violations: list[dict] = []
        self.excursions: list[dict] = []   # informational only
        self.note = note

    def _tol_for(self, ref, tol):
        if self.cls == "graph":
            return TOL_GRAPH_REL * max(1.0, abs(float(ref)))
        return self.tol if tol is None else tol

    def add(self, key, ref, got, tol: float | None = None):
        if self.cls == "skipped":
            self.n += 1
            return
        self.n += 1
        if self.cls == "exact":
            if not _equal(ref, got):
                self.violations.append({"key": key, "ref": ref, "got": got})
            return
        if ref is None and got is None:
            return
        if ref is None or got is None:
            self.violations.append({"key": key, "ref": ref, "got": got, "reason": "null on one side"})
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
        tol = self._tol_for(ref, tol)
        self.max_abs_diff = max(self.max_abs_diff, diff)
        if tol and tol > 0:
            self.max_ratio = max(self.max_ratio, diff / tol)
        if diff > tol:
            item = {"key": key, "ref": ref, "got": got, "abs_diff": diff, "tol": tol}
            (self.excursions if self.cls == "informational" else self.violations).append(item)

    def to_json(self) -> dict:
        d = {"field": self.field, "class": self.cls, "n": self.n, "violations": len(self.violations)}
        if self.cls == "graph":
            d["tol"] = f"{TOL_GRAPH_REL:g} * max(1, |ref|)"
        elif self.cls == "graph-delta":
            d["tol"] = GRAPH_DELTA_RULE
        elif self.cls not in ("exact", "skipped"):
            d["tol"] = self.tol
        if self.cls not in ("exact", "skipped"):
            d["max_abs_diff"] = self.max_abs_diff
            d["max_ratio_to_tol"] = round(self.max_ratio, 4)
        if self.note:
            d["note"] = self.note
        if self.violations:
            d["first_violations"] = self.violations[:20]
        if self.excursions:
            d["excursions"] = len(self.excursions)
            d["first_excursions"] = self.excursions[:20]
        return d


def _equal(a, b) -> bool:
    """Exact class: equality after canonicalisation (int 1 == float 1.0; lists elementwise)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    return a == b


def stat_tol(p_ref: float, b_ref: int, b_got: int) -> float:
    """3 sigma of the difference of two independent Monte Carlo estimates at B_ref and B_got."""
    b_min = max(1, min(b_ref, b_got))
    p = min(max(float(p_ref), 1.0 / b_min), 1.0 - 1.0 / b_min)
    return STAT_SIGMAS * math.sqrt(p * (1.0 - p) * (1.0 / b_ref + 1.0 / b_got))


def _get(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def scalar_checks(ref: dict, got: dict, spec: list[tuple], scale: float = 1.0) -> list[Check]:
    """spec entries: (field, class, tol) with class in exact|graph|tolerance|sum|skipped."""
    checks = []
    for field, cls, tol in spec:
        note = BUSTED_NEURAL_NOTE if field in BUSTED_NEURAL and cls == "skipped" else None
        if cls == "sum":
            c = Check(field, "tolerance", tol * scale, note=f"sum over L={int(scale)} graph-class values: L * {tol:g}")
        else:
            c = Check(field, cls, tol, note=note)
        r, g = _get(ref, field), _get(got, field)
        if r is None and g is None and cls != "skipped":
            c.note = "absent on both sides"
            checks.append(c)
            continue
        c.add(field, r, g)
        checks.append(c)
    return checks


def _keyed_collection(items: list[dict], key_fields: tuple[str, ...]) -> tuple[list, dict]:
    keys = [tuple(it.get(k) for k in key_fields) for it in items]
    return keys, dict(zip(keys, items))


def _order_check(name: str, r_keys: list, g_keys: list) -> Check:
    c = Check(name, "exact")
    c.n = max(len(r_keys), len(g_keys))
    if r_keys != g_keys:
        first = next((i for i, (a, b) in enumerate(zip(r_keys, g_keys)) if a != b), min(len(r_keys), len(g_keys)))
        c.violations.append({"key": f"index {first}",
                             "ref": r_keys[first] if first < len(r_keys) else None,
                             "got": g_keys[first] if first < len(g_keys) else None,
                             "ref_len": len(r_keys), "got_len": len(g_keys),
                             "missing_in_surface": sorted(set(r_keys) - set(g_keys), key=str)[:10],
                             "extra_in_surface": sorted(set(g_keys) - set(r_keys), key=str)[:10]})
    return c


def _keyed_check(name: str, cls: str, tol, keys: list, ref: dict, got: dict, note: str | None = None) -> Check:
    c = Check(name, cls, tol, note=note)
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


def _keyed_checks(keys, r_map, g_map, exact=(), graph=(), abs_tol=(), tol=TOL_DERIVED) -> list[Check]:
    out = []
    for name in exact:
        out.append(_keyed_check(name, "exact", None, keys, r_map, g_map))
    for name in graph:
        out.append(_keyed_check(name, "graph", None, keys, r_map, g_map))
    for name in abs_tol:
        out.append(_keyed_check(name, "tolerance", tol, keys, r_map, g_map))
    return out


def _taxa_gate(ref: dict, got: dict, field: str) -> str | None:
    """None when the two files analysed the same taxon set; otherwise the incomparable reason."""
    r, g = ref.get(field), got.get(field)
    if r is None or g is None or r == g:
        return None
    return (f"{field} differs (reference {r}, surface {g}): the surface analysed a different taxon set "
            "(a taxon cap or a different duplicate policy), so nothing below the counts can be aligned")


# --- meme ----------------------------------------------------------------------------------
def compare_meme(ref: dict, got: dict, ctx: dict) -> tuple[list[Check], list[str], str | None]:
    pvals_from_lrt_meme, benjamini_hochberg = _reference_stats()
    gate = _taxa_gate(ref, got, "taxa_count")
    if gate:
        return [], [], gate
    notes: list[str] = []
    checks = scalar_checks(ref, got, [("taxa_count", "exact", None), ("codon_count", "exact", None)])

    rs, gs = ref.get("sites") or [], got.get("sites") or []
    order = _order_check("site order", [s.get("site") for s in rs], [s.get("site") for s in gs])
    checks.append(order)
    if order.violations:
        notes.append("site order differs; per-site fields not compared")
        return checks, notes, None

    inv = Check("is_invariable", "exact")
    lrt = Check("hyphaeon_lrt", "graph")
    for r, g in zip(rs, gs):
        inv.add(r["site"], r.get("is_invariable"), g.get("is_invariable"))
        lrt.add(r["site"], r.get("hyphaeon_lrt"), g.get("hyphaeon_lrt"))
    checks += [inv, lrt]
    checks += _pq_given_lrt(gs, pvals_from_lrt_meme, benjamini_hochberg)
    return checks, notes, None


def _pq_given_lrt(sites: list[dict], pvals_from_lrt_meme, benjamini_hochberg) -> list[Check]:
    """p and q at 1e-9 given the surface's own LRT (see the header)."""
    p = Check("p_value", "special", TOL_SPECIAL)
    q = Check("q_value", "special", TOL_SPECIAL)
    p.note = "expected = float32(pvals_from_lrt_meme(surface lrt)); surface p rounded to float32"
    q.note = "expected = float32(benjamini_hochberg(float32(surface p))); surface q rounded to float32"
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
    ("rate_distributions.omega_1", "exact", None),       # constants 0.10 / 1.00
    ("rate_distributions.omega_2", "exact", None),
    ("omnibus_lrt", "sum", TOL_GRAPH_REL),               # sum of L graph-class values
    ("total_selection_energy", "sum", TOL_GRAPH_REL),
    ("p_value_acat", "tolerance", TOL_DERIVED),
    ("p_value_simes", "tolerance", TOL_DERIVED),
] + [(f, "skipped", None) for f in BUSTED_NEURAL]


def compare_busted(ref: dict, got: dict, ctx: dict) -> tuple[list[Check], list[str], str | None]:
    # The CLI writes a list for batch runs and a dict for a single alignment; parity runs one.
    if isinstance(ref, list):
        ref = ref[0]
    if isinstance(got, list):
        got = got[0]
    gate = _taxa_gate(ref, got, "taxa")
    if gate:
        return [], [], gate
    L = int(ref.get("sites") or 1)
    notes = [f"{len(BUSTED_NEURAL)} neural-head fields skipped (unseeded upstream); the statistical half is compared"]
    return scalar_checks(ref, got, BUSTED_SPEC, scale=L), notes, None


# --- shared collections: edges, sectors, plasticity ----------------------------------------
EDGE_EXACT = ("ref_u", "ref_v", "shared_taxa", "shared_branches")
EDGE_GRAPH = ("lrt_u", "lrt_v", "cesi")
EDGE_ABS = ("similarity", "p_val", "hyper_p", "fdr_q")
SECTOR_EXACT = ("size", "sites", "shared_taxa", "shared_branches", "pars_signature", "consensus_signature")
SECTOR_MOMENTS = ("null_coherence_mean", "null_coherence_std", "null_coherence_95")
PLAST_EXACT = ("wt_aa",)
PLAST_DELTAS = ("intrinsic_plasticity", "mean_delta_lrt", "max_delta_lrt", "min_delta_lrt")
GRAPH_DELTA_RULE = f"{TOL_GRAPH_REL:g} * (max(1, |baseline|) + max(1, |mutant|)): a difference of two graph-class LRTs"


def delta_tol(baseline: float, *mutants: float) -> float:
    """Budget of `mutant_lrt - baseline_lrt` (epistasis.py:560): each operand carries the graph class."""
    worst = max((max(1.0, abs(float(m))) for m in mutants), default=1.0)
    return TOL_GRAPH_REL * (max(1.0, abs(float(baseline))) + worst)


def compare_edges(ref_edges: list, got_edges: list, label: str, abs_fields=EDGE_ABS, exact=EDGE_EXACT) -> list[Check]:
    r_keys, r_map = _keyed_collection(ref_edges or [], ("site_u", "site_v"))
    g_keys, g_map = _keyed_collection(got_edges or [], ("site_u", "site_v"))
    checks = [_order_check(f"{label} (site_u, site_v) set and order", r_keys, g_keys)]
    checks += _keyed_checks(r_keys, r_map, g_map, exact=exact, graph=EDGE_GRAPH, abs_tol=abs_fields)
    return checks


def compare_sectors(ref_sectors: list, got_sectors: list, label: str, b_ref: int, b_got: int | None,
                    notes: list[str]) -> list[Check]:
    r_keys, r_map = _keyed_collection(ref_sectors or [], ("sector_id",))
    g_keys, g_map = _keyed_collection(got_sectors or [], ("sector_id",))
    checks = [_order_check(f"{label} sector_id order", r_keys, g_keys)]
    checks += _keyed_checks(r_keys, r_map, g_map, exact=SECTOR_EXACT, graph=("mean_lrt",))
    checks.append(_keyed_check("isotropic_baseline", "tolerance", 1e-12, r_keys, r_map, g_map, note="1/K"))
    checks.append(_keyed_check("spectral_coherence", "tolerance", TOL_EIGEN, r_keys, r_map, g_map,
                               note="eigenvalue ratio: the MDS/eigen class"))

    have_ref = any("p_perm" in r_map[k] for k in r_keys)
    have_got = any("p_perm" in g_map.get(k, {}) for k in r_keys)
    if not (r_keys and have_ref and have_got):
        why = "no sectors" if not r_keys else ("absent in reference" if not have_ref else "absent in surface")
        notes.append(f"{label}: statistical class (p_perm, null_coherence_*) skipped: {why}")
        return checks

    if b_got is None:
        b_got = b_ref
        notes.append(f"{label}: surface B not recorded; assumed equal to the reference's B={b_ref}")
    pp = Check("p_perm", "statistical")
    pp.note = (f"|dp| <= {STAT_SIGMAS:g}*sqrt(p(1-p)(1/B_ref + 1/B_surface)), B_ref={b_ref}, "
               f"B_surface={b_got}, p floored at 1/min(B)")
    for k in r_keys:
        r, g = r_map[k], g_map.get(k)
        if g is None or "p_perm" not in r or "p_perm" not in g:
            continue
        pp.add(k[0], r["p_perm"], g["p_perm"], tol=stat_tol(r["p_perm"], b_ref, b_got))
    checks.append(pp)

    moments_enforced = min(b_ref, b_got) >= STAT_NULL_MIN_B
    for name in SECTOR_MOMENTS:
        mc = Check(name, "statistical" if moments_enforced else "informational")
        mc.note = (f"relative difference <= {STAT_NULL_REL:g}" if moments_enforced else
                   f"relative difference vs {STAT_NULL_REL:g}, informational: min(B_ref, B_surface)={min(b_ref, b_got)} "
                   f"< {STAT_NULL_MIN_B} (PHASE2A.md: the std estimate's own SE is ~2.2% at B=1000)")
        for k in r_keys:
            r, g = r_map[k], g_map.get(k)
            if g is None or name not in r or name not in g:
                continue
            mc.add(k[0], r[name], g[name], tol=STAT_NULL_REL * max(abs(float(r[name])), 1e-12))
        checks.append(mc)
    if not moments_enforced:
        notes.append(f"{label}: null_coherence moments informational (B_surface={b_got}, B_ref={b_ref}; "
                     f"the 2% class needs both >= {STAT_NULL_MIN_B})")
    return checks


def compare_plasticity(ref_pl: list, got_pl: list, label: str, notes: list[str], surface_optional: bool) -> list[Check]:
    r_keys, r_map = _keyed_collection(ref_pl or [], ("site",))
    g_keys, g_map = _keyed_collection(got_pl or [], ("site",))
    if not r_keys and not g_keys:
        notes.append(f"{label}: absent on both sides")
        return []
    if r_keys and not g_keys and surface_optional:
        notes.append(f"{label}: absent in surface (the runtime runs the sector DMS as its own analysis, "
                     "compared under `dms`); reference has {n} sites".replace("{n}", str(len(r_keys))))
        return []
    checks = [_order_check(f"{label} site order", r_keys, g_keys)]
    checks += _keyed_checks(r_keys, r_map, g_map, exact=PLAST_EXACT, graph=("baseline_lrt",))
    checks.append(_keyed_check("p_value", "tolerance", TOL_DERIVED, r_keys, r_map, g_map,
                               note="Self-Liang p of a graph-class LRT"))
    # Every other field is a difference of two forward passes (epistasis.py:560-575): delta = mutant
    # LRT - baseline LRT, intrinsic_plasticity = mean |delta|, mean/max/min over the 19 deltas. Each
    # operand carries the graph class, so the budget is the sum of both operands' budgets, the
    # mutant taken as the largest one behind the statistic.
    summary_checks = {name: Check(name, "graph-delta") for name in PLAST_DELTAS}
    md = Check("mutant_deltas", "graph-delta")
    for k in r_keys:
        r, g = r_map[k], g_map.get(k)
        if g is None:
            continue
        b = float(r.get("baseline_lrt") or 0.0)
        rd, gd = r.get("mutant_deltas") or {}, g.get("mutant_deltas") or {}
        mutants = [b + float(v) for v in rd.values()]
        site_tol = delta_tol(b, *mutants)
        for name, c in summary_checks.items():
            if name in r or name in g:
                c.add(k[0], r.get(name), g.get(name), tol=site_tol)
        if set(rd) != set(gd):
            md.violations.append({"key": k[0], "ref": sorted(rd), "got": sorted(gd), "reason": "mutant set differs"})
            continue
        for aa in rd:
            md.add((k[0], aa), rd[aa], gd[aa], tol=delta_tol(b, b + float(rd[aa])))
    checks += list(summary_checks.values()) + [md]
    return checks


# --- epistasis -----------------------------------------------------------------------------
EPI_TOP = [
    ("taxa_count", "exact", None),
    ("codon_count", "exact", None),
    ("evaluated_taxa", "exact", None),
    ("coselection_edges_count", "exact", None),
    ("discovered_sectors_count", "exact", None),
]


def compare_epistasis(ref: dict, got: dict, ctx: dict) -> tuple[list[Check], list[str], str | None]:
    gate = _taxa_gate(ref, got, "taxa_count")
    if gate:
        return [], [], gate
    notes: list[str] = []
    checks = scalar_checks(ref, got, EPI_TOP)
    checks += compare_edges(ref.get("edges"), got.get("edges"), "edges")
    checks += compare_sectors(ref.get("sectors"), got.get("sectors"), "sectors", ctx["b_ref"], ctx["b_got"], notes)
    checks += compare_plasticity(ref.get("plasticity"), got.get("plasticity"), "plasticity", notes, surface_optional=True)
    notes.append("coselection_edges / epistatic_sectors / selection_dms_plasticity are the writer's copies of "
                 "edges / sectors / plasticity and are not compared twice")
    return checks, notes, None


# --- dms -----------------------------------------------------------------------------------
DMS_TOP = [
    ("taxa_count", "exact", None),
    ("codon_count", "exact", None),
    ("focal_taxon", "exact", None),
    ("total_mutations", "exact", None),
]


def compare_dms(ref: dict, got: dict, ctx: dict) -> tuple[list[Check], list[str], str | None]:
    gate = _taxa_gate(ref, got, "taxa_count")
    if gate:
        return [], [], gate
    notes: list[str] = []
    checks = scalar_checks(ref, got, DMS_TOP)
    checks += compare_plasticity(ref.get("plasticity"), got.get("plasticity"), "plasticity", notes, surface_optional=False)
    notes.append("selection_dms_plasticity is the writer's copy of plasticity and is not compared twice")
    return checks, notes, None


# --- phenotype -----------------------------------------------------------------------------
PHENO_TOP = [
    ("taxa_count", "exact", None),
    ("codon_count", "exact", None),
    ("phenotype_meta", "exact", None),
    ("compact_pars_signature", "exact", None),
    ("permulations_count", "exact", None),
    ("significant_sites_count", "exact", None),
    ("coselection_pairs_count", "exact", None),
    ("trait_sectors_count", "exact", None),
    ("spectral_energy", "tolerance", TOL_DERIVED),
    ("norm_spectral_ratio", "tolerance", TOL_DERIVED),
    ("max_assoc", "tolerance", TOL_DERIVED),
    ("p_evd_length_adjusted", "tolerance", TOL_DERIVED),
    ("score_track_b", "tolerance", TOL_DERIVED),
    ("score_track_a", "graph", None),            # sqrt(LRT)-weighted, a function of the LRTs
    ("dual_track_composite", "graph", None),
]
PHENO_SITE_EXACT = ("ref_aa", "derived_aa")
PHENO_SITE_GRAPH = ("hyphaeon_lrt",)
PHENO_SITE_ABS = ("p_lrt", "attribution_norm", "fg_mean_attn", "bg_mean_attn", "association_rho",
                  "p_value", "p_assoc", "p_assoc_parametric", "score", "q_value",
                  "foreground_freq_pct", "background_freq_pct")
PHENO_PAIR_EXACT = ("ref_u", "ref_v", "shared_branches")
PHENO_PAIR_ABS = ("similarity", "p_value", "q_value")


def compare_phenotype(ref: dict, got: dict, ctx: dict) -> tuple[list[Check], list[str], str | None]:
    gate = _taxa_gate(ref, got, "taxa_count")
    if gate:
        return [], [], gate
    notes: list[str] = []
    checks = scalar_checks(ref, got, PHENO_TOP)

    # gene-level permulation p: null on both sides when permulations were 0; statistical otherwise
    p_ref, p_got = ref.get("gene_p_value_perm"), got.get("gene_p_value_perm")
    m_ref, m_got = int(ref.get("permulations_count") or 0), int(got.get("permulations_count") or 0)
    gp = Check("gene_p_value_perm", "exact" if (p_ref is None or p_got is None) else "statistical")
    if gp.cls == "exact":
        gp.note = "null on both sides (permulations 0)" if p_ref is None and p_got is None else "null on one side"
        gp.add("gene", p_ref, p_got)
    else:
        gp.note = f"B = permulations_count: reference {m_ref}, surface {m_got}"
        gp.add("gene", p_ref, p_got, tol=stat_tol(p_ref, max(1, m_ref), max(1, m_got)))
    checks.append(gp)

    # site table, in the file's (score-descending) order
    r_keys, r_map = _keyed_collection(ref.get("sites") or [], ("site",))
    g_keys, g_map = _keyed_collection(got.get("sites") or [], ("site",))
    checks.append(_order_check("sites site order", r_keys, g_keys))
    checks += _keyed_checks(r_keys, r_map, g_map, exact=PHENO_SITE_EXACT, graph=PHENO_SITE_GRAPH, abs_tol=PHENO_SITE_ABS)
    perm = Check("p_assoc_perm", "statistical" if m_ref and m_got else "exact")
    perm.note = (f"B = permulations_count: reference {m_ref}, surface {m_got}" if perm.cls == "statistical"
                 else "null on both sides (permulations 0)")
    for k in r_keys:
        r, g = r_map[k], g_map.get(k)
        if g is None:
            continue
        rv, gv = r.get("p_assoc_perm"), g.get("p_assoc_perm")
        if perm.cls == "statistical" and rv is not None and gv is not None:
            perm.add(k[0], rv, gv, tol=stat_tol(rv, m_ref, m_got))
        else:
            perm.add(k[0], rv, gv)
    checks.append(perm)

    checks += compare_edges(ref.get("coselection_pairs"), got.get("coselection_pairs"), "coselection_pairs",
                            abs_fields=PHENO_PAIR_ABS, exact=PHENO_PAIR_EXACT)
    checks += compare_sectors(ref.get("trait_sectors"), got.get("trait_sectors"), "trait_sectors",
                              ctx["b_ref"], ctx["b_got"], notes)
    return checks, notes, None


COMPARATORS = {"meme": compare_meme, "busted": compare_busted, "epistasis": compare_epistasis,
               "dms": compare_dms, "phenotype": compare_phenotype}


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


def surface_B(doc, sdir: Path) -> int | None:
    """The permutation count a surface file was produced with, if it says so."""
    prov = doc.get("provenance") if isinstance(doc, dict) else None
    opts = (prov or {}).get("options") or {}
    for key in ("permutations", "n_permutations", "nPermutations"):
        if isinstance(opts.get(key), (int, float)):
            return int(opts[key])
    summary = sdir / "summary.json"
    if summary.exists():
        try:
            s = load_json(summary)
            for key in ("n_permutations", "permutations"):
                if isinstance(s.get(key), (int, float)):
                    return int(s[key])
        except (OSError, ValueError):
            pass
    return None


def normalise_surface(name: str) -> str:
    base, suffix = (name[: -len(TN93)], TN93) if name.endswith(TN93) else (name, "")
    base = SURFACE_ALIASES.get(base, base)
    return base + suffix


def surface_dir(out_dir: Path, surface: str) -> Path:
    """The directory holding a surface's files; `web` falls back to the e2e's `browser` name."""
    base, suffix = (surface[: -len(TN93)], TN93) if surface.endswith(TN93) else (surface, "")
    candidates = [base] + [alias for alias, target in SURFACE_ALIASES.items() if target == base]
    for cand in candidates:
        d = out_dir / (cand + suffix)
        if d.exists():
            return d
    return out_dir / (base + suffix)


def parse_list(value: str, universe: tuple[str, ...] | None, name: str) -> list[str]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    if universe is not None:
        bad = [v for v in items if v not in universe]
        if bad:
            print(f"[parity] unknown {name}: {', '.join(bad)} (choose from {', '.join(universe)})", file=sys.stderr)
            sys.exit(2)
    return items


def wanted_pairs(examples: list[str], analyses: list[str], args) -> list[tuple[str, str]]:
    """(example, analysis) pairs: dms and phenotype only on their own example lists."""
    dms_ex = parse_list(args.dms_examples, None, "dms example")
    ph_ex = parse_list(args.phenotype_examples, None, "phenotype example")
    pairs = []
    for ex in examples:
        for an in analyses:
            if an == "dms" and ex not in dms_ex:
                continue
            if an == "phenotype" and ex not in ph_ex:
                continue
            pairs.append((ex, an))
    return pairs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", default="all",
                    help="comma-separated example names from examples/ (stem of the .fasta), or 'all' (default)")
    ap.add_argument("--surfaces", default="python,node,node-tn93",
                    help=f"comma-separated surfaces to compare; '{REFERENCE}' is the reference and is always run. "
                         f"Known: {', '.join(BASE_SURFACES)} and their tree-free siblings <surface>{TN93}; "
                         f"'browser' is an alias of 'web' (default: python,node,node{TN93})")
    ap.add_argument("--analyses", default=",".join(ANALYSES),
                    help=f"comma-separated analyses (default: {','.join(ANALYSES)})")
    ap.add_argument("--dms-examples", default="Smc6",
                    help="examples the dms analysis is run/compared on (19 x L forward passes; default Smc6)")
    ap.add_argument("--phenotype-examples", default="RHO",
                    help="examples the phenotype analysis is run/compared on (default RHO)")
    ap.add_argument("--phenotype-fg", default=RHO_MARINE_FOREGROUND,
                    help="foreground list passed to `phenotype -fg` (default: the README Example 3 marine list)")
    ap.add_argument("--permulations", type=int, default=0,
                    help="passed to `phenotype --permulations` (default 0: parametric p, as the fixture and the runner)")
    ap.add_argument("--tn93-examples", default="auto",
                    help="examples run tree-free for the <surface>-tn93 comparisons: 'auto' (default) = every example "
                         "whose .nwk has no branch lengths (D22) plus any example a requested -tn93 surface has files for; "
                         "'none'; or a comma-separated list")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="parity directory (default: <repo>/parity)")
    ap.add_argument("--weights", default=os.environ.get("HYPHAEON_WEIGHTS") or None,
                    help="passed to the CLI as -w (default: $HYPHAEON_WEIGHTS; unset means the CLI's own resolution)")
    ap.add_argument("--n-permutations", type=int, default=DEFAULT_B,
                    help=f"B for the reference's epistasis/phenotype sector null and for the statistical class (default {DEFAULT_B})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"seed passed to epistasis/phenotype --seed (default {DEFAULT_SEED})")
    ap.add_argument("--no-run", action="store_true", help="do not run the Python CLI; reuse parity/python*/*.json")
    ap.add_argument("--reuse", action="store_true", help="run the Python CLI only for reference files that do not exist yet")
    ap.add_argument("--strict-missing", action="store_true",
                    help="treat a missing or incomparable surface file as a violation (off: reported, exit 0)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out).resolve()
    examples = discover_examples()
    wanted = list(examples) if args.examples == "all" else parse_list(args.examples, tuple(examples), "example")
    surfaces = [normalise_surface(s) for s in parse_list(args.surfaces, None, "surface")]
    for s in surfaces:
        base = s[: -len(TN93)] if s.endswith(TN93) else s
        if base not in BASE_SURFACES:
            print(f"[parity] unknown surface: {s} (choose from {', '.join(BASE_SURFACES)}, "
                  f"their {TN93} siblings, or the alias 'browser')", file=sys.stderr)
            return 2
    if REFERENCE not in surfaces:
        surfaces.insert(0, REFERENCE)
    others = [s for s in surfaces if s != REFERENCE]
    tn93_surfaces = [s for s in others if s.endswith(TN93)]
    analyses = parse_list(args.analyses, ANALYSES, "analysis")

    # tree-free examples (D22)
    if args.tn93_examples == "none":
        tn93_examples: list[str] = []
    elif args.tn93_examples == "auto":
        tn93_examples = [ex for ex in wanted if examples[ex]["tree_free"]]
        for s in tn93_surfaces:
            d = surface_dir(out_dir, s)
            if d.exists():
                for f in d.glob("*.json"):
                    ex = f.name.split(".")[0]
                    if ex in wanted and ex not in tn93_examples:
                        tn93_examples.append(ex)
    else:
        tn93_examples = parse_list(args.tn93_examples, tuple(examples), "example")
    tn93_examples = [ex for ex in wanted if ex in tn93_examples]

    report = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": git_commit(),
        "reference": REFERENCE,
        "surfaces_requested": surfaces,
        "surfaces_present": [],
        "surfaces_missing": [],
        "surface_dirs": {},
        "examples": wanted,
        "tree_free_examples": tn93_examples,
        "analyses": analyses,
        "dms_examples": args.dms_examples,
        "phenotype_examples": args.phenotype_examples,
        "weights": args.weights,
        "n_permutations": args.n_permutations,
        "permulations": args.permulations,
        "seed": args.seed,
        "classes": {
            "exact": "equality after canonicalisation",
            "graph": f"{TOL_GRAPH_REL:g} * max(1, |ref|): LRT and everything computed from it through the network",
            "sum": f"L * {TOL_GRAPH_REL:g}: omnibus_lrt, total_selection_energy",
            "eigen": TOL_EIGEN,
            "special": f"{TOL_SPECIAL:g} given the surface's own LRT, after float32 rounding: meme p_value, q_value",
            "derived": f"{TOL_DERIVED:g} absolute: float32 cosine networks and p-values downstream of graph-class inputs",
            "statistical": {"sigmas": STAT_SIGMAS, "bound": "3*sqrt(p(1-p)(1/B_ref + 1/B_surface)), p floored at 1/min(B)",
                            "null_moment_rel": STAT_NULL_REL, "null_moments_enforced_from_B": STAT_NULL_MIN_B},
            "skipped": list(BUSTED_NEURAL),
        },
        "runs": [],
        "self_checks": [],
        "comparisons": [],
        "notes": [],
        "summary": {},
    }

    # epistasis / phenotype need a CLI seed to be reproducible on the Python side (D17)
    for an in ("epistasis", "phenotype"):
        if an in analyses and not args.no_run and not cli_has_flag(an, "--seed"):
            report["notes"].append(f"{an} skipped: the CLI has no --seed flag (PLAN.md section 7, item 7)")
            analyses = [a for a in analyses if a != an]
    report["analyses"] = analyses

    tree_pairs = wanted_pairs(wanted, analyses, args)
    tn93_pairs = wanted_pairs(tn93_examples, analyses, args) if tn93_surfaces else []

    # 1. reference runs
    ref_failed = 0
    for tn93, pairs in ((False, tree_pairs), (True, tn93_pairs)):
        ref_dir = out_dir / (REFERENCE + (TN93 if tn93 else ""))
        for ex, an in pairs:
            rec = run_reference(ex, examples[ex], an, ref_dir, args, tn93)
            report["runs"].append(rec)
            print(f"[parity] {rec['surface']:12s} {ex:10s} {an:9s} {rec['status']}"
                  + (f" ({rec['seconds']}s)" if rec.get("seconds") else "")
                  + (f" -- {rec['note']}" if rec.get("note") else ""))
            if rec["status"] == "failed":
                ref_failed += 1

    # reference self-consistency: the harness's p/q recomputation must reproduce the CLI's own file
    if "meme" in analyses:
        pv, bh = _reference_stats()
        for tn93 in (False, True):
            ref_dir = out_dir / (REFERENCE + (TN93 if tn93 else ""))
            for ex in (tn93_examples if tn93 else wanted):
                path = ref_dir / f"{ex}.meme.json"
                if not path.exists():
                    continue
                checks = _pq_given_lrt(load_json(path).get("sites") or [], pv, bh)
                report["self_checks"].append({"surface": REFERENCE + (TN93 if tn93 else ""), "example": ex,
                                              "analysis": "meme", "checks": [c.to_json() for c in checks]})

    # 2. other surfaces
    violations = 0
    missing = 0
    incomparable = 0
    redirected = 0
    informational = 0
    for surface in others:
        tn93 = surface.endswith(TN93)
        sdir = surface_dir(out_dir, surface)
        report["surface_dirs"][surface] = rel(sdir)
        ref_dir = out_dir / (REFERENCE + (TN93 if tn93 else ""))
        present_any = False
        for ex, an in (tn93_pairs if tn93 else tree_pairs):
            ref_path = ref_dir / f"{ex}.{an}.json"
            got_path = sdir / f"{ex}.{an}.json"
            entry = {"surface": surface, "example": ex, "analysis": an, "path": rel(got_path),
                     "status": None, "checks": [], "notes": []}
            if not got_path.exists():
                sibling = surface_dir(out_dir, surface + TN93) / f"{ex}.{an}.json"
                if not tn93 and ex in tn93_examples and sibling.exists():
                    entry["status"] = "redirected"
                    entry["notes"] = [f"tree-free example (D22): the run is {rel(sibling)}, compared under {surface}{TN93}"]
                    redirected += 1
                else:
                    entry["status"] = "missing"
                    missing += 1
                report["comparisons"].append(entry)
                continue
            present_any = True
            if not ref_path.exists():
                entry["status"] = "no-reference"
                entry["notes"] = [f"{rel(ref_path)} does not exist (reference run failed or was not requested)"]
                report["comparisons"].append(entry)
                print(f"[parity] {surface:12s} {ex:10s} {an:9s} no-reference")
                continue
            ref, got = load_json(ref_path), load_json(got_path)
            ctx = {"b_ref": args.n_permutations, "b_got": surface_B(got, sdir)}
            checks, notes, gate = COMPARATORS[an](ref, got, ctx)
            if gate:
                entry["status"] = "incomparable"
                entry["notes"] = [gate]
                incomparable += 1
                report["comparisons"].append(entry)
                print(f"[parity] {surface:12s} {ex:10s} {an:9s} incomparable -- {gate}")
                continue
            n_viol = sum(len(c.violations) for c in checks)
            n_info = sum(len(c.excursions) for c in checks)
            violations += n_viol
            informational += n_info
            entry["status"] = "pass" if n_viol == 0 else "fail"
            entry["violations"] = n_viol
            entry["informational_excursions"] = n_info
            entry["surface_B"] = ctx["b_got"]
            entry["checks"] = [c.to_json() for c in checks]
            entry["notes"] = notes
            worst = max((c for c in checks if c.cls not in ("exact", "skipped", "informational") and c.n),
                        key=lambda c: c.max_ratio, default=None)
            worst_s = f"; worst {worst.field} at {worst.max_ratio:.2f} of its bound" if worst else ""
            info_s = f"; {n_info} informational" if n_info else ""
            report["comparisons"].append(entry)
            print(f"[parity] {surface:12s} {ex:10s} {an:9s} {entry['status']} ({n_viol} violations{info_s}{worst_s})")
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
        "passed_comparisons": sum(1 for c in report["comparisons"] if c["status"] == "pass"),
        "failed_comparisons": sum(1 for c in report["comparisons"] if c["status"] == "fail"),
        "missing": missing,
        "redirected": redirected,
        "incomparable": incomparable,
        "violations": violations,
        "informational_excursions": informational,
        "strict_missing": args.strict_missing,
    }
    ok = (ref_failed == 0 and violations == 0 and self_viol == 0
          and ((missing == 0 and incomparable == 0) or not args.strict_missing))
    report["summary"]["passed"] = ok

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print()
    for note in report["notes"]:
        print(f"[parity] note: {note}")
    for c in report["comparisons"]:
        if c["status"] == "incomparable":
            print(f"[parity] incomparable: {c['surface']} {c['example']} {c['analysis']}: {c['notes'][0]}")
    s = report["summary"]
    print(f"[parity] reference runs: {s['reference_runs']} ({s['reference_failed']} failed); "
          f"self-check violations: {s['self_check_violations']}; comparisons: {s['comparisons']} "
          f"({s['passed_comparisons']} pass, {s['failed_comparisons']} fail); missing: {s['missing']}; "
          f"redirected: {s['redirected']}; incomparable: {s['incomparable']}; "
          f"violations: {s['violations']}; informational excursions: {s['informational_excursions']}")
    print(f"[parity] report: {report_path}")
    print(f"[parity] {'PASS' if ok else 'FAIL'}")
    if ref_failed:
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
