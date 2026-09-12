#!/usr/bin/env python
"""
scripts/gen_fixtures.py — per-function parity fixtures for the JavaScript port.

WHY THIS FILE EXISTS

The JavaScript library under js/ mirrors hyphaeon/*.py function by function.
End-to-end parity alone cannot say which of a dozen preprocessing or
statistics steps drifted, so every ported function gets its own fixture:
this script imports the Python function directly, runs it on the bundled
examples plus seeded synthetic inputs, and writes inputs and outputs as JSON
under fixtures/<module>/<function>.json. js/test replays those files in the
same CI run (PLAN.md 5.3 rule 2, 5.4, 7 item 2).

Fixture record shape (one JSON list per function):

    {"name": str, "inputs": {...}, "outputs": {...},
     "tolerance": "exact" | "1e-5" | "1e-6" | "1e-9" | "statistical",
     "tolerance_relative": true, "tolerance_floor": 1.0     (MDS cases only: 1e-5 * max(floor, |compared value|))
     "notes": str}

Arrays are nested lists; NaN and infinities are the strings "NaN",
"Infinity", "-Infinity" because JSON has no encoding for them. Each file is
kept under 2 MB; a case that would exceed that is downsampled and says so in
its notes.

Decisions and measurements behind the non-obvious choices:

  * Every model run is forced onto the CPU (torch.device('cpu')). MPS and CPU
    float32 kernels disagree at the 1e-6 level, and the ONNX runtime the port
    uses is a CPU reference too. Model-dependent cases carry tolerance 1e-5,
    the class PLAN.md 5.4 reserves for LRT and attention through ORT.
  * Synthetic inputs come from numpy.random.default_rng(SYNTH_SEED) and are
    written INTO the fixture, so a replay never needs to reproduce numpy's
    PCG64 stream. The only outputs that depend on a Python RNG are the
    permutation and permulation nulls, which are tolerance "statistical".
  * Timing fields in end-to-end outputs (runtime_sec, elapsed_seconds) are
    nulled so regenerated fixtures diff cleanly; absolute paths are replaced
    with basenames for the same reason.
  * examples/camelid.nwk has no branch lengths, so dataset.load_alignment_and_tree
    shells out to HyPhy (HKY85) when `hyphy` is on PATH. Any fixture that
    depends on that tree records the HyPhy version and captures the resulting
    distance matrix as an INPUT, so the function-level fixture is
    self-contained even though the e2e one is HyPhy-dependent.
  * The Python is ported as-is. Behaviour that looks like a bug is replicated
    and listed under "known_quirks" in fixtures/manifest.json rather than
    fixed here (PLAN.md 5.3 rule 3).

Regenerate from the repository root with the same environment CI uses:

    HYPHAEON_WEIGHTS=model.safetensors HF_HUB_OFFLINE=1 \
        python scripts/gen_fixtures.py

Options: --only <module> (repeatable), --skip-model, --skip-e2e.

  * `--only dates` needs neither weights nor torch: the four date parsers of
    temporal.py and dating.py are lifted out of the checked-out source by `ast`
    (load_date_reference) and executed in a namespace holding only re, datetime,
    numpy and pandas, because both modules import torch at module scope and the
    parsers use none of it. A run that touches no model-dependent module keeps
    the previous manifest's weights record rather than nulling it.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from importlib.metadata import version as importlib_version

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "fixtures"
EXAMPLES = REPO / "examples"
MODEL_PATH = Path(os.environ.get("HYPHAEON_WEIGHTS", str(REPO / "model.safetensors")))
MEME_CACHE = REPO / "model_eval" / "_cache"

SYNTH_SEED = 20260904          # seed for synthetic inputs (inputs are written into fixtures)
PERM_SEED = 42                 # the engine's own default rng_seed / seed
MAX_FILE_BYTES = 2 * 1024 * 1024

# README Example 3 marine foreground for examples/RHO.fasta (README.md, "Example 3").
RHO_MARINE_FOREGROUND = "turTru,balMus,balPhys,orcOrc,delDelp,phyCat,phoVit,halGryp,mirLeo,zalCali,odoRos"

sys.path.insert(0, str(REPO))

# --------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------

def jsonable(obj: Any) -> Any:
    """Recursively convert numpy / path / non-finite values into JSON-safe data."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return jsonable(float(obj))
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if isinstance(obj, Path):
        return obj.name
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    raise TypeError(f"cannot serialise {type(obj)!r}")


def case(name: str, inputs: Dict[str, Any], outputs: Dict[str, Any], tolerance: str, notes: str, **extra: Any) -> Dict[str, Any]:
    """One fixture case. `extra` keys are recorded after `notes`; the one in use is the MDS class's
    `tolerance_relative=True, tolerance_floor=1.0` (see mds_relative())."""
    assert tolerance in ("exact", "1e-5", "1e-6", "1e-9", "statistical"), tolerance
    return {"name": name, "inputs": jsonable(inputs), "outputs": jsonable(outputs), "tolerance": tolerance, "notes": notes, **jsonable(extra)}


# The MDS class is RELATIVE to the magnitude of the compared coordinate, floor 1: the reference forms
# B and runs eigh in float32, so its own rounding at a coordinate of magnitude m is ~m * 6e-8, and the
# unrescaled bat_oas1 distances give coordinates of magnitude ~60 where an absolute 1e-5 is below the
# reference's own ULP (PHASE1A.md fixture defect 1). For everything of order 1 or less (every rescaled
# input) the floor makes it the plain absolute 1e-5.
MDS_RELATIVE = {"tolerance_relative": True, "tolerance_floor": 1.0}


class Writer:
    def __init__(self) -> None:
        self.counts: Dict[str, Dict[str, int]] = {}
        self.bytes: Dict[str, int] = {}

    def write(self, module: str, function: str, cases: List[Dict[str, Any]]) -> Path:
        out_dir = FIXTURES / module
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{function}.json"
        text = json.dumps(cases, indent=1, allow_nan=False) + "\n"
        size = len(text.encode("utf-8"))  # the size of the file as written, trailing newline included
        if size > MAX_FILE_BYTES:
            raise RuntimeError(f"{path} would be {size} bytes (> {MAX_FILE_BYTES}); downsample it")
        path.write_text(text)
        self.counts.setdefault(module, {})[function] = len(cases)
        self.bytes[str(path.relative_to(FIXTURES))] = size
        print(f"  wrote {path.relative_to(REPO)}  ({len(cases)} cases, {size/1024:.1f} KB)")
        return path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def graph_to_json(G) -> Dict[str, Any]:
    """networkx Graph -> node/edge lists in insertion order (order is part of the spec)."""
    import networkx as nx  # noqa: F401
    return {
        "nodes": [[n, dict(d)] for n, d in G.nodes(data=True)],
        "edges": [[u, v, dict(d)] for u, v, d in G.edges(data=True)],
    }


def graph_from_json(d: Dict[str, Any]):
    import networkx as nx
    G = nx.Graph()
    for n, attrs in d["nodes"]:
        G.add_node(n, **attrs)
    for u, v, attrs in d["edges"]:
        G.add_edge(u, v, **attrs)
    return G


def basenames(obj: Any) -> Any:
    """Replace absolute path strings anywhere in a result with their basename."""
    if isinstance(obj, dict):
        return {k: basenames(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [basenames(v) for v in obj]
    if isinstance(obj, str) and (obj.startswith("/") or obj.startswith("examples/")) and "/" in obj:
        return os.path.basename(obj)
    return obj


def null_timings(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: (None if k in ("runtime_sec", "elapsed_seconds") else null_timings(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [null_timings(v) for v in obj]
    return obj


def python_env() -> Dict[str, str]:
    import Bio, networkx, scipy, torch
    hyphy = shutil.which("hyphy")
    hyphy_version = None
    if hyphy:
        try:
            hyphy_version = subprocess.run([hyphy, "--version"], capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
        except Exception:
            hyphy_version = "unknown"
    try:
        tn93_package = importlib_version("tn93")
    except Exception:
        tn93_package = None
    tn93_bin = shutil.which("tn93")
    tn93_binary = None
    if tn93_bin:
        try:
            proc = subprocess.run([tn93_bin, "--version"], capture_output=True, text=True, timeout=30)
            tn93_binary = (proc.stdout.strip() or proc.stderr.strip()).splitlines()[0]   # the binary prints its version on stderr
        except Exception:
            tn93_binary = "unknown"
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "networkx": networkx.__version__,
        "torch": torch.__version__,
        "biopython": Bio.__version__,
        "platform": platform.platform(),
        "hyphy": hyphy_version,
        # The TN93 fixtures hide the binary and pin the PACKAGE path; both are recorded so a
        # disagreement between the two can be attributed (they measured identical here).
        "tn93_package": tn93_package,
        "tn93_binary_present_but_hidden": tn93_binary,
    }


@contextlib.contextmanager
def tn93_python_package_path():
    """
    Force `dataset.compute_tn93_distance_matrix` onto the Python `tn93` package by hiding the
    compiled `tn93` binary from `shutil.which` (dataset.py:505). The binary is PREFERRED by the
    reference, but it is not a dependency of the engine and not everyone has it, so the fixtures
    pin the package path, which the JS port mirrors. MEASURED on this machine (tn93 binary v1.0.15
    against tn93 package 1.2.2): the two agree exactly on bat_oas1 and HIV1_RT, max |delta| = 0.0.
    """
    import shutil as _shutil
    orig = _shutil.which

    def which(cmd, *args, **kwargs):
        return None if cmd == "tn93" else orig(cmd, *args, **kwargs)

    _shutil.which = which
    try:
        yield
    finally:
        _shutil.which = orig


def path_without_tn93_binary() -> str:
    """PATH with every directory holding a `tn93` executable removed, for the CLI subprocesses."""
    kept = []
    for d in os.environ.get("PATH", "").split(os.pathsep):
        cand = os.path.join(d, "tn93") if d else "tn93"
        if d and os.path.isfile(cand) and os.access(cand, os.X_OK):
            continue
        kept.append(d)
    return os.pathsep.join(kept)


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------

def with_dtype(inputs: Dict[str, Any], arr: np.ndarray) -> Dict[str, Any]:
    """Record `inputs.dtype` for a float32 case: the Python did float32 arithmetic on it (BH, CCT) or
    the model emitted it (LRTs), and a float64 replay misses the 1e-9 class by ~4e-8..9e-8
    (PHASE1A.md fixture defect 2). float64 cases carry no dtype key, as before."""
    if arr.dtype == np.float32:
        return {**inputs, "dtype": "float32"}
    return inputs


def gen_stats(w: Writer) -> None:
    from hyphaeon.stats import (
        benjamini_hochberg,
        cauchy_combination_p,
        pvals_from_lrt_meme,
        pvals_from_lrt_self_liang,
    )

    rng = np.random.default_rng(SYNTH_SEED)
    lrt_sets = {
        "zeros": np.zeros(5),
        "single_zero": np.array([0.0]),
        "single_positive": np.array([3.841]),
        "empty": np.zeros(0),
        "ties": np.array([2.0, 2.0, 0.0, 2.0, 0.5, 0.5]),
        "small_and_negative": np.array([-1.0, -1e-9, 0.0, 1e-9, 1e-6, 1e-3]),
        "typical": np.array([0.0, 0.5, 1.0, 2.0, 3.841, 5.0, 6.635, 10.83, 15.0, 20.0]),
        "huge": np.array([50.0, 100.0, 500.0, 1000.0, 1e4, 1e6, float("inf")]),
        "random_exponential_200": rng.exponential(2.0, size=200),
        "float32_model_like": rng.exponential(1.5, size=64).astype(np.float32),
    }

    meme_cases, sl_cases = [], []
    for name, lrts in lrt_sets.items():
        note = "float32 input as the model emits it; p computed in float64" if lrts.dtype == np.float32 else ""
        if name == "small_and_negative":
            note = "negative LRTs are not > 0 so they take the point-mass branch"
        if name == "huge":
            note = "chi2.sf underflows to 0 for very large LRT; inf gives 0"
        meme_cases.append(case(name, with_dtype({"lrts": lrts}, lrts), {"pvals": pvals_from_lrt_meme(lrts)}, "1e-9",
                               ("MEME null 1/3 delta0 + 2/3 (0.45 chi2_1 + 0.55 chi2_2); LRT<=0 -> 2/3. " + note).strip()))
        sl_cases.append(case(name, with_dtype({"lrts": lrts}, lrts), {"pvals": pvals_from_lrt_self_liang(lrts)}, "1e-9",
                             ("Self-Liang null 0.5 delta0 + 0.5 chi2_1; LRT<=0 -> 1. " + note).strip()))
    w.write("stats", "pvals_from_lrt_meme", meme_cases)
    w.write("stats", "pvals_from_lrt_self_liang", sl_cases)

    p_sets = {
        "empty": np.zeros(0),
        "single": np.array([0.03]),
        "single_one": np.array([1.0]),
        "all_ones": np.ones(6),
        "all_zeros": np.zeros(4),
        "ties": np.array([0.01, 0.01, 0.5, 0.01, 0.5, 1.0]),
        "monotone_step": np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.5, 0.6, 0.7, 0.8, 0.9]),
        "with_p_equal_one_and_zero": np.array([0.0, 1.0, 0.2, 0.05]),
        "random_uniform_100": rng.random(100),
        "float32_meme_like": pvals_from_lrt_meme(rng.exponential(2.0, size=300)).astype(np.float32),
        "clip_above_one": np.array([0.9, 0.95, 1.0, 0.3]),
    }
    bh_cases = [case(n, with_dtype({"pvals": p}, p), {"qvals": benjamini_hochberg(p)}, "1e-9",
                     "argsort is numpy default (quicksort, unstable) but tied p give identical q so order does not matter; result clipped to [0,1]")
                for n, p in p_sets.items()]
    w.write("stats", "benjamini_hochberg", bh_cases)

    cct_sets = {
        "empty": np.zeros(0),
        "single_half": np.array([0.5]),
        "single_small": np.array([1e-6]),
        "single_one": np.array([1.0]),
        "all_ones": np.ones(5),
        "all_zeros": np.zeros(5),
        "mixed_with_zero_and_one": np.array([0.0, 1.0, 0.5, 0.2]),
        "ties": np.array([0.05, 0.05, 0.05]),
        "typical": np.array([0.2, 0.7, 0.03, 0.5, 0.9, 0.01]),
        "random_uniform_50": rng.random(50),
        "one_extreme_among_many": np.concatenate([[1e-12], np.full(999, 0.5)]),
        "float32_input": rng.random(20).astype(np.float32),
    }
    cct_cases = [case(n, with_dtype({"pvals": p}, p), {"p_cct": cauchy_combination_p(p)}, "1e-9",
                      "p clipped to [1e-15, 1-1e-15] before tan((0.5-p)*pi); mean; back-transform; clip to [1e-15, 1]; empty -> 1.0")
                 for n, p in cct_sets.items()]
    w.write("stats", "cauchy_combination_p", cct_cases)


# --------------------------------------------------------------------------
# filter
# --------------------------------------------------------------------------

def gen_filter(w: Writer) -> None:
    from hyphaeon.filter import scan_hypergeometric_patches

    rng = np.random.default_rng(SYNTH_SEED + 1)
    sig = ("signature: scan_hypergeometric_patches(site_pvals: ndarray, alpha_site=0.05, min_k=3, max_span=35, "
           "p_local_thresh=0.01) -> list of {start, end, k, d, p_local} with 0-indexed inclusive start/end; "
           "p_local = 1 - hypergeom.cdf(k-1, L, K, d) with L=len(site_pvals), K=#sites<=alpha_site, d=end-start+1; "
           "candidates sorted by start then merged when start<=curr.end (p_local=min, k recounted over merged span). "
           "p_local is a special-function value (1e-9); start/end/k/d are exact integers.")

    def planted(L: int, patches: List[List[int]], background_sig_frac: float = 0.02) -> np.ndarray:
        p = rng.uniform(0.06, 1.0, size=L)
        n_bg = int(background_sig_frac * L)
        bg = rng.choice(L, size=n_bg, replace=False)
        p[bg] = rng.uniform(0.001, 0.05, size=n_bg)
        for sites in patches:
            p[sites] = rng.uniform(1e-4, 0.03, size=len(sites))
        return p

    cases = []
    p1 = planted(300, [[100, 102, 104, 107, 109]])
    cases.append(case("single_planted_patch_L300", {"site_pvals": p1, "alpha_site": 0.05, "min_k": 3, "max_span": 35, "p_local_thresh": 0.01},
                      {"patches": scan_hypergeometric_patches(p1)}, "1e-9", sig + " Planted 5 significant sites at 100..109."))
    p2 = planted(500, [[40, 41, 43, 44, 46, 48], [60, 62, 65], [400, 401, 402, 403]], background_sig_frac=0.03)
    cases.append(case("three_planted_patches_L500", {"site_pvals": p2, "alpha_site": 0.05, "min_k": 3, "max_span": 35, "p_local_thresh": 0.01},
                      {"patches": scan_hypergeometric_patches(p2)}, "1e-9", sig + " Patches at 40..48, 60..65 (may merge with the first via the 35-codon span) and 400..403."))
    p3 = rng.uniform(0.051, 1.0, size=200); p3[[10, 50]] = 0.01
    cases.append(case("fewer_than_min_k_significant", {"site_pvals": p3, "alpha_site": 0.05, "min_k": 3, "max_span": 35, "p_local_thresh": 0.01},
                      {"patches": scan_hypergeometric_patches(p3)}, "exact", "K=2 < min_k -> []"))
    p4 = rng.uniform(0.06, 1.0, size=120); p4[::10] = 0.01
    cases.append(case("evenly_spread_no_cluster", {"site_pvals": p4, "alpha_site": 0.05, "min_k": 3, "max_span": 35, "p_local_thresh": 0.01},
                      {"patches": scan_hypergeometric_patches(p4)}, "1e-9", sig + " 12 significant sites spaced 10 apart; whether any window passes p_local<=0.01 is what the fixture pins."))
    p5 = planted(80, [[5, 6, 7, 8, 9, 10]])
    cases.append(case("custom_thresholds", {"site_pvals": p5, "alpha_site": 0.10, "min_k": 4, "max_span": 12, "p_local_thresh": 0.05},
                      {"patches": scan_hypergeometric_patches(p5, alpha_site=0.10, min_k=4, max_span=12, p_local_thresh=0.05)}, "1e-9", sig))
    p6 = np.full(50, 0.01)
    cases.append(case("all_significant", {"site_pvals": p6, "alpha_site": 0.05, "min_k": 3, "max_span": 35, "p_local_thresh": 0.01},
                      {"patches": scan_hypergeometric_patches(p6)}, "exact", sig + " Every site significant (K=L): any window of d sites holds exactly d significant sites, so P(X>=k)=1 for every candidate and NOTHING is reported."))
    # example-derived: p-values from the committed Smc6 results CSV
    smc6 = np.array([float(r["p_value"]) for r in csv.DictReader(open(EXAMPLES / "Smc6_results.csv"))])
    cases.append(case("example_Smc6_results_csv", {"site_pvals": smc6, "alpha_site": 0.05, "min_k": 3, "max_span": 35, "p_local_thresh": 0.01},
                      {"patches": scan_hypergeometric_patches(smc6)}, "1e-9", sig + " Input is the p_value column of examples/Smc6_results.csv (1097 sites)."))
    bat = np.array([float(r["p_value"]) for r in csv.DictReader(open(EXAMPLES / "bat_oas1_results.csv"))])
    cases.append(case("example_bat_oas1_results_csv", {"site_pvals": bat, "alpha_site": 0.05, "min_k": 3, "max_span": 35, "p_local_thresh": 0.01},
                      {"patches": scan_hypergeometric_patches(bat)}, "1e-9", sig + " Input is the p_value column of examples/bat_oas1_results.csv (351 sites)."))
    w.write("filter", "scan_hypergeometric_patches", cases)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def _write_prediction_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["site", "hyphaeon_lrt", "p_value", "q_value", "is_invariable"])
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


def gen_evaluation(w: Writer) -> None:
    from hyphaeon.evaluation import (
        _correlations,
        _roc_auc,
        evaluate_files,
        load_meme_json,
        load_prediction_csv,
    )
    from hyphaeon.stats import benjamini_hochberg, pvals_from_lrt_meme

    inputs_dir = FIXTURES / "evaluation" / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SYNTH_SEED + 2)

    # --- geneA: 12 sites, non-default header positions, one negative MEME LRT, ties, invariable rows
    lrtA = np.array([0.0, 1.6, 0.0, 7.2, 3.9, 3.9, 0.4, 12.5, 0.0, 2.2, 5.1, 0.0])
    pA = pvals_from_lrt_meme(lrtA)
    qA = benjamini_hochberg(pA)
    invA = [True, False, True, False, False, False, False, False, True, False, False, True]
    _write_prediction_csv(inputs_dir / "geneA.csv", [
        {"site": i + 1, "hyphaeon_lrt": float(lrtA[i]), "p_value": float(pA[i]), "q_value": float(qA[i]), "is_invariable": str(invA[i])}
        for i in range(12)])
    meme_lrtA = [0.0, 2.1, -0.04, 6.8, 3.9, 1.0, 0.0, 15.3, 0.0, 2.2, 0.7, -0.0]
    meme_pA = [1.0, 0.19, 0.6667, 0.02, 0.06, 0.31, 0.6667, 0.0002, 1.0, 0.16, 0.45, 0.6667]
    geneA_json = {
        "MLE": {
            "headers": [["alpha;", "Synonymous rate"], ["beta&#45;", "Neg"], ["<b>LRT</b>", "Likelihood ratio test"],
                        ["p-value", "Asymptotic p"], ["# branches", "count"]],
            "content": {"0": [[0.3, 0.1, meme_lrtA[i], meme_pA[i], 1] for i in range(12)]},
        },
        "data partitions": {"0": {"name": "geneA", "coverage": [list(range(12))]}},
    }
    (inputs_dir / "geneA.MEME.json").write_text(json.dumps(geneA_json, indent=1) + "\n")

    # --- geneB: two partitions with interleaved coverage, no headers (fallback columns 5 and 6),
    #     prediction has one extra site (13) so the strict call fails and allow_site_mismatch is needed
    lrtB = rng.exponential(2.0, size=13)
    pB = pvals_from_lrt_meme(lrtB)
    qB = benjamini_hochberg(pB)
    invB = [bool(x) for x in (rng.random(13) < 0.25)]
    _write_prediction_csv(inputs_dir / "geneB.csv", [
        {"site": i + 1, "hyphaeon_lrt": float(lrtB[i]), "p_value": float(pB[i]), "q_value": float(qB[i]), "is_invariable": "1" if invB[i] else "0"}
        for i in range(13)])
    cov0 = [0, 2, 4, 6, 8, 10]
    cov1 = [1, 3, 5, 7, 9, 11]
    meme_lrtB = rng.exponential(2.0, size=12)
    meme_lrtB[3] = -0.02
    meme_pB = pvals_from_lrt_meme(np.maximum(meme_lrtB, 0.0))
    def row(i):
        return [0.2, 0.1, 0.5, 1.0, 0.5, float(meme_lrtB[i]), float(meme_pB[i]), 2]
    geneB_json = {
        "MLE": {"content": {"1": [row(i) for i in cov1], "0": [row(i) for i in cov0]}},
        "data partitions": {"0": {"coverage": cov0}, "1": {"coverage": cov1}},
    }
    (inputs_dir / "geneB.MEME.json").write_text(json.dumps(geneB_json, indent=1) + "\n")

    # --- Smc6: committed prediction CSV + a MEME JSON synthesised from model_eval/_cache (1097 sites)
    shutil.copyfile(EXAMPLES / "Smc6_results.csv", inputs_dir / "Smc6.csv")
    cache_file = next(p for p in sorted(MEME_CACHE.glob("*.json")) if len(json.load(open(p))) == 1097)
    cache = json.load(open(cache_file))
    rows = []
    for i in range(1097):
        e = cache[str(i)]
        rows.append([0.5, 0.2, 0.5, 1.0, 0.5, e["lrt"], e["p_value"], 1])
    smc6_json = {
        "MLE": {"headers": [["alpha;", ""], ["&beta;<sup>-</sup>", ""], ["p<sup>-</sup>", ""], ["&beta;<sup>+</sup>", ""],
                            ["p<sup>+</sup>", ""], ["LRT", ""], ["p-value", ""], ["# branches under selection", ""]],
                "content": {"0": rows}},
        "data partitions": {"0": {"name": "Smc6", "coverage": [list(range(1097))]}},
        "_provenance": f"LRT/p per site copied from {cache_file.name}; other columns are placeholders",
    }
    (inputs_dir / "Smc6.MEME.json").write_text(json.dumps(smc6_json) + "\n")

    # load_* fixtures
    def sites_to_json(d):
        return {str(k): v.__dict__ for k, v in sorted(d.items())}
    w.write("evaluation", "load_meme_json", [
        case("geneA_headers_nondefault_columns", {"file": "geneA.MEME.json"}, {"sites": sites_to_json(load_meme_json(inputs_dir / "geneA.MEME.json"))}, "exact",
             "headers carry HTML/entities; normalised header 'lrt' is column 2 and 'pvalue' column 3; site -0.0 is not < 0 so not clamped; -0.04 clamped to 0 with lrt_was_clamped"),
        case("geneB_two_partitions_no_headers", {"file": "geneB.MEME.json"}, {"sites": sites_to_json(load_meme_json(inputs_dir / "geneB.MEME.json"))}, "exact",
             "no headers -> fallback columns 5 (LRT) and 6 (p); partitions sorted numerically; coverage flattened; sites are coverage index + 1"),
        case("Smc6_synthesised_from_cache", {"file": "Smc6.MEME.json"}, {"sites": sites_to_json(load_meme_json(inputs_dir / "Smc6.MEME.json"))}, "exact",
             "1097 sites; 4 negative LRTs clamped"),
    ])
    w.write("evaluation", "load_prediction_csv", [
        case("geneA", {"file": "geneA.csv"}, {"sites": sites_to_json(load_prediction_csv(inputs_dir / "geneA.csv"))}, "exact", "is_invariable parsed from 'True'/'False'"),
        case("geneB_numeric_booleans", {"file": "geneB.csv"}, {"sites": sites_to_json(load_prediction_csv(inputs_dir / "geneB.csv"))}, "exact", "is_invariable parsed from '1'/'0'"),
    ])

    # helper fixtures with ties
    labels = np.array([True, False, True, True, False, False, True, False])
    scores = np.array([5.0, 5.0, 2.0, 9.0, 0.0, 2.0, 2.0, 1.0])
    w.write("evaluation", "roc_auc", [
        case("ties_average_rank", {"labels": labels, "scores": scores}, {"auc": _roc_auc(labels, scores)}, "1e-9", "Mann-Whitney AUC with rankdata(method='average') so tied scores share the mean rank"),
        case("all_positive", {"labels": np.ones(4, bool), "scores": np.arange(4.0)}, {"auc": _roc_auc(np.ones(4, bool), np.arange(4.0))}, "exact", "no negatives -> None"),
        case("perfect", {"labels": np.array([False, False, True, True]), "scores": np.array([0.1, 0.2, 0.8, 0.9])}, {"auc": _roc_auc(np.array([False, False, True, True]), np.array([0.1, 0.2, 0.8, 0.9]))}, "exact", ""),
    ])
    x = np.array([1.0, 2.0, 2.0, 3.0, 5.0, 5.0, 5.0, 8.0])
    y = np.array([0.5, 1.9, 2.4, 2.9, 4.0, 6.0, 4.5, 7.0])
    xr = rng.exponential(2.0, size=60); yr = xr * 0.8 + rng.normal(0, 1.0, size=60); yr = np.maximum(yr, 0.0)
    w.write("evaluation", "correlations", [
        case("ties_in_predicted", {"predicted": x, "observed": y}, {"pearson_r": _correlations(x, y)[0], "spearman_rho": _correlations(x, y)[1]}, "1e-9", "Spearman uses average ranks for ties"),
        case("random_60_with_zeros", {"predicted": xr, "observed": yr}, {"pearson_r": _correlations(xr, yr)[0], "spearman_rho": _correlations(xr, yr)[1]}, "1e-9", "observed clipped at 0 produces ties"),
        case("constant_observed", {"predicted": x, "observed": np.ones(8)}, {"pearson_r": None, "spearman_rho": None}, "exact", "ptp==0 -> (None, None)"),
        case("too_short", {"predicted": np.array([1.0]), "observed": np.array([2.0])}, {"pearson_r": None, "spearman_rho": None}, "exact", "n<2 -> (None, None)"),
    ])

    # evaluate_files
    def ev(pred, meme, **kw):
        return basenames(evaluate_files(inputs_dir / pred, inputs_dir / meme, **kw))
    w.write("evaluation", "evaluate_files", [
        case("geneA_all_sites", {"prediction_file": "geneA.csv", "meme_result_file": "geneA.MEME.json", "allow_site_mismatch": False, "variable_only": False},
             {"result": ev("geneA.csv", "geneA.MEME.json")}, "1e-9",
             "prediction_file/meme_result_file are basenames of the files under fixtures/evaluation/inputs/; correlations and AUC are 1e-9, counts and confusion matrices exact"),
        case("geneA_variable_only", {"prediction_file": "geneA.csv", "meme_result_file": "geneA.MEME.json", "allow_site_mismatch": False, "variable_only": True},
             {"result": ev("geneA.csv", "geneA.MEME.json", variable_only=True)}, "1e-9", "invariable rows excluded from metrics, still counted in totals"),
        case("geneB_allow_site_mismatch", {"prediction_file": "geneB.csv", "meme_result_file": "geneB.MEME.json", "allow_site_mismatch": True, "variable_only": False},
             {"result": ev("geneB.csv", "geneB.MEME.json", allow_site_mismatch=True)}, "1e-9", "prediction site 13 has no MEME counterpart; intersection used and a warning emitted"),
        case("Smc6_example", {"prediction_file": "Smc6.csv", "meme_result_file": "Smc6.MEME.json", "allow_site_mismatch": False, "variable_only": False},
             {"result": ev("Smc6.csv", "Smc6.MEME.json")}, "1e-9", "prediction = examples/Smc6_results.csv; MEME JSON synthesised from model_eval/_cache (matching 1097 sites)"),
        case("Smc6_example_variable_only", {"prediction_file": "Smc6.csv", "meme_result_file": "Smc6.MEME.json", "allow_site_mismatch": False, "variable_only": True},
             {"result": ev("Smc6.csv", "Smc6.MEME.json", variable_only=True)}, "1e-9", ""),
    ])
    # error cases are recorded as expected exception messages (exact class + message prefix)
    from hyphaeon.evaluation import EvaluationError
    errs = []
    for name, args in [("site_mismatch_strict", ("geneB.csv", "geneB.MEME.json", {})),
                       ("gene_name_mismatch", ("geneA.csv", "geneB.MEME.json", {}))]:
        try:
            evaluate_files(inputs_dir / args[0], inputs_dir / args[1], **args[2])
            msg = None
        except EvaluationError as e:
            msg = basenames(str(e))
        errs.append(case(name, {"prediction_file": args[0], "meme_result_file": args[1], **args[2]}, {"raises": "EvaluationError", "message": msg}, "exact", "message has absolute paths replaced by basenames"))
    w.write("evaluation", "evaluate_files_errors", errs)


# --------------------------------------------------------------------------
# epistasis
# --------------------------------------------------------------------------

def _synthetic_attributions(rng, L=40, N=12):
    """Sparse non-negative L x N float32 'attention * delta' matrix with planted co-selected sites."""
    attr = rng.random((L, N)) * (rng.random((L, N)) < 0.35)
    pattern1 = np.zeros(N); pattern1[[0, 1, 2, 5]] = [0.9, 0.8, 0.7, 0.6]
    pattern2 = np.zeros(N); pattern2[[7, 8, 9]] = [0.85, 0.75, 0.65]
    for s in (2, 6, 10):
        attr[s] = pattern1 + 0.05 * rng.random(N) * (rng.random(N) < 0.3)
    for s in (20, 21, 22):
        attr[s] = pattern2 + 0.05 * rng.random(N) * (rng.random(N) < 0.3)
    attr[30] = 0.0  # inactive site (zero norm)
    attr[31] = 0.0
    lrts = rng.uniform(0.0, 3.0, size=L)
    lrts[[2, 6, 10]] = [6.5, 8.1, 5.2]
    lrts[[20, 21, 22]] = [4.4, 3.9, 7.7]
    lrts[30] = 0.0
    aas = [ "ACDEFGHIKLMNPQRSTVWY"[i] for i in rng.integers(0, 20, size=L)]
    aas[30] = "-"
    return attr.astype(np.float32), lrts.astype(np.float32), aas


def _coherence(attr, site_indices):
    sub = attr[site_indices, :]
    cov = sub @ sub.T
    ev = np.maximum(np.linalg.eigvalsh(cov), 0.0)
    return float(ev[-1] / ev.sum()) if ev.sum() > 0 else 0.0


def gen_epistasis(w: Writer) -> None:
    import networkx as nx
    from hyphaeon.epistasis import (
        compute_branch_coselection_network,
        compute_sector_permutation_test,
        extract_epistatic_sectors_tse,
    )

    rng = np.random.default_rng(SYNTH_SEED + 3)
    attr, lrts, aas = _synthetic_attributions(rng)
    L, N = attr.shape
    taxa = [f"taxon_{i:02d}" for i in range(N)]

    sig = ("signature: compute_branch_coselection_network(attributions[L,N] float32, lrts[L] float32, branch_names[N], consensus_aas[L], "
           "min_sim=0.35, min_shared=2, max_fdr=0.05, min_lrt=1.0, min_cesi=2.0) -> (sig_pairs sorted by cesi desc, networkx Graph with nodes 1..L "
           "attrs ref/lrt and edges attrs weight/shared/cesi/fdr_q). Cosine on float32 inputs; t = sim*sqrt(df/(1-sim^2)) with df=max(1,N-2); "
           "p = t.sf; BH over all V(V-1)/2 active pairs (q cast to float32). Graph node order is 1..L insertion order. "
           "NOTE the CLI passes min_sim=0.30 (cli.py epistasis parser default) while the function default is 0.35.")

    net_cases = []
    defaults = dict(min_sim=0.35, min_shared=2, max_fdr=0.05, min_lrt=1.0, min_cesi=2.0)
    for name, kw in [("planted_function_defaults", defaults),
                     ("planted_cli_defaults_min_sim_0.30", {**defaults, "min_sim": 0.30}),
                     ("planted_loose_filters", dict(min_sim=0.10, min_shared=1, max_fdr=0.50, min_lrt=0.0, min_cesi=0.0)),
                     ("planted_strict_cesi", {**defaults, "min_cesi": 5.0})]:
        edges, G = compute_branch_coselection_network(attr, lrts, taxa, aas, **kw)
        net_cases.append(case(name, {"attributions": attr, "lrts": lrts, "branch_names": taxa, "consensus_aas": aas, **kw},
                              {"sig_pairs": edges, "graph": graph_to_json(G)}, "1e-6",
                              sig + " Planted: sites 3,7,11 share taxa {1,2,3,6}; sites 21,22,23 share taxa {8,9,10}; sites 31,32 inactive (1-indexed)."))
    small = np.zeros((4, 5), np.float32); small[1, 0] = 1.0
    edges, G = compute_branch_coselection_network(small, np.array([1, 2, 3, 4], np.float32), [f"t{i}" for i in range(5)], list("ACDE"))
    net_cases.append(case("fewer_than_two_active_sites", {"attributions": small, "lrts": [1, 2, 3, 4], "branch_names": [f"t{i}" for i in range(5)], "consensus_aas": list("ACDE"), **defaults},
                          {"sig_pairs": edges, "graph": graph_to_json(G)}, "exact", "V<2 -> ([], graph with nodes only)"))
    w.write("epistasis", "compute_branch_coselection_network", net_cases)

    # sectors on the graph from the first network case
    edges, G = compute_branch_coselection_network(attr, lrts, taxa, aas, **defaults)
    edges_cli, G_cli = compute_branch_coselection_network(attr, lrts, taxa, aas, **{**defaults, "min_sim": 0.30})
    a_np = rng.integers(0, 20, size=(L, N)).astype(np.int64)
    a_np[rng.random((L, N)) < 0.08] = 20
    sec_sig = ("signature: extract_epistatic_sectors_tse(G, attributions, lrts, consensus_aas, min_clique_size=3, max_overlap=0.50, min_coherence=0.50, "
               "focal_taxon=None, a_np=None, taxa=None, n_permutations=10000, max_perm_p=None, rng_seed=42) -> sectors sorted by (coherence, size) desc. "
               "Communities: networkx greedy_modularity_communities(G_sub, weight='weight') (networkx " + nx.__version__ + " tie-breaking is part of the spec) "
               "when >= min_clique_size nodes have degree>0, else connected components; members sorted ascending; sites with |v_dom|<0.10 pruned along the top eigenvector; "
               "coherence = lambda_max / trace of A_S A_S^T. max_overlap is accepted but unused by the Python.")
    sec_cases = []
    for name, Gx, kw in [("planted_no_permutations", G, dict(n_permutations=0)),
                         ("planted_cli_graph_no_permutations", G_cli, dict(n_permutations=0)),
                         ("planted_low_coherence_threshold", G_cli, dict(n_permutations=0, min_coherence=0.10)),
                         ("planted_focal_taxon", G, dict(n_permutations=0, focal_taxon="TAXON_05", a_np=a_np, taxa=taxa))]:
        secs = extract_epistatic_sectors_tse(Gx, attr, lrts, aas, **kw)
        inp = {"graph": graph_to_json(Gx), "attributions": attr, "lrts": lrts, "consensus_aas": aas, "min_clique_size": 3, "max_overlap": 0.5,
               "min_coherence": kw.get("min_coherence", 0.5), "n_permutations": 0, "max_perm_p": None, "rng_seed": PERM_SEED,
               "focal_taxon": kw.get("focal_taxon"), "a_np": kw.get("a_np"), "taxa": kw.get("taxa")}
        sec_cases.append(case(name, inp, {"sectors": secs}, "1e-6",
                              sec_sig + " n_permutations=0 makes the permutation block degenerate (p_perm=1, null mean = observed, std 0) so the case is deterministic; "
                              "site lists and sector membership are exact, coherence is 1e-6."))
    secs = extract_epistatic_sectors_tse(G, attr, lrts, aas, n_permutations=2000, rng_seed=PERM_SEED)
    sec_cases.append(case("planted_with_permutations_B2000_seed42", {"graph": graph_to_json(G), "attributions": attr, "lrts": lrts, "consensus_aas": aas,
                                                                     "min_clique_size": 3, "max_overlap": 0.5, "min_coherence": 0.5, "n_permutations": 2000, "max_perm_p": None, "rng_seed": PERM_SEED},
                          {"sectors": secs}, "statistical",
                          sec_sig + " p_perm/null_* come from numpy.random.default_rng(42) (PCG64) via rng.choice(pool, K, replace=False) per draw; "
                          "compare per PLAN.md 5.4: |dp| <= 3*sqrt(p(1-p)/B), null moments within 2%. sites/size/coherence are still exact/1e-6."))
    secs = extract_epistatic_sectors_tse(G, attr, lrts, aas, n_permutations=2000, rng_seed=PERM_SEED, max_perm_p=0.05)
    sec_cases.append(case("planted_max_perm_p_0.05", {"graph": graph_to_json(G), "attributions": attr, "lrts": lrts, "consensus_aas": aas,
                                                      "min_clique_size": 3, "max_overlap": 0.5, "min_coherence": 0.5, "n_permutations": 2000, "max_perm_p": 0.05, "rng_seed": PERM_SEED},
                          {"sectors": secs}, "statistical", sec_sig + " sectors with p_perm > 0.05 dropped; membership of the surviving set is expected to match."))
    empty = nx.Graph(); empty.add_nodes_from(range(1, L + 1))
    sec_cases.append(case("no_edges", {"graph": graph_to_json(empty), "attributions": attr, "lrts": lrts, "consensus_aas": aas, "n_permutations": 0},
                          {"sectors": extract_epistatic_sectors_tse(empty, attr, lrts, aas, n_permutations=0)}, "exact", "number_of_edges()==0 -> []"))
    G2 = nx.Graph(); G2.add_edge(3, 7, weight=0.9)
    sec_cases.append(case("two_nodes_below_min_clique_size_uses_components", {"graph": graph_to_json(G2), "attributions": attr, "lrts": lrts, "consensus_aas": aas, "n_permutations": 0},
                          {"sectors": extract_epistatic_sectors_tse(G2, attr, lrts, aas, n_permutations=0)}, "1e-6",
                          "2 nodes with degree>0 < min_clique_size=3 -> connected components path instead of greedy modularity"))
    w.write("epistasis", "extract_epistatic_sectors_tse", sec_cases)

    # permutation test
    perm_sig = ("signature: compute_sector_permutation_test(attributions, site_indices, observed_coherence, n_permutations=10000, active_only=True, rng_seed=42) -> "
                "{p_perm, null_coherence_mean, null_coherence_std, null_coherence_95, isotropic_baseline}. RNG: numpy.random.default_rng(rng_seed) i.e. PCG64; "
                "each draw is rng.choice(candidate_pool, size=K, replace=False); candidate_pool = sites with L2 norm > 1e-9 when active_only; batches of <= 25000; "
                "coherence per draw = max eig / trace of the K x K Gram (float32 accumulate then float32 stored); p_perm counts null >= observed - 1e-7; "
                "null_std is population std (ddof=0); null_95 is numpy.percentile linear interpolation.")
    perm_cases = []
    for name, sites in [("planted_sector_3_7_11", [2, 6, 10]), ("random_triplet", [0, 4, 15]), ("planted_pair_21_22", [20, 21])]:
        obs = _coherence(attr, sites)
        res = compute_sector_permutation_test(attr, sites, obs, n_permutations=5000, active_only=True, rng_seed=PERM_SEED)
        perm_cases.append(case(name, {"attributions": attr, "site_indices": sites, "observed_coherence": obs, "n_permutations": 5000, "active_only": True, "rng_seed": PERM_SEED},
                               res, "statistical", perm_sig + " isotropic_baseline is exact (1/K)."))
    obs = _coherence(attr, [2, 6, 10])
    res = compute_sector_permutation_test(attr, [2, 6, 10], obs, n_permutations=0, active_only=True, rng_seed=PERM_SEED)
    perm_cases.append(case("n_permutations_zero", {"attributions": attr, "site_indices": [2, 6, 10], "observed_coherence": obs, "n_permutations": 0, "active_only": True, "rng_seed": PERM_SEED},
                           res, "exact", "early return: p_perm=1, null mean = observed, std 0, 95 = observed"))
    tiny = attr[:3]
    res = compute_sector_permutation_test(tiny, [0, 1, 2], 0.9, n_permutations=100, active_only=False, rng_seed=PERM_SEED)
    perm_cases.append(case("pool_equals_K_active_only_false", {"attributions": tiny, "site_indices": [0, 1, 2], "observed_coherence": 0.9, "n_permutations": 100, "active_only": False, "rng_seed": PERM_SEED},
                           res, "statistical", "pool size == K so every draw is the same set in some order; coherence is permutation invariant so the null is constant"))
    w.write("epistasis", "compute_sector_permutation_test", perm_cases)


# --------------------------------------------------------------------------
# phenotype
# --------------------------------------------------------------------------

def gen_phenotype(w: Writer) -> None:
    from Bio import Phylo
    from hyphaeon.dataset import parse_alignment_sequences
    from hyphaeon.phenotype import compute_phylogenetic_covariance, generate_permulations, resolve_phenotype_vector

    inputs_dir = FIXTURES / "phenotype" / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    rho_taxa = list(parse_alignment_sequences(str(EXAMPLES / "RHO.fasta")).keys())
    bat_taxa = list(parse_alignment_sequences(str(EXAMPLES / "bat_oas1.fasta")).keys())
    smc6_taxa = list(parse_alignment_sequences(str(EXAMPLES / "Smc6.fasta")).keys())

    # small phenotype tables
    with open(inputs_dir / "smc6_traits.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["species", "hominoid", "body_mass_kg", "note"])
        rows = [("homSap_293T", "yes", 62.0, "human"), ("hg18", "1", 62.0, "human build"), ("panTro4", "true", 45.0, ""),
                ("panPan", "case", 39.0, ""), ("ponAbe2", "foreground", 60.0, ""), ("nomLeu3", "no", 6.0, ""),
                ("macMul", "0", 8.0, ""), ("rheMac", "false", 8.0, ""), ("calJac", "no", 0.35, ""),
                ("saiBol", "", 0.8, "missing trait"), ("chlSab", "NO", 4.5, "case-insensitive"), ("aotTri", "yes", 1.0, "partial name: matches aotTri_OMK by substring")]
        for r in rows:
            wr.writerow(r)
    with open(inputs_dir / "smc6_traits.tsv", "w", newline="") as f:
        f.write("taxon\ttrait\n")
        for t, v in [("homSap_293T", "1"), ("panTro4", "1"), ("macMul", "0"), ("calJac", "0")]:
            f.write(f"{t}\t{v}\n")

    sig = ("signature: resolve_phenotype_vector(taxa, preset=None, foreground=None, background=None, phenotype_file=None, trait_col=None, species_col=None, continuous=False) "
           "-> (y float[N], meta{mode, foreground_count, background_count, description}). Priority: phenotype_file, then preset, then foreground. "
           "Preset: fnmatch(t.lower(), pat.lower()) or substring. Foreground string split on '|' when it has '|' and no ',', else on ','; each pattern tried as "
           "re.search(pat, t, IGNORECASE) then fnmatch/substring with leading/trailing '.*' stripped. background is accepted and ignored. "
           "File: sep tab for .tsv/.tab else comma; species column auto-detected from a fixed name list else column 0; trait = first other column; "
           "values matched exact, lower, then substring either way in dict insertion order; discrete truthy set {1,true,yes,case,foreground,target,positive} plus foreground tokens.")
    cases = []
    def add(name, taxa_name, taxa, tol="exact", **kw):
        y, meta = resolve_phenotype_vector(taxa, **kw)
        inputs = {"taxa": taxa, "preset": kw.get("preset"), "foreground": kw.get("foreground"), "background": kw.get("background"),
                  "phenotype_file": kw.get("phenotype_file"), "trait_col": kw.get("trait_col"), "species_col": kw.get("species_col"), "continuous": kw.get("continuous", False)}
        cases.append(case(name, inputs, {"y": y, "meta": meta, "foreground_taxa": [t for t, v in zip(taxa, y) if v > 0]}, tol, sig + f" taxa = parse order of examples/{taxa_name}."))
    add("preset_marine_RHO", "RHO.fasta", rho_taxa, preset="marine")
    add("preset_dim_light_RHO", "RHO.fasta", rho_taxa, preset="dim-light")
    add("preset_echolocation_bat_oas1", "bat_oas1.fasta", bat_taxa, preset="echolocation")
    add("preset_longevity_Smc6", "Smc6.fasta", smc6_taxa, preset="longevity")
    add("foreground_readme_marine_list_RHO", "RHO.fasta", rho_taxa, foreground=RHO_MARINE_FOREGROUND)
    add("foreground_python_list_Smc6", "Smc6.fasta", smc6_taxa, foreground=["homSap_293T", "panTro4", "hg18"])
    add("foreground_regex_bat_oas1", "bat_oas1.fasta", bat_taxa, foreground="^M_.*|R_sin")
    add("foreground_pipe_separated_bat_oas1", "bat_oas1.fasta", bat_taxa, foreground="M_lyra|P_alec")
    add("foreground_glob_Smc6", "Smc6.fasta", smc6_taxa, foreground="pan*,mac*")
    add("foreground_dotstar_prefix_Smc6", "Smc6.fasta", smc6_taxa, foreground=".*Anu,cer.*")
    add("file_csv_autodetect_discrete_Smc6", "Smc6.fasta", smc6_taxa, phenotype_file=str(inputs_dir / "smc6_traits.csv"))
    add("file_csv_explicit_cols_discrete_Smc6", "Smc6.fasta", smc6_taxa, phenotype_file=str(inputs_dir / "smc6_traits.csv"), trait_col="hominoid", species_col="species")
    add("file_csv_continuous_body_mass_Smc6", "Smc6.fasta", smc6_taxa, tol="1e-9", phenotype_file=str(inputs_dir / "smc6_traits.csv"), trait_col="body_mass_kg", continuous=True)
    add("file_csv_discrete_with_extra_foreground_tokens_Smc6", "Smc6.fasta", smc6_taxa, phenotype_file=str(inputs_dir / "smc6_traits.csv"), trait_col="note", foreground="human")
    add("file_tsv_discrete_Smc6", "Smc6.fasta", smc6_taxa, phenotype_file=str(inputs_dir / "smc6_traits.tsv"))
    for c in cases:
        if c["inputs"]["phenotype_file"]:
            c["inputs"]["phenotype_file"] = os.path.basename(c["inputs"]["phenotype_file"])
            c["notes"] += " phenotype_file is relative to fixtures/phenotype/inputs/. continuous mode z-scores y over ALL taxa (unmatched taxa stay 0 before standardisation)."
    w.write("phenotype", "resolve_phenotype_vector", cases)

    cov_sig = ("signature: compute_phylogenetic_covariance(tree: Bio.Phylo tree, taxa) -> V[M,M] float64 with V[i,i] = root-to-tip distance and "
               "V[i,j] = root-to-MRCA distance (Bio.Phylo tree.distance / common_ancestor; a None branch length counts as 0). taxa order = tree terminal order.")
    cov_cases = []
    trees = {}
    for nwk in ("Smc6.nwk", "bat_oas1.nwk"):
        tree = Phylo.read(str(EXAMPLES / nwk), "newick")
        taxa = [t.name for t in tree.get_terminals()]
        trees[nwk] = (tree, taxa)
        V = compute_phylogenetic_covariance(tree, taxa)
        cov_cases.append(case(f"example_{nwk}", {"newick": (EXAMPLES / nwk).read_text().strip(), "taxa": taxa}, {"V": V}, "1e-9", cov_sig))
    tree, taxa = trees["Smc6.nwk"]
    sub = taxa[::3]
    cov_cases.append(case("Smc6_subset_reordered", {"newick": (EXAMPLES / "Smc6.nwk").read_text().strip(), "taxa": sub[::-1]}, {"V": compute_phylogenetic_covariance(tree, sub[::-1])}, "1e-9", cov_sig + " Subset of taxa in reversed order."))
    w.write("phenotype", "compute_phylogenetic_covariance", cov_cases)

    perm_sig = ("signature: generate_permulations(original_y, tree, taxa, n_perm=1000, seed=42) -> [n_perm, M]. RNG is numpy's LEGACY global state: np.random.seed(seed) then "
                "np.random.randn(M, n_perm) (MT19937 RandomState, Box-Muller via legacy gauss). Z = cholesky(V + 1e-7 I) @ randn. Binary y (values subset of {0,1}, exactly 2 unique): "
                "the top-K entries of each column by argsort become 1 (K = #foreground). Otherwise rank-matching: sorted(original_y)[argsort(argsort(Z[:,p]))]. "
                "Statistical parity: compare per-taxon foreground frequency (binary) or per-taxon mean (continuous) against the *_summary outputs; the matrix itself will not match.")
    tree, taxa = trees["Smc6.nwk"]
    yb = np.array([1.0 if t in ("homSap_293T", "hg18", "panTro4", "panPan") else 0.0 for t in taxa])
    P = generate_permulations(yb, tree, taxa, n_perm=200, seed=PERM_SEED)
    V = compute_phylogenetic_covariance(tree, taxa)
    perm_cases = [case("Smc6_binary_hominini_B200_seed42", {"original_y": yb, "newick": (EXAMPLES / "Smc6.nwk").read_text().strip(), "taxa": taxa, "n_perm": 200, "seed": PERM_SEED},
                       {"perm_matrix": P, "per_taxon_foreground_frequency": P.mean(axis=0), "row_sums": P.sum(axis=1), "V": V},
                       "statistical", perm_sig + " row_sums are exact (every row has K=4 ones).")]
    rngc = np.random.default_rng(SYNTH_SEED + 4)
    yc = rngc.normal(size=len(taxa))
    Pc = generate_permulations(yc, tree, taxa, n_perm=200, seed=PERM_SEED)
    perm_cases.append(case("Smc6_continuous_B200_seed42", {"original_y": yc, "newick": (EXAMPLES / "Smc6.nwk").read_text().strip(), "taxa": taxa, "n_perm": 200, "seed": PERM_SEED},
                           {"perm_matrix": Pc, "per_taxon_mean": Pc.mean(axis=0), "sorted_row_values": np.sort(yc), "V": V},
                           "statistical", perm_sig + " Every row is a permutation of original_y (sorted_row_values is exact)."))
    tree_b, taxa_b = trees["bat_oas1.nwk"]
    ybb = np.array([1.0 if t.startswith("M_") else 0.0 for t in taxa_b])
    Pb = generate_permulations(ybb, tree_b, taxa_b, n_perm=200, seed=PERM_SEED)
    perm_cases.append(case("bat_oas1_binary_Myotis_B200_seed42", {"original_y": ybb, "newick": (EXAMPLES / "bat_oas1.nwk").read_text().strip(), "taxa": taxa_b, "n_perm": 200, "seed": PERM_SEED},
                           {"perm_matrix": Pb, "per_taxon_foreground_frequency": Pb.mean(axis=0), "row_sums": Pb.sum(axis=1), "V": compute_phylogenetic_covariance(tree_b, taxa_b)},
                           "statistical", perm_sig + " bat_oas1.nwk branch lengths are in the tens (raw units); V is NOT rescaled here (only dataset.load_alignment_and_tree rescales)."))
    w.write("phenotype", "generate_permulations", perm_cases)


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------

def gen_dataset(w: Writer) -> None:
    import torch
    from Bio import Phylo
    from hyphaeon.dataset import (
        AA_MAP, CODON_TO_AA, GENETIC_CODE,
        compute_fast_dist_matrix, compute_mds_coordinates, downsample_taxa_faith_pd,
        enforce_nonzero_branch_lengths, extract_tree_from_string_or_file, get_aa_token, get_codon_token,
        has_nonzero_branch_lengths, load_alignment_and_tree, parse_alignment_sequences, prune_identical_sequences,
    )

    inputs_dir = FIXTURES / "dataset" / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    # --- tokenizer
    codons = [a + b + c for a in "TCAG" for b in "TCAG" for c in "TCAG"]
    extra = ["---", "NNN", "ANN", "AN-", "???", "??A", "atg", "Atg", "AUG", "aug", "TTU", "RAT", "YTG", "A-G", "--A", "", "AT", "ATGC", "XXX", "*", "TAa"]
    tok_cases = [case("all_64_codons_TCAG_order", {"codons": codons},
                      {"codon_tokens": [get_codon_token(c) for c in codons], "aa_tokens": [get_aa_token(c) for c in codons], "amino_acids": [CODON_TO_AA[c] for c in codons]},
                      "exact", "GENETIC_CODE: 61 sense codons -> 0..60 in TCAG table order skipping stops; stops TAA/TAG/TGA -> 64 (the unknown token, not a separate stop token). "
                               "AA_MAP: 20 standard amino acids alphabetical 0..19; stop '*' and everything else -> 20."),
                 case("gaps_ambiguity_lowercase_U", {"codons": extra},
                      {"codon_tokens": [get_codon_token(c) for c in extra], "aa_tokens": [get_aa_token(c) for c in extra]},
                      "exact", "get_codon_token upper-cases then dict lookup with default 64; 'AUG' -> 64 at the token level (U is only replaced by T in parse_alignment_sequences); "
                               "lengths other than 3 -> 64/20."),
                 case("tables", {}, {"GENETIC_CODE": GENETIC_CODE, "AA_MAP": AA_MAP, "CODON_TO_AA": CODON_TO_AA}, "exact", "the three lookup tables verbatim")]
    w.write("dataset", "tokenizer", tok_cases)

    # --- parse_alignment_sequences on examples + synthetic formats
    parse_cases = []
    for fa in sorted(EXAMPLES.glob("*.fasta")):
        seqs = parse_alignment_sequences(str(fa))
        names = list(seqs.keys())
        parse_cases.append(case(f"example_{fa.name}", {"file": fa.name},
                                {"n_sequences": len(names), "names": names, "lengths": [len(seqs[n]) for n in names],
                                 "first_5": {n: seqs[n] for n in names[:5]}, "sha256_of_concatenated_sequences": hashlib.sha256("".join(seqs[n] for n in names).encode()).hexdigest()},
                                "exact", "file is examples/<file>; sequences upper-cased with U->T; names = first whitespace token after '>' with surrounding quotes stripped (FASTA) "
                                         "or TAXLABELS/matrix names (NEXUS). Only the first 5 sequences are stored; the sha256 covers all of them concatenated in order."))
    synthetic_files = {
        "phylip_oneline.phy": "2 12\nsp_one ATGAAACCCGGG\nsp_two ATGAAACCCGGT\n",
        "phylip_sequential.phy": "3 12\nsp_one      ATGAAA\nCCCGGG\nsp_two      ATGAAACCCGGT\nsp_three    ATGAAACCCGGA\n",
        "phylip_interleaved.phy": "2 18\ntaxA ATGAAA CCCGGG\ntaxB ATGAAA CCCGGT\n\nTTTAAA\nTTTAAC\n",
        "fasta_with_embedded_tree.fa": ">A desc\nATGAAACCC\nGGG\n>'B'\nATG AAA CCC GGT\n>C\nauguuuccc ggu\n\n(A:0.1,(B:0.2,C:0.3):0.05);\n",
        "nexus_labels.nex": "#NEXUS\nBEGIN DATA;\nDIMENSIONS NTAX=3 NCHAR=9;\nFORMAT DATATYPE=DNA MISSING=? GAP=-;\nMATRIX\n'sp one' ATGAAACCC\nsp_two   ATG---CCC\nsp_three ATGAAANNN\n;\nEND;\nBEGIN TREES;\nTREE t1 = ((sp_one,sp_two),sp_three);\nEND;\n",
        "nexus_nolabels.nex": "#NEXUS\nBEGIN TAXA;\nTAXLABELS 'x1' x2 \"x3\";\nEND;\nBEGIN CHARACTERS;\nFORMAT DATATYPE=DNA NOLABELS;\nMATRIX\nATGAAACCC\nATGAAACCT\nATGAAACCG\n;\nEND;\n",
        "nexus_interleaved_labels.nex": "#NEXUS\nBEGIN DATA;\nFORMAT DATATYPE=DNA INTERLEAVE;\nMATRIX\n[comment]\nsp1 ATGAAA\nsp2 ATGAAA\n\nsp1 CCCGGG\nsp2 CCCGGT\n;\nEND;\n",
        "fasta_gzipped.fa.gz": None,
    }
    import gzip
    for name, text in synthetic_files.items():
        p = inputs_dir / name
        if text is None:
            with gzip.open(p, "wt") as f:
                f.write(">g1\nATGAAACCC\n>g2\nATGAAACCT\n")
        else:
            p.write_text(text)
        seqs = parse_alignment_sequences(str(p))
        quirk = ""
        if name in ("phylip_sequential.phy", "phylip_interleaved.phy"):
            quirk = (" QUIRK replicated: a sequence-only continuation line that is purely alphanumeric and <= 35 chars is taken as a taxon NAME, and a 'name seq' line whose "
                     "sequence part is <= 10 chars is not recognised as a record start; the resulting garbage dict is what the Python returns and what the fixture pins.")
        if name == "nexus_labels.nex":
            quirk = " QUIRK replicated: a quoted NEXUS matrix label containing a space ('sp one') is split on whitespace, giving name \"sp\" and a sequence that starts with ONE'."
        parse_cases.append(case(f"synthetic_{name}", {"file": name, "text": text}, {"sequences": seqs}, "exact",
                                "file is fixtures/dataset/inputs/<file>; text is its content (null for the gzipped one). Format detection order: PHYLIP header 'ntaxa nsites' -> FASTA ('>') -> NEXUS." + quirk))
    w.write("dataset", "parse_alignment_sequences", parse_cases)

    # --- tree extraction
    def tree_summary(tree):
        if tree is None:
            return None
        return {"terminals": [t.name for t in tree.get_terminals()],
                "terminal_branch_lengths": [t.branch_length for t in tree.get_terminals()],
                "n_clades": sum(1 for _ in tree.find_clades()),
                "has_nonzero_branch_lengths": has_nonzero_branch_lengths(tree)}
    tree_cases = []
    tree_inputs = {
        "plain_newick": "((A:0.1,B:0.2):0.05,(C:0.3,D:0.4):0.06);",
        "hyphy_annotated": "((A:0.1{Foreground},B:0.2):0.05{Foreground},(C:0.3,D:0.4):0.06);",
        "nexus_comments": "((A:0.1[&label=x],B:0.2):0.05,(C:0.3,D:0.4)[&support=1]:0.06);",
        "topology_only": "((A,B),(C,D));",
        "no_trailing_semicolon": "((A:0.1,B:0.2):0.05,(C:0.3,D:0.4):0.06)",
        "nexus_tree_block": "#NEXUS\nBEGIN TREES;\n  TREE tree1 = [&R] ((A:0.1,B:0.2):0.05,(C:0.3,D:0.4):0.06);\nEND;\n",
        "single_pair_not_a_tree": "(A:0.1,B:0.2);",
        "quoted_names": "(('sp one':0.1,\"sp two\":0.2):0.05,sp_three:0.3);",
        "example_Smc6.nwk": (EXAMPLES / "Smc6.nwk").read_text(),
        "example_bat_oas1.nwk": (EXAMPLES / "bat_oas1.nwk").read_text(),
        "example_camelid.nwk": (EXAMPLES / "camelid.nwk").read_text(),
    }
    for name, text in tree_inputs.items():
        tree = extract_tree_from_string_or_file(text)
        note = ("signature: extract_tree_from_string_or_file(source) -> Bio.Phylo tree or None. Tries: NEXUS/HyPhy 'TREE name = (...);' regex; then any line starting with '(' having >= 2 '(' "
                "(adds ';' if missing); then the whole content. {..} and [..] comments are stripped. A single-pair newick has one '(' so it is rejected (None).")
        if tree is not None and name == "topology_only":
            enforce_nonzero_branch_lengths(tree, min_len=1e-4)
            note += " Output after enforce_nonzero_branch_lengths(min_len=1e-4): None -> 1e-3, < 1e-4 -> 1e-4."
        if name == "nexus_tree_block":
            note += " QUIRK replicated: the '[&R]' rooting annotation sits between '=' and '(' so the TREE regex does not match, and the line does not start with '(' -> None."
        if name == "quoted_names":
            note += " Bio.Phylo newick: single-quoted names keep their spaces; a double-quoted name with a space is split by the tokenizer (terminal 'two\"')."
        tree_cases.append(case(name, {"source": text}, {"tree": tree_summary(tree)}, "exact", note))
    w.write("dataset", "extract_tree_from_string_or_file", tree_cases)

    # --- patristic distances + MDS on the example trees
    dist_cases, mds_cases = [], []
    mds_sig = ("TOLERANCE IS RELATIVE (tolerance_relative=true, tolerance_floor=1): 1e-5 * max(1, |compared magnitude|), because the reference's float32 eigh rounds at ~6e-8 of the coordinate's magnitude and the raw bat_oas1 coordinates are ~60. "
               "signature: compute_mds_coordinates(dist_matrix float32 [N,N], n_components=4, mds_sign='canonical') -> float32 [N,4]. Dense path (N<=500): H = I - 1/N, B = -0.5 H D^2 H in float32, "
               "numpy.linalg.eigh (LAPACK syevd), eigenvalues sorted descending, coords = eigvecs[:, :4] * sqrt(max(eigval, 0)). "
               "SIGN CONVENTION (mds_sign='canonical', the default): before the sqrt(eigval) scaling each kept eigenvector is flipped so that its largest-magnitude entry "
               "is positive (np.argmax(np.abs(col)): first index on ties; an all-zero column is left alone). Columns are therefore comparable EXACTLY (max|c_js - c_py| <= 1e-5) "
               "with no sign-flip allowance; the sign-invariant Gram matrix coords @ coords.T is still given as 'gram' for degenerate eigenspaces. "
               "mds_sign='lapack' keeps the solver's signs (the pre-canonical behaviour) and is not what these fixtures record. "
               "N > 500 uses scipy eigsh (Lanczos) on an implicit operator; same convention, same tolerance.")
    example_trees = {}
    for nwk, fa in (("Smc6.nwk", "Smc6.fasta"), ("bat_oas1.nwk", "bat_oas1.fasta"), ("HIV1_RT.nwk", "HIV1_RT.fasta")):
        tree = extract_tree_from_string_or_file(str(EXAMPLES / nwk))
        enforce_nonzero_branch_lengths(tree, min_len=1e-4)
        seqs = parse_alignment_sequences(str(EXAMPLES / fa))
        taxa = [t.name.strip("'\"") for t in tree.get_terminals() if t.name and t.name.strip("'\"") in seqs]
        D = compute_fast_dist_matrix(tree, taxa)
        example_trees[nwk] = (tree, taxa, D)
        if nwk != "HIV1_RT.nwk":
            dist_cases.append(case(f"example_{nwk}", {"newick": (EXAMPLES / nwk).read_text().strip(), "taxa": taxa}, {"dist_matrix": D, "max": float(D.max())}, "1e-6",
                                   "signature: compute_fast_dist_matrix(tree, taxa) -> float32 [N,N]; depths accumulated in float64 from the root (None branch length = 0), "
                                   "d(i,j) = depth_i + depth_j - 2 depth_lca, stored float32. Tree first passed through enforce_nonzero_branch_lengths(min_len=1e-4). taxa = tree terminal order filtered to alignment names."))
            coords = compute_mds_coordinates(D, 4, mds_sign="canonical")
            mds_cases.append(case(f"example_{nwk}_raw_distances", {"dist_matrix": D, "n_components": 4, "mds_sign": "canonical"},
                                  {"coords": coords, "gram": coords @ coords.T, "abs_coords": np.abs(coords)}, "1e-5", mds_sig, **MDS_RELATIVE))
    # rescaled bat (what the pipeline actually feeds the model)
    tree, taxa, D = example_trees["bat_oas1.nwk"]
    Lb = len(next(iter(parse_alignment_sequences(str(EXAMPLES / "bat_oas1.fasta")).values()))) // 3
    Db = (D / Lb).astype(np.float32)
    coords = compute_mds_coordinates(Db, 4, mds_sign="canonical")
    mds_cases.append(case("example_bat_oas1_rescaled_by_L", {"dist_matrix": Db, "n_components": 4, "mds_sign": "canonical"}, {"coords": coords, "gram": coords @ coords.T, "abs_coords": np.abs(coords)}, "1e-5",
                          mds_sig + f" Input is the bat_oas1 patristic matrix divided by L={Lb} (the >10 rescale rule).", **MDS_RELATIVE))
    rng = np.random.default_rng(SYNTH_SEED + 5)
    pts = rng.normal(size=(6, 2))
    Ds = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)).astype(np.float32)
    coords = compute_mds_coordinates(Ds, 4, mds_sign="canonical")
    mds_cases.append(case("synthetic_euclidean_2d_points_6", {"dist_matrix": Ds, "n_components": 4, "mds_sign": "canonical"}, {"coords": coords, "gram": coords @ coords.T, "abs_coords": np.abs(coords)}, "1e-5",
                          mds_sig + " Exact Euclidean distances from 2-D points: components 3 and 4 have eigenvalue ~0 and are numerically noise-scaled (sqrt of tiny positive or clipped 0).", **MDS_RELATIVE))
    D3 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], np.float32)
    coords = compute_mds_coordinates(D3, 4, mds_sign="canonical")
    mds_cases.append(case("equilateral_3_points_padding", {"dist_matrix": D3, "n_components": 4, "mds_sign": "canonical"}, {"coords": coords, "gram": coords @ coords.T, "abs_coords": np.abs(coords)}, "1e-5",
                          mds_sig + " N=3 < n_components: eigvecs has only 3 columns so the 4th is zero-padded; degenerate eigenvalues make individual columns basis-dependent, use 'gram'.", **MDS_RELATIVE))
    w.write("dataset", "compute_fast_dist_matrix", dist_cases)
    w.write("dataset", "compute_mds_coordinates", mds_cases)

    # --- load_alignment_and_tree on Smc6 and bat_oas1 (full assembly)
    lat_sig = ("signature: load_alignment_and_tree(fa_path, nwk_path=None, max_species=None, prune_duplicates=True, use_tn93=False, mds_sign=None -> HYPHAEON_MDS_SIGN or 'canonical') -> "
               "(c [L,N,1] int64, a [L,N,1] int64, d [1,N,N] float32, z [1,N,4] float32, is_aa_invariable [L] bool, taxa, L). "
               "Steps: parse -> tree -> enforce_nonzero_branch_lengths -> taxa = tree terminal order kept if in alignment (fallbacks: quote-stripped, then case-insensitive) -> "
               "prune identical sequences -> L = len(first seq)//3 -> patristic matrix -> if max > 10: divide by L -> MDS -> tokens -> invariable = <=1 distinct valid aa (tokens < 20).")
    lat_cases = []
    for fa, nwk in (("Smc6.fasta", "Smc6.nwk"), ("bat_oas1.fasta", "bat_oas1.nwk")):
        c, a, d, z, inv, taxa, L = load_alignment_and_tree(str(EXAMPLES / fa), str(EXAMPLES / nwk), mds_sign="canonical")
        raw_D = example_trees[nwk][2]
        zc = z[0].numpy()
        lat_cases.append(case(f"example_{fa}", {"alignment": fa, "tree": nwk, "max_species": None, "prune_duplicates": True, "mds_sign": "canonical"},
                              {"taxa": taxa, "L": L, "N": len(taxa), "codon_tokens": c[:, :, 0].numpy(), "aa_tokens": a[:, :, 0].numpy(),
                               "dist_matrix": d[0].numpy(), "rescaled_by_L": bool(raw_D.max() > 10.0), "raw_dist_max": float(raw_D.max()),
                               "mds_coords": zc, "mds_gram": zc @ zc.T, "is_aa_invariable": inv, "n_invariable": int(inv.sum())},
                              "1e-5", lat_sig + " Tokens, taxa, L and the invariable mask are exact; dist_matrix 1e-6; mds_coords 1e-5 EXACT per column under the canonical sign convention (see compute_mds_coordinates notes)."))
    w.write("dataset", "load_alignment_and_tree", lat_cases)

    # --- >10 rescale rule on a synthetic tree
    fa_text = ">s1\nATGAAACCCGGGTTT\n>s2\nATGAAACCCGGGTTC\n>s3\nATGAAACCTGGGTTC\n>s4\nATGAAGCCTGGATTC\n>s5\nATGAAGCCTGGATTA\n"
    (inputs_dir / "rescale_5taxa.fa").write_text(fa_text)
    rescale_cases = []
    for name, nwk_text, note in (
        ("rescale_triggered", "((s1:12.0,s2:8.0):4.0,(s3:6.5,s4:3.0):2.0,s5:9.0);", "max patristic distance 12+4+2+6.5 = 24.5 > 10 -> every entry divided by L=5"),
        ("rescale_not_triggered", "((s1:1.2,s2:0.8):0.4,(s3:0.65,s4:0.3):0.2,s5:0.9);", "same topology scaled by 0.1: max 2.45 <= 10 -> unchanged"),
        ("rescale_boundary_exactly_10", "((s1:5.0,s2:5.0):0.0,(s3:1.0,s4:1.0):1.0,s5:1.0);", "max path s1-s2 = 10.0 exactly, which is NOT > 10, so no rescale; the zero-length internal branch is raised to 1e-4 but is not on that path"),
        ("rescale_zero_branches_tip_over_10", "((s1:5.0,s2:1.0):0.0,(s3:5.0,s4:1.0):0.0,s5:1.0);", "max path s1-s3 sums to 10.0 in the file but crosses two zero-length internal branches that enforce_nonzero_branch_lengths raises to 1e-4, so max = 10.0002 > 10 and the rescale fires"),
    ):
        (inputs_dir / f"{name}.nwk").write_text(nwk_text + "\n")
        c, a, d, z, inv, taxa, L = load_alignment_and_tree(str(inputs_dir / "rescale_5taxa.fa"), str(inputs_dir / f"{name}.nwk"))
        tree = extract_tree_from_string_or_file(nwk_text); enforce_nonzero_branch_lengths(tree, min_len=1e-4)
        raw = compute_fast_dist_matrix(tree, taxa)
        outputs = {"taxa": taxa, "L": L, "raw_dist_matrix": raw, "raw_dist_max": float(raw.max()), "rescaled": bool(raw.max() > 10.0), "dist_matrix": d[0].numpy()}
        rescale_cases.append(case(name, {"alignment": "rescale_5taxa.fa", "tree": f"{name}.nwk", "newick": nwk_text, "fasta": fa_text}, outputs, "1e-6",
                                  "rule in load_alignment_and_tree: `if dist_mat.max() > 10.0: dist_mat = dist_mat / L` (L = codon count, here 5). " + note))
    w.write("dataset", "rescale_rule", rescale_cases)

    # --- prune_identical_sequences
    seqs = {"a": "ATGAAA", "b": "ATGAAA", "c": "ATGAAC", "d": "ATGAAA", "e": "ATGAAC", "f": "atgaaa", "g": "ATG---"}
    prune_cases = [
        case("planted_duplicates", {"seq_dict": seqs, "taxa": list("abcdefg")}, dict(zip(("unique_taxa", "dup_map", "num_pruned"), prune_identical_sequences(seqs, list("abcdefg")))), "exact",
             "signature: prune_identical_sequences(seq_dict, taxa) -> (unique_taxa in first-seen order, dup_map rep -> [dups], num_pruned). Byte-identical only: 'atgaaa' != 'ATGAAA' at this level (parse upper-cases earlier); gapped variants are distinct."),
        case("taxa_order_matters", {"seq_dict": seqs, "taxa": list("gfedcba")}, dict(zip(("unique_taxa", "dup_map", "num_pruned"), prune_identical_sequences(seqs, list("gfedcba")))), "exact", "representative is the first taxon in the given order"),
        case("subset_of_taxa", {"seq_dict": seqs, "taxa": ["b", "d"]}, dict(zip(("unique_taxa", "dup_map", "num_pruned"), prune_identical_sequences(seqs, ["b", "d"]))), "exact", ""),
        case("no_duplicates", {"seq_dict": {"x": "AAA", "y": "AAC"}, "taxa": ["x", "y"]}, dict(zip(("unique_taxa", "dup_map", "num_pruned"), prune_identical_sequences({"x": "AAA", "y": "AAC"}, ["x", "y"]))), "exact", ""),
    ]
    w.write("dataset", "prune_identical_sequences", prune_cases)
    # duplicates through load_alignment_and_tree (tree pruning side effect: dropped taxa disappear from taxa list, tree untouched)
    dup_fa = ">s1\nATGAAACCCGGGTTT\n>s2\nATGAAACCCGGGTTT\n>s3\nATGAAACCTGGGTTC\n>s4\nATGAAACCTGGGTTC\n>s5\nATGAAGCCTGGATTA\n"
    (inputs_dir / "duplicates_5taxa.fa").write_text(dup_fa)
    nwk_text = "((s1:0.1,s2:0.1):0.2,(s3:0.15,s4:0.05):0.1,s5:0.3);"
    (inputs_dir / "duplicates_5taxa.nwk").write_text(nwk_text + "\n")
    out = []
    for prune in (True, False):
        c, a, d, z, inv, taxa, L = load_alignment_and_tree(str(inputs_dir / "duplicates_5taxa.fa"), str(inputs_dir / "duplicates_5taxa.nwk"), prune_duplicates=prune)
        out.append(case(f"prune_duplicates_{prune}", {"alignment": "duplicates_5taxa.fa", "tree": "duplicates_5taxa.nwk", "fasta": dup_fa, "newick": nwk_text, "prune_duplicates": prune},
                        {"taxa": taxa, "L": L, "dist_matrix": d[0].numpy(), "codon_tokens": c[:, :, 0].numpy(), "is_aa_invariable": inv}, "1e-6",
                        "duplicates s2 (of s1) and s4 (of s3) are dropped from taxa when prune_duplicates; the distance matrix is computed over the retained taxa only (the tree itself is not re-pruned)."))
    w.write("dataset", "load_alignment_and_tree_duplicates", out)

    # --- Faith's PD downsampling
    pd_sig = ("signature: downsample_taxa_faith_pd(dist_mat, taxa, max_species) -> (sub_dist_mat, selected_taxa). Farthest-point traversal: start with the argmax pair "
              "(np.unravel_index(np.argmax(dist_mat)) i.e. first maximum in row-major order), then repeatedly add argmax of the running min-distance vector (first index on ties). "
              "Returns the input unchanged when max_species >= n or <= 0. Selected order is the selection order, not the original order.")
    pd_cases = []
    rng = np.random.default_rng(SYNTH_SEED + 6)
    pts = rng.normal(size=(10, 3))
    Dsyn = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)).astype(np.float32)
    syn_taxa = [f"t{i}" for i in range(10)]
    sub, sel = downsample_taxa_faith_pd(Dsyn, syn_taxa, 4)
    pd_cases.append(case("synthetic_10_to_4", {"dist_mat": Dsyn, "taxa": syn_taxa, "max_species": 4}, {"sub_dist_mat": sub, "selected_taxa": sel}, "1e-6", pd_sig))
    sub, sel = downsample_taxa_faith_pd(Dsyn, syn_taxa, 10)
    pd_cases.append(case("max_species_equals_n", {"dist_mat": Dsyn, "taxa": syn_taxa, "max_species": 10}, {"sub_dist_mat": sub, "selected_taxa": sel}, "exact", pd_sig))
    Dtie = np.array([[0, 2, 2, 1], [2, 0, 2, 1], [2, 2, 0, 1], [1, 1, 1, 0]], np.float32)
    sub, sel = downsample_taxa_faith_pd(Dtie, list("abcd"), 3)
    pd_cases.append(case("ties_first_index_wins", {"dist_mat": Dtie, "taxa": list("abcd"), "max_species": 3}, {"sub_dist_mat": sub, "selected_taxa": sel}, "exact", pd_sig))
    # camelid through the pipeline (HyPhy-estimated branch lengths) at max_species 64
    env = python_env()
    c, a, d, z, inv, taxa64, L = load_alignment_and_tree(str(EXAMPLES / "camelid.fasta"), str(EXAMPLES / "camelid.nwk"), max_species=64)
    tree = extract_tree_from_string_or_file(str(EXAMPLES / "camelid.nwk"))
    seqs = parse_alignment_sequences(str(EXAMPLES / "camelid.fasta"))
    all_taxa = [t.name.strip("'\"") for t in tree.get_terminals() if t.name and t.name.strip("'\"") in seqs]
    from hyphaeon.dataset import estimate_tree_branch_lengths_hyphy
    est = estimate_tree_branch_lengths_hyphy(seqs, tree) if shutil.which("hyphy") else None
    if est is not None:
        enforce_nonzero_branch_lengths(est, min_len=1e-4)
        unique_taxa, _, _ = prune_identical_sequences(seqs, all_taxa)
        stride = max(1, len(unique_taxa) // (64 * 2))
        pre = unique_taxa[::stride][:64 * 2]
        Dpre = compute_fast_dist_matrix(est, pre)
        if Dpre.max() > 10.0:
            Dpre = Dpre / L
        sub, sel = downsample_taxa_faith_pd(Dpre, pre, 64)
        pd_cases.append(case("camelid_128_to_64_hyphy_branch_lengths", {"dist_mat": Dpre, "taxa": pre, "max_species": 64},
                             {"sub_dist_mat_diagonal_check": float(np.abs(np.diag(sub)).max()), "selected_taxa": sel, "selected_taxa_via_load_alignment_and_tree": taxa64,
                              "match": sel == taxa64}, "exact",
                             pd_sig + f" examples/camelid.nwk has no branch lengths, so load_alignment_and_tree estimated them with HyPhy ({env['hyphy']}); "
                             "the resulting 128x128 matrix is captured here as the INPUT so this case does not depend on HyPhy at replay time. "
                             "QUIRK replicated: before the Faith's PD step load_alignment_and_tree strides to at most 2*max_species taxa "
                             "(stride = max(1, n // (2*max_species)) which is 1 for 212 taxa, so it simply keeps the FIRST 128 unique taxa in tree order); "
                             "sub_dist_mat omitted for size (64x64 of the given input rows/cols), diagonal check only."))
    else:
        pd_cases.append(case("camelid_128_to_64_hyphy_branch_lengths_SKIPPED", {}, {"selected_taxa_via_load_alignment_and_tree": taxa64}, "exact", "hyphy not on PATH; only the pipeline result is recorded"))
    w.write("dataset", "downsample_taxa_faith_pd", pd_cases)

    # ------------------------------------------------------------------
    # TN93 (the tree-free path, PLAN.md D22)
    # ------------------------------------------------------------------
    from hyphaeon.dataset import compute_tn93_distance_matrix
    from tn93.tn93 import TN93
    import tn93 as tn93_pkg

    tn = TN93()   # exactly dataset.py:541: verbose=0, ignore_gaps=False, max_ambig_fraction=1.0, minimum_overlap=500
    tn93_version = getattr(tn93_pkg, "__version__", None) or importlib_version("tn93")
    dist_sig = (
        "signature (as dataset.py:544-547 composes it, NOT the package's tn93_distance wrapper): "
        "tn = TN93(); counts = tn.get_counts(seq1, seq2, 'resolve'); freq = tn.get_nucleotide_frequency(counts); "
        "d = tn.calculate_distance(counts, freq). Options are the TN93() constructor defaults the reference takes: "
        "match_mode 'resolve', max_ambig_fraction 1.0, ignore_gaps False, verbose 0; minimum_overlap (500) is never "
        f"applied because tn93_distance is not called. tn93 package {tn93_version}. float64; every returned distance is "
        "rounded to 6 SIGNIFICANT digits (np.format_float_positional(precision=6, unique=False, fractional=False)). "
        "counts and nucleotide_frequency are the intermediates, pinned so a port can localise a disagreement. "
        "An 'error' output means the Python RAISES there and dataset.py does not catch it."
    )

    def tn93_case(name, seq1, seq2, note="", match_mode="resolve", ignore_gaps=False):
        engine = tn if not ignore_gaps else TN93(ignore_gaps=True)
        try:
            counts = engine.get_counts(seq1, seq2, match_mode)
            freq = engine.get_nucleotide_frequency(counts)
            d = engine.calculate_distance(counts, freq)
            outputs = {"counts": counts, "nucleotide_frequency": freq, "distance": float(d)}
        except Exception as exc:
            outputs = {"error": type(exc).__name__, "error_message": str(exc)}
        inputs = {"seq1": seq1, "seq2": seq2, "match_mode": match_mode}
        if ignore_gaps:
            inputs["ignore_gaps"] = True
        return case(name, inputs, outputs, "1e-9", dist_sig + (" " + note if note else ""))

    stem = "ATGAAACCCGGGTTTACGTACGTAAACCCGGGTTTAAACCCGGGTTTAAACCCGGGTTT"
    dist_cases_tn93 = [
        tn93_case("identical", stem, stem, "Identical sequences: dist is exactly 0.0 (the `dist > -0.0` test at tn93.py:225 sends it to 0.0)."),
        tn93_case("one_transition", stem, stem[:-1] + "C", "One T->C transition in the last position."),
        tn93_case("one_transversion", stem, stem[:-1] + "A", "One T->A transversion."),
        tn93_case("gaps_in_one", "ATG---" + stem[6:], stem[:-1] + "C", "RESOLVE counts a gap facing a base (only gap/gap is skipped, tn93.py:331) but the gap's resolutionsCount is 0.0, so the position adds nothing."),
        tn93_case("terminal_gaps", "---" + stem[3:-3] + "---", stem, "find_terminal_gaps runs but RESOLVE never reads its result (only GAPMM with ignore_gaps does)."),
        tn93_case("N_in_one", stem[:9] + "NNN" + stem[12:], stem[:-1] + "C", "N (4-fold) facing a base resolves to a MATCH under RESOLVE."),
        tn93_case("iupac_every_code", "ATGRYSWKMBDHVN" + stem[14:], stem, "One of every IUPAC code: those containing the other sequence's base become matches, the others spread 1/|resolutions| over their bases."),
        tn93_case("iupac_vs_iupac", "RYSWKMBDHVNN" + stem[12:], "YRWSMKVDHBNN" + stem[12:], "Both ambiguous: shared resolutions add 1/k to the DIAGONAL cells, otherwise the product of the weights is spread over k1 x k2 cells."),
        tn93_case("lowercase", stem.lower(), stem[:-1] + "C", "map_character maps both cases (tn93.py:542-557)."),
        tn93_case("uracil", stem.replace("T", "U"), stem[:-1] + "C", "U is code 4, resolving to T only; note parse_alignment_sequences already replaces U with T upstream."),
        tn93_case("question_mark_and_unknown_char", "AT?XZ" + stem[5:], stem, "'?' and any character not in the table map to 16, which resolves to any base (tn93.py:524, 577)."),
        tn93_case("unequal_lengths_longer_truncated", stem + "AAACCCGGG", stem, "range(min(len1, len2)): the longer sequence is simply cut, never padded."),
        tn93_case("single_base_only", "AAAAAAAAAAAA", "AAAAAAAAAAAG", "C and T never occur, so `0 in nucleotide_frequency` takes the degenerate branch (tn93.py:199-205), not the TN93 formula."),
        tn93_case("all_positions_resolvable_forces_average", "RRRRRRRRRRRR", "AAAAAAAAAAAA", "ambig_fraction_too_high: every non-gap position is resolvable, so with max_ambig_fraction 1.0 the pair is scored by get_counts_AVERAGE (tn93.py:293-297)."),
        tn93_case("saturated_raises", "ATGCATGCATGCATGC", "GCTAGCTAGCTAGCTA", "SATURATED: the corrected proportion goes negative and math.log RAISES ValueError. dataset.py does not catch it, so a reference run dies here."),
        tn93_case("no_overlap_raises", "ATGCATGC--------", "--------ATGCATGC", "No position where both are non-gap: the count matrix is empty and `2 / sum(freq)` raises ZeroDivisionError (tn93.py:190)."),
        tn93_case("all_gap_raises", "------------", "------------", "Same ZeroDivisionError from the other direction."),
        tn93_case("all_N_raises", "NNNNNNNNNNNN", "NNNNNNNNNNNN", "Every position resolvable -> AVERAGE -> counts spread evenly, no zero frequency, and math.log(0.0) raises."),
    ]
    _base = "ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC"
    gappy1 = "--" + _base[2:20] + "RYN" + _base[23:]
    gappy2 = _base[:5] + "YRS" + _base[8:38] + "--"
    dist_cases_tn93 += [
        tn93_case("mode_average", gappy1, gappy2, "match_mode='average' (tn93.py:383-431): ANY gap skips the position and an ambiguity is always spread over its resolutions, never matched first. Not the reference's mode; pinned because RESOLVE falls back to it via ambig_fraction_too_high.", match_mode="average"),
        tn93_case("mode_gapmm", gappy1, gappy2, "match_mode='gapmm' (tn93.py:433-490): a gap facing a base is treated as an N. Not on the reference's path.", match_mode="gapmm"),
        tn93_case("mode_gapmm_ignore_terminal_gaps", gappy1, gappy2, "match_mode='gapmm' with TN93(ignore_gaps=True): the loop runs over [first_nongap, last_nongap) where last_nongap = min(end1, end2) - 1, so the LAST non-gap position is excluded - a package quirk, replicated.", match_mode="gapmm", ignore_gaps=True),
        tn93_case("mode_skip", gappy1, gappy2, "match_mode='skip' (tn93.py:492-518): only positions where BOTH characters are plain bases are counted. Not on the reference's path.", match_mode="skip"),
    ]
    ex_seqs = parse_alignment_sequences(str(EXAMPLES / "bat_oas1.fasta"))
    ex_names = list(ex_seqs.keys())
    dist_cases_tn93.append(tn93_case("example_bat_oas1_first_pair", ex_seqs[ex_names[0]], ex_seqs[ex_names[1]],
                                     f"Real data, full length: examples/bat_oas1.fasta taxa '{ex_names[0]}' and '{ex_names[1]}'."))

    matrix_sig = (
        "signature: compute_tn93_distance_matrix(seq_dict, taxa, fa_path=None, threshold=100.0) -> float32 [n, n]. dataset.py:715-822. "
        "PREFERS the compiled `tn93` binary (`tn93 -t 100.0 -l 1 -q -o out.csv in.fa`, default ambiguity strategy resolve, "
        "retried at -t 1.0 when a stock build refuses the threshold); these fixtures hide it so the Python `tn93` package "
        "path runs. The matrix starts at -1.0 with a zeroed diagonal, so an entry never written is outside the range of "
        "any distance. Off-diagonal i<j only, mirrored; `calculate_distance` is wrapped in "
        "`except (ValueError, OverflowError): d = 1.0`, then `if d is None or d == '-' or d < 0 or isnan(d): d = 1.0` "
        "(dead on this path); then the imputation: ONLY entries still below zero are filled, with max(1.0, max_d) where "
        "max_d is read once from the filled matrix; the diagonal is zeroed last. A MEASURED 0.0 survives as 0.0, whether "
        "the two sequences are byte-identical or merely at distance zero. Since the package path writes every pair, "
        "nothing is ever imputed on it; only a pair the binary omits from its CSV can be. n <= 1 returns the -1.0 matrix "
        "with its zeroed diagonal, which for n == 1 is the zero matrix."
    )
    matrix_cases = []
    with tn93_python_package_path():
        for fa in ("bat_oas1.fasta", "Smc6.fasta", "camelid.fasta"):
            seqs = parse_alignment_sequences(str(EXAMPLES / fa))
            taxa = list(seqs.keys())
            D = compute_tn93_distance_matrix(seqs, taxa)
            sat = int(sum(1 for i in range(len(taxa)) for j in range(i + 1, len(taxa)) if D[i, j] == 1.0))
            matrix_cases.append(case(f"example_{fa}", {"alignment": fa, "taxa": taxa},
                                     {"dist_matrix": D, "max": float(D.max()), "saturated_pairs_at_1.0": sat}, "1e-9",
                                     matrix_sig + " taxa are list(seq_dict.keys()), the alignment order the tree-free path uses."))
        seqs = parse_alignment_sequences(str(EXAMPLES / "HIV1_RT.fasta"))
        taxa = list(seqs.keys())
        Dh = compute_tn93_distance_matrix(seqs, taxa)
        sat_pairs = [[i, j] for i in range(len(taxa)) for j in range(i + 1, len(taxa)) if Dh[i, j] == 1.0]
        matrix_cases.append(case("example_HIV1_RT.fasta_reduced", {"alignment": "HIV1_RT.fasta", "taxa": taxa},
                                 {"n": len(taxa), "first_row": Dh[0], "last_row": Dh[-1], "max": float(Dh.max()),
                                  "min_off_diagonal": float(np.min(Dh[~np.eye(len(taxa), dtype=bool)])),
                                  "saturated_pairs_index": sat_pairs, "saturated_pairs_at_1.0": len(sat_pairs)},
                                 "1e-9", matrix_sig + " REDUCED FOR SIZE: a 476x476 matrix does not fit the 2 MB cap, so only the first and last rows, "
                                                     "the extrema and the indices of the saturated pairs are stored; replay recomputes the whole matrix and checks those."))
        if sat_pairs:
            i, j = sat_pairs[0]
            dist_cases_tn93.append(tn93_case(f"example_HIV1_RT_pair_at_the_sentinel", seqs[taxa[i]], seqs[taxa[j]],
                                             f"An HIV1_RT pair sitting at 1.0 in the matrix: taxa '{taxa[i]}' and '{taxa[j]}' (indices {i}, {j} in alignment order). "
                                             "Across all five bundled alignments NO pair reaches the degenerate 1.0 sentinel of calculate_distance, and under the "
                                             "reconciled imputation no measured zero is raised either, so this case is generated only if such a pair exists. HIV1_RT has "
                                             "1 byte-identical pair and 66 different-sequence pairs at distance 0, RHO has 78 and 8; all of them now stay at 0.0. "
                                             "load_alignment_and_tree prunes the identical ones before the matrix is built."))
        # synthetic cases for the assembly rules
        synth = {
            "distinct_sequences_at_zero_stay_zero": ({"s1": "ATGCATGCATGCATGCATGC", "s2": "ATGCATGCATGCATGCATG-", "s3": "ATGCATGCATGCATGCTTGC"},
                                     "s1/s2 differ only by a terminal gap, which RESOLVE counts as nothing: their distance is a MEASURED 0.0 and stays 0.0. Under the rule this replaced it was raised to 1e-4 because the strings differ."),
            "identical_strings_stay_zero": ({"s1": "ATGCATGCATGCATGCATGC", "s2": "ATGCATGCATGCATGCATGC", "s3": "ATGCATGCATGCATGCTTGC"},
                                                  "s1 and s2 are byte-identical and measure 0.0, which is kept. Under the rule this replaced, that zero was indistinguishable from an unwritten entry and became max(1.0, max_d) = 1.0, LARGER than every real distance in the matrix - the reference bug this repairs. load_alignment_and_tree prunes duplicates before this anyway, so it needed prune_duplicates=False to be reached."),
            "all_identical_stays_zero": ({"s1": "ATGCATGCATGCATGCATGC", "s2": "ATGCATGCATGCATGCATGC", "s3": "ATGCATGCATGCATGCATGC"},
                                         "Every pair measures 0 and every pair is written, so nothing is imputed: the matrix stays all zeros."),
            "single_taxon": ({"s1": "ATGCATGCATGCATGCATGC"}, "n <= 1 returns without computing anything (dataset.py:734-735); the 1x1 matrix is its own zeroed diagonal."),
        }
        for name, (sd, note) in synth.items():
            tx = list(sd.keys())
            D = compute_tn93_distance_matrix(sd, tx)
            matrix_cases.append(case(name, {"seq_dict": sd, "taxa": tx}, {"dist_matrix": D, "max": float(D.max())}, "1e-9", matrix_sig + " " + note))
    w.write("dataset", "tn93_distance", dist_cases_tn93)
    w.write("dataset", "tn93_distance_matrix", matrix_cases)

    # --- load_alignment_and_tree(use_tn93=True): the whole tree-free assembly
    tn93_lat_sig = (
        "signature: load_alignment_and_tree(fa_path, nwk_path=None, max_species=None, prune_duplicates=True, use_tn93=True, mds_sign='canonical'). "
        "dataset.py:598-636 then the shared tail: taxa = list(seq_dict.keys()) in ALIGNMENT order (no tree, no matching) -> prune identical sequences -> "
        "L = len(first seq)//3 -> stride pre-selection when max_species -> compute_tn93_distance_matrix -> Faith's PD when still over max_species -> MDS -> tokens -> invariable. "
        "NO `> 10` rescale and NO enforce_nonzero_branch_lengths on this path. The tn93 binary is hidden so the Python package computes the distances."
    )
    lat_tn93 = []
    with tn93_python_package_path():
        for fa in ("bat_oas1.fasta", "camelid.fasta"):
            c, a, d, z, inv, taxa, L = load_alignment_and_tree(str(EXAMPLES / fa), None, use_tn93=True, mds_sign="canonical")
            zc = z[0].numpy()
            dm = d[0].numpy()
            ct = c[:, :, 0].numpy()
            at = a[:, :, 0].numpy()
            token_sha = lambda arr: hashlib.sha256(",".join(str(int(v)) for v in arr.reshape(-1)).encode()).hexdigest()
            outputs = {
                "taxa": taxa, "L": L, "N": len(taxa), "dist_matrix": dm, "max_distance": float(dm.max()),
                "saturated_pairs_at_1.0": int(sum(1 for i in range(len(taxa)) for j in range(i + 1, len(taxa)) if dm[i, j] == 1.0)),
                "mds_coords": zc, "is_aa_invariable": inv, "n_invariable": int(inv.sum()),
                "codon_tokens_sha256": token_sha(ct), "aa_tokens_sha256": token_sha(at),
            }
            note = tn93_lat_sig + " Tokens are pinned by the sha256 of their row-major [L, N] values joined with ',' (the full arrays would blow the 2 MB cap); taxa, L and the invariable mask are exact, distances 1e-9, mds_coords 1e-5 EXACT per column under the canonical sign convention."
            if fa == "bat_oas1.fasta":
                outputs["codon_tokens"] = ct
                outputs["aa_tokens"] = at
                outputs["mds_gram"] = zc @ zc.T
                note += " bat_oas1 also carries the full token arrays and the sign-invariant Gram matrix; camelid omits both for size (212x212 twice does not fit the 2 MB cap)."
            lat_tn93.append(case(f"example_{fa}", {"alignment": fa, "tree": None, "use_tn93": True, "max_species": None, "prune_duplicates": True, "mds_sign": "canonical"}, outputs, "1e-5", note))
    w.write("dataset", "load_alignment_and_tree_tn93", lat_tn93)


# --------------------------------------------------------------------------
# dates (no model, no torch, no weights: four pure string parsers)
# --------------------------------------------------------------------------

# The four date parsers of the time-aware pillars. They are ordinary ported reference functions, so
# their tables live here rather than under js/test/data/ (fixtures/README.md: test/data/<module>/ is
# for cases this generator CANNOT produce).
DATE_FUNCTIONS = {
    "hyphaeon/temporal.py": ["parse_date_to_decimal", "extract_date_from_string"],
    "hyphaeon/dating.py": ["parse_header_timestamp", "_parse_timestamp_flexible"],
}


def load_date_reference(engine_root: Path = REPO):
    """
    Execute the four date functions out of the checked-out Python, WITHOUT importing hyphaeon.

    Both modules import torch at module scope (temporal.py:42; dating.py imports inference), and the
    date parsers need none of it: importing either would make four string tables depend on a full
    model environment. So the function definitions are lifted out of the source by `ast` — by name,
    with their exact line ranges returned alongside — and executed in a namespace holding only what
    their bodies reference (`re`, `datetime`, `np`, `pd`). Nothing is retyped, so a drift between the
    reference body and the generated table is impossible, and the line ranges are the audit trail
    (js/test/dates.test.js asserts them).

    `js/test/data/dates/gen.py` imports this function for the one table it still owns.

    @returns (namespace, {function name: "<file>:<first line>-<last line>"})
    """
    import datetime as _datetime
    import re as _re

    import pandas as _pd

    ns: Dict[str, Any] = {"re": _re, "datetime": _datetime, "np": np, "pd": _pd, "Any": Any}
    provenance: Dict[str, str] = {}
    for rel, names in DATE_FUNCTIONS.items():
        path = engine_root / rel
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        by_name = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name in names:
            node = by_name[name]
            provenance[name] = f"{rel}:{node.lineno}-{node.end_lineno}"
            exec(compile(ast.get_source_segment(source, node), str(path), "exec"), ns)  # noqa: S102
    return ns, provenance


# Each block is (what it pins, [values]). The blocks are flattened in order, so the tables' lengths
# and order are fixed by this file; a case's `notes` carries its block's sentence.

# parse_date_to_decimal, calendar (temporal.py:73-151).
DATE_CALENDAR_BLOCKS = [
    ("plain numbers and the [1800, 2100] gate (temporal.py:104-108, 115-120); a Python bool is an int, so True is 1.0 and fails the gate",
     [2021.25, 1959.5, 1800, 2100, 1799.999, 2100.001, 0, -1, 1e9, True, False]),
    ("missing / null tokens (temporal.py:86, 111); the token test is case-folded, so 'NaN' and 'NA' are null but '??' is not",
     [None, "", "   ", "unknown", "UNKNOWN", "nan", "NaN", "none", "NA", "na", "?", "??"]),
    ("decimal-year strings (temporal.py:115-120) through CPython float(): underscore separators, exponents, a leading '+', and the inf spellings",
     ["2021.25", " 2021.25 ", "2021", "2_021", "2.0212e3", "+2021", "inf", "-inf"]),
    ("ISO and partial ISO on the delimiter path (temporal.py:126-147) -- Q2: a missing month defaults to June and a missing day to the 15th, silently",
     ["2021-04-15", "2021-04", "2021-01-01", "2021-12-31", "2020-02-29", "2021-02-28",
      "2021-XX-XX", "2021-04-XX", "2021-4-5", "2021-004-015"]),
    ("'/' and '.' are both normalised to '-' (temporal.py:123) -- Q5, and only AFTER float() has claimed every plain decimal year",
     ["2021/04/15", "2021.04.15", "2021.4.15", "2021/4", "2021.5", "2021.05"]),
    ("the day cap min(day, 28 if month == 2 else 30) (temporal.py:144) -- Q3: the 31st of a 31-day month moves back a day and 29 February is unreachable",
     ["2021-01-31", "2021-03-31", "2021-02-29", "2020-02-29", "2021-02-30", "2021-12-30"]),
    ("an out-of-range month or day is discarded and the default kept, with no error (temporal.py:133-142) -- Q4",
     ["2021-13-15", "2021-00-15", "2021-04-00", "2021-04-32", "2021-04-31"]),
    ("out of the gate, and unparseable (temporal.py:129-130, 151) -- Q6: the reference returns NaN for both and the two are indistinguishable to it",
     ["1700-04-15", "2500-04-15", "0021-04-15", "21-04-15", "202-04-15", "20210415",
      "notadate", "2021-04-15T09:30:00Z", "15-04-2021", "April 2021"]),
    ("the leap rule and the 365/366 divisor (temporal.py:146) across a non-leap century, a leap century and an ordinary leap year",
     ["1900-03-01", "2000-03-01", "2004-03-01", "1996-12-31"]),
]

# parse_date_to_decimal, non-calendar (temporal.py:90-102). Replayed under each of the three units.
DATE_NON_CALENDAR_BLOCKS = [
    ("any non-negative real, else the first embedded number; no year gate -- Q10: float('inf') >= 0.0 is True, so Infinity is a valid coordinate and NaN is not",
     [0, 5000, 5000.5, -1, -0.0, "5000", "5000.5", "-5000", "gen_5000", "generation 42",
      "t=17.5", "abc", "", None, "nan", "inf", "-inf", "  12  ", 1e9, True]),
]

# extract_date_from_string, calendar (temporal.py:217-243).
DATE_HEADER_CALENDAR_BLOCKS = [
    ("the four calendar header patterns and the delimiter class [|/_\\s] they require -- Q7: '-' is NOT a delimiter, pattern 2 needs two to four decimals, and pattern 4 is anchored at the end with no '^' alternative",
     ["A/USA/CA-1/2021|2021-05-14", "seq_2021-05-14", "seq 2021-05-14", "A/USA/2021-05-14/2021",
      "2021-05-14", "A-2021-05-14", "hCoV-19/England/X|2021.35", "hCoV-19/England/X|2021.3512",
      "hCoV-19/England/X|2021.5", "strain_2021-05", "strain|2021", "strain/2021", "2021",
      "strain_2021", "strain-2021", "EPI_ISL_402124|2019-12-30|China", "no_date_here", "",
      "X|9999-05-14", "X|1700-05-14", "X|2021-13-14"]),
]

# extract_date_from_string, non-calendar (temporal.py:204-215).
DATE_HEADER_NON_CALENDAR_BLOCKS = [
    ("the three non-calendar header patterns, whose delimiter class DOES include '-' and which are case-insensitive; the bare-number rule takes the FIRST delimiter-bound number in the name",
     ["pop1_gen2000", "pop1|gen_5000", "pop1_20000gen", "pop1_g50000", "lineage|5000",
      "lineage_generation_120", "sample_day-7", "sample_d30", "sample_t12.5", "sample_D30",
      "clone7", "7clone", "no_number", "", "x_2021-05-14"]),
]

# parse_header_timestamp (dating.py:317-364). Every name containing Z59, ZR59 or 1959 is answered
# 1959.5 by the reference before any pattern is tried -- Q1.
DATE_HEADER_TIMESTAMP_BLOCKS = [
    ("the hard-coded archival anchor (dating.py:330-332) -- Q1: an unbounded substring test that overwrites a real date and claims an unrelated accession",
     ["Z59ZR.ZHU", "ZR59", "AZ59012", "A/Kinshasa/1959-03-04", "X|1959-03-04", "X|2019-03-04"]),
    ("the Korber HIV-1 isolate code ^[A-Za-z]\\d\\d[A-Za-z]{2}[._] with the two-digit year pivoted at 30 and mid-year added (dating.py:341-345) -- Q8",
     ["B86US.SFMHS18", "F93BE_VI850", "C86ET.ETH2220", "A85UG.U455", "D84ZR.84ZR085",
      "B29US.X", "B30US.X", "b86us.SFMHS18"]),
    ("the LANL _XX_86_ year, which requires an UPPERCASE country code (dating.py:348-352) -- Q8",
     ["Ref_B_FR_83_HXB2", "Ref_C_ET_86_ETH2220", "Ref_C_et_86_ETH2220"]),
    ("weeks and days post infection, returned raw onto the same axis as a decimal year (dating.py:355-362) -- Q9: (?:WPI|wpi) matches neither 'Wpi' nor 'wPI'",
     ["patient1_16WPI", "patient1_16 WPI", "patient1_16wpi", "patient1_16Wpi",
      "patient1_120DPI", "patient1_120.5dpi"]),
    ("the calendar patterns it delegates to first (dating.py:335), and the two ways of having no date at all",
     ["seq_2021-05-14", "", "nothing_here"]),
]

# _parse_timestamp_flexible (dating.py:367-384).
DATE_FLEXIBLE_BLOCKS = [
    ("the calendar reading first, then a RAW float() with no gate at all (dating.py:378-381): this is the only way the dating pillar reaches a non-calendar coordinate, and it accepts negatives",
     [None, "", "unknown", "nan", "2021-04-15", "2021.25", 2021.25, "5000", 5000, 0.25, "0.25",
      "-3", -3, "1700", 1700, "inf", "abc", True, "1e6"]),
]


def _flat(blocks):
    return [v for _, values in blocks for v in values]


# Flat lists, for js/test/data/dates/gen.py, which replays the same inputs through one hand-written
# variant of parse_header_timestamp.
CALENDAR_VALUES = _flat(DATE_CALENDAR_BLOCKS)
NON_CALENDAR_VALUES = _flat(DATE_NON_CALENDAR_BLOCKS)
HEADER_NAMES_CALENDAR = _flat(DATE_HEADER_CALENDAR_BLOCKS)
HEADER_NAMES_NON_CALENDAR = _flat(DATE_HEADER_NON_CALENDAR_BLOCKS)
HEADER_TIMESTAMP_NAMES = _flat(DATE_HEADER_TIMESTAMP_BLOCKS)
FLEXIBLE_VALUES = _flat(DATE_FLEXIBLE_BLOCKS)

# The measured real-data cases, BY INDEX into the bundled alignment so the string itself is never
# retyped. MEASURED at this commit with the reference's own extract_date_from_string:
#   examples/H1N1_2009_pandemic.fasta   95 of 100 headers dated; these five are the misses, for two
#                                       distinct reasons (see the notes on each block)
#   examples/korber_env_gp160.fasta     0 of 143 dated by temporal's parser; all 142 that dating's
#                                       parse_header_timestamp dates come from the Korber rule,
#                                       which temporal does not have (CONSENSUS is the 143rd)
H1N1_UNDATED_HEADER_INDICES = [46, 57, 76, 87, 97]
KORBER_HEADER_INDICES = [0, 1, 142]

_H1N1_ISOLATE_NUMBER_SHADOW = (
    "pattern 2 matches the FIRST \\d{4}\\.\\d{2,4} in the header, which here is the isolate number "
    "(4218.01, 1127.02, 4016.02); that fails the [1800, 2100] gate, re.search never offers the second "
    "match, and the real decimal year in the last pipe field is then unreachable because pattern 4 is "
    "anchored at '$' -- Q6 + Q7."
)
_H1N1_ONE_DECIMAL = (
    "pattern 2 requires TWO to four decimals, so a decimal year written to one place (|2009.4, |2009.8) "
    "is not matched at all; pattern 4's '$' anchor then leaves the header undated -- Q7."
)
# MEASURED per header, not derived: which of the two defects drops this one.
H1N1_UNDATED_REASONS = {
    46: _H1N1_ISOLATE_NUMBER_SHADOW, 57: _H1N1_ISOLATE_NUMBER_SHADOW, 76: _H1N1_ISOLATE_NUMBER_SHADOW,
    87: _H1N1_ONE_DECIMAL, 97: _H1N1_ONE_DECIMAL,
}


def fasta_headers(path: Path) -> List[str]:
    """The '>' lines of a FASTA, marker stripped and stripped, in file order."""
    return [line[1:].strip() for line in path.read_text().splitlines() if line.startswith(">")]


def _date_case_name(index: int, value: Any) -> str:
    """`NNN_<repr>`: the index keeps names unique (a value may appear in two blocks), the repr shows
    whitespace and quoting, which is half of what these cases are about."""
    body = "".join(ch if ch.isprintable() else "?" for ch in repr(value))
    if len(body) > 60:
        body = body[:57] + "..."
    return f"{index:03d}_{body}"


def gen_dates(w: Writer) -> None:
    ns, provenance = load_date_reference()
    parse_date_to_decimal = ns["parse_date_to_decimal"]
    extract_date_from_string = ns["extract_date_from_string"]
    parse_header_timestamp = ns["parse_header_timestamp"]
    parse_timestamp_flexible = ns["_parse_timestamp_flexible"]

    def sig(fn: str, call: str) -> str:
        return f"signature: {call}. Reference {provenance[fn]}, lifted by ast and executed without importing hyphaeon."

    # ---- parse_date_to_decimal -------------------------------------------------------------
    cases: List[Dict[str, Any]] = []
    i = 0
    for note, values in DATE_CALENDAR_BLOCKS:
        for v in values:
            cases.append(case(_date_case_name(i, v), {"val": v, "time_units": "years"},
                              {"result": parse_date_to_decimal(v, "years")}, "exact",
                              sig("parse_date_to_decimal", "parse_date_to_decimal(val, time_units='years') -> float") + " " + note))
            i += 1
    for units in ("generations", "days", "arbitrary"):
        for note, values in DATE_NON_CALENDAR_BLOCKS:
            for v in values:
                cases.append(case(_date_case_name(i, v), {"val": v, "time_units": units},
                                  {"result": parse_date_to_decimal(v, units)}, "exact",
                                  sig("parse_date_to_decimal", f"parse_date_to_decimal(val, time_units={units!r}) -> float") + " " + note))
                i += 1
    # An unrecognised time_units takes the CALENDAR path, because the reference's test is
    # `if time_units in ('generations', 'days', 'arbitrary')` and not a membership of the known set.
    for v in ["2021-04-15", "5000", 5000]:
        cases.append(case(_date_case_name(i, v), {"val": v, "time_units": "fortnights"},
                          {"result": parse_date_to_decimal(v, "fortnights")}, "exact",
                          sig("parse_date_to_decimal", "parse_date_to_decimal(val, time_units='fortnights') -> float")
                          + " an unknown time_units is calendar, because the reference tests membership of the non-calendar tuple only."))
        i += 1
    w.write("dates", "parse_date_to_decimal", cases)

    # ---- extract_date_from_string ----------------------------------------------------------
    cases = []
    i = 0
    for note, values in DATE_HEADER_CALENDAR_BLOCKS:
        for v in values:
            cases.append(case(_date_case_name(i, v), {"name": v, "time_units": "years"},
                              {"result": extract_date_from_string(v, "years")}, "exact",
                              sig("extract_date_from_string", "extract_date_from_string(name, time_units='years') -> float") + " " + note))
            i += 1
    for note, values in DATE_HEADER_NON_CALENDAR_BLOCKS:
        for v in values:
            cases.append(case(_date_case_name(i, v), {"name": v, "time_units": "generations"},
                              {"result": extract_date_from_string(v, "generations")}, "exact",
                              sig("extract_date_from_string", "extract_date_from_string(name, time_units='generations') -> float") + " " + note))
            i += 1

    # The real-data proof. Five headers of the 100 in the shipped H1N1 alignment are undated by this
    # function, for two distinct reasons, and both are quirks a reading of the source can miss.
    h1n1 = fasta_headers(EXAMPLES / "H1N1_2009_pandemic.fasta")
    for k in H1N1_UNDATED_HEADER_INDICES:
        name = h1n1[k]
        reason = H1N1_UNDATED_REASONS[k]
        cases.append(case(_date_case_name(i, name), {"name": name, "time_units": "years"},
                          {"result": extract_date_from_string(name, "years")}, "exact",
                          sig("extract_date_from_string", "extract_date_from_string(name, time_units='years') -> float")
                          + f" MEASURED: examples/H1N1_2009_pandemic.fasta header {k} (0-based) is one of the 5 of 100 this function cannot date. " + reason))
        i += 1

    korber = fasta_headers(EXAMPLES / "korber_env_gp160.fasta")
    for k in KORBER_HEADER_INDICES:
        name = korber[k]
        cases.append(case(_date_case_name(i, name), {"name": name, "time_units": "years"},
                          {"result": extract_date_from_string(name, "years")}, "exact",
                          sig("extract_date_from_string", "extract_date_from_string(name, time_units='years') -> float")
                          + f" MEASURED: examples/korber_env_gp160.fasta header {k} (0-based). This function dates 0 of that file's 143 headers;"
                            " dating.parse_header_timestamp dates 142 of them, all through the Korber rule temporal.py does not have."))
        i += 1
    w.write("dates", "extract_date_from_string", cases)

    # ---- parse_header_timestamp ------------------------------------------------------------
    # The reference VERBATIM, 1959 anchor included. js/src/dates.js makes that anchor opt-in, so the
    # port reproduces this table only under {archival1959: true}; the default is measured against
    # js/test/data/dates/parsers.json, which has no reference function of its own (Q1).
    cases = []
    i = 0
    for note, values in DATE_HEADER_TIMESTAMP_BLOCKS:
        for v in values:
            cases.append(case(_date_case_name(i, v), {"name": v},
                              {"result": parse_header_timestamp(v)}, "exact",
                              sig("parse_header_timestamp", "parse_header_timestamp(name) -> float") + " " + note))
            i += 1
    w.write("dates", "parse_header_timestamp", cases)

    # ---- _parse_timestamp_flexible ---------------------------------------------------------
    cases = []
    i = 0
    for note, values in DATE_FLEXIBLE_BLOCKS:
        for v in values:
            cases.append(case(_date_case_name(i, v), {"val": v},
                              {"result": parse_timestamp_flexible(v)}, "exact",
                              sig("_parse_timestamp_flexible", "_parse_timestamp_flexible(val) -> float") + " " + note))
            i += 1
    w.write("dates", "_parse_timestamp_flexible", cases)

    return {"date_reference_source": provenance}


# --------------------------------------------------------------------------
# attribution + dms (model-dependent, bat_oas1 only)
# --------------------------------------------------------------------------

def _load_bat_model():
    import torch
    from hyphaeon.inference import load_model, prepare_alignment
    device = torch.device("cpu")
    torch.manual_seed(0)
    model = load_model(weights=str(MODEL_PATH), variant=None, device=device)
    c, a, d, z, inv, taxa, L, cache = prepare_alignment(str(EXAMPLES / "bat_oas1.fasta"), str(EXAMPLES / "bat_oas1.nwk"), model=model, device=device)
    return device, model, c, a, d, z, inv, taxa, L, cache


def gen_attribution(w: Writer, model_sha: str) -> None:
    from hyphaeon.attribution import attribute_selection
    from hyphaeon.inference import predict_site_lrts
    device, model, c, a, d, z, inv, taxa, L, cache = _load_bat_model()
    lrts = predict_site_lrts(model, c, a, d, z, inv, tree_cache=cache, batch_size=64, device=device, progress=False)
    attrs = attribute_selection(model, c, a, d, z, inv, taxa=taxa, min_lrt=3.84, base_lrts=lrts, cache=cache)
    w.write("attribution", "attribute_selection", [case(
        "bat_oas1_min_lrt_3.84_cpu",
        {"alignment": "bat_oas1.fasta", "tree": "bat_oas1.nwk", "min_lrt": 3.84, "focal_sites": None, "model_safetensors_sha256": model_sha, "device": "cpu", "batch_size": 64},
        {"base_lrts": lrts, "attributions": {str(k): v for k, v in attrs.items()}, "n_sites_attributed": len(attrs)},
        "1e-5",
        "signature: attribute_selection(model, c, a, d, z, inv, taxa=None, focal_sites=None, min_lrt=3.84, base_lrts=None, cache=None) -> {site0 -> record}. "
        "For each site with base LRT >= min_lrt: consensus codon = most frequent codon token (np.unique + argmax: lowest token on ties); every non-consensus taxon is reverted "
        "to the consensus codon/aa one at a time and re-scored; delta = site_lrt - mod_lrt; pct = max(0, delta/site_lrt*100); mean_patristic_depth = row mean of d; "
        "epoch by weighted depth / max distance (>=0.60 tip, >=0.35 intermediate, else deep); mode = >=2 drivers with pct>=10 -> recurrent. Everything numeric is model output (1e-5); strings and integers exact. "
        "base_lrts come from inference.predict_site_lrts over variable sites (invariable -> 0).")])


def gen_dms(w: Writer, model_sha: str) -> None:
    from hyphaeon.epistasis import run_insilico_selection_dms
    from hyphaeon.inference import predict_site_lrts
    device, model, c, a, d, z, inv, taxa, L, cache = _load_bat_model()
    lrts = predict_site_lrts(model, c, a, d, z, inv, tree_cache=cache, batch_size=64, device=device, progress=False)
    target = [int(s) for s in np.where(lrts >= 3.84)[0]]
    res = run_insilico_selection_dms(model, c, a, cache, taxa, device, focal_taxon=None, batch_size=64, progress=False, target_sites=target)
    res2 = run_insilico_selection_dms(model, c, a, cache, taxa, device, focal_taxon="r_ferr", batch_size=64, progress=False, target_sites=target[:2])
    notes = ("signature: run_insilico_selection_dms(model, c, a, tree_cache, taxa, device, focal_taxon=None, batch_size=None, progress=True, target_sites=None) -> list of per-site records. "
             "target_sites (0-indexed) IS supported. focal taxon = first taxon whose lower-cased name contains focal_taxon, else index 0. Baseline LRT is recomputed over ALL sites "
             "(not just variable ones, unlike meme). For each target site, the focal taxon is replaced by the canonical codon of each of the 19 other amino acids "
             "(CANONICAL_AA_TO_CODON); wt with token >= 20 falls back to the site's majority aa (bincount argmax) or 'A'. p_value = Self-Liang of baseline. Numeric 1e-5, keys/strings exact.")
    w.write("dms", "run_insilico_selection_dms", [
        case("bat_oas1_sites_lrt_ge_3.84_focal_default", {"alignment": "bat_oas1.fasta", "tree": "bat_oas1.nwk", "focal_taxon": None, "target_sites": target, "batch_size": 64,
                                                          "model_safetensors_sha256": model_sha, "device": "cpu"},
             {"focal_index": 0, "focal_name": taxa[0], "plasticity": res}, "1e-5", notes),
        case("bat_oas1_two_sites_focal_r_ferr", {"alignment": "bat_oas1.fasta", "tree": "bat_oas1.nwk", "focal_taxon": "r_ferr", "target_sites": target[:2], "batch_size": 64,
                                                 "model_safetensors_sha256": model_sha, "device": "cpu"},
             {"focal_index": [i for i, t in enumerate(taxa) if "r_ferr" in t.lower()][0], "focal_name": [t for t in taxa if "r_ferr" in t.lower()][0], "plasticity": res2}, "1e-5", notes),
    ])


# --------------------------------------------------------------------------
# e2e via the CLI
# --------------------------------------------------------------------------

def gen_e2e(w: Writer, model_sha: str, only_cases: List[str] | None = None) -> None:
    import torch
    hy = shutil.which("hyphaeon") or str(Path(sys.executable).parent / "hyphaeon")
    env = {**os.environ, "HYPHAEON_WEIGHTS": str(MODEL_PATH), "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}
    penv = python_env()

    wall_times: Dict[str, float] = {}

    def wanted(name: str) -> bool:
        return only_cases is None or any(sub in name for sub in only_cases)

    def run_cli(name: str, argv: List[str], notes: str, tolerance: str = "1e-5", postprocess=None, env_extra: Dict[str, str] | None = None,
                post_kwargs=None) -> None:
        if not wanted(name):
            return
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.json"
            full = [hy] + argv + ["-o", str(out)]
            t0 = time.time()
            proc = subprocess.run(full, cwd=REPO, env={**env, **(env_extra or {})}, capture_output=True, text=True)
            if proc.returncode != 0 or not out.exists():
                raise RuntimeError(f"{name}: exit {proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
            data = json.load(open(out))
            wall_times[name] = round(time.time() - t0, 1)
        data = null_timings(basenames(data))
        if postprocess is not None:
            data, notes = postprocess(data, notes, **(post_kwargs(argv) if post_kwargs else {}))
        cmd_display = ["hyphaeon"] + [os.path.basename(x) if x.startswith("examples/") else x for x in argv] + ["-o", "<out.json>"]
        w.write("e2e", name, [case(name, {"argv": cmd_display, "model_safetensors_sha256": model_sha, "device": "cpu", "torch": penv["torch"], "hyphy": penv["hyphy"]},
                                   data, tolerance, notes + " Timing fields nulled (wall times are in manifest.json e2e_wall_seconds); paths reduced to basenames.")])

    # BustedMultiTaskHead parameters that model.safetensors does not carry are randomly initialised on every run
    # (cmd_busted loads the head with strict=False and never seeds torch), so those outputs cannot be pinned.
    from safetensors import safe_open
    from hyphaeon.model import BustedMultiTaskHead
    from hyphaeon.weights import load_arch_config
    with safe_open(str(MODEL_PATH), "pt") as f:
        present = {k[len("head_busted."):] for k in f.keys() if k.startswith("head_busted.")}
    needed = set(BustedMultiTaskHead(embed_dim=load_arch_config(str(MODEL_PATH))["embed_dim"]).state_dict().keys())
    busted_missing = sorted(needed - present)
    BUSTED_RANDOM_FIELDS = ("predicted_gene_lrt", "selection_probability", "synonymous_rate_variation", "positive_selection_detected")

    def busted_site_lrts(argv: List[str], env_extra: Dict[str, str] | None) -> Dict[str, Any]:
        """`hyphaeon meme` on the busted run's inputs, with cmd_busted's own taxon cap (-s 512, cli.py:1108):
        cmd_busted computes per-site LRTs (cli.py:454-467) but never writes them, and cmd_meme's are the same
        forward passes on the same tokens. The caller asserts that the float32 sums reproduce the busted
        record's omnibus_lrt and total_selection_energy bit for bit, so the equivalence is checked at
        generation rather than assumed."""
        meme_argv = ["meme"] + [a for a in argv[1:]] + ["-s", "512"]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "meme.json"
            proc = subprocess.run([hy] + meme_argv + ["-o", str(out)], cwd=REPO, env={**env, **(env_extra or {})}, capture_output=True, text=True)
            if proc.returncode != 0 or not out.exists():
                raise RuntimeError(f"busted_site_lrts: meme exit {proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
            meme = json.load(open(out))
        sites = meme["sites"]
        return {"site_lrts": np.array([s["hyphaeon_lrt"] for s in sites], dtype=np.float32),
                "is_invariable": [bool(s["is_invariable"]) for s in sites], "taxa_count": meme["taxa_count"]}

    def busted_post(data, notes, sites=None):
        if sites is not None:
            lrts = sites["site_lrts"]
            energy = float(np.sum(lrts))
            omnibus = float(np.sum(np.maximum(0.0, lrts - 3.841)))
            if sites["taxa_count"] != data["taxa"] or len(lrts) != data["sites"]:
                raise RuntimeError(f"busted_post: meme run has {sites['taxa_count']} taxa x {len(lrts)} sites, busted {data['taxa']} x {data['sites']}")
            if energy != data["total_selection_energy"] or omnibus != data["omnibus_lrt"]:
                raise RuntimeError(f"busted_post: the meme LRTs do not reproduce the busted sums: energy {energy!r} vs {data['total_selection_energy']!r}, "
                                   f"omnibus {omnibus!r} vs {data['omnibus_lrt']!r}")
            data["site_lrts"] = lrts
            data["is_invariable"] = sites["is_invariable"]
            notes += (" site_lrts (float32, one per codon; invariable sites 0) and is_invariable are `hyphaeon meme` on the same inputs with cmd_busted's -s 512 cap: "
                      "cmd_busted computes them (cli.py:454-467) but does not write them; float32 np.sum of site_lrts and of max(0, site_lrts - 3.841) reproduce "
                      "total_selection_energy and omnibus_lrt EXACTLY (asserted when this fixture was generated), so a replay can start from these two arrays.")
        if busted_missing:
            for k in BUSTED_RANDOM_FIELDS:
                data[k] = None
            data["rate_distributions"] = {k: (None if k in ("proportion_1", "proportion_2", "proportion_3", "omega_3") else v) for k, v in data["rate_distributions"].items()}
            notes += (" NEURAL HEAD FIELDS NULLED: model.safetensors lacks these BustedMultiTaskHead parameters " + ", ".join(busted_missing) +
                      " so they are random on every run (strict=False, no seed); predicted_gene_lrt, selection_probability, synonymous_rate_variation, "
                      "omega_3/proportion_* and positive_selection_detected (which depends on selection_probability) are therefore null here and must not be compared. "
                      "omega_1/omega_2 are constants (0.10, 1.00).")
        return data, notes

    meme_note = ("`hyphaeon meme` JSON (cli.py cmd_meme). p_value/q_value are float32 casts of the float64 stats. Per-site LRT through the model is 1e-5; "
                 "is_invariable exact. RHO has an embedded NEXUS tree and 710 taxa (> 500, so MDS uses the Lanczos path). "
                 "MDS eigenvector signs are canonical (--mds-sign canonical: largest-|entry| of each kept eigenvector positive); the model is not sign-invariant, so a port must use the same convention.")
    MDS = ["--mds-sign", "canonical"]   # the fixtures' sign convention, recorded in argv (also the CLI default)
    SEED = ["--seed", str(PERM_SEED)]
    run_cli("meme_bat_oas1", ["meme", "-a", "examples/bat_oas1.fasta", "-t", "examples/bat_oas1.nwk", "--cpu"] + MDS, meme_note)
    run_cli("meme_Smc6", ["meme", "-a", "examples/Smc6.fasta", "-t", "examples/Smc6.nwk", "--cpu"] + MDS, meme_note)
    run_cli("meme_camelid", ["meme", "-a", "examples/camelid.fasta", "-t", "examples/camelid.nwk", "--cpu"] + MDS, meme_note + " camelid.nwk has no branch lengths: HyPhy HKY85 estimation ran first, so this fixture depends on the HyPhy version recorded in inputs.")
    run_cli("meme_HIV1_RT", ["meme", "-a", "examples/HIV1_RT.fasta", "-t", "examples/HIV1_RT.nwk", "--cpu"] + MDS, meme_note)
    run_cli("meme_RHO", ["meme", "-a", "examples/RHO.fasta", "--cpu"] + MDS, meme_note)
    run_cli("meme_bat_oas1_attribute_filter", ["meme", "-a", "examples/bat_oas1.fasta", "-t", "examples/bat_oas1.nwk", "--cpu", "--attribute", "--filter"] + MDS,
            meme_note + " --attribute adds attributions keyed by 1-indexed site; --filter runs the cli.py copy of the hypergeometric+OCI screen (which differs slightly from filter.py: no '?' or length check on consensus codons).")
    busted_note = ("`hyphaeon busted` JSON (cli.py cmd_busted): Self-Liang site p, ACAT over variable sites, Simes over all sites, omnibus_lrt = sum max(0, lrt-3.841), "
                   "neural BustedMultiTaskHead outputs (selection_probability, predicted_gene_lrt, synonymous_rate_variation, omega proportions). Head outputs 1e-5; counts exact.")
    busted_sites = lambda argv: {"sites": busted_site_lrts(argv, None)}
    run_cli("busted_Smc6", ["busted", "-a", "examples/Smc6.fasta", "-t", "examples/Smc6.nwk", "--cpu"] + MDS, busted_note, postprocess=busted_post, post_kwargs=busted_sites)
    run_cli("busted_HIV1_RT", ["busted", "-a", "examples/HIV1_RT.fasta", "-t", "examples/HIV1_RT.nwk", "--cpu"] + MDS, busted_note, postprocess=busted_post, post_kwargs=busted_sites)
    run_cli("epistasis_Smc6_n_permutations_1000",
            ["epistasis", "-a", "examples/Smc6.fasta", "-t", "examples/Smc6.nwk", "--cpu", "--n-permutations", "1000"] + SEED + MDS,
            "`hyphaeon epistasis` JSON (epistasis.run_epistatic_analysis). CLI defaults: min_sim 0.30, min_shared 2, max_fdr 0.05, min_lrt 1.0, min_cesi 2.0 (function default, CLI has no flag), "
            "min_coherence 0.50, rng_seed 42 (--seed 42, the CLI default). MDS signs canonical (--mds-sign canonical). edges: cosine/t/BH on float32 attributions (1e-5 through the model); sectors: membership exact given identical edges, "
            "p_perm/null_* statistical (B=1000, PCG64 seed 42); plasticity: DMS over sector sites (1e-5). Duplicate collections (edges/coselection_edges etc.) are the Python's.", tolerance="statistical")
    run_cli("phenotype_RHO_marine_n_permutations_0",
            ["phenotype", "-a", "examples/RHO.fasta", "-fg", RHO_MARINE_FOREGROUND, "--n-permutations", "0", "--cpu"] + SEED + MDS,
            "`hyphaeon phenotype` JSON (phenotype.run_phenotype_association) with the README Example 3 foreground, --permulations 0 (parametric p) and --n-permutations 0 "
            "(trait sector permutation block degenerate); --seed 42 (the CLI default; seeds permulations and the sector permutation null) and --mds-sign canonical. Site stats 1e-5 through the model; p_evd/score tracks 1e-9 given identical inputs; sector membership exact.")

    # --- tree-free TN93 (--use-tn93, PLAN.md D22). The compiled tn93 binary is hidden from the
    # subprocess so the Python `tn93` package computes the distances, which is what the JS port
    # mirrors (they measured identical here; see tn93_python_package_path).
    TN93_ENV = {"PATH": path_without_tn93_binary()}
    tn93_note = (
        " TREE-FREE: --use-tn93 skips the tree entirely (dataset.py:598-636). taxa are list(seq_dict.keys()) in ALIGNMENT order "
        "(no tree matching, no tree terminal order), the distance matrix is compute_tn93_distance_matrix (match mode 'resolve', "
        "max_ambig_fraction 1.0, no minimum overlap) and there is no `> 10` rescale. The tn93 BINARY is hidden from this run's PATH "
        "so the Python tn93 package path is what is pinned."
    )
    run_cli("meme_camelid_tn93", ["meme", "-a", "examples/camelid.fasta", "--use-tn93", "--cpu"] + MDS,
            meme_note + tn93_note + " camelid.nwk is NOT used, so unlike meme_camelid this case does not depend on HyPhy at all.", env_extra=TN93_ENV)
    run_cli("meme_HIV1_RT_tn93", ["meme", "-a", "examples/HIV1_RT.fasta", "--use-tn93", "--cpu"] + MDS,
            meme_note + tn93_note + " HIV1_RT has one byte-identical taxon pair; duplicate pruning removes it before the matrix is built (475 taxa here), so none of the "
                                    "1.0 imputations of fixtures/dataset/tn93_distance_matrix.json survive into this run.", env_extra=TN93_ENV)
    run_cli("busted_Smc6_tn93", ["busted", "-a", "examples/Smc6.fasta", "--use-tn93", "--cpu"] + MDS,
            busted_note + tn93_note, postprocess=busted_post, env_extra=TN93_ENV,
            post_kwargs=lambda argv: {"sites": busted_site_lrts(argv, TN93_ENV)})
    run_cli("epistasis_Smc6_tn93",
            ["epistasis", "-a", "examples/Smc6.fasta", "--use-tn93", "--cpu", "--n-permutations", "1000"] + SEED + MDS,
            "`hyphaeon epistasis` JSON with the tree replaced by TN93 distances; CLI defaults as in epistasis_Smc6_n_permutations_1000 (min_sim 0.30, min_shared 2, max_fdr 0.05, "
            "min_lrt 1.0, min_cesi 2.0, min_coherence 0.50, rng_seed 42), B = 1000." + tn93_note, tolerance="statistical", env_extra=TN93_ENV)

    # filter: the `filter` subcommand writes only a FASTA and an audit CSV (no JSON), so the result dict is taken from
    # filter.run_alignment_filter, the function cmd_filter calls, with the CLI's default thresholds.
    from hyphaeon.filter import run_alignment_filter
    for name, fa, nwk in (("filter_bat_oas1", "bat_oas1.fasta", "bat_oas1.nwk"), ("filter_camelid", "camelid.fasta", "camelid.nwk")):
        if not wanted(name):
            continue
        with tempfile.TemporaryDirectory() as td:
            res = run_alignment_filter(str(EXAMPLES / fa), str(EXAMPLES / nwk), weights_path=str(MODEL_PATH), output_alignment_path=str(Path(td) / "clean.fa"),
                                       audit_csv_path=str(Path(td) / "audit.csv"), device=torch.device("cpu"), progress=False)
            cleaned = (Path(td) / "clean.fa").read_text()
        res = null_timings(basenames(res))
        masked = {}
        for art in res["artifacts"]:
            masked.setdefault(art["outlier_taxon"], []).append([art["patch_start_1idx"], art["patch_end_1idx"]])
        w.write("e2e", name, [case(name, {"call": "hyphaeon.filter.run_alignment_filter(alignment, tree, alpha_site=0.05, min_k=3, max_span=35, min_oci=0.25, min_run_length=3)",
                                          "alignment": fa, "tree": nwk, "model_safetensors_sha256": model_sha, "device": "cpu", "hyphy": penv["hyphy"]},
                                   {"result": res, "masked_codon_ranges_1idx_by_taxon": masked, "cleaned_alignment_sha256": hashlib.sha256(cleaned.encode()).hexdigest()},
                                   "1e-5",
                                   "OCI is inline in filter.run_alignment_filter (not a separate function) so this is an end-to-end fixture: baseline LRTs (model, 1e-5) -> MEME p -> "
                                   "scan_hypergeometric_patches -> per-patch consensus codons (pandas mode: lowest codon string on ties) -> per-taxon max mismatch run and count -> "
                                   "top taxon by (run, count) -> OCI = count/total -> artifact if (run>=3 and OCI>=0.25) or run>=4 -> NNN mask -> re-score. "
                                   "artifacts list, masked ranges and cleaned-alignment hash are exact; metrics 1e-5. "
                                   + ("camelid depends on HyPhy branch-length estimation (version in inputs)." if "camelid" in name else "bat_oas1 yields 1 candidate patch and 0 artifacts, exercising the no-mask path."))])
    return {"e2e_wall_seconds": wall_times, "busted_head_missing_keys": busted_missing}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", action="append", choices=["stats", "filter", "evaluation", "epistasis", "phenotype", "dataset", "dates", "attribution", "dms", "e2e"], help="generate only these modules")
    ap.add_argument("--e2e-case", action="append", help="within e2e, generate only the cases whose name contains one of these substrings; the manifest keeps the other cases' counts, sizes and wall times")
    ap.add_argument("--skip-model", action="store_true", help="skip attribution, dms and e2e (no weights needed)")
    ap.add_argument("--skip-e2e", action="store_true")
    args = ap.parse_args()

    wanted = set(args.only) if args.only else {"stats", "filter", "evaluation", "epistasis", "phenotype", "dataset", "dates", "attribution", "dms", "e2e"}
    # Fixtures always record the canonical MDS sign convention: every in-process load_alignment_and_tree call
    # (attribution, dms, the dataset cases that do not pass mds_sign explicitly, filter e2e) reads this env var.
    os.environ["HYPHAEON_MDS_SIGN"] = "canonical"
    if args.skip_model:
        wanted -= {"attribution", "dms", "e2e"}
    if args.skip_e2e:
        wanted -= {"e2e"}

    FIXTURES.mkdir(exist_ok=True)
    manifest_path = FIXTURES / "manifest.json"
    previous = json.load(open(manifest_path)) if manifest_path.exists() else {}
    w = Writer()
    model_sha = sha256_of(MODEL_PATH) if MODEL_PATH.exists() else None
    model_bytes = MODEL_PATH.stat().st_size if MODEL_PATH.exists() else None
    if wanted & {"attribution", "dms", "e2e"} and model_sha is None:
        raise SystemExit(f"model weights not found at {MODEL_PATH}; set HYPHAEON_WEIGHTS or use --skip-model")
    # A partial run that touches no model-dependent module keeps the previous run's weights record:
    # attribution/, dms/ and e2e/ are still on disk and their provenance is still the hash that made
    # them. Nulling it because THIS run needed no weights would falsify those files (and
    # js/test/fixtures.test.js asserts the field is a sha256). `dates` is the module this is for: it
    # is four string parsers and needs no weights, no torch and no model.
    if model_sha is None and not (wanted & {"attribution", "dms", "e2e"}):
        model_sha = previous.get("model_safetensors_sha256")
        model_bytes = previous.get("model_safetensors_bytes")
    t0 = time.time()
    steps = [("stats", lambda: gen_stats(w)), ("filter", lambda: gen_filter(w)), ("evaluation", lambda: gen_evaluation(w)),
             ("epistasis", lambda: gen_epistasis(w)), ("phenotype", lambda: gen_phenotype(w)), ("dataset", lambda: gen_dataset(w)),
             ("dates", lambda: gen_dates(w)),
             ("attribution", lambda: gen_attribution(w, model_sha)), ("dms", lambda: gen_dms(w, model_sha)), ("e2e", lambda: gen_e2e(w, model_sha, args.e2e_case))]
    extra: Dict[str, Any] = {}
    for name, fn in steps:
        if name in wanted:
            print(f"[{name}]")
            extra.update(fn() or {})

    manifest = dict(previous)
    # a partial e2e run keeps the wall times of the cases it did not re-run
    if "e2e_wall_seconds" in extra:
        extra["e2e_wall_seconds"] = {**manifest.get("e2e_wall_seconds", {}), **extra["e2e_wall_seconds"]}
    manifest.update(extra)
    # a partial run (--only / --skip-* / --e2e-case) keeps the previous run's counts for everything
    # it did not touch: the merge is per FUNCTION, not per module, or an --e2e-case run would drop
    # every e2e case it did not regenerate.
    counts = {mod: dict(fns) for mod, fns in manifest.get("counts", {}).items()}
    for mod, fns in w.counts.items():
        counts.setdefault(mod, {}).update(fns)
    sizes = {**manifest.get("bytes", {}), **w.bytes}
    # and any file on disk that no run has ever counted is counted here, so the manifest always
    # describes every fixture (js/test/fixtures.test.js checks the counts against the files).
    for rel in sizes:
        mod, fn = rel.split("/", 1)
        fn = fn[: -len(".json")]
        if fn not in counts.get(mod, {}):
            counts.setdefault(mod, {})[fn] = len(json.load(open(FIXTURES / rel)))
    manifest.update({
        "generator": "scripts/gen_fixtures.py",
        "engine_commit": git_head(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_safetensors_sha256": model_sha,
        "model_safetensors_bytes": model_bytes,
        "environment": python_env(),
        "synthetic_seed": SYNTH_SEED,
        "engine_default_rng_seed": PERM_SEED,
        "mds_sign": "canonical",
        "mds_sign_rule": "dataset.compute_mds_coordinates(mds_sign='canonical'): each kept eigenvector is flipped so its largest-magnitude entry is positive (np.argmax(np.abs(col)), first index on ties) before scaling by sqrt(eigenvalue); CLI --mds-sign / env HYPHAEON_MDS_SIGN, default canonical. 'lapack' keeps the solver's signs and is not what the fixtures record.",
        "rng": {
            "epistasis.compute_sector_permutation_test": "numpy.random.default_rng(rng_seed) -> PCG64; rng.choice(pool, size=K, replace=False)",
            "phenotype.generate_permulations": "numpy legacy global RandomState: np.random.seed(seed); np.random.randn(M, n_perm) (MT19937)",
        },
        "tolerance_classes": {
            "exact": "equality after canonicalisation (tokens, names, integer counts, graph membership, strings)",
            "1e-5": "max |delta| <= 1e-5: anything that passed through the neural model (LRT, attention, attributions) and MDS coordinates (exact per column under the canonical sign convention)",
            "1e-6": "max |delta| <= 1e-6: float32 arithmetic on float32 inputs (patristic distances, cosine networks, coherence)",
            "1e-9": "max |delta| <= 1e-9: special functions and float64 statistics (chi2/t/hypergeometric survival, BH, CCT, correlations, AUC)",
            "statistical": "Monte Carlo outputs: |p_js - p_py| <= 3*sqrt(p(1-p)/B), null moments within 2%; membership/exact fields still compared exactly",
        },
        "known_quirks": [
            "dataset.compute_tn93_distance_matrix PREFERS the compiled `tn93` binary on PATH (`tn93 -t 100.0 -l 1 -q -o out.csv in.fa`, retried at -t 1.0 on CalledProcessError for stock builds <= v1.0.15) and falls back to the Python `tn93` package (TN93().get_counts(a, b, 'resolve') -> get_nucleotide_frequency -> calculate_distance). The fixtures hide the binary so the PACKAGE path is pinned. The earlier measurement that the two agree exactly (max |delta| = 0.0 on bat_oas1 and HIV1_RT) was taken at threshold 1.0 with the old imputation and has NOT been re-taken: the binary now reports pairs it used to drop, and only it can leave an entry unwritten for the imputation to fill, so the two paths are no longer known to be identical.",
            "dataset.compute_tn93_distance_matrix used to impute a zero distance between two BYTE-IDENTICAL sequences as max(1.0, max_d) - the LARGEST distance in the matrix - and raise two different sequences at zero distance to 1e-4. FIXED UPSTREAM: the matrix now starts at -1.0 so a measured zero is distinguishable from an entry that was never written, and only the latter is imputed (dataset.py:816-821). A measured 0.0 survives, from either kind of pair. HIV1_RT has 1 byte-identical pair and 66 different-sequence pairs at distance 0, RHO has 78 and 8; all now stay 0.0.",
            "The tn93 package RAISES rather than returning a sentinel on two inputs: math.log of a non-positive corrected proportion (a saturated pair) -> ValueError, and `2 / sum(nucleotide_frequency)` on a pair with no overlapping non-gap position -> ZeroDivisionError. dataset.py now catches (ValueError, OverflowError) around calculate_distance and calls such a pair maximally distant (d = 1.0); ZeroDivisionError is still uncaught and still kills the run. Its `if d is None or d == '-' or d < 0 or isnan(d)` guard remains dead code on that path. No pair in the five bundled alignments raises either.",
            "tn93.calculate_distance falls back to a fixed 1.0 SENTINEL (not a distance) whenever some base is absent from the pairwise counts and the degenerate correction goes non-positive: 1 pair in HIV1_RT, 78 in RHO, 0 in bat_oas1/Smc6/camelid.",
            "load_alignment_and_tree(use_tn93=True) takes taxa in ALIGNMENT order and applies no `> 10` rescale, where the tree path uses tree terminal order and rescales.",
            "cli.py cmd_busted loads BustedMultiTaskHead with strict=False; model.safetensors lacks 11 of its parameters (see busted_head_missing_keys) and torch is never seeded, so predicted_gene_lrt, selection_probability, synonymous_rate_variation, omega_3, proportion_* and positive_selection_detected differ on every run. They are nulled in fixtures/e2e/busted_*.json.",
            "dataset.load_alignment_and_tree with max_species strides taxa to at most 2*max_species BEFORE Faith's PD; for 212 camelid taxa the stride is 1 so it keeps the first 128 taxa in tree order.",
            "dataset.load_alignment_and_tree rescale rule fires on dist_mat.max() > 10.0 AFTER enforce_nonzero_branch_lengths, so a zero-length branch can tip a max of exactly 10.0 over the threshold.",
            "cli.py cmd_meme --filter duplicates the OCI logic of filter.run_alignment_filter with two differences: consensus codons are not checked for '?' or length 3, and the artifact rule uses --min-patch-consec.",
            "epistasis: CLI default min_sim is 0.30 while compute_branch_coselection_network's default is 0.35; the CLI never passes min_cesi (function default 2.0) or min_shared beyond its own default.",
            "epistasis.extract_epistatic_sectors_tse accepts max_overlap but never uses it.",
            "epistasis and phenotype CLIs expose --seed (default 42) feeding rng_seed / seed; the e2e fixtures pass --seed 42 explicitly.",
            "phenotype.resolve_phenotype_vector ignores `background`; continuous mode z-scores across all taxa including unmatched ones (which contribute 0 before standardisation).",
            "phenotype.resolve_phenotype_vector file mode falls back to substring matching in both directions over dict insertion order, so a short species name can match a longer taxon name.",
            "evaluation.load_meme_json treats -0.0 as not negative (no clamp flag) because the test is raw_lrt < 0.0.",
            "dataset.parse_alignment_sequences PHYLIP branch: an alphanumeric sequence-only continuation line (<= 35 chars) is taken as a taxon name, and a 'name seq' line with <= 10 sequence characters is not a record start; multi-line PHYLIP therefore parses into garbage (pinned in parse_alignment_sequences.json).",
            "dataset.parse_alignment_sequences NEXUS branch: quoted matrix labels containing spaces are split on whitespace.",
            "dataset.extract_tree_from_string_or_file: a NEXUS 'TREE name = [&R] (...);' line is rejected because the rooting annotation precedes '(' (returns None).",
            "filter.scan_hypergeometric_patches reports nothing when every site is significant (P(X>=k)=1 for every window).",
            "stats.benjamini_hochberg uses numpy argsort (quicksort) which is unstable; tied p-values still get identical q so results are order-independent.",
            "DATES Q1: dating.parse_header_timestamp opens with `if 'Z59' in name or 'ZR59' in name or '1959' in name: return 1959.5` (dating.py:330-332), before any date pattern. It is an unbounded substring test on the whole header, so A/Kinshasa/1959-03-04 loses the date that is written there and an accession like AZ59012 is dated to the archival ZR59 isolate. fixtures/dates/parse_header_timestamp.json pins it VERBATIM; js/src/dates.js makes it opt-in ({archival1959: true}) because it is data loss rather than a convention, and js/test/data/dates/parsers.json carries the anchor-off table so both directions stay pinned.",
            "DATES Q2: temporal.parse_date_to_decimal defaults a missing month to 6 and a missing day to 15 on the delimiter path (temporal.py:132, 138) with no flag of any kind, so 2021-XX-XX becomes 2021.45205. Its OWN DOCSTRING (temporal.py:80) is wrong about the commonest case: it promises '2021' -> 2021.5, but a bare year string is claimed by float() at temporal.py:116, passes the gate, and comes back as 2021.0 with nothing imputed.",
            "DATES Q3: the day is capped at min(day, 28 if month == 2 else 30) (temporal.py:144), in EVERY month. 2021-01-31 is returned as 30 January and 29 February of a leap year is unreachable. r0.parse_calendar_date, which uses strptime, disagrees on exactly those days.",
            "DATES Q4: an out-of-range month or day is discarded and the default kept, with no error (temporal.py:133-142), so 2021-13-40 comes back as mid-2021, indistinguishable by value from a masked 2021-XX-XX.",
            "DATES Q5: temporal.parse_date_to_decimal normalises '.' to '-' as well as '/' (temporal.py:123), but only AFTER float() has claimed every plain decimal year. So 2021.25 is a decimal year and 2021.4.15 is 15 April 2021; a decimal year just outside the gate is re-read as a date rather than rejected.",
            "DATES Q6: a year outside [1800, 2100] returns NaN with no distinction from an unparseable string (temporal.py:108, 130). fixtures/dates/parse_date_to_decimal.json pins both; js/src/dates.js separates them into rule 'out_of_range' vs 'unparsed' without changing the value, because a page has to say which happened.",
            "DATES Q7: extract_date_from_string's calendar delimiter class is [\\|/_\\s] and does NOT include '-' (temporal.py:218-239), pattern 2 requires two to four decimals, and pattern 4 has no '^' alternative. MEASURED on examples/H1N1_2009_pandemic.fasta: 5 of 100 headers are undated, three because pattern 2 matches an isolate number (4218.01) that then fails the gate and re.search never offers the second match, and two because their decimal year is written to one place (|2009.4, |2009.8). The non-calendar patterns use a WIDER class that does include '-', and re.IGNORECASE.",
            "DATES Q8: the Korber/LANL two-digit year pivots at 30 (dating.py:344, 351), hard-coded, so 2030 onwards reads as 1930; both rules add a flat 0.5 for mid-year. MEASURED on examples/korber_env_gp160.fasta: temporal.extract_date_from_string dates 0 of 143 headers and dating.parse_header_timestamp dates 142 (CONSENSUS is the miss), all through this rule.",
            "DATES Q9: (?:WPI|wpi) and (?:DPI|dpi) (dating.py:355, 360) match neither 'Wpi' nor 'wPI', while the non-calendar unit patterns are re.IGNORECASE throughout. Both return the elapsed time RAW onto the same axis a decimal year is on, with nothing downstream distinguishing them.",
            "DATES Q10: the non-calendar path accepts Infinity, because float('inf') >= 0.0 is True (temporal.py:92-93), and rejects a negative real and a literal 'nan' by the same comparison without falling through to the embedded-number search.",
        ],
        "counts": counts,
        "bytes": sizes,
        "total_bytes": sum(sizes.values()),
        "elapsed_seconds": round(time.time() - t0, 1),
    })
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"\nmanifest: {manifest_path.relative_to(REPO)}")
    for mod, fns in w.counts.items():
        print(f"  {mod}: {sum(fns.values())} cases in {len(fns)} files")
    print(f"  total {sum(w.bytes.values())/1024:.0f} KB in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
