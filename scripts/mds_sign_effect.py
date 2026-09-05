#!/usr/bin/env python
"""
scripts/mds_sign_effect.py — measure what the MDS eigenvector sign convention does to the reference.

WHY THIS FILE EXISTS

`hyphaeon/dataset.py:compute_mds_coordinates` (cf838ab lines 354-394; now with an `mds_sign`
parameter) feeds the model's `mds_coords` input. Eigenvectors are only defined up to sign, and the
model is NOT sign-invariant (`mds_proj = nn.Linear(4, ...)` on the raw coordinates plus Tree-RoPE),
so whichever sign the eigensolver happens to return changes per-site LRTs. Before this change the
reference kept LAPACK's (or ARPACK's, N > 500) signs and the JavaScript library kept tred2/tql2's;
the two disagreed on bat_oas1 (columns 1, 2) and RHO (columns 2, 3) and missed LRT parity by up to
1e-1 (hyphaeon-app PHASE1.md, gap 1). `mds_sign="canonical"` (largest-magnitude entry of each kept
eigenvector positive, first index on ties, applied before the sqrt(eigenvalue) scaling) makes the
signs a property of the matrix rather than of the solver.

This script quantifies the consequence of switching the DEFAULT from the old behaviour
(`--mds-sign lapack`) to the new one (`--mds-sign canonical`) for existing CLI users, so the ML team
can decide with numbers in front of them. For each bundled example it runs `hyphaeon meme` twice
with the bundled weights on the CPU and reports:

  * which MDS columns flipped (probe: the sign of the largest-|entry| of each LAPACK column),
  * max |ΔLRT| over all sites,
  * median |ΔLRT| / max(1, |LRT_lapack|) over variable sites ("relative", the PLAN.md §5.4 scaling),
  * Spearman ρ between the two LRT vectors over variable sites,
  * how many sites changed their p <= 0.05 call,
  * for Smc6 additionally `hyphaeon busted`'s ACAT / Simes / omnibus LRT under both conventions.

WHAT IT DELIBERATELY DOES NOT DO: it does not touch the JavaScript library, the fixtures, or any
example output. It runs the CLI as a subprocess (fresh process per run) so that the ARPACK start
vector on the Lanczos path (RHO: 655 taxa after duplicate pruning) is the same for both runs — in
one process ARPACK's internal seed advances between calls and a second in-process call would not
reproduce the CLI's signs. The probe for flipped columns runs in its own fresh subprocess for the
same reason (`--probe`).

REQUIREMENTS: the engine installed (`pip install -e .`), `HYPHAEON_WEIGHTS` pointing at the bundled
`model.safetensors`, `HF_HUB_OFFLINE=1`. `examples/camelid.nwk` and `examples/HIV1_RT.nwk` carry no
branch lengths, so those two examples need `hyphy` on PATH (HKY85 branch-length estimation); when it
is absent they are skipped and the report says so.

USAGE (from the repository root):

    HYPHAEON_WEIGHTS=model.safetensors HF_HUB_OFFLINE=1 python scripts/mds_sign_effect.py
    python scripts/mds_sign_effect.py --examples Smc6,bat_oas1 --out /tmp/mds_sign

Writes `<out>/<example>.<analysis>.<mode>.json` (the CLI's -o files), `<out>/report.json`, and
prints the Markdown table that MDS_SIGN.md carries.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"
MODES = ("lapack", "canonical")
ALPHA = 0.05
NEEDS_HYPHY = {"camelid", "HIV1_RT"}   # trees without branch lengths (fixtures/README.md)


# --------------------------------------------------------------------------------------------
# probe: which columns does canonicalisation flip on this example?
# --------------------------------------------------------------------------------------------

def probe(fasta: str, tree: str | None) -> dict:
    """Load the example once with mds_sign='lapack' and report, per MDS column, the sign of the
    largest-magnitude entry (np.argmax on |z|, first index on ties). Canonicalisation flips exactly
    the columns whose pivot is negative; scaling by sqrt(eigenvalue) >= 0 does not move the pivot."""
    sys.path.insert(0, str(REPO))
    from hyphaeon.dataset import load_alignment_and_tree
    c, a, d, z, inv, taxa, L = load_alignment_and_tree(fasta, tree, mds_sign="lapack")
    zc = z[0].numpy()
    flipped = []
    pivots = []
    for j in range(zc.shape[1]):
        col = zc[:, j]
        p = int(np.argmax(np.abs(col)))
        pivots.append({"index": p, "taxon": taxa[p], "value": float(col[p])})
        if col[p] < 0:
            flipped.append(j)
    return {"N": len(taxa), "L": int(L), "n_variable": int((~inv).sum()), "flipped_columns": flipped,
            "pivots": pivots, "solver": "scipy.sparse.linalg.eigsh (Lanczos)" if len(taxa) > 500 else "numpy.linalg.eigh (LAPACK)"}


# --------------------------------------------------------------------------------------------
# running the CLI
# --------------------------------------------------------------------------------------------

def cli_env() -> dict:
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}
    env.setdefault("HYPHAEON_WEIGHTS", str(REPO / "model.safetensors"))
    return env


def run_cli(analysis: str, example: str, spec: dict, mode: str, out_dir: Path, log: list) -> dict:
    out_json = out_dir / f"{example}.{analysis}.{mode}.json"
    cmd = [sys.executable, "-m", "hyphaeon.cli", analysis, "-a", str(spec["fasta"]), "--cpu", "--mds-sign", mode, "-o", str(out_json)]
    if spec["tree"] is not None:
        cmd += ["-t", str(spec["tree"])]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO, env=cli_env(), capture_output=True, text=True)
    secs = round(time.time() - t0, 1)
    log.append({"example": example, "analysis": analysis, "mode": mode, "seconds": secs, "returncode": proc.returncode,
                "command": " ".join(cmd)})
    if proc.returncode != 0 or not out_json.exists():
        raise RuntimeError(f"{analysis} {example} --mds-sign {mode} failed (exit {proc.returncode}):\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
    with open(out_json) as fh:
        return json.load(fh)


def run_probe(example: str, spec: dict) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--probe", str(spec["fasta"])]
    if spec["tree"] is not None:
        cmd.append(str(spec["tree"]))
    proc = subprocess.run(cmd, cwd=REPO, env=cli_env(), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"probe {example} failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
    # the loader prints progress lines; the probe result is the last line
    return json.loads(proc.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------------------------

def compare_meme(ref: dict, alt: dict) -> dict:
    """ref = lapack (today's behaviour), alt = canonical."""
    from scipy.stats import spearmanr
    rs, as_ = ref["sites"], alt["sites"]
    assert [s["site"] for s in rs] == [s["site"] for s in as_], "site order differs"
    lr = np.array([s["hyphaeon_lrt"] for s in rs], dtype=np.float64)
    la = np.array([s["hyphaeon_lrt"] for s in as_], dtype=np.float64)
    pr = np.array([s["p_value"] for s in rs], dtype=np.float64)
    pa = np.array([s["p_value"] for s in as_], dtype=np.float64)
    var = ~np.array([s["is_invariable"] for s in rs], dtype=bool)
    delta = np.abs(la - lr)
    rel = delta[var] / np.maximum(1.0, np.abs(lr[var]))
    rho = float(spearmanr(lr[var], la[var]).statistic) if var.sum() > 2 else float("nan")
    sig_r, sig_a = pr <= ALPHA, pa <= ALPHA
    changed = int((sig_r != sig_a).sum())
    imax = int(np.argmax(delta))
    return {
        "sites": int(len(rs)), "variable_sites": int(var.sum()),
        "max_abs_delta_lrt": float(delta.max()), "max_abs_delta_site": int(rs[imax]["site"]),
        "lrt_lapack_at_max": float(lr[imax]), "lrt_canonical_at_max": float(la[imax]),
        "median_rel_delta_lrt_variable": float(np.median(rel)) if rel.size else float("nan"),
        "spearman_variable": rho,
        "sig_p05_lapack": int(sig_r.sum()), "sig_p05_canonical": int(sig_a.sum()), "sig_p05_calls_changed": changed,
        "sites_beyond_1e-5_class": int((delta > 1e-5 * np.maximum(1.0, np.abs(lr))).sum()),
    }


def compare_busted(ref: dict, alt: dict) -> dict:
    keys = ("p_value_acat", "p_value_simes", "omnibus_lrt", "total_selection_energy", "sig_sites_p05", "sig_sites_p10")
    out = {}
    for k in keys:
        if k in ref and k in alt and ref[k] is not None and alt[k] is not None:
            out[k] = {"lapack": ref[k], "canonical": alt[k], "delta": float(alt[k]) - float(ref[k])}
    return out


# --------------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------------

def fmt_e(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2e}"


def markdown(report: dict) -> str:
    lines = ["| Example | N (after pruning) | Solver | Columns flipped by canonical | max \\|ΔLRT\\| (site) | median rel. \\|ΔLRT\\| (variable) | Spearman ρ (variable) | p ≤ 0.05 calls changed (lapack → canonical) |",
             "|---|---|---|---|---|---|---|---|"]
    for ex in report["examples"]:
        if ex.get("skipped"):
            lines.append(f"| {ex['name']} | — | — | — | — | — | — | skipped: {ex['skipped']} |")
            continue
        pr, m = ex["probe"], ex["meme"]
        cols = ", ".join(str(c) for c in pr["flipped_columns"]) if pr["flipped_columns"] else "none"
        lines.append(f"| {ex['name']} | {pr['N']} | {'Lanczos' if 'eigsh' in pr['solver'] else 'LAPACK eigh'} | {cols} | "
                     f"{fmt_e(m['max_abs_delta_lrt'])} ({m['max_abs_delta_site']}) | {fmt_e(m['median_rel_delta_lrt_variable'])} | "
                     f"{m['spearman_variable']:.6f} | {m['sig_p05_calls_changed']} / {m['variable_sites']} variable ({m['sig_p05_lapack']} → {m['sig_p05_canonical']} significant) |")
    for ex in report["examples"]:
        if ex.get("busted"):
            lines += ["", f"`hyphaeon busted` on {ex['name']} (statistical bridge only; the neural head loads unseeded and is random on every run):", "",
                      "| field | lapack | canonical | Δ |", "|---|---|---|---|"]
            for k, v in ex["busted"].items():
                lines.append(f"| `{k}` | {v['lapack']} | {v['canonical']} | {v['delta']:+.3e} |")
    return "\n".join(lines)


def discover(examples_filter: list[str] | None) -> dict[str, dict]:
    specs = {}
    for fa in sorted(EXAMPLES.glob("*.fasta")):
        name = fa.stem
        if examples_filter and name not in examples_filter:
            continue
        nwk = EXAMPLES / f"{name}.nwk"
        specs[name] = {"fasta": fa.relative_to(REPO), "tree": nwk.relative_to(REPO) if nwk.exists() else None}
    return specs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", nargs="+", metavar="PATH", help=argparse.SUPPRESS)
    ap.add_argument("--examples", default="all", help="comma-separated example stems (default: all examples/*.fasta)")
    ap.add_argument("--out", default=None, help="directory for the CLI outputs and report.json (default: a temporary directory)")
    ap.add_argument("--busted-examples", default="Smc6", help="comma-separated stems to also run `busted` on (default: Smc6)")
    args = ap.parse_args(argv)

    if args.probe:
        print(json.dumps(probe(args.probe[0], args.probe[1] if len(args.probe) > 1 else None)))
        return 0

    out_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="mds_sign_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = None if args.examples == "all" else [x.strip() for x in args.examples.split(",") if x.strip()]
    busted_wanted = {x.strip() for x in args.busted_examples.split(",") if x.strip()}
    specs = discover(wanted)
    hyphy = shutil.which("hyphy")
    hyphy_version = None
    if hyphy:
        try:
            hyphy_version = subprocess.run([hyphy, "--version"], capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
        except Exception:
            hyphy_version = "unknown"

    import torch
    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "weights": cli_env()["HYPHAEON_WEIGHTS"],
              "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(), "platform": platform.platform(),
              "hyphy": hyphy_version, "alpha": ALPHA, "runs": [], "examples": []}
    try:
        report["repo_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip() or None
    except Exception:
        report["repo_commit"] = None

    for name, spec in specs.items():
        entry = {"name": name, "alignment": str(spec["fasta"]), "tree": str(spec["tree"]) if spec["tree"] else "(embedded)"}
        if name in NEEDS_HYPHY and not hyphy:
            entry["skipped"] = f"{spec['tree']} has no branch lengths and hyphy is not on PATH"
            report["examples"].append(entry)
            print(f"[-] {name}: skipped ({entry['skipped']})", flush=True)
            continue
        print(f"[*] {name}: probing MDS columns...", flush=True)
        entry["probe"] = run_probe(name, spec)
        outs = {}
        for mode in MODES:
            print(f"[*] {name}: hyphaeon meme --mds-sign {mode}", flush=True)
            outs[mode] = run_cli("meme", name, spec, mode, out_dir, report["runs"])
        entry["meme"] = compare_meme(outs["lapack"], outs["canonical"])
        if name in busted_wanted:
            bouts = {}
            for mode in MODES:
                print(f"[*] {name}: hyphaeon busted --mds-sign {mode}", flush=True)
                bouts[mode] = run_cli("busted", name, spec, mode, out_dir, report["runs"])
            entry["busted"] = compare_busted(bouts["lapack"], bouts["canonical"])
        report["examples"].append(entry)
        m = entry["meme"]
        print(f"    flipped {entry['probe']['flipped_columns']}  max|dLRT| {m['max_abs_delta_lrt']:.3e}  median rel {m['median_rel_delta_lrt_variable']:.3e}  "
              f"rho {m['spearman_variable']:.6f}  calls changed {m['sig_p05_calls_changed']}", flush=True)

    (out_dir / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nreport: {out_dir / 'report.json'}\n")
    print(markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
