"""``python -m model_eval`` — concordance evaluation CLI.

Compares HyphAeon predictions against HyPhy MEME on user-supplied data. This
CLI is *not* shipped with the ``hyphaeon`` package (``model_eval`` is absent
from ``pyproject.toml``'s ``packages`` list and has no console-script entry
point); it only works from a source clone.

Two usage patterns:

  Pattern 1 — "I have an alignment, check it against MEME" (runs the model +
  HyPhy automatically)::

      python -m model_eval concordance -a gene.fasta -t tree.nwk
      python -m model_eval concordance -a gene.fasta -t tree.nwk \\
          --meme-result gene.json   # skip HyPhy, use pre-computed MEME

  Pattern 2 — "I already ran both tools, just compare" (batch file mode)::

      python -m model_eval concordance \\
          --predictions-dir hyphaeon_results/ --meme-dir meme_results/

All parsing, site alignment, and metrics come from the shared backend in
``concordance_compare.py``, which the pytest concordance tests also use.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from . import concordance_compare
from .concordance_compare import ConcordanceError


# ---------------------------------------------------------------------------
# Model loading (Pattern 1 only)
# ---------------------------------------------------------------------------

def _load_model(weights):
    """Load a HyphAeon checkpoint in eval mode on CPU.

    Resolution order: an explicit ``--weights`` path (must exist), then
    ``HYPHAEON_WEIGHTS`` (falls back to Hugging Face if stale), then the
    default Hugging Face variant.

    ``weights`` may be a str or pathlib.Path (argparse produces Path);
    it is cast to str before passing to ``load_model`` because the
    downstream weight-loading code calls ``path.endswith()``.
    """
    from hyphaeon.inference import load_model

    if weights:
        weights = str(weights)
        if not os.path.exists(weights):
            raise ConcordanceError(f"Weights file not found: {weights}")
        return load_model(weights=weights, device=torch.device("cpu"), strict=True)

    env_weights = os.environ.get("HYPHAEON_WEIGHTS")
    if env_weights and os.path.exists(env_weights):
        return load_model(weights=env_weights, device=torch.device("cpu"), strict=True)

    return load_model(weights=None, device=torch.device("cpu"), strict=True)


def _load_and_predict(model, alignment: Path, tree: Path):
    """Load an alignment+tree, run the model, and return per-site arrays.

    Returns ``(axo_lrts, axo_pvals, inv, taxa, L)``.
    """
    from ._harness import load_tensors, predict, pvals_from_lrt

    c, a, d, z, inv, taxa, L = load_tensors(str(alignment), str(tree))
    axo_lrts = predict(model, c, a, d, z, inv)
    axo_pvals = pvals_from_lrt(axo_lrts)
    return axo_lrts, axo_pvals, inv, taxa, L


# ---------------------------------------------------------------------------
# Pattern 1 — live prediction + MEME comparison
# ---------------------------------------------------------------------------

def _compare_live(
    axo_lrts: np.ndarray,
    axo_pvals: np.ndarray,
    inv: np.ndarray,
    meme_sites: Mapping[int, concordance_compare.MemeSite],
    *,
    alignment: Path,
    tree: Path,
    meme_source: str,
    variable_only: bool,
    stat_set: str,
) -> Dict[str, object]:
    """Align live HyphAeon predictions with MEME sites and compute metrics."""
    n_sites = len(axo_lrts)
    meme_lrts, meme_pvals, meme_tested = concordance_compare.meme_sites_to_arrays(
        meme_sites, n_sites
    )
    tested = ~inv if variable_only else np.ones(n_sites, dtype=bool)

    # Always compute concordance stats (counts + correlations) in the base
    # call; threshold stats are computed per-alpha in the loop below. This
    # avoids redundant computation when stat_set="threshold".
    base = concordance_compare.concordance_metrics(
        axo_lrts, axo_pvals, meme_lrts, meme_pvals, tested,
        meme_tested=meme_tested, stat_set="concordance",
    )
    n_meme_sites = len(meme_sites)
    result: Dict[str, object] = {
        "input_mode": "live",
        "alignment": str(Path(alignment).resolve()),
        "tree": str(Path(tree).resolve()),
        "meme_source": meme_source,
        "matched_genes": 1,
        "genes": [Path(alignment).stem],
        "total_sites": n_sites,
        "total_prediction_sites": n_sites,
        "total_meme_sites": n_meme_sites,
        "invariable_sites": int(inv.sum()),
        "variable_sites": int((~inv).sum()),
        "evaluated_sites": base["n_concordance_sites"],
        "meme_tested_sites": base["n_meme_tested_sites"],
        "evaluation_scope": "variable sites only" if variable_only else "all sites",
        "correlation_definition": (
            "HyphAeon hyphaeon_lrt versus MEME LRT over pooled evaluated sites; "
            "negative MEME LRT numerical artifacts are clamped to zero"
        ),
        "pearson_r": base.get("pearson_r") if stat_set in {"concordance", "all"} else None,
        "spearman_rho": base.get("spearman_rho") if stat_set in {"concordance", "all"} else None,
        "thresholds": {},
        "per_gene": [{
            "gene": Path(alignment).stem,
            "prediction_sites": n_sites,
            "meme_sites": n_meme_sites,
            "matched_sites": n_sites,
            "evaluated_sites": base["n_concordance_sites"],
            "prediction_only_sites": 0,
            "meme_only_sites": 0,
        }],
        "warnings": [],
        "clamped_negative_meme_lrts": sum(s.lrt_was_clamped for s in meme_sites.values()),
    }
    if stat_set in {"threshold", "all"}:
        for alpha in (0.05, 0.10):
            result["thresholds"][f"{alpha:g}"] = concordance_compare.concordance_metrics(
                axo_lrts, axo_pvals, meme_lrts, meme_pvals, tested,
                meme_tested=meme_tested, alpha=alpha, stat_set="threshold",
            )
    return result


def _run_live(args) -> Dict[str, object]:
    """Pattern 1: load model, predict, run/load MEME, compare."""
    print(f"Loading model"
          + (f" from {args.weights}" if args.weights else " (default Hugging Face variant)")
          + " ...")
    model = _load_model(args.weights)

    print(f"Loading alignment: {args.alignment} ...")
    axo_lrts, axo_pvals, inv, taxa, L = _load_and_predict(model, args.alignment, args.tree)
    print(f"  {len(taxa)} taxa, {L} codons, {int((~inv).sum())} variable "
          f"({100 * int((~inv).sum()) / max(L, 1):.0f}%)")

    if args.meme_result:
        meme_sites = concordance_compare.load_meme_json(args.meme_result)
        meme_source = str(args.meme_result)
        print(f"Loaded MEME results: {args.meme_result}")
    else:
        from .concordance._common import run_hyphy_meme
        print("Running HyPhy MEME ...")
        meme_sites = run_hyphy_meme(str(args.alignment), str(args.tree))
        if meme_sites is None or len(meme_sites) == 0:
            raise ConcordanceError(
                "HyPhy MEME failed or produced no sites. Install hyphy or pass "
                "--meme-result with a pre-computed MEME JSON file."
            )
        meme_source = "hyphy"

    return _compare_live(
        axo_lrts, axo_pvals, inv, meme_sites,
        alignment=args.alignment, tree=args.tree, meme_source=meme_source,
        variable_only=args.variable_only, stat_set=args.stats,
    )


# ---------------------------------------------------------------------------
# Pattern 2 — batch file comparison
# ---------------------------------------------------------------------------

def _run_batch(args) -> Dict[str, object]:
    """Pattern 2: pair directories of pre-computed results and pool."""
    pairs, prediction_only, meme_only = concordance_compare.match_gene_files(
        args.predictions_dir, args.meme_dir,
        prediction_suffix=args.prediction_suffix,
        meme_suffix=args.meme_suffix,
        allow_unmatched=args.allow_unmatched,
    )
    return concordance_compare.evaluate_pairs(
        pairs,
        variable_only=args.variable_only,
        allow_site_mismatch=args.allow_site_mismatch,
        stat_set=args.stats,
        prediction_only=prediction_only,
        meme_only=meme_only,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _display(value: object) -> str:
    return "undefined" if value is None else f"{float(value):.3f}"


def _format_text_report(result: Mapping[str, object]) -> str:
    lines = [
        "",
        "── Concordance: HyphAeon vs MEME ──",
        "",
        f"Matched sites:          {result.get('total_sites', '?')}",
        f"Evaluated sites:        {result.get('evaluated_sites', '?')} ({result.get('evaluation_scope', '')})",
    ]
    if "invariable_sites" in result:
        lines.append(f"Invariable sites:       {result['invariable_sites']}")
    if "variable_sites" in result:
        lines.append(f"Variable sites:         {result['variable_sites']}")
    if result.get("matched_genes") is not None:
        lines.append(f"Matched genes:          {result['matched_genes']}")

    if result.get("pearson_r") is not None or result.get("spearman_rho") is not None:
        lines.extend([
            "",
            "Correlation:",
            f"  Pearson r (LRT):      {_display(result.get('pearson_r'))}",
            f"  Spearman rho (LRT):   {_display(result.get('spearman_rho'))}",
        ])

    thresholds = result.get("thresholds") or {}
    for alpha_key in sorted(thresholds):
        block = thresholds[alpha_key]
        tag = alpha_key.replace(".", "")
        lines.extend([
            "",
            f"Threshold metrics (alpha={alpha_key}):",
            f"  ROC-AUC:              {_display(block.get('roc_auc'))}",
            f"  PPV:                  {_display(block.get('ppv'))}",
            f"  FPR:                  {_display(block.get('fpr'))}",
            f"  Cohen's kappa:        {_display(block.get(f'cohen_kappa_{tag}'))}",
            f"  F1:                   {_display(block.get(f'f1_{tag}'))}",
            f"  MEME positive sites:  {block.get(f'meme_significant_{tag}', '?')}",
            f"  HyphAeon positive:    {block.get(f'hyphaeon_significant_{tag}', '?')}",
        ])

    warnings = result.get("warnings") or []
    for warning in warnings:
        lines.extend(["", f"Warning: {warning}"])
    return "\n".join(lines)


def _print_result(result: Mapping[str, object], fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        print(_format_text_report(result))


def _write_json(result: Mapping[str, object], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


# ---------------------------------------------------------------------------
# Argument parsing & dispatch
# ---------------------------------------------------------------------------

def configure_parser(parser: argparse.ArgumentParser) -> None:
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("-a", "--alignment", type=Path,
                      help="Alignment FASTA (Pattern 1: live model + MEME run)")
    mode.add_argument("--predictions-dir", type=Path,
                      help="Directory of *.csv prediction files (Pattern 2: batch compare)")

    # Pattern 1 companions
    parser.add_argument("-t", "--tree", type=Path,
                        help="Tree Newick (required for Pattern 1)")
    parser.add_argument("--meme-result", type=Path,
                        help="Pre-computed MEME JSON (skip running HyPhy)")
    parser.add_argument("--weights", type=Path,
                        help="Model weights (default: HYPHAEON_WEIGHTS or Hugging Face)")

    # Pattern 2 companions
    parser.add_argument("--meme-dir", type=Path,
                        help="Directory of MEME JSON files (Pattern 2)")
    parser.add_argument("--prediction-suffix", default=".csv",
                        help="Suffix removed to get prediction gene names")
    parser.add_argument("--meme-suffix", default=".json",
                        help="Suffix removed to get MEME gene names (HyPhy MEME writes .json; "
                             "use .MEME.json for that naming convention)")

    # Shared options
    parser.add_argument("--stats", choices=("all", "concordance", "threshold"), default="all",
                        help="Which metric set to compute")
    parser.add_argument("--variable-only", action="store_true",
                        help="Exclude invariable sites from metrics (site totals still include them)")
    parser.add_argument("-o", "--output", type=Path, help="Write JSON results to file")
    parser.add_argument("--format", choices=("text", "json"), default="text",
                        help="Standard-output format")
    parser.add_argument("--allow-unmatched", action="store_true",
                        help="Ignore unpaired files in directory mode")
    parser.add_argument("--allow-site-mismatch", action="store_true",
                        help="Use site intersection instead of failing on unequal site sets")


def command(args: argparse.Namespace) -> Dict[str, object]:
    if args.alignment is not None:
        if args.tree is None:
            raise ConcordanceError("Pattern 1 requires --tree together with --alignment")
        result = _run_live(args)
    else:
        if args.meme_dir is None:
            raise ConcordanceError(
                "Pattern 2 requires --meme-dir together with --predictions-dir"
            )
        result = _run_batch(args)

    if args.output:
        _write_json(result, args.output)
    _print_result(result, args.format)
    if args.output and args.format == "text":
        print(f"\nJSON results written to: {args.output}")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m model_eval",
        description="Evaluate HyphAeon concordance against HyPhy MEME",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    concordance_parser = subparsers.add_parser(
        "concordance",
        help="Compare HyphAeon predictions with HyPhy MEME",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    configure_parser(concordance_parser)
    args = parser.parse_args(argv)
    try:
        command(args)
    except ConcordanceError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
