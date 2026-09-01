"""Pooled evaluation of HyphAeon ``predict`` results against HyPhy MEME.

This module powers ``hyphaeon evaluate`` and can also be run directly with
``python -m hyphaeon.evaluation``. Folder inputs are paired by gene name after
removing ``.csv`` from predictions and ``.MEME.json`` from MEME results. A
single matched pair can instead be supplied with ``--prediction`` and
``--meme-result``.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

from .io import ensure_parent_directory


class EvaluationError(ValueError):
    """Raised when inputs cannot be paired or safely aligned by site."""


@dataclass(frozen=True)
class PredictionSite:
    lrt: float
    p_value: float
    is_invariable: bool


@dataclass(frozen=True)
class MemeSite:
    lrt: float
    p_value: float
    lrt_was_clamped: bool = False


def _finite_float(value: object, label: str, path: Path) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"{path}: {label} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise EvaluationError(f"{path}: {label} must be finite, got {value!r}")
    return parsed


def _probability(value: object, label: str, path: Path) -> float:
    parsed = _finite_float(value, label, path)
    if not 0.0 <= parsed <= 1.0:
        raise EvaluationError(f"{path}: {label} must be between 0 and 1, got {parsed}")
    return parsed


def _site_number(value: object, path: Path) -> int:
    parsed = _finite_float(value, "site", path)
    site = int(parsed)
    if parsed != site or site < 1:
        raise EvaluationError(f"{path}: site must be a positive integer, got {value!r}")
    return site


def _boolean(value: object, label: str, path: Path) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise EvaluationError(f"{path}: {label} must be true or false, got {value!r}")


def load_prediction_csv(path: Path) -> Dict[int, PredictionSite]:
    """Load site-indexed results written by ``hyphaeon meme``."""
    required = {"site", "hyphaeon_lrt", "p_value", "is_invariable"}
    sites: Dict[int, PredictionSite] = {}
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise EvaluationError(
                    f"{path}: missing required CSV column(s): {', '.join(sorted(missing))}"
                )
            for line_number, row in enumerate(reader, start=2):
                site = _site_number(row["site"], path)
                if site in sites:
                    raise EvaluationError(f"{path}:{line_number}: duplicate site {site}")
                lrt = _finite_float(row["hyphaeon_lrt"], "hyphaeon_lrt", path)
                if lrt < 0.0:
                    raise EvaluationError(f"{path}:{line_number}: hyphaeon_lrt cannot be negative")
                sites[site] = PredictionSite(
                    lrt=lrt,
                    p_value=_probability(row["p_value"], "p_value", path),
                    is_invariable=_boolean(row["is_invariable"], "is_invariable", path),
                )
    except OSError as exc:
        raise EvaluationError(f"Could not read prediction file {path}: {exc}") from exc
    if not sites:
        raise EvaluationError(f"{path}: prediction CSV contains no sites")
    return sites


def _normalized_header(value: object) -> str:
    label = value[0] if isinstance(value, list) and value else value
    label = html.unescape(re.sub(r"<[^>]+>", "", str(label)))
    return re.sub(r"[^a-z0-9]+", "", label.lower())


def _column_index(headers: object, wanted: str, fallback: int, path: Path) -> int:
    if isinstance(headers, list):
        matches = [i for i, header in enumerate(headers) if _normalized_header(header) == wanted]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise EvaluationError(f"{path}: MEME has more than one {wanted!r} column")
    # Official HyPhy MEME output has LRT and p-value at zero-based columns 5 and 6.
    return fallback


def _partition_key(value: str) -> Tuple[int, object]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _flatten_coverage(value: object, path: Path, partition: str) -> List[int]:
    flattened: List[int] = []

    def visit(item: object) -> None:
        if isinstance(item, list):
            for child in item:
                visit(child)
            return
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise EvaluationError(
                f"{path}: invalid site index in data partition {partition!r} coverage"
            )
        index = int(item)
        if index != item or index < 0:
            raise EvaluationError(
                f"{path}: invalid site index {item!r} in partition {partition!r} coverage"
            )
        flattened.append(index)

    visit(value)
    return flattened


def load_meme_json(path: Path) -> Dict[int, MemeSite]:
    """Load MEME sites, using partition coverage to recover global site IDs."""
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Could not read MEME JSON {path}: {exc}") from exc

    mle = data.get("MLE")
    if not isinstance(mle, dict) or not isinstance(mle.get("content"), dict):
        raise EvaluationError(f"{path}: expected an MLE.content object")
    content = mle["content"]
    if not content:
        raise EvaluationError(f"{path}: MLE.content contains no partitions")

    lrt_column = _column_index(mle.get("headers"), "lrt", 5, path)
    p_column = _column_index(mle.get("headers"), "pvalue", 6, path)
    data_partitions = data.get("data partitions", {})
    sites: Dict[int, MemeSite] = {}
    next_fallback_index = 0

    for partition in sorted(content, key=_partition_key):
        rows = content[partition]
        if not isinstance(rows, list):
            raise EvaluationError(f"{path}: MLE.content[{partition!r}] must be a list")

        partition_info = data_partitions.get(partition, {}) if isinstance(data_partitions, dict) else {}
        coverage = partition_info.get("coverage") if isinstance(partition_info, dict) else None
        if coverage is None:
            site_indices = list(range(next_fallback_index, next_fallback_index + len(rows)))
        else:
            site_indices = _flatten_coverage(coverage, path, partition)
            if len(site_indices) != len(rows):
                raise EvaluationError(
                    f"{path}: partition {partition!r} has {len(rows)} MLE rows but "
                    f"{len(site_indices)} covered sites"
                )
        if site_indices:
            next_fallback_index = max(next_fallback_index, max(site_indices) + 1)

        for row_number, (zero_based_site, row) in enumerate(zip(site_indices, rows), start=1):
            if not isinstance(row, list) or max(lrt_column, p_column) >= len(row):
                raise EvaluationError(
                    f"{path}: partition {partition!r} row {row_number} lacks LRT/p-value columns"
                )
            site = zero_based_site + 1
            if site in sites:
                raise EvaluationError(f"{path}: duplicate MEME site {site} across partitions")
            raw_lrt = _finite_float(row[lrt_column], "MEME LRT", path)
            # An LRT is non-negative by definition, but MEME can emit small
            # negative values when its two numerical likelihood fits cross.
            lrt = max(0.0, raw_lrt)
            sites[site] = MemeSite(
                lrt=lrt,
                p_value=_probability(row[p_column], "MEME p-value", path),
                lrt_was_clamped=raw_lrt < 0.0,
            )

    if not sites:
        raise EvaluationError(f"{path}: MEME JSON contains no sites")
    return sites


def _files_by_gene(directory: Path, suffix: str, label: str) -> Dict[str, Path]:
    if not directory.is_dir():
        raise EvaluationError(f"{label} directory does not exist or is not a directory: {directory}")
    files: Dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or not path.name.endswith(suffix):
            continue
        gene = path.name[: -len(suffix)] if suffix else path.name
        if not gene:
            raise EvaluationError(f"{path}: file name has no gene name before {suffix!r}")
        if gene in files:
            raise EvaluationError(f"{label} directory has duplicate files for gene {gene!r}")
        files[gene] = path
    if not files:
        raise EvaluationError(f"No *{suffix} files found in {label} directory {directory}")
    return files


def match_gene_files(
    prediction_directory: Path,
    meme_directory: Path,
    prediction_suffix: str = ".csv",
    meme_suffix: str = ".MEME.json",
    allow_unmatched: bool = False,
) -> Tuple[List[Tuple[str, Path, Path]], List[str], List[str]]:
    predictions = _files_by_gene(prediction_directory, prediction_suffix, "prediction")
    meme_results = _files_by_gene(meme_directory, meme_suffix, "MEME")
    prediction_only = sorted(predictions.keys() - meme_results.keys())
    meme_only = sorted(meme_results.keys() - predictions.keys())
    if (prediction_only or meme_only) and not allow_unmatched:
        details = []
        if prediction_only:
            details.append("prediction-only genes: " + ", ".join(prediction_only))
        if meme_only:
            details.append("MEME-only genes: " + ", ".join(meme_only))
        raise EvaluationError("Unmatched input files; " + "; ".join(details))
    genes = sorted(predictions.keys() & meme_results.keys())
    if not genes:
        raise EvaluationError("The two directories contain no matching gene names")
    return [(gene, predictions[gene], meme_results[gene]) for gene in genes], prediction_only, meme_only


def _optional_float(value: float) -> Optional[float]:
    return float(value) if math.isfinite(float(value)) else None


def _correlations(predicted: np.ndarray, observed: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if len(predicted) < 2 or np.ptp(predicted) == 0.0 or np.ptp(observed) == 0.0:
        return None, None
    pearson = _optional_float(pearsonr(predicted, observed).statistic)
    spearman = _optional_float(spearmanr(predicted, observed).statistic)
    return pearson, spearman


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(scores, method="average")
    rank_sum = float(ranks[labels].sum())
    auc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def _confusion(truth: np.ndarray, predicted: np.ndarray) -> Dict[str, int]:
    return {
        "true_positive": int(np.sum(truth & predicted)),
        "false_positive": int(np.sum(~truth & predicted)),
        "true_negative": int(np.sum(~truth & ~predicted)),
        "false_negative": int(np.sum(truth & ~predicted)),
    }


def _ppv(confusion: Mapping[str, int]) -> Optional[float]:
    denominator = confusion["true_positive"] + confusion["false_positive"]
    return confusion["true_positive"] / denominator if denominator else None


def _fpr(confusion: Mapping[str, int]) -> Optional[float]:
    denominator = confusion["false_positive"] + confusion["true_negative"]
    return confusion["false_positive"] / denominator if denominator else None


def _threshold_metrics(
    predicted_lrt: np.ndarray,
    predicted_p: np.ndarray,
    meme_p: np.ndarray,
    alpha: float,
) -> Dict[str, object]:
    inclusive_truth = meme_p <= alpha
    inclusive_predictions = predicted_p <= alpha
    ppv_confusion = _confusion(inclusive_truth, inclusive_predictions)
    fpr_confusion = _confusion(inclusive_truth, inclusive_predictions)

    return {
        "roc_auc": _roc_auc(inclusive_truth, predicted_lrt),
        "roc_auc_definition": f"ground truth MEME p_value <= {alpha:g}; score = HyphAeon hyphaeon_lrt",
        "ppv": _ppv(ppv_confusion),
        "ppv_definition": f"MEME and HyphAeon p_value <= {alpha:g}",
        "fpr": _fpr(fpr_confusion),
        "fpr_definition": f"MEME and HyphAeon p_value <= {alpha:g}",
        "meme_positive_sites": int(inclusive_truth.sum()),
        "hyphaeon_positive_sites": int(inclusive_predictions.sum()),
        "ppv_confusion_matrix": ppv_confusion,
        "fpr_confusion_matrix": fpr_confusion,
    }


def _evaluate_pairs(
    pairs: Sequence[Tuple[str, Path, Path]],
    *,
    input_metadata: Mapping[str, object],
    prediction_only: Sequence[str] = (),
    meme_only: Sequence[str] = (),
    allow_site_mismatch: bool = False,
    variable_only: bool = False,
) -> Dict[str, object]:
    """Pool already-paired gene files and calculate dataset-level metrics."""
    pooled_prediction_lrt: List[float] = []
    pooled_prediction_p: List[float] = []
    pooled_meme_lrt: List[float] = []
    pooled_meme_p: List[float] = []
    per_gene: List[Dict[str, object]] = []
    total_prediction_sites = 0
    total_meme_sites = 0
    total_matched_sites = 0
    total_invariable_sites = 0
    total_clamped_meme_lrts = 0

    for gene, prediction_path, meme_path in pairs:
        predictions = load_prediction_csv(prediction_path)
        meme_sites = load_meme_json(meme_path)
        total_prediction_sites += len(predictions)
        total_meme_sites += len(meme_sites)

        prediction_only_sites = sorted(predictions.keys() - meme_sites.keys())
        meme_only_sites = sorted(meme_sites.keys() - predictions.keys())
        if (prediction_only_sites or meme_only_sites) and not allow_site_mismatch:
            details = []
            if prediction_only_sites:
                details.append(f"prediction-only sites {prediction_only_sites[:10]}")
            if meme_only_sites:
                details.append(f"MEME-only sites {meme_only_sites[:10]}")
            raise EvaluationError(f"Gene {gene!r} has mismatched sites: " + "; ".join(details))

        matched_sites = sorted(predictions.keys() & meme_sites.keys())
        if not matched_sites:
            raise EvaluationError(f"Gene {gene!r} has no sites shared by its two result files")
        total_matched_sites += len(matched_sites)
        total_invariable_sites += sum(predictions[site].is_invariable for site in matched_sites)
        total_clamped_meme_lrts += sum(meme_sites[site].lrt_was_clamped for site in matched_sites)

        evaluated_sites = [
            site for site in matched_sites if not variable_only or not predictions[site].is_invariable
        ]
        for site in evaluated_sites:
            prediction = predictions[site]
            meme = meme_sites[site]
            pooled_prediction_lrt.append(prediction.lrt)
            pooled_prediction_p.append(prediction.p_value)
            pooled_meme_lrt.append(meme.lrt)
            pooled_meme_p.append(meme.p_value)

        per_gene.append(
            {
                "gene": gene,
                "prediction_sites": len(predictions),
                "meme_sites": len(meme_sites),
                "matched_sites": len(matched_sites),
                "evaluated_sites": len(evaluated_sites),
                "prediction_only_sites": len(prediction_only_sites),
                "meme_only_sites": len(meme_only_sites),
            }
        )

    if not pooled_prediction_lrt:
        scope = "variable" if variable_only else "matched"
        raise EvaluationError(f"No {scope} sites are available for evaluation")

    prediction_lrt = np.asarray(pooled_prediction_lrt, dtype=float)
    prediction_p = np.asarray(pooled_prediction_p, dtype=float)
    meme_lrt = np.asarray(pooled_meme_lrt, dtype=float)
    meme_p = np.asarray(pooled_meme_p, dtype=float)
    pearson, spearman = _correlations(prediction_lrt, meme_lrt)

    warnings: List[str] = []
    if prediction_only:
        warnings.append("Ignored prediction-only genes: " + ", ".join(prediction_only))
    if meme_only:
        warnings.append("Ignored MEME-only genes: " + ", ".join(meme_only))
    if allow_site_mismatch:
        dropped_prediction = sum(int(item["prediction_only_sites"]) for item in per_gene)
        dropped_meme = sum(int(item["meme_only_sites"]) for item in per_gene)
        if dropped_prediction or dropped_meme:
            warnings.append(
                f"Site intersection used; dropped {dropped_prediction} prediction-only and "
                f"{dropped_meme} MEME-only sites"
            )
    if total_clamped_meme_lrts:
        warnings.append(
            f"Clamped {total_clamped_meme_lrts} negative MEME LRT numerical artifact(s) to zero"
        )

    return {
        **input_metadata,
        "matched_genes": len(pairs),
        "genes": [gene for gene, _, _ in pairs],
        "total_sites": total_matched_sites,
        "total_prediction_sites": total_prediction_sites,
        "total_meme_sites": total_meme_sites,
        "evaluated_sites": len(prediction_lrt),
        "variable_sites": total_matched_sites - total_invariable_sites,
        "invariable_sites": total_invariable_sites,
        "evaluation_scope": "variable sites only" if variable_only else "all matched sites",
        "correlation_definition": (
            "HyphAeon hyphaeon_lrt versus MEME LRT over pooled evaluated sites; "
            "negative MEME LRT numerical artifacts are clamped to zero"
        ),
        "clamped_negative_meme_lrts": total_clamped_meme_lrts,
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "thresholds": {
            "0.05": _threshold_metrics(prediction_lrt, prediction_p, meme_p, 0.05),
            "0.10": _threshold_metrics(prediction_lrt, prediction_p, meme_p, 0.10),
        },
        "per_gene": per_gene,
        "warnings": warnings,
    }


def evaluate_directories(
    prediction_directory: Path,
    meme_directory: Path,
    *,
    prediction_suffix: str = ".csv",
    meme_suffix: str = ".MEME.json",
    allow_unmatched: bool = False,
    allow_site_mismatch: bool = False,
    variable_only: bool = False,
) -> Dict[str, object]:
    """Pair genes from two directories, pool their sites, and evaluate."""
    prediction_directory = Path(prediction_directory)
    meme_directory = Path(meme_directory)
    pairs, prediction_only, meme_only = match_gene_files(
        prediction_directory,
        meme_directory,
        prediction_suffix=prediction_suffix,
        meme_suffix=meme_suffix,
        allow_unmatched=allow_unmatched,
    )
    return _evaluate_pairs(
        pairs,
        input_metadata={
            "input_mode": "directories",
            "prediction_directory": str(prediction_directory.resolve()),
            "meme_directory": str(meme_directory.resolve()),
        },
        prediction_only=prediction_only,
        meme_only=meme_only,
        allow_site_mismatch=allow_site_mismatch,
        variable_only=variable_only,
    )


def _gene_name_from_file(path: Path, suffix: str, label: str) -> str:
    if not path.is_file():
        raise EvaluationError(f"{label} file does not exist or is not a file: {path}")
    if suffix and not path.name.endswith(suffix):
        raise EvaluationError(f"{label} file must end with {suffix!r}: {path}")
    gene = path.name[: -len(suffix)] if suffix else path.name
    if not gene:
        raise EvaluationError(f"{path}: file name has no gene name before {suffix!r}")
    return gene


def evaluate_files(
    prediction_file: Path,
    meme_result_file: Path,
    *,
    prediction_suffix: str = ".csv",
    meme_suffix: str = ".MEME.json",
    allow_site_mismatch: bool = False,
    variable_only: bool = False,
) -> Dict[str, object]:
    """Evaluate one matched HyphAeon prediction and MEME result file."""
    prediction_file = Path(prediction_file)
    meme_result_file = Path(meme_result_file)
    prediction_gene = _gene_name_from_file(
        prediction_file, prediction_suffix, "prediction"
    )
    meme_gene = _gene_name_from_file(meme_result_file, meme_suffix, "MEME result")
    if prediction_gene != meme_gene:
        raise EvaluationError(
            "Single-gene files do not match: "
            f"prediction gene {prediction_gene!r}, MEME result gene {meme_gene!r}"
        )
    return _evaluate_pairs(
        [(prediction_gene, prediction_file, meme_result_file)],
        input_metadata={
            "input_mode": "files",
            "prediction_file": str(prediction_file.resolve()),
            "meme_result_file": str(meme_result_file.resolve()),
        },
        allow_site_mismatch=allow_site_mismatch,
        variable_only=variable_only,
    )


def _display(value: object) -> str:
    return "undefined" if value is None else f"{float(value):.6f}"


def format_text_report(result: Mapping[str, object]) -> str:
    thresholds = result["thresholds"]
    alpha_005 = thresholds["0.05"]
    alpha_010 = thresholds["0.10"]
    lines = [
        f"Matched genes: {result['matched_genes']}",
        f"Total sites: {result['total_sites']}",
        f"Evaluated sites: {result['evaluated_sites']} ({result['evaluation_scope']})",
        f"Pearson r (LRT): {_display(result['pearson_r'])}",
        f"Spearman rho (LRT): {_display(result['spearman_rho'])}",
        "",
        "Metric                 p <= 0.05    p <= 0.10",
        f"ROC-AUC                {_display(alpha_005['roc_auc']):>10}    {_display(alpha_010['roc_auc']):>10}",
        f"PPV                    {_display(alpha_005['ppv']):>10}    {_display(alpha_010['ppv']):>10}",
        f"FPR                    {_display(alpha_005['fpr']):>10}    {_display(alpha_010['fpr']):>10}",
    ]
    warnings = result.get("warnings", [])
    if warnings:
        lines.extend(["", *(f"Warning: {warning}" for warning in warnings)])
    return "\n".join(lines)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    prediction_input = parser.add_mutually_exclusive_group(required=True)
    prediction_input.add_argument(
        "--predictions-dir",
        type=Path,
        help="Folder containing Gene.csv HyphAeon predict results",
    )
    prediction_input.add_argument(
        "--prediction",
        type=Path,
        help="Single Gene.csv HyphAeon predict result",
    )
    meme_input = parser.add_mutually_exclusive_group(required=True)
    meme_input.add_argument(
        "--meme-dir",
        type=Path,
        help="Folder containing matched Gene.MEME.json results",
    )
    meme_input.add_argument(
        "--meme-result",
        type=Path,
        help="Single matched Gene.MEME.json result",
    )
    parser.add_argument("-o", "--output", type=Path, help="Optional JSON output file")
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Standard-output format",
    )
    parser.add_argument("--prediction-suffix", default=".csv", help="Suffix removed to get prediction gene names")
    parser.add_argument("--meme-suffix", default=".MEME.json", help="Suffix removed to get MEME gene names")
    parser.add_argument(
        "--variable-only",
        action="store_true",
        help="Exclude rows marked is_invariable from metrics (site totals still include them)",
    )
    parser.add_argument(
        "--allow-unmatched",
        action="store_true",
        help="Ignore files whose gene name has no counterpart instead of failing",
    )
    parser.add_argument(
        "--allow-site-mismatch",
        action="store_true",
        help="Use each gene's site intersection instead of failing on unequal site sets",
    )


def command(args: argparse.Namespace) -> Dict[str, object]:
    direct_file_mode = args.prediction is not None or args.meme_result is not None
    if direct_file_mode:
        if args.prediction is None or args.meme_result is None:
            raise EvaluationError(
                "Single-gene mode requires --prediction together with --meme-result; "
                "directory and file inputs cannot be mixed"
            )
        if args.allow_unmatched:
            raise EvaluationError("--allow-unmatched is only valid with directory inputs")
        result = evaluate_files(
            args.prediction,
            args.meme_result,
            prediction_suffix=args.prediction_suffix,
            meme_suffix=args.meme_suffix,
            allow_site_mismatch=args.allow_site_mismatch,
            variable_only=args.variable_only,
        )
    else:
        if args.predictions_dir is None or args.meme_dir is None:
            raise EvaluationError(
                "Directory mode requires --predictions-dir together with --meme-dir; "
                "directory and file inputs cannot be mixed"
            )
        result = evaluate_directories(
            args.predictions_dir,
            args.meme_dir,
            prediction_suffix=args.prediction_suffix,
            meme_suffix=args.meme_suffix,
            allow_unmatched=args.allow_unmatched,
            allow_site_mismatch=args.allow_site_mismatch,
            variable_only=args.variable_only,
        )
    if args.output:
        ensure_parent_directory(str(args.output))
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    if args.format == "json":
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        print(format_text_report(result))
        if args.output:
            print(f"\nJSON results written to: {args.output}")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pool and evaluate HyphAeon site predictions against matched MEME results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    configure_parser(parser)
    args = parser.parse_args(argv)
    try:
        command(args)
    except EvaluationError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
