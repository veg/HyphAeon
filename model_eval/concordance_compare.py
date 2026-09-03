"""Shared backend for HyphAeon-vs-MEME concordance evaluation.

This module is pytest-free and model-free: it only parses result files and
computes metrics. Both the ``python -m model_eval`` CLI and the existing
pytest concordance tests (``concordance/test_hyphaeon_vs_meme.py``) import
from here so file parsing, site alignment, and metrics are defined in exactly
one place.

It is *not* part of the installed ``hyphaeon`` package — ``model_eval`` stays
unpackaged (see ``pyproject.toml``: ``packages = ["hyphaeon"]``) and only
works from a source clone.

The parsing logic (partition coverage to recover global site IDs, negative
LRT clamping, multi-gene pooling) is adapted from the pooled-evaluation code
that previously lived in ``hyphaeon/evaluation.py``; this module supersedes
that ad-hoc path for concordance work.
"""
from __future__ import annotations

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

try:
    from sklearn.metrics import cohen_kappa_score, f1_score
except ImportError:  # sklearn is optional for the CLI; tests skip without it
    cohen_kappa_score = None
    f1_score = None


class ConcordanceError(ValueError):
    """Raised when inputs cannot be paired or safely aligned by site."""


# ---------------------------------------------------------------------------
# Site records
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Value coercion helpers
# ---------------------------------------------------------------------------

def _finite_float(value: object, label: str, path: Path) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConcordanceError(f"{path}: {label} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ConcordanceError(f"{path}: {label} must be finite, got {value!r}")
    return parsed


def _probability(value: object, label: str, path: Path) -> float:
    parsed = _finite_float(value, label, path)
    if not 0.0 <= parsed <= 1.0:
        raise ConcordanceError(f"{path}: {label} must be between 0 and 1, got {parsed}")
    return parsed


def _site_number(value: object, path: Path) -> int:
    parsed = _finite_float(value, "site", path)
    site = int(parsed)
    if parsed != site or site < 1:
        raise ConcordanceError(f"{path}: site must be a positive integer, got {value!r}")
    return site


def _boolean(value: object, label: str, path: Path) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ConcordanceError(f"{path}: {label} must be true or false, got {value!r}")


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

def load_prediction_csv(path: Path) -> Dict[int, PredictionSite]:
    """Load site-indexed results written by ``hyphaeon meme``.

    Validates required columns, rejects duplicate sites, and enforces
    non-negative LRT. Site IDs are 1-based, matching the CSV contract.
    """
    path = Path(path)
    required = {"site", "hyphaeon_lrt", "p_value", "is_invariable"}
    sites: Dict[int, PredictionSite] = {}
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ConcordanceError(
                    f"{path}: missing required CSV column(s): {', '.join(sorted(missing))}"
                )
            for line_number, row in enumerate(reader, start=2):
                site = _site_number(row["site"], path)
                if site in sites:
                    raise ConcordanceError(f"{path}:{line_number}: duplicate site {site}")
                lrt = _finite_float(row["hyphaeon_lrt"], "hyphaeon_lrt", path)
                if lrt < 0.0:
                    raise ConcordanceError(f"{path}:{line_number}: hyphaeon_lrt cannot be negative")
                sites[site] = PredictionSite(
                    lrt=lrt,
                    p_value=_probability(row["p_value"], "p_value", path),
                    is_invariable=_boolean(row["is_invariable"], "is_invariable", path),
                )
    except OSError as exc:
        raise ConcordanceError(f"Could not read prediction file {path}: {exc}") from exc
    if not sites:
        raise ConcordanceError(f"{path}: prediction CSV contains no sites")
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
            raise ConcordanceError(f"{path}: MEME has more than one {wanted!r} column")
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
            raise ConcordanceError(
                f"{path}: invalid site index in data partition {partition!r} coverage"
            )
        index = int(item)
        if index != item or index < 0:
            raise ConcordanceError(
                f"{path}: invalid site index {item!r} in partition {partition!r} coverage"
            )
        flattened.append(index)

    visit(value)
    return flattened


def load_meme_json(path: Path) -> Dict[int, MemeSite]:
    """Load MEME sites, using partition coverage to recover global site IDs.

    Handles multi-partition output. Negative LRT numerical artifacts (which
    occur when MEME's two numerical likelihood fits cross) are clamped to
    zero and flagged via ``MemeSite.lrt_was_clamped``. Site IDs are 1-based.
    """
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConcordanceError(f"Could not read MEME JSON {path}: {exc}") from exc

    mle = data.get("MLE")
    if not isinstance(mle, dict) or not isinstance(mle.get("content"), dict):
        raise ConcordanceError(f"{path}: expected an MLE.content object")
    content = mle["content"]
    if not content:
        raise ConcordanceError(f"{path}: MLE.content contains no partitions")

    lrt_column = _column_index(mle.get("headers"), "lrt", 5, path)
    p_column = _column_index(mle.get("headers"), "pvalue", 6, path)
    data_partitions = data.get("data partitions", {})
    sites: Dict[int, MemeSite] = {}
    next_fallback_index = 0

    for partition in sorted(content, key=_partition_key):
        rows = content[partition]
        if not isinstance(rows, list):
            raise ConcordanceError(f"{path}: MLE.content[{partition!r}] must be a list")

        partition_info = data_partitions.get(partition, {}) if isinstance(data_partitions, dict) else {}
        coverage = partition_info.get("coverage") if isinstance(partition_info, dict) else None
        if coverage is None:
            site_indices = list(range(next_fallback_index, next_fallback_index + len(rows)))
        else:
            site_indices = _flatten_coverage(coverage, path, partition)
            if len(site_indices) != len(rows):
                raise ConcordanceError(
                    f"{path}: partition {partition!r} has {len(rows)} MLE rows but "
                    f"{len(site_indices)} covered sites"
                )
        if site_indices:
            next_fallback_index = max(next_fallback_index, max(site_indices) + 1)

        for row_number, (zero_based_site, row) in enumerate(zip(site_indices, rows), start=1):
            if not isinstance(row, list) or max(lrt_column, p_column) >= len(row):
                raise ConcordanceError(
                    f"{path}: partition {partition!r} row {row_number} lacks LRT/p-value columns"
                )
            site = zero_based_site + 1
            if site in sites:
                raise ConcordanceError(f"{path}: duplicate MEME site {site} across partitions")
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
        raise ConcordanceError(f"{path}: MEME JSON contains no sites")
    return sites


# ---------------------------------------------------------------------------
# File pairing (batch mode)
# ---------------------------------------------------------------------------

def _files_by_gene(directory: Path, suffix: str, label: str) -> Dict[str, Path]:
    if not directory.is_dir():
        raise ConcordanceError(f"{label} directory does not exist or is not a directory: {directory}")
    files: Dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or not path.name.endswith(suffix):
            continue
        gene = path.name[: -len(suffix)] if suffix else path.name
        if not gene:
            raise ConcordanceError(f"{path}: file name has no gene name before {suffix!r}")
        if gene in files:
            raise ConcordanceError(f"{label} directory has duplicate files for gene {gene!r}")
        files[gene] = path
    if not files:
        raise ConcordanceError(f"No *{suffix} files found in {label} directory {directory}")
    return files


def match_gene_files(
    prediction_dir: Path,
    meme_dir: Path,
    prediction_suffix: str = ".csv",
    meme_suffix: str = ".json",
    allow_unmatched: bool = False,
) -> Tuple[List[Tuple[str, Path, Path]], List[str], List[str]]:
    """Pair prediction and MEME files by gene name.

    Returns ``(matched_pairs, prediction_only, meme_only)``. The MEME suffix
    defaults to ``.json`` because that is the terminator HyPhy MEME actually
    writes; the ``.MEME.json`` convention is a personal naming choice and is
    not enforced here — pass ``meme_suffix=".MEME.json"`` to match such files.
    """
    predictions = _files_by_gene(Path(prediction_dir), prediction_suffix, "prediction")
    meme_results = _files_by_gene(Path(meme_dir), meme_suffix, "MEME")
    prediction_only = sorted(predictions.keys() - meme_results.keys())
    meme_only = sorted(meme_results.keys() - predictions.keys())
    if (prediction_only or meme_only) and not allow_unmatched:
        details = []
        if prediction_only:
            details.append("prediction-only genes: " + ", ".join(prediction_only))
        if meme_only:
            details.append("MEME-only genes: " + ", ".join(meme_only))
        raise ConcordanceError("Unmatched input files; " + "; ".join(details))
    genes = sorted(predictions.keys() & meme_results.keys())
    if not genes:
        raise ConcordanceError("The two directories contain no matching gene names")
    return (
        [(gene, predictions[gene], meme_results[gene]) for gene in genes],
        prediction_only,
        meme_only,
    )


# ---------------------------------------------------------------------------
# Site dict -> aligned arrays (bridge for in-memory / live-run predictions)
# ---------------------------------------------------------------------------

def meme_sites_to_arrays(
    meme_sites: Mapping[int, MemeSite], n_sites: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a ``{site: MemeSite}`` dict into arrays aligned with HyphAeon's
    per-site LRT/p-value arrays (length ``n_sites``, 1-based site IDs).

    Sites with no MEME entry default to ``lrt=0, p_value=1`` (non-significant).
    Returns ``(meme_lrts, meme_pvals, meme_tested)`` where ``meme_tested`` is a
    boolean mask of sites MEME actually produced a result for. Use this mask to
    avoid penalizing HyphAeon for sites MEME skipped.
    """
    meme_lrts = np.zeros(n_sites)
    meme_pvals = np.ones(n_sites)
    meme_tested = np.zeros(n_sites, dtype=bool)
    for site_idx, info in meme_sites.items():
        i = site_idx - 1  # 1-based site ID -> 0-based array index
        if 0 <= i < n_sites:
            meme_lrts[i] = info.lrt
            meme_pvals[i] = info.p_value
            meme_tested[i] = True
    return meme_lrts, meme_pvals, meme_tested


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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


def concordance_metrics(
    axo_lrts: np.ndarray,
    axo_pvals: np.ndarray,
    meme_lrts: np.ndarray,
    meme_pvals: np.ndarray,
    tested: np.ndarray,
    meme_tested: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    stat_set: str = "all",
) -> Dict[str, object]:
    """Compute concordance metrics between HyphAeon and MEME.

    ``stat_set`` controls which metric groups are returned:
      - ``"concordance"``: Spearman rho and Pearson r on LRTs
      - ``"threshold"``: ROC-AUC, PPV, FPR, Cohen's kappa, F1 at matched alpha
      - ``"all"``: both groups

    If ``meme_tested`` is provided, only sites where HyphAeon has a variable
    site AND MEME produced a result are scored. This avoids penalizing
    HyphAeon for sites MEME skipped (which default to p=1 and would count as
    discordances). Metrics that cannot be computed (e.g. ROC-AUC when one
    class is empty, or kappa/F1 when scikit-learn is unavailable) are ``None``.
    """
    if stat_set not in {"all", "concordance", "threshold"}:
        raise ConcordanceError(f"stat_set must be 'all', 'concordance', or 'threshold', got {stat_set!r}")

    concordance_tested = (tested & meme_tested) if meme_tested is not None else tested
    axo_var = np.asarray(axo_lrts)[concordance_tested]
    meme_var = np.asarray(meme_lrts)[concordance_tested]
    axo_p_var = np.asarray(axo_pvals)[concordance_tested]
    meme_p_var = np.asarray(meme_pvals)[concordance_tested]

    alpha_tag = f"{alpha:g}".replace(".", "")  # 0.05 -> "005"
    result: Dict[str, object] = {
        "n_concordance_sites": int(concordance_tested.sum()),
        "n_variable_sites": int(tested.sum()),
        "n_meme_tested_sites": int(meme_tested.sum()) if meme_tested is not None else int(tested.sum()),
    }

    if stat_set in {"concordance", "all"}:
        pearson, spearman = _correlations(axo_var, meme_var)
        result["pearson_r"] = pearson
        result["spearman_rho"] = spearman

    if stat_set in {"threshold", "all"}:
        meme_sig = meme_p_var <= alpha
        axo_sig = axo_p_var <= alpha
        confusion = _confusion(meme_sig, axo_sig)
        # sklearn raises on empty arrays and returns nan for kappa when all
        # labels are identical; normalize both to None so the result is always
        # JSON-safe (allow_nan=False).
        if len(meme_sig) == 0:
            kappa = None
            f1 = None
        elif cohen_kappa_score is not None:
            kappa = _optional_float(cohen_kappa_score(meme_sig, axo_sig))
            f1 = _optional_float(f1_score(meme_sig, axo_sig, zero_division=0))
        else:
            kappa = None
            f1 = None
        result.update({
            "roc_auc": _roc_auc(meme_sig, axo_var),
            "ppv": _ppv(confusion),
            "fpr": _fpr(confusion),
            f"cohen_kappa_{alpha_tag}": kappa,
            f"f1_{alpha_tag}": f1,
            f"hyphaeon_significant_{alpha_tag}": int(axo_sig.sum()),
            f"meme_significant_{alpha_tag}": int(meme_sig.sum()),
            f"confusion_matrix_{alpha_tag}": confusion,
        })

    return result


# ---------------------------------------------------------------------------
# Pooling (multi-gene batch mode)
# ---------------------------------------------------------------------------

def evaluate_pairs(
    pairs: Sequence[Tuple[str, Path, Path]],
    variable_only: bool = False,
    allow_site_mismatch: bool = False,
    stat_set: str = "all",
    prediction_only: Sequence[str] = (),
    meme_only: Sequence[str] = (),
) -> Dict[str, object]:
    """Pool already-paired gene files and compute dataset-level metrics.

    Each pair is ``(gene, prediction_csv, meme_json)``. Sites are pooled across
    all genes before metrics are computed (metrics are not averaged per gene).
    Threshold metrics are reported at both alpha=0.05 and alpha=0.10.
    """
    pooled_axo_lrt: List[float] = []
    pooled_axo_p: List[float] = []
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
            raise ConcordanceError(f"Gene {gene!r} has mismatched sites: " + "; ".join(details))

        matched_sites = sorted(predictions.keys() & meme_sites.keys())
        if not matched_sites:
            raise ConcordanceError(f"Gene {gene!r} has no sites shared by its two result files")
        total_matched_sites += len(matched_sites)
        total_invariable_sites += sum(predictions[site].is_invariable for site in matched_sites)
        total_clamped_meme_lrts += sum(meme_sites[site].lrt_was_clamped for site in matched_sites)

        evaluated_sites = [
            site for site in matched_sites
            if not variable_only or not predictions[site].is_invariable
        ]
        for site in evaluated_sites:
            pooled_axo_lrt.append(predictions[site].lrt)
            pooled_axo_p.append(predictions[site].p_value)
            pooled_meme_lrt.append(meme_sites[site].lrt)
            pooled_meme_p.append(meme_sites[site].p_value)

        per_gene.append({
            "gene": gene,
            "prediction_sites": len(predictions),
            "meme_sites": len(meme_sites),
            "matched_sites": len(matched_sites),
            "evaluated_sites": len(evaluated_sites),
            "prediction_only_sites": len(prediction_only_sites),
            "meme_only_sites": len(meme_only_sites),
        })

    if not pooled_axo_lrt:
        scope = "variable" if variable_only else "matched"
        raise ConcordanceError(f"No {scope} sites are available for evaluation")

    axo_lrt = np.asarray(pooled_axo_lrt, dtype=float)
    axo_p = np.asarray(pooled_axo_p, dtype=float)
    meme_lrt = np.asarray(pooled_meme_lrt, dtype=float)
    meme_p = np.asarray(pooled_meme_p, dtype=float)
    tested = np.ones(len(axo_lrt), dtype=bool)  # all pooled sites are scored

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

    # Concordance stats (counts + correlations) are alpha-independent; compute
    # once. Threshold stats are computed per-alpha in the loop below.
    base = concordance_metrics(
        axo_lrt, axo_p, meme_lrt, meme_p, tested,
        stat_set="concordance",
    )
    result: Dict[str, object] = {
        "matched_genes": len(pairs),
        "genes": [gene for gene, _, _ in pairs],
        "total_sites": total_matched_sites,
        "total_prediction_sites": total_prediction_sites,
        "total_meme_sites": total_meme_sites,
        "evaluated_sites": len(axo_lrt),
        "variable_sites": total_matched_sites - total_invariable_sites,
        "invariable_sites": total_invariable_sites,
        "evaluation_scope": "variable sites only" if variable_only else "all matched sites",
        "correlation_definition": (
            "HyphAeon hyphaeon_lrt versus MEME LRT over pooled evaluated sites; "
            "negative MEME LRT numerical artifacts are clamped to zero"
        ),
        "clamped_negative_meme_lrts": total_clamped_meme_lrts,
        "pearson_r": base.get("pearson_r") if stat_set in {"concordance", "all"} else None,
        "spearman_rho": base.get("spearman_rho") if stat_set in {"concordance", "all"} else None,
        "thresholds": {},
        "per_gene": per_gene,
        "warnings": warnings,
    }

    if stat_set in {"threshold", "all"}:
        for alpha in (0.05, 0.10):
            result["thresholds"][f"{alpha:g}"] = concordance_metrics(
                axo_lrt, axo_p, meme_lrt, meme_p, tested, alpha=alpha, stat_set="threshold"
            )

    return result
