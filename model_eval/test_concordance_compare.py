"""Unit tests for the concordance shared backend (concordance_compare.py).

These are pure parsing/metrics tests — no model weights, no HyPhy. They
verify the shared backend that both the ``python -m model_eval`` CLI and the
pytest concordance tests rely on.
"""
import csv
import json

import numpy as np
import pytest

import concordance_compare as cc


MEME_HEADERS = [
    ["alpha", ""],
    ["beta1", ""],
    ["p1", ""],
    ["beta+", ""],
    ["p+", ""],
    ["LRT", "Likelihood ratio test statistic"],
    ["p-value", "Asymptotic p-value"],
]


def _write_predictions(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["site", "hyphaeon_lrt", "p_value", "q_value", "is_invariable"],
        )
        writer.writeheader()
        for site, lrt, p_value, is_invariable in rows:
            writer.writerow({
                "site": site,
                "hyphaeon_lrt": lrt,
                "p_value": p_value,
                "q_value": p_value,
                "is_invariable": is_invariable,
            })


def _meme_row(lrt, p_value):
    return [0.0, 0.0, 1.0, 0.0, 1.0, lrt, p_value]


def _write_meme(path, rows, coverage=None, partition="0"):
    if coverage is None:
        coverage = list(range(len(rows)))
    path.write_text(json.dumps({
        "input": {"number of sites": len(rows)},
        "MLE": {"headers": MEME_HEADERS, "content": {partition: rows}},
        "data partitions": {partition: {"coverage": [coverage]}},
    }))


# ---------------------------------------------------------------------------
# load_prediction_csv
# ---------------------------------------------------------------------------

def test_load_prediction_csv_basic(tmp_path):
    path = tmp_path / "Gene.csv"
    _write_predictions(path, [
        (1, 4.0, 0.01, False),
        (2, 0.0, 1.0, True),
    ])
    sites = cc.load_prediction_csv(path)
    assert set(sites) == {1, 2}
    assert sites[1].lrt == 4.0
    assert sites[1].p_value == 0.01
    assert sites[1].is_invariable is False
    assert sites[2].is_invariable is True


def test_load_prediction_csv_rejects_negative_lrt(tmp_path):
    path = tmp_path / "Gene.csv"
    _write_predictions(path, [(1, -1.0, 0.5, False)])
    with pytest.raises(cc.ConcordanceError, match="cannot be negative"):
        cc.load_prediction_csv(path)


def test_load_prediction_csv_rejects_duplicate_site(tmp_path):
    path = tmp_path / "Gene.csv"
    _write_predictions(path, [(1, 1.0, 0.5, False), (1, 2.0, 0.4, False)])
    with pytest.raises(cc.ConcordanceError, match="duplicate site"):
        cc.load_prediction_csv(path)


def test_load_prediction_csv_rejects_missing_column(tmp_path):
    path = tmp_path / "Gene.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["site", "hyphaeon_lrt", "p_value"])
        writer.writeheader()
        writer.writerow({"site": 1, "hyphaeon_lrt": 1.0, "p_value": 0.5})
    with pytest.raises(cc.ConcordanceError, match="missing required CSV column"):
        cc.load_prediction_csv(path)


# ---------------------------------------------------------------------------
# load_meme_json
# ---------------------------------------------------------------------------

def test_load_meme_json_uses_partition_coverage(tmp_path):
    path = tmp_path / "Gene.json"
    rows = [_meme_row(3.0, 0.02), _meme_row(5.0, 0.01)]
    # coverage [5, 9] -> global zero-based sites 5 and 9 -> 1-based 6 and 10
    _write_meme(path, rows, coverage=[5, 9])
    sites = cc.load_meme_json(path)
    assert set(sites) == {6, 10}
    assert sites[6].lrt == 3.0
    assert sites[10].p_value == 0.01


def test_load_meme_json_clamps_negative_lrt(tmp_path):
    path = tmp_path / "Gene.json"
    rows = [_meme_row(-1e-9, 0.5)]
    _write_meme(path, rows, coverage=[0])
    sites = cc.load_meme_json(path)
    assert sites[1].lrt == 0.0
    assert sites[1].lrt_was_clamped is True


def test_load_meme_json_multi_partition(tmp_path):
    path = tmp_path / "Gene.json"
    rows_a = [_meme_row(1.0, 0.4)]
    rows_b = [_meme_row(2.0, 0.3)]
    path.write_text(json.dumps({
        "MLE": {"headers": MEME_HEADERS, "content": {"0": rows_a, "1": rows_b}},
        "data partitions": {
            "0": {"coverage": [[0]]},
            "1": {"coverage": [[1]]},
        },
    }))
    sites = cc.load_meme_json(path)
    assert set(sites) == {1, 2}
    assert sites[1].lrt == 1.0
    assert sites[2].lrt == 2.0


# ---------------------------------------------------------------------------
# match_gene_files — Hannah's .json default
# ---------------------------------------------------------------------------

def test_match_gene_files_default_json_suffix(tmp_path):
    pred = tmp_path / "pred"; pred.mkdir()
    meme = tmp_path / "meme"; meme.mkdir()
    _write_predictions(pred / "GeneA.csv", [(1, 1.0, 0.5, False)])
    (meme / "GeneA.json").write_text(json.dumps({
        "MLE": {"headers": MEME_HEADERS, "content": {"0": [_meme_row(1.0, 0.5)]}},
        "data partitions": {"0": {"coverage": [[0]]}},
    }))
    pairs, pred_only, meme_only = cc.match_gene_files(pred, meme)
    assert [g for g, _, _ in pairs] == ["GeneA"]
    assert pred_only == [] and meme_only == []


def test_match_gene_files_memee_json_convention_configurable(tmp_path):
    pred = tmp_path / "pred"; pred.mkdir()
    meme = tmp_path / "meme"; meme.mkdir()
    _write_predictions(pred / "GeneB.csv", [(1, 1.0, 0.5, False)])
    (meme / "GeneB.MEME.json").write_text(json.dumps({
        "MLE": {"headers": MEME_HEADERS, "content": {"0": [_meme_row(1.0, 0.5)]}},
        "data partitions": {"0": {"coverage": [[0]]}},
    }))
    # Default .json suffix would leave gene name "GeneB.MEME" (no match).
    with pytest.raises(cc.ConcordanceError, match="Unmatched"):
        cc.match_gene_files(pred, meme)
    # Explicit .MEME.json suffix pairs correctly.
    pairs, _, _ = cc.match_gene_files(pred, meme, meme_suffix=".MEME.json")
    assert [g for g, _, _ in pairs] == ["GeneB"]


def test_match_gene_files_allow_unmatched(tmp_path):
    pred = tmp_path / "pred"; pred.mkdir()
    meme = tmp_path / "meme"; meme.mkdir()
    _write_predictions(pred / "GeneA.csv", [(1, 1.0, 0.5, False)])
    _write_predictions(pred / "GeneC.csv", [(1, 1.0, 0.5, False)])
    (meme / "GeneA.json").write_text(json.dumps({
        "MLE": {"headers": MEME_HEADERS, "content": {"0": [_meme_row(1.0, 0.5)]}},
        "data partitions": {"0": {"coverage": [[0]]}},
    }))
    pairs, pred_only, meme_only = cc.match_gene_files(pred, meme, allow_unmatched=True)
    assert [g for g, _, _ in pairs] == ["GeneA"]
    assert pred_only == ["GeneC"]


# ---------------------------------------------------------------------------
# meme_sites_to_arrays
# ---------------------------------------------------------------------------

def test_meme_sites_to_arrays_alignment():
    sites = {1: cc.MemeSite(3.0, 0.02), 3: cc.MemeSite(5.0, 0.01)}
    lrts, pvals, tested = cc.meme_sites_to_arrays(sites, 4)
    assert lrts.tolist() == [3.0, 0.0, 5.0, 0.0]
    assert pvals.tolist() == [0.02, 1.0, 0.01, 1.0]
    assert tested.tolist() == [True, False, True, False]


# ---------------------------------------------------------------------------
# concordance_metrics
# ---------------------------------------------------------------------------

def test_concordance_metrics_all():
    axo_lrt = np.array([4.0, 3.0, 2.0, 1.0, 0.0])
    axo_p = np.array([0.01, 0.04, 0.10, 0.50, 1.0])
    meme_lrt = np.array([5.0, 2.0, 2.5, 0.5, 0.0])
    meme_p = np.array([0.01, 0.10, 0.04, 0.80, 1.0])
    tested = np.array([True, True, True, True, False])
    m = cc.concordance_metrics(axo_lrt, axo_p, meme_lrt, meme_p, tested,
                               alpha=0.05, stat_set="all")
    assert m["n_concordance_sites"] == 4
    assert m["spearman_rho"] is not None
    assert m["pearson_r"] is not None
    assert "roc_auc" in m
    assert "cohen_kappa_005" in m
    assert "f1_005" in m


def test_concordance_metrics_concordance_only():
    axo_lrt = np.array([4.0, 3.0, 2.0, 1.0])
    axo_p = np.array([0.01, 0.04, 0.10, 0.50])
    meme_lrt = np.array([5.0, 2.0, 2.5, 0.5])
    meme_p = np.array([0.01, 0.10, 0.04, 0.80])
    tested = np.array([True, True, True, True])
    m = cc.concordance_metrics(axo_lrt, axo_p, meme_lrt, meme_p, tested,
                               stat_set="concordance")
    assert "spearman_rho" in m
    assert "roc_auc" not in m


def test_concordance_metrics_threshold_only():
    axo_lrt = np.array([4.0, 3.0, 2.0, 1.0])
    axo_p = np.array([0.01, 0.04, 0.10, 0.50])
    meme_lrt = np.array([5.0, 2.0, 2.5, 0.5])
    meme_p = np.array([0.01, 0.10, 0.04, 0.80])
    tested = np.array([True, True, True, True])
    m = cc.concordance_metrics(axo_lrt, axo_p, meme_lrt, meme_p, tested,
                               stat_set="threshold")
    assert "roc_auc" in m
    assert "spearman_rho" not in m


def test_concordance_metrics_meme_tested_mask():
    # MEME only tested sites 0 and 2; site 1 should be excluded from scoring.
    axo_lrt = np.array([4.0, 3.0, 2.0])
    axo_p = np.array([0.01, 0.04, 0.10])
    meme_lrt = np.array([5.0, 0.0, 2.5])
    meme_p = np.array([0.01, 1.0, 0.04])
    tested = np.array([True, True, True])
    meme_tested = np.array([True, False, True])
    m = cc.concordance_metrics(axo_lrt, axo_p, meme_lrt, meme_p, tested,
                               meme_tested=meme_tested, stat_set="concordance")
    assert m["n_concordance_sites"] == 2
    assert m["n_meme_tested_sites"] == 2


def test_concordance_metrics_rejects_bad_stat_set():
    arr = np.array([1.0, 2.0])
    with pytest.raises(cc.ConcordanceError, match="stat_set"):
        cc.concordance_metrics(arr, arr, arr, arr, np.array([True, True]), stat_set="bogus")


def test_concordance_metrics_empty_tested_does_not_crash():
    """No sites pass the tested mask — sklearn must not be called on empty arrays."""
    axo_lrt = np.array([1.0, 2.0, 3.0])
    axo_p = np.array([0.5, 0.5, 0.5])
    meme_lrt = np.array([1.0, 2.0, 3.0])
    meme_p = np.array([0.5, 0.5, 0.5])
    tested = np.array([False, False, False])
    m = cc.concordance_metrics(axo_lrt, axo_p, meme_lrt, meme_p, tested, stat_set="all")
    assert m["n_concordance_sites"] == 0
    assert m["pearson_r"] is None
    assert m["spearman_rho"] is None
    assert m["roc_auc"] is None
    assert m["cohen_kappa_005"] is None
    assert m["f1_005"] is None


# ---------------------------------------------------------------------------
# evaluate_pairs (pooling)
# ---------------------------------------------------------------------------

def _write_pair(tmp_path, gene, pred_rows, meme_rows, coverage=None):
    pred = tmp_path / "pred"; pred.mkdir(exist_ok=True)
    meme = tmp_path / "meme"; meme.mkdir(exist_ok=True)
    _write_predictions(pred / f"{gene}.csv", pred_rows)
    _write_meme(meme / f"{gene}.json", meme_rows, coverage=coverage)


def test_evaluate_pairs_pools_sites(tmp_path):
    _write_pair(tmp_path, "GeneA",
                [(1, 4.0, 0.01, False), (2, 0.0, 1.0, True)],
                [_meme_row(5.0, 0.01), _meme_row(0.0, 1.0)])
    pairs, _, _ = cc.match_gene_files(tmp_path / "pred", tmp_path / "meme")
    result = cc.evaluate_pairs(pairs, stat_set="all")
    assert result["matched_genes"] == 1
    assert result["total_sites"] == 2
    assert result["pearson_r"] is not None
    assert "0.05" in result["thresholds"]
    assert "0.1" in result["thresholds"]


def test_evaluate_pairs_variable_only_excludes_invariable(tmp_path):
    _write_pair(tmp_path, "GeneA",
                [(1, 4.0, 0.01, False), (2, 0.0, 1.0, True)],
                [_meme_row(5.0, 0.01), _meme_row(0.0, 1.0)])
    pairs, _, _ = cc.match_gene_files(tmp_path / "pred", tmp_path / "meme")
    result = cc.evaluate_pairs(pairs, variable_only=True, stat_set="all")
    assert result["evaluated_sites"] == 1
    assert result["invariable_sites"] == 1
    assert result["evaluation_scope"] == "variable sites only"


def test_evaluate_pairs_site_mismatch(tmp_path):
    _write_pair(tmp_path, "GeneA",
                [(1, 4.0, 0.01, False), (2, 0.0, 1.0, True)],
                [_meme_row(5.0, 0.01)], coverage=[0])
    pairs, _, _ = cc.match_gene_files(tmp_path / "pred", tmp_path / "meme")
    with pytest.raises(cc.ConcordanceError, match="mismatched sites"):
        cc.evaluate_pairs(pairs)
    # With allow_site_mismatch, the intersection is used.
    result = cc.evaluate_pairs(pairs, allow_site_mismatch=True, stat_set="all")
    assert result["total_sites"] == 1


def test_evaluate_pairs_concordance_only(tmp_path):
    _write_pair(tmp_path, "GeneA",
                [(1, 4.0, 0.01, False), (2, 3.0, 0.04, False)],
                [_meme_row(5.0, 0.01), _meme_row(2.0, 0.10)])
    pairs, _, _ = cc.match_gene_files(tmp_path / "pred", tmp_path / "meme")
    result = cc.evaluate_pairs(pairs, stat_set="concordance")
    assert result["pearson_r"] is not None
    assert result["spearman_rho"] is not None
    assert result["thresholds"] == {}


def test_evaluate_pairs_threshold_only(tmp_path):
    _write_pair(tmp_path, "GeneA",
                [(1, 4.0, 0.01, False), (2, 3.0, 0.04, False)],
                [_meme_row(5.0, 0.01), _meme_row(2.0, 0.10)])
    pairs, _, _ = cc.match_gene_files(tmp_path / "pred", tmp_path / "meme")
    result = cc.evaluate_pairs(pairs, stat_set="threshold")
    assert result["pearson_r"] is None
    assert result["spearman_rho"] is None
    assert "0.05" in result["thresholds"]
    assert "0.1" in result["thresholds"]
