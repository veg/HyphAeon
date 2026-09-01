import csv
import json

import pytest

from hyphaeon.evaluation import (
    EvaluationError,
    evaluate_directories,
    evaluate_files,
    load_meme_json,
    main,
)


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
            writer.writerow(
                {
                    "site": site,
                    "hyphaeon_lrt": lrt,
                    "p_value": p_value,
                    "q_value": p_value,
                    "is_invariable": is_invariable,
                }
            )


def _meme_row(lrt, p_value):
    return [0.0, 0.0, 1.0, 0.0, 1.0, lrt, p_value]


def _write_meme(path, rows, coverage=None):
    if coverage is None:
        coverage = list(range(len(rows)))
    path.write_text(
        json.dumps(
            {
                "input": {"number of sites": len(rows)},
                "MLE": {"headers": MEME_HEADERS, "content": {"0": rows}},
                "data partitions": {"0": {"coverage": [coverage]}},
            }
        )
    )


def _paired_directories(tmp_path):
    predictions = tmp_path / "predictions"
    meme = tmp_path / "meme"
    predictions.mkdir()
    meme.mkdir()
    return predictions, meme


def test_calculates_requested_metrics_with_inclusive_threshold_rules(tmp_path):
    predictions, meme = _paired_directories(tmp_path)
    _write_predictions(
        predictions / "Gene1.csv",
        [
            (1, 4.0, 0.01, False),
            (2, 3.0, 0.10, False),
            (3, 2.0, 0.04, False),
            (4, 1.0, 0.80, True),
        ],
    )
    _write_meme(
        meme / "Gene1.MEME.json",
        [
            _meme_row(40.0, 0.01),
            _meme_row(30.0, 0.10),
            _meme_row(20.0, 0.20),
            _meme_row(10.0, 0.50),
        ],
    )

    result = evaluate_directories(predictions, meme)

    assert result["matched_genes"] == 1
    assert result["total_sites"] == 4
    assert result["evaluated_sites"] == 4
    assert result["variable_sites"] == 3
    assert result["invariable_sites"] == 1
    assert result["pearson_r"] == pytest.approx(1.0)
    assert result["spearman_rho"] == pytest.approx(1.0)

    alpha_005 = result["thresholds"]["0.05"]
    assert alpha_005["roc_auc"] == pytest.approx(1.0)
    assert alpha_005["ppv"] == pytest.approx(0.5)
    assert alpha_005["fpr"] == pytest.approx(1 / 3)

    alpha_010 = result["thresholds"]["0.10"]
    assert alpha_010["roc_auc"] == pytest.approx(1.0)
    assert alpha_010["ppv"] == pytest.approx(2 / 3)
    assert alpha_010["ppv_confusion_matrix"] == {
        "true_positive": 2,
        "false_positive": 1,
        "true_negative": 1,
        "false_negative": 0,
    }
    # p == 0.10 is significant for every metric, including FPR.
    assert alpha_010["fpr"] == pytest.approx(1 / 2)
    assert alpha_010["fpr_confusion_matrix"] == {
        "true_positive": 2,
        "false_positive": 1,
        "true_negative": 1,
        "false_negative": 0,
    }
    assert alpha_010["fpr_definition"] == "MEME and HyphAeon p_value <= 0.1"


def test_pools_sites_across_genes_and_can_limit_metrics_to_variable_sites(tmp_path):
    predictions, meme = _paired_directories(tmp_path)
    _write_predictions(
        predictions / "GeneA.csv",
        [(1, 1.0, 0.2, False), (2, 2.0, 0.1, True)],
    )
    _write_meme(
        meme / "GeneA.MEME.json",
        [_meme_row(10.0, 0.2), _meme_row(20.0, 0.1)],
    )
    _write_predictions(
        predictions / "GeneB.csv",
        [(1, 3.0, 0.05, False), (2, 4.0, 0.01, False)],
    )
    _write_meme(
        meme / "GeneB.MEME.json",
        [_meme_row(30.0, 0.05), _meme_row(40.0, 0.01)],
    )

    all_sites = evaluate_directories(predictions, meme)
    variable_sites = evaluate_directories(predictions, meme, variable_only=True)

    assert all_sites["genes"] == ["GeneA", "GeneB"]
    assert all_sites["total_sites"] == 4
    assert all_sites["evaluated_sites"] == 4
    assert all_sites["pearson_r"] == pytest.approx(1.0)
    assert variable_sites["total_sites"] == 4
    assert variable_sites["evaluated_sites"] == 3
    assert variable_sites["evaluation_scope"] == "variable sites only"


def test_meme_partition_coverage_maps_rows_to_global_sites(tmp_path):
    path = tmp_path / "Gene.MEME.json"
    path.write_text(
        json.dumps(
            {
                "MLE": {
                    "headers": MEME_HEADERS,
                    "content": {
                        "1": [_meme_row(5.0, 0.5)],
                        "0": [_meme_row(1.0, 0.1), _meme_row(3.0, 0.3)],
                    },
                },
                "data partitions": {
                    "0": {"coverage": [[0, 2]]},
                    "1": {"coverage": [[4]]},
                },
            }
        )
    )

    sites = load_meme_json(path)

    assert sorted(sites) == [1, 3, 5]
    assert sites[3].lrt == pytest.approx(3.0)
    assert sites[5].p_value == pytest.approx(0.5)


def test_negative_meme_lrt_numerical_artifacts_are_clamped(tmp_path):
    predictions, meme = _paired_directories(tmp_path)
    _write_predictions(predictions / "Gene1.csv", [(1, 0.0, 1.0, False), (2, 1.0, 0.5, False)])
    _write_meme(meme / "Gene1.MEME.json", [_meme_row(-0.001, 1.0), _meme_row(1.0, 0.5)])

    result = evaluate_directories(predictions, meme)

    assert result["clamped_negative_meme_lrts"] == 1
    assert result["pearson_r"] == pytest.approx(1.0)
    assert "Clamped 1 negative MEME LRT" in result["warnings"][0]


def test_unmatched_genes_fail_by_default_and_can_be_ignored(tmp_path):
    predictions, meme = _paired_directories(tmp_path)
    _write_predictions(predictions / "Gene1.csv", [(1, 1.0, 0.2, False)])
    _write_predictions(predictions / "PredictionOnly.csv", [(1, 1.0, 0.2, False)])
    _write_meme(meme / "Gene1.MEME.json", [_meme_row(1.0, 0.2)])

    with pytest.raises(EvaluationError, match="prediction-only genes: PredictionOnly"):
        evaluate_directories(predictions, meme)

    result = evaluate_directories(predictions, meme, allow_unmatched=True)
    assert result["matched_genes"] == 1
    assert result["warnings"] == ["Ignored prediction-only genes: PredictionOnly"]


def test_evaluates_one_gene_from_directly_supplied_files(tmp_path):
    prediction = tmp_path / "Gene1.csv"
    meme = tmp_path / "Gene1.MEME.json"
    _write_predictions(prediction, [(1, 2.0, 0.1, False), (2, 1.0, 0.2, False)])
    _write_meme(meme, [_meme_row(2.0, 0.1), _meme_row(1.0, 0.2)])

    result = evaluate_files(prediction, meme)

    assert result["input_mode"] == "files"
    assert result["prediction_file"] == str(prediction.resolve())
    assert result["meme_result_file"] == str(meme.resolve())
    assert result["genes"] == ["Gene1"]
    assert result["matched_genes"] == 1
    assert result["total_sites"] == 2


def test_direct_file_mode_rejects_mismatched_gene_names(tmp_path):
    prediction = tmp_path / "Gene1.csv"
    meme = tmp_path / "Gene2.MEME.json"
    _write_predictions(prediction, [(1, 1.0, 0.2, False)])
    _write_meme(meme, [_meme_row(1.0, 0.2)])

    with pytest.raises(EvaluationError, match="prediction gene 'Gene1'.*MEME result gene 'Gene2'"):
        evaluate_files(prediction, meme)


def test_direct_file_cli_flags_emit_single_gene_json(tmp_path, capsys):
    prediction = tmp_path / "Gene1.csv"
    meme = tmp_path / "Gene1.MEME.json"
    _write_predictions(prediction, [(1, 2.0, 0.1, False), (2, 1.0, 0.2, False)])
    _write_meme(meme, [_meme_row(2.0, 0.1), _meme_row(1.0, 0.2)])

    exit_code = main(
        [
            "--prediction",
            str(prediction),
            "--meme-result",
            str(meme),
            "--format",
            "json",
        ]
    )

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["input_mode"] == "files"
    assert report["genes"] == ["Gene1"]


def test_site_mismatches_fail_by_default_and_intersection_is_reported(tmp_path):
    predictions, meme = _paired_directories(tmp_path)
    _write_predictions(
        predictions / "Gene1.csv",
        [(1, 2.0, 0.1, False), (2, 1.0, 0.2, False)],
    )
    _write_meme(
        meme / "Gene1.MEME.json",
        [_meme_row(2.0, 0.1), _meme_row(1.0, 0.2)],
        coverage=[0, 2],
    )

    with pytest.raises(EvaluationError, match="mismatched sites"):
        evaluate_directories(predictions, meme)

    result = evaluate_directories(predictions, meme, allow_site_mismatch=True)
    assert result["total_sites"] == 1
    assert result["total_prediction_sites"] == 2
    assert result["total_meme_sites"] == 2
    assert "dropped 1 prediction-only and 1 MEME-only sites" in result["warnings"][0]


def test_undefined_metrics_are_json_safe_none(tmp_path):
    predictions, meme = _paired_directories(tmp_path)
    _write_predictions(predictions / "Gene1.csv", [(1, 0.0, 1.0, True), (2, 0.0, 1.0, True)])
    _write_meme(meme / "Gene1.MEME.json", [_meme_row(0.0, 1.0), _meme_row(0.0, 1.0)])

    result = evaluate_directories(predictions, meme)

    assert result["pearson_r"] is None
    assert result["spearman_rho"] is None
    assert result["thresholds"]["0.05"]["roc_auc"] is None
    assert result["thresholds"]["0.05"]["ppv"] is None
    json.dumps(result, allow_nan=False)
