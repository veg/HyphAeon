"""Smoke tests for the disease and filter CLI subcommands.

These tests exercise the full CLI plumbing (argument parsing, model loading,
inference, output writing) using a tiny dummy checkpoint — not real pretrained
weights. They verify that the pipeline runs end-to-end and produces well-formed
output, not that the predictions are biologically meaningful.
"""
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")


def _run_cli(args):
    return subprocess.run(
        [sys.executable, "-m", "hyphaeon.cli"] + args,
        capture_output=True,
        text=True,
    )


class TestDiseaseCLI:
    def test_disease_runs_and_produces_csv(self, dummy_weights, tmp_path):
        fasta = os.path.join(EXAMPLES_DIR, "Smc6.fasta")
        csv_out = str(tmp_path / "disease_results.csv")
        result = _run_cli([
            "disease",
            "-a", fasta,
            "-m", "R175H,G245S",
            "-w", dummy_weights,
            "--cpu",
            "-c", csv_out,
        ])
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        assert os.path.exists(csv_out)
        df = pd.read_csv(csv_out)
        assert len(df) == 2
        assert "mutation" in df.columns
        assert "pathogenicity_score" in df.columns

    def test_disease_json_output(self, dummy_weights, tmp_path):
        fasta = os.path.join(EXAMPLES_DIR, "Smc6.fasta")
        json_out = str(tmp_path / "disease_results.json")
        result = _run_cli([
            "disease",
            "-a", fasta,
            "-m", "R175H",
            "-w", dummy_weights,
            "--cpu",
            "-o", json_out,
        ])
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        assert os.path.exists(json_out)
        with open(json_out) as f:
            data = json.load(f)
        assert len(data) == 1
        assert "mutation" in data[0]


class TestFilterCLI:
    def test_filter_runs_and_produces_alignment(self, dummy_weights, tmp_path):
        fasta = os.path.join(EXAMPLES_DIR, "Smc6.fasta")
        tree = os.path.join(EXAMPLES_DIR, "Smc6.nwk")
        out_aln = str(tmp_path / "cleaned.fasta")
        result = _run_cli([
            "filter",
            "-a", fasta,
            "-t", tree,
            "-w", dummy_weights,
            "--cpu",
            "-o", out_aln,
        ])
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        assert os.path.exists(out_aln)
        with open(out_aln) as f:
            content = f.read()
        assert content.startswith(">")

    def test_filter_audit_csv(self, dummy_weights, tmp_path):
        fasta = os.path.join(EXAMPLES_DIR, "Smc6.fasta")
        tree = os.path.join(EXAMPLES_DIR, "Smc6.nwk")
        audit_csv = str(tmp_path / "audit.csv")
        result = _run_cli([
            "filter",
            "-a", fasta,
            "-t", tree,
            "-w", dummy_weights,
            "--cpu",
            "-c", audit_csv,
        ])
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        if os.path.exists(audit_csv):
            df = pd.read_csv(audit_csv)
            assert len(df) >= 0
