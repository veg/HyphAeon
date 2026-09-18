"""
test_cli.py
-----------
End-to-end CLI integration tests for chronaeon subcommands that lack
dedicated CLI tests: date, triage, autoclock, sketch, align.

Existing CLI coverage:
  - test_geo.py::test_geo_worked_example_cli  (geo subcommand)
  - test_r0.py::test_cli_r0_example           (r0 subcommand)
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _run_cli(args):
    return subprocess.run(
        [sys.executable, "-m", "chronaeon.cli"] + args,
        capture_output=True,
        text=True,
    )


class TestDateCLI:
    def test_date_example(self, tmp_path):
        """Test the date subcommand with the bundled H1N1 example."""
        aln = EXAMPLES_DIR / "H1N1_2009_pandemic.fasta"
        if not aln.exists():
            pytest.skip("H1N1 example not found")

        out_json = str(tmp_path / "dating.json")
        result = _run_cli([
            "date",
            "-a", str(aln),
            "--clock-model", "auto",
            "--method", "ols",
            "-o", out_json,
        ])
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        assert os.path.exists(out_json)

    def test_date_nonlinear_clocks(self, tmp_path):
        """Test the date subcommand with --nonlinear-clocks."""
        import json
        aln = EXAMPLES_DIR / "H1N1_2009_pandemic.fasta"
        if not aln.exists():
            pytest.skip("H1N1 example not found")

        out_json = str(tmp_path / "dating_nl.json")
        result = _run_cli([
            "date",
            "-a", str(aln),
            "--clock-model", "linear",
            "--method", "ols",
            "--nonlinear-clocks",
            "-o", out_json,
        ])
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        assert os.path.exists(out_json)
        with open(out_json) as f:
            data = json.load(f)
        assert "nonlinear_clocks" in data
        assert "dudas_models" in data
        assert "quadratic" in data["nonlinear_clocks"]
        assert "exponential" in data["nonlinear_clocks"]
        assert "bilinear_crash" in data["nonlinear_clocks"]
        assert "polyepoch" in data["nonlinear_clocks"]


class TestTriageCLI:
    def test_triage_runs(self, tmp_path):
        """Test the triage subcommand with a synthetic alignment (>=10 dated taxa required)."""
        ref_seq = "ACGT" * 50

        ref_lines = []
        date_lines = ["genome_id,collection_date"]
        for i in range(12):
            date = 2000.0 + i * 2.0
            seq = ref_seq[:180] + "G" * i + ref_seq[180 + i:]
            ref_lines.append(f">taxon_{i}\n{seq}")
            date_lines.append(f"taxon_{i},{date}")
        ref_lines.append(f">ref_2020\n{ref_seq}")
        date_lines.append("ref_2020,2020.0")

        ref_path = tmp_path / "ref.fasta"
        ref_path.write_text("\n".join(ref_lines) + "\n")

        stream_seq = "ACGT" * 45 + "NNNN" * 5
        stream_path = tmp_path / "stream.fasta"
        stream_path.write_text(f">query_2021\n{stream_seq}\n")

        dates_path = tmp_path / "dates.csv"
        dates_path.write_text("\n".join(date_lines) + "\n")

        out_json = str(tmp_path / "triage.json")
        result = _run_cli([
            "triage",
            "-a", str(ref_path),
            "-d", str(dates_path),
            "-s", str(stream_path),
            "--root-taxon", "ref_2020",
            "-o", out_json,
        ])
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        assert os.path.exists(out_json)


class TestSketchCLI:
    def test_sketch_runs(self, tmp_path):
        """Test the sketch subcommand with a small synthetic FASTA."""
        seq = "ATGGAGAAAATAGTGCTTCTTTAGCGATCGATCGATCGATCGATCGATC"
        fasta_path = tmp_path / "input.fasta"
        fasta_path.write_text(f">seq1\n{seq}\n>seq2\n{seq[:40] + 'AAA' + seq[43:]}\n")

        out_prefix = str(tmp_path / "sketch_out")
        result = _run_cli([
            "sketch",
            "-a", str(fasta_path),
            "-o", out_prefix,
            "--quiet",
        ])
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"


class TestAlignCLI:
    def test_align_runs(self, tmp_path):
        """Test the align subcommand with a small reference and query."""
        ref_cds = "ATGGAGAAAATAGTGCTTCTTCTTGCAATAGTCAGTCTTGTTAAAAGT"
        query = ref_cds[:12] + ref_cds[15:]

        ref_path = tmp_path / "ref.fasta"
        ref_path.write_text(f">ref\n{ref_cds}\n")

        query_path = tmp_path / "query.fasta"
        query_path.write_text(f">query\n{query}\n")

        out_path = str(tmp_path / "aligned.fasta")
        result = _run_cli([
            "align",
            "-r", str(ref_path),
            "-q", str(query_path),
            "-o", out_path,
            "--quiet",
        ])
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        assert os.path.exists(out_path)


class TestAutoclockCLI:
    def test_autoclock_runs(self, tmp_path):
        """Test the autoclock subcommand with a small synthetic FASTA."""
        fasta_content = ""
        for i in range(5):
            date = 2000 + i * 5
            seq = "ATGGCC" + "A" * (20 - i) + "C" * i + "T" * 20
            fasta_content += f">seq_{date}\n{seq}\n"

        fasta_path = tmp_path / "aln.fasta"
        fasta_path.write_text(fasta_content)

        out_dir = str(tmp_path / "autoclock_out")
        result = _run_cli([
            "autoclock",
            "-a", str(fasta_path),
            "-k", "2",
            "--manifold", "tn93",
            "--output-dir", out_dir,
            "--quiet",
        ])
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
