"""Unit tests for the rhizaeon CLI."""

import os
import sys
import json
import tempfile
import pytest
from unittest.mock import patch

from rhizaeon.cli import main


def test_cli_help(capsys):
    with patch.object(sys, "argv", ["rhizaeon", "--help"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "RhizAeon: Tree-Free Reticulate Evolution" in captured.out
    assert "scan" in captured.out
    assert "rp-fda" in captured.out
    assert "alluvial" in captured.out
    assert "visualize" in captured.out


def test_cli_scan_toy():
    # Create temporary toy FASTA
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "toy.fasta")
        json_path = os.path.join(tmpdir, "report.json")
        with open(fasta_path, "w") as f:
            f.write(">Taxon_Rec\n" + ("AAA" * 20 + "GGG" * 20) + "\n")
            f.write(">Taxon_P1\n" + ("AAA" * 40) + "\n")
            f.write(">Taxon_P2\n" + ("GGG" * 40) + "\n")
            f.write(">Taxon_Out\n" + ("CCC" * 40) + "\n")

        with patch.object(sys, "argv", [
            "rhizaeon", "scan", fasta_path,
            "--engine", "scalar",
            "--window", "10",
            "--min-tract", "10",
            "--json", json_path
        ]):
            main()

        assert os.path.exists(json_path)
        with open(json_path) as f:
            data = json.load(f)
        assert data["num_taxa"] == 4
        assert data["units"] == 40


def test_cli_rpfda_toy():
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "toy.fasta")
        json_path = os.path.join(tmpdir, "rpfda.json")
        with open(fasta_path, "w") as f:
            f.write(">Taxon_Rec\n" + ("A" * 60 + "G" * 60) + "\n")
            f.write(">Taxon_P1\n" + ("A" * 120) + "\n")
            f.write(">Taxon_P2\n" + ("G" * 120) + "\n")
            f.write(">Taxon_Out\n" + ("C" * 120) + "\n")

        with patch.object(sys, "argv", [
            "rhizaeon", "rp-fda", fasta_path,
            "--min-len", "20",
            "--min-z", "1.0",
            "--json", json_path
        ]):
            main()

        assert os.path.exists(json_path)
        with open(json_path) as f:
            data = json.load(f)
        assert data["num_taxa"] == 4
        assert data["length"] == 120


def test_cli_visualize():
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "toy.fasta")
        html_path = os.path.join(tmpdir, "dashboard.html")
        with open(fasta_path, "w") as f:
            f.write(">Taxon_Rec\n" + ("A" * 60 + "G" * 60) + "\n")
            f.write(">Taxon_P1\n" + ("A" * 120) + "\n")
            f.write(">Taxon_P2\n" + ("G" * 120) + "\n")
            f.write(">Taxon_Out\n" + ("C" * 120) + "\n")

        with patch.object(sys, "argv", [
            "rhizaeon", "visualize", fasta_path,
            "--output", html_path
        ]):
            main()

        assert os.path.exists(html_path)
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "<!DOCTYPE html>" in content
        assert "RhizAeon" in content


def test_cli_scan_and_rpfda_exports():
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "toy.fasta")
        with open(fasta_path, "w") as f:
            f.write(">Taxon_Rec\n" + ("A" * 60 + "G" * 60) + "\n")
            f.write(">Taxon_P1\n" + ("A" * 120) + "\n")
            f.write(">Taxon_P2\n" + ("G" * 120) + "\n")
            f.write(">Taxon_Out\n" + ("C" * 120) + "\n")

        nex_path = os.path.join(tmpdir, "part.nex")
        hyphy_json = os.path.join(tmpdir, "part.json")
        hyphy_bf = os.path.join(tmpdir, "part.bf")
        part_dir = os.path.join(tmpdir, "parts")

        with patch.object(sys, "argv", [
            "rhizaeon", "rp-fda", fasta_path,
            "--min-len", "20",
            "--min-z", "1.0",
            "--no-html",
            "--export-nexus", nex_path,
            "--export-hyphy-json", hyphy_json,
            "--export-hyphy-bf", hyphy_bf,
            "--export-partitions", part_dir
        ]):
            main()

        assert os.path.exists(nex_path)
        assert os.path.exists(hyphy_json)
        assert os.path.exists(hyphy_bf)
        assert os.path.isdir(part_dir)
        part_files = os.listdir(part_dir)
        assert len(part_files) >= 1

