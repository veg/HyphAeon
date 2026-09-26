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


def test_cli_html_flags_scan():
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "sample.fasta")
        with open(fasta_path, "w") as f:
            f.write(">Taxon_Rec\n" + ("AAA" * 20 + "GGG" * 20) + "\n")
            f.write(">Taxon_P1\n" + ("AAA" * 40) + "\n")
            f.write(">Taxon_P2\n" + ("GGG" * 40) + "\n")
            f.write(">Taxon_Out\n" + ("CCC" * 40) + "\n")

        # 1. Custom HTML path
        custom_html = os.path.join(tmpdir, "custom_scan.html")
        with patch.object(sys, "argv", [
            "rhizaeon", "scan", fasta_path,
            "--engine", "scalar",
            "--window", "10",
            "--min-tract", "10",
            "--html", custom_html
        ]):
            main()
        assert os.path.exists(custom_html)
        with open(custom_html, "r", encoding="utf-8") as f:
            content = f.read()
        assert "RhizAeon Recombination Explorer" in content
        assert "Contiguous Ancestral Mosaic Architecture" in content

        # 2. Flag --no-html suppresses HTML
        no_html_target = os.path.join(tmpdir, "should_not_exist.html")
        with patch.object(sys, "argv", [
            "rhizaeon", "scan", fasta_path,
            "--engine", "scalar",
            "--window", "10",
            "--min-tract", "10",
            "--html", no_html_target,
            "--no-html"
        ]):
            main()
        assert not os.path.exists(no_html_target)


def test_cli_html_flags_rpfda():
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "sample.fasta")
        with open(fasta_path, "w") as f:
            f.write(">Taxon_Rec\n" + ("A" * 60 + "G" * 60) + "\n")
            f.write(">Taxon_P1\n" + ("A" * 120) + "\n")
            f.write(">Taxon_P2\n" + ("G" * 120) + "\n")
            f.write(">Taxon_Out\n" + ("C" * 120) + "\n")

        # 1. Custom HTML path
        custom_html = os.path.join(tmpdir, "custom_rpfda.html")
        with patch.object(sys, "argv", [
            "rhizaeon", "rp-fda", fasta_path,
            "--min-len", "20",
            "--min-z", "1.0",
            "--html", custom_html
        ]):
            main()
        assert os.path.exists(custom_html)
        with open(custom_html, "r", encoding="utf-8") as f:
            content = f.read()
        assert "RhizAeon Recombination Explorer" in content
        assert "Alluvial Corridors" in content

        # 2. --no-html suppresses HTML
        no_html_target = os.path.join(tmpdir, "suppressed.html")
        with patch.object(sys, "argv", [
            "rhizaeon", "rp-fda", fasta_path,
            "--min-len", "20",
            "--min-z", "1.0",
            "--html", no_html_target,
            "--no-html"
        ]):
            main()
        assert not os.path.exists(no_html_target)


def test_cli_visualize_flags():
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "sample.fasta")
        with open(fasta_path, "w") as f:
            f.write(">Taxon_Rec\n" + ("A" * 60 + "G" * 60) + "\n")
            f.write(">Taxon_P1\n" + ("A" * 120) + "\n")
            f.write(">Taxon_P2\n" + ("G" * 120) + "\n")
            f.write(">Taxon_Out\n" + ("C" * 120) + "\n")

        # 1. --no-html suppresses generation
        out_html = os.path.join(tmpdir, "suppressed_viz.html")
        with patch.object(sys, "argv", [
            "rhizaeon", "visualize", fasta_path,
            "--output", out_html,
            "--no-html"
        ]):
            main()
        assert not os.path.exists(out_html)


def test_cli_rpfda_kal153(capsys):
    example_path = os.path.join(os.path.dirname(__file__), "..", "examples", "hiv1_kal153.fasta")
    if not os.path.exists(example_path):
        pytest.skip("examples/hiv1_kal153.fasta not found")

    with tempfile.TemporaryDirectory() as tmpdir:
        json_out = os.path.join(tmpdir, "kal153.json")
        with patch.object(sys, "argv", [
            "rhizaeon", "rp-fda", example_path,
            "--no-html",
            "--json", json_out
        ]):
            main()

        assert os.path.exists(json_out)
        with open(json_out) as f:
            data = json.load(f)

        captured = capsys.readouterr()
        # Verify terminal output includes Primary Mosaic Genomes with R
        assert "Primary Mosaic Genomes" in captured.out
        assert "R" in captured.out
        assert "[A]──2,800──[B]──8,842──[A]" in captured.out

        # Verify JSON contains R's breakpoints
        bps = data.get("breakpoints", [])
        r_bps = [b for b in bps if b.get("recombinant") == "R"]
        assert len(r_bps) == 2
        coords = sorted([b.get("breakpoint_nt", b.get("breakpoint")) for b in r_bps])
        assert abs(coords[0] - 2800) <= 20
        assert abs(coords[1] - 8842) <= 20



