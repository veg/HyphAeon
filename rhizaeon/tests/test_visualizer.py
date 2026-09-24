"""Unit tests for the rhizaeon visualizer module."""

import os
import pytest

from rhizaeon.visualizer import (
    generate_interactive_html,
    compute_alignment_signals,
    compute_classical_mds_subspaces
)


@pytest.fixture
def toy_alignment(tmp_path):
    fasta_path = tmp_path / "toy.fasta"
    # Create 4 taxa with a clean simulated recombination event
    # Taxon_Rec has P1 (A) on left, P2 (G) on right
    seq_rec = "A" * 60 + "G" * 60
    seq_p1 = "A" * 120
    seq_p2 = "G" * 120
    seq_out = "C" * 120

    content = (
        f">Taxon_Rec\n{seq_rec}\n"
        f">Taxon_P1\n{seq_p1}\n"
        f">Taxon_P2\n{seq_p2}\n"
        f">Taxon_Out\n{seq_out}\n"
    )
    fasta_path.write_text(content, encoding="utf-8")
    return str(fasta_path), {
        "Taxon_Rec": seq_rec,
        "Taxon_P1": seq_p1,
        "Taxon_P2": seq_p2,
        "Taxon_Out": seq_out
    }


def test_compute_alignment_signals(toy_alignment):
    _, seqs = toy_alignment
    signals = compute_alignment_signals(
        r_seq=seqs["Taxon_Rec"],
        p1_seq=seqs["Taxon_P1"],
        p2_seq=seqs["Taxon_P2"],
        window_size=20,
        step=5
    )
    assert "trajectory_x" in signals
    assert "trajectory_y" in signals
    assert "informative_snps" in signals
    assert "walk_x" in signals
    assert len(signals["trajectory_x"]) > 0


def test_compute_classical_mds_subspaces(toy_alignment):
    _, seqs = toy_alignment
    taxa = list(seqs.keys())
    partitions = [(0, 60), (60, 120)]
    subspaces = compute_classical_mds_subspaces(taxa=taxa, seqs=seqs, partitions=partitions)
    assert len(subspaces) == 2
    assert "Taxon_Rec" in subspaces[0]
    assert "Taxon_P1" in subspaces[0]
    assert len(subspaces[0]["Taxon_Rec"]) == 2  # 2D coordinates


def test_generate_interactive_html(toy_alignment, tmp_path):
    fasta_path, _ = toy_alignment
    out_html = str(tmp_path / "dashboard.html")
    res_path = generate_interactive_html(
        alignment_path=fasta_path,
        detection_results=None,
        output_html_path=out_html,
        title="Test Interactive Dashboard"
    )
    assert os.path.exists(res_path)
    with open(res_path, "r", encoding="utf-8") as f:
        html = f.read()
    assert "<!DOCTYPE html>" in html
    assert "Test Interactive Dashboard" in html
    assert "RhizAeon Recombination Explorer" in html
    assert "Taxon_Rec" in html
