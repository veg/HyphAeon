"""Unit tests for load_alignment_and_tree in hyphaeon.dataset."""
import numpy as np
import pytest
import torch
from hyphaeon.dataset import load_alignment_and_tree


class TestLoadAlignmentAndTreeFasta:
    def test_basic_fasta_newick(self, fasta_file, newick_file):
        c, a, d, z, inv, taxa, L = load_alignment_and_tree(fasta_file, newick_file)
        assert L == 4  # 12 bp / 3
        assert set(taxa) == {"seq1", "seq2", "seq3"}
        assert len(taxa) == 3

    def test_tensor_shapes(self, fasta_file, newick_file):
        c, a, d, z, inv, taxa, L = load_alignment_and_tree(fasta_file, newick_file)
        n = len(taxa)
        assert c.shape == (L, n, 1)
        assert a.shape == (L, n, 1)
        # d and z are broadcastable (1, N, N) / (1, N, 4) — the CLI expands
        # them per site-chunk at inference time, avoiding L× memory.
        assert d.shape == (1, n, n)
        assert z.shape == (1, n, 4)
        assert inv.shape == (L,)

    def test_tensor_dtypes(self, fasta_file, newick_file):
        c, a, d, z, inv, taxa, L = load_alignment_and_tree(fasta_file, newick_file)
        assert c.dtype == torch.long
        assert a.dtype == torch.long
        assert d.dtype == torch.float32
        assert z.dtype == torch.float32
        assert inv.dtype == np.bool_

    def test_invariable_detection(self, fasta_file, newick_file):
        c, a, d, z, inv, taxa, L = load_alignment_and_tree(fasta_file, newick_file)
        # seq1=ATGTTTCTTGGT, seq2=ATGTTCCTTGGT, seq3=ATGTTTCTCGGT
        # Codons: seq1=ATG TTT CTT GGT, seq2=ATG TTC CTT GGT, seq3=ATG TTT CTC GGT
        # Site 1: M,M,M -> invariable
        # Site 2: F,F,F -> invariable
        # Site 3: L,L,L -> invariable
        # Site 4: G,G,G -> invariable
        assert inv[0] == True  # ATG
        assert inv[1] == True  # TTT/TTC -> F/F/F
        assert inv[2] == True  # CTT/CTT/CTC -> L/L/L
        assert inv[3] == True  # GGT/GGT/GGT -> G/G/G

    def test_variable_site_detection(self, tmp_path, newick_file):
        fa = tmp_path / "var.fa"
        fa.write_text(">seq1\nATGAAATTTGGT\n>seq2\nATGAGGTTTGGT\n>seq3\nATGAAATTTGGT\n")
        c, a, d, z, inv, taxa, L = load_alignment_and_tree(str(fa), newick_file)
        # Site 2: AAA/AGG/AAA -> K/R/K -> variable
        assert inv[1] == False

    def test_embedded_tree_from_nexus(self, nexus_file):
        c, a, d, z, inv, taxa, L = load_alignment_and_tree(nexus_file, None)
        assert L == 4
        assert set(taxa) == {"seq1", "seq2", "seq3"}

    def test_no_tree_no_embedded_raises(self, fasta_file):
        with pytest.raises(ValueError, match="No tree specified"):
            load_alignment_and_tree(fasta_file, None)

    def test_empty_alignment_raises(self, tmp_path, newick_file):
        fa = tmp_path / "empty.fa"
        fa.write_text("")
        with pytest.raises(ValueError, match="Could not parse"):
            load_alignment_and_tree(str(fa), newick_file)

    def test_no_matching_taxa_raises(self, tmp_path, newick_file):
        fa = tmp_path / "mismatch.fa"
        fa.write_text(">completely_different\nATGTTTCTTGGT\n")
        with pytest.raises(ValueError, match="No matching taxa"):
            load_alignment_and_tree(str(fa), newick_file)

    def test_short_sequence_raises(self, tmp_path, newick_file):
        fa = tmp_path / "short.fa"
        fa.write_text(">seq1\nAT\n>seq2\nAT\n>seq3\nAT\n")
        with pytest.raises(ValueError, match="less than 1 codon"):
            load_alignment_and_tree(str(fa), newick_file)

    def test_branchless_tree_gets_enforced(self, fasta_file, tmp_path):
        nwk = tmp_path / "no_branch.nwk"
        nwk.write_text("((seq1,seq2),seq3);\n")
        c, a, d, z, inv, taxa, L = load_alignment_and_tree(fasta_file, str(nwk))
        # Distance matrix should be non-zero (enforced minimum branch lengths)
        assert not np.allclose(d[0].numpy(), 0.0)

    def test_distance_matrix_consistency(self, fasta_file, newick_file):
        c, a, d, z, inv, taxa, L = load_alignment_and_tree(fasta_file, newick_file)
        dist = d[0].numpy()  # d is (1, N, N), d[0] is the (N, N) matrix
        assert np.allclose(np.diag(dist), 0.0)
        assert np.allclose(dist, dist.T)
