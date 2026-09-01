"""Unit tests for alignment and tree parsing functions in hyphaeon.dataset."""
import gzip
import pytest
from Bio import Phylo
from hyphaeon.dataset import (
    parse_alignment_sequences,
    extract_tree_from_string_or_file,
    has_nonzero_branch_lengths,
    enforce_nonzero_branch_lengths,
)


class TestParseAlignmentSequences:
    def test_fasta_basic(self, fasta_file):
        seqs = parse_alignment_sequences(fasta_file)
        assert len(seqs) == 3
        assert "seq1" in seqs
        assert "seq2" in seqs
        assert "seq3" in seqs
        assert seqs["seq1"] == "ATGTTTCTTGGT"
        assert seqs["seq2"] == "ATGTTCCTTGGT"

    def test_fasta_uppercase(self, tmp_path):
        p = tmp_path / "lower.fa"
        p.write_text(">seq1\natgtttcttggt\n>seq2\natgttccttggt\n")
        seqs = parse_alignment_sequences(str(p))
        assert seqs["seq1"] == "ATGTTTCTTGGT"

    def test_fasta_gzipped(self, tmp_path, fasta_content):
        p = tmp_path / "test.fa.gz"
        with gzip.open(p, "wt") as f:
            f.write(fasta_content)
        seqs = parse_alignment_sequences(str(p))
        assert len(seqs) == 3
        assert seqs["seq1"] == "ATGTTTCTTGGT"

    def test_fasta_with_embedded_tree_stops_at_tree(self, tmp_path):
        p = tmp_path / "embedded.fa"
        p.write_text(">seq1\nATGTTTCTTGGT\n>seq2\nATGTTCCTTGGT\n\n((seq1:0.1,seq2:0.1):0.05,seq3:0.15);\n")
        seqs = parse_alignment_sequences(str(p))
        assert len(seqs) == 2
        assert "seq1" in seqs

    def test_nexus_labeled_matrix(self, nexus_file):
        seqs = parse_alignment_sequences(nexus_file)
        assert len(seqs) == 3
        assert seqs["seq1"] == "ATGTTTCTTGGT"
        assert seqs["seq2"] == "ATGTTCCTTGGT"
        assert seqs["seq3"] == "ATGTTTCTCGGT"

    def test_nexus_nolabels(self, tmp_path):
        content = """#NEXUS
BEGIN TAXA;
    DIMENSIONS NTAX=2;
    TAXLABELS taxA taxB;
END;
BEGIN CHARACTERS;
    DIMENSIONS NCHAR=6;
    FORMAT DATATYPE=DNA MISSING=? GAP=- NOLABELS;
    MATRIX
ATGTTT
ATGTTC
    ;
END;
"""
        p = tmp_path / "nolabels.nex"
        p.write_text(content)
        seqs = parse_alignment_sequences(str(p))
        assert len(seqs) == 2
        assert seqs["taxA"] == "ATGTTT"
        assert seqs["taxB"] == "ATGTTC"

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.fa"
        p.write_text("")
        seqs = parse_alignment_sequences(str(p))
        assert len(seqs) == 0

    def test_multiline_fasta(self, tmp_path):
        p = tmp_path / "multi.fa"
        p.write_text(">seq1\nATGTTT\nCTTGGT\n>seq2\nATGTTC\nCTTGGT\n")
        seqs = parse_alignment_sequences(str(p))
        assert seqs["seq1"] == "ATGTTTCTTGGT"
        assert seqs["seq2"] == "ATGTTCCTTGGT"


class TestExtractTreeFromStringOrFile:
    def test_newick_file(self, newick_file):
        tree = extract_tree_from_string_or_file(newick_file)
        assert tree is not None
        terminals = [t.name for t in tree.get_terminals()]
        assert set(terminals) == {"seq1", "seq2", "seq3"}

    def test_newick_string(self, newick_content):
        tree = extract_tree_from_string_or_file(newick_content)
        assert tree is not None
        assert len(tree.get_terminals()) == 3

    def test_nexus_file(self, nexus_file):
        tree = extract_tree_from_string_or_file(nexus_file)
        assert tree is not None
        terminals = [t.name for t in tree.get_terminals()]
        assert set(terminals) == {"seq1", "seq2", "seq3"}

    def test_hyphy_annotations_stripped(self, newick_hyphy_annotations):
        tree = extract_tree_from_string_or_file(newick_hyphy_annotations)
        assert tree is not None
        terminals = [t.name for t in tree.get_terminals()]
        assert "seq1" in terminals
        assert "{Foreground}" not in terminals
        assert "seq1{Foreground}" not in terminals

    def test_no_tree_returns_none(self):
        tree = extract_tree_from_string_or_file("no tree here")
        assert tree is None

    def test_gzipped_newick(self, tmp_path, newick_content):
        p = tmp_path / "tree.nwk.gz"
        with gzip.open(p, "wt") as f:
            f.write(newick_content)
        tree = extract_tree_from_string_or_file(str(p))
        assert tree is not None
        assert len(tree.get_terminals()) == 3


class TestHasNonzeroBranchLengths:
    def test_with_branch_lengths(self, newick_file):
        tree = extract_tree_from_string_or_file(newick_file)
        assert has_nonzero_branch_lengths(tree) is True

    def test_without_branch_lengths(self, newick_no_branch_lengths):
        tree = extract_tree_from_string_or_file(newick_no_branch_lengths)
        assert has_nonzero_branch_lengths(tree) is False

    def test_partial_branch_lengths(self):
        tree = extract_tree_from_string_or_file("((seq1:0.1,seq2):0.05,seq3:0.15);")
        assert has_nonzero_branch_lengths(tree) is True


class TestEnforceNonzeroBranchLengths:
    def test_sets_missing_branch_lengths(self, newick_no_branch_lengths):
        tree = extract_tree_from_string_or_file(newick_no_branch_lengths)
        enforce_nonzero_branch_lengths(tree)
        for clade in tree.find_clades():
            if clade is not tree.root:
                assert clade.branch_length is not None
                assert clade.branch_length > 0

    def test_enforces_minimum(self):
        tree = extract_tree_from_string_or_file("((seq1:0.0,seq2:0.0):0.0,seq3:0.0);")
        enforce_nonzero_branch_lengths(tree, min_len=1e-4)
        for clade in tree.find_clades():
            if clade is not tree.root:
                assert clade.branch_length >= 1e-4

    def test_preserves_valid_branch_lengths(self, newick_file):
        tree = extract_tree_from_string_or_file(newick_file)
        original = {c.name: c.branch_length for c in tree.find_clades() if c is not tree.root}
        enforce_nonzero_branch_lengths(tree)
        for c in tree.find_clades():
            if c is not tree.root and c.name in original and original[c.name] >= 1e-4:
                assert c.branch_length == original[c.name]
