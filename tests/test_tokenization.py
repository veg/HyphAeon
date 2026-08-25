"""Unit tests for tokenization functions in hyphaeon.dataset."""
import pytest
from hyphaeon.dataset import get_codon_token, get_aa_token, GENETIC_CODE, AA_MAP


class TestGetCodonToken:
    def test_standard_codons(self):
        assert get_codon_token("TTT") == 0
        assert get_codon_token("ATG") == 32
        assert get_codon_token("GGG") == 60

    def test_case_insensitive(self):
        assert get_codon_token("ttt") == 0
        assert get_codon_token("AtG") == 32
        assert get_codon_token("gGg") == 60

    def test_stop_codons(self):
        assert get_codon_token("TAA") == 64
        assert get_codon_token("TAG") == 64
        assert get_codon_token("TGA") == 64

    def test_unknown_codon_returns_gap_token(self):
        assert get_codon_token("NNN") == 64
        assert get_codon_token("---") == 64
        assert get_codon_token("XYZ") == 64
        assert get_codon_token("AT") == 64

    def test_all_standard_codons_covered(self):
        for codon, token in GENETIC_CODE.items():
            assert get_codon_token(codon) == token


class TestGetAAToken:
    def test_standard_codons(self):
        assert get_aa_token("TTT") == AA_MAP["F"]
        assert get_aa_token("ATG") == AA_MAP["M"]
        assert get_aa_token("GGG") == AA_MAP["G"]

    def test_case_insensitive(self):
        assert get_aa_token("ttt") == AA_MAP["F"]
        assert get_aa_token("atg") == AA_MAP["M"]

    def test_stop_codons_return_gap_aa(self):
        assert get_aa_token("TAA") == 20
        assert get_aa_token("TAG") == 20
        assert get_aa_token("TGA") == 20

    def test_unknown_codon_returns_gap_aa(self):
        assert get_aa_token("NNN") == 20
        assert get_aa_token("---") == 20
        assert get_aa_token("XYZ") == 20

    def test_all_aa_tokens_in_range(self):
        for codon in GENETIC_CODE:
            token = get_aa_token(codon)
            assert 0 <= token <= 20
