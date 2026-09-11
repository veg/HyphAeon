"""
Unit tests for HyphAeon Alignment-Free Sketch & Reference-Guided Codon-Aware Threader.
"""
import pytest
import numpy as np
from hyphaeon.sketch import (
    get_canonical_kmer,
    hash_kmer_64,
    CanonicalMinHashSketcher,
    AlignmentFreeBinner,
    AlignmentFreeCentrifuge,
)
from hyphaeon.alignment import ReferenceCodonAligner


def test_canonical_kmer_symmetry():
    # ACGT -> Reverse complement is ACGT (palindrome)
    assert get_canonical_kmer("ACGT") == "ACGT"
    # AAAA -> TTTT -> min is AAAA
    assert get_canonical_kmer("AAAA") == "AAAA"
    assert get_canonical_kmer("TTTT") == "AAAA"
    # 64-bit hash consistency
    h1 = hash_kmer_64("AAAA")
    h2 = hash_kmer_64("AAAA")
    assert h1 == h2
    assert isinstance(h1, int)


def test_minhash_sketcher():
    sketcher = CanonicalMinHashSketcher(k=7, sketch_size=32)
    seq = "ATGGAGAAAATAGTGCTTCTTTAGCGATCGATCGATCGATCGATCGATC"
    sketch = sketcher.sketch(seq)
    assert len(sketch) > 0
    assert len(sketch) <= 32
    assert np.all(sketch[:-1] <= sketch[1:])  # sorted


def test_centrifuge_binning_and_quarantine():
    ref_h1 = "ATGAAGGCAATACTAGTAGTTCTGCTATATACATTTGCAACCGCAAATGCAGACACATTATGTATAGGTTATCAT"
    ref_h3 = "ATGAAGACTATCATTGCTTTGAGCTACATTTTCTGTCTGGTTTTCGCTCAAAAACTTCCCGGAAATGACAACAGC"
    non_target_na = "ATGAATCCAAATCAGAAAATAATAACCATTGGATCAATCTGTCTGGTAGTCGGACTAATTAGCCTAATATTGCAA"

    seq_dict = {
        "h1_sample_1": ref_h1,
        "h1_sample_2": ref_h1[:60] + "AAA" + ref_h1[63:],
        "h3_sample_1": ref_h3,
        "contaminant_na": non_target_na,
    }

    centrifuge = AlignmentFreeCentrifuge(
        k=9,
        sketch_size=64,
        alignable_threshold=0.10,
        quarantine_threshold=0.015,
        quiet=True,
    )

    results = centrifuge.bin_dataset(
        seq_dict=seq_dict,
        references={"H1": ref_h1, "H3": ref_h3}
    )

    assert "H1" in results["bins"]
    assert "h1_sample_1" in results["bins"]["H1"]
    assert "h1_sample_2" in results["bins"]["H1"]
    assert "H3" in results["bins"]
    assert "h3_sample_1" in results["bins"]["H3"]
    # NA should be quarantined because J < 0.015
    assert "contaminant_na" in results["quarantined"]


def test_reference_codon_aligner():
    # Canonical reference: 15 codons (45 nt >= 30 bp)
    ref_cds = "ATGGAGAAAATAGTGCTTCTTCTTGCAATAGTCAGTCTTGTTAAAAGT"
    aligner = ReferenceCodonAligner(ref_seq=ref_cds, ref_name="test_ref")

    # Exact match query
    query_exact = ref_cds
    aligned, score, status = aligner.align_single(query_exact)
    assert aligned == ref_cds
    assert score > 0
    assert "ok" in status

    # Query with 1 codon deletion in the middle
    # Deleting codon 5 (CTT)
    query_del = ref_cds[:12] + ref_cds[15:]
    aligned_del, score_del, status_del = aligner.align_single(query_del)
    assert len(aligned_del) == len(ref_cds)  # Frame-locked length preserved
    assert aligned_del[12:15] == "---"      # Codon deletion padded with '---'

    # Batch alignment
    batch_res = aligner.align_batch({"q1": query_exact, "q2": query_del}, quiet=True)
    assert len(batch_res["aligned_seqs"]) == 2
    assert len(batch_res["aligned_seqs"]["q1"]) == len(ref_cds)
    assert len(batch_res["aligned_seqs"]["q2"]) == len(ref_cds)
