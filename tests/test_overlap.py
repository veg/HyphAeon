"""Tests for overlapping reading frame detection (hyphaeon/overlap.py)."""
from collections import OrderedDict

from hyphaeon import overlap


def test_revcomp_alignment():
    rc = overlap.revcomp_alignment(OrderedDict(a="ATG---AACN"))
    assert rc["a"] == "NGTT---CAT"


def test_genetic_code_stops():
    assert overlap.GENETIC_CODE["TAA"] == "*"
    assert overlap.GENETIC_CODE["ATG"] == "M"
    assert len(overlap.GENETIC_CODE) == 64


def test_passes_signature_thresholds():
    # CDKN2A/p14ARF-like: real double constraint on all axes
    assert overlap.passes_signature({"n_overlap_subs": 1200, "syn_in_B": 0.66,
                                     "pos3_ratio": 0.41, "density_ratio": 0.38,
                                     "synB_at_nonsynA": 0.79, "nonsynA_subs": 80})
    # passenger: frame B free
    assert not overlap.passes_signature({"n_overlap_subs": 1000, "syn_in_B": 0.05,
                                         "pos3_ratio": 1.1, "density_ratio": 0.9,
                                         "synB_at_nonsynA": 0.1, "nonsynA_subs": 50})
    # KRAS-like inflation: high syn_in_B but frame B not constrained at nonsynA sites
    assert not overlap.passes_signature({"n_overlap_subs": 60, "syn_in_B": 0.95,
                                         "pos3_ratio": 0.06, "density_ratio": 0.46,
                                         "synB_at_nonsynA": 0.2, "nonsynA_subs": 55})


def test_detect_synthetic_overlap(tmp_path):
    # frame 0 conserved protein; construct so frame +1 is a stop-free, constrained ORF.
    import random
    random.seed(0)
    codons = ["ATG"] + [random.choice(["GCT", "GGT", "CTG", "ACC", "GTT", "AAG"])
                        for _ in range(40)] + ["TGG"]
    base = "".join(codons)
    seqs = {"ref": base}
    for i in range(1, 12):
        s = list(base)
        # a few synonymous-ish tweaks at 3rd positions
        for p in range(2, len(s), 7):
            s[p] = random.choice("ACGT")
        seqs[f"sp{i}"] = "".join(s)
    fa = tmp_path / "toy.fa"
    fa.write_text("".join(f">{n}\n{seqs[n]}\n" for n in seqs))
    hits, ref = overlap.detect(str(fa), reference="ref", min_codons=10, min_subs=10)
    assert ref == "ref"
    assert isinstance(hits, list)  # runs end-to-end and returns ranked hits
