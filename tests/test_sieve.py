#!/usr/bin/env python3
import sys
from pathlib import Path
import tempfile
import numpy as np

REPO_ROOT = Path('/Users/sergei/Projects/TOGA_MEME/axomeme_repo')
sys.path.insert(0, str(REPO_ROOT))
from hyphaeon.sieve import ChronAeonSieve

def test_chronaeon_sieve_quality_and_missing_data():
    """Verify that Gate 1 properly flags sequences with >5% missing data (Ns, gaps, IUPAC codes)."""
    ref_seq = "ACGT" * 500  # 2000 bp
    taxa = [f"anc_{i}" for i in range(15)]
    aln_dict = {"ref_2020": ref_seq}
    dates_dict = {"ref_2020": 2020.0}
    for i in range(15):
        aln_dict[f"anc_{i}"] = "ACGT" * (500 - i) + "TCGT" * i
        dates_dict[f"anc_{i}"] = 2020.0 + i * 0.2

    with tempfile.TemporaryDirectory() as td:
        fa_path = Path(td) / "aln.fasta"
        meta_path = Path(td) / "meta.csv"

        with open(fa_path, "w") as f:
            for k, v in aln_dict.items():
                f.write(f">{k}\n{v}\n")

        with open(meta_path, "w") as f:
            f.write("genome_id,collection_date\n")
            for k, v in dates_dict.items():
                f.write(f"{k},{v}\n")

        sieve = ChronAeonSieve.build_from_alignment(
            alignment_path=fa_path,
            dates_path=meta_path,
            root_taxon="ref_2020",
            max_ambig_ratio=0.05
        )

        # 1. Clean identical sequence should PASS
        res = sieve.screen_sequence("clean_test", ref_seq, 2020.0)
        assert res['status'] == 'PASS', f"Expected PASS, got {res['status']}"

        # 2. Sequence with 10% Ns should fail at Gate 1
        seq_10pct_n = "N" * 200 + ref_seq[200:]
        res_n = sieve.screen_sequence("high_n_test", seq_10pct_n, 2020.0)
        assert res_n['status'] == 'SUS'
        assert "SUS_LOW_QUALITY" in res_n['sus_reason']
        assert "missing/degenerate data: 10.0%" in res_n['sus_reason']

        # 3. Sequence with 10% gaps '-' (e.g. unaligned/soft-clipped) should fail at Gate 1
        seq_10pct_gap = "-" * 200 + ref_seq[200:]
        res_gap = sieve.screen_sequence("high_gap_test", seq_10pct_gap, 2020.0)
        assert res_gap['status'] == 'SUS'
        assert "SUS_LOW_QUALITY" in res_gap['sus_reason']

        # 4. Sequence with 10% degenerate IUPAC characters should fail at Gate 1
        seq_10pct_iupac = "R" * 200 + ref_seq[200:]
        res_iupac = sieve.screen_sequence("high_iupac_test", seq_10pct_iupac, 2020.0)
        assert res_iupac['status'] == 'SUS'
        assert "SUS_LOW_QUALITY" in res_iupac['sus_reason']

        print("All Gate 1 quality & missing data tests passed successfully!")

if __name__ == '__main__':
    test_chronaeon_sieve_quality_and_missing_data()
