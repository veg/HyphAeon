#!/usr/bin/env python3
"""
scripts/run_sars2_streaming_sieve.py
------------------------------------
Builds a realistic streaming public health surveillance challenge suite for SARS-CoV-2
and benchmarks ChronAeon Sieve on:
1. Valid contemporary submissions (PASS)
2. Mislabeled collection dates (SUS_DATE_MISMATCH)
3. Archival / lab escape revertants (SUS_ARCHIVAL_OR_LAB_LEAK)
4. Degraded / excessive missing data (SUS_LOW_QUALITY)
5. Non-target contaminants (SUS_NON_TARGET_CONTAMINANT)
6. Chimeric recombinants (SUS_CHIMERIC_RECOMBINANT)
7. Biological hypermutators (SUS_HYPERMUTATED)

Outputs:
- Diagnostic triage CSV
- Segregated FASTAs (clean vs quarantined)
- Latency and throughput benchmarks
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
from Bio import SeqIO
from typing import Dict, List, Tuple
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hyphaeon.sieve import ChronAeonSieve


BENCHMARK_DIR = REPO_ROOT / "benchmarks" / "sars2_chronaeon_sieve"
ANCHOR_FA = BENCHMARK_DIR / "sars2_anchor_aligned.fasta"
ANCHOR_META = BENCHMARK_DIR / "sars2_anchor_metadata.csv"
ROOT_TAXON = "2697049.107626|2019-12-26"


def generate_challenge_suite(anchor_taxa: List[str], seq_dict: Dict[str, str], root_seq: str) -> Tuple[str, List[Dict]]:
    rng = random.Random(42)
    challenge_records = []
    metadata = []
    L = len(root_seq)

    # Separate taxa by approximate year
    taxa_2020 = [t for t in anchor_taxa if "2020" in t]
    taxa_2021 = [t for t in anchor_taxa if "2021" in t]
    taxa_2026 = [t for t in anchor_taxa if "2026" in t]

    # 1. Valid Contemporary Isolates (PASS) - 15 clean 2026 genomes
    for i in range(min(15, len(taxa_2026))):
        t = taxa_2026[i]
        qid = f"SURVEILLANCE_2026_GENUINE_{i+1:02d}|2026-05-15"
        challenge_records.append((qid, seq_dict[t]))
        metadata.append({"query_id": qid, "expected_category": "PASS", "scenario": "Clean 2026 isolate"})

    # 2. Mislabeled Collection Date (SUS_DATE_MISMATCH) - 5 early 2020 genomes stamped as 2026
    for i in range(min(5, len(taxa_2020))):
        t = taxa_2020[i]
        qid = f"ERROR_MISLABELED_DATE_{i+1:02d}|2026-06-01"
        challenge_records.append((qid, seq_dict[t]))
        metadata.append({"query_id": qid, "expected_category": "SUS_DATE_MISMATCH", "scenario": "2020 sequence labeled 2026"})

    # 3. Archival / Lab Escape Revertant (SUS_ARCHIVAL_OR_LAB_LEAK) - Ancestral Wuhan-Hu-1 stamped as 2026
    for i in range(3):
        # Wuhan-Hu-1 with 0-1 private mutations
        seq_mut = list(root_seq)
        if i > 0:
            seq_mut[rng.randint(100, L-100)] = rng.choice(['A', 'C', 'G', 'T'])
        qid = f"SUS_ARCHIVAL_LEAK_{i+1:02d}|2026-06-10"
        challenge_records.append((qid, "".join(seq_mut)))
        metadata.append({"query_id": qid, "expected_category": "SUS_ARCHIVAL_OR_LAB_LEAK", "scenario": "Ancestral root sequence labeled 2026"})

    # 4. Excessive Missing Data (SUS_LOW_QUALITY) - 4 genomes with 2,500 Ns (8.3% missing)
    for i in range(4):
        base_t = taxa_2026[i % len(taxa_2026)]
        seq_chars = list(seq_dict[base_t])
        # Inject 2500 Ns across 5 blocks
        for _ in range(5):
            start = rng.randint(500, L - 1000)
            for pos in range(start, start + 500):
                seq_chars[pos] = 'N'
        qid = f"DEGRADED_ASSEMBLY_{i+1:02d}|2026-05-20"
        challenge_records.append((qid, "".join(seq_chars)))
        metadata.append({"query_id": qid, "expected_category": "SUS_LOW_QUALITY", "scenario": "High missing data (2500 Ns)"})

    # 5. Non-Target Foreign Contaminants (SUS_NON_TARGET_CONTAMINANT) - 3 foreign sequences
    for i in range(3):
        # Synthetic divergent sequence with 15% divergence from root
        div_seq = list(root_seq)
        for pos in range(0, L, 6):
            div_seq[pos] = rng.choice(['A', 'C', 'G', 'T'])
        qid = f"FOREIGN_CONTAMINANT_{i+1:02d}|2026-05-01"
        challenge_records.append((qid, "".join(div_seq)))
        metadata.append({"query_id": qid, "expected_category": "SUS_NON_TARGET_CONTAMINANT", "scenario": "Foreign / highly divergent genome"})

    # 6. Chimeric Recombinant (SUS_CHIMERIC_RECOMBINANT) - 3 Delta-Omicron mosaics
    for i in range(min(3, len(taxa_2021), len(taxa_2026))):
        seq_5p = seq_dict[taxa_2021[i]][:L//2]
        seq_3p = seq_dict[taxa_2026[i]][L//2:]
        mosaic = seq_5p + seq_3p
        qid = f"CHIMERIC_RECOMBINANT_{i+1:02d}|2026-05-25"
        challenge_records.append((qid, mosaic))
        metadata.append({"query_id": qid, "expected_category": "SUS_CHIMERIC_RECOMBINANT", "scenario": "5' 2021 Delta + 3' 2026 Contemporary recombinant"})

    # 7. Biological Hypermutator (SUS_HYPERMUTATED) - 3 clean sequences with 50 private substitutions
    for i in range(min(3, len(taxa_2026))):
        seq_hyp = list(seq_dict[taxa_2026[i]])
        # Add 50 private mutations in clean ACGT
        for _ in range(50):
            pos = rng.randint(100, L - 100)
            orig = seq_hyp[pos]
            alts = [b for b in "ACGT" if b != orig]
            seq_hyp[pos] = rng.choice(alts)
        qid = f"BIOLOGICAL_HYPERMUTATOR_{i+1:02d}|2026-05-30"
        challenge_records.append((qid, "".join(seq_hyp)))
        metadata.append({"query_id": qid, "expected_category": "SUS_HYPERMUTATED", "scenario": "Clean sequence with 50 private mutations"})

    # Write challenge FASTA
    challenge_fa = BENCHMARK_DIR / "sars2_streaming_challenge.fasta"
    with open(challenge_fa, "w") as f:
        for qid, s in challenge_records:
            f.write(f">{qid}\n{s}\n")

    return str(challenge_fa), metadata


def run_benchmark():
    print("=" * 80)
    print("CHRONAEON SIEVE: SARS-CoV-2 REAL-WORLD SURVEILLANCE BENCHMARK")
    print("=" * 80)

    # 1. Build & Calibrate Sieve
    print("\n[1/3] Initializing ChronAeon Sieve Anchor Skeleton...")
    t_start = time.time()
    sieve = ChronAeonSieve.build_from_alignment(
        alignment_path=ANCHOR_FA,
        dates_path=ANCHOR_META,
        root_taxon=ROOT_TAXON,
        n_anchor=128,
        n_temporal_bins=32,
        tolerance_days=90.0,
        z_threshold=2.5,
        max_ambig_ratio=0.05
    )
    t_calib = time.time() - t_start
    print(f"[✓] Anchor calibrated in {t_calib:.3f}s:")
    print(f"    - Ancestral Root: {sieve.anchor_taxa[0]}")
    print(f"    - Rate μ:        {sieve.mu:.6f} subs/site/yr ({sieve.mu*sieve.seq_len:.2f} subs/yr)")
    print(f"    - t_MRCA:        {sieve.t_mrca:.4f} (Fieller g = {sieve.clock_params['fieller_g']:.5f})")

    # 2. Build Challenge Suite
    print("\n[2/3] Generating Multidimensional Surveillance Challenge Suite...")
    seq_dict = {rec.id: str(rec.seq).upper() for rec in SeqIO.parse(str(ANCHOR_FA), "fasta")}
    root_seq = seq_dict[ROOT_TAXON]
    challenge_fa, meta_records = generate_challenge_suite(list(seq_dict.keys()), seq_dict, root_seq)
    print(f"[✓] Challenge suite generated: {len(meta_records)} sequences across 7 challenge scenarios.")

    # 3. Stream & Triage Sequences
    print("\n[3/3] Streaming sequences through ChronAeon Sieve...")
    clean_fa = BENCHMARK_DIR / "sars2_clean_surveillance.fasta"
    sus_fa = BENCHMARK_DIR / "sars2_quarantined_sus.fasta"
    report_csv = BENCHMARK_DIR / "sars2_sieve_triage_report.csv"

    t0_stream = time.time()
    df_results = sieve.screen_stream(
        stream_fasta=challenge_fa,
        clean_fasta_out=str(clean_fa),
        sus_fasta_out=str(sus_fa)
    )
    t_stream = time.time() - t0_stream

    # Join metadata
    df_meta = pd.DataFrame(meta_records)
    df_merged = df_results.merge(df_meta, on="query_id", how="left")
    df_merged.to_csv(report_csv, index=False)

    n_tot = len(df_results)
    n_pass = int((df_results['status'] == 'PASS').sum())
    n_sus = int((df_results['status'] == 'SUS').sum())
    ms_per_seq = (t_stream / n_tot) * 1000.0
    throughput_hr = (n_tot / t_stream) * 3600.0

    print("\n" + "=" * 80)
    print("BENCHMARK TRIAGE SUMMARY")
    print("=" * 80)
    print(f"Total Sequences Screened:  {n_tot}")
    print(f"Passed Clean Filter:       {n_pass} ({n_pass/n_tot*100:.1f}%)")
    print(f"Quarantined Anomalies:     {n_sus} ({n_sus/n_tot*100:.1f}%)")
    print(f"Streaming Wallclock Time:  {t_stream:.3f} s")
    print(f"Latency per Genome:        {ms_per_seq:.2f} ms")
    print(f"Streaming Throughput:      {throughput_hr:,.0f} complete genomes / hour")
    print("=" * 80)

    print("\nDetailed Scenario Breakdown:")
    for sc, grp in df_merged.groupby("scenario"):
        reasons = grp['sus_reason'].tolist()
        print(f"\n* Scenario: {sc} (N={len(grp)})")
        print(f"  Expected: {grp['expected_category'].iloc[0]}")
        print(f"  Status:   PASS={sum(grp['status']=='PASS')}, SUS={sum(grp['status']=='SUS')}")
        for idx, row in grp.iterrows():
            print(f"    - {row['query_id'][:35]:<36} -> [{row['status']}] {row['sus_reason'][:55]}")

    print(f"\n[✓] Diagnostic triage report saved to: {report_csv}")
    print(f"[✓] Clean FASTA saved to: {clean_fa}")
    print(f"[✓] Quarantined FASTA saved to: {sus_fa}")


if __name__ == "__main__":
    run_benchmark()
