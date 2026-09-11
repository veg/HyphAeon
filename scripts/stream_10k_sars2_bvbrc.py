#!/usr/bin/env python3
"""
scripts/stream_10k_sars2_bvbrc.py
---------------------------------
High-throughput streaming quality control and clock manifold triage of 10,000
real-world SARS-CoV-2 genomes from BV-BRC.

Features:
- Micro-batch streaming directly in RAM (zero multi-gigabyte disk buffering).
- Stratified sampling across 2020-2026 (or customizable via CLI).
- Dynamic coordinate alignment via multi-block minimap2.
- Real-time classification: PASS vs SUS (Date error, Archival leak, Low quality,
  Recombinant, Contaminant, Hypermutator).
- Real-time streaming progress and anomaly logging to CSV and quarantined FASTA.
"""

import os
import sys
import time
import json
import argparse
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hyphaeon.sieve import ChronAeonSieve
from hyphaeon.temporal import parse_date_to_decimal, extract_date_from_string

BASE_URL = "https://www.bv-brc.org/api"
BENCHMARK_DIR = REPO_ROOT / "benchmarks" / "sars2_chronaeon_sieve"
ANCHOR_FA = BENCHMARK_DIR / "sars2_anchor_aligned.fasta"
ANCHOR_META = BENCHMARK_DIR / "sars2_anchor_metadata.csv"
ROOT_TAXON = "2697049.107626|2019-12-26"

OUT_CSV = BENCHMARK_DIR / "sars2_10k_empirical_triage_report.csv"
OUT_SUS_FA = BENCHMARK_DIR / "sars2_10k_empirical_quarantined_sus.fasta"


def fetch_bvbrc_json(url: str, timeout: int = 45, retries: int = 3) -> List[Dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            if attempt == retries - 1:
                print(f"\n[!] Network warning on attempt {attempt+1}: {e}", flush=True)
                return []
            time.sleep(1.5 * (attempt + 1))
    return []


def stream_10k_sieve(target_total: int = 10000, batch_size: int = 100, specific_year: int = None):
    print("=" * 80, flush=True)
    print(f"CHRONAEON SIEVE: STREAMING {target_total:,} REAL-WORLD SARS-CoV-2 GENOMES (BV-BRC)", flush=True)
    print("=" * 80, flush=True)

    # 1. Initialize calibrated sieve anchor skeleton
    print("\n[1/3] Loading calibrated ChronAeon Sieve anchor manifold...", flush=True)
    t0_init = time.time()
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
    print(f"[✓] Calibrated anchor loaded in {time.time()-t0_init:.2f}s:", flush=True)
    print(f"    - Rate μ: {sieve.mu:.6f} subs/site/yr ({sieve.mu*sieve.seq_len:.2f} subs/yr)", flush=True)
    print(f"    - t_MRCA: {sieve.t_mrca:.4f} (Fieller g = {sieve.clock_params['fieller_g']:.5f})", flush=True)

    # 2. Plan temporal stratification
    if specific_year:
        years = [specific_year]
        per_year = target_total
        print(f"\n[2/3] Target Scope: {target_total:,} genomes strictly from year {specific_year}.", flush=True)
    else:
        years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
        per_year = int(np.ceil(target_total / len(years)))
        print(f"\n[2/3] Temporal Sampling Plan: ~{per_year:,} genomes/yr across {years[0]}–{years[-1]}.", flush=True)

    # Setup outputs
    headers_written = False
    if OUT_CSV.exists():
        OUT_CSV.unlink()
    if OUT_SUS_FA.exists():
        OUT_SUS_FA.unlink()

    total_screened = 0
    total_pass = 0
    total_sus = 0
    anomaly_counts = {}
    t_start_streaming = time.time()

    print(f"\n[3/3] Streaming Micro-batches from BV-BRC directly into ChronAeon Sieve in RAM...", flush=True)

    with open(OUT_SUS_FA, "a") as f_sus:
        for yr in years:
            year_harvested = 0
            offset = 0
            print(f"\n>>> Streaming Year {yr} (Target: up to {per_year:,} genomes)...", flush=True)

            while year_harvested < per_year and total_screened < target_total:
                current_k = min(batch_size, per_year - year_harvested, target_total - total_screened)
                if current_k <= 0:
                    break

                # Fetch metadata batch
                meta_query = (
                    f"and(eq(taxon_lineage_ids,2697049),"
                    f"eq(collection_year,{yr}),"
                    f"gt(genome_length,28000))"
                    f"&select(genome_id,genome_name,collection_date,collection_year,genome_length)"
                    f"&limit({current_k},{offset})"
                )
                meta_url = f"{BASE_URL}/genome/?{meta_query}"
                meta_records = fetch_bvbrc_json(meta_url)

                if not meta_records:
                    print(f"  [!] No more records found for year {yr} at offset {offset}.", flush=True)
                    break

                gids = [str(r["genome_id"]) for r in meta_records if "genome_id" in r]
                dates_dict = {}
                name_dict = {}
                for r in meta_records:
                    gid = str(r["genome_id"])
                    name_dict[gid] = r.get("genome_name", "")
                    d_raw = r.get("collection_date") or str(yr)
                    d_dec = parse_date_to_decimal(str(d_raw))
                    if d_dec is None or np.isnan(d_dec):
                        d_dec = extract_date_from_string(str(d_raw))
                    if d_dec is not None and not np.isnan(d_dec):
                        dates_dict[gid] = d_dec

                # Fetch sequences for these genomes
                c_str = ",".join(gids)
                seq_url = f"{BASE_URL}/genome_sequence/?in(genome_id,({c_str}))&select(genome_id,sequence)&limit({len(gids)*2})"
                seq_data = fetch_bvbrc_json(seq_url)

                batch_seqs = {}
                for item in seq_data:
                    gid = str(item.get("genome_id"))
                    seq = item.get("sequence", "")
                    if gid and seq and len(seq) > 20000:
                        batch_seqs[gid] = seq.upper()

                if not batch_seqs:
                    offset += current_k
                    continue

                # Dynamic alignment via multi-block minimap2
                t0_compute = time.time()
                aligned_map, pre_failed = sieve.align_queries_minimap2(batch_seqs)

                # Sieve screening
                batch_records = []
                for qid, sus_reason in pre_failed.items():
                    r_date = dates_dict.get(qid, np.nan)
                    rec = {
                        'query_id': qid,
                        'status': 'SUS',
                        'sus_reason': sus_reason,
                        'reported_date': r_date,
                        'predicted_date': np.nan,
                        'temporal_error_days': np.nan,
                        'temporal_error_years': np.nan,
                        'divergence_z': np.nan,
                        'root_divergence': np.nan,
                        'nearest_neighbor': None,
                        'nn_date': np.nan,
                        'nn_distance': np.nan,
                        'elapsed_ms': 0.1,
                        'genome_name': name_dict.get(qid, ''),
                        'collection_year': yr
                    }
                    batch_records.append(rec)
                    f_sus.write(f">{qid}|{sus_reason}\n{batch_seqs[qid]}\n")

                for qid, aln_seq in aligned_map.items():
                    r_date = dates_dict.get(qid, np.nan)
                    rec = sieve.screen_sequence(qid, aln_seq, r_date)
                    rec['genome_name'] = name_dict.get(qid, '')
                    rec['collection_year'] = yr
                    batch_records.append(rec)
                    if rec['status'] != 'PASS':
                        f_sus.write(f">{qid}|{rec['sus_reason']}\n{batch_seqs[qid]}\n")

                f_sus.flush()

                # Tally stats
                df_b = pd.DataFrame(batch_records)
                b_pass = int((df_b['status'] == 'PASS').sum())
                b_sus = int((df_b['status'] != 'PASS').sum())

                total_screened += len(df_b)
                total_pass += b_pass
                total_sus += b_sus
                year_harvested += len(df_b)
                offset += current_k

                # Append to CSV
                df_b.to_csv(OUT_CSV, mode='a', header=not headers_written, index=False)
                headers_written = True

                for _, r in df_b[df_b['status'] != 'PASS'].iterrows():
                    cat = r['sus_reason'].split()[0]
                    anomaly_counts[cat] = anomaly_counts.get(cat, 0) + 1

                elapsed_tot = time.time() - t_start_streaming
                rate_overall = total_screened / max(0.01, elapsed_tot)
                pct_clean = (total_pass / total_screened) * 100.0

                # Print progress update
                print(f"  [{total_screened:>5,}/{target_total:,}] (yr {yr}) "
                      f"Pass: {total_pass:>5} ({pct_clean:>5.1f}%) | "
                      f"SUS: {total_sus:>4} | "
                      f"Rate: {rate_overall:>5.1f} seq/s ({rate_overall*3600:>8,.0f}/hr)", flush=True)

                # If new anomalies in this batch, highlight up to 2 examples
                sus_in_batch = df_b[df_b['status'] != 'PASS']
                for _, srow in sus_in_batch.head(2).iterrows():
                    err_str = f"Δt={srow['temporal_error_days']:+.0f}d, Z={srow['divergence_z']:+.1f}" if not np.isnan(srow['temporal_error_days']) else ""
                    print(f"      ↳ [SUS] {srow['query_id']} -> {srow['sus_reason'][:48]} {err_str}", flush=True)

    elapsed_all = time.time() - t_start_streaming
    throughput_hr = (total_screened / max(0.01, elapsed_all)) * 3600.0

    print("\n" + "=" * 80, flush=True)
    print(f"EMPIRICAL REAL-WORLD GENOME TRIAGE COMPLETE ({elapsed_all:.1f}s)", flush=True)
    print("=" * 80, flush=True)
    print(f"Total Streamed & Screened: {total_screened:,}", flush=True)
    print(f"Clean (PASS Analysis-Ready): {total_pass:,} ({total_pass/max(1,total_screened)*100:.2f}%)", flush=True)
    print(f"Quarantined (SUS Anomalies):  {total_sus:,} ({total_sus/max(1,total_screened)*100:.2f}%)", flush=True)
    print(f"Effective Throughput:        {throughput_hr:,.0f} complete genomes / hour", flush=True)
    print(f"Mean Ingestion + Triage Latency: {(elapsed_all / max(1,total_screened))*1000.0:.2f} ms / genome", flush=True)
    print("-" * 80, flush=True)
    print("Empirical Anomaly Breakdown:", flush=True)
    for cat, cnt in sorted(anomaly_counts.items(), key=lambda x: -x[1]):
        print(f"  • {cat:<30}: {cnt:>5} ({cnt/max(1,total_sus)*100:>5.1f}% of anomalies)", flush=True)
    print("=" * 80, flush=True)
    print(f"\n[✓] Diagnostic triage CSV saved: {OUT_CSV}", flush=True)
    print(f"[✓] Quarantined anomalies FASTA: {OUT_SUS_FA}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream real-world SARS-CoV-2 genomes from BV-BRC through ChronAeon Sieve")
    parser.add_argument("--total", type=int, default=10000, help="Target total genomes to stream (default: 10000)")
    parser.add_argument("--batch-size", type=int, default=100, help="Micro-batch size per HTTP request (default: 100)")
    parser.add_argument("--year", type=int, default=None, help="Optional: restrict to a specific collection year (e.g. 2026)")
    args = parser.parse_args()

    stream_10k_sieve(target_total=args.total, batch_size=args.batch_size, specific_year=args.year)
