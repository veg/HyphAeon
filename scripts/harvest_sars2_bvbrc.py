#!/usr/bin/env python3
"""
scripts/harvest_sars2_bvbrc.py
------------------------------
Harvests a temporally stratified anchor set and contemporary query set of complete
SARS-CoV-2 genomes from BV-BRC, and projects them to Wuhan-Hu-1 coordinates via minimap2.
"""

import os
import sys
import time
import json
import subprocess
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


OUTPUT_DIR = Path("/Users/sergei/Projects/TOGA_MEME/axomeme_repo/benchmarks/sars2_chronaeon_sieve")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_URL = "https://www.bv-brc.org/api"


def fetch_bvbrc_json(url: str, timeout: int = 60) -> List[Dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def harvest_sars2_cohort():
    print("=" * 80)
    print("HARVESTING SARS-CoV-2 GENOMES FROM BV-BRC (2019-2026)")
    print("=" * 80)

    # 1. Fetch Wuhan-Hu-1 as reference root
    print("[1/4] Fetching ancestral root: Wuhan-Hu-1 (2697049.107626)...")
    root_gid = "2697049.107626"
    root_seq_url = f"{BASE_URL}/genome_sequence/?eq(genome_id,{root_gid})&select(genome_id,sequence)&limit(1)"
    root_data = fetch_bvbrc_json(root_seq_url)
    root_seq = root_data[0]["sequence"].upper()
    print(f"  [✓] Wuhan-Hu-1 sequence loaded: {len(root_seq)} bp.")

    # Save Wuhan-Hu-1 reference FASTA
    ref_fasta = OUTPUT_DIR / "wuhan_hu_1.fasta"
    with open(ref_fasta, "w") as f:
        f.write(f">Wuhan-Hu-1|NC_045512.2|2019-12-26\n{root_seq}\n")

    # 2. Sample anchor metadata across 2020-2026 (35 per year)
    years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    anchor_metadata = [{
        "genome_id": root_gid,
        "genome_name": "Severe acute respiratory syndrome coronavirus 2 Wuhan-Hu-1",
        "collection_date": "2019-12-26",
        "collection_year": 2019,
        "genome_length": len(root_seq),
        "is_root": True
    }]

    print("\n[2/4] Sampling stratified anchor metadata across 2020-2026...")
    for yr in years:
        # Sample 35 complete genomes with length >= 29400 bp
        query = (
            f"and(eq(taxon_lineage_ids,2697049),"
            f"eq(genome_quality,Good),"
            f"eq(collection_year,{yr}),"
            f"gt(genome_length,29400))"
            f"&select(genome_id,genome_name,collection_date,collection_year,genome_length)"
            f"&limit(35)"
        )
        url = f"{BASE_URL}/genome/?{query}"
        try:
            records = fetch_bvbrc_json(url)
            for r in records:
                r["is_root"] = False
                anchor_metadata.append(r)
            print(f"  [✓] {yr}: Sampled {len(records)} complete genomes.")
        except Exception as e:
            print(f"  [!] Failed for year {yr}: {e}")

    df_meta = pd.DataFrame(anchor_metadata)
    print(f"\nTotal sampled genomes: {len(df_meta)}.")

    # 3. Batch download full sequences
    print("\n[3/4] Batch downloading full genomes from BV-BRC...")
    gids = [str(r["genome_id"]) for r in anchor_metadata if r["genome_id"] != root_gid]
    seq_dict = {root_gid: root_seq}

    # Fetch in chunks of 50
    chunk_size = 50
    for i in range(0, len(gids), chunk_size):
        chunk = gids[i : i + chunk_size]
        c_str = ",".join(chunk)
        url = f"{BASE_URL}/genome_sequence/?in(genome_id,({c_str}))&select(genome_id,sequence)&limit({len(chunk)*2})"
        try:
            res = fetch_bvbrc_json(url)
            for item in res:
                gid = str(item.get("genome_id"))
                s = item.get("sequence", "").upper()
                if gid and s:
                    seq_dict[gid] = s
            print(f"  Retrieved chunk {i//chunk_size + 1}/{(len(gids)-1)//chunk_size + 1} ({len(seq_dict)}/{len(anchor_metadata)} total).")
        except Exception as e:
            print(f"  [!] Chunk error: {e}")

    # Write unaligned fasta
    unaligned_fa = OUTPUT_DIR / "sars2_anchor_unaligned.fasta"
    with open(unaligned_fa, "w") as f:
        for r in anchor_metadata:
            gid = str(r["genome_id"])
            if gid in seq_dict:
                hdr = f"{gid}|{r.get('collection_date', '2020-01-01')}"
                f.write(f">{hdr}\n{seq_dict[gid]}\n")

    print(f"  [✓] Saved unaligned FASTA with {len(seq_dict)} genomes.")

    # 4. Align to Wuhan-Hu-1 reference coordinates via minimap2
    print("\n[4/4] Aligning anchor sequences to Wuhan-Hu-1 coordinates via minimap2...")
    aligned_fa = OUTPUT_DIR / "sars2_anchor_aligned.fasta"
    paf_path = OUTPUT_DIR / "sars2_anchor.paf"

    cmd_mm2 = ["minimap2", "-c", "--eqx", "-x", "asm5", str(ref_fasta), str(unaligned_fa)]
    with open(paf_path, "w") as out_paf:
        subprocess.run(cmd_mm2, stdout=out_paf, stderr=subprocess.DEVNULL, check=True)

    # Project coordinates using PAF CIGAR
    print("  Projecting PAF alignments onto reference coordinates...")
    project_paf_to_reference(str(paf_path), str(unaligned_fa), root_seq, str(aligned_fa))

    # Save final metadata
    df_meta.to_csv(OUTPUT_DIR / "sars2_anchor_metadata.csv", index=False)
    print(f"\n[✓] Finished! Aligned anchor set saved to {aligned_fa}")


def project_paf_to_reference(paf_path: str, query_fa: str, ref_seq: str, out_fa: str):
    from Bio import SeqIO
    import re

    queries = {rec.id: str(rec.seq).upper() for rec in SeqIO.parse(query_fa, "fasta")}
    ref_len = len(ref_seq)

    aligned_bufs = {qid: ["-"] * ref_len for qid in queries}

    with open(paf_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 12:
                continue
            qid = parts[0]
            if qid not in queries:
                continue
            q_start = int(parts[2])
            strand = parts[4]
            r_start = int(parts[7])
            cigar = None
            for tag in parts[12:]:
                if tag.startswith("cg:Z:"):
                    cigar = tag[5:]
                    break
            if not cigar:
                continue

            qseq = queries[qid]
            if strand == "-":
                rc_map = str.maketrans("ACGTNacgtn", "TGCANtgcan")
                qseq = qseq.translate(rc_map)[::-1]

            ops = re.findall(r"(\d+)([MIDNSHP=X])", cigar)
            q_pos = q_start
            r_pos = r_start
            for length_str, op in ops:
                length = int(length_str)
                if op in ("M", "=", "X"):
                    for _ in range(length):
                        if 0 <= r_pos < ref_len and 0 <= q_pos < len(qseq):
                            aligned_bufs[qid][r_pos] = qseq[q_pos]
                        r_pos += 1
                        q_pos += 1
                elif op in ("D", "N"):
                    r_pos += length
                elif op in ("I", "S"):
                    q_pos += length

    with open(out_fa, "w") as out:
        for qid, qseq in queries.items():
            if qid.startswith("2697049.107626") or "Wuhan-Hu-1" in qid:
                out.write(f">{qid}\n{ref_seq}\n")
            else:
                out.write(f">{qid}\n{''.join(aligned_bufs[qid])}\n")


if __name__ == "__main__":
    harvest_sars2_cohort()
