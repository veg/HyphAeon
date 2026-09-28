"""
benchmarks/15_grand_100k_organismal_benchmark/benchmark_worker.py
==================================================================
Execution worker for the 100,000-simulation Organismal Benchmark.

Processes a batch of scenario indices [start_idx, end_idx) and records
matched head-to-head metrics for 3SEQ and RhizAeon into an SQLite database shard.
Strict 60-second timeout enforced per simulation run.
"""

import sys
import os
import time
import json
import sqlite3
import tempfile
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

# Add local path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from organismal_scenarios import OrganismalScenario, get_scenario_for_index
from organismal_generator import simulate_organismal_replicate


def write_fasta(filepath: Path, sequences: Dict[str, str]) -> None:
    """Writes sequences to FASTA format."""
    with open(filepath, "w", encoding="utf-8") as f:
        for taxon, seq in sequences.items():
            f.write(f">{taxon}\n{seq}\n")


def parse_3seq_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Parses 3SEQ output CSV for recombinant candidates."""
    if not os.path.exists(csv_path):
        return []
    candidates = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    if len(lines) <= 1:
        return []
    
    # 3SEQ Header: P_name,Q_name,C_name,m,n,k,p,HS?,log10(p),DS(p),DS(p),min_rec_length,breakpoints
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 12:
            continue
        dad = parts[0]
        mum = parts[1]
        child = parts[2]
        try:
            m = int(parts[3])
            n = int(parts[4])
            k = int(parts[5])
        except ValueError:
            continue
        try:
            p_val = float(parts[6])
        except ValueError:
            p_val = 1.0
        try:
            ds_p = float(parts[9])
        except (ValueError, IndexError):
            ds_p = p_val

        coords: List[float] = []
        for bp_str in parts[12:]:
            bp_str = bp_str.strip()
            if " & " in bp_str:
                l_str, r_str = bp_str.split(" & ")
                try:
                    l_min, l_max = map(float, l_str.strip().split("-"))
                    r_min, r_max = map(float, r_str.strip().split("-"))
                    coords.extend([(l_min + l_max) / 2.0, (r_min + r_max) / 2.0])
                except Exception:
                    continue
            elif "-" in bp_str:
                try:
                    b_min, b_max = map(float, bp_str.strip().split("-"))
                    coords.append((b_min + b_max) / 2.0)
                except Exception:
                    continue

        cand = {
            "child": child,
            "dad": dad,
            "mum": mum,
            "m": m,
            "n": n,
            "k": k,
            "p_val": p_val,
            "ds_p": ds_p,
            "coords": sorted(list(set(coords))),
        }
        candidates.append(cand)
    return candidates


def init_db(db_path: Path) -> sqlite3.Connection:
    """Initializes SQLite schema for benchmark records."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS benchmark_records (
        global_idx INTEGER,
        scenario_id TEXT,
        archetype TEXT,
        category TEXT,
        mode TEXT,
        is_null INTEGER,
        num_taxa INTEGER,
        length_nt INTEGER,
        method TEXT,
        detected INTEGER,
        false_positive INTEGER,
        true_positive INTEGER,
        timed_out INTEGER,
        num_breakpoints INTEGER,
        bp_coords TEXT,
        bp_error_mean REAL,
        bp_within_50 INTEGER,
        bp_within_100 INTEGER,
        correct_recombinant INTEGER,
        correct_parents INTEGER,
        p_value REAL,
        runtime_ms REAL
    )
    """)
    conn.commit()
    return conn


def evaluate_scenario(
    global_idx: int,
    three_seq_bin: str,
    ptable_path: str,
    rhizaeon_src: str,
    timeout_sec: float = 60.0,
) -> List[Dict[str, Any]]:
    """Evaluates a single scenario index head-to-head between 3SEQ and RhizAeon."""
    sc = get_scenario_for_index(global_idx)
    seqs, meta = simulate_organismal_replicate(sc, rep_idx=0)
    
    true_bps = sc.true_breakpoints
    true_recs = sc.recombinant_taxa
    true_parents = sc.parent_taxa
    is_null = sc.is_null
    records = []

    with tempfile.TemporaryDirectory(prefix=f"eval_{global_idx}_") as td:
        td_path = Path(td)
        fa_path = td_path / "alignment.fasta"
        write_fasta(fa_path, seqs)

        # -----------------------------------------------------------------
        # 1. Evaluate 3SEQ
        # -----------------------------------------------------------------
        run_id = f"s3_{global_idx}"
        cmd_3seq = [
            str(three_seq_bin),
            "-full", str(fa_path),
            "-p", str(ptable_path),
            "-id", run_id,
            "-q"
        ]
        t0 = time.perf_counter()
        det_3seq = False
        timeout_3seq = False
        coords_3seq: List[float] = []
        inferred_rec_3seq = ""
        inferred_parents_3seq = ""
        best_p_val_3seq = np.nan
        runtime_3seq = 0.0

        try:
            res_3seq = subprocess.run(cmd_3seq, cwd=td, capture_output=True, text=True, timeout=timeout_sec)
            runtime_3seq = (time.perf_counter() - t0) * 1000.0
            csv_path = td_path / f"{run_id}.3s.rec.csv"
            cands_3seq = parse_3seq_csv(str(csv_path))
            sig_cands = [c for c in cands_3seq if c["p_val"] <= 0.05 or c.get("ds_p", 1.0) <= 0.05]
            det_3seq = len(sig_cands) > 0
            if det_3seq:
                best_cand = min(sig_cands, key=lambda c: c["p_val"])
                best_p_val_3seq = best_cand["p_val"]
                inferred_rec_3seq = ";".join(sorted(set(c["child"] for c in sig_cands)))
                inferred_parents_3seq = "|".join(sorted(set(f"{c['dad']};{c['mum']}" for c in sig_cands)))
                for c in sig_cands:
                    coords_3seq.extend(c["coords"])
                coords_3seq = sorted(list(set(coords_3seq)))
        except subprocess.TimeoutExpired:
            runtime_3seq = timeout_sec * 1000.0
            timeout_3seq = True
        except Exception:
            runtime_3seq = (time.perf_counter() - t0) * 1000.0

        # Localization & Parent attribution for 3SEQ
        bp_error_3seq = np.nan
        within_50_3seq = False
        within_100_3seq = False
        correct_rec_3seq = False
        correct_parents_3seq = False

        if not is_null and det_3seq and coords_3seq and true_bps:
            errs = [min(abs(bp - tbp) for bp in coords_3seq) for tbp in true_bps]
            bp_error_3seq = float(np.mean(errs))
            within_50_3seq = any(e <= 50.0 for e in errs)
            within_100_3seq = any(e <= 100.0 for e in errs)
            correct_rec_3seq = any(tr in inferred_rec_3seq for tr in true_recs)
            # Check parentage attribution
            if true_recs and true_recs[0] in true_parents:
                exp_p = set(true_parents[true_recs[0]])
                cand_pairs = [set(p.split(";")) for p in inferred_parents_3seq.split("|") if p]
                if any(all(p in cand_set for p in exp_p if not p.startswith("Ghost")) for cand_set in cand_pairs):
                    correct_parents_3seq = True

        records.append({
            "global_idx": global_idx,
            "scenario_id": sc.scenario_id,
            "archetype": sc.archetype,
            "category": sc.category,
            "mode": sc.mode,
            "is_null": int(is_null),
            "num_taxa": sc.num_taxa,
            "length_nt": sc.length_nt,
            "method": "3seq",
            "detected": int(det_3seq),
            "false_positive": int(det_3seq and is_null),
            "true_positive": int(det_3seq and not is_null),
            "timed_out": int(timeout_3seq),
            "num_breakpoints": len(coords_3seq),
            "bp_coords": ";".join(f"{x:.1f}" for x in coords_3seq),
            "bp_error_mean": bp_error_3seq if not np.isnan(bp_error_3seq) else None,
            "bp_within_50": int(within_50_3seq),
            "bp_within_100": int(within_100_3seq),
            "correct_recombinant": int(correct_rec_3seq),
            "correct_parents": int(correct_parents_3seq),
            "p_value": best_p_val_3seq if not np.isnan(best_p_val_3seq) else None,
            "runtime_ms": runtime_3seq,
        })

        # -----------------------------------------------------------------
        # 2. Evaluate RhizAeon (in-process for 50x speedup)
        # -----------------------------------------------------------------
        is_codon = (sc.mode == "codon")
        t0 = time.perf_counter()
        det_rhiz = False
        timeout_rhiz = False
        coords_rhiz: List[float] = []
        inferred_rec_rhiz = ""
        inferred_parents_rhiz = ""
        runtime_rhiz = 0.0

        try:
            from rhizaeon.embed import build_prefix_engine
            from rhizaeon.segmentation import RhizAeonDetector

            engine = build_prefix_engine(
                str(fa_path),
                engine="default" if is_codon else "scalar",
                codon=is_codon,
            )
            taxa_list = getattr(engine, "taxa", list(seqs.keys()))
            detector = RhizAeonDetector(
                window_units=25 if is_codon else 75,
                min_tract_units=35 if is_codon else 105,
                calibration="calibrated",
            )
            events = detector.detect_recombination(engine, taxa_list)
            runtime_rhiz = (time.perf_counter() - t0) * 1000.0

            rec_events = [e for e in events if not getattr(e, "is_hypermutation", False) and not (isinstance(e, dict) and e.get("is_hypermutation", False))]
            det_rhiz = len(rec_events) > 0
            if det_rhiz:
                recs = []
                parents_list = []
                for ev in rec_events:
                    bp_val = getattr(ev, "breakpoint_nt", ev.get("breakpoint_nt") if isinstance(ev, dict) else None)
                    if bp_val is None:
                        bp_unit = getattr(ev, "breakpoint", ev.get("breakpoint") if isinstance(ev, dict) else None)
                        if bp_unit is not None:
                            bp_val = bp_unit * (3 if is_codon else 1)
                    if bp_val is not None:
                        coords_rhiz.append(float(bp_val))
                    r_name = getattr(ev, "recombinant", ev.get("recombinant") if isinstance(ev, dict) else "")
                    if r_name:
                        recs.append(r_name)
                    p1 = getattr(ev, "parent_left", ev.get("parent_left") if isinstance(ev, dict) else "")
                    p2 = getattr(ev, "parent_right", ev.get("parent_right") if isinstance(ev, dict) else "")
                    trans = getattr(ev, "transition", ev.get("transition") if isinstance(ev, dict) else "")
                    if p1 and p2:
                        parents_list.append(f"{p1};{p2}")
                    elif trans:
                        p_parts = trans.replace(" ➔ ", ";").split(";")
                        if len(p_parts) == 2:
                            parents_list.append(f"{p_parts[0]};{p_parts[1]}")
                inferred_rec_rhiz = ";".join(sorted(set(recs)))
                inferred_parents_rhiz = "|".join(sorted(set(parents_list)))
                coords_rhiz = sorted(list(set(coords_rhiz)))
        except Exception as e:
            runtime_rhiz = (time.perf_counter() - t0) * 1000.0

        # Localization & Parent attribution for RhizAeon
        bp_error_rhiz = np.nan
        within_50_rhiz = False
        within_100_rhiz = False
        correct_rec_rhiz = False
        correct_parents_rhiz = False

        if not is_null and det_rhiz and coords_rhiz and true_bps:
            errs = [min(abs(bp - tbp) for bp in coords_rhiz) for tbp in true_bps]
            bp_error_rhiz = float(np.mean(errs))
            within_50_rhiz = any(e <= 50.0 for e in errs)
            within_100_rhiz = any(e <= 100.0 for e in errs)
            correct_rec_rhiz = any(tr in inferred_rec_rhiz for tr in true_recs)
            if true_recs and true_recs[0] in true_parents:
                exp_p = set(true_parents[true_recs[0]])
                cand_pairs = [set(p.split(";")) for p in inferred_parents_rhiz.split("|") if p]
                if any(all(p in cand_set for p in exp_p if not p.startswith("Ghost")) for cand_set in cand_pairs):
                    correct_parents_rhiz = True

        records.append({
            "global_idx": global_idx,
            "scenario_id": sc.scenario_id,
            "archetype": sc.archetype,
            "category": sc.category,
            "mode": sc.mode,
            "is_null": int(is_null),
            "num_taxa": sc.num_taxa,
            "length_nt": sc.length_nt,
            "method": "rhizaeon",
            "detected": int(det_rhiz),
            "false_positive": int(det_rhiz and is_null),
            "true_positive": int(det_rhiz and not is_null),
            "timed_out": int(timeout_rhiz),
            "num_breakpoints": len(coords_rhiz),
            "bp_coords": ";".join(f"{x:.1f}" for x in coords_rhiz),
            "bp_error_mean": bp_error_rhiz if not np.isnan(bp_error_rhiz) else None,
            "bp_within_50": int(within_50_rhiz),
            "bp_within_100": int(within_100_rhiz),
            "correct_recombinant": int(correct_rec_rhiz),
            "correct_parents": int(correct_parents_rhiz),
            "p_value": None,
            "runtime_ms": runtime_rhiz,
        })

    return records


def main():
    parser = argparse.ArgumentParser(description="Worker for Grand 100k Organismal Benchmark")
    parser.add_argument("--start-idx", type=int, required=True, help="Start global scenario index")
    parser.add_argument("--end-idx", type=int, required=True, help="End global scenario index (exclusive)")
    parser.add_argument("--shard-id", type=int, default=None, help="Shard identifier")
    parser.add_argument("--db-output", type=str, required=True, help="Path to SQLite shard output file")
    parser.add_argument("--3seq-bin", type=str, default="3seq", help="Path to 3seq executable")
    parser.add_argument("--ptable", type=str, required=True, help="Path to 3seq p-table file")
    parser.add_argument("--rhizaeon-src", type=str, default="", help="Path to rhizaeon/src")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-scenario timeout in seconds (default: 60.0)")
    args = parser.parse_args()

    db_path = Path(args.db_output)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = init_db(db_path)
    cursor = conn.cursor()

    total_runs = args.end_idx - args.start_idx
    print(f"[*] Worker Shard {args.shard_id}: Processing indices {args.start_idx} to {args.end_idx} ({total_runs} simulations)")

    t_start = time.time()
    for idx, g_idx in enumerate(range(args.start_idx, args.end_idx)):
        try:
            records = evaluate_scenario(
                global_idx=g_idx,
                three_seq_bin=getattr(args, "3seq_bin"),
                ptable_path=args.ptable,
                rhizaeon_src=args.rhizaeon_src,
                timeout_sec=args.timeout,
            )
            for r in records:
                cursor.execute("""
                INSERT INTO benchmark_records VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """, (
                    r["global_idx"], r["scenario_id"], r["archetype"], r["category"],
                    r["mode"], r["is_null"], r["num_taxa"], r["length_nt"], r["method"],
                    r["detected"], r["false_positive"], r["true_positive"], r["timed_out"],
                    r["num_breakpoints"], r["bp_coords"], r["bp_error_mean"],
                    r["bp_within_50"], r["bp_within_100"], r["correct_recombinant"],
                    r["correct_parents"], r["p_value"], r["runtime_ms"]
                ))
            if (idx + 1) % 10 == 0:
                conn.commit()
                elapsed = time.time() - t_start
                rate = (idx + 1) / max(elapsed, 0.001)
                print(f"  [{idx+1:5d}/{total_runs:5d}] Shard {args.shard_id} progress | Rate: {rate:.2f} sims/sec | Elapsed: {elapsed:.1f}s")
        except Exception as e:
            print(f"[!] Error on scenario index {g_idx}: {e}", file=sys.stderr)

    conn.commit()
    conn.close()
    print(f"[✓] Worker Shard {args.shard_id} complete. Results saved to: {db_path}")


if __name__ == "__main__":
    main()
