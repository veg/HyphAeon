"""
benchmarks/15_grand_100k_organismal_benchmark/merge_100k_shards.py
==================================================================
Merges SQLite shards from the 100,000-simulation Organismal Benchmark
into a master database and generates comprehensive summary statistics and tables.
"""

import sys
import os
import glob
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np


def merge_shards(shards_dir: Path, master_db: Path) -> pd.DataFrame:
    """Merges all shard_*.db files into master_db."""
    shard_files = sorted(glob.glob(str(shards_dir / "shard_*.db")))
    if not shard_files:
        raise FileNotFoundError(f"No shard databases found in {shards_dir}")

    print(f"[*] Found {len(shard_files)} shard files in {shards_dir}")
    if master_db.exists():
        master_db.unlink()

    conn_master = sqlite3.connect(str(master_db))
    cur_master = conn_master.cursor()

    cur_master.execute("""
    CREATE TABLE benchmark_records (
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

    total_rows = 0
    for sf in shard_files:
        conn_s = sqlite3.connect(sf)
        df_s = pd.read_sql_query("SELECT * FROM benchmark_records", conn_s)
        conn_s.close()
        df_s.to_sql("benchmark_records", conn_master, if_exists="append", index=False)
        total_rows += len(df_s)

    conn_master.commit()
    print(f"[✓] Merged {len(shard_files)} shards ({total_rows} records) into {master_db}")

    df_all = pd.read_sql_query("SELECT * FROM benchmark_records", conn_master)
    conn_master.close()
    return df_all


def compute_summary_tables(df: pd.DataFrame, output_dir: Path) -> None:
    """Computes global scorecard and breakdown tables across archetypes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Summary by Archetype and Method
    records = []
    for (arch, cat, method), sub in df.groupby(["archetype", "category", "method"]):
        n_sims = len(sub)
        null_sub = sub[sub["is_null"] == 1]
        rec_sub = sub[sub["is_null"] == 0]

        fpr = (null_sub["false_positive"].mean() * 100.0) if len(null_sub) > 0 else np.nan
        power = (rec_sub["true_positive"].mean() * 100.0) if len(rec_sub) > 0 else np.nan
        mae = rec_sub["bp_error_mean"].dropna().mean() if len(rec_sub) > 0 else np.nan
        w50 = (rec_sub["bp_within_50"].mean() * 100.0) if len(rec_sub) > 0 else np.nan
        rec_acc = (rec_sub["correct_recombinant"].mean() * 100.0) if len(rec_sub) > 0 else np.nan
        par_acc = (rec_sub["correct_parents"].mean() * 100.0) if len(rec_sub) > 0 else np.nan
        timeouts = (sub["timed_out"].mean() * 100.0)
        mean_runtime = sub["runtime_ms"].mean()

        records.append({
            "archetype": arch,
            "category": cat,
            "method": method,
            "n_sims": n_sims,
            "fpr_pct": fpr,
            "power_pct": power,
            "mae_nt": mae,
            "within_50_pct": w50,
            "rec_acc_pct": rec_acc,
            "par_acc_pct": par_acc,
            "timeout_pct": timeouts,
            "runtime_ms": mean_runtime,
        })

    df_summary = pd.DataFrame(records)
    csv_path = output_dir / "grand_100k_summary_by_archetype.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"[✓] Summary saved to: {csv_path}")

    # Generate LaTeX table
    tex_path = output_dir / "tab_grand_100k_summary.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated Grand 100,000 Organismal Recombination Benchmark Summary\n")
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{llccccccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Organismal Archetype} & \\textbf{Method} & \\textbf{Regime} & \\textbf{FPR (\\%)} & \\textbf{Power (\\%)} & \\textbf{MAE (nt)} & \\textbf{Rec ID (\\%)} & \\textbf{Parents (\\%)} & \\textbf{Time (ms)} \\\\\n")
        f.write("\\midrule\n")
        for arch in sorted(df_summary["archetype"].unique()):
            sub = df_summary[df_summary["archetype"] == arch]
            for _, r in sub.iterrows():
                fpr_s = f"{r['fpr_pct']:.1f}\\%" if not np.isnan(r['fpr_pct']) else "---"
                pwr_s = f"{r['power_pct']:.1f}\\%" if not np.isnan(r['power_pct']) else "---"
                mae_s = f"{r['mae_nt']:.1f}" if not np.isnan(r['mae_nt']) else "---"
                rec_s = f"{r['rec_acc_pct']:.1f}\\%" if not np.isnan(r['rec_acc_pct']) else "---"
                par_s = f"{r['par_acc_pct']:.1f}\\%" if not np.isnan(r['par_acc_pct']) else "---"
                f.write(f"{r['archetype']} & {r['method']} & {r['category']} & {fpr_s} & {pwr_s} & {mae_s} & {rec_s} & {par_s} & {r['runtime_ms']:.1f} \\\\\n")
            f.write("\\midrule\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{\\textbf{Empirical Grand 100,000 Organismal Recombination Benchmark Summary.} Matched performance of 3SEQ and RhizAeon across the 6 organismal archetypes under default operational settings.}\n")
        f.write("\\label{tab:grand_100k_summary}\n")
        f.write("\\end{table*}\n")
    print(f"[✓] LaTeX table saved to: {tex_path}")


def main():
    base_dir = Path(__file__).resolve().parent
    shards_dir = base_dir / "shards"
    master_db = base_dir / "grand_100k_results.db"
    df = merge_shards(shards_dir, master_db)
    compute_summary_tables(df, base_dir)


if __name__ == "__main__":
    main()
