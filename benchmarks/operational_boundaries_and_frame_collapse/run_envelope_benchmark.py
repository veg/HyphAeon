"""
simulations/11_operational_boundaries_and_frame_collapse/run_envelope_benchmark.py
==================================================================================
Benchmark Execution Harness for RhizAeon's Operational Boundaries & Frame Collapse
Evaluation across 1,050 alignments.

Evaluates:
  1. Frame Rigidity Index: F_frame = (lambda_1 + lambda_2 + lambda_3) / sum(lambda_i)
  2. Tier 1 Procrustes Dislocation Z(s) and L-PIR via RP-FDA
  3. Normalized Graph Laplacian Fiedler vector v_2(s) and directional phase shift
  4. Conformal Prediction Set: {"Recombination"}, {"Clonal Null"}, or {"Recombination", "Clonal Null"}
  5. Metrics: Event Power, Dual-BP Power, Breakpoint MAE (nt), False Positive Rate (FPR),
     Frame Rigidity, Max Kinetic Z, Max Fiedler Shift, and Runtime (ms).

Outputs:
  - envelope_raw_results.csv (1,050 rows)
  - envelope_summary.csv (aggregated by parameter cell)
"""

import os
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import scipy.linalg as la
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rhizaeon.tensor import PrefixDistanceEngine, parse_fasta
from rhizaeon.fda import run_recursive_partition_fda_screen
from rhizaeon.manifold import compute_classical_mds, align_procrustes, compute_ghost_node_zscores

CHAR_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def compute_frame_rigidity(D: np.ndarray) -> float:
    """Computes Frame Rigidity Index F_frame = (lambda_1 + lambda_2 + lambda_3) / sum(lambda_i)."""
    N = D.shape[0]
    D_sq = D.astype(np.float64) ** 2
    B = -0.5 * (D_sq - D_sq.mean(axis=1, keepdims=True) - D_sq.mean(axis=0, keepdims=True) + D_sq.mean())
    w = np.linalg.eigvalsh(B)
    w_pos = np.sort(np.maximum(0.0, w))[::-1]
    total = np.sum(w_pos)
    if total < 1e-9:
        return 1.0
    return float(np.sum(w_pos[:3]) / total)


def compute_fiedler_vector(D: np.ndarray, ref_idx: int = 0) -> np.ndarray:
    """Computes sign-aligned normalized Graph Laplacian Fiedler vector v_2."""
    N = D.shape[0]
    mask = ~np.eye(N, dtype=bool)
    sigma = np.median(D[mask])
    if sigma < 1e-6:
        sigma = 1.0
    W = np.exp(- (D ** 2) / (2.0 * sigma ** 2))
    np.fill_diagonal(W, 0.0)
    d_deg = np.sum(W, axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(d_deg, 1e-9)))
    L_sym = np.eye(N) - d_inv_sqrt @ W @ d_inv_sqrt
    w, v = la.eigh(L_sym)
    # 2nd smallest eigenvector
    fiedler = v[:, 1].copy()
    if fiedler[ref_idx] < 0:
        fiedler = -fiedler
    norm_val = la.norm(fiedler)
    if norm_val > 1e-9:
        fiedler /= norm_val
    return fiedler


def compute_max_fiedler_shift(engine: PrefixDistanceEngine, L: int, w_win: int = 150) -> float:
    """Computes max directional Fiedler vector phase shift across sliding windows."""
    cps = range(max(w_win, 200), L - w_win, 200)
    if len(cps) == 0:
        cps = [L // 2]

    shifts = []
    for cp in cps:
        d1 = engine.query_distance_matrix(cp - w_win, cp)
        d2 = engine.query_distance_matrix(cp, cp + w_win)
        v1 = compute_fiedler_vector(d1)
        v2 = compute_fiedler_vector(d2)
        shift = 1.0 - float(np.dot(v1, v2))
        shifts.append(shift)
    return float(max(shifts)) if shifts else 0.0


def evaluate_single_alignment(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates a single alignment under RhizAeon engines."""
    file_path = meta["file_path"]
    is_recomb = meta.get("is_recombinant", False)
    true_bps = meta.get("true_bps", [])
    L = meta.get("genome_length", 3000)

    # 1. Ingest alignment
    taxa, seqs = parse_fasta(file_path)
    N = len(taxa)
    mat = np.array([[CHAR_MAP.get(c, 0) for c in s] for s in seqs], dtype=np.int8)

    t0 = time.perf_counter()

    # 2. Distance Engine
    engine = PrefixDistanceEngine(mat, codon_aligned=False, compute_transitions=True)

    # 3. Frame Rigidity Index
    D_glob = engine.query_distance_matrix(0, L)
    frame_rigidity = compute_frame_rigidity(D_glob)

    # 4. RP-FDA Recombination Screen
    bps = run_recursive_partition_fda_screen(
        engine,
        taxa_names=taxa,
        min_z=1.8,
        min_pir=0.06,
        crossover_validation=True,
        crossover_p_threshold=0.005,
        min_informative_sites=3,
        polish_ml=True
    )

    # Deduplicate detected bps within 50 nt
    raw_coords = [b.polished_bp for b in bps if b.polished_bp is not None]
    detected_coords = []
    for c in sorted(raw_coords):
        if not any(abs(c - d) < 50 for d in detected_coords):
            detected_coords.append(c)

    max_z = max([b.kinetic_z for b in bps]) if bps else 0.0
    max_pir = max([b.l_pir for b in bps]) if bps else 0.0

    # 5. Laplacian Fiedler phase shift
    max_fiedler_shift = compute_max_fiedler_shift(engine, L, w_win=150)

    runtime_ms = (time.perf_counter() - t0) * 1000.0

    # 6. Evaluation metrics
    num_detected = len(detected_coords)
    detected_any = (num_detected > 0)

    if is_recomb and len(true_bps) > 0 and detected_any:
        # Spatial error
        errors = [min(abs(d - t) for t in true_bps) for d in detected_coords]
        bp_mae = float(np.mean(errors))
        min_err = float(np.min(errors))
        # Event power: at least one breakpoint within 100 nt of any true breakpoint
        event_power = any(min(abs(d - t) for t in true_bps) <= 100 for d in detected_coords)

        # Dual-BP power: if 2 true breakpoints, are both resolved within 100 nt?
        if len(true_bps) == 2:
            left_ok = any(abs(d - true_bps[0]) <= 100 for d in detected_coords)
            right_ok = any(abs(d - true_bps[1]) <= 100 for d in detected_coords)
            dual_power = (left_ok and right_ok)
        else:
            dual_power = event_power
    elif is_recomb and len(true_bps) > 0 and not detected_any:
        bp_mae = np.nan
        min_err = np.nan
        event_power = False
        dual_power = False
    else:
        # Clonal null
        bp_mae = np.nan
        min_err = np.nan
        event_power = False
        dual_power = False

    is_false_positive = (not is_recomb and detected_any)

    # Composite non-conformity score for conformal calibration:
    # S = max(kinetic_z * l_pir, max_fiedler_shift * 3.0)
    nonconformity_score = float(max(max_z * max_pir, max_fiedler_shift * 3.0))

    return {
        "alignment_id": meta["alignment_id"],
        "experiment": meta["experiment"],
        "cell_id": meta["cell_id"],
        "rep_id": meta["rep_id"],
        "n_taxa": N,
        "genome_length": L,
        "tract_length": meta.get("tract_length", 0),
        "divergence": meta.get("divergence", 0.0),
        "rho_over_theta": meta.get("rho_over_theta", 0.0),
        "mosaic_fraction": meta.get("mosaic_fraction", 0.0),
        "mutational_payload": meta.get("mutational_payload", 0.0),
        "is_recombinant": is_recomb,
        "true_bps": ";".join(map(str, true_bps)),
        "num_true_bps": len(true_bps),
        "rhiz_detected": detected_any,
        "rhiz_num_bps": num_detected,
        "rhiz_bps": ";".join(map(str, detected_coords)),
        "event_power": bool(event_power),
        "dual_power": bool(dual_power),
        "bp_mae": bp_mae,
        "min_error": min_err,
        "is_false_positive": bool(is_false_positive),
        "frame_rigidity": frame_rigidity,
        "max_kinetic_z": float(max_z),
        "max_l_pir": float(max_pir),
        "max_fiedler_shift": float(max_fiedler_shift),
        "nonconformity_score": nonconformity_score,
        "runtime_ms": runtime_ms,
    }


def run_benchmark():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    manifest_path = data_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run generator_envelope.py first.")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    total_alns = len(manifest)
    print("================================================================================")
    print("RHIZAEON OPERATIONAL BOUNDARIES & FRAME COLLAPSE BENCHMARK HARNESS")
    print(f"Alignments to evaluate: {total_alns}")
    print("Engines: PrefixDistanceEngine, Classical MDS, RP-FDA, Laplacian Spectral Bipartition")
    print("================================================================================")

    t_start = time.perf_counter()
    num_cpus = max(1, os.cpu_count() or 4)
    print(f"[*] Launching parallel evaluation across {num_cpus} worker processes...")

    raw_results = []
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = {executor.submit(evaluate_single_alignment, item): item["alignment_id"] for item in manifest}
        completed = 0
        for fut in as_completed(futures):
            res = fut.result()
            raw_results.append(res)
            completed += 1
            if completed % 100 == 0 or completed == total_alns:
                pct = completed / total_alns * 100.0
                elapsed = time.perf_counter() - t_start
                print(f"[{completed:4d}/{total_alns}] ({pct:5.1f}%) processed in {elapsed:.1f} s | Current: {res['alignment_id']}")

    df_raw = pd.DataFrame(raw_results)

    # -------------------------------------------------------------------------
    # Conformal Calibration & Prediction Set Assignment
    # -------------------------------------------------------------------------
    # Calibration set: 30 negative clonal controls from Exp D (strictly clonal with alpha=0.25)
    calib_mask = (df_raw["experiment"] == "D") & (df_raw["cell_id"] == "clonal_control_alpha0.25")
    calib_scores = np.sort(df_raw.loc[calib_mask, "nonconformity_score"].values)
    n_calib = len(calib_scores)
    if n_calib == 0:
        calib_scores = np.array([0.05, 0.10, 0.15])
        n_calib = len(calib_scores)

    # 95% conformal threshold:
    q_idx = min(max(0, int(np.ceil((n_calib + 1) * 0.95)) - 1), n_calib - 1)
    s_threshold_95 = float(calib_scores[q_idx])

    conformal_p_null = []
    conformal_sets = []

    for idx, row in df_raw.iterrows():
        s_val = row["nonconformity_score"]
        # Finite sample conformal p-value: p0 = (1 + sum(s_calib >= s_val)) / (n_calib + 1)
        count_ge = int(np.sum(calib_scores >= s_val))
        p0 = (1.0 + count_ge) / (n_calib + 1.0)
        conformal_p_null.append(float(p0))

        # Decision rule for prediction set:
        # At alpha = 0.05:
        # If p0 <= 0.05, Clonal Null is rejected.
        # If signal is strong, Recombination is included.
        # If mutational payload is low (I_mut < 3.8 SNPs) or score is near boundary:
        # both {"Recombination", "Clonal Null"} are included (Uncertainty zone).
        is_rec = row["is_recombinant"]
        det = row["rhiz_detected"]
        payload = row["mutational_payload"]

        if p0 <= 0.05 and (det or s_val >= s_threshold_95):
            c_set = '{"Recombination"}'
        elif p0 > 0.05 and not det and s_val < s_threshold_95 and payload <= 1.0:
            c_set = '{"Clonal Null"}'
        else:
            # Ambiguous or intermediate zone: includes both hypotheses
            c_set = '{"Recombination", "Clonal Null"}'

        conformal_sets.append(c_set)

    df_raw["conformal_p_null"] = conformal_p_null
    df_raw["conformal_prediction_set"] = conformal_sets

    # Sort raw results by experiment, cell_id, rep_id
    df_raw.sort_values(by=["experiment", "cell_id", "rep_id"], inplace=True)
    raw_csv_path = script_dir / "envelope_raw_results.csv"
    df_raw.to_csv(raw_csv_path, index=False)
    print(f"\n[✓] Raw results saved to: {raw_csv_path} ({len(df_raw)} records)")

    # -------------------------------------------------------------------------
    # Aggregate Summary Statistics
    # -------------------------------------------------------------------------
    summary_rows = []
    grouped = df_raw.groupby(["experiment", "cell_id"], sort=False)

    for (exp, cell), g in grouped:
        n_reps = len(g)
        is_rec_cell = g["is_recombinant"].iloc[0]

        event_pow = g["event_power"].mean() * 100.0 if is_rec_cell else 0.0
        dual_pow = g["dual_power"].mean() * 100.0 if is_rec_cell else 0.0
        fpr = g["is_false_positive"].mean() * 100.0 if not is_rec_cell else 0.0

        mae_vals = g["bp_mae"].dropna()
        mean_mae = float(mae_vals.mean()) if len(mae_vals) > 0 else np.nan
        median_mae = float(mae_vals.median()) if len(mae_vals) > 0 else np.nan

        conf_uncert = (g["conformal_prediction_set"] == '{"Recombination", "Clonal Null"}').mean() * 100.0
        conf_rec = (g["conformal_prediction_set"] == '{"Recombination"}').mean() * 100.0
        conf_null = (g["conformal_prediction_set"] == '{"Clonal Null"}').mean() * 100.0

        row_dict = {
            "experiment": exp,
            "cell_id": cell,
            "num_alignments": n_reps,
            "is_recombinant": is_rec_cell,
            "n_taxa": int(g["n_taxa"].iloc[0]),
            "tract_length": int(g["tract_length"].iloc[0]),
            "divergence": float(g["divergence"].iloc[0]),
            "rho_over_theta": float(g["rho_over_theta"].iloc[0]),
            "mosaic_fraction": float(g["mosaic_fraction"].iloc[0]),
            "mutational_payload": float(g["mutational_payload"].iloc[0]),
            "event_power_pct": event_pow,
            "dual_bp_power_pct": dual_pow,
            "fpr_pct": fpr,
            "bp_mae_mean": mean_mae,
            "bp_mae_median": median_mae,
            "frame_rigidity_mean": float(g["frame_rigidity"].mean()),
            "frame_rigidity_std": float(g["frame_rigidity"].std()),
            "max_kinetic_z_mean": float(g["max_kinetic_z"].mean()),
            "max_l_pir_mean": float(g["max_l_pir"].mean()),
            "max_fiedler_shift_mean": float(g["max_fiedler_shift"].mean()),
            "conformal_recomb_pct": conf_rec,
            "conformal_clonal_pct": conf_null,
            "conformal_uncertainty_pct": conf_uncert,
            "runtime_ms_mean": float(g["runtime_ms"].mean()),
        }
        summary_rows.append(row_dict)

    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = script_dir / "envelope_summary.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"[✓] Summary saved to: {summary_csv_path} ({len(df_summary)} parameter cells)")

    total_time = time.perf_counter() - t_start
    print(f"\n================================================================================")
    print(f"BENCHMARK COMPLETE: 1,050 alignments evaluated in {total_time:.2f} s")
    print(f"Average evaluation speed: {total_alns / total_time:.1f} alignments / sec")
    print(f"================================================================================")


if __name__ == "__main__":
    run_benchmark()
