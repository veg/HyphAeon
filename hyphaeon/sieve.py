"""
hyphaeon/sieve.py
-----------------
High-Throughput Streaming Pathogen Genomic Quality Control & Clock Manifold Sieve.

Workflow:
1. Anchor Calibration:
   Selects a temporally and phylogenetically representative anchor skeleton (N = 256 to 4096).
   Calibrates a robust heterochronous molecular clock (with Fieller identifiability audit).
   Establishes ancestral root consensus and reference distance distributions.

2. Streaming Ingestion & Triage:
   Streams candidate genomes one-at-a-time (or in micro-batches).
   Computes vector distances against root and anchor references via fast vectorized TN93.
   Projects query onto the molecular clock manifold:
     - Predicts expected collection date (t_hat).
     - Computes temporal error (delta_t) and divergence residual Z-score.
     - Identifies phylogenetic nearest neighbor (r*) and clade affiliation.

3. Outlier Taxonomy (SUS Classification):
   - PASS: Clean, temporally and phylogenetically consistent.
   - SUS_DATE_MISMATCH: Genetically normal, but reported date is severely discrepant.
   - SUS_ARCHIVAL_OR_LAB_LEAK: Recent sample date, but divergence matches ancestral root.
   - SUS_HYPERMUTATED_DEGRADED: Massively elevated divergence or excess private mutations.
   - SUS_CHIMERIC_RECOMBINANT: Discordant nearest-neighbor clade affiliation across 5' vs 3' halves.
   - SUS_NON_TARGET_CONTAMINANT: Sequence fails to match any reference in the anchor skeleton.
   - SUS_DEGRADED_SEQUENCE: Severe ambiguity (>20% Ns) or excessive gap ratio.
"""

import os
import sys
import time
import math
import re
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import scipy.stats as stats

from .dataset import parse_alignment_sequences
from .temporal import parse_date_to_decimal, extract_date_from_string
from .dating import run_ols_dating, compute_fieller_mrca_interval


def compute_vector_tn93(
    query_seq: str,
    ref_seqs: List[str],
    seq_names: Optional[List[str]] = None
) -> np.ndarray:
    """
    Computes Tamura-Nei 93 (TN93) genetic distances from a single query sequence
    to a list of M reference sequences.
    Prefers the compiled 'tn93' binary when available, falling back to a vectorized
    NumPy implementation.
    """
    m = len(ref_seqs)
    if m == 0:
        return np.array([], dtype=np.float64)

    tn93_bin = shutil.which("tn93")
    if tn93_bin and m >= 64:
        with tempfile.TemporaryDirectory() as tmpdir:
            q_fa = os.path.join(tmpdir, "query.fa")
            ref_fa = os.path.join(tmpdir, "refs.fa")
            out_csv = os.path.join(tmpdir, "dists.csv")

            with open(q_fa, "w") as f:
                f.write(f">QUERY\n{query_seq}\n")
            with open(ref_fa, "w") as f:
                for idx, r in enumerate(ref_seqs):
                    name = seq_names[idx] if seq_names else f"REF_{idx}"
                    f.write(f">{name}\n{r}\n")

            cmd = [tn93_bin, "-s", ref_fa, "-t", "1.0", "-l", "50", "-q", "-o", out_csv, q_fa]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
                    df = pd.read_csv(out_csv, usecols=["ID1", "ID2", "Distance"])
                    ref_name_map = {name: idx for idx, name in enumerate(seq_names)} if seq_names else {f"REF_{idx}": idx for idx in range(m)}
                    dists = np.full(m, 1.0, dtype=np.float64)
                    for _, row in df.iterrows():
                        r_name = row["ID2"] if row["ID1"] == "QUERY" else row["ID1"]
                        if r_name in ref_name_map:
                            dists[ref_name_map[r_name]] = float(row["Distance"])
                    return dists
            except Exception:
                pass

    # Fast Vectorized NumPy Fallback
    q_chars = np.frombuffer(query_seq.upper().encode('ascii'), dtype=np.uint8)
    L = len(q_chars)

    # Encode: A=1, C=2, G=3, T=4, others=0
    lut = np.zeros(256, dtype=np.uint8)
    lut[ord('A')] = 1
    lut[ord('C')] = 2
    lut[ord('G')] = 3
    lut[ord('T')] = 4
    lut[ord('U')] = 4  # Treat U as T

    q_enc = lut[q_chars]
    is_q_valid = (q_enc > 0)

    dists = np.zeros(m, dtype=np.float64)

    for i in range(m):
        r_str = ref_seqs[i].upper()
        if len(r_str) != L:
            # Length mismatch; truncate or pad
            min_len = min(L, len(r_str))
            r_chars = np.frombuffer(r_str[:min_len].encode('ascii'), dtype=np.uint8)
            r_enc = np.zeros(L, dtype=np.uint8)
            r_enc[:min_len] = lut[r_chars]
        else:
            r_chars = np.frombuffer(r_str.encode('ascii'), dtype=np.uint8)
            r_enc = lut[r_chars]

        valid = is_q_valid & (r_enc > 0)
        tot_valid = int(np.sum(valid))

        if tot_valid < 50:
            dists[i] = 1.0
            continue

        q_v = q_enc[valid]
        r_v = r_enc[valid]

        diff = (q_v != r_v)
        tot_diff = np.sum(diff)

        if tot_diff == 0:
            dists[i] = 0.0
            continue

        # Transitions: (A<->G: 1<->3) or (C<->T: 2<->4)
        is_ag = ((q_v == 1) & (r_v == 3)) | ((q_v == 3) & (r_v == 1))
        is_ct = ((q_v == 2) & (r_v == 4)) | ((q_v == 4) & (r_v == 2))
        transversions = diff & (~is_ag) & (~is_ct)

        P1 = float(np.sum(is_ag)) / tot_valid
        P2 = float(np.sum(is_ct)) / tot_valid
        Q = float(np.sum(transversions)) / tot_valid

        # Base frequencies across valid sites of the pair
        fA = (np.sum(q_v == 1) + np.sum(r_v == 1)) / (2.0 * tot_valid)
        fC = (np.sum(q_v == 2) + np.sum(r_v == 2)) / (2.0 * tot_valid)
        fG = (np.sum(q_v == 3) + np.sum(r_v == 3)) / (2.0 * tot_valid)
        fT = (np.sum(q_v == 4) + np.sum(r_v == 4)) / (2.0 * tot_valid)

        fR = fA + fG
        fY = fC + fT

        if fR <= 1e-6 or fY <= 1e-6 or fA <= 1e-6 or fG <= 1e-6 or fC <= 1e-6 or fT <= 1e-6:
            # Fallback to simple Hamming p-distance
            dists[i] = float(tot_diff) / tot_valid
            continue

        a1 = (fA * fG) / fR
        a2 = (fC * fT) / fY

        arg1 = 1.0 - (P1 / (2.0 * a1)) - (Q / (2.0 * fR))
        arg2 = 1.0 - (P2 / (2.0 * a2)) - (Q / (2.0 * fY))
        arg3 = 1.0 - (Q / (2.0 * fR * fY))

        if arg1 <= 0.0 or arg2 <= 0.0 or arg3 <= 0.0:
            # Saturated pair
            dists[i] = 1.0
        else:
            d = (-2.0 * a1 * math.log(arg1)
                 - 2.0 * a2 * math.log(arg2)
                 - 2.0 * (fR * fY - a1 * fY - a2 * fR) * math.log(arg3))
            dists[i] = float(max(0.0, d))

    return dists


class ChronAeonSieve:
    """
    High-Throughput Quality Control and Molecular Clock Streaming Sieve.
    """

    def __init__(
        self,
        anchor_taxa: List[str],
        anchor_seqs: List[str],
        anchor_dates: np.ndarray,
        root_seq: str,
        clock_params: Dict[str, Any],
        tolerance_days: float = 60.0,
        z_threshold: float = 2.0,
        max_ambig_ratio: float = 0.05
    ):
        self.anchor_taxa = list(anchor_taxa)
        self.anchor_seqs = list(anchor_seqs)
        self.anchor_dates = np.asarray(anchor_dates, dtype=np.float64)
        self.root_seq = root_seq.upper()
        self.clock_params = clock_params
        self.tolerance_days = float(tolerance_days)
        self.z_threshold = float(z_threshold)
        self.max_ambig_ratio = float(max_ambig_ratio)

        self.mu = float(clock_params['mu'])
        self.d0 = float(clock_params['d0'])
        self.t_ref = float(clock_params['t_ref'])
        self.t_mrca = float(clock_params['t_mrca'])
        self.sigma_res = float(max(1e-6, clock_params.get('sigma_res', 0.001)))
        self.seq_len = len(root_seq)
        self.timespan = float(np.max(self.anchor_dates) - np.min(self.anchor_dates))

        # Build Look-Up Table: A=1, C=2, G=3, T/U=4, else=0
        self.lut = np.zeros(256, dtype=np.uint8)
        self.lut[ord('A')] = 1
        self.lut[ord('C')] = 2
        self.lut[ord('G')] = 3
        self.lut[ord('T')] = 4
        self.lut[ord('U')] = 4

        # Pre-encode root sequence
        r_chars = np.frombuffer(self.root_seq.encode('ascii'), dtype=np.uint8)
        self.root_enc = self.lut[r_chars]

        # Pre-encode all anchor sequences into contiguous 2D NumPy array (M, L)
        m = len(self.anchor_seqs)
        print(f"[*] Pre-encoding {m} anchor sequences ({self.seq_len} nt) into memory...")
        self.anchor_enc = np.zeros((m, self.seq_len), dtype=np.uint8)
        for i, s in enumerate(self.anchor_seqs):
            chars = np.frombuffer(s[:self.seq_len].upper().encode('ascii'), dtype=np.uint8)
            self.anchor_enc[i, :len(chars)] = self.lut[chars]

        # Compute baseline root-to-tip divergences for anchor set
        print(f"[*] Calibrating baseline anchor divergences (M={m})...")
        self.anchor_root_dists = self._compute_distances_matrix(self.root_enc[None, :], self.anchor_enc)[0]
        self.median_nn_dist = float(np.median(self.anchor_root_dists)) if len(self.anchor_root_dists) > 0 else 0.005

    @classmethod
    def build_from_alignment(
        cls,
        alignment_path: Union[str, Path],
        dates_path: Optional[Union[str, Path]] = None,
        dates_dict: Optional[Dict[str, float]] = None,
        date_col: Optional[str] = None,
        strain_col: Optional[str] = None,
        n_anchor: int = 1024,
        n_temporal_bins: int = 32,
        root_taxon: Optional[str] = None,
        seed: int = 42,
        tolerance_days: float = 60.0,
        z_threshold: float = 2.0,
        max_ambig_ratio: float = 0.05
    ) -> "ChronAeonSieve":
        """
        Extracts a stratified, high-diversity anchor skeleton and calibrates a pristine clock.
        """
        print(f"[*] Loading alignment: {alignment_path}...")
        seq_dict = parse_alignment_sequences(str(alignment_path))
        taxa_all = list(seq_dict.keys())

        dates_map = {}
        if dates_dict is not None:
            dates_map.update(dates_dict)
        elif dates_path is not None:
            df = pd.read_csv(dates_path)
            s_col = strain_col if strain_col and strain_col in df.columns else df.columns[0]
            d_col = date_col if date_col and date_col in df.columns else ("collection_date" if "collection_date" in df.columns else df.columns[1])
            for _, r in df.iterrows():
                parsed = parse_date_to_decimal(str(r[d_col]))
                if parsed is not None:
                    raw_id = str(r[s_col]).strip("'\"")
                    dates_map[raw_id] = parsed
            # Also map headers that have pipe-delimited prefixes or fallback to header date
            for t in taxa_all:
                if t not in dates_map:
                    prefix = t.split("|")[0].strip()
                    if prefix in dates_map:
                        dates_map[t] = dates_map[prefix]
                    else:
                        parsed = parse_date_to_decimal(t)
                        if parsed is None or np.isnan(parsed):
                            parsed = extract_date_from_string(t)
                        if parsed is not None and not np.isnan(parsed):
                            dates_map[t] = parsed
        else:
            # Attempt FASTA header extraction
            for t in taxa_all:
                parsed = parse_date_to_decimal(t)
                if parsed is None or np.isnan(parsed):
                    parsed = extract_date_from_string(t)
                if parsed is not None and not np.isnan(parsed):
                    dates_map[t] = parsed

        # Filter for dated taxa with sufficient sequence completeness
        dated_taxa = []
        for t in taxa_all:
            if t in dates_map and not np.isnan(dates_map[t]):
                s = seq_dict[t].upper()
                ambig_cnt = sum(s.count(c) for c in 'N-?')
                if ambig_cnt / max(1, len(s)) <= max_ambig_ratio:
                    dated_taxa.append(t)

        if len(dated_taxa) < 10:
            dated_taxa = [t for t in taxa_all if t in dates_map and not np.isnan(dates_map[t])]
            if len(dated_taxa) < 10:
                raise ValueError(f"Fewer than 10 dated taxa found in candidate pool ({len(dated_taxa)} valid).")

        print(f"[*] Valid dated sequences: {len(dated_taxa)}/{len(taxa_all)} (filtered for completeness <= {max_ambig_ratio*100:.1f}% missing).")

        # Stratified Temporal Selection
        rng = np.random.default_rng(seed)
        t_vals = np.array([dates_map[t] for t in dated_taxa])
        t_min, t_max = float(np.min(t_vals)), float(np.max(t_vals))
        bins = np.linspace(t_min, t_max, n_temporal_bins + 1)

        selected_taxa = []
        target_per_bin = max(2, int(math.ceil(n_anchor / n_temporal_bins)))

        for b_idx in range(n_temporal_bins):
            b_start, b_end = bins[b_idx], bins[b_idx + 1]
            in_bin = [dated_taxa[i] for i in range(len(dated_taxa)) if b_start <= t_vals[i] <= b_end]
            if not in_bin:
                continue
            if len(in_bin) <= target_per_bin:
                selected_taxa.extend(in_bin)
            else:
                chosen = rng.choice(in_bin, size=target_per_bin, replace=False)
                selected_taxa.extend(chosen)

        selected_taxa = list(dict.fromkeys(selected_taxa))
        if len(selected_taxa) > n_anchor:
            selected_taxa = list(rng.choice(selected_taxa, size=n_anchor, replace=False))

        print(f"[✓] Stratified Anchor Skeleton: Selected {len(selected_taxa)} representative taxa across {t_min:.1f} - {t_max:.1f}.")

        # Define root sequence (Time-Decay Weighted Consensus or Earliest Sequence)
        earliest_taxon = min(selected_taxa, key=lambda t: dates_map[t])
        root_seq = seq_dict.get(root_taxon, seq_dict[earliest_taxon])

        anc_seqs = [seq_dict[t] for t in selected_taxa]
        anc_times = np.array([dates_map[t] for t in selected_taxa], dtype=np.float64)
        anc_dists = compute_vector_tn93(root_seq, anc_seqs, selected_taxa)

        # Fit OLS clock on pristine anchor set
        ols_fit = run_ols_dating(anc_times, anc_dists, ci_method="fieller")
        print(f"[✓] Anchor Clock Calibrated: t_MRCA = {ols_fit['t_mrca']:.2f}, μ = {ols_fit['mu']:.6f} subs/site/yr (R^2 = {ols_fit['r2']:.3f}, g = {ols_fit['fieller_g']:.4f})")

        clock_params = {
            'mu': ols_fit['mu'],
            'd0': ols_fit['d0'],
            't_ref': ols_fit['t_ref'],
            't_mrca': ols_fit['t_mrca'],
            'ci_mrca': ols_fit['ci_mrca'],
            'fieller_g': ols_fit['fieller_g'],
            'sigma_res': float(np.std(ols_fit['residuals'])),
            'r2': ols_fit['r2'],
            'method': 'OLS'
        }

        return cls(
            anchor_taxa=selected_taxa,
            anchor_seqs=anc_seqs,
            anchor_dates=anc_times,
            root_seq=root_seq,
            clock_params=clock_params,
            tolerance_days=tolerance_days,
            z_threshold=z_threshold,
            max_ambig_ratio=max_ambig_ratio
        )


    def _compute_distances_matrix(self, q_enc_2d: np.ndarray, ref_enc_2d: np.ndarray) -> np.ndarray:
        """
        Pure NumPy vectorized TN93 calculation between queries (Q, L) and references (M, L).
        Zero disk I/O, zero subprocess overhead.
        """
        Q_n, L = q_enc_2d.shape
        M_n, _ = ref_enc_2d.shape
        dists = np.zeros((Q_n, M_n), dtype=np.float64)

        for q_idx in range(Q_n):
            q_v = q_enc_2d[q_idx]
            q_valid = (q_v > 0)

            # Broadcast q_v against ref_enc_2d: shape (M_n, L)
            valid = q_valid[None, :] & (ref_enc_2d > 0)
            tot_valid = np.sum(valid, axis=1)

            diff = valid & (q_v[None, :] != ref_enc_2d)
            tot_diff = np.sum(diff, axis=1)

            # Transitions: A(1)<->G(3), C(2)<->T(4)
            is_ag = valid & (((q_v[None, :] == 1) & (ref_enc_2d == 3)) | ((q_v[None, :] == 3) & (ref_enc_2d == 1)))
            is_ct = valid & (((q_v[None, :] == 2) & (ref_enc_2d == 4)) | ((q_v[None, :] == 4) & (ref_enc_2d == 2)))
            transversions = diff & (~is_ag) & (~is_ct)

            P1 = np.sum(is_ag, axis=1) / np.maximum(1, tot_valid)
            P2 = np.sum(is_ct, axis=1) / np.maximum(1, tot_valid)
            Q = np.sum(transversions, axis=1) / np.maximum(1, tot_valid)

            # Base frequencies
            fA = (np.sum(valid & (q_v[None, :] == 1), axis=1) + np.sum(valid & (ref_enc_2d == 1), axis=1)) / (2.0 * np.maximum(1, tot_valid))
            fC = (np.sum(valid & (q_v[None, :] == 2), axis=1) + np.sum(valid & (ref_enc_2d == 2), axis=1)) / (2.0 * np.maximum(1, tot_valid))
            fG = (np.sum(valid & (q_v[None, :] == 3), axis=1) + np.sum(valid & (ref_enc_2d == 3), axis=1)) / (2.0 * np.maximum(1, tot_valid))
            fT = (np.sum(valid & (q_v[None, :] == 4), axis=1) + np.sum(valid & (ref_enc_2d == 4), axis=1)) / (2.0 * np.maximum(1, tot_valid))

            fR = fA + fG
            fY = fC + fT

            safe_denom = (fR > 1e-6) & (fY > 1e-6) & (fA > 1e-6) & (fG > 1e-6) & (fC > 1e-6) & (fT > 1e-6)
            a1 = np.divide(fA * fG, fR, where=safe_denom, out=np.zeros_like(fA))
            a2 = np.divide(fC * fT, fY, where=safe_denom, out=np.zeros_like(fC))

            arg1 = 1.0 - np.divide(P1, 2.0 * a1, where=(a1 > 1e-6), out=np.zeros_like(P1)) - np.divide(Q, 2.0 * fR, where=(fR > 1e-6), out=np.zeros_like(Q))
            arg2 = 1.0 - np.divide(P2, 2.0 * a2, where=(a2 > 1e-6), out=np.zeros_like(P2)) - np.divide(Q, 2.0 * fY, where=(fY > 1e-6), out=np.zeros_like(Q))
            arg3 = 1.0 - np.divide(Q, 2.0 * fR * fY, where=((fR * fY) > 1e-6), out=np.zeros_like(Q))

            pos_args = (arg1 > 0) & (arg2 > 0) & (arg3 > 0) & safe_denom & (tot_valid >= 50)
            d_tn93 = np.zeros(M_n, dtype=np.float64)

            # Apply TN93 where valid
            idx_pos = np.where(pos_args)[0]
            if len(idx_pos) > 0:
                d_tn93[idx_pos] = (
                    -2.0 * a1[idx_pos] * np.log(arg1[idx_pos])
                    - 2.0 * a2[idx_pos] * np.log(arg2[idx_pos])
                    - 2.0 * (fR[idx_pos] * fY[idx_pos] - a1[idx_pos] * fY[idx_pos] - a2[idx_pos] * fR[idx_pos]) * np.log(arg3[idx_pos])
                )

            # Fallback to p-distance for non-TN93 or invalid
            idx_p = np.where((~pos_args) & (tot_valid >= 50))[0]
            if len(idx_p) > 0:
                d_tn93[idx_p] = tot_diff[idx_p] / tot_valid[idx_p]

            idx_bad = np.where(tot_valid < 50)[0]
            if len(idx_bad) > 0:
                d_tn93[idx_bad] = 1.0

            dists[q_idx] = np.maximum(0.0, d_tn93)

        return dists

    def screen_sequence(
        self,
        query_id: str,
        query_seq: str,
        reported_date: Union[float, str]
    ) -> Dict[str, Any]:
        """
        Screens an individual streaming query sequence against the calibrated clock manifold.
        """
        t0 = time.time()
        rep_date = parse_date_to_decimal(str(reported_date)) if isinstance(reported_date, str) else float(reported_date)
        if rep_date is None or np.isnan(rep_date):
            rep_date = extract_date_from_string(str(reported_date))
        if rep_date is None or np.isnan(rep_date):
            rep_date = extract_date_from_string(query_id)

        if rep_date is None or np.isnan(rep_date):
            return {
                'query_id': query_id,
                'status': 'FLAG_MISSING_DATE',
                'sus_reason': 'Missing or unparseable collection date',
                'reported_date': np.nan,
                'predicted_date': np.nan,
                'temporal_error_days': np.nan,
                'divergence_z': np.nan,
                'nearest_neighbor': None,
                'nn_distance': np.nan,
                'elapsed_ms': (time.time() - t0) * 1000.0
            }

        # 1. Quality & Missing Data Audit (Any non-ACGTU character, gap, or ambiguity is missing data)
        q_upper = query_seq.upper()
        n_missing = sum(1 for c in q_upper if c not in 'ACGTU')
        missing_ratio = n_missing / max(1, len(q_upper))
        if missing_ratio > self.max_ambig_ratio:
            return {
                'query_id': query_id,
                'status': 'SUS',
                'sus_reason': f'SUS_LOW_QUALITY (excess missing/degenerate data: {missing_ratio*100:.1f}% > {self.max_ambig_ratio*100:.1f}% threshold)',
                'reported_date': rep_date,
                'predicted_date': np.nan,
                'temporal_error_days': np.nan,
                'temporal_error_years': np.nan,
                'divergence_z': np.nan,
                'root_divergence': np.nan,
                'nearest_neighbor': None,
                'nn_date': np.nan,
                'nn_distance': np.nan,
                'elapsed_ms': (time.time() - t0) * 1000.0
            }

        # Encode query
        q_chars = np.frombuffer(q_upper[:self.seq_len].encode('ascii'), dtype=np.uint8)
        q_enc = np.zeros(self.seq_len, dtype=np.uint8)
        q_enc[:len(q_chars)] = self.lut[q_chars]

        # 2. Compute Root Divergence
        d_root = float(self._compute_distances_matrix(q_enc[None, :], self.root_enc[None, :])[0, 0])

        # 3. Vectorized Nearest-Neighbor Search against Anchor Matrix
        anc_dists = self._compute_distances_matrix(q_enc[None, :], self.anchor_enc)[0]
        nn_idx = int(np.argmin(anc_dists))
        nn_dist = float(anc_dists[nn_idx])
        nn_taxon = self.anchor_taxa[nn_idx]
        nn_date = float(self.anchor_dates[nn_idx])

        # 4. Invert Molecular Clock
        if self.mu > 1e-7:
            pred_date = self.t_ref + (d_root - self.d0) / self.mu
            err_years = pred_date - rep_date
            err_days = err_years * 365.25
        else:
            pred_date = np.nan
            err_years = np.nan
            err_days = np.nan

        expected_d = self.d0 + self.mu * (rep_date - self.t_ref)
        res_d = d_root - expected_d
        z_score = float(res_d / self.sigma_res)

        # 5. Multidimensional Anomaly Classification
        status = "PASS"
        sus_reason = "Consistent with molecular clock"

        # A. Non-target contaminant (very distant from all references)
        if nn_dist > max(0.08, 3.5 * self.median_nn_dist):
            status = "SUS"
            sus_reason = f"SUS_NON_TARGET_CONTAMINANT (nn_dist={nn_dist:.4f} > threshold)"

        # B. Archival isolate / laboratory vector carryover
        # (strictly near-zero root distance sampled late in outbreak timeline)
        elif d_root <= 0.0002 and rep_date >= (self.t_mrca + 0.60 * self.timespan) and z_score <= -2.5:
            status = "SUS"
            sus_reason = f"SUS_ARCHIVAL_OR_LAB_LEAK (d_root={d_root:.5f} ≈ 0, sampled {rep_date:.2f})"

        # C. Over-diverged: Distinguish biological hypermutation from missing data ('N') artifacts
        elif z_score > 2.5 and (nn_dist > (1.25 * self.median_nn_dist) or abs(nn_date - rep_date) <= 1.5):
            status = "SUS"
            if n_missing > 10 or missing_ratio > 0.02:
                sus_reason = f"SUS_LOW_QUALITY (Z={z_score:+.2f}, divergence artifact driven by {n_missing} missing/degenerate bases)"
            else:
                sus_reason = f"SUS_HYPERMUTATED (Z={z_score:+.2f}, biological excess divergence with high sequence completeness)"

        # D. Split-window recombination check (5' vs 3' halves)
        elif self.seq_len >= 1000:
            half = self.seq_len // 2
            d_5p = self._compute_distances_matrix(q_enc[None, :half], self.anchor_enc[:, :half])[0]
            d_3p = self._compute_distances_matrix(q_enc[None, half:], self.anchor_enc[:, half:])[0]
            nn_5p = int(np.argmin(d_5p))
            nn_3p = int(np.argmin(d_3p))
            # Test reciprocal topological discordance
            d5_disc = d_5p[nn_3p] - d_5p[nn_5p]
            d3_disc = d_3p[nn_5p] - d_3p[nn_3p]
            date_diff = abs(self.anchor_dates[nn_5p] - self.anchor_dates[nn_3p])

            if nn_5p != nn_3p and d5_disc >= 0.0015 and d3_disc >= 0.0015 and date_diff >= (0.25 * self.timespan):
                status = "SUS"
                sus_reason = f"SUS_CHIMERIC_RECOMBINANT (5' nn={self.anchor_taxa[nn_5p]} vs 3' nn={self.anchor_taxa[nn_3p]}, Δt={date_diff:.2f}yr)"

        # E. Date typo or metadata discrepancy
        if status == "PASS" and not np.isnan(err_days):
            is_large_date_err = abs(err_days) > self.tolerance_days
            is_stat_sig_z = abs(z_score) >= self.z_threshold
            is_extreme_err = abs(err_days) > max(365.25, 0.75 * self.timespan * 365.25)
            is_normal_seq = nn_dist <= (2.5 * max(self.median_nn_dist, 0.001))

            if ((is_large_date_err and is_stat_sig_z) or is_extreme_err) and is_normal_seq:
                status = "SUS"
                sus_reason = f"SUS_DATE_MISMATCH (discrepancy={err_days:+.1f} days, Z={z_score:+.2f}, nn_date={nn_date:.2f})"

        elapsed_ms = (time.time() - t0) * 1000.0

        return {
            'query_id': query_id,
            'status': status,
            'sus_reason': sus_reason,
            'reported_date': rep_date,
            'predicted_date': pred_date,
            'temporal_error_days': err_days,
            'temporal_error_years': err_years,
            'divergence_z': z_score,
            'root_divergence': d_root,
            'nearest_neighbor': nn_taxon,
            'nn_date': nn_date,
            'nn_distance': nn_dist,
            'elapsed_ms': elapsed_ms
        }

    def align_queries_minimap2(
        self,
        query_dict: Dict[str, str],
        min_identity: float = 0.50
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Ultra-fast reference-coordinate alignment using minimap2.
        Projects raw, unaligned streaming query genomes onto the anchor coordinate system.
        """
        mm2_bin = shutil.which("minimap2")
        if not mm2_bin:
            raise RuntimeError("minimap2 executable not found on system PATH. Please install minimap2.")

        aligned_dict = {}
        failed_dict = {}

        with tempfile.TemporaryDirectory() as td:
            ref_fa = os.path.join(td, "ref.fa")
            q_fa = os.path.join(td, "queries.fa")

            with open(ref_fa, "w") as f:
                f.write(f">REF\n{self.root_seq}\n")

            with open(q_fa, "w") as f:
                for qid, qseq in query_dict.items():
                    clean_s = "".join(qseq.split())
                    f.write(f">{qid}\n{clean_s}\n")

            cmd = [mm2_bin, "-c", "--eqx", "-x", "asm5", ref_fa, q_fa]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        aligned_buffers = {qid: ["-"] * self.seq_len for qid in query_dict}
        total_matches = {qid: 0 for qid in query_dict}

        for line in proc.stdout.splitlines():
            if not line:
                continue
            parts = line.split('\t')
            qid = parts[0]
            if qid not in query_dict:
                continue

            strand = parts[4]
            tstart = int(parts[7])
            matches = int(parts[9])
            total_matches[qid] += matches

            cigar = None
            for tag in parts[12:]:
                if tag.startswith("cg:Z:"):
                    cigar = tag[5:]
                    break

            if cigar is None:
                continue

            q_seq = "".join(query_dict[qid].split())
            if strand == "-":
                rc_map = str.maketrans("ACGTNacgtn", "TGCANtgcan")
                q_seq = q_seq.translate(rc_map)[::-1]

            ops = re.findall(r"(\d+)([=XIDMSH])", cigar)
            r_pos = tstart
            q_pos = int(parts[2])

            for length_str, op in ops:
                l = int(length_str)
                if op in ("=", "X", "M"):
                    for i in range(l):
                        if 0 <= r_pos + i < self.seq_len and q_pos + i < len(q_seq):
                            aligned_buffers[qid][r_pos + i] = q_seq[q_pos + i]
                    r_pos += l
                    q_pos += l
                elif op == "D":
                    r_pos += l
                elif op == "I":
                    q_pos += l
                elif op == "S":
                    q_pos += l

        for qid in query_dict:
            m = total_matches[qid]
            if m < (min_identity * self.seq_len):
                if m == 0:
                    failed_dict[qid] = "SUS_NON_TARGET_CONTAMINANT (minimap2 unmapped / completely foreign sequence)"
                else:
                    failed_dict[qid] = f"SUS_NON_TARGET_CONTAMINANT (minimap2 low identity: {m}/{self.seq_len} matches)"
            else:
                aligned_dict[qid] = "".join(aligned_buffers[qid])

        return aligned_dict, failed_dict

    def screen_stream(
        self,
        stream_fasta: Union[str, Path],
        stream_dates: Optional[Union[str, Path, Dict[str, float]]] = None,
        date_col: Optional[str] = None,
        strain_col: Optional[str] = None,
        align_minimap2: bool = False,
        clean_fasta_out: Optional[Union[str, Path]] = None,
        sus_fasta_out: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Screens an incoming batch of streaming sequences, returning a diagnostic DataFrame.
        If align_minimap2 is True, raw unaligned sequences are dynamically mapped and coordinate-
        standardized against the ancestral reference using minimap2.
        If clean_fasta_out or sus_fasta_out are provided, writes segregated FASTAs directly.
        """
        t_start = time.time()
        stream_dict = parse_alignment_sequences(str(stream_fasta))
        raw_dict = dict(stream_dict)
        dates_map = {}

        if isinstance(stream_dates, dict):
            dates_map = dict(stream_dates)
        elif stream_dates and Path(stream_dates).exists():
            df_d = pd.read_csv(stream_dates)
            s_col = strain_col if strain_col and strain_col in df_d.columns else df_d.columns[0]
            d_col = date_col if date_col and date_col in df_d.columns else df_d.columns[1]
            for _, r in df_d.iterrows():
                p = parse_date_to_decimal(str(r[d_col]))
                if p is None or np.isnan(p):
                    p = extract_date_from_string(str(r[d_col]))
                if p is not None and not np.isnan(p):
                    raw_id = str(r[s_col]).strip("'\"")
                    dates_map[raw_id] = p
            for k in stream_dict:
                if k not in dates_map:
                    prefix = k.split("|")[0].strip()
                    if prefix in dates_map:
                        dates_map[k] = dates_map[prefix]
                    else:
                        p = parse_date_to_decimal(k)
                        if p is None or np.isnan(p):
                            p = extract_date_from_string(k)
                        if p is not None and not np.isnan(p):
                            dates_map[k] = p
        else:
            for k in stream_dict:
                p = parse_date_to_decimal(k)
                if p is None or np.isnan(p):
                    p = extract_date_from_string(k)
                if p is not None and not np.isnan(p):
                    dates_map[k] = p

        pre_failed = {}
        if align_minimap2:
            print(f"[*] ChronAeon Sieve: Aligning {len(stream_dict)} raw sequences to root reference using minimap2...")
            t0_mm2 = time.time()
            aligned_map, pre_failed = self.align_queries_minimap2(stream_dict)
            elapsed_mm2 = time.time() - t0_mm2
            rate_mm2 = len(stream_dict) / max(0.001, elapsed_mm2)
            print(f"[✓] Minimap2 Alignment Complete in {elapsed_mm2*1000:.1f}ms ({rate_mm2:.1f} seq/s): {len(aligned_map)} aligned, {len(pre_failed)} unmapped/contaminants.")
            stream_dict = aligned_map

        records = []
        n_tot = len(stream_dict) + len(pre_failed)
        print(f"[*] ChronAeon Sieve: Screening {n_tot} streaming sequences against clock manifold...")

        for q_id, sus_reason in pre_failed.items():
            r_date = dates_map.get(q_id, np.nan)
            records.append({
                'query_id': q_id,
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
                'elapsed_ms': 0.1
            })

        for i, (q_id, q_seq) in enumerate(stream_dict.items()):
            r_date = dates_map.get(q_id, np.nan)
            rec = self.screen_sequence(q_id, q_seq, r_date)
            records.append(rec)
            if (i + 1) % 500 == 0 or (i + 1) == len(stream_dict):
                rate = (i + 1) / max(0.001, time.time() - t_start)
                print(f"    Processed {i + 1}/{len(stream_dict)} ({rate:.1f} seq/s)...")

        df_res = pd.DataFrame(records)
        elapsed = time.time() - t_start
        rate_tot = n_tot / max(0.001, elapsed)

        n_pass = int(np.sum(df_res['status'] == 'PASS'))
        n_sus = int(np.sum(df_res['status'] == 'SUS'))
        print(f"[✓] Sieve Complete in {elapsed:.2f}s ({rate_tot:.1f} seq/s): PASS={n_pass} ({n_pass/max(1,n_tot)*100:.1f}%), SUS={n_sus} ({n_sus/max(1,n_tot)*100:.1f}%).")

        # Export segregated FASTAs if requested
        if clean_fasta_out:
            clean_ids = set(df_res[df_res['status'] == 'PASS']['query_id'])
            with open(clean_fasta_out, "w") as f:
                for qid in clean_ids:
                    s = stream_dict.get(qid, raw_dict.get(qid, ""))
                    f.write(f">{qid}\n{s}\n")
            print(f"[✓] Clean analysis-ready FASTA written to: {clean_fasta_out}")

        if sus_fasta_out:
            sus_ids = set(df_res[df_res['status'] != 'PASS']['query_id'])
            with open(sus_fasta_out, "w") as f:
                for qid in sus_ids:
                    s = stream_dict.get(qid, raw_dict.get(qid, ""))
                    f.write(f">{qid}\n{s}\n")
            print(f"[✓] Quarantined FASTA written to: {sus_fasta_out}")

        return df_res
