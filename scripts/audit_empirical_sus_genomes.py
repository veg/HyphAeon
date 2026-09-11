#!/usr/bin/env python3
"""
scripts/audit_empirical_sus_genomes.py
--------------------------------------
Audits all empirical viral benchmark datasets to identify temporal outliers
("sus" genomes) and diagnose zero-signal regimes using Studentized residuals
and ChronAeon Sieve anomaly classification.
"""

import os
import glob
import json
import argparse
from typing import Dict, List, Any, Optional

import numpy as np
import scipy.stats as stats
import pandas as pd
import scipy.linalg as la
from Bio import SeqIO


def audit_benchmarks(benchmark_dir: str, output_csv: Optional[str] = None, z_threshold: float = 3.0):
    concordance_path = os.path.join(benchmark_dir, 'compiled_empirical_concordance.json')
    concordance_map = {}
    if os.path.exists(concordance_path):
        with open(concordance_path) as f:
            concordance = json.load(f)
            concordance_map = {s['id']: s for s in concordance}

    subdirs = sorted([d for d in os.listdir(benchmark_dir) if os.path.isdir(os.path.join(benchmark_dir, d)) and d[:2].isdigit()])

    outlier_records = []
    study_summaries = []

    for d in subdirs:
        sid = d[:2]
        c_entry = concordance_map.get(sid, {})
        p = os.path.join(benchmark_dir, d)
        times, dists, taxa = None, None, None

        # 1. Load from JSON
        jsons = glob.glob(os.path.join(p, '*dating*.json')) + glob.glob(os.path.join(p, 'dating_results.json')) + glob.glob(os.path.join(p, 'results*.json'))
        for jf in jsons:
            try:
                with open(jf) as f:
                    jd = json.load(f)
                    if 'taxa_records' in jd:
                        times = np.array([r['sampling_date'] for r in jd['taxa_records']])
                        dists = np.array([r['root_divergence'] for r in jd['taxa_records']])
                        taxa = [r['taxon'] for r in jd['taxa_records']]
                        break
                    elif 'times' in jd and 'dists' in jd and 'taxa' in jd:
                        times = np.array(jd['times'])
                        dists = np.array(jd['dists'])
                        taxa = list(jd['taxa'])
                        break
            except Exception:
                pass

        # 2. Load from CSV
        if times is None or len(times) == 0:
            for cf in sorted(glob.glob(os.path.join(p, '*taxa*.csv')) + glob.glob(os.path.join(p, '*.csv'))):
                try:
                    df = pd.read_csv(cf)
                    if 'sampling_date' in df.columns and 'root_divergence' in df.columns:
                        val = ~df['sampling_date'].isna() & ~df['root_divergence'].isna()
                        if val.sum() > 3:
                            times = df['sampling_date'].values[val]
                            dists = df['root_divergence'].values[val]
                            taxa_col = 'taxon' if 'taxon' in df.columns else ('strain' if 'strain' in df.columns else df.columns[0])
                            taxa = df[taxa_col].astype(str).values[val]
                            break
                except Exception:
                    pass

        # 3. Load FASTA if available
        seqs = {}
        fasta_files = glob.glob(os.path.join(p, '*.fasta')) + glob.glob(os.path.join(p, '*.fas'))
        if fasta_files:
            try:
                for rec in SeqIO.parse(fasta_files[0], 'fasta'):
                    seqs[rec.id] = str(rec.seq).upper()
            except Exception:
                pass

        if times is None or len(times) < 5:
            study_summaries.append({
                'id': sid, 'name': c_entry.get('label', d), 'N': 0 if times is None else len(times),
                'status': 'NO_DATA_OR_DATES', 'R2': np.nan, 'Fieller_g': np.nan, 'mu': np.nan, 'n_sus': 0
            })
            continue

        N = len(times)
        t_ref = np.mean(times)
        x = times - t_ref
        y = dists
        X = np.column_stack([np.ones(N), x])

        try:
            XtX_inv = la.inv(X.T @ X)
        except Exception:
            study_summaries.append({
                'id': sid, 'name': c_entry.get('label', d), 'N': N,
                'status': 'SINGULAR_DESIGN', 'R2': np.nan, 'Fieller_g': np.nan, 'mu': np.nan, 'n_sus': 0
            })
            continue

        beta = XtX_inv @ (X.T @ y)
        d0, mu = beta[0], beta[1]
        e_d = y - X @ beta
        s2 = np.sum(e_d**2) / max(1, N - 2)
        se_d = np.sqrt(s2)

        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1.0 - (np.sum(e_d**2) / ss_tot) if ss_tot > 0 else 0.0

        t_crit = stats.t.ppf(0.975, df=max(1, N - 2))
        var_mu = s2 * XtX_inv[1, 1]
        g = (t_crit**2 * var_mu) / (mu**2) if mu != 0 else np.inf

        if g >= 1.0 or mu <= 0 or r2 < 0.05:
            status = 'NO_TEMPORAL_RESOLUTION'
        else:
            status = 'TEMPORALLY_RESOLVED'

        H = X @ XtX_inv @ X.T
        h_diag = np.diag(H)

        n_sus = 0
        for i in range(N):
            h_ii = min(0.9999, h_diag[i])
            stud_res = e_d[i] / (se_d * np.sqrt(1.0 - h_ii)) if se_d > 0 else 0.0

            b_mi = beta - (XtX_inv @ X[i]) * (e_d[i] / (1.0 - h_ii))
            d0_i, mu_i = b_mi[0], b_mi[1]
            if abs(mu_i) > 1e-12:
                pred_t_loo = t_ref + (y[i] - d0_i) / mu_i
                err_t_loo = pred_t_loo - times[i]
            else:
                err_t_loo = 0.0

            if abs(stud_res) >= z_threshold:
                n_sus += 1
                taxon_id = taxa[i] if taxa is not None and i < len(taxa) else f'tip_{i}'
                seq_match = seqs.get(taxon_id, '')
                gaps = seq_match.count('-') if seq_match else -1
                n_ambig = seq_match.count('N') if seq_match else -1
                seq_len = len(seq_match) if seq_match else 1
                ambig_ratio = n_ambig / max(1, seq_len) if seq_match else 0.0

                if stud_res > 0:
                    if n_ambig > 10 or ambig_ratio > 0.02 or gaps > 10:
                        anomaly_type = f'SUS_LOW_QUALITY (Missing Data / {n_ambig} Ns)'
                    else:
                        anomaly_type = 'SUS_HYPERMUTATED (Biological Excess Divergence / Clean)'
                else:
                    anomaly_type = 'SUS_LAGGING_OR_FROZEN (Under-Diverged / Archival / Misdated)'

                outlier_records.append({
                    'study_id': sid,
                    'study_name': c_entry.get('label', d),
                    'taxon': taxon_id,
                    'date': times[i],
                    'divergence': y[i],
                    'z_score': stud_res,
                    'err_years_loo': err_t_loo,
                    'anomaly_type': anomaly_type,
                    'gaps': gaps,
                    'ambig_N': n_ambig,
                    'ambig_pct': ambig_ratio * 100.0,
                    'study_status': status
                })

        study_summaries.append({
            'id': sid,
            'name': c_entry.get('label', d),
            'N': N,
            'status': status,
            'R2': r2,
            'Fieller_g': g,
            'mu': mu,
            'n_sus': n_sus
        })

    sum_df = pd.DataFrame(study_summaries)
    out_df = pd.DataFrame(outlier_records)

    print('=== STUDY-LEVEL SUMMARY ===')
    print(sum_df.to_string(index=False))
    print(f'\nTotal SUS genomes across all studies (|z| >= {z_threshold}): {len(out_df)}')

    if output_csv:
        out_df.to_csv(output_csv, index=False)
        print(f'Wrote outlier records to {output_csv}')

    return sum_df, out_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audit empirical benchmarks for anomalous (sus) genomes.')
    parser.add_argument('--benchmarks-dir', default='/Users/sergei/Projects/TOGA_MEME/PheWAS/verified_empirical_beast_benchmarks', help='Path to benchmarks directory')
    parser.add_argument('--output-csv', default='empirical_sus_genomes.csv', help='Path to output CSV')
    parser.add_argument('--threshold', type=float, default=3.0, help='Z-score threshold for sus flag')
    args = parser.parse_args()
    audit_benchmarks(args.benchmarks_dir, args.output_csv, args.threshold)
