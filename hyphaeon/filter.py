"""
hyphaeon/filter.py
------------------
Automated Dataset-Wide Alignment Quality Control, Spatial Error Detection,
Outlier Contamination Index (OCI) Attribution, and Surgical Masking.

Hardware-Aware Memory Management & OOM Prevention:
- Dynamically scales sitewise batch sizes via compute_adaptive_safe_batch_size.
- Slices tensor batches on-the-fly onto GPU/MPS accelerators, retaining host alignment on CPU.
- Invokes explicit cache flushes and garbage collection across all inference boundaries.
"""

import os
import sys
import gc
import time
import tempfile
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

import torch
from .stats import pvals_from_lrt_meme as calc_asymptotic_pvals, benjamini_hochberg as calc_fdr_qvals, cauchy_combination_p as calc_cauchy_omnibus_p

from .model import PhyloAxialTransformer
from .dataset import (
    load_alignment_and_tree,
    parse_alignment_sequences,
    CODON_TO_AA,
    GENETIC_CODE,
)
from .weights import resolve_weights_path, DEFAULT_VARIANT
from .inference import get_device, load_model
from .epistasis import compute_adaptive_safe_batch_size

def cleanup_device_memory(device: torch.device):
    """Flushes accelerator cache and invokes host garbage collection."""
    gc.collect()
    if device.type == 'mps':
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    elif device.type == 'cuda':
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

def scan_hypergeometric_patches(
    site_pvals: np.ndarray,
    alpha_site: float = 0.05,
    min_k: int = 3,
    max_span: int = 35,
    p_local_thresh: float = 0.01
) -> List[Dict[str, Any]]:
    """
    Scans for spatial clustering of nominally significant sites using the exact
    upper-tail hypergeometric distribution.
    """
    L = len(site_pvals)
    sel_sites = np.where(site_pvals <= alpha_site)[0]
    K = len(sel_sites)
    if K < min_k:
        return []
    
    candidates = []
    for idx_i in range(len(sel_sites)):
        pos_i = sel_sites[idx_i]
        for idx_j in range(idx_i + min_k - 1, len(sel_sites)):
            pos_j = sel_sites[idx_j]
            d = pos_j - pos_i + 1
            if d > max_span:
                break
            k = idx_j - idx_i + 1
            p_local = 1.0 - hypergeom.cdf(k - 1, L, K, d)
            if p_local <= p_local_thresh:
                candidates.append({'start': int(pos_i), 'end': int(pos_j), 'k': int(k), 'd': int(d), 'p_local': float(p_local)})
                
    if not candidates:
        return []
        
    candidates.sort(key=lambda x: x['start'])
    merged = []
    curr = candidates[0].copy()
    for c in candidates[1:]:
        if c['start'] <= curr['end']:
            curr['end'] = max(curr['end'], c['end'])
            curr['p_local'] = min(curr['p_local'], c['p_local'])
        else:
            curr['d'] = curr['end'] - curr['start'] + 1
            curr['k'] = int(np.sum((sel_sites >= curr['start']) & (sel_sites <= curr['end'])))
            merged.append(curr)
            curr = c.copy()
    curr['d'] = curr['end'] - curr['start'] + 1
    curr['k'] = int(np.sum((sel_sites >= curr['start']) & (sel_sites <= curr['end'])))
    merged.append(curr)
    return merged

def run_alignment_filter(
    alignment_path: str,
    tree_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    model_variant: Optional[str] = None,
    output_alignment_path: Optional[str] = None,
    audit_csv_path: Optional[str] = None,
    alpha_site: float = 0.05,
    min_k: int = 3,
    max_span: int = 35,
    min_oci: float = 0.25,
    min_run_length: int = 3,
    batch_size: Optional[int] = None,
    max_species: Optional[int] = None,
    device: Optional[torch.device] = None,
    progress: bool = True,
) -> Dict[str, Any]:
    """
    Executes automated alignment quality control, artifact detection,
    surgical masking, and optional neural re-evaluation with strict memory safety.
    """
    if device is None:
        device = get_device()

    t_start = time.time()

    # 1. Load model
    model = load_model(weights=weights_path, variant=model_variant, device=device)
    
    # 2. Load Alignment and Tree
    c_tensor, a_tensor, d_mat, z_coords, inv_mask, taxa, L = load_alignment_and_tree(
        alignment_path, tree_path, max_species=max_species, prune_duplicates=True
    )
    raw_seqs = parse_alignment_sequences(alignment_path)
    N = len(taxa)
    
    # 3. Adaptive Batch Sizing for Memory Safety
    eff_batch_size = compute_adaptive_safe_batch_size(N, user_batch_size=batch_size, device=device)
    if N >= 600:
        eff_batch_size = min(eff_batch_size, 16)
    elif N >= 400:
        eff_batch_size = min(eff_batch_size, 32)
        
    var_indices = np.where(~inv_mask)[0]
    n_var = len(var_indices)
    
    if progress:
        print(f"[*] Step 1/4: Running Baseline Neural Inference across {L} codons ({n_var} variable, batch_size={eff_batch_size})...")
        sys.stdout.flush()
        
    # Baseline Inference
    tree_cache = model.precompute_tree_cache(d_mat.to(device), z_coords.to(device))
    lrts_raw = np.zeros(L, dtype=np.float32)
    t_base_start = time.time()
    
    with torch.no_grad():
        for b_start in range(0, n_var, eff_batch_size):
            b_end = min(b_start + eff_batch_size, n_var)
            batch_idx = var_indices[b_start:b_end]
            c_ch = c_tensor[batch_idx].to(device)
            a_ch = a_tensor[batch_idx].to(device)
            y_soft, _ = model.forward_cached(c_ch, a_ch, tree_cache)
            lrts_raw[batch_idx] = torch.clamp(y_soft.squeeze(-1), min=0.0).cpu().numpy().flatten()
            del c_ch, a_ch, y_soft
            
            if progress:
                n_done = b_end
                pct = (n_done / max(1, n_var)) * 100.0
                elapsed = time.time() - t_base_start
                rate = n_done / max(1e-3, elapsed)
                eta = (n_var - n_done) / max(1e-3, rate)
                sys.stdout.write(f"\r[Filter: Baseline] Codon {n_done:5d}/{n_var:5d} ({pct:5.1f}%) | {rate:6.1f} cod/s | Elapsed: {elapsed:5.1f}s | ETA: {eta:5.1f}s")
                sys.stdout.flush()
                
    if progress:
        sys.stdout.write("\n")
        sys.stdout.flush()
        
    # Free Step 1 accelerator cache
    del tree_cache, d_mat, z_coords
    cleanup_device_memory(device)
    
    pvals_raw = calc_asymptotic_pvals(lrts_raw)
    qvals_raw = calc_fdr_qvals(pvals_raw)
    cct_p_raw = calc_cauchy_omnibus_p(pvals_raw)
    sig_p05_raw = int(np.sum(pvals_raw <= 0.05))
    sig_q10_raw = int(np.sum(qvals_raw <= 0.10))
    
    # 4. Spatial Hypergeometric Clustering Scan
    if progress:
        print(f"[*] Step 2/4: Spatial Hypergeometric Cluster Scanning (d <= {max_span}, k >= {min_k})...")
        sys.stdout.flush()
        
    patches = scan_hypergeometric_patches(
        pvals_raw, alpha_site=alpha_site, min_k=min_k, max_span=max_span, p_local_thresh=0.01
    )
    
    if progress:
        print(f"    • Candidate Spatial Clusters Identified: {len(patches)}")
        sys.stdout.flush()
        
    # Compute consensus codon per site
    consensus_codons = []
    for s in range(L):
        scds = [raw_seqs[t][s*3:(s+1)*3].upper() for t in taxa if t in raw_seqs and len(raw_seqs[t]) >= (s+1)*3]
        valid = [cd for cd in scds if '-' not in cd and 'N' not in cd and '?' not in cd and len(cd) == 3]
        consensus_codons.append(pd.Series(valid).mode().iloc[0] if valid else 'NNN')
        
    cleaned_seqs = {t: list(raw_seqs[t]) for t in raw_seqs}
    artifacts_audit = []
    masked_codons_total = 0
    
    # 5. Outlier Contamination Index (OCI) Attribution
    if progress and patches:
        print(f"[*] Step 3/4: Evaluating Outlier Contamination Index (OCI) across {len(patches)} patches...")
        sys.stdout.flush()
    elif progress:
        print(f"[*] Step 3/4: OCI Attribution skipped (0 candidate patches).")
        sys.stdout.flush()
        
    for p_idx, p in enumerate(patches):
        p_start, p_end = p['start'], p['end']
        tax_max_run, tax_total_muts, total_patch_muts = {}, {}, 0
        for t in taxa:
            if t not in raw_seqs:
                continue
            run, max_r, m_cnt = 0, 0, 0
            for s in range(p_start, p_end + 1):
                obs_cd = raw_seqs[t][s*3:(s+1)*3].upper()
                con_cd = consensus_codons[s]
                obs_aa = CODON_TO_AA.get(obs_cd, '-')
                con_aa = CODON_TO_AA.get(con_cd, '-')
                if obs_aa not in '-?' and con_aa not in '-?' and obs_aa != con_aa:
                    m_cnt += 1
                    run += 1
                    max_r = max(max_r, run)
                else:
                    run = 0
            tax_max_run[t] = max_r
            tax_total_muts[t] = m_cnt
            total_patch_muts += m_cnt
            
        if not taxa:
            continue
            
        top_tax = max(taxa, key=lambda t: (tax_max_run.get(t, 0), tax_total_muts.get(t, 0)))
        top_run = tax_max_run.get(top_tax, 0)
        top_muts = tax_total_muts.get(top_tax, 0)
        oci = top_muts / (total_patch_muts + 1e-8)
        
        is_artifact = (top_run >= min_run_length and oci >= min_oci) or (top_run >= 4)
        if is_artifact:
            obs_patch_str = ''.join([CODON_TO_AA.get(raw_seqs[top_tax][s*3:(s+1)*3].upper(), '-') for s in range(p_start, p_end+1)])
            con_patch_str = ''.join([CODON_TO_AA.get(consensus_codons[s], '-') for s in range(p_start, p_end+1)])
            
            artifacts_audit.append({
                'patch_start_1idx': p_start + 1,
                'patch_end_1idx': p_end + 1,
                'span_codons': p['d'],
                'significant_sites_k': p['k'],
                'p_hypergeom': p['p_local'],
                'outlier_taxon': top_tax,
                'consecutive_mismatches': top_run,
                'outlier_contamination_index': oci,
                'outlier_aa_sequence': obs_patch_str,
                'consensus_aa_sequence': con_patch_str,
                'mean_raw_patch_lrt': float(np.mean(lrts_raw[p_start:p_end+1]))
            })
            
            # Mask the offending taxon in the patch with NNN
            for s in range(p_start, p_end + 1):
                cleaned_seqs[top_tax][s*3 : (s+1)*3] = ['N', 'N', 'N']
                masked_codons_total += 1
                
        if progress and patches:
            sys.stdout.write(f"\r[Filter: OCI Audit] Patch {p_idx+1:3d}/{len(patches):3d} | Top Outlier: {top_tax[:16]:16s} (Run={top_run:2d}, OCI={oci:4.2f}) | Artifacts: {len(artifacts_audit):3d}")
            sys.stdout.flush()
            
    if progress and patches:
        sys.stdout.write("\n")
        sys.stdout.flush()
        
    # 6. Cleaned Alignment Neural Re-evaluation
    num_artifacts = len(artifacts_audit)
    if num_artifacts > 0:
        if progress:
            print(f"[*] Step 4/4: Re-Evaluating Cleaned Alignment after Surgically Masking {masked_codons_total} codons...")
            sys.stdout.flush()
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp_clean_f:
            for t, seq_chars in cleaned_seqs.items():
                tmp_clean_f.write(f">{t}\n{''.join(seq_chars)}\n")
            tmp_clean_fasta = tmp_clean_f.name
            
        effective_tree = tree_path if tree_path is not None else alignment_path
        c_cl, a_cl, d_cl, z_cl, inv_cl, taxa_cl, L_cl = load_alignment_and_tree(
            tmp_clean_fasta, effective_tree, max_species=max_species, prune_duplicates=True
        )
        tree_cache_cl = model.precompute_tree_cache(d_cl.to(device), z_cl.to(device))
        
        var_indices_cl = np.where(~inv_cl)[0]
        n_var_cl = len(var_indices_cl)
        lrts_clean = np.zeros(L_cl, dtype=np.float32)
        t_clean_start = time.time()
        
        with torch.no_grad():
            for b_start in range(0, n_var_cl, eff_batch_size):
                b_end = min(b_start + eff_batch_size, n_var_cl)
                batch_idx = var_indices_cl[b_start:b_end]
                c_ch = c_cl[batch_idx].to(device)
                a_ch = a_cl[batch_idx].to(device)
                y_soft_cl, _ = model.forward_cached(c_ch, a_ch, tree_cache_cl)
                lrts_clean[batch_idx] = torch.clamp(y_soft_cl.squeeze(-1), min=0.0).cpu().numpy().flatten()
                del c_ch, a_ch, y_soft_cl
                
                if progress:
                    n_done = b_end
                    pct = (n_done / max(1, n_var_cl)) * 100.0
                    elapsed = time.time() - t_clean_start
                    rate = n_done / max(1e-3, elapsed)
                    eta = (n_var_cl - n_done) / max(1e-3, rate)
                    sys.stdout.write(f"\r[Filter: Re-eval]  Codon {n_done:5d}/{n_var_cl:5d} ({pct:5.1f}%) | {rate:6.1f} cod/s | Elapsed: {elapsed:5.1f}s | ETA: {eta:5.1f}s")
                    sys.stdout.flush()
                    
        if progress:
            sys.stdout.write("\n")
            sys.stdout.flush()
            
        del tree_cache_cl, d_cl, z_cl, c_cl, a_cl
        cleanup_device_memory(device)
        
        pvals_clean = calc_asymptotic_pvals(lrts_clean)
        qvals_clean = calc_fdr_qvals(pvals_clean)
        cct_p_clean = calc_cauchy_omnibus_p(pvals_clean)
        sig_p05_clean = int(np.sum(pvals_clean <= 0.05))
        sig_q10_clean = int(np.sum(qvals_clean <= 0.10))
        
        try:
            os.remove(tmp_clean_fasta)
        except Exception:
            pass
    else:
        if progress:
            print(f"[*] Step 4/4: No sequencing artifacts detected. Re-evaluation skipped.")
            sys.stdout.flush()
        lrts_clean = lrts_raw
        pvals_clean = pvals_raw
        qvals_clean = qvals_raw
        cct_p_clean = cct_p_raw
        sig_p05_clean = sig_p05_raw
        sig_q10_clean = sig_q10_raw
        
    elapsed_sec = time.time() - t_start
    
    # 7. Save Cleaned Alignment if requested
    if output_alignment_path:
        out_dir = os.path.dirname(os.path.abspath(output_alignment_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_alignment_path, 'w') as f_out:
            for t, seq_chars in cleaned_seqs.items():
                f_out.write(f">{t}\n{''.join(seq_chars)}\n")
                
    # 8. Save Audit CSV if requested
    if audit_csv_path and artifacts_audit:
        out_dir = os.path.dirname(os.path.abspath(audit_csv_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(artifacts_audit).to_csv(audit_csv_path, index=False)
        
    cleanup_device_memory(device)
    
    result = {
        'num_taxa': N,
        'num_codons': L,
        'num_patches_detected': len(patches),
        'num_artifacts_masked': num_artifacts,
        'masked_codons_count': masked_codons_total,
        'raw_metrics': {
            'cct_p_value': cct_p_raw,
            'sig_sites_p05': sig_p05_raw,
            'sig_sites_q10': sig_q10_raw,
            'mean_lrt': float(np.mean(lrts_raw))
        },
        'cleaned_metrics': {
            'cct_p_value': cct_p_clean,
            'sig_sites_p05': sig_p05_clean,
            'sig_sites_q10': sig_q10_clean,
            'mean_lrt': float(np.mean(lrts_clean))
        },
        'suppressed_spurious_sites': max(0, sig_p05_raw - sig_p05_clean),
        'artifacts': artifacts_audit,
        'elapsed_seconds': elapsed_sec,
        'output_alignment': output_alignment_path,
        'audit_csv': audit_csv_path
    }
    return result
