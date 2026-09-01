#!/usr/bin/env python3
"""
scripts/run_avian_genome_filter.py
----------------------------------
Runs full-scale end-to-end alignment error detection, surgical masking, and 
neural re-evaluation across all 8,699 avian gene families from Shultz & Sackton (2019).

Outputs:
  - benchmark/avian_shultz2019/avian_genome_wide_filter_results.csv
  - benchmark/avian_shultz2019/avian_filtered_artifacts_audit.csv
"""

import os
import sys
import time
import zipfile
import gzip
import tempfile
import numpy as np
import pandas as pd
import torch
from scipy.stats import hypergeom, chi2, cauchy
from Bio import Phylo

sys.path.insert(0, 'axomeme_repo')
from axomeme.weights import load_arch_config, load_weights
from axomeme.model import PhyloAxialTransformer
from axomeme.dataset import load_alignment_and_tree, CODON_TO_AA, parse_alignment_sequences

def calc_pvals(lrts):
    pvals = np.full(len(lrts), 2.0 / 3.0, dtype=np.float32)
    pos = lrts > 0.0
    if np.any(pos):
        pvals[pos] = (2.0 / 3.0) * (0.45 * chi2.sf(lrts[pos], df=1) + 0.55 * chi2.sf(lrts[pos], df=2))
    return pvals

def calc_fdr_qvals(pvals):
    n = len(pvals)
    if n == 0: return np.array([])
    sorted_idx = np.argsort(pvals)
    sorted_p = pvals[sorted_idx]
    qvals = np.zeros(n, dtype=np.float32)
    min_q = 1.0
    for i in range(n - 1, -1, -1):
        q = sorted_p[i] * n / (i + 1)
        if q < min_q: min_q = q
        qvals[i] = min_q
    res = np.zeros(n, dtype=np.float32)
    res[sorted_idx] = qvals
    return np.clip(res, 0.0, 1.0)

def calc_cauchy_p(pvals):
    p_clipped = np.clip(pvals, 1e-15, 1.0 - 1e-15)
    t = np.mean(np.tan((0.5 - p_clipped) * np.pi))
    p_cct = 0.5 - (np.arctan(t) / np.pi)
    return float(np.clip(p_cct, 1e-15, 1.0))

def scan_hypergeometric_patches(site_pvals, alpha_site=0.05, min_k=3, max_span=35):
    L = len(site_pvals)
    sel_sites = np.where(site_pvals <= alpha_site)[0]
    K = len(sel_sites)
    if K < min_k: return []
    candidates = []
    for idx_i in range(len(sel_sites)):
        pos_i = sel_sites[idx_i]
        for idx_j in range(idx_i + min_k - 1, len(sel_sites)):
            pos_j = sel_sites[idx_j]
            d = pos_j - pos_i + 1
            if d > max_span: break
            k = idx_j - idx_i + 1
            p_local = 1.0 - hypergeom.cdf(k - 1, L, K, d)
            if p_local <= 0.01:
                candidates.append({'start': pos_i, 'end': pos_j, 'k': k, 'd': d, 'p_local': p_local})
    if not candidates: return []
    candidates.sort(key=lambda x: x['start'])
    merged, curr = [], candidates[0].copy()
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

def main():
    print("======================================================================")
    print("🚀 Starting Dataset-Wide Avian Genome Filter (8,699 Orthologs)")
    print("======================================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"[*] Compute Device: {device.type.upper()}")
    
    weights_path = 'axomeme_5_dim384_nonull.pt'
    config = load_arch_config(weights=weights_path)
    model = PhyloAxialTransformer(
        embed_dim=config['embed_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        window_size=config['window_size']
    ).to(device)
    state_dict = load_weights(weights=weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    tree_path = 'benchmark/avian_shultz2019/avian_39_species_tree_clean.nwk'
    tree = Phylo.read(tree_path, 'newick')
    tree_terms = set(c.name for c in tree.get_terminals() if c.name)
    print(f"[*] Master Phylogenetic Tree: {len(tree_terms)} species")
    
    meta_df = pd.read_csv('benchmark/avian_shultz2019/avian_genome_wide_hyphaeon_merged_shultz2019.csv')
    print(f"[*] Loaded metadata for {len(meta_df)} avian genes")
    
    zip_path = 'benchmark/avian_shultz2019/comparative_genomics_hog_alignments.zip'
    zf = zipfile.ZipFile(zip_path, 'r')
    zip_map = {os.path.basename(n).replace('.phy.gz', '').replace('.phy', ''): n for n in zf.namelist() if n.endswith('.phy.gz') or n.endswith('.phy')}
    
    out_results_csv = 'benchmark/avian_shultz2019/avian_genome_wide_filter_results.csv'
    out_audit_csv = 'benchmark/avian_shultz2019/avian_filtered_artifacts_audit.csv'
    
    results = []
    artifacts = []
    
    t_start_global = time.time()
    total_codons_evaluated = 0
    total_artifacts_masked = 0
    total_genes_with_artifacts = 0
    
    for idx, row in meta_df.iterrows():
        hog_id = str(row['hog'])
        if hog_id not in zip_map:
            continue
            
        zip_fname = zip_map[hog_id]
        with zf.open(zip_fname) as f:
            gz = gzip.GzipFile(fileobj=f)
            content = gz.read().decode('utf-8')
            
        # Parse phylip lines and standardize headers
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        if len(lines) < 2: continue
        
        parsed_seqs = {}
        for l in lines[1:]:
            parts = l.split()
            if len(parts) >= 2:
                raw_h = parts[0]
                sp_code = raw_h.split('_')[0] if '_' in raw_h else raw_h
                seq_str = ''.join(parts[1:]).upper()
                if sp_code in tree_terms:
                    parsed_seqs[sp_code] = seq_str
                    
        if len(parsed_seqs) < 5: continue
        L_check = len(next(iter(parsed_seqs.values()))) // 3
        if L_check < 20: continue
        
        # Write standardized FASTA
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp_f:
            for sp, seq_str in parsed_seqs.items():
                tmp_f.write(f">{sp}\n{seq_str}\n")
            tmp_fasta = tmp_f.name
            
        try:
            c, a, d_mat, z, inv, taxa, L = load_alignment_and_tree(tmp_fasta, tree_path, max_species=64, prune_duplicates=True)
            if L < 20 or len(taxa) < 5:
                os.remove(tmp_fasta)
                continue
                
            total_codons_evaluated += L
            tree_cache = model.precompute_tree_cache(d_mat.to(device), z.to(device))
            
            # 1. Raw Baseline Inference (Batched for Memory Safety)
            var_idx = np.where(~inv)[0]
            n_var = len(var_idx)
            b_size = 64
            lrts_raw = np.zeros(L, dtype=np.float32)
            with torch.no_grad():
                for b_start in range(0, n_var, b_size):
                    b_end = min(b_start + b_size, n_var)
                    b_sub = var_idx[b_start:b_end]
                    c_ch = c[b_sub].to(device)
                    a_ch = a[b_sub].to(device)
                    y_soft, _ = model.forward_cached(c_ch, a_ch, tree_cache)
                    lrts_raw[b_sub] = torch.clamp(y_soft.squeeze(-1), min=0.0).cpu().numpy().flatten()
                    del c_ch, a_ch, y_soft
                
            pvals_raw = calc_pvals(lrts_raw)
            qvals_raw = calc_fdr_qvals(pvals_raw)
            cct_p_raw = calc_cauchy_p(pvals_raw)
            sig_p05_raw = int(np.sum(pvals_raw <= 0.05))
            sig_p01_raw = int(np.sum(pvals_raw <= 0.01))
            sig_q10_raw = int(np.sum(qvals_raw <= 0.10))
            density_p05_raw = sig_p05_raw / L
            
            # 2. Dual-Stage Error Filtering
            patches = scan_hypergeometric_patches(pvals_raw)
            raw_seqs = parse_alignment_sequences(tmp_fasta)
            cleaned_seqs = {t: list(raw_seqs[t]) for t in taxa if t in raw_seqs}
            
            # Consensus sequence
            consensus_codons = []
            for s in range(L):
                scds = [raw_seqs[t][s*3:(s+1)*3].upper() for t in taxa if t in raw_seqs and len(raw_seqs[t]) >= (s+1)*3]
                valid = [cd for cd in scds if '-' not in cd and 'N' not in cd and len(cd)==3]
                consensus_codons.append(pd.Series(valid).mode().iloc[0] if valid else 'NNN')
                
            gene_artifacts = 0
            for p in patches:
                p_start, p_end = p['start'], p['end']
                tax_max_run, tax_total_muts, total_patch_muts = {}, {}, 0
                for t in taxa:
                    if t not in raw_seqs: continue
                    run, max_r, m_cnt = 0, 0, 0
                    for s in range(p_start, p_end + 1):
                        obs_cd = raw_seqs[t][s*3:(s+1)*3].upper()
                        con_cd = consensus_codons[s]
                        obs_aa = CODON_TO_AA.get(obs_cd, '-')
                        con_aa = CODON_TO_AA.get(con_cd, '-')
                        if obs_aa not in '-?' and con_aa not in '-?' and obs_aa != con_aa:
                            m_cnt += 1; run += 1; max_r = max(max_r, run)
                        else: run = 0
                    tax_max_run[t] = max_r; tax_total_muts[t] = m_cnt; total_patch_muts += m_cnt
                    
                top_tax = max(taxa, key=lambda t: (tax_max_run.get(t, 0), tax_total_muts.get(t, 0)))
                top_run = tax_max_run.get(top_tax, 0)
                top_muts = tax_total_muts.get(top_tax, 0)
                oci = top_muts / (total_patch_muts + 1e-8)
                
                is_artifact = (top_run >= 3 and oci >= 0.25) or (top_run >= 4)
                if is_artifact:
                    gene_artifacts += 1
                    total_artifacts_masked += 1
                    # Record audit
                    obs_patch_str = "".join([CODON_TO_AA.get(raw_seqs[top_tax][s*3:(s+1)*3].upper(), '-') for s in range(p_start, p_end+1)])
                    con_patch_str = "".join([CODON_TO_AA.get(consensus_codons[s], '-') for s in range(p_start, p_end+1)])
                    artifacts.append({
                        'hog': row['hog'],
                        'entrezgene': row.get('entrezgene', np.nan),
                        'patch_start_1idx': p_start + 1,
                        'patch_end_1idx': p_end + 1,
                        'span_d': p['d'],
                        'k_sites': p['k'],
                        'p_local': p['p_local'],
                        'outlier_taxon': top_tax,
                        'consecutive_mismatches': top_run,
                        'oci': oci,
                        'outlier_aa_seq': obs_patch_str,
                        'consensus_aa_seq': con_patch_str,
                        'mean_patch_lrt_raw': float(np.mean(lrts_raw[p_start:p_end+1]))
                    })
                    # Mask outlier
                    for s in range(p_start, p_end + 1):
                        cleaned_seqs[top_tax][s*3 : (s+1)*3] = ['N', 'N', 'N']
                        
            # 3. Post-Cleaning Re-evaluation
            if gene_artifacts > 0:
                total_genes_with_artifacts += 1
                with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp_clean_f:
                    for t in taxa:
                        if t in cleaned_seqs:
                            tmp_clean_f.write(f">{t}\n{''.join(cleaned_seqs[t])}\n")
                    tmp_clean_fasta = tmp_clean_f.name
                    
                c_cl, a_cl, d_cl, z_cl, inv_cl, taxa_cl, L_cl = load_alignment_and_tree(tmp_clean_fasta, tree_path, max_species=64, prune_duplicates=True)
                tree_cache_cl = model.precompute_tree_cache(d_cl.to(device), z_cl.to(device))
                var_idx_cl = np.where(~inv_cl)[0]
                n_var_cl = len(var_idx_cl)
                lrts_clean = np.zeros(L_cl, dtype=np.float32)
                with torch.no_grad():
                    for b_start in range(0, n_var_cl, b_size):
                        b_end = min(b_start + b_size, n_var_cl)
                        b_sub = var_idx_cl[b_start:b_end]
                        c_ch = c_cl[b_sub].to(device)
                        a_ch = a_cl[b_sub].to(device)
                        y_soft_cl, _ = model.forward_cached(c_ch, a_ch, tree_cache_cl)
                        lrts_clean[b_sub] = torch.clamp(y_soft_cl.squeeze(-1), min=0.0).cpu().numpy().flatten()
                        del c_ch, a_ch, y_soft_cl
                pvals_clean = calc_pvals(lrts_clean)
                qvals_clean = calc_fdr_qvals(pvals_clean)
                cct_p_clean = calc_cauchy_p(pvals_clean)
                sig_p05_clean = int(np.sum(pvals_clean <= 0.05))
                sig_p01_clean = int(np.sum(pvals_clean <= 0.01))
                sig_q10_clean = int(np.sum(qvals_clean <= 0.10))
                density_p05_clean = sig_p05_clean / L
                mean_lrt_clean = float(np.mean(lrts_clean))
                max_lrt_clean = float(np.max(lrts_clean))
                os.remove(tmp_clean_fasta)
                del tree_cache_cl, d_cl, z_cl, c_cl, a_cl
            else:
                pvals_clean = pvals_raw
                qvals_clean = qvals_raw
                cct_p_clean = cct_p_raw
                sig_p05_clean = sig_p05_raw
                sig_p01_clean = sig_p01_raw
                sig_q10_clean = sig_q10_raw
                density_p05_clean = density_p05_raw
                mean_lrt_clean = float(np.mean(lrts_raw))
                max_lrt_clean = float(np.max(lrts_raw))

            del tree_cache, d_mat, z, c, a
            if device.type == 'mps':
                torch.mps.empty_cache()
            elif device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
                
            results.append({
                'hog': row['hog'],
                'entrezgene': row.get('entrezgene', np.nan),
                'taxa': len(taxa),
                'codons': L,
                'patches_detected': len(patches),
                'artifacts_masked': gene_artifacts,
                # Raw metrics
                'cct_pval_raw': cct_p_raw,
                'neglog10_cct_raw': -np.log10(cct_p_raw),
                'sig_p05_raw': sig_p05_raw,
                'sig_p01_raw': sig_p01_raw,
                'sig_q10_raw': sig_q10_raw,
                'density_p05_raw': density_p05_raw,
                'mean_lrt_raw': float(np.mean(lrts_raw)),
                'max_lrt_raw': float(np.max(lrts_raw)),
                # Cleaned metrics
                'cct_pval_clean': cct_p_clean,
                'neglog10_cct_clean': -np.log10(cct_p_clean),
                'sig_p05_clean': sig_p05_clean,
                'sig_p01_clean': sig_p01_clean,
                'sig_q10_clean': sig_q10_clean,
                'density_p05_clean': density_p05_clean,
                'mean_lrt_clean': mean_lrt_clean,
                'max_lrt_clean': max_lrt_clean,
                # Delta metrics
                'delta_sig_p05': sig_p05_raw - sig_p05_clean,
                'delta_cct_neglog': -np.log10(cct_p_raw) - (-np.log10(cct_p_clean)),
                # Original benchmark metrics for comparison
                'omega_m0': row.get('Omega_m0', np.nan),
                'pval_busted': row.get('pval_busted', np.nan),
                'fdr_busted': row.get('FDRPval_busted', np.nan),
                'pval_m8': row.get('PVal_m8m8a', np.nan),
                'fdr_m8': row.get('FDRPval_m8m8a', np.nan),
            })
            os.remove(tmp_fasta)
            
        except Exception as e:
            if os.path.exists(tmp_fasta): os.remove(tmp_fasta)
            print(f"[!] Error on HOG {hog_id}: {e}")
            
        if (idx + 1) % 250 == 0 or (idx + 1) == len(meta_df):
            elapsed = time.time() - t_start_global
            throughput = total_codons_evaluated / elapsed
            print(f"[{idx+1:4d}/{len(meta_df)}] Processed {total_codons_evaluated:,} codons ({throughput:.1f} cod/s) | Artifacts masked: {total_artifacts_masked} across {total_genes_with_artifacts} genes | Elapsed: {elapsed/60:.2f} min")
            # Intermediate checkpoint save
            pd.DataFrame(results).to_csv(out_results_csv, index=False)
            pd.DataFrame(artifacts).to_csv(out_audit_csv, index=False)
            
    # Final save
    df_final = pd.DataFrame(results)
    df_audit = pd.DataFrame(artifacts)
    df_final.to_csv(out_results_csv, index=False)
    df_audit.to_csv(out_audit_csv, index=False)
    
    total_time = time.time() - t_start_global
    print("\n" + "=" * 76)
    print(f"🎉 Complete Avian Genome Filter Finished in {total_time/60:.2f} minutes ({total_time:.2f} s)!")
    print(f"   • Total Evaluated Genes: {len(df_final):,} / {len(meta_df):,}")
    print(f"   • Total Codons Evaluated: {total_codons_evaluated:,}")
    print(f"   • Total Artifact Patches Surgically Masked: {len(df_audit):,}")
    print(f"   • Genes with >= 1 Masked Artifact: {total_genes_with_artifacts:,} ({total_genes_with_artifacts/len(df_final)*100:.2f}%)")
    print(f"   • Results CSV: {out_results_csv}")
    print(f"   • Audit CSV: {out_audit_csv}")
    print("=" * 76)

if __name__ == '__main__':
    main()
