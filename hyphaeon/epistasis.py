"""
hyphaeon/epistasis.py
--------------------
Phylogenetic Epistatic Co-Selection Networks, Sector Mining, and
In Silico Selection Deep Mutational Scanning (ESSM / Digital DMS).

Strictly uses continuous Neural Transformer Attribution Vectors (multi-head
root-to-leaf axial attention weights combined with contextual selection drive)
with ZERO character mapping, ZERO ancestral parsimony heuristics, and ZERO
discrete substitution counting.
"""

import os
import sys
import math
import json
import time
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import networkx as nx
from Bio import Phylo

from .dataset import (
    AA_MAP,
    GENETIC_CODE,
    CODON_TO_AA,
    load_alignment_and_tree,
    get_codon_token,
    get_aa_token
)
from .model import PhyloAxialTransformer
from .weights import load_weights, load_arch_config
from .stats import pvals_from_lrt_self_liang, benjamini_hochberg
from .inference import get_device, load_model, get_device_memory_budget, compute_adaptive_safe_batch_size
from ._progress import ChunkProgress

REV_AA_MAP = {v: k for k, v in AA_MAP.items()}

# Canonical sense codons for all 20 standard amino acids
CANONICAL_AA_TO_CODON = {
    'A': 'GCC', 'C': 'TGC', 'D': 'GAC', 'E': 'GAG', 'F': 'TTC',
    'G': 'GGC', 'H': 'CAC', 'I': 'ATC', 'K': 'AAG', 'L': 'CTG',
    'M': 'ATG', 'N': 'AAC', 'P': 'CCC', 'Q': 'CAG', 'R': 'CGC',
    'S': 'AGC', 'T': 'ACC', 'V': 'GTG', 'W': 'TGG', 'Y': 'TAC'
}

def compute_transformer_attributions(
    model: PhyloAxialTransformer,
    c_tensor: torch.Tensor,
    a_tensor: torch.Tensor,
    tree_cache: Dict[str, Any],
    taxa: List[str],
    device: torch.device,
    batch_size: Optional[int] = None,
    progress: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Computes continuous Transformer Attribution Vectors directly from axial attention maps:
      a_{s, n} = alpha_{root->n, s} * delta_{s, n}
    where alpha is the multi-head root-to-leaf attention weight and delta is the non-consensus mutational indicator.
    Returns: (leaf_attributions [L, N], lrts [L], pvals [L], consensus_aas [L]).
    """
    L, n_taxa, _ = c_tensor.shape
    a_np = a_tensor.squeeze(-1).numpy() # [L, N]
    
    safe_batch_size = compute_adaptive_safe_batch_size(n_taxa, batch_size)
    
    # 1. Determine consensus amino acid per site (Vectorized)
    consensus_aas = []
    delta = np.zeros((L, n_taxa), dtype=np.float32)
    valid_mask = (a_np < 20)
    for s in range(L):
        valid_row = a_np[s, valid_mask[s]]
        if len(valid_row) > 0:
            major_aa = int(np.argmax(np.bincount(valid_row, minlength=20)))
            consensus_aas.append(REV_AA_MAP.get(major_aa, '-'))
            delta[s] = (valid_mask[s] & (a_np[s] != major_aa)).astype(np.float32)
        else:
            consensus_aas.append('-')
                    
    # 3. Batched neural inference with root attention extraction
    lrts = np.zeros(L, dtype=np.float32)
    mean_attns = np.zeros((L, n_taxa), dtype=np.float32)
    
    model.eval()
    pb = ChunkProgress(L, 'Attribution', 'codon', enabled=progress and L > 0)
    with torch.no_grad():
        for start_idx in range(0, L, safe_batch_size):
            end_idx = min(start_idx + safe_batch_size, L)
            c_chunk = c_tensor[start_idx:end_idx].to(device)
            a_chunk = a_tensor[start_idx:end_idx].to(device)

            y_soft, _, root_attns = model.forward_cached(
                c_chunk, a_chunk, tree_cache, return_attentions=True
            )
            lrts[start_idx:end_idx] = torch.clamp(y_soft, min=0.0).cpu().numpy().flatten()
            mean_attns[start_idx:end_idx] = root_attns.cpu().numpy()
            pb.update(end_idx)
    pb.finish()

    if device.type == 'mps':
        torch.mps.empty_cache()
            
    # 4. Asymptotic p-values (Self & Liang for attribution LRTs)
    pvals = pvals_from_lrt_self_liang(lrts).astype(np.float32)
    
    # 5. Continuous Leaf attributions: A = alpha * delta [L, N]
    leaf_attributions = mean_attns * delta
    
    return leaf_attributions, lrts, pvals, consensus_aas

# Backward-compatibility alias
compute_phylogenetic_branch_attributions = lambda model, c_tensor, a_tensor, tree_cache, tree_obj, taxa, device, batch_size=64: (
    compute_transformer_attributions(model, c_tensor, a_tensor, tree_cache, taxa, device, batch_size) + (taxa,)
)

def compute_branch_coselection_network(
    attributions: np.ndarray,
    lrts: np.ndarray,
    branch_names: List[str],
    consensus_aas: List[str],
    min_sim: float = 0.35,
    min_shared: int = 2,
    max_fdr: float = 0.05,
    min_lrt: float = 1.0,
    min_cesi: float = 2.0
) -> Tuple[List[Dict[str, Any]], nx.Graph]:
    """
    Computes pairwise co-selection directly from continuous Transformer Attribution Vectors.
    Calculates cosine attribution alignment, Student t-statistic significance,
    global Benjamini-Hochberg FDR across all M_total site pairs,
    and Composite Epistatic Selection Index (CESI).
    """
    L, N = attributions.shape
    df = max(1, N - 2)
    
    G = nx.Graph()
    for s in range(L):
        G.add_node(s + 1, ref=consensus_aas[s], lrt=float(lrts[s]))
        
    norms = np.linalg.norm(attributions, axis=1) # [L]
    active_mask = (norms > 0)
    valid_sites = np.where(active_mask)[0]
    V = len(valid_sites)
    if V < 2:
        return [], G
        
    M_total = V * (V - 1) // 2
    
    A_active = attributions[valid_sites]        # [V, N]
    A_bin = (A_active > 0).astype(np.float32)   # [V, N]
    norms_active = norms[valid_sites]           # [V]
    
    dot_matrix = A_active @ A_active.T          # [V, V]
    shared_matrix = A_bin @ A_bin.T             # [V, V]
    norm_outer = np.outer(norms_active, norms_active)
    sim_matrix = dot_matrix / np.maximum(norm_outer, 1e-9)
    
    tri_i, tri_j = np.triu_indices(V, k=1)
    shared_arr = shared_matrix[tri_i, tri_j]
    sim_arr = sim_matrix[tri_i, tri_j]
    s1_arr = valid_sites[tri_i]
    s2_arr = valid_sites[tri_j]
    
    # Continuous Student's t-test for attribution cosine alignment
    t_stat = sim_arr * np.sqrt(df / np.maximum(1e-9, 1.0 - sim_arr**2))
    p_vals = stats.t.sf(t_stat, df=df)
    cesi_vals = sim_arr * np.sqrt(np.maximum(lrts[s1_arr], 0.1) * np.maximum(lrts[s2_arr], 0.1))

    # Proper Global Benjamini-Hochberg FDR over all M_total hypothesis tests
    from .stats import benjamini_hochberg
    q_vals = benjamini_hochberg(p_vals).astype(np.float32)
    
    pass_filter = (
        (q_vals <= max_fdr) & 
        (shared_arr >= min_shared) & 
        (sim_arr >= min_sim) & 
        (cesi_vals >= min_cesi) &
        (np.maximum(lrts[s1_arr], lrts[s2_arr]) >= min_lrt)
    )
    pass_indices = np.where(pass_filter)[0]
    
    sig_pairs = []
    for idx in pass_indices:
        s1 = int(s1_arr[idx])
        s2 = int(s2_arr[idx])
        shared = int(shared_arr[idx])
        sim = float(sim_arr[idx])
        cesi = float(cesi_vals[idx])
        q_v = float(q_vals[idx])
        
        pair_dict = {
            "site_u": s1 + 1,
            "site_v": s2 + 1,
            "ref_u": consensus_aas[s1],
            "ref_v": consensus_aas[s2],
            "lrt_u": float(lrts[s1]),
            "lrt_v": float(lrts[s2]),
            "similarity": sim,
            "shared_taxa": shared,
            "shared_branches": shared,
            "p_val": float(p_vals[idx]),
            "hyper_p": float(p_vals[idx]),
            "fdr_q": q_v,
            "cesi": cesi
        }
        sig_pairs.append(pair_dict)
        G.add_edge(
            s1 + 1,
            s2 + 1,
            weight=sim,
            shared=shared,
            cesi=cesi,
            fdr_q=q_v
        )
        
    sig_pairs.sort(key=lambda x: x["cesi"], reverse=True)
    return sig_pairs, G

def compute_sector_permutation_test(
    attributions: np.ndarray,
    site_indices: List[int],
    observed_coherence: float,
    n_permutations: int = 10000,
    active_only: bool = True,
    rng_seed: Optional[int] = 42
) -> Dict[str, float]:
    """
    Evaluates statistical significance of an epistatic sector's spectral coherence
    against an empirical null distribution of random K-site subsets drawn from
    the same alignment attributions.
    
    Computes:
      p_perm = (1 / B) * sum( I( C(S_rand) >= C(S_obs) ) )
      null_mean = E[ C(S_rand) ]
      null_std  = Std[ C(S_rand) ]
      null_95   = 95th percentile of C(S_rand)
      isotropic_baseline = 1 / K
    """
    K = len(site_indices)
    L, N = attributions.shape
    
    if active_only:
        site_norms = np.linalg.norm(attributions, axis=1)
        candidate_pool = np.where(site_norms > 1e-9)[0]
    else:
        candidate_pool = np.arange(L)
        
    num_candidates = len(candidate_pool)
    if num_candidates < K or K < 2 or n_permutations <= 0:
        return {
            "p_perm": 1.0,
            "null_coherence_mean": float(observed_coherence),
            "null_coherence_std": 0.0,
            "null_coherence_95": float(observed_coherence),
            "isotropic_baseline": float(1.0 / K) if K > 0 else 1.0
        }
        
    rng = np.random.default_rng(rng_seed)
    batch_size = min(n_permutations, 25000)
    greater_equal_count = 0
    null_coherences = []
    
    while sum(len(x) for x in null_coherences) < n_permutations:
        cur_b = min(batch_size, n_permutations - sum(len(x) for x in null_coherences))
        if cur_b <= 0:
            break
            
        perm_indices = np.empty((cur_b, K), dtype=np.int64)
        for i in range(cur_b):
            perm_indices[i] = rng.choice(candidate_pool, size=K, replace=False)
            
        sub_A = attributions[perm_indices]  # [cur_b, K, N]
        if K > N:
            cov = np.einsum('bkn,bkm->bnm', sub_A, sub_A)  # [cur_b, N, N]
        else:
            cov = np.einsum('bkn,bln->bkl', sub_A, sub_A)  # [cur_b, K, K]
        traces = np.trace(cov, axis1=1, axis2=2)
        eigs = np.linalg.eigvalsh(cov)[:, -1]
        eigs = np.maximum(eigs, 0.0)
        
        valid_mask = traces > 1e-9
        batch_coherences = np.zeros(cur_b, dtype=np.float32)
        batch_coherences[valid_mask] = eigs[valid_mask] / traces[valid_mask]
        batch_coherences[~valid_mask] = 1.0 / K
        
        greater_equal_count += int(np.sum(batch_coherences >= (observed_coherence - 1e-7)))
        null_coherences.append(batch_coherences)
        
    all_null = np.concatenate(null_coherences)
    actual_b = len(all_null)
    p_val = float(greater_equal_count / actual_b) if actual_b > 0 else 1.0
    null_mean = float(np.mean(all_null))
    null_std = float(np.std(all_null))
    null_95 = float(np.percentile(all_null, 95))
    
    return {
        "p_perm": p_val,
        "null_coherence_mean": null_mean,
        "null_coherence_std": null_std,
        "null_coherence_95": null_95,
        "isotropic_baseline": float(1.0 / K)
    }

def extract_epistatic_sectors_tse(
    G: nx.Graph,
    attributions: np.ndarray,
    lrts: np.ndarray,
    consensus_aas: List[str],
    min_clique_size: int = 3,
    max_overlap: float = 0.50,
    min_coherence: float = 0.50,
    focal_taxon: Optional[str] = None,
    a_np: Optional[np.ndarray] = None,
    taxa: Optional[List[str]] = None,
    n_permutations: int = 10000,
    max_perm_p: Optional[float] = None,
    rng_seed: Optional[int] = 42
) -> List[Dict[str, Any]]:
    """
    Two-Stage Seed-and-Extend (TSE) epistatic sector mining using greedy modularity
    community decomposition, spectral coherence filtering (C(S) >= 0.50), and
    Monte Carlo random K-site permutation testing (p_perm) on the continuous
    attribution co-selection graph.
    """
    if G.number_of_edges() == 0:
        return []
        
    sub_nodes = [n for n, d in G.degree() if d > 0]
    if len(sub_nodes) < min_clique_size:
        components = [list(c) for c in nx.connected_components(G.subgraph(sub_nodes)) if len(c) >= 2]
    else:
        G_sub = G.subgraph(sub_nodes)
        try:
            communities = nx.algorithms.community.greedy_modularity_communities(G_sub, weight='weight')
            components = [list(c) for c in communities if len(c) >= 2]
        except Exception:
            components = [list(c) for c in nx.connected_components(G_sub) if len(c) >= 2]
            
    sectors = []
    sector_id_counter = 1
    for comp_idx, members in enumerate(components):
        site_indices = [n - 1 for n in sorted(members)]
        sub_A = attributions[site_indices, :]
        
        # Spectral coherence C(S) = lambda_1 / sum(lambda_k)
        if sub_A.shape[0] >= 2 and np.linalg.norm(sub_A) > 0:
            cov = sub_A @ sub_A.T
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 0.0)
            tot_eig = np.sum(eigvals)
            coherence = float(eigvals[-1] / tot_eig) if tot_eig > 0 else 0.0
            
            # Prune weakly-loaded peripheral nodes along primary eigenvector mode
            v_dom = np.abs(eigvecs[:, -1])
            retained_sites = [site_indices[i] for i in range(len(site_indices)) if v_dom[i] >= 0.10]
            if len(retained_sites) >= 2:
                sub_A2 = attributions[retained_sites, :]
                cov2 = sub_A2 @ sub_A2.T
                eigvals2 = np.maximum(np.linalg.eigvalsh(cov2), 0.0)
                tot2 = np.sum(eigvals2)
                coherence = float(eigvals2[-1] / tot2) if tot2 > 0 else 0.0
                site_indices = retained_sites
        else:
            coherence = 1.0
            
        # Filter out loose/noisy background clusters below empirical null baseline
        if coherence < min_coherence or len(site_indices) < 2:
            continue
            
        # Monte Carlo random K-site subset permutation test
        perm_stats = compute_sector_permutation_test(
            attributions=attributions,
            site_indices=site_indices,
            observed_coherence=coherence,
            n_permutations=n_permutations,
            active_only=True,
            rng_seed=rng_seed
        )
        p_perm = perm_stats["p_perm"]
        
        # Optional permutation significance filter
        if max_perm_p is not None and p_perm > max_perm_p:
            continue
            
        shared_taxa_cnt = int(np.sum(np.all(sub_A > 0, axis=0)))
        mean_lrt_val = float(np.mean(lrts[site_indices]))
        
        sorted_sites = sorted(site_indices)
        sig_tokens = [f"{consensus_aas[s]}{s+1}" for s in sorted_sites]
        pars_sig = f"[ {' - '.join(sig_tokens[:10])} ]"
        
        # Focal taxon cluster signature extraction
        focal_name = None
        focal_sig = None
        focal_diffs = []
        if focal_taxon and taxa is not None and a_np is not None:
            focal_idx = 0
            for t_i, t_name in enumerate(taxa):
                if focal_taxon.lower() in t_name.lower():
                    focal_idx = t_i
                    focal_name = t_name
                    break
            if focal_name:
                focal_tokens = []
                for s in sorted_sites:
                    c_aa = consensus_aas[s]
                    f_tok = a_np[s, focal_idx]
                    f_aa = REV_AA_MAP.get(f_tok, '-') if f_tok < 20 else c_aa
                    if f_aa != c_aa and f_aa != '-':
                        focal_tokens.append(f"{f_aa}{s+1}*")
                        focal_diffs.append(f"{c_aa}{s+1}->{f_aa}")
                    else:
                        focal_tokens.append(f"{f_aa}{s+1}")
                focal_sig = f"[ {' - '.join(focal_tokens[:10])} ]"
        
        sector_dict = {
            "sector_id": sector_id_counter,
            "size": len(sorted_sites),
            "sites": [s + 1 for s in sorted_sites],
            "spectral_coherence": coherence,
            "p_perm": p_perm,
            "null_coherence_mean": perm_stats["null_coherence_mean"],
            "null_coherence_std": perm_stats["null_coherence_std"],
            "null_coherence_95": perm_stats["null_coherence_95"],
            "isotropic_baseline": perm_stats["isotropic_baseline"],
            "shared_taxa": shared_taxa_cnt,
            "shared_branches": shared_taxa_cnt,
            "mean_lrt": mean_lrt_val,
            "pars_signature": pars_sig,
            "consensus_signature": pars_sig
        }
        if focal_name:
            sector_dict["focal_taxon"] = focal_name
            sector_dict["focal_signature"] = focal_sig
            sector_dict["focal_mutations"] = focal_diffs
            sector_dict["focal_mutations_count"] = len(focal_diffs)
            
        sectors.append(sector_dict)
        sector_id_counter += 1
        
    sectors.sort(key=lambda x: (x["spectral_coherence"], x["size"]), reverse=True)
    return sectors

def run_insilico_selection_dms(
    model: PhyloAxialTransformer,
    c_tensor: torch.Tensor,
    a_tensor: torch.Tensor,
    tree_cache: Dict[str, Any],
    taxa: List[str],
    device: torch.device,
    focal_taxon: Optional[str] = None,
    batch_size: Optional[int] = None,
    progress: bool = True,
    target_sites: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """
    Performs 19-amino-acid in silico Deep Mutational Scanning (Selection DMS / ESSM)
    by computing delta LRT across all possible point substitutions at a focal taxon.
    If target_sites is specified (0-indexed list of codon indices), only those focal sites are evaluated.
    Features real-time progress tracking, adaptive hardware safeguards, and high-speed tensor chunking.
    """
    L, n_taxa, _ = c_tensor.shape
    
    # Filter target sites
    if target_sites is not None:
        sites_to_eval = sorted(list(set([int(s) for s in target_sites if 0 <= s < L])))
    else:
        sites_to_eval = list(range(L))
        
    if not sites_to_eval:
        return []
    
    # 1. Automated Safe Batch Sizing Safeguard
    safe_batch_size = compute_adaptive_safe_batch_size(n_taxa, batch_size, device=device)
    sites_per_chunk = max(1, safe_batch_size // 19)
    
    focal_idx = 0
    focal_name = taxa[0] if taxa else "consensus"
    if focal_taxon:
        for idx, t in enumerate(taxa):
            if focal_taxon.lower() in t.lower():
                focal_idx = idx
                focal_name = t
                break
                
    a_np = a_tensor.squeeze(-1).numpy() # [L, N]
    model.eval()
    baseline_lrts = np.zeros(L, dtype=np.float32)
    
    # Baseline inference with safe chunking
    base_eval_chunk = max(1, min(64, safe_batch_size))
    with torch.no_grad():
        for start_idx in range(0, L, base_eval_chunk):
            end_idx = min(start_idx + base_eval_chunk, L)
            c_chunk = c_tensor[start_idx:end_idx].to(device)
            a_chunk = a_tensor[start_idx:end_idx].to(device)
            y_soft, _ = model.forward_cached(c_chunk, a_chunk, tree_cache)
            baseline_lrts[start_idx:end_idx] = torch.clamp(y_soft, min=0.0).cpu().numpy().flatten()
            
    if device.type == 'mps':
        torch.mps.empty_cache()
        
    standard_aas = list('ACDEFGHIKLMNPQRSTVWY')
    total_sites = len(sites_to_eval)
    total_mutants_est = 19 * total_sites
    
    if progress:
        site_scope_str = f"{total_sites} epistatic codons" if target_sites is not None else f"{L} codons"
        print(f"[*] Starting Selection Deep Mutational Scanning (ESSM / Digital DMS)...", flush=True)
        print(f"    Target Taxon: {focal_name} (index {focal_idx}) | Scope: {site_scope_str} | Total Mutants: {total_mutants_est} | Batch Size: {safe_batch_size} ({sites_per_chunk} sites/pass)", flush=True)
        
    plasticity_results = []
    t_start = time.time()
    last_update_time = t_start
    muts_processed = 0
    is_tty = sys.stdout.isatty()
    
    for chunk_idx in range(0, total_sites, sites_per_chunk):
        chunk_sites = sites_to_eval[chunk_idx : min(chunk_idx + sites_per_chunk, total_sites)]
        n_chunk_sites = len(chunk_sites)
        n_chunk_muts = 19 * n_chunk_sites
        
        # High-performance tensor repeat + in-place mutation (no python list/stack overhead)
        c_chunk_muts = c_tensor[chunk_sites].repeat_interleave(19, dim=0)
        a_chunk_muts = a_tensor[chunk_sites].repeat_interleave(19, dim=0)
        
        chunk_mut_aas = []
        for i, s in enumerate(chunk_sites):
            wt_tok = a_np[s, focal_idx]
            wt_aa = REV_AA_MAP.get(wt_tok, '-')
            if wt_tok >= 20:
                valid = a_np[s][a_np[s] < 20]
                wt_tok = int(np.argmax(np.bincount(valid))) if len(valid) > 0 else 0
                wt_aa = REV_AA_MAP.get(wt_tok, 'A')
                
            cand_mut_aas = [aa for aa in standard_aas if aa != wt_aa]
            chunk_mut_aas.append(cand_mut_aas)
            for m_i, m_aa in enumerate(cand_mut_aas):
                c_str = CANONICAL_AA_TO_CODON[m_aa]
                c_chunk_muts[i * 19 + m_i, focal_idx, 0] = GENETIC_CODE.get(c_str, 64)
                a_chunk_muts[i * 19 + m_i, focal_idx, 0] = AA_MAP.get(CODON_TO_AA.get(c_str, '-'), 20)
                
        with torch.no_grad():
            y_mut_soft, _ = model.forward_cached(c_chunk_muts.to(device), a_chunk_muts.to(device), tree_cache)
            mut_lrts = torch.clamp(y_mut_soft, min=0.0).cpu().numpy().flatten()
            
        for i, s in enumerate(chunk_sites):
            wt_tok = a_np[s, focal_idx]
            wt_aa = REV_AA_MAP.get(wt_tok, '-')
            if wt_tok >= 20:
                valid = a_np[s][a_np[s] < 20]
                wt_tok = int(np.argmax(np.bincount(valid))) if len(valid) > 0 else 0
                wt_aa = REV_AA_MAP.get(wt_tok, 'A')
                
            s_mut_lrts = mut_lrts[i * 19 : (i + 1) * 19]
            s_mut_aas = chunk_mut_aas[i]
            
            delta_lrts = s_mut_lrts - baseline_lrts[s]
            abs_deltas = np.abs(delta_lrts)
            plasticity = float(np.mean(abs_deltas)) if len(abs_deltas) > 0 else 0.0
            p_val = float(pvals_from_lrt_self_liang(np.array([max(0.0, baseline_lrts[s])]))[0])
            
            mut_scores = {str(aa): float(d) for aa, d in zip(s_mut_aas, delta_lrts)}
            
            plasticity_results.append({
                "site": s + 1,
                "wt_aa": wt_aa,
                "baseline_lrt": float(baseline_lrts[s]),
                "p_value": p_val,
                "intrinsic_plasticity": plasticity,
                "mean_delta_lrt": float(np.mean(delta_lrts)) if len(delta_lrts) > 0 else 0.0,
                "max_delta_lrt": float(np.max(delta_lrts)) if len(delta_lrts) > 0 else 0.0,
                "min_delta_lrt": float(np.min(delta_lrts)) if len(delta_lrts) > 0 else 0.0,
                "mutant_deltas": mut_scores
            })
            
        muts_processed += n_chunk_muts
        
        # Periodic Progress reporting
        now = time.time()
        sites_done = min(chunk_idx + sites_per_chunk, total_sites)
        if progress and (now - last_update_time >= 0.5 or sites_done == total_sites):
            elapsed = max(1e-3, now - t_start)
            pct = (sites_done / total_sites) * 100.0
            rate = muts_processed / elapsed
            eta = (total_sites - sites_done) * (19.0 / max(1e-3, rate))
            curr_mean_phi = np.mean([r["intrinsic_plasticity"] for r in plasticity_results]) if plasticity_results else 0.0
            
            if is_tty:
                sys.stdout.write(
                    f"\r\033[K[ESSM DMS] Codon {sites_done:4d}/{total_sites} ({pct:5.1f}%) | "
                    f"Mutants: {muts_processed:5d}/{total_mutants_est} | "
                    f"{rate:5.1f} mut/s | "
                    f"Mean Φ: {curr_mean_phi:.3f} | "
                    f"Elapsed: {elapsed:4.1f}s | ETA: {eta:4.1f}s"
                )
                sys.stdout.flush()
            else:
                print(
                    f"[ESSM DMS] Codon {sites_done:4d}/{total_sites} ({pct:5.1f}%) | "
                    f"Mutants: {muts_processed:5d}/{total_mutants_est} | "
                    f"{rate:5.1f} mut/s | "
                    f"Mean Φ: {curr_mean_phi:.3f} | "
                    f"Elapsed: {elapsed:4.1f}s | ETA: {eta:4.1f}s",
                    flush=True
                )
            last_update_time = now
            
    if device.type == 'mps':
        torch.mps.empty_cache()
        
    if progress:
        tot_time = time.time() - t_start
        overall_rate = muts_processed / max(1e-3, tot_time)
        final_mean_phi = np.mean([r["intrinsic_plasticity"] for r in plasticity_results]) if plasticity_results else 0.0
        final_max_phi = np.max([r["intrinsic_plasticity"] for r in plasticity_results]) if plasticity_results else 0.0
        if is_tty:
            sys.stdout.write("\n")
        print(
            f"[✓] Selection DMS Complete: {total_sites} codons ({muts_processed} point mutations) "
            f"in {tot_time:.2f}s ({overall_rate:.1f} mut/s). Mean Plasticity Φ = {final_mean_phi:.3f}, Max Φ = {final_max_phi:.3f}",
            flush=True
        )
        
    return plasticity_results

def run_epistatic_analysis(
    alignment_path: str,
    tree_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    variant: Optional[str] = None,
    focal_taxon: Optional[str] = None,
    min_sim: float = 0.35,
    min_shared: int = 2,
    max_fdr: float = 0.05,
    min_lrt: float = 1.0,
    min_cesi: float = 2.0,
    min_clique_size: int = 3,
    max_overlap: float = 0.50,
    min_coherence: float = 0.50,
    n_permutations: int = 10000,
    max_perm_p: Optional[float] = None,
    rng_seed: Optional[int] = 42,
    run_dms: bool = True,
    skip_dms: bool = False,
    cpu: bool = False,
    batch_size: int = 64,
    progress: bool = True,
    use_tn93: bool = False
) -> Dict[str, Any]:
    """
    Executes pure Transformer Attribution Co-Selection Networks, Epistatic Sector Mining,
    and Selection Deep Mutational Scanning (ESSM) on the identified epistatic sector positions.
    """
    # 1. Device Selection
    device = get_device(cpu=cpu)

    # 2. Load Alignment and Tree
    c_tensor, a_tensor, d_mat, z_coords, inv_mask, taxa, L = load_alignment_and_tree(
        alignment_path, tree_path, prune_duplicates=True, use_tn93=use_tn93
    )
    N = len(taxa)

    # 3. Load Model
    model = load_model(weights=weights_path, variant=variant, device=device)

    tree_cache = model.precompute_tree_cache(d_mat.to(device), z_coords.to(device))

    # 4. Compute Transformer Attributions and Co-Selection Network
    if progress:
        print(f"[*] Computing Transformer Attributions across {L} codons...", flush=True)
    leaf_attr, lrts, pvals, consensus_aas = compute_transformer_attributions(
        model, c_tensor, a_tensor, tree_cache, taxa, device, batch_size=batch_size, progress=progress
    )

    sig_edges, G = compute_branch_coselection_network(
        leaf_attr, lrts, taxa, consensus_aas,
        min_sim=min_sim, min_shared=min_shared, max_fdr=max_fdr, min_lrt=min_lrt, min_cesi=min_cesi
    )

    sectors = extract_epistatic_sectors_tse(
        G, leaf_attr, lrts, consensus_aas,
        min_clique_size=min_clique_size, max_overlap=max_overlap, min_coherence=min_coherence,
        focal_taxon=focal_taxon, a_np=a_tensor.squeeze(-1).numpy(), taxa=taxa,
        n_permutations=n_permutations, max_perm_p=max_perm_p, rng_seed=rng_seed
    )

    # 5. Run ESSM specifically on the identified epistatic sector positions
    plasticity = []
    if run_dms and not skip_dms:
        epistatic_sites = sorted(list(set([s - 1 for sec in sectors for s in sec["sites"]])))
        if not epistatic_sites and sig_edges:
            high_cesi_edges = [e for e in sig_edges if e.get("cesi", 0.0) >= 3.0]
            epistatic_sites = sorted(list(set(
                [e["site_u"] - 1 for e in high_cesi_edges] + 
                [e["site_v"] - 1 for e in high_cesi_edges]
            )))
            
        if epistatic_sites:
            plasticity = run_insilico_selection_dms(
                model, c_tensor, a_tensor, tree_cache, taxa, device,
                focal_taxon=focal_taxon, batch_size=batch_size, progress=progress,
                target_sites=epistatic_sites
            )

    return {
        "alignment": alignment_path,
        "tree": tree_path,
        "taxa_count": N,
        "codon_count": L,
        "evaluated_taxa": N,
        "coselection_edges_count": len(sig_edges),
        "discovered_sectors_count": len(sectors),
        "edges": sig_edges,
        "sectors": sectors,
        "plasticity": plasticity,
        "coselection_edges": sig_edges,
        "epistatic_sectors": sectors,
        "selection_dms_plasticity": plasticity
    }

def run_digital_dms_analysis(
    alignment_path: str,
    tree_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    variant: Optional[str] = None,
    focal_taxon: Optional[str] = None,
    cpu: bool = False,
    batch_size: int = 64,
    progress: bool = True,
    use_tn93: bool = False
) -> Dict[str, Any]:
    """
    Executes standalone in silico Selection Deep Mutational Scanning (Digital DMS / ESSM)
    for all 19 single-point substitutions across all codon sites.
    """
    # 1. Device Selection
    device = get_device(cpu=cpu)

    # 2. Load Alignment and Tree
    c_tensor, a_tensor, d_mat, z_coords, inv_mask, taxa, L = load_alignment_and_tree(
        alignment_path, tree_path, prune_duplicates=True, use_tn93=use_tn93
    )
    N = len(taxa)

    # 3. Load Model
    model = load_model(weights=weights_path, variant=variant, device=device)

    tree_cache = model.precompute_tree_cache(d_mat.to(device), z_coords.to(device))

    # 4. Run Selection DMS
    plasticity = run_insilico_selection_dms(
        model, c_tensor, a_tensor, tree_cache, taxa, device,
        focal_taxon=focal_taxon, batch_size=batch_size, progress=progress
    )

    return {
        "alignment": alignment_path,
        "tree": tree_path,
        "taxa_count": N,
        "codon_count": L,
        "focal_taxon": focal_taxon or (taxa[0] if taxa else "consensus"),
        "total_mutations": 19 * L,
        "plasticity": plasticity,
        "selection_dms_plasticity": plasticity
    }

# Backwards compatibility aliases
run_dms_analysis = run_digital_dms_analysis
run_epistasis_analysis = run_epistatic_analysis
run_epistatic_sector_mining = run_epistatic_analysis
compute_selection_dms_essm = run_insilico_selection_dms
extract_epistatic_sectors = extract_epistatic_sectors_tse
compute_sector_permutation = compute_sector_permutation_test
