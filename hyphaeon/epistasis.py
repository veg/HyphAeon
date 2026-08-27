"""
axomeme/epistasis.py
--------------------
Phylogenetic Epistatic Co-Selection Networks, Sector Mining, and
In Silico Selection Deep Mutational Scanning (ESSM / Digital DMS).

Strictly uses continuous Neural Transformer Attribution Vectors (multi-head
root-to-leaf axial attention weights combined with contextual selection drive)
with ZERO character mapping, ZERO ancestral parsimony heuristics, and ZERO
discrete substitution counting.
"""

import math
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import scipy.stats as stats
import torch
import networkx as nx

from .dataset import (
    load_alignment_and_tree,
    get_codon_token,
    get_aa_token
)
from .inference import compute_transformer_attributions, load_model
from .utils import REV_AA_MAP, lrt_to_pvals, benjamini_hochberg, find_taxon_index

# Canonical sense codons for all 20 standard amino acids
CANONICAL_AA_TO_CODON = {
    'A': 'GCC', 'C': 'TGC', 'D': 'GAC', 'E': 'GAG', 'F': 'TTC',
    'G': 'GGC', 'H': 'CAC', 'I': 'ATC', 'K': 'AAG', 'L': 'CTG',
    'M': 'ATG', 'N': 'AAC', 'P': 'CCC', 'Q': 'CAG', 'R': 'CGC',
    'S': 'AGC', 'T': 'ACC', 'V': 'GTG', 'W': 'TGG', 'Y': 'TAC'
}

# Backward-compatibility alias
compute_phylogenetic_branch_attributions = lambda model, c_tensor, a_tensor, tree_cache, tree_obj, taxa, device, batch_size=64: (
    compute_transformer_attributions(model, c_tensor, a_tensor, tree_cache, taxa, device, batch_size) + (taxa,)
)

def compute_branch_coselection_network(
    attributions: np.ndarray,
    lrts: np.ndarray,
    branch_names: List[str],
    consensus_aas: List[str],
    min_sim: float = 0.30,
    min_shared: int = 2,
    max_fdr: float = 0.05,
    min_lrt: float = 1.0
) -> Tuple[List[Dict[str, Any]], nx.Graph]:
    """
    Computes pairwise co-selection directly from continuous Transformer Attribution Vectors.
    Calculates cosine attribution alignment, Student t-statistic significance,
    and Composite Epistatic Selection Index (CESI).
    """
    L, N = attributions.shape
    df = max(1, N - 2)
    
    G = nx.Graph()
    for s in range(L):
        G.add_node(s + 1, ref=consensus_aas[s], lrt=float(lrts[s]))
        
    norms = np.linalg.norm(attributions, axis=1) # [L]
    k_counts = np.sum(attributions > 0, axis=1)  # [L]
    
    active_mask = (norms > 0)
    valid_sites = np.where(active_mask)[0]
    if len(valid_sites) < 2:
        return [], G
        
    A_active = attributions[valid_sites]        # [V, N]
    A_bin = (A_active > 0).astype(np.float32)   # [V, N]
    norms_active = norms[valid_sites]           # [V]
    
    dot_matrix = A_active @ A_active.T          # [V, V]
    shared_matrix = A_bin @ A_bin.T             # [V, V]
    norm_outer = np.outer(norms_active, norms_active)
    sim_matrix = dot_matrix / np.maximum(norm_outer, 1e-9)
    
    tri_i, tri_j = np.triu_indices(len(valid_sites), k=1)
    shared_arr = shared_matrix[tri_i, tri_j]
    sim_arr = sim_matrix[tri_i, tri_j]
    s1_arr = valid_sites[tri_i]
    s2_arr = valid_sites[tri_j]
    
    lrt_s1 = lrts[s1_arr]
    lrt_s2 = lrts[s2_arr]
    max_lrt_arr = np.maximum(lrt_s1, lrt_s2)
    
    pass_filter = (shared_arr >= min_shared) & (sim_arr >= min_sim) & (max_lrt_arr >= min_lrt)
    pass_indices = np.where(pass_filter)[0]
    
    candidate_pairs = []
    for idx in pass_indices:
        s1 = int(s1_arr[idx])
        s2 = int(s2_arr[idx])
        shared = int(shared_arr[idx])
        sim = float(sim_arr[idx])
        
        # Continuous Student's t-test for attribution cosine alignment
        t_stat = sim * np.sqrt(df / max(1e-9, 1.0 - sim**2))
        p_t = float(stats.t.sf(t_stat, df=df))
        cesi = sim * math.sqrt(max(float(lrts[s1]), 0.1) * max(float(lrts[s2]), 0.1))
        
        candidate_pairs.append({
            "site_u": s1 + 1,
            "site_v": s2 + 1,
            "ref_u": consensus_aas[s1],
            "ref_v": consensus_aas[s2],
            "lrt_u": float(lrts[s1]),
            "lrt_v": float(lrts[s2]),
            "similarity": sim,
            "shared_taxa": shared,
            "shared_branches": shared,
            "p_val": p_t,
            "hyper_p": p_t,
            "cesi": float(cesi)
        })
        
    if not candidate_pairs:
        return [], G
        
    candidate_pairs.sort(key=lambda x: x["p_val"])
    m_tests = len(candidate_pairs)
    sig_pairs = []
    
    p_arr = np.array([p["p_val"] for p in candidate_pairs], dtype=np.float32)
    q_arr = benjamini_hochberg(p_arr)
    for i, q_val in enumerate(q_arr):
        candidate_pairs[i]["fdr_q"] = float(q_val)
        
    for p in candidate_pairs:
        if p["fdr_q"] <= max_fdr:
            sig_pairs.append(p)
            G.add_edge(
                p["site_u"],
                p["site_v"],
                weight=p["similarity"],
                shared=p["shared_taxa"],
                cesi=p["cesi"],
                fdr_q=p["fdr_q"]
            )
            
    sig_pairs.sort(key=lambda x: x["cesi"], reverse=True)
    return sig_pairs, G

def extract_epistatic_sectors_tse(
    G: nx.Graph,
    attributions: np.ndarray,
    lrts: np.ndarray,
    consensus_aas: List[str],
    min_clique_size: int = 3,
    max_overlap: float = 0.50,
    focal_taxon: Optional[str] = None,
    a_np: Optional[np.ndarray] = None,
    taxa: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Two-Stage Seed-and-Extend (TSE) epistatic sector mining using greedy modularity
    community decomposition on the continuous attribution co-selection graph.
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
    for comp_idx, members in enumerate(components):
        site_indices = [n - 1 for n in members]
        sub_A = attributions[site_indices, :]
        
        # Spectral coherence C(S) = lambda_1 / sum(lambda_k)
        if sub_A.shape[0] >= 2 and np.linalg.norm(sub_A) > 0:
            cov = sub_A @ sub_A.T
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.maximum(eigvals, 0.0)
            tot_eig = np.sum(eigvals)
            coherence = float(eigvals[-1] / tot_eig) if tot_eig > 0 else 0.0
        else:
            coherence = 1.0
            
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
            focal_idx = find_taxon_index(taxa, focal_taxon)
            if focal_idx is not None:
                focal_name = taxa[focal_idx]
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
            "sector_id": comp_idx + 1,
            "size": len(members),
            "sites": [s + 1 for s in sorted_sites],
            "spectral_coherence": coherence,
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
        
    sectors.sort(key=lambda x: (x["spectral_coherence"], x["size"]), reverse=True)
    return sectors

def run_insilico_selection_dms(
    model: "PhyloAxialTransformer",
    c_tensor: torch.Tensor,
    a_tensor: torch.Tensor,
    tree_cache: Dict[str, Any],
    taxa: List[str],
    device: torch.device,
    focal_taxon: Optional[str] = None,
    batch_size: int = 64
) -> List[Dict[str, Any]]:
    """
    Performs 19-amino-acid in silico Deep Mutational Scanning (Selection DMS / ESSM)
    by computing delta LRT across all possible point substitutions at a focal taxon.
    """
    L, n_taxa, _ = c_tensor.shape
    
    focal_idx = 0
    if focal_taxon:
        idx = find_taxon_index(taxa, focal_taxon)
        if idx is not None:
            focal_idx = idx
                
    a_np = a_tensor.squeeze(-1).numpy() # [L, N]
    model.eval()
    baseline_lrts = np.zeros(L, dtype=np.float32)
    
    with torch.no_grad():
        for start_idx in range(0, L, batch_size):
            end_idx = min(start_idx + batch_size, L)
            c_chunk = c_tensor[start_idx:end_idx].to(device)
            a_chunk = a_tensor[start_idx:end_idx].to(device)
            y_soft, _ = model.forward_cached(c_chunk, a_chunk, tree_cache)
            baseline_lrts[start_idx:end_idx] = torch.clamp(y_soft, min=0.0).cpu().numpy().flatten()
            
    plasticity_results = []
    standard_aas = list('ACDEFGHIKLMNPQRSTVWY')
    
    for s in range(L):
        wt_tok = a_np[s, focal_idx]
        wt_aa = REV_AA_MAP.get(wt_tok, '-')
        if wt_tok >= 20:
            valid = a_np[s][a_np[s] < 20]
            wt_tok = int(np.argmax(np.bincount(valid))) if len(valid) > 0 else 0
            wt_aa = REV_AA_MAP.get(wt_tok, 'A')
            
        cand_mut_aas = [aa for aa in standard_aas if aa != wt_aa]
        n_muts = len(cand_mut_aas)
        
        c_batch = c_tensor[s:s+1].repeat(n_muts, 1, 1).to(device)
        a_batch = a_tensor[s:s+1].repeat(n_muts, 1, 1).to(device)
        
        for m_i, m_aa in enumerate(cand_mut_aas):
            codon_str = CANONICAL_AA_TO_CODON[m_aa]
            c_tok = get_codon_token(codon_str)
            aa_tok = get_aa_token(codon_str)
            c_batch[m_i, focal_idx, 0] = c_tok
            a_batch[m_i, focal_idx, 0] = aa_tok
            
        with torch.no_grad():
            y_mut_soft, _ = model.forward_cached(c_batch, a_batch, tree_cache)
            mut_lrts = torch.clamp(y_mut_soft, min=0.0).cpu().numpy().flatten()
            
        delta_lrts = mut_lrts - baseline_lrts[s]
        plasticity = float(np.mean(np.abs(delta_lrts)))
        p_val = float(lrt_to_pvals(np.array([max(0.0, baseline_lrts[s])], dtype=np.float32))[0])
        
        plasticity_results.append({
            "site": s + 1,
            "wt_aa": wt_aa,
            "baseline_lrt": float(baseline_lrts[s]),
            "p_value": p_val,
            "intrinsic_plasticity": plasticity,
            "mean_delta_lrt": float(np.mean(delta_lrts)),
            "max_delta_lrt": float(np.max(delta_lrts)),
            "min_delta_lrt": float(np.min(delta_lrts))
        })
        
    return plasticity_results

def run_epistatic_analysis(
    alignment_path: str,
    tree_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    variant: Optional[str] = None,
    focal_taxon: Optional[str] = None,
    min_sim: float = 0.30,
    min_shared: int = 2,
    max_fdr: float = 0.05,
    min_lrt: float = 1.0,
    min_clique_size: int = 3,
    max_overlap: float = 0.50,
    run_dms: bool = True,
    skip_dms: bool = False,
    cpu: bool = False,
    batch_size: int = 64
) -> Dict[str, Any]:
    """
    Executes pure Transformer Attribution Co-Selection Networks, Epistatic Sector Mining,
    and Selection Deep Mutational Scanning (Digital DMS).
    """
    # 1. Device & Model
    model, device = load_model(weights=weights_path, variant=variant, cpu=cpu)

    # 2. Load Alignment and Tree
    c_tensor, a_tensor, d_mat, z_coords, inv_mask, taxa, L = load_alignment_and_tree(
        alignment_path, tree_path, prune_duplicates=True
    )
    N = len(taxa)

    tree_cache = model.precompute_tree_cache(d_mat.to(device), z_coords.to(device))

    # 4. Compute Transformer Attributions and Co-Selection Network
    leaf_attr, lrts, pvals, consensus_aas = compute_transformer_attributions(
        model, c_tensor, a_tensor, tree_cache, taxa, device, batch_size=batch_size
    )

    sig_edges, G = compute_branch_coselection_network(
        leaf_attr, lrts, taxa, consensus_aas,
        min_sim=min_sim, min_shared=min_shared, max_fdr=max_fdr, min_lrt=min_lrt
    )

    sectors = extract_epistatic_sectors_tse(
        G, leaf_attr, lrts, consensus_aas,
        min_clique_size=min_clique_size, max_overlap=max_overlap,
        focal_taxon=focal_taxon, a_np=a_tensor.squeeze(-1).numpy(), taxa=taxa
    )

    plasticity = []
    if run_dms and not skip_dms:
        plasticity = run_insilico_selection_dms(
            model, c_tensor, a_tensor, tree_cache, taxa, device,
            focal_taxon=focal_taxon, batch_size=batch_size
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

# Backwards compatibility aliases
run_epistasis_analysis = run_epistatic_analysis
run_epistatic_sector_mining = run_epistatic_analysis
compute_selection_dms_essm = run_insilico_selection_dms
extract_epistatic_sectors = extract_epistatic_sectors_tse
