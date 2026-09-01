"""
attribution.py:
Mechanistic Feature Attribution Engine for HyphAeon / HyphAeon.
Answers two core biological questions for positively selected codons:
  1. 'Which species & mutations drive the selection signal?': Single-taxon counterfactual sensitivity (Delta-LRT)
     and percentage of total selection evidence explained.
  2. 'When did selection occur?': Phylogenetic patristic depth to root and evolutionary horizon
     (Recent Terminal / Tip Sweep vs Intermediate Subclade Burst vs Deep Ancestral Divergence).
"""

import numpy as np
import torch
from .dataset import CODON_TO_AA, AA_MAP, GENETIC_CODE
from ._progress import ChunkProgress

INV_GENETIC_CODE = {v: k for k, v in GENETIC_CODE.items()}


def attribute_selection(
    model,
    c: torch.Tensor,
    a: torch.Tensor,
    d: torch.Tensor,
    z: torch.Tensor,
    inv: np.ndarray,
    taxa=None,
    focal_sites=None,
    min_lrt: float = 3.84,
    base_lrts: np.ndarray = None,
    cache=None
):
    """Computes mechanistic feature attribution across candidate positively selected sites.
    
    Args:
        model: PhyloAxialTransformer instance
        c: Codon tensor [L, N, 1]
        a: Amino acid tensor [L, N, 1]
        d: Distance matrix tensor [1, N, N]
        z: 4D MDS coordinate tensor [1, N, 4]
        inv: Invariable site mask [L]
        taxa: List of taxon names [N]
        focal_sites: Explicit list of 0-indexed site indices to attribute (defaults to lrts >= min_lrt)
        min_lrt: Minimum LRT threshold to trigger attribution (default: 3.84, nominal p <= 0.05)
        base_lrts: Pre-computed baseline LRT scores [L]
        cache: Precomputed tree attention cache (optional)
        
    Returns:
        Dictionary mapping site_idx -> attribution metadata dictionary
    """
    device = next(model.parameters()).device
    
    if cache is None:
        cache = model.precompute_tree_cache(d.to(device), z.to(device))
        
    if focal_sites is None:
        if base_lrts is not None:
            focal_sites = np.where(base_lrts >= min_lrt)[0]
        else:
            focal_sites = np.arange(c.shape[0])
    else:
        focal_sites = np.array(focal_sites)
        
    num_sites, num_taxa, _ = c.shape
    if taxa is None:
        taxa = [f"Taxon_{i+1:03d}" for i in range(num_taxa)]
        
    # Distance matrix [N, N]
    d_mat_np = d[0].cpu().numpy()
    mean_dist_to_others = d_mat_np.mean(axis=1)
    max_tree_dist = float(np.max(d_mat_np)) if np.max(d_mat_np) > 0 else 1.0
    
    attributions = {}

    _pb_attr = ChunkProgress(len(focal_sites), 'Attribute', 'site', enabled=len(focal_sites) > 0)
    _attr_done = 0
    with torch.no_grad():
        for site in focal_sites:
            _attr_done += 1
            _pb_attr.update(_attr_done)
            if base_lrts is not None:
                site_lrt = float(base_lrts[site])
            else:
                c_site = c[site:site+1].to(device)
                a_site = a[site:site+1].to(device)
                y_base, _ = model.forward_cached(c_site, a_site, cache)
                site_lrt = float(torch.clamp(y_base, min=0.0).cpu().numpy().ravel()[0])
                
            if site_lrt < min_lrt:
                continue
                
            site_codons = c[site, :, 0].cpu().numpy()
            vals, counts = np.unique(site_codons, return_counts=True)
            cons_codon_tok = int(vals[np.argmax(counts)])
            cons_codon_str = INV_GENETIC_CODE.get(cons_codon_tok, 'NNN')
            cons_aa_str = CODON_TO_AA.get(cons_codon_str, '-')
            cons_aa_tok = AA_MAP.get(cons_aa_str, 20)
            
            driving_species = []
            non_cons_taxa = np.where(site_codons != cons_codon_tok)[0]
            
            for t_idx in non_cons_taxa:
                orig_codon_tok = int(site_codons[t_idx])
                orig_codon_str = INV_GENETIC_CODE.get(orig_codon_tok, 'NNN')
                orig_aa_str = CODON_TO_AA.get(orig_codon_str, '-')
                # Single-taxon counterfactual mutation to consensus
                c_np = c[site:site+1].cpu().numpy().copy()
                a_np = a[site:site+1].cpu().numpy().copy()
                c_np[0, t_idx, 0] = cons_codon_tok
                a_np[0, t_idx, 0] = cons_aa_tok
                
                c_mod = torch.from_numpy(c_np).long().to(device)
                a_mod = torch.from_numpy(a_np).long().to(device)
                
                y_mod, _ = model.forward_cached(c_mod, a_mod, cache)
                mod_lrt = float(torch.clamp(y_mod.squeeze(-1), min=0.0).cpu().numpy().ravel()[0])
                
                delta_lrt = site_lrt - mod_lrt
                pct_contrib = max(0.0, (delta_lrt / site_lrt) * 100.0) if site_lrt > 0 else 0.0
                
                driving_species.append({
                    'taxon': taxa[t_idx],
                    'taxon_index': int(t_idx),
                    'observed_codon': orig_codon_str,
                    'observed_aa': orig_aa_str,
                    'consensus_codon': cons_codon_str,
                    'consensus_aa': cons_aa_str,
                    'delta_lrt': float(delta_lrt),
                    'pct_signal_explained': float(pct_contrib),
                    'mean_patristic_depth': float(mean_dist_to_others[t_idx])
                })
                
            # Sort driving species by marginal selection evidence (Delta-LRT)
            driving_species.sort(key=lambda x: x['delta_lrt'], reverse=True)
            
            # Determine 'When did selection occur?'
            pos_drivers = [d_sp for d_sp in driving_species if d_sp['delta_lrt'] > 0]
            if len(pos_drivers) > 0:
                weights = np.array([d_sp['delta_lrt'] for d_sp in pos_drivers])
                depths = np.array([d_sp['mean_patristic_depth'] for d_sp in pos_drivers])
                weighted_depth = float(np.sum(weights * depths) / (np.sum(weights) + 1e-8))
                depth_ratio = weighted_depth / max_tree_dist
                
                if depth_ratio >= 0.60:
                    epoch = "Recent Terminal / Tip Sweep"
                elif depth_ratio >= 0.35:
                    epoch = "Intermediate Subclade Burst"
                else:
                    epoch = "Deep Ancestral / Basal Divergence"
                    
                num_major_drivers = sum(1 for d_sp in pos_drivers if d_sp['pct_signal_explained'] >= 10.0)
                mode = "Recurrent / Multi-Lineage Adaptation" if num_major_drivers >= 2 else "Single-Lineage Clade Sweep"
            else:
                weighted_depth = 0.0
                depth_ratio = 0.0
                epoch = "Diffuse / Unresolved"
                mode = "Diffuse Background Variation"
                
            attributions[int(site)] = {
                'site_0indexed': int(site),
                'site_1indexed': int(site + 1),
                'predicted_lrt': float(site_lrt),
                'consensus_codon': cons_codon_str,
                'consensus_aa': cons_aa_str,
                'num_mutated_taxa': int(len(non_cons_taxa)),
                'driving_species': driving_species,
                'when_selection_occurred': {
                    'evolutionary_epoch': epoch,
                    'mode_of_adaptation': mode,
                    'weighted_patristic_depth': float(weighted_depth),
                    'tree_depth_ratio': float(depth_ratio)
                }
            }

    _pb_attr.finish()
    return attributions
