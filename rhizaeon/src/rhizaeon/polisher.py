"""
rhizaeon.polisher
=================
Data-driven Maximum Likelihood (ML) Breakpoint Polisher.

Pinpoints recombination boundaries down to the intrinsic information-theoretic
limit imposed by nucleotide variability, returning the exact maximum-likelihood
interval [ci_left, ci_right] where sequence likelihood is maximally flat,
along with the principled ML midpoint and profile likelihood bounds.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


@dataclass
class PolishedBreakpoint:
    """Encapsulates a data-driven polished recombination breakpoint."""
    coarse_bp: int
    polished_bp: int
    ci_left: int
    ci_right: int
    plateau_width: int
    num_informative_sites: int
    flanking_p1_site: Optional[int]
    flanking_p2_site: Optional[int]
    log_likelihood_gain: float
    recombinant_taxon: str
    parent_1: str
    parent_2: str


def polish_breakpoint_ml(
    seq_matrix: np.ndarray,
    coarse_bp: int,
    r_idx: int,
    p1_idx: int,
    p2_idx: int,
    search_window: int = 500,
    error_rate: float = 0.01,
    taxa_names: Optional[List[str]] = None
) -> PolishedBreakpoint:
    """
    Polishes a candidate breakpoint using a localized 2-state likelihood profile.
    
    Parameters:
      seq_matrix: shape [N, L] integer encoded nucleotides (0=A, 1=C, 2=G, 3=T, 4=Gap/Ambiguity).
      coarse_bp: initial inferred breakpoint coordinate.
      r_idx: index of recombinant taxon.
      p1_idx: index of left donor parent (P1).
      p2_idx: index of right donor parent (P2).
      search_window: radius in base pairs around coarse_bp to inspect.
      error_rate: assumed sequencing / private mutation rate epsilon (default: 0.01).
      taxa_names: optional list of taxon names for display.
      
    Returns:
      PolishedBreakpoint dataclass with exact ML plateau bounds and midpoint.
    """
    N, L = seq_matrix.shape
    w_start = max(0, coarse_bp - search_window)
    w_end = min(L, coarse_bp + search_window)
    
    r_seq = seq_matrix[r_idx, w_start:w_end]
    p1_seq = seq_matrix[p1_idx, w_start:w_end]
    p2_seq = seq_matrix[p2_idx, w_start:w_end]
    
    eps = max(1e-5, min(0.49, error_rate))
    gamma = np.log(3.0 * (1.0 - eps) / eps)
    
    win_len = w_end - w_start
    delta_l = np.zeros(win_len, dtype=np.float64)
    
    valid = (r_seq < 4) & (p1_seq < 4) & (p2_seq < 4)
    diff_parents = valid & (p1_seq != p2_seq)
    
    matches_p1 = diff_parents & (r_seq == p1_seq)
    matches_p2 = diff_parents & (r_seq == p2_seq)
    
    delta_l[matches_p1] = +gamma
    delta_l[matches_p2] = -gamma
    
    num_inf = int(np.sum(diff_parents))
    
    # Cumulative profile log-likelihood
    profile_ll = np.cumsum(delta_l)
    max_ll = float(np.max(profile_ll))
    
    # Identify all positions achieving the maximum likelihood
    # (The uninformative sequence plateau where likelihood is flat)
    tol = 1e-7
    plateau_rel_indices = np.where(np.abs(profile_ll - max_ll) <= tol)[0]
    
    if len(plateau_rel_indices) == 0:
        # Fallback if no informative sites found
        ci_left = coarse_bp
        ci_right = coarse_bp
        ml_mid = coarse_bp
        p1_site = None
        p2_site = None
        gain = 0.0
    else:
        ci_left = w_start + int(plateau_rel_indices[0])
        ci_right = w_start + int(plateau_rel_indices[-1])
        ml_mid = int(round((ci_left + ci_right) / 2.0))
        
        # Log likelihood gain over the coarse starting point
        coarse_rel = min(max(0, coarse_bp - w_start), win_len - 1)
        gain = float(max_ll - profile_ll[coarse_rel])
        
        # Find nearest informative sites flanking the plateau
        inf_indices = np.where(diff_parents)[0] + w_start
        p1_indices = np.where(matches_p1)[0] + w_start
        p2_indices = np.where(matches_p2)[0] + w_start
        
        p1_flank = [s for s in p1_indices if s <= ci_left]
        p2_flank = [s for s in p2_indices if s >= ci_right]
        
        p1_site = int(p1_flank[-1]) if p1_flank else None
        p2_site = int(p2_flank[0]) if p2_flank else None

    r_name = taxa_names[r_idx] if taxa_names else f"taxon_{r_idx}"
    p1_name = taxa_names[p1_idx] if taxa_names else f"taxon_{p1_idx}"
    p2_name = taxa_names[p2_idx] if taxa_names else f"taxon_{p2_idx}"

    return PolishedBreakpoint(
        coarse_bp=coarse_bp,
        polished_bp=ml_mid,
        ci_left=ci_left,
        ci_right=ci_right,
        plateau_width=max(0, ci_right - ci_left),
        num_informative_sites=num_inf,
        flanking_p1_site=p1_site,
        flanking_p2_site=p2_site,
        log_likelihood_gain=gain,
        recombinant_taxon=r_name,
        parent_1=p1_name,
        parent_2=p2_name
    )


def polish_all_breakpoints(
    seq_matrix: np.ndarray,
    taxa_names: List[str],
    breakpoints: List[Any],
    search_window: Optional[int] = None,
    error_rate: float = 0.01
) -> List[PolishedBreakpoint]:
    """
    Applies data-driven ML polishing to a list of detected FDABreakpoints.
    """
    N, L = seq_matrix.shape
    taxa_map = {name: i for i, name in enumerate(taxa_names)}
    
    # Adaptive search window based on sequence scale
    win = search_window if search_window is not None else max(150, min(1000, L // 20))
    
    polished = []
    for b in breakpoints:
        r_name = b.recombinant_taxon
        p1_name = b.parent_1
        p2_name = b.parent_2
        
        if r_name not in taxa_map or p1_name not in taxa_map or p2_name not in taxa_map:
            continue
            
        r_idx = taxa_map[r_name]
        p1_idx = taxa_map[p1_name]
        p2_idx = taxa_map[p2_name]
        
        pol = polish_breakpoint_ml(
            seq_matrix,
            coarse_bp=b.breakpoint_nt,
            r_idx=r_idx,
            p1_idx=p1_idx,
            p2_idx=p2_idx,
            search_window=win,
            error_rate=error_rate,
            taxa_names=taxa_names
        )
        polished.append(pol)
        
    return polished
