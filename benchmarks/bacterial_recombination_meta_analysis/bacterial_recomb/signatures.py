"""
signatures.py
=============
Taxonomy and classification engine for bacterial recombination signatures.
Distinguishes between biophysical delivery vehicles and integration pathways:
1. Homologous Conversion (Natural Transformation)
2. Micro-Conversion (Domain Shuffling / Mismatch Repair)
3. Specialized Transduction (Bacteriophage Lysogeny / att-anchored)
4. Generalized Transduction (Capsid-bounded Homologous Swaps)
5. Conjugative Mega-Islands (ICEs / Integrative Conjugative Elements)
6. Transposition & Insertion Sequences (IS Elements / TSDs)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np


class RecombinationSignature(str, Enum):
    """The 6 fundamental signatures of bacterial sequence exchange."""
    HOMOLOGOUS_CONVERSION = "Homologous Conversion (Transformation)"
    MICRO_CONVERSION = "Micro-Conversion (Patch / Domain Shuffle)"
    SPECIALIZED_TRANSDUCTION = "Specialized Transduction (Prophage / att-anchored)"
    GENERALIZED_TRANSDUCTION = "Generalized Transduction (Capsid-bounded Swap)"
    CONJUGATIVE_ICE = "Conjugative Mega-Island (ICE / T4SS)"
    TRANSPOSITION = "Transposition / IS Element"
    UNCLASSIFIED = "Unclassified Reticulation"


@dataclass
class RecombinationEvent:
    """Represents a detected recombination tract on a bacterial chromosome."""
    event_id: str
    recombinant_taxon: str
    start_pos: int
    end_pos: int
    length_bp: int
    gap_gradient: float
    orthogonal_departure: float
    l_pir: float
    kinetic_z: float
    directional_attention_drift: float
    flanking_chi_score: float
    flanking_att_score: float
    signature: RecombinationSignature
    confidence: float
    metadata: Optional[Dict[str, Any]] = None


def compute_gap_gradient(
    seq_mat: np.ndarray,
    isolate_idx: int,
    start_pos: int,
    end_pos: int,
    flank_bp: int = 150
) -> float:
    """
    Computes the spatial gap gradient |∇g| across tract boundaries.
    Detects presence/absence block insertions (ICEs, prophages) vs gapless swaps.
    
    Args:
        seq_mat: (N, L) array of encoded sequence characters (4 represents gap '-')
        isolate_idx: Row index of the recombinant isolate
        start_pos: Start coordinate (0-indexed)
        end_pos: End coordinate (0-indexed)
        flank_bp: Window half-width to measure gap fraction transition
        
    Returns:
        Max absolute gap frequency transition across left and right boundaries (0.0 to 1.0)
    """
    N, L = seq_mat.shape
    row = seq_mat[isolate_idx]

    # Left boundary transition: [start - flank, start] vs [start, start + flank]
    l_ext = row[max(0, start_pos - flank_bp) : start_pos]
    l_int = row[start_pos : min(L, start_pos + flank_bp)]
    gap_l_ext = np.mean(l_ext == 4) if len(l_ext) > 0 else 0.0
    gap_l_int = np.mean(l_int == 4) if len(l_int) > 0 else 0.0
    grad_left = abs(gap_l_int - gap_l_ext)

    # Right boundary transition: [end - flank, end] vs [end, end + flank]
    r_int = row[max(0, end_pos - flank_bp) : end_pos]
    r_ext = row[end_pos : min(L, end_pos + flank_bp)]
    gap_r_int = np.mean(r_int == 4) if len(r_int) > 0 else 0.0
    gap_r_ext = np.mean(r_ext == 4) if len(r_ext) > 0 else 0.0
    grad_right = abs(gap_r_ext - gap_r_int)

    return float(max(grad_left, grad_right))


def compute_orthogonal_departure(
    tract_distances: np.ndarray,
    core_gram_eigenvectors: np.ndarray,
    k_core: int = 4
) -> float:
    """
    Evaluates whether the recombinant tract departs from the core phylogenetic
    hyperplane into an orthogonal accessory subspace.
    
    Args:
        tract_distances: (N, N) pairwise distance matrix within the tract
        core_gram_eigenvectors: (N, k) eigenvectors of the whole-genome core Gram matrix
        k_core: Number of core dimensions to evaluate
        
    Returns:
        Orthogonal residual norm P_perp >= 0.0
    """
    N = tract_distances.shape[0]
    # Double center tract distance matrix
    H = np.eye(N) - (1.0 / N) * np.ones((N, N))
    B_tract = -0.5 * H @ (tract_distances ** 2) @ H

    # Project B_tract onto orthogonal complement of core subspace
    V_core = core_gram_eigenvectors[:, :k_core]
    P_core = V_core @ V_core.T
    P_perp = np.eye(N) - P_core

    residual_matrix = P_perp @ B_tract @ P_perp
    residual_norm = float(np.linalg.norm(residual_matrix, ord="fro") / (np.linalg.norm(B_tract, ord="fro") + 1e-8))
    return residual_norm


def scan_attachment_motifs(
    sequence: str,
    start_pos: int,
    end_pos: int,
    search_window: int = 150
) -> Tuple[float, float]:
    """
    Scans tract boundaries for:
    1. Chi sites (RecBCD regulation)
    2. Direct repeats / attachment sites (attL/attR, tRNA 3' ends)
    
    Returns:
        (chi_score, att_score)
    """
    L = len(sequence)
    seq_upper = sequence.upper()

    # Known species Chi motifs (S. pneumoniae, E. coli, etc.)
    chi_motifs = ["GAGAATGA", "TCATTCTC", "GCTGGTGG", "CCACCAGC"]
    
    # Check 100 bp around left and right boundaries
    l_win = seq_upper[max(0, start_pos - search_window) : min(L, start_pos + search_window)]
    r_win = seq_upper[max(0, end_pos - search_window) : min(L, end_pos + search_window)]

    chi_hits = 0
    for motif in chi_motifs:
        chi_hits += l_win.count(motif) + r_win.count(motif)
    chi_score = float(min(1.0, chi_hits / 2.0))

    # Simple direct repeat heuristic between left and right flanks (attL vs attR)
    # Phage integration creates 12-45 bp identical direct repeats
    att_score = 0.0
    kmer_len = 15
    if len(l_win) >= kmer_len and len(r_win) >= kmer_len:
        kmers_l = set(l_win[i : i + kmer_len] for i in range(len(l_win) - kmer_len + 1))
        kmers_r = set(r_win[i : i + kmer_len] for i in range(len(r_win) - kmer_len + 1))
        shared = kmers_l.intersection(kmers_r)
        if len(shared) > 0:
            att_score = 1.0

    return chi_score, att_score


def classify_recombination_event(
    length_bp: int,
    gap_gradient: float,
    orthogonal_departure: float,
    l_pir: float,
    kinetic_z: float,
    directional_attention_drift: float = 0.0,
    chi_score: float = 0.0,
    att_score: float = 0.0,
    max_capsid_size_bp: int = 45000
) -> Tuple[RecombinationSignature, float]:
    """
    Applies the decision rules to classify a candidate recombination event into
    the 6-signature biological taxonomy.
    
    Returns:
        (Signature, Confidence score in [0.0, 1.0])
    """
    is_gapless = gap_gradient < 0.20
    is_structural_indel = gap_gradient >= 0.50
    stays_in_core = orthogonal_departure < 0.35
    is_orthogonal = orthogonal_departure >= 0.35

    # 1. Micro-conversion
    if length_bp < 200 and is_gapless and directional_attention_drift >= 0.10:
        return RecombinationSignature.MICRO_CONVERSION, 0.92

    # 2. Homologous Conversion (Transformation)
    if is_gapless and stays_in_core and length_bp <= 15000 and (l_pir >= 0.50 or chi_score > 0.0):
        conf = 0.85 + (0.10 if chi_score > 0.0 else 0.0)
        return RecombinationSignature.HOMOLOGOUS_CONVERSION, min(0.99, conf)

    # 3. Generalized Transduction (Capsid bounded homologous swap)
    if is_gapless and stays_in_core and 15000 < length_bp <= max_capsid_size_bp:
        return RecombinationSignature.GENERALIZED_TRANSDUCTION, 0.88

    # 4. Specialized Transduction (Lysogenic prophage / att-anchored)
    if is_structural_indel and 20000 <= length_bp <= 55000 and (att_score > 0.0 or is_orthogonal):
        conf = 0.85 + (0.10 if att_score > 0.0 else 0.0)
        return RecombinationSignature.SPECIALIZED_TRANSDUCTION, min(0.98, conf)

    # 5. Conjugative Mega-Island (ICE / T4SS)
    if is_structural_indel and length_bp >= 20000 and is_orthogonal:
        return RecombinationSignature.CONJUGATIVE_ICE, 0.94

    # 6. Transposition / Insertion Sequence
    if is_structural_indel and length_bp < 5000:
        return RecombinationSignature.TRANSPOSITION, 0.85

    # Fallback based on synteny
    if is_gapless and stays_in_core:
        return RecombinationSignature.HOMOLOGOUS_CONVERSION, 0.70
    elif is_structural_indel:
        return RecombinationSignature.CONJUGATIVE_ICE if length_bp >= 20000 else RecombinationSignature.TRANSPOSITION, 0.70

    return RecombinationSignature.UNCLASSIFIED, 0.50
