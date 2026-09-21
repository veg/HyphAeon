"""
rhizaeon.wavelet
================
Continuous Multiresolution Wavelet & Scalogram Engine for Recombination Detection.

Replaces fixed sliding windows with scale-space Haar-integral continuous wavelet
transforms over genomic coordinates. Detects topological handovers across dyadic
spatial scales (from micro-conversions to macro-chimeras) using Wavelet Transform
Modulus Maxima (WTMM) ridge tracing down the cone of influence.
"""

from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks

from rhizaeon.tensor import PrefixDistanceEngine
from rhizaeon.manifold import compute_classical_mds, align_procrustes, compute_ghost_node_zscores
from rhizaeon.pir import evaluate_triplets_for_taxon, refine_breakpoint_codon


@dataclass
class ScalogramResult:
    """Encapsulates the continuous scale-space manifold scalogram."""
    scales: np.ndarray             # [S] scales (window widths) in nucleotides
    positions: np.ndarray          # [P] genomic coordinates
    z_surface: np.ndarray          # [N, S, P] Ghost Node Z-scores per taxon
    max_z_surface: np.ndarray      # [S, P] envelope of maximum Ghost Node Z-scores
    top_taxa_surface: np.ndarray   # [S, P] index of taxon driving the maximum leverage
    taxa_names: List[str]


@dataclass
class WaveletRidge:
    """Represents a continuous Modulus Maxima ridge tracing a jump singularity."""
    taxon_idx: int
    taxon_name: str
    singularity_nt: int            # Fine-scale converged coordinate
    peak_scale: int                # Scale with maximum topological leverage
    max_z_score: float             # Peak Ghost Node leverage
    holder_alpha: float            # Estimated Hölder regularity exponent
    ridge_positions: List[int]     # Trajectory of cutpoints across scales
    ridge_scales: List[int]        # Dyadic scales along the ridge
    ridge_z_scores: List[float]    # Z-scores along the ridge
    l_pir: float                   # Latent Parental Incongruence Ratio
    parent_1: str                  # Upstream parental lineage
    parent_2: str                  # Downstream parental lineage
    scale_regime: str              # 'micro' (<200nt), 'meso' (200-1500nt), 'macro' (>1500nt)


def generate_dyadic_scales(
    min_scale: int = 32,
    max_scale: int = 2048,
    num_per_octave: int = 2
) -> np.ndarray:
    """
    Generates geometrically spaced dyadic scales:
      a_m = min_scale * 2^(m / num_per_octave)
    """
    octaves = np.log2(max_scale / min_scale)
    total_scales = int(np.round(octaves * num_per_octave)) + 1
    exponents = np.linspace(0, octaves, total_scales)
    scales = np.round(min_scale * (2.0 ** exponents)).astype(int)
    return np.unique(scales)


def compute_haar_scalogram(
    engine: PrefixDistanceEngine,
    taxa_names: List[str],
    scales: Optional[np.ndarray] = None,
    step: int = 20,
    k_mds: int = 4
) -> ScalogramResult:
    """
    Evaluates the continuous Procrustes-Haar scalogram across all scales and positions.
    
    Thanks to PrefixDistanceEngine, each scale-position query is evaluated in O(1)
    using integral difference lookups.
    """
    N = engine.N
    L = engine.num_units

    if scales is None:
        max_s = min(2048, L // 2)
        min_s = min(32, max_s // 4)
        scales = generate_dyadic_scales(min_scale=min_s, max_scale=max_s, num_per_octave=2)

    positions = np.arange(step, L - step, step)
    S = len(scales)
    P = len(positions)

    z_surface = np.zeros((N, S, P), dtype=np.float32)
    max_z_surface = np.zeros((S, P), dtype=np.float32)
    top_taxa_surface = np.zeros((S, P), dtype=np.int32)

    k_eff = min(k_mds, N - 1)

    for s_idx, a in enumerate(scales):
        half_a = a // 2
        for p_idx, b in enumerate(positions):
            left_start = b - half_a
            right_end = b + half_a

            if left_start < 0 or right_end > L:
                continue

            # O(1) integral difference query
            D1 = engine.query_distance_matrix(left_start, b)
            D2 = engine.query_distance_matrix(b, right_end)

            if np.max(D1) < 1e-4 or np.max(D2) < 1e-4:
                continue

            Z1 = compute_classical_mds(D1, k=k_eff)
            Z2 = compute_classical_mds(D2, k=k_eff)
            _, residuals = align_procrustes(Z1, Z2)
            z_scores = compute_ghost_node_zscores(residuals)

            top_t = int(np.argmax(z_scores))
            top_z = float(z_scores[top_t])

            z_surface[:, s_idx, p_idx] = z_scores
            max_z_surface[s_idx, p_idx] = top_z
            top_taxa_surface[s_idx, p_idx] = top_t

    return ScalogramResult(
        scales=scales,
        positions=positions,
        z_surface=z_surface,
        max_z_surface=max_z_surface,
        top_taxa_surface=top_taxa_surface,
        taxa_names=taxa_names
    )


def trace_modulus_maxima_ridges(
    scalogram: ScalogramResult,
    engine: PrefixDistanceEngine,
    min_prominence: float = 0.5,
    min_z_threshold: float = 2.2,
    pir_threshold: float = 0.20,
    bound_factor: float = 1.05
) -> List[WaveletRidge]:
    """
    Traces Wavelet Transform Modulus Maxima (WTMM) ridges down the cone of influence
    from coarse scales (macro-topology) to fine scales (single-base singularity).
    """
    scales = scalogram.scales
    positions = scalogram.positions
    S = len(scales)
    taxa = scalogram.taxa_names

    # Identify seed peaks across all scale octaves (capturing micro, meso, and macro events)
    seed_peaks = []
    for s_idx in range(S):
        curve = scalogram.max_z_surface[s_idx]
        peaks, _ = find_peaks(curve, height=min_z_threshold, prominence=min_prominence, distance=8)
        for p in peaks:
            pos_nt = positions[p]
            top_tax = scalogram.top_taxa_surface[s_idx, p]
            seed_peaks.append((pos_nt, s_idx, p, top_tax, float(curve[p])))

    # Deduplicate seeds within 150 nt for the same taxon, prioritizing strongest peaks and coarser scales
    seed_peaks.sort(key=lambda x: (x[4], x[1]), reverse=True)
    dedup_seeds = []
    for sp in seed_peaks:
        pos_nt, s_idx, p, top_tax, val = sp
        if not any(abs(pos_nt - d[0]) < 150 and top_tax == d[3] for d in dedup_seeds):
            dedup_seeds.append(sp)

    ridges: List[WaveletRidge] = []

    for seed in dedup_seeds:
        seed_pos, seed_s_idx, _, seed_tax, _ = seed

        curr_pos = seed_pos
        ridge_pos = []
        ridge_sc = []
        ridge_z = []

        # Trace down from seed_s_idx to scale 0 (coarse to fine)
        for s_idx in range(seed_s_idx, -1, -1):
            a = scales[s_idx]
            half_a = a // 2
            mask = (positions >= curr_pos - max(40, half_a // 2)) & (positions <= curr_pos + max(40, half_a // 2))
            valid_p_indices = np.where(mask)[0]

            if len(valid_p_indices) == 0:
                continue

            taxon_z = scalogram.z_surface[seed_tax, s_idx, valid_p_indices]
            best_local_idx = np.argmax(taxon_z)
            best_p = valid_p_indices[best_local_idx]
            curr_pos = positions[best_p]

            ridge_pos.append(int(curr_pos))
            ridge_sc.append(int(a))
            ridge_z.append(float(taxon_z[best_local_idx]))

        # Trace up from seed_s_idx + 1 to S - 1 (fine to coarse)
        curr_pos_up = seed_pos
        ridge_pos_up = []
        ridge_sc_up = []
        ridge_z_up = []
        for s_idx in range(seed_s_idx + 1, S):
            a = scales[s_idx]
            half_a = a // 2
            mask = (positions >= curr_pos_up - max(40, half_a // 2)) & (positions <= curr_pos_up + max(40, half_a // 2))
            valid_p_indices = np.where(mask)[0]

            if len(valid_p_indices) == 0:
                continue

            taxon_z = scalogram.z_surface[seed_tax, s_idx, valid_p_indices]
            best_local_idx = np.argmax(taxon_z)
            best_p = valid_p_indices[best_local_idx]
            curr_pos_up = positions[best_p]

            ridge_pos_up.append(int(curr_pos_up))
            ridge_sc_up.append(int(a))
            ridge_z_up.append(float(taxon_z[best_local_idx]))

        # Combine: sorted from coarsest scale to finest scale
        if ridge_sc_up:
            ridge_pos = ridge_pos_up[::-1] + ridge_pos
            ridge_sc = ridge_sc_up[::-1] + ridge_sc
            ridge_z = ridge_z_up[::-1] + ridge_z

        if len(ridge_sc) < 2:
            continue

        # Estimate Hölder regularity alpha via log-log regression:
        # log(Z) ~ alpha * log(scale) + const
        log_a = np.log(np.array(ridge_sc, dtype=np.float64))
        log_z = np.log(np.maximum(ridge_z, 1e-4))
        # Linear fit slope
        if len(log_a) >= 3 and (log_a.max() - log_a.min()) > 0.5:
            alpha, _ = np.polyfit(log_a, log_z, 1)
        else:
            alpha = 0.0

        # Fine-scale singularity coordinate (nt)
        singularity_bp = ridge_pos[-1]
        max_z = float(np.max(ridge_z))
        peak_scale = ridge_sc[int(np.argmax(ridge_z))]

        # Classify scale regime
        if peak_scale < 200:
            scale_regime = "micro"
        elif peak_scale <= 1500:
            scale_regime = "meso"
        else:
            scale_regime = "macro"

        # Evaluate L-PIR and parental attribution at the converged singularity
        eval_flank = min(peak_scale // 2, singularity_bp, engine.num_units - singularity_bp)
        eval_flank = max(30, eval_flank)
        D1 = engine.query_distance_matrix(singularity_bp - eval_flank, singularity_bp)
        D2 = engine.query_distance_matrix(singularity_bp, singularity_bp + eval_flank)
        trip = evaluate_triplets_for_taxon(D1, D2, seed_tax, bound_factor=bound_factor)

        if trip is not None and trip["pir"] >= pir_threshold:
            p1_name = taxa[trip["parent_left_idx"]]
            p2_name = taxa[trip["parent_right_idx"]]
            pir_val = float(trip["pir"])
        else:
            p1_name = "Unknown"
            p2_name = "Unknown"
            pir_val = 0.0

        # Reject if PIR does not meet significance threshold
        if pir_val < pir_threshold:
            continue

        ridges.append(WaveletRidge(
            taxon_idx=seed_tax,
            taxon_name=taxa[seed_tax],
            singularity_nt=singularity_bp,
            peak_scale=peak_scale,
            max_z_score=max_z,
            holder_alpha=float(alpha),
            ridge_positions=ridge_pos,
            ridge_scales=ridge_sc,
            ridge_z_scores=ridge_z,
            l_pir=pir_val,
            parent_1=p1_name,
            parent_2=p2_name,
            scale_regime=scale_regime
        ))

    # Sort ridges by composite significance: max_z * l_pir
    ridges.sort(key=lambda r: r.max_z_score * r.l_pir, reverse=True)
    return ridges


def run_wavelet_recombination_screen(
    engine: PrefixDistanceEngine,
    taxa_names: List[str],
    min_scale: int = 32,
    max_scale: int = 2048,
    step: int = 25,
    min_prominence: float = 0.5,
    min_z_threshold: float = 2.2,
    pir_threshold: float = 0.20,
    bound_factor: float = 1.05
) -> Tuple[ScalogramResult, List[WaveletRidge]]:
    """
    Executes a complete continuous multiresolution wavelet recombination screen.
    """
    scales = generate_dyadic_scales(min_scale=min_scale, max_scale=max_scale, num_per_octave=2)
    scalogram = compute_haar_scalogram(engine, taxa_names, scales=scales, step=step)
    ridges = trace_modulus_maxima_ridges(
        scalogram,
        engine,
        min_prominence=min_prominence,
        min_z_threshold=min_z_threshold,
        pir_threshold=pir_threshold,
        bound_factor=bound_factor
    )
    return scalogram, ridges
