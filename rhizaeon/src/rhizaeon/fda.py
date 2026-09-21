"""
rhizaeon.fda
============
Functional Data Analysis (FDA) and Manifold Trajectory Tracking for Recombination.

Transforms sequence alignments into continuous vector-valued functional curves
Y_i(s) in R^k along the genomic coordinate s. Recombination events are detected
as jump discontinuities in functional trajectories via:
  1. Global spectral manifold projection (O(1) per coordinate via BLAS)
  2. Functional kinetic velocity fields E_i(s) = || dY_i / ds ||^2
  3. 1D Total Variation (Fused Lasso) piecewise-constant filtering (Condat 2013)
  4. Vectorized Latent Parental Incongruence Ratio (L-PIR) attribution
"""

from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import fisher_exact

from rhizaeon.tensor import PrefixDistanceEngine
from rhizaeon.manifold import compute_classical_mds, align_procrustes, compute_ghost_node_zscores
from rhizaeon.pir import evaluate_triplets_for_taxon
from rhizaeon.polisher import polish_breakpoint_ml


@dataclass
class FDAResult:
    """Encapsulates the continuous functional trajectory and changepoint analysis."""
    cutpoints: np.ndarray       # [M] genomic coordinates
    trajectories: np.ndarray    # [N, M, k] functional coordinates along genome
    kinetic_energy: np.ndarray  # [N, M] functional velocity squared
    z_energy: np.ndarray        # [N, M] robust Studentized kinetic energy Z-scores
    tv_trajectories: np.ndarray # [N, M, k] Total Variation filtered piecewise constant curves
    taxa_names: List[str]


@dataclass
class FDABreakpoint:
    """Represents a detected functional changepoint / recombination event."""
    breakpoint_nt: int
    recombinant_taxon: str
    taxon_idx: int
    kinetic_z: float
    l_pir: float
    parent_1: str
    parent_2: str
    jump_magnitude: float
    # Data-driven ML Polishing fields
    coarse_bp: Optional[int] = None
    polished_bp: Optional[int] = None
    ci_left: Optional[int] = None
    ci_right: Optional[int] = None
    plateau_width: Optional[int] = None
    log_likelihood_gain: Optional[float] = None
    flanking_p1_site: Optional[int] = None
    flanking_p2_site: Optional[int] = None


def tv1d(y: np.ndarray, lam: float) -> np.ndarray:
    """
    Condat's direct O(L) 1D Total Variation / Fused Lasso denoising filter.
    Reference: Condat, L. (2013). IEEE Signal Processing Letters, 20(11), 1054-1057.
    """
    n = len(y)
    x = np.empty(n, dtype=np.float64)
    k = 0
    k0 = 0
    vmin = y[0] - lam
    vmax = y[0] + lam
    umin = lam
    umax = -lam
    kplus = 0
    kminus = 0

    while True:
        while k < n - 1:
            if y[k + 1] + umin < vmin - lam:
                for i in range(k0, kminus + 1):
                    x[i] = vmin
                k = k0 = kminus = kplus = kminus + 1
                vmin = y[k]
                vmax = y[k] + 2.0 * lam
                umin = lam
                umax = -lam
            elif y[k + 1] + umax > vmax + lam:
                for i in range(k0, kplus + 1):
                    x[i] = vmax
                k = k0 = kminus = kplus = kplus + 1
                vmin = y[k] - 2.0 * lam
                vmax = y[k]
                umin = lam
                umax = -lam
            else:
                k += 1
                umin += y[k] - vmin
                umax += y[k] - vmax
                if umin >= lam:
                    vmin += (umin - lam) / (k - k0 + 1)
                    umin = lam
                    kminus = k
                if umax <= -lam:
                    vmax += (umax + lam) / (k - k0 + 1)
                    umax = -lam
                    kplus = k

        if umin < 0.0:
            for i in range(k0, kminus + 1):
                x[i] = vmin
            k = k0 = kminus = kplus = kminus + 1
            vmin = y[k]
            vmax = y[k] + 2.0 * lam
            umin = lam
            umax = -lam
        elif umax > 0.0:
            for i in range(k0, kplus + 1):
                x[i] = vmax
            k = k0 = kminus = kplus = kplus + 1
            vmin = y[k] - 2.0 * lam
            vmax = y[k]
            umin = lam
            umax = -lam
        else:
            for i in range(k0, n):
                x[i] = vmin + umin / (k - k0 + 1)
            break
    return x


def compute_fda_trajectories(
    engine: PrefixDistanceEngine,
    taxa_names: List[str],
    k: int = 5,
    bin_size: int = 300,
    step: int = 30,
    tv_lambda_factor: float = 0.5
) -> FDAResult:
    """
    Computes continuous functional trajectory curves Y_i(s) in R^k for all taxa
    via global spectral manifold projection and applies 1D Total Variation filtering.
    """
    N = engine.N
    L = engine.num_units
    k_eff = min(k, N - 1)

    # 1. Global manifold embedding (computed once in O(1) prefix time)
    D_glob = engine.query_distance_matrix(0, L)
    Z_glob = compute_classical_mds(D_glob, k=k_eff)

    H = np.eye(N, dtype=np.float64) - (1.0 / N) * np.ones((N, N), dtype=np.float64)
    # Projection operator W in R^{N x k}
    W = H @ Z_glob @ np.linalg.pinv(Z_glob.T @ Z_glob)

    # 2. Continuous functional projection across coordinates
    flank = bin_size // 2
    cutpoints = np.arange(flank, L - flank, step)
    M = len(cutpoints)

    Y = np.zeros((N, M, k_eff), dtype=np.float64)
    for m, cp in enumerate(cutpoints):
        D_m = engine.query_distance_matrix(cp - flank, cp + flank)
        B_m = -0.5 * (H @ (D_m ** 2) @ H)
        Y[:, m, :] = B_m @ W

    # 3. Functional Velocity and Kinetic Energy Field
    dY = np.gradient(Y, step, axis=1)
    kinetic_energy = np.sum(dY ** 2, axis=2)  # [N, M]

    # Robust Studentized Z-score of functional kinetic energy per taxon
    med_E = np.median(kinetic_energy, axis=1, keepdims=True)
    mad_E = np.median(np.abs(kinetic_energy - med_E), axis=1, keepdims=True)
    iqr_E = np.maximum(mad_E * 1.4826, 1e-6)
    z_energy = (kinetic_energy - med_E) / iqr_E  # [N, M]

    # 4. 1D Total Variation Denoising per taxon
    Y_tv = np.zeros_like(Y)
    for i in range(N):
        for j in range(k_eff):
            std_yj = float(np.std(Y[i, :, j]))
            lam = tv_lambda_factor * std_yj
            Y_tv[i, :, j] = tv1d(Y[i, :, j], lam)

    return FDAResult(
        cutpoints=cutpoints,
        trajectories=Y,
        kinetic_energy=kinetic_energy,
        z_energy=z_energy,
        tv_trajectories=Y_tv,
        taxa_names=taxa_names
    )


def verify_crossover_support(
    seq_mat: np.ndarray,
    r_idx: int,
    p1_idx: int,
    p2_idx: int,
    bp: int,
    flank: int = 150,
    min_informative: int = 3,
    p_critical: float = 0.005
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluates whether candidate breakpoint 'bp' for recombinant 'r_idx'
    and candidate parents ('p1_idx', 'p2_idx') represents a statistically significant
    parental crossover or spurious intra-clade coalescent noise.

    Constructs a 2x2 contingency table of flanking parental match counts:
      [ [k_L(P1), k_L(P2)],
        [k_R(P1), k_R(P2)] ]
    and performs a one-sided Fisher exact test for parental allele switching.
    """
    N, L = seq_mat.shape
    l_start = max(0, bp - flank)
    r_end = min(L, bp + flank)

    l_sub = slice(l_start, bp)
    r_sub = slice(bp, r_end)

    r_l = seq_mat[r_idx, l_sub]
    p1_l = seq_mat[p1_idx, l_sub]
    p2_l = seq_mat[p2_idx, l_sub]

    r_r = seq_mat[r_idx, r_sub]
    p1_r = seq_mat[p1_idx, r_sub]
    p2_r = seq_mat[p2_idx, r_sub]

    valid_l = (p1_l < 4) & (p2_l < 4) & (r_l < 4)
    valid_r = (p1_r < 4) & (p2_r < 4) & (r_r < 4)

    diff_l = valid_l & (p1_l != p2_l)
    diff_r = valid_r & (p1_r != p2_r)

    k_l1 = int(np.sum(diff_l & (r_l == p1_l)))
    k_l2 = int(np.sum(diff_l & (r_l == p2_l)))
    k_r1 = int(np.sum(diff_r & (r_r == p1_r)))
    k_r2 = int(np.sum(diff_r & (r_r == p2_r)))

    s1 = k_l1 + k_r2
    v1 = k_l2 + k_r1
    s2 = k_l2 + k_r1
    v2 = k_l1 + k_r2

    stats = {"k_l1": k_l1, "k_l2": k_l2, "k_r1": k_r1, "k_r2": k_r2}

    if s1 > v1 and k_l1 >= min_informative and k_r2 >= min_informative:
        _, p_val = fisher_exact([[k_l1, k_l2], [k_r1, k_r2]], alternative="greater")
        stats["orientation"] = 1
        stats["p_value"] = float(p_val)
        if p_val <= p_critical:
            return True, float(p_val), stats

    elif s2 > v2 and k_l2 >= min_informative and k_r1 >= min_informative:
        _, p_val = fisher_exact([[k_l2, k_l1], [k_r2, k_r1]], alternative="greater")
        stats["orientation"] = 2
        stats["p_value"] = float(p_val)
        if p_val <= p_critical:
            return True, float(p_val), stats

    return False, 1.0, stats


def extract_fda_breakpoints(
    fda_result: FDAResult,
    engine: PrefixDistanceEngine,
    min_kinetic_z: float = 3.0,
    min_pir: float = 0.10,
    min_prominence: float = 1.0,
    distance_bins: int = 4,
    crossover_validation: bool = True,
    crossover_p_threshold: Optional[float] = None,
    min_informative_sites: int = 3,
    polish_ml: bool = True,
    search_window: Optional[int] = None,
    error_rate: float = 0.01
) -> List[FDABreakpoint]:
    """
    Identifies recombination breakpoints from functional kinetic energy surges
    and Total Variation jumps, cross-verified with L-PIR parental handover tests
    and statistical 2x2 Fisher crossover gates.
    Optionally polishes boundaries with the data-driven ML polisher.
    """
    cutpoints = fda_result.cutpoints
    z_energy = fda_result.z_energy
    Y_tv = fda_result.tv_trajectories
    taxa = fda_result.taxa_names
    N, M, k = Y_tv.shape
    L = engine.num_units
    seq_mat = getattr(engine, "full_seq_matrix", getattr(engine, "seq_matrix", None))

    # Envelope of maximum kinetic energy across taxa
    max_z = np.max(z_energy, axis=0)
    top_taxa = np.argmax(z_energy, axis=0)

    # Calculate Total Variation discrete jump magnitudes: || Y_tv(m+1) - Y_tv(m) ||
    tv_jumps = np.zeros((N, M))
    tv_jumps[:, :-1] = np.linalg.norm(np.diff(Y_tv, axis=1), axis=2)

    peaks, _ = find_peaks(max_z, height=min_kinetic_z, prominence=min_prominence, distance=distance_bins)
    breakpoints: List[FDABreakpoint] = []

    for p in peaks:
        bp = int(cutpoints[p])
        t_idx = int(top_taxa[p])
        rec_name = taxa[t_idx]

        # Evaluate flanking L-PIR handover
        flank = min(250, bp, L - bp)
        if flank < 40:
            continue

        D1 = engine.query_distance_matrix(bp - flank, bp)
        D2 = engine.query_distance_matrix(bp, bp + flank)

        cands = evaluate_triplets_for_taxon(
            D1, D2, t_idx, weight_by_divergence=True, top_k=5 if seq_mat is not None else 1
        )
        if not cands:
            continue
        if isinstance(cands, dict):
            cands = [cands]

        valid_trip = None
        for trip in cands:
            if trip["pir"] < min_pir:
                continue
            if seq_mat is not None and crossover_validation:
                p_crit = crossover_p_threshold if crossover_p_threshold is not None else min(0.005, 0.15 / max(1, N))
                scale_coord = 3 if getattr(engine, "codon_aligned", False) else 1
                ok, _, _ = verify_crossover_support(
                    seq_mat, t_idx, trip["parent_left_idx"], trip["parent_right_idx"],
                    bp * scale_coord, flank=flank * scale_coord, min_informative=min_informative_sites, p_critical=p_crit
                )
                if ok:
                    valid_trip = trip
                    break
            else:
                valid_trip = trip
                break

        if valid_trip is None:
            continue

        trip = valid_trip
        p1_name = taxa[trip["parent_left_idx"]]
        p2_name = taxa[trip["parent_right_idx"]]
        jump_mag = float(tv_jumps[t_idx, p])

        breakpoints.append(FDABreakpoint(
            breakpoint_nt=bp,
            recombinant_taxon=rec_name,
            taxon_idx=t_idx,
            kinetic_z=float(max_z[p]),
            l_pir=float(trip["pir"]),
            parent_1=p1_name,
            parent_2=p2_name,
            jump_magnitude=jump_mag,
            coarse_bp=bp
        ))

    # Sort by composite significance: kinetic_z * l_pir
    breakpoints.sort(key=lambda b: b.kinetic_z * b.l_pir, reverse=True)

    if polish_ml:
        seq_mat = getattr(engine, "full_seq_matrix", getattr(engine, "seq_matrix", None))
        if seq_mat is not None:
            win = search_window if search_window is not None else max(150, min(1000, L // 20))
            taxa_map = {name: i for i, name in enumerate(taxa)}
            scale_coord = 3 if getattr(engine, "codon_aligned", False) else 1
            for b in breakpoints:
                if b.recombinant_taxon in taxa_map and b.parent_1 in taxa_map and b.parent_2 in taxa_map:
                    r_idx = taxa_map[b.recombinant_taxon]
                    p1_idx = taxa_map[b.parent_1]
                    p2_idx = taxa_map[b.parent_2]
                    raw_c_bp = b.coarse_bp if b.coarse_bp is not None else b.breakpoint_nt
                    pol = polish_breakpoint_ml(
                        seq_mat,
                        coarse_bp=raw_c_bp * scale_coord,
                        r_idx=r_idx,
                        p1_idx=p1_idx,
                        p2_idx=p2_idx,
                        search_window=win * scale_coord,
                        error_rate=error_rate,
                        taxa_names=taxa
                    )
                    b.polished_bp = pol.polished_bp // scale_coord if scale_coord > 1 else pol.polished_bp
                    b.ci_left = pol.ci_left // scale_coord if scale_coord > 1 else pol.ci_left
                    b.ci_right = pol.ci_right // scale_coord if scale_coord > 1 else pol.ci_right
                    b.plateau_width = pol.plateau_width // scale_coord if scale_coord > 1 else pol.plateau_width
                    b.log_likelihood_gain = pol.log_likelihood_gain
                    b.flanking_p1_site = pol.flanking_p1_site // scale_coord if scale_coord > 1 else pol.flanking_p1_site
                    b.flanking_p2_site = pol.flanking_p2_site // scale_coord if scale_coord > 1 else pol.flanking_p2_site
                    b.breakpoint_nt = b.polished_bp

    return breakpoints


def run_fda_recombination_screen(
    engine: PrefixDistanceEngine,
    taxa_names: List[str],
    k: int = 5,
    bin_size: int = 300,
    step: int = 30,
    min_kinetic_z: float = 3.0,
    min_pir: float = 0.10,
    polish_ml: bool = True
) -> Tuple[FDAResult, List[FDABreakpoint]]:
    """
    Executes a complete Functional Data Analysis (FDA) recombination screen.
    """
    fda_result = compute_fda_trajectories(
        engine=engine,
        taxa_names=taxa_names,
        k=k,
        bin_size=bin_size,
        step=step
    )
    breakpoints = extract_fda_breakpoints(
        fda_result=fda_result,
        engine=engine,
        min_kinetic_z=min_kinetic_z,
        min_pir=min_pir,
        polish_ml=polish_ml
    )
    return fda_result, breakpoints


def run_recursive_partition_fda_screen(
    engine: PrefixDistanceEngine,
    taxa_names: List[str],
    min_len: int = 40,
    max_depth: int = 5,
    min_z: float = 1.8,
    min_pir: float = 0.08,
    min_flank: Optional[int] = None,
    max_flank: Optional[int] = None,
    step: Optional[int] = None,
    frobenius_triage: bool = False,
    triage_threshold: float = 0.02,
    crossover_validation: bool = True,
    crossover_p_threshold: Optional[float] = None,
    min_informative_sites: int = 3,
    polish_ml: bool = True,
    search_window: Optional[int] = None,
    error_rate: float = 0.01,
    min_parent_dist: float = 1e-4
) -> List[FDABreakpoint]:
    """
    Recursive Binary Partitioning FDA (RP-FDA):
    Recursively segments genomic intervals by identifying optimal bilateral
    functional incongruence changepoints, refining boundaries down to single-base
    precision in O(L log L) time. Optionally applies Frobenius triage (156x speedup),
    crossover validation gates (eliminates false partition cascades), and
    data-driven Maximum Likelihood (ML) breakpoint polishing.
    """
    N = engine.N
    L = engine.num_units
    taxa = taxa_names
    k_eff = min(4, N - 1)
    seq_mat = getattr(engine, "full_seq_matrix", getattr(engine, "seq_matrix", None))
    
    # Adaptive scale-aware defaults
    cur_min_flank = min_flank if min_flank is not None else max(15, min(50, L // 20))
    cur_max_flank = max_flank if max_flank is not None else max(40, min(200, L // 5))
    cur_step = step if step is not None else max(5, min(25, cur_min_flank // 2))
    
    detected_bps: List[FDABreakpoint] = []

    def recursive_fda_split(s_start: int, s_end: int, depth: int = 0):
        if depth >= max_depth or (s_end - s_start) < min_len:
            return
            
        step_sz = cur_step
        flank = max(cur_min_flank, min(cur_max_flank, (s_end - s_start) // 4))
        if s_end - s_start < 2 * flank:
            return
            
        cps = np.arange(s_start + flank, s_end - flank + 1, step_sz)
        if len(cps) < 3:
            return
            
        z_curve = []
        top_tax = []
        for cp in cps:
            d1 = engine.query_distance_matrix(cp - flank, cp)
            d2 = engine.query_distance_matrix(cp, cp + flank)
            if np.max(d1) < 1e-4 or np.max(d2) < 1e-4:
                z_curve.append(0.0)
                top_tax.append(0)
                continue
            if frobenius_triage:
                f_diff = np.linalg.norm(d1 - d2)
                f_sum = np.linalg.norm(d1 + d2) + 1e-9
                if (f_diff / f_sum) < triage_threshold:
                    z_curve.append(0.0)
                    top_tax.append(0)
                    continue
            if N <= 4:
                # In small alignments (N <= 4), consensus degrees of freedom are insufficient
                # for empirical IQR Z-scores. Evaluate bilateral L-PIR handover directly:
                best_pir = 0.0
                best_t = 0
                for t_i in range(N):
                    tr = evaluate_triplets_for_taxon(d1, d2, t_i, min_parent_dist=min_parent_dist)
                    if tr and tr["pir"] > best_pir:
                        best_pir = tr["pir"]
                        best_t = t_i
                f_incong = np.linalg.norm(d1 - d2) / (np.linalg.norm(d1 + d2) + 1e-9)
                score = float(best_pir * 5.0 * (1.0 + f_incong)) if best_pir > 0.0 else 0.0
                z_curve.append(score)
                top_tax.append(best_t)
            else:
                z1 = compute_classical_mds(d1, k=k_eff)
                z2 = compute_classical_mds(d2, k=k_eff)
                _, res = align_procrustes(z1, z2)
                z_sc = compute_ghost_node_zscores(res)
                top_t = int(np.argmax(z_sc))
                z_curve.append(float(z_sc[top_t]))
                top_tax.append(top_t)
            
        z_curve_arr = np.array(z_curve)
        peaks, _ = find_peaks(z_curve_arr, height=min_z, prominence=0.5, distance=3)
        if len(peaks) == 0:
            return
            
        peak_order = sorted(peaks, key=lambda p: z_curve_arr[p], reverse=True)
        valid_splits = []
        for p in peak_order:
            bp = int(cps[p])
            t_idx = top_tax[p]
            d1 = engine.query_distance_matrix(bp - flank, bp)
            d2 = engine.query_distance_matrix(bp, bp + flank)
            
            cands = evaluate_triplets_for_taxon(
                d1, d2, t_idx, min_parent_dist=min_parent_dist, weight_by_divergence=True, top_k=5 if seq_mat is not None else 1
            )
            if not cands:
                continue
            if isinstance(cands, dict):
                cands = [cands]

            valid_trip = None
            for trip in cands:
                if trip["pir"] < min_pir:
                    continue
                if seq_mat is not None and crossover_validation:
                    p_crit = crossover_p_threshold if crossover_p_threshold is not None else min(0.005, 0.05 / max(1, N))
                    val_flank = max(flank, bp - s_start, s_end - bp)
                    scale_coord = 3 if getattr(engine, "codon_aligned", False) else 1
                    ok, _, _ = verify_crossover_support(
                        seq_mat, t_idx, trip["parent_left_idx"], trip["parent_right_idx"],
                        bp * scale_coord, flank=val_flank * scale_coord, min_informative=min_informative_sites, p_critical=p_crit
                    )
                    if ok:
                        valid_trip = trip
                        break
                else:
                    valid_trip = trip
                    break

            if valid_trip is None:
                continue

            trip = valid_trip
            if (bp - s_start) >= min_len and (s_end - bp) >= min_len:
                val_flank_left = min(bp - s_start, max(min_len, 3 * flank))
                val_flank_right = min(s_end - bp, max(min_len, 3 * flank))
                dl_full = engine.query_distance_matrix(bp - val_flank_left, bp)
                dr_full = engine.query_distance_matrix(bp, bp + val_flank_right)
                trip_full = evaluate_triplets_for_taxon(dl_full, dr_full, t_idx, min_parent_dist=min_parent_dist, weight_by_divergence=True)
                if not trip_full or trip_full["pir"] < (min_pir * 0.70):
                    continue
                p1 = trip["parent_left_idx"]
                p2 = trip["parent_right_idx"]
                best_b = bp
                min_loss = float("inf")
                for b in range(max(flank, bp - 20), min(L - flank, bp + 21), 2):
                    dl = engine.query_distance_matrix(b - flank, b)
                    dr = engine.query_distance_matrix(b, b + flank)
                    loss = dl[t_idx, p1] + dr[t_idx, p2]
                    if loss < min_loss:
                        min_loss = loss
                        best_b = b
                        
                detected_bps.append(FDABreakpoint(
                    breakpoint_nt=best_b,
                    recombinant_taxon=taxa[t_idx],
                    taxon_idx=t_idx,
                    kinetic_z=float(z_curve_arr[p]),
                    l_pir=float(trip["pir"]),
                    parent_1=taxa[p1],
                    parent_2=taxa[p2],
                    jump_magnitude=float(z_curve_arr[p]),
                    coarse_bp=best_b
                ))
                valid_splits.append(best_b)
                
        if not valid_splits:
            return
            
        split_pts = sorted(list(set([s_start] + valid_splits + [s_end])))
        for i in range(len(split_pts) - 1):
            left_sub = split_pts[i]
            right_sub = split_pts[i + 1]
            if (right_sub - left_sub) >= min_len:
                recursive_fda_split(left_sub, right_sub, depth + 1)

    recursive_fda_split(0, L, depth=0)
    
    # Deduplicate within scale-adaptive window
    min_dedup = max(10, min(50, L // 50))
    dedup: List[FDABreakpoint] = []
    for d in sorted(detected_bps, key=lambda x: x.breakpoint_nt):
        if not any(abs(d.breakpoint_nt - c.breakpoint_nt) < min_dedup for c in dedup):
            dedup.append(d)
            
    dedup.sort(key=lambda b: b.kinetic_z * b.l_pir, reverse=True)

    if polish_ml:
        seq_mat = getattr(engine, "full_seq_matrix", getattr(engine, "seq_matrix", None))
        if seq_mat is not None:
            win = search_window if search_window is not None else max(150, min(1000, L // 20))
            taxa_map = {name: i for i, name in enumerate(taxa)}
            scale_coord = 3 if getattr(engine, "codon_aligned", False) else 1
            for b in dedup:
                if b.recombinant_taxon in taxa_map and b.parent_1 in taxa_map and b.parent_2 in taxa_map:
                    r_idx = taxa_map[b.recombinant_taxon]
                    p1_idx = taxa_map[b.parent_1]
                    p2_idx = taxa_map[b.parent_2]
                    raw_c_bp = b.coarse_bp if b.coarse_bp is not None else b.breakpoint_nt
                    pol = polish_breakpoint_ml(
                        seq_mat,
                        coarse_bp=raw_c_bp * scale_coord,
                        r_idx=r_idx,
                        p1_idx=p1_idx,
                        p2_idx=p2_idx,
                        search_window=win * scale_coord,
                        error_rate=error_rate,
                        taxa_names=taxa
                    )
                    b.polished_bp = pol.polished_bp // scale_coord if scale_coord > 1 else pol.polished_bp
                    b.ci_left = pol.ci_left // scale_coord if scale_coord > 1 else pol.ci_left
                    b.ci_right = pol.ci_right // scale_coord if scale_coord > 1 else pol.ci_right
                    b.plateau_width = pol.plateau_width // scale_coord if scale_coord > 1 else pol.plateau_width
                    b.log_likelihood_gain = pol.log_likelihood_gain
                    b.flanking_p1_site = pol.flanking_p1_site // scale_coord if scale_coord > 1 else pol.flanking_p1_site
                    b.flanking_p2_site = pol.flanking_p2_site // scale_coord if scale_coord > 1 else pol.flanking_p2_site
                    b.breakpoint_nt = b.polished_bp

    return dedup


def run_multiscale_fda_screen(
    engine: PrefixDistanceEngine,
    taxa_names: List[str],
    scales: Optional[List[int]] = None,
    step: int = 20,
    min_z: float = 1.8,
    min_pir: float = 0.08,
    frobenius_triage: bool = False,
    triage_threshold: float = 0.02,
    polish_ml: bool = True,
    search_window: Optional[int] = None,
    error_rate: float = 0.01
) -> List[FDABreakpoint]:
    """
    Multiscale Bilateral Functional Data Analysis (MB-FDA):
    Evaluates continuous bilateral left-right functional incongruence across
    dyadic spatial scales, achieving sub-codon / fine single-base spatial resolution.
    Optionally applies Frobenius triage and ML breakpoint polishing.
    """
    if scales is None:
        scales = [100, 240, 500, 1000]

    N = engine.N
    L = engine.num_units
    taxa = taxa_names
    k_eff = min(4, N - 1)
    
    candidates_dict = {}
    for w in scales:
        flank = w // 2
        if L < 2 * flank:
            continue
        cps = np.arange(flank, L - flank + 1, step)
        if len(cps) < 3:
            continue
            
        z_curve = []
        tax_curve = []
        for cp in cps:
            d1 = engine.query_distance_matrix(cp - flank, cp)
            d2 = engine.query_distance_matrix(cp, cp + flank)
            if np.max(d1) < 1e-4 or np.max(d2) < 1e-4:
                z_curve.append(0.0)
                tax_curve.append(0)
                continue
            if frobenius_triage:
                f_diff = np.linalg.norm(d1 - d2)
                f_sum = np.linalg.norm(d1 + d2) + 1e-9
                if (f_diff / f_sum) < triage_threshold:
                    z_curve.append(0.0)
                    tax_curve.append(0)
                    continue
            z1 = compute_classical_mds(d1, k=k_eff)
            z2 = compute_classical_mds(d2, k=k_eff)
            _, res = align_procrustes(z1, z2)
            z_sc = compute_ghost_node_zscores(res)
            top_t = int(np.argmax(z_sc))
            z_curve.append(float(z_sc[top_t]))
            tax_curve.append(top_t)
            
        z_curve_arr = np.array(z_curve)
        peaks, _ = find_peaks(z_curve_arr, height=min_z, prominence=0.6, distance=4)
        for p in peaks:
            bp = int(cps[p])
            z_val = float(z_curve_arr[p])
            t_idx = tax_curve[p]
            if bp not in candidates_dict or z_val > candidates_dict[bp][1]:
                candidates_dict[bp] = (t_idx, z_val, w)
                
    detected_bps: List[FDABreakpoint] = []
    for bp, (t_idx, z_val, w) in candidates_dict.items():
        flank = min(200, w // 2, bp, L - bp)
        d1 = engine.query_distance_matrix(bp - flank, bp)
        d2 = engine.query_distance_matrix(bp, bp + flank)
        trip = evaluate_triplets_for_taxon(d1, d2, t_idx)
        if trip and trip["pir"] >= min_pir:
            p1 = trip["parent_left_idx"]
            p2 = trip["parent_right_idx"]
            best_b = bp
            min_loss = float("inf")
            for b in range(max(flank, bp - 20), min(L - flank, bp + 21), 2):
                dl = engine.query_distance_matrix(b - flank, b)
                dr = engine.query_distance_matrix(b, b + flank)
                loss = dl[t_idx, p1] + dr[t_idx, p2]
                if loss < min_loss:
                    min_loss = loss
                    best_b = b
                    
            detected_bps.append(FDABreakpoint(
                breakpoint_nt=best_b,
                recombinant_taxon=taxa[t_idx],
                taxon_idx=t_idx,
                kinetic_z=z_val,
                l_pir=float(trip["pir"]),
                parent_1=taxa[p1],
                parent_2=taxa[p2],
                jump_magnitude=z_val,
                coarse_bp=best_b
            ))
            
    # Deduplicate within 50 nt
    dedup: List[FDABreakpoint] = []
    for d in sorted(detected_bps, key=lambda x: x.breakpoint_nt):
        if not any(abs(d.breakpoint_nt - c.breakpoint_nt) < 50 for c in dedup):
            dedup.append(d)
            
    dedup.sort(key=lambda b: b.kinetic_z * b.l_pir, reverse=True)

    if polish_ml:
        seq_mat = getattr(engine, "full_seq_matrix", getattr(engine, "seq_matrix", None))
        if seq_mat is not None:
            win = search_window if search_window is not None else max(150, min(1000, L // 20))
            taxa_map = {name: i for i, name in enumerate(taxa)}
            for b in dedup:
                if b.recombinant_taxon in taxa_map and b.parent_1 in taxa_map and b.parent_2 in taxa_map:
                    r_idx = taxa_map[b.recombinant_taxon]
                    p1_idx = taxa_map[b.parent_1]
                    p2_idx = taxa_map[b.parent_2]
                    pol = polish_breakpoint_ml(
                        seq_mat,
                        coarse_bp=b.coarse_bp if b.coarse_bp is not None else b.breakpoint_nt,
                        r_idx=r_idx,
                        p1_idx=p1_idx,
                        p2_idx=p2_idx,
                        search_window=win,
                        error_rate=error_rate,
                        taxa_names=taxa
                    )
                    b.polished_bp = pol.polished_bp
                    b.ci_left = pol.ci_left
                    b.ci_right = pol.ci_right
                    b.plateau_width = pol.plateau_width
                    b.log_likelihood_gain = pol.log_likelihood_gain
                    b.flanking_p1_site = pol.flanking_p1_site
                    b.flanking_p2_site = pol.flanking_p2_site
                    b.breakpoint_nt = pol.polished_bp

    return dedup

