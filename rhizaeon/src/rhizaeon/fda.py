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

from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import fisher_exact

from rhizaeon.tensor import PrefixDistanceEngine
from rhizaeon.manifold import compute_classical_mds, align_procrustes, compute_ghost_node_zscores, compute_grubbs_effective_z
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
    nt_bp: Optional[int] = None
    nt_ci_left: Optional[int] = None
    nt_ci_right: Optional[int] = None
    nt_plateau_width: Optional[int] = None
    nt_flanking_p1: Optional[int] = None
    nt_flanking_p2: Optional[int] = None
    num_informative_sites: Optional[int] = None
    tier2_resolved: bool = False
    is_ambiguous: bool = False
    ambiguity_reason: Optional[str] = None
    is_hypermutation: bool = False
    mechanism: Optional[str] = None


class BreakpointList(list):
    """List of confirmed breakpoints with optional ambiguity queue attached."""
    ambiguous_candidates: List["FDABreakpoint"]


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


def evaluate_deaminase_hypermutation(
    seq_mat: np.ndarray,
    t_idx: int,
    bp: int,
    flank: int,
    min_mismatches: int = 4
) -> Dict[str, Any]:
    """
    Directional Deaminase Hypermutation Test.
    
    Examines the substitution spectrum of private mutations in taxon t_idx
    relative to its closest flanking reference across flanking and local windows:
      Left flank:  [max(0, bp - flank), bp)
      Right flank: [bp, min(L, bp + flank))
      Full window: [max(0, bp - flank), min(L, bp + flank))
    
    Computes the dominant transition fraction:
      f_deam = max(N_{C->T}, N_{G->A}, N_{A->G}) / sum(Mismatches)
      
    If mismatches >= min_mismatches (default 4), f_deam >= 0.85, and transversions <= 1:
      Performs a one-sided Binomial test against neutral expectation p_0 = 0.35.
      If significant (p < 0.005) and the mutations do not match synapomorphies of
      any other sampled taxon in the alignment, tag the candidate with:
        is_hypermutation = True, mechanism = "Deaminase Hypermutation"
    """
    from scipy.stats import binom
    from rhizaeon.tensor import MUT_CLASS

    N, L = seq_mat.shape
    empty_res = {
        "is_hypermutation": False,
        "mechanism": None,
        "dominant_transition": None,
        "f_deam": 0.0,
        "mismatches": 0,
        "transversions": 0,
        "p_value": 1.0,
        "ref_taxon_idx": None
    }
    if N < 2:
        return empty_res

    other_taxa = [j for j in range(N) if j != t_idx]

    intervals = []
    if bp > 0:
        intervals.append((max(0, bp - flank), bp))
    if bp < L:
        intervals.append((bp, min(L, bp + flank)))
    intervals.append((max(0, bp - flank), min(L, bp + flank)))

    for start, end in intervals:
        if end - start < min_mismatches:
            continue

        t_sub = seq_mat[t_idx, start:end]
        valid_t = (t_sub >= 0) & (t_sub < 4)
        if np.sum(valid_t) < min_mismatches:
            continue

        # Find closest flanking reference taxon among other taxa
        best_ref = None
        min_mismatch_rate = float("inf")

        for j in other_taxa:
            j_sub = seq_mat[j, start:end]
            valid_pair = valid_t & (j_sub >= 0) & (j_sub < 4)
            n_valid = int(np.sum(valid_pair))
            if n_valid < min_mismatches:
                continue
            n_diff = int(np.sum(valid_pair & (t_sub != j_sub)))
            rate = n_diff / float(n_valid)
            if rate < min_mismatch_rate:
                min_mismatch_rate = rate
                best_ref = j

        if best_ref is None:
            continue

        ref_sub = seq_mat[best_ref, start:end]
        valid_pair = valid_t & (ref_sub >= 0) & (ref_sub < 4)
        mismatch_mask = valid_pair & (t_sub != ref_sub)
        M = int(np.sum(mismatch_mask))

        if M < min_mismatches:
            continue

        m_pos = np.where(mismatch_mask)[0]
        ref_alleles = ref_sub[m_pos]
        t_alleles = t_sub[m_pos]

        # Nucleotide mapping: 0=A, 1=C, 2=G, 3=T
        n_c_to_t = int(np.sum((ref_alleles == 1) & (t_alleles == 3)))
        n_g_to_a = int(np.sum((ref_alleles == 2) & (t_alleles == 0)))
        n_a_to_g = int(np.sum((ref_alleles == 0) & (t_alleles == 2)))

        transitions = [
            ("C->T", n_c_to_t, (ref_alleles == 1) & (t_alleles == 3)),
            ("G->A", n_g_to_a, (ref_alleles == 2) & (t_alleles == 0)),
            ("A->G", n_a_to_g, (ref_alleles == 0) & (t_alleles == 2)),
        ]
        dom_name, dom_count, dom_mask = max(transitions, key=lambda x: x[1])

        # Transversions: MUT_CLASS == 2
        n_tv = int(np.sum(MUT_CLASS[ref_alleles, t_alleles] == 2))

        f_deam = float(dom_count) / float(M)

        if M >= min_mismatches and f_deam >= 0.85 and n_tv <= 1 and dom_count >= min_mismatches:
            # One-sided Binomial test against neutral expectation p_0 = 0.35
            p_val = float(binom.sf(dom_count - 1, M, 0.35))

            if p_val < 0.005:
                # Check if mutations match synapomorphies of any other sampled taxon
                dom_m_pos = m_pos[dom_mask]
                global_pos = start + dom_m_pos
                t_dom_alleles = seq_mat[t_idx, global_pos]

                is_synapomorphy = False
                for other in other_taxa:
                    if other == best_ref:
                        continue
                    other_alleles = seq_mat[other, global_pos]
                    valid_other = (other_alleles >= 0) & (other_alleles < 4)
                    matches_other = int(np.sum(valid_other & (other_alleles == t_dom_alleles)))
                    if matches_other >= 2 and matches_other >= int(np.ceil(0.60 * len(dom_m_pos))):
                        is_synapomorphy = True
                        break

                if not is_synapomorphy:
                    return {
                        "is_hypermutation": True,
                        "mechanism": "Deaminase Hypermutation",
                        "dominant_transition": dom_name,
                        "f_deam": float(f_deam),
                        "mismatches": int(M),
                        "transversions": int(n_tv),
                        "p_value": float(p_val),
                        "ref_taxon_idx": int(best_ref)
                    }

    return empty_res


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
                    b.flanking_p1_site = pol.flanking_p1_site // scale_coord if scale_coord > 1 and pol.flanking_p1_site is not None else pol.flanking_p1_site
                    b.flanking_p2_site = pol.flanking_p2_site // scale_coord if scale_coord > 1 and pol.flanking_p2_site is not None else pol.flanking_p2_site
                    b.breakpoint_nt = b.polished_bp

    # Directional Deaminase Hypermutation Test
    if seq_mat is not None:
        scale_coord = 3 if getattr(engine, "codon_aligned", False) else 1
        for b in breakpoints:
            bp_nt = (b.polished_bp if b.polished_bp is not None else b.breakpoint_nt) * scale_coord
            flank_nt = min(200 * scale_coord, max(50 * scale_coord, bp_nt, (L * scale_coord) - bp_nt))
            hyp_res = evaluate_deaminase_hypermutation(
                seq_mat=seq_mat,
                t_idx=b.taxon_idx,
                bp=bp_nt,
                flank=flank_nt,
                min_mismatches=4
            )
            if hyp_res.get("is_hypermutation", False):
                b.is_hypermutation = True
                b.mechanism = "Deaminase Hypermutation"

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
    min_len: Optional[Union[int, str]] = "auto",
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
    crossover validation gates (eliminates false partition cascades), finite-sample
    Thompson/Grubbs scaling for small cohorts (N <= 5), and data-driven Maximum
    Likelihood (ML) breakpoint polishing.
    """
    N = engine.N
    L = engine.num_units
    taxa = taxa_names
    if N < 3:
        return BreakpointList()
    k_eff = min(3, N - 1)
    seq_mat = getattr(engine, "full_seq_matrix", getattr(engine, "seq_matrix", None))

    # Dynamic Poisson Information Limit:
    # If min_len is None or min_len == "auto", compute mean pairwise divergence D_bar across sequence:
    # L_min = max(15, ceil(3.0 / max(D_bar, 1e-4)))
    if min_len is None or min_len == "auto":
        D_all = engine.query_distance_matrix(0, L)
        if N <= 1:
            d_mean = 0.05
        else:
            triu = np.triu_indices(N, k=1)
            d_mean = float(np.mean(D_all[triu])) if len(triu[0]) > 0 else 0.05
        eff_min_len = max(15, int(np.ceil(3.0 / max(d_mean, 1e-4))))
    elif isinstance(min_len, str) and min_len.isdigit():
        eff_min_len = int(min_len)
    else:
        eff_min_len = int(min_len)

    # Finite-Sample Thompson/Grubbs Normalization:
    # In any sample of size N, the maximum standardized residual from a sample mean
    # is mathematically bounded by (N - 1) / sqrt(N) (Thompson 1935, Grubbs 1950).
    # Applies exact Student t critical values for Grubbs test across small to medium cohorts.
    eff_min_z = compute_grubbs_effective_z(N, nominal_z=min_z, alpha=0.005)
    
    # Information-aware adaptive flanking bounds
    cur_min_flank = min_flank if min_flank is not None else 30
    cur_max_flank = max_flank if max_flank is not None else 250
    cur_step = step if step is not None else max(5, cur_min_flank // 3)

    # Precompute segregating site cumulative array for Symmetrical Fisher Information Flank Balancing
    scale_coord = 3 if getattr(engine, "codon_aligned", False) else 1
    cum_seg = None
    if seq_mat is not None:
        valid_mask = (seq_mat >= 0) & (seq_mat < 4)
        clean_mat = np.where(valid_mask, seq_mat, -1)
        max_val = np.max(clean_mat, axis=0)
        min_mat = np.where(valid_mask, seq_mat, 99)
        min_val = np.min(min_mat, axis=0)
        seg_mask = (max_val > min_val) & (min_val >= 0) & (max_val <= 3)
        cum_seg = np.zeros(len(seg_mask) + 1, dtype=np.int32)
        cum_seg[1:] = np.cumsum(seg_mask.astype(np.int32))
    
    detected_bps: List[FDABreakpoint] = []
    ambiguous_bps: List[FDABreakpoint] = []

    def recursive_fda_split(s_start: int, s_end: int, depth: int = 0):
        if depth >= max_depth or (s_end - s_start) < eff_min_len:
            return
            
        scale_coord = 3 if getattr(engine, "codon_aligned", False) else 1
        step_sz = cur_step
        if (s_end - s_start) < 2 * cur_min_flank:
            return
            
        cps = np.arange(s_start + cur_min_flank, s_end - cur_min_flank + 1, step_sz)
        if len(cps) < 2:
            return
            
        z_curve = []
        valid_cps = []
        candidates_at_cp = []
        for cp in cps:
            avail = min(cp - s_start, s_end - cp)
            if avail < cur_min_flank:
                continue
            flank = min(cur_max_flank, avail)

            # Symmetrical Fisher Information Flank Balancing:
            # Verify that both left flank [cp - flank, cp) and right flank [cp, cp + flank)
            # possess >= min_informative_sites (default 3) informative segregating sites
            if cum_seg is not None:
                l_s = max(0, (cp - flank) * scale_coord)
                l_e = max(0, cp * scale_coord)
                r_s = max(0, cp * scale_coord)
                r_e = min(len(cum_seg) - 1, (cp + flank) * scale_coord)
                if l_s >= l_e or r_s >= r_e:
                    continue
                n_left = int(cum_seg[l_e] - cum_seg[l_s])
                n_right = int(cum_seg[r_e] - cum_seg[r_s])
                if (n_left + n_right) < min_informative_sites or min(n_left, n_right) < 1:
                    continue

            d1 = engine.query_distance_matrix(cp - flank, cp)
            d2 = engine.query_distance_matrix(cp, cp + flank)
            if np.max(d1) < 1e-4 or np.max(d2) < 1e-4:
                continue
            if frobenius_triage:
                # Max-row discrepancy scales as O(1) with respect to N, immune to cohort dilution
                row_diffs = np.linalg.norm(d1 - d2, axis=1)
                row_sums = np.linalg.norm(d1 + d2, axis=1) + 1e-9
                max_row_ratio = float(np.max(row_diffs / row_sums))
                if max_row_ratio < triage_threshold:
                    continue
            valid_cps.append((cp, flank))
            k_eff = min(4, N - 1)
            z1 = compute_classical_mds(d1, k=k_eff)
            z2 = compute_classical_mds(d2, k=k_eff)
            _, res = align_procrustes(z1, z2)
            mean_div = float(0.5 * (np.mean(d1) + np.mean(d2)))
            z_sc = compute_ghost_node_zscores(res, mean_divergence=mean_div, window_len=flank * scale_coord)
            top_indices = np.argsort(-z_sc)[:min(5, N)]
            top_taxa = [(int(t), float(z_sc[t])) for t in top_indices if z_sc[t] >= eff_min_z * 0.75]
            if not top_taxa:
                top_taxa = [(int(top_indices[0]), float(z_sc[top_indices[0]]))]
            z_curve.append(float(np.max(z_sc)))
            candidates_at_cp.append(top_taxa)
            
        if len(z_curve) < 3:
            return

        z_curve_arr = np.array(z_curve)
        prom_threshold = min(0.5, eff_min_z * 0.25) if N <= 5 else 0.5
        peaks, _ = find_peaks(z_curve_arr, height=eff_min_z, prominence=prom_threshold, distance=3)
        if len(peaks) == 0:
            return
            
        peak_order = sorted(peaks, key=lambda p: z_curve_arr[p], reverse=True)
        best_split_call = None
        best_split_score = -1.0

        for p in peak_order:
            bp, flank = valid_cps[p]
            d1 = engine.query_distance_matrix(bp - flank, bp)
            d2 = engine.query_distance_matrix(bp, bp + flank)

            cand_taxa = candidates_at_cp[p]

            for t_idx, t_z in cand_taxa:
                if t_z < eff_min_z:
                    continue
                if hasattr(engine, "query_coverage"):
                    cov = engine.query_coverage(max(0, bp - flank), min(L, bp + flank))
                    if cov[t_idx] < 0.15:
                        continue
                cands = evaluate_triplets_for_taxon(
                    d1, d2, t_idx, min_parent_dist=min_parent_dist, weight_by_divergence=True, top_k=5 if seq_mat is not None else 1
                )
                if not cands:
                    ambiguous_bps.append(FDABreakpoint(
                        breakpoint_nt=bp,
                        recombinant_taxon=taxa[t_idx],
                        taxon_idx=t_idx,
                        kinetic_z=float(t_z),
                        l_pir=0.0,
                        parent_1="Ghost",
                        parent_2="Ghost",
                        jump_magnitude=float(t_z),
                        coarse_bp=bp,
                        is_ambiguous=True,
                        ambiguity_reason="ghost_donor_no_sampled_triplet"
                    ))
                    continue
                if isinstance(cands, dict):
                    cands = [cands]

                for trip in cands:
                    # Allow 30% margin at preliminary coarse changepoint because uncentered flanks straddle the junction
                    if trip["pir"] < (min_pir * 0.70):
                        continue
                    p1 = trip["parent_left_idx"]
                    p2 = trip["parent_right_idx"]

                    # Information-aware adaptive coordinate refinement around coarse changepoint
                    search_r = min(flank // 2, 80)
                    best_b = bp
                    min_loss = float("inf")
                    for b in range(max(cur_min_flank, bp - search_r), min(L - cur_min_flank, bp + search_r + 1), 2):
                        f_ref = min(flank, b - s_start, s_end - b)
                        if f_ref < cur_min_flank:
                            continue
                        dl = engine.query_distance_matrix(b - f_ref, b)
                        dr = engine.query_distance_matrix(b, b + f_ref)
                        loss = dl[t_idx, p1] + dr[t_idx, p2]
                        if loss < min_loss:
                            min_loss = loss
                            best_b = b

                    # Re-evaluate PIR at centered refined coordinate best_b
                    f_ref_val = min(flank, best_b - s_start, s_end - best_b)
                    dl_ref = engine.query_distance_matrix(best_b - f_ref_val, best_b)
                    dr_ref = engine.query_distance_matrix(best_b, best_b + f_ref_val)
                    ref_trip = evaluate_triplets_for_taxon(dl_ref, dr_ref, t_idx, min_parent_dist=min_parent_dist, weight_by_divergence=True)
                    if ref_trip is None or ref_trip["pir"] < min_pir:
                        continue

                    ok = True
                    if seq_mat is not None and crossover_validation:
                        p_crit = crossover_p_threshold if crossover_p_threshold is not None else 0.01
                        ok, _, _ = verify_crossover_support(
                            seq_mat, t_idx, p1, p2,
                            best_b * scale_coord, flank=f_ref_val * scale_coord, min_informative=min_informative_sites, p_critical=p_crit
                        )
                        if not ok:
                            # Fallback check at original coarse changepoint
                            ok, _, _ = verify_crossover_support(
                                seq_mat, t_idx, p1, p2,
                                bp * scale_coord, flank=min(flank, bp - s_start, s_end - bp) * scale_coord, min_informative=min_informative_sites, p_critical=p_crit
                            )

                    if ok:
                        score = float(t_z * ref_trip["pir"])
                        if score > best_split_score:
                            best_split_score = score
                            best_split_call = (best_b, FDABreakpoint(
                                breakpoint_nt=best_b,
                                recombinant_taxon=taxa[t_idx],
                                taxon_idx=t_idx,
                                kinetic_z=float(t_z),
                                l_pir=float(ref_trip["pir"]),
                                parent_1=taxa[p1],
                                parent_2=taxa[p2],
                                jump_magnitude=float(t_z),
                                coarse_bp=best_b,
                                nt_bp=best_b * scale_coord,
                                nt_ci_left=best_b * scale_coord,
                                nt_ci_right=best_b * scale_coord
                            ))
                            break

        if best_split_call is None:
            return

        split_bp, bp_record = best_split_call
        detected_bps.append(bp_record)

        # Recursive Binary Partitioning on left and right subsegments
        if (split_bp - s_start) >= eff_min_len:
            recursive_fda_split(s_start, split_bp, depth + 1)
        if (s_end - split_bp) >= eff_min_len:
            recursive_fda_split(split_bp, s_end, depth + 1)

    recursive_fda_split(0, L, depth=0)
    
    # Deduplicate candidate breakpoints, prioritizing higher composite statistical evidence (kinetic_z * l_pir)
    # Deduplicate within 40 nt per recombinant lineage to avoid cross-lineage masking
    dedup: BreakpointList = BreakpointList()
    for d in sorted(detected_bps, key=lambda b: b.kinetic_z * b.l_pir, reverse=True):
        duplicate = False
        for c in dedup:
            if abs(d.breakpoint_nt - c.breakpoint_nt) < 25 and d.recombinant_taxon == c.recombinant_taxon:
                # Same parental transition direction indicates duplicate call of the same boundary
                if (d.parent_1 == c.parent_1 and d.parent_2 == c.parent_2) or (d.parent_1 == "Ghost" and c.parent_1 == "Ghost"):
                    duplicate = True
                    break
        if not duplicate:
            dedup.append(d)
            
    dedup.sort(key=lambda b: b.kinetic_z * b.l_pir, reverse=True)

    # Deduplicate ambiguous candidates by proximity (within eff_min_len) and filter out any covered by confirmed
    dedup_ambiguous: List[FDABreakpoint] = []
    for amb in sorted(ambiguous_bps, key=lambda b: b.kinetic_z, reverse=True):
        near_confirmed = any(
            amb.recombinant_taxon == c.recombinant_taxon and abs(amb.breakpoint_nt - c.breakpoint_nt) < max(20, eff_min_len)
            for c in dedup
        )
        if near_confirmed:
            continue
        near_amb = any(
            amb.recombinant_taxon == a.recombinant_taxon and abs(amb.breakpoint_nt - a.breakpoint_nt) < max(15, eff_min_len // 2)
            for a in dedup_ambiguous
        )
        if not near_amb:
            dedup_ambiguous.append(amb)

    dedup.ambiguous_candidates = dedup_ambiguous

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
                    b.flanking_p1_site = pol.flanking_p1_site // scale_coord if scale_coord > 1 and pol.flanking_p1_site is not None else pol.flanking_p1_site
                    b.flanking_p2_site = pol.flanking_p2_site // scale_coord if scale_coord > 1 and pol.flanking_p2_site is not None else pol.flanking_p2_site
                    b.breakpoint_nt = b.polished_bp
                    b.nt_bp = pol.polished_bp
                    b.nt_ci_left = pol.ci_left
                    b.nt_ci_right = pol.ci_right
                    b.nt_plateau_width = pol.plateau_width
                    b.nt_flanking_p1 = pol.flanking_p1_site
                    b.nt_flanking_p2 = pol.flanking_p2_site
                    b.num_informative_sites = pol.num_informative_sites

            # Merge overlapping likelihood plateaus for identical recombinant taxon and parental transition
            merged_plateaus: List[FDABreakpoint] = []
            for b in sorted(dedup, key=lambda x: x.breakpoint_nt):
                merged_into = None
                for m in merged_plateaus:
                    if m.recombinant_taxon == b.recombinant_taxon and m.parent_1 == b.parent_1 and m.parent_2 == b.parent_2:
                        if m.ci_left is not None and m.ci_right is not None and b.ci_left is not None and b.ci_right is not None:
                            if max(m.ci_left, b.ci_left) <= min(m.ci_right, b.ci_right):
                                merged_into = m
                                break
                if merged_into is not None:
                    merged_into.ci_left = min(merged_into.ci_left, b.ci_left)
                    merged_into.ci_right = max(merged_into.ci_right, b.ci_right)
                    merged_into.plateau_width = merged_into.ci_right - merged_into.ci_left
                    merged_into.polished_bp = (merged_into.ci_left + merged_into.ci_right) // 2
                    merged_into.breakpoint_nt = merged_into.polished_bp
                    merged_into.log_likelihood_gain = max(merged_into.log_likelihood_gain or 0.0, b.log_likelihood_gain or 0.0)
                    merged_into.kinetic_z = max(merged_into.kinetic_z, b.kinetic_z)
                    merged_into.l_pir = max(merged_into.l_pir, b.l_pir)
                    if scale_coord == 1:
                        merged_into.nt_bp = merged_into.polished_bp
                        merged_into.nt_ci_left = merged_into.ci_left
                        merged_into.nt_ci_right = merged_into.ci_right
                        merged_into.nt_plateau_width = merged_into.plateau_width
                    else:
                        merged_into.nt_bp = merged_into.polished_bp * scale_coord
                        merged_into.nt_ci_left = merged_into.ci_left * scale_coord
                        merged_into.nt_ci_right = merged_into.ci_right * scale_coord
                        merged_into.nt_plateau_width = merged_into.plateau_width * scale_coord
                else:
                    merged_plateaus.append(b)
            dedup = merged_plateaus
    else:
        scale_coord = 3 if getattr(engine, "codon_aligned", False) else 1
        for b in dedup:
            if b.nt_bp is None:
                b.nt_bp = b.breakpoint_nt * scale_coord
                b.nt_ci_left = (b.ci_left if b.ci_left is not None else b.breakpoint_nt) * scale_coord
                b.nt_ci_right = (b.ci_right if b.ci_right is not None else b.breakpoint_nt) * scale_coord
                b.nt_plateau_width = (b.plateau_width if b.plateau_width is not None else 0) * scale_coord
        for amb in getattr(dedup, "ambiguous_candidates", []):
            if amb.nt_bp is None:
                amb.nt_bp = amb.breakpoint_nt * scale_coord
                amb.nt_ci_left = (amb.ci_left if amb.ci_left is not None else amb.breakpoint_nt) * scale_coord
                amb.nt_ci_right = (amb.ci_right if amb.ci_right is not None else amb.breakpoint_nt) * scale_coord
                amb.nt_plateau_width = (amb.plateau_width if amb.plateau_width is not None else 0) * scale_coord

    # Directional Deaminase Hypermutation Test
    if seq_mat is not None:
        for b in dedup:
            raw_c_bp = b.coarse_bp if b.coarse_bp is not None else b.breakpoint_nt
            bp_nt = (b.polished_bp if b.polished_bp is not None else raw_c_bp) * scale_coord
            flank_nt = min(200 * scale_coord, max(50 * scale_coord, bp_nt, (L * scale_coord) - bp_nt))
            hyp_res = evaluate_deaminase_hypermutation(
                seq_mat=seq_mat,
                t_idx=b.taxon_idx,
                bp=bp_nt,
                flank=flank_nt,
                min_mismatches=4
            )
            if hyp_res.get("is_hypermutation", False):
                b.is_hypermutation = True
                b.mechanism = "Deaminase Hypermutation"

        for amb in getattr(dedup, "ambiguous_candidates", []):
            raw_c_bp = amb.coarse_bp if amb.coarse_bp is not None else amb.breakpoint_nt
            bp_nt = raw_c_bp * scale_coord
            flank_nt = min(200 * scale_coord, max(50 * scale_coord, bp_nt, (L * scale_coord) - bp_nt))
            hyp_res = evaluate_deaminase_hypermutation(
                seq_mat=seq_mat,
                t_idx=amb.taxon_idx,
                bp=bp_nt,
                flank=flank_nt,
                min_mismatches=4
            )
            if hyp_res.get("is_hypermutation", False):
                amb.is_hypermutation = True
                amb.mechanism = "Deaminase Hypermutation"

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
                row_diffs = np.linalg.norm(d1 - d2, axis=1)
                row_sums = np.linalg.norm(d1 + d2, axis=1) + 1e-9
                max_row_ratio = float(np.max(row_diffs / row_sums))
                if max_row_ratio < triage_threshold:
                    z_curve.append(0.0)
                    tax_curve.append(0)
                    continue
            z1 = compute_classical_mds(d1, k=k_eff)
            z2 = compute_classical_mds(d2, k=k_eff)
            _, res = align_procrustes(z1, z2)
            mean_div = float(0.5 * (np.mean(d1) + np.mean(d2)))
            scale_coord = 3 if getattr(engine, "codon_aligned", False) else 1
            z_sc = compute_ghost_node_zscores(res, mean_divergence=mean_div, window_len=flank * scale_coord)
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

    scale_coord = 3 if getattr(engine, "codon_aligned", False) else 1
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
                    b.flanking_p1_site = pol.flanking_p1_site // scale_coord if scale_coord > 1 and pol.flanking_p1_site is not None else pol.flanking_p1_site
                    b.flanking_p2_site = pol.flanking_p2_site // scale_coord if scale_coord > 1 and pol.flanking_p2_site is not None else pol.flanking_p2_site
                    b.breakpoint_nt = b.polished_bp
                    b.nt_bp = pol.polished_bp
                    b.nt_ci_left = pol.ci_left
                    b.nt_ci_right = pol.ci_right
                    b.nt_plateau_width = pol.plateau_width
                    b.nt_flanking_p1 = pol.flanking_p1_site
                    b.nt_flanking_p2 = pol.flanking_p2_site
                    b.num_informative_sites = pol.num_informative_sites
    else:
        for b in dedup:
            if b.nt_bp is None:
                b.nt_bp = b.breakpoint_nt * scale_coord
                b.nt_ci_left = (b.ci_left if b.ci_left is not None else b.breakpoint_nt) * scale_coord
                b.nt_ci_right = (b.ci_right if b.ci_right is not None else b.breakpoint_nt) * scale_coord
                b.nt_plateau_width = (b.plateau_width if b.plateau_width is not None else 0) * scale_coord

    # Directional Deaminase Hypermutation Test
    if seq_mat is not None:
        for b in dedup:
            raw_c_bp = b.coarse_bp if b.coarse_bp is not None else b.breakpoint_nt
            bp_nt = (b.polished_bp if b.polished_bp is not None else raw_c_bp) * scale_coord
            flank_nt = min(200 * scale_coord, max(50 * scale_coord, bp_nt, (L * scale_coord) - bp_nt))
            hyp_res = evaluate_deaminase_hypermutation(
                seq_mat=seq_mat,
                t_idx=b.taxon_idx,
                bp=bp_nt,
                flank=flank_nt,
                min_mismatches=4
            )
            if hyp_res.get("is_hypermutation", False):
                b.is_hypermutation = True
                b.mechanism = "Deaminase Hypermutation"

    return dedup

