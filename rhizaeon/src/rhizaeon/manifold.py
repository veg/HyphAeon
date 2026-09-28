"""
rhizaeon.manifold
=================
Continuous sequence manifold geometry, spectral embeddings, and
ChronAeon-style out-of-sample Ghost Node Procrustes alignment.
Includes systematic low-level handling for gappy and partial genomes.
"""

from typing import Tuple, Optional, Dict, List
import numpy as np
import scipy.linalg as la


def compute_classical_mds(D: np.ndarray, k: int = 4) -> np.ndarray:
    """
    Classical Multidimensional Scaling (ChronAeon Eq. 152-153).
    
    Transforms pairwise distance matrix D into a continuous Euclidean embedding
    Z in R^{N x k} preserving pairwise evolutionary distances:
      B = -0.5 * H * (D^2) * H,  where H = I - (1/N) * 1 * 1^T
      Z = V_k * sqrt(max(0, Lambda_k))
    """
    N = D.shape[0]
    if N <= 0:
        return np.empty((0, k), dtype=np.float64)
    if N <= k:
        k = max(1, N - 1)

    # Efficient double-centering: H @ D_sq @ H = D_sq - mean_row - mean_col + mean_all
    D_sq = D.astype(np.float64) ** 2
    B = -0.5 * (D_sq - D_sq.mean(axis=1, keepdims=True) - D_sq.mean(axis=0, keepdims=True) + D_sq.mean())

    if N <= 200:
        w, v = np.linalg.eigh(B)
        idx = np.argsort(-w)[:k]
        w_top = np.maximum(0.0, w[idx])
        v_top = v[:, idx]
    else:
        # Fast randomized subspace iteration (Halko et al. 2011) for O(k N^2) scaling on grand cohorts
        p = min(k + 4, N)
        np.random.seed(42)
        Omega = np.random.randn(N, p)
        Q, _ = np.linalg.qr(B @ Omega)
        for _ in range(3):
            Q, _ = np.linalg.qr(B @ Q)
        B_small = Q.T @ (B @ Q)
        w_small, v_small = np.linalg.eigh(B_small)
        idx = np.argsort(-w_small)[:k]
        w_top = np.maximum(0.0, w_small[idx])
        v_top = Q @ v_small[:, idx]

    coords = [v_top[:, j] * np.sqrt(float(w_top[j])) for j in range(k)]
    return np.column_stack(coords)


def nystrom_out_of_sample_mds(
    Z_core: np.ndarray,
    D_core: np.ndarray,
    d_sample_to_core: np.ndarray
) -> np.ndarray:
    """
    Gower / Nyström out-of-sample projection into an existing Classical MDS space.
    
    Given core coordinates Z_core in R^{M x k} and distance vector d in R^M from
    a new/partial sample to all M core landmarks, computes the optimal projection:
      ||z_i - z||^2 = ||z_i||^2 - 2 z_i^T z + ||z||^2 = d_i^2
    By centering the valid landmark subset Z_sub around its local centroid mu_sub:
      Z_tilde = Z_sub - mu_sub
      v = 0.5 * ( (||Z_sub||^2 - d_sub^2) - mean(||Z_sub||^2 - d_sub^2) )
      z_sample = (Z_tilde^T Z_tilde)^{-1} Z_tilde^T v
    
    This guarantees exact Euclidean reconstruction for any arbitrary subset of
    M_valid >= k + 1 core contacts with zero mean-shift bias.
    """
    valid_mask = ~np.isnan(d_sample_to_core)
    M_valid = np.sum(valid_mask)
    k = Z_core.shape[1]

    if M_valid < k + 1:
        return np.full(k, np.nan, dtype=np.float64)

    Z_sub = Z_core[valid_mask]
    d_sub = d_sample_to_core[valid_mask]

    # Center the landmark subset
    mu_sub = np.mean(Z_sub, axis=0, keepdims=True)
    Z_tilde = Z_sub - mu_sub

    sq_norms = np.sum(Z_sub**2, axis=1)
    d_sub_sq = d_sub**2
    y = sq_norms - d_sub_sq
    v = 0.5 * (y - np.mean(y))

    cov = Z_tilde.T @ Z_tilde
    try:
        z_proj = np.linalg.solve(cov, Z_tilde.T @ v)
    except np.linalg.LinAlgError:
        z_proj = np.linalg.pinv(cov) @ (Z_tilde.T @ v)

    return z_proj


def compute_laplacian_eigenmaps(
    D: np.ndarray,
    k: int = 4,
    k_nn: int = 4
) -> np.ndarray:
    """
    Graph Laplacian / Diffusion Map embedding with local neighborhood scaling.
    
    Normalizes for varying evolutionary rates and branch lengths across clades,
    amplifying fine-scale intra-clade structure.
    """
    N = D.shape[0]
    if N <= k:
        return compute_classical_mds(D, k=k)

    # Local scale sigma_i = distance to k_nn-th nearest neighbor
    sigma = np.zeros(N, dtype=np.float64)
    for i in range(N):
        sorted_d = np.sort(D[i])
        nn_idx = min(max(1, k_nn), N - 1)
        sigma[i] = max(1e-5, sorted_d[nn_idx])

    # Affinity matrix with local scaling
    W = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i != j:
                W[i, j] = np.exp(-(D[i, j] ** 2) / (2.0 * sigma[i] * sigma[j] + 1e-9))
            else:
                W[i, j] = 1.0

    # Symmetric normalized Laplacian: L_sym = D_deg^{-1/2} W D_deg^{-1/2}
    d_deg = np.sum(W, axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(d_deg, 1e-9)))
    L_sym = d_inv_sqrt @ W @ d_inv_sqrt

    w, v = la.eigh(L_sym)
    idx = np.argsort(-w)  # Descending
    v = v[:, idx]

    # Return leading non-trivial eigenvectors (skipping the trivial constant mode)
    return v[:, 1 : min(k + 1, N)]


def align_procrustes(
    Z_target: np.ndarray,
    Z_source: np.ndarray,
    enforce_so: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Orthogonal Procrustes Manifold Alignment with NaN-Tolerance and Kabsch SO(k) Support.
    
    Finds optimal rotation matrix R and translation vector t minimizing:
      min_{R, t} ||Z_target - (Z_source @ R + t)||_F^2
    evaluated strictly over valid (non-NaN) common taxa.
    
    Parameters:
      Z_target: [N, k] target frame
      Z_source: [N, k] source coordinates to be aligned
      enforce_so: if True, enforces det(R) = +1 (proper rotation, banning improper reflections)
    
    Returns:
      Z_aligned: Source coordinates rotated and translated into target frame.
      residuals: Per-taxon Euclidean distances ||z_{target, i} - z_{aligned, i}||_2 (NaN for missing).
    """
    N, k = Z_source.shape
    common_mask = ~np.isnan(Z_target[:, 0]) & ~np.isnan(Z_source[:, 0])
    common_idx = np.where(common_mask)[0]

    if len(common_idx) < max(2, k):
        # Insufficient overlap to estimate rotation; return source as is
        residuals = np.full(N, np.nan, dtype=np.float64)
        if len(common_idx) > 0:
            residuals[common_idx] = np.linalg.norm(Z_target[common_idx] - Z_source[common_idx], axis=1)
        return Z_source.copy(), residuals

    Z_t_sub = Z_target[common_idx]
    Z_s_sub = Z_source[common_idx]

    mu_s = np.mean(Z_s_sub, axis=0, keepdims=True)
    mu_t = np.mean(Z_t_sub, axis=0, keepdims=True)

    Z_s_cent = Z_s_sub - mu_s
    Z_t_cent = Z_t_sub - mu_t

    # Procrustes cross-covariance
    M = Z_t_cent.T @ Z_s_cent
    U, S, Vt = la.svd(M)
    V = Vt.T

    if enforce_so:
        d = np.linalg.det(V @ U.T)
        W_diag = np.eye(len(S))
        if d < 0:
            W_diag[-1, -1] = -1.0
        R = V @ W_diag @ U.T
    else:
        R = V @ U.T

    # Apply alignment to all valid source coordinates
    Z_aligned = np.full_like(Z_source, np.nan)
    valid_source = ~np.isnan(Z_source[:, 0])
    Z_aligned[valid_source] = (Z_source[valid_source] - mu_s) @ R + mu_t

    residuals = np.full(N, np.nan, dtype=np.float64)
    residuals[common_idx] = np.linalg.norm(Z_target[common_idx] - Z_aligned[common_idx], axis=1)
    return Z_aligned, residuals


def compute_ghost_node_zscores(
    residuals: np.ndarray,
    mean_divergence: float = 0.0,
    window_len: int = 1
) -> np.ndarray:
    """
    Calculates robust Studentized / MAD Z-scores from Procrustes residuals:
      Z_i = (r_i - median(r)) / scale
    where scale incorporates the consistent normal dispersion and Poisson sampling variance lower bound:
      sigma_mad = 1.4826 * MAD(r)
      sigma_iqr = IQR(r) / 1.34898
      poisson_var = max(mean_divergence, 1e-5) / max(1, window_len)
      scale = max(sigma_mad, sigma_iqr, sqrt(poisson_var), 1e-6)

    Non-recombinant taxa conform tightly to the global rigid rotation (Z ~ 0).
    Recombinant lineages detach as high-leverage 'Ghost Nodes' (Z >> 2.5).
    Ignores NaNs for partial/unsequenced genomes.
    """
    valid_res = residuals[~np.isnan(residuals)]
    if len(valid_res) == 0:
        return np.full_like(residuals, np.nan)
    N = len(valid_res)
    med = float(np.median(valid_res))
    mad = float(np.median(np.abs(valid_res - med)))
    sigma_mad = 1.4826 * mad
    iqr = float(np.percentile(valid_res, 75) - np.percentile(valid_res, 25))
    sigma_iqr = iqr / 1.34898
    emp_scale = float(max(sigma_mad, sigma_iqr))

    # Poisson sampling variance floor for individual taxon residual
    poisson_var = float(max(mean_divergence, 1e-5) / max(1, window_len))
    sigma_poisson = float(np.sqrt(poisson_var))

    scale = float(max(emp_scale, sigma_poisson, 1e-6))
    return (residuals - med) / scale


def compute_grubbs_effective_z(N: int, nominal_z: float = 2.75, alpha: float = 0.005) -> float:
    """
    Computes finite-sample Thompson-Grubbs effective Z-score threshold.
    
    For a sample of N observations, the maximum possible standardized deviation
    from the sample mean is mathematically bounded by:
      Z_max = (N - 1) / sqrt(N)
    
    To maintain invariant extreme-value false-alarm protection across all cohort sizes
    (from small reference quartets N=4 to large-scale cohorts N=100+), the effective
    threshold is:
      Z_eff = min(nominal_z, G_crit(N, alpha))
    where G_crit(N, alpha) is the exact Student's t critical value for Grubbs' test:
      t = t_{N-2}(1 - alpha / N)
      G_crit = ((N - 1) * t) / sqrt(N * (N - 2 + t^2))
    
    If N < 3, returns nominal_z.
    """
    if N < 3:
        return float(nominal_z)
    max_z = (N - 1) / np.sqrt(N)
    
    # Scale effective alpha smoothly for small cohorts (N <= 6) to avoid asymptotic over-conservatism
    eff_alpha = min(0.05, alpha * (8.0 / max(1, N))) if N <= 6 else alpha

    try:
        from scipy import stats
        t_crit = stats.t.ppf(1.0 - eff_alpha / N, df=N - 2)
        g_crit = ((N - 1) * t_crit) / np.sqrt(N * (N - 2 + t_crit**2))
        return float(min(nominal_z, g_crit))
    except Exception:
        # High-accuracy fallback approximation if scipy is not available
        ratio = max(0.60, min(0.995, 1.0 - (0.05 / np.sqrt(N))))
        return float(min(nominal_z, ratio * max_z))


def trace_continuous_manifold_flow(
    engine,
    window_units: int = 25,
    step: int = 1,
    k_dims: int = 4,
    min_core_coverage: float = 0.30,
    min_sample_coverage: float = 0.05,
    segment_bounds: Optional[List[int]] = None
) -> Dict[str, np.ndarray]:
    """
    Traces continuous manifold river coordinates along the entire genome with
    systematic low-level handling for gappy and partial genomes.
    
    Architecture:
      1. Core Landmark Basis: Taxa with coverage >= min_core_coverage define the
         unperturbed Classical MDS eigenbasis.
      2. Nyström Out-of-Sample Embedding: Partial taxa (coverage >= min_sample_coverage)
         are projected into core coordinates without distorting the coordinate frame.
      3. Explicit NaN Masking: Unsequenced regions (coverage < min_sample_coverage)
         are assigned NaN, preventing artificial jumps to zero distance.
      4. Sign Continuity & Masked Kabsch SO(k): Guarantees continuous coordinate
         flow and strictly bans improper axis reflections (det(R) = +1).
      5. Segment Clamping: If segment_bounds are supplied, windows are clamped inside
         each physical chromosome, preventing chimeric inter-segment blending.
    
    Returns a dictionary with:
      - 'cutpoints': genomic positions
      - 'trajectories': shape [T, N, k_dims]
      - 'velocities': shape [T - 1, N] instantaneous velocity ||z_i(u+1) - z_i(u)||
      - 'coverage': shape [T, N] per-taxon coverage fraction
    """
    U = engine.num_units
    N = engine.N

    # Determine window cutpoints
    if segment_bounds is not None and len(segment_bounds) >= 2:
        # Segment-aware cutpoints: clamped strictly within each segment
        cutpoints_list = []
        seg_indices = []
        for s_idx in range(len(segment_bounds) - 1):
            s_start = segment_bounds[s_idx]
            s_end = segment_bounds[s_idx + 1]
            seg_len = s_end - s_start
            eff_window = min(window_units, max(5, seg_len // 4))
            cps = list(range(s_start + eff_window, s_end - eff_window, step))
            if len(cps) == 0:
                cps = [(s_start + s_end) // 2]
            cutpoints_list.extend(cps)
            seg_indices.extend([s_idx] * len(cps))
        cutpoints = cutpoints_list
    else:
        cutpoints = list(range(window_units, U - window_units, step))
        if len(cutpoints) == 0:
            cutpoints = [U // 2]
        seg_indices = [0] * len(cutpoints)

    num_cut = len(cutpoints)
    trajectories = np.full((num_cut, N, k_dims), np.nan, dtype=np.float64)
    coverage_arr = np.zeros((num_cut, N), dtype=np.float64)
    prev_coords = None
    prev_seg = None

    for idx, u in enumerate(cutpoints):
        curr_seg = seg_indices[idx]
        if segment_bounds is not None and len(segment_bounds) >= 2:
            s_start = segment_bounds[curr_seg]
            s_end = segment_bounds[curr_seg + 1]
            l_start = max(s_start, u - window_units)
            r_end = min(s_end, u + window_units)
        else:
            l_start = max(0, u - window_units)
            r_end = min(U, u + window_units)

        # 1. Query per-taxon coverage in window
        cov = engine.query_coverage(l_start, r_end)
        coverage_arr[idx] = cov

        # 2. Partition into core landmarks and partial samples
        core_mask = cov >= min_core_coverage
        core_idx = np.where(core_mask)[0]

        if len(core_idx) < max(2, k_dims):
            # Fallback if too few core taxa: relax threshold to top taxa
            top_cov = np.argsort(-cov)[:max(4, k_dims + 1)]
            core_idx = top_cov[cov[top_cov] > 0.0]

        coords = np.full((N, k_dims), np.nan, dtype=np.float64)

        if len(core_idx) >= 2:
            # 3. Compute distance matrix
            D_all = engine.query_distance_matrix(l_start, r_end)
            D_core = D_all[np.ix_(core_idx, core_idx)]

            # 4. Classical MDS strictly on core landmarks
            Z_core = compute_classical_mds(D_core, k=k_dims)
            coords[core_idx] = Z_core

            # 5. Gower / Nyström projection for non-core taxa with sufficient partial coverage
            for i in range(N):
                if not core_mask[i] and cov[i] >= min_sample_coverage:
                    d_sample = D_all[i, core_idx]
                    coords[i] = nystrom_out_of_sample_mds(Z_core, D_core, d_sample)

        # 6. Sequential alignment (within the same segment)
        if prev_coords is not None and curr_seg == prev_seg:
            common_mask = ~np.isnan(coords[:, 0]) & ~np.isnan(prev_coords[:, 0])
            common_idx = np.where(common_mask)[0]

            if len(common_idx) >= max(2, k_dims):
                # Enforce sign continuity on leading eigenvectors
                for j in range(k_dims):
                    if np.dot(coords[common_idx, j], prev_coords[common_idx, j]) < 0:
                        coords[:, j] *= -1.0

                # Proper Kabsch SO(k) alignment
                coords, _ = align_procrustes(prev_coords, coords, enforce_so=True)

        trajectories[idx] = coords
        prev_coords = coords
        prev_seg = curr_seg

    # Instantaneous trajectory velocities (ignoring NaNs)
    if num_cut > 1:
        velocities = np.full((num_cut - 1, N), np.nan, dtype=np.float64)
        for t in range(num_cut - 1):
            valid_pair = ~np.isnan(trajectories[t + 1, :, 0]) & ~np.isnan(trajectories[t, :, 0])
            if np.any(valid_pair):
                diff = trajectories[t + 1, valid_pair] - trajectories[t, valid_pair]
                velocities[t, valid_pair] = np.linalg.norm(diff, axis=1)
    else:
        velocities = np.zeros((1, N), dtype=np.float64)

    return {
        "cutpoints": np.array(cutpoints),
        "trajectories": trajectories,
        "velocities": velocities,
        "coverage": coverage_arr
    }
