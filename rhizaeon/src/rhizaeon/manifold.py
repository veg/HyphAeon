"""
rhizaeon.manifold
=================
Continuous sequence manifold geometry, spectral embeddings, and
ChronAeon-style out-of-sample Ghost Node Procrustes alignment.
"""

from typing import Tuple, Optional, Dict
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
    if N <= k:
        k = max(1, N - 1)

    # Centering matrix H
    H = np.eye(N, dtype=np.float64) - (1.0 / N) * np.ones((N, N), dtype=np.float64)
    D_sq = D.astype(np.float64) ** 2
    B = -0.5 * (H @ D_sq @ H)

    # Symmetric eigendecomposition
    w, v = la.eigh(B)

    # Sort descending
    idx = np.argsort(-w)
    w = w[idx]
    v = v[:, idx]

    coords = []
    for j in range(k):
        val = max(0.0, float(w[j]))
        coords.append(v[:, j] * np.sqrt(val))

    return np.column_stack(coords)


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
    Z_source: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Orthogonal Procrustes Manifold Alignment.
    
    Finds optimal rotation matrix R in O(k) and translation vector t minimizing:
      min_{R, t} ||Z_target - (Z_source @ R + t)||_F^2
    
    Returns:
      Z_aligned: Source coordinates rotated and translated into target frame.
      residuals: Per-taxon Euclidean distances ||z_{target, i} - z_{aligned, i}||_2.
    """
    mu_s = np.mean(Z_source, axis=0, keepdims=True)
    mu_t = np.mean(Z_target, axis=0, keepdims=True)

    Z_s_cent = Z_source - mu_s
    Z_t_cent = Z_target - mu_t

    # Procrustes cross-covariance
    M = Z_t_cent.T @ Z_s_cent
    U, _, Vt = la.svd(M)
    R = Vt.T @ U.T

    Z_aligned = Z_s_cent @ R + mu_t
    residuals = np.linalg.norm(Z_target - Z_aligned, axis=1)
    return Z_aligned, residuals


def compute_ghost_node_zscores(residuals: np.ndarray) -> np.ndarray:
    """
    Calculates robust Studentized / IQR Z-scores from Procrustes residuals:
      Z_i = (r_i - median(r)) / max(IQR(r), 1e-5)
    
    Non-recombinant taxa conform tightly to the global rigid rotation (Z ~ 0).
    Recombinant lineages detach as high-leverage 'Ghost Nodes' (Z >> 3.0).
    """
    med = np.median(residuals)
    iqr = np.percentile(residuals, 75) - np.percentile(residuals, 25)
    return (residuals - med) / max(iqr, 1e-5)


def trace_continuous_manifold_flow(
    engine,
    window_units: int = 25,
    step: int = 1,
    k_dims: int = 4
) -> Dict[str, np.ndarray]:
    """
    Traces continuous manifold river coordinates along the entire genome.
    
    At each position u, embeds all N taxa into low-dimensional manifold coordinates
    Z(u) in R^{N x k} and sequentially aligns each step via Orthogonal Procrustes
    to ensure smooth, continuous trajectories.
    
    Returns a dictionary with:
      - 'cutpoints': genomic positions
      - 'trajectories': shape [T, N, k_dims]
      - 'velocities': shape [T - 1, N] instantaneous velocity ||z_i(u+1) - z_i(u)||
    """
    U = engine.num_units
    N = engine.N

    cutpoints = list(range(window_units, U - window_units, step))
    if len(cutpoints) == 0:
        cutpoints = [U // 2]

    trajectories = np.zeros((len(cutpoints), N, k_dims), dtype=np.float64)
    prev_coords = None

    for idx, u in enumerate(cutpoints):
        l_start = max(0, u - window_units)
        r_end = min(U, u + window_units)
        D_w = engine.query_distance_matrix(l_start, r_end)

        coords = compute_classical_mds(D_w, k=k_dims)

        if prev_coords is not None:
            coords, _ = align_procrustes(prev_coords, coords)

        trajectories[idx] = coords
        prev_coords = coords

    # Instantaneous trajectory velocities
    num_steps = len(cutpoints)
    if num_steps > 1:
        velocities = np.zeros((num_steps - 1, N), dtype=np.float64)
        for t in range(num_steps - 1):
            velocities[t] = np.linalg.norm(trajectories[t + 1] - trajectories[t], axis=1)
    else:
        velocities = np.zeros((1, N), dtype=np.float64)

    return {
        "cutpoints": np.array(cutpoints),
        "trajectories": trajectories,
        "velocities": velocities
    }
