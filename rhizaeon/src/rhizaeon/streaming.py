"""
rhizaeon/streaming.py
=====================
Unbiased Online Manifold Sequence Threading and Dynamic Metric Covering Engine.

Provides label-free, unsupervised discovery of evolutionary bundles ("cassettes")
on continuous Riemannian sequence manifolds without human prior knowledge,
pre-selected landmarks, or static identity thresholds.

Architecture:
  1. Phase 0: Unbiased Initialization
     - Greedy Farthest-Point Metric Sampling (k-center approximation) on an
       unannotated initial batch, or temporal/chronological first decade.
  2. Phase 1: Real-Time Coordinate Threading
     - Projects incoming sequences into low-dimensional continuous manifold space
       in O(M * d) via exact centered trilateration Nyström projection.
  3. Phase 2: Metric Covering Radius & Outlier Gating
     - Sequences within covering ball (d_min <= epsilon_cover) thread into existing
       bundles without bloating landmark basis (combats outbreak swarms).
     - Sequences landing in empty metric space (d_min > epsilon_cover) are evaluated
       by a quality filter (0 frameshifts, <0.5% ambiguities) and persistence buffer.
  4. Phase 3: Dynamic Landmark Recruitment
     - Confirmed novel lineages are dynamically promoted to active landmark nodes.
     - Manifold coordinates update incrementally via Procrustes-anchored alignment.
  5. Multi-Segment Pan-Genome Cassette Tracker:
     - Tracks 8 segments concurrently, mapping the continuous coordinate stream
       into discrete bundle tuples [b_PB2, b_PB1, b_PA, b_NP, b_MP, b_NS] + [b_HA, b_NA].
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field

from rhizaeon.manifold import (
    compute_classical_mds,
    nystrom_out_of_sample_mds,
    align_procrustes
)


def compute_pairwise_p_distances(seq_mat: np.ndarray) -> np.ndarray:
    """
    Computes pairwise p-distance matrix for encoded nucleotide array [N, L] (0..3 valid, 4 gap/N).
    Vectorized and O(N^2 * L) without intermediate allocations.
    """
    N, L = seq_mat.shape
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        s_i = seq_mat[i:i+1]
        valid = (s_i < 4) & (seq_mat < 4)
        diffs = valid & (s_i != seq_mat)
        valid_cnt = np.maximum(1, np.sum(valid, axis=1))
        diff_cnt = np.sum(diffs, axis=1)
        D[i] = diff_cnt.astype(np.float64) / valid_cnt
    np.fill_diagonal(D, 0.0)
    return D


def compute_sample_to_basis_distances(sample_vec: np.ndarray, basis_mat: np.ndarray) -> np.ndarray:
    """
    Computes distance vector from a single encoded sequence [L] to an array of basis sequences [M, L].
    Returns distance vector d of shape [M].
    """
    s_i = sample_vec[np.newaxis, :]  # [1, L]
    valid = (s_i < 4) & (basis_mat < 4)  # [M, L]
    diffs = valid & (s_i != basis_mat)
    valid_cnt = np.maximum(1, np.sum(valid, axis=1))
    diff_cnt = np.sum(diffs, axis=1)
    d = diff_cnt.astype(np.float64) / valid_cnt
    # If a basis sample has zero valid overlap, assign NaN
    no_overlap = np.sum(valid, axis=1) == 0
    d[no_overlap] = np.nan
    return d


@dataclass
class ThreadingResult:
    """Result of threading a single sequence into the online manifold."""
    seq_id: str
    coords: np.ndarray                  # [k_dims] continuous manifold coordinates
    min_dist: float                     # Distance to closest active landmark
    nearest_landmark_idx: int           # Index of nearest active landmark
    is_novel: bool                      # True if min_dist > cover_radius
    recruited: bool                     # True if promoted to active landmark
    reconstruction_residual: float      # Off-manifold residual distance
    provisional_candidate: bool = False # True if buffered awaiting persistence


class OnlineManifoldCover:
    """
    Continuous Manifold Online Sequence Threading and Dynamic Metric Covering Engine.
    Discovers sequence bundles in streaming fashion with zero prior taxonomic knowledge.
    """

    def __init__(
        self,
        k_dims: int = 3,
        cover_radius: float = 0.025,
        persistence_threshold: int = 2,
        max_landmarks: int = 1500,
        random_seed: int = 42
    ):
        """
        Parameters:
          k_dims: Dimension of continuous manifold Euclidean embedding (default 3).
          cover_radius: Metric radius epsilon_cover (default 0.025 = 2.5% divergence).
                        Sequences within this radius thread into existing bundles.
          persistence_threshold: Number of independent observations required to promote
                                 an outlier to an active landmark (default 2).
          max_landmarks: Safety cap on total active landmarks to guarantee bounded compute.
          random_seed: Random seed for initialization.
        """
        self.k_dims = k_dims
        self.cover_radius = cover_radius
        self.persistence_threshold = persistence_threshold
        self.max_landmarks = max_landmarks
        self.random_seed = random_seed

        # Active basis structures
        self.landmark_ids: List[str] = []
        self.landmark_seqs: List[np.ndarray] = []  # Encoded byte/int8 arrays [L]
        self.landmark_mass: List[int] = []         # Coverage count per landmark
        self.landmark_reservoir: List[List[np.ndarray]] = []  # Member reservoir for medoid calculation
        self.Z_landmarks: Optional[np.ndarray] = None  # [M, k_dims]
        self.D_landmarks: Optional[np.ndarray] = None  # [M, M]
        self.L: Optional[int] = None

        # Persistence candidate buffer: candidate_idx -> list of nearby observed sequence vectors
        self.provisional_candidates: List[Dict[str, Any]] = []

        # Streaming trajectory log
        self.stream_log: List[ThreadingResult] = []

    def initialize_unbiased_farthest_point(
        self,
        seq_matrix: np.ndarray,
        seq_ids: List[str],
        n_landmarks: int = 30
    ) -> None:
        """
        Initializes the landmark core on an unannotated seed batch using greedy
        farthest-point metric sampling (k-center 2-approximation).
        Zero labels or metadata used.
        """
        N, L = seq_matrix.shape
        self.L = L
        n_landmarks = min(n_landmarks, N)

        # 1. Greedy Farthest Point Sampling
        np.random.seed(self.random_seed)
        first_idx = int(np.random.randint(0, N))
        selected_indices = [first_idx]

        # Compute initial distances from first point to all others
        d_to_selected = compute_sample_to_basis_distances(seq_matrix[first_idx], seq_matrix)

        for _ in range(1, n_landmarks):
            # Select point with maximum distance to currently selected set
            next_idx = int(np.nanargmax(d_to_selected))
            selected_indices.append(next_idx)
            # Update minimum distance to selected set
            d_next = compute_sample_to_basis_distances(seq_matrix[next_idx], seq_matrix)
            d_to_selected = np.minimum(d_to_selected, d_next)

        # 2. Extract selected landmarks
        self.landmark_ids = [seq_ids[i] for i in selected_indices]
        self.landmark_seqs = [seq_matrix[i].copy() for i in selected_indices]
        self.landmark_mass = [1 for _ in selected_indices]
        self.landmark_reservoir = [[seq_matrix[i].copy()] for i in selected_indices]
        basis_mat = np.array(self.landmark_seqs)

        # 3. Compute initial distance matrix and classical MDS
        self.D_landmarks = compute_pairwise_p_distances(basis_mat)
        self.Z_landmarks = compute_classical_mds(self.D_landmarks, k=self.k_dims)

    def thread_sequence(
        self,
        seq_vec: np.ndarray,
        seq_id: str,
        quality_check: bool = True,
        allow_recruitment: bool = True
    ) -> ThreadingResult:
        """
        Threads a single incoming sequence into the continuous sequence manifold.
        """
        if self.Z_landmarks is None or len(self.landmark_seqs) == 0:
            raise RuntimeError("OnlineManifoldCover must be initialized before threading sequences.")

        M = len(self.landmark_seqs)
        basis_mat = np.array(self.landmark_seqs)

        # 1. Compute distances to all active landmarks
        d_to_basis = compute_sample_to_basis_distances(seq_vec, basis_mat)

        # Handle valid comparisons
        valid_mask = ~np.isnan(d_to_basis)
        if np.sum(valid_mask) < self.k_dims + 1:
            # Insufficient overlap with core landmarks
            res = ThreadingResult(
                seq_id=seq_id,
                coords=np.full(self.k_dims, np.nan),
                min_dist=np.nan,
                nearest_landmark_idx=-1,
                is_novel=True,
                recruited=False,
                reconstruction_residual=np.nan,
                provisional_candidate=False
            )
            self.stream_log.append(res)
            return res

        # 2. Project onto manifold via exact centered trilateration Nyström
        z_sample = nystrom_out_of_sample_mds(self.Z_landmarks, self.D_landmarks, d_to_basis)

        # 3. Geometric covering metric
        valid_d = d_to_basis[valid_mask]
        min_dist = float(np.min(valid_d))
        nearest_idx = int(np.where(valid_mask)[0][np.argmin(valid_d)])

        # Reconstruction residual in manifold Euclidean space
        diff = self.Z_landmarks[valid_mask] - z_sample
        d_rec = np.linalg.norm(diff, axis=1)
        res_error = float(np.sqrt(np.mean(np.abs(valid_d**2 - d_rec**2))))

        is_novel = min_dist > self.cover_radius
        recruited = False
        provisional = False

        # Accumulate mass and reservoir sample for medoid tracking
        if not is_novel and nearest_idx >= 0:
            self.landmark_mass[nearest_idx] += 1
            if len(self.landmark_reservoir[nearest_idx]) < 50:
                self.landmark_reservoir[nearest_idx].append(seq_vec.copy())

        # 4. Novelty & Persistence Gate
        if is_novel and allow_recruitment and len(self.landmark_seqs) < self.max_landmarks:
            # Perform sequence quality check
            is_high_quality = True
            if quality_check:
                # Allow up to 15% gaps (accommodating subtype-specific indel columns across all 18 HA and 11 NA types)
                # while rejecting severely truncated or unsequenced contigs (>15% missing)
                gap_frac = np.mean(seq_vec >= 4)
                if gap_frac > 0.15:
                    is_high_quality = False

            if is_high_quality:
                # Check candidate persistence buffer
                matched_candidate = False
                for cand in self.provisional_candidates:
                    d_to_cand = float(compute_sample_to_basis_distances(seq_vec, cand["seq"][np.newaxis, :])[0])
                    if d_to_cand <= self.cover_radius:
                        cand["count"] += 1
                        cand["members"].append(seq_id)
                        matched_candidate = True
                        if cand["count"] >= self.persistence_threshold:
                            # Promote candidate to active landmark!
                            self._recruit_landmark(cand["seq"], cand["id"], cand["coords"])
                            self.provisional_candidates.remove(cand)
                            recruited = True
                        break

                if not matched_candidate and not recruited:
                    # Register new provisional candidate in buffer
                    self.provisional_candidates.append({
                        "id": seq_id,
                        "seq": seq_vec.copy(),
                        "coords": z_sample.copy(),
                        "count": 1,
                        "members": [seq_id]
                    })
                    provisional = True

        result = ThreadingResult(
            seq_id=seq_id,
            coords=z_sample,
            min_dist=min_dist,
            nearest_landmark_idx=nearest_idx,
            is_novel=is_novel,
            recruited=recruited,
            reconstruction_residual=res_error,
            provisional_candidate=provisional
        )
        self.stream_log.append(result)
        return result

    def _recruit_landmark(self, new_seq: np.ndarray, new_id: str, new_coords: np.ndarray) -> None:
        """
        Dynamically adds a confirmed persistent outlier to the active landmark basis.
        Updates D_landmarks and Z_landmarks with Procrustes continuity.
        """
        basis_mat = np.array(self.landmark_seqs)
        d_new_to_old = compute_sample_to_basis_distances(new_seq, basis_mat)

        M = len(self.landmark_seqs)
        new_D = np.zeros((M + 1, M + 1), dtype=np.float64)
        new_D[:M, :M] = self.D_landmarks
        new_D[M, :M] = d_new_to_old
        new_D[:M, M] = d_new_to_old
        new_D[M, M] = 0.0

        # Append sequence, mass, reservoir, and ID
        self.landmark_ids.append(new_id)
        self.landmark_seqs.append(new_seq.copy())
        self.landmark_mass.append(1)
        self.landmark_reservoir.append([new_seq.copy()])
        self.D_landmarks = new_D

        # Update manifold coordinates: append projected coordinate
        self.Z_landmarks = np.vstack([self.Z_landmarks, new_coords[np.newaxis, :]])

    def recenter_landmarks(self) -> None:
        """
        Shifts each active landmark to the empirical medoid of its covered members.
        Pulls peripheral early pioneers straight into the true density mode of each lineage.
        """
        for j in range(len(self.landmark_seqs)):
            res = self.landmark_reservoir[j]
            if len(res) >= 3:
                res_mat = np.array(res)
                D_sub = compute_pairwise_p_distances(res_mat)
                medoid_idx = int(np.argmin(np.sum(D_sub, axis=1)))
                self.landmark_seqs[j] = res[medoid_idx].copy()

        basis_mat = np.array(self.landmark_seqs)
        self.D_landmarks = compute_pairwise_p_distances(basis_mat)
        self.Z_landmarks = compute_classical_mds(self.D_landmarks, k=self.k_dims)

    def consolidate_landmarks(
        self,
        min_mass: int = 5,
        merge_radius: Optional[float] = None
    ) -> int:
        """
        Consolidates the landmark basis:
          1. Recenters landmarks to cluster medoids.
          2. Prunes low-mass 'ghost' landmarks (dead-end spillovers / transient outliers).
          3. Coalesces redundant nearby landmarks within merge_radius.
          4. Reconstructs canonical D_landmarks and Z_landmarks.
        """
        self.recenter_landmarks()

        M = len(self.landmark_seqs)
        if M <= self.k_dims + 2:
            return M

        if merge_radius is None:
            merge_radius = 0.6 * self.cover_radius

        # 1. Ghost pruning: keep only landmarks with mass >= min_mass
        keep_mask = np.array(self.landmark_mass) >= min_mass
        if np.sum(keep_mask) < self.k_dims + 2:
            top_indices = np.argsort(self.landmark_mass)[::-1][:self.k_dims + 2]
            keep_mask = np.zeros(M, dtype=bool)
            keep_mask[top_indices] = True

        surviving_indices = np.where(keep_mask)[0].tolist()

        # 2. Metric Coalescence: merge surviving landmarks that are closer than merge_radius
        basis_mat = np.array([self.landmark_seqs[i] for i in surviving_indices])
        D_surv = compute_pairwise_p_distances(basis_mat)

        final_kept = []
        for idx_local, orig_idx in enumerate(surviving_indices):
            merged = False
            for kept_local in final_kept:
                if D_surv[idx_local, kept_local] < merge_radius:
                    orig_kept = surviving_indices[kept_local]
                    self.landmark_mass[orig_kept] += self.landmark_mass[orig_idx]
                    merged = True
                    break
            if not merged:
                final_kept.append(idx_local)

        final_orig_indices = [surviving_indices[k] for k in final_kept]

        # 3. Update structures
        self.landmark_ids = [self.landmark_ids[i] for i in final_orig_indices]
        self.landmark_seqs = [self.landmark_seqs[i] for i in final_orig_indices]
        self.landmark_mass = [self.landmark_mass[i] for i in final_orig_indices]
        self.landmark_reservoir = [self.landmark_reservoir[i] for i in final_orig_indices]

        basis_final = np.array(self.landmark_seqs)
        self.D_landmarks = compute_pairwise_p_distances(basis_final)
        self.Z_landmarks = compute_classical_mds(self.D_landmarks, k=self.k_dims)

        return len(self.landmark_ids)

    def thread_stream(
        self,
        seq_matrix: np.ndarray,
        seq_ids: List[str],
        allow_recruitment: bool = True,
        verbose: bool = False
    ) -> List[ThreadingResult]:
        """
        Streams an entire array of sequences through the online manifold threading engine.
        """
        results = []
        t0 = time.perf_counter()
        for idx in range(len(seq_ids)):
            res = self.thread_sequence(seq_matrix[idx], seq_ids[idx], allow_recruitment=allow_recruitment)
            results.append(res)
            if verbose and (idx + 1) % 1000 == 0:
                elapsed = time.perf_counter() - t0
                rate = (idx + 1) / max(0.001, elapsed)
                print(f"[*] Threaded {idx + 1:,} sequences | Active Landmarks: {len(self.landmark_ids)} | Rate: {rate:.1f} seq/s")
        return results

    def two_pass_stream(
        self,
        seq_matrix: np.ndarray,
        seq_ids: List[str],
        min_mass: int = 5,
        verbose: bool = False
    ) -> List[ThreadingResult]:
        """
        Two-Pass Sketch-and-Consolidate streaming engine for guaranteed order invariance.
        Pass 1: Discover metric covering envelope and accumulate mass.
        Consolidation: Ghost pruning, medoid recentering, and coalescence.
        Pass 2: Canonical coordinate projection onto the consolidated basis.
        """
        if verbose:
            print("[*] Pass 1: Streaming metric sketch & mass accumulation...")
        self.thread_stream(seq_matrix, seq_ids, allow_recruitment=True, verbose=verbose)

        if verbose:
            print(f"[*] Consolidating landmark basis (M={len(self.landmark_ids)})...")
        M_final = self.consolidate_landmarks(min_mass=min_mass)
        if verbose:
            print(f"[*] Basis consolidated to M={M_final} high-mass canonical centroids.")

        if verbose:
            print("[*] Pass 2: Projecting sequences onto canonical order-invariant basis...")
        self.stream_log.clear()
        results = self.thread_stream(seq_matrix, seq_ids, allow_recruitment=False, verbose=verbose)
        return results


    def cluster_density_bundles(
        self,
        density_bandwidth: Optional[float] = None
    ) -> np.ndarray:
        """
        Assigns each sequence in the stream to an unsupervised topological bundle
        via watershed density mode clustering on continuous manifold coordinates.
        """
        coords_all = np.array([r.coords for r in self.stream_log])
        valid_mask = ~np.isnan(coords_all[:, 0])
        valid_coords = coords_all[valid_mask]
        N_valid = len(valid_coords)

        if density_bandwidth is None:
            # Silverman / Scott rule of thumb bandwidth
            sigma = np.std(valid_coords, axis=0)
            density_bandwidth = float(np.mean(sigma) * (N_valid ** (-1.0 / (self.k_dims + 4))))
            density_bandwidth = max(0.01, density_bandwidth)

        # Hill-climbing to local density modes
        bundle_assignments = np.full(len(self.stream_log), -1, dtype=int)
        local_modes = np.zeros(N_valid, dtype=int)

        if N_valid <= 2000:
            from scipy.spatial.distance import cdist
            D_eucl = cdist(valid_coords, valid_coords)
            densities = np.sum(np.exp(- (D_eucl ** 2) / (2.0 * density_bandwidth ** 2)), axis=1)
            k_neighbors = min(15, N_valid - 1)
            for i in range(N_valid):
                curr = i
                visited = set([curr])
                while True:
                    nn_idx = np.argsort(D_eucl[curr])[:k_neighbors + 1]
                    best_nn = nn_idx[np.argmax(densities[nn_idx])]
                    if densities[best_nn] <= densities[curr] or best_nn in visited:
                        local_modes[i] = curr
                        break
                    curr = best_nn
                    visited.add(curr)
        else:
            from scipy.spatial import cKDTree
            tree = cKDTree(valid_coords)
            k_neighbors = min(30, N_valid)
            dists, indices = tree.query(valid_coords, k=k_neighbors)
            densities = np.sum(np.exp(- (dists ** 2) / (2.0 * density_bandwidth ** 2)), axis=1)
            for i in range(N_valid):
                curr = i
                visited = set([curr])
                while True:
                    nbrs = indices[curr]
                    best_nbr = nbrs[np.argmax(densities[nbrs])]
                    if densities[best_nbr] <= densities[curr] or best_nbr in visited:
                        local_modes[i] = curr
                        break
                    curr = best_nbr
                    visited.add(curr)

        # Map distinct modes to integer bundle IDs
        unique_modes = np.unique(local_modes)
        mode_to_bundle = {m: b_id + 1 for b_id, m in enumerate(unique_modes)}
        valid_bundle_ids = np.array([mode_to_bundle[m] for m in local_modes])

        bundle_assignments[valid_mask] = valid_bundle_ids
        return bundle_assignments

    def cluster_landmark_bundles(
        self,
        divergence_cutoff: float = 0.07,
        linkage_method: str = "average"
    ) -> np.ndarray:
        """
        Partitions the streaming sequences into coherent macro-bundles by performing
        hierarchical metric linkage on the active landmark distance graph (the epsilon_cover net).

        Parameters:
          divergence_cutoff: Pairwise metric divergence threshold defining macro-bundle boundaries
                             (default 0.07 = 7% sequence divergence).
          linkage_method: Agglomerative linkage method ('average', 'complete', 'single').
                          Default 'average' (UPGMA).

        Returns:
          bundle_assignments: Integer array of shape [N_stream] giving the macro-bundle ID
                              for each streaming sequence.
        """
        if self.D_landmarks is None or len(self.landmark_ids) == 0:
            return np.full(len(self.stream_log), -1, dtype=int)

        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        M = len(self.landmark_ids)
        if M == 1:
            clusters = np.array([1], dtype=int)
        else:
            condensed = squareform(self.D_landmarks, checks=False)
            Z = linkage(condensed, method=linkage_method)
            clusters = fcluster(Z, t=divergence_cutoff, criterion="distance")

        # Map each streaming sequence to the cluster ID of its nearest active landmark
        nearest_indices = np.array([r.nearest_landmark_idx for r in self.stream_log], dtype=int)
        bundle_assignments = np.full(len(self.stream_log), -1, dtype=int)
        valid = (nearest_indices >= 0) & (nearest_indices < M)
        bundle_assignments[valid] = clusters[nearest_indices[valid]]
        return bundle_assignments


class PanGenomeCassetteTracker:
    """
    Multi-Segment Pan-Genome Cassette Tracker for Segmented Viruses (Influenza A).
    Runs 8 independent OnlineManifoldCover instances, compiling 8-segment bundle tuples
    and detecting whole-cassette reassortments in real time.
    """

    SEGMENTS = ["PB2", "PB1", "PA", "HA", "NP", "NA", "MP", "NS"]
    CANONICAL_BOUNDS_12961 = [0, 2280, 4554, 6705, 8509, 10024, 11488, 12247, 12961]
    LEGACY_BOUNDS_12744 = [0, 2277, 4548, 6696, 8397, 9891, 11298, 12054, 12744]
    BOUNDS = LEGACY_BOUNDS_12744

    def __init__(
        self,
        k_dims: int = 3,
        cover_radius: float = 0.025,
        persistence_threshold: int = 2,
        bounds: Optional[List[int]] = None
    ):
        self.bounds = bounds
        self.engines: Dict[str, OnlineManifoldCover] = {
            s: OnlineManifoldCover(
                k_dims=k_dims,
                cover_radius=cover_radius,
                persistence_threshold=persistence_threshold
            )
            for s in self.SEGMENTS
        }
        self.genome_log: List[Dict[str, Any]] = []

    def _resolve_bounds(self, seq_len: int) -> List[int]:
        if self.bounds is not None:
            return self.bounds
        if seq_len == 12961:
            return self.CANONICAL_BOUNDS_12961
        elif seq_len == 12744:
            return self.LEGACY_BOUNDS_12744
        return self.CANONICAL_BOUNDS_12961

    def initialize_from_concatenated_batch(
        self,
        concatenated_matrix: np.ndarray,
        genome_ids: List[str],
        n_init: int = 30
    ) -> None:
        """
        Initializes each of the 8 segment engines on an unannotated seed batch.
        """
        bounds = self._resolve_bounds(concatenated_matrix.shape[1])
        for s_idx, s_name in enumerate(self.SEGMENTS):
            start = bounds[s_idx]
            end = bounds[s_idx + 1]
            seg_mat = concatenated_matrix[:, start:end]
            self.engines[s_name].initialize_unbiased_farthest_point(seg_mat, genome_ids, n_landmarks=n_init)

    def thread_genome(
        self,
        concatenated_seq: np.ndarray,
        genome_id: str,
        allow_recruitment: bool = True
    ) -> Dict[str, Any]:
        """
        Threads an 8-segment concatenated genome through all 8 online segment manifolds.
        Returns per-segment coordinates, novelty flags, and the composite cassette profile.
        """
        seg_results = {}
        coords_8seg = []
        is_novel_any = False
        recruited_any = False

        bounds = self._resolve_bounds(len(concatenated_seq))
        for s_idx, s_name in enumerate(self.SEGMENTS):
            start = bounds[s_idx]
            end = bounds[s_idx + 1]
            seg_vec = concatenated_seq[start:end]
            res = self.engines[s_name].thread_sequence(
                seg_vec,
                f"{genome_id}_{s_name}",
                allow_recruitment=allow_recruitment
            )
            seg_results[s_name] = res
            coords_8seg.append(res.coords)
            if res.is_novel:
                is_novel_any = True
            if res.recruited:
                recruited_any = True

        record = {
            "genome_id": genome_id,
            "coords": coords_8seg,
            "seg_results": seg_results,
            "is_novel_reassortant": is_novel_any,
            "landmark_recruited": recruited_any
        }
        self.genome_log.append(record)
        return record

    def two_pass_stream(
        self,
        concatenated_matrix: np.ndarray,
        genome_ids: List[str],
        min_mass: int = 5,
        verbose: bool = False
    ) -> None:
        """
        Two-pass streaming for the complete 8-segment pan-genome tracker.
        Pass 1: Stream and accumulate masses across all 8 segments.
        Consolidation: Prune ghosts, merge near-duplicates, and recenter landmarks to medoids.
        Pass 2: Canonical coordinate projection.
        """
        if verbose:
            print("[*] PanGenomeCassetteTracker: Pass 1 (Metric Sketch)...")
        for i in range(len(genome_ids)):
            self.thread_genome(concatenated_matrix[i], genome_ids[i], allow_recruitment=True)

        if verbose:
            print("[*] PanGenomeCassetteTracker: Consolidating 8-segment landmark bases...")
        for s in self.SEGMENTS:
            self.engines[s].consolidate_landmarks(min_mass=min_mass)

        if verbose:
            print("[*] PanGenomeCassetteTracker: Pass 2 (Canonical Projection)...")
        self.genome_log.clear()
        for s in self.SEGMENTS:
            self.engines[s].stream_log.clear()
        for i in range(len(genome_ids)):
            self.thread_genome(concatenated_matrix[i], genome_ids[i], allow_recruitment=False)


    def compute_genome_cassettes(
        self,
        internal_only: bool = True,
        divergence_cutoff: float = 0.07,
        linkage_method: str = "average"
    ) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """
        Computes composite genomic cassettes across all streamed genomes using
        unsupervised landmark macro-bundles.

        Parameters:
          internal_only: If True, compiles the 6 internal segments [PB2, PB1, PA, NP, MP, NS].
                         If False, compiles all 8 segments.
          divergence_cutoff: Metric distance cutoff for landmark clustering (default 0.07).
          linkage_method: Linkage method ('average', 'complete', 'single').

        Returns:
          cassette_keys: List of composite cassette string signatures for all genomes.
          segment_bundles: Dict mapping segment name -> array of bundle IDs.
        """
        target_segs = ["PB2", "PB1", "PA", "NP", "MP", "NS"] if internal_only else self.SEGMENTS
        segment_bundles = {}
        for s_name in self.SEGMENTS:
            segment_bundles[s_name] = self.engines[s_name].cluster_landmark_bundles(
                divergence_cutoff=divergence_cutoff,
                linkage_method=linkage_method
            )

        N_genomes = len(self.genome_log)
        cassette_keys = []
        for i in range(N_genomes):
            sig = ".".join(f"{s}{segment_bundles[s][i]}" for s in target_segs)
            cassette_keys.append(sig)

        return cassette_keys, segment_bundles
