"""
Conformal Prediction Engine for Genomic Metric Spaces (rhizaeon.conformal)

Implements distribution-free non-conformity scoring, finite-sample coverage
guarantees, and multi-segment hypothesis aggregation (Simes / Bonferroni / Benjamini-Hochberg)
for autonomous novelty and reassortant detection in viral genomic surveillance.
"""

from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd


class ConformalMetricCalibrator:
    """
    Distribution-Free Conformal Calibrator on Finite Metric Spaces.
    Computes non-conformity scores relative to exemplar cluster medoids
    and evaluates finite-sample valid conformal p-values.
    """

    def __init__(self, normalized: bool = False):
        self.normalized = normalized
        self.medoid_indices: Optional[List[int]] = None
        self.calibration_scores: Optional[np.ndarray] = None
        self.cluster_scales: Optional[np.ndarray] = None
        self.n_calib: int = 0

    def fit(self, D_calib: np.ndarray, medoid_indices: List[int]) -> "ConformalMetricCalibrator":
        """
        Calibrate on intra-cluster distances to exemplar medoids.
        
        Parameters
        ----------
        D_calib : np.ndarray, shape (N_calib, N_calib)
            Pairwise distance matrix for the calibration set.
        medoid_indices : List[int]
            Indices of the exemplar medoids within D_calib.
        """
        self.medoid_indices = list(medoid_indices)
        self.n_calib = D_calib.shape[0]
        K = len(self.medoid_indices)

        # Distances from all calibration samples to the K medoids: shape (N_calib, K)
        D_to_medoids = D_calib[:, self.medoid_indices]

        if self.normalized:
            # Estimate intra-cluster dispersion (median distance to medoid)
            closest_cluster = np.argmin(D_to_medoids, axis=1)
            scales = np.zeros(K, dtype=float)
            for c in range(K):
                members = np.where(closest_cluster == c)[0]
                if len(members) > 1:
                    dists = D_to_medoids[members, c]
                    # Median absolute distance to medoid (avoiding zero)
                    scales[c] = max(float(np.median(dists)), 1e-6)
                else:
                    scales[c] = 1e-3
            self.cluster_scales = scales
            
            # Normalized non-conformity: min_c (d_c / scale_c)
            norm_dists = D_to_medoids / self.cluster_scales[np.newaxis, :]
            scores = np.min(norm_dists, axis=1)
        else:
            # Raw minimum distance to nearest medoid
            scores = np.min(D_to_medoids, axis=1)

        self.calibration_scores = np.sort(scores)
        return self

    def compute_p_value(self, query_distances_to_medoids: np.ndarray) -> np.ndarray:
        """
        Compute finite-sample conformal p-values for query sequences.
        
        Parameters
        ----------
        query_distances_to_medoids : np.ndarray, shape (N_queries, K)
            Distance from each query sequence to the K reference medoids.
            
        Returns
        -------
        p_values : np.ndarray, shape (N_queries,)
            Valid conformal p-values under exchangeability.
        """
        if self.calibration_scores is None:
            raise ValueError("Calibrator has not been fitted.")

        if self.normalized:
            scaled_dists = query_distances_to_medoids / self.cluster_scales[np.newaxis, :]
            query_scores = np.min(scaled_dists, axis=1)
        else:
            query_scores = np.min(query_distances_to_medoids, axis=1)

        # Finite-sample conformal p-value formula:
        # p(Q) = (1 + count(s_calib >= s_Q)) / (N_calib + 1)
        n = self.n_calib
        # searchsorted with side='left' gives count of elements < query_score
        counts_less = np.searchsorted(self.calibration_scores, query_scores, side="left")
        counts_greater_or_equal = n - counts_less
        p_values = (1.0 + counts_greater_or_equal) / (n + 1.0)
        return np.clip(p_values, 0.0, 1.0)

    def get_threshold(self, epsilon: float) -> float:
        """
        Get conformal distance cutoff for a specified significance level epsilon.
        """
        if self.calibration_scores is None:
            raise ValueError("Calibrator has not been fitted.")
        q_idx = int(np.ceil((self.n_calib + 1) * (1.0 - epsilon))) - 1
        q_idx = min(max(0, q_idx), self.n_calib - 1)
        return float(self.calibration_scores[q_idx])


class MultiSegmentConformalEngine:
    """
    Multi-Segment Conformal Prediction Engine for Segmented Viral Genomes.
    Evaluates per-segment non-conformity and performs omnibus hypothesis
    aggregation (Simes, Bonferroni, FDR) to detect novel segment introductions.
    """

    def __init__(
        self,
        segment_names: List[str],
        segment_bounds: List[Tuple[int, int]],
        normalized: bool = False
    ):
        self.segment_names = segment_names
        self.segment_bounds = segment_bounds
        self.n_segments = len(segment_names)
        self.calibrators: Dict[str, ConformalMetricCalibrator] = {
            s: ConformalMetricCalibrator(normalized=normalized) for s in segment_names
        }
        self.medoids_per_segment: Dict[str, List[int]] = {}

    def fit_from_prefix_engine(
        self,
        engine,
        core_taxa_indices: List[int],
        medoids_per_segment: Dict[str, List[int]]
    ) -> "MultiSegmentConformalEngine":
        """
        Calibrate all segment calibrators using prefix distance tensor slices.
        """
        self.medoids_per_segment = medoids_per_segment
        core_arr = np.array(core_taxa_indices)

        for s_idx, s_name in enumerate(self.segment_names):
            start, end = self.segment_bounds[s_idx]
            D_full = engine.query_distance_matrix(start, end)
            D_core = D_full[np.ix_(core_arr, core_arr)]
            
            # Map global medoid indices to local core indices
            medoid_globals = medoids_per_segment[s_name]
            local_medoids = []
            for mg in medoid_globals:
                pos = np.where(core_arr == mg)[0]
                if len(pos) > 0:
                    local_medoids.append(int(pos[0]))
                else:
                    raise ValueError(f"Medoid {mg} not found in diversity core.")
                    
            self.calibrators[s_name].fit(D_core, local_medoids)

        return self

    def score_queries(
        self,
        engine,
        query_taxa: List[str],
        query_indices: Optional[List[int]] = None,
        alpha_fdr: float = 0.05
    ) -> pd.DataFrame:
        """
        Score all queries across all 8 segments and compute omnibus genome-level novelty.
        """
        if query_indices is None:
            query_indices = list(range(len(query_taxa)))
            
        N_queries = len(query_taxa)
        segment_p_values = np.zeros((N_queries, self.n_segments), dtype=float)
        segment_min_dists = np.zeros((N_queries, self.n_segments), dtype=float)
        closest_medoid_names = []

        # Evaluate each segment independently
        for s_idx, s_name in enumerate(self.segment_names):
            start, end = self.segment_bounds[s_idx]
            D_full = engine.query_distance_matrix(start, end)
            
            medoids = self.medoids_per_segment[s_name]
            # Slices: (N_queries, K)
            D_q_to_medoids = D_full[np.ix_(query_indices, medoids)]
            
            p_vals = self.calibrators[s_name].compute_p_value(D_q_to_medoids)
            min_d = np.min(D_q_to_medoids, axis=1)
            
            segment_p_values[:, s_idx] = p_vals
            segment_min_dists[:, s_idx] = min_d

        # Omnibus Hypothesis Aggregation:
        # 1. Simes test for intersection null H_0: all segments conform
        # p_simes = min_{k=1..8} (8 / k * p_(k))
        sorted_p = np.sort(segment_p_values, axis=1) # shape (N, 8)
        k_factors = 8.0 / np.arange(1, 9, dtype=float)
        simes_matrix = sorted_p * k_factors[np.newaxis, :]
        p_simes = np.min(simes_matrix, axis=1)
        p_simes = np.clip(p_simes, 0.0, 1.0)

        # 2. Bonferroni conservative bound
        p_bonf = np.clip(np.min(segment_p_values, axis=1) * self.n_segments, 0.0, 1.0)

        # 3. Minimum segment p-value
        p_min = np.min(segment_p_values, axis=1)

        # Build output DataFrame
        df_records = []
        for i in range(N_queries):
            rec = {
                "Taxon": query_taxa[i],
                "Omnibus_Simes_P": float(p_simes[i]),
                "Omnibus_Bonf_P": float(p_bonf[i]),
                "Min_Segment_P": float(p_min[i]),
                "Is_Novel_Genome": bool(p_simes[i] < alpha_fdr)
            }
            # Add per-segment metrics
            outlier_segments = []
            for s_idx, s_name in enumerate(self.segment_names):
                p_s = segment_p_values[i, s_idx]
                d_s = segment_min_dists[i, s_idx]
                rec[f"{s_name}_P"] = float(p_s)
                rec[f"{s_name}_MinDist"] = float(d_s)
                if p_s < (alpha_fdr / self.n_segments): # Bonferroni segment threshold
                    outlier_segments.append(s_name)
                    
            rec["Novel_Segments"] = ",".join(outlier_segments) if outlier_segments else "None"
            df_records.append(rec)

        return pd.DataFrame(df_records)

    def stream_queries_dynamic(
        self,
        engine,
        query_taxa: List[str],
        query_indices: Optional[List[int]] = None,
        alpha_fdr: float = 0.05,
        alpha_recruit: Optional[float] = None,
        min_recruit_distance: float = 0.005
    ) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, Any]]]]:
        """
        Stream query genomes with Dynamic Medoid Recruitment.
        When an incoming isolate exhibits statistically verified segment novelty
        (p_s <= alpha_recruit), the engine promotes the query sequence into the active
        segment reference medoid set. Subsequent query sequences belonging to the same
        emergent outbreak or clade match the newly recruited medoid, preventing alarm fatigue
        and dynamically tracking cluster expansion in real time.

        Parameters
        ----------
        engine : PrefixDistanceEngine
            Precomputed prefix distance engine.
        query_taxa : List[str]
            List of query taxon identifiers in arrival order.
        query_indices : Optional[List[int]]
            Global indices in the engine corresponding to query_taxa.
        alpha_fdr : float
            Genome-level significance threshold for omnibus Simes test.
        alpha_recruit : Optional[float]
            Segment-level threshold for recruiting novel medoids. Defaults to alpha_fdr / 8.
        min_recruit_distance : float
            Minimum distance to existing medoids required to spawn a new cluster.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, List[Dict[str, Any]]]]
            DataFrame of typing results and dictionary of recruited medoid audit logs.
        """
        if query_indices is None:
            query_indices = list(range(len(query_taxa)))

        if alpha_recruit is None:
            alpha_recruit = alpha_fdr / float(self.n_segments)

        N_queries = len(query_taxa)
        active_medoid_indices = {s: list(self.medoids_per_segment[s]) for s in self.segment_names}
        recruited_info: Dict[str, List[Dict[str, Any]]] = {s: [] for s in self.segment_names}

        # Precompute segment distance matrices
        seg_dist_matrices = {}
        for s_idx, s_name in enumerate(self.segment_names):
            start, end = self.segment_bounds[s_idx]
            seg_dist_matrices[s_name] = engine.query_distance_matrix(start, end)

        df_records = []
        cluster_counts = {s: {i: 0 for i in range(len(active_medoid_indices[s]))} for s in self.segment_names}

        for i, q_idx in enumerate(query_indices):
            taxon = query_taxa[i]
            q_segment_p = np.zeros(self.n_segments, dtype=float)
            q_segment_dist = np.zeros(self.n_segments, dtype=float)
            q_segment_call = []
            flagged_novel_segments = []
            recruited_this_genome = []

            for s_idx, s_name in enumerate(self.segment_names):
                D_seg = seg_dist_matrices[s_name]
                current_medoids = active_medoid_indices[s_name]
                dists = D_seg[q_idx, current_medoids]

                min_idx = int(np.argmin(dists))
                min_d = float(dists[min_idx])

                p_s = float(self.calibrators[s_name].compute_p_value(dists.reshape(1, -1))[0])
                q_segment_p[s_idx] = p_s
                q_segment_dist[s_idx] = min_d

                # Dynamic recruitment check: verified novelty and minimum separation
                if p_s <= alpha_recruit and min_d >= min_recruit_distance:
                    # Spawn new medoid
                    new_medoid_idx = q_idx
                    current_medoids.append(new_medoid_idx)
                    new_cluster_id = len(current_medoids) - 1
                    cluster_counts[s_name][new_cluster_id] = 1

                    rec_log = {
                        "Step": i,
                        "Taxon": taxon,
                        "Segment": s_name,
                        "Global_Idx": q_idx,
                        "P_Value": p_s,
                        "Min_Dist": min_d,
                        "Cluster_ID": f"C{new_cluster_id + 1}",
                    }
                    recruited_info[s_name].append(rec_log)
                    recruited_this_genome.append(s_name)
                    q_segment_call.append(f"C{new_cluster_id + 1}*") # asterisk denotes founder
                    flagged_novel_segments.append(s_name)
                else:
                    cluster_counts[s_name][min_idx] += 1
                    q_segment_call.append(f"C{min_idx + 1}")
                    if p_s <= alpha_recruit:
                        flagged_novel_segments.append(s_name)

            # Simes test for omnibus novelty
            sorted_p = np.sort(q_segment_p)
            k_factors = float(self.n_segments) / np.arange(1, self.n_segments + 1, dtype=float)
            p_simes = float(np.min(sorted_p * k_factors))
            p_simes = min(max(p_simes, 0.0), 1.0)
            p_bonf = float(np.clip(np.min(q_segment_p) * self.n_segments, 0.0, 1.0))

            rec = {
                "Taxon": taxon,
                "Constellation": "-".join(q_segment_call),
                "Omnibus_Simes_P": p_simes,
                "Omnibus_Bonf_P": p_bonf,
                "Min_Segment_P": float(np.min(q_segment_p)),
                "Is_Novel_Genome": bool(p_simes < alpha_fdr),
                "Novel_Segments": ",".join(flagged_novel_segments) if flagged_novel_segments else "None",
                "Recruited_Segments": ",".join(recruited_this_genome) if recruited_this_genome else "None",
            }
            for s_idx, s_name in enumerate(self.segment_names):
                rec[f"{s_name}_P"] = q_segment_p[s_idx]
                rec[f"{s_name}_MinDist"] = q_segment_dist[s_idx]

            df_records.append(rec)

        return pd.DataFrame(df_records), recruited_info

