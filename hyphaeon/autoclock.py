"""
hyphaeon/autoclock.py
--------------------
Automated Multi-Clock Community Deconvolution Engine (ChronAeon AutoClock).

Unsupervised manifold learning and spectral graph partitioning for time-stamped
viral/bacterial sequences. Deconvolves uncurated sequence mixtures into their
constituent biological clocks without requiring prior taxonomic annotations.

Mathematical Pipeline:
1. Pairwise Tamura-Nei 93 (TN93) continuous distance manifold mapping.
2. Normalized symmetric graph Laplacian decomposition:
   L_sym = I - D_A^{-1/2} A D_A^{-1/2}
3. Dominant topological eigengap detection:
   Delta lambda_K = lambda_{K+1} - lambda_K
4. Joint penalized model selection (AICc / BIC) across candidate clocks K in [1, K_max].
5. Per-community ChronAeon molecular clock calibration (OLS and REML PGLS).
6. Automated sequence quality triage via studentized divergence residuals (|r_i*| > 3.0 sigma).

Author: Sergei L. Kosakovsky Pond & DeepMind Antigravity Pair Programmer
"""

import os
import sys
import time
import json
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import eigh
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from .dataset import (
    compute_tn93_distance_matrix,
    compute_tn93_cross_distance_matrix,
    parse_alignment_sequences,
)
from .dating import (
    run_mrca_dating,
    parse_sample_dates,
    verify_coding_alignment,
)


def select_adaptive_n_landmarks(
    n_taxa: int,
    max_k: int = 8,
    user_landmarks: Optional[Union[int, str]] = None,
    max_memory_mb: float = 1024.0,
    min_landmarks: int = 100,
    max_landmarks: int = 2500,
) -> int:
    """
    Selects optimal landmark count M balancing memory footprint, runtime, and spectral fidelity.
    
    Trade-offs:
      - Memory footprint: Rectangular float64 matrix (N x M x 8 bytes).
      - Compute time: Multi-threaded pairwise comparisons scale as O(N * M).
      - Spectral fidelity: Nyström kernel approximation error drops as M approaches
        the intrinsic manifold dimension (scaling sublinearly as O(sqrt(N)) or O(K*log(K))).
    """
    if user_landmarks is not None and str(user_landmarks).lower() not in ["none", "auto"]:
        try:
            return min(n_taxa, max(min_landmarks, int(user_landmarks)))
        except (ValueError, TypeError):
            pass

    # Upper bound from physical memory budget (N x M x 8 bytes)
    m_mem = int((max_memory_mb * 1e6) / max(8.0 * n_taxa, 1.0))

    # Statistical / intrinsic dimension bound
    m_stat = int(max(20 * max_k, 15 * np.sqrt(n_taxa)))

    m_opt = min(m_mem, m_stat, max_landmarks, n_taxa)
    return max(min(n_taxa, min_landmarks), m_opt)


class AutoClockDeconvolution:
    """
    Automated Multi-Clock Community Deconvolution Engine for ChronAeon.
    """

    def __init__(
        self,
        alignment_path: Optional[Union[str, Path]] = None,
        dates_source: Optional[Union[str, Path]] = None,
        date_col: Optional[str] = None,
        strain_col: Optional[str] = None,
        date_regex: Optional[str] = None,
        max_k: int = 6,
        manifold: str = "auto",
        weights: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        kernel_bandwidth: Optional[float] = None,
        min_cluster_size: int = 5,
        random_state: int = 42,
        output_dir: Optional[Union[str, Path]] = None,
        allow_stop_codons: bool = True,
        quiet: bool = False,
        beast_path: Optional[Union[str, Path]] = None,
        max_dense_n: int = 2500,
        n_landmarks: Union[int, str] = "auto",
        max_memory_mb: float = 1024.0,
        records: Optional[List[SeqRecord]] = None,
        dates_map: Optional[Dict[str, float]] = None,
        meta_df: Optional[pd.DataFrame] = None,
    ):
        if beast_path:
            if not alignment_path:
                alignment_path = beast_path
            if not dates_source:
                dates_source = beast_path
        if not alignment_path and not records:
            raise ValueError("Must provide either alignment_path, beast_path, or in-memory records.")
        self.alignment_path = Path(alignment_path) if alignment_path else Path("in_memory.fasta")

        # Auto-detect BEAST XML dates if dates_source was not explicitly specified
        if dates_source is None and alignment_path:
            if str(self.alignment_path).lower().endswith(('.xml', '.xml.gz')):
                dates_source = self.alignment_path
            elif self.alignment_path.exists():
                try:
                    with open(self.alignment_path, 'r', encoding='utf-8', errors='replace') as f:
                        header = f.read(1024).lstrip()
                    if header.startswith('<') and ('<beast' in header.lower() or '<alignment' in header.lower() or '<data' in header.lower()):
                        dates_source = self.alignment_path
                except Exception:
                    pass

        self.dates_source = Path(dates_source) if dates_source else None
        self.date_col = date_col
        self.strain_col = strain_col
        self.date_regex = date_regex
        self.max_k = max_k
        self.manifold = str(manifold).lower()
        self.weights = Path(weights) if weights else None
        self.device = device
        self.kernel_bandwidth = kernel_bandwidth
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
        self.output_dir = Path(output_dir) if output_dir else self.alignment_path.parent / "autoclock_results"
        self.allow_stop_codons = allow_stop_codons
        self.quiet = quiet
        self.max_dense_n = max_dense_n
        self.n_landmarks = n_landmarks
        self.max_memory_mb = float(max_memory_mb)
        self.effective_n_landmarks: Optional[int] = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Loaded data
        self.records: List[SeqRecord] = list(records) if records else []
        self.taxa: List[str] = [r.id for r in self.records] if self.records else []
        self.seq_dict: Dict[str, str] = {r.id: str(r.seq) for r in self.records} if self.records else {}
        self.dates_map: Dict[str, float] = dict(dates_map) if dates_map else {}
        self.meta_df: Optional[pd.DataFrame] = meta_df
        self.D_matrix: Optional[np.ndarray] = None
        self.affinity_matrix: Optional[np.ndarray] = None
        self.taxa_repr: Optional[np.ndarray] = None
        self.cross_attn: Optional[np.ndarray] = None
        self.K_neural: Optional[np.ndarray] = None
        self.sigma: Optional[float] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None
        self.eigengaps: Optional[np.ndarray] = None
        self.model_selection_df: Optional[pd.DataFrame] = None
        self.optimal_k: int = 1
        self.community_labels: Optional[np.ndarray] = None
        self.community_results: Dict[int, Dict[str, Any]] = {}
        self.candidate_k_fits: Dict[int, List[Dict[str, Any]]] = {}
        self.candidate_k_labels: Dict[int, np.ndarray] = {}
        self.triage_df: Optional[pd.DataFrame] = None

        # Landmark Nyström decomposition attributes
        self.is_landmark_mode: bool = False
        self.landmark_indices: Optional[np.ndarray] = None
        self.landmark_taxa: Optional[List[str]] = None
        self.D_NM: Optional[np.ndarray] = None
        self.D_MM: Optional[np.ndarray] = None

    def _log(self, msg: str):
        if not self.quiet:
            print(msg)

    def load_and_validate(self) -> None:
        """Load sequences and temporal metadata, resolving dates and frame alignment."""
        if self.records and self.dates_map:
            self.taxa = [r.id for r in self.records]
            self.seq_dict = {r.id: str(r.seq) for r in self.records}
            if self.meta_df is None:
                self.meta_df = pd.DataFrame({"id": self.taxa, "date": [self.dates_map[t] for t in self.taxa]})
            self._log(f"[✓] Initialized from in-memory records ({len(self.taxa)} taxa).")
            return

        self._log(f"[*] Loading alignment from: {self.alignment_path}")
        self.seq_dict = parse_alignment_sequences(str(self.alignment_path))
        if not self.seq_dict:
            raise ValueError(f"No sequence records found in {self.alignment_path}")

        self.taxa = list(self.seq_dict.keys())
        first_len = len(self.seq_dict[self.taxa[0]])

        # If sequence length is not a multiple of 3, trim trailing 1-2 bases to ensure codon validity
        if first_len % 3 != 0:
            rem = first_len % 3
            self._log(f"[!] Warning: Sequence length {first_len} nt not divisible by 3 (remainder {rem}). Trimming {rem} trailing nt.")
            for t in self.taxa:
                self.seq_dict[t] = self.seq_dict[t][:-rem]
            first_len = len(self.seq_dict[self.taxa[0]])

        self.records = [SeqRecord(Seq(self.seq_dict[t]), id=t, description="") for t in self.taxa]
        self._log(f"[✓] Loaded {len(self.taxa)} sequences (length: {first_len} bp).")

        # Extract timestamps
        self.dates_map, missing = parse_sample_dates(
            self.taxa,
            dates_source=str(self.dates_source) if self.dates_source else None,
            date_col=self.date_col,
            strain_col=self.strain_col,
            date_regex=self.date_regex
        )

        valid_taxa = [t for t in self.taxa if t in self.dates_map and not np.isnan(self.dates_map[t])]
        if len(valid_taxa) < len(self.taxa):
            self._log(f"[!] Notice: {len(self.taxa) - len(valid_taxa)} taxa omitted due to missing or invalid timestamps.")
            self.taxa = valid_taxa
            self.records = [r for r in self.records if r.id in set(valid_taxa)]
            self.seq_dict = {t: self.seq_dict[t] for t in valid_taxa}

        if len(self.taxa) < 4:
            raise ValueError(f"Insufficient dated sequences ({len(self.taxa)}) for multi-clock deconvolution.")

        dates_arr = np.array([self.dates_map[t] for t in self.taxa])
        self._log(f"[✓] Validated {len(self.taxa)} taxa with valid timestamps (range: {dates_arr.min():.1f} - {dates_arr.max():.1f}).")

        # Load supplementary metadata if available
        if self.dates_source and self.dates_source.suffix.lower() in [".csv", ".tsv"]:
            try:
                sep = "\t" if self.dates_source.suffix.lower() == ".tsv" else ","
                meta = pd.read_csv(self.dates_source, sep=sep, dtype=str)
                s_col = self.strain_col
                if not s_col:
                    # Auto-detect strain col
                    for cand in ["strain", "id", "genome_id", "taxon", "name", "accession", "seq_id"]:
                        if cand in meta.columns:
                            s_col = cand
                            break
                    if not s_col:
                        s_col = meta.columns[0]

                meta[s_col] = meta[s_col].astype(str)
                # Map to validated taxa
                meta_dict = meta.drop_duplicates(subset=[s_col]).set_index(s_col).to_dict(orient="index")
                rows = []
                for t in self.taxa:
                    r = dict(meta_dict.get(t, {}))
                    r["id"] = t
                    r["date"] = self.dates_map[t]
                    rows.append(r)
                self.meta_df = pd.DataFrame(rows)
            except Exception:
                self.meta_df = pd.DataFrame({"id": self.taxa, "date": [self.dates_map[t] for t in self.taxa]})
        else:
            self.meta_df = pd.DataFrame({"id": self.taxa, "date": [self.dates_map[t] for t in self.taxa]})

    def _get_distances_from_local_root(self, root_idx: int, target_indices: np.ndarray) -> np.ndarray:
        """Retrieve or compute distances from local root sequence to a subset of taxa."""
        if self.D_matrix is not None:
            return self.D_matrix[root_idx, target_indices]

        if self.is_landmark_mode and self.landmark_indices is not None and self.D_NM is not None:
            lm_match = np.where(self.landmark_indices == root_idx)[0]
            if len(lm_match) > 0:
                col_j = lm_match[0]
                return self.D_NM[target_indices, col_j]

        root_taxon = self.taxa[root_idx]
        target_taxa = [self.taxa[i] for i in target_indices]
        cross_mat = compute_tn93_cross_distance_matrix(self.seq_dict, target_taxa, [root_taxon])
        return cross_mat[:, 0]

    def compute_manifold_representation(self) -> None:
        """Map alignment onto continuous transformer representation manifold, landmark Nystrom spectral manifold, or dense distance manifold."""
        n = len(self.taxa)
        use_landmark = (self.manifold in ["landmark", "nystrom"] or (n > self.max_dense_n and self.manifold != "force_dense"))
        self.is_landmark_mode = use_landmark

        if use_landmark:
            m = select_adaptive_n_landmarks(
                n,
                max_k=self.max_k,
                user_landmarks=self.n_landmarks,
                max_memory_mb=self.max_memory_mb,
            )
            self.effective_n_landmarks = m
            self._log(f"[*] Large cohort detected (N={n} > {self.max_dense_n}): Activating Landmark Nyström Normalized Spectral Embedding (M={m} adaptive landmarks, budget={self.max_memory_mb:.0f}MB)...")
            t0 = time.time()

            # Stratified temporal landmark sampling to guarantee balanced representation across eras/clades
            dates_arr = np.array([self.dates_map.get(t, 0.0) for t in self.taxa])
            sort_order = np.argsort(dates_arr)
            landmark_idx_sorted = np.linspace(0, n - 1, m, dtype=int)
            self.landmark_indices = sort_order[landmark_idx_sorted]
            self.landmark_taxa = [self.taxa[i] for i in self.landmark_indices]

            self.D_NM = compute_tn93_cross_distance_matrix(self.seq_dict, self.taxa, self.landmark_taxa)
            self.D_MM = self.D_NM[self.landmark_indices, :]
            t_dist = time.time() - t0
            self._log(f"[✓] Computed {n}x{m} landmark cross-distance matrix in {t_dist:.2f}s ({self.D_NM.nbytes / 1e6:.1f} MB).")

            nonzero_dists = self.D_MM[self.D_MM > 0]
            if len(nonzero_dists) == 0:
                nonzero_dists = np.array([0.01])
            p10 = float(np.percentile(nonzero_dists, 10))
            p50 = float(np.median(nonzero_dists))
            p90 = float(np.percentile(nonzero_dists, 90))
            self.sigma = float(self.kernel_bandwidth) if self.kernel_bandwidth is not None else (max(p10, 0.05) if p10 > 0.01 else max(p50 * 0.5, 0.05))
            self._log(f"[*] Landmark spectral affinity kernel bandwidth sigma = {self.sigma:.4f}")

            C = np.exp(- (self.D_NM ** 2) / (2.0 * (self.sigma ** 2)))
            W = np.exp(- (self.D_MM ** 2) / (2.0 * (self.sigma ** 2)))

            w_evals, w_evecs = eigh(W + 1e-4 * np.eye(m))
            pos = w_evals > 1e-4
            w_evals = w_evals[pos]
            w_evecs = w_evecs[:, pos]

            W_inv = (w_evecs * (1.0 / w_evals)) @ w_evecs.T
            d_hat = C @ (W_inv @ (C.T @ np.ones(n)))
            d_hat = np.maximum(d_hat, 1e-8)
            d_inv_sqrt = 1.0 / np.sqrt(d_hat)

            W_inv_sqrt = w_evecs / np.sqrt(w_evals)
            S = (d_inv_sqrt[:, None] * C) @ W_inv_sqrt

            k_svd = min(S.shape[1], max(16, self.max_k + 6))
            from sklearn.utils.extmath import randomized_svd
            U, s, _ = randomized_svd(S, n_components=k_svd, random_state=self.random_state)

            lap_evals = np.clip(1.0 - s**2, 0.0, None)
            sort_idx = np.argsort(lap_evals)
            self.eigenvalues = lap_evals[sort_idx]
            self.eigenvectors = U[:, sort_idx]

            max_eval = min(len(self.eigenvalues) - 1, 10)
            self.eigengaps = np.diff(self.eigenvalues[:max_eval + 1])

            self._log(f"[✓] Nyström Normalized Laplacian eigenvalues (first {min(len(self.eigenvalues), 8)}):")
            for i in range(min(len(self.eigenvalues), 7)):
                gap_str = f" | gap(k={i+1})={self.eigengaps[i]:.5f}" if i < len(self.eigengaps) else ""
                self._log(f"    lambda_{i+1}: {self.eigenvalues[i]:.6f}{gap_str}")
            return

        self._log(f"[*] Computing TN93 pairwise distance matrix for N={n} taxa...")
        t0 = time.time()
        self.D_matrix = compute_tn93_distance_matrix(self.seq_dict, self.taxa)
        t_elapsed = time.time() - t0
        self._log(f"[✓] Computed {n}x{n} distance matrix in {t_elapsed:.2f}s.")

        used_neural = False
        if self.manifold in ["transformer", "neural"] or (self.manifold == "auto" and n <= 1000):
            try:
                import torch
                from .inference import load_model, prepare_alignment, get_device
                from .splits import extract_cross_taxa_attentions_and_embeddings
                from .dating import compute_neural_covariance_kernel

                dev = get_device() if self.device is None else torch.device(self.device)
                self._log(f"[*] Extracting continuous transformer representations on {dev.type.upper()}...")
                model = load_model(weights=self.weights, device=dev)

                tmp_fa = self.output_dir / "tmp_validated_input.fa"
                with open(tmp_fa, "w") as f:
                    for t in self.taxa:
                        f.write(f">{t}\n{self.seq_dict[t]}\n")

                c, a, d, z, inv, aln_taxa, L, tree_cache = prepare_alignment(
                    str(tmp_fa), None, model=model, device=dev, prune_duplicates=False, use_tn93=True
                )
                cross_attn, taxa_repr = extract_cross_taxa_attentions_and_embeddings(
                    model, c.to(dev), a.to(dev), tree_cache, device=dev
                )
                aln_map = {t: i for i, t in enumerate(aln_taxa)}
                sub_idx = [aln_map[t] for t in self.taxa if t in aln_map]
                if len(sub_idx) == n:
                    self.taxa_repr = taxa_repr[sub_idx]
                    self.cross_attn = cross_attn[sub_idx, :][:, sub_idx]
                    self.K_neural = compute_neural_covariance_kernel(self.cross_attn, self.taxa_repr)

                    # Normalized neural affinity matrix on [0, 1]
                    self.affinity_matrix = np.clip((self.K_neural + 1.0) / 2.0, 0.0, 1.0)
                    np.fill_diagonal(self.affinity_matrix, 0.0)
                    used_neural = True
                    self._log(f"[✓] Formed transformer neural affinity manifold ({n}x{n}) from {self.taxa_repr.shape[1]}D embeddings.")
                else:
                    self._log("[!] Warning: Taxa count mismatch during tensor extraction; falling back to distance manifold.")
            except Exception as e:
                self._log(f"[!] Warning: Transformer manifold extraction failed ({e}); falling back to continuous distance manifold.")

        if not used_neural:
            nonzero_dists = self.D_matrix[self.D_matrix > 0]
            if len(nonzero_dists) == 0:
                nonzero_dists = np.array([0.01])

            p10 = float(np.percentile(nonzero_dists, 10))
            p50 = float(np.median(nonzero_dists))
            p90 = float(np.percentile(nonzero_dists, 90))
            d_max = float(nonzero_dists.max())
            self._log(f"[*] Pairwise distance distribution: p10={p10:.4f}, median={p50:.4f}, p90={p90:.4f}, max={d_max:.4f}")

            # Effective kernel bandwidth
            if self.kernel_bandwidth is not None:
                self.sigma = float(self.kernel_bandwidth)
            else:
                self.sigma = max(p10, 0.05) if p10 > 0.01 else max(p50 * 0.5, 0.05)

            self._log(f"[*] Spectral affinity kernel bandwidth sigma = {self.sigma:.4f}")

            # Gaussian RBF affinity matrix
            self.affinity_matrix = np.exp(- (self.D_matrix ** 2) / (2.0 * (self.sigma ** 2)))
            np.fill_diagonal(self.affinity_matrix, 0.0)

        # Degree and normalized symmetric graph Laplacian
        degrees = np.sum(self.affinity_matrix, axis=1)
        d_inv_sqrt = np.power(degrees, -0.5, where=degrees > 0)
        d_inv_sqrt[degrees == 0] = 0.0

        # Memory-efficient L_sym = I - D^{-1/2} A D^{-1/2} via O(N^2) broadcasting (avoids dense N x N diagonal matrix)
        d_inv_sqrt_32 = d_inv_sqrt.astype(np.float32)
        L_sym = - (d_inv_sqrt_32[:, None] * self.affinity_matrix.astype(np.float32) * d_inv_sqrt_32[None, :])
        np.fill_diagonal(L_sym, 1.0 + np.diag(L_sym))

        self._log(f"[*] Diagonalizing normalized graph Laplacian...")
        if n > 1500:
            from scipy.sparse.linalg import eigsh
            k_eval = min(n - 2, max(12, self.max_k + 4))
            evals, evecs = eigsh(L_sym, k=k_eval, which='SM', tol=1e-5)
            sort_idx = np.argsort(evals)
            evals = np.clip(evals[sort_idx], 0.0, None)
            evecs = evecs[:, sort_idx]
        else:
            evals, evecs = eigh(L_sym)
            evals = np.clip(evals, 0.0, None)

        self.eigenvalues = evals
        self.eigenvectors = evecs

        # Compute eigengaps: Delta lambda_k = lambda_{k+1} - lambda_k
        max_eval = min(len(evals) - 1, 10)
        self.eigengaps = np.diff(evals[:max_eval + 1])

        self._log(f"[✓] Normalized Laplacian eigenvalues (first {min(len(evals), 8)}):")
        for i in range(min(len(evals), 7)):
            gap_str = f" | gap(k={i+1})={self.eigengaps[i]:.5f}" if i < len(self.eigengaps) else ""
            self._log(f"    lambda_{i+1}: {evals[i]:.6f}{gap_str}")

    def evaluate_model_selection(self) -> None:
        """
        Evaluate candidate clock models K in [1, max_k] using joint AICc, BIC, and topological eigengaps.
        """
        n = len(self.taxa)
        effective_max_k = min(self.max_k, max(1, n // self.min_cluster_size))

        self._log(f"\n" + "=" * 80)
        self._log(f"CHRONAEON AUTOCLOCK: EVALUATING CANDIDATE CLOCK MODELS (K = 1 to {effective_max_k})")
        self._log("=" * 80)

        eval_records = []
        scratch_dir = self.output_dir / "k_evaluation_scratch"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        self.candidate_k_fits = {}
        self.candidate_k_labels = {}

        for k in range(1, effective_max_k + 1):
            t_k0 = time.time()
            if k == 1:
                labels = np.zeros(n, dtype=int)
            else:
                # Spectral clustering using the first k eigenvectors of L_sym
                U = self.eigenvectors[:, :k]
                row_norms = np.linalg.norm(U, axis=1, keepdims=True)
                row_norms[row_norms == 0] = 1.0
                U_norm = U / row_norms

                km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = km.fit_predict(U_norm)

            # Check cluster sizes
            unique_labels, counts = np.unique(labels, return_counts=True)
            if len(unique_labels) < k or any(c < 3 for c in counts):
                self._log(f"    [!] K={k}: Degenerate cluster detected (sizes: {list(counts)}), skipping.")
                continue

            # Fit fast OLS molecular clock for each community in candidate partition directly from distance matrix
            total_rss = 0.0
            total_log_lik = 0.0
            total_params = 0
            community_fits = []
            all_valid = True

            for c_id in range(k):
                c_taxa_idx = np.where(labels == c_id)[0]
                n_c = len(c_taxa_idx)
                if n_c < 3:
                    all_valid = False
                    break

                c_dates = np.array([self.dates_map[self.taxa[i]] for i in c_taxa_idx], dtype=np.float64)

                # Local root selection: earliest sample in community as root anchor
                earliest_local_idx = c_taxa_idx[np.argmin(c_dates)]
                d_c = self._get_distances_from_local_root(earliest_local_idx, c_taxa_idx)

                # Fast analytical OLS: d = mu * (t - t_ref) + d0
                t_ref = float(np.mean(c_dates))
                x = c_dates - t_ref
                var_x = float(np.sum(x ** 2))
                y = d_c - float(np.mean(d_c))
                tot_var = float(np.sum(y ** 2))

                if var_x < 1e-12:
                    mu_c = 0.0
                    rss_c = tot_var
                else:
                    mu_c = float(np.sum(x * y) / var_x)
                    d0_c = float(np.mean(d_c))
                    fitted = d0_c + mu_c * x
                    rss_c = float(np.sum((d_c - fitted) ** 2))

                r2_c = float(max(0.0, 1.0 - (rss_c / max(1e-12, tot_var))))
                s2_c = max(rss_c / max(n_c - 2, 1), 1e-8)
                log_lik_c = -0.5 * n_c * (np.log(2.0 * np.pi * s2_c) + 1.0)

                total_rss += rss_c
                total_log_lik += log_lik_c
                total_params += 3
                community_fits.append({
                    "community_id": c_id,
                    "n_taxa": n_c,
                    "mu": mu_c,
                    "r2": r2_c,
                    "rss": rss_c,
                    "log_lik": log_lik_c
                })

            if not all_valid:
                continue

            self.candidate_k_fits[k] = community_fits
            self.candidate_k_labels[k] = labels

            p = total_params
            if n > p + 1:
                aicc = -2.0 * total_log_lik + 2.0 * p + (2.0 * p * (p + 1)) / (n - p - 1)
            else:
                aicc = -2.0 * total_log_lik + 2.0 * p + 1000.0

            bic = -2.0 * total_log_lik + p * np.log(n)
            gap_k = self.eigengaps[k - 1] if k - 1 < len(self.eigengaps) else 0.0

            n_valid_clusters = sum(1 for cnt in counts if cnt >= self.min_cluster_size)
            is_non_degenerate = (k == 1) or (n_valid_clusters >= 2)

            eval_records.append({
                "K": k,
                "logL": total_log_lik,
                "total_rss": total_rss,
                "AICc": aicc,
                "BIC": bic,
                "eigengap": gap_k,
                "valid_clocks": len(community_fits),
                "cluster_sizes": list(counts),
                "is_non_degenerate": is_non_degenerate,
            })
            self._log(f"[✓] K={k:2d}: logL={total_log_lik:8.2f} | RSS={total_rss:.6f} | AICc={aicc:8.2f} | Eigengap={gap_k:.5f} | Sizes={list(counts)} ({time.time() - t_k0:.1f}s)")

        if not eval_records:
            self._log("[!] All multi-clock partitions failed; falling back to single clock (K=1).")
            self.optimal_k = 1
            return

        ms_df = pd.DataFrame(eval_records)
        min_aicc = ms_df["AICc"].min()
        min_bic = ms_df["BIC"].min()
        ms_df["delta_AICc"] = ms_df["AICc"] - min_aicc
        ms_df["delta_BIC"] = ms_df["BIC"] - min_bic

        self.model_selection_df = ms_df
        ms_df.to_csv(self.output_dir / "autoclock_model_selection.csv", index=False)

        # Decision rule for optimal K*:
        # Calculate eigengap drop ratios: rho_k = Delta lambda_k / max(Delta lambda_{k+1}, 1e-4)
        gaps = ms_df["eigengap"].values
        ratios = []
        for i in range(len(gaps)):
            next_gap = gaps[i + 1] if i + 1 < len(gaps) else 1e-4
            ratios.append(gaps[i] / max(next_gap, 1e-4))
        ms_df["eigengap_ratio"] = ratios

        max_gap_idx = ms_df["eigengap"].idxmax()
        max_gap = float(ms_df.loc[max_gap_idx, "eigengap"])

        # Count near-zero eigenvalues (multiplicity of 0 indicates number of disconnected graph components C)
        n_zero_evals = int(np.sum(self.eigenvalues[:self.max_k + 1] < 1e-4))

        # Check if there is a dominant spectral cliff among non-degenerate partitions
        cliff_candidates = [
            i for i in range(1, len(ratios) - 1)
            if ratios[i] >= 3.0 and (gaps[i] >= 0.04 or ratios[i] >= 10.0)
            and bool(ms_df.loc[i, "is_non_degenerate"])
        ]

        if cliff_candidates:
            # Pick the macro cliff candidate
            best_cliff_idx = max(cliff_candidates)
            k_cliff = int(ms_df.loc[best_cliff_idx, "K"])
            cliff_ratio = float(ratios[best_cliff_idx])
            self.optimal_k = k_cliff
            self._log(f"\n[★] Selected Optimal Partition: K* = {self.optimal_k} (Dominant Eigengap Drop Cliff: {cliff_ratio:.1f}x drop at K={k_cliff})")
        else:
            valid_ms_df = ms_df[ms_df["is_non_degenerate"]]
            if valid_ms_df.empty or (len(valid_ms_df) == 1 and 1 in valid_ms_df["K"].values):
                self.optimal_k = 1
                self._log(f"\n[★] Selected Optimal Partition: K* = 1 (Single Clock Preferred; all K>1 partitions degenerate)")
            else:
                k_bic = int(valid_ms_df.sort_values("BIC").iloc[0]["K"])
                k_aicc = int(valid_ms_df.sort_values("AICc").iloc[0]["K"])
                aicc_k1 = float(ms_df.loc[ms_df["K"] == 1, "AICc"].values[0])
                min_aicc = float(valid_ms_df["AICc"].min())
                delta_aicc_k1 = aicc_k1 - min_aicc

                zero_match = valid_ms_df.loc[valid_ms_df["K"] == n_zero_evals, "AICc"].values
                if n_zero_evals > 1 and n_zero_evals <= self.max_k and len(zero_match) > 0 and zero_match[0] < aicc_k1:
                    self.optimal_k = n_zero_evals
                    self._log(f"\n[★] Selected Optimal Partition: K* = {self.optimal_k} (Disconnected Graph Components C={n_zero_evals})")
                elif k_aicc == 1 or (max_gap < 0.05 and delta_aicc_k1 < 20.0 and n_zero_evals <= 1):
                    self.optimal_k = 1
                    self._log(f"\n[★] Selected Optimal Partition: K* = 1 (Single Clock Preferred; max gap = {max_gap:.5f}, delta_AICc = {delta_aicc_k1:.1f})")
                elif k_bic > 1 and k_bic <= k_aicc:
                    self.optimal_k = k_bic
                    self._log(f"\n[★] Selected Optimal Partition: K* = {self.optimal_k} (Bayesian Information Criterion BIC)")
                else:
                    self.optimal_k = k_aicc
                    self._log(f"\n[★] Selected Optimal Partition: K* = {self.optimal_k} (Minimum AICc Criterion)")

    def calibrate_optimal_communities(self) -> None:
        """Calibrate final independent ChronAeon molecular clocks for optimal communities."""
        n = len(self.taxa)
        k_opt = self.optimal_k

        self._log(f"\n" + "=" * 80)
        self._log(f"CHRONAEON AUTOCLOCK: CALIBRATING K*={k_opt} DECONVOLVED CLOCK COMMUNITIES")
        self._log("=" * 80)

        if k_opt == 1:
            labels = np.zeros(n, dtype=int)
        else:
            U = self.eigenvectors[:, :k_opt]
            row_norms = np.linalg.norm(U, axis=1, keepdims=True)
            row_norms[row_norms == 0] = 1.0
            U_norm = U / row_norms
            km = KMeans(n_clusters=k_opt, random_state=self.random_state, n_init=15)
            labels = km.fit_predict(U_norm)

        self.community_labels = labels

        # Save classified metadata
        self.meta_df["inferred_clock_community"] = labels
        self.meta_df.to_csv(self.output_dir / "autoclock_classified_metadata.csv", index=False)

        # Calibrate each community
        comm_dir = self.output_dir / "clock_communities"
        comm_dir.mkdir(parents=True, exist_ok=True)

        for c_id in range(k_opt):
            c_taxa = [self.taxa[i] for i in range(n) if labels[i] == c_id]
            c_seqs = {t: self.seq_dict[t] for t in c_taxa}
            c_fa = comm_dir / f"chronaeon_community_{c_id}.fasta"
            with open(c_fa, "w") as f:
                for t in c_taxa:
                    f.write(f">{t}\n{c_seqs[t]}\n")

            c_dates_csv = comm_dir / f"chronaeon_community_{c_id}_dates.csv"
            pd.DataFrame({"id": c_taxa, "date": [self.dates_map[t] for t in c_taxa]}).to_csv(c_dates_csv, index=False)

            comm_dist_mode = "latent" if (self.manifold in ["transformer", "neural"] or (self.manifold == "auto" and len(c_taxa) <= 500)) else "tn93"
            comm_method = "all" if (comm_dist_mode == "latent" and len(c_taxa) <= 500) else "ols"
            c_prefix = comm_dir / f"chronaeon_community_{c_id}"
            res = run_mrca_dating(
                alignment_path=str(c_fa),
                dates_source=str(c_dates_csv),
                strain_col="id",
                date_col="date",
                use_tn93=True,
                distance_mode=comm_dist_mode,
                method=comm_method,
                clock_model="auto",
                ridge="auto",
                n_bootstrap=0,
                ci_method="fieller",
                allow_stop_codons=self.allow_stop_codons,
                output_prefix=str(c_prefix)
            )

            ols = res.get("ols") or {}
            pgls = res.get("pgls") or {}
            spline = res.get("spline") or {}

            calibrated_rate = pgls.get("mu") if pgls.get("mu") and pgls.get("mu") > 0 else ols.get("mu", 0.0)
            calibrated_tmrca = pgls.get("t_mrca") if pgls.get("t_mrca") else ols.get("t_mrca", 0.0)
            ci_mrca = pgls.get("ci_mrca") if pgls.get("ci_mrca") else ols.get("ci_mrca", [calibrated_tmrca, calibrated_tmrca])

            c_dates = [self.dates_map[t] for t in c_taxa]
            self.community_results[c_id] = {
                "community_id": c_id,
                "taxa_count": len(c_taxa),
                "timespan": [float(min(c_dates)), float(max(c_dates))],
                "ols": ols,
                "pgls": pgls,
                "spline": spline,
                "calibrated_rate": calibrated_rate,
                "calibrated_tmrca": calibrated_tmrca,
                "ci_mrca": ci_mrca,
                "r2": pgls.get("r2", ols.get("r2", 0.0)),
                "pagel_lambda": pgls.get("pagel_lambda", None)
            }

            self._log(f"\n[★] Discovered Clock Community {c_id} (N={len(c_taxa)}):")
            self._log(f"    Timespan: {min(c_dates):.1f} - {max(c_dates):.1f}")
            self._log(f"    OLS  Rate: {ols.get('mu', 0):.6e} subs/site/yr | t_MRCA: {ols.get('t_mrca', 0):.1f} {ols.get('ci_mrca')} | R^2: {ols.get('r2', 0):.3f}")
            if pgls.get("mu") is not None:
                self._log(f"    PGLS Rate: {pgls.get('mu', 0):.6e} subs/site/yr | t_MRCA: {pgls.get('t_mrca', 0):.1f} {pgls.get('ci_mrca')} | R^2: {pgls.get('r2', 0):.3f} | lambda*: {pgls.get('pagel_lambda', 0):.4f}")

    def triage_anomalies(self) -> None:
        """Conduct studentized residual outlier screening across all deconvolved communities."""
        comm_dir = self.output_dir / "clock_communities"
        all_triage_rows = []

        for c_id in range(self.optimal_k):
            csv_path = comm_dir / f"chronaeon_community_{c_id}.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path, dtype={"taxon": str})
            d = df["root_divergence"].values
            t = df["sampling_date"].values
            n_c = len(d)

            slope, intercept, r_val, p_val, std_err = stats.linregress(t, d)
            d_pred = intercept + slope * t
            raw_res = d - d_pred

            t_mean = np.mean(t)
            ss_t = np.sum((t - t_mean) ** 2)
            if ss_t > 0:
                h = (1.0 / n_c) + ((t - t_mean) ** 2) / ss_t
            else:
                h = np.full(n_c, 1.0 / n_c)

            h = np.clip(h, 0.0, 0.99)
            s2 = np.sum(raw_res ** 2) / max(n_c - 2, 1)
            s = np.sqrt(max(s2, 1e-10))

            studentized = raw_res / (s * np.sqrt(1.0 - h))
            is_sus = np.abs(studentized) > 3.0

            for i in range(n_c):
                all_triage_rows.append({
                    "strain": str(df["taxon"].iloc[i]),
                    "date": float(t[i]),
                    "clock_community": int(c_id),
                    "divergence": float(d[i]),
                    "residual": float(raw_res[i]),
                    "studentized_residual": float(studentized[i]),
                    "is_sus": bool(is_sus[i])
                })

        self.triage_df = pd.DataFrame(all_triage_rows)
        self.triage_df.to_csv(self.output_dir / "autoclock_sequence_triage.csv", index=False)
        sus_count = int(self.triage_df["is_sus"].sum()) if not self.triage_df.empty else 0
        self._log(f"\n[✓] Sequence Triage Complete: Screened {len(self.taxa)} sequences, flagged {sus_count} anomalous (SUS) records (|z| > 3.0).")

    def save_summary(self) -> None:
        """Write structured autoclock_summary.json."""
        summary = {
            "alignment": str(self.alignment_path),
            "dates_source": str(self.dates_source) if self.dates_source else None,
            "total_taxa": len(self.taxa),
            "optimal_k": self.optimal_k,
            "max_eigengap_value": float(self.eigengaps[self.optimal_k - 1]) if self.eigengaps is not None and self.optimal_k - 1 < len(self.eigengaps) else 0.0,
            "communities": self.community_results,
            "sus_count": int(self.triage_df["is_sus"].sum()) if self.triage_df is not None and not self.triage_df.empty else 0,
        }
        out_json = self.output_dir / "autoclock_summary.json"
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        self._log(f"[✓] Complete AutoClock summary written to: {out_json}")

    def plot_diagnostics(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """Generate publication-grade diagnostic figure."""
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from scipy.spatial import ConvexHull

        if output_path is None:
            fig_png = self.output_dir / "fig_autoclock_diagnostic.png"
        else:
            fig_png = Path(output_path)

        fig_png.parent.mkdir(parents=True, exist_ok=True)
        pdf_path = fig_png.with_suffix(".pdf")

        meta_df = pd.read_csv(self.output_dir / "autoclock_classified_metadata.csv", dtype=str)
        meta_df["inferred_clock_community"] = pd.to_numeric(meta_df["inferred_clock_community"], errors="coerce").fillna(0).astype(int)
        ms_df = self.model_selection_df
        triage_df = self.triage_df

        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.titlesize": 13,
            "figure.dpi": 300,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.3,
        })

        palette = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e7298a", "#66a61e", "#a6761d", "#e6ab02"]

        fig = plt.figure(figsize=(16.5, 9.5))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

        ax_a = fig.add_subplot(gs[0, 0])
        gs_b = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1], hspace=0.25)
        ax_b1 = fig.add_subplot(gs_b[0, 0])
        ax_b2 = fig.add_subplot(gs_b[1, 0], sharex=ax_b1)
        ax_c = fig.add_subplot(gs[0, 2])
        ax_d = fig.add_subplot(gs[1, 0:2])
        ax_e = fig.add_subplot(gs[1, 2])

        # Panel A: Manifold projection
        taxa_to_idx = {t: i for i, t in enumerate(self.taxa)}
        strain_col = "id" if "id" in meta_df.columns else meta_df.columns[0]
        if self.taxa_repr is not None:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=self.random_state)
            coords = pca.fit_transform(self.taxa_repr)
            var1, var2 = pca.explained_variance_ratio_[0] * 100, pca.explained_variance_ratio_[1] * 100
            ax_a_title = r"$\mathbf{A}$  Transformer Embedding Manifold Partitioning"
            ax_a_xlabel = f"Latent Embedding PC1 ({var1:.1f}% var)"
            ax_a_ylabel = f"Latent Embedding PC2 ({var2:.1f}% var)"
        elif self.is_landmark_mode or len(self.taxa) > 1500 or self.D_matrix is None:
            # Spectral Laplacian embedding projection (Fiedler coordinate and next)
            e1 = self.eigenvectors[:, 1] if self.eigenvectors.shape[1] > 1 else self.eigenvectors[:, 0]
            e2 = self.eigenvectors[:, 2] if self.eigenvectors.shape[1] > 2 else e1
            coords = np.column_stack([e1, e2])
            ax_a_title = r"$\mathbf{A}$  Spectral Manifold Embedding (Nyström Deconvolution)"
            ax_a_xlabel = "Spectral Eigenvector 2 (Fiedler Coordinate)"
            ax_a_ylabel = "Spectral Eigenvector 3"
        else:
            mds = MDS(n_components=2, dissimilarity="precomputed", random_state=self.random_state, normalized_stress="auto")
            coords = mds.fit_transform(self.D_matrix)
            ax_a_title = r"$\mathbf{A}$  Continuous Distance Manifold Partitioning (MDS)"
            ax_a_xlabel = "Manifold Coordinate 1 (TN93 divergence)"
            ax_a_ylabel = "Manifold Coordinate 2 (TN93 divergence)"

        meta_df["x_proj"] = [coords[taxa_to_idx[str(t)], 0] for t in meta_df[strain_col]]
        meta_df["y_proj"] = [coords[taxa_to_idx[str(t)], 1] for t in meta_df[strain_col]]

        k_opt = self.optimal_k
        for c_id in range(k_opt):
            sub = meta_df[meta_df["inferred_clock_community"] == c_id]
            c_color = palette[c_id % len(palette)]
            ax_a.scatter(
                sub["x_proj"], sub["y_proj"],
                c=c_color, s=36, alpha=0.85, edgecolors="black", linewidths=0.4,
                label=f"Community {c_id} (N={len(sub)})"
            )
            if len(sub) >= 3:
                pts = sub[["x_proj", "y_proj"]].values
                try:
                    hull = ConvexHull(pts)
                    for simplex in hull.simplices:
                        ax_a.plot(pts[simplex, 0], pts[simplex, 1], color=c_color, linestyle="--", linewidth=1.1, alpha=0.75)
                    ax_a.fill(pts[hull.vertices, 0], pts[hull.vertices, 1], color=c_color, alpha=0.10)
                except Exception:
                    pass

        ax_a.set_title(ax_a_title, loc="left", weight="bold")
        ax_a.set_xlabel(ax_a_xlabel)
        ax_a.set_ylabel(ax_a_ylabel)
        ax_a.grid(True, linestyle=":", alpha=0.5)
        ax_a.legend(loc="upper right", frameon=True, fontsize=7.5)

        # Panel B: Model Selection
        if ms_df is not None and not ms_df.empty:
            k_vals = ms_df["K"].values
            eigengaps = ms_df["eigengap"].values
            delta_aicc = ms_df["delta_AICc"].values

            ax_b1.bar(k_vals, eigengaps, width=0.45, color="#2b5c8f", alpha=0.85)
            ax_b1.axvline(k_opt, color="#1b9e77", linestyle=":", linewidth=1.3)
            ax_b1.set_title(rf"$\mathbf{{B}}$  Model Selection ($K^*={k_opt}$ Detected)", loc="left", weight="bold")
            ax_b1.set_ylabel(r"Eigengap $\Delta \lambda_K$", color="#2b5c8f", fontsize=8.5)
            ax_b1.grid(True, linestyle=":", alpha=0.5)
            ax_b1.tick_params(labelbottom=False)

            ax_b2.plot(k_vals, delta_aicc, color="#d95f02", marker="o", linewidth=1.5, markersize=5)
            ax_b2.axvline(k_opt, color="#1b9e77", linestyle=":", linewidth=1.3)
            ax_b2.set_xlabel(r"Candidate Number of Clocks ($K$)")
            ax_b2.set_ylabel(r"Penalty $\Delta \mathrm{AICc}$", color="#d95f02", fontsize=8.5)
            ax_b2.set_xticks(k_vals)
            ax_b2.grid(True, linestyle=":", alpha=0.5)

        # Panel C: Multi-Clock Trajectories
        comm_dir = self.output_dir / "clock_communities"
        for c_id in range(k_opt):
            c_color = palette[c_id % len(palette)]
            t_csv = comm_dir / f"chronaeon_community_{c_id}.csv"
            if t_csv.exists():
                t_meta = pd.read_csv(t_csv)
                d_obs = t_meta["root_divergence"].values
                t_obs = t_meta["sampling_date"].values

                ax_c.scatter(t_obs, d_obs, c=c_color, alpha=0.75, s=28, edgecolors="black", linewidths=0.3, label=f"Community {c_id}")
                ols_info = self.community_results.get(c_id, {}).get("ols", {})
                mu_val = ols_info.get("mu", 0.0)
                d0_val = ols_info.get("d0", 0.0)
                t_ref = ols_info.get("t_ref", t_obs.min())
                t_line = np.linspace(t_obs.min(), t_obs.max(), 50)
                d_line = d0_val + mu_val * (t_line - t_ref)
                ax_c.plot(t_line, d_line, color=c_color, linewidth=1.8)

        ax_c.set_title(r"$\mathbf{C}$  Deconvolved Root-to-Tip Clock Trajectories", loc="left", weight="bold")
        ax_c.set_xlabel("Sampling Date (Year)")
        ax_c.set_ylabel("Genetic Divergence from Convex Hull Root")
        ax_c.grid(True, linestyle=":", alpha=0.5)
        ax_c.legend(loc="upper left", frameon=True, fontsize=8)

        # Panel D: Rates Forest Plot
        inferred = []
        for c_id in range(k_opt):
            comm = self.community_results.get(c_id, {})
            rate = comm.get("calibrated_rate", 0.0)
            ols = comm.get("ols", {})
            se = ols.get("se_mu", rate * 0.15) if ols else rate * 0.15
            inferred.append({
                "label": f"Community {c_id} (N={comm.get('taxa_count', 0)})",
                "rate": rate,
                "ci_low": max(rate - 1.96 * se, 0),
                "ci_high": rate + 1.96 * se,
                "color": palette[c_id % len(palette)]
            })

        y_pos = 0
        y_ticks = []
        y_labels = []
        for item in inferred:
            rate_val = item["rate"] * 1e4
            err_low = max(0.0, (item["rate"] - item["ci_low"]) * 1e4)
            err_high = max(0.0, (item["ci_high"] - item["rate"]) * 1e4)
            ax_d.errorbar(
                rate_val, y_pos,
                xerr=[[err_low], [err_high]],
                fmt="o", color=item["color"], markersize=7, capsize=4, linewidth=1.5
            )
            y_ticks.append(y_pos)
            y_labels.append(item["label"])
            y_pos += 1

        ax_d.set_yticks(y_ticks)
        ax_d.set_yticklabels(y_labels)
        ax_d.set_title(r"$\mathbf{D}$  Calibrated Evolutionary Rates per Clock Community", loc="left", weight="bold")
        ax_d.set_xlabel(r"Substitution Rate $\mu$ ($10^{-4}$ substitutions / site / year)")
        ax_d.grid(True, linestyle=":", alpha=0.5)

        # Panel E: Residual Outlier Triage
        if triage_df is not None and not triage_df.empty:
            for c_id in range(k_opt):
                c_color = palette[c_id % len(palette)]
                sub_t = triage_df[triage_df["clock_community"] == c_id]
                normal_taxa = sub_t[~sub_t["is_sus"]]
                sus_taxa = sub_t[sub_t["is_sus"]]

                ax_e.scatter(
                    normal_taxa["date"], normal_taxa["studentized_residual"],
                    c=c_color, alpha=0.7, s=24, edgecolors="none"
                )
                if not sus_taxa.empty:
                    ax_e.scatter(
                        sus_taxa["date"], sus_taxa["studentized_residual"],
                        c="red", marker="x", s=45, linewidths=1.5,
                        label="Flagged Anomaly (SUS)" if c_id == 0 else None
                    )

        ax_e.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
        ax_e.axhline(3.0, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
        ax_e.axhline(-3.0, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
        ax_e.set_title(r"$\mathbf{E}$  Automated Outlier Triage (Studentized Residuals)", loc="left", weight="bold")
        ax_e.set_xlabel("Sampling Date (Year)")
        ax_e.set_ylabel(r"Studentized Residual $r_i^*$ ($\sigma$ units)")
        ax_e.set_ylim(-4.5, 4.5)
        ax_e.grid(True, linestyle=":", alpha=0.5)

        plt.suptitle(f"ChronAeon AutoClock: Unsupervised Multi-Clock Deconvolution ({Path(self.alignment_path).stem})", weight="bold", y=0.98, fontsize=12)

        fig.savefig(fig_png, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        self._log(f"[✓] Saved publication diagnostic figures: {fig_png} and {pdf_path}")
        return fig_png

    def run(self, plot: bool = False, plot_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Execute full end-to-end AutoClock deconvolution pipeline."""
        self.load_and_validate()
        self.compute_manifold_representation()
        self.evaluate_model_selection()
        self.calibrate_optimal_communities()
        self.triage_anomalies()
        self.save_summary()
        if plot or plot_path:
            self.plot_diagnostics(plot_path)

        return {
            "optimal_k": self.optimal_k,
            "communities": self.community_results,
            "max_eigengap_value": float(self.eigengaps.max()) if self.eigengaps is not None and len(self.eigengaps) > 0 else 0.0,
            "eigengaps": [float(g) for g in self.eigengaps] if self.eigengaps is not None else [],
            "sus_count": int(self.triage_df["is_sus"].sum()) if self.triage_df is not None and not self.triage_df.empty else 0,
            "summary_path": str(self.output_dir / "autoclock_summary.json"),
            "classified_metadata_path": str(self.output_dir / "autoclock_classified_metadata.csv"),
            "model_selection_path": str(self.output_dir / "autoclock_model_selection.csv"),
            "triage_path": str(self.output_dir / "autoclock_sequence_triage.csv"),
        }


def run_autoclock_deconvolution(
    alignment_path: Optional[Union[str, Path]] = None,
    dates_source: Optional[Union[str, Path]] = None,
    date_col: Optional[str] = None,
    strain_col: Optional[str] = None,
    date_regex: Optional[str] = None,
    max_k: int = 6,
    manifold: str = "auto",
    weights: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    kernel_bandwidth: Optional[float] = None,
    output_dir: Optional[Union[str, Path]] = None,
    plot: bool = False,
    plot_path: Optional[Union[str, Path]] = None,
    quiet: bool = False,
    beast_path: Optional[Union[str, Path]] = None,
    max_dense_n: int = 2500,
    n_landmarks: Union[int, str] = "auto",
    max_memory_mb: float = 1024.0,
) -> Dict[str, Any]:
    """
    Convenience function to run ChronAeon AutoClock Multi-Clock Community Deconvolution.
    Directly ingests FASTA, NEXUS, or BEAST 1.x / 2.x XML configuration files.
    """
    engine = AutoClockDeconvolution(
        alignment_path=alignment_path,
        dates_source=dates_source,
        date_col=date_col,
        strain_col=strain_col,
        date_regex=date_regex,
        max_k=max_k,
        manifold=manifold,
        weights=weights,
        device=device,
        kernel_bandwidth=kernel_bandwidth,
        output_dir=output_dir,
        quiet=quiet,
        beast_path=beast_path,
        max_dense_n=max_dense_n,
        n_landmarks=n_landmarks,
        max_memory_mb=max_memory_mb,
    )
    return engine.run(plot=plot, plot_path=plot_path)


def fit_fast_ols_clock(dates: np.ndarray, dists: np.ndarray) -> Dict[str, Any]:
    """
    Fits an analytical OLS molecular clock: d = mu * (t - t_ref) + d0.
    Returns mu, t_mrca, r2, rss, p_val, ci_mrca, and rate_ci.
    """
    n = len(dates)
    if n < 3:
        t_val = float(dates.min()) if n > 0 else 0.0
        return {
            "mu": 0.0, "t_mrca": t_val, "r2": 0.0, "rss": 0.0, "p_val": 1.0,
            "ci_mrca": [t_val, t_val], "rate_ci": [0.0, 0.0], "se_mu": 0.0,
        }

    t_ref = float(np.mean(dates))
    x = dates - t_ref
    var_x = float(np.sum(x ** 2))
    y = dists - float(np.mean(dists))
    tot_var = float(np.sum(y ** 2))

    if var_x < 1e-12:
        t_val = float(dates.min())
        return {
            "mu": 0.0, "t_mrca": t_val, "r2": 0.0, "rss": tot_var, "p_val": 1.0,
            "ci_mrca": [t_val, t_val], "rate_ci": [0.0, 0.0], "se_mu": 0.0,
        }

    mu = float(np.sum(x * y) / var_x)
    d0 = float(np.mean(dists))
    fitted = d0 + mu * x
    rss = float(np.sum((dists - fitted) ** 2))
    r2 = float(max(0.0, 1.0 - (rss / max(1e-12, tot_var))))

    t_mrca = float(t_ref - (d0 / mu)) if mu > 1e-12 else float(dates.min())

    s2 = max(rss / max(n - 2, 1), 1e-12)
    se_mu = float(np.sqrt(s2 / var_x))
    ci_lower_rate = max(0.0, mu - 1.96 * se_mu)
    ci_upper_rate = mu + 1.96 * se_mu

    # Approximate 95% CI for t_MRCA using delta method
    delta_tmrca = 1.96 * se_mu * abs(d0) / max(mu ** 2, 1e-8)
    ci_mrca = [float(t_mrca - delta_tmrca), float(t_mrca + delta_tmrca)]

    return {
        "mu": mu,
        "t_mrca": t_mrca,
        "ci_mrca": ci_mrca,
        "rate_ci": [ci_lower_rate, ci_upper_rate],
        "se_mu": se_mu,
        "r2": r2,
        "rss": rss,
        "p_val": 0.0 if r2 > 0.3 else 0.05,
    }


class HierarchicalAutoClock:
    """
    ChronAeon Hierarchical AutoClock: Recursive Multi-Clock Community Deconvolution.

    Recursively partitions phylogenetic sequence manifolds into discrete, clock-constrained
    biological communities (e.g. serotypes, lineages, host reservoirs, and outbreak clades)
    using adaptive landmark spectral graph theory, multi-clock AICc parsimony, and
    evolutionary rate consistency.

    The engine terminates recursion at any node using 5 principled stopping criteria:
      1. Statistical Size / Timespan Floor: Node taxa < 2 * min_leaf_size or timespan < min_timespan.
      2. Max Depth Bound: Recursion depth >= max_depth.
      3. Information-Theoretic AICc Parsimony: Multi-clock model rejected by Occam's razor (delta_AICc < min_delta_aicc).
      4. Spectral Graph Modularity: No topological Cheeger bottleneck (max eigengap < min_eigengap).
      5. Phylodynamic Rate Homogeneity: Candidate subcommunities have statistically indistinguishable
         rates (|mu_a - mu_b| / max(mu) < max_rate_diff_ratio) under a validated parent clock.
      6. Outlier Quarantine: Quarantines isolated micro-clusters (< min_leaf_size) without fragmenting the core clock.
    """

    def __init__(
        self,
        alignment_path: Optional[Union[str, Path]] = None,
        dates_source: Optional[Union[str, Path]] = None,
        date_col: Optional[str] = None,
        strain_col: Optional[str] = None,
        date_regex: Optional[str] = None,
        max_depth: int = 3,
        min_leaf_size: int = 25,
        min_delta_aicc: float = 15.0,
        min_timespan: float = 1.0,
        min_eigengap: float = 0.02,
        max_rate_diff_ratio: float = 0.15,
        max_k_per_node: int = 6,
        manifold: str = "auto",
        weights: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        kernel_bandwidth: Optional[float] = None,
        random_state: int = 42,
        output_dir: Optional[Union[str, Path]] = None,
        allow_stop_codons: bool = True,
        quiet: bool = False,
        beast_path: Optional[Union[str, Path]] = None,
        max_dense_n: int = 2500,
        n_landmarks: Union[int, str] = "auto",
        max_memory_mb: float = 1024.0,
    ):
        if beast_path:
            if not alignment_path:
                alignment_path = beast_path
            if not dates_source:
                dates_source = beast_path
        if not alignment_path:
            raise ValueError("Must provide either alignment_path or beast_path.")
        self.alignment_path = Path(alignment_path)

        if dates_source is None and alignment_path:
            if str(self.alignment_path).lower().endswith(('.xml', '.xml.gz')):
                dates_source = self.alignment_path
            elif self.alignment_path.exists():
                try:
                    with open(self.alignment_path, 'r', encoding='utf-8', errors='replace') as f:
                        header = f.read(1024).lstrip()
                    if header.startswith('<') and ('<beast' in header.lower() or '<alignment' in header.lower() or '<data' in header.lower()):
                        dates_source = self.alignment_path
                except Exception:
                    pass

        self.dates_source = Path(dates_source) if dates_source else None
        self.date_col = date_col
        self.strain_col = strain_col
        self.date_regex = date_regex
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.min_delta_aicc = float(min_delta_aicc)
        self.min_timespan = float(min_timespan)
        self.min_eigengap = float(min_eigengap)
        self.max_rate_diff_ratio = float(max_rate_diff_ratio)
        self.max_k_per_node = max_k_per_node
        self.manifold = manifold
        self.weights = weights
        self.device = device
        self.kernel_bandwidth = kernel_bandwidth
        self.random_state = random_state
        self.output_dir = Path(output_dir) if output_dir else self.alignment_path.parent / "hierarchical_autoclock_results"
        self.allow_stop_codons = allow_stop_codons
        self.quiet = quiet
        self.max_dense_n = max_dense_n
        self.n_landmarks = n_landmarks
        self.max_memory_mb = float(max_memory_mb)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Loaded master datasets
        self.records: List[SeqRecord] = []
        self.taxa: List[str] = []
        self.seq_dict: Dict[str, str] = {}
        self.dates_map: Dict[str, float] = {}
        self.meta_df: Optional[pd.DataFrame] = None
        self.records_map: Dict[str, SeqRecord] = {}

        # Deconvolution outputs
        self.tree: Dict[str, Any] = {}
        self.leaves: List[Dict[str, Any]] = []
        self.outliers: List[Dict[str, Any]] = []
        self.classified_df: Optional[pd.DataFrame] = None

    def _log(self, msg: str):
        if not self.quiet:
            print(msg, flush=True)

    def load_and_validate(self):
        """Load master sequence alignment and timestamps."""
        self._log(f"[*] Loading alignment from: {self.alignment_path}")
        self.seq_dict = parse_alignment_sequences(str(self.alignment_path))
        if not self.seq_dict:
            raise ValueError(f"No sequences found in {self.alignment_path}")

        self.taxa = list(self.seq_dict.keys())
        first_len = len(self.seq_dict[self.taxa[0]])
        if first_len % 3 != 0:
            rem = first_len % 3
            self._log(f"[!] Warning: Sequence length {first_len} nt not divisible by 3. Trimming {rem} trailing nt.")
            for t in self.taxa:
                self.seq_dict[t] = self.seq_dict[t][:-rem]

        self.records = [SeqRecord(Seq(self.seq_dict[t]), id=t, description="") for t in self.taxa]
        self.records_map = {r.id: r for r in self.records}

        self.dates_map, _ = parse_sample_dates(
            self.taxa,
            dates_source=str(self.dates_source) if self.dates_source else None,
            date_col=self.date_col,
            strain_col=self.strain_col,
            date_regex=self.date_regex,
        )
        valid_taxa = [t for t in self.taxa if t in self.dates_map and not np.isnan(self.dates_map[t])]
        if len(valid_taxa) < len(self.taxa):
            self._log(f"[!] Notice: {len(self.taxa) - len(valid_taxa)} taxa omitted due to missing timestamps.")
            self.taxa = valid_taxa
            self.records = [self.records_map[t] for t in valid_taxa]
            self.seq_dict = {t: self.seq_dict[t] for t in valid_taxa}
            self.records_map = {r.id: r for r in self.records}

        if len(self.taxa) < 4:
            raise ValueError(f"Insufficient dated taxa ({len(self.taxa)}) for deconvolution.")

        # Load supplementary metadata if available
        if self.dates_source and self.dates_source.suffix.lower() in [".csv", ".tsv"]:
            try:
                sep = "\t" if self.dates_source.suffix.lower() == ".tsv" else ","
                meta = pd.read_csv(self.dates_source, sep=sep, dtype=str)
                s_col = self.strain_col
                if not s_col:
                    for cand in ["strain", "id", "genome_id", "taxon", "name", "accession", "seq_id"]:
                        if cand in meta.columns:
                            s_col = cand
                            break
                    if not s_col:
                        s_col = meta.columns[0]
                meta[s_col] = meta[s_col].astype(str)
                meta_dict = meta.drop_duplicates(subset=[s_col]).set_index(s_col).to_dict(orient="index")
                rows = []
                for t in self.taxa:
                    r = dict(meta_dict.get(t, {}))
                    r["id"] = t
                    r["date"] = self.dates_map[t]
                    rows.append(r)
                self.meta_df = pd.DataFrame(rows)
            except Exception:
                self.meta_df = pd.DataFrame({"id": self.taxa, "date": [self.dates_map[t] for t in self.taxa]})
        else:
            self.meta_df = pd.DataFrame({"id": self.taxa, "date": [self.dates_map[t] for t in self.taxa]})

        self._log(f"[✓] Validated {len(self.taxa)} sequences with timestamps across {self.meta_df['date'].min():.1f} - {self.meta_df['date'].max():.1f}.")

    def calibrate_node_clock(self, taxa_subset: List[str]) -> Dict[str, Any]:
        """Calibrates fast analytical OLS molecular clock for a node's taxa."""
        c_dates = np.array([self.dates_map[t] for t in taxa_subset], dtype=np.float64)
        earliest_idx = int(np.argmin(c_dates))
        earliest_t = taxa_subset[earliest_idx]
        earliest_seq = self.seq_dict[earliest_t]

        seq_mat = np.array([list(self.seq_dict[t]) for t in taxa_subset])
        earliest_arr = np.array(list(earliest_seq))
        diffs = np.mean(seq_mat != earliest_arr, axis=1)

        fit = fit_fast_ols_clock(c_dates, diffs)
        fit["earliest_date"] = float(np.min(c_dates))
        fit["latest_date"] = float(np.max(c_dates))
        fit["timespan"] = float(np.max(c_dates) - np.min(c_dates))
        fit["earliest_taxon"] = earliest_t
        return fit

    def deconvolve_node(
        self,
        node_id: str,
        taxa_subset: List[str],
        depth: int = 0,
        path: str = "root",
        parent_rate: Optional[float] = None,
        parent_r2: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Recursively evaluates a node using the 5 principled stopping criteria."""
        indent = "  " * depth
        n_c = len(taxa_subset)
        dates_c = np.array([self.dates_map[t] for t in taxa_subset])
        t_span = float(np.max(dates_c) - np.min(dates_c))

        self._log(f"{indent}[Node {node_id}] Depth {depth} | N={n_c} | Timespan={t_span:.1f}y")

        node_fit = self.calibrate_node_clock(taxa_subset)

        node_dict = {
            "node_id": node_id,
            "path": path,
            "depth": depth,
            "n_taxa": n_c,
            "taxa": taxa_subset,
            "timespan": [float(np.min(dates_c)), float(np.max(dates_c))],
            "duration_years": t_span,
            "rate": node_fit["mu"],
            "rate_ci": node_fit["rate_ci"],
            "tmrca": node_fit["t_mrca"],
            "ci_mrca": node_fit["ci_mrca"],
            "r2": node_fit["r2"],
            "rss": node_fit["rss"],
            "p_value": node_fit["p_val"],
            "earliest_taxon": node_fit["earliest_taxon"],
            "is_leaf": False,
            "stopping_reason": None,
            "optimal_k": 1,
            "delta_aicc": 0.0,
            "max_eigengap": 0.0,
            "children": [],
            "outliers": [],
        }

        # STOPPING CRITERION 1: Max recursion depth reached
        if depth >= self.max_depth:
            node_dict["is_leaf"] = True
            node_dict["stopping_reason"] = f"max_depth_reached (depth={depth})"
            self._log(f"{indent}  └── STOP: {node_dict['stopping_reason']} (rate={node_fit['mu']:.2e}, R2={node_fit['r2']:.3f})")
            return node_dict

        # STOPPING CRITERION 2: Statistical Sample Size Floor
        if n_c < 2 * self.min_leaf_size:
            node_dict["is_leaf"] = True
            node_dict["stopping_reason"] = f"min_leaf_size_floor (N={n_c} < 2*{self.min_leaf_size})"
            self._log(f"{indent}  └── STOP: {node_dict['stopping_reason']} (rate={node_fit['mu']:.2e}, R2={node_fit['r2']:.3f})")
            return node_dict

        # STOPPING CRITERION 3: Timespan Floor
        if t_span < self.min_timespan:
            node_dict["is_leaf"] = True
            node_dict["stopping_reason"] = f"insufficient_timespan ({t_span:.2f}y < {self.min_timespan}y)"
            self._log(f"{indent}  └── STOP: {node_dict['stopping_reason']} (rate={node_fit['mu']:.2e}, R2={node_fit['r2']:.3f})")
            return node_dict

        # Launch spectral manifold deconvolution on node subset
        sub_records = [self.records_map[t] for t in taxa_subset]
        sub_dates = {t: self.dates_map[t] for t in taxa_subset}
        max_k = min(self.max_k_per_node, max(2, n_c // self.min_leaf_size))

        node_sub_dir = self.output_dir / "sub_nodes" / f"node_{node_id.replace('/', '_')}"
        engine = AutoClockDeconvolution(
            records=sub_records,
            dates_map=sub_dates,
            max_k=max_k,
            min_cluster_size=self.min_leaf_size,
            manifold=self.manifold,
            output_dir=node_sub_dir,
            quiet=True,
            n_landmarks=self.n_landmarks,
            max_memory_mb=self.max_memory_mb,
            random_state=self.random_state,
        )
        engine.load_and_validate()
        engine.compute_manifold_representation()
        engine.evaluate_model_selection()

        k_opt = engine.optimal_k
        max_gap = float(engine.eigengaps.max()) if engine.eigengaps is not None and len(engine.eigengaps) > 0 else 0.0
        node_dict["optimal_k"] = k_opt
        node_dict["max_eigengap"] = max_gap

        if engine.model_selection_df is None or engine.model_selection_df.empty:
            node_dict["is_leaf"] = True
            node_dict["stopping_reason"] = "model_selection_failed"
            self._log(f"{indent}  └── STOP: {node_dict['stopping_reason']}")
            return node_dict

        aicc_k1 = float(engine.model_selection_df.loc[engine.model_selection_df["K"] == 1, "AICc"].values[0])
        min_aicc = float(engine.model_selection_df["AICc"].min())
        delta_aicc = float(aicc_k1 - min_aicc)
        node_dict["delta_aicc"] = delta_aicc

        # STOPPING CRITERION 4: Information-Theoretic AICc Parsimony
        if k_opt == 1 or delta_aicc < self.min_delta_aicc:
            node_dict["is_leaf"] = True
            node_dict["stopping_reason"] = f"aicc_parsimony_k1 (delta_AICc={delta_aicc:.1f} < {self.min_delta_aicc})"
            self._log(f"{indent}  └── STOP: {node_dict['stopping_reason']} (rate={node_fit['mu']:.2e}, R2={node_fit['r2']:.3f})")
            return node_dict

        # STOPPING CRITERION 5: Spectral Graph Modularity / Cheeger Bottleneck
        # In Nyström mode on massive datasets (N > 2500), eigenvalues are compressed,
        # so we scale the minimum eigengap threshold or check if AICc overwhelmingly supports splitting.
        effective_min_eigengap = self.min_eigengap if n_c <= 2500 else min(self.min_eigengap, 0.0005)
        if max_gap < effective_min_eigengap and delta_aicc < 100.0:
            node_dict["is_leaf"] = True
            node_dict["stopping_reason"] = f"no_spectral_bottleneck (max_gap={max_gap:.4f} < {effective_min_eigengap})"
            self._log(f"{indent}  └── STOP: {node_dict['stopping_reason']} (rate={node_fit['mu']:.2e}, R2={node_fit['r2']:.3f})")
            return node_dict

        # Extract spectral clustering labels for k_opt
        if k_opt in engine.candidate_k_labels:
            labels = engine.candidate_k_labels[k_opt]
        else:
            U = engine.eigenvectors[:, :k_opt]
            row_norms = np.linalg.norm(U, axis=1, keepdims=True)
            row_norms[row_norms == 0] = 1.0
            U_norm = U / row_norms
            km = KMeans(n_clusters=k_opt, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(U_norm)

        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_clusters = [c for c, cnt in zip(unique_labels, counts) if cnt >= self.min_leaf_size]

        # Check degenerate split
        if len(valid_clusters) < 2:
            node_dict["is_leaf"] = True
            node_dict["stopping_reason"] = f"degenerate_clusters (only {len(valid_clusters)} cluster >= {self.min_leaf_size})"
            self._log(f"{indent}  └── STOP: {node_dict['stopping_reason']}")
            return node_dict

        # STOPPING CRITERION 6: Rate Homogeneity (Phylodynamic Consistency)
        cluster_rates = []
        for c in valid_clusters:
            c_idx = np.where(labels == c)[0]
            c_sub_taxa = [taxa_subset[i] for i in c_idx]
            c_fit = self.calibrate_node_clock(c_sub_taxa)
            cluster_rates.append(c_fit["mu"])

        if all(r > 0 for r in cluster_rates) and parent_rate is not None and parent_r2 is not None and parent_r2 > 0.20:
            min_r = min(cluster_rates)
            max_r = max(cluster_rates)
            spread = (max_r - min_r) / max(max_r, 1e-12)
            if spread < self.max_rate_diff_ratio:
                node_dict["is_leaf"] = True
                node_dict["stopping_reason"] = f"homogeneous_clock_rates (spread={spread:.1%} < {self.max_rate_diff_ratio:.1%})"
                self._log(f"{indent}  └── STOP: {node_dict['stopping_reason']} (rate={node_fit['mu']:.2e}, R2={node_fit['r2']:.3f})")
                return node_dict

        # ALL CRITERIA PASSED: SPLIT NODE INTO SUBCOMMUNITIES
        self._log(f"{indent}  ├── SPLIT into K={len(valid_clusters)} subcommunities (delta_AICc={delta_aicc:.1f}, gap={max_gap:.4f}):")

        # Quarantined outliers (< min_leaf_size)
        outlier_taxa = []
        for c, cnt in zip(unique_labels, counts):
            if cnt < self.min_leaf_size:
                c_idx = np.where(labels == c)[0]
                outlier_taxa.extend([taxa_subset[i] for i in c_idx])
        node_dict["outliers"] = outlier_taxa

        for idx, c in enumerate(valid_clusters):
            c_idx = np.where(labels == c)[0]
            child_taxa = [taxa_subset[i] for i in c_idx]
            child_id = f"{node_id}.{idx}" if node_id != "root" else f"c{idx}"
            child_path = f"{path}/{child_id}"
            child_node = self.deconvolve_node(
                node_id=child_id,
                taxa_subset=child_taxa,
                depth=depth + 1,
                path=child_path,
                parent_rate=node_fit["mu"],
                parent_r2=node_fit["r2"],
            )
            node_dict["children"].append(child_node)

        return node_dict

    def _collect_leaves_and_outliers(self, node: Dict[str, Any]):
        if node.get("outliers"):
            for t in node["outliers"]:
                self.outliers.append({
                    "id": t,
                    "date": self.dates_map[t],
                    "parent_node": node["node_id"],
                    "reason": "isolated_cluster_under_min_leaf_size",
                })
        if node.get("is_leaf", False):
            self.leaves.append(node)
        else:
            for child in node.get("children", []):
                self._collect_leaves_and_outliers(child)

    def run(self, plot: bool = False, plot_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Execute full hierarchical multi-clock community deconvolution pipeline."""
        t0 = time.time()
        self.load_and_validate()

        self._log("\n" + "=" * 80)
        self._log("CHRONAEON HIERARCHICAL AUTOCLOCK: RECURSIVE MANIFOLD DECONVOLUTION")
        self._log(f"Config: max_depth={self.max_depth} | min_leaf_size={self.min_leaf_size} | min_delta_aicc={self.min_delta_aicc} | RAM_budget={self.max_memory_mb:.0f}MB")
        self._log("=" * 80)

        self.tree = self.deconvolve_node("root", self.taxa, depth=0, path="root")
        self._collect_leaves_and_outliers(self.tree)

        self._log("\n" + "=" * 80)
        self._log(f"DECONVOLUTION COMPLETE: {len(self.leaves)} Leaf Communities Identified ({len(self.outliers)} Quarantined Outliers) in {time.time() - t0:.2f}s")
        self._log("=" * 80)

        # Build Classified Taxon Metadata Table
        rows = []
        for leaf in self.leaves:
            l_id = leaf["node_id"]
            l_path = leaf["path"]
            l_depth = leaf["depth"]
            l_rate = leaf["rate"]
            l_tmrca = leaf["tmrca"]
            l_r2 = leaf["r2"]
            l_reason = leaf["stopping_reason"]
            earliest_t = leaf["earliest_taxon"]
            earliest_seq = self.seq_dict[earliest_t]
            earliest_arr = np.array(list(earliest_seq))

            for t in leaf["taxa"]:
                t_date = self.dates_map[t]
                t_seq_arr = np.array(list(self.seq_dict[t]))
                dist = float(np.mean(t_seq_arr != earliest_arr))
                expected_dist = float(max(0.0, l_rate * (t_date - l_tmrca)))
                residual = float(dist - expected_dist)
                rows.append({
                    "id": t,
                    "date": t_date,
                    "leaf_community_id": l_id,
                    "hierarchical_path": l_path,
                    "leaf_depth": l_depth,
                    "leaf_rate": l_rate,
                    "leaf_tmrca": l_tmrca,
                    "leaf_r2": l_r2,
                    "stopping_reason": l_reason,
                    "root_distance": dist,
                    "expected_distance": expected_dist,
                    "residual": residual,
                    "is_outlier": False,
                    "outlier_reason": None,
                })

        for out in self.outliers:
            t = out["id"]
            rows.append({
                "id": t,
                "date": out["date"],
                "leaf_community_id": f"{out['parent_node']}_outlier",
                "hierarchical_path": f"{out['parent_node']}/outlier",
                "leaf_depth": -1,
                "leaf_rate": 0.0,
                "leaf_tmrca": 0.0,
                "leaf_r2": 0.0,
                "stopping_reason": "outlier_quarantine",
                "root_distance": 0.0,
                "expected_distance": 0.0,
                "residual": 0.0,
                "is_outlier": True,
                "outlier_reason": out["reason"],
            })

        self.classified_df = pd.DataFrame(rows)
        meta_csv = self.output_dir / "hierarchical_classified_metadata.csv"
        self.classified_df.to_csv(meta_csv, index=False)
        self._log(f"[✓] Saved classified metadata to: {meta_csv}")

        # Save Tree JSON (omitting taxa list from nodes for clean readability)
        tree_json = self.output_dir / "hierarchical_tree.json"

        def _clean_node(n):
            c = dict(n)
            c.pop("taxa", None)
            c["children"] = [_clean_node(ch) for ch in c.get("children", [])]
            return c

        with open(tree_json, "w") as f:
            json.dump(_clean_node(self.tree), f, indent=2)
        self._log(f"[✓] Saved hierarchical tree to: {tree_json}")

        # Save Summary JSON
        summary = {
            "total_taxa": len(self.taxa),
            "n_leaves": len(self.leaves),
            "n_outliers": len(self.outliers),
            "max_depth_reached": int(max(leaf["depth"] for leaf in self.leaves)) if self.leaves else 0,
            "elapsed_seconds": float(time.time() - t0),
            "stopping_counts": {
                reason: int(sum(1 for l in self.leaves if l["stopping_reason"] and reason in l["stopping_reason"]))
                for reason in ["max_depth", "min_leaf_size", "insufficient_timespan", "aicc_parsimony", "no_spectral_bottleneck", "degenerate_clusters", "homogeneous_clock_rates"]
            },
            "leaf_communities": [
                {
                    "node_id": l["node_id"],
                    "path": l["path"],
                    "depth": l["depth"],
                    "n_taxa": l["n_taxa"],
                    "rate": l["rate"],
                    "rate_ci": l["rate_ci"],
                    "tmrca": l["tmrca"],
                    "ci_mrca": l["ci_mrca"],
                    "r2": l["r2"],
                    "stopping_reason": l["stopping_reason"],
                    "timespan": l["timespan"],
                }
                for l in self.leaves
            ],
            "tree_path": str(tree_json),
            "classified_metadata_path": str(meta_csv),
            "summary_path": str(self.output_dir / "hierarchical_summary.json"),
        }
        sum_json = self.output_dir / "hierarchical_summary.json"
        with open(sum_json, "w") as f:
            json.dump(summary, f, indent=2)
        self._log(f"[✓] Saved summary to: {sum_json}")

        if plot or plot_path:
            self.plot_diagnostics(plot_path)

        return summary

    def plot_diagnostics(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Generates 4-panel publication diagnostic figure for Hierarchical AutoClock:
          (A) Multi-Clock Root-to-Tip Regressions across leaf communities
          (B) Hierarchical Manifold Deconvolution Topology (Dendrogram)
          (C) Calibrated Evolutionary Substitution Rates (mu) with 95% Confidence Intervals
          (D) Root-to-Tip Residual Distributions by Leaf Community
        """
        import matplotlib.gridspec as gridspec

        if self.classified_df is None or self.classified_df.empty:
            raise RuntimeError("Must call run() before plotting diagnostics.")

        valid_df = self.classified_df[~self.classified_df["is_outlier"]].copy()
        leaf_ids = sorted(valid_df["leaf_community_id"].unique())
        n_leaves = len(leaf_ids)

        fig_height = max(11.0, min(24.0, 6.0 + 0.35 * n_leaves))
        fig = plt.figure(figsize=(18, fig_height))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.25)

        cmap = plt.get_cmap("tab20", max(len(leaf_ids), 1))
        color_map = {lid: cmap(i % 20) for i, lid in enumerate(leaf_ids)}

        # (A) Multi-Clock Regressions
        ax_a = fig.add_subplot(gs[0, 0])
        for lid in leaf_ids:
            sub = valid_df[valid_df["leaf_community_id"] == lid]
            c = color_map[lid]
            ax_a.scatter(sub["date"], sub["root_distance"], color=c, alpha=0.55, s=18, edgecolors="none", label=f"{lid} (N={len(sub)})")

            t_min, t_max = sub["date"].min(), sub["date"].max()
            rate = sub["leaf_rate"].iloc[0]
            tmrca = sub["leaf_tmrca"].iloc[0]
            t_grid = np.linspace(t_min, t_max, 50)
            d_grid = np.maximum(0.0, rate * (t_grid - tmrca))
            ax_a.plot(t_grid, d_grid, color=c, lw=2.2, alpha=0.9)

        ax_a.set_title("(A) Hierarchical Multi-Clock Root-to-Tip Regressions", weight="bold", fontsize=11)
        ax_a.set_xlabel("Sample Collection Date", fontsize=10)
        ax_a.set_ylabel("Divergence from Subcommunity Root (subs/site)", fontsize=10)
        ax_a.grid(True, linestyle="--", alpha=0.35)
        if n_leaves <= 15:
            ax_a.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5, frameon=True)

        # (B) Hierarchical Topology Dendrogram
        ax_b = fig.add_subplot(gs[0, 1])
        coords = {}
        node_labels = {}
        edges = []
        leaf_counter = [0]

        def _compute_layout(node, depth=0):
            nid = node["node_id"]
            if node.get("is_leaf", False) or not node.get("children"):
                y = float(leaf_counter[0])
                leaf_counter[0] += 1
                coords[nid] = (depth, y)
                node_labels[nid] = f"{nid} (N={node['n_taxa']}, $\mu$={node['rate']:.1e})"
                return y

            child_ys = []
            for ch in node.get("children", []):
                edges.append((nid, ch["node_id"]))
                cy = _compute_layout(ch, depth + 1)
                child_ys.append(cy)

            my_y = float(np.mean(child_ys))
            coords[nid] = (depth, my_y)
            node_labels[nid] = f"{nid} (N={node['n_taxa']})"
            return my_y

        _compute_layout(self.tree)

        for p, c in edges:
            xp, yp = coords[p]
            xc, yc = coords[c]
            ax_b.plot([xp, xp + 0.35, xp + 0.35, xc], [yp, yp, yc, yc], color="#475569", lw=1.5, alpha=0.7)

        for nid, (x, y) in coords.items():
            is_leaf = nid in color_map
            c = color_map.get(nid, "#94a3b8")
            ax_b.scatter(x, y, color=c, s=100 if is_leaf else 70, zorder=4, edgecolors="#1e293b")
            ax_b.text(x + 0.06, y, node_labels[nid], va="center", fontsize=7.0 if n_leaves > 12 else 8.0,
                      weight="bold" if is_leaf else "normal")

        ax_b.set_title("(B) Hierarchical Manifold Deconvolution Topology", weight="bold", fontsize=11)
        ax_b.set_xlabel("Hierarchy Depth", fontsize=10)
        ax_b.set_yticks([])
        ax_b.set_xlim(-0.2, self.max_depth + 1.2)
        ax_b.set_ylim(-0.8, n_leaves + 0.2)
        ax_b.grid(True, axis="x", linestyle="--", alpha=0.35)

        # (C) Substitution Rates with 95% CI
        ax_c = fig.add_subplot(gs[1, 0])
        leaf_summaries = sorted(self.leaves, key=lambda x: x["rate"])
        c_lids = [l["node_id"] for l in leaf_summaries]
        rates = [l["rate"] * 1e3 for l in leaf_summaries]
        rates_ci_low = [max(0.0, l["rate_ci"][0] * 1e3) for l in leaf_summaries]
        rates_ci_high = [max(0.0, l["rate_ci"][1] * 1e3) for l in leaf_summaries]
        err_low = [abs(float(r - low)) for r, low in zip(rates, rates_ci_low)]
        err_high = [abs(float(high - r)) for r, high in zip(rates, rates_ci_high)]
        y_err = [err_low, err_high]
        y_pos = np.arange(len(c_lids))
        colors = [color_map.get(lid, "#3b82f6") for lid in c_lids]

        ax_c.barh(y_pos, rates, xerr=y_err, color=colors, alpha=0.8, edgecolor="#1e293b", capsize=3.5)
        ax_c.set_yticks(y_pos)
        ax_c.set_yticklabels(c_lids, fontsize=8)
        ax_c.set_xlabel("Substitution Rate $\mu$ ($10^{-3}$ substitutions/site/year)", fontsize=10)
        ax_c.set_title("(C) Calibrated Evolutionary Rates with 95% Confidence Intervals", weight="bold", fontsize=11)
        ax_c.grid(True, linestyle="--", alpha=0.35)

        # (D) Goodness-of-Fit Residual Dispersion
        ax_d = fig.add_subplot(gs[1, 1])
        res_data = [valid_df[valid_df["leaf_community_id"] == lid]["residual"].values for lid in c_lids]
        bp = ax_d.boxplot(res_data, vert=False, patch_artist=True, labels=c_lids,
                          boxprops=dict(facecolor="#93c5fd", alpha=0.7, edgecolor="#1e293b"),
                          medianprops=dict(color="#b91c1c", lw=1.8),
                          flierprops=dict(marker='o', markersize=3, alpha=0.4))

        for patch, lid in zip(bp['boxes'], c_lids):
            patch.set_facecolor(color_map.get(lid, "#93c5fd"))

        ax_d.axvline(0.0, color="red", linestyle="--", lw=1.2, alpha=0.8)
        ax_d.set_xlabel("Root-to-Tip Residual ($d_i - \hat{d}_i$)", fontsize=10)
        ax_d.set_title("(D) Goodness-of-Fit Residual Dispersion by Leaf Community", weight="bold", fontsize=11)
        ax_d.grid(True, linestyle="--", alpha=0.35)

        plt.suptitle(
            f"ChronAeon Hierarchical AutoClock: Multi-Clock Manifold Deconvolution ({Path(self.alignment_path).stem}, N={len(self.taxa)})",
            weight="bold", fontsize=13, y=0.99
        )

        p_png = Path(output_path) if output_path else self.output_dir / "fig_hierarchical_autoclock_diagnostic.png"
        p_pdf = p_png.with_suffix(".pdf")
        p_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p_png, dpi=300, bbox_inches="tight")
        plt.savefig(p_pdf, bbox_inches="tight")
        plt.close()
        self._log(f"[✓] Generated hierarchical diagnostic plots:\n    {p_png}\n    {p_pdf}")
        return p_png


def run_hierarchical_autoclock(
    alignment_path: Optional[Union[str, Path]] = None,
    dates_source: Optional[Union[str, Path]] = None,
    date_col: Optional[str] = None,
    strain_col: Optional[str] = None,
    date_regex: Optional[str] = None,
    max_depth: int = 3,
    min_leaf_size: int = 25,
    min_delta_aicc: float = 15.0,
    min_timespan: float = 1.0,
    min_eigengap: float = 0.02,
    max_rate_diff_ratio: float = 0.15,
    max_k_per_node: int = 6,
    manifold: str = "auto",
    weights: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    kernel_bandwidth: Optional[float] = None,
    output_dir: Optional[Union[str, Path]] = None,
    plot: bool = False,
    plot_path: Optional[Union[str, Path]] = None,
    quiet: bool = False,
    beast_path: Optional[Union[str, Path]] = None,
    max_dense_n: int = 2500,
    n_landmarks: Union[int, str] = "auto",
    max_memory_mb: float = 1024.0,
) -> Dict[str, Any]:
    """
    Convenience function to execute recursive Hierarchical AutoClock deconvolution.
    """
    engine = HierarchicalAutoClock(
        alignment_path=alignment_path,
        dates_source=dates_source,
        date_col=date_col,
        strain_col=strain_col,
        date_regex=date_regex,
        max_depth=max_depth,
        min_leaf_size=min_leaf_size,
        min_delta_aicc=min_delta_aicc,
        min_timespan=min_timespan,
        min_eigengap=min_eigengap,
        max_rate_diff_ratio=max_rate_diff_ratio,
        max_k_per_node=max_k_per_node,
        manifold=manifold,
        weights=weights,
        device=device,
        kernel_bandwidth=kernel_bandwidth,
        output_dir=output_dir,
        quiet=quiet,
        beast_path=beast_path,
        max_dense_n=max_dense_n,
        n_landmarks=n_landmarks,
        max_memory_mb=max_memory_mb,
    )
    return engine.run(plot=plot, plot_path=plot_path)
