"""
simulations/11_operational_boundaries_and_frame_collapse/generator_envelope.py
==============================================================================
Definitive Generative Experimental Matrix (1,050 Alignments) for RhizAeon's
Operational Boundaries and Frame Collapse Benchmark.

Simulates synthetic alignments (L = 3,000 nt) under continuous-time Markov
substitutions (HKY85, kappa=2.5, stationary base freqs pi=[0.3, 0.2, 0.2, 0.3])
with continuous Gamma site-to-site rate variation (alpha = 0.5, or alpha = 0.25).

Experiments:
  - Experiment A (Mutational Information Floor):
      5 tract lengths (50, 100, 250, 500, 1000 bp) x 6 divergence levels
      (0.002, 0.005, 0.01, 0.03, 0.10, 0.25). 30 cells x 15 reps = 450 alignments.
  - Experiment B (Panmictic Frame Collapse):
      rho / theta in {0.0, 0.05, 0.2, 0.5, 1.0, 2.5, 5.0, 10.0} across N in {20, 50}.
      16 cells x 15 reps = 240 alignments. (Multi-fragment patchworks from multiple ancestors).
  - Experiment C (Distant Parents & Mutational Saturation):
      Delta d(P1, P2) in {0.05, 0.15, 0.30, 0.45, 0.60} across 2 tract lengths (250, 1000 bp).
      10 cells x 15 reps = 150 alignments.
  - Experiment D (Clade Torque & Negative Clonal Controls):
      Mosaic fraction f_recomb in {1/N, 0.10, 0.25, 0.50} across Delta d in {0.02, 0.05, 0.15}.
      12 cells x 15 reps = 180 alignments.
      Plus 30 negative clonal controls with spatial Gamma rate variation (alpha = 0.25).
      Total Exp D = 210 alignments.

Total: exactly 1,050 alignments.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import scipy.linalg as la

NT_CHARS = ["A", "C", "G", "T"]
CHAR_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}
INT_TO_CHAR = {0: "A", 1: "C", 2: "G", 3: "T"}

PI_DEFAULT = np.array([0.30, 0.20, 0.20, 0.30], dtype=np.float64)
KAPPA_DEFAULT = 2.5


class VectorizedMarkovEvolver:
    """
    Continuous-time Markov chain substitution evolver with site-specific
    Gamma rate heterogeneity.
    """

    def __init__(
        self,
        kappa: float = KAPPA_DEFAULT,
        pi: np.ndarray = PI_DEFAULT,
        alpha: float = 0.5,
        L: int = 3000,
        seed: Optional[int] = None
    ):
        self.L = L
        self.pi = pi / np.sum(pi)
        self.kappa = kappa
        self.alpha = alpha

        if seed is not None:
            np.random.seed(seed)

        # Build HKY85 infinitesimal rate matrix Q
        Q = np.zeros((4, 4), dtype=np.float64)
        ti = [(0, 2), (2, 0), (1, 3), (3, 1)]
        tv = [(0, 1), (1, 0), (0, 3), (3, 0), (2, 1), (1, 2), (2, 3), (3, 2)]
        for i, j in ti:
            Q[i, j] = self.kappa * self.pi[j]
        for i, j in tv:
            Q[i, j] = 1.0 * self.pi[j]
        for i in range(4):
            Q[i, i] = -np.sum(Q[i])

        scale = -np.sum(self.pi * np.diag(Q))
        self.Q = Q / scale

        # Spectral decomposition Q = V diag(w) V^-1
        self.w, self.V = np.linalg.eig(self.Q)
        self.V_inv = np.linalg.inv(self.V)

        # Sample site rates from Gamma(alpha, 1/alpha) with mean 1.0
        if self.alpha > 0:
            self.site_rates = np.random.gamma(self.alpha, 1.0 / self.alpha, size=self.L)
        else:
            self.site_rates = np.ones(self.L, dtype=np.float64)

    def sample_root(self) -> np.ndarray:
        """Samples ancestral sequence from stationary distribution pi."""
        return np.random.choice(4, size=self.L, p=self.pi).astype(np.int8)

    def evolve_branch(self, parent_seq: np.ndarray, branch_len: float) -> np.ndarray:
        """
        Evolves parent_seq along branch of length branch_len using exact
        continuous-time Markov transition probabilities.
        """
        if branch_len <= 1e-9:
            return parent_seq.copy()

        diag_exp = np.exp(np.outer(self.site_rates * branch_len, self.w))
        P = np.einsum("ik, sk, kj -> sij", self.V, diag_exp, self.V_inv).real
        P = np.maximum(0.0, P)

        # Sample from categorical transition probabilities
        probs = P[np.arange(self.L), parent_seq]
        cum = np.cumsum(probs, axis=1)
        r = np.random.rand(self.L, 1)
        return (r > cum).sum(axis=1).astype(np.int8)


def simulate_kingman_tree(
    n: int = 20,
    total_depth: float = 0.05,
    seed: Optional[int] = None
) -> Tuple[int, Dict[int, List[int]], Dict[int, float]]:
    """Simulates a Kingman coalescent tree for n taxa scaled to total_depth."""
    if seed is not None:
        np.random.seed(seed)

    active = list(range(n))
    node_time = {i: 0.0 for i in range(n)}
    children = {}
    branch_len = {}
    next_node = n
    t = 0.0

    while len(active) > 1:
        k = len(active)
        dt = np.random.exponential(2.0 / (k * (k - 1)))
        t += dt
        pair = np.random.choice(len(active), size=2, replace=False)
        u, v = active[pair[0]], active[pair[1]]
        node_time[next_node] = t
        children[next_node] = [u, v]
        branch_len[u] = t - node_time[u]
        branch_len[v] = t - node_time[v]
        active.remove(u)
        active.remove(v)
        active.append(next_node)
        next_node += 1

    root = active[0]
    scale = total_depth / max(1e-6, node_time[root])
    for k in branch_len:
        branch_len[k] *= scale
    return root, children, branch_len


# -----------------------------------------------------------------------------
# Experiment A: Mutational Information Floor
# -----------------------------------------------------------------------------

def generate_experiment_a(
    tract_len: int,
    divergence: float,
    rep_id: int,
    n_taxa: int = 20,
    L: int = 3000,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Experiment A: Mutational Information Floor (I_mut = L_tract * Delta_d).
    Recombinant taxon 0 inherits flanks [0, s1) and [s2, L) from Clade 1,
    and cassette [s1, s2) from Clade 2.
    """
    if seed is None:
        seed = 100000 + int(tract_len * 100) + int(divergence * 10000) + rep_id

    evolver = VectorizedMarkovEvolver(alpha=0.5, L=L, seed=seed)
    np.random.seed(seed)

    # 2 clades: Clade 1 (0..n1-1), Clade 2 (n1..N-1)
    n1 = n_taxa // 2
    n2 = n_taxa - n1

    d_between = divergence
    d_within = max(0.001, 0.15 * divergence)

    root1, ch1, bl1 = simulate_kingman_tree(n1, total_depth=d_within, seed=seed + 1)
    root2, ch2, bl2 = simulate_kingman_tree(n2, total_depth=d_within, seed=seed + 2)

    # Evolve Clade 1 and Clade 2 from common root with separation d_between
    anc_root = evolver.sample_root()
    root1_seq = evolver.evolve_branch(anc_root, d_between / 2.0)
    root2_seq = evolver.evolve_branch(anc_root, d_between / 2.0)

    def evolve_clade(root_n, ch_map, bl_map, start_seq, offset):
        node_seqs = {root_n: start_seq}
        queue = [root_n]
        while queue:
            curr = queue.pop(0)
            if curr in ch_map:
                for ch in ch_map[curr]:
                    node_seqs[ch] = evolver.evolve_branch(node_seqs[curr], bl_map[ch])
                    queue.append(ch)
        return [node_seqs[i] for i in range(len(ch_map) // 2 + 1 if len(ch_map) > 0 else 1)]

    # We evolve Clade 1
    node_seqs1 = {root1: root1_seq}
    q1 = [root1]
    while q1:
        c = q1.pop(0)
        if c in ch1:
            for ch in ch1[c]:
                node_seqs1[ch] = evolver.evolve_branch(node_seqs1[c], bl1[ch])
                q1.append(ch)
    clade1_seqs = [node_seqs1[i] for i in range(n1)]

    # We evolve Clade 2
    node_seqs2 = {root2: root2_seq}
    q2 = [root2]
    while q2:
        c = q2.pop(0)
        if c in ch2:
            for ch in ch2[c]:
                node_seqs2[ch] = evolver.evolve_branch(node_seqs2[c], bl2[ch])
                q2.append(ch)
    clade2_seqs = [node_seqs2[i] for i in range(n2)]

    # Assemble population matrix
    mat = np.zeros((n_taxa, L), dtype=np.int8)
    for i in range(n1):
        mat[i] = clade1_seqs[i]
    for i in range(n2):
        mat[n1 + i] = clade2_seqs[i]

    # Recombinant taxon 0 inherits cassette from Clade 2 (taxon n1)
    s1 = 1000
    s2 = s1 + tract_len
    donor_idx = n1  # First taxon of Clade 2
    mat[0, s1:s2] = mat[donor_idx, s1:s2]

    taxa = [f"Taxon_{i}" for i in range(n_taxa)]
    taxa[0] = "Rec_0"

    mut_payload = float(tract_len * divergence)
    meta = {
        "experiment": "A",
        "cell_id": f"L{tract_len}_d{divergence}",
        "rep_id": rep_id,
        "n_taxa": n_taxa,
        "genome_length": L,
        "tract_length": tract_len,
        "divergence": divergence,
        "rho_over_theta": 0.0,
        "mosaic_fraction": 1.0 / n_taxa,
        "mutational_payload": mut_payload,
        "is_recombinant": True,
        "true_bps": [s1, s2],
        "recombinant_taxon": "Rec_0",
        "parent_left": "Taxon_1",
        "parent_right": f"Taxon_{donor_idx}",
    }
    return mat, taxa, meta


# -----------------------------------------------------------------------------
# Experiment B: Panmictic Frame Collapse
# -----------------------------------------------------------------------------

def generate_experiment_b(
    rho_over_theta: float,
    n_taxa: int,
    rep_id: int,
    L: int = 3000,
    theta: float = 0.02,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Experiment B: Panmictic Frame Collapse under coalescent with gene conversion.
    At rho/theta = 0.0, evolution is strictly clonal.
    At high rho/theta (up to 10.0), genomes are multi-fragment patchworks from
    multiple ancestors, leading to breakdown of a single Euclidean coordinate frame.
    """
    if seed is None:
        seed = 200000 + int(rho_over_theta * 1000) + (n_taxa * 100) + rep_id

    evolver = VectorizedMarkovEvolver(alpha=0.5, L=L, seed=seed)
    np.random.seed(seed)

    # Base clonal tree
    total_depth = max(0.02, theta * 2.5)
    root, ch, bl = simulate_kingman_tree(n_taxa, total_depth=total_depth, seed=seed)

    # Evolve initial clonal sequences for all nodes
    root_seq = evolver.sample_root()
    node_seqs = {root: root_seq}
    queue = [root]
    while queue:
        c = queue.pop(0)
        if c in ch:
            for child in ch[c]:
                node_seqs[child] = evolver.evolve_branch(node_seqs[c], bl[child])
                queue.append(child)

    leaf_mat = np.array([node_seqs[i] for i in range(n_taxa)], dtype=np.int8)
    true_bps = set()

    # If rho_over_theta > 0, introduce homologous gene conversion imports
    num_events = 0
    if rho_over_theta > 0.0:
        # Expected number of imports scales with rho/theta * tree depth * log(N)
        mean_imports = rho_over_theta * (3.0 + 2.0 * np.log(n_taxa))
        n_imports = np.random.poisson(mean_imports)
        num_events = n_imports

        for _ in range(n_imports):
            rec_idx = np.random.randint(0, n_taxa)
            donor_idx = np.random.randint(0, n_taxa)
            while donor_idx == rec_idx:
                donor_idx = np.random.randint(0, n_taxa)

            # Tract length ~ Exponential(500) bounded [50, 1200]
            t_len = int(np.clip(np.random.exponential(500.0), 50, 1200))
            start_pos = np.random.randint(0, max(1, L - t_len))
            end_pos = min(L, start_pos + t_len)

            leaf_mat[rec_idx, start_pos:end_pos] = leaf_mat[donor_idx, start_pos:end_pos]
            true_bps.add(start_pos)
            if end_pos < L:
                true_bps.add(end_pos)

    taxa = [f"Taxon_{i}" for i in range(n_taxa)]
    sorted_bps = sorted(list(true_bps))

    meta = {
        "experiment": "B",
        "cell_id": f"rho_{rho_over_theta}_N{n_taxa}",
        "rep_id": rep_id,
        "n_taxa": n_taxa,
        "genome_length": L,
        "tract_length": 500 if rho_over_theta > 0 else 0,
        "divergence": total_depth,
        "rho_over_theta": rho_over_theta,
        "mosaic_fraction": min(1.0, num_events / max(1, n_taxa)),
        "mutational_payload": float(500 * total_depth if rho_over_theta > 0 else 0.0),
        "is_recombinant": (rho_over_theta > 0.0 and len(sorted_bps) > 0),
        "true_bps": sorted_bps,
        "num_imports": num_events,
    }
    return leaf_mat, taxa, meta


# -----------------------------------------------------------------------------
# Experiment C: Distant Parents & Mutational Saturation
# -----------------------------------------------------------------------------

def generate_experiment_c(
    divergence: float,
    tract_len: int,
    rep_id: int,
    n_taxa: int = 20,
    L: int = 3000,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Experiment C: Distant Parents & Mutational Saturation.
    Delta d(P1, P2) in {0.05, 0.15, 0.30, 0.45, 0.60} across tract_len in {250, 1000}.
    Evaluates whether distance linearization and single-base polishing break
    under deep saturation and multiple substitutions.
    """
    if seed is None:
        seed = 300000 + int(divergence * 10000) + int(tract_len * 10) + rep_id

    evolver = VectorizedMarkovEvolver(alpha=0.5, L=L, seed=seed)
    np.random.seed(seed)

    n1 = n_taxa // 2
    n2 = n_taxa - n1

    d_within = 0.015  # Fixed moderate within-clade divergence
    root1, ch1, bl1 = simulate_kingman_tree(n1, total_depth=d_within, seed=seed + 1)
    root2, ch2, bl2 = simulate_kingman_tree(n2, total_depth=d_within, seed=seed + 2)

    # Distant parents separated by divergence Delta_d
    anc_root = evolver.sample_root()
    root1_seq = evolver.evolve_branch(anc_root, divergence / 2.0)
    root2_seq = evolver.evolve_branch(anc_root, divergence / 2.0)

    # Clade 1
    node_seqs1 = {root1: root1_seq}
    q1 = [root1]
    while q1:
        c = q1.pop(0)
        if c in ch1:
            for ch in ch1[c]:
                node_seqs1[ch] = evolver.evolve_branch(node_seqs1[c], bl1[ch])
                q1.append(ch)

    # Clade 2
    node_seqs2 = {root2: root2_seq}
    q2 = [root2]
    while q2:
        c = q2.pop(0)
        if c in ch2:
            for ch in ch2[c]:
                node_seqs2[ch] = evolver.evolve_branch(node_seqs2[c], bl2[ch])
                q2.append(ch)

    mat = np.zeros((n_taxa, L), dtype=np.int8)
    for i in range(n1):
        mat[i] = node_seqs1[i]
    for i in range(n2):
        mat[n1 + i] = node_seqs2[i]

    s1 = 1000
    s2 = s1 + tract_len
    donor_idx = n1
    mat[0, s1:s2] = mat[donor_idx, s1:s2]

    taxa = [f"Taxon_{i}" for i in range(n_taxa)]
    taxa[0] = "Rec_0"

    meta = {
        "experiment": "C",
        "cell_id": f"dist_d{divergence}_L{tract_len}",
        "rep_id": rep_id,
        "n_taxa": n_taxa,
        "genome_length": L,
        "tract_length": tract_len,
        "divergence": divergence,
        "rho_over_theta": 0.0,
        "mosaic_fraction": 1.0 / n_taxa,
        "mutational_payload": float(tract_len * divergence),
        "is_recombinant": True,
        "true_bps": [s1, s2],
        "recombinant_taxon": "Rec_0",
        "parent_left": "Taxon_1",
        "parent_right": f"Taxon_{donor_idx}",
    }
    return mat, taxa, meta


# -----------------------------------------------------------------------------
# Experiment D: Clade Torque & Negative Clonal Controls
# -----------------------------------------------------------------------------

def generate_experiment_d(
    mosaic_fraction: float,
    divergence: float,
    rep_id: int,
    n_taxa: int = 20,
    L: int = 3000,
    tract_len: int = 500,
    is_clonal_control: bool = False,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Experiment D: Clade Torque & Negative Clonal Controls.
    - Part 1 (Clade Torque): f_recomb in {1/N, 0.10, 0.25, 0.50} across Delta d in {0.02, 0.05, 0.15}.
      A sub-clade of size k_rec = round(f_recomb * N) inherits the cassette.
      Under high f_recomb, Procrustes rotation Q suffers torque (Z drops),
      while the Graph Laplacian Fiedler vector v_2(s) phase shift surges to 1.0!
    - Part 2 (Negative Clonal Controls): 30 clonal alignments under spatial Gamma rate
      variation (alpha = 0.25) with zero recombination.
    """
    if seed is None:
        if is_clonal_control:
            seed = 490000 + rep_id
        else:
            seed = 400000 + int(mosaic_fraction * 1000) + int(divergence * 10000) + rep_id

    # For clonal controls, use stronger rate heterogeneity (alpha = 0.25)
    alpha = 0.25 if is_clonal_control else 0.5
    evolver = VectorizedMarkovEvolver(alpha=alpha, L=L, seed=seed)
    np.random.seed(seed)

    if is_clonal_control:
        # Strictly clonal tree, no recombination
        root, ch, bl = simulate_kingman_tree(n_taxa, total_depth=0.08, seed=seed)
        root_seq = evolver.sample_root()
        node_seqs = {root: root_seq}
        queue = [root]
        while queue:
            c = queue.pop(0)
            if c in ch:
                for child in ch[c]:
                    node_seqs[child] = evolver.evolve_branch(node_seqs[c], bl[child])
                    queue.append(child)

        mat = np.array([node_seqs[i] for i in range(n_taxa)], dtype=np.int8)
        taxa = [f"Taxon_{i}" for i in range(n_taxa)]
        meta = {
            "experiment": "D",
            "cell_id": "clonal_control_alpha0.25",
            "rep_id": rep_id,
            "n_taxa": n_taxa,
            "genome_length": L,
            "tract_length": 0,
            "divergence": 0.08,
            "rho_over_theta": 0.0,
            "mosaic_fraction": 0.0,
            "mutational_payload": 0.0,
            "is_recombinant": False,
            "true_bps": [],
            "k_recomb": 0,
        }
        return mat, taxa, meta

    # Clade Torque: 2 divergent clades
    n1 = n_taxa // 2
    n2 = n_taxa - n1

    d_within = max(0.002, 0.15 * divergence)
    root1, ch1, bl1 = simulate_kingman_tree(n1, total_depth=d_within, seed=seed + 1)
    root2, ch2, bl2 = simulate_kingman_tree(n2, total_depth=d_within, seed=seed + 2)

    anc_root = evolver.sample_root()
    root1_seq = evolver.evolve_branch(anc_root, divergence / 2.0)
    root2_seq = evolver.evolve_branch(anc_root, divergence / 2.0)

    node_seqs1 = {root1: root1_seq}
    q1 = [root1]
    while q1:
        c = q1.pop(0)
        if c in ch1:
            for ch in ch1[c]:
                node_seqs1[ch] = evolver.evolve_branch(node_seqs1[c], bl1[ch])
                q1.append(ch)

    node_seqs2 = {root2: root2_seq}
    q2 = [root2]
    while q2:
        c = q2.pop(0)
        if c in ch2:
            for ch in ch2[c]:
                node_seqs2[ch] = evolver.evolve_branch(node_seqs2[c], bl2[ch])
                q2.append(ch)

    mat = np.zeros((n_taxa, L), dtype=np.int8)
    for i in range(n1):
        mat[i] = node_seqs1[i]
    for i in range(n2):
        mat[n1 + i] = node_seqs2[i]

    s1 = 1000
    s2 = s1 + tract_len
    k_rec = max(1, int(round(mosaic_fraction * n_taxa)))

    # Taxa 0..k_rec-1 inherit cassette from Clade 2 (taxa n1..n1+k_rec-1)
    for idx in range(k_rec):
        donor = n1 + (idx % n2)
        mat[idx, s1:s2] = mat[donor, s1:s2]

    taxa = [f"Taxon_{i}" for i in range(n_taxa)]
    for idx in range(k_rec):
        taxa[idx] = f"Rec_{idx}"

    meta = {
        "experiment": "D",
        "cell_id": f"torque_f{mosaic_fraction:.2f}_d{divergence}",
        "rep_id": rep_id,
        "n_taxa": n_taxa,
        "genome_length": L,
        "tract_length": tract_len,
        "divergence": divergence,
        "rho_over_theta": 0.0,
        "mosaic_fraction": mosaic_fraction,
        "mutational_payload": float(tract_len * divergence),
        "is_recombinant": True,
        "true_bps": [s1, s2],
        "k_recomb": k_rec,
    }
    return mat, taxa, meta


# -----------------------------------------------------------------------------
# Master Generator CLI and Batch Orchestration
# -----------------------------------------------------------------------------

def save_fasta(mat: np.ndarray, taxa: List[str], filepath: Path):
    """Writes integer nucleotide matrix to FASTA file."""
    with open(filepath, "w") as f:
        for i, name in enumerate(taxa):
            seq = "".join(INT_TO_CHAR[b] for b in mat[i])
            f.write(f">{name}\n{seq}\n")


def generate_all_envelope_datasets(output_dir: Path) -> List[Dict[str, Any]]:
    """
    Generates all 1,050 benchmark alignments:
      Exp A: 450 alignments
      Exp B: 240 alignments
      Exp C: 150 alignments
      Exp D: 210 alignments (180 torque + 30 clonal controls)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    t0 = time.perf_counter()

    print("================================================================================")
    print("RHIZAEON OPERATIONAL BOUNDARIES & FRAME COLLAPSE BENCHMARK GENERATOR")
    print(f"Target Directory: {output_dir}")
    print("Generating exact matrix of 1,050 alignments (L = 3,000 nt)...")
    print("================================================================================")

    # -------------------------------------------------------------------------
    # Experiment A: Mutational Floor (30 cells x 15 reps = 450)
    # -------------------------------------------------------------------------
    tract_lengths_a = [50, 100, 250, 500, 1000]
    divergences_a = [0.002, 0.005, 0.01, 0.03, 0.10, 0.25]
    count_a = 0
    for t_len in tract_lengths_a:
        for div in divergences_a:
            for rep in range(15):
                mat, taxa, meta = generate_experiment_a(t_len, div, rep)
                aln_id = f"expA_L{t_len}_d{div:.3f}_rep{rep:02d}"
                fasta_path = output_dir / f"{aln_id}.fasta"
                save_fasta(mat, taxa, fasta_path)
                meta["alignment_id"] = aln_id
                meta["file_path"] = str(fasta_path)
                manifest.append(meta)
                count_a += 1
    print(f"[✓] Experiment A Complete: {count_a} alignments generated.")

    # -------------------------------------------------------------------------
    # Experiment B: Panmictic Frame Collapse (16 cells x 15 reps = 240)
    # -------------------------------------------------------------------------
    rhos_b = [0.0, 0.05, 0.2, 0.5, 1.0, 2.5, 5.0, 10.0]
    ns_b = [20, 50]
    count_b = 0
    for n in ns_b:
        for rho in rhos_b:
            for rep in range(15):
                mat, taxa, meta = generate_experiment_b(rho, n, rep)
                aln_id = f"expB_N{n}_rho{rho}_rep{rep:02d}"
                fasta_path = output_dir / f"{aln_id}.fasta"
                save_fasta(mat, taxa, fasta_path)
                meta["alignment_id"] = aln_id
                meta["file_path"] = str(fasta_path)
                manifest.append(meta)
                count_b += 1
    print(f"[✓] Experiment B Complete: {count_b} alignments generated.")

    # -------------------------------------------------------------------------
    # Experiment C: Distant Parents & Saturation (10 cells x 15 reps = 150)
    # -------------------------------------------------------------------------
    divergences_c = [0.05, 0.15, 0.30, 0.45, 0.60]
    tract_lengths_c = [250, 1000]
    count_c = 0
    for div in divergences_c:
        for t_len in tract_lengths_c:
            for rep in range(15):
                mat, taxa, meta = generate_experiment_c(div, t_len, rep)
                aln_id = f"expC_d{div:.2f}_L{t_len}_rep{rep:02d}"
                fasta_path = output_dir / f"{aln_id}.fasta"
                save_fasta(mat, taxa, fasta_path)
                meta["alignment_id"] = aln_id
                meta["file_path"] = str(fasta_path)
                manifest.append(meta)
                count_c += 1
    print(f"[✓] Experiment C Complete: {count_c} alignments generated.")

    # -------------------------------------------------------------------------
    # Experiment D: Clade Torque & Negative Clonal Controls (180 + 30 = 210)
    # -------------------------------------------------------------------------
    mosaic_fractions_d = [0.05, 0.10, 0.25, 0.50]  # 0.05 = 1/20
    divergences_d = [0.02, 0.05, 0.15]
    count_d = 0
    for frac in mosaic_fractions_d:
        for div in divergences_d:
            for rep in range(15):
                mat, taxa, meta = generate_experiment_d(frac, div, rep, is_clonal_control=False)
                aln_id = f"expD_torque_f{frac:.2f}_d{div:.2f}_rep{rep:02d}"
                fasta_path = output_dir / f"{aln_id}.fasta"
                save_fasta(mat, taxa, fasta_path)
                meta["alignment_id"] = aln_id
                meta["file_path"] = str(fasta_path)
                manifest.append(meta)
                count_d += 1

    # 30 negative clonal controls
    for rep in range(30):
        mat, taxa, meta = generate_experiment_d(0.0, 0.08, rep, is_clonal_control=True)
        aln_id = f"expD_clonal_control_rep{rep:02d}"
        fasta_path = output_dir / f"{aln_id}.fasta"
        save_fasta(mat, taxa, fasta_path)
        meta["alignment_id"] = aln_id
        meta["file_path"] = str(fasta_path)
        manifest.append(meta)
        count_d += 1
    print(f"[✓] Experiment D Complete: {count_d} alignments generated.")

    total_time = time.perf_counter() - t0
    total_count = len(manifest)
    print(f"\n================================================================================")
    print(f"GENERATION COMPLETE: Exactly {total_count} alignments generated in {total_time:.2f} s")
    print(f"================================================================================")

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[✓] Manifest saved to: {manifest_path}")

    return manifest


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    out_data = script_dir / "data"
    generate_all_envelope_datasets(out_data)
