"""
HyphAeon Alignment-Free Sketch & Topological Binner
===================================================
Provides ultra-fast canonical MinHash sketching and alignment-free binning
for massive unaligned genomic surveillance feeds.

Key Capabilities:
  1. Canonical k-mer MinHash (strand-invariant, O(L) per genome).
  2. Instant Jaccard & Mash distance estimation (microseconds per pair).
  3. Upstream Centrifuge Binner: separates subtypes (H1, H3, H5, H7),
     auto-routes sequences to optimal structural references, and
     quarantines non-target contaminants before alignment.

Author: Sergei L. Kosakovsky Pond & DeepMind Antigravity Pair Programmer
"""

import os
import sys
import zlib
import time
from typing import Dict, List, Tuple, Optional, Set, Any
import numpy as np
import pandas as pd


def get_canonical_kmer(kmer: str) -> str:
    """Returns the lexicographically smaller of kmer and its reverse complement."""
    tr = str.maketrans("ACGTURYKMSWBDHVNacgturykmswbdhvn", "TGCAAYRMKSWVHDBNtgcaayrmkswvhdbn")
    rc = kmer.translate(tr)[::-1]
    return kmer if kmer <= rc else rc


def hash_kmer_64(kmer: str) -> int:
    """Computes a 64-bit integer hash for a canonical k-mer using dual-seed CRC32."""
    b = kmer.encode('ascii', errors='ignore')
    h1 = zlib.crc32(b, 0x12345678)
    h2 = zlib.crc32(b, 0x87654321)
    return (h1 << 32) | h2


class CanonicalMinHashSketcher:
    """Fast canonical MinHash sketch generator for unaligned nucleotide/protein sequences."""

    def __init__(self, k: int = 15, sketch_size: int = 1024, is_protein: bool = False):
        self.k = k
        self.sketch_size = sketch_size
        self.is_protein = is_protein

    def sketch(self, seq: str) -> np.ndarray:
        """
        Extracts all canonical k-mers and returns sorted array of bottom-s unique hashes.
        If sequence length < k, returns empty array.
        """
        seq_clean = seq.upper().replace("-", "").replace(".", "").replace(" ", "")
        n_kmers = len(seq_clean) - self.k + 1
        if n_kmers <= 0:
            return np.array([], dtype=np.uint64)

        seen_hashes: Set[int] = set()
        k = self.k
        is_prot = self.is_protein

        for i in range(n_kmers):
            kmer = seq_clean[i:i + k]
            if not is_prot:
                canon = get_canonical_kmer(kmer)
            else:
                canon = kmer
            seen_hashes.add(hash_kmer_64(canon))

        sorted_hashes = np.array(sorted(seen_hashes), dtype=np.uint64)
        if len(sorted_hashes) > self.sketch_size:
            return sorted_hashes[:self.sketch_size]
        return sorted_hashes

    @staticmethod
    def jaccard(sketch_a: np.ndarray, sketch_b: np.ndarray) -> float:
        """Computes unbiased bottom-s MinHash Jaccard similarity between two sketches."""
        if len(sketch_a) == 0 or len(sketch_b) == 0:
            return 0.0

        s_a = set(sketch_a)
        s_b = set(sketch_b)
        joint = sorted(s_a | s_b)
        s_max = max(len(sketch_a), len(sketch_b))
        joint_s = set(joint[:s_max])

        shared = len(s_a & s_b & joint_s)
        return float(shared / max(1, len(joint_s)))

    def mash_distance(self, sketch_a: np.ndarray, sketch_b: np.ndarray) -> float:
        """Estimates substitution distance D from Jaccard similarity via Mash formula."""
        j = self.jaccard(sketch_a, sketch_b)
        if j <= 0.0:
            return 1.0
        if j >= 1.0:
            return 0.0
        # Mash formula: D = -1/k * ln(2J / (1 + J))
        ratio = (2.0 * j) / (1.0 + j)
        return max(0.0, float(- (1.0 / self.k) * np.log(max(1e-9, ratio))))


class AlignmentFreeBinner:
    """
    Topological Centrifuge: Partitions unaligned sequences into alignable cliques,
    auto-routes each clique to an optimal reference profile, and discards sludge.
    """

    def __init__(
        self,
        k: int = 15,
        sketch_size: int = 1024,
        alignable_threshold: float = 0.12,  # Jaccard >= 0.12 corresponds to ~85-90% nucleotide identity
        quarantine_threshold: float = 0.02, # Jaccard < 0.02 indicates non-target / distant contaminant
        min_bin_size: int = 5,
        quiet: bool = False,
    ):
        self.k = k
        self.sketch_size = sketch_size
        self.alignable_threshold = alignable_threshold
        self.quarantine_threshold = quarantine_threshold
        self.min_bin_size = min_bin_size
        self.quiet = quiet
        self.sketcher = CanonicalMinHashSketcher(k=k, sketch_size=sketch_size)

    def _log(self, msg: str):
        if not self.quiet:
            print(msg, flush=True)

    def bin_dataset(
        self,
        seq_dict: Dict[str, str],
        references: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Bins unaligned sequences into alignable sub-cohorts.

        Args:
            seq_dict: Map of {taxon_id: unaligned_sequence}
            references: Optional map of {ref_name: ref_sequence} (canonical subtype references)

        Returns:
            Dict containing:
              - 'bins': {bin_name: [taxa]}
              - 'bin_references': {bin_name: ref_seq}
              - 'quarantined': {taxon: reason}
              - 'taxa_assignments': pd.DataFrame with per-taxon bin, best ref, and Jaccard similarity
        """
        n_total = len(seq_dict)
        self._log("=" * 80)
        self._log(f"HYPHAEON ALIGNMENT-FREE CENTRIFUGE: BINNING {n_total:,} UNALIGNED SEQUENCES")
        self._log("=" * 80)

        t0 = time.time()
        # 1. Compute MinHash sketches
        self._log(f"[*] Extracting canonical {self.k}-mer MinHash sketches (s={self.sketch_size})...")
        sketches: Dict[str, np.ndarray] = {}
        for t, s in seq_dict.items():
            sketches[t] = self.sketcher.sketch(s)

        t_sketch = time.time() - t0
        self._log(f"[✓] Sketched {n_total:,} sequences in {t_sketch:.2f}s ({n_total/max(0.01, t_sketch):.1f} seqs/s).")

        # 2. Sketch reference profiles
        ref_sketches: Dict[str, np.ndarray] = {}
        if references:
            for rname, rseq in references.items():
                ref_sketches[rname] = self.sketcher.sketch(rseq)
            self._log(f"[*] Sketched {len(references)} canonical subtype reference profiles.")

        # 3. Match against reference library (if provided)
        assigned_bins: Dict[str, List[str]] = {}
        quarantined: Dict[str, str] = {}
        unassigned: List[str] = []
        assignment_records = []

        if references:
            for t in seq_dict:
                q_sk = sketches[t]
                if len(q_sk) == 0:
                    quarantined[t] = "empty_or_too_short"
                    continue

                best_ref = None
                best_j = 0.0

                for rname, r_sk in ref_sketches.items():
                    j = self.sketcher.jaccard(q_sk, r_sk)
                    if j > best_j:
                        best_j = j
                        best_ref = rname

                mash_d = self.sketcher.mash_distance(q_sk, ref_sketches[best_ref]) if best_ref else 1.0

                if best_j >= self.alignable_threshold:
                    bin_name = best_ref
                    if bin_name not in assigned_bins:
                        assigned_bins[bin_name] = []
                    assigned_bins[bin_name].append(t)
                    assignment_records.append({
                        "taxon": t, "bin": bin_name, "best_reference": best_ref,
                        "jaccard": best_j, "mash_distance": mash_d, "status": "assigned"
                    })
                elif best_j < self.quarantine_threshold:
                    quarantined[t] = f"non_target_contaminant (J={best_j:.4f} < {self.quarantine_threshold})"
                    assignment_records.append({
                        "taxon": t, "bin": "quarantined", "best_reference": best_ref,
                        "jaccard": best_j, "mash_distance": mash_d, "status": "quarantined"
                    })
                else:
                    unassigned.append(t)
                    assignment_records.append({
                        "taxon": t, "bin": "unassigned_divergent", "best_reference": best_ref,
                        "jaccard": best_j, "mash_distance": mash_d, "status": "unassigned"
                    })
        else:
            unassigned = list(seq_dict.keys())

        # 4. De novo clustering for unassigned sequences (or when no reference library exists)
        if unassigned:
            self._log(f"[*] Clustering {len(unassigned):,} unassigned sequences de novo via landmark graphs...")
            # Pick landmarks among unassigned
            n_un = len(unassigned)
            n_lm = min(n_un, 64)
            lm_idx = np.random.choice(n_un, size=n_lm, replace=False)
            lm_taxa = [unassigned[i] for i in lm_idx]

            # Fast greedy single-pass clustering on Jaccard threshold
            de_novo_clusters: List[List[str]] = []
            cluster_medoids: List[str] = []

            for t in unassigned:
                q_sk = sketches[t]
                if len(q_sk) == 0:
                    quarantined[t] = "empty_or_too_short"
                    continue

                matched = False
                for c_idx, medoid in enumerate(cluster_medoids):
                    j = self.sketcher.jaccard(q_sk, sketches[medoid])
                    if j >= self.alignable_threshold:
                        de_novo_clusters[c_idx].append(t)
                        matched = True
                        break

                if not matched:
                    # New de novo cluster
                    cluster_medoids.append(t)
                    de_novo_clusters.append([t])

            # Assign valid de novo clusters
            for c_idx, (medoid, members) in enumerate(zip(cluster_medoids, de_novo_clusters)):
                if len(members) >= self.min_bin_size:
                    bin_name = f"de_novo_bin_{c_idx+1}"
                    assigned_bins[bin_name] = members
                    # Medoid sequence serves as its reference
                    if references is None:
                        references = {}
                    references[bin_name] = seq_dict[medoid]
                    for m in members:
                        for r in assignment_records:
                            if r["taxon"] == m:
                                r["bin"] = bin_name
                                r["status"] = "assigned_de_novo"
                else:
                    for m in members:
                        quarantined[m] = f"singleton_or_small_bin (size={len(members)})"
                        for r in assignment_records:
                            if r["taxon"] == m:
                                r["bin"] = "quarantined"
                                r["status"] = "quarantined"

        # 5. Populate bin references
        bin_references = {}
        for b_name in assigned_bins:
            if references and b_name in references:
                bin_references[b_name] = references[b_name]
            else:
                # Pick internal medoid as reference
                bin_taxa = assigned_bins[b_name]
                bin_references[b_name] = seq_dict[bin_taxa[0]]

        df_assign = pd.DataFrame(assignment_records)

        self._log("\n" + "=" * 80)
        self._log(f"CENTRIFUGE RESULTS: {len(assigned_bins)} ALIGNABLE BINS | {len(quarantined)} QUARANTINED")
        self._log("=" * 80)
        for b_name, members in assigned_bins.items():
            self._log(f"  ├── Bin '{b_name}': {len(members):,} sequences (Reference: {b_name})")
        self._log(f"  └── Quarantined Sieve: {len(quarantined):,} sequences")

        return {
            "bins": assigned_bins,
            "bin_references": bin_references,
            "quarantined": quarantined,
            "taxa_assignments": df_assign,
            "elapsed_seconds": time.time() - t0,
        }


# Aliases for convenience
AlignmentFreeCentrifuge = AlignmentFreeBinner
