"""
rhizaeon.segmentation
=====================
Recursive Binary Segmentation (RBS), Multi-Breakpoint Localization,
and Single-Codon Boundary Refinement.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
from scipy.signal import find_peaks

from rhizaeon.tensor import PrefixDistanceEngine
from rhizaeon.manifold import (
    compute_classical_mds,
    compute_laplacian_eigenmaps,
    align_procrustes,
    compute_ghost_node_zscores
)
from rhizaeon.pir import evaluate_triplets_for_taxon, refine_breakpoint_codon


class RhizAeonDetector:
    """
    Tree-Free Reticulate Evolution & Recombination Detection Engine.
    
    Combines continuous sequence manifold geometry, out-of-sample Ghost Node
    Procrustes triage, and bounded Latent Parental Incongruence Ratio (L-PIR).
    """

    def __init__(
        self,
        window_units: int = 25,
        min_tract_units: int = 35,
        ghost_z_threshold: float = 3.0,
        pir_threshold: float = 0.25,
        k_dims: int = 4,
        bound_factor: float = 1.25,
        min_parent_dist: float = 0.025,
        step: int = 5,
        concordance_tolerance: Optional[int] = 60,
        embedding_method: str = "mds"
    ):
        self.window_units = window_units
        self.min_tract_units = min_tract_units
        self.ghost_z_threshold = ghost_z_threshold
        self.pir_threshold = pir_threshold
        self.k_dims = k_dims
        self.bound_factor = bound_factor
        self.min_parent_dist = min_parent_dist
        self.step = step
        self.concordance_tolerance = concordance_tolerance
        self.embedding_method = embedding_method


    def find_best_split_in_segment(
        self,
        engine: PrefixDistanceEngine,
        taxa: List[str],
        start_u: int,
        end_u: int,
        step: int = 5
    ) -> Optional[Dict]:
        """
        Evaluates candidate cut points in interval [start_u, end_u) and returns
        the optimal changepoint with validated Ghost Node Z and L-PIR.
        """
        if end_u - start_u < 2 * self.min_tract_units:
            return None

        N = engine.N
        best_call = None
        best_score = -1.0

        cutpoints = list(range(start_u + self.min_tract_units, end_u - self.min_tract_units, step))

        # Finite-sample Thomson-Smirnov bound adjustment for small N
        eff_z_thresh = min(self.ghost_z_threshold, 0.85 * (N - 1) / np.sqrt(max(N, 2)))

        for bp in cutpoints:
            flank = min(75, bp - start_u, end_u - bp)
            if flank < self.min_tract_units:
                continue

            D1 = engine.query_distance_matrix(bp - flank, bp)
            D2 = engine.query_distance_matrix(bp, bp + flank)

            if np.max(D1) < 1e-4 or np.max(D2) < 1e-4:
                continue

            if self.embedding_method == "laplacian":
                Z1 = compute_laplacian_eigenmaps(D1, k=self.k_dims, k_nn=4)
                Z2 = compute_laplacian_eigenmaps(D2, k=self.k_dims, k_nn=4)
            else:
                Z1 = compute_classical_mds(D1, k=self.k_dims)
                Z2 = compute_classical_mds(D2, k=self.k_dims)

            _, residuals = align_procrustes(Z1, Z2)
            z_scores = compute_ghost_node_zscores(residuals)

            # Test top drifting taxa
            top_indices = np.argsort(-z_scores)[:5]

            for t_idx in top_indices:
                t_z = z_scores[t_idx]
                if t_z < eff_z_thresh:
                    continue

                triplet_res = evaluate_triplets_for_taxon(
                    D1, D2, t_idx,
                    bound_factor=self.bound_factor,
                    min_parent_dist=self.min_parent_dist
                )

                if triplet_res is not None and triplet_res["pir"] >= self.pir_threshold:
                    pir_val = triplet_res["pir"]
                    # Composite changepoint score combining Ghost Z and L-PIR
                    composite_score = pir_val * np.log1p(t_z)

                    if composite_score > best_score:
                        best_score = composite_score
                        best_call = {
                            "raw_breakpoint": int(bp),
                            "segment_start": start_u,
                            "segment_end": end_u,
                            "recombinant_idx": t_idx,
                            "recombinant": taxa[t_idx],
                            "parent_left_idx": triplet_res["parent_left_idx"],
                            "parent_left": taxa[triplet_res["parent_left_idx"]],
                            "parent_right_idx": triplet_res["parent_right_idx"],
                            "parent_right": taxa[triplet_res["parent_right_idx"]],
                            "l_pir": float(pir_val),
                            "ghost_z": float(t_z),
                            "composite_score": float(composite_score),
                            "term1": float(triplet_res["term1"]),
                            "term2": float(triplet_res["term2"])
                        }

        return best_call

    def recursive_binary_segmentation(
        self,
        engine: PrefixDistanceEngine,
        taxa: List[str],
        start_u: int,
        end_u: int,
        discovered: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Recursively bisects the alignment into subsegments whenever a statistically
        significant recombination breakpoint is identified.
        """
        if discovered is None:
            discovered = []

        split = self.find_best_split_in_segment(engine, taxa, start_u, end_u, step=self.step)
        if split is not None:
            bp = split["raw_breakpoint"]
            discovered.append(split)

            # Recurse into left subsegment [start_u, bp]
            self.recursive_binary_segmentation(engine, taxa, start_u, bp, discovered)
            # Recurse into right subsegment [bp, end_u]
            self.recursive_binary_segmentation(engine, taxa, bp, end_u, discovered)

        return discovered

    def detect_recombination(
        self,
        engine: Any,
        taxa: List[str],
        tier2_engine: Optional[Any] = None,
        concordance_tolerance: Optional[int] = None
    ) -> List[Dict]:
        """
        Complete end-to-end detection pipeline:
          - If Two-Tier (engine has .tier1 and .tier2, or tier2_engine is provided):
            1. Tier 1 fast screening sieve: RBS over scalar prefix tensor.
               If 0 raw candidate events found, returns [] immediately (fast negative path).
            2. Tier 2 verification & segmentation: RBS over continuous 384D manifold.
               If 0 candidate events found on Tier 2, returns [] (rejects Tier 1 false alarms).
            3. Boundary refinement & spatial concordance: filters by spatial concordance
               with Tier 1 candidates and refines to single-codon precision on Tier 2.
          - If Standalone (scalar, embed-static, or embed-contextual):
            1. Recursive Binary Segmentation
            2. Single-codon boundary refinement
            3. Deduplication and sorting
        """
        tol = self.concordance_tolerance if concordance_tolerance is None else concordance_tolerance

        is_two_tier = hasattr(engine, "tier1") and (hasattr(engine, "_tier2") or hasattr(engine, "_tier2_factory"))

        if is_two_tier or tier2_engine is not None:
            tier1 = engine.tier1 if is_two_tier else engine

            # === Two-Tier Pipeline ===
            # Step 1: Tier 1 fast screening sieve
            raw_events_t1 = self.recursive_binary_segmentation(tier1, taxa, 0, tier1.num_units)
            if len(raw_events_t1) == 0:
                return []

            t1_breakpoints = [ev["raw_breakpoint"] for ev in raw_events_t1]

            # Step 2: Now that Tier 1 screened positive, instantiate/evaluate Tier 2
            tier2 = engine.tier2 if is_two_tier else tier2_engine
            raw_events_t2 = self.recursive_binary_segmentation(tier2, taxa, 0, tier2.num_units)
            if len(raw_events_t2) == 0:
                return []

            # Step 3: Spatial concordance filtering & single-codon refinement on Tier 2
            raw_events_t2.sort(key=lambda x: x["raw_breakpoint"])
            validated = []
            U = tier2.num_units

            for ev in raw_events_t2:
                # Enforce spatial concordance with Tier 1 screening candidates
                if tol is not None and tol > 0:
                    if not any(abs(ev["raw_breakpoint"] - bp1) <= tol for bp1 in t1_breakpoints):
                        continue

                flank = min(80, max(35, ev["raw_breakpoint"], U - ev["raw_breakpoint"]))
                refined_bp, refined_pir = refine_breakpoint_codon(
                    tier2,
                    bp=ev["raw_breakpoint"],
                    r_idx=ev["recombinant_idx"],
                    p1_idx=ev["parent_left_idx"],
                    p2_idx=ev["parent_right_idx"],
                    flank_len=flank,
                    search_radius=15
                )

                if refined_pir < self.pir_threshold:
                    continue

                ev["breakpoint"] = refined_bp
                ev["refined_pir"] = refined_pir
                ev["tier"] = "two-tier"

                # Deduplication: if another event was found within 20 codons, keep the higher L-PIR
                duplicate = False
                for v in validated:
                    if abs(v["breakpoint"] - ev["breakpoint"]) <= 20:
                        duplicate = True
                        if ev["refined_pir"] > v["refined_pir"]:
                            v.update(ev)
                        break

                if not duplicate:
                    validated.append(ev)

            validated.sort(key=lambda x: x["breakpoint"])
            return validated

        # === Single-Tier Standalone Pipeline ===
        U = engine.num_units
        raw_events = self.recursive_binary_segmentation(engine, taxa, 0, U)

        if len(raw_events) == 0:
            return []

        # Sort by genomic coordinate
        raw_events.sort(key=lambda x: x["raw_breakpoint"])

        # Refine each breakpoint to single-codon precision
        validated = []
        for ev in raw_events:
            flank = min(80, max(35, ev["raw_breakpoint"], U - ev["raw_breakpoint"]))
            refined_bp, refined_pir = refine_breakpoint_codon(
                engine,
                bp=ev["raw_breakpoint"],
                r_idx=ev["recombinant_idx"],
                p1_idx=ev["parent_left_idx"],
                p2_idx=ev["parent_right_idx"],
                flank_len=flank,
                search_radius=15
            )

            if refined_pir < self.pir_threshold:
                continue

            ev["breakpoint"] = refined_bp
            ev["refined_pir"] = refined_pir

            # Deduplication: if another event was found within 20 codons, keep the higher L-PIR
            duplicate = False
            for v in validated:
                if abs(v["breakpoint"] - ev["breakpoint"]) <= 20:
                    duplicate = True
                    if ev["refined_pir"] > v["refined_pir"]:
                        v.update(ev)
                    break

            if not duplicate:
                validated.append(ev)

        # Final sort
        validated.sort(key=lambda x: x["breakpoint"])
        return validated
