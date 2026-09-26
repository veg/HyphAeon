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
    compute_ghost_node_zscores,
    compute_grubbs_effective_z
)
from rhizaeon.pir import evaluate_triplets_for_taxon, refine_breakpoint_codon


CALIBRATION_PROFILES = {
    # High-sensitivity calibrated profile (optimal F1 on benchmark, <=0.5% FPR, detects micro-tracts):
    "calibrated": {
        "ghost_z": 2.75,
        "pir": 0.20,
    },
    # Conservative strict profile (100% false positive elimination, 0.0% FPR):
    "strict": {
        "ghost_z": 3.00,
        "pir": 0.25,
    },
}


class RhizAeonDetector:
    """
    Tree-Free Reticulate Evolution & Recombination Detection Engine.
    
    Combines continuous sequence manifold geometry, out-of-sample Ghost Node
    Procrustes triage, and bounded Latent Parental Incongruence Ratio (L-PIR).
    Supports data-driven adaptive tract length estimation and principled calibration profiles.
    """

    def __init__(
        self,
        window_units: int = 25,
        min_tract_units: Union[int, str] = "auto",
        ghost_z_threshold: Optional[float] = None,
        pir_threshold: Optional[float] = None,
        calibration: str = "calibrated",
        k_dims: int = 4,
        bound_factor: float = 1.25,
        min_parent_dist: float = 0.025,
        step: int = 5,
        concordance_tolerance: Optional[int] = 60,
        embedding_method: str = "mds"
    ):
        calib_key = calibration.lower() if isinstance(calibration, str) else "calibrated"
        profile = CALIBRATION_PROFILES.get(calib_key, CALIBRATION_PROFILES["calibrated"])

        self.calibration = calib_key
        self.window_units = window_units
        self.min_tract_units = min_tract_units
        self.ghost_z_threshold = ghost_z_threshold if ghost_z_threshold is not None else profile["ghost_z"]
        self.pir_threshold = pir_threshold if pir_threshold is not None else profile["pir"]
        self.k_dims = k_dims
        self.bound_factor = bound_factor
        self.min_parent_dist = min_parent_dist
        self.step = step
        self.concordance_tolerance = concordance_tolerance
        self.embedding_method = embedding_method

    def get_effective_min_tract(self, engine: Any) -> int:
        """
        Calculates the physical Poisson mutation information limit on tract length:
          L_min = max(15, ceil(3.0 / mean_divergence))
        If min_tract_units was explicitly specified as an integer, returns that value.
        """
        if isinstance(self.min_tract_units, (int, float)):
            val = int(self.min_tract_units)
            if val > 0:
                return val
        if isinstance(self.min_tract_units, str) and self.min_tract_units.isdigit():
            val = int(self.min_tract_units)
            if val > 0:
                return val

        # Data-driven calculation based on alignment divergence
        if hasattr(engine, "get_mean_divergence"):
            d_mean = engine.get_mean_divergence()
        elif hasattr(engine, "tier1") and hasattr(engine.tier1, "get_mean_divergence"):
            d_mean = engine.tier1.get_mean_divergence()
        else:
            d_mean = 0.10

        # Minimum 3 informative substitutions between parents; geometric stability floor of 15
        return max(15, int(np.ceil(3.0 / max(d_mean, 1e-4))))


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
        eff_tract = self.get_effective_min_tract(engine)
        if end_u - start_u < 2 * eff_tract:
            return None

        N = engine.N
        best_call = None
        best_score = -1.0

        cutpoints = list(range(start_u + eff_tract, end_u - eff_tract, step))

        # Finite-sample Thompson-Grubbs bound adjustment for small N
        eff_z_thresh = compute_grubbs_effective_z(N, self.ghost_z_threshold, alpha=0.005)

        for bp in cutpoints:
            flank = min(max(self.window_units, 75), bp - start_u, end_u - bp)
            if flank < min(eff_tract, 25):
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
            # Step 1: Tier 1 RP-FDA screening sieve
            from rhizaeon.fda import run_recursive_partition_fda_screen
            eff_min_tract = self.get_effective_min_tract(engine)
            bps_t1 = run_recursive_partition_fda_screen(
                engine=tier1,
                taxa_names=taxa,
                min_len=max(15, eff_min_tract),
                min_z=max(1.5, self.ghost_z_threshold * 0.70),
                min_pir=max(0.08, self.pir_threshold * 0.70),
                crossover_validation=True,
                polish_ml=True
            )
            if len(bps_t1) == 0:
                return []

            # Step 2: Tier 2 Manifold Verification & Handoff
            from rhizaeon.adaptive import DualArchitectureConfig, evaluate_tier2_trigger
            cfg = DualArchitectureConfig()
            tier2 = engine.tier2 if is_two_tier else tier2_engine
            U = tier2.num_units
            N = len(taxa)
            eff_z_thresh = compute_grubbs_effective_z(N, self.ghost_z_threshold, alpha=0.005)
            validated = []

            for b in bps_t1:
                target_bp = b.breakpoint_nt if b.breakpoint_nt is not None else b.coarse_bp
                if target_bp is None or target_bp <= 0 or target_bp >= U:
                    continue

                if float(b.kinetic_z) < eff_z_thresh:
                    continue

                flank = min(max(self.window_units, 50), max(20, min(target_bp, U - target_bp)))
                if target_bp - flank < 0 or target_bp + flank > U:
                    flank = min(target_bp, U - target_bp)
                if flank < 5:
                    continue

                D1 = tier1.query_distance_matrix(target_bp - flank, target_bp)
                D2 = tier1.query_distance_matrix(target_bp, target_bp + flank)

                trip2 = evaluate_triplets_for_taxon(
                    D1, D2, b.taxon_idx,
                    bound_factor=self.bound_factor,
                    min_parent_dist=self.min_parent_dist
                )
                if trip2 is None:
                    continue
                if trip2["pir"] < self.pir_threshold:
                    continue

                p1_idx = trip2["parent_left_idx"]
                p2_idx = trip2["parent_right_idx"]
                p1_name = taxa[p1_idx]
                p2_name = taxa[p2_idx]

                refined_bp, refined_pir = refine_breakpoint_codon(
                    tier2,
                    bp=target_bp,
                    r_idx=b.taxon_idx,
                    p1_idx=p1_idx,
                    p2_idx=p2_idx,
                    flank_len=flank,
                    search_radius=15
                )

                if refined_pir < self.pir_threshold:
                    continue

                # Evaluate formal Tier 1 breakdown trigger
                should_t2, reason = evaluate_tier2_trigger(
                    plateau_width=b.plateau_width or 0,
                    num_snps=b.num_informative_sites or 10,
                    pir_val=refined_pir,
                    config=cfg
                )

                if tol is not None and tol > 0:
                    if abs(refined_bp - target_bp) > tol:
                        continue

                is_codon = getattr(tier1, "codon_aligned", getattr(engine, "codon_aligned", False))
                scale_coord = 3 if is_codon else 1
                ev = {
                    "breakpoint": refined_bp,
                    "raw_breakpoint": target_bp,
                    "breakpoint_nt": refined_bp * scale_coord,
                    "recombinant_idx": b.taxon_idx,
                    "recombinant": b.recombinant_taxon,
                    "parent_left_idx": p1_idx,
                    "parent_left": p1_name,
                    "parent_right_idx": p2_idx,
                    "parent_right": p2_name,
                    "l_pir": float(b.l_pir),
                    "refined_pir": float(refined_pir),
                    "ghost_z": float(b.kinetic_z),
                    "kinetic_z": float(b.kinetic_z),
                    "tier": "two-tier",
                    "tier2_resolved": should_t2,
                    "ci_left": b.ci_left,
                    "ci_right": b.ci_right,
                    "ci_left_nt": b.nt_ci_left if b.nt_ci_left is not None else (b.ci_left * scale_coord if b.ci_left is not None else None),
                    "ci_right_nt": b.nt_ci_right if b.nt_ci_right is not None else (b.ci_right * scale_coord if b.ci_right is not None else None),
                    "plateau_width": b.plateau_width,
                    "plateau_width_nt": b.nt_plateau_width if b.nt_plateau_width is not None else (b.plateau_width * scale_coord if b.plateau_width is not None else None),
                    "log_likelihood_gain": b.log_likelihood_gain,
                    "flanking_p1_site": b.flanking_p1_site,
                    "flanking_p2_site": b.flanking_p2_site,
                    "flanking_p1_site_nt": b.nt_flanking_p1 if b.nt_flanking_p1 is not None else (b.flanking_p1_site * scale_coord if b.flanking_p1_site is not None else None),
                    "flanking_p2_site_nt": b.nt_flanking_p2 if b.nt_flanking_p2 is not None else (b.flanking_p2_site * scale_coord if b.flanking_p2_site is not None else None),
                }

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
