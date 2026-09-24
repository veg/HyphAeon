"""
pipeline.py
===========
End-to-end pipeline runner for bacterial chromosome recombination meta-analysis.
Orchestrates:
1. Multi-scale detection of candidate recombination events
2. 5-signature biological classification
3. Chromosomal polar mapping and biophysical feature extraction
4. Mechanistic vs. selective deconvolution
"""

import time
from typing import List, Dict, Any, Optional
import numpy as np

from .signatures import (
    RecombinationSignature,
    RecombinationEvent,
    classify_recombination_event,
    compute_gap_gradient
)
from .chromosomal_map import (
    ChromosomalCoordinates,
    compute_gc_skew,
    compute_chi_density
)
from .deconvolution import (
    fit_mechanistic_prior,
    compute_selective_sieve,
    DeconvolutionResults
)


class BacterialRecombinationPipeline:
    """Orchestrates chromosome-wide bacterial recombination analysis."""

    def __init__(
        self,
        chromosome_length: int,
        ori_pos: int = 0,
        ter_pos: Optional[int] = None,
        species: str = "Streptococcus pneumoniae"
    ):
        self.chromosome_length = chromosome_length
        self.species = species
        self.chrom_coords = ChromosomalCoordinates(chromosome_length, ori_pos, ter_pos)

    def analyze_chromosome(
        self,
        raw_events: List[Dict[str, Any]],
        ref_sequence: Optional[str] = None,
        seq_mat: Optional[np.ndarray] = None,
        grid_step_bp: int = 2500
    ) -> Dict[str, Any]:
        """
        Runs the full signature classification and deconvolution analysis.
        
        Args:
            raw_events: List of raw detected events with coordinates and metrics
            ref_sequence: Complete reference chromosome sequence (optional, for Chi & Skew)
            seq_mat: (N, L) alignment matrix (optional, for exact gap gradients)
            grid_step_bp: Step size along chromosome for continuous density modeling
            
        Returns:
            Dictionary containing classified events, spatial features, and deconvolution results
        """
        t0 = time.time()
        print(f"[*] Running Bacterial Recombination Pipeline across {self.chromosome_length:,} bp chromosome...")

        # 1. Classify all events into biological signatures
        classified_events: List[RecombinationEvent] = []
        for idx, ev in enumerate(raw_events):
            start = ev["start"]
            end = ev["end"]
            length = end - start + 1
            gap_grad = ev.get("gap_gradient", 0.0)
            ortho = ev.get("orthogonal_departure", 0.0)
            l_pir = ev.get("l_pir", 1.0)
            z = ev.get("kinetic_z", 3.0)
            drift = ev.get("directional_attention_drift", 0.0)
            chi = ev.get("chi_score", 0.0)
            att = ev.get("att_score", 0.0)

            sig, conf = classify_recombination_event(
                length_bp=length,
                gap_gradient=gap_grad,
                orthogonal_departure=ortho,
                l_pir=l_pir,
                kinetic_z=z,
                directional_attention_drift=drift,
                chi_score=chi,
                att_score=att
            )

            c_ev = RecombinationEvent(
                event_id=f"EV_{idx+1:04d}",
                recombinant_taxon=ev.get("taxon", f"Taxon_{idx}"),
                start_pos=start,
                end_pos=end,
                length_bp=length,
                gap_gradient=gap_grad,
                orthogonal_departure=ortho,
                l_pir=l_pir,
                kinetic_z=z,
                directional_attention_drift=drift,
                flanking_chi_score=chi,
                flanking_att_score=att,
                signature=sig,
                confidence=conf,
                metadata=ev.get("metadata", {})
            )
            classified_events.append(c_ev)

        # 2. Extract Chromosomal Features
        grid_coords = np.arange(0, self.chromosome_length, grid_step_bp)
        M = len(grid_coords)

        dist_ori = np.array([self.chrom_coords.dist_to_origin_bp(c) for c in grid_coords], dtype=float)
        dist_ter = np.array([self.chrom_coords.dist_to_terminus_bp(c) for c in grid_coords], dtype=float)

        if ref_sequence is not None:
            _, gc_skew_raw = compute_gc_skew(ref_sequence, window_bp=15000, step_bp=grid_step_bp)
            gc_skew = gc_skew_raw[:M] if len(gc_skew_raw) >= M else np.pad(gc_skew_raw, (0, M - len(gc_skew_raw)))
            _, chi_dens_raw = compute_chi_density(ref_sequence, species=self.species, grid_step_bp=grid_step_bp)
            chi_density = chi_dens_raw[:M] if len(chi_dens_raw) >= M else np.pad(chi_dens_raw, (0, M - len(chi_dens_raw)))
        else:
            gc_skew = np.zeros(M)
            chi_density = np.ones(M)

        # 3. Compute Observed Recombination Rate Field lambda_obs(s)
        lambda_obs = np.zeros(M, dtype=float)
        sigma_kernel = 15000.0  # 15 kb smoothing
        for ev in classified_events:
            mid = (ev.start_pos + ev.end_pos) / 2.0
            diffs = np.abs(grid_coords - mid)
            circ_diffs = np.minimum(diffs, self.chromosome_length - diffs)
            lambda_obs += np.exp(-(circ_diffs ** 2) / (2.0 * sigma_kernel ** 2))

        # 4. Deconvolve: Fit Mechanistic Prior and Compute Selective Sieve
        is_trna = np.zeros(M, dtype=bool)  # default if annotations not supplied
        lambda_mech = fit_mechanistic_prior(
            coordinates=grid_coords,
            event_counts=lambda_obs,
            chi_density=chi_density,
            dist_to_ori=dist_ori,
            is_tRNA_att=is_trna,
            abs_gc_skew=np.abs(gc_skew)
        )

        deconv_res = compute_selective_sieve(
            coordinates=grid_coords,
            lambda_observed=lambda_obs,
            lambda_mechanistic=lambda_mech
        )

        t_elapsed = time.time() - t0
        print(f"[✓] Analysis complete in {t_elapsed:.2f} s. {len(classified_events)} events classified, {len(deconv_res.hotspots)} adaptive hotspots identified.")

        return {
            "events": classified_events,
            "coordinates": grid_coords,
            "dist_to_ori": dist_ori,
            "dist_to_ter": dist_ter,
            "gc_skew": gc_skew,
            "chi_density": chi_density,
            "deconvolution": deconv_res,
            "runtime_sec": t_elapsed
        }
