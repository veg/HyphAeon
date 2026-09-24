#!/usr/bin/env python3
"""
test_signatures.py
==================
Unit tests verifying the 5-signature bacterial recombination classification engine,
chromosomal coordinates, and mechanistic deconvolution.
"""

import math
import numpy as np
import pytest

from bacterial_recomb.signatures import (
    RecombinationSignature,
    classify_recombination_event,
    compute_gap_gradient,
    compute_orthogonal_departure,
    scan_attachment_motifs
)
from bacterial_recomb.chromosomal_map import (
    ChromosomalCoordinates,
    compute_gc_skew,
    compute_chi_density
)
from bacterial_recomb.deconvolution import (
    fit_mechanistic_prior,
    compute_selective_sieve
)


def test_homologous_transformation():
    """Authentic natural transformation: gapless, core manifold, 3.5 kb."""
    sig, conf = classify_recombination_event(
        length_bp=3500,
        gap_gradient=0.02,
        orthogonal_departure=0.08,
        l_pir=0.98,
        kinetic_z=5.2,
        chi_score=1.0
    )
    assert sig == RecombinationSignature.HOMOLOGOUS_CONVERSION
    assert conf >= 0.90


def test_micro_conversion():
    """Micro-conversion: ultra short (120 bp), gapless, directional attention drift."""
    sig, conf = classify_recombination_event(
        length_bp=120,
        gap_gradient=0.00,
        orthogonal_departure=0.05,
        l_pir=0.60,
        kinetic_z=2.1,
        directional_attention_drift=0.18
    )
    assert sig == RecombinationSignature.MICRO_CONVERSION
    assert conf >= 0.85


def test_specialized_transduction_prophage():
    """Lysogenic prophage: 38 kb, structural indel step, att direct repeats."""
    sig, conf = classify_recombination_event(
        length_bp=38000,
        gap_gradient=0.95,
        orthogonal_departure=0.60,
        l_pir=0.00,
        kinetic_z=6.5,
        att_score=1.0
    )
    assert sig == RecombinationSignature.SPECIALIZED_TRANSDUCTION
    assert conf >= 0.90


def test_generalized_transduction():
    """Generalized transduction: capsid-bounded (32 kb) homologous allelic swap."""
    sig, conf = classify_recombination_event(
        length_bp=32000,
        gap_gradient=0.05,
        orthogonal_departure=0.12,
        l_pir=0.92,
        kinetic_z=4.8
    )
    assert sig == RecombinationSignature.GENERALIZED_TRANSDUCTION
    assert conf >= 0.85


def test_conjugative_ice():
    """Conjugative mega-island (ICE): 81 kb (like ICESp23FST81), massive indel, orthogonal departure."""
    sig, conf = classify_recombination_event(
        length_bp=81000,
        gap_gradient=1.00,
        orthogonal_departure=0.72,
        l_pir=0.00,
        kinetic_z=7.8
    )
    assert sig == RecombinationSignature.CONJUGATIVE_ICE
    assert conf >= 0.90


def test_transposition_is_element():
    """Transposon / IS element: 1.4 kb structural indel."""
    sig, conf = classify_recombination_event(
        length_bp=1400,
        gap_gradient=0.88,
        orthogonal_departure=0.45,
        l_pir=0.10,
        kinetic_z=3.4
    )
    assert sig == RecombinationSignature.TRANSPOSITION
    assert conf >= 0.80


def test_chromosomal_coordinates():
    """Tests circular chromosome distances to oriC and ter."""
    coords = ChromosomalCoordinates(chromosome_length=2000000, ori_pos=0, ter_pos=1000000)
    
    # Distance to origin
    assert coords.dist_to_origin_bp(100) == 100
    assert coords.dist_to_origin_bp(1999900) == 100
    assert coords.dist_to_origin_bp(1000000) == 1000000

    # Distance to terminus
    assert coords.dist_to_terminus_bp(1000000) == 0
    assert coords.dist_to_terminus_bp(999900) == 100
    assert coords.dist_to_terminus_bp(1000100) == 100

    # Polar angle
    assert math.isclose(coords.polar_angle_rad(0), 0.0)
    assert math.isclose(coords.polar_angle_rad(500000), math.pi / 2.0)
    assert math.isclose(coords.polar_angle_rad(1000000), math.pi)


def test_deconvolution_engine():
    """Tests fitting mechanistic prior and detecting adaptive hotspots vs deserts."""
    coords = np.linspace(0, 1000000, 200)
    # Synthetic observed events: flat baseline with one massive hotspot at 300 kb and a desert at 700 kb
    obs_rates = np.ones(200) * 1.0
    # Hotspot at index 60 (300 kb)
    obs_rates[55:65] = 12.0
    # Desert at index 140 (700 kb)
    obs_rates[135:145] = 0.05

    chi_dens = np.ones(200)
    dist_ori = np.abs(coords - 0)
    is_trna = np.zeros(200, dtype=bool)
    abs_skew = np.ones(200) * 0.1

    lambda_mech = fit_mechanistic_prior(coords, obs_rates, chi_dens, dist_ori, is_trna, abs_skew)
    results = compute_selective_sieve(coords, obs_rates, lambda_mech, hotspot_threshold=2.0, desert_threshold=-1.5)

    assert len(results.hotspots) >= 1
    assert any(h["start"] <= 300000 <= h["end"] for h in results.hotspots)
    assert len(results.deserts) >= 1
    assert any(d["start"] <= 700000 <= d["end"] for d in results.deserts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
