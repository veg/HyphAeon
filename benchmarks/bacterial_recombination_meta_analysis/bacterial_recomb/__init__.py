"""
bacterial_recomb
================
A computational framework for bacterial recombination meta-analysis.
Deconvolves foreign DNA delivery mechanisms (transformation, transduction, conjugation)
from evolutionary natural selection across complete circular chromosomes.
"""

from .signatures import (
    RecombinationSignature,
    RecombinationEvent,
    classify_recombination_event,
    compute_gap_gradient,
    compute_orthogonal_departure
)
from .chromosomal_map import (
    ChromosomalCoordinates,
    compute_gc_skew,
    compute_chi_density,
    SPECIES_CHI_MOTIFS
)
from .deconvolution import (
    fit_mechanistic_prior,
    compute_selective_sieve,
    DeconvolutionResults
)

__all__ = [
    "RecombinationSignature",
    "RecombinationEvent",
    "classify_recombination_event",
    "compute_gap_gradient",
    "compute_orthogonal_departure",
    "ChromosomalCoordinates",
    "compute_gc_skew",
    "compute_chi_density",
    "SPECIES_CHI_MOTIFS",
    "fit_mechanistic_prior",
    "compute_selective_sieve",
    "DeconvolutionResults",
]
