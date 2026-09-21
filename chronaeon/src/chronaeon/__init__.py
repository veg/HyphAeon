"""
ChronAeon: Ultra-Fast Molecular Clock Dating, Phylodynamics, and Genomic Surveillance.
"""

from .dating import (
    run_mrca_dating,
    run_ols_dating,
    run_pgls_dating,
    run_restricted_spline_clock_dating,
    run_powerlaw_clock_dating,
    verify_coding_alignment,
    parse_sample_dates,
    generate_consensus_sequence,
    generate_time_decay_consensus_sequence,
)
from .geo import (
    run_phylogeography_analysis,
    estimate_spatial_pgls_epicenter,
    parse_geo_metadata,
)
from .r0 import (
    run_r0_analysis,
    compute_reproduction_numbers,
    plot_r0_diagnostics,
    PATHOGEN_PRESETS,
)
from .autoclock import (
    AutoClockDeconvolution,
    run_autoclock_deconvolution,
    HierarchicalAutoClock,
    run_hierarchical_autoclock,
    fit_clock,
    recursive_spectral_autoclock,
    classify_community,
    classify_leaf_community,
)
from .triage import ChronAeonSieve
from .sketch import (
    CanonicalMinHashSketcher,
    AlignmentFreeBinner,
    AlignmentFreeCentrifuge,
)
from .alignment import (
    ReferenceCodonAligner,
    ReferenceGuidedCodonThreader,
)

__version__ = "0.1.1"
__all__ = [
    "run_mrca_dating", "run_ols_dating", "run_pgls_dating",
    "run_restricted_spline_clock_dating", "run_powerlaw_clock_dating",
    "verify_coding_alignment", "parse_sample_dates",
    "generate_consensus_sequence", "generate_time_decay_consensus_sequence",
    "run_phylogeography_analysis", "estimate_spatial_pgls_epicenter",
    "parse_geo_metadata",
    "run_r0_analysis", "compute_reproduction_numbers",
    "plot_r0_diagnostics", "PATHOGEN_PRESETS",
    "AutoClockDeconvolution", "run_autoclock_deconvolution",
    "HierarchicalAutoClock", "run_hierarchical_autoclock",
    "fit_clock", "recursive_spectral_autoclock",
    "classify_community", "classify_leaf_community",
    "ChronAeonSieve",
    "CanonicalMinHashSketcher", "AlignmentFreeBinner", "AlignmentFreeCentrifuge",
    "ReferenceCodonAligner", "ReferenceGuidedCodonThreader",
]
