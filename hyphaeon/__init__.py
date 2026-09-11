"""
HyphAeon: Ultra-Fast Neural Inference of Episodic Selection,
Phenotype-Genotype Association Mapping, and Epistatic Sector Mining.
"""

from .model import PhyloAxialTransformer
from .dataset import load_alignment_and_tree, parse_alignment_sequences, compute_tn93_distance_matrix, parse_beast_xml
from .phenotype import run_phenotype_association, resolve_phenotype_vector, PRESETS
from .epistasis import (
    run_epistasis_analysis,
    run_digital_dms_analysis,
    run_dms_analysis,
    compute_phylogenetic_branch_attributions,
    compute_branch_coselection_network,
    compute_selection_dms_essm,
    extract_epistatic_sectors
)
from .disease import predict_disease_pathogenicity
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
)
from .sketch import (
    CanonicalMinHashSketcher,
    AlignmentFreeBinner,
    AlignmentFreeCentrifuge,
)
from .alignment import (
    ReferenceCodonAligner,
    ReferenceGuidedCodonThreader,
)

__version__ = "1.0.0"
__all__ = [
    "PhyloAxialTransformer",
    "load_alignment_and_tree",
    "compute_tn93_distance_matrix",
    "parse_alignment_sequences",
    "parse_beast_xml",
    "run_phenotype_association",
    "resolve_phenotype_vector",
    "run_epistasis_analysis",
    "run_digital_dms_analysis",
    "run_dms_analysis",
    "compute_phylogenetic_branch_attributions",
    "compute_branch_coselection_network",
    "compute_selection_dms_essm",
    "extract_epistatic_sectors",
    "predict_disease_pathogenicity",
    "run_mrca_dating",
    "run_ols_dating",
    "run_pgls_dating",
    "run_restricted_spline_clock_dating",
    "run_powerlaw_clock_dating",
    "verify_coding_alignment",
    "parse_sample_dates",
    "generate_consensus_sequence",
    "generate_time_decay_consensus_sequence",
    "run_phylogeography_analysis",
    "estimate_spatial_pgls_epicenter",
    "parse_geo_metadata",
    "run_r0_analysis",
    "compute_reproduction_numbers",
    "plot_r0_diagnostics",
    "PATHOGEN_PRESETS",
    "PRESETS",
    "AutoClockDeconvolution",
    "run_autoclock_deconvolution",
    "HierarchicalAutoClock",
    "run_hierarchical_autoclock",
    "CanonicalMinHashSketcher",
    "AlignmentFreeBinner",
    "AlignmentFreeCentrifuge",
    "ReferenceCodonAligner",
    "ReferenceGuidedCodonThreader",
]
