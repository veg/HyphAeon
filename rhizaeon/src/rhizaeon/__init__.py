"""
RhizAeon: Tree-Free Reticulate Evolution and Continuous Manifold Recombination Detection.

Part of the HyphAeon foundation ecosystem.
"""

__version__ = "0.1.0"
__author__ = "Antigravity & Sergei L. Kosakovsky Pond"

from rhizaeon.tensor import (
    PrefixDistanceEngine,
    SNPCompressedPrefixEngine,
    encode_alignment_matrix,
    parse_fasta
)
from rhizaeon.manifold import (
    compute_classical_mds,
    compute_laplacian_eigenmaps,
    align_procrustes,
    compute_ghost_node_zscores,
    trace_continuous_manifold_flow
)
from rhizaeon.pir import compute_pir, evaluate_triplets_for_taxon, refine_breakpoint_codon
from rhizaeon.segmentation import RhizAeonDetector
from rhizaeon.alluvial import render_alluvial_genome_river
from rhizaeon.export import (
    build_partition_intervals,
    export_hyphy_partition_json,
    export_nexus_partitions,
    export_hyphy_batchfile
)
from rhizaeon.wavelet import (
    ScalogramResult,
    WaveletRidge,
    compute_haar_scalogram,
    trace_modulus_maxima_ridges,
    run_wavelet_recombination_screen
)
from rhizaeon.fda import (
    FDAResult,
    FDABreakpoint,
    tv1d,
    compute_fda_trajectories,
    extract_fda_breakpoints,
    run_fda_recombination_screen,
    run_recursive_partition_fda_screen,
    run_multiscale_fda_screen
)
from rhizaeon.adaptive import (
    AdaptiveBreakpoint,
    run_adaptive_hybrid_screen
)
from rhizaeon.polisher import (
    PolishedBreakpoint,
    polish_breakpoint_ml,
    polish_all_breakpoints
)
from rhizaeon.embed import (
    TwoTierPrefixDistanceEngine,
    EmbeddingPrefixDistanceEngine,
    ContextualPrefixDistanceEngine,
    build_prefix_engine,
    load_embedding_matrices
)

__all__ = [
    "PrefixDistanceEngine",
    "SNPCompressedPrefixEngine",
    "encode_alignment_matrix",
    "parse_fasta",
    "compute_classical_mds",
    "compute_laplacian_eigenmaps",
    "align_procrustes",
    "compute_ghost_node_zscores",
    "trace_continuous_manifold_flow",
    "compute_pir",
    "evaluate_triplets_for_taxon",
    "refine_breakpoint_codon",
    "RhizAeonDetector",
    "render_alluvial_genome_river",
    "build_partition_intervals",
    "export_hyphy_partition_json",
    "export_nexus_partitions",
    "export_hyphy_batchfile",
    "ScalogramResult",
    "WaveletRidge",
    "compute_haar_scalogram",
    "trace_modulus_maxima_ridges",
    "run_wavelet_recombination_screen",
    "FDAResult",
    "FDABreakpoint",
    "tv1d",
    "compute_fda_trajectories",
    "extract_fda_breakpoints",
    "run_fda_recombination_screen",
    "run_recursive_partition_fda_screen",
    "run_multiscale_fda_screen",
    "AdaptiveBreakpoint",
    "run_adaptive_hybrid_screen",
    "PolishedBreakpoint",
    "polish_breakpoint_ml",
    "polish_all_breakpoints",
    "TwoTierPrefixDistanceEngine",
    "EmbeddingPrefixDistanceEngine",
    "ContextualPrefixDistanceEngine",
    "build_prefix_engine",
    "load_embedding_matrices"
]

