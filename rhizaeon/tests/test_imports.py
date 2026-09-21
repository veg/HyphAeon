"""Smoke test: verify all rhizaeon submodules import cleanly."""

import rhizaeon
from rhizaeon import (
    PrefixDistanceEngine,
    SNPCompressedPrefixEngine,
    compute_classical_mds,
    compute_laplacian_eigenmaps,
    compute_ghost_node_zscores,
    trace_continuous_manifold_flow,
    compute_pir,
    evaluate_triplets_for_taxon,
    refine_breakpoint_codon,
    RhizAeonDetector,
    render_alluvial_genome_river,
    export_hyphy_partition_json,
    export_nexus_partitions,
    export_hyphy_batchfile,
    compute_haar_scalogram,
    trace_modulus_maxima_ridges,
    run_wavelet_recombination_screen,
    run_fda_recombination_screen,
    run_recursive_partition_fda_screen,
    run_adaptive_hybrid_screen,
    polish_breakpoint_ml,
    polish_all_breakpoints,
    TwoTierPrefixDistanceEngine,
    EmbeddingPrefixDistanceEngine,
    ContextualPrefixDistanceEngine,
    build_prefix_engine,
    load_embedding_matrices,
)
from rhizaeon.conformal import ConformalMetricCalibrator, MultiSegmentConformalEngine


def test_version():
    assert rhizaeon.__version__ == "0.1.0"


def test_all_imports():
    assert PrefixDistanceEngine is not None
    assert SNPCompressedPrefixEngine is not None
    assert compute_classical_mds is not None
    assert compute_laplacian_eigenmaps is not None
    assert compute_ghost_node_zscores is not None
    assert trace_continuous_manifold_flow is not None
    assert compute_pir is not None
    assert evaluate_triplets_for_taxon is not None
    assert refine_breakpoint_codon is not None
    assert RhizAeonDetector is not None
    assert render_alluvial_genome_river is not None
    assert export_hyphy_partition_json is not None
    assert export_nexus_partitions is not None
    assert export_hyphy_batchfile is not None
    assert compute_haar_scalogram is not None
    assert trace_modulus_maxima_ridges is not None
    assert run_wavelet_recombination_screen is not None
    assert run_fda_recombination_screen is not None
    assert run_recursive_partition_fda_screen is not None
    assert run_adaptive_hybrid_screen is not None
    assert polish_breakpoint_ml is not None
    assert polish_all_breakpoints is not None
    assert TwoTierPrefixDistanceEngine is not None
    assert EmbeddingPrefixDistanceEngine is not None
    assert ContextualPrefixDistanceEngine is not None
    assert build_prefix_engine is not None
    assert load_embedding_matrices is not None
    assert ConformalMetricCalibrator is not None
    assert MultiSegmentConformalEngine is not None
