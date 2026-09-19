"""
chronaeon/dating.py
-------------------
Heterochronous Molecular Clock Calibration and Ancestor Dating (t_MRCA)
for Pathogen Genomics.

This module is a facade that re-exports from the dating submodules:
- dating_kernels: Neural covariance kernel & metricity diagnostics
- dating_io: Alignment validation, consensus, timestamp parsing
- dating_divergence: Tree/TN93/latent divergence computation
- dating_models: OLS/PGLS/spline/power-law dating estimators
- dating_plots: Diagnostic visualization
- dating_pipeline: Master MRCA dating pipeline
"""

from .dating_kernels import (
    compute_neural_covariance_kernel,
    compute_attention_covariance_kernel,
    compute_transformer_metricity_diagnostics,
)
from .dating_io import (
    verify_coding_alignment,
    generate_consensus_sequence,
    generate_time_decay_consensus_sequence,
    compute_time_decay_profile_divergences,
    parse_header_timestamp,
    parse_sample_dates,
)
from .dating_divergence import (
    extract_tree_root_to_tip,
    compute_tree_free_divergences,
    optimize_latent_convex_hull_root,
)
from .dating_models import (
    _spectral_eigh,
    _gls_fit,
    _generalized_r2,
    compute_fieller_mrca_interval,
    compute_poisson_mrca_interval,
    compute_residual_bootstrap_mrca_interval,
    run_dating_loocv,
    run_ols_dating,
    run_pgls_dating,
    estimate_reml_pagel_lambda,
    tune_ridge_for_pgls,
    run_powerlaw_clock_dating,
    compute_rcs_basis,
    run_restricted_spline_clock_dating,
)
from .dating_plots import (
    plot_mrca_dating,
    plot_alluvial_phylogeny,
)
from .dating_pipeline import (
    run_mrca_dating,
    _compute_precision_weighted_ensemble,
    _select_clock_model,
    _compute_taxon_predictions,
    _export_dating_results,
)
