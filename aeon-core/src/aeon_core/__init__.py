"""
Aeon-Core: Shared model and bioinformatics infrastructure for Aeon-family packages.
"""

from .model import PhyloAxialTransformer, BustedMultiTaskHead, decode_soft_ordinal_lrt
from .weights import (
    resolve_weights_path, load_arch_config, load_weights,
    load_model_config, list_available_variants, print_available_variants, get_variant_filename,
    HF_REPO_ID, DEFAULT_VARIANT, CACHE_DIR,
)
from .inference import (
    get_device, get_device_memory_budget,
    get_live_available_memory, compute_live_adaptive_chunk_sizes,
    compute_adaptive_safe_batch_size,
    load_model, prepare_alignment,
)
from .splits import (
    extract_cross_taxa_attentions_and_embeddings,
    compute_fused_affinity_matrix,
)
from .dataset import (
    GENETIC_CODE, CODON_TO_AA, AA_MAP,
    get_codon_token, get_aa_token,
    parse_alignment_sequences,
    compute_tn93_distance_matrix,
    compute_tn93_cross_distance_matrix,
    parse_beast_xml,
    extract_tree_from_string_or_file,
    has_nonzero_branch_lengths,
    estimate_tree_branch_lengths_hyphy,
    enforce_nonzero_branch_lengths,
    compute_fast_dist_matrix,
    compute_mds_coordinates,
    downsample_taxa_faith_pd,
    prune_identical_sequences,
    load_alignment_and_tree,
)
from .temporal import (
    parse_date_to_decimal,
    parse_dates_from_auspice_json,
    extract_date_from_string,
)
from .io import ensure_parent_directory, write_json, write_csv, format_pq
from .stats import (
    pvals_from_lrt_meme,
    pvals_from_lrt_self_liang,
    benjamini_hochberg,
    cauchy_combination_p,
)

__version__ = "0.1.0"
