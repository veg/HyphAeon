"""
HyphAeon: Ultra-Fast Neural Inference of Episodic Selection,
Phenotype-Genotype Association Mapping, and Epistatic Sector Mining.
"""

from .model import PhyloAxialTransformer
from .dataset import load_alignment_and_tree, parse_alignment_sequences
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

__version__ = "1.0.0"
__all__ = [
    "PhyloAxialTransformer",
    "load_alignment_and_tree",
    "parse_alignment_sequences",
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
    "PRESETS",
]
