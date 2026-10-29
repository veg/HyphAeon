use serde::{Deserialize, Serialize};

/// Encoded alignment representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alignment {
    pub taxa: Vec<String>,
    pub num_taxa: usize,
    pub length: usize,
    /// Row-major flattened sequence matrix: index = i * length + u
    /// Encoding: 0: gap/ambiguity, 1: A, 2: C, 3: G, 4: T/U
    pub matrix: Vec<u8>,
}

impl Alignment {
    pub fn new(taxa: Vec<String>, length: usize, matrix: Vec<u8>) -> Self {
        let num_taxa = taxa.len();
        assert_eq!(matrix.len(), num_taxa * length);
        Self {
            taxa,
            num_taxa,
            length,
            matrix,
        }
    }

    #[inline(always)]
    pub fn get(&self, taxon_idx: usize, site_idx: usize) -> u8 {
        self.matrix[taxon_idx * self.length + site_idx]
    }

    #[inline(always)]
    pub fn row(&self, taxon_idx: usize) -> &[u8] {
        let start = taxon_idx * self.length;
        &self.matrix[start..start + self.length]
    }
}

/// Clonal landmark set on Delta^K
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LandmarkSet {
    pub indices: Vec<usize>,
    pub taxa: Vec<String>,
    pub num_landmarks: usize,
    /// 2D polar/metric coordinates (K+1, 2) where index 0 is ROOT [0, 0]
    pub coords: Vec<f64>,
    pub root_to_tip: Vec<f64>,
}

/// Screening summary output from Pass 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreeningResult {
    pub v_cont: Vec<f64>,
    pub v_root: Vec<f64>,
    pub r_root: Vec<f64>,
    pub gamma_root: Vec<f64>,
    pub rate_burst_indices: Vec<usize>,
    pub ghost_candidate_indices: Vec<usize>,
    pub candidate_recombinant_indices: Vec<usize>,
    pub clonal_backbone_indices: Vec<usize>,
    pub median_v_cont: f64,
    pub is_non_recombinant_h0: bool,
}

/// Physically deconvolved tract from Pass 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalTract {
    pub candidate_idx: usize,
    pub home_channel: usize,
    pub donor_channel: usize,
    pub is_crossover: bool,
    pub u1_discrete: usize, // 1-indexed
    pub u2_discrete: usize, // 1-indexed
    pub u1_continuous: f64,
    pub u2_continuous: f64,
    pub delta_c_max: f64,
    pub z_phys: f64,
    pub z_crit: f64,
    pub is_physically_significant: bool,
}

/// Verified recombination event from Pass 3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecombinationEvent {
    pub event_id: usize,
    pub isolates: Vec<String>,
    pub candidate_idx: usize,
    pub candidate_name: String,
    pub home_idx: Option<usize>,
    pub home_name: String,
    pub donor_idx: Option<usize>,
    pub donor_name: String,
    pub is_ghost_donor: bool,
    pub is_crossover: bool,
    pub u1: usize, // 1-indexed
    pub u2: usize, // 1-indexed
    pub u1_continuous: f64,
    pub u2_continuous: f64,
    pub tract_length: usize,
    pub s_informative: usize,
    pub p_fisher: f64,
    pub z_phys: f64,
    pub d_flank_home: f64,
    pub d_flank_donor: f64,
    pub d_tract_home: f64,
    pub d_tract_donor: f64,
    pub is_verified: bool,
}

/// Complete scan result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanResult {
    pub num_taxa: usize,
    pub length: usize,
    pub informative_sites: usize,
    pub is_non_recombinant_h0: bool,
    pub screening: ScreeningResult,
    pub events: Vec<RecombinationEvent>,
    pub run_time_ms: f64,
}

use std::collections::BTreeMap;

/// Metadata describing the alignment and recombination scan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationMetadata {
    pub title: String,
    pub engine_version: String,
    pub alignment_length: usize,
    pub taxa_count: usize,
    pub query_id: String,
    pub p1_id: String,
    pub p2_id: String,
    pub unique_events_count: usize,
    pub recombinants_count: usize,
    pub breakpoints_count: usize,
    pub informative_snps_count: usize,
    pub mean_plateau_width: f64,
    pub max_support: f64,
    pub run_time_ms: f64,
    pub is_non_recombinant_h0: bool,
}

/// Taxon metadata and display attributes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxonMeta {
    pub label: String,
    #[serde(rename = "type")]
    pub taxon_type: String,
    pub color: String,
    #[serde(default)]
    pub input_order: usize,
}

/// Macro genomic domain or partition segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroSegment {
    pub name: String,
    pub start: usize,
    pub end: usize,
    pub color: String,
}

/// Detailed breakpoint record for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationBreakpoint {
    pub breakpoint_id: String,
    pub event_id: usize,
    pub isolates: Vec<String>,
    pub recombinant: String,
    pub parent_1: String,
    pub parent_2: String,
    pub breakpoint_nt: usize,
    pub coarse_bp: usize,
    pub ci_left: usize,
    pub ci_right: usize,
    pub plateau_width: usize,
    pub tract_start: usize,
    pub tract_end: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flanking_p1_site: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flanking_p2_site: Option<usize>,
    pub log_likelihood_gain: f64,
    pub kinetic_z: f64,
    pub l_pir: f64,
    pub p_fisher: f64,
    pub is_crossover: bool,
    pub is_ghost_donor: bool,
}

/// Segment in a taxon's mosaic representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MosaicSegment {
    pub start: usize,
    pub end: usize,
    pub lineage: String,
    pub color: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_plateau: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub breakpoint_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_id: Option<usize>,
}

/// Informative SNP site record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformativeSnpRecord {
    pub pos: usize,
    #[serde(rename = "type")]
    pub match_type: String,
    pub base_r: char,
    pub base_p1: char,
    pub base_p2: char,
}

/// Continuous signals across the genome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationSignals {
    pub trajectory_x: Vec<usize>,
    pub trajectory_y: Vec<f64>,
    pub velocity_dy: Vec<f64>,
    pub informative_snps: Vec<InformativeSnpRecord>,
    pub walk_x: Vec<usize>,
    pub walk_s: Vec<i32>,
    pub drawdown: Vec<f64>,
    pub brownian_bridge: Vec<f64>,
}

/// Metric subspace projections (Classical MDS on partitions)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationSubspaces {
    pub partition_1: BTreeMap<String, Vec<f64>>,
    pub partition_2: BTreeMap<String, Vec<f64>>,
    pub p1_title: String,
    pub p2_title: String,
}

/// Gower out-of-sample continuous 2D projection point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GowerPoint {
    pub x: f64,
    pub y: f64,
    pub u: usize,
}

/// Time-invariant Canonical Metric Manifold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalManifold {
    pub ref_coords: BTreeMap<String, Vec<f64>>,
    pub gower_trajectory: Vec<GowerPoint>,
    pub home_coord: Vec<f64>,
    pub donor_coord: Vec<f64>,
    pub chord_vector: Vec<f64>,
    pub cos_theta: f64,
    pub theta_deg: f64,
    pub is_colinear: bool,
}

/// Enriched hotspot or conserved coldspot interval in the genome-wide recombination profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecombinationRegion {
    pub start: usize,
    pub end: usize,
    pub peak_nt: usize,
    pub peak_flux: f64,
    pub mean_flux: f64,
    pub event_count: usize,
    pub label: String,
}

/// Genome-wide recombination hotspot / coldspot flux and cumulative strain profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecombinationLandscape {
    pub grid_x: Vec<usize>,
    pub flux_phi: Vec<f64>,
    pub cumulative_strain: Vec<f64>,
    pub hotspot_threshold: f64,
    pub coldspot_threshold: f64,
    pub hotspots: Vec<RecombinationRegion>,
    pub coldspots: Vec<RecombinationRegion>,
}

/// Complete rich visualization dossier emitted by RhizAeon
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationDossier {
    pub metadata: VisualizationMetadata,
    pub taxa: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_taxa: Vec<String>,
    pub taxa_meta: BTreeMap<String, TaxonMeta>,
    pub macro_segments: Vec<MacroSegment>,
    pub breakpoints: Vec<VisualizationBreakpoint>,
    pub mosaic_taxa: BTreeMap<String, Vec<MosaicSegment>>,
    pub signals: VisualizationSignals,
    pub subspaces: VisualizationSubspaces,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical_manifold: Option<CanonicalManifold>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recombination_landscape: Option<RecombinationLandscape>,
    pub sequences: BTreeMap<String, String>,
}
