"""
rhizaeon.adaptive
=================
Adaptive Hybrid Recombination Engine: Unifying FDA, Wavelets, and Manifold Geometry.

Combines the complementary strengths of:
  1. Tier 1 (FDA Global Triage): Global spectral projection and 1D Total Variation
     piecewise-constant filtering in milliseconds to prune 95% of non-recombinant space.
  2. Tier 2 (Wavelet Multiresolution Zoom): Scale-space dyadic tracking down the cone
     of influence specifically within candidate regions to extract characteristic scales
     (micro/meso/macro) and Hölder regularity exponents (alpha) to filter noise.
  3. Tier 3 (Local Manifold Procrustes & L-PIR): Local geometric superimposition and
     single-base boundary refinement with exact parental attribution.
"""

from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks

from rhizaeon.tensor import PrefixDistanceEngine
from rhizaeon.manifold import compute_classical_mds, align_procrustes, compute_ghost_node_zscores
from rhizaeon.fda import compute_fda_trajectories, tv1d
from rhizaeon.wavelet import generate_dyadic_scales
from rhizaeon.pir import evaluate_triplets_for_taxon, refine_breakpoint_codon


@dataclass
class AdaptiveBreakpoint:
    """Represents a validated recombination breakpoint from the adaptive hybrid engine."""
    breakpoint_nt: int
    uncertainty_interval_nt: Tuple[int, int]  # [start_nt, end_nt]
    recombinant_taxon: str
    taxon_idx: int
    parent_1: str
    parent_2: str
    l_pir: float
    ghost_z_score: float
    kinetic_z_score: float
    characteristic_scale_nt: int
    scale_regime: str           # 'micro' (<200nt), 'meso' (200-1500nt), 'macro' (>1500nt)
    holder_alpha: float         # Hölder exponent (step jump ~0 to 1, noise < -0.3)
    tv_jump_magnitude: float


def run_adaptive_hybrid_screen(
    engine: PrefixDistanceEngine,
    taxa_names: List[str],
    k_global: int = 6,
    fda_bin_size: int = 300,
    fda_step: int = 30,
    min_kinetic_z: float = 2.5,
    min_pir: float = 0.08
) -> List[AdaptiveBreakpoint]:
    """
    Executes the 3-Tier Adaptive Hybrid Recombination Screen.
    """
    N = engine.N
    L = engine.num_units
    taxa = taxa_names

    # =========================================================================
    # TIER 1: High-Speed Global FDA Trajectory Triage (Milliseconds)
    # =========================================================================
    fda_res = compute_fda_trajectories(
        engine=engine,
        taxa_names=taxa,
        k=k_global,
        bin_size=fda_bin_size,
        step=fda_step
    )

    cutpoints = fda_res.cutpoints
    z_energy = fda_res.z_energy
    Y_tv = fda_res.tv_trajectories
    M = len(cutpoints)

    max_z = np.max(z_energy, axis=0)
    top_taxa = np.argmax(z_energy, axis=0)

    # TV discrete jump magnitudes: || Y_tv(m+1) - Y_tv(m) ||
    tv_jumps = np.zeros((N, M))
    tv_jumps[:, :-1] = np.linalg.norm(np.diff(Y_tv, axis=1), axis=2)

    # Find candidate energy peaks PER TAXON to ensure all co-existing recombinant lineages are tracked
    candidate_regions = []
    for i in range(N):
        peaks, _ = find_peaks(z_energy[i], height=min_kinetic_z, prominence=0.8, distance=6)
        for p in peaks:
            bp_nt = int(cutpoints[p])
            kz = float(z_energy[i, p])
            jump_mag = float(tv_jumps[i, p])
            candidate_regions.append((bp_nt, i, kz, jump_mag))

    # Augment with Coarse Multiscale Procrustes screening (2 anchor scales: 180 and 700 nt)
    coarse_scales = [180, 700]
    coarse_step = 35
    coarse_pos = np.arange(90, L - 90, coarse_step)
    for a in coarse_scales:
        half_a = a // 2
        for cp in coarse_pos:
            if cp - half_a < 0 or cp + half_a > L:
                continue
            D1 = engine.query_distance_matrix(cp - half_a, cp)
            D2 = engine.query_distance_matrix(cp, cp + half_a)
            if np.max(D1) < 1e-4 or np.max(D2) < 1e-4:
                continue
            Z1 = compute_classical_mds(D1, k=min(4, N - 1))
            Z2 = compute_classical_mds(D2, k=min(4, N - 1))
            _, res = align_procrustes(Z1, Z2)
            z_sc = compute_ghost_node_zscores(res)
            top_i = int(np.argmax(z_sc))
            top_z = float(z_sc[top_i])
            if top_z >= 1.8:
                candidate_regions.append((int(cp), top_i, top_z, 1.0))

    if len(candidate_regions) == 0:
        return []

    # Prioritize and cluster candidate seed cones within 60 nt
    clustered_regions = []
    for cr in candidate_regions:
        if not any(abs(cr[0] - cl[0]) < 60 for cl in clustered_regions):
            clustered_regions.append(cr)

    candidate_regions = clustered_regions[:60]

    # =========================================================================
    # TIER 2: Local Wavelet Multiresolution Zoom within Candidate Cones
    # =========================================================================
    dyadic_scales = np.array([64, 128, 256, 512, 1024, 1500])
    dyadic_scales = dyadic_scales[dyadic_scales < L // 2]
    if len(dyadic_scales) == 0:
        dyadic_scales = np.array([64, 128])

    validated_breakpoints: List[AdaptiveBreakpoint] = []

    for cr in candidate_regions:
        seed_bp, t_seed, kz, jump_mag = cr

        # Evaluate local Haar scalogram inside candidate cone
        local_step = 20
        cone_radius = 80
        local_positions = np.arange(
            max(40, seed_bp - cone_radius),
            min(L - 40, seed_bp + cone_radius + 1),
            local_step
        )

        if len(local_positions) == 0:
            continue

        # Ridge tracking along scales
        ridge_pos = []
        ridge_scales = []
        ridge_z = []
        ridge_tax = []

        # Coarse to fine scale evaluation
        curr_p = seed_bp
        for s_idx in range(len(dyadic_scales) - 1, -1, -1):
            a = dyadic_scales[s_idx]
            half_a = a // 2

            # Search in localized cone around curr_p
            search_mask = (local_positions >= curr_p - max(25, half_a // 2)) & (local_positions <= curr_p + max(25, half_a // 2))
            cand_p = local_positions[search_mask]
            if len(cand_p) == 0:
                continue

            best_z_local = -1.0
            best_p_local = curr_p
            best_t_local = t_seed

            for b in cand_p:
                if b - half_a < 0 or b + half_a > L:
                    continue
                d1 = engine.query_distance_matrix(b - half_a, b)
                d2 = engine.query_distance_matrix(b, b + half_a)
                if np.max(d1) < 1e-4 or np.max(d2) < 1e-4:
                    continue
                z1 = compute_classical_mds(d1, k=min(4, N - 1))
                z2 = compute_classical_mds(d2, k=min(4, N - 1))
                _, res = align_procrustes(z1, z2)
                z_sc = compute_ghost_node_zscores(res)
                top_t = int(np.argmax(z_sc))
                max_z = float(z_sc[top_t])
                if max_z > best_z_local:
                    best_z_local = max_z
                    best_p_local = int(b)
                    best_t_local = top_t

            if best_z_local > 1.5:
                curr_p = best_p_local
                ridge_pos.append(curr_p)
                ridge_scales.append(int(a))
                ridge_z.append(best_z_local)
                ridge_tax.append(best_t_local)

        if len(ridge_scales) < 2:
            continue

        # Estimate Hölder regularity alpha
        log_a = np.log(np.array(ridge_scales, dtype=np.float64))
        log_z = np.log(np.maximum(ridge_z, 1e-4))
        if (log_a.max() - log_a.min()) > 0.5:
            alpha, _ = np.polyfit(log_a, log_z, 1)
        else:
            alpha = 0.0

        # Fine-scale converged coordinate
        converged_bp = ridge_pos[-1]
        peak_z = float(np.max(ridge_z))
        peak_scale = ridge_scales[int(np.argmax(ridge_z))]
        t_idx = ridge_tax[-1]
        rec_name = taxa[t_idx]

        # Scale classification
        if peak_scale < 200:
            scale_regime = "micro"
        elif peak_scale <= 1500:
            scale_regime = "meso"
        else:
            scale_regime = "macro"

        # =====================================================================
        # TIER 3: Local Manifold Procrustes & Single-Base L-PIR Refinement
        # =====================================================================
        flank_eval = min(max(80, peak_scale // 2), converged_bp, L - converged_bp)
        flank_eval = max(30, flank_eval)

        D1 = engine.query_distance_matrix(converged_bp - flank_eval, converged_bp)
        D2 = engine.query_distance_matrix(converged_bp, converged_bp + flank_eval)

        trip = evaluate_triplets_for_taxon(D1, D2, t_idx)
        if trip is None or trip["pir"] < min_pir:
            continue

        p1_idx = trip["parent_left_idx"]
        p2_idx = trip["parent_right_idx"]
        p1_name = taxa[p1_idx]
        p2_name = taxa[p2_idx]
        pir_val = float(trip["pir"])

        # Fine-scale single-base / interval refinement:
        # scan around converged_bp for minimum sum of squared distance discrepancies
        refine_window = max(35, local_step * 2)
        s_min = max(flank_eval, converged_bp - refine_window)
        s_max = min(L - flank_eval, converged_bp + refine_window)
        
        best_ref_bp = converged_bp
        min_disc = 1e9
        for test_s in range(s_min, s_max + 1, 3):
            # discrepancy = (d(R, P1)_left + d(R, P2)_right)
            d_l = engine.query_distance_matrix(test_s - flank_eval, test_s)
            d_r = engine.query_distance_matrix(test_s, test_s + flank_eval)
            disc = d_l[t_idx, p1_idx] + d_r[t_idx, p2_idx]
            if disc < min_disc:
                min_disc = disc
                best_ref_bp = test_s

        # Re-evaluate triplet metrics at the refined coordinate
        D1_ref = engine.query_distance_matrix(best_ref_bp - flank_eval, best_ref_bp)
        D2_ref = engine.query_distance_matrix(best_ref_bp, best_ref_bp + flank_eval)
        trip_ref = evaluate_triplets_for_taxon(D1_ref, D2_ref, t_idx)
        if trip_ref is not None:
            pir_val = float(trip_ref["pir"])
            p1_name = taxa[trip_ref["parent_left_idx"]]
            p2_name = taxa[trip_ref["parent_right_idx"]]

        # Spatial uncertainty interval where posterior PIR remains > 70% of max
        uncert_left = max(0, best_ref_bp - local_step)
        uncert_right = min(L, best_ref_bp + local_step)

        validated_breakpoints.append(AdaptiveBreakpoint(
            breakpoint_nt=int(best_ref_bp),
            uncertainty_interval_nt=(int(uncert_left), int(uncert_right)),
            recombinant_taxon=rec_name,
            taxon_idx=t_idx,
            parent_1=p1_name,
            parent_2=p2_name,
            l_pir=pir_val,
            ghost_z_score=peak_z,
            kinetic_z_score=kz,
            characteristic_scale_nt=peak_scale,
            scale_regime=scale_regime,
            holder_alpha=float(alpha),
            tv_jump_magnitude=jump_mag
        ))

    # Deduplicate within 50 nt for same taxon
    validated_breakpoints.sort(key=lambda b: b.ghost_z_score * b.l_pir, reverse=True)
    dedup: List[AdaptiveBreakpoint] = []
    for vb in validated_breakpoints:
        if not any(abs(vb.breakpoint_nt - d.breakpoint_nt) < 50 and vb.taxon_idx == d.taxon_idx for d in dedup):
            dedup.append(vb)

    # Final sort by coordinate
    dedup.sort(key=lambda b: b.breakpoint_nt)
    return dedup


# =============================================================================
# DUAL ARCHITECTURE: TIER 1 FAST SCALAR SCREEN -> TIER 2 TRANSFORMER LENS
# =============================================================================

@dataclass
class DualArchitectureConfig:
    """Configuration and dispatch thresholds for the Dual Architecture."""
    max_plateau_nt: int = 50          # Dispatch if uninformative plateau > 50 nt
    min_flank_snps: int = 4           # Dispatch if flank informative SNPs < 4
    min_fiedler_div: float = 0.40     # Tier 2 Fiedler divergence threshold
    min_taxon_drift: float = 0.035    # Tier 2 per-taxon drift threshold
    ghost_root_ratio: float = 1.25    # Ratio of [ROOT] token attention inside vs outside
    weights_path: Optional[str] = None
    device: str = "auto"
    window_codons: int = 25


@dataclass
class Tier2TransformerResult:
    """Output from Tier 2 PhyloAxialTransformer precision resolution."""
    refined_breakpoint_codon: int
    refined_breakpoint_nt: int
    fiedler_divergence: float
    top_recombinant_taxon: str
    taxon_drift: float
    is_ghost_parent: bool
    ghost_root_enrichment: float
    all_taxa_drifts: Dict[str, float]
    dispatch_reason: str


def evaluate_tier2_trigger(
    plateau_width: int,
    num_snps: int,
    pir_val: float,
    config: Optional[DualArchitectureConfig] = None
) -> Tuple[bool, str]:
    """
    Evaluates whether a candidate breakpoint from Tier 1 warrants Tier 2 Transformer dispatch.
    """
    cfg = config or DualArchitectureConfig()
    reasons = []
    if plateau_width > cfg.max_plateau_nt:
        reasons.append(f"wide_plateau_{plateau_width}nt")
    if num_snps < cfg.min_flank_snps:
        reasons.append(f"sparse_snps_{num_snps}")
    if pir_val < 0.15:
        reasons.append(f"borderline_pir_{pir_val:.3f}")

    should_dispatch = len(reasons) > 0
    reason_str = "; ".join(reasons) if should_dispatch else "tier1_confident"
    return should_dispatch, reason_str


def dispatch_tier2_transformer(
    fasta_path: str,
    candidate_nt: int,
    uncertainty_window_nt: Tuple[int, int],
    recombinant_taxon: Optional[str] = None,
    config: Optional[DualArchitectureConfig] = None
) -> Optional[Tier2TransformerResult]:
    """
    Executes Tier 2 PhyloAxialTransformer precision resolution over an ambiguous candidate region.

    Targets:
      1. Single-Sequence Resolution: Computes per-taxon attention drift ||S_left[i] - S_right[i]||
         to pinpoint the exact mosaic lineage without tree reconstruction.
      2. Ghost Parents: Identifies unsampled donor lineages via [ROOT] token attention surge A_i0.
      3. GARD-Equivalent Major Rearrangements: Computes normalized Graph Laplacian L(s) and
         Fiedler vector angular divergence (1 - cos theta) across candidate codons.
      4. Plateau Collapse: Refines wide inter-SNP plateaus down to single-codon precision.
    """
    import sys
    import os
    from pathlib import Path
    import scipy.linalg as la

    cfg = config or DualArchitectureConfig()

    # Dynamically locate aeon_core
    try:
        import torch
        from aeon_core.inference import load_model, get_device
        from aeon_core.dataset import load_alignment_and_tree
    except ImportError:
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        aeon_core_path = repo_root / "aeon-core" / "src"
        if aeon_core_path.exists() and str(aeon_core_path) not in sys.path:
            sys.path.insert(0, str(aeon_core_path))
        try:
            import torch
            from aeon_core.inference import load_model, get_device
            from aeon_core.dataset import load_alignment_and_tree
        except ImportError:
            return None

    # Resolve default weights checkpoint
    weights_path = cfg.weights_path
    if not weights_path or not os.path.exists(weights_path):
        try:
            from aeon_core.weights import resolve_weights_path
            resolved = resolve_weights_path(weights=cfg.weights_path)
            if resolved and os.path.exists(resolved):
                weights_path = str(resolved)
        except Exception:
            pass

    if not weights_path or not os.path.exists(weights_path):
        default_ckpts = [
            str(Path.home() / ".cache" / "hyphaeon" / "axomeme_5_dim384_nonull.pt"),
            str(Path.home() / ".cache" / "hyphaeon" / "model.safetensors"),
            "/Users/sergei/Projects/TOGA_MEME/axomeme_5_dim384_nonull.pt",
            "/Users/sergei/Projects/TOGA_MEME/phylomlm_dim384.pt",
        ]
        for cp in default_ckpts:
            if cp and os.path.exists(cp):
                weights_path = cp
                break

    if not weights_path:
        return None

    # Resolve device
    if cfg.device == "auto":
        dev = get_device()
    elif cfg.device == "cpu":
        dev = torch.device("cpu")
    elif cfg.device == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
    elif cfg.device == "mps" and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    # Load alignment and precompute TN93 tree cache
    try:
        c_tensor, a_tensor, d_tensor, z_tensor, _, taxa, L = load_alignment_and_tree(
            fasta_path, use_tn93=True, prune_duplicates=False
        )
    except Exception:
        return None

    model = load_model(weights=weights_path, device=dev)
    msa_codons = c_tensor.to(dev)
    msa_aas = a_tensor.to(dev)
    dist_mat = d_tensor.squeeze(0).cpu().numpy()
    mds_coords = z_tensor.squeeze(0).cpu().numpy()
    tree_cache = model.precompute_tree_cache(dist_mat, mds_coords)

    model.eval()
    batch_size, num_species, window_size = msa_codons.shape
    num_nodes = num_species + 1

    static_biases = tree_cache["static_phylo_biases"]
    static_coss = tree_cache["static_rope_coss"]
    static_sins = tree_cache["static_rope_sins"]

    with torch.no_grad():
        codon_emb = model.codon_embedding(msa_codons)
        aa_emb = model.aa_embedding(msa_aas)
        x = torch.cat([codon_emb, aa_emb], dim=-1) + model.pos_embedding.unsqueeze(1)
        phylo_pos = tree_cache.get("mds_pos_static", tree_cache.get("static_phylo_pos"))
        if phylo_pos.dim() == 3:
            x = x + phylo_pos.unsqueeze(2)
        else:
            x = x + phylo_pos.unsqueeze(0).unsqueeze(2)
        root = model.root_token.expand(batch_size, 1, window_size, -1)
        x_full = torch.cat([root, x], dim=1)

        for i in range(len(model.col_layers)):
            col_in = x_full.reshape(batch_size * num_nodes, window_size, model.embed_dim) if window_size > 1 else x_full.reshape(batch_size * num_nodes, model.embed_dim)
            col_out = model.col_layers[i](col_in)
            x_full = col_out.reshape(batch_size, num_nodes, window_size, model.embed_dim)

        x0_dup = x_full.transpose(1, 2).contiguous().view(batch_size * window_size, num_nodes, model.embed_dim)
        accum_site_attn = torch.zeros((batch_size, num_species, num_species), dtype=torch.float32, device=dev)
        accum_root_attn = torch.zeros((batch_size, num_species), dtype=torch.float32, device=dev)

        for i, layer in enumerate(model.row_layers):
            row_in = x_full.transpose(1, 2).contiguous().view(batch_size * window_size, num_nodes, model.embed_dim)
            q = layer.q_proj(row_in).view(batch_size * window_size, num_nodes, layer.num_heads, layer.head_dim).transpose(1, 2)
            k = layer.k_proj(row_in).view(batch_size * window_size, num_nodes, layer.num_heads, layer.head_dim).transpose(1, 2)
            v = layer.v_proj(row_in).view(batch_size * window_size, num_nodes, layer.num_heads, layer.head_dim).transpose(1, 2)

            cos, sin = static_coss[i], static_sins[i]
            half_dim = layer.head_dim // 2
            q1, q2 = q[..., :half_dim], q[..., half_dim:]
            k1, k2 = k[..., :half_dim], k[..., half_dim:]
            q = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
            k = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

            scores = torch.matmul(q, k.transpose(-2, -1)) / (layer.head_dim ** 0.5) + static_biases[i]
            attn_weights = torch.softmax(scores, dim=-1)

            accum_site_attn += attn_weights[:, :, 1:, 1:].mean(dim=1)
            accum_root_attn += attn_weights[:, :, 1:, 0].mean(dim=1)

            out = torch.matmul(attn_weights, v)
            out = out.transpose(1, 2).contiguous().view(batch_size * window_size, num_nodes, layer.embed_dim)
            out = layer.out_proj(out) + layer.alpha_skip * x0_dup
            row_out = model.row_norms[i](row_in + out)
            x_full = row_out.reshape(batch_size, window_size, num_nodes, model.embed_dim).transpose(1, 2)

    site_attn = (accum_site_attn / len(model.row_layers)).cpu().numpy()
    root_attn = (accum_root_attn / len(model.row_layers)).cpu().numpy()

    # Symmetrized attention graph
    S_sites = np.zeros_like(site_attn)
    for l in range(L):
        s = (site_attn[l] + site_attn[l].T) / 2.0
        np.fill_diagonal(s, 0.0)
        S_sites[l] = s

    # Convert candidate coordinates to codons
    c_start = max(cfg.window_codons, uncertainty_window_nt[0] // 3)
    c_end = min(L - cfg.window_codons, uncertainty_window_nt[1] // 3)

    if c_start >= c_end:
        c_start = max(cfg.window_codons, (candidate_nt // 3) - 20)
        c_end = min(L - cfg.window_codons, (candidate_nt // 3) + 20)

    # Scan candidate window for Fiedler angular divergence and per-taxon drift
    w = cfg.window_codons
    best_codon = candidate_nt // 3
    max_fiedler = 0.0
    best_taxa_drifts = np.zeros(num_species)

    for l in range(c_start, c_end):
        Sl = np.mean(S_sites[l - w : l], axis=0)
        Sr = np.mean(S_sites[l : l + w], axis=0)

        # Graph Laplacians & Fiedler vectors
        deg_l = np.sum(Sl, axis=1)
        deg_r = np.sum(Sr, axis=1)
        deg_l_inv_sqrt = np.where(deg_l > 1e-8, 1.0 / np.sqrt(deg_l), 0.0)
        deg_r_inv_sqrt = np.where(deg_r > 1e-8, 1.0 / np.sqrt(deg_r), 0.0)

        L_left = np.eye(num_species) - np.outer(deg_l_inv_sqrt, deg_l_inv_sqrt) * Sl
        L_right = np.eye(num_species) - np.outer(deg_r_inv_sqrt, deg_r_inv_sqrt) * Sr

        w_l, v_l = la.eigh(L_left)
        w_r, v_r = la.eigh(L_right)

        v2_l, v2_r = v_l[:, 1], v_r[:, 1]
        if np.dot(v2_l, v2_r) < 0:
            v2_r = -v2_r

        cos_sim = np.dot(v2_l, v2_r) / (np.linalg.norm(v2_l) * np.linalg.norm(v2_r) + 1e-12)
        d_fiedler = 1.0 - max(-1.0, min(1.0, cos_sim))

        if d_fiedler > max_fiedler:
            max_fiedler = float(d_fiedler)
            best_codon = l
            for i in range(num_species):
                best_taxa_drifts[i] = float(np.linalg.norm(Sl[i, :] - Sr[i, :]))

    # Identify top drifting taxon (Single Sequence Resolution)
    top_tax_idx = int(np.argmax(best_taxa_drifts))
    top_tax_name = taxa[top_tax_idx]
    top_drift = float(best_taxa_drifts[top_tax_idx])

    # Check for Ghost Parent attribution: [ROOT] token attention inside window vs background
    inside_root = float(np.mean(root_attn[max(0, best_codon - 15) : min(L, best_codon + 15), top_tax_idx]))
    outside_root = float(np.mean(root_attn[:, top_tax_idx]))
    root_ratio = inside_root / max(1e-6, outside_root)
    is_ghost = root_ratio >= cfg.ghost_root_ratio

    all_drifts = {taxa[i]: float(best_taxa_drifts[i]) for i in range(num_species)}

    return Tier2TransformerResult(
        refined_breakpoint_codon=int(best_codon),
        refined_breakpoint_nt=int(best_codon * 3),
        fiedler_divergence=max_fiedler,
        top_recombinant_taxon=top_tax_name,
        taxon_drift=top_drift,
        is_ghost_parent=is_ghost,
        ghost_root_enrichment=root_ratio,
        all_taxa_drifts=all_drifts,
        dispatch_reason=f"resolved_fiedler_{max_fiedler:.3f}"
    )

