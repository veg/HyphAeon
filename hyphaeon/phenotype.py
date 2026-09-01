"""
hyphaeon/phenotype.py
--------------------
Directional Phenotype-Genotype Association Mapping (PhyloWAS) and
Phenotype-Associated Residue Signature (PARS) extraction.

Uses continuous Transformer Attribution Vectors (root-to-leaf multi-head
axial attention weights combined with phylogenetic branch projections)
to discover convergent and directional trait-associated molecular adaptations.
"""

import os
import re
import sys
import json
import math
import fnmatch
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from Bio import Phylo

from .dataset import (
    AA_MAP,
    CODON_TO_AA,
    load_alignment_and_tree,
    parse_alignment_sequences,
    extract_tree_from_string_or_file
)
from .model import PhyloAxialTransformer
from .stats import cauchy_combination_p, benjamini_hochberg
from .inference import get_device, load_model
from .weights import (
    load_weights,
    load_arch_config,
    resolve_weights_path,
    DEFAULT_VARIANT
)

DEFAULT_WEIGHTS = "weights/hyphaeon_v1.pt"
from .epistasis import compute_transformer_attributions

REV_AA_MAP = {v: k for k, v in AA_MAP.items()}

PRESETS = {
    "echolocation": {
        "title": "Mammalian Echolocation Convergence",
        "description": "Microchiropteran bats and odontocete toothed whales.",
        "foreground": [
            "rhi*", "hip*", "myo*", "pte*", "mor*", "min*", "emb*", "cra*", "meg*", "mol*",
            "turTru", "delLeu", "orcOrc", "gloMel", "phaSin", "graGri", "neoPho", "phoPho",
            "phyCat", "kogBre", "kogSim", "mesBid", "zipCav", "plaGan", "iniGeo", "lipVex", "ponBla"
        ],
        "controls": ["pteAle", "pteRod", "pteRuf", "pteGig", "pteVam", "ptePse", "bal*", "megNov", "eubGla", "escRob"]
    },
    "marine": {
        "title": "Marine Mammal Transition & Deep Diving Hypoxia",
        "description": "Cetaceans, Pinnipeds, Sirenians, and Sea Otters.",
        "foreground": [
            "enhLut*", "pusHis*", "pusSib*", "halGryp*", "phoVit*", "phoLar*", "eriBar*", "neoSch*",
            "odoRos*", "zalCal*", "eumJub*", "arcAus*", "arcGaz*", "otoFla*", "mirLeo*", "mirAng*",
            "lepWed*", "hydLep*", "lobCar*", "ommRos*", "triMan*", "triSen*", "triInu*", "dugDug*",
            "turTru*", "delLeu*", "orcOrc*", "gloMel*", "phaSin*", "graGri*", "neoPho*", "phoPho*",
            "balMys*", "balAcu*", "balPhy*", "balMus*", "megNov*", "eubGla*", "escRob*", "phyCat*",
            "kogBre*", "kogSim*", "mesBid*", "zipCav*", "plaGan*", "iniGeo*", "lipVex*", "ponBla*"
        ]
    },
    "fossorial": {
        "title": "Subterranean Fossoriality & Hypercapnic Hypoxia",
        "description": "Naked mole-rats, blind mole-rats, golden moles, star-nosed moles, pocket gophers.",
        "foreground": [
            "hetGla*", "fukDam*", "cryAns*", "nanGal*", "nanEhr*", "spaCar*", "conCri*", "talEur*",
            "talOcc*", "scaMos*", "scaAqu*", "chrAsi*", "chrSta*", "uroGra*", "geoBur*", "thoTal*",
            "canTub*", "ellLut*", "ellTal*"
        ]
    },
    "hibernation": {
        "title": "True Hibernation & Metabolic Torpor",
        "description": "Marmots, ground squirrels, dormice, tenrecs, hedgehogs, Myotis bats.",
        "foreground": [
            "ictTri*", "uroPar*", "speCit*", "speDau*", "marFla*", "marMar*", "marVan*", "marMon*",
            "gliGli*", "dryNit*", "musAve*", "eriEur*", "tenEca*", "echTel*", "micTal*", "myoLuc*",
            "myoDau*", "myoMyo*", "myoNat*", "myoBra*", "ursArc*"
        ]
    },
    "longevity": {
        "title": "Extreme Longevity & Peto's Paradox Centenarians",
        "description": "Bowhead whale, naked mole-rat, Brandt's bat, elephants, humans.",
        "foreground": [
            "balMys*", "hetGla*", "myoBra*", "loxAfr*", "eleMax*", "homSap*"
        ]
    },
    "high_altitude": {
        "title": "High-Altitude Hypoxia Adaptation",
        "description": "Yak, Tibetan antelope, snow leopard, vicuna, pikas, chinchilla.",
        "foreground": [
            "bosGru*", "bosMut*", "panHod*", "panUnc*", "vicVic*", "vicPac*", "chiLan*", "ochCur*",
            "ochPri*", "ochArg*"
        ]
    },
    "cardenolide": {
        "title": "Insect Cardenolide Resistance (ATP1a)",
        "description": "Chrysochus, Tetraopes, Danaus (Monarch), Oncopeltus.",
        "foreground": [
            "chrysochus*", "tetraopes*", "danaus*", "oncopeltus*", "chrysomela*"
        ]
    },
    "dim_light": {
        "title": "Low-Light & Deep-Sea Rhodopsin Vision",
        "description": "Deep-sea teleosts, cavefish, coelacanth, marine diving mammals.",
        "foreground": [
            "*eel*", "*conger*", "*scabbard*", "*blackdragon*", "*viperfish*", "*loosejaw*",
            "*lampfish*", "*thornyhead*", "*cavefish*", "*dolphin*", "*coelacanth*"
        ]
    }
}

def resolve_phenotype_vector(
    taxa: List[str],
    preset: Optional[str] = None,
    foreground: Optional[Union[str, List[str]]] = None,
    background: Optional[Union[str, List[str]]] = None,
    phenotype_file: Optional[str] = None,
    trait_col: Optional[str] = None,
    species_col: Optional[str] = None,
    continuous: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Constructs the phenotypic trait vector y in R^N across all N taxa.
    Supports flexible presets, metadata table parsing, and inline pattern matching.
    """
    N = len(taxa)
    y = np.zeros(N, dtype=float)
    meta = {
        "mode": "discrete",
        "foreground_count": 0,
        "background_count": 0,
        "description": ""
    }

    # 1. Phenotype Metadata File
    if phenotype_file:
        if not os.path.exists(phenotype_file):
            raise FileNotFoundError(f"Phenotype file not found: {phenotype_file}")

        sep = "	" if phenotype_file.endswith((".tsv", ".tab")) else ","
        df_pheno = pd.read_csv(phenotype_file, sep=sep)

        # Auto-detect species column if not specified
        if not species_col:
            for c in ["species", "taxon", "taxa", "tree_leaf_name", "assembly", "id", "name", "species_name"]:
                match = [col for col in df_pheno.columns if col.lower() == c]
                if match:
                    species_col = match[0]
                    break
            if not species_col:
                species_col = df_pheno.columns[0]

        # Auto-detect trait column if not specified
        if not trait_col:
            cand_cols = [c for c in df_pheno.columns if c != species_col]
            if not cand_cols:
                raise ValueError(f"No valid trait column found in {phenotype_file}")
            trait_col = cand_cols[0]

        trait_dict = {}
        for _, row in df_pheno.iterrows():
            sp = str(row[species_col]).strip()
            val = row[trait_col]
            trait_dict[sp] = val
            trait_dict[sp.lower()] = val

        # Match taxa against dictionary
        matched_values = []
        for i, t in enumerate(taxa):
            val = None
            if t in trait_dict:
                val = trait_dict[t]
            elif t.lower() in trait_dict:
                val = trait_dict[t.lower()]
            else:
                # Substring / pattern matching
                for k, v in trait_dict.items():
                    if k in t or t in k:
                        val = v
                        break

            if val is not None and not pd.isna(val):
                if not continuous:
                    val_str = str(val).strip().lower()
                    fg_targets = ["1", "true", "yes", "case", "foreground", "target", "positive"]
                    if foreground:
                        if isinstance(foreground, str):
                            fg_targets.extend([p.strip().lower() for p in foreground.split(",") if p.strip()])
                        elif isinstance(foreground, (list, tuple)):
                            fg_targets.extend([str(p).strip().lower() for p in foreground])
                    y[i] = 1.0 if val_str in fg_targets else 0.0
                    matched_values.append(y[i])
                else:
                    try:
                        num_val = float(val)
                        if not np.isnan(num_val):
                            y[i] = num_val
                            matched_values.append(num_val)
                    except ValueError:
                        pass

        if continuous:
            meta["mode"] = "continuous"
            if len(matched_values) > 0 and np.std(matched_values) > 0:
                y = (y - np.mean(y)) / np.std(y)
            meta["description"] = f"Continuous trait '{trait_col}' from {os.path.basename(phenotype_file)}"
        else:
            meta["mode"] = "discrete"
            meta["foreground_count"] = int(np.sum(y > 0))
            meta["background_count"] = int(np.sum(y <= 0))
            meta["description"] = f"Discrete trait '{trait_col}' from {os.path.basename(phenotype_file)}"

        return y, meta

    # 2. Curated Presets
    if preset:
        preset_key = preset.lower().replace("-", "_").strip()
        if preset_key not in PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}")
        p_info = PRESETS[preset_key]
        patterns = p_info["foreground"]
        meta["description"] = f"{p_info['title']} ({p_info['description']})"
        for i, t in enumerate(taxa):
            for pat in patterns:
                if fnmatch.fnmatch(t.lower(), pat.lower()) or (pat.lower() in t.lower()):
                    y[i] = 1.0
                    break

        meta["mode"] = "discrete"
        meta["foreground_count"] = int(np.sum(y > 0))
        meta["background_count"] = int(np.sum(y <= 0))
        return y, meta

    # 3. Inline Foreground / Background Patterns
    if foreground:
        if isinstance(foreground, str):
            if "|" in foreground and "," not in foreground:
                fg_list = [p.strip() for p in foreground.split("|") if p.strip()]
            else:
                fg_list = [p.strip() for p in foreground.split(",") if p.strip()]
        else:
            fg_list = list(foreground)

        for i, t in enumerate(taxa):
            for pat in fg_list:
                clean_pat = pat.rstrip(".*").lstrip(".*") if pat.startswith(".*") or pat.endswith(".*") else pat
                try:
                    if re.search(pat, t, re.IGNORECASE):
                        y[i] = 1.0
                        break
                except Exception:
                    pass
                if fnmatch.fnmatch(t.lower(), pat.lower()) or (clean_pat.lower() in t.lower()):
                    y[i] = 1.0
                    break

        meta["mode"] = "discrete"
        meta["foreground_count"] = int(np.sum(y > 0))
        meta["background_count"] = int(np.sum(y <= 0))
        meta["description"] = f"User-specified foreground patterns: {fg_list}"
        return y, meta

    raise ValueError("Must provide one of --preset, --phenotype-file, or --foreground.")


def compute_phylogenetic_covariance(tree: Phylo.BaseTree.Tree, taxa: List[str]) -> np.ndarray:
    """
    Computes the phylogenetic variance-covariance matrix V (Saputra et al. 2021):
    V[i, j] = shared patristic distance from root to MRCA(taxon_i, taxon_j).
    """
    M = len(taxa)
    V = np.zeros((M, M), dtype=np.float64)
    root = tree.root
    root_dists = {t.name: tree.distance(root, t) for t in tree.get_terminals()}

    for i in range(M):
        t1 = tree.find_any(name=taxa[i])
        for j in range(i, M):
            t2 = tree.find_any(name=taxa[j])
            if i == j:
                V[i, i] = root_dists.get(taxa[i], 1.0)
            elif t1 is not None and t2 is not None:
                mrca = tree.common_ancestor(t1, t2)
                shared_d = tree.distance(root, mrca)
                V[i, j] = shared_d
                V[j, i] = shared_d
    return V


def generate_permulations(
    original_y: np.ndarray,
    tree: Phylo.BaseTree.Tree,
    taxa: List[str],
    n_perm: int = 1000,
    seed: int = 42
) -> np.ndarray:
    """
    Generates n_perm phylogenetic permulations preserving the tree covariance structure
    under Brownian motion (Saputra et al. 2021 / RERconverge null model).

    - For binary phenotypes: Uses the threshold liability model, setting the top-K
      largest simulated liability scores to 1.0 (matching exact foreground count K).
    - For continuous phenotypes: Uses rank-matching transformation, sorting the simulated
      Brownian motion vector and mapping back to the exact empirical values.

    Returns:
        np.ndarray of shape (n_perm, len(taxa))
    """
    np.random.seed(seed)
    M = len(taxa)
    V = compute_phylogenetic_covariance(tree, taxa)

    # Cholesky factorization with tiny ridge for numerical stability
    L_chol = np.linalg.cholesky(V + 1e-7 * np.eye(M))

    # Simulate Brownian motion paths: shape (M, n_perm)
    Z = np.dot(L_chol, np.random.randn(M, n_perm))

    unique_vals = np.unique(original_y)
    is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})

    perm_matrix = np.zeros((n_perm, M), dtype=float)

    if is_binary:
        k_foreground = int(np.sum(original_y > 0))
        for p in range(n_perm):
            top_k_idx = np.argsort(Z[:, p])[-k_foreground:]
            perm_matrix[p, top_k_idx] = 1.0
    else:
        sorted_orig = np.sort(original_y)
        for p in range(n_perm):
            ranks = np.argsort(np.argsort(Z[:, p]))
            perm_matrix[p] = sorted_orig[ranks]

    return perm_matrix


def run_phenotype_association(
    alignment_path: str,
    tree_path: Optional[str] = None,
    weights_path: Optional[str] = DEFAULT_WEIGHTS,
    variant: Optional[str] = None,
    preset: Optional[str] = None,
    foreground: Optional[Union[str, List[str]]] = None,
    background: Optional[Union[str, List[str]]] = None,
    phenotype_file: Optional[str] = None,
    trait_col: Optional[str] = None,
    species_col: Optional[str] = None,
    continuous: bool = False,
    permulations: int = 0,
    min_taxa_per_site: int = 4,
    alpha: float = 0.05,
    cpu: bool = False,
    batch_size: int = 64,
    progress: bool = True
) -> Dict[str, Any]:
    """
    Executes directional Phenotype-Genotype association (PhyloWAS) on a codon alignment
    using continuous Transformer Attribution Vectors (multi-head phylogenetic attention
    attributions and branch projections) rather than binary string substitution counts.
    """
    # 1. Device Selection
    device = get_device(cpu=cpu)

    # 2. Load Alignment, Tree, and Extract Tree Cache
    c_tensor, a_tensor, d_mat, z_coords, inv_mask, taxa, L = load_alignment_and_tree(
        alignment_path, tree_path, prune_duplicates=True
    )
    tree_obj = Phylo.read(tree_path, 'newick') if tree_path and os.path.exists(tree_path) else None
    if tree_obj is None:
        tree_obj = extract_tree_from_string_or_file(tree_path if tree_path else alignment_path)

    N = len(taxa)

    # 3. Resolve Phenotype Vector
    y, meta = resolve_phenotype_vector(
        taxa=taxa,
        preset=preset,
        foreground=foreground,
        background=background,
        phenotype_file=phenotype_file,
        trait_col=trait_col,
        species_col=species_col,
        continuous=continuous
    )

    is_fg = (y > 0)
    fg_count = int(np.sum(is_fg))
    if fg_count < 2 and not continuous:
        raise ValueError(f"Insufficient foreground taxa ({fg_count}) matching criteria among {N} taxa.")

    # 4. Load Neural Architecture and Pretrained Weights
    model = load_model(weights=weights_path, variant=variant, device=device)
    tree_cache = model.precompute_tree_cache(d_mat.to(device), z_coords.to(device))

    # 5. Extract Transformer Phylogenetic Attributions
    if progress:
        print(f"[*] Computing Transformer Attributions across {L} codons...", flush=True)
    leaf_attr, lrts, pvals, cons_aas = compute_transformer_attributions(
        model, c_tensor, a_tensor, tree_cache, taxa, device, batch_size=batch_size, progress=progress
    )

    # 6. Directional Unit-Hypersphere Attribution Projection
    y_norm = y / np.linalg.norm(y) if np.linalg.norm(y) > 0 else y
    proj = leaf_attr @ y_norm # [L]
    spectral_energy = float(np.linalg.norm(proj))
    frob_norm = float(np.linalg.norm(leaf_attr))
    norm_spectral_ratio = float(spectral_energy / frob_norm) if frob_norm > 0 else 0.0

    a_np = a_tensor.squeeze(-1).numpy() # [L, N] amino acid token matrix

    # 6b. Vectorized Phylogenetic Permulations (Saputra et al. 2021 / RERconverge null model)
    null_rhos = None
    gene_p_perm = None
    if permulations > 0 and tree_obj is not None:
        try:
            Y_perms = generate_permulations(y, tree_obj, taxa, n_perm=permulations) # [P, N]
            norm_y_perms = np.linalg.norm(Y_perms, axis=1) # [P]
            
            # Site-level null correlations: [L, P]
            norm_leaf_mat = np.linalg.norm(leaf_attr, axis=1, keepdims=True) # [L, 1]
            null_rhos = (leaf_attr @ Y_perms.T) / (norm_leaf_mat @ norm_y_perms[None, :] + 1e-15)
            
            # Gene-level null spectral energies: [P]
            null_projs = (leaf_attr @ Y_perms.T) / (norm_y_perms[None, :] + 1e-15)
            null_spectral_energies = np.linalg.norm(null_projs, axis=0)
            gene_p_perm = float((1.0 + np.sum(null_spectral_energies >= spectral_energy)) / (1.0 + permulations))
        except Exception as e:
            null_rhos = None
            gene_p_perm = None

    # 7. Site-Level Transformer Attribution Associations
    site_results = []
    for s in range(L):
        a_s = leaf_attr[s, :] # [N] continuous transformer attribution vector
        norm_a = float(np.linalg.norm(a_s))
        
        valid_mask = (a_np[s] < 20)
        N_valid = int(np.sum(valid_mask))
        
        if norm_a > 0 and N_valid >= min_taxa_per_site:
            rho = float(np.dot(a_s, y) / (norm_a * np.linalg.norm(y) + 1e-15))
            
            # Continuous Attribution Student's t-statistic (Parametric Null)
            df = max(1, N_valid - 2)
            t_stat = rho * np.sqrt(df / max(1e-15, 1.0 - rho**2))
            p_assoc_parametric = float(stats.t.sf(t_stat, df=df))
            
            # Empirical Phylogenetic Permulation p-value
            if null_rhos is not None:
                p_assoc_perm = float((1.0 + np.sum(null_rhos[s] >= rho)) / (1.0 + permulations))
                p_assoc = p_assoc_perm
            else:
                p_assoc_perm = None
                p_assoc = p_assoc_parametric
            
            lrt_val = float(lrts[s])
            p_lrt = float(pvals[s])
            score = float(np.sqrt(max(0.0, lrt_val)) * max(0.0, rho))
            
            # ACAT Cauchy combination: combines omnibus selection LRT + directional trait attribution
            p_combined = cauchy_combination_p(np.array([p_lrt, p_assoc]))

            ref_aa = cons_aas[s]
            fg_valid_aa = a_np[s, is_fg][a_np[s, is_fg] < 20]
            if len(fg_valid_aa) > 0:
                derived_tok = int(np.argmax(np.bincount(fg_valid_aa)))
                derived_aa = REV_AA_MAP.get(derived_tok, ref_aa)
                fg_freq = float(np.sum(fg_valid_aa == derived_tok) / len(fg_valid_aa) * 100.0)
            else:
                derived_aa = ref_aa
                fg_freq = 0.0

            bg_valid_aa = a_np[s, ~is_fg][a_np[s, ~is_fg] < 20]
            if len(bg_valid_aa) > 0 and derived_aa != '-':
                derived_tok = AA_MAP.get(derived_aa, 20)
                bg_freq = float(np.sum(bg_valid_aa == derived_tok) / len(bg_valid_aa) * 100.0)
            else:
                bg_freq = 0.0

            fg_mean_attn = float(np.mean(a_s[is_fg])) if fg_count > 0 else 0.0
            bg_mean_attn = float(np.mean(a_s[~is_fg])) if (N - fg_count) > 0 else 0.0

            site_results.append({
                "site": s + 1,
                "ref_aa": ref_aa,
                "derived_aa": derived_aa,
                "hyphaeon_lrt": lrt_val,
                "p_lrt": p_lrt,
                "attribution_norm": norm_a,
                "fg_mean_attn": fg_mean_attn,
                "bg_mean_attn": bg_mean_attn,
                "association_rho": rho,
                "p_value": p_combined,
                "p_assoc": p_assoc,
                "p_assoc_parametric": p_assoc_parametric,
                "p_assoc_perm": p_assoc_perm,
                "score": score,
                "foreground_freq_pct": fg_freq,
                "background_freq_pct": bg_freq
            })

    site_results.sort(key=lambda x: x["score"], reverse=True)

    # 8. Benjamini-Hochberg FDR
    if site_results:
        p_arr = np.array([x["p_value"] for x in site_results])
        q_arr = benjamini_hochberg(p_arr)
        for i, s in enumerate(site_results):
            s["q_value"] = float(q_arr[i])
    
    # 9. Dual-Track Extreme-Value Statistics
    max_assoc = float(site_results[0]["association_rho"]) if site_results else 0.0
    sigma_null = 1.0 / np.sqrt(max(10, N))
    z_single = max_assoc / sigma_null if sigma_null > 0 else 0.0
    p_single = float(2.0 * stats.norm.sf(abs(z_single)))
    p_single = max(1e-15, min(1.0, p_single))
    
    p_evd = float(-np.expm1(-L * p_single))
    p_evd = max(1e-15, min(1.0, p_evd))
    
    score_track_a = float(-np.log10(p_evd))
    score_track_b = float(norm_spectral_ratio)
    dual_track_composite = float(max(score_track_a / 10.0, score_track_b))

    # 10. Extract PARS Signature Bracket
    top_pars_sites = [f"{x['ref_aa']}{x['site']}{x['derived_aa']}" for x in site_results if x["association_rho"] >= 0.40 and x["score"] >= 0.50][:15]
    compact_pars = f"[ {' - '.join(top_pars_sites)} ]" if top_pars_sites else "[]"

    # 11. Directional Epistatic Co-Selection & Sector Mining Across Trait-Associated Sites
    sig_trait_sites = [x for x in site_results if x.get("q_value", 1.0) <= alpha and x["association_rho"] > 0]
    trait_site_indices = [x["site"] - 1 for x in sig_trait_sites]
    
    coselection_pairs = []
    trait_sectors = []
    
    if len(trait_site_indices) >= 2:
        sub_indices = np.array(trait_site_indices, dtype=np.int64)
        sub_A = leaf_attr[sub_indices, :]
        norms = np.linalg.norm(sub_A, axis=1, keepdims=True)
        valid_n = (norms > 1e-12).squeeze()
        
        if np.sum(valid_n) >= 2:
            norm_sub_A = np.zeros_like(sub_A)
            norm_sub_A[valid_n] = sub_A[valid_n] / norms[valid_n]
            C_trait = norm_sub_A @ norm_sub_A.T
            
            K_t = len(sub_indices)
            import networkx as nx
            G_trait = nx.Graph()
            pair_list = []
            
            for i in range(K_t):
                s1 = sub_indices[i]
                for j in range(i + 1, K_t):
                    s2 = sub_indices[j]
                    sim = float(C_trait[i, j])
                    if sim > 0.15:
                        lrt_1 = float(lrts[s1])
                        lrt_2 = float(lrts[s2])
                        cesi = float(sim * np.sqrt(max(0.1, lrt_1) * max(0.1, lrt_2)))
                        
                        df_pair = max(1, N - 2)
                        t_pair = sim * np.sqrt(df_pair / max(1e-15, 1.0 - sim**2))
                        p_pair = float(stats.t.sf(t_pair, df=df_pair))
                        
                        a1_mut = (a_np[s1] < 20) & (a_np[s1] != AA_MAP.get(cons_aas[s1], 20))
                        a2_mut = (a_np[s2] < 20) & (a_np[s2] != AA_MAP.get(cons_aas[s2], 20))
                        shared_branches = int(np.sum(a1_mut & a2_mut))
                        
                        pair_dict = {
                            "site_u": int(s1 + 1),
                            "site_v": int(s2 + 1),
                            "ref_u": cons_aas[s1],
                            "ref_v": cons_aas[s2],
                            "lrt_u": lrt_1,
                            "lrt_v": lrt_2,
                            "similarity": sim,
                            "cesi": cesi,
                            "p_value": p_pair,
                            "shared_branches": shared_branches
                        }
                        pair_list.append(pair_dict)
                        if sim >= 0.25 and cesi >= 1.0:
                            G_trait.add_edge(int(s1 + 1), int(s2 + 1), weight=sim, cesi=cesi)
            
            if pair_list:
                p_arr = np.array([x["p_value"] for x in pair_list])
                q_arr = benjamini_hochberg(p_arr)
                for i, pair in enumerate(pair_list):
                    pair["q_value"] = float(q_arr[i])
                
                pair_list.sort(key=lambda x: x["cesi"], reverse=True)
                coselection_pairs = pair_list
                
            if G_trait.number_of_edges() > 0:
                from .epistasis import extract_epistatic_sectors_tse
                trait_sectors = extract_epistatic_sectors_tse(
                    G_trait, leaf_attr, lrts, cons_aas,
                    min_clique_size=2,
                    min_coherence=0.45,
                    a_np=a_np,
                    taxa=taxa
                )

    return {
        "alignment": alignment_path,
        "tree": tree_path,
        "taxa_count": N,
        "codon_count": L,
        "phenotype_meta": meta,
        "spectral_energy": spectral_energy,
        "norm_spectral_ratio": norm_spectral_ratio,
        "max_assoc": max_assoc,
        "p_evd_length_adjusted": p_evd,
        "score_track_a": score_track_a,
        "score_track_b": score_track_b,
        "dual_track_composite": dual_track_composite,
        "compact_pars_signature": compact_pars,
        "permulations_count": permulations if null_rhos is not None else 0,
        "gene_p_value_perm": gene_p_perm,
        "significant_sites_count": len(sig_trait_sites),
        "coselection_pairs_count": len(coselection_pairs),
        "trait_sectors_count": len(trait_sectors),
        "coselection_pairs": coselection_pairs,
        "trait_sectors": trait_sectors,
        "sites": site_results
    }
