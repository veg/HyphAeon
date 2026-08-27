"""
axomeme/phenotype.py
--------------------
Directional Phenotype-Genotype Association Mapping (PhyloWAS) and
Phenotype-Associated Residue Signature (PARS) extraction.

Uses continuous Transformer Attribution Vectors (root-to-leaf multi-head
axial attention weights combined with phylogenetic branch projections)
to discover convergent and directional trait-associated molecular adaptations.
"""

import os
import fnmatch
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from Bio import Phylo

from .dataset import (
    AA_MAP,
    load_alignment_and_tree,
    extract_tree_from_string_or_file
)
from .inference import compute_transformer_attributions, load_model
from .utils import REV_AA_MAP, benjamini_hochberg

DEFAULT_WEIGHTS = "weights/axomeme_v1.pt"

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
            fg_list = [p.strip() for p in foreground.split(",") if p.strip()]
        else:
            fg_list = foreground

        for i, t in enumerate(taxa):
            for pat in fg_list:
                if fnmatch.fnmatch(t.lower(), pat.lower()) or (pat.lower() in t.lower()):
                    y[i] = 1.0
                    break

        meta["mode"] = "discrete"
        meta["foreground_count"] = int(np.sum(y > 0))
        meta["background_count"] = int(np.sum(y <= 0))
        meta["description"] = f"User-specified foreground patterns: {fg_list}"
        return y, meta

    raise ValueError("Must provide one of --preset, --phenotype-file, or --foreground.")

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
    min_taxa_per_site: int = 4,
    alpha: float = 0.05,
    cpu: bool = False,
    batch_size: int = 64
) -> Dict[str, Any]:
    """
    Executes directional Phenotype-Genotype association (PhyloWAS) on a codon alignment
    using continuous Transformer Attribution Vectors (multi-head phylogenetic attention
    attributions and branch projections) rather than binary string substitution counts.
    """
    # 1. Device & Model
    model, device = load_model(weights=weights_path, variant=variant, cpu=cpu)

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

    d_dev = d_mat.to(device)
    z_dev = z_coords.to(device)
    tree_cache = model.precompute_tree_cache(d_dev, z_dev)

    # 5. Extract Transformer Phylogenetic Attributions
    leaf_attr, lrts, pvals, cons_aas = compute_transformer_attributions(
        model, c_tensor, a_tensor, tree_cache, taxa, device, batch_size=batch_size
    )

    # 6. Directional Unit-Hypersphere Attribution Projection
    y_norm = y / np.linalg.norm(y) if np.linalg.norm(y) > 0 else y
    proj = leaf_attr @ y_norm # [L]
    spectral_energy = float(np.linalg.norm(proj))
    frob_norm = float(np.linalg.norm(leaf_attr))
    norm_spectral_ratio = float(spectral_energy / frob_norm) if frob_norm > 0 else 0.0

    a_np = a_tensor.squeeze(-1).numpy() # [L, N] amino acid token matrix

    # 7. Site-Level Transformer Attribution Associations
    site_results = []
    for s in range(L):
        a_s = leaf_attr[s, :] # [N] continuous transformer attribution vector
        norm_a = float(np.linalg.norm(a_s))
        
        valid_mask = (a_np[s] < 20)
        N_valid = int(np.sum(valid_mask))
        
        if norm_a > 0 and N_valid >= min_taxa_per_site:
            rho = float(np.dot(a_s, y) / (norm_a * np.linalg.norm(y) + 1e-15))
            
            # Continuous Attribution Student's t-statistic
            df = max(1, N_valid - 2)
            t_stat = rho * np.sqrt(df / max(1e-15, 1.0 - rho**2))
            p_assoc = float(stats.t.sf(t_stat, df=df))
            
            lrt_val = float(lrts[s])
            p_lrt = float(pvals[s])
            score = float(np.sqrt(max(0.0, lrt_val)) * max(0.0, rho))
            
            # ACAT Cauchy combination: combines omnibus selection LRT + directional trait attribution
            c_p = 0.5 * (np.tan((0.5 - p_lrt) * np.pi) + np.tan((0.5 - p_assoc) * np.pi))
            p_combined = float(0.5 - np.arctan(c_p) / np.pi)
            p_combined = max(1e-15, min(1.0, p_combined))

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
                "axomeme_lrt": lrt_val,
                "p_lrt": p_lrt,
                "attribution_norm": norm_a,
                "fg_mean_attn": fg_mean_attn,
                "bg_mean_attn": bg_mean_attn,
                "association_rho": rho,
                "p_value": p_combined,
                "p_assoc": p_assoc,
                "score": score,
                "foreground_freq_pct": fg_freq,
                "background_freq_pct": bg_freq
            })

    site_results.sort(key=lambda x: x["score"], reverse=True)

    # 8. Benjamini-Hochberg FDR
    m = len(site_results)
    if m > 0:
        p_arr = np.array([x["p_value"] for x in site_results], dtype=np.float32)
        q_arr = benjamini_hochberg(p_arr)
        for idx, q_val in enumerate(q_arr):
            site_results[idx]["q_value"] = float(q_val)
    
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
        "significant_sites_count": len([x for x in site_results if x.get("q_value", 1.0) <= alpha]),
        "sites": site_results
    }
