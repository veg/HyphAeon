"""
Mode I baseline: raw binary substitution counting (phylogeny-blind).

Extracted from git commit 8b5b6c9 (pre-neural-attribution version of
axomeme/phenotype.py and axomeme/epistasis.py). This is the
"Mode I: Raw Mutation Baseline" — the approach PhyloWAS was designed
to replace.

Mode I projects a binary {0,1}^{L×N} substitution matrix onto a phenotype
vector using cosine similarity and a Poisson test. It does not use a
phylogenetic tree, neural model, or attention weights. Sister taxa
sharing substitutions by common descent create spurious associations.

This module exists solely as a baseline for comparison against Mode II
(neural transformer attributions) in model_eval tests. It is NOT part
of the package's public API and should not be imported by production code.

Dependencies: numpy, scipy, pandas, biopython (for parse_alignment_sequences).
No neural model weights required.
"""
import os
import fnmatch
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
from scipy.stats import poisson, norm
import networkx as nx

from hyphaeon.dataset import (
    AA_MAP,
    CODON_TO_AA,
    parse_alignment_sequences,
)

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


def _resolve_phenotype_vector(
    taxa: List[str],
    preset: Optional[str] = None,
    foreground: Optional[Union[str, List[str]]] = None,
    background: Optional[Union[str, List[str]]] = None,
    phenotype_file: Optional[str] = None,
    trait_col: Optional[str] = None,
    species_col: Optional[str] = None,
    continuous: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Constructs the phenotypic trait vector y in R^N across all N taxa."""
    N = len(taxa)
    y = np.zeros(N, dtype=float)
    meta = {
        "mode": "discrete",
        "foreground_count": 0,
        "background_count": 0,
        "description": ""
    }

    if phenotype_file:
        if not os.path.exists(phenotype_file):
            raise FileNotFoundError(f"Phenotype file not found: {phenotype_file}")

        sep = "\t" if phenotype_file.endswith((".tsv", ".tab")) else ","
        df_pheno = pd.read_csv(phenotype_file, sep=sep)

        if not species_col:
            for c in ["species", "taxon", "taxa", "tree_leaf_name", "assembly", "id", "name", "species_name"]:
                match = [col for col in df_pheno.columns if col.lower() == c]
                if match:
                    species_col = match[0]
                    break
            if not species_col:
                species_col = df_pheno.columns[0]

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

        matched_values = []
        for i, t in enumerate(taxa):
            val = None
            if t in trait_dict:
                val = trait_dict[t]
            elif t.lower() in trait_dict:
                val = trait_dict[t.lower()]
            else:
                for k, v in trait_dict.items():
                    if k in t or t in k:
                        val = v
                        break

            if val is not None:
                try:
                    num_val = float(val)
                    y[i] = num_val
                    matched_values.append(num_val)
                except ValueError:
                    y[i] = 1.0 if str(val).lower() in ["1", "true", "yes", "case", "foreground", "target", "positive"] else 0.0
                    matched_values.append(y[i])

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


def run_phenotype_association_mode_i(
    alignment_path: str,
    preset: Optional[str] = None,
    foreground: Optional[Union[str, List[str]]] = None,
    background: Optional[Union[str, List[str]]] = None,
    phenotype_file: Optional[str] = None,
    trait_col: Optional[str] = None,
    species_col: Optional[str] = None,
    continuous: bool = False,
    min_taxa_per_site: int = 10,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """Mode I PhyloWAS: binary substitution counting + Poisson test.

    No neural model, no tree, no weights. Phylogeny-blind by construction.
    """
    seq_dict = parse_alignment_sequences(alignment_path)
    if not seq_dict:
        raise ValueError(f"Could not parse sequences from {alignment_path}")
    taxa = list(seq_dict.keys())
    seqs = list(seq_dict.values())
    N = len(taxa)
    L = len(seqs[0]) // 3

    y, meta = _resolve_phenotype_vector(
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

    aa_mat = np.zeros((N, L), dtype=int)
    for i in range(N):
        s_seq = seqs[i]
        for s in range(L):
            codon = s_seq[s*3:s*3+3].upper()
            aa = CODON_TO_AA.get(codon, '-')
            aa_mat[i, s] = AA_MAP.get(aa, 20)

    bg_consensus = np.zeros(L, dtype=int)
    for s in range(L):
        valid_bg = aa_mat[~is_fg, s][aa_mat[~is_fg, s] < 20] if fg_count > 0 else aa_mat[:, s][aa_mat[:, s] < 20]
        bg_consensus[s] = np.argmax(np.bincount(valid_bg)) if len(valid_bg) > 0 else 20

    B = np.zeros((L, N), dtype=float)
    for s in range(L):
        if bg_consensus[s] < 20:
            for i in range(N):
                if aa_mat[i, s] < 20 and aa_mat[i, s] != bg_consensus[s]:
                    B[s, i] = 1.0

    y_norm = y / np.linalg.norm(y) if np.linalg.norm(y) > 0 else y
    proj = B @ y_norm
    spectral_energy = float(np.linalg.norm(proj))
    frob_norm = float(np.linalg.norm(B))
    norm_spectral_ratio = spectral_energy / frob_norm if frob_norm > 0 else 0.0

    site_results = []
    for s in range(L):
        if bg_consensus[s] < 20:
            valid_mask = (aa_mat[:, s] < 20)
            tot_mut = int(np.sum(B[s, :]))
            if np.sum(valid_mask) >= min_taxa_per_site and tot_mut > 0:
                b_sub = B[s, valid_mask]
                y_sub = y[valid_mask]
                b_norm = b_sub / np.linalg.norm(b_sub) if np.linalg.norm(b_sub) > 0 else b_sub
                y_sub_norm = y_sub / np.linalg.norm(y_sub) if np.linalg.norm(y_sub) > 0 else y_sub
                assoc = float(b_norm @ y_sub_norm)

                shared = int(np.sum((B[s, :] > 0) & is_fg))
                N_valid = int(np.sum(valid_mask))
                N_fg_valid = int(np.sum(valid_mask & is_fg))
                exp_shared = (tot_mut * N_fg_valid) / N_valid if N_valid > 0 else 0.0
                p_val = 1.0 - poisson.cdf(shared - 1, exp_shared) if exp_shared > 0 else 1.0

                ref_aa = REV_AA_MAP.get(bg_consensus[s], '-')
                fg_valid_aa = aa_mat[is_fg, s][aa_mat[is_fg, s] < 20]
                derived_aa = REV_AA_MAP.get(np.argmax(np.bincount(fg_valid_aa)), '-') if len(fg_valid_aa) > 0 else ref_aa
                fg_freq = (np.sum(fg_valid_aa == AA_MAP.get(derived_aa, 20)) / len(fg_valid_aa) * 100.0) if len(fg_valid_aa) > 0 else 0.0
                bg_valid_aa = aa_mat[~is_fg, s][aa_mat[~is_fg, s] < 20]
                bg_freq = (np.sum(bg_valid_aa == AA_MAP.get(derived_aa, 20)) / len(bg_valid_aa) * 100.0) if len(bg_valid_aa) > 0 else 0.0

                site_results.append({
                    "site": s + 1,
                    "ref_aa": ref_aa,
                    "derived_aa": derived_aa,
                    "total_mutations": tot_mut,
                    "shared_foreground_mutations": shared,
                    "expected_shared": float(exp_shared),
                    "association_rho": float(assoc),
                    "p_value": float(p_val),
                    "foreground_freq_pct": float(fg_freq),
                    "background_freq_pct": float(bg_freq)
                })

    site_results.sort(key=lambda x: x["association_rho"], reverse=True)

    m = len(site_results)
    if m > 0:
        p_sorted_idx = np.argsort([x["p_value"] for x in site_results])
        min_q = 1.0
        for rank, idx in reversed(list(enumerate(p_sorted_idx))):
            p_val = site_results[idx]["p_value"]
            q_val = (p_val * m) / (rank + 1)
            if q_val < min_q:
                min_q = q_val
            site_results[idx]["q_value"] = min(min_q, 1.0)

    max_assoc = float(site_results[0]["association_rho"]) if site_results else 0.0
    sigma_null = 1.0 / np.sqrt(max(10, N))
    z_single = max_assoc / sigma_null if sigma_null > 0 else 0.0
    p_single = float(2.0 * norm.sf(abs(z_single)))
    p_single = max(1e-15, min(1.0, p_single))

    p_evd = float(-np.expm1(-L * p_single))
    p_evd = max(1e-15, min(1.0, p_evd))

    score_track_a = float(-np.log10(p_evd))
    score_track_b = float(norm_spectral_ratio)
    dual_track_composite = float(max(score_track_a / 10.0, score_track_b))

    top_pars_sites = [f"{x['ref_aa']}{x['site']}{x['derived_aa']}" for x in site_results if x["association_rho"] >= 0.50][:15]
    compact_pars = f"[ {' - '.join(top_pars_sites)} ]" if top_pars_sites else "[]"

    return {
        "alignment": alignment_path,
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


def run_epistatic_sector_mining_mode_i(
    alignment_path: str,
    tree_path: Optional[str] = None,
    min_clique_size: int = 3,
    min_sim: float = 0.50,
    max_p_pair: float = 0.05,
    min_mutations: int = 2,
    max_sectors: int = 15
) -> Dict[str, Any]:
    """Mode I ESSM: binary substitution cosine similarity + Poisson test + clique finding.

    No neural model, no tree, no weights. Phylogeny-blind by construction.
    The tree_path parameter is accepted but ignored — Mode I does not use the tree.
    """
    seq_dict = parse_alignment_sequences(alignment_path)
    if not seq_dict:
        raise ValueError(f"Could not parse sequences from {alignment_path}")
    taxa = list(seq_dict.keys())
    seqs = list(seq_dict.values())
    N = len(taxa)
    L = len(seqs[0]) // 3

    aa_mat = np.zeros((N, L), dtype=int)
    for i in range(N):
        s_seq = seqs[i]
        for s in range(L):
            codon = s_seq[s*3:s*3+3].upper()
            aa = CODON_TO_AA.get(codon, '-')
            aa_mat[i, s] = AA_MAP.get(aa, 20)

    consensus = np.zeros(L, dtype=int)
    for s in range(L):
        valid = aa_mat[:, s][aa_mat[:, s] < 20]
        consensus[s] = np.argmax(np.bincount(valid)) if len(valid) > 0 else 20

    B = np.zeros((L, N), dtype=float)
    for s in range(L):
        if consensus[s] < 20:
            for i in range(N):
                if aa_mat[i, s] < 20 and aa_mat[i, s] != consensus[s]:
                    B[s, i] = 1.0

    active_sites = [s for s in range(L) if np.sum(B[s, :]) >= min_mutations]
    if len(active_sites) < min_clique_size:
        return {
            "alignment": alignment_path,
            "taxa_count": N,
            "codon_count": L,
            "active_sites_count": len(active_sites),
            "sectors": [],
            "edges": []
        }

    G = nx.Graph()
    for s in active_sites:
        G.add_node(s + 1, ref=REV_AA_MAP.get(consensus[s], '-'))

    edge_list = []
    for i_idx, s1 in enumerate(active_sites):
        b1 = B[s1, :]
        n1 = np.linalg.norm(b1)
        if n1 == 0:
            continue
        for s2 in active_sites[i_idx + 1:]:
            b2 = B[s2, :]
            n2 = np.linalg.norm(b2)
            if n2 == 0:
                continue

            sim = float((b1 @ b2) / (n1 * n2))
            shared = int(np.sum((b1 > 0) & (b2 > 0)))
            k1, k2 = int(np.sum(b1 > 0)), int(np.sum(b2 > 0))
            exp_shared = (k1 * k2) / N
            p_val = 1.0 - poisson.cdf(shared - 1, exp_shared) if exp_shared > 0 else 1.0

            if p_val <= max_p_pair and sim >= min_sim:
                G.add_edge(s1 + 1, s2 + 1, weight=sim, p_value=p_val, shared=shared)
                edge_list.append({
                    "site_u": s1 + 1,
                    "site_v": s2 + 1,
                    "ref_u": REV_AA_MAP.get(consensus[s1], '-'),
                    "ref_v": REV_AA_MAP.get(consensus[s2], '-'),
                    "similarity": sim,
                    "p_value": float(p_val),
                    "shared_taxa": shared
                })

    raw_cliques = list(nx.find_cliques(G))
    cliques = [c for c in raw_cliques if len(c) >= min_clique_size]
    cliques.sort(key=lambda c: len(c), reverse=True)

    sectors = []
    seen_subsets = set()

    for c_idx, clq in enumerate(cliques[:max_sectors]):
        clq_sorted = sorted(clq)
        clq_key = tuple(clq_sorted)
        if clq_key in seen_subsets:
            continue
        seen_subsets.add(clq_key)

        site_0based = [s - 1 for s in clq_sorted]
        sub_B = B[site_0based, :]
        cov = sub_B @ sub_B.T
        eigvals = np.linalg.eigvalsh(cov)
        coherence = float(eigvals[-1] / (np.sum(eigvals) + 1e-15))

        u = np.min(sub_B, axis=0)
        simul_taxa = int(np.sum(u > 0))
        exp_simul = N * np.prod([np.sum(B[s, :] > 0) / N for s in site_0based])
        p_multi = float(1.0 - poisson.cdf(simul_taxa - 1, exp_simul) if exp_simul > 0 else 1.0)

        pars_elements = []
        for s in site_0based:
            ref = REV_AA_MAP.get(consensus[s], '-')
            mut_taxa = np.where(B[s, :] > 0)[0]
            mut_aas = [REV_AA_MAP.get(aa_mat[i, s], '-') for i in mut_taxa if aa_mat[i, s] < 20]
            derived = max(set(mut_aas), key=mut_aas.count) if mut_aas else ref
            pars_elements.append(f"{ref}{s+1}{derived}")

        compact_pars = f"[ {' - '.join(pars_elements)} ]"

        sectors.append({
            "sector_id": len(sectors) + 1,
            "size": len(clq_sorted),
            "sites": clq_sorted,
            "coherence": coherence,
            "simultaneous_taxa": simul_taxa,
            "expected_simultaneous": float(exp_simul),
            "multi_way_p_value": p_multi,
            "pars_signature": compact_pars
        })

    return {
        "alignment": alignment_path,
        "taxa_count": N,
        "codon_count": L,
        "active_sites_count": len(active_sites),
        "coevolution_edges_count": len(edge_list),
        "sectors_discovered": len(sectors),
        "sectors": sectors,
        "edges": edge_list
    }


def mode_i_phylowas_pvals(result):
    """Extract p-values from Mode I PhyloWAS result dict."""
    sites = result.get("sites", [])
    return np.array([s.get("p_value", 1.0) for s in sites], dtype=np.float64)


def mode_i_essm_edge_pvals(result):
    """Extract edge p-values from Mode I ESSM result dict."""
    edges = result.get("edges", [])
    return np.array([e.get("p_value", 1.0) for e in edges], dtype=np.float64)
