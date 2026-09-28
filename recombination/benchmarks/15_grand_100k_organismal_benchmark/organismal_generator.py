"""
benchmarks/15_grand_100k_organismal_benchmark/organismal_generator.py
=====================================================================
Deterministic, high-performance simulation engine for the 100,000-simulation
Organismal Recombination Benchmark.

Reproducible via SHA-256 seeding:
  seed = hash(scenario.scenario_id, rep_idx)
"""

import hashlib
import random
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pyvolve

try:
    from .organismal_scenarios import OrganismalScenario, get_scenario_for_index
except ImportError:
    from organismal_scenarios import OrganismalScenario, get_scenario_for_index


def get_deterministic_seed(scenario_id: str, rep_idx: int = 0, base_seed: int = 42) -> int:
    """Computes a 31-bit integer seed via SHA-256."""
    token = f"{base_seed}_{scenario_id}_rep{rep_idx}".encode("utf-8")
    digest = hashlib.sha256(token).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def _make_hky_model(kappa: float = 2.5, alpha: Optional[float] = None, num_cats: int = 4, gc: float = 0.5) -> Any:
    """Constructs a pyvolve nucleotide model with HKY parameters."""
    g = gc / 2.0
    a = (1.0 - gc) / 2.0
    freqs = [a, g, g, a] # T, C, A, G
    params = {"parameters": {"kappa": kappa}, "state_freqs": freqs}
    if alpha is not None and alpha > 0:
        return pyvolve.Model("nucleotide", params, alpha=alpha, num_categories=num_cats)
    return pyvolve.Model("nucleotide", params)


def _make_gy94_model(kappa: float = 2.5, omega: float = 0.20, alpha: Optional[float] = None) -> Any:
    """Constructs a pyvolve GY94 codon model."""
    params = {"omega": omega, "kappa": kappa}
    if alpha is not None and alpha > 0:
        return pyvolve.Model("GY", params, alpha=alpha, num_categories=4)
    return pyvolve.Model("GY", params)


def _evolve_partition(length: int, newick_tree: str, model: Any, seed: int) -> Dict[str, str]:
    """Evolves a single genomic partition on a tree."""
    tree = pyvolve.read_tree(tree=newick_tree)
    partition = pyvolve.Partition(models=model, size=length)
    evolver = pyvolve.Evolver(partitions=partition, tree=tree)
    evolver(seed=seed, seqfile=None, ratefile=None, infofile=None)
    raw = evolver.get_sequences()
    return {k: v.upper() for k, v in raw.items()}


def simulate_organismal_replicate(
    scenario: OrganismalScenario,
    rep_idx: int = 0,
    base_seed: int = 42,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Simulates one exact stochastic alignment replicate for the given OrganismalScenario.
    Returns:
        (sequences_dict, metadata_dict)
    """
    seed = get_deterministic_seed(scenario.scenario_id, rep_idx, base_seed)
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    L = scenario.length_nt
    N = scenario.num_taxa
    params = scenario.params
    is_null = scenario.is_null
    mode = scenario.mode
    is_codon = (mode == "codon")
    size_units = (L // 3) if is_codon else L

    kappa = params.get("kappa", 2.5)
    d = params.get("d", 0.05)
    gc = params.get("gc_content", 0.5)
    alpha = params.get("alpha", None)
    omega = params.get("omega", 0.18)

    # 1. Base model
    if is_codon:
        base_model = _make_gy94_model(kappa=kappa, omega=omega, alpha=alpha)
    else:
        base_model = _make_hky_model(kappa=kappa, alpha=alpha, gc=gc)

    sequences: Dict[str, str] = {}
    metadata: Dict[str, Any] = {
        "scenario_id": scenario.scenario_id,
        "archetype": scenario.archetype,
        "category": scenario.category,
        "mode": scenario.mode,
        "is_null": is_null,
        "num_taxa": N,
        "length_nt": L,
        "true_breakpoints": scenario.true_breakpoints,
        "recombinant_taxa": scenario.recombinant_taxa,
        "parent_taxa": scenario.parent_taxa,
        "seed": seed,
    }

    # -------------------------------------------------------------------------
    # CASE A: STRICT TREE-LIKE CLONAL NULLS (H0)
    # -------------------------------------------------------------------------
    if is_null and not params.get("gamma") and not params.get("k_burst") and not params.get("omega_burst") and not params.get("num_convergent_sites"):
        # Standard balanced or star tree
        taxa = [f"P{i+1}" for i in range(N - 1)] + ["O"]
        if N == 4:
            tree_str = f"((P1:{d:.5f}, P2:{d:.5f}):{d:.5f}, P3:{d:.5f}, O:{d*2:.5f});"
        elif N == 8:
            tree_str = f"(((P1:{d:.5f}, P2:{d:.5f}):{d*0.5:.5f}, (P3:{d:.5f}, P4:{d:.5f}):{d*0.5:.5f}):{d*0.5:.5f}, ((P5:{d:.5f}, P6:{d:.5f}):{d*0.5:.5f}, P7:{d:.5f}):{d*0.5:.5f}, O:{d*2:.5f});"
        elif N == 16:
            # 16-taxon tree
            clades = []
            for c in range(4):
                clades.append(f"((P{4*c+1}:{d:.5f}, P{4*c+2}:{d:.5f}):{d*0.5:.5f}, (P{4*c+3}:{d:.5f}, P{4*c+4 if c < 3 else 'O'}:{d:.5f}):{d*0.5:.5f})")
            tree_str = f"(({clades[0]}:{d*0.5:.5f}, {clades[1]}:{d*0.5:.5f}):{d*0.5:.5f}, ({clades[2]}:{d*0.5:.5f}, {clades[3]}:{d*0.5:.5f}):{d*0.5:.5f});"
        else:
            # General N-star tree
            sub_strs = [f"P{i+1}:{d:.5f}" for i in range(N - 1)]
            tree_str = f"({', '.join(sub_strs)}, O:{d*2:.5f});"

        # Check for Felsenstein Saturation Zone
        if params.get("tree_length"):
            T = params["tree_length"]
            b_long = T / 4.0
            b_short = 0.02
            b_int = 0.01
            tree_str = f"((P1:{b_long:.4f}, P2:{b_short:.4f}):{b_int:.4f}, (P3:{b_long:.4f}, O:{b_short:.4f}):{b_int:.4f});"
            alpha = params["alpha"]
            base_model = _make_hky_model(kappa=kappa, alpha=alpha, gc=gc)

        raw = _evolve_partition(size_units, tree_str, base_model, seed)
        return raw, metadata

    # -------------------------------------------------------------------------
    # CASE B: ADVERSARIAL ATTACK 1 - DARREN TRAP COUPLED HETEROTACHY (H0)
    # -------------------------------------------------------------------------
    if is_null and params.get("gamma"):
        gamma = params["gamma"]
        L_half = size_units // 2
        
        # Left half: P1 accelerates by gamma, P2 standard
        tree1 = f"((P1:{d*gamma:.5f}, P2:{d:.5f}):{d:.5f}, P3:{d:.5f}, O:{d*2:.5f});"
        # Right half: P2 accelerates by gamma, P1 standard
        tree2 = f"((P1:{d:.5f}, P2:{d*gamma:.5f}):{d:.5f}, P3:{d:.5f}, O:{d*2:.5f});"

        raw1 = _evolve_partition(L_half, tree1, base_model, seed)
        raw2 = _evolve_partition(size_units - L_half, tree2, base_model, seed + 1000)

        for k in raw1:
            sequences[k] = raw1[k] + raw2[k]
        return sequences, metadata

    # -------------------------------------------------------------------------
    # CASE C: ADVERSARIAL ATTACK 2 - DEAMINASE KATAEGIS / HYPERMUTATION (H0)
    # -------------------------------------------------------------------------
    if is_null and params.get("k_burst"):
        k_burst = params["k_burst"]
        tree_str = f"((P1:{d:.5f}, P2:{d:.5f}):{d:.5f}, R:{d:.5f}, O:{d*2:.5f});"
        raw = _evolve_partition(size_units, tree_str, base_model, seed)
        
        # Inject directional transitions into R in window [1400, 1550]
        seq_r = list(raw["R"])
        w_start, w_end = 1400, 1550
        candidates = [i for i in range(w_start, min(w_end, len(seq_r))) if seq_r[i] == ('C' if params.get('deaminase') == 'APOBEC3G' else 'A')]
        if len(candidates) < k_burst:
            candidates = list(range(w_start, min(w_end, len(seq_r))))
        
        chosen = rng.choice(candidates, size=min(k_burst, len(candidates)), replace=False)
        target_nuc = 'T' if params.get('deaminase') == 'APOBEC3G' else 'G'
        for pos in chosen:
            seq_r[pos] = target_nuc
        raw["R"] = "".join(seq_r)
        return raw, metadata

    # -------------------------------------------------------------------------
    # CASE D: ADVERSARIAL ATTACK 3 - DARWINIAN MIMIC (H0)
    # -------------------------------------------------------------------------
    if is_null and params.get("omega_burst"):
        # Evolve cassette under high omega on lineage R
        omega_burst = params["omega_burst"]
        c_burst = params.get("burst_len_codons", 150)
        c_flank = (size_units - c_burst) // 2
        c_right = size_units - c_flank - c_burst

        tree_pur = f"((P1:{d:.5f}, P2:{d:.5f}):{d:.5f}, R:{d:.5f}, O:{d*2:.5f});"
        tree_burst = f"((P1:{d:.5f}, P2:{d:.5f}):{d:.5f}, R:{d*2.5:.5f}, O:{d*2:.5f});"

        mod_pur = _make_gy94_model(kappa=kappa, omega=params.get("omega_0", 0.05))
        mod_burst = _make_gy94_model(kappa=kappa, omega=omega_burst)

        raw1 = _evolve_partition(c_flank, tree_pur, mod_pur, seed)
        raw2 = _evolve_partition(c_burst, tree_burst, mod_burst, seed + 2000)
        raw3 = _evolve_partition(c_right, tree_pur, mod_pur, seed + 3000)

        for k in raw1:
            sequences[k] = raw1[k] + raw2[k] + raw3[k]
        return sequences, metadata

    # -------------------------------------------------------------------------
    # CASE E: ADVERSARIAL ATTACK 4 - CONVERGENT DRUG RESISTANCE (H0)
    # -------------------------------------------------------------------------
    if is_null and params.get("num_convergent_sites"):
        tree_str = f"((P1:{d:.5f}, P2:{d:.5f}):{d:.5f}, R:{d:.5f}, O:{d*2:.5f});"
        raw = _evolve_partition(size_units, tree_str, base_model, seed)
        n_conv = params["num_convergent_sites"]
        
        # Inject convergent mutations at identical sites in P1 and R
        seq_p1 = list(raw["P1"])
        seq_r = list(raw["R"])
        site_pool = rng.choice(range(50, len(seq_p1) - 50), size=n_conv, replace=False)
        for s in site_pool:
            curr = seq_p1[s]
            mut = rng.choice([b for b in "ACGT" if b != curr])
            seq_p1[s] = mut
            seq_r[s] = mut
        raw["P1"] = "".join(seq_p1)
        raw["R"] = "".join(seq_r)
        return raw, metadata

    # -------------------------------------------------------------------------
    # CASE F: AUTHENTIC RECOMBINATION (H1) - SINGLE CROSSOVER, CASSETTE, NESTED
    # -------------------------------------------------------------------------
    bps = scenario.true_breakpoints
    if len(bps) == 1:
        bp = bps[0]
        bp_units = (bp // 3) if is_codon else bp
        L1 = bp_units
        L2 = size_units - bp_units

        is_ghost = params.get("is_ghost", False)
        if is_ghost:
            tree1 = f"((P1:{d:.5f}, R:{d:.5f}):{d*0.5:.5f}, P2:{d*2:.5f}, (Ghost_G:{d*2:.5f}, O:{d*3:.5f}):{d:.5f});"
            tree2 = f"(P1:{d*2:.5f}, P2:{d*2:.5f}, ((Ghost_G:{d:.5f}, R:{d:.5f}):{d*0.5:.5f}, O:{d*3:.5f}):{d:.5f});"
        else:
            if N == 4:
                tree1 = f"((P1:{d:.5f}, R:{d:.5f}):{d*0.5:.5f}, P2:{d*2:.5f}, O:{d*3:.5f});"
                tree2 = f"(P1:{d*2:.5f}, (P2:{d:.5f}, R:{d:.5f}):{d*0.5:.5f}, O:{d*3:.5f});"
            else:
                tree1 = f"(((P1:{d:.5f}, R:{d:.5f}):{d*0.5:.5f}, P2:{d*1.5:.5f}):{d*0.5:.5f}, (P3:{d:.5f}, P4:{d:.5f}):{d*0.5:.5f}, O:{d*2:.5f});"
                tree2 = f"(((P2:{d:.5f}, R:{d:.5f}):{d*0.5:.5f}, P1:{d*1.5:.5f}):{d*0.5:.5f}, (P3:{d:.5f}, P4:{d:.5f}):{d*0.5:.5f}, O:{d*2:.5f});"

        raw1 = _evolve_partition(L1, tree1, base_model, seed)
        raw2 = _evolve_partition(L2, tree2, base_model, seed + 4000)

        for k in raw1:
            if is_ghost and k == "Ghost_G":
                continue # Omit ghost lineage from final FASTA matrix
            sequences[k] = raw1[k] + raw2[k]
        return sequences, metadata

    elif len(bps) == 2:
        bp1, bp2 = bps
        bp1_u = (bp1 // 3) if is_codon else bp1
        bp2_u = (bp2 // 3) if is_codon else bp2
        L1 = bp1_u
        L2 = bp2_u - bp1_u
        L3 = size_units - bp2_u

        # Check if Nested Reticulation (Attack 6)
        if scenario.archetype == "hiv_second_gen_crf":
            # R1: rec(P1, P2) at bp1; R2: rec(R1, P3) at bp2
            # Part 1 [0, bp1): R1 with P1, R2 with P1
            t1 = f"(((P1:{d:.5f}, R1:{d:.5f}):0.005, R2:{d:.5f}):0.005, P2:{d*2:.5f}, P3:{d*2:.5f}, O:{d*3:.5f});"
            # Part 2 [bp1, bp2): R1 with P2, R2 with P2
            t2 = f"(P1:{d*2:.5f}, ((P2:{d:.5f}, R1:{d:.5f}):0.005, R2:{d:.5f}):0.005, P3:{d*2:.5f}, O:{d*3:.5f});"
            # Part 3 [bp2, L): R1 with P2, R2 with P3
            t3 = f"(P1:{d*2:.5f}, (P2:{d:.5f}, R1:{d:.5f}):0.005, (P3:{d:.5f}, R2:{d:.5f}):0.005, O:{d*3:.5f});"

            raw1 = _evolve_partition(L1, t1, base_model, seed)
            raw2 = _evolve_partition(L2, t2, base_model, seed + 5000)
            raw3 = _evolve_partition(L3, t3, base_model, seed + 6000)
            for k in raw1:
                sequences[k] = raw1[k] + raw2[k] + raw3[k]
            return sequences, metadata

        else:
            # Standard Cassette insertion: [bp1, bp2] from P2 into P1 background
            tree_bg = f"((P1:{d:.5f}, R:{d:.5f}):{d*0.5:.5f}, P2:{d*2:.5f}, O:{d*3:.5f});"
            tree_ins = f"(P1:{d*2:.5f}, (P2:{d:.5f}, R:{d:.5f}):{d*0.5:.5f}, O:{d*3:.5f});"

            raw1 = _evolve_partition(L1, tree_bg, base_model, seed)
            raw2 = _evolve_partition(L2, tree_ins, base_model, seed + 7000)
            raw3 = _evolve_partition(L3, tree_bg, base_model, seed + 8000)

            for k in raw1:
                sequences[k] = raw1[k] + raw2[k] + raw3[k]
            return sequences, metadata

    raise ValueError(f"Unsupported breakpoint topology: {bps}")
