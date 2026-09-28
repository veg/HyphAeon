"""
benchmarks/15_grand_100k_organismal_benchmark/organismal_scenarios.py
====================================================================
Declarative specification and deterministic registry for the 100,000-simulation
Organismal Recombination Benchmark:

Partitioning:
  • 50% Standard Public Surveillance Regimes (50,000 simulations)
      - Archetype 1: Positive-Sense ssRNA Viruses (HIV-1, SARS-CoV-2, Norovirus, HCV) [20,000]
      - Archetype 2: Negative-Sense ssRNA Intragenic Nulls (Influenza A PA/PB2, Ebola) [10,000]
      - Archetype 3: Bacterial Natural Competence (S. pneumoniae cps & pbp2x, N. meningitidis) [12,000]
      - Archetype 4: Strictly Clonal Bacteria Nulls (M. tuberculosis complex) [4,000]
      - Archetype 5: Eukaryotic Organellar & Paralogous Copies (Human mtDNA, Primate gamma-globin) [4,000]

  • 50% 'Hell Week' Adversarial Torture Regimes (50,000 simulations)
      - Attack 1: Host-Shift Darren Trap Heterotachy Waves (Flu host shifts, gamma=1.5..10) [10,000]
      - Attack 2: Host Deaminase Kataegis / Hypermutation Showers (APOBEC/ADAR bursts, k=6..48) [9,000]
      - Attack 3: Episodic Positive Selection Darwinian Mimics (omega=2.0..15.0 in env/HA) [8,000]
      - Attack 4: Convergent Drug-Resistance Hotspot Traps (M. tuberculosis MDR mimics) [6,000]
      - Attack 5: Sub-Window Micro-Conversion Limits (RecA tracts 20..120 nt) [8,000]
      - Attack 6: Second-Generation CRFs / Nested Reticulations (recombinant of recombinant) [5,000]
      - Attack 7: Felsenstein Saturation Zones (T=4.0..10.0, extreme Gamma alpha <= 0.12) [4,000]

Total: Exactly 100,000 simulations.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import hashlib


@dataclass
class OrganismalScenario:
    scenario_id: str
    archetype: str          # e.g., 'hiv1', 'sars2', 'norovirus', 'flu_pa', 'spneumoniae', 'mtbc', 'mtdna'
    category: str           # 'standard' or 'adversarial'
    mode: str               # 'codon' or 'nt'
    is_null: bool           # True for H0, False for H1
    num_taxa: int
    length_nt: int
    true_breakpoints: List[int]
    recombinant_taxa: List[str]
    parent_taxa: Dict[str, List[str]]
    params: Dict[str, Any]
    description: str


def get_scenario_for_index(global_idx: int) -> OrganismalScenario:
    """
    Deterministically computes the exact scenario configuration for any index in [0, 99999].
    Guarantees O(1) memory mapping for distributed Slurm array workers.
    """
    if not (0 <= global_idx < 100000):
        raise IndexError(f"global_idx {global_idx} out of bounds [0, 99999]")

    is_adversarial = (global_idx >= 50000)
    category = "adversarial" if is_adversarial else "standard"
    local_idx = global_idx - 50000 if is_adversarial else global_idx

    # Seed for deterministic parameter sampling
    seed_hash = int(hashlib.sha256(f"organismal_{global_idx}".encode()).hexdigest()[:8], 16)
    
    # -------------------------------------------------------------------------
    # PART 1: STANDARD PRACTICE REGIMES (Indices 0 .. 49999)
    # -------------------------------------------------------------------------
    if not is_adversarial:
        # Sub-block 1: Positive-Sense RNA Viruses (0 .. 19999) [20,000 runs]
        if local_idx < 20000:
            sub = local_idx
            if sub < 8000:
                # HIV-1 Archetype: L=9180 nt (3060 codons) or L=3000 nt
                is_full = (sub % 2 == 0)
                L_nt = 9180 if is_full else 3000
                N = 4 if sub < 4000 else (8 if sub < 6500 else 16)
                is_null = (sub % 5 == 0) # 20% null controls within archetype
                is_ghost = (not is_null and (sub % 4 == 1))
                d = 0.03 + (sub % 50) * 0.002
                
                taxa = [f"P{i+1}" for i in range(N - (2 if not is_null else 1))] + (["R"] if not is_null else []) + ["O"]
                true_bps = [] if is_null else ([1500] if not is_full else [2400, 6800])
                recs = ["R"] if not is_null else []
                parents = {"R": ["P1", "Ghost_G" if is_ghost else "P2"]} if not is_null else {}

                return OrganismalScenario(
                    scenario_id=f"std_hiv1_{global_idx}",
                    archetype="hiv1",
                    category=category,
                    mode="codon" if (sub % 3 != 0) else "nt",
                    is_null=is_null,
                    num_taxa=N,
                    length_nt=L_nt,
                    true_breakpoints=true_bps,
                    recombinant_taxa=recs,
                    parent_taxa=parents,
                    params={
                        "d": d, "omega": 0.18, "kappa": 4.5, "is_ghost": is_ghost,
                        "gc_content": 0.42, "model": "GY94" if (sub % 3 != 0) else "HKY85"
                    },
                    description=f"HIV-1 {'Null' if is_null else 'Mosaic'} (N={N}, L={L_nt}, d={d:.3f})"
                )

            elif sub < 14000:
                # SARS-CoV-2 Archetype: L=29903 nt, low divergence (d=0.001 .. 0.006)
                s_sub = sub - 8000
                N = 4 if s_sub < 3000 else (8 if s_sub < 5000 else 16)
                is_null = (s_sub % 4 == 0)
                d = 0.001 + (s_sub % 50) * 0.0001
                taxa = [f"P{i+1}" for i in range(N - (2 if not is_null else 1))] + (["R"] if not is_null else []) + ["O"]
                true_bps = [] if is_null else [21500] # Spike NTD/RBD boundary
                recs = ["R"] if not is_null else []
                parents = {"R": ["P1", "P2"]} if not is_null else {}

                return OrganismalScenario(
                    scenario_id=f"std_sars2_{global_idx}",
                    archetype="sars2",
                    category=category,
                    mode="nt",
                    is_null=is_null,
                    num_taxa=N,
                    length_nt=29903,
                    true_breakpoints=true_bps,
                    recombinant_taxa=recs,
                    parent_taxa=parents,
                    params={"d": d, "c_to_t_bias": 4.0, "kappa": 3.0, "gc_content": 0.38},
                    description=f"SARS-CoV-2 Whole Genome {'Null' if is_null else 'XBB-mimic'} (N={N}, L=29903, d={d:.4f})"
                )

            elif sub < 17000:
                # Norovirus Archetype: L=7500 nt, ORF1/ORF2 junction hotspot (~5050 nt)
                s_sub = sub - 14000
                N = 4 if s_sub < 1500 else 8
                is_null = (s_sub % 4 == 0)
                d = 0.04 + (s_sub % 40) * 0.002
                true_bps = [] if is_null else [5050]
                recs = ["R"] if not is_null else []
                parents = {"R": ["P1", "P2"]} if not is_null else {}

                return OrganismalScenario(
                    scenario_id=f"std_norovirus_{global_idx}",
                    archetype="norovirus",
                    category=category,
                    mode="nt",
                    is_null=is_null,
                    num_taxa=N,
                    length_nt=7500,
                    true_breakpoints=true_bps,
                    recombinant_taxa=recs,
                    parent_taxa=parents,
                    params={"d": d, "kappa": 2.8, "hotspot_coord": 5050},
                    description=f"Norovirus Polyprotein {'Null' if is_null else 'ORF1/2 Crossover'} (N={N}, L=7500, d={d:.3f})"
                )

            else:
                # HCV Archetype: L=9600 nt
                s_sub = sub - 17000
                N = 4 if s_sub < 1500 else 8
                is_null = (s_sub % 4 == 0)
                d = 0.05 + (s_sub % 40) * 0.0025
                true_bps = [] if is_null else [3400]
                recs = ["R"] if not is_null else []
                parents = {"R": ["P1", "P2"]} if not is_null else {}

                return OrganismalScenario(
                    scenario_id=f"std_hcv_{global_idx}",
                    archetype="hcv",
                    category=category,
                    mode="nt",
                    is_null=is_null,
                    num_taxa=N,
                    length_nt=9600,
                    true_breakpoints=true_bps,
                    recombinant_taxa=recs,
                    parent_taxa=parents,
                    params={"d": d, "kappa": 3.2},
                    description=f"HCV Genotype Mosaic {'Null' if is_null else 'NS2 Crossover'} (N={N}, L=9600, d={d:.3f})"
                )

        # Sub-block 2: Negative-Sense RNA Intragenic Nulls (20000 .. 29999) [10,000 runs]
        elif local_idx < 30000:
            sub = local_idx - 20000
            if sub < 7000:
                # Influenza A Single Segments (PA L=2151 nt coding, PB2 L=2280 nt)
                is_pa = (sub % 2 == 0)
                L_nt = 2151 if is_pa else 2280
                N = 4 if sub < 3500 else (8 if sub < 5500 else 16)
                d = 0.02 + (sub % 50) * 0.002
                alpha = 0.4 + (sub % 30) * 0.03
                
                return OrganismalScenario(
                    scenario_id=f"std_flu_{global_idx}",
                    archetype="flu_single_segment",
                    category=category,
                    mode="codon" if (sub % 2 == 0) else "nt",
                    is_null=True, # STRICT CLONAL NULL
                    num_taxa=N,
                    length_nt=L_nt,
                    true_breakpoints=[],
                    recombinant_taxa=[],
                    parent_taxa={},
                    params={"d": d, "alpha": alpha, "segment": "PA" if is_pa else "PB2", "kappa": 3.8},
                    description=f"Influenza A {'PA' if is_pa else 'PB2'} Segment Clonal Null (N={N}, L={L_nt}, d={d:.3f}, alpha={alpha:.2f})"
                )
            else:
                # Filovirus (Ebola L=6000 nt or L=18960 nt)
                L_nt = 6000 if (sub % 3 != 0) else 18960
                N = 4 if sub < 8500 else 8
                d = 0.005 + (sub % 40) * 0.0005
                return OrganismalScenario(
                    scenario_id=f"std_ebola_{global_idx}",
                    archetype="ebola",
                    category=category,
                    mode="nt",
                    is_null=True, # STRICT CLONAL NULL
                    num_taxa=N,
                    length_nt=L_nt,
                    true_breakpoints=[],
                    recombinant_taxa=[],
                    parent_taxa={},
                    params={"d": d, "kappa": 2.5},
                    description=f"Ebola Clonal Monocistronic Null (N={N}, L={L_nt}, d={d:.4f})"
                )

        # Sub-block 3: Bacterial Natural Competence & Transformation (30000 .. 41999) [12,000 runs]
        elif local_idx < 42000:
            sub = local_idx - 30000
            if sub < 6000:
                # S. pneumoniae cps Capsular Switch Macro-Recombination (L=15000 bp)
                N = 4 if sub < 3000 else (8 if sub < 5000 else 16)
                is_null = (sub % 4 == 0)
                d = 0.02 + (sub % 40) * 0.0015
                true_bps = [] if is_null else [4000, 11000] # 7 kb cassette
                recs = ["R"] if not is_null else []
                parents = {"R": ["P1", "P2"]} if not is_null else {}

                return OrganismalScenario(
                    scenario_id=f"std_spneumo_cps_{global_idx}",
                    archetype="spneumoniae_cps",
                    category=category,
                    mode="nt",
                    is_null=is_null,
                    num_taxa=N,
                    length_nt=15000,
                    true_breakpoints=true_bps,
                    recombinant_taxa=recs,
                    parent_taxa=parents,
                    params={"d": d, "tract_length": 7000, "kappa": 2.2},
                    description=f"S. pneumoniae cps Macro-Cassette {'Null' if is_null else '19A Switch'} (N={N}, L=15000, d={d:.3f})"
                )
            else:
                # S. pneumoniae & N. meningitidis Housekeeping Loci (pbp2x / folA, L=2100 nt)
                s_sub = sub - 6000
                N = 4 if s_sub < 3000 else 8
                is_null = (s_sub % 4 == 0)
                d = 0.03 + (s_sub % 40) * 0.002
                true_bps = [] if is_null else [600, 1500] # 900 nt cassette
                recs = ["R"] if not is_null else []
                parents = {"R": ["P1", "P2"]} if not is_null else {}

                return OrganismalScenario(
                    scenario_id=f"std_spneumo_pbp_{global_idx}",
                    archetype="spneumoniae_pbp",
                    category=category,
                    mode="codon",
                    is_null=is_null,
                    num_taxa=N,
                    length_nt=2100,
                    true_breakpoints=true_bps,
                    recombinant_taxa=recs,
                    parent_taxa=parents,
                    params={"d": d, "omega": 0.15, "kappa": 2.4},
                    description=f"S. pneumoniae pbp2x Allelic Exchange {'Null' if is_null else 'Mosaic'} (N={N}, L=2100, d={d:.3f})"
                )

        # Sub-block 4: Strictly Clonal Bacterial Nulls (42000 .. 45999) [4,000 runs]
        elif local_idx < 46000:
            sub = local_idx - 42000
            N = 4 if sub < 2000 else (8 if sub < 3500 else 16)
            d = 0.0003 + (sub % 40) * 0.00003
            return OrganismalScenario(
                scenario_id=f"std_mtbc_{global_idx}",
                archetype="mtbc",
                category=category,
                mode="codon" if (sub % 2 == 0) else "nt",
                is_null=True, # STRICT CLONAL NULL
                num_taxa=N,
                length_nt=3000,
                true_breakpoints=[],
                recombinant_taxa=[],
                parent_taxa={},
                params={"d": d, "kappa": 2.0},
                description=f"M. tuberculosis Complex Strictly Clonal Null (N={N}, L=3000, d={d:.5f})"
            )

        # Sub-block 5: Eukaryotic Organellar & Paralogous Copies (46000 .. 49999) [4,000 runs]
        else:
            sub = local_idx - 46000
            if sub < 2500:
                # Human mtDNA (L=16569 bp, Awadalla Null Control)
                N = 4 if sub < 1500 else 8
                d = 0.003 + (sub % 30) * 0.0002
                alpha = 0.15 + (sub % 20) * 0.01
                return OrganismalScenario(
                    scenario_id=f"std_mtdna_{global_idx}",
                    archetype="human_mtdna",
                    category=category,
                    mode="nt",
                    is_null=True, # STRICT MATERNAL CLONAL NULL
                    num_taxa=N,
                    length_nt=16569,
                    true_breakpoints=[],
                    recombinant_taxa=[],
                    parent_taxa={},
                    params={"d": d, "alpha": alpha, "kappa": 8.5},
                    description=f"Human mtDNA Maternal Clonal Null (N={N}, L=16569, d={d:.4f}, alpha={alpha:.2f})"
                )
            else:
                # Primate gamma-globin Duplicate Pair (L=3200 nt)
                s_sub = sub - 2500
                N = 4 if s_sub < 1000 else 8
                is_null = (s_sub % 4 == 0)
                d = 0.02 + (s_sub % 30) * 0.001
                true_bps = [] if is_null else [1600]
                recs = ["R"] if not is_null else []
                parents = {"R": ["P1", "P2"]} if not is_null else {}

                return OrganismalScenario(
                    scenario_id=f"std_globin_{global_idx}",
                    archetype="globin_duplication",
                    category=category,
                    mode="nt",
                    is_null=is_null,
                    num_taxa=N,
                    length_nt=3200,
                    true_breakpoints=true_bps,
                    recombinant_taxa=recs,
                    parent_taxa=parents,
                    params={"d": d, "kappa": 2.5},
                    description=f"Primate gamma-globin Duplication {'Null' if is_null else 'Ectopic Conversion'} (N={N}, L=3200, d={d:.3f})"
                )

    # -------------------------------------------------------------------------
    # PART 2: 'HELL WEEK' ADVERSARIAL TORTURE BATTERY (Indices 50000 .. 99999)
    # -------------------------------------------------------------------------
    else:
        # Attack 1: Host-Shift Darren Trap Heterotachy Waves (50000 .. 59999) [10,000 runs]
        if local_idx < 10000:
            sub = local_idx
            N = 4 if sub < 5000 else (8 if sub < 8500 else 16)
            gamma = 1.5 + (sub % 40) * 0.25 # gamma from 1.5 to 11.5
            L_nt = 2151 if (sub % 2 == 0) else 3000
            return OrganismalScenario(
                scenario_id=f"adv_darren_trap_{global_idx}",
                archetype="flu_host_shift_trap",
                category=category,
                mode="nt",
                is_null=True, # STRICT TREE-LIKE CLONAL NULL
                num_taxa=N,
                length_nt=L_nt,
                true_breakpoints=[],
                recombinant_taxa=[],
                parent_taxa={},
                params={"gamma": gamma, "d": 0.03, "kappa": 3.5},
                description=f"Influenza Avian-to-Human Heterotachy Darren Trap (N={N}, gamma={gamma:.1f}, Null H0)"
            )

        # Attack 2: Host Deaminase Kataegis / Hypermutation Showers (60000 .. 68999) [9,000 runs]
        elif local_idx < 19000:
            sub = local_idx - 10000
            N = 4 if sub < 4500 else (8 if sub < 7500 else 16)
            k_burst = 6 + (sub % 35) * 1 # k from 6 to 40 transitions
            deaminase = "APOBEC3G" if (sub % 2 == 0) else "ADAR"
            return OrganismalScenario(
                scenario_id=f"adv_deaminase_{global_idx}",
                archetype="sars2_host_kataegis",
                category=category,
                mode="nt",
                is_null=True, # STRICT CLONAL NULL WITH PRIVATE SHOWER
                num_taxa=N,
                length_nt=3000,
                true_breakpoints=[],
                recombinant_taxa=[],
                parent_taxa={},
                params={"k_burst": k_burst, "deaminase": deaminase, "window_len": 150, "kappa": 2.5},
                description=f"Host {deaminase} Deaminase Shower (N={N}, k={k_burst} in 150nt, Null H0)"
            )

        # Attack 3: Episodic Positive Selection Darwinian Mimics (69000 .. 76999) [8,000 runs]
        elif local_idx < 27000:
            sub = local_idx - 19000
            N = 4 if sub < 4000 else 8
            omega_burst = 2.0 + (sub % 30) * 0.45 # omega from 2.0 to 15.5
            is_codon = (sub % 2 == 0)
            return OrganismalScenario(
                scenario_id=f"adv_darwinian_{global_idx}",
                archetype="hiv_env_selection_burst",
                category=category,
                mode="codon" if is_codon else "nt",
                is_null=True, # STRICT CLONAL GENEALOGY WITH SELECTION BURST
                num_taxa=N,
                length_nt=3000,
                true_breakpoints=[],
                recombinant_taxa=[],
                parent_taxa={},
                params={"omega_0": 0.05, "omega_burst": omega_burst, "burst_len_codons": 150, "kappa": 3.0},
                description=f"HIV gp120 Episodic Darwinian Selection Burst (N={N}, omega={omega_burst:.1f}, mode={'codon' if is_codon else 'nt'})"
            )

        # Attack 4: Convergent Drug-Resistance Hotspot Traps (77000 .. 82999) [6,000 runs]
        elif local_idx < 33000:
            sub = local_idx - 27000
            N = 4 if sub < 3000 else (8 if sub < 5000 else 16)
            num_convergent_sites = 3 + (sub % 8) * 1 # 3 to 10 convergent sites
            return OrganismalScenario(
                scenario_id=f"adv_convergent_{global_idx}",
                archetype="mtbc_convergent_amr",
                category=category,
                mode="codon",
                is_null=True, # STRICT CLONAL NULL WITH CONVERGENT SITES
                num_taxa=N,
                length_nt=3000,
                true_breakpoints=[],
                recombinant_taxa=[],
                parent_taxa={},
                params={"num_convergent_sites": num_convergent_sites, "d": 0.001, "kappa": 2.5},
                description=f"M. tuberculosis Convergent AMR Resistance Trap (N={N}, n_sites={num_convergent_sites}, Null H0)"
            )

        # Attack 5: Sub-Window Micro-Conversion Limits (83000 .. 90999) [8,000 runs]
        elif local_idx < 41000:
            sub = local_idx - 33000
            N = 4 if sub < 4500 else 8
            tract_len = 20 + (sub % 45) * 2 # tract length 20 to 108 nt
            d = 0.08 + (sub % 25) * 0.003
            w_start = 1450
            w_end = w_start + tract_len
            return OrganismalScenario(
                scenario_id=f"adv_micro_conv_{global_idx}",
                archetype="spneumo_micro_conversion",
                category=category,
                mode="nt",
                is_null=False, # RECOMBINANT MICRO-TRACT
                num_taxa=N,
                length_nt=3000,
                true_breakpoints=[w_start, w_end],
                recombinant_taxa=["R"],
                parent_taxa={"R": ["P1", "P2"]},
                params={"tract_len": tract_len, "w_start": w_start, "w_end": w_end, "d": d, "kappa": 2.5},
                description=f"S. pneumoniae RecA Minimal Micro-Conversion (N={N}, L_tract={tract_len}nt, d={d:.3f})"
            )

        # Attack 6: Second-Generation CRFs / Nested Reticulations (91000 .. 95999) [5,000 runs]
        elif local_idx < 46000:
            sub = local_idx - 41000
            N = 6 if sub < 3000 else 8
            d = 0.04 + (sub % 30) * 0.002
            # R1 is rec(P1, P2) at 1000 nt; R2 is rec(R1, P3) at 2000 nt
            return OrganismalScenario(
                scenario_id=f"adv_nested_crf_{global_idx}",
                archetype="hiv_second_gen_crf",
                category=category,
                mode="codon",
                is_null=False, # NESTED RECOMBINANT
                num_taxa=N,
                length_nt=3000,
                true_breakpoints=[1000, 2000],
                recombinant_taxa=["R1", "R2"],
                parent_taxa={"R1": ["P1", "P2"], "R2": ["R1", "P3"]},
                params={"d": d, "bp1": 1000, "bp2": 2000, "kappa": 3.5},
                description=f"HIV-1 Nested Second-Generation CRF (N={N}, R1=rec(P1,P2), R2=rec(R1,P3))"
            )

        # Attack 7: Felsenstein Saturation Zones (96000 .. 99999) [4,000 runs]
        else:
            sub = local_idx - 46000
            N = 4 if sub < 2500 else 8
            tree_length = 4.0 + (sub % 30) * 0.2 # T from 4.0 to 10.0 subs/site
            alpha = 0.08 + (sub % 15) * 0.005 # severe ASRV alpha <= 0.155
            return OrganismalScenario(
                scenario_id=f"adv_felsenstein_{global_idx}",
                archetype="ancient_viral_saturation",
                category=category,
                mode="nt",
                is_null=True, # STRICT CLONAL NULL WITH HOMOPLASY SATURATION
                num_taxa=N,
                length_nt=3000,
                true_breakpoints=[],
                recombinant_taxa=[],
                parent_taxa={},
                params={"tree_length": tree_length, "alpha": alpha, "kappa": 2.0},
                description=f"Felsenstein Saturation Zone Null (N={N}, T={tree_length:.1f}, alpha={alpha:.3f}, Null H0)"
            )
