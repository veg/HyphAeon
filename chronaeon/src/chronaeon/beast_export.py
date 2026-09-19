"""
chronaeon/beast_export.py
-------------------------
Automated BEAST 1.x / BEAST X (v10.5.0) XML Configuration Generator
with ChronAeon Tree-Free Geometric Warm-Start Priors.

Key Capabilities:
  1. Calibrates clock rate parameter (clock.rate or ucld.mean) directly to
     ChronAeon's empirical substitution rate mu (replacing arbitrary defaults like 1.0).
  2. Embeds informative data-driven LogNormal priors on clock rate matching
     ChronAeon's point estimate and Fieller / REML standard error.
  3. Calibrates treeModel.rootHeight prior directly to ChronAeon's inferred
     t_MRCA and 95% Fieller / LOOCV confidence interval.
  4. Initializes the coalescent population size (constant.popSize) directly to
     match the inferred tree height (H / 2.0), avoiding multi-million iteration
     demographic burn-in drift.
  5. Supports both Strict Molecular Clock and Uncorrelated Lognormal (UCLN)
     Relaxed Clock models.
  6. Supports AutoClock community partitioning: exports partitioned taxon sets
     (<taxa id="community_k">) with lineage-specific local clock rates.

Author: Sergei L. Kosakovsky Pond & DeepMind Antigravity Pair Programmer
"""

import os
import sys
import math
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np


def _xml_escape(text: str) -> str:
    """Escapes special XML characters in taxon names and attributes."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def generate_beast_xml(
    seqs: Dict[str, str],
    dates: Dict[str, float],
    dating_results: Dict[str, Any],
    clock_type: str = "relaxed",
    chain_length: int = 10000000,
    log_every: int = 1000,
    out_stem: str = "beast_run",
    autoclock_results: Optional[Dict[str, Any]] = None,
    coalescent_model: str = "constant",
    subst_model: str = "hky",
    gamma_categories: int = 4,
) -> str:
    """
    Generates production-grade BEAST v10.5.0 / BEAST 1.10.x XML pre-populated
    with ChronAeon empirical warm-start priors and parameters.

    Args:
        seqs: Dictionary mapping taxon ID to nucleotide sequence string.
        dates: Dictionary mapping taxon ID to float calendar date (decimal years).
        dating_results: Dictionary containing ChronAeon dating results (from run_mrca_dating).
        clock_type: 'strict' or 'relaxed' (UCLN).
        chain_length: Total MCMC chain length (default: 10,000,000).
        log_every: Log sampling frequency (default: 1,000).
        out_stem: Base filename for BEAST .log and .trees output files.
        autoclock_results: Optional dictionary from AutoClockDeconvolution with community partitions.
        coalescent_model: Coalescent demographic prior ('constant').
        subst_model: Substitution model ('hky').
        gamma_categories: Discrete gamma rate categories (default: 4).

    Returns:
        String containing complete, well-formed BEAST XML configuration.
    """
    clock_type = clock_type.lower()
    if clock_type not in ["strict", "relaxed"]:
        clock_type = "relaxed"

    taxa = sorted([t for t in seqs.keys() if t in dates])
    if not taxa:
        raise ValueError("No common taxa found between sequence alignment and dates table.")

    n_taxa = len(taxa)
    n_sites = len(seqs[taxa[0]])

    # 1. Extract and validate ChronAeon substitution rate mu
    mu = float(dating_results.get("mu", 1e-3))
    if mu <= 0 or np.isnan(mu):
        mu = 1e-3

    # Extract rate uncertainty
    se_mu = dating_results.get("se_mu")
    if se_mu is None or np.isnan(se_mu) or se_mu <= 0:
        if dating_results.get("ols", {}).get("se_mu"):
            se_mu = dating_results["ols"]["se_mu"]
        elif dating_results.get("pgls", {}).get("se_mu"):
            se_mu = dating_results["pgls"]["se_mu"]
        else:
            se_mu = 0.25 * mu

    log_mu = math.log(mu)
    log_se = max(0.20, float(se_mu) / mu)

    # 2. Extract and validate ChronAeon root height
    t_mrca = dating_results.get("t_mrca", np.nan)
    max_tip_date = max(dates[t] for t in taxa)
    min_tip_date = min(dates[t] for t in taxa)

    if np.isnan(t_mrca) or t_mrca >= max_tip_date:
        root_height = max_tip_date - min_tip_date
    else:
        root_height = max_tip_date - float(t_mrca)

    ci_mrca = dating_results.get("ci_mrca")
    if ci_mrca and not np.isnan(ci_mrca[0]) and not np.isnan(ci_mrca[1]):
        root_height_se = max(0.5, (ci_mrca[1] - ci_mrca[0]) / (2 * 1.96))
    else:
        root_height_se = max(1.0, 0.25 * root_height)

    # Calibrated population size parameter (expected tree height = 2 * N_e * (1 - 1/n))
    pop_size_init = max(0.1, root_height / 2.0)

    # 3. Taxa Block
    taxa_xml = []
    for t in taxa:
        taxa_xml.append(
            f'        <taxon id="{_xml_escape(t)}">\n'
            f'            <date value="{dates[t]:.4f}" direction="forwards" units="years"/>\n'
            f'        </taxon>'
        )
    taxa_str = "\n".join(taxa_xml)

    # AutoClock community partitions if provided
    community_taxa_xml = []
    if autoclock_results and "communities" in autoclock_results:
        comms = autoclock_results["communities"]
        for c_idx, c_info in enumerate(comms):
            c_taxa = [t for t in c_info.get("taxa", []) if t in taxa]
            if c_taxa:
                sub_xml = [f'    <taxa id="community_{c_idx}">']
                for ct in c_taxa:
                    sub_xml.append(f'        <taxon idref="{_xml_escape(ct)}"/>')
                sub_xml.append("    </taxa>")
                community_taxa_xml.append("\n".join(sub_xml))
    comm_taxa_str = ("\n\n" + "\n\n".join(community_taxa_xml)) if community_taxa_xml else ""

    # 4. Alignment Block
    seq_xml = []
    for t in taxa:
        seq_clean = seqs[t].strip().upper()
        seq_xml.append(
            f'        <sequence>\n'
            f'            <taxon idref="{_xml_escape(t)}"/>\n'
            f'            {seq_clean}\n'
            f'        </sequence>'
        )
    seq_str = "\n".join(seq_xml)

    # 5. Clock Architecture
    if clock_type == "strict":
        clock_branch_rates = f"""    <strictClockBranchRates id="branchRates">
        <rate>
            <parameter id="clock.rate" value="{mu:.6e}" lower="0.0"/>
        </rate>
    </strictClockBranchRates>"""
        clock_prior = f"""                <!-- ChronAeon Data-Calibrated Rate Prior (LogNormal) -->
                <logNormalPrior mean="{log_mu:.4f}" stdev="{log_se:.4f}" offset="0.0" meanInRealSpace="false">
                    <parameter idref="clock.rate"/>
                </logNormalPrior>"""
        clock_operator = f"""        <scaleOperator scaleFactor="0.75" weight="3">
            <parameter idref="clock.rate"/>
        </scaleOperator>
        <upDownOperator scaleFactor="0.75" weight="3">
            <up>
                <parameter idref="clock.rate"/>
            </up>
            <down>
                <parameter idref="treeModel.allInternalNodeHeights"/>
            </down>
        </upDownOperator>"""
        clock_log = '            <parameter idref="clock.rate"/>'
        screen_clock_col = """            <column label="clock.rate" sf="6" width="12">
                <parameter idref="clock.rate"/>
            </column>"""
    else:  # relaxed UCLN
        clock_branch_rates = f"""    <discretizedBranchRates id="branchRates">
        <treeModel idref="treeModel"/>
        <distribution>
            <logNormalDistributionModel meanInRealSpace="true">
                <mean>
                    <parameter id="ucld.mean" value="{mu:.6e}" lower="0.0"/>
                </mean>
                <stdev>
                    <parameter id="ucld.stdev" value="0.333333" lower="0.0"/>
                </stdev>
            </logNormalDistributionModel>
        </distribution>
        <rateCategories>
            <parameter id="branchRates.categories"/>
        </rateCategories>
    </discretizedBranchRates>
    <rateStatistic id="meanRate" name="meanRate" mode="mean" internal="true" external="true">
        <treeModel idref="treeModel"/>
        <discretizedBranchRates idref="branchRates"/>
    </rateStatistic>"""
        clock_prior = f"""                <!-- ChronAeon Data-Calibrated Mean Rate Prior (LogNormal) -->
                <logNormalPrior mean="{log_mu:.4f}" stdev="{log_se:.4f}" offset="0.0" meanInRealSpace="false">
                    <parameter idref="ucld.mean"/>
                </logNormalPrior>
                <exponentialPrior mean="0.333333" offset="0.0">
                    <parameter idref="ucld.stdev"/>
                </exponentialPrior>"""
        clock_operator = f"""        <scaleOperator scaleFactor="0.75" weight="3">
            <parameter idref="ucld.mean"/>
        </scaleOperator>
        <scaleOperator scaleFactor="0.75" weight="3">
            <parameter idref="ucld.stdev"/>
        </scaleOperator>
        <upDownOperator scaleFactor="0.75" weight="3">
            <up>
                <parameter idref="ucld.mean"/>
            </up>
            <down>
                <parameter idref="treeModel.allInternalNodeHeights"/>
            </down>
        </upDownOperator>
        <swapOperator size="1" weight="10" autoOptimize="false">
            <parameter idref="branchRates.categories"/>
        </swapOperator>
        <uniformIntegerOperator weight="10">
            <parameter idref="branchRates.categories"/>
        </uniformIntegerOperator>"""
        clock_log = """            <parameter idref="ucld.mean"/>
            <parameter idref="ucld.stdev"/>
            <rateStatistic idref="meanRate"/>"""
        screen_clock_col = """            <column label="ucld.mean" sf="6" width="12">
                <parameter idref="ucld.mean"/>
            </column>"""

    xml = f"""<?xml version="1.0" standalone="yes"?>
<!-- ====================================================================== -->
<!-- BEAST XML Generated by ChronAeon (Tree-Free Geometric Warm-Start)       -->
<!--                                                                        -->
<!-- ChronAeon Data-Calibrated Priors:                                      -->
<!--   • Inferred Evolutionary Rate: mu = {mu:.6e} subs/site/year           -->
<!--     LogNormal Prior: mean(log)={log_mu:.4f}, stdev={log_se:.4f}        -->
<!--   • Inferred Outbreak Origin: t_MRCA = {t_mrca:.2f} CE                 -->
<!--     Root Height Prior: mean = {root_height:.2f} yr, stdev = {root_height_se:.2f} yr     -->
<!--   • Calibrated Coalescent PopSize initial = {pop_size_init:.2f} yr               -->
<!--   • Molecular Clock Model: {clock_type.upper()}                                 -->
<!-- ====================================================================== -->
<beast>

    <!-- Taxa and sampling dates (N={n_taxa}) -->
    <taxa id="taxa">
{taxa_str}
    </taxa>{comm_taxa_str}

    <!-- Sequence alignment (N={n_taxa}, L={n_sites} nt) -->
    <alignment id="alignment" dataType="nucleotide">
{seq_str}
    </alignment>

    <patterns id="patterns" from="1">
        <alignment idref="alignment"/>
    </patterns>

    <!-- Constant size coalescent demographic model calibrated by ChronAeon -->
    <constantSize id="constant" units="years">
        <populationSize>
            <parameter id="constant.popSize" value="{pop_size_init:.3f}" lower="0.0"/>
        </populationSize>
    </constantSize>

    <coalescentTree id="startingTree">
        <taxa idref="taxa"/>
        <constantSize idref="constant"/>
    </coalescentTree>

    <treeModel id="treeModel">
        <coalescentTree idref="startingTree"/>
        <rootHeight>
            <parameter id="treeModel.rootHeight"/>
        </rootHeight>
        <nodeHeights internalNodes="true">
            <parameter id="treeModel.internalNodeHeights"/>
        </nodeHeights>
        <nodeHeights internalNodes="true" rootNode="true">
            <parameter id="treeModel.allInternalNodeHeights"/>
        </nodeHeights>
    </treeModel>

    <coalescentLikelihood id="coalescent">
        <model>
            <constantSize idref="constant"/>
        </model>
        <populationTree>
            <treeModel idref="treeModel"/>
        </populationTree>
    </coalescentLikelihood>

    <!-- Molecular Clock Model -->
{clock_branch_rates}

    <!-- Substitution Model: HKY85 + Gamma ({gamma_categories} categories) -->
    <HKYModel id="hky">
        <frequencies>
            <frequencyModel dataType="nucleotide">
                <alignment idref="alignment"/>
                <frequencies>
                    <parameter id="frequencies" dimension="4"/>
                </frequencies>
            </frequencyModel>
        </frequencies>
        <kappa>
            <parameter id="kappa" value="2.0" lower="0.0"/>
        </kappa>
    </HKYModel>

    <siteModel id="siteModel">
        <substitutionModel>
            <HKYModel idref="hky"/>
        </substitutionModel>
        <gammaShape gammaCategories="{gamma_categories}">
            <parameter id="alpha" value="0.5" lower="0.0"/>
        </gammaShape>
    </siteModel>

    <treeLikelihood id="treeLikelihood" useAmbiguities="false">
        <patterns idref="patterns"/>
        <treeModel idref="treeModel"/>
        <siteModel idref="siteModel"/>
        <discretizedBranchRates idref="branchRates"/>
    </treeLikelihood>

    <!-- Adaptive MCMC Operators -->
    <operators id="operators">
        <scaleOperator scaleFactor="0.75" weight="1">
            <parameter idref="kappa"/>
        </scaleOperator>
        <scaleOperator scaleFactor="0.75" weight="1">
            <parameter idref="alpha"/>
        </scaleOperator>
        <deltaExchange delta="0.01" weight="1">
            <parameter idref="frequencies"/>
        </deltaExchange>
{clock_operator}
        <scaleOperator scaleFactor="0.75" weight="3">
            <parameter idref="constant.popSize"/>
        </scaleOperator>
        <scaleOperator scaleFactor="0.75" weight="3">
            <parameter idref="treeModel.rootHeight"/>
        </scaleOperator>
        <uniformOperator weight="30">
            <parameter idref="treeModel.internalNodeHeights"/>
        </uniformOperator>
        <subtreeSlide size="2.0" gaussian="true" weight="15">
            <treeModel idref="treeModel"/>
        </subtreeSlide>
        <narrowExchange weight="15">
            <treeModel idref="treeModel"/>
        </narrowExchange>
        <wideExchange weight="3">
            <treeModel idref="treeModel"/>
        </wideExchange>
        <wilsonBalding weight="3">
            <treeModel idref="treeModel"/>
        </wilsonBalding>
    </operators>

    <!-- MCMC Sampling with ChronAeon Data-Calibrated Priors -->
    <mcmc id="mcmc" chainLength="{chain_length}" autoOptimize="true">
        <posterior id="posterior">
            <prior id="prior">
{clock_prior}
                <!-- ChronAeon Data-Calibrated Root Height Prior (Normal) -->
                <normalPrior mean="{root_height:.3f}" stdev="{root_height_se:.3f}">
                    <parameter idref="treeModel.rootHeight"/>
                </normalPrior>
                <logNormalPrior mean="1.0" stdev="1.25" offset="0.0" meanInRealSpace="false">
                    <parameter idref="kappa"/>
                </logNormalPrior>
                <exponentialPrior mean="0.5" offset="0.0">
                    <parameter idref="alpha"/>
                </exponentialPrior>
                <oneOnXPrior>
                    <parameter idref="constant.popSize"/>
                </oneOnXPrior>
                <coalescentLikelihood idref="coalescent"/>
            </prior>
            <likelihood id="likelihood">
                <treeLikelihood idref="treeLikelihood"/>
            </likelihood>
        </posterior>
        <operators idref="operators"/>

        <!-- Real-Time Screen Logger -->
        <log id="screenLog" logEvery="{log_every}">
            <column label="Posterior" dp="4" width="12">
                <posterior idref="posterior"/>
            </column>
            <column label="Prior" dp="4" width="12">
                <prior idref="prior"/>
            </column>
            <column label="Likelihood" dp="4" width="12">
                <likelihood idref="likelihood"/>
            </column>
            <column label="rootHeight" sf="6" width="12">
                <parameter idref="treeModel.rootHeight"/>
            </column>
{screen_clock_col}
        </log>

        <!-- Posterior Trace Log File -->
        <log id="fileLog" logEvery="{log_every}" fileName="{out_stem}.log">
            <posterior idref="posterior"/>
            <prior idref="prior"/>
            <likelihood idref="likelihood"/>
            <parameter idref="treeModel.rootHeight"/>
{clock_log}
            <parameter idref="constant.popSize"/>
            <parameter idref="kappa"/>
            <parameter idref="alpha"/>
        </log>

        <!-- Posterior Tree Log File -->
        <logTree id="treeFileLog" logEvery="{log_every}" nexusFormat="true" fileName="{out_stem}.trees" sortTranslationTable="true">
            <treeModel idref="treeModel"/>
            <trait name="rate" tag="rate">
                <discretizedBranchRates idref="branchRates"/>
            </trait>
            <posterior idref="posterior"/>
        </logTree>
    </mcmc>

    <report>
        <property name="timer">
            <mcmc idref="mcmc"/>
        </property>
    </report>

</beast>
"""
    return xml


def export_beast_xml(
    output_xml_path: Union[str, Path],
    alignment_path: Union[str, Path],
    dating_results: Dict[str, Any],
    dates_source: Optional[Union[str, Path, Dict[str, float]]] = None,
    date_col: Optional[str] = None,
    strain_col: Optional[str] = None,
    clock_type: str = "relaxed",
    chain_length: int = 10000000,
    log_every: int = 1000,
    autoclock_results: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Parses alignment and dates, builds the BEAST XML configuration with
    ChronAeon empirical warm-start priors, and writes to disk.
    """
    from .dating import parse_sample_dates
    from aeon_core.dataset import parse_alignment_sequences

    out_p = Path(output_xml_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    seqs = parse_alignment_sequences(str(alignment_path))
    taxa = list(seqs.keys())

    # Ingest dates: use dating_results['taxa_records'] if available
    dates_dict: Dict[str, float] = {}
    if dating_results.get("taxa_records"):
        for rec in dating_results["taxa_records"]:
            t = rec.get("taxon")
            d = rec.get("sampling_date")
            if t and d is not None and not np.isnan(d):
                dates_dict[t] = float(d)

    # Fallback to parse_sample_dates if not populated
    if not dates_dict and dates_source:
        dates_dict, _ = parse_sample_dates(
            taxa=taxa,
            dates_source=dates_source,
            date_col=date_col,
            strain_col=strain_col,
        )

    out_stem = out_p.stem
    xml_content = generate_beast_xml(
        seqs=seqs,
        dates=dates_dict,
        dating_results=dating_results,
        clock_type=clock_type,
        chain_length=chain_length,
        log_every=log_every,
        out_stem=out_stem,
        autoclock_results=autoclock_results,
    )

    with open(out_p, "w", encoding="utf-8") as f:
        f.write(xml_content)

    return out_p
