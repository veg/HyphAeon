"""
tests/test_r0.py
----------------
Unit and integration test suite for HyphAeon Phylodynamics (r, R0, Rt) inference.
"""

import os
import sys
import subprocess
import pytest
import numpy as np
from Bio import Phylo
from Bio.Phylo.BaseTree import Tree, Clade

from hyphaeon.r0 import (
    parse_calendar_date,
    load_dates,
    time_calibrate_tree,
    extract_coalescent_intervals,
    estimate_exponential_growth,
    compute_reproduction_numbers,
    estimate_dynamic_rt,
    run_r0_analysis,
    PATHOGEN_PRESETS,
)


def test_date_parsing():
    # Decimal
    assert parse_calendar_date(2009.332) == pytest.approx(2009.332)
    assert parse_calendar_date("2009.5") == pytest.approx(2009.5)
    
    # ISO date: 2014-07-02 is approximately mid-2014
    d_iso = parse_calendar_date("2014-07-02")
    assert d_iso is not None
    assert 2014.49 <= d_iso <= 2014.52
    
    # Slash dates
    d_slash = parse_calendar_date("15/04/2009")
    assert d_slash is not None
    assert 2009.28 <= d_slash <= 2009.30

    # Invalid / None
    assert parse_calendar_date(None) is None
    assert parse_calendar_date("unknown_date") is None


def test_pathogen_presets():
    assert "h1n1" in PATHOGEN_PRESETS
    assert "ebola" in PATHOGEN_PRESETS
    assert "sars_cov_2" in PATHOGEN_PRESETS
    assert "measles" in PATHOGEN_PRESETS
    
    h1n1 = PATHOGEN_PRESETS["h1n1"]
    assert h1n1["generation_time"] == pytest.approx(2.8)
    assert h1n1["generation_sd"] == pytest.approx(1.3)


def test_reproduction_number_formulas():
    # If r = 0, R0 must be exactly 1.0
    res_zero = compute_reproduction_numbers(
        r_day=0.0,
        r_ci_day=[-0.01, 0.01],
        generation_time=5.0,
        generation_sd=2.5,
    )
    assert res_zero["R0_sir"] == pytest.approx(1.0)
    assert res_zero["R0_gamma"] == pytest.approx(1.0)
    
    # Positive growth: r = 0.1 / day, Tg = 5.0 days
    # SIR: 1 + 0.1 * 5 = 1.5
    res_pos = compute_reproduction_numbers(
        r_day=0.10,
        r_ci_day=[0.08, 0.12],
        generation_time=5.0,
        generation_sd=2.5,
        latent_time=2.0,
    )
    assert res_pos["R0_sir"] == pytest.approx(1.50)
    # Gamma: alpha = (5/2.5)^2 = 4, beta = 5/6.25 = 0.8. R0 = (1 + 0.1 / 0.8)^4 = (1.125)^4 ~ 1.6018
    assert res_pos["R0_gamma"] > res_pos["R0_sir"]
    assert res_pos["R0_seir"] is not None
    # Doubling time: ln(2) / 0.1 ~ 6.93 days
    assert res_pos["doubling_time_days"] == pytest.approx(6.9315, rel=1e-3)


def test_profile_likelihood_exponential_growth():
    # Create synthetic intervals where population is rapidly expanding into the past
    # 5 coalescent events with higher rate in past
    # Intervals: (tau1, tau2, k)
    intervals = [
        (0.0, 0.1, 2),
        (0.1, 0.2, 3),
        (0.2, 0.3, 4),
        (0.3, 0.4, 5),
    ]
    coal_taus = [0.15, 0.25, 0.35, 0.38]
    
    fit = estimate_exponential_growth(intervals, coal_taus)
    assert fit["status"] == "success"
    assert "r_year" in fit
    assert "r_day" in fit
    assert fit["C"] == 4
    assert fit["r_ci_year"][0] <= fit["r_year"] <= fit["r_ci_year"][1]


def test_time_calibration_and_induced_coalescence():
    # Build a simple synthetic tree: ((A:0.02, B:0.01):0.01, C:0.04);
    clade_a = Clade(branch_length=0.02, name="A")
    clade_b = Clade(branch_length=0.01, name="B")
    clade_ab = Clade(branch_length=0.01, clades=[clade_a, clade_b])
    clade_c = Clade(branch_length=0.04, name="C")
    root = Clade(clades=[clade_ab, clade_c])
    tree = Tree(root=root)
    
    tip_dates = {
        "A": 2020.5,
        "B": 2020.4,
        "C": 2020.8,
    }
    
    node_dates, mu, tmrca = time_calibrate_tree(tree, tip_dates)
    assert tmrca < min(tip_dates.values())
    assert node_dates[clade_ab] < tip_dates["A"]
    assert node_dates[clade_ab] < tip_dates["B"]
    
    intervals, coal_taus, t_max, _ = extract_coalescent_intervals(tree, node_dates)
    # 3 leaves -> exactly 2 coalescent events
    assert len(coal_taus) == 2
    assert t_max == pytest.approx(2020.8)


def test_h1n1_benchmark_example_end_to_end(tmp_path):
    # Test on the bundled H1N1 dataset
    aln_path = os.path.join(os.path.dirname(__file__), "..", "examples", "H1N1_2009_pandemic.fasta")
    tree_path = os.path.join(os.path.dirname(__file__), "..", "examples", "H1N1_2009_pandemic.nwk")
    
    if not os.path.exists(aln_path) or not os.path.exists(tree_path):
        pytest.skip("H1N1 example files not found.")
        
    out_plot = str(tmp_path / "test_r0.png")
    out_json = str(tmp_path / "test_r0.json")
    
    res = run_r0_analysis(
        alignment_path=aln_path,
        tree_path=tree_path,
        pathogen="h1n1",
        window_size=0.20,
        step_size=0.05,
    )
    
    assert res["status"] == "success"
    assert res["n_taxa"] == 100
    assert res["n_coalescent_events"] == 99
    assert res["t_mrca"] == pytest.approx(2008.989, abs=0.05)
    
    # Inferred growth rate should be positive and concordant
    growth = res["growth_rate"]
    assert 5.0 <= growth["r_year"] <= 10.0
    
    # R0 should be greater than 1.0
    rep = res["reproduction_number"]
    assert 1.02 <= rep["R0_gamma"] <= 1.15
    assert len(res["rt_skyline"]) > 0


def test_cli_r0_example(tmp_path):
    cmd = [
        sys.executable,
        "-m",
        "hyphaeon.cli",
        "r0",
        "--example",
        "--output",
        str(tmp_path / "cli_r0.json"),
        "--plot-path",
        str(tmp_path / "cli_r0.png"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "HYPHAEON R0 ESTIMATION RESULTS" in proc.stdout
    assert os.path.exists(tmp_path / "cli_r0.json")
    assert os.path.exists(tmp_path / "cli_r0.png")
