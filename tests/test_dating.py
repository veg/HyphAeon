"""Unit tests for HyphAeon dating module and non-linear clock models."""
import numpy as np
import pytest

from hyphaeon.dating import (
    run_ols_dating,
    run_pgls_dating,
    run_powerlaw_clock_dating,
    compute_rcs_basis,
    run_restricted_spline_clock_dating,
    run_mrca_dating,
)


class TestClockDatingModels:
    def test_ols_clock_linear(self):
        np.random.seed(42)
        true_t0 = 1950.0
        true_mu = 0.002
        times = np.linspace(1970, 2020, 50)
        noise = np.random.normal(0, 0.002, size=len(times))
        dists = true_mu * (times - true_t0) + noise

        res = run_ols_dating(times, dists, n_boot=100)
        assert abs(res['t_mrca'] - true_t0) < 5.0
        assert abs(res['mu'] - true_mu) < 0.0005
        assert res['r2'] > 0.90

    def test_rcs_basis_ancestral_linearity(self):
        """Verify Harrell RCS basis vanishes identically for t <= t_min, ensuring strictly linear ancestral extrapolation."""
        knots = np.array([1980.0, 2000.0, 2020.0])
        ancestral_times = np.array([1900.0, 1950.0, 1979.9, 1980.0])
        B, dB = compute_rcs_basis(ancestral_times, knots)
        assert np.all(B == 0.0)
        assert np.all(dB == 0.0)

    def test_restricted_spline_on_linear_synthetic(self):
        """When data is strictly linear, restricted spline should fail to reject linearity and delta_AIC < 0."""
        np.random.seed(42)
        true_t0 = 1960.0
        true_mu = 0.0015
        times = np.linspace(1970, 2020, 60)
        noise = np.random.normal(0, 0.001, size=len(times))
        dists = true_mu * (times - true_t0) + noise

        res = run_restricted_spline_clock_dating(times, dists, n_boot=50)
        assert abs(res['t_mrca'] - true_t0) < 5.0
        assert not res['is_nonlinear_preferred']

    def test_restricted_spline_on_decelerating_synthetic(self):
        """When data has genuine rate deceleration across the observation window,

        restricted spline should detect it (p < 0.05, delta_AIC > 2) without boundary collapse.
        """
        np.random.seed(123)
        times = np.linspace(1980, 2020, 80)
        knots = np.array([1980.0, 2000.0, 2015.0])
        B, _ = compute_rcs_basis(times, knots)
        # beta_0 = -1.98, beta_1 = 0.0010 (ancestral rate), beta_2 = -0.0005 (deceleration)
        true_dists = -1.98 + 0.0010 * times - 0.0005 * B[:, 0]
        noise = np.random.normal(0, 0.0002, size=len(times))
        dists = true_dists + noise

        res = run_restricted_spline_clock_dating(times, dists, n_boot=50)
        assert res['is_nonlinear_preferred']
        assert res['p_f_test'] < 0.01
        assert res['delta_aic'] > 2.0
        assert res['rate_recent'] < res['rate_ancestral']
        # Ancestral extrapolation is linear and stable
        assert 1970.0 < res['t_mrca'] < 1982.0

    def test_run_mrca_dating_auto_selection(self, tmp_path):
        """Test run_mrca_dating end-to-end with mock alignment and auto clock selection."""
        fasta_path = tmp_path / 'mock_dated.fasta'
        fasta_content = (
            ">seq1_1980\nATGGCC\n"
            ">seq2_1990\nATGGCA\n"
            ">seq3_2000\nATGGTA\n"
            ">seq4_2010\nTTGGTA\n"
            ">seq5_2020\nTTGGTT\n"
        )
        fasta_path.write_text(fasta_content)

        res = run_mrca_dating(
            alignment_path=str(fasta_path),
            use_tn93=True,
            clock_model='auto',
            method='ols',
            n_bootstrap=50,
        )

        assert 'ols' in res
        assert 'spline' in res
        assert res['clock_model'] == 'auto'
        assert 'selected_clock' in res
        assert len(res['taxa_records']) == 5

    def test_time_decay_consensus(self):
        """Test that time-decay consensus properly upweights early ancestral isolates."""
        from hyphaeon.dating import generate_time_decay_consensus_sequence
        seq_dict = {
            'seq1_1920': 'AAAAAA',
            'seq2_2020': 'TTTTTT',
            'seq3_2021': 'TTTTTT',
            'seq4_2022': 'TTTTTT',
        }
        dates_map = {'seq1_1920': 1920.0, 'seq2_2020': 2020.0, 'seq3_2021': 2021.0, 'seq4_2022': 2022.0}
        # In unweighted consensus, 'T' is 3/4 (75%), so unweighted would produce 'TTTTTT'.
        # But with time-decay (gamma=0.05, delta_t=100 -> weight of 1920 is exp(0)=1.0 vs ~exp(-5)=0.0067),
        # 1920 dominates and produces 'AAAAAA'.
        con_seq, eff_g = generate_time_decay_consensus_sequence(seq_dict, dates_map, gamma=0.05)
        assert con_seq == 'AAAAAA'
        assert eff_g == 0.05

    def test_parse_beast_1_xml(self, tmp_path):
        """Test parsing of BEAST 1.x XML alignment, sampling dates, and starting tree."""
        from hyphaeon.dataset import parse_beast_xml

        xml_content = """<?xml version="1.0" standalone="yes"?>
<beast version="1.10.4">
    <taxa id="taxa">
        <taxon id="taxon_A"><date value="1980.0" direction="forwards" units="years"/></taxon>
        <taxon id="taxon_B"><date value="1990.0" direction="forwards" units="years"/></taxon>
        <taxon id="taxon_C"><date value="2000.0" direction="forwards" units="years"/></taxon>
        <taxon id="taxon_D"><date value="2010.0" direction="forwards" units="years"/></taxon>
    </taxa>
    <alignment id="alignment" dataType="nucleotide">
        <sequence><taxon idref="taxon_A"/>ATGGCC</sequence>
        <sequence><taxon idref="taxon_B"/>ATGGCA</sequence>
        <sequence><taxon idref="taxon_C"/>ATGGTA</sequence>
        <sequence><taxon idref="taxon_D"/>TTGGTA</sequence>
    </alignment>
    <newick id="startingTree">((taxon_A:0.01,taxon_B:0.02):0.05,(taxon_C:0.03,taxon_D:0.04):0.05);</newick>
</beast>"""
        xml_file = tmp_path / "test_beast1.xml"
        xml_file.write_text(xml_content)

        data = parse_beast_xml(xml_file)
        assert data["version"] == "BEAST 1"
        assert len(data["sequences"]) == 4
        assert data["sequences"]["taxon_A"] == "ATGGCC"
        assert len(data["dates"]) == 4
        assert data["dates"]["taxon_A"] == 1980.0
        assert data["dates"]["taxon_D"] == 2010.0
        assert data["tree_newick"] is not None
        assert "taxon_A" in data["tree_newick"]

    def test_parse_beast_2_xml(self, tmp_path):
        """Test parsing of BEAST 2.x XML data and TraitSet date trait values."""
        from hyphaeon.dataset import parse_beast_xml

        xml_content = """<beast version="2.6" namespace="beast.evolution.alignment:beast.evolution.tree">
    <data id="h1n1" name="alignment">
        <sequence id="seq_A" taxon="isolate_A" value="ATGGCC"/>
        <sequence id="seq_B" taxon="isolate_B" value="ATGGCA"/>
        <sequence id="seq_C" taxon="isolate_C" value="ATGGTA"/>
        <sequence id="seq_D" taxon="isolate_D" value="TTGGTA"/>
    </data>
    <trait id="dateTrait" spec="beast.evolution.tree.TraitSet" traitname="date" value="
        isolate_A=1990.25,
        isolate_B=2000.50,
        isolate_C=2010.75,
        isolate_D=2020.00
    "/>
</beast>"""
        xml_file = tmp_path / "test_beast2.xml"
        xml_file.write_text(xml_content)

        data = parse_beast_xml(xml_file)
        assert data["version"] == "BEAST 2"
        assert len(data["sequences"]) == 4
        assert data["sequences"]["isolate_A"] == "ATGGCC"
        assert len(data["dates"]) == 4
        assert data["dates"]["isolate_A"] == 1990.25
        assert data["dates"]["isolate_D"] == 2020.00

    def test_run_mrca_dating_beast(self, tmp_path):
        """Test run_mrca_dating directly ingesting a BEAST XML file via beast_path."""
        xml_content = """<beast version="1.10.4">
    <taxa id="taxa">
        <taxon id="t1"><date value="1980.0"/></taxon>
        <taxon id="t2"><date value="1990.0"/></taxon>
        <taxon id="t3"><date value="2000.0"/></taxon>
        <taxon id="t4"><date value="2010.0"/></taxon>
        <taxon id="t5"><date value="2020.0"/></taxon>
    </taxa>
    <alignment id="alignment">
        <sequence><taxon idref="t1"/>ATGGCC</sequence>
        <sequence><taxon idref="t2"/>ATGGCA</sequence>
        <sequence><taxon idref="t3"/>ATGGTA</sequence>
        <sequence><taxon idref="t4"/>TTGGTA</sequence>
        <sequence><taxon idref="t5"/>TTGGTT</sequence>
    </alignment>
</beast>"""
        xml_file = tmp_path / "synthetic_dating.xml"
        xml_file.write_text(xml_content)

        res = run_mrca_dating(
            beast_path=str(xml_file),
            use_tn93=True,
            method="ols",
            clock_model="linear",
            n_bootstrap=50,
        )
        assert "ols" in res
        assert len(res["taxa_records"]) == 5
        assert res["ols"]["mu"] > 0
        assert res["ols"]["t_mrca"] < 1980.0

    def test_autoclock_with_beast_xml(self, tmp_path):
        """Test run_autoclock_deconvolution directly ingesting a BEAST XML file."""
        from hyphaeon.autoclock import run_autoclock_deconvolution

        xml_content = """<beast version="1.10.4">
    <taxa id="taxa">
        <taxon id="t1"><date value="1980.0"/></taxon>
        <taxon id="t2"><date value="1990.0"/></taxon>
        <taxon id="t3"><date value="2000.0"/></taxon>
        <taxon id="t4"><date value="2010.0"/></taxon>
        <taxon id="t5"><date value="2020.0"/></taxon>
    </taxa>
    <alignment id="alignment">
        <sequence><taxon idref="t1"/>ATGGCC</sequence>
        <sequence><taxon idref="t2"/>ATGGCA</sequence>
        <sequence><taxon idref="t3"/>ATGGTA</sequence>
        <sequence><taxon idref="t4"/>TTGGTA</sequence>
        <sequence><taxon idref="t5"/>TTGGTT</sequence>
    </alignment>
</beast>"""
        xml_file = tmp_path / "autoclock_beast.xml"
        xml_file.write_text(xml_content)

        res = run_autoclock_deconvolution(
            beast_path=str(xml_file),
            max_k=2,
            manifold="tn93",
            quiet=True,
            output_dir=tmp_path / "out",
        )
        assert res["optimal_k"] >= 1
        assert len(res["communities"]) >= 1

    def test_select_adaptive_n_landmarks(self):
        """Test adaptive Nyström landmark selection under memory budgets and dataset sizes."""
        from hyphaeon.autoclock import select_adaptive_n_landmarks

        # Small dataset: clamped to n_taxa
        m_small = select_adaptive_n_landmarks(n_taxa=50, max_k=4)
        assert m_small == 50

        # Medium dataset (N=2,873): should choose sublinear statistical bound
        m_med = select_adaptive_n_landmarks(n_taxa=2873, max_k=8, max_memory_mb=1024.0)
        assert 500 <= m_med <= 2500

        # Huge dataset (N=100,000) under tight memory budget (100 MB)
        # M_mem = (100 * 1e6) / (8 * 100,000) = 125
        m_tight = select_adaptive_n_landmarks(n_taxa=100000, max_k=8, max_memory_mb=100.0)
        assert m_tight == 125

        # Explicit user landmark count override
        m_user = select_adaptive_n_landmarks(n_taxa=5000, user_landmarks=500)
        assert m_user == 500

    def test_hierarchical_autoclock_deconvolution(self, tmp_path):
        """Test recursive hierarchical multi-clock community deconvolution."""
        from pathlib import Path
        import pandas as pd
        from hyphaeon.autoclock import HierarchicalAutoClock, run_hierarchical_autoclock

        # Create synthetic multi-community alignment: 2 distinct clades with distinct mutation rates
        clade1_taxa = [f"c1_t{i}" for i in range(1, 11)]
        clade2_taxa = [f"c2_t{i}" for i in range(1, 11)]
        all_taxa = clade1_taxa + clade2_taxa

        dates_map = {}
        seqs = {}
        for i, t in enumerate(clade1_taxa):
            dates_map[t] = 1990.0 + i * 2.0
            # Clade 1 background: base prefix AAAA, slow rate
            seqs[t] = "AAAA" + "A" * (10 - i) + "C" * i + "T" * 20
        for i, t in enumerate(clade2_taxa):
            dates_map[t] = 2000.0 + i * 1.5
            # Clade 2 background: base prefix GGGG, fast rate
            seqs[t] = "GGGG" + "G" * (10 - i) + "T" * i + "C" * 20

        fasta_path = tmp_path / "hierarchical_test.fasta"
        with open(fasta_path, "w") as f:
            for t in all_taxa:
                f.write(f">{t}\n{seqs[t]}\n")

        dates_csv = tmp_path / "hierarchical_dates.csv"
        pd.DataFrame({"id": all_taxa, "date": [dates_map[t] for t in all_taxa]}).to_csv(dates_csv, index=False)

        res = run_hierarchical_autoclock(
            alignment_path=fasta_path,
            dates_source=dates_csv,
            max_depth=2,
            min_leaf_size=5,
            min_delta_aicc=5.0,
            manifold="tn93",
            output_dir=tmp_path / "hier_out",
            plot=True,
            quiet=True,
        )

        assert res["total_taxa"] == 20
        assert res["n_leaves"] >= 1
        assert "leaf_communities" in res
        assert Path(res["tree_path"]).exists()
        assert Path(res["classified_metadata_path"]).exists()
        assert (tmp_path / "hier_out" / "fig_hierarchical_autoclock_diagnostic.png").exists()
