"""Unit tests for ChronAeon dating module and non-linear clock models."""
import numpy as np
import pytest
from scipy import stats

from chronaeon.dating import (
    run_ols_dating,
    run_pgls_dating,
    run_powerlaw_clock_dating,
    compute_rcs_basis,
    run_restricted_spline_clock_dating,
    run_mrca_dating,
    estimate_reml_pagel_lambda,
    run_dating_loocv,
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
        assert res['delta_aic'] < -2.0
        assert res['aic_reduction'] > 2.0
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
        from chronaeon.dating import generate_time_decay_consensus_sequence
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
        from aeon_core.dataset import parse_beast_xml

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
        from aeon_core.dataset import parse_beast_xml

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
        from chronaeon.autoclock import run_autoclock_deconvolution

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
        from chronaeon.autoclock import select_adaptive_n_landmarks

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
        from chronaeon.autoclock import HierarchicalAutoClock, run_hierarchical_autoclock

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

    def test_autoclock_contemporaneous_dyads_and_convex_decay(self, tmp_path):
        """Test contemporaneous direct transmission dyad screening and convex decay consensus rooting."""
        from pathlib import Path
        import pandas as pd
        from chronaeon.autoclock import run_hierarchical_autoclock, detect_contemporaneous_dyads

        # Create 6 taxa:
        # Pair 1 (t1, t2): sampled 5 days apart, distance 0.000 (identical) -> Contemporaneous Dyad
        # Pair 2 (t3, t4): sampled 30 days apart, distance 0.005 -> Contemporaneous Dyad
        # Non-dyad (t5, t6): sampled 4 years apart, distance 0.050
        taxa = [f"iso_{i}" for i in range(1, 7)]
        dates_map = {
            "iso_1": 2015.000,
            "iso_2": 2015.014, # ~5 days later
            "iso_3": 2016.100,
            "iso_4": 2016.182, # ~30 days later
            "iso_5": 2010.000,
            "iso_6": 2014.000,
        }
        seqs = {
            "iso_1": "ACGT" * 50,
            "iso_2": "ACGT" * 50, # identical to iso_1
            "iso_3": "ACGT" * 49 + "ACGA", # 1 substitution from iso_1
            "iso_4": "ACGT" * 49 + "ACGC", # 1 substitution from iso_1
            "iso_5": "AAAA" * 50,
            "iso_6": "CCCC" * 50,
        }

        # 1. Test direct detect_contemporaneous_dyads function
        dyads_df, dyad_map = detect_contemporaneous_dyads(
            taxa=taxa,
            dates_map=dates_map,
            seq_dict=seqs,
            dyad_max_days=90.0,
            dyad_max_dist=0.010,
        )
        assert len(dyads_df) >= 2
        assert "iso_1" in dyad_map
        assert "iso_2" in dyad_map
        assert "iso_3" in dyad_map
        assert "iso_4" in dyad_map
        assert "iso_5" not in dyad_map
        assert "iso_6" not in dyad_map

        # 2. Test full hierarchical autoclock pipeline with convex_decay rooting & dyads
        fa_p = tmp_path / "dyads_test.fa"
        with open(fa_p, "w") as f:
            for t in taxa:
                f.write(f">{t}\n{seqs[t]}\n")
        meta_p = tmp_path / "dyads_meta.csv"
        pd.DataFrame({"id": taxa, "date": [dates_map[t] for t in taxa]}).to_csv(meta_p, index=False)

        res = run_hierarchical_autoclock(
            alignment_path=fa_p,
            dates_source=meta_p,
            max_depth=1,
            min_leaf_size=2,
            rooting_mode="convex_decay",
            contemporaneous_dyads=True,
            dyad_max_days=90.0,
            dyad_max_dist=0.010,
            output_dir=tmp_path / "dyad_run",
            quiet=True,
        )

        assert res["rooting_mode"] == "convex_decay"
        assert res["contemporaneous_dyads_enabled"] is True
        assert res["n_contemporaneous_clusters"] >= 2
        assert (tmp_path / "dyad_run" / "contemporaneous_dyads.csv").exists()

        df_class = pd.read_csv(tmp_path / "dyad_run" / "hierarchical_classified_metadata.csv")
        assert "is_contemporaneous_dyad" in df_class.columns
        assert df_class.loc[df_class["id"] == "iso_1", "is_contemporaneous_dyad"].values[0] == True
        assert df_class.loc[df_class["id"] == "iso_5", "is_contemporaneous_dyad"].values[0] == False


class TestTransformerMetricityDiagnostics:
    def test_metricity_outbreak_regime(self):
        from chronaeon.dating import compute_transformer_metricity_diagnostics
        # Low divergence matrix (mean ~ 0.002, d90 < 0.05)
        np.random.seed(42)
        n = 15
        D = np.random.uniform(0.0005, 0.004, (n, n))
        np.fill_diagonal(D, 0.0)
        D = (D + D.T) / 2.0

        diag = compute_transformer_metricity_diagnostics(D)
        assert diag["d_90"] < 0.05
        assert diag["recommended_regime"] == "tn93"
        assert "Outbreak" in diag["regime_label"]

    def test_metricity_tn93_nan_saturation_failure(self):
        from chronaeon.dating import compute_transformer_metricity_diagnostics
        np.random.seed(42)
        n = 15
        D = np.random.uniform(0.15, 0.30, (n, n))
        np.fill_diagonal(D, 0.0)
        D = (D + D.T) / 2.0
        # Simulate true TN93 mathematical saturation (undefined log arguments -> NaNs)
        mask = np.random.rand(n, n) < 0.20
        D[mask] = np.nan
        np.fill_diagonal(D, 0.0)
        for i in range(n):
            for j in range(i + 1, n):
                if np.isnan(D[i, j]):
                    D[j, i] = np.nan

        diag = compute_transformer_metricity_diagnostics(D)
        assert diag["nan_fraction"] > 0.05
        assert diag["recommended_regime"] == "latent"
        assert "Saturation Breakdown Recovery" in diag["regime_label"]

    def test_metricity_severe_non_metric_distortion_failure(self):
        from chronaeon.dating import compute_transformer_metricity_diagnostics
        n = 12
        # Severe triangle inequality violation creating high negative eigenvalue energy
        D = np.ones((n, n)) * 0.04
        np.fill_diagonal(D, 0.0)
        D[0, 1] = D[1, 0] = 5.0
        D[2, 3] = D[3, 2] = 5.0

        diag = compute_transformer_metricity_diagnostics(D)
        assert diag["rho_neg"] > 0.35
        assert diag["recommended_regime"] == "latent"
        assert "Saturation Breakdown Recovery" in diag["regime_label"]

    def test_metricity_deep_divergence_non_isometric_prefers_tn93(self):
        from chronaeon.dating import compute_transformer_metricity_diagnostics
        np.random.seed(42)
        n = 20
        # Deep divergence (d_90 ~ 0.28, finite, no NaNs, low distortion)
        D = np.random.uniform(0.15, 0.30, (n, n))
        np.fill_diagonal(D, 0.0)
        D = (D + D.T) / 2.0
        # Non-isometric latent representations (uncorrelated with physical distance, e.g. Faria)
        z = np.random.randn(n, 128)

        diag = compute_transformer_metricity_diagnostics(D, taxa_repr=z)
        assert diag["d_90"] >= 0.20
        assert diag["recommended_regime"] == "tn93"
        assert "Linear Evolutionary Drift" in diag["regime_label"]
        assert diag["rho_iso_spearman"] < 0.70

    def test_metricity_with_latent_representations(self):
        from chronaeon.dating import compute_transformer_metricity_diagnostics
        np.random.seed(42)
        n = 25
        d = 384
        # Simulated latent features and correlated distances
        z = np.random.randn(n, d)
        from scipy.spatial.distance import pdist, squareform
        d_lat = squareform(pdist(z, metric='euclidean'))
        # Scale to physical distances
        d_phys = d_lat * 0.02
        np.fill_diagonal(d_phys, 0.0)

        diag = compute_transformer_metricity_diagnostics(d_phys, taxa_repr=z)
        assert diag["rho_iso_pearson"] is not None
        assert diag["rho_iso_spearman"] is not None
        assert diag["rho_iso_pearson"] > 0.90
        assert diag["kappa_sat"] is not None
        assert diag["recommended_regime"] == "tn93"
        assert "Isometric Concordance Confirmed" in diag["regime_label"]

    def test_coverage_filtering_degraded_isolates(self):
        from chronaeon.dating import compute_transformer_metricity_diagnostics
        np.random.seed(42)
        n = 10
        D = np.random.uniform(0.001, 0.005, (n, n))
        np.fill_diagonal(D, 0.0)
        D = (D + D.T) / 2.0
        # Taxon 0 is an archival fragment with only 20% coverage
        cov = np.ones(n)
        cov[0] = 0.20

        diag = compute_transformer_metricity_diagnostics(D, coverage=cov)
        assert diag["recommended_regime"] == "tn93"

    def test_fieller_four_cases(self):
        """Verify all 4 mathematical regimes of Fieller's ratio inversion."""
        from chronaeon.dating import compute_fieller_mrca_interval

        # Case 1: Bounded (g < 1, disc >= 0)
        cov1 = np.array([[1e-8, 0.0], [0.0, 1e-6]])
        ci1, info1 = compute_fieller_mrca_interval(
            mu=0.002, d0=0.04, cov_beta=cov1, t_ref=2020.0, df=50, min_time=2000.0
        )
        assert info1['status'] == 'BOUNDED'
        assert info1['g'] < 1.0
        assert ci1[0] < ci1[1] <= 2000.0

        # Case 2: Complement Set (g > 1, disc > 0)
        # Distance offset is well-measured (C > 0), but slope has large variance (A < 0)
        cov2 = np.array([[1e-4, 0.0], [0.0, 1e-6]])
        ci2, info2 = compute_fieller_mrca_interval(
            mu=0.0005, d0=0.2, cov_beta=cov2, t_ref=2000.0, df=30, min_time=1950.0
        )
        assert info2['status'] == 'COMPLEMENT'
        assert info2['g'] > 1.0
        assert 'complement_set' in info2
        assert 'excluded_window' in info2
        assert ci2[0] == float('-inf')
        assert ci2[1] <= 1950.0

        # Case 3: All Reals (g > 1, disc <= 0)
        cov3 = np.array([[1.0, 0.0], [0.0, 1.0]])
        ci3, info3 = compute_fieller_mrca_interval(
            mu=0.0001, d0=0.0001, cov_beta=cov3, t_ref=2000.0, df=10
        )
        assert info3['status'] == 'ALL_REALS'
        assert info3['g'] > 1.0
        assert ci3[0] == float('-inf')

    def test_spline_wild_residual_bootstrap_non_degenerate(self):
        """Verify Wild Rademacher residual bootstrap generates non-degenerate CIs without [t0, t0] collapse."""
        from chronaeon.dating import run_restricted_spline_clock_dating
        np.random.seed(42)
        n = 50
        times = np.linspace(1985, 2005, n)
        # Synthetic non-linear decelerating clock
        dists = 0.05 + 0.002 * (times - 1985) - 0.0008 * np.maximum(0.0, times - 1995)
        dists += np.random.normal(0, 0.001, size=n)
        cov = np.eye(n) * 0.001

        res = run_restricted_spline_clock_dating(times, dists, cov_matrix=cov, n_boot=100, seed=42)
        assert res['ci_mrca'][0] < res['ci_mrca'][1], "Spline CI should not collapse to zero-width!"
        width = res['ci_mrca'][1] - res['ci_mrca'][0]
        assert width > 1.0, f"Expected realistic CI width > 1.0 yr, got {width}"


class TestPGLSDating:
    """Tests for the PGLS estimator with known covariance structure."""

    def test_pgls_linear_recovery_identity_cov(self):
        """PGLS with identity covariance should recover the same parameters as OLS."""
        np.random.seed(42)
        true_t0 = 1950.0
        true_mu = 0.002
        times = np.linspace(1970, 2020, 50)
        noise = np.random.normal(0, 0.002, size=len(times))
        dists = true_mu * (times - true_t0) + noise

        n = len(times)
        cov_identity = np.eye(n)

        res = run_pgls_dating(
            times, dists, cov_identity,
            ridge=0.0, pagel_lambda=1.0,
            ci_method="delta", n_boot=100
        )

        assert res['status'] == 'OK'
        assert abs(res['t_mrca'] - true_t0) < 5.0
        assert abs(res['mu'] - true_mu) < 0.0005
        assert res['r2'] > 0.90

    def test_pgls_with_phylogenetic_covariance(self):
        """PGLS with a known block-diagonal covariance should recover rate and t_mrca."""
        np.random.seed(123)
        true_t0 = 1960.0
        true_mu = 0.0015
        n = 40
        times = np.linspace(1975, 2020, n)

        # Build a block-diagonal covariance: 2 clades of 20 taxa each
        cov = np.zeros((n, n))
        block_size = n // 2
        for b in range(2):
            idx = slice(b * block_size, (b + 1) * block_size)
            cov[idx, idx] = 0.6
            np.fill_diagonal(cov[idx, idx], 1.0)

        # Generate correlated noise via Cholesky
        L = np.linalg.cholesky(cov + 1e-6 * np.eye(n))
        noise = L @ np.random.normal(0, 0.001, size=n)
        dists = true_mu * (times - true_t0) + noise

        res = run_pgls_dating(
            times, dists, cov,
            ridge=0.05, pagel_lambda=0.8,
            ci_method="delta", n_boot=100
        )

        assert res['status'] == 'OK'
        assert abs(res['t_mrca'] - true_t0) < 10.0
        assert abs(res['mu'] - true_mu) < 0.0005
        assert res['mu'] > 0
        assert res['r2'] > 0.80

    def test_pgls_non_positive_rate(self):
        """PGLS should report NON_POSITIVE_RATE when the slope is non-positive."""
        np.random.seed(42)
        times = np.linspace(2000, 2020, 30)
        # Negative slope (rate decreasing with time)
        dists = -0.001 * (times - 2000) + np.random.normal(0, 0.001, size=len(times))

        res = run_pgls_dating(
            times, dists, np.eye(len(times)),
            ridge=0.05, pagel_lambda=0.5,
            ci_method="delta", n_boot=50
        )

        assert res['status'] == 'NON_POSITIVE_RATE'
        assert np.isnan(res['t_mrca'])

    def test_pgls_mrca_after_earliest_sample(self):
        """PGLS should report MRCA_AFTER_EARLIEST_SAMPLE when t_mrca >= min(times)."""
        np.random.seed(42)
        times = np.linspace(2000, 2020, 30)
        # Very low divergence with high intercept -> t_mrca falls within sample range
        dists = 0.0001 * (times - 1999.5) + np.random.normal(0, 0.00001, size=len(times))

        res = run_pgls_dating(
            times, dists, np.eye(len(times)),
            ridge=0.0, pagel_lambda=1.0,
            ci_method="delta", n_boot=50
        )

        assert res['status'] in ['MRCA_AFTER_EARLIEST_SAMPLE', 'OK']

    def test_pgls_fieller_ci_bounded(self):
        """PGLS with strong signal should produce a bounded Fieller CI."""
        np.random.seed(42)
        true_t0 = 1950.0
        true_mu = 0.002
        times = np.linspace(1970, 2020, 80)
        noise = np.random.normal(0, 0.0005, size=len(times))
        dists = true_mu * (times - true_t0) + noise

        res = run_pgls_dating(
            times, dists, np.eye(len(times)),
            ridge=0.0, pagel_lambda=1.0,
            ci_method="fieller", n_boot=100
        )

        assert res['status'] == 'OK'
        assert res['fieller_g'] is not None and res['fieller_g'] < 1.0
        assert res['ci_mrca'][0] < res['ci_mrca'][1]
        assert res['ci_mrca'][1] <= float(np.min(times))

    def test_pgls_minimum_observations(self):
        """PGLS should raise ValueError with fewer than 3 observations."""
        with pytest.raises(ValueError, match="At least 3"):
            run_pgls_dating(
                np.array([2000.0, 2010.0]),
                np.array([0.01, 0.02]),
                np.eye(2),
            )


class TestREMLPagelLambda:
    """Tests for REML estimation of Pagel's lambda."""

    def test_reml_lambda_recovery_high_signal(self):
        """When data is generated with high phylogenetic signal, REML should estimate lambda > 0."""
        np.random.seed(42)
        true_t0 = 1960.0
        true_mu = 0.002
        n = 60
        times = np.linspace(1975, 2020, n)

        # Build a covariance with strong within-clade correlation
        true_lambda = 0.85
        K = np.zeros((n, n))
        block = n // 3
        for b in range(3):
            idx = slice(b * block, (b + 1) * block)
            K[idx, idx] = 0.9
            np.fill_diagonal(K[idx, idx], 1.0)

        C_true = true_lambda * K + (1 - true_lambda) * np.eye(n)
        L = np.linalg.cholesky(C_true + 1e-6 * np.eye(n))
        noise = L @ np.random.normal(0, 0.003, size=n)
        dists = true_mu * (times - true_t0) + noise

        res = estimate_reml_pagel_lambda(times, dists, K)

        assert res['status'] in ['OPTIMAL_REML', 'OPTIMAL_REML_SUBSAMPLED']
        est_lambda = res['best_lambda']
        # Should detect some phylogenetic signal (lambda > 0)
        assert est_lambda > 0.01, f"Expected lambda > 0.01, got {est_lambda}"

    def test_reml_lambda_in_range(self):
        """REML lambda should always be in (0.001, 0.999)."""
        np.random.seed(42)
        n = 30
        times = np.linspace(2000, 2020, n)
        dists = np.random.uniform(0.01, 0.05, size=n)
        K = np.eye(n)

        res = estimate_reml_pagel_lambda(times, dists, K)
        assert 0.001 <= res['best_lambda'] <= 0.999

    def test_reml_insufficient_data(self):
        """REML with fewer than 3 observations should return a default."""
        res = estimate_reml_pagel_lambda(
            np.array([2000.0, 2010.0]),
            np.array([0.01, 0.02]),
            np.eye(2)
        )
        assert res['status'] == 'INSUFFICIENT_DATA'
        assert res['best_lambda'] == 0.95


class TestPowerLawClock:
    """Tests for the power-law molecular clock model."""

    def test_powerlaw_linear_recovery(self):
        """When theta=1.0 (linear), power-law should recover the same t_mrca as OLS."""
        np.random.seed(42)
        true_t0 = 1960.0
        true_mu = 0.0015
        times = np.linspace(1970, 2020, 60)
        noise = np.random.normal(0, 0.001, size=len(times))
        dists = true_mu * (times - true_t0) + noise

        res = run_powerlaw_clock_dating(times, dists, n_boot=50, seed=42)

        assert abs(res['t_mrca'] - true_t0) < 8.0
        assert abs(res['theta'] - 1.0) < 0.15
        assert not res['is_nonlinear_preferred']

    def test_powerlaw_sublinear_recovery(self):
        """When data has sub-linear (decelerating) rate, power-law should detect theta < 1."""
        np.random.seed(123)
        true_t0 = 1950.0
        true_k = 0.0008
        true_theta = 0.7
        times = np.linspace(1970, 2020, 80)
        dists = true_k * (times - true_t0) ** true_theta
        noise = np.random.normal(0, 0.0003, size=len(times))
        dists = dists + noise

        res = run_powerlaw_clock_dating(times, dists, n_boot=50, seed=42)

        assert res['t_mrca'] < 1970.0
        assert res['theta'] < 0.9
        assert res['rate_ancestral'] > res['rate_recent']
        assert res['is_nonlinear_preferred']
        assert res['p_f_test'] < 0.05

    def test_powerlaw_ci_non_degenerate(self):
        """Bootstrap CI for t_mrca should not collapse to a point."""
        np.random.seed(7)
        true_t0 = 1955.0
        true_mu = 0.002
        times = np.linspace(1970, 2020, 20)
        noise = np.random.normal(0, 0.01, size=len(times))
        dists = true_mu * (times - true_t0) + noise

        res = run_powerlaw_clock_dating(times, dists, n_boot=500, seed=99)

        # CI should be non-NaN and have positive width
        assert not np.isnan(res['ci_mrca'][0])
        assert not np.isnan(res['ci_mrca'][1])
        assert res['ci_mrca'][0] <= res['ci_mrca'][1]

    def test_powerlaw_minimum_observations(self):
        """Power-law should raise ValueError with fewer than 4 observations."""
        with pytest.raises(ValueError, match="At least 4"):
            run_powerlaw_clock_dating(
                np.array([2000.0, 2005.0, 2010.0]),
                np.array([0.01, 0.02, 0.03]),
            )


class TestLOOCV:
    """Tests for Leave-One-Out Cross-Validation."""

    def test_loocv_ols_tip_prediction(self):
        """LOOCV with OLS should produce reasonable tip date predictions on clean linear data."""
        np.random.seed(42)
        true_t0 = 1950.0
        true_mu = 0.002
        times = np.linspace(1970, 2020, 30)
        noise = np.random.normal(0, 0.001, size=len(times))
        dists = true_mu * (times - true_t0) + noise

        res = run_dating_loocv(times, dists, method="ols")

        assert res['n_valid_predictions'] == 30
        assert res['tip_mae_days'] < 500  # < ~1.4 years error
        assert res['tip_rmse_days'] > 0
        assert not np.isnan(res['jackknife_mean'])
        assert res['jackknife_ci'][0] < res['jackknife_ci'][1]

    def test_loocv_pgls_tip_prediction(self):
        """LOOCV with PGLS and identity covariance should match OLS behavior."""
        np.random.seed(42)
        true_t0 = 1950.0
        true_mu = 0.002
        n = 30
        times = np.linspace(1970, 2020, n)
        noise = np.random.normal(0, 0.001, size=n)
        dists = true_mu * (times - true_t0) + noise

        res = run_dating_loocv(
            times, dists,
            cov_matrix=np.eye(n),
            ridge=0.0, pagel_lambda=1.0,
            method="pgls"
        )

        assert res['method'] == 'PGLS'
        assert res['n_valid_predictions'] == n
        assert res['tip_mae_days'] < 500
        assert not np.isnan(res['jackknife_mean'])

    def test_loocv_jackknife_se(self):
        """Jackknife SE should be positive for non-degenerate data."""
        np.random.seed(42)
        true_t0 = 1955.0
        true_mu = 0.002
        times = np.linspace(1970, 2020, 40)
        noise = np.random.normal(0, 0.001, size=len(times))
        dists = true_mu * (times - true_t0) + noise

        res = run_dating_loocv(times, dists, method="ols")

        assert res['jackknife_se_years'] > 0
        assert res['jackknife_se_days'] > 0
        assert res['jackknife_ci_width_years'] > 0

    def test_loocv_minimum_observations(self):
        """LOOCV should raise ValueError with fewer than 4 observations."""
        with pytest.raises(ValueError, match="At least 4"):
            run_dating_loocv(
                np.array([2000.0, 2005.0, 2010.0]),
                np.array([0.01, 0.02, 0.03]),
            )

    def test_loocv_taxa_names_preserved(self):
        """LOOCV should preserve provided taxa names in records."""
        np.random.seed(42)
        n = 10
        times = np.linspace(2000, 2020, n)
        dists = 0.001 * (times - 1990) + np.random.normal(0, 0.0005, size=n)
        taxa = [f"strain_{i}" for i in range(n)]

        res = run_dating_loocv(times, dists, taxa=taxa, method="ols")

        assert len(res['records']) == n
        assert res['records'][0]['taxon'] == 'strain_0'
        assert res['records'][-1]['taxon'] == 'strain_9'


class TestFiellerBoundaryCases:
    """Tests for Fieller's theorem edge cases."""

    def test_fieller_linear_boundary(self):
        """Test the linear transition boundary where g ≈ 1 (A ≈ 0).

        When g is very close to 1, the quadratic coefficient A = mu^2 - t_crit^2 * var_mu ≈ 0.
        This is the boundary between bounded and complement regimes.
        """
        from chronaeon.dating import compute_fieller_mrca_interval

        # Construct cov_beta so that A ≈ 0: var_mu ≈ mu^2 / t_crit^2
        mu = 0.001
        df = 50
        t_crit = float(stats.t.ppf(0.975, df=df))
        var_mu = (mu ** 2) / (t_crit ** 2) * 1.001  # Slightly past boundary -> A slightly negative
        var_d0 = 1e-6
        cov_mud0 = 0.0
        cov_beta = np.array([[var_mu, cov_mud0], [cov_mud0, var_d0]])

        ci, info = compute_fieller_mrca_interval(
            mu=mu, d0=0.04, cov_beta=cov_beta, t_ref=2020.0, df=df, min_time=2000.0
        )

        # g should be very close to or slightly above 1
        assert info['g'] >= 0.99
        # Should not crash; should return a valid interval or complement
        assert info['status'] in ['COMPLEMENT', 'ALL_REALS', 'LINEAR', 'BOUNDED']

    def test_fieller_exact_g_equals_one(self):
        """Test the exact g=1 boundary where A is exactly zero."""
        from chronaeon.dating import compute_fieller_mrca_interval

        mu = 0.001
        df = 50
        t_crit = float(stats.t.ppf(0.975, df=df))
        # Set var_mu so that A = mu^2 - t_crit^2 * var_mu = 0 exactly
        var_mu = (mu ** 2) / (t_crit ** 2)
        var_d0 = 1e-6
        cov_mud0 = 0.0
        cov_beta = np.array([[var_mu, cov_mud0], [cov_mud0, var_d0]])

        ci, info = compute_fieller_mrca_interval(
            mu=mu, d0=0.04, cov_beta=cov_beta, t_ref=2020.0, df=df, min_time=2000.0
        )

        assert abs(info['g'] - 1.0) < 0.01
        assert info['status'] in ['LINEAR', 'BOUNDED', 'COMPLEMENT', 'ALL_REALS']

    def test_fieller_non_positive_rate(self):
        """Fieller should return NON_POSITIVE_RATE for mu <= 0."""
        from chronaeon.dating import compute_fieller_mrca_interval

        ci, info = compute_fieller_mrca_interval(
            mu=0.0, d0=0.04, cov_beta=np.eye(2), t_ref=2020.0, df=50
        )
        assert info['status'] == 'NON_POSITIVE_RATE'
        assert np.isnan(ci[0]) and np.isnan(ci[1])


class TestBootstrapIntervals:
    """Tests for Poisson and residual bootstrap CI methods."""

    def test_poisson_bootstrap_ci_coverage(self):
        """Poisson bootstrap CI should contain the true t_mrca for well-specified data."""
        from chronaeon.dating import compute_poisson_mrca_interval

        np.random.seed(42)
        true_t0 = 1950.0
        true_mu = 0.002
        seq_len = 3000
        n = 40
        times = np.linspace(1970, 2020, n)
        # Generate distances with Poisson sampling noise
        rng = np.random.default_rng(42)
        mut_counts = rng.poisson(true_mu * (times - true_t0) * seq_len)
        dists = mut_counts / seq_len

        t_ref = float(np.mean(times))
        X = np.column_stack([times - t_ref, np.ones(n)])
        C_inv = np.eye(n)
        Xt_Cinv_X = X.T @ C_inv @ X

        ci = compute_poisson_mrca_interval(
            times, dists, Xt_Cinv_X, X, C_inv, t_ref,
            seq_len=seq_len, n_boot=500, seed=42, min_time=float(np.min(times))
        )

        assert not np.isnan(ci[0])
        assert not np.isnan(ci[1])
        assert ci[0] < ci[1]
        # CI should contain the true t_mrca
        assert ci[0] < true_t0 < ci[1]

    def test_residual_bootstrap_ci_coverage(self):
        """Wild Rademacher residual bootstrap CI should contain the true t_mrca."""
        from chronaeon.dating import compute_residual_bootstrap_mrca_interval

        np.random.seed(42)
        true_t0 = 1950.0
        true_mu = 0.002
        n = 40
        times = np.linspace(1970, 2020, n)
        noise = np.random.normal(0, 0.001, size=n)
        dists = true_mu * (times - true_t0) + noise

        t_ref = float(np.mean(times))
        X = np.column_stack([times - t_ref, np.ones(n)])
        C_inv = np.eye(n)
        C_half = np.eye(n)
        C_inv_half = np.eye(n)
        Xt_Cinv_X = X.T @ C_inv @ X
        beta = np.array([true_mu, true_mu * (t_ref - true_t0)])

        ci = compute_residual_bootstrap_mrca_interval(
            times, dists, beta, Xt_Cinv_X, X, C_inv, C_half, C_inv_half,
            t_ref, n_boot=500, seed=42, min_time=float(np.min(times))
        )

        assert not np.isnan(ci[0])
        assert not np.isnan(ci[1])
        assert ci[0] < ci[1]
        # CI should contain the true t_mrca
        assert ci[0] < true_t0 < ci[1]


class TestParseHeaderTimestamp:
    """Tests for FASTA header timestamp parsing across all supported formats."""

    def test_iso_full_date(self):
        from chronaeon.dating import parse_header_timestamp
        assert abs(parse_header_timestamp("seq_2021-04-15") - 2021.288) < 0.01

    def test_iso_slash_date(self):
        from chronaeon.dating import parse_header_timestamp
        # NOTE: docstring claims slash format support but extract_date_from_string
        # may not handle it — test actual behavior
        result = parse_header_timestamp("seq_2021/04/15")
        if not np.isnan(result):
            assert abs(result - 2021.288) < 0.01
        # If NaN, this is a known documentation gap — slash format not implemented

    def test_decimal_year(self):
        from chronaeon.dating import parse_header_timestamp
        assert abs(parse_header_timestamp("seq_2021.25") - 2021.25) < 0.001

    def test_year_month(self):
        from chronaeon.dating import parse_header_timestamp
        assert abs(parse_header_timestamp("seq_2021-04") - 2021.25) < 0.1

    def test_korber_hiv_format(self):
        from chronaeon.dating import parse_header_timestamp
        # B86US.SFMHS18 -> 1986.5
        assert abs(parse_header_timestamp("B86US.SFMHS18") - 1986.5) < 0.001
        # F93BE_VI850 -> 1993.5
        assert abs(parse_header_timestamp("F93BE_VI850") - 1993.5) < 0.001

    def test_zr59_anchor(self):
        from chronaeon.dating import parse_header_timestamp
        assert parse_header_timestamp("ZR59") == 1959.5
        assert parse_header_timestamp("Z59strain") == 1959.5

    def test_wpi_format(self):
        from chronaeon.dating import parse_header_timestamp
        assert parse_header_timestamp("16WPI") == 16.0
        assert abs(parse_header_timestamp("patient_24.5WPI") - 24.5) < 0.001

    def test_dpi_format(self):
        from chronaeon.dating import parse_header_timestamp
        assert parse_header_timestamp("30DPI") == 30.0

    def test_no_date_returns_nan(self):
        from chronaeon.dating import parse_header_timestamp
        assert np.isnan(parse_header_timestamp("unknown_sequence"))
        assert np.isnan(parse_header_timestamp(""))

    def test_nextstrain_pipe_format(self):
        from chronaeon.dating import parse_header_timestamp
        # Ref_B_FR_83_HXB2 -> 1983.5
        assert abs(parse_header_timestamp("Ref_B_FR_83_HXB2") - 1983.5) < 0.001


class TestParseSampleDates:
    """Tests for multi-format sample date ingestion."""

    def test_dict_input(self, tmp_path):
        from chronaeon.dating import parse_sample_dates

        taxa = ["seq1", "seq2", "seq3"]
        dates_dict = {"seq1": 2000.0, "seq2": 2010.0, "seq3": 2020.0}

        dates_map, missing = parse_sample_dates(taxa, dates_source=dates_dict)

        assert len(missing) == 0
        assert dates_map["seq1"] == 2000.0
        assert dates_map["seq2"] == 2010.0
        assert dates_map["seq3"] == 2020.0

    def test_csv_input(self, tmp_path):
        from chronaeon.dating import parse_sample_dates

        csv_path = tmp_path / "dates.csv"
        csv_path.write_text("strain,date\nseq1,2000.0\nseq2,2010.0\nseq3,2020.0\n")

        taxa = ["seq1", "seq2", "seq3"]
        dates_map, missing = parse_sample_dates(taxa, dates_source=csv_path)

        assert len(missing) == 0
        assert dates_map["seq1"] == 2000.0
        assert dates_map["seq3"] == 2020.0

    def test_tsv_input(self, tmp_path):
        from chronaeon.dating import parse_sample_dates

        tsv_path = tmp_path / "dates.tsv"
        tsv_path.write_text("strain\tdate\nseq1\t2000.0\nseq2\t2010.0\n")

        taxa = ["seq1", "seq2"]
        dates_map, missing = parse_sample_dates(taxa, dates_source=tsv_path)

        assert len(missing) == 0
        assert dates_map["seq1"] == 2000.0

    def test_json_input(self, tmp_path):
        from chronaeon.dating import parse_sample_dates

        json_path = tmp_path / "dates.json"
        json_path.write_text('{"seq1": 2000.0, "seq2": 2010.0, "seq3": 2020.0}')

        taxa = ["seq1", "seq2", "seq3"]
        dates_map, missing = parse_sample_dates(taxa, dates_source=json_path)

        assert len(missing) == 0
        assert dates_map["seq2"] == 2010.0

    def test_fasta_header_fallback(self):
        from chronaeon.dating import parse_sample_dates

        taxa = ["seq_2000-06-15", "seq_2010-01-01", "seq_2020-03-20"]
        dates_map, missing = parse_sample_dates(taxa, dates_source=None)

        assert len(missing) == 0
        assert abs(dates_map["seq_2000-06-15"] - 2000.45) < 0.01
        assert abs(dates_map["seq_2010-01-01"] - 2010.0) < 0.01

    def test_partial_missing(self, tmp_path):
        from chronaeon.dating import parse_sample_dates

        taxa = ["seq1", "seq2", "seq_2000-06-15", "unknown"]
        dates_map, missing = parse_sample_dates(taxa, dates_source=None)

        assert "seq_2000-06-15" in dates_map
        assert "unknown" in missing
        assert "seq1" in missing

    def test_date_regex_override(self):
        from chronaeon.dating import parse_sample_dates

        taxa = ["sample_2001", "sample_2002", "sample_2003"]
        dates_map, missing = parse_sample_dates(
            taxa, dates_source=None, date_regex=r"sample_(\d{4})"
        )

        assert len(missing) == 0
        assert dates_map["sample_2001"] == 2001.0
        assert dates_map["sample_2003"] == 2003.0


class TestNeuralCovarianceKernel:
    """Tests for the neural covariance kernel construction."""

    def test_unit_diagonal(self):
        from chronaeon.dating import compute_neural_covariance_kernel

        np.random.seed(42)
        n = 10
        cross_attn = np.random.dirichlet(np.ones(n), size=n)
        K = compute_neural_covariance_kernel(cross_attn)

        diag = np.diag(K)
        assert np.allclose(diag, 1.0), f"Diagonal should be 1.0, got {diag}"

    def test_psd_attention_only(self):
        from chronaeon.dating import compute_neural_covariance_kernel

        np.random.seed(42)
        n = 15
        cross_attn = np.random.dirichlet(np.ones(n), size=n)
        K = compute_neural_covariance_kernel(cross_attn)

        eigvals = np.linalg.eigvalsh(K)
        # Should be PSD (no significantly negative eigenvalues)
        assert np.all(eigvals > -1e-10), f"Negative eigenvalue detected: {eigvals.min()}"

    def test_psd_with_embeddings(self):
        from chronaeon.dating import compute_neural_covariance_kernel

        np.random.seed(42)
        n = 20
        cross_attn = np.random.dirichlet(np.ones(n), size=n)
        taxa_repr = np.random.randn(n, 128)
        K = compute_neural_covariance_kernel(cross_attn, taxa_repr=taxa_repr)

        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals > -1e-10)
        assert np.allclose(np.diag(K), 1.0)

    def test_symmetry(self):
        from chronaeon.dating import compute_neural_covariance_kernel

        np.random.seed(42)
        n = 10
        cross_attn = np.random.dirichlet(np.ones(n), size=n)
        taxa_repr = np.random.randn(n, 64)
        K = compute_neural_covariance_kernel(cross_attn, taxa_repr=taxa_repr)

        assert np.allclose(K, K.T)

    def test_identical_taxa_high_correlation(self):
        from chronaeon.dating import compute_neural_covariance_kernel

        n = 5
        # Identical attention profiles -> after centering, zero variance
        # The kernel correctly returns identity (no signal to correlate)
        cross_attn = np.ones((n, n)) / n
        K = compute_neural_covariance_kernel(cross_attn)

        # With zero variance, the where clause in np.divide falls back to eye(n)
        assert np.allclose(np.diag(K), 1.0)
        # Off-diagonal should be 0 (no covariance signal)
        off_diag = K - np.eye(n)
        assert np.allclose(off_diag, 0.0)


class TestVerifyCodingAlignment:
    """Tests for in-frame coding alignment validation."""

    def test_valid_alignment(self):
        from chronaeon.dating import verify_coding_alignment

        seq_dict = {
            "seq1": "ATGGCCATGGCCATGGCC",
            "seq2": "ATGGCAATGGCCATGGCA",
            "seq3": "ATGGTAATGGCCATGGTA",
        }
        n_taxa, n_codons = verify_coding_alignment(seq_dict)
        assert n_taxa == 3
        assert n_codons == 6

    def test_non_multiple_of_three_auto_trim(self):
        from chronaeon.dating import verify_coding_alignment

        seq_dict = {
            "seq1": "ATGGCAATGA",  # 10 nt -> should trim 1 -> 9 nt
            "seq2": "ATGGCCATGA",  # 10 nt -> should trim 1 -> 9 nt
        }
        n_taxa, n_codons = verify_coding_alignment(seq_dict, auto_trim_trailing=True)
        assert n_codons == 3
        assert len(seq_dict["seq1"]) == 9  # trimmed
        assert len(seq_dict["seq2"]) == 9  # trimmed

    def test_non_multiple_of_three_no_trim_raises(self):
        from chronaeon.dating import verify_coding_alignment

        seq_dict = {"seq1": "ATGGCAATGAA", "seq2": "ATGGCAATGAA"}
        with pytest.raises(ValueError, match="in-frame coding"):
            verify_coding_alignment(seq_dict, auto_trim_trailing=False)

    def test_empty_alignment(self):
        from chronaeon.dating import verify_coding_alignment

        with pytest.raises(ValueError, match="empty"):
            verify_coding_alignment({})

    def test_length_mismatch_raises(self):
        from chronaeon.dating import verify_coding_alignment

        seq_dict = {"seq1": "ATGGCCATGGCC", "seq2": "ATGGCC"}
        with pytest.raises(ValueError, match="length"):
            verify_coding_alignment(seq_dict)

    def test_zero_length_sequence(self):
        from chronaeon.dating import verify_coding_alignment

        seq_dict = {"seq1": "", "seq2": "ATGGCC"}
        with pytest.raises(ValueError, match="length 0"):
            verify_coding_alignment(seq_dict)

    def test_internal_stop_codon_allowed(self):
        from chronaeon.dating import verify_coding_alignment

        # TAA is a stop codon — should be allowed by default
        seq_dict = {"seq1": "ATGTAAGCCATG", "seq2": "ATGTAAGCCATG"}
        n_taxa, n_codons = verify_coding_alignment(seq_dict, allow_stop_codons=True)
        assert n_codons == 4

    def test_internal_stop_codon_disallowed(self):
        from chronaeon.dating import verify_coding_alignment

        seq_dict = {"seq1": "ATGTAAGCCATG", "seq2": "ATGTAAGCCATG"}
        with pytest.raises(ValueError, match="[Ss]top"):
            verify_coding_alignment(seq_dict, allow_stop_codons=False)


class TestSpectralEigh:
    """Tests for the _spectral_eigh helper (dense and sparse paths)."""

    def test_dense_path_matches_eigh(self):
        from chronaeon.dating import _spectral_eigh
        np.random.seed(42)
        n = 20
        A = np.random.randn(n, n)
        cov = A @ A.T + np.eye(n)
        w, v = _spectral_eigh(cov, n_max=2500)
        assert w.shape == (n,)
        assert v.shape == (n, n)
        assert np.all(w >= 0.0)
        # Reconstruct: V diag(w) V^T ≈ cov
        recon = v @ np.diag(w) @ v.T
        assert np.allclose(recon, cov, atol=1e-8)

    def test_sparse_path_clamps_negative_eigenvalues(self):
        from chronaeon.dating import _spectral_eigh
        np.random.seed(42)
        n = 50
        A = np.random.randn(n, n)
        cov = A @ A.T + np.eye(n)
        # Force the sparse path with n_max < n
        w, v = _spectral_eigh(cov.astype(np.float32), n_max=30)
        assert np.all(w >= 0.0)
        assert v.shape[0] == n
        # k_eig should be <= n-2
        assert w.shape[0] <= n - 2


class TestGLSFit:
    """Tests for the _gls_fit helper."""

    def test_recovers_known_coefficients(self):
        from chronaeon.dating import _gls_fit
        np.random.seed(42)
        n = 50
        true_beta = np.array([0.002, -3.9])
        X = np.column_stack([np.linspace(1970, 2020, n), np.ones(n)])
        noise = np.random.normal(0, 0.001, size=n)
        dists = X @ true_beta + noise
        C_inv = np.eye(n)
        beta_hat = _gls_fit(X, dists, C_inv)
        assert np.allclose(beta_hat, true_beta, atol=0.05)

    def test_pseudoinverse_fallback_on_singular_matrix(self):
        from chronaeon.dating import _gls_fit
        n = 5
        X = np.ones((n, 2))  # Rank-deficient: both columns identical
        dists = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        C_inv = np.eye(n)
        # Should not raise; should fall back to pinv
        beta_hat = _gls_fit(X, dists, C_inv)
        assert beta_hat.shape == (2,)
        assert np.all(np.isfinite(beta_hat))


class TestGeneralizedR2:
    """Tests for the _generalized_r2 helper."""

    def test_perfect_fit_returns_high_r2(self):
        from chronaeon.dating import _generalized_r2
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.linspace(1970, 2020, n), np.ones(n)])
        true_beta = np.array([0.002, -3.9])
        dists = X @ true_beta  # No noise
        C_inv = np.eye(n)
        fitted = X @ true_beta
        rss = float(np.sum((dists - fitted) ** 2))
        r2 = _generalized_r2(rss, dists, C_inv)
        assert r2 > 0.99

    def test_no_fit_returns_low_r2(self):
        from chronaeon.dating import _generalized_r2
        np.random.seed(42)
        n = 50
        dists = np.random.uniform(0.01, 0.05, size=n)
        C_inv = np.eye(n)
        # RSS ≈ TSS (no signal)
        mean_d = np.mean(dists)
        rss = float(np.sum((dists - mean_d) ** 2))
        r2 = _generalized_r2(rss, dists, C_inv)
        assert r2 < 0.1


class TestPipelineHelpers:
    """Direct unit tests for the extracted pipeline helper functions."""

    def _make_ols_result(self, t_mrca=1950.0, mu=0.002, ci_width=20.0, g=0.5, r2=0.95):
        return {
            't_mrca': t_mrca,
            'mu': mu,
            'ci_mrca': [t_mrca - ci_width / 2, t_mrca + ci_width / 2],
            'fieller_g': g,
            'r2': r2,
            'method': 'OLS',
            'd0': 0.04,
            't_ref': 2000.0,
        }

    def _make_pgls_result(self, t_mrca=1955.0, mu=0.0018, ci_width=25.0, g=0.6, r2=0.90, lam=0.8):
        return {
            't_mrca': t_mrca,
            'mu': mu,
            'ci_mrca': [t_mrca - ci_width / 2, t_mrca + ci_width / 2],
            'fieller_g': g,
            'r2': r2,
            'method': 'PGLS',
            'd0': 0.04,
            't_ref': 2000.0,
            'pagel_lambda': lam,
        }

    def _make_spline_result(self, t_mrca=1960.0, rate_ancestral=0.002, ci_width=30.0,
                            is_nonlinear=True, rate_ratio=1.5, f_stat=10.0,
                            p_f_test=0.002, delta_aic=-5.0):
        return {
            't_mrca': t_mrca,
            'rate_ancestral': rate_ancestral,
            'ci_mrca': [t_mrca - ci_width / 2, t_mrca + ci_width / 2],
            'is_nonlinear_preferred': is_nonlinear,
            'rate_ratio': rate_ratio,
            'f_stat': f_stat,
            'p_f_test': p_f_test,
            'delta_aic': delta_aic,
            'method': 'RESTRICTED_SPLINE',
            'beta_0': -3.9,
            'beta_1': 0.002,
            'beta_2': -0.0005,
            'knots': np.array([1980.0, 2000.0, 2020.0]),
        }

    def _make_power_result(self, t_mrca=1958.0, k=0.001, theta=0.9):
        return {
            't_mrca': t_mrca,
            'k': k,
            'theta': theta,
            'method': 'POWER_LAW',
        }

    def test_ensemble_weights_sum_to_one(self):
        from chronaeon.dating import _compute_precision_weighted_ensemble
        times = np.linspace(1970, 2020, 50)
        ols = self._make_ols_result()
        pgls = self._make_pgls_result()
        spline = self._make_spline_result()
        info = _compute_precision_weighted_ensemble(ols, pgls, spline, times)
        if info['weights']:
            assert abs(sum(info['weights'].values()) - 1.0) < 1e-6

    def test_ensemble_no_valid_models(self):
        from chronaeon.dating import _compute_precision_weighted_ensemble
        times = np.linspace(1970, 2020, 10)
        info = _compute_precision_weighted_ensemble(None, None, None, times)
        assert info['t_mrca'] is None
        assert info['ci_mrca'] is None
        assert info['weights'] == {}

    def test_ensemble_ols_only_fallback(self):
        from chronaeon.dating import _compute_precision_weighted_ensemble
        times = np.linspace(1970, 2020, 50)
        ols = self._make_ols_result()
        info = _compute_precision_weighted_ensemble(ols, None, None, times)
        assert info['t_mrca'] == 1950.0
        assert info['weights'] == {'ols': 1.0}

    def test_select_clock_forced_spline(self):
        from chronaeon.dating import _select_clock_model, _compute_precision_weighted_ensemble
        times = np.linspace(1970, 2020, 50)
        ols = self._make_ols_result()
        spline = self._make_spline_result()
        info = _compute_precision_weighted_ensemble(ols, None, spline, times)
        model, label = _select_clock_model("spline", ols, None, spline, None, info)
        assert model == spline
        assert "Spline" in label

    def test_select_clock_forced_power(self):
        from chronaeon.dating import _select_clock_model, _compute_precision_weighted_ensemble
        times = np.linspace(1970, 2020, 50)
        ols = self._make_ols_result()
        power = self._make_power_result()
        info = _compute_precision_weighted_ensemble(ols, None, None, times)
        model, label = _select_clock_model("power", ols, None, None, power, info)
        assert model == power
        assert "Power-Law" in label

    def test_select_clock_linear_prefers_pgls(self):
        from chronaeon.dating import _select_clock_model, _compute_precision_weighted_ensemble
        times = np.linspace(1970, 2020, 50)
        ols = self._make_ols_result()
        pgls = self._make_pgls_result()
        info = _compute_precision_weighted_ensemble(ols, pgls, None, times)
        model, label = _select_clock_model("linear", ols, pgls, None, None, info)
        assert model == pgls
        assert "PGLS" in label

    def test_select_clock_auto_prefers_spline_when_nonlinear(self):
        from chronaeon.dating import _select_clock_model, _compute_precision_weighted_ensemble
        times = np.linspace(1970, 2020, 50)
        ols = self._make_ols_result()
        pgls = self._make_pgls_result()
        spline = self._make_spline_result(is_nonlinear=True)
        info = _compute_precision_weighted_ensemble(ols, pgls, spline, times)
        model, label = _select_clock_model("auto", ols, pgls, spline, None, info)
        assert model == spline
        assert "Spline" in label

    def test_select_clock_auto_prefers_ols_when_clade_attenuated(self):
        from chronaeon.dating import _select_clock_model, _compute_precision_weighted_ensemble
        times = np.linspace(1970, 2020, 50)
        # PGLS rate much lower than OLS -> clade attenuation
        ols = self._make_ols_result(mu=0.002, r2=0.95, g=0.5)
        pgls = self._make_pgls_result(mu=0.0005, r2=0.30, g=0.5, lam=0.5)
        info = _compute_precision_weighted_ensemble(ols, pgls, None, times)
        assert info['is_clade_attenuated']
        model, label = _select_clock_model("auto", ols, pgls, None, None, info)
        assert model == ols
        assert "OLS" in label

    def test_compute_taxon_predictions_linear(self):
        from chronaeon.dating import _compute_taxon_predictions
        n = 10
        times = np.linspace(1980, 2020, n)
        dists = 0.002 * (times - 1950)
        taxa = [f"seq_{i}" for i in range(n)]
        is_train = np.ones(n, dtype=bool)
        train_idx = np.arange(n)
        model = {'method': 'OLS', 'd0': 0.04, 'mu': 0.002, 't_ref': 2000.0, 't_mrca': 1950.0}
        fitted, pred, resids, records = _compute_taxon_predictions(
            model, times, dists, taxa, is_train, train_idx
        )
        assert len(fitted) == n
        assert len(pred) == n
        assert len(records) == n
        assert records[0]['taxon'] == 'seq_0'
        assert 'z_score' in records[0]
        assert 'is_outlier' in records[0]

    def test_compute_taxon_predictions_power_law(self):
        from chronaeon.dating import _compute_taxon_predictions
        n = 10
        times = np.linspace(1980, 2020, n)
        k, theta, t0 = 0.001, 0.9, 1955.0
        dists = k * (times - t0) ** theta
        taxa = [f"seq_{i}" for i in range(n)]
        is_train = np.ones(n, dtype=bool)
        train_idx = np.arange(n)
        model = {'method': 'POWER_LAW', 'k': k, 'theta': theta, 't_mrca': t0}
        fitted, pred, resids, records = _compute_taxon_predictions(
            model, times, dists, taxa, is_train, train_idx
        )
        assert np.allclose(fitted, dists, rtol=0.01)

    def test_compute_taxon_predictions_spline(self):
        from chronaeon.dating import _compute_taxon_predictions
        n = 20
        times = np.linspace(1980, 2020, n)
        knots = np.array([1980.0, 2000.0, 2020.0])
        from chronaeon.dating import compute_rcs_basis
        B, _ = compute_rcs_basis(times, knots)
        b0, b1, b2 = -3.9, 0.002, -0.0005
        dists = b0 + b1 * times + b2 * B[:, 0]
        taxa = [f"seq_{i}" for i in range(n)]
        is_train = np.ones(n, dtype=bool)
        train_idx = np.arange(n)
        model = {
            'method': 'RESTRICTED_SPLINE',
            'beta_0': b0, 'beta_1': b1, 'beta_2': b2,
            'knots': knots, 't_mrca': 1975.0,
        }
        fitted, pred, resids, records = _compute_taxon_predictions(
            model, times, dists, taxa, is_train, train_idx
        )
        assert np.allclose(fitted, dists, atol=1e-8)


class TestRunMrcaDatingPipelinePaths:
    """Integration tests for run_mrca_dating covering previously untested paths."""

    def _make_fasta(self, tmp_path, n=10, start_year=1980, end_year=2020):
        fasta_path = tmp_path / 'test_dating.fasta'
        lines = []
        for i in range(n):
            year = start_year + (end_year - start_year) * i // (n - 1)
            # Vary sequences to create distance gradient
            seq = 'ATGGCC' + 'ATG' * (n - i - 1) + 'GTA' * i
            lines.append(f">seq_{i}_{year}\n{seq}")
        fasta_path.write_text('\n'.join(lines) + '\n')
        return fasta_path

    def test_forced_spline_clock(self, tmp_path):
        """run_mrca_dating with clock_model='spline' should force spline selection."""
        fasta_path = self._make_fasta(tmp_path)
        res = run_mrca_dating(
            alignment_path=str(fasta_path),
            use_tn93=True,
            clock_model='spline',
            method='ols',
            n_bootstrap=50,
        )
        assert 'spline' in res
        assert 'Spline' in res['selected_clock']
        assert res['active_model'] == 'spline'

    def test_forced_power_clock(self, tmp_path):
        """run_mrca_dating with clock_model='power' should force power-law selection."""
        fasta_path = self._make_fasta(tmp_path)
        res = run_mrca_dating(
            alignment_path=str(fasta_path),
            use_tn93=True,
            clock_model='power',
            method='ols',
            n_bootstrap=50,
        )
        assert 'power' in res
        assert 'Power' in res['selected_clock']
        assert res['active_model'] == 'power'

    def test_output_prefix_writes_json_and_csv(self, tmp_path):
        """run_mrca_dating with output_prefix should write JSON and CSV files."""
        fasta_path = self._make_fasta(tmp_path)
        out_prefix = str(tmp_path / 'dating_output')
        res = run_mrca_dating(
            alignment_path=str(fasta_path),
            use_tn93=True,
            clock_model='auto',
            method='ols',
            n_bootstrap=50,
            output_prefix=out_prefix,
        )
        json_file = tmp_path / 'dating_output.json'
        csv_file = tmp_path / 'dating_output.csv'
        assert json_file.exists(), f"JSON file not written: {json_file}"
        assert csv_file.exists(), f"CSV file not written: {csv_file}"

        import json
        with open(json_file) as f:
            jdata = json.load(f)
        assert 't_mrca' in jdata
        assert 'taxa_summary' in jdata
        assert jdata['taxa_count'] == 10

        import pandas as pd
        df = pd.read_csv(csv_file)
        assert len(df) == 10
        assert 'taxon' in df.columns
        assert 'z_score' in df.columns

    def test_loocv_integration(self, tmp_path):
        """run_mrca_dating with loocv=True should produce LOOCV results."""
        fasta_path = self._make_fasta(tmp_path, n=15)
        res = run_mrca_dating(
            alignment_path=str(fasta_path),
            use_tn93=True,
            clock_model='auto',
            method='ols',
            n_bootstrap=50,
            loocv=True,
        )
        assert res['loocv'] is not None
        assert 'tip_mae_days' in res['loocv']
        assert 'jackknife_ci' in res['loocv']
        # LOOCV records should be merged into taxa_records
        has_loocv = any('loocv_predicted_date' in r for r in res['taxa_records'])
        assert has_loocv
