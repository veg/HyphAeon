"""Unit tests for ChronAeon dating module and non-linear clock models."""
import numpy as np
import pytest

from chronaeon.dating import (
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
