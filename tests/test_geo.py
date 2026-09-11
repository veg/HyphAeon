"""Unit tests for HyphAeon Geo discrete phylogeography module."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from Bio import Phylo

from hyphaeon.geo import (
    parse_geo_metadata,
    reconstruct_ancestral_states_parsimony,
    compute_attention_migration_flux,
    run_permutation_bssvs,
    estimate_spatial_pgls_epicenter,
    generate_geojson,
    run_phylogeography_analysis,
)


class TestHyphaeonGeo:
    def test_parse_geo_metadata(self, tmp_path):
        csv_p = tmp_path / "metadata.csv"
        csv_p.write_text(
            "taxon,location,date,lat,lon\n"
            "seq1,Guangdong,2001.0,23.1,113.2\n"
            "seq2,Guangdong,2002.0,23.1,113.2\n"
            "seq3,Hunan,2003.0,28.1,113.0\n"
            "seq4,Fujian,2004.0,26.1,119.3\n"
        )
        taxa = ["seq1", "seq2", "seq3", "seq4"]
        df_clean, unique_locs, centroids = parse_geo_metadata(csv_p, taxa)
        assert len(df_clean) == 4
        assert unique_locs == ["Fujian", "Guangdong", "Hunan"]
        assert "Guangdong" in centroids
        assert abs(centroids["Guangdong"][0] - 23.1) < 0.1

    def test_parsimony_and_migration_routes(self, tmp_path):
        # Build synthetic tree: ((seq1:0.1, seq2:0.1):0.2, (seq3:0.1, seq4:0.1):0.2);
        tree_p = tmp_path / "test.nwk"
        tree_p.write_text("((seq1:0.1, seq2:0.1):0.2, (seq3:0.1, seq4:0.1):0.2);")
        tree = Phylo.read(str(tree_p), "newick")

        unique_locs = ["Guangdong", "Hunan"]
        loc_map = {"seq1": "Guangdong", "seq2": "Guangdong", "seq3": "Hunan", "seq4": "Hunan"}
        root_probs, transitions = reconstruct_ancestral_states_parsimony(tree, loc_map, unique_locs)

        assert len(root_probs) == 2
        assert np.isclose(root_probs.sum(), 1.0)
        assert transitions.shape == (2, 2)

    def test_permutation_bssvs_synthetic(self):
        # Create synthetic transition matrix
        K = 3
        N = 30
        unique_locs = ["LocA", "LocB", "LocC"]
        taxa = [f"seq_{i}" for i in range(N)]
        loc_indices = np.array([0] * 10 + [1] * 10 + [2] * 10)

        # Synthetic attention matrix with strong LocA -> LocB directed edge
        A = np.random.uniform(0.001, 0.005, size=(N, N))
        A[:10, 10:20] += 0.05  # LocA -> LocB strong attention
        np.fill_diagonal(A, 0.0)

        M_obs, Z, p_vals, bf_mat, routes = run_permutation_bssvs(
            tree=None,
            attn_matrix=A,
            taxa=taxa,
            loc_indices=loc_indices,
            unique_locs=unique_locs,
            n_perms=200,
            seed=42,
        )

        assert M_obs.shape == (3, 3)
        assert Z[0, 1] > 2.0  # Strong Z-score for LocA -> LocB
        assert bf_mat[0, 1] > 3.0  # Significant Bayes Factor

    def test_spatial_pgls_epicenter(self):
        # 3 points in 2D space
        coords = np.array([
            [20.0, 100.0],
            [22.0, 102.0],
            [25.0, 105.0],
        ])
        cov = np.eye(3)
        root_xy, cov_xy, radius_km = estimate_spatial_pgls_epicenter(coords, cov)
        # With identity covariance, PGLS is the arithmetic mean
        assert np.allclose(root_xy, np.mean(coords, axis=0))
        assert radius_km > 0.0

    def test_geojson_export(self):
        unique_locs = ["LocA", "LocB"]
        centroids = {"LocA": (20.0, 100.0), "LocB": (25.0, 105.0)}
        root_probs = np.array([0.8, 0.2])
        counts = np.array([10, 5])
        sig_routes = [{
            "source": "LocA",
            "target": "LocB",
            "rate": 0.05,
            "z_score": 3.5,
            "p_value": 0.001,
            "fdr_q": 0.002,
            "bayes_factor": 50.0,
            "support": "Strong (10 <= BF < 100)",
        }]
        geo_dict = generate_geojson(unique_locs, centroids, root_probs, counts, sig_routes)
        assert geo_dict["type"] == "FeatureCollection"
        assert len(geo_dict["features"]) == 3  # 2 points + 1 arc

    def test_geo_worked_example_cli(self, tmp_path):
        """Test the built-in worked example execution end-to-end."""
        from hyphaeon.cli import main
        import sys
        
        out_json = tmp_path / "example_res.json"
        out_geojson = tmp_path / "example_res.geojson"
        
        test_args = [
            "hyphaeon", "geo", "--example", "--no-neural",
            "--n-perms", "100",
            "-o", str(out_json),
            "--geojson", str(out_geojson),
        ]
        
        orig_argv = sys.argv
        try:
            sys.argv = test_args
            main()
        finally:
            sys.argv = orig_argv
            
        assert out_json.exists()
        assert out_geojson.exists()

