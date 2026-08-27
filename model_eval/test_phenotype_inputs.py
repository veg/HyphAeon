"""
Input handling tests for hyphaeon/phenotype.py (Mode II).

Tests the phenotype vector resolution paths: continuous mode, phenotype
file parsing (CSV/TSV with auto column detection), curated presets, and
the background parameter. Uses the Smc6 example dataset.

These test the resolve_phenotype_vector function's contract: given
various input modalities, it should produce a correct foreground/background
partition (discrete) or standardized trait vector (continuous).

Error handling (nonexistent foreground, missing tree) is tested in
TestPhenotypeErrorHandling in test_phenotype_outputs.py.
"""
import numpy as np
import pytest

from hyphaeon.phenotype import PRESETS, resolve_phenotype_vector
from _harness import get_taxa, fg_string


class TestContinuousMode:
    """When continuous=True with a phenotype file, the trait vector should
    be standardized (mean 0, std 1) and meta["mode"] should be "continuous".
    """

    def test_continuous_mode_flag(self, phylowas_runner, smc6_paths, tmp_path):
        fa, nwk = smc6_paths
        taxa = get_taxa(fa, nwk)

        pheno_path = str(tmp_path / "test_continuous_pheno.tsv")
        with open(pheno_path, "w") as f:
            f.write("species\tbody_mass\n")
            for i, t in enumerate(taxa):
                f.write(f"{t}\t{10.0 * (i + 1)}\n")

        result = phylowas_runner(fa, tree_path=nwk,
                                  phenotype_file=pheno_path, continuous=True)
        meta = result["phenotype_meta"]
        assert meta["mode"] == "continuous"
        assert meta["foreground_count"] == 0
        assert len(result["sites"]) > 0

    def test_continuous_standardizes_trait(self, tmp_path):
        """resolve_phenotype_vector with continuous=True should return a
        standardized y (mean ~0, std ~1) when values have variance.
        """
        taxa = [f"taxon_{i:02d}" for i in range(10)]
        pheno_path = str(tmp_path / "pheno.tsv")
        with open(pheno_path, "w") as f:
            f.write("species\tvalue\n")
            for i, t in enumerate(taxa):
                f.write(f"{t}\t{10.0 * (i + 1)}\n")

        y, meta = resolve_phenotype_vector(
            taxa=taxa, phenotype_file=pheno_path,
            continuous=True, species_col="species", trait_col="value"
        )
        assert meta["mode"] == "continuous"
        assert np.mean(y) == pytest.approx(0.0, abs=1e-6)
        assert np.std(y) == pytest.approx(1.0, abs=1e-6)


class TestPhenotypeFileParsing:
    """Phenotype file parsing: CSV, TSV, auto column detection."""

    def test_csv_with_header(self, phylowas_runner, smc6_paths, tmp_path):
        fa, nwk = smc6_paths
        taxa = get_taxa(fa, nwk)

        pheno_path = str(tmp_path / "pheno.csv")
        with open(pheno_path, "w") as f:
            f.write("species,status\n")
            for i, t in enumerate(taxa):
                status = "case" if i < 5 else "control"
                f.write(f"{t},{status}\n")

        result = phylowas_runner(fa, tree_path=nwk, phenotype_file=pheno_path)
        meta = result["phenotype_meta"]
        assert meta["foreground_count"] == 5

    def test_tsv_with_header(self, phylowas_runner, smc6_paths, tmp_path):
        fa, nwk = smc6_paths
        taxa = get_taxa(fa, nwk)

        pheno_path = str(tmp_path / "pheno.tsv")
        with open(pheno_path, "w") as f:
            f.write("species\tgroup\n")
            for i, t in enumerate(taxa):
                group = "case" if i < 3 else "control"
                f.write(f"{t}\t{group}\n")

        result = phylowas_runner(fa, tree_path=nwk, phenotype_file=pheno_path)
        meta = result["phenotype_meta"]
        assert meta["foreground_count"] == 3


class TestPresets:
    """All curated presets should resolve to valid foreground/background splits."""

    def test_all_presets_resolve(self, smc6_paths):
        fa, nwk = smc6_paths
        taxa = get_taxa(fa, nwk)

        any_matched = False
        for preset_name in PRESETS:
            y, meta = resolve_phenotype_vector(taxa=taxa, preset=preset_name)
            assert len(y) == len(taxa)
            # foreground_count >= 0 is trivially true; verify it's an int
            assert isinstance(meta["foreground_count"], int)
            if meta["foreground_count"] > 0:
                any_matched = True
        # At least one preset should match some Smc6 taxa — otherwise
        # the preset patterns are all mismatched with the alignment.
        assert any_matched, (
            "No preset matched any Smc6 taxa — preset patterns may be "
            "stale or the alignment taxon names don't match any pattern"
        )

    def test_preset_foreground_is_binary(self, smc6_paths):
        fa, nwk = smc6_paths
        taxa = get_taxa(fa, nwk)

        for preset_name in PRESETS:
            y, meta = resolve_phenotype_vector(taxa=taxa, preset=preset_name)
            unique_vals = set(np.unique(y))
            assert unique_vals.issubset({0.0, 1.0}), \
                f"Preset {preset_name} produced non-binary y: {unique_vals}"


class TestForegroundBackground:
    """Direct foreground/background string specification."""

    def test_foreground_string(self, phylowas_runner, smc6_paths):
        fa, nwk = smc6_paths
        taxa = get_taxa(fa, nwk)
        result = phylowas_runner(fa, tree_path=nwk, foreground=fg_string(taxa, 2))
        meta = result["phenotype_meta"]
        assert meta["foreground_count"] == 2

    def test_foreground_list(self, phylowas_runner, smc6_paths):
        fa, nwk = smc6_paths
        taxa = get_taxa(fa, nwk)
        result = phylowas_runner(fa, tree_path=nwk, foreground=taxa[:3])
        meta = result["phenotype_meta"]
        assert meta["foreground_count"] == 3

    def test_background_excludes_from_foreground(self, smc6_paths):
        """When background is specified, those taxa should be excluded from
        foreground and have y=0 in the result."""
        fa, nwk = smc6_paths
        taxa = get_taxa(fa, nwk)
        fg_taxa = fg_string(taxa, 2).split(",")
        bg_taxa = fg_string(taxa[2:4], 2).split(",")

        y, meta = resolve_phenotype_vector(
            taxa=taxa, foreground=fg_string(taxa, 2),
            background=fg_string(taxa[2:4], 2)
        )
        assert meta["foreground_count"] == 2
        # Verify background taxa are not in foreground (y=0)
        for t in bg_taxa:
            idx = taxa.index(t)
            assert y[idx] == 0.0, (
                f"Background taxon {t} has y={y[idx]} — should be 0"
            )
        # Verify foreground taxa have y=1
        for t in fg_taxa:
            idx = taxa.index(t)
            assert y[idx] == 1.0, (
                f"Foreground taxon {t} has y={y[idx]} — should be 1"
            )


class TestInsufficientForeground:
    """Should raise ValueError when fewer than 2 foreground taxa match."""

    def test_zero_foreground_raises(self, phylowas_runner, smc6_paths):
        fa, nwk = smc6_paths
        with pytest.raises(ValueError, match="Insufficient foreground"):
            phylowas_runner(fa, tree_path=nwk, foreground="nonexistent_taxon")

    def test_one_foreground_raises(self, phylowas_runner, smc6_paths):
        fa, nwk = smc6_paths
        taxa = get_taxa(fa, nwk)
        with pytest.raises(ValueError, match="Insufficient foreground"):
            phylowas_runner(fa, tree_path=nwk, foreground=taxa[0])
