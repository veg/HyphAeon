"""
Tests for hyphaeon/weights.py — HF weights discovery, download, and loading.

These tests mock the huggingface_hub API calls so they don't require network
access or a valid HF token. Tests that do require HF access are skipped if
HF_TOKEN is not set or the repo is unreachable.
"""

import os
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from hyphaeon.weights import (
    HF_REPO_ID,
    DEFAULT_VARIANT,
    get_variant_filename,
    resolve_weights_path,
    load_model_config,
    load_arch_config,
    load_weights,
    list_available_variants,
    CACHE_DIR,
)


class TestGetVariantFilename:
    def test_general_variant(self):
        assert get_variant_filename("general") == "model.safetensors"

    def test_viral_variant(self):
        assert get_variant_filename("viral") == "model.viral.safetensors"

    def test_arbitrary_variant(self):
        assert get_variant_filename("custom") == "model.custom.safetensors"


class TestResolveWeightsPath:
    def test_explicit_path_takes_precedence(self, tmp_path):
        """If --weights points to an existing file, use it directly."""
        weights_file = tmp_path / "model.pt"
        weights_file.write_text("dummy")
        result = resolve_weights_path(weights=str(weights_file), variant="general")
        assert result == str(weights_file)

    def test_nonexistent_explicit_path_falls_through(self, tmp_path):
        """If --weights points to a nonexistent file, fall through to variant resolution."""
        nonexistent = str(tmp_path / "does_not_exist.pt")
        with patch("hyphaeon.weights.hf_hub_download") as mock_dl, \
             patch("hyphaeon.weights.CACHE_DIR", tmp_path):
            mock_dl.return_value = "/fake/cache/model.safetensors"
            result = resolve_weights_path(weights=nonexistent, variant="general")
            mock_dl.assert_called_once()

    def test_cached_variant_skips_download(self, tmp_path):
        """If the variant is already in the HF cache, don't download."""
        cache_file = str(tmp_path / "model.safetensors")
        Path(cache_file).write_text("cached")
        with patch("hyphaeon.weights.CACHE_DIR", tmp_path), \
             patch("huggingface_hub.try_to_load_from_cache", return_value=cache_file), \
             patch("hyphaeon.weights.hf_hub_download") as mock_dl:
            result = resolve_weights_path(weights=None, variant="general")
            assert result == cache_file
            mock_dl.assert_not_called()

    def test_download_failure_raises_error(self, tmp_path):
        """If HF download fails and no local weights exist, raise RuntimeError."""
        with patch("hyphaeon.weights.CACHE_DIR", tmp_path), \
             patch("hyphaeon.weights.hf_hub_download", side_effect=Exception("401 Unauthorized")):
            with pytest.raises(RuntimeError, match="Could not download weights"):
                resolve_weights_path(weights=None, variant="general")


class TestListAvailableVariants:
    def test_parses_safetensors_files(self):
        """Should extract variant names from model.*.safetensors filenames."""
        mock_files = [
            ".gitattributes",
            "README.md",
            "config.json",
            "model.safetensors",
            "model.viral.safetensors",
            "model.viral.pt",
            "model.viral.onnx",
        ]
        with patch("hyphaeon.weights.list_repo_files", return_value=mock_files):
            variants = list_available_variants()
        names = [v["variant"] for v in variants]
        assert "general" in names
        assert "viral" in names
        assert len(variants) == 2  # only .safetensors files

    def test_general_listed_first(self):
        """General variant should be listed first."""
        mock_files = ["model.viral.safetensors", "model.safetensors", "config.json"]
        with patch("hyphaeon.weights.list_repo_files", return_value=mock_files):
            variants = list_available_variants()
        assert variants[0]["variant"] == "general"

    def test_no_safetensors_returns_empty(self):
        mock_files = ["config.json", "README.md"]
        with patch("hyphaeon.weights.list_repo_files", return_value=mock_files):
            variants = list_available_variants()
        assert len(variants) == 0


class TestLoadModelConfig:
    def test_loads_shared_config(self, tmp_path):
        """Should load config.json from HF when no variant-specific config exists."""
        config = {"embed_dim": 384, "num_layers": 6, "num_heads": 12}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        mock_files = ["config.json", "model.safetensors"]
        with patch("hyphaeon.weights.list_repo_files", return_value=mock_files), \
             patch("hyphaeon.weights.hf_hub_download", return_value=str(config_path)):
            result = load_model_config(variant="general")
        assert result["embed_dim"] == 384
        assert result["num_layers"] == 6

    def test_variant_specific_config_preferred(self, tmp_path):
        """If model.{variant}.config.json exists, use it over the shared config.json."""
        general_config = {"embed_dim": 384, "num_layers": 6}
        viral_config = {"embed_dim": 512, "num_layers": 8}

        general_path = tmp_path / "config.json"
        general_path.write_text(json.dumps(general_config))
        viral_path = tmp_path / "model.viral.config.json"
        viral_path.write_text(json.dumps(viral_config))

        mock_files = ["config.json", "model.safetensors", "model.viral.safetensors", "model.viral.config.json"]

        def fake_download(repo_id, filename, **kwargs):
            return str(tmp_path / filename)

        with patch("hyphaeon.weights.list_repo_files", return_value=mock_files), \
             patch("hyphaeon.weights.hf_hub_download", side_effect=fake_download):
            result = load_model_config(variant="viral")
        assert result["embed_dim"] == 512
        assert result["num_layers"] == 8

    def test_no_config_returns_empty(self):
        """If no config files exist on HF, return empty dict (CLI uses defaults)."""
        mock_files = ["model.safetensors"]
        with patch("hyphaeon.weights.list_repo_files", return_value=mock_files):
            result = load_model_config(variant="general")
        assert result == {}


class TestLoadWeights:
    def test_load_safetensors(self, tmp_path):
        """Should load a .safetensors file directly as a state_dict."""
        pytest.importorskip("safetensors")
        from safetensors.torch import save_file

        state_dict = {"weight": torch.zeros(3, 3)}
        st_path = tmp_path / "model.safetensors"
        save_file(state_dict, str(st_path))

        result = load_weights(weights=str(st_path), variant="general")
        assert "weight" in result
        assert result["weight"].shape == (3, 3)

    def test_load_pt_with_model_state_dict(self, tmp_path):
        """Should extract model_state_dict from a .pt checkpoint."""
        state_dict = {"weight": torch.ones(2, 2)}
        ckpt = {"epoch": 10, "model_state_dict": state_dict, "optimizer_state_dict": {}}
        pt_path = tmp_path / "model.pt"
        torch.save(ckpt, str(pt_path))

        result = load_weights(weights=str(pt_path), variant="general")
        assert "weight" in result
        assert result["weight"].shape == (2, 2)

    def test_load_pt_with_state_dict_key(self, tmp_path):
        """Should extract state_dict from a .pt checkpoint with 'state_dict' key."""
        state_dict = {"weight": torch.ones(2, 2)}
        ckpt = {"state_dict": state_dict}
        pt_path = tmp_path / "model.pt"
        torch.save(ckpt, str(pt_path))

        result = load_weights(weights=str(pt_path), variant="general")
        assert "weight" in result

    def test_load_pt_raw_state_dict(self, tmp_path):
        """Should handle a .pt file that is just a raw state_dict (no wrapper)."""
        state_dict = {"weight": torch.ones(2, 2)}
        pt_path = tmp_path / "model.pt"
        torch.save(state_dict, str(pt_path))

        result = load_weights(weights=str(pt_path), variant="general")
        assert "weight" in result


class TestLoadArchConfig:
    def test_load_from_pt_checkpoint(self, tmp_path):
        """Should extract architecture config from a .pt checkpoint's args dict."""
        state_dict = {"weight": torch.ones(2, 2)}
        ckpt = {"args": {"embed_dim": 512, "layers": 8, "heads": 16, "window_size": 2},
                "model_state_dict": state_dict}
        pt_path = tmp_path / "model.pt"
        torch.save(ckpt, str(pt_path))

        config = load_arch_config(weights=str(pt_path))
        assert config["embed_dim"] == 512
        assert config["num_layers"] == 8
        assert config["num_heads"] == 16
        assert config["window_size"] == 2

    def test_pt_with_num_layers_key(self, tmp_path):
        """Should handle checkpoints that use 'num_layers' instead of 'layers'."""
        state_dict = {"weight": torch.ones(2, 2)}
        ckpt = {"args": {"embed_dim": 256, "num_layers": 4, "num_heads": 8, "window_size": 1},
                "model_state_dict": state_dict}
        pt_path = tmp_path / "model.pt"
        torch.save(ckpt, str(pt_path))

        config = load_arch_config(weights=str(pt_path))
        assert config["num_layers"] == 4
        assert config["num_heads"] == 8

    def test_pt_without_args_uses_defaults(self, tmp_path):
        """Should fall back to architecture defaults if .pt has no args dict."""
        state_dict = {"weight": torch.ones(2, 2)}
        pt_path = tmp_path / "model.pt"
        torch.save(state_dict, str(pt_path))

        config = load_arch_config(weights=str(pt_path))
        assert config["embed_dim"] == 384
        assert config["num_layers"] == 6


@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="HF_TOKEN not set")
class TestHFAccess:
    """Integration tests that actually hit Hugging Face. Skipped without HF_TOKEN."""

    def test_list_variants_real(self):
        variants = list_available_variants()
        assert len(variants) >= 1
        assert any(v["variant"] == "general" for v in variants)

    def test_load_config_real(self):
        config = load_model_config(variant="general")
        assert "embed_dim" in config
        assert config["embed_dim"] == 384
