"""Unit tests for BUSTED omnibus selection — BustedMultiTaskHead and CLI.

Tests the neural head's forward pass (output shapes, ranges, keys) and the
full busted CLI subcommand end-to-end with dummy weights.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from hyphaeon.model import BustedMultiTaskHead, PhyloAxialTransformer
from hyphaeon.cli import _IS_DEV


# ---------------------------------------------------------------------------
# BustedMultiTaskHead — forward pass structure and ranges
# ---------------------------------------------------------------------------

class TestBustedMultiTaskHead:
    def test_forward_output_keys(self):
        head = BustedMultiTaskHead(embed_dim=8)
        x = torch.randn(1, 100, 8)
        out = head(x)
        expected_keys = {
            "cls_prob", "logits_lrt", "logits_logp", "logits_omega3",
            "pred_lrt", "pred_logp", "pred_omega3", "syn_var", "omega_prop",
        }
        assert set(out.keys()) == expected_keys

    def test_forward_output_ranges(self):
        head = BustedMultiTaskHead(embed_dim=8)
        x = torch.randn(1, 50, 8)
        out = head(x)
        assert 0.0 <= float(out["cls_prob"]) <= 1.0
        assert float(out["pred_lrt"]) >= 0.0
        assert float(out["pred_logp"]) >= 0.0
        assert float(out["pred_omega3"]) >= 0.0
        assert float(out["syn_var"]) >= 0.0
        props = out["omega_prop"].squeeze().detach().numpy()
        assert np.all(props >= 0.0)
        assert abs(np.sum(props) - 1.0) < 1e-5

    def test_forward_2d_input_auto_unsqueezed(self):
        head = BustedMultiTaskHead(embed_dim=8)
        x = torch.randn(100, 8)  # 2D, should be auto-unsqueezed
        out = head(x)
        assert out["cls_prob"].dim() == 0 or out["cls_prob"].shape[0] == 1

    def test_forward_batch_mode(self):
        head = BustedMultiTaskHead(embed_dim=8)
        x = torch.randn(4, 50, 8)
        out = head(x)
        assert out["cls_prob"].shape[0] == 4
        assert out["omega_prop"].shape[0] == 4

    def test_forward_finite_output(self):
        head = BustedMultiTaskHead(embed_dim=8)
        x = torch.randn(2, 80, 8)
        out = head(x)
        for key in ["cls_prob", "pred_lrt", "pred_logp", "pred_omega3", "syn_var"]:
            assert torch.isfinite(out[key]).all(), f"{key} has non-finite values"
        assert torch.isfinite(out["omega_prop"]).all()

    def test_eval_mode_no_dropout(self):
        head = BustedMultiTaskHead(embed_dim=8, dropout=0.5)
        head.eval()
        x = torch.randn(1, 50, 8)
        out1 = head(x)
        out2 = head(x)
        assert torch.allclose(out1["cls_prob"], out2["cls_prob"], atol=1e-6)

    def test_different_embed_dims(self):
        for dim in [8, 16, 32, 64]:
            head = BustedMultiTaskHead(embed_dim=dim)
            x = torch.randn(1, 30, dim)
            out = head(x)
            assert out["cls_prob"].shape[0] == 1


# ---------------------------------------------------------------------------
# BUSTED CLI — end-to-end with dummy weights
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _IS_DEV, reason="busted subcommand is gated behind dev mode")
class TestBustedCLI:
    def test_busted_single_alignment(self, examples_dir, dummy_weights, tmp_path):
        fa = os.path.join(examples_dir, "Smc6.fasta")
        nwk = os.path.join(examples_dir, "Smc6.nwk")
        out_json = str(tmp_path / "busted.json")
        out_csv = str(tmp_path / "busted.csv")

        cmd = [
            sys.executable, "-m", "hyphaeon.cli", "busted",
            "-a", fa, "-t", nwk, "-w", dummy_weights,
            "-o", out_json, "-c", out_csv,
            "--cpu",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"busted CLI failed: {result.stderr}"
        assert os.path.exists(out_json)
        assert os.path.exists(out_csv)

        with open(out_json) as f:
            data = json.load(f)
        assert "p_value_acat" in data
        assert "p_value_simes" in data
        assert "rate_distributions" in data
        assert "selection_probability" in data
        assert "positive_selection_detected" in data
        assert 0.0 <= data["p_value_acat"] <= 1.0
        assert 0.0 <= data["p_value_simes"] <= 1.0
        assert 0.0 <= data["selection_probability"] <= 1.0
        assert isinstance(data["positive_selection_detected"], bool)

        rd = data["rate_distributions"]
        for k in ["omega_1", "omega_2", "omega_3", "proportion_1", "proportion_2", "proportion_3"]:
            assert k in rd

        df = pd.read_csv(out_csv)
        assert "p_ACAT" in df.columns
        assert "p_Simes" in df.columns
        assert "Omnibus_LRT" in df.columns
        assert len(df) == 1

    def test_busted_batch_directory(self, examples_dir, dummy_weights, tmp_path):
        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()
        for name in ["Smc6", "bat_oas1"]:
            fa_src = os.path.join(examples_dir, f"{name}.fasta")
            nwk_src = os.path.join(examples_dir, f"{name}.nwk")
            if os.path.exists(fa_src) and os.path.exists(nwk_src):
                import shutil
                shutil.copy(fa_src, batch_dir / f"{name}.fasta")
                shutil.copy(nwk_src, batch_dir / f"{name}.nwk")

        out_json = str(tmp_path / "batch_busted.json")
        out_csv = str(tmp_path / "batch_busted.csv")

        cmd = [
            sys.executable, "-m", "hyphaeon.cli", "busted",
            "-d", str(batch_dir),
            "--pattern", "*.fasta",
            "--tree-suffix", ".nwk",
            "-w", dummy_weights,
            "-o", out_json, "-c", out_csv,
            "--cpu",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"busted batch CLI failed: {result.stderr}"
        assert os.path.exists(out_json)

        with open(out_json) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) >= 1
        for record in data:
            assert "p_value_acat" in record
            assert "gene" in record

        df = pd.read_csv(out_csv)
        assert len(df) == len(data)
