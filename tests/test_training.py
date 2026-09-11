from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest
import torch

from hyphaeon.model import PhyloAxialTransformer, decode_soft_ordinal_lrt
from hyphaeon.training_data import TRAINING_SCHEMA_VERSION
from training.train import iter_site_indices, load_initial_checkpoint, train_epoch


def make_gene(site_count, taxon_count, targets=None, eligible=None, gene_name="gene"):
    if targets is None:
        targets = torch.arange(site_count, dtype=torch.float32)
    if eligible is None:
        eligible = torch.ones(site_count, dtype=torch.bool)
    return {
        "c": torch.randint(0, 61, (site_count, taxon_count, 1)),
        "a": torch.randint(0, 20, (site_count, taxon_count, 1)),
        "d": torch.zeros(taxon_count, taxon_count),
        "z": torch.zeros(taxon_count, 4),
        "target_lrt": targets,
        "eligible_mask": eligible,
        "gene_name": gene_name,
    }


def test_ordinal_decoder_is_differentiable():
    logits = torch.randn(2, 16, requires_grad=True)
    decoded, _ = decode_soft_ordinal_lrt(logits)
    decoded.sum().backward()

    assert decoded.shape == (2,)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0


def test_site_batches_visit_every_eligible_site_once():
    eligible = torch.tensor([True, False, True, True, False, True, True])
    original_mask = eligible.clone()

    cases = ((1, [1, 1, 1, 1, 1]), (2, [2, 2, 1]), (20, [5]))
    for batch_size, expected_sizes in cases:
        batches = list(
            iter_site_indices(
                eligible,
                batch_size=batch_size,
                generator=torch.Generator().manual_seed(7),
            )
        )
        visited = torch.cat(batches).tolist()
        assert sorted(visited) == [0, 2, 3, 5, 6]
        assert len(visited) == len(set(visited))
        assert [len(batch) for batch in batches] == expected_sizes
    assert torch.equal(eligible, original_mask)


def test_train_epoch_batches_sites_with_heterogeneous_gene_shapes():
    model = PhyloAxialTransformer(
        embed_dim=8, num_layers=1, num_heads=1, window_size=1
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    genes = [make_gene(3, 3, gene_name="short"), make_gene(5, 5, gene_name="long")]

    loss = train_epoch(
        model,
        genes,
        optimizer,
        scaler,
        torch.device("cpu"),
        SimpleNamespace(fp16=False, batch_size=2),
    )

    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))


def test_train_epoch_reports_site_weighted_loss():
    class ConstantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(16))

        def forward(self, c, a, d, z):
            return self.logits.unsqueeze(0).expand(c.shape[0], -1)

    targets = torch.tensor([0.0, 1.0, 10.0])
    gene = make_gene(3, 2, targets=targets)
    model = ConstantModel()
    prediction, _ = decode_soft_ordinal_lrt(model.logits.unsqueeze(0))
    expected = torch.nn.functional.smooth_l1_loss(
        prediction.expand_as(targets), targets, reduction="mean"
    ).item()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    actual = train_epoch(
        model,
        [gene],
        optimizer,
        scaler,
        torch.device("cpu"),
        SimpleNamespace(fp16=False, batch_size=2),
    )

    assert actual == pytest.approx(expected)


def test_batch_size_larger_than_gene_uses_one_step():
    class CountingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(16))
            self.batch_sizes = []

        def forward(self, c, a, d, z):
            self.batch_sizes.append(c.shape[0])
            return self.logits.unsqueeze(0).expand(c.shape[0], -1)

    model = CountingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    gene = make_gene(3, 4)

    train_epoch(
        model,
        [gene],
        optimizer,
        scaler,
        torch.device("cpu"),
        SimpleNamespace(fp16=False, batch_size=50),
    )

    assert model.batch_sizes == [3]


@pytest.mark.parametrize("wrapped", [True, False])
def test_load_initial_checkpoint_strictly_loads_weights(tmp_path, wrapped):
    source = PhyloAxialTransformer(
        embed_dim=8, num_layers=1, num_heads=1, window_size=1
    )
    checkpoint = (
        {"model_state_dict": source.state_dict()} if wrapped else source.state_dict()
    )
    path = tmp_path / "initial.pt"
    torch.save(checkpoint, path)
    target = PhyloAxialTransformer(
        embed_dim=8, num_layers=1, num_heads=1, window_size=1
    )

    load_initial_checkpoint(target, str(path))

    for expected, actual in zip(source.parameters(), target.parameters()):
        assert torch.equal(expected, actual)


def test_load_initial_checkpoint_rejects_incompatible_model(tmp_path):
    source = PhyloAxialTransformer(
        embed_dim=8, num_layers=1, num_heads=1, window_size=1
    )
    path = tmp_path / "initial.pt"
    torch.save({"model_state_dict": source.state_dict()}, path)
    incompatible = PhyloAxialTransformer(
        embed_dim=16, num_layers=1, num_heads=1, window_size=1
    )

    with pytest.raises(ValueError, match="architecture is incompatible"):
        load_initial_checkpoint(incompatible, str(path))


def test_training_cli_runs_two_epochs_from_checkpoint(tmp_path):
    data_dir = tmp_path / "npz"
    output_dir = tmp_path / "weights"
    data_dir.mkdir()
    np.savez_compressed(
        data_dir / "gene.npz",
        c=np.zeros((2, 3, 1), dtype=np.int64),
        a=np.zeros((2, 3, 1), dtype=np.int64),
        d=np.zeros((3, 3), dtype=np.float32),
        z=np.zeros((3, 4), dtype=np.float32),
        target_lrt=np.asarray([1.0, 2.0], dtype=np.float32),
        eligible_mask=np.asarray([True, True], dtype=np.bool_),
        taxa=np.asarray(["a", "b", "c"]),
        gene_name=np.asarray("gene"),
        schema_version=np.asarray(TRAINING_SCHEMA_VERSION, dtype=np.int64),
    )
    model = PhyloAxialTransformer(
        embed_dim=8, num_layers=1, num_heads=1, window_size=1
    )
    checkpoint = tmp_path / "initial.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)

    result = subprocess.run(
        [
            sys.executable,
            "training/train.py",
            "--data_dir",
            str(data_dir),
            "--init_checkpoint",
            str(checkpoint),
            "--epochs",
            "2",
            "--batch_size",
            "1",
            "--embed_dim",
            "8",
            "--layers",
            "1",
            "--heads",
            "1",
            "--output_dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Epoch [ 1/ 2]" in result.stdout
    assert "Epoch [ 2/ 2]" in result.stdout
    assert (output_dir / "hyphaeon_best.pt").exists()
