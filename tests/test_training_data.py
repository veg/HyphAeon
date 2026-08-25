import gzip
import json
import warnings

import numpy as np
import pytest

from hyphaeon.dataset import load_alignment_and_tree
from hyphaeon.training_data import (
    GeneTensorsDataset,
    TRAINING_SCHEMA_VERSION,
    build_gene_npz,
    build_training_directory,
    pair_training_inputs,
    read_meme_lrt,
)


HEADERS = [
    ["alpha", "synonymous rate"],
    ["beta-", "lower nonsynonymous rate"],
    ["p-", "lower rate weight"],
    ["beta+", "upper nonsynonymous rate"],
    ["p+", "upper rate weight"],
    ["LRT", "likelihood ratio test statistic"],
    ["p-value", "asymptotic p-value"],
]


def write_inputs(tmp_path, gene="gene", targets=None, compressed_json=False):
    if targets is None:
        targets = [2.0, 1.0, 3.0, 4.0, 5.0]
    alignment_dir = tmp_path / "alignments"
    tree_dir = tmp_path / "trees"
    meme_dir = tmp_path / "meme"
    alignment_dir.mkdir()
    tree_dir.mkdir()
    meme_dir.mkdir()

    alignment = alignment_dir / f"{gene}.fasta"
    alignment.write_text(
        ">seq1\nATGTTTCTTGGTGAA\n"
        ">seq2\nATATTTATTGATGAA\n"
        ">seq3\nATGTTACTTGGTGAA\n"
    )
    tree = tree_dir / f"{gene}.nwk"
    tree.write_text("((seq1:0.1,seq2:0.1):0.05,seq3:0.15);\n")
    payload = {
        "input": {"number of sites": 5, "number of sequences": 3},
        "MLE": {
            "headers": HEADERS,
            "content": {
                "0": [[1.0, 0.5, 0.5, 2.0, 0.5, value, 0.1] for value in targets]
            },
        },
    }
    suffix = ".MEME.json.gz" if compressed_json else ".MEME.json"
    meme = meme_dir / f"{gene}{suffix}"
    if compressed_json:
        with gzip.open(meme, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
    else:
        meme.write_text(json.dumps(payload))
    return alignment_dir, tree_dir, meme_dir, alignment, tree, meme


def test_build_gene_npz_shapes_tokens_and_eligibility(tmp_path):
    targets = [2.0, -1e-9, float("nan"), -0.1, 5.0]
    _, _, _, alignment, tree, meme = write_inputs(tmp_path, targets=targets)
    output = tmp_path / "gene.npz"

    summary = build_gene_npz(
        "gene", str(alignment), str(meme), str(output), str(tree)
    )
    expected_c, expected_a, expected_d, expected_z, _, taxa, _ = load_alignment_and_tree(
        str(alignment), str(tree)
    )

    with np.load(output, allow_pickle=False) as data:
        assert data["c"].shape == (5, 3, 1)
        assert data["a"].shape == (5, 3, 1)
        assert data["d"].shape == (3, 3)
        assert data["z"].shape == (3, 4)
        assert data["target_lrt"].shape == (5,)
        assert data["eligible_mask"].tolist() == [True, True, False, False, False]
        assert data["target_lrt"][1] == 0.0
        assert np.isnan(data["target_lrt"][2])
        assert data["target_lrt"][3] == pytest.approx(-0.1)
        assert data["c"].dtype == np.int64
        assert data["a"].dtype == np.int64
        assert data["d"].dtype == np.float32
        assert data["z"].dtype == np.float32
        assert data["target_lrt"].dtype == np.float32
        assert data["eligible_mask"].dtype == np.bool_
        assert data["schema_version"].item() == TRAINING_SCHEMA_VERSION
        assert data["gene_name"].item() == "gene"
        assert data["taxa"].tolist() == taxa
        np.testing.assert_array_equal(data["c"], expected_c.numpy())
        np.testing.assert_array_equal(data["a"], expected_a.numpy())
        np.testing.assert_allclose(data["d"], expected_d.squeeze(0).numpy())
        np.testing.assert_allclose(data["z"], expected_z.squeeze(0).numpy())

    assert summary == {
        "sites": 5,
        "eligible": 2,
        "invariable": 1,
        "nonfinite": 1,
        "materially_negative": 1,
        "clamped_negative": 1,
    }


def test_reads_plain_and_gzipped_meme_json(tmp_path):
    plain_root = tmp_path / "plain"
    gzip_root = tmp_path / "gzip"
    plain_root.mkdir()
    gzip_root.mkdir()
    *_, plain = write_inputs(plain_root, compressed_json=False)
    *_, compressed = write_inputs(gzip_root, compressed_json=True)

    np.testing.assert_allclose(read_meme_lrt(str(plain), 5), [2, 1, 3, 4, 5])
    np.testing.assert_allclose(read_meme_lrt(str(compressed), 5), [2, 1, 3, 4, 5])


def test_canonical_lrt_column_fallback_without_headers(tmp_path):
    *_, meme = write_inputs(tmp_path)
    payload = json.loads(meme.read_text())
    del payload["MLE"]["headers"]
    meme.write_text(json.dumps(payload))

    np.testing.assert_allclose(read_meme_lrt(str(meme), 5), [2, 1, 3, 4, 5])


def test_pairing_requires_meme_for_every_alignment(tmp_path):
    alignment_dir, tree_dir, meme_dir, *_ = write_inputs(tmp_path)
    (alignment_dir / "missing.fasta").write_text(">seq1\nATG\n")

    with pytest.raises(ValueError, match="Missing MEME result.*missing"):
        pair_training_inputs(str(alignment_dir), str(meme_dir), str(tree_dir))


def test_pairing_supports_msa_alignments_with_nwk_trees(tmp_path):
    alignment_dir = tmp_path / "alignments"
    tree_dir = tmp_path / "trees"
    meme_dir = tmp_path / "meme"
    for directory in (alignment_dir, tree_dir, meme_dir):
        directory.mkdir()
    (alignment_dir / "gene.msa").write_text(">a\nATG\n>b\nATA\n>c\nATG\n")
    (tree_dir / "gene.nwk").write_text("((a:0.1,b:0.1):0.05,c:0.15);\n")
    (meme_dir / "gene.MEME.json").write_text("{}")

    pairs = pair_training_inputs(str(alignment_dir), str(meme_dir), str(tree_dir))

    assert len(pairs) == 1
    assert pairs[0][0] == "gene"
    assert pairs[0][1].name == "gene.msa"
    assert pairs[0][2].name == "gene.nwk"
    assert pairs[0][3].name == "gene.MEME.json"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data["input"].update({"number of sites": 4}), "Site-count mismatch"),
        (lambda data: data["MLE"].pop("content"), "missing MLE.content"),
        (lambda data: data["MLE"].update({"headers": [["not-lrt", "bad"]] * 7}), "LRT column"),
        (lambda data: data["MLE"]["content"]["0"][0].__setitem__(5, None), "Invalid LRT"),
    ],
)
def test_rejects_invalid_meme_data(tmp_path, mutation, message):
    *_, meme = write_inputs(tmp_path)
    payload = json.loads(meme.read_text())
    mutation(payload)
    meme.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        read_meme_lrt(str(meme), 5)


def test_training_directory_builds_every_alignment_gene(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = write_inputs(first_root, gene="alpha")
    second = write_inputs(second_root, gene="beta")
    alignment_dir, tree_dir, meme_dir = first[:3]
    (alignment_dir / "beta.fasta").write_text(second[3].read_text())
    (tree_dir / "beta.nwk").write_text(second[4].read_text())
    (meme_dir / "beta.MEME.json").write_text(second[5].read_text())
    output_dir = tmp_path / "npz"

    summaries = build_training_directory(
        str(alignment_dir), str(meme_dir), str(output_dir), str(tree_dir)
    )

    assert [gene for gene, _ in summaries] == ["alpha", "beta"]
    assert sorted(path.name for path in output_dir.glob("*.npz")) == [
        "alpha.npz",
        "beta.npz",
    ]
    dataset = GeneTensorsDataset(str(output_dir))
    assert len(dataset) == 2
    assert {dataset[0]["gene_name"], dataset[1]["gene_name"]} == {"alpha", "beta"}


def test_training_directory_does_not_warn_for_clean_output(tmp_path):
    alignment_dir, tree_dir, meme_dir, *_ = write_inputs(tmp_path)
    output_dir = tmp_path / "npz"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_training_directory(
            str(alignment_dir), str(meme_dir), str(output_dir), str(tree_dir)
        )

    assert caught == []


def test_training_directory_does_not_warn_for_expected_existing_archive(tmp_path):
    alignment_dir, tree_dir, meme_dir, *_ = write_inputs(tmp_path)
    output_dir = tmp_path / "npz"
    output_dir.mkdir()
    (output_dir / "gene.npz").write_text("old archive will be overwritten")
    (output_dir / "notes.txt").write_text("non-NPZ files are ignored")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_training_directory(
            str(alignment_dir), str(meme_dir), str(output_dir), str(tree_dir)
        )

    assert caught == []
    with np.load(output_dir / "gene.npz", allow_pickle=False) as data:
        assert data["gene_name"].item() == "gene"


def test_training_directory_warns_once_for_unexpected_archives(tmp_path):
    alignment_dir, tree_dir, meme_dir, *_ = write_inputs(tmp_path)
    output_dir = tmp_path / "npz"
    output_dir.mkdir()
    stale_paths = []
    for index in range(12):
        path = output_dir / f"stale_{index:02d}.npz"
        np.savez(path, marker=np.asarray(index))
        stale_paths.append(path)

    with pytest.warns(UserWarning) as caught:
        build_training_directory(
            str(alignment_dir), str(meme_dir), str(output_dir), str(tree_dir)
        )

    assert len(caught) == 1
    message = str(caught[0].message)
    assert "12 NPZ archive(s)" in message
    assert "stale_00.npz" in message
    assert "stale_09.npz" in message
    assert "and 2 more" in message
    assert "every .npz in --data_dir" in message
    assert all(path.exists() for path in stale_paths)
    assert (output_dir / "gene.npz").exists()


def test_dataset_rejects_legacy_per_site_npz(tmp_path):
    np.savez(tmp_path / "legacy.npz", c=np.zeros((3, 1), dtype=np.int64))

    with pytest.raises(ValueError, match="missing required key"):
        GeneTensorsDataset(str(tmp_path))
