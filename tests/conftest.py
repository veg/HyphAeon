import os
import tempfile
import pytest
import torch

from hyphaeon.model import PhyloAxialTransformer

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")
EXPECTED_DIR = os.path.join(EXAMPLES_DIR, "expected_results")


@pytest.fixture
def examples_dir():
    return EXAMPLES_DIR


@pytest.fixture
def expected_dir():
    return EXPECTED_DIR


@pytest.fixture(scope="session")
def dummy_weights(tmp_path_factory):
    """A tiny checkpoint with random weights and minimal architecture.

    Real pretrained weights (23 MB) won't always live in the repo — they'll
    move to Hugging Face / GitHub releases, and new model versions will ship
    different weights. Package tests should not depend on them. This fixture
    builds a minimal model (embed_dim=8, 1 layer, 1 head), saves it as a
    checkpoint with the same structure the CLI expects, and returns the path.

    The forward pass produces meaningless but finite, well-shaped output —
    enough to exercise the full CLI code path (checkpoint loading, model
    instantiation, inference, post-processing, output writing) without
    coupling tests to any particular pretrained model.
    """
    model = PhyloAxialTransformer(embed_dim=8, num_layers=1, num_heads=1, window_size=1)
    ckpt_path = str(tmp_path_factory.mktemp("weights") / "dummy.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": {"embed_dim": 8, "layers": 1, "heads": 1, "window_size": 1},
    }, ckpt_path)
    return ckpt_path


@pytest.fixture
def fasta_content():
    return """>seq1
ATGTTTCTTGGT
>seq2
ATGTTCCTTGGT
>seq3
ATGTTTCTCGGT
"""


@pytest.fixture
def fasta_file(tmp_path, fasta_content):
    p = tmp_path / "test.fasta"
    p.write_text(fasta_content)
    return str(p)


@pytest.fixture
def nexus_content():
    return """#NEXUS
BEGIN TAXA;
    DIMENSIONS NTAX=3;
    TAXLABELS seq1 seq2 seq3;
END;
BEGIN CHARACTERS;
    DIMENSIONS NCHAR=12;
    FORMAT DATATYPE=DNA MISSING=? GAP=-;
    MATRIX
    seq1 ATGTTTCTTGGT
    seq2 ATGTTCCTTGGT
    seq3 ATGTTTCTCGGT
    ;
END;
BEGIN TREES;
    TREE tree1 = ((seq1:0.1,seq2:0.1):0.05,seq3:0.15);
END;
"""


@pytest.fixture
def nexus_file(tmp_path, nexus_content):
    p = tmp_path / "test.nex"
    p.write_text(nexus_content)
    return str(p)


@pytest.fixture
def newick_content():
    return "((seq1:0.1,seq2:0.1):0.05,seq3:0.15);"


@pytest.fixture
def newick_file(tmp_path, newick_content):
    p = tmp_path / "test.nwk"
    p.write_text(newick_content)
    return str(p)


@pytest.fixture
def newick_no_branch_lengths():
    return "((seq1,seq2),seq3);"


@pytest.fixture
def newick_hyphy_annotations():
    return "((seq1{Foreground}:0.1,seq2{Background}:0.1):0.05,seq3:0.15);"


@pytest.fixture
def example_dataset(examples_dir):
    """Yields (name, fasta_path, tree_path, expected_csv_path) for each example."""
    datasets = []
    for name in ["Smc6", "bat_oas1", "camelid"]:
        fa = os.path.join(examples_dir, f"{name}.fasta")
        nwk = os.path.join(examples_dir, f"{name}.nwk")
        csv = os.path.join(EXPECTED_DIR, f"{name}_results.csv")
        if os.path.exists(fa) and os.path.exists(nwk) and os.path.exists(csv):
            datasets.append((name, fa, nwk, csv))
    return datasets
