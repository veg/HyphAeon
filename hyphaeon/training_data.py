"""Per-gene training tensor extraction and loading utilities."""

from __future__ import annotations

import gzip
import json
import re
import warnings
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from hyphaeon.dataset import load_alignment_and_tree


TRAINING_SCHEMA_VERSION = 1
NEGATIVE_LRT_TOLERANCE = 1e-8

REQUIRED_GENE_KEYS = {
    "c",
    "a",
    "d",
    "z",
    "target_lrt",
    "eligible_mask",
    "taxa",
    "gene_name",
    "schema_version",
}

ALIGNMENT_SUFFIXES = (
    ".fasta.gz",
    ".nexus.gz",
    ".msa.gz",
    ".fas.gz",
    ".nex.gz",
    ".fa.gz",
    ".fasta",
    ".nexus",
    ".msa",
    ".fas",
    ".nex",
    ".fa",
    ".gz",
)
MEME_SUFFIXES = (".meme.json.gz", ".json.gz", ".meme.json", ".json")
TREE_SUFFIXES = (
    ".newick.gz",
    ".tree.gz",
    ".nwk.gz",
    ".tre.gz",
    ".newick",
    ".tree",
    ".nwk",
    ".tre",
)


def _strip_known_suffix(filename: str, suffixes: Iterable[str]) -> Optional[str]:
    lower_name = filename.lower()
    for suffix in suffixes:
        if lower_name.endswith(suffix):
            return filename[: -len(suffix)]
    return None


def _scan_by_gene(directory: Path, suffixes: Iterable[str], kind: str) -> Dict[str, Path]:
    if not directory.is_dir():
        raise ValueError(f"{kind.capitalize()} directory does not exist: {directory}")

    files: Dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        gene_name = _strip_known_suffix(path.name, suffixes)
        if gene_name is None:
            continue
        if not gene_name:
            raise ValueError(f"Could not derive a gene name from {kind} file: {path}")
        if gene_name in files:
            raise ValueError(
                f"Multiple {kind} files map to gene '{gene_name}': "
                f"{files[gene_name].name}, {path.name}"
            )
        files[gene_name] = path
    return files


def pair_training_inputs(
    alignment_dir: str,
    meme_dir: str,
    tree_dir: Optional[str] = None,
) -> Tuple[Tuple[str, Path, Optional[Path], Path], ...]:
    """Pair alignment, optional external tree, and MEME files by gene stem."""
    alignments = _scan_by_gene(Path(alignment_dir), ALIGNMENT_SUFFIXES, "alignment")
    if not alignments:
        raise ValueError(f"No supported alignment files found in: {alignment_dir}")

    meme_results = _scan_by_gene(Path(meme_dir), MEME_SUFFIXES, "MEME result")
    trees = _scan_by_gene(Path(tree_dir), TREE_SUFFIXES, "tree") if tree_dir else {}

    missing_meme = sorted(set(alignments) - set(meme_results))
    if missing_meme:
        raise ValueError(
            "Missing MEME result for alignment gene(s): " + ", ".join(missing_meme)
        )

    return tuple(
        (gene, alignments[gene], trees.get(gene), meme_results[gene])
        for gene in sorted(alignments)
    )


def _read_json(path: Path) -> dict:
    open_fn = gzip.open if path.name.lower().endswith(".gz") else open
    try:
        with open_fn(path, "rt", encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read MEME JSON '{path}': {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"MEME JSON root must be an object: {path}")
    return result


def _header_name(header: object) -> str:
    if isinstance(header, str):
        return header
    if isinstance(header, (list, tuple)) and header:
        return str(header[0])
    if isinstance(header, dict):
        for key in ("name", "header", "label"):
            if key in header:
                return str(header[key])
        if len(header) == 1:
            return str(next(iter(header)))
    return ""


def _normalize_header(header: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", _header_name(header).lower())


def _resolve_lrt_column(mle: dict, rows: list, path: Path) -> int:
    headers = mle.get("headers")
    if headers is not None:
        if not isinstance(headers, list):
            raise ValueError(f"MEME MLE.headers must be a list: {path}")
        matches = [
            index
            for index, header in enumerate(headers)
            if _normalize_header(header)
            in {"lrt", "likelihoodratiotest", "likelihoodratioteststatistic"}
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Could not resolve exactly one LRT column from MEME MLE.headers in: {path}"
            )
        return matches[0]

    # Canonical MEME output stores LRT at index 5 and p-value at index 6.
    if rows and all(isinstance(row, list) and len(row) >= 7 for row in rows):
        return 5
    raise ValueError(
        f"MEME JSON has no usable MLE.headers and does not match the canonical row layout: {path}"
    )


def read_meme_lrt(path: str, expected_sites: int) -> np.ndarray:
    """Read physical site-level LRT values from partition 0 of a MEME JSON."""
    json_path = Path(path)
    data = _read_json(json_path)

    try:
        reported_sites = int(data["input"]["number of sites"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"MEME JSON is missing a valid input.number of sites value: {json_path}"
        ) from exc
    if reported_sites != expected_sites:
        raise ValueError(
            f"Site-count mismatch for '{json_path}': alignment has {expected_sites}, "
            f"MEME reports {reported_sites}"
        )

    try:
        mle = data["MLE"]
        rows = mle["content"]["0"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"MEME JSON is missing MLE.content['0']: {json_path}"
        ) from exc
    if not isinstance(mle, dict) or not isinstance(rows, list):
        raise ValueError(f"MEME MLE.content['0'] must be a list: {json_path}")
    if len(rows) != expected_sites:
        raise ValueError(
            f"Site-count mismatch for '{json_path}': alignment has {expected_sites}, "
            f"MLE content has {len(rows)} rows"
        )

    lrt_column = _resolve_lrt_column(mle, rows, json_path)
    targets = np.empty(expected_sites, dtype=np.float64)
    for site_index, row in enumerate(rows, start=1):
        if not isinstance(row, list) or lrt_column >= len(row):
            raise ValueError(
                f"Malformed MEME row for site {site_index} in '{json_path}'"
            )
        try:
            targets[site_index - 1] = float(row[lrt_column])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid LRT value for site {site_index} in '{json_path}': "
                f"{row[lrt_column]!r}"
            ) from exc
    return targets


def build_gene_npz(
    gene_name: str,
    alignment_path: str,
    meme_path: str,
    output_path: str,
    tree_path: Optional[str] = None,
) -> dict:
    """Build one schema-versioned per-gene training archive."""
    c, a, d, z, invariable, taxa, site_count = load_alignment_and_tree(
        alignment_path, tree_path
    )
    target_lrt = read_meme_lrt(meme_path, site_count)

    finite = np.isfinite(target_lrt)
    tiny_negative = finite & (target_lrt < 0.0) & (
        target_lrt >= -NEGATIVE_LRT_TOLERANCE
    )
    materially_negative = finite & (target_lrt < -NEGATIVE_LRT_TOLERANCE)
    target_lrt[tiny_negative] = 0.0
    eligible_mask = (~np.asarray(invariable, dtype=bool)) & finite & (~materially_negative)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        c=c.cpu().numpy().astype(np.int64, copy=False),
        a=a.cpu().numpy().astype(np.int64, copy=False),
        d=d.squeeze(0).cpu().numpy().astype(np.float32, copy=False),
        z=z.squeeze(0).cpu().numpy().astype(np.float32, copy=False),
        target_lrt=target_lrt.astype(np.float32),
        eligible_mask=eligible_mask.astype(np.bool_),
        taxa=np.asarray(taxa, dtype=str),
        gene_name=np.asarray(gene_name),
        schema_version=np.asarray(TRAINING_SCHEMA_VERSION, dtype=np.int64),
    )

    return {
        "sites": site_count,
        "eligible": int(eligible_mask.sum()),
        "invariable": int(np.asarray(invariable).sum()),
        "nonfinite": int((~finite).sum()),
        "materially_negative": int(materially_negative.sum()),
        "clamped_negative": int(tiny_negative.sum()),
    }


def build_training_directory(
    alignment_dir: str,
    meme_dir: str,
    output_dir: str,
    tree_dir: Optional[str] = None,
) -> Tuple[Tuple[str, dict], ...]:
    """Build an NPZ for every gene alignment in ``alignment_dir``."""
    pairs = pair_training_inputs(alignment_dir, meme_dir, tree_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    expected_archives = {f"{gene}.npz" for gene, *_ in pairs}
    unexpected_archives = sorted(
        path.name
        for path in output.glob("*.npz")
        if path.name not in expected_archives
    )
    if unexpected_archives:
        preview_limit = 10
        preview = ", ".join(unexpected_archives[:preview_limit])
        remaining = len(unexpected_archives) - preview_limit
        if remaining > 0:
            preview = f"{preview}, and {remaining} more"
        warnings.warn(
            f"Output directory contains {len(unexpected_archives)} NPZ archive(s) "
            f"that are not outputs of the current alignment set: {preview}. "
            "train.py will include every .npz in --data_dir; remove these files "
            "or use a clean output directory if they should not be trained on.",
            UserWarning,
            stacklevel=2,
        )

    summaries = []
    for gene, alignment, tree, meme in pairs:
        summary = build_gene_npz(
            gene_name=gene,
            alignment_path=str(alignment),
            tree_path=str(tree) if tree else None,
            meme_path=str(meme),
            output_path=str(output / f"{gene}.npz"),
        )
        summaries.append((gene, summary))
    return tuple(summaries)


def _scalar_value(data: np.lib.npyio.NpzFile, key: str, path: Path):
    value = np.asarray(data[key])
    if value.ndim != 0:
        raise ValueError(f"'{key}' must be a scalar in training archive: {path}")
    return value.item()


def validate_gene_npz(path: str) -> None:
    """Validate a per-gene archive without retaining its arrays in memory."""
    archive_path = Path(path)
    try:
        with np.load(archive_path, allow_pickle=False) as data:
            missing = REQUIRED_GENE_KEYS - set(data.files)
            if missing:
                raise ValueError(
                    f"missing required key(s): {', '.join(sorted(missing))}"
                )

            version = _scalar_value(data, "schema_version", archive_path)
            if version != TRAINING_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported schema_version {version!r}; expected {TRAINING_SCHEMA_VERSION}"
                )
            gene_name = _scalar_value(data, "gene_name", archive_path)
            if not isinstance(gene_name, str) or not gene_name:
                raise ValueError("'gene_name' must be a non-empty string scalar")

            c, a = data["c"], data["a"]
            d, z = data["d"], data["z"]
            target, eligible = data["target_lrt"], data["eligible_mask"]
            taxa = data["taxa"]

            if c.dtype != np.int64 or a.dtype != np.int64:
                raise ValueError("'c' and 'a' must have dtype int64")
            if c.ndim != 3 or c.shape != a.shape or c.shape[2] != 1:
                raise ValueError("'c' and 'a' must share shape [sites, taxa, 1]")
            site_count, taxon_count, _ = c.shape
            if site_count == 0 or taxon_count == 0:
                raise ValueError("training archives must contain at least one site and taxon")
            if d.dtype != np.float32 or d.shape != (taxon_count, taxon_count):
                raise ValueError("'d' must have dtype float32 and shape [taxa, taxa]")
            if z.dtype != np.float32 or z.shape != (taxon_count, 4):
                raise ValueError("'z' must have dtype float32 and shape [taxa, 4]")
            if target.dtype != np.float32 or target.shape != (site_count,):
                raise ValueError("'target_lrt' must have dtype float32 and shape [sites]")
            if eligible.dtype != np.bool_ or eligible.shape != (site_count,):
                raise ValueError("'eligible_mask' must have dtype bool and shape [sites]")
            if taxa.ndim != 1 or taxa.shape != (taxon_count,) or taxa.dtype.kind not in "US":
                raise ValueError("'taxa' must be a string array with shape [taxa]")
            if len(set(taxa.tolist())) != taxon_count:
                raise ValueError("'taxa' entries must be unique")
            if not np.isfinite(d).all() or not np.isfinite(z).all():
                raise ValueError("'d' and 'z' must contain only finite values")
            if eligible.any():
                eligible_targets = target[eligible]
                if not np.isfinite(eligible_targets).all() or (eligible_targets < 0).any():
                    raise ValueError(
                        "eligible target_lrt values must be finite and non-negative"
                    )
    except ValueError as exc:
        raise ValueError(f"Invalid per-gene training archive '{archive_path}': {exc}") from exc
    except (OSError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(
            f"Could not open per-gene training archive '{archive_path}': {exc}"
        ) from exc


class GeneTensorsDataset(Dataset):
    """Load every schema-compatible per-gene NPZ directly from a directory."""

    def __init__(self, npz_dir: str):
        directory = Path(npz_dir)
        if not directory.is_dir():
            raise ValueError(f"Training data directory does not exist: {directory}")
        self.files = sorted(directory.glob("*.npz"))
        if not self.files:
            raise ValueError(f"No .npz training archives found in: {directory}")
        for path in self.files:
            validate_gene_npz(str(path))
        print(f"[*] Loaded {len(self.files)} per-gene training archives from: {directory}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict:
        path = self.files[index]
        with np.load(path, allow_pickle=False) as data:
            return {
                "c": torch.from_numpy(data["c"].copy()).long(),
                "a": torch.from_numpy(data["a"].copy()).long(),
                "d": torch.from_numpy(data["d"].copy()).float(),
                "z": torch.from_numpy(data["z"].copy()).float(),
                "target_lrt": torch.from_numpy(data["target_lrt"].copy()).float(),
                "eligible_mask": torch.from_numpy(data["eligible_mask"].copy()).bool(),
                "gene_name": str(np.asarray(data["gene_name"]).item()),
            }
