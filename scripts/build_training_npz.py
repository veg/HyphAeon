#!/usr/bin/env python3
"""Build one HyphAeon training NPZ per alignment gene."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyphaeon.training_data import build_training_directory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-gene HyphAeon training tensors from alignments and HyPhy MEME JSON"
    )
    parser.add_argument(
        "--alignment_dir", required=True, help="Directory containing FASTA or NEXUS alignments"
    )
    parser.add_argument(
        "--meme_dir", required=True, help="Directory containing HyPhy MEME .json or .json.gz results"
    )
    parser.add_argument(
        "--tree_dir",
        help="Optional directory containing matching Newick trees; omit for embedded trees",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory in which to write one .npz per gene"
    )
    args = parser.parse_args()

    summaries = build_training_directory(
        alignment_dir=args.alignment_dir,
        tree_dir=args.tree_dir,
        meme_dir=args.meme_dir,
        output_dir=args.output_dir,
    )
    total_sites = 0
    total_eligible = 0
    for gene, summary in summaries:
        total_sites += summary["sites"]
        total_eligible += summary["eligible"]
        exclusions = (
            f"invariable={summary['invariable']}, nonfinite={summary['nonfinite']}, "
            f"negative={summary['materially_negative']}"
        )
        print(
            f"[✓] {gene}: {summary['eligible']}/{summary['sites']} eligible sites "
            f"({exclusions}, clamped={summary['clamped_negative']})"
        )
    print(
        f"[✓] Wrote {len(summaries)} genes with {total_eligible}/{total_sites} "
        f"eligible sites to: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
