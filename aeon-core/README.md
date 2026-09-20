# aeon-core

Shared infrastructure for the HyphAeon family of packages (HyphAeon, ChronAeon).

Contains the `PhyloAxialTransformer` foundation model, pretrained weight resolution (HuggingFace Hub), and core utilities for alignment processing, inference, spectral splits, temporal parsing, I/O, and statistics.

## Installation

```bash
pip install hyphaeon-core
```

## Components

- **`model`** — `PhyloAxialTransformer`, `BustedMultiTaskHead`, `decode_soft_ordinal_lrt`
- **`weights`** — Weight resolution from HuggingFace Hub (`datamonkey/hyphaeon`), safetensors loading
- **`inference`** — Device management, adaptive batch sizing, model loading, alignment preparation
- **`dataset`** — Alignment parsing, TN93 distance matrices, tree handling, MDS coordinates
- **`splits`** — Cross-taxa attention extraction, fused affinity matrices
- **`temporal`** — Date parsing utilities (`parse_date_to_decimal`, `extract_date_from_string`)
- **`io`** — JSON/CSV writers, directory helpers
- **`stats`** — LRT p-value computation, Benjamini-Hochberg, Cauchy combination
- **`_progress`** — `ChunkProgress` progress bar utility
