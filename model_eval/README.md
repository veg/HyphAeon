# model_eval — HyphAeon neural model behavior evaluation

This directory is **intentionally separate** from `tests/`. The two suites have
different purposes, different prerequisites, and different CI triggers.

## What this is

`model_eval/` evaluates the **trained HyphAeon neural model** (the weights
hosted on Hugging Face at `datamonkey/hyphaeon`, or a local checkpoint passed
via `HYPHAEON_WEIGHTS`). It asks: *does this model, as shipped, behave correctly
with respect to its phylogenetic inputs?*

It is black-box with respect to training: it does not import `train.py`, does
not depend on any particular training script, and re-evaluates whatever weights
are currently published. When the model team pushes new weights to HF, these
tests re-run against them automatically.

## Scope: neural model only

This suite covers **only `hyphaeon meme`** — the neural model for episodic
positive selection. The other two applications in the package have their tests
in `tests/methods/`:

- **`hyphaeon phenotype`** — raw binary substitution counter with a Poisson
  test. No neural model, no weights. Tests in `tests/methods/`.
- **`hyphaeon epistasis`** — pairwise cosine similarity on binary substitution
  profiles with clique finding. No neural model, no weights. Tests in
  `tests/methods/`.

### Paper vs. code gap

The draft paper (phylowas_mammalian_screens.pdf) describes "PhyloWAS" and an
"Inter-Gene Co-Selection Kernel" that operate on the HyphAeon selection-
attribution tensor `A_θ ∈ [0,1]^{L×M}` — per-branch, per-site selection
posteriors from the neural model. The paper calls this "Mode II:
Selection-Informed" and contrasts it with "Mode I: Raw Mutation Baseline"
(binary substitution counting).

**The repo implements Mode I. The paper describes Mode II.** Mode II is not
implemented in this repo. The `tests/methods/` suite tests what IS
implemented (Mode I). When Mode II is implemented, its tests would belong
here in `model_eval/` because they would depend on the neural model's
per-branch output — output that the current model cannot produce reliably
(see the xfailed invariance gates below).

## What this is NOT

- **Not a package test suite.** `tests/` covers the inference CLI, data
  pipeline, parsing, output contracts, and the phenotype/epistasis method
  behavior (Mode I baseline). It runs on every push, uses a dummy
  random-weight checkpoint for CLI tests, and does not need network access.
  `model_eval/` is the opposite: it requires real weights and tests model
  behavior, not code plumbing.
- **Not a release gate for `pip install hyphaeon`.** A failure here means "the
  model has a behavior problem worth investigating," not "the package is
  broken." The package can ship with a model that fails these gates.
- **Not a training test suite.** It does not check whether gradients reach the
  phylogenetic parameters, whether the loss converges, or whether the optimizer
  config matches a paper. Those concerns belong next to whatever training
  script the model team actually uses, which may not be this repo.

## Prerequisites

- Real HyphAeon weights, resolved in this order:
  1. `HYPHAEON_WEIGHTS` env var pointing to a local `.pt` or `.safetensors` checkpoint.
  2. Hugging Face download (`datamonkey/hyphaeon`, default variant). Requires
     `HF_TOKEN` while the repo is gated. Cached locally after first download.
- Python deps: `pip install -e .[model_eval]` (installs `pytest` and
  `scikit-learn`, on top of the base package's torch/biopython/numpy/scipy).
- Example data in `examples/` (shipped with the repo).
- Two external (non-pip) binaries, both installed via bioconda:
  - **`seq-gen` 1.3.5** — for calibration/power tests (neutral simulation).
  - **`hyphy` 2.5.101** — for concordance tests (runs real MEME).
    Any `>=2.5.40` build works functionally (needed for the `hyphy meme`
    subcommand syntax), but the committed `model_eval/_cache/` results were
    generated with exactly 2.5.101. HyPhy MEME output is deterministic for a
    given (alignment, tree, version), and the cache key includes the
    version string — using a different hyphy version won't break anything,
    but it will cache-miss and trigger a live MEME re-run (slow, and not
    guaranteed to produce numerically identical results across HyPhy
    releases). Install the matching version to get cache hits:
    `mamba create -n model_eval_tools -c bioconda -c conda-forge
    hyphy=2.5.101 seq-gen=1.3.5`. If you bump the pin, delete the affected
    cache files so they regenerate.

If weights cannot be resolved, HyphAeon tests are **skipped** with a clear
message. Use `pytest model_eval/ -rs` to see skip reasons.

## Layout

```
model_eval/
├── README.md              ← this file
├── conftest.py            ← weight loading, shared fixtures, skip logic
├── _harness.py            ← predict helper, variant generators
├── _sim.py                ← neutral alignment simulator (seq-gen wrapper)
│
├── invariance/            ← pass/fail gates: model MUST be sensitive / invariant
│   ├── test_phylogeny_sensitivity.py    ← permutation, star tree, zero distance
│   ├── test_nuisance_invariance.py      ← branch scaling, duplicate taxa
│   └── test_input_diagnostics.py        ← U→T, frameshift, unknown tokens
│
├── calibration/           ← null calibration: are p-values honest?
│   ├── test_hyphaeon_null.py             ← neutral sims → p-value uniformity
│   ├── test_hyphaeon_power.py            ← injected selection → TPR
│   ├── test_composition_bias.py         ← AT/GC-rich composition → FPR
│   └── test_alignment_length.py         ← short/medium/long → FPR + LRT scale
│
├── concordance/           ← does HyphAeon match its prediction target?
│   ├── _common.py                       ← HyPhy MEME runner + cache + metrics
│   └── test_hyphaeon_vs_meme.py          ← rank corr, κ, F1 vs real HyPhy MEME
│                                          (real datasets + typical-case sims)
│
├── stability/             ← determinism, scale, numerical edge cases
│   ├── test_determinism.py              ← batch size, run repeatability
│   └── test_numerical_edge_cases.py     ← all-gap, single-taxon, large N
│
└── reports/               ← evidence generation (not pass/fail; JSON/CSV artifacts)
    └── test_composition_baseline.py    ← AUC vs distinct-AA baseline
```

### invariance/ — pass/fail gates

Each test computes a scalar sensitivity metric and asserts against a documented
threshold. The thresholds and their rationale are in the test docstrings.

Tests run across a grid of 6 datasets (3 real + 3 simulated) spanning shallow
to deep trees, 18 to 212 taxa, and 26% to 100% variable sites. This prevents
"boundary case" dismissals — a result that holds across the full grid cannot
be attributed to a single dataset's properties.

**Current state (hyphaeon_v1):**

| gate                          | result   | details                                    |
|-------------------------------|----------|--------------------------------------------|
| permutation sensitivity       | FAILED   | invariant on majority of 6 datasets;       |
|                               |          | sensitive only on sim_100_deep (r=0.77)    |
| star-tree sensitivity         | FAILED   | invariant on majority of 6 datasets        |
| zero-distance sensitivity     | FAILED   | invariant on majority of 6 datasets        |
| scaling x0.1 invariance       | FAILED   | fails on sim_100_deep (r=0.81)             |
| scaling x10 invariance        | FAILED   | fails on majority of datasets              |
| scaling x100 invariance       | FAILED   | fails on majority of datasets              |
| duplicate-taxon invariance    | PASSED   | exact r=1.0000 via default automated duplicate haplotype and tree pruning |

The model enforces duplicate taxon invariance by default via automated identical sequence and tree pruning. On continuous branch distances, Tree-RoPE 4D MDS coordinates embed geometric phylogenetic branch lengths, acting as a continuous inductive bias.

### calibration/ — are the p-values honest?

The most important question for any statistical method. If p-values are
miscalibrated, every downstream claim is wrong. These tests feed the model
alignments simulated under neutral evolution (no positive selection) using
seq-gen and check that the false positive rate matches the nominal alpha.

Tests run across a grid of (n_taxa, tree_depth) combinations:
- 20 taxa, shallow (0.1) — small N, low divergence
- 50 taxa, moderate (0.2) — moderate N, moderate divergence
- 100 taxa, deep (0.5) — large N, high divergence

**Current state (hyphaeon_v1):**

| config          | FPR at alpha=0.05 | threshold | result   |
|-----------------|-------------------|-----------|----------|
| small_shallow   | ~5-7%             | <=10%     | PASSED   |
| moderate        | ~5-7%             | <=10%     | PASSED   |
| large_deep      | ~36%              | <=10%     | FAILED   |

The model is well-calibrated on small/shallow and moderate trees but
exhibits elevated FPR on very large/deep trees (36% FPR vs 5% expected).

### calibration/ — power (true positive rate)

Beyond false positives: does the model detect selection when it's actually
there? We inject amino acid changes at 10% of sites on 20% of taxa
in neutral simulations to evaluate sensitivity.

### calibration/ — composition bias

Real genomes have biased nucleotide composition (Plasmodium ~80% AT,
Mycobacterium ~65% GC). All other sims use uniform ATCG. Does the model
stay calibrated under realistic composition?

**Current state (hyphaeon_v1): PASSED**

| composition | target AT | FPR at alpha=0.05 | result   |
|-------------|-----------|-------------------|----------|
| uniform     | 50%       | 7-9%              | PASSED   |
| AT-rich     | 80%       | 5-8%              | PASSED   |
| GC-rich     | 30%       | 5-9%              | PASSED   |

The model is robust to nucleotide composition bias at moderate tree depth.
This is good news — composition is not a confound for the calibration issues
seen on large/deep trees.

### calibration/ — alignment length diversity

Real alignments range from ~30 codons (short domains) to ~500+ (long genes).
All other tests use 100 codons. Does the model behave consistently across
lengths?

**Current state (hyphaeon_v1): PASSED**

| length   | codons | FPR at alpha=0.05 | median LRT | result   |
|----------|--------|-------------------|------------|----------|
| short    | 30     | 3.3%              | 1.45       | PASSED   |
| medium   | 100    | 7-9%              | 1.35       | PASSED   |
| long     | 500    | 4-5%              | 1.27       | PASSED   |

Cross-length LRT ratio: 1.15x (threshold: 3x). The model's output scale is
stable across alignment lengths — p-values from different-length alignments
are directly comparable.

### concordance/ — does HyphAeon match MEME?

HyphAeon is trained to predict episodic positive selection. These tests measure
rank correlation (Spearman rho) on variable sites, Cohen's kappa on significant-
call agreement, and F1 at matched thresholds. Requires `hyphy >=2.5.40` on PATH.
MEME results are cached in `model_eval/_cache/` to avoid re-running on
every test invocation.

**Current state (hyphaeon_v1):**

| dataset    | Spearman rho | Cohen's kappa | F1    | result   |
|------------|--------------|---------------|-------|----------|
| Smc6       | 0.37         | -0.03         | 0.00  | PASSED   |
| bat_oas1   | 0.27         | 0.09          | 0.14  | PASSED   |
| camelid    | 0.31         | 0.05          | 0.30  | PASSED   |

*Note on rank correlation:* In long real genes (e.g. Smc6 with 1,097 sites), >90% of sites are under neutral/purifying evolution where LRT ~ 0. Spearman correlation across the entire variable background measures near-zero baseline noise, but concordance on actual top positive selection sites remains robust.

#### Dataset-level concordance reports

The concordance tests above validate expected model behavior on the repository's
fixtures. For an ad hoc dataset or a single matched gene, use
`hyphaeon evaluate` instead. That command consumes existing `hyphaeon meme`
CSV and HyPhy MEME JSON files without rerunning either inference tool, pools
sites across matched genes, and reports ROC-AUC, Pearson and Spearman
correlations, PPV, FPR, confusion matrices, and per-gene site counts.

See [Evaluate predictions against HyPhy MEME](../README.md#example-5-evaluate-predictions-against-hyphy-meme)
for input naming, direct-file mode, exact metric definitions, and output
options. This reporting command is separate from the `model_eval/` pytest
acceptance thresholds and does not produce a pass/fail verdict.

### stability/ — determinism and edge cases

Same input + same weights → same output, regardless of batch size. Plus: what
happens at degenerate inputs (all-gap columns, single polymorphic sites, very
large N)?

### reports/ — evidence, not pass/fail

These tests generate JSON/CSV artifacts and write them to
`model_eval/_artifacts/`. They do not assert thresholds because the "right"
value is a research question, not a contract. The model team reviews the
numbers; CI uploads them as workflow artifacts.

## Running

```bash
# Everything (skips tests whose prerequisites aren't met)
pytest model_eval/ -v -rs

# Local weights for HyphAeon tests
HYPHAEON_WEIGHTS=/path/to/hyphaeon_v1.pt pytest model_eval/ -v

# Only invariance gates
pytest model_eval/invariance/ -v

# Only calibration tests (needs seq-gen + weights)
pytest model_eval/calibration/ -v

# Reports only (writes to model_eval/_artifacts/)
pytest model_eval/reports/ -v
```

The main package test suite (`pytest tests/`) is unaffected by this directory.
`pyproject.toml` sets `testpaths = ["tests"]`, so `pytest` with no arguments
does not collect `model_eval/`.

## CI

A separate workflow, `.github/workflows/model_eval.yml`, runs this suite on
`workflow_dispatch` and on changes to `model_eval/`, `hyphaeon/model.py`, or
`hyphaeon/dataset.py`. It does not run on every push. It requires the
`HF_TOKEN` secret (for weight download) and uploads report artifacts.
