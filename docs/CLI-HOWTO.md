# HyphAeon CLI How-To: Every Analysis, End to End

This is the practical, task-by-task reference for the `hyphaeon` command-line
interface. It documents **every analysis the manuscript describes** and maps
each one to the exact CLI invocation, inputs, outputs (CSV columns / JSON
schema), and gotchas. Where a paper analysis has **no dedicated CLI
subcommand**, that gap is flagged explicitly (see
[Paper-vs-CLI coverage matrix](#paper-vs-cli-coverage-matrix)).

All example commands use the bundled files in [`examples/`](../examples/) so you
can copy-paste and run them directly after `pip install -e .`.

> **Note on the model name.** The package/CLI is `hyphaeon` (formerly
> `axomeme`). Internally several columns and fields still carry the `axomeme_`
> prefix (e.g. `axomeme_lrt`); these are the model's predicted MEME-style
> likelihood-ratio scores and are documented as such below.

---

## Table of contents

- [Quick start & global conventions](#quick-start--global-conventions)
- [Bundled example datasets](#bundled-example-datasets)
- [Weights, model variants, and device selection](#weights-model-variants-and-device-selection)
- [Subcommand how-tos](#subcommand-how-tos)
  - [`predict` — site-level episodic diversifying selection (MEME emulation)](#predict--site-level-episodic-diversifying-selection-meme-emulation)
  - [`predict --attribute` — mechanistic feature attribution](#predict---attribute--mechanistic-feature-attribution)
  - [`predict --filter` / `filter` — automated alignment-error screening](#predict---filter--filter--automated-alignment-error-screening)
  - [`busted` — alignment-wide omnibus selection (BUSTED / BUSTED+S emulation)](#busted--alignment-wide-omnibus-selection-busted--busteds-emulation)
  - [`epistasis` — branch co-selection, contacts & epistatic sectors](#epistasis--branch-co-selection-contacts--epistatic-sectors)
  - [`dms` — digital deep mutational scanning (dDMS / ESSM)](#dms--digital-deep-mutational-scanning-ddms--essm)
  - [`disease` — zero-shot clinical variant pathogenicity](#disease--zero-shot-clinical-variant-pathogenicity)
  - [`phenotype` — directional phenotype–genotype association (PhyloWAS / PARS)](#phenotype--directional-phenotypegenotype-association-phylowas--pars)
  - [`evaluate` — HyPhy MEME concordance](#evaluate--hyphy-meme-concordance)
  - [`list-models` — enumerate Hugging Face variants](#list-models--enumerate-hugging-face-variants)
- [Paper-vs-CLI coverage matrix](#paper-vs-cli-coverage-matrix)
- [Paper-ahead-of-code gaps](#paper-ahead-of-code-gaps)

---

## Quick start & global conventions

Install and confirm the CLI is on your `PATH`:

```bash
pip install -e .
hyphaeon --help
```

**Conventions shared by every subcommand:**

| Flag | Meaning |
| :--- | :--- |
| `-a, --alignment` | In-frame codon alignment (FASTA or NEXUS). Autodetected by content. |
| `-t, --tree` | Newick/NEXUS tree. **Optional** — if omitted, the tree embedded in the alignment is used; if it has no branch lengths they are estimated. |
| `-w, --weights` | Explicit local weights file. Overrides HF download. Can also be set via `HYPHAEON_WEIGHTS` (or legacy `AXOMEME_WEIGHTS`). |
| `--model-variant` | Named variant to fetch from Hugging Face (see [`list-models`](#list-models--enumerate-hugging-face-variants)). Env: `HYPHAEON_VARIANT`. |
| `-o, --output` | Path to write a detailed **JSON** report. |
| `-c, --csv` | Path to write a flattened **CSV** report. |
| `--cpu` | Force CPU inference (default auto-selects CUDA → MPS → CPU). |

Input alignments must be **in-frame codon** alignments (length divisible by 3).
Trees are matched to alignment taxa by name.

---

## Bundled example datasets

Everything in the examples below ships in [`examples/`](../examples/):

| Dataset | Alignment | Tree | Taxa | Codons | Domain |
| :--- | :--- | :--- | ---: | ---: | :--- |
| HIV-1 RT | `examples/HIV1_RT.fasta` | `examples/HIV1_RT.nwk` | 476 | 335 | Retroviral reverse transcriptase (drug resistance & epistasis) |
| Rhodopsin | `examples/RHO.fasta` | embedded / auto | 710 | 349 | Mammalian visual pigment (diving/dim-light adaptation) |
| Smc6 | `examples/Smc6.fasta` | `examples/Smc6.nwk` | 20 | 1,097 | Primate antiviral host restriction |
| Bat OAS1 | `examples/bat_oas1.fasta` | `examples/bat_oas1.nwk` | 18 | 351 | Chiropteran innate immunity |
| Camelid VHH | `examples/camelid.fasta` | `examples/camelid.nwk` | 212 | 96 | Single-domain antibody variable domain |

Precomputed reference outputs for several of these live in
[`examples/expected_results/`](../examples/expected_results/).

---

## Weights, model variants, and device selection

- If `model.safetensors` is present at the repository root (it is, in this
  checkout), it is the **default weights** and no network access is needed.
- Otherwise weights are downloaded from the Hugging Face repo
  `datamonkey/axomeme`, verified against sha256, and cached. Gated repos need
  `HF_TOKEN`.
- Override precedence: explicit `--weights PATH` **>** `--model-variant NAME`
  (HF download) **>** the default packaged weights.
- Device: auto-selects **CUDA**, then **Apple MPS**, then **CPU**; force CPU
  with `--cpu`.

List everything available on the Hub:

```bash
hyphaeon list-models
```

---

## Subcommand how-tos

### `predict` — site-level episodic diversifying selection (MEME emulation)

**Paper pillar:** Pillar 1 — *Ultra-Fast Identification of Codon Sites Under
Episodic Diversifying Selection* (`03_pillar1_site_selection_meme.tex`).

**What it does.** Emulates the HyPhy **MEME** per-site likelihood-ratio test
(LRT) for episodic diversifying selection with a single forward pass of the
`PhyloAxialTransformer`, at `>1,000×` per-locus and `>10,000×` proteome-scale
speedup. For each codon column it predicts a non-negative LRT, converts it to a
p-value via the MEME asymptotic mixture null
(`1/3·δ(0) + 2/3·(0.45·χ²₁ + 0.55·χ²₂)`), and applies Benjamini–Hochberg to get
FDR q-values. Invariant columns are forced to `LRT = 0`.

**Basic invocation:**

```bash
hyphaeon predict \
  -a examples/Smc6.fasta \
  -t examples/Smc6.nwk \
  -o examples/Smc6_results.json \
  -c examples/Smc6_results.csv
```

**Inputs:** in-frame codon alignment (`-a`), optional tree (`-t`).

**Key options:**

| Flag | Purpose |
| :--- | :--- |
| `-s, --max-species N` | Cap taxa via Faith's-PD downsampling (default: all). |
| `-b, --batch-size N` | Sites per chunk (default: hardware-adaptive). |
| `--no-prune-duplicates` | Keep 100%-identical duplicate sequences (default: collapse them). |
| `--attribute` | Add mechanistic attribution (see next section). |
| `--filter` | Add automated alignment-error screening (see below). |

**CSV output columns** (one row per codon site):

| Column | Meaning |
| :--- | :--- |
| `site` | 1-indexed codon position. |
| `axomeme_lrt` | Predicted MEME-style LRT (≥ 0). |
| `p_value` | Mixture-null p-value. |
| `q_value` | Benjamini–Hochberg FDR q-value. |
| `is_invariable` | `True` if the column is invariant (LRT forced to 0). |

With `--attribute`, additional columns appear on attributed rows:
`evolutionary_epoch`, `adaptation_mode`, `top_driver`, `top_mutation`.

**JSON output schema:**

```jsonc
{
  "alignment": "examples/Smc6.fasta",
  "tree": "examples/Smc6.nwk",         // or "embedded_in_alignment"
  "taxa_count": 20,
  "codon_count": 1097,
  "runtime_sec": 0.42,
  "filter_enabled": false,
  "artifacts_masked": [],               // populated only with --filter
  "attribution_enabled": false,
  "attributions": { },                  // keyed by 1-indexed site, only with --attribute
  "sites": [
    { "site": 1, "axomeme_lrt": 0.0, "p_value": 1.0, "q_value": 1.0, "is_invariable": true }
  ]
}
```

**Gotchas.**
- The tree is required conceptually but not as a flag: if `-t` is omitted the
  embedded NEXUS tree is used; branch lengths are estimated if missing.
- p-values are asymptotic-null approximations, not likelihood MLEs — treat them
  as MEME-concordant scores (validate with [`evaluate`](#evaluate--hyphy-meme-concordance)).

---

### `predict --attribute` — mechanistic feature attribution

**Paper pillar:** Pillar 1, *Mechanistic Probing of Learned Representations*.

**What it does.** For every site above `--attribution-min-lrt` (default `3.84`,
nominal p ≤ 0.05), it performs **single-taxon counterfactual perturbation**:
each non-consensus species is reverted to the ancestral state and the marginal
`ΔLRT` is measured, ranking the taxa that *drive* the selection signal and the
fraction of signal each explains. It also performs **evolutionary-epoch
decomposition** by weighted root patristic depth — *Recent Terminal / Tip
Sweep* (≥ 0.60), *Intermediate Subclade Burst* (0.35–0.60), and *Deep Ancestral
/ Basal Divergence* (< 0.35) — separating recurrent multi-lineage adaptation
from single-lineage sweeps.

```bash
hyphaeon predict \
  -a examples/Smc6.fasta -t examples/Smc6.nwk \
  --attribute --attribution-min-lrt 3.84 \
  -o examples/Smc6_attributed.json
```

**Output.** In the JSON, an `attributions` object keyed by 1-indexed site;
each record holds `predicted_lrt`, `consensus_aa`,
`when_selection_occurred.evolutionary_epoch`,
`when_selection_occurred.mode_of_adaptation`, and a ranked
`driving_species` list (`taxon`, `observed_aa`, `delta_lrt`,
`pct_signal_explained`). CSV rows gain `evolutionary_epoch`,
`adaptation_mode`, `top_driver`, `top_mutation`.

**Gotcha.** Attribution scales with the number of driving taxa per site; on
deep trees with many candidate sites this is the slow part of `predict`.

---

### `predict --filter` / `filter` — automated alignment-error screening

**Paper pillar:** Pillar 1 — *Cross-Clade Macroevolutionary Generalization and
Alignment Error Filtering* (the counterfactual-attribution artifact filter).

There are **two ways** to run this analysis:

1. **Inline** with `predict --filter` (screen, mask, and re-score in one pass).
2. **Standalone** with the dedicated `filter` subcommand (QC-only; emits a
   cleaned alignment plus an audit log with raw-vs-cleaned metric comparison).

**What it does.** A dual-stage detector: (1) an exact upper-tail
**hypergeometric scan** finds 1D spatial clusters of selected sites
(`p_local ≤ 0.01`); (2) within each cluster it computes the **Outlier
Contamination Index (OCI)** and flags a private frameshift when a single
isolated leaf carries ≥ 3 contiguous radical mutations against conserved
species. The guilty taxon's span is **surgically masked with `NNN`** and the
cleaned alignment is re-scored.

**Inline (predict):**

```bash
hyphaeon predict \
  -a examples/Smc6.fasta -t examples/Smc6.nwk \
  --filter --filter-out-aln examples/Smc6_cleaned.fasta \
  -c examples/Smc6_clean.csv
```

Relevant `predict` flags: `--filter`, `--filter-out-aln PATH`,
`--filter-p-thresh` (default `0.01`), `--min-patch-consec` (default `3`).
Masked artifacts are listed in the JSON under `artifacts_masked`.

**Standalone (`filter`):** aliases `mask`, `qc`, `clean`.

```bash
hyphaeon filter \
  -a examples/Smc6.fasta -t examples/Smc6.nwk \
  -o examples/Smc6_cleaned.fasta \
  -c examples/Smc6_audit.csv
```

For `filter`, `-o/--output` is the **cleaned alignment (FASTA)** and
`-c/--csv` is the **artifact audit log**. Tuning flags: `--alpha-site`
(default `0.05`), `--min-k` (default `3`), `--max-span` (default `35`),
`--min-oci` (default `0.25`), `--min-run-length` (default `3`).

**`filter` audit output** reports, per masked patch: `patch_start_1idx`,
`patch_end_1idx`, `span_codons`, `significant_sites_k`, `p_hypergeom`,
`outlier_taxon`, `consecutive_mismatches`, `outlier_contamination_index`, plus a
raw→cleaned metric comparison (Cauchy/CCT omnibus p-value, significant-site
counts, mean LRT).

**Gotcha.** `filter` outputs a **FASTA alignment** at `-o`, not a JSON report —
this is the one subcommand where `-o` is not JSON. Use `predict --filter -o` if
you want the JSON selection report *and* filtering in one run.

---

### `busted` — alignment-wide omnibus selection (BUSTED / BUSTED+S emulation)

**Paper capability:** whole-gene episodic selection with SRV false-positive
filtering (Architecture capability #1 / gene-level selection).

**What it does.** Pools site representations through a multi-query
cross-attention `BustedMultiTaskHead` to produce a **gene-level** verdict:
a positive-selection probability, a predicted gene LRT, a synonymous-rate
variation (SRV) estimate, an inferred 3-class ω mixture, and two omnibus
p-values (**ACAT** Cauchy combination and **Simes**). Runs a single alignment
or a whole directory in **batch** mode.

**Single alignment:**

```bash
hyphaeon busted \
  -a examples/HIV1_RT.fasta -t examples/HIV1_RT.nwk \
  -o examples/HIV1_RT_busted.json \
  -c examples/HIV1_RT_busted.csv
```

**Batch directory (high-throughput):**

```bash
hyphaeon busted \
  -d examples/ --pattern "*.fasta" \
  --tree-suffix .nwk \
  -c busted_batch.csv -o busted_batch.json
```

**Input options:** `-a` (single file, or comma-separated list) **or** `-d`
(directory) — one is required. In batch mode, trees are auto-located by
appending `--tree-suffix` (default `.raxml.bestTree`) to each alignment name, or
searched in `--tree-dir`; `.nwk` / `.tree` are also tried. `-s/--max-species`
defaults to `512` (farthest-point-traversal subsampling above that).

**CSV summary columns:** `Gene`, `Taxa`, `Sites`, `p_ACAT`, `p_Simes`,
`Selection_Prob`, `Pred_Gene_LRT`, `Omnibus_LRT`, `Omega_3`, `Prop_Positive`,
`Sig_Sites_p05`, `Selected`, `Time_ms`.

**JSON per-gene record** includes the CSV fields plus
`p_value_acat`, `p_value_simes`, `omnibus_lrt`, `predicted_gene_lrt`,
`selection_probability`, `synonymous_rate_variation`,
`total_selection_energy`, `sig_sites_p05`, `sig_sites_p10`,
`rate_distributions` (`omega_1..3`, `proportion_1..3`),
`positive_selection_detected`, `elapsed_seconds`. In batch mode the JSON is a
list; single mode is a single object.

**Gotchas.**
- Batch tree matching is suffix-based; if your trees don't follow the
  `<alignment><suffix>` convention, pass `--tree-dir` or run per-gene with `-t`.
- A gene is called positive when `p_ACAT < 0.05` **or** the neural probability
  `> 0.50`.

---

### `epistasis` — branch co-selection, contacts & epistatic sectors

**Paper pillar:** Pillar 2 — *Epistatic Co-Selection, Macromolecular Contacts,
and Epistatic Sectors* (`04_pillar2_epistatic_coevolution.tex`).

**What it does.** Recovers inter-site co-evolution *without* any pairwise
training: it computes per-branch attribution vectors, then measures pairwise
**cosine co-selection** between site attribution vectors, tests shared
mutated-branch overlap with an exact **tree hypergeometric** test, and combines
them into the **Composite Epistatic Selection Index (CESI)**. It then does
graph-theoretic **sector mining** (two-stage seed-and-extend with Jaccard
overlap suppression) yielding coherent allosteric sectors, and by default a
per-site **Selection DMS plasticity** sweep. Emergent `Cβ–Cβ < 8 Å` contact
recovery is reported via Average Product Correction (APC) in the paper.

```bash
hyphaeon epistasis \
  -a examples/HIV1_RT.fasta -t examples/HIV1_RT.nwk \
  -o examples/HIV1_RT_epistasis.json \
  -c examples/HIV1_RT_edges.csv \
  --graphml examples/HIV1_RT_coselection.graphml
```

**Aliases:** `coselection`, `sector`, `network`.

**Key options:** `--min-sim` (cosine threshold, default `0.30`),
`--min-shared` (min shared mutated branches, default `2`), `--max-fdr`
(default `0.05`), `--min-lrt` (site drive threshold, default `1.0`),
`--min-clique-size` (sector seed, default `3`), `--max-overlap` (max Jaccard
between sectors, default `0.50`), `--focal-taxon`, `--no-dms` (skip the
per-site DMS plasticity sweep), `--graphml PATH` (export network for
Cytoscape/Gephi).

**CSV (edges) columns:** `site_u`, `site_v`, `ref_u`, `ref_v`, `lrt_u`,
`lrt_v`, `similarity`, `shared_branches`, `branches_u`, `branches_v`, `cesi`,
`p_hyper`, `fdr_q`.

**JSON schema** top-level keys: `taxa_count`, `codon_count`, `branch_count`,
`edges` (list, as above), `sectors` (each with `sector_id`, `size`, `sites`,
`spectral_coherence`, `mean_lrt`, `pars_signature`, optional `focal_taxon`/
`focal_signature`/`focal_mutations`), and `plasticity` (per-site DMS; see
[`dms`](#dms--digital-deep-mutational-scanning-ddms--essm) columns).

**GraphML.** `--graphml` writes an undirected co-selection network with edge
attributes `weight` (=similarity), `cesi`, `shared`, `fdr_q`.

**Gotcha.** Sector mining and the DMS sweep make `epistasis` heavier than
`predict`; use `--no-dms` when you only need the co-selection network.

---

### `dms` — digital deep mutational scanning (dDMS / ESSM)

**Paper pillar:** Pillar 3 — *Digital Deep Mutational Scanning (dDMS) and
Experimental Mutational Landscapes* (`05_pillar3_digital_dms_clinical_cpd.tex`).

**What it does.** Sweeps all 19 alternative amino acids at every codon position
in seconds, producing the **Epistatic Selection Sensitivity Matrix (ESSM)** and
per-site **Intrinsic Mutational Plasticity** — a scalar constraint score
distinguishing permissive/evolvable sites (high plasticity) from rigid
catalytic-backbone sites (low plasticity). Zero-shot: no fitness supervision.

```bash
hyphaeon dms \
  -a examples/HIV1_RT.fasta -t examples/HIV1_RT.nwk \
  -o examples/HIV1_RT_dms.json \
  -c examples/HIV1_RT_dms.csv
```

**Aliases:** `essm`, `digital-dms`. **Option:** `--focal-taxon` (default
auto/consensus).

**CSV columns:** `site`, `wt_aa`, `focal_taxon`, `baseline_lrt`, `p_value`,
`intrinsic_plasticity`, `max_shift`.

**JSON schema:** `taxa_count`, `codon_count`, `total_mutations`, and
`plasticity` (list of per-site records; the JSON additionally carries the full
per-mutant `mutant_deltas` that the CSV drops for compactness).

**Gotcha.** The per-mutant ΔLRT matrix (`mutant_deltas`) is JSON-only; the CSV
is the compact per-site summary.

**CPD note (paper-ahead-of-code):** the README advertises `dms` as also
predicting *Compensated Pathogenic Deviations* / de novo compensatory partners
(`s_comp`). That rescue-partner discovery is **not** surfaced by the `dms`
CLI output; the compensatory/co-selection signal is instead accessible via
[`epistasis`](#epistasis--branch-co-selection-contacts--epistatic-sectors)
(CESI edges). See [gaps](#paper-ahead-of-code-gaps).

---

### `disease` — zero-shot clinical variant pathogenicity

**Paper pillar:** Pillar 3 — *Zero-Shot Clinical Pathogenicity and Variant
Effect Prediction: The ProteinGym Benchmark* (same file, clinical half).

**What it does.** Scores missense variants for pathogenicity in [0, 1] from
evolutionary constraint, mapping each variant's canonical human position onto
the alignment column and modulating an evolutionary log-odds by a
co-evolution-centrality term. No labels or training — zero-shot.

```bash
hyphaeon disease \
  -a examples/Smc6.fasta \
  -m "R175H,G245S,P250L" \
  -o Smc6_variants.json \
  -c Smc6_variants.csv
```

**Aliases:** `pathogenicity`, `variant`, `clinvar`.

**Inputs:** `-a` alignment; `-m/--mutations` accepts an inline list
(`"R175H,G245S"`), a CSV file, or a Parquet path. Optional
`--canonical-seq` (canonical human reference sequence or FASTA path) and
`--human-taxon` (name of the human row; auto-detected otherwise).
`-b/--batch-size` defaults to `64`.

**Output columns** (`-c` CSV / `-o` JSON records): `mutation`,
`canonical_pos`, `wt_aa`, `mut_aa`, `aln_col`, `is_mapped`, `concordant_wt`,
`coevol_centrality`, `evolutionary_log_odds`, `pathogenicity_score`,
`prediction` (`Pathogenic` if score ≥ 0.50 else `Benign`; unmapped variants in
deleted exons/isoforms are flagged `Unmapped ...`).

**Gotchas.**
- Provide a canonical human reference (`--canonical-seq`) when the alignment's
  coordinates don't match the mutation numbering, or mapping may fail
  (`is_mapped = False`).
- `--mutations` is required and must use 1-letter WT/MUT codes with the
  canonical position, e.g. `R175H`.

---

### `phenotype` — directional phenotype–genotype association (PhyloWAS / PARS)

**Paper pillar:** *Phenotype–Genotype Association via Directional Attribution
Projection* (`05b_phenotype_genotype_attribution.tex`).

**What it does.** Directional trait mapping on the unit hypersphere: projects
per-branch attribution onto a foreground/background (or continuous) trait
contrast to find codon sites whose selection signal aligns with the phenotype.
Reports spectral trait energy (Ψ), the normalized spectral ratio, exact
sequenced-taxa-null p-values with Benjamini–Hochberg FDR q-values, a compact
**Phenotype-Associated Residue Signature (PARS)**, trait-restricted epistatic
sectors, and co-selection pairs. Supports optional Brownian-motion
**permulations** (RERconverge-style null) for a gene-level empirical p-value.

**Inline foreground list:**

```bash
hyphaeon phenotype \
  -a examples/RHO.fasta \
  -fg "turTru,balMus,balPhys,orcOrc,delDelp,phyCat,phoVit,halGryp,mirLeo,zalCali,odoRos" \
  -o examples/RHO_marine_phenotype.json \
  -c examples/RHO_marine_sites.csv
```

**Curated preset:**

```bash
hyphaeon phenotype -a examples/RHO.fasta -p marine -o rho_marine.json -c rho_marine.csv
```

**Aliases:** `phylowas`, `trait`.

**Trait-specification options (choose one):**

| Flag | Meaning |
| :--- | :--- |
| `-p, --preset` | Curated foreground set. Available: `echolocation`, `marine`, `fossorial`, `hibernation`, `longevity`, `high_altitude`, `cardenolide`, `dim_light`. |
| `-fg, --foreground` | Inline comma-separated taxa or a regex/glob pattern. |
| `-bg, --background` | Explicit background/control set (optional). |
| `-pf, --phenotype-file` with `-tc/--trait-col`, `-sc/--species-col` | Trait table (CSV/TSV) mapping taxa → values. |
| `--continuous` | Treat trait values as continuous phylogenetic contrasts. |

**Other options:** `--permulations N` (Brownian permulations for empirical
gene p-value; default `0` = parametric), `--min-taxa` (min sequenced taxa per
site, default `4`), `--alpha` (FDR threshold, default `0.05`).

**CSV (sites) columns:** `site`, `ref_aa`, `derived_aa`, `total_mutations`,
`shared_foreground_mutations`, `expected_shared`, `association_rho`, `p_value`,
`foreground_freq_pct`, `background_freq_pct`, `q_value`.

**JSON schema** top-level keys include: `taxa_count`, `codon_count`,
`spectral_energy`, `norm_spectral_ratio`, `permulations_count`,
`gene_p_value_perm`, `significant_sites_count`, `compact_pars_signature`,
`phenotype_meta` (`description`, `foreground_count`, …), `sites` (as above,
plus a ranking `score`), `trait_sectors`, and `coselection_pairs`.

**Gotchas.**
- Exactly one trait source is used; presets are name-globbed against alignment
  taxa, so foreground membership depends on your taxon-naming scheme.
- Permulations add runtime linearly in `N` — start with a few hundred.

---

### `evaluate` — HyPhy MEME concordance

**Paper context:** the site-level benchmarking of `predict` against HyPhy MEME
throughout Pillar 1 (ROC-AUC / PR / correlation reporting).

**What it does.** Compares `hyphaeon predict` site CSVs against matched HyPhy
MEME JSON results, **pooling all matched sites across genes** before computing
metrics (not per-gene-then-averaged). "True" here means *concordant with MEME*,
not independent ground truth.

**Folder mode** (pairs `Gene1.csv` ↔ `Gene1.MEME.json` by stem):

```bash
hyphaeon evaluate \
  --predictions-dir /path/to/hyphaeon_predictions/ \
  --meme-dir /path/to/meme_results/ \
  --output pooled_metrics.json
```

**Single-gene mode:**

```bash
hyphaeon evaluate \
  --prediction /path/to/Gene1.csv \
  --meme-result /path/to/Gene1.MEME.json \
  --output Gene1_metrics.json
```

**Inputs:** mutually exclusive `--predictions-dir` **or** `--prediction`, and
`--meme-dir` **or** `--meme-result` (directory and single-file flags cannot be
mixed).

**Options:** `--format {text,json}` (stdout format; default `text`),
`--output FILE.json` (detailed report), `--variable-only` (drop
`is_invariable` rows from metrics; totals still count them),
`--allow-unmatched` (folder mode; ignore genes without a counterpart),
`--allow-site-mismatch` (use the site intersection and report dropped counts),
`--prediction-suffix` (default `.csv`), `--meme-suffix` (default `.MEME.json`).

**Metrics reported:** total/evaluated sites, Pearson r and Spearman ρ on LRT,
and at α ∈ {0.05, 0.10}: ROC-AUC, PPV, FPR, with full TP/FP/TN/FN confusion
matrices in the JSON. A TP is a site called significant by both MEME and
HyphAeon. Undefined metrics (e.g. ROC-AUC with a single MEME class) are JSON
`null`. Negative MEME LRT artifacts are clamped to zero with a warning.

**Gotcha.** By default any unmatched gene or unequal site set **stops**
evaluation to prevent silent misalignment — opt into leniency with
`--allow-unmatched` / `--allow-site-mismatch`.

---

### `list-models` — enumerate Hugging Face variants

```bash
hyphaeon list-models
```

Lists variant names/descriptions from the `datamonkey/axomeme` Hub repo and
marks the default. Gated repos require `HF_TOKEN`. Use a listed name with
`--model-variant`.

---

## Paper-vs-CLI coverage matrix

Every analysis discussed in the manuscript, mapped to the CLI. ✅ = fully
covered by a subcommand; ⚠️ = partially covered / reachable via a different
subcommand than the README implies; ❌ = no CLI subcommand (paper-ahead-of-code).

| Paper analysis (section) | CLI subcommand | Status |
| :--- | :--- | :---: |
| Site-level episodic diversifying selection, MEME emulation (Pillar 1) | `predict` | ✅ |
| Mechanistic feature attribution / driving-taxon & epoch decomposition (Pillar 1) | `predict --attribute` | ✅ |
| Automated alignment-error filtering via counterfactual attribution (Pillar 1) | `predict --filter`, `filter` | ✅ |
| Cross-clade generalization (avian/OrthoMaM screens) (Pillar 1) | `predict` / `busted` (batch) | ✅ |
| Clade-specific selection profiling across mammalian orders (Pillar 1) | `predict` per-clade / `phenotype` | ⚠️ |
| Statistical power & error calibration under simulation (Pillar 1) | *(analysis/benchmark workflow — uses `predict`)* | ⚠️ |
| Mechanistic probing of learned representations (Pillar 1) | `predict --attribute` | ✅ |
| Alignment-wide omnibus / gene-level selection + SRV filtering | `busted` | ✅ |
| Pairwise epistatic co-selection & CESI (Pillar 2) | `epistasis` | ✅ |
| Macromolecular contact recovery via APC (Pillar 2) | `epistasis` (from co-selection edges) | ⚠️ |
| Graph-theoretic sector mining & spectral coherence (Pillar 2) | `epistasis` | ✅ |
| Direct-coupling-analysis comparison (Pillar 2) | *(external benchmark using `epistasis`)* | ⚠️ |
| Digital DMS / ESSM intrinsic plasticity (Pillar 3) | `dms` | ✅ |
| Compensated Pathogenic Deviations / compensatory partners (`s_comp`) (Pillar 3, README) | *(no dedicated output)* | ❌ |
| Zero-shot clinical pathogenicity / ProteinGym / ClinVar (Pillar 3) | `disease` | ✅ |
| Directional phenotype–genotype association / PhyloWAS / PARS | `phenotype` | ✅ |
| Gene-level trait permutation significance | `phenotype --permulations` | ✅ |
| CSUBST / ESL-PSC / convergence-benchmark comparisons | *(external benchmark using `phenotype`)* | ⚠️ |
| HyPhy MEME concordance evaluation | `evaluate` | ✅ |
| Retraining / fine-tuning | `scripts/build_training_npz.py` + `train.py` | ✅ (scripts, not a subcommand) |

---

## Paper-ahead-of-code gaps

These paper analyses have **no corresponding first-class CLI output**:

1. **Compensated Pathogenic Deviations (CPD) / de novo compensatory partner
   prediction (`s_comp`).** The README's `dms` bullet claims the tool
   "de novo predicts compensatory partners that rescue human disease
   mutations." The `dms` subcommand (`hyphaeon dms`) emits only per-site
   plasticity / ESSM (`site, wt_aa, focal_taxon, baseline_lrt, p_value,
   intrinsic_plasticity, max_shift`) — there is **no compensatory-partner or
   `s_comp` field** in its CSV/JSON. The closest available signal is co-selection
   edge discovery via `epistasis` (CESI), but that is not the same as
   variant-specific rescue-partner prediction. **Flagged: paper-ahead-of-code.**

2. **README/CLI naming drift for MEME concordance.** The README's capability
   list #6 calls it `hyphaeon evaluate` in its prose but titles it
   "`hyphaeon evaluate` (HyPhy MEME Concordance)" — the subcommand does exist
   and is named `evaluate` (confirmed in `cli.py`). No gap, but note the
   capability was numbered separately from the CLI reference table. *(Not a gap;
   noted for reviewers.)*

3. **Contact-map (Cβ–Cβ < 8 Å) and DCA/CSUBST/ESL-PSC/ProteinGym benchmark
   comparisons** are *analyses in the paper* rather than CLI subcommands: they
   are produced by running the relevant subcommand (`epistasis`, `phenotype`,
   `disease`) and comparing to external tools/datasets offline. The raw model
   outputs (co-selection edges, association scores, pathogenicity scores) are
   available, but there is no turnkey `hyphaeon contacts` / `hyphaeon benchmark`
   command. Marked ⚠️ in the matrix.

Everything else in the manuscript maps cleanly onto a subcommand.
