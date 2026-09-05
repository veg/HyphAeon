# Reference (Python) defects and quirks found while porting HyphAeon to JavaScript

One consolidated, deduplicated checklist for the ML team: every bug, silent fallback, dead
parameter and CLI/library disagreement the JavaScript port met between Phase 0 and Phase 3a, with
the line it lives on, what it does to a user, where the port reproduces it, what to do about it,
and whether fixing it changes numbers (and therefore needs `fixtures/` regenerated).

**Nothing in this list has been fixed.** PLAN.md §5.3 rule 3 is the reason: *a Python bug is
replicated in the port, flagged, fixed upstream, the fixtures regenerated, and only then fixed in
JavaScript* — otherwise "parity" becomes a matter of opinion and the port silently becomes the
second opinion. The library therefore reproduces every item marked "replicated" on purpose, with
the citation in the module header and a test that pins it.

## How to read an entry

* **Where** — `file:line` at commit `61d681a` (branch `feat/js-port`, tag `phase-3a`).
  `hyphaeon/cli.py`, `hyphaeon/dataset.py` and `hyphaeon/phenotype.py` changed during the port
  (the `--mds-sign`, `--seed` and export additions); every other module is unchanged since
  `main@3cb9cc6`, so its line numbers are valid on `main` too.
* **In the port** — the JavaScript file that reproduces the behaviour, so a fix upstream has a
  matching place to land.
* **Output change** — *yes* means fixing it changes numbers or keys a user sees, so
  `scripts/gen_fixtures.py` must be re-run, `js/test` re-pinned, and the affected
  `fixtures/e2e/*.json` and `examples/*` regenerated (see "Regenerating" at the end). *no* means
  the fix is invisible on inputs that work today.
* **Severity** — *user-visible* (a wrong number, a crash, or a silently different answer reaches
  the user), *latent* (unreachable today, would bite after an unrelated change), or *cost/clarity*.

## Where this came from

`PHASE0.md`, `PHASE1A.md`, `PHASE2A.md`, `PHASE3A.md`, `MDS_SIGN.md`, `fixtures/manifest.json`
(`known_quirks`), the `WHY THIS FILE EXISTS` headers under `js/src/`, and the app repository's
`PHASE0–3.md` measurements. Each item was re-checked against the source at `61d681a` for this
document; anything that turned out to be a misreading is in **§F, checked and benign**.

---

## Summary

| # | Item | Group | Severity | Output change |
|---|---|---|---|---|
| [C1](#c1) | TN93 raises where `dataset.py` expects a sentinel | crash | user-visible | no (bundled examples unaffected) |
| [C2](#c2) | `meme --filter` crashes at the cleaned re-load with an embedded tree | crash | user-visible | yes |
| [C3](#c3) | A bare `>` FASTA header raises `IndexError` | crash | user-visible | no |
| [C4](#c4) | `--max-species 0` divides by zero | crash | user-visible | no |
| [C5](#c5) | DMS writes past the site block for a non-standard wild-type residue | crash | latent | no |
| [W1](#w1) | The codon vocabulary is not published, and DM3/AxoMEME disagrees with it | wrong results | user-visible (elsewhere) | no |
| [W2](#w2) | Identical sequences get the LARGEST TN93 distance | wrong results | user-visible | yes |
| [W3](#w3) | Tier-2/3 name matching leaves zero distance rows (#9) | wrong results | user-visible | no |
| [W4](#w4) | The `> 10` rescale is decided after branch lengths are nudged | wrong results | user-visible | conditional |
| [W5](#w5) | `--max-species 1` returns two taxa; a degenerate matrix repeats a taxon | wrong results | user-visible | conditional |
| [W6](#w6) | The stride pre-selection defeats Faith's PD | wrong results | user-visible | yes |
| [W7](#w7) | A float trait column empties the phenotype foreground | wrong results | user-visible | conditional |
| [W8](#w8) | Trait tables match taxa by substring, both directions, in row order | wrong results | user-visible | conditional |
| [W9](#w9) | Continuous traits are z-scored over unmatched taxa | wrong results | user-visible | conditional |
| [W10](#w10) | `find_any(name=…)` treats a taxon name as a regular expression | wrong results | user-visible | no |
| [W11](#w11) | Sector `shared_taxa` describes the unpruned site set | wrong results | user-visible | yes |
| [W12](#w12) | Float32 cosine > 1 and float32 threshold comparisons decide edges | wrong results | user-visible | yes |
| [W13](#w13) | BUSTED: Simes over all sites, ACAT over variable sites | wrong results | user-visible | yes |
| [W14](#w14) | Attribution reports a gap consensus as codon `TGA` / `*` | wrong results | user-visible | yes |
| [W15](#w15) | The DMS record names a focal taxon it did not sweep | wrong results | user-visible | yes |
| [W16](#w16) | Trait co-selection pairs are emitted in score order | wrong results | cost/clarity | yes (key order) |
| [W17](#w17) | The artifact scan reports nothing when every site is significant | wrong results | user-visible | conditional |
| [W18](#w18) | Parsers: PHYLIP, NEXUS labels, `TREE x = [&R] (…)` | wrong results | user-visible | conditional |
| [W19](#w19) | `forward()`'s micro-batch branch does not slice the tree tensors | wrong results | latent | no |
| [S1](#s1) | A missing `--weights` / `HYPHAEON_WEIGHTS` path is silently ignored | silent fallback | user-visible | no |
| [S2](#s2) | **The BUSTED head is loaded incomplete and unseeded** | silent fallback | user-visible | yes |
| [S3](#s3) | `load_model(strict=False)` accepts any checkpoint | silent fallback | user-visible | no |
| [S4](#s4) | Branch lengths come from whatever HyPhy is on PATH, or from 1e-3 | silent fallback | user-visible | yes |
| [S5](#s5) | The Lanczos MDS path falls back to dense on any exception | silent fallback | user-visible | no |
| [S6](#s6) | A malformed tree is reported as a missing tree | silent fallback | user-visible | no |
| [S7](#s7) | TN93 comes from the binary or the package depending on `PATH` | silent fallback | user-visible | no (today) |
| [S8](#s8) | Permulation failures are swallowed whole | silent fallback | user-visible | no |
| [D1](#d1) | `max_overlap` is declared, threaded through the CLI, never read | dead code | cost/clarity | yes if implemented |
| [D2](#d2) | `branch_names`, `shared_branches`, `hyper_p`: an older formulation's names | dead code | cost/clarity | keys only |
| [D3](#d3) | `--background` / `background=` is accepted and never read | dead code | user-visible | yes if implemented |
| [D4](#d4) | The `d == "-"` TN93 guard is unreachable | dead code | cost/clarity | no |
| [D5](#d5) | `[-k_foreground:]` with k = 0 would return every taxon | dead code | latent | no |
| [D6](#d6) | The permulation projection is computed twice | cost | cost/clarity | no |
| [D7](#d7) | The DMS wild-type residue is computed twice per site | cost | cost/clarity | no |
| [D8](#d8) | `plasticity` and `selection_dms_plasticity` are one list | cost | cost/clarity | keys only |
| [D9](#d9) | `ppv_confusion` and `fpr_confusion` are the same computation | dead code | cost/clarity | no |
| [I1](#i1) | `meme --filter` is a second, divergent copy of `filter.py` | CLI vs library | user-visible | yes |
| [I2](#i2) | The epistasis CLI and the function disagree on `min_sim` | CLI vs library | user-visible | no |
| [I3](#i3) | No library entry points; `sys.exit` instead of exceptions | CLI vs library | user-visible | no |
| [I4](#i4) | `--seed`: the history, and what is still unreproducible | CLI vs library | user-visible | no |
| [I5](#i5) | `--mds-sign canonical` landed; the committed outputs did not follow | CLI vs library | user-visible | yes (already) |
| [I6](#i6) | Four copies of the amino-acid tables | CLI vs library | cost/clarity | no |

---

## A. Crashes

### C1
**TN93 raises where `dataset.py` expects a sentinel, and nothing catches it**

* **Where** `hyphaeon/dataset.py:540-547` — `from tn93.tn93 import TN93`, then per pair
  `get_counts` / `get_nucleotide_frequency` / `calculate_distance` (544-546) guarded by
  `if d is None or d == "-" or d < 0 or np.isnan(d): d = 1.0` (547).
* **What happens** The guard anticipates a sentinel, but the package never returns one on this
  path: it *raises*. A saturated pair (the corrected proportions go non-positive) reaches
  `math.log` of a non-positive number → `ValueError`; a pair with no overlapping non-gap position
  divides by zero in `2 / sum(freq)` → `ZeroDivisionError`. Neither is caught anywhere between
  here and `main()`.
* **Why it matters** User-visible: `hyphaeon meme --use-tn93` (and every other `--use-tn93`
  command) dies with a raw traceback on an alignment containing one bad pair, after loading the
  model. Under PLAN.md D22 the app takes this path by default whenever an upload has no usable
  tree, so it is the first thing a real user's messy alignment hits.
* **In the port** `js/src/preprocess/tn93.js:258` (ValueError) and `:551` (ZeroDivisionError)
  throw with the Python exception name; `js/src/diagnostics.js` catches them and reports
  `TN93_SATURATED_PAIRS` at refuse level so the app can say which sequences are the problem.
* **Fix** Wrap the per-pair call in `try/except (ValueError, ZeroDivisionError)` and use the
  `d = 1.0` the guard already writes, or better, count and report those pairs. Keep the sentinel
  distinguishable from a real distance of 1.0 if you can.
* **Output change** No — on the five bundled examples no pair reaches it (measured), so the
  fixtures do not move; only alignments that crash today change, from a traceback to a result.

### C2
**`hyphaeon meme --filter` crashes at the cleaned re-load when the tree was embedded**

* **Where** `hyphaeon/cli.py:210` — the cleaned temporary FASTA is re-loaded with `args.tree`,
  which is `None` when the tree came embedded in the alignment; the cleaned file carries no tree,
  so `dataset.py:646-651` raises "No tree specified …". `hyphaeon/filter.py:297` does it correctly
  (`effective_tree = tree_path if (tree_path is not None or use_tn93) else alignment_path`).
* **What happens** Every artifact-masking run on an alignment with an embedded tree — RHO, for
  instance — does the full scan, masks, writes the temporary file and then dies.
  `cli.py:214-215` compounds it: the cleaned re-score reuses the *baseline* `tree_cache`, so if
  masking ever changed the taxon set the model would fail on a shape mismatch instead.
* **Why it matters** User-visible: a documented flag (README Example 4) is unusable on one of the
  bundled examples' input shapes, and the failure comes after the expensive part.
* **In the port** `js/src/filter.js` behind `{cliVariant: true}` (header items 5–6);
  `fixtures/e2e/meme_bat_oas1_attribute_filter.json` pins `cmd_meme: null`. The app's runtime
  completes that re-load through the tree-free path — the one place the port deliberately outruns
  the reference (`PHASE3A.md`, "Deliberate divergences", item 4).
* **Fix** Use `filter.py:297`'s expression in `cmd_meme`, and recompute the tree cache from the
  cleaned distances rather than reusing the baseline. Better: delete the copy (see [I1](#i1)).
* **Output change** Yes — a result appears where there was a crash. Regenerate
  `fixtures/e2e/meme_*_filter*.json`.

### C3
**A FASTA header line that is exactly `>` raises `IndexError`**

* **Where** `hyphaeon/dataset.py:107` — `curr_id = l_strip[1:].split()[0].strip("'\"")` on an
  empty split list.
* **What happens** `IndexError: list index out of range`, with no mention of the file or the line.
* **Why it matters** User-visible but small: an empty header is a corrupt file, and the user
  deserves to be told which line.
* **In the port** `js/src/preprocess/parse.js:172` throws the same way, pinned by
  `fixtures/dataset/parse_alignment_sequences.json`.
* **Fix** Skip the record with a warning, or raise `ValueError` naming the line number.
* **Output change** No, on inputs that parse today; the fixture case that pins the raise changes.

### C4
**`--max-species 0` divides by zero**

* **Where** `hyphaeon/dataset.py:627` (tree-free) and `:726` (tree path) —
  `stride = max(1, len(taxa) // (max_species * 2))`.
* **What happens** `ZeroDivisionError` before anything useful. `downsample_taxa_faith_pd`
  (`:446-448`) already treats `max_species <= 0` as "no cap", so the two disagree.
* **Why it matters** User-visible: a plausible typo on `-s` becomes a traceback.
* **In the port** `js/src/preprocess/downsample.js:129-132` reproduces the exception and its
  message.
* **Fix** Skip the stride when `max_species <= 0` (matching the downsampler), or reject the value
  in argparse.
* **Output change** No.

### C5
**The DMS sweep would write past the site's row block for a non-standard wild-type residue**

* **Where** `hyphaeon/epistasis.py:531-536` (and its verbatim second copy at `:550-555`) — the
  19 mutants of a site are written into a fixed 19-row block indexed by the position of the
  wild-type residue among the 20 standard ones.
* **What happens** Nothing today: the wild-type fallback always yields one of the 20 residues, so
  the index is always in range. If that fallback ever changed, the sweep would overwrite the next
  site's rows and report a neighbouring site's mutants as this site's.
* **Why it matters** Latent, and silent if it ever fires — no exception, just wrong deltas.
* **In the port** `js/src/dms.js` throws a documented guard instead of reproducing the
  out-of-bounds write (the one place the port refuses to mirror the Python, recorded in
  `PHASE2A.md`).
* **Fix** Assert the residue is standard, or size the block from the candidate list.
* **Output change** No.

---

## B. Wrong results

### W1
**The codon vocabulary is not published anywhere, and DataMonkey 3 / AxoMEME uses a different one**

* **Where** `hyphaeon/dataset.py:25-34` (`GENETIC_CODE`: 61 sense codons → 0..60 in TCAG order,
  stops/gaps/unknown → 64), `:36-39` (`AA_MAP`), `:41-50` (`CODON_TO_AA`), `:52-57` (the two token
  functions). Nothing exports them; `models/manifest.json` does not carry them.
* **What happens** DataMonkey 3's in-browser AxoMEME port — byte-identical to the AxoMEME 2.0
  training scripts — numbers all 64 codons in TCAG order with gap 64 / unknown 65 and amino-acid
  sentinels 20/21/22. Measured at Phase 0: **54 of 64 codon tokens differ** from `dataset.py`, as
  do 17 of 21 codon sentinels and 18 of 21 amino-acid sentinels; the 20 residues agree. The
  shipped checkpoint wants `dataset.py`'s tables — swapping them (with the `> 10` rescale) lifted
  variable-site Spearman against `hyphaeon meme` on bat_oas1 from **0.13 to 0.94** (viral) and
  0.10 to 0.69 (general). `training_data.py` goes through `load_alignment_and_tree`, so
  `dataset.py` *is* the training vocabulary and the driver is the one that is wrong.
* **Why it matters** User-visible, in the wrong repository: DataMonkey 3's Predict panel is live
  and scoring with the wrong table today, and its own parity test passed — this is the handoff's
  central hazard, measured. Any future driver can repeat it, because there is no published
  contract to check against.
* **In the port** `js/src/preprocess/tokenizer.js` generates the tables from the TCAG codon list
  and `fixtures/dataset/tokenizer.json` compares them entry by entry against a verbatim dump of
  the Python's; `js/test/modelContract.test.js` pins the sentinels.
* **Fix** Publish the vocabulary as part of the model contract — the three tables plus the
  sentinels in `models/manifest.json` (and in the HuggingFace `config.json`) — and have every
  driver read it rather than transcribe it. Then fix DM3's copy (PLAN.md §8 phase 4 replaces it
  with `@veg/hyphaeon-js`).
* **Output change** No: this is documentation of what the reference already does.

### W2
**Two identical sequences are given the LARGEST distance in the TN93 matrix**

* **Where** `hyphaeon/dataset.py:559-568` — after every pair is scored,
  `max_d = dist.max() or 0.1`; then for `i < j`, a zero distance between **different** sequences
  becomes `1e-4` (563-565), and a zero distance between **byte-identical** sequences becomes
  `max(1.0, max_d)` (566-568).
* **What happens** The `elif` is reachable only when the two raw sequence strings are identical,
  so a duplicate pair — genuinely at distance 0 — is recorded as the most distant pair in the
  matrix, at least 1.0. Everything downstream reads it: the MDS embedding, Faith's PD selection,
  Tree-RoPE.
* **Why it matters** User-visible whenever duplicates survive:
  `load_alignment_and_tree` prunes identical sequences first (`:603-607`), so the CLI path is
  usually safe, but `prune_duplicates=False`, a direct call to
  `compute_tn93_distance_matrix`, or duplicates that differ only in a gap column all reach it.
  Measured on the bundled examples: 1 identical pair in HIV1_RT and 78 in RHO before pruning.
* **In the port** `js/src/preprocess/tn93.js:610-640` (`tn93DistanceMatrix`, `maxD`), replicated
  exactly, including the fact that `dist.max()` inside the loop is live while `max_d` is not.
* **Fix** Impute identical sequences to `0.0`, or to the same `1e-4` floor as the distinct case if
  a strictly positive distance is required. The current value looks like an inverted comparison.
* **Output change** Yes for any run that reaches it. Regenerate
  `fixtures/dataset/tn93_distance_matrix.json` and any `*_tn93` e2e case with duplicates.

### W3
**A taxon matched by tier-2/3 name matching keeps a zero distance row (issue #9)**

* **Where** `hyphaeon/dataset.py:309` builds `terminals` keyed by the **quote-stripped tree**
  name; `:330` and `:342` `continue` on a miss, leaving that taxon's row and column zero. The
  taxa list, however, can hold **alignment** spellings: `load_alignment_and_tree`'s second and
  third matching tiers (`:671-687`) map quote-stripped or case-folded tree names back to the
  alignment's keys.
* **What happens** Whenever matching needed tier 2 or 3, one or more taxa are at distance 0 from
  everything — they sit at the origin of the MDS embedding and the model is told they are
  identical to every other sequence. No warning; the run completes.
* **Why it matters** User-visible and silent: exactly the "well-formed tensor that means the wrong
  thing" class. Alignments whose FASTA names are quoted, or differ in case from the tree, are
  common.
* **In the port** `js/src/preprocess/patristic.js` (header, and the `ti === null` branch) keeps
  the zero rows deliberately, with the issue number in the comment.
* **Fix** Key `terminals` by the same normalisation that produced `taxa`, or raise when a taxon
  has no terminal. Reporting the count would be enough to make it visible.
* **Output change** No for the bundled examples (all match at tier 1); yes for affected
  alignments.

### W4
**The `> 10` rescale is decided after branch lengths have been nudged**

* **Where** `hyphaeon/dataset.py:668` (`enforce_nonzero_branch_lengths`, raising every length
  below `1e-4`, and every missing one to `1e-3`) runs before `:734-735`
  (`if dist_mat.max() > 10.0: dist_mat = dist_mat / L`).
* **What happens** A tree whose true maximum patristic distance is exactly 10.0 and whose path
  crosses a zero-length branch is nudged over the threshold, and the *entire matrix* is divided by
  the codon count. The rescale is all-or-nothing and unit-changing.
* **Why it matters** User-visible: the rescale is load-bearing (a chronogram in Mya — bat_oas1's
  max is 123 — scores at Spearman 0.68 with it and 0.13 without), so a threshold decided by a
  1e-4 nudge is a cliff. It is also undocumented in the CLI help.
* **In the port** `js/src/preprocess/patristic.js` (`rescaleDistances`, applied after enforcement)
  and `fixtures/dataset/rescale_rule.json`, which pins exactly this ordering.
* **Fix** Decide the rescale on the raw matrix, before enforcement; better, make branch-length
  units explicit (a flag, or a warning naming the inferred unit) instead of inferring from a
  magnitude.
* **Output change** Conditional — only for a matrix whose maximum sits within the nudge of 10.0.
  Regenerate `fixtures/dataset/rescale_rule.json` either way, since it pins the order.

### W5
**`--max-species 1` returns two taxa; a degenerate matrix selects the same taxon repeatedly**

* **Where** `hyphaeon/dataset.py:456-461` — the most distant pair is placed *before*
  `for _ in range(2, max_species)` runs, and each subsequent pick is `argmax` of the running
  minimum-distance vector.
* **What happens** `max_species = 1` yields 2 taxa. And when the remaining minimum distances are
  all zero (an alignment of identical or near-identical sequences), `argmax` returns index 0 every
  time, so the selection repeats a taxon; the duplicates are fed to the model as separate rows.
* **Why it matters** User-visible: `-s 1` silently ignores the cap, and a low-diversity panel gets
  a subsample containing the same sequence several times — precisely the shallow-panel regime
  where the model already returns nothing (issue #33).
* **In the port** `js/src/preprocess/downsample.js` (both quirks in the header, pinned by tests).
* **Fix** Slice the selection to `max_species`, and stop (or warn) when the running maximum
  distance is 0.
* **Output change** Conditional: only for `-s 1` and for degenerate matrices.

### W6
**The stride pre-selection truncates in tree order and defeats Faith's PD**

* **Where** `hyphaeon/dataset.py:625-628` (tree-free) and `:724-728` (tree path) —
  `stride = max(1, n // (max_species * 2))`, then `taxa = taxa[::stride][:max_species * 2]`,
  before the distance matrix is computed and before `downsample_taxa_faith_pd` runs.
* **What happens** For `n < 4 * max_species` the stride is 1, so this keeps the **first**
  `2 * max_species` taxa in tree (or alignment) order and throws the rest away *before* diversity
  is measured. The greedy PD selection then maximises diversity over a prefix.
* **Why it matters** User-visible whenever `--max-species` is set on a moderately sized panel —
  which is the app's default path (cap 256, hard max 512 per `models/manifest.json`), so RHO (710
  taxa) and HIV-1 RT (476) are decided partly by file order. Two orderings of the same alignment
  give different results.
* **In the port** `js/src/preprocess/downsample.js:129` (`stridePreselect`), pinned by the camelid
  case of `fixtures/dataset/downsample_taxa_faith_pd.json`.
* **Fix** Skip the pre-stride unless `n > 4 * max_species` (its evident intent — a cheap guard for
  "massive collections"), or sample evenly instead of truncating.
* **Output change** Yes for any capped run in that range. Regenerate
  `fixtures/dataset/downsample_taxa_faith_pd.json` and any capped e2e case.

### W7
**A float-typed trait column empties the phenotype foreground**

* **Where** `hyphaeon/phenotype.py:193` — discrete mode compares `str(val).strip().lower()`
  against `["1","true","yes","case","foreground","target","positive"]`.
* **What happens** pandas types a numeric column with any missing cell as `float64`, so `1`
  renders as `"1.0"`, matches nothing, and every taxon lands in the background. An `int64` column
  gives `"1"` and works. One blank row is the difference.
* **Why it matters** User-visible and silent: the run completes, `foreground_count` is 0, and every
  association statistic is computed against an all-background trait.
* **In the port** `js/src/phenotype.js` (`parsePhenotypeTable` reproduces pandas' dtype inference
  closely enough to preserve the distinction; the quirk is in the header).
* **Fix** Coerce numerically before the string comparison (`float(val) == 1`), or refuse when the
  resolved foreground is empty — a trait with no foreground is never a valid analysis.
* **Output change** Conditional; regenerate `fixtures/phenotype/resolve_phenotype_vector.json` if
  a case covers it.

### W8
**Trait tables match taxa by substring, in both directions, in row order**

* **Where** `hyphaeon/phenotype.py:187` — after an exact and a lower-cased lookup,
  `for k, v in trait_dict.items(): if k in t or t in k`. Plus `:173-174`, which stores every
  species under both its original and its lower-cased key, so two rows differing only in case
  overwrite each other (last row wins).
* **What happens** The row `aotTri` claims the taxon `aotTri_OMK`; a one-letter species name would
  claim almost everything; which row wins depends on file order.
* **Why it matters** User-visible: a trait table silently mislabels taxa, and the resulting
  foreground looks plausible.
* **In the port** `js/src/phenotype.js` (header quirk list), pinned by
  `fixtures/phenotype/resolve_phenotype_vector.json`.
* **Fix** Exact, then case-insensitive, then stop — and report the taxa that matched no row.
  If substring matching is wanted, make it opt-in and require uniqueness.
* **Output change** Conditional.

### W9
**Continuous traits are z-scored over all taxa, including unmatched ones**

* **Where** `hyphaeon/phenotype.py:213-214` — the *decision* to standardise uses
  `np.std(matched_values)` (matched taxa only), the *transform* uses `np.mean(y)` / `np.std(y)`
  over all N, where unmatched taxa are still 0. `foreground_count` and `background_count` are left
  at 0 in this mode (`:218` runs only in the discrete branch).
* **What happens** A continuous trait with any unmatched taxon is standardised against a vector
  padded with zeros, shifting and shrinking the real values.
* **Why it matters** User-visible: the association ρ per site is computed against a distorted
  trait, and nothing in the output says how many taxa were matched.
* **In the port** `js/src/phenotype.js:703` marks the line; the header states the rule.
* **Fix** Standardise over the matched values and treat unmatched taxa as missing (excluded), and
  record the matched count in `phenotype_meta`.
* **Output change** Conditional (any continuous run with unmatched taxa).

### W10
**`Bio.Phylo`'s `find_any(name=…)` treats a taxon name as a regular expression**

* **Where** `hyphaeon/phenotype.py:286` and `:288` (`compute_phylogenetic_covariance`), against
  `:283`, which builds `root_dists` by exact dictionary key.
* **What happens** Biopython's `_attribute_matcher` does `re.match(pattern + "$", target)`, so a
  taxon literally named `a.c` matches the terminal `aXc`, `a|b` matches a terminal named `a`, and
  a name containing an unbalanced bracket raises `re.error` out of the whole function. The
  diagonal (dict lookup) and the off-diagonal (regex) can therefore disagree about which tip a
  name means. Match order is `find_clades()` preorder, so an internal node can win over a tip.
* **Why it matters** User-visible for names with regex metacharacters — `|` and `.` are common in
  sequence identifiers. The Brownian covariance, and every permulation drawn from it, is then
  wrong; or the run dies inside the swallowed `except` of [S8](#s8) and silently loses its
  permulations.
* **In the port** `js/src/permulations.js` (`findAnyByName`) reproduces the regex semantics
  exactly, with a fast path for names containing no metacharacter.
* **Fix** Look tips up by exact name (a dict, as `root_dists` already does).
* **Output change** No for the bundled examples; yes for affected names.

### W11
**A sector's `shared_taxa` describes the site set before pruning**

* **Where** `hyphaeon/epistasis.py:345` binds `sub_A = attributions[site_indices, :]`; `:356-364`
  prunes weakly loaded sites and rebinds `site_indices` **without** rebinding `sub_A`; `:387`
  computes `shared_taxa_cnt` from the stale `sub_A`, while `:389-392`'s `pars_signature`,
  `mean_lrt` (`:388`) and the emitted `sites` use the pruned set.
* **What happens** One record describes two different site sets. `shared_branches` (`:429`) is the
  same number, so both fields are affected.
* **Why it matters** User-visible: `shared_taxa` is displayed as "how many taxa carry the whole
  sector", and for a pruned sector it counts a set that is not the sector.
* **In the port** `js/src/sectors.js` (step 8 in the header) replicates it, with a test.
* **Fix** Recompute `sub_A` from the retained sites before `:387`.
* **Output change** Yes on any sector that prunes. Regenerate
  `fixtures/epistasis/extract_epistatic_sectors_tse.json` and the epistasis e2e cases.

### W12
**A float32 cosine can exceed 1, and five thresholds are compared in float32**

* **Where** `hyphaeon/epistasis.py:161` (`sim = dot / max(outer(norms, norms), 1e-9)`, all
  float32), `:170` (`t = sim * sqrt(df / max(1e-9, 1 - sim**2))`), `:181-185` (the five threshold
  comparisons).
* **What happens** Two identical attribution rows give `sim = 1.0000001192`, so `1 - sim**2` is
  negative, the `max(1e-9, ·)` guard fires, and collinear pairs get *different* p-values (at
  df = 1, p = 1.0066e-5 rather than 0). Separately, numpy 2 weak-scalar promotion casts each
  threshold to float32 before comparing, and `float32(0.35) = 0.34999999404 < 0.35`, so a pair
  whose cosine is exactly the threshold **passes** where a float64 comparison would fail. The same
  applies to `min_cesi`, `max_fdr`, `min_lrt`, and to `v_dom >= 0.10` in the sector miner
  (`:357`).
* **Why it matters** User-visible: edge membership — which pairs appear in the network at all —
  is decided by float32 rounding within ~5e-7 of a threshold. It also makes bit-parity between any
  two implementations impossible in that band (measured: BLAS `sgemm`'s blocked accumulation is
  not reproducible by any fixed summation order).
* **In the port** `js/src/epistasis.js:550-670` compares against `Math.fround(threshold)` and
  keeps the guard; the header records the measurement.
* **Fix** Clip `sim` to `[-1, 1]` and compare thresholds in float64 (`sim_arr.astype(np.float64) >= min_sim`).
* **Output change** Yes, potentially adding or removing edges. Regenerate
  `fixtures/epistasis/compute_branch_coselection_network.json` and the epistasis e2e cases.

### W13
**BUSTED combines Simes over all sites and ACAT over variable sites**

* **Where** `hyphaeon/cli.py:493` (`var_p = pvals[variable_indices]` → `p_acat`) against
  `:496-499` (`sorted_p = np.sort(pvals)` over all L → `p_simes`). Invariable sites have LRT 0 and
  therefore p = 1 under Self–Liang.
* **What happens** Simes is diluted by every invariable site in the gene: Smc6 reports
  `p_value_simes = 1.0` while `p_value_acat` is 0.118. Related, at `:504`: the omnibus threshold
  is the rounded `3.841` rather than the χ²(1) 95 % quantile 3.8414588, which shifts
  `omnibus_lrt` slightly on every gene.
* **Why it matters** User-visible: two gene-level p-values with different meanings are printed
  side by side, and the more conservative one is an artefact of the site set, not the data.
* **In the port** `js/src/omnibus.js` reproduces both (the constant is named `BUSTED_LRT_THRESHOLD`
  with the rounding noted).
* **Fix** Run both combinations over the same site set (variable sites); use the exact quantile,
  or state that 3.841 is deliberate.
* **Output change** Yes. Regenerate `fixtures/e2e/busted_*.json` and `examples/*` busted outputs.

### W14
**Attribution reports a gap or unknown consensus as codon `TGA`, amino acid `*`**

* **Where** `hyphaeon/attribution.py:16` — `INV_GENETIC_CODE = {v: k for k, v in GENETIC_CODE.items()}`
  inverts a many-to-one mapping: all three stops (and every gap/ambiguous codon) map to token 64,
  and the last key wins. `:94-96` then reads token 64 back as `'TGA'` → `'*'` → AA token 20.
* **What happens** At a site whose most frequent codon token is 64 — a gappy column — the
  attribution record's consensus codon is `TGA` and its consensus amino acid is `*`, and the
  per-taxon mutation strings are written against that.
* **Why it matters** User-visible: `top_mutation` and `attribution_details` say something specific
  and false about a column that is mostly gaps.
* **In the port** `js/src/attribution.js:69` builds the same inverse and the header names the
  consequence; pinned by `fixtures/attribution/attribute_selection.json`.
* **Fix** Map token 64 to `'NNN'` / `'-'` explicitly, and skip the site or mark it rather than
  reporting a codon.
* **Output change** Yes on gappy sites. Regenerate the attribution fixtures and any
  `--attribute` e2e case.

### W15
**The DMS record names a focal taxon it did not sweep**

* **Where** `hyphaeon/epistasis.py:481-486` — the focal index defaults to 0 and is only replaced
  when `focal_taxon.lower() in t.lower()` matches; a miss is silent. `:764` then records
  `focal_taxon or (taxa[0] if taxa else "consensus")` — the **caller's string**, not the taxon
  swept. `:765` reports `total_mutations = 19 * L` whatever subset was actually swept.
* **What happens** `--focal-taxon beta` on an alignment with no such taxon sweeps `taxa[0]` and
  reports `"beta"`. A case difference (`Beta`) sweeps the right taxon and still reports the query.
* **Why it matters** User-visible: the DMS heatmap is labelled with a taxon that may not be the
  one the mutations were applied to.
* **In the port** `js/src/dms.js:147` (`resolveFocalTaxon`) and `digitalDmsRecord` reproduce both,
  with tests.
* **Fix** Report the resolved taxon and warn (or fail) on a miss; report the swept-site count.
* **Output change** Yes (a string and a count). Regenerate
  `fixtures/dms/run_insilico_selection_dms.json` and the DMS e2e cases.

### W16
**Trait co-selection pairs are emitted in score order, not site order**

* **Where** `hyphaeon/phenotype.py:553` — `sub_indices` comes from the score-sorted site list, and
  the pair loop at `:560-605` indexes it directly, so `site_u > site_v` is common (RHO's first
  pair is `(325, 83)`).
* **Why it matters** Cost/clarity, but it surprises every consumer: a table keyed on
  `(site_u, site_v)` cannot be joined with the epistasis pillar's, which is in site order.
* **In the port** `js/src/phenotype.js` replicates it; the e2e fixture pins `(325, 83)`.
* **Fix** Sort `sub_indices` before the pair loop, or normalise each pair.
* **Output change** Yes — pair order and the `(u, v)` orientation. Regenerate
  `fixtures/e2e/phenotype_*.json`.

### W17
**The artifact scan reports nothing when every site is significant**

* **Where** `hyphaeon/filter.py:52-100` (`scan_hypergeometric_patches`) — the local
  hypergeometric p is 1 for every window when the significant fraction is 1, so no patch passes
  `p_local <= 0.01`.
* **What happens** On a strongly selected (or badly misaligned) gene where nearly every site
  crosses α, the screen finds no candidate patches at all.
* **Why it matters** User-visible: the filter reports "no artifacts" most confidently exactly when
  something is wrong with the alignment.
* **In the port** `js/src/filter.js` (`scanHypergeometricPatches`), pinned by
  `fixtures/filter/scan_hypergeometric_patches.json`.
* **Fix** Detect the degenerate case and say so (a warning, not a silent empty list).
* **Output change** Conditional.

### W18
**Three parser defects: PHYLIP continuations, quoted NEXUS labels, `TREE x = [&R] (…)`**

* **Where** `hyphaeon/dataset.py:78` and `:83` (PHYLIP: an alphanumeric line of ≤ 35 characters is
  taken as a taxon name, and a `name seq` line whose sequence part is ≤ 10 characters is not a
  record start); `:144` and `:156` (NEXUS: a quoted matrix label containing a space is split on
  whitespace, so `'sp one'` becomes taxon `sp` with sequence `ONE'…`); `:177` (the tree regex
  requires `(` immediately after `=`, so the standard rooting annotation in
  `TREE x = [&R] (…);` makes the line unmatched — and it does not start with `(`, so step 2 misses
  it too, and the function returns `None`).
* **What happens** Multi-line PHYLIP parses into garbage sequences under invented taxon names; a
  quoted NEXUS label silently loses its tail into the sequence; a perfectly ordinary NEXUS tree
  block is reported as "no tree found".
* **Why it matters** User-visible: two of the three formats the docstring advertises mis-parse
  common files, and the third failure is reported as a missing tree ([S6](#s6)).
* **In the port** `js/src/preprocess/parse.js:172` and `js/src/preprocess/tree.js:344`, pinned by
  `fixtures/dataset/parse_alignment_sequences.json` and
  `fixtures/dataset/extract_tree_from_string_or_file.json`.
* **Fix** PHYLIP: track the expected sequence length from the header instead of guessing from line
  shape. NEXUS: strip quotes before splitting. Tree: allow `[…]` between `=` and `(` (the comment
  stripper already exists two lines later).
* **Output change** Conditional — only for the inputs that mis-parse today; the fixture cases that
  pin the current behaviour must be regenerated.

### W19
**`forward()`'s micro-batch branch does not slice the per-batch tree tensors**

* **Where** `hyphaeon/model.py:489-513` — the automatic micro-batching slices `msa_codons` and
  `msa_aas` (`:499-500`) but passes `dist_matrix=dist_matrix` and `mds_coords=mds_coords` whole
  (`:501-502`). `forward_cached` (`:355-375`) has the same shape but passes an already-shared
  `tree_cache`, so it is unaffected.
* **What happens** Nothing on the CLI path: `dataset.py` hands over `[1, N, N]` and `[1, N, 4]`
  tensors that broadcast across any batch. A caller who supplies per-site tree tensors (a batch
  dimension > 1) gets every micro-batch scored against the whole stack — silently, if broadcasting
  happens to succeed.
* **Why it matters** Latent: the export path and the CLI both avoid it, but it is a trap for the
  next driver.
* **In the port** Not applicable (the ONNX graph takes one distance matrix per call).
* **Fix** Slice both tensors alongside the token tensors when their first dimension is not 1.
* **Output change** No.

---

## C. Silent fallbacks

### S1
**A `--weights` / `HYPHAEON_WEIGHTS` path that does not exist is silently ignored**

* **Where** `hyphaeon/cli.py:44` (`DEFAULT_WEIGHTS_ENV` reads the variable without checking it)
  and `hyphaeon/weights.py:94-99` — an explicit path is used **only if it exists** (or resolves
  relative to the package root); otherwise control falls through to `:102-131`, which picks the
  variant, looks in the HuggingFace cache and downloads.
* **What happens** A typo in `HYPHAEON_WEIGHTS`, or a checkpoint that has moved, scores the
  alignment with a *different* model and prints the path it actually used in one line among many.
* **Why it matters** User-visible and dangerous in exactly the setting this matters most —
  benchmark runs, fine-tuned checkpoints, CI. It is also the opposite of what every runtime
  surface does: the app verifies the sha256 of the graph it loads against `models/manifest.json`
  and refuses to score on a mismatch.
* **In the port** Not applicable (the library loads nothing); the app's `runtime/src/manifest.js`
  is the contrasting policy.
* **Fix** Raise when an explicitly supplied path does not exist. Fall back only when nothing was
  asked for.
* **Output change** No, for correct usage.

### S2
**The BUSTED head is loaded incomplete and unseeded, so its numbers are random on every run**

* **Where** `hyphaeon/cli.py:375-378` — `BustedMultiTaskHead(...)` then
  `busted_head.load_state_dict(busted_dict, strict=False)`; `hyphaeon/model.py:657` defines the
  head; `hyphaeon/cli.py:371` does the same for the backbone. `model.safetensors` is missing 11 of
  the head's parameters (`fixtures/manifest.json` `busted_head_missing_keys`):
  `in_proj.{weight,bias}`, `head_prop.{weight,bias}`, `head_syn_var.{0,2}.{weight,bias}`,
  `coral_{logp,lrt,omega3}.theta_steps`. Nothing seeds torch before the head is constructed.
* **What happens** Those parameters keep their random initialisation, freshly drawn per process.
  Measured on Smc6: `selection_probability` 0.995 in one run and 0.601 in the next. The fields
  affected are `predicted_gene_lrt`, `selection_probability`, `synonymous_rate_variation`,
  `rate_distributions.omega_3`, `rate_distributions.proportion_*`, and — through
  `is_significant = p_acat < 0.05 or pred_prob_pos > 0.50` (`cli.py:505`) — the
  `positive_selection_detected` **verdict**.
* **Why it matters** User-visible and the most serious item in this list: `hyphaeon busted` prints
  a gene-level selection call that is partly a random draw, with no indication that it is. It is
  also unfixable downstream: every surface either reproduces the arithmetic on random inputs or
  omits the fields.
* **In the port** `js/src/omnibus.js:287` (`bustedHeadFields`) reproduces the arithmetic only, and
  says so; `models/busted_head.onnx` is **one seeded draw** (`export.py`,
  `BUSTED_HEAD_INIT_SEED = 0`); the e2e fixtures null those fields; the app marks them
  `deterministic_upstream: false` and does not surface them as results.
* **Fix** Publish a checkpoint that contains the 11 parameters (the real fix), and until then make
  `cmd_busted` refuse — or emit `null` with a warning — rather than print numbers. `strict=True`
  on the head would have caught it at load.
* **Output change** Yes, entirely, for the neural fields: `busted_head.onnx` must be re-exported,
  `models/manifest.json`'s `busted_head_onnx_sha256` updated, `fixtures/e2e/busted_*.json`
  regenerated with real values, and the app's pinned hash bumped.

### S3
**`load_model` defaults to `strict=False`**

* **Where** `hyphaeon/inference.py:117` and `:134`.
* **What happens** A checkpoint whose keys do not match the architecture loads anyway, with
  randomly initialised layers for whatever is missing; the CLI does the same at `cli.py:371`.
* **Why it matters** User-visible in the same way as [S2](#s2), for the backbone: a mismatched
  `--weights` or `--model-variant` produces confident output from a partly random model.
* **In the port** Not applicable.
* **Fix** Default to `strict=True`; keep `strict=False` as an explicit opt-in, and log the missing
  and unexpected keys either way.
* **Output change** No, when the checkpoint matches.

### S4
**Branch lengths come from whichever HyPhy is on `PATH`, or from constants that are not distances**

* **Where** `hyphaeon/dataset.py:656-667` — if `shutil.which("hyphy")` finds a binary, the tree's
  branch lengths are fitted by shelling out (`estimate_tree_branch_lengths_hyphy`, `:224-287`);
  if not, `enforce_nonzero_branch_lengths` (`:668`, `:289-300`) gives every missing branch `1e-3`
  and every zero branch `1e-4` and the run proceeds.
* **What happens** The same command on two machines produces different numbers, and on a machine
  without HyPhy produces a distance matrix built from two constants — a nearly star-shaped tree —
  with no indication in the output. Measured: HyPhy 2.5.98 (WebAssembly) against 2.5.65 (native)
  on camelid moved the patristic matrix by max |Δ| 3.7e-4 and per-site LRTs by up to 2e-2
  relative, enough to change 44 of 96 sites' agreement class.
* **Why it matters** User-visible: reproducibility of every result on a topology-only tree depends
  on an undeclared external dependency and its version. Two of the five bundled examples
  (camelid, HIV1_RT) are exactly this shape.
* **In the port** Not reproduced, deliberately: PLAN.md D22 sends a tree without usable branch
  lengths down the reference's own `--use-tn93` path instead
  (`js/src/preprocess/assemble.js`), which both sides compute bit-identically. That closed the
  last parity gap in the app (camelid and HIV1_RT: 0/96 and 0/335 sites outside the class, from
  44/96 and 149/335).
* **Fix** Record the estimator and its version in the result, and refuse rather than substitute
  `1e-3`: point the user at `--use-tn93`, which is a real distance estimate and needs no external
  binary. Making `--use-tn93` the default for a tree without branch lengths would align the
  reference with every runtime surface.
* **Output change** Yes for topology-only trees. Regenerate the camelid and HIV1_RT tree-path
  fixtures and example outputs.

### S5
**The Lanczos MDS path falls back to dense on any exception**

* **Where** `hyphaeon/dataset.py:404-427` — for `n > 500`, `scipy.sparse.linalg.eigsh(..., k=4,
  which='LA', maxiter=300)` inside a `try` whose `except Exception: pass` (`:427`) drops through to
  the dense `eigh`.
* **What happens** Non-convergence, a scipy import failure or an operator error all produce the
  dense answer with no message. The two paths do not agree exactly (measured on RHO, 655 taxa:
  max |Δ| 6.4e-7 per MDS column after canonicalisation), so which one ran is part of the result.
* **Why it matters** User-visible at the 1e-7 level and unrecorded: two runs on the same data can
  differ for a reason nothing reports.
* **In the port** `js/src/preprocess/mds.js:87` always runs dense (exact at these sizes) and the
  header records the divergence; `MDS_SIGN.md` measures it.
* **Fix** Log which solver produced the coordinates, and let a genuine failure raise.
* **Output change** No.

### S6
**A malformed tree is reported as a missing tree**

* **Where** `hyphaeon/dataset.py:184`, `:197`, `:209` — each parse attempt in
  `extract_tree_from_string_or_file` swallows every exception, and the function returns `None`.
  The caller (`:642`, `:646-651`) then says "Could not parse … from specified path" or "No tree
  specified …, and no embedded tree found".
* **What happens** A Newick file with one unbalanced parenthesis is indistinguishable from no file
  at all; the user is advised to supply a tree they did supply.
* **Why it matters** User-visible: an actionable error becomes a confusing one. Combined with
  [W18](#w18)'s `[&R]` rejection, valid NEXUS trees land here too.
* **In the port** `js/src/preprocess/tree.js:344` reproduces which strings fail (the fallthrough
  order is behaviour), but the app reports `TREE_UNPARSEABLE` separately from a missing tree.
* **Fix** Keep the last exception and include it in the raised message.
* **Output change** No.

### S7
**TN93 distances come from the compiled binary or the Python package, depending on `PATH`**

* **Where** `hyphaeon/dataset.py:505-537` — `shutil.which("tn93")`, then
  `tn93 -t 1.0 -l 1 -q -o distances.csv subset.fa`; a failure prints a warning (`:535`) and falls
  back to the package (`:540-547`). Nothing records which ran. The binary's `-t 1.0` threshold
  means pairs at or above 1.0 are **not written** and stay 0.0, to be imputed by [W2](#w2)'s rules.
* **What happens** Two installations of the same version produce numbers by two different code
  paths. Measured today (binary v1.0.15 against package 1.2.2, through
  `compute_tn93_distance_matrix`): **identical**, max |Δ| = 0.0 on bat_oas1 and HIV1_RT, because
  the binary writes the same six significant digits and every pair it drops is imputed back to the
  1.0 the package returns. Nothing pins that, and a future binary release could diverge silently.
* **Why it matters** User-visible if it ever diverges, and invisible when it does. The port
  mirrors the *package*, so a divergence would appear as a parity failure with no obvious cause.
* **In the port** `js/src/preprocess/tn93.js` implements the package's algorithm (there is no
  subprocess in a library); `scripts/gen_fixtures.py` hides the binary from `PATH` so the fixtures
  pin the package path on any machine.
* **Fix** Record the implementation and version in the result. A CI job comparing the two on the
  bundled examples would turn a future divergence into a failing build (the `parity` workflow
  already reports which path it took).
* **Output change** No today.

### S8
**Permulation failures are swallowed whole**

* **Where** `hyphaeon/phenotype.py:430-445` — the entire permulation block is inside
  `try: … except Exception: null_rhos = None; gene_p_perm = None`.
* **What happens** A non-positive-definite covariance (`LinAlgError` from the Cholesky at `:323`),
  a regex error from [W10](#w10), or an out-of-memory allocation all silently downgrade the run to
  parametric p-values. The only hint is `permulations_count`, which is set to 0.
* **Why it matters** User-visible: a user who asked for 10,000 permulations gets parametric
  p-values and no explanation, in a field where the permulation p is the headline number.
* **In the port** `js/src/phenotype.js` mirrors the swallow but the app's `runPhenotype` records a
  reason (`skipped.reason`) and the UI shows it.
* **Fix** Catch the specific exceptions, record the reason in `phenotype_meta`, and let the rest
  propagate.
* **Output change** No (a new field only).

---

## D. Dead code and duplicated work

### D1
**`max_overlap` is declared, threaded through the CLI, and never read**

* **Where** `hyphaeon/epistasis.py:312` (the parameter, documented as sector overlap suppression),
  `:641` and `:685` (passed through `run_epistatic_analysis`), `hyphaeon/cli.py` `--max-overlap`.
  Nothing in `:306-445` reads it: there is no Jaccard suppression in the reference.
* **Why it matters** A documented CLI flag does nothing, silently. Users tuning it see no change
  and conclude the method is insensitive.
* **In the port** `js/src/sectors.js:428` — `void options.maxOverlap;` with the citation, and a
  test that says so.
* **Fix** Implement the suppression, or delete the parameter and the flag.
* **Output change** Yes if implemented (sector membership changes) — regenerate the epistasis
  fixtures.

### D2
**`branch_names`, `shared_branches` and `hyper_p` are an older formulation's names**

* **Where** `hyphaeon/epistasis.py:124` (`branch_names` is never read — only `attributions.shape`
  defines N), `:205` (`shared_branches` duplicates `shared_taxa`), `:207` (`hyper_p` duplicates
  `p_val`). There is no branch projection and no hypergeometric test in `:121-222`; "branches"
  means taxa, and the p-value is Student's t on a cosine.
* **Why it matters** Clarity, but it misleads every reader of the output schema — including the
  app's UI, which had to be told that "shared branches" is a taxon count.
* **In the port** `js/src/epistasis.js:550-670` emits both duplicates for byte parity, with the
  explanation in the header.
* **Fix** Rename in the schema (a breaking change, so do it with the `schema_version` PLAN.md §7
  item 5 asks for) or document them as aliases.
* **Output change** Keys only.

### D3
**`--background` is accepted and never read**

* **Where** `hyphaeon/phenotype.py:125` (`resolve_phenotype_vector`), `:354` and `:395`
  (`run_phenotype_association` accepts it and passes it on), `hyphaeon/cli.py:1075` (`-bg`,
  `--background`), `cli.py:640` (passed in). Nothing reads it: anything not matched as foreground
  is background.
* **Why it matters** User-visible: a user who carefully names control species gets a silently
  different analysis from the one they asked for.
* **In the port** `js/src/phenotype.js` accepts and ignores it, with a test.
* **Fix** Implement it (restrict the background to the named taxa and exclude the rest) or remove
  the flag.
* **Output change** Yes if implemented.

### D4
**The `d == "-"` TN93 guard is unreachable**

* **Where** `hyphaeon/dataset.py:547`. The `"-"` sentinel belongs to the package's
  `tn93_distance` wrapper (with its 500-nucleotide minimum-overlap test), which the reference
  never calls; `calculate_distance` returns a float or raises. `d is None` and `np.isnan(d)` are
  unreachable for the same reason, and negatives are already clamped inside the package.
* **Why it matters** Clarity, and actively misleading: the line looks like saturation handling,
  while the real saturation path raises ([C1](#c1)).
* **In the port** Replicated anyway (`js/src/preprocess/tn93.js`), with the note.
* **Fix** Replace it with the `try/except` of [C1](#c1).
* **Output change** No.

### D5
**`[-k_foreground:]` with k = 0 would select every taxon**

* **Where** `hyphaeon/phenotype.py:336` — `np.argsort(Z[:, p])[-k_foreground:]`; in Python
  `[-0:]` is `[0:]`, the whole column, so every permulation row would become all-ones.
* **Why it matters** Latent: unreachable today because the binary test requires exactly two unique
  values from {0, 1}, which forces k ≥ 1 (an all-zero trait has one unique value and takes the
  rank-matching branch). It would fire the moment the binary test changed.
* **In the port** `js/src/permulations.js:336` keeps the slice semantics and pins the all-zero
  case's actual output.
* **Fix** `if k_foreground: …` or an explicit guard.
* **Output change** No.

### D6
**The permulation projection is computed twice**

* **Where** `hyphaeon/phenotype.py:437` and `:440` — `leaf_attr @ Y_perms.T` appears in both the
  site-level and the gene-level expression.
* **Why it matters** Cost: that product is the largest allocation in the pillar (L × P float64 —
  about 28 MB at RHO's L = 349 and P = 10,000, twice).
* **In the port** Computed once (`js/src/phenotype.js`); numpy returns the same values both times.
* **Fix** Hoist it.
* **Output change** No.

### D7
**The DMS wild-type residue and candidate list are computed twice per site**

* **Where** `hyphaeon/epistasis.py:531-536` (to build the tensors) and `:550-555` (to read the
  results) — identical code on identical inputs.
* **Why it matters** Cost only, but it reads like a difference and sends every reader looking for
  one.
* **In the port** One call site (`js/src/dms.js`), noted in the header.
* **Fix** Compute once and reuse.
* **Output change** No.

### D8
**`plasticity` and `selection_dms_plasticity` are the same list object**

* **Where** `hyphaeon/epistasis.py:766-767` — one list bound to two keys, so the JSON writer
  serialises it twice and a mutation through one key shows in the other.
* **Why it matters** Cost/clarity: it doubles the largest section of a DMS result file for no
  information.
* **In the port** `js/src/dms.js` emits both, as the writers must.
* **Fix** Emit one, or make the second an explicit alias documented in the schema.
* **Output change** Keys only. Regenerate the DMS fixtures if removed.

### D9
**`ppv_confusion` and `fpr_confusion` are the same computation**

* **Where** `hyphaeon/evaluation.py:313-314` — both call
  `_confusion(inclusive_truth, inclusive_predictions)`; the report then prints two identical
  matrices (`ppv_confusion_matrix`, `fpr_confusion_matrix`).
* **Why it matters** Clarity: the duplication implies the FPR is measured against a different
  threshold or truth set, and it is not. The numbers are correct — FP/(FP+TN) of that matrix *is*
  the classifier's FPR — but the schema promises something it does not deliver.
* **In the port** `js/src/evaluate.js:805` emits both, byte-equal to the Python.
* **Fix** Emit one matrix, or make the second genuinely different (e.g. FPR at a second α) if that
  was the intent.
* **Output change** Keys only.

---

## E. CLI / library inconsistencies

### I1
**`hyphaeon meme --filter` is a second, divergent copy of `filter.py`**

* **Where** `hyphaeon/cli.py:133-237` against `hyphaeon/filter.py:102-399`. Seven differences,
  all measured and pinned:
  1. consensus codons — `cli.py:152` drops only codons containing `-` or `N`; `filter.py:208` also
     drops `?` and anything not of length 3;
  2. the artifact rule reads `min_patch_consec` (`cli.py:155`, `:186`) and hard-codes
     `min_oci = 0.25`, α = 0.05, `min_k = 3`, `max_span = 35`, where `filter.py:253` takes them as
     parameters; only `--filter-p-thresh` is configurable on `meme`;
  3. the p-values scanned are float32 (`cli.py:123-124`), so `<= 0.05` compares against
     `float32(0.05)`;
  4. the cleaned FASTA holds only the matched `taxa` in taxa order, where `filter.py:293-294`
     writes every parsed sequence in file order;
  5. the cleaned re-load crashes with an embedded tree ([C2](#c2));
  6. the cleaned re-score reuses the original tree cache (`cli.py:214`);
  7. `cmd_meme` computes no CCT or mean-LRT metrics, so its `artifacts_masked` records carry
     fewer fields than `hyphaeon filter`'s audit.
* **Why it matters** User-visible: `hyphaeon meme --filter` and `hyphaeon filter` can mask
  different patches on the same alignment, and only one of them can be right.
* **In the port** `js/src/filter.js:247` — `{cliVariant: true}` selects the `cmd_meme` copy; both
  are implemented because both are reachable from the product.
* **Fix** Delete the copy: have `cmd_meme --filter` call `run_alignment_filter` and surface its
  records. That fixes [C2](#c2) with it.
* **Output change** Yes. Regenerate `fixtures/e2e/meme_*_filter*.json`.

### I2
**The epistasis CLI and the function disagree on `min_sim`**

* **Where** `hyphaeon/cli.py:745` and `:1099` pass and default `--min-sim 0.30`;
  `hyphaeon/epistasis.py:126` and `:635` default to `0.35`. `min_cesi` is never passed by the CLI,
  so the function default 2.0 always applies, and `min_shared` likewise.
* **Why it matters** User-visible: the same alignment gives a different network through the CLI
  and through `run_epistatic_analysis`, which is what any downstream consumer calls. Neither value
  is documented as the recommended one.
* **In the port** `js/src/epistasis.js` keeps the function default 0.35; the app's runner and MCP
  pass 0.30 explicitly to reproduce the CLI, and say so.
* **Fix** One default, in one place.
* **Output change** No if the chosen default matches what the fixtures were generated with
  (`--min-sim 0.30` via the CLI); otherwise regenerate the epistasis e2e fixtures.

### I3
**There are no library entry points, and errors are `sys.exit`**

* **Where** `hyphaeon/cli.py:70-347` (`cmd_meme`) and `:349-589` (`cmd_busted`) implement the
  whole pipeline inline — device selection, batching, statistics, writing — with no
  `hyphaeon.api.run_meme(...)` equivalent; `epistasis`, `dms` and `phenotype` do have module-level
  drivers. Failures call `sys.exit(1)` after printing (`cli.py:76-78`, `:101-103`), and progress
  goes to stdout.
* **Why it matters** User-visible for any embedder: the two most-used analyses can only be reached
  by shelling out and parsing printed text, exit codes carry no taxonomy, and stdout cannot be
  used as a protocol channel. This is PLAN.md §7 items 3–4, and it is why the app's MCP bridge had
  to classify errors from printed `[!]` markers before the port replaced it.
* **In the port** Not applicable — the library is functions only, which is the shape being asked
  for here.
* **Fix** `hyphaeon.api.run_meme(...)` / `run_busted(...)` returning the result dict, typed
  exceptions, and a `progress(phase, done, total, message)` callback (with `--progress-json` on the
  CLI). The CLI becomes a thin wrapper.
* **Output change** No.

### I4
**`--seed`: what landed, and what is still not reproducible**

* **Where** `hyphaeon/cli.py:55` (`SEED_DEFAULT = 42`), `:1085` (`phenotype --seed`), `:1108`
  (`epistasis --seed`) — added during Phase 2a. Before that, `rng_seed=42`
  (`epistasis.py:230`, `:319`, `:645`) and the permulation seed were hard-wired defaults the CLI
  never surfaced, so a Python run's randomness could not be recorded or reproduced from its own
  output, and `scripts/parity.py` **skipped epistasis entirely** for that reason.
* **What is still open**
  1. `hyphaeon/phenotype.py:318` seeds numpy's **legacy global state** (`np.random.seed`), which is
     process-wide: any other library that draws from `np.random` between the seed and `:326`'s
     `randn` changes the result. `epistasis.py` uses `default_rng` correctly.
  2. Nothing records the seed in the output. A results file cannot say which draw produced it
     (PLAN.md §3.5 asks for `seed` in provenance).
  3. `hyphaeon dms` has no `--seed` — correct today, since it draws nothing, but worth stating.
* **Why it matters** User-visible: Monte Carlo p-values that cannot be reproduced from the file
  they appear in.
* **In the port** `js/src/numeric/prng.js` (xoshiro256\*\*, seed 42 by default, recorded in
  `models/manifest.json` `prng`); every randomised function takes an explicit seed and the app
  records it in provenance.
* **Fix** Replace `np.random.seed` with a `default_rng(seed)` instance, and write the seed into the
  result.
* **Output change** Yes for the permulation draws if the generator changes — but permulation
  parity is statistical by construction (PLAN.md §5.4), so no fixture pins the values; only the
  recorded per-taxon frequencies in `fixtures/phenotype/generate_permulations.json` would move.

### I5
**`--mds-sign canonical` landed; the committed outputs did not follow**

* **Where** `hyphaeon/dataset.py:358-390` (`resolve_mds_sign`,
  `canonicalize_eigenvector_signs`), `:391` (`compute_mds_coordinates`), `hyphaeon/cli.py:48`,
  `:59-63` and the `--mds-sign` flag on every alignment-loading subcommand. Documented in
  `MDS_SIGN.md`.
* **What happens** The default changed from "whatever the eigensolver returned" to canonical, so
  every reference output changed: up to 0.59 absolute LRT on the bundled examples (median 3.5e-4
  to 1.7e-2 relative), Spearman ≥ 0.998, and 2 of 145 variable RHO sites crossing p = 0.05.
  `fixtures/` were regenerated. **Not** regenerated: `examples/*_results.csv`, `examples/*.json`,
  `examples/expected_results/`, and the `model_eval/` outputs — they no longer reproduce.
* **Why it matters** User-visible: the shipped example outputs disagree with what the shipped code
  produces, which is the first thing a new user checks.
* **In the port** `js/src/preprocess/mds.js` implements the identical rule and defaults to it, so
  the two implementations now agree per column with no sign allowance.
* **Fix** Regenerate `examples/*` and `expected_results/` under the default with a test that
  compares them (PLAN.md §7 item 5), and record `mds_sign` in result provenance. Worth
  considering: evaluate — or fine-tune — on canonical coordinates, since `training_data.py` now
  produces them automatically, which would remove the model's sensitivity to an arbitrary
  convention rather than merely pinning it.
* **Output change** Already happened; the remaining work is regenerating the committed outputs.

### I6
**Four copies of the amino-acid tables**

* **Where** `REV_AA_MAP` / `standard_aas` are defined at the top of `hyphaeon/epistasis.py`
  (`:41`), `hyphaeon/phenotype.py`, `hyphaeon/disease.py` and derived again in
  `hyphaeon/dataset.py:36-39`.
* **Why it matters** Clarity and drift risk: the vocabulary is the model contract ([W1](#w1)), and
  it exists four times.
* **In the port** Defined once (`js/src/preprocess/tokenizer.js` and `modelContract.js`); the
  barrel refuses to re-export a second binding under the same name.
* **Fix** One definition in `dataset.py`, imported.
* **Output change** No.

---

## F. Checked and benign

Examined during the port, found harmless, listed so nobody re-opens them:

* `hyphaeon/stats.py:58` — `benjamini_hochberg` sorts with `np.argsort` (quicksort, unstable).
  Tied p-values still receive identical q-values, so the result is order-independent.
* `hyphaeon/evaluation.py:209`, `:213` — `max(0.0, raw_lrt)` returns the *first* argument on a tie,
  so a MEME LRT of `-0.0` becomes `0.0` and `lrt_was_clamped` stays False. Cosmetic.
* `hyphaeon/stats.py:16-48` — the MEME and Self–Liang mixtures, checked against scipy to 1e-12; no
  defect. The float32 casts in `cmd_meme` (`cli.py:123-124`) are deliberate and are reproduced.
* `hyphaeon/epistasis.py:170` — the Student-t test is applied to a cosine of non-negative vectors,
  so t ≥ 0 in practice; scipy's full `t.sf` is kept because the function is generic.
* `hyphaeon/dataset.py:289-300` — `enforce_nonzero_branch_lengths` raises a *negative* branch to
  `1e-4` as well (`< min_len`). Defensible, and it is what the port does.
* Sector ordering depends on CPython's `set` iteration order, because networkx's subgraph view
  iterates a `set` of node ids. Reproduced exactly in `js/src/numeric/graph.js`; not a defect, but
  it means sector ids are an implementation detail and should not be treated as stable labels.

---

## Suggested order

1. **[S2](#s2)** — the incomplete, unseeded BUSTED head. It is the only item that makes a printed
   result a random draw, and it needs a checkpoint, so it has the longest lead time.
2. **[C1](#c1), [C2](#c2)** — the two crashes a normal user reaches (`--use-tn93` on a messy
   alignment; `--filter` on an embedded tree). [C2](#c2) is fixed for free by [I1](#i1).
3. **[W3](#w3), [W6](#w6), [S4](#s4)** — the three silent "the model was given something else"
   defects: zero distance rows, the stride truncation, and the branch-length substitution.
4. **[W2](#w2), [W11](#w11), [W12](#w12), [W13](#w13), [W14](#w14), [W15](#w15)** — wrong values
   in shipped fields. Batch them into one regeneration of `fixtures/`.
5. **[W7](#w7)–[W10](#w10), [D3](#d3), [S8](#s8)** — the phenotype pillar's trait resolution,
   which is where a user's own data meets the code.
6. **[I3](#i3), [I4](#i4), [I5](#i5)** — the API, the seeds and the stale committed outputs
   (PLAN.md §7's remaining items).
7. **[D1](#d1), [D2](#d2), [D4](#d4)–[D9](#d9), [I2](#i2), [I6](#i6), [W1](#w1)** — dead
   parameters, duplicated work, and publishing the vocabulary as a contract.

## Regenerating, when a fix changes outputs

The order in PLAN.md §5.3 rule 3 is not negotiable, because the fixtures are the only thing that
keeps the two implementations honest:

```bash
# 1. fix the Python, with a test in tests/
python -m pytest tests/ -q

# 2. regenerate the fixtures from the fixed reference (~75 s; needs the tn93 extra)
HYPHAEON_WEIGHTS=$PWD/model.safetensors HF_HUB_OFFLINE=1 python scripts/gen_fixtures.py

# 3. the JS suite now FAILS where the behaviour changed -- that is the point.
#    Port the fix, then:
cd js && npm test && npm run typecheck && node scripts/fixture-coverage.mjs   # 48/48

# 4. end to end, reference side
HYPHAEON_WEIGHTS=$PWD/model.safetensors HF_HUB_OFFLINE=1 \
  python scripts/parity.py --examples all --analyses meme,busted --surfaces python

# 5. if models/ changed (a re-export): hyphaeon export-onnx --variant all, then
python scripts/verify_onnx.py    # also updates models/manifest.json hashes
```

Then tell the app repository: it pins an exact library version, so a behaviour change is a version
bump there, and any change under `models/` is a new hash its sessions verify against.
