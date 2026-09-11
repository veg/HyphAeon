# Reconciliation, phase 1 of the temporal and dating plan

Branch `reconcile/phase-5a`, head **`e49fe4f`**, working tree clean.

This document is the record of what was merged, what had to be decided by hand, what the
verification actually measured, and what is still open. It summarises work already done; nothing
here is an instruction to re-run a merge.

---

## 1. What the reconciled line contains

Three divergent lines were brought onto one branch:

- **`feat/js-port`** — the ONNX export (`hyphaeon export-onnx`), the JavaScript library under
  `js/` (`@veg/hyphaeon-js`), `fixtures/`, the parity harness (`scripts/parity.py`), and the MDS
  sign convention of decision **D20** (`resolve_mds_sign`, `canonicalize_eigenvector_signs`,
  `--mds-sign` on every subparser that takes it).
- **`origin/main`** — the temporal pillar (`temporal.py`), `splits.py`, the v0.1.x release
  metadata, the move of `train.py` into `training/`, `_warn_internal_stops`, the `_IS_DEV` gate on
  the BUSTED subcommand, and the mainline test suite.
- **`origin/feature/dating-mrca-module`** — the dating suite (`dating.py`, `autoclock.py`,
  `geo.py`, `r0.py`, `sieve.py`, `sketch.py`, `alignment.py`), the dated examples, the BEAST XML
  reader, and the rewritten TN93 distance matrix.

Four commits make up the reconciliation:

| Commit | What it is |
|---|---|
| `58d4b16` | Merge the dating/MRCA line into the JavaScript-port line. One conflict (`cli.py`). |
| `261e7da` | Merge the mainline in. Three conflicts (`dataset.py`, `cli.py`, `README.md`). |
| `fda67f0` | Packaging fix: import scikit-learn and matplotlib where `autoclock` uses them, not at module scope, so `import hyphaeon` works on the declared base dependencies alone. |
| `e49fe4f` | Follow the reconciled `dataset.py` in the JavaScript library — two behavioural mirrors, a new internal-stop notice, regenerated dataset fixtures, and a full line-citation remap of the library's provenance headers. |

The CLI now carries **17 subcommands** with no duplicate names: `autoclock, busted, dating,
disease, dms, epistasis, evaluate, export-onnx, filter, geo, list-models, meme, phenotype, r0,
sieve, splits, temporal`. `export-onnx` — the application's only hard CLI requirement — is intact
along with its dispatch arm.

---

## 2. Conflicts and auto-merge corrections

### 2.1 Conflicts resolved by hand

**`hyphaeon/cli.py`, dating merge, the subparser block.** `HEAD` appended `export-onnx`; the
dating branch appended `splits`, `dating`, `geo`, `r0`, `sieve`, `autoclock`. Both append at the
same anchor with no semantic overlap, so both sides were kept.

**`hyphaeon/cli.py`, mainline merge, hunk A — the constants block (~line 45).** `HEAD` had
`DEFAULT_WEIGHTS_ENV` with the repo-local `model.safetensors` fallback plus
`MDS_SIGN_DEFAULT` / `MDS_SIGN_HELP` / `SEED_DEFAULT` / `SEED_HELP` / `_apply_mds_sign`;
`origin/main` had two lines narrowing `DEFAULT_WEIGHTS_ENV` to the environment variable alone.
**Took `HEAD` whole.** Main's narrowing is redundant — `weights.py:93-99` already resolves a bare
filename against the package root — while taking main's side would have deleted all of D20 and the
seed constants from the CLI.

**`hyphaeon/cli.py`, mainline merge, hunk B — the BUSTED subparser (~line 1770).** `HEAD`
registered it unconditionally with `--mds-sign`; `origin/main` had the same block indented under
`if _IS_DEV:` ("dev-only, not in preprint"), reworded, without `--mds-sign`. **Took main's gate and
re-inserted `--mds-sign` inside it at main's indent.** The gate is main's deliberate release
decision and costs nothing in practice — `_IS_DEV` is true whenever `<repo>/.git` exists, so BUSTED
stays available in every checkout, editable install and CI run — but dropping `--mds-sign` would
have left the BUSTED parity surface unable to name its sign convention. Verified:
`grep -c mds-sign hyphaeon/cli.py` is **8**.

**`hyphaeon/dataset.py`, mainline merge, the insertion point after `get_aa_token` (lines
61-253).** `HEAD` carried the dating branch's `_parse_numeric_or_calendar_date` and
`parse_beast_xml`; `origin/main` carried `_warn_internal_stops`. Pure both-added; **both kept**.
Each has live call sites in the merged file (`parse_alignment_sequences` calls `parse_beast_xml`;
`load_alignment_and_tree` calls `_warn_internal_stops` at `:982` and `:1080`), so either `--ours`
or `--theirs` would have produced a file that imports cleanly and raises `NameError` on the first
alignment parsed.

**`README.md`, hunk 1 — the example table.** Both sides carried an HIV-1 RT row with different
taxon counts: the dating branch said 52, the mainline 476. `grep -c '^>' examples/HIV1_RT.fasta`
returns **476**, so the mainline row is the accurate one. Kept its five rows and appended only the
dating branch's three genuinely new rows (`korber_env_gp160`, `H5N1_HA_geo`,
`H1N1_2009_pandemic`, all three files present), dropping the stale row.

**`README.md`, hunk 2 — the subcommand table.** Additive; both sides kept.

### 2.2 Auto-merge corrections — where git was silently wrong, or silently right

**`compute_tn93_distance_matrix` (`hyphaeon/dataset.py`).** Git resolved this with **no conflict
marker even though both branches rewrote it**, because the mainline's delta is largely a textual
subset of the dating branch's. The outcome is right — the dating branch's version is the
mainline's successor, not a competitor — but nothing checked it, so it was verified explicitly:
the merged file has `np.full((n, n), -1.0)` at `:732`, `missing = (dist_mat < 0.0)` at `:817`,
`dtype={"ID1": str, "ID2": str}` on both `read_csv` calls, and **zero** occurrences of the
mainline's `off_zero`. A hybrid carrying the mainline's `<= 0.0` mask against the `-1.0` sentinel
would have imputed **every** off-diagonal entry to maximum distance, silently and unrecoverably.
Section 3 records what that would have cost, measured.

**`hyphaeon/dataset.py`, exactness of the textual merge.** Proved rather than assumed: the delta
from the dating branch's file to the first merge result is byte-equal to the delta from the merge
base to `feat/js-port`, so the MDS sign block and `mds_sign` threading came through untouched.
`hyphaeon/phenotype.py` was verified the same way. After the mainline merge every tripwire lands
on its predicted value: 1144 lines, 19 top-level defs, 12 `mds_sign` occurrences,
`_warn_internal_stops` 3×, `parse_beast_xml` 3×, no `off_zero`.

**The deleted `model.safetensors`.** `origin/main` commit `4a3f339` deletes the tracked
`model.safetensors` — it has always been covered by `.gitignore`'s `*.safetensors`, so the mainline
was cleaning up a file tracked in violation of its own ignore rule. The merge applied that
deletion silently, and it broke `scripts/gen_fixtures.py`, which could then not hash the weights
and wrote `model_safetensors_sha256: null` into the manifest. No survey flagged this. **The
mainline's decision was kept** — the file stays untracked — and a working copy was restored to the
working tree from `feat/js-port`'s blob; its sha256 is `0278dff0…`, byte-for-byte the one the
existing fixtures were generated with. `git status` is clean because the file is ignored. The
other nine files that commit deleted (BUSTED and orphaned example outputs) were checked: none is
referenced from `fixtures/`, `js/`, `scripts/` or `hyphaeon/`.

**`js/src/preprocess/assemble.js` header.** It quoted *two* commits' line numbering side by side
("LINE NUMBERS IN THIS BLOCK ARE `61d30e3`'s … the list above is `267f5cf`'s"), so a blind remap
would apply one file's diff map to the other file's numbers. The whole header was rewritten
against the reconciled `dataset.py` with a single numbering and a note saying so; every step's
range was re-derived by locating the statement in the merged file.

**`js/src/preprocess/tn93.js:13`.** "PLAN.md §5.1 cites `dataset.py:665-766, 544-580`; those were
the line numbers at the commit the plan was written against" is a deliberately **historical**
citation that the mechanical remap rewrote as if it were current. The historical values
(443-521 / 544-580) were restored and the paragraph rephrased so historical and current numbers
are unambiguous.

**`README.md` lead-in.** Both sides' sentence above the example table survived the textual merge,
leaving two consecutive introductions. Folded into one.

**Order of work.** The citation remap was run on the pristine library **first** and the
behavioural edits applied afterwards. A first attempt in the opposite order double-mapped the
citations just written in reconciled coordinates; `js/` was reverted and redone in the safe order
rather than hand-patched.

### 2.3 The library synced to the reconciled reference (`e49fe4f`)

- **`js/src/preprocess/tree.js`** — `hasNonzeroBranchLengths` gains `minPosRatio = 0.05`, a ≥50%
  numeric-coverage precondition, and a denominator of *numeric* branches rather than all branches,
  mirroring `dataset.py:430-444`. See section 4: this is the gate that decides the tree-free path.
- **`js/src/preprocess/tn93.js`** — `tn93DistanceMatrix` rewritten to the `-1.0` sentinel form of
  `dataset.py:715-821`: fill with `-1.0`, zero the diagonal, write measured values (a measured
  `0.0` now survives as `0.0`), impute only entries still negative, `max_d` read once from the
  filled matrix. The `1e-4` floor and the live-max re-read are gone. This is the reference
  **fixing a bug this file had flagged and replicated** — identical sequences were being given the
  largest distance in the matrix — so the port follows the fix.
- `TN93_FALLBACK_MAX` `0.1 → 1.0` (`dataset.py:818`); `TN93_MIN_POSITIVE_DISTANCE` kept exported
  with a deprecation note rather than deleted, so the export surface `index.test.js` pins does not
  churn; `TN93_MISSING_SENTINEL` (`-1.0`) and `TN93_REPORTING_THRESHOLD` (`100.0`) added
  module-private for the same reason.
- The `pairwiseDistances` provider hook: `null`/`undefined` for a pair now means "not written"
  (to be imputed) rather than the saturation answer `1.0`, and the hook receives the threshold as
  a third argument. At the reference's new cutoff of `100.0` those are different numbers.
  `tn93SaturatedPairs` keeps its exact-equality definition with the caveat documented, since the
  package path this library implements never leaves an entry unwritten.
- **`js/src/preprocess/assemble.js`** — new `notices.internalStops {worstTaxon, worstCount}`,
  mirroring `_warn_internal_stops` (`dataset.py:235-249`, called at `:982` and `:1080`).
  `assemble.js`'s stated contract is that everything the reference prints becomes a notice field,
  and this is a genuinely different statistic from the existing `inFrameStops` total: per taxon,
  over the first L−1 codons only, worst taxon reported, ties to the first taxon as the reference's
  strict `>` does.
- **Fixtures** — `fixtures/dataset/tn93_distance_matrix.json`, `tn93_distance.json`,
  `parse_alignment_sequences.json` and `fixtures/manifest.json` regenerated with
  `scripts/gen_fixtures.py --only dataset` against the reconciled reference. The two synthetic
  cases that pinned the old imputation were renamed to what they now test
  (`distinct_sequences_at_zero_stay_zero`, `identical_strings_stay_zero`); the HIV1_RT reduced
  matrix no longer has a saturated pair, so the `tn93_distance` case that existed only to document
  that pair is gone; `parse_alignment_sequences` gained three cases because the dating branch
  brought three new example alignments.
- **`scripts/gen_fixtures.py`** — the matrix signature note, the synthetic case names and notes,
  and three of the manifest's `known_quirks` rewritten to describe the rule actually in force.
  Quirk 1 now says the binary-vs-package equality measurement was taken at threshold 1.0 under the
  old imputation and has **not** been re-taken; quirk 2 records the old rule as fixed upstream;
  quirk 3 records that `ValueError`/`OverflowError` is now caught (`ZeroDivisionError` still is
  not).
- **`js/test/tn93.test.js`** — the imputation-rules assertion rewritten to pin the new rule, plus a
  sweep asserting no fixture matrix entry reaches the retired `1e-4` floor.
- **`js/test/data/preprocess/edge_cases.json`** — `topology_only_half_zero` and
  `one_positive_of_six` sit squarely in the flip band the new predicate opens (the surveys had
  concluded no case did). Their `has_nonzero_branch_lengths` / `needs_branch_lengths` values were
  recomputed by running the reconciled Python function over the same Newick strings through
  Biopython, not by hand.
- **Line-citation remap across 26 files of `js/src` and `js/test`** — every `dataset.py`,
  `model.py`, `cli.py`, `phenotype.py` and `epistasis.py` citation rewritten through a per-module
  `difflib` map from the pre-merge file to the reconciled one (**369 citations**). All five of
  those modules moved; `stats.py`, `filter.py`, `attribution.py`, `evaluation.py`, `io.py` and
  `inference.py` are byte-identical, so their numbers hold at either ref. Anchors spot-checked in
  both directions. Provenance refs changed from `267f5cf` / `61d30e3` to `reconcile/phase-5a` in
  19 files.

---

## 3. Verification

### 3.1 Parity — the gate that matters, and the one number that moved

**`runtime/scripts/parity-node.mjs` (node, node-tn93) then `scripts/parity.py --surfaces
python,node,node-tn93 --analyses meme`, on Smc6 (tree), camelid (tree-free) and HIV1_RT
(tree-free). Verdict: PASS, exit 0.** Reference runs 5 (0 failed); self-check violations 0;
comparisons 3 (3 pass, 0 fail); missing 0; redirected 2; incomparable 0; violations 0;
informational excursions 0. ~80 s wall.

**One number moved, and it is on the Python reference, not the port.** HIV1_RT tree-free,
`hyphaeon_lrt`, max |ΔLRT| **9.53674e-06** against the pre-merge reference (worst at site 219,
`17.295930862426758 → 17.295921325683594`), **0/335 sites beyond the graph class**, called sites
at p ≤ 0.05 unchanged at 35. That is the merge's TN93 imputation rewrite in
`compute_tn93_distance_matrix` — the `-1.0` sentinel form, a measured `0.0` surviving as `0.0`,
`TN93_FALLBACK_MAX` `0.1 → 1.0`, the `1e-4` floor gone. It lands exactly where the survey
predicted: the reconciled reference is **bit-identical** to the survey's dating-rule probe, which
had forecast `9.53674e-06` and `0/335`. The JavaScript library followed the same fix, so
`node-tn93` HIV1_RT still passes at 0.2298 of its bound.

Everything else is bit-identical to the pre-merge baseline on every per-site numeric field, on
both the reference and the node side: Smc6 (node, tree) max |Δ| `7.62939453125e-06` (0.1979 of
bound), camelid (node-tn93) `1.0967254638671875e-05` (0.2249) — exactly the baseline's recorded
figures. `p_value` and `q_value` comparisons are exactly 0.0 across all three examples
(n = 1097 + 96 + 335). All exact-class checks (taxa count, codon count, site order,
`is_invariable`) pass with 0 violations.

The two "skipped" comparisons are `redirected` pairs, not failures: camelid and HIV1_RT are
tree-free by D22, so the tree-based `node` surface has no file for them by construction and
`parity.py` redirects to `node-tn93`. The harness working as designed.

**The silent auto-merge on `compute_tn93_distance_matrix` is now measured, not just argued.**
Against the survey's `origin/main`-rule probe, the hybrid resolution git could have produced would
have moved HIV1_RT's reference by max |ΔLRT| **0.326771** (site 151:
`7.354361057281494 → 7.02760124206543`), put **151/335** sites beyond the graph class, and flipped
the called set 35 → 36. The merged tree does not do that: the reconciled reference is
bit-identical to the dating-rule probe, and `node-tn93` measured against the `origin/main` probe
reproduces that 0.326771 / 151-site / 36-vs-35 signature exactly — confirming the port is on the
taken rule and not the rejected one.

**D20 is exercised end to end by this gate and produced no signal.** Every `hyphaeon_lrt`
comparison sits at 0.20–0.23 of its bound with 0 violations, the same band as the baseline. The
standing instruction to treat any parity regression as a D20 signal is not triggered.

Two limits on this run, stated plainly. **RHO was not run.** Zero-distance pairs after duplicate
pruning are Smc6 0, bat_oas1 0, camelid 0, HIV1_RT 66, RHO 5, so HIV1_RT and RHO are the only two
examples with exposure to the imputation rewrite; RHO meme + phenotype is the remaining surface
worth confirming and would be expected to move by the same order. And **this gate cannot see the
tree-gate change at all** — see section 4.

*Gate infrastructure note, worth fixing before CI relies on this environment:* the scratchpad venv
has no `pyvenv.cfg` and its `bin/python` is a bare symlink to the pyenv 3.14.0 interpreter, so its
`site-packages` is not on `sys.path`. `import hyphaeon` only resolved because the CLI ran with
cwd = the engine checkout, and `import tn93` did not resolve at all; the first `parity.py` run
exited 2 with "The 'tn93' tool or python package is required to skip the tree" and
"cannot import hyphaeon.stats". Worked around for this run with a scratch directory holding only
a symlink to the venv's `tn93` package plus `PYTHONPATH=<engine root>:<that dir>` — deliberately
minimal, because adding the whole venv `site-packages` would have shadowed numpy 2.3.3 with 2.5.3
and changed the numerical stack under the reference mid-comparison.

### 3.2 Python suite

`pytest` at `e49fe4f`: **294 passed, 3 failed, 11 skipped**, 308 collected, 63.4 s. Baseline
re-established in a throwaway worktree at `feat/js-port` (`5848336`) with the same interpreter:
**252 passed, 2 skipped, 0 failed**, 254 collected, 31.7 s.

**Nothing regressed.** Every test that passed at baseline still passes. All three failures are
environmental — missing optional dependencies, or `hyphaeon` not pip-installed in the throwaway
venv — not merge defects:

1. `test_dating.py::test_autoclock_with_beast_xml` — `ModuleNotFoundError: sklearn` at
   `autoclock.py:477`. scikit-learn is an optional extra (`model_eval`, `all`). The file is new to
   this branch; it cannot have regressed. This is the exact consequence flagged for the lazy-import
   packaging fix (`fda67f0`): autoclock paths now fail at first use rather than at import.
2. `test_dating.py::test_hierarchical_autoclock_deconvolution` — same `sklearn` error here, but
   **masking a real upstream bug**. `autoclock.py:1809` calls `ax_d.boxplot(..., labels=c_lids)`;
   matplotlib removed `labels` in 3.11 (renamed `tick_labels`) and this machine has 3.11.0,
   verified directly: `TypeError: Axes.boxplot() got an unexpected keyword argument 'labels'`.
   With scikit-learn installed this test would still fail. Dating-line code the reconciliation did
   not touch; it bites any user on matplotlib ≥ 3.11.
3. `test_training.py::test_training_cli_runs_two_epochs_from_checkpoint` — environmental, but a
   fragility the baseline did not have. The test shells out to `python training/train.py`; the
   subprocess dies with `ModuleNotFoundError: hyphaeon`. Proven environmental: it passes in 2.5 s
   with `PYTHONPATH` set to the repo root. Cause: `origin/main` commit `2feafd4` moved `train.py`
   from the repo root into `training/`, so the subprocess's `sys.path[0]` is now `<repo>/training`
   and the script can no longer find the package without an install. Latent in `origin/main` too,
   not a reconciliation decision.

The two runs reconcile exactly with the merge report's own figures (1 failed / 304 passed /
3 skipped): a machine with scikit-learn, the `tn93` package and an editable `hyphaeon` install
converts the 8 `tn93` skips and 2 of the 3 failures into passes, leaving precisely the one
matplotlib failure. Nothing in either run is unexplained.

**Coverage loss worth a follow-up.** Twelve tests that exist at `feat/js-port` are gone:
`tests/test_batch_size.py` (8, deleted outright) and `TestDeterminAdaptiveShim` (4) in
`tests/test_gpu_mem_guard.py`. Both came from `origin/main` `de1702b`, whose message says it
removed the deprecated `determine_adaptive_batch_size` from `cli.py` along with the tests. But the
reconciled tree **still defines `determine_adaptive_batch_size` at `hyphaeon/cli.py:68`** — the
`cli.py` hunk-A resolution took `HEAD` whole for that region — so the branch carries a function
main deliberately deleted, with no caller anywhere and no test covering it. Either drop the
function or restore its tests; today it is untested dead code.

**Side effect on a tracked file.** Running the suite from the repo root dirties
`h5n1_geo_example.png`: `tests/test_geo.py:127` calls the geo CLI's worked example, and
`cli.py:1359-1362` defaults its outputs to relative paths, so they land in the cwd and overwrite
the tracked PNG. Restored with `git checkout --`. The dating/geo line should write these into
`tmp_path`.

### 3.3 JavaScript library suite

`cd js && npx vitest run` at `e49fe4f`: **29 files / 1069 passed, 0 failed, 0 skipped**, 8.5 s —
against a pre-merge baseline of 1067. The +2 is fully accounted for by regenerated fixtures, not
by new source behaviour going untested: `parse_alignment_sequences.json` 13 → 16 cases (three new
example alignments from the dating branch) and `tn93_distance.json` 24 → 23 (the case documenting
HIV1_RT's saturated pair is gone, since the reduced matrix no longer has one). 13 + (−1) + 3 = +2.

The three test-side edits that could have hidden a real divergence were inspected, and each pins
the reconciled Python rather than papering over it: `tn93.test.js`'s imputation assertion now
asserts a measured `0.0` survives as `0.0` where it previously asserted the `1e-4` floor and the
`max(1.0, max_d)` bug; `edge_cases.json` flips two cases to the widened predicate;
`assemble.test.js` changed only citation line numbers. Statistical replays printed inside class
and did not flake (sectors `p_perm` 0.0002 vs ref 0.0004, 0.3316 vs 0.3282, 0.0004 vs 0.0014).

### 3.4 Application workspaces

`npm run test --workspaces --if-present` against the linked reconciled library
(`node_modules/@veg/hyphaeon-js -> ../../../HyphAeon/js`): **runtime 23 files / 306, web 13 / 93,
mcp 10 / 113, server 5 / 57 — 569 total, 0 failed, 0 skipped, exit 0**, 80 s.

One failure on the first of two runs: `server/test/oauth.test.js` "rejects a token bound to another
resource (RFC 8707)" timed out at 120 s. **A proven load flake, not a regression.** No assertion
failed; the file's other 11 tests passed in 347 ms while the file reported 120361 ms, and it runs
concurrently with `jobs.test.js` scoring bat_oas1 through onnxruntime-node. Re-run alone: 12/12 in
251 ms. Re-run of the workspace: 57/57. Re-run of the full four-workspace command: green, exit 0.
The path touches only `server/src/oauth.js` and the `/mcp` bearer check — it parses no alignment,
loads no graph and never reaches `@veg/hyphaeon-js`.

runtime is 306 against the 297 in the Phase 4 release note because that number is stale: app
commit `9907b0b` (WebAssembly TN93) landed afterwards and took the suite to 306. So the reconciled
library moved **zero** application tests in either direction.

---

## 4. What changed behaviourally for the application

**The branch-length predicate moved, and it decides when an upload goes tree-free.** This is the
first-order change in this reconciliation and it needs the owner's sign-off.

`hasNonzeroBranchLengths` (`js/src/preprocess/tree.js`, mirroring `dataset.py:430-444`) is now
strictly more permissive: it requires at least 50% of branches to carry a number at all, and at
least `minPosRatio = 0.05` of *those numeric* branches to be strictly positive — a denominator of
numeric branches, not all branches. It never rejects a tree the old predicate accepted.

This is the gate `runtime/src/pipeline.js:258` reads to choose the tree-free path, so leaving the
library on the old rule would have made the browser and the reconciled CLI disagree
**categorically, not numerically**, about whether a given upload is tree-free at all. That is why
it was synced.

The consequence: an upload whose tree carries numbers on at least half its branches but is
positive on between 5% and 50% of them — a dense outbreak tree of near-identical isolates — now
flips from the tree-free TN93 path to using its own tree, whose zero branches
`enforce_nonzero_branch_lengths` then raises to the `1e-4` floor, handing the model a near-star
patristic matrix. **Nobody has measured whether that is better.**

**Every bundled example is silent on it** — Smc6 0.946 positive, bat_oas1 1.000, camelid and
HIV1_RT carry no branch lengths at all — so no fixture, gallery record, e2e assertion or parity
file moves. Which is exactly why it could ship unnoticed: the green parity gate, the green library
suite and the green application suites are all, by construction, no evidence at all about this
change. Worth saying to the ML team alongside D20: the widened input class is near-degenerate, and
eigenvector signs in a near-degenerate eigenspace are the case the canonicalisation cannot
stabilise.

Other behavioural changes, smaller:

- **TN93 imputation.** A measured distance of `0.0` now survives as `0.0` instead of being floored
  to `1e-4`; identical sequences are no longer given the largest distance in the matrix. Unwritten
  entries are imputed at `max(1.0, max_d)` with `TN93_FALLBACK_MAX` at 1.0 rather than 0.1.
  Measured effect on the application's surfaces: the 9.5e-06 movement on HIV1_RT in §3.1, inside
  the graph class, no called site changed.
- **A new notice, `notices.internalStops {worstTaxon, worstCount}`**, available on the notices
  object. No diagnostics code was emitted to surface it (see §5).
- **`import hyphaeon` works on base dependencies alone** again (`fda67f0`). The cost is that
  autoclock paths now fail at first use rather than at import when scikit-learn is absent.

**Application-repository follow-up, out of scope for this branch:** `runtime/src/tn93-wasm.js`
still invokes the compiled tool with `TN93_ARGV = ['-t','1.0',...]` where the reference now uses
`-t 100.0` with a retry at 1.0, so the WASM engine still drops pairs the reference would report.
It never returns `null`/`undefined` (it throws on an unwritten pair), so the provider-hook
semantics change is inert for it today. The application suites are green with that divergence in
place.

---

## 5. Deferred, and open

### Deliberately deferred

**D20, the MDS sign convention — untouched, as instructed.** Verified byte-identical to
`feat/js-port`: `resolve_mds_sign`, `canonicalize_eigenvector_signs` and `compute_mds_coordinates`
are unchanged function for function; all 12 `mds_sign` occurrences survive in `dataset.py`;
`cli.py` keeps `MDS_SIGN_DEFAULT` / `MDS_SIGN_HELP` / `_apply_mds_sign` and `--mds-sign` on all
seven subparsers that had it, BUSTED included; `js/src/preprocess/mds.js` differs from
`feat/js-port` only in citation and provenance strings. There was never any merge pressure on it —
neither `origin/main` nor the dating branch touches `compute_mds_coordinates`, and the mainline did
not "drop" the block, it simply never had it. **Status: preserved, not resolved. It remains the ML
team's to revisit**, now with the parity evidence of §3.1 (no signal at 0.20–0.23 of bound) and the
near-degenerate-eigenspace concern of §4 in front of them.

**Version left at the mechanical merge result, `0.1.1`**, `requires-python >=3.8`. The CLI survey
recommended 0.2.0 on the grounds that the reconciliation adds five pillars plus the ONNX export to
a released package. That is a release decision with no correct mechanical answer, so it was not
taken; 0.1.1 is the mainline's deliberate value, and `1.0.0` — the untouched placeholder on the
other two lines — correctly lost. See §6.

**Not ported to JavaScript, on purpose:**

- `parse_beast_xml`, `_parse_numeric_or_calendar_date` and the two BEAST step-0 branches. Both
  guards require a leading `<` in the content or an `.xml` suffix, and no FASTA, NEXUS, PHYLIP or
  Newick input can produce that, so they are inert for every format the application accepts.
  Porting them is a product decision about BEAST XML upload, not a reconciliation.
- `compute_tn93_cross_distance_matrix`. Verified called only from `dating.py` and `autoclock.py`,
  never from `load_alignment_and_tree` and never from any path the MEME / BUSTED / epistasis / DMS
  / phenotype pillars take. Porting it would be porting a new pillar.
- `temporal.py`, `splits.py`, `dating.py`, `autoclock.py`, `geo.py`, `r0.py`, `sieve.py`,
  `sketch.py`, `alignment.py` — new pillars, later work by definition.
- No `diagnostics.js` code for the new `internalStops` notice. Called optional for this phase; the
  number is on the notices object for whoever wants to surface it.

### Open

1. **D20 needs the ML team.** Preserved, not resolved. Parity gives it a clean bill at this
   change, but the widened tree gate (§4) enlarges the input class toward near-degeneracy, which
   is precisely where the canonicalisation is weakest. These two should be looked at together.
2. **The widened branch-length predicate needs the owner's sign-off** on its own terms (§4). No
   suite in this repository or the application can see it.
3. **The application cannot pin this line until it is tagged.** `runtime/package.json` links the
   library by `file:../../HyphAeon/js`, and the application's CI `ENGINE_REF` is still `phase-3a`,
   whose `parity.py` refuses the `node-tn93` surface. The reference must be bumped past it, and
   that bump wants a tag, not a branch name.
4. **`HYPHAEON_WEIGHTS`.** The application's `CLAUDE.md` documents
   `HYPHAEON_WEIGHTS=../HyphAeon/model.safetensors`, which is now an **untracked** file. A fresh
   clone of the engine will not have it and must fetch the weights from Hugging Face. The
   documentation needs updating on the application side.
5. **`runtime/src/tn93-wasm.js` threshold divergence** (§4, last paragraph).
6. **Upstream defects recorded, not fixed**, all on dating-branch code this reconciliation did not
   touch:
   - `autoclock.py:1809` `boxplot(labels=…)` breaks on matplotlib ≥ 3.11 (§3.2).
   - In `compute_tn93_distance_matrix`, `'*' -> '-'` sanitisation (the LANL MASE gap character) is
     applied **only** to the FASTA written for the compiled binary; the Python-package fallback
     passes `seq_dict[t]` raw. The two paths therefore disagree on MASE-style alignments, which
     also breaks the "binary and package agree exactly" property the fixtures previously asserted.
     The library has no binary path so has nothing to replicate; manifest quirk 1 was rewritten to
     say the equality measurement has not been re-taken rather than leave a claim no longer known
     to hold.
   - `SyntaxWarning: invalid escape sequence` under Python 3.12+ in `dating.py` (257, 2302, 2304,
     2380, 2381) and `autoclock.py` (1739, 1790, 1806). Cosmetic.
   - `tests/test_sieve.py:7` hard-codes an absolute path (`/Users/sergei/Projects/...`) into
     `sys.path` and passes only because `conftest.py` already inserted the real repo root.
   - `tests/test_geo.py` writes into the cwd and dirties a tracked PNG (§3.2).
7. **`determine_adaptive_batch_size` is untested dead code** (§3.2). Drop it or restore its tests.
8. **No test asserts `import hyphaeon` succeeds on base dependencies alone.** One should be added;
   that is the property `fda67f0` restored, and nothing guards it. (Also noted:
   `SpectralClustering` was imported at module scope and referenced nowhere, so it is not
   re-imported anywhere.)
9. **RHO parity not run** (§3.1) — the one remaining example with exposure to the TN93 imputation
   rewrite.
10. **The gate environment needs repair** before CI leans on it (§3.1, closing note).

---

## 6. Recommendation

**Yes — tag it. `phase-5a`.**

The case for tagging: the merge is complete and the tree is clean; every conflict was resolved on
stated reasoning rather than a side preference, and the one hunk git resolved silently has been
checked explicitly and then *measured* against the rejected alternative. Parity passes at exit 0
with 0 violations, and the single number that moved — 9.5e-06 on HIV1_RT's reference — is a
predicted consequence of an upstream bug fix, inside the graph class, changing no called site, and
bit-identical to the probe that forecast it. The library suite is green at 1069, the four
application workspaces are green at 569 with zero movement in either direction, and the Python
suite has no regression: every failure is environmental or a pre-existing upstream defect, each one
traced to its cause.

The case against waiting: the two things that genuinely remain open — D20 and the widened
branch-length predicate — are **not merge questions and will not be settled by more merging**.
They need the ML team and the product owner respectively, and both of those conversations are
easier against a fixed, citable ref than against a moving branch. Meanwhile the application is
blocked on exactly this: `ENGINE_REF` is still `phase-3a`, whose `parity.py` refuses the
`node-tn93` surface, and the application cannot pin a branch.

`phase-5a` is the right name: it matches the branch, it follows `phase-3a` in the sequence the
application's CI already reads, and it makes no claim about the temporal and dating work being
finished — which it is not; this is phase 1 of that plan.

Two conditions on the tag, neither blocking:

- **Tag the line, not the release.** Leave the version at `0.1.1`. Bumping to `0.2.0` is a
  separate, deliberate decision about a published package and should be taken by whoever owns the
  release, not inherited from a reconciliation.
- **Carry §4 and §5 forward with the tag.** The widened predicate must not reach users as a silent
  change. It is silent on every bundled example — which is the argument for writing it down
  loudly, not for assuming it is harmless.
