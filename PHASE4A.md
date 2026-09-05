# Phase 4a — CI that can fail, one upstream list, and a release that can be run

Phase 4 of the plan of record (`../hyphaeon-app/PLAN.md` §8: "harden and share") needs three
things from this repository before anything can be published: continuous integration that fails
when the port loses a fixture or the reference stops reproducing itself, one document the ML team
can work from instead of five phase reports, and a release path with no secrets in it. This page
is the map of what changed, what each check printed, and the checklist for cutting `v1.0.0`.

`PHASE0.md`, `PHASE1A.md`, `PHASE2A.md` and `PHASE3A.md` are the previous maps; `MDS_SIGN.md` and
`PARITY.md` are the two documents this one leans on.

## What changed

| Path | What changed |
|---|---|
| `.github/workflows/js.yml` | Third gate added: `node scripts/fixture-coverage.mjs`, which fails when any `fixtures/**/*.json` is read by no test (48 of 48 today). `npm test` and `npm run typecheck` were already there; the npm cache is keyed on `js/package-lock.json`, the repository's only lockfile. `timeout-minutes: 20`. The header now says what each of the three gates catches that the others cannot: a wrong value, a wrong shape, a fixture nothing replays. |
| `.github/workflows/parity.yml` | Rewritten as **reference-only**, and rewritten again against the rebuilt `scripts/parity.py` that landed in this phase (tree-free surfaces, `dms` and `phenotype` comparators, the PLAN §5.4 graph class, skipped BUSTED head fields). Three steps: sites and gene on every example (tree **and** `--use-tn93`), epistasis + DMS, phenotype. The header documents the contract with `veg/hyphaeon-app` — who writes `parity/<surface>/`, who compares, and that the comparison runs `scripts/parity.py` from this repository at the engine tag the app pins — and deliberately does **not** restate the tolerance classes, so it cannot drift from `PARITY.md` and the script's docstring. `js/**` is no longer a trigger: nothing in the job runs JavaScript. |
| `.github/workflows/release.yml` | The three versions (tag, `pyproject.toml`, `js/package.json`) were already checked; added a hashlib check that every ONNX file named by `models/manifest.json` matches its recorded sha256, so a re-export without a regenerated manifest cannot ship (every surface verifies that hash before scoring and would refuse *after* the release). `MDS_SIGN.md` is attached to the GitHub release beside `models/manifest.json`. The npm job publishes over **OIDC trusted publishing** like the PyPI job, so `secrets.NPM_TOKEN` is gone and there is now no secret of any kind in the file; the npm job also runs the same three gates as `js.yml` before publishing. |
| `UPSTREAM.md` (new) | One consolidated, deduplicated checklist of every reference (Python) bug and quirk the port found: 47 items in five groups (crashes, wrong results, silent fallbacks, dead code, CLI/library inconsistencies) plus a "checked and benign" list, each with `file:line` at `61d681a`, what happens, who sees it, where the port reproduces it, a suggested fix, and whether fixing it moves the fixtures. Mined from `PHASE0.md`, `PHASE1A.md`, `PHASE2A.md`, `PHASE3A.md`, `MDS_SIGN.md`, `fixtures/manifest.json` `known_quirks` and the `WHY THIS FILE EXISTS` headers under `js/src/`, then **re-checked against the source** — every line number in it was verified at `61d681a`, and several from the phase docs had moved. |
| `README.md` | The "JavaScript library" section gained what CI checks (the three gates, the reference-only parity run and where the other surfaces come from), the one-line local command, and links to `MDS_SIGN.md` and `UPSTREAM.md` with the port rule stated: fix the Python, regenerate the fixtures, then fix the port. |

Nothing else was touched: no method, no fixture, no test, no `scripts/`.

## What UPSTREAM.md says, in one paragraph

47 items. The one that matters most is **S2**: `model.safetensors` is missing 11 of
`BustedMultiTaskHead`'s parameters, `cmd_busted` loads it with `strict=False` and never seeds
torch, so `selection_probability`, `predicted_gene_lrt`, `synonymous_rate_variation`, the ω-class
fields and the `positive_selection_detected` verdict are a fresh random draw on every run
(measured on Smc6: 0.995, then 0.601). Everything downstream inherits it — `busted_head.onnx` is
one seeded draw, the fixtures null those fields, the app marks them non-deterministic — and
nothing can be at parity with an unseeded reference. After that: two crashes a normal user reaches
(`--use-tn93` on a saturated alignment, `--filter` on an embedded tree), three silent
"the model was given something else" defects (zero distance rows from tier-2/3 name matching, the
stride pre-selection that truncates before Faith's PD measures diversity, branch lengths that come
from whatever HyPhy is on `PATH` or from the constant 1e-3), and a long tail of wrong values, dead
parameters (`max_overlap`, `background`, `branch_names`) and CLI/library disagreements. §"Suggested
order" ranks them; §"Regenerating" gives the exact commands for the ones that move fixtures.

## How to run each check, and what it printed

From the repository root, Python 3.11+ with the engine installed editable
(`pip install -e ".[dev,tn93]"`), Node 22, and:

```bash
export HYPHAEON_WEIGHTS=$PWD/model.safetensors HF_HUB_OFFLINE=1
```

| # | Check | Command | Printed |
|---|---|---|---|
| 1 | Workflows are valid YAML | `python -c "import yaml,sys; [yaml.safe_load(open(f)) for f in sys.argv[1:]]" .github/workflows/*.yml` | silent (all five files, including the two this phase did not touch) |
| 2 | Every `run:` block is valid shell | `bash -n` over each `run` extracted from the three workflows | `shell syntax: all run: blocks parse` |
| 3 | JS suite (js.yml gate 1) | `cd js && npm test` | `Test Files 29 passed (29)` / `Tests 1067 passed (1067)` / `Duration 5.92s` |
| 4 | Typecheck (gate 2) | `cd js && npm run typecheck` | `tsc --noEmit -p tsconfig.json` silent, exit 0 |
| 5 | Fixture coverage (gate 3) | `cd js && node scripts/fixture-coverage.mjs` | `48/48 fixture files read by a test`, **exit 0** |
| 6 | Release verify job, locally | the `verify` job's version checks and the manifest hash block, run by hand | `tag / js/package.json 1.0.0`; `tag / pyproject.toml 1.0.0`; `checked 3 artifact hash(es) against models/manifest.json`, exit 0 |
| 7 | parity.yml step 1 (sites and gene) | `python scripts/parity.py --examples all --analyses meme,busted --surfaces python,node-tn93` | `reference runs: 14 (0 failed); self-check violations: 0; … violations: 0` / `PASS`, 1 min 57 s. Ten tree runs plus four `python-tn93` runs (camelid and HIV1_RT, the two examples whose `.nwk` has no branch lengths), each noting `tn93 binary hidden from PATH: the Python tn93 package computes the distances, as the fixtures and the port do`. Seven p/q self-checks, five tree and two tree-free, all `max_abs_diff 0.0` |
| 8 | parity.yml step 2 (epistasis, DMS) | `python scripts/parity.py --examples Smc6,bat_oas1 --analyses epistasis,dms --surfaces python,node-tn93 --tn93-examples Smc6` | `reference runs: 5 (0 failed); … violations: 0` / `PASS`, 32 s: Smc6 and bat_oas1 epistasis at B = 10,000, Smc6 DMS, plus the two tree-free Smc6 runs the `*_tn93` fixtures pin |
| 9 | parity.yml step 3 (phenotype) | `python scripts/parity.py --examples RHO --analyses phenotype --surfaces python,node-tn93` | `reference runs: 1 (0 failed); … violations: 0` / `PASS`, 10 s (RHO, the README marine foreground, `--permulations 0`) |
| 10 | The artifact the job uploads | `ls parity/{report-*.json,python,python-tn93}` | three reports; `python/` 14 JSON + 14 logs; `python-tn93/` 6 JSON + 6 logs |
| 11 | Attribution strings | `grep -rniE "claude\|anthropic\|co-authored\|generated with" .github/workflows UPSTREAM.md PHASE4A.md README.md` | no `claude` / `anthropic` / `co-authored` anywhere; the `generated with` hits are prose ("the fixtures were generated with", "regenerated with real values") |

Total parity wall time on an Apple M4 Pro CPU: **2 min 39 s** for all three steps; the job's
`timeout-minutes: 60` leaves a wide margin for a GitHub runner, where HIV1_RT is the long pole
(17.8 s tree, 25.3 s tree-free, locally).

Two caveats on the numbers above. Check 7's `--surfaces python,node-tn93` asks for a surface this
repository does not produce **on purpose**: `parity.py` runs the tree-free reference only when a
`-tn93` surface is requested, and an absent surface is reported as missing, which is not a
failure. And checks 3–5 ran against a working tree in which another builder was regenerating
`fixtures/` and `js/test/`; the counts (29 files, 1067 tests, 48 fixtures) are that tree's, and the
gates are what matters, not the totals.

## Release checklist

### Once, before the first tag

1. **PyPI trusted publisher** on the `hyphaeon` project: owner `veg`, repository `HyphAeon`,
   workflow `release.yml`, environment `pypi`. No token is stored anywhere.
2. **npm trusted publisher** on `@veg/hyphaeon-js`: GitHub Actions, owner `veg`, repository
   `HyphAeon`, workflow `release.yml`. Provenance is then attached automatically, which is why the
   job passes no `--provenance` flag; the job installs `npm@latest` because Node 22 ships an npm
   too old for OIDC. If the npm job fails with an authentication error, configure the publisher —
   **do not add a token to the workflow.**
3. **`CODEOWNERS`** names `@veg/hyphaeon-maintainers`, a placeholder. GitHub silently ignores a
   CODEOWNERS entry for a team that does not exist; create the team or replace it with usernames.

### Every release

1. **Decide the version** `X.Y.Z` and bump it in **both** files in one commit:
   `pyproject.toml` `[project].version` and `js/package.json` `version`. The `verify` job checks
   both against the tag and refuses on a mismatch — that is the rule that keeps the app's exact
   pin honest (PLAN.md D19).
2. **If a method changed**: fix the Python first, then
   `python scripts/gen_fixtures.py` (≈75 s), then the port, then
   `cd js && npm test && npm run typecheck && node scripts/fixture-coverage.mjs` — the third
   command is what catches a new fixture that nothing replays.
3. **If the graph or the weights changed**: `hyphaeon export-onnx --variant all`, then
   `python scripts/verify_onnx.py` (PyTorch vs onnxruntime on Smc6 and camelid, both variants,
   the BUSTED head, and every manifest hash). `models/manifest.json` is rewritten by the export;
   the release job re-checks the committed hashes so the two cannot disagree.
4. **Regenerate `fixtures/manifest.json` at the release commit.** It records `engine_commit`
   (`61d681a` after this phase's regeneration, matching `HEAD`) — a fixture set that says which
   commit generated it is only useful if that is still true at the tag, so re-run
   `scripts/gen_fixtures.py` after the version bump lands.
5. **Run the reference parity job locally** (checks 7–9 above) and confirm
   `reference runs: N (0 failed)` and `self-check violations: 0`.
6. **Tag the HuggingFace repository.** `models/manifest.json` records
   `variants.*.safetensors_sha256` for weights that live at `datamonkey/hyphaeon`; tag that repo
   with the same `vX.Y.Z` (PLAN.md D2) so the checkpoint a released manifest refers to is
   recoverable by revision rather than by "whatever `main` held that day".
7. **Tag and push**: `git tag vX.Y.Z && git push origin vX.Y.Z`. `release.yml` then runs
   `verify` → `pypi` + `npm` in parallel → `github-release`, which creates the release if the tag
   has none and attaches `models/manifest.json` and `MDS_SIGN.md`.
8. **Confirm**: `pip download hyphaeon==X.Y.Z`, `npm view @veg/hyphaeon-js@X.Y.Z`,
   `gh release view vX.Y.Z` (both assets present).
9. **Tell the app**: `veg/hyphaeon-app` pins the library exactly and links it by
   `file:../../HyphAeon/js` until a tag exists. Bump `runtime/package.json` to
   `@veg/hyphaeon-js@X.Y.Z`, re-run its suites, and check that the ONNX hashes its sessions verify
   against are the ones in the released manifest.

## Known gaps for Phase 4b

1. **`release.yml` has never run end to end.** The repository's tags are `phase-0`…`phase-3a`; no
   `v*` tag exists, so the trusted-publisher configuration of both registries, the `pypi`
   environment and the `gh release upload` step are all untested. The first release is the test —
   cut it from a branch tag first if that is too sharp.
2. **Nothing checks `fixtures/manifest.json` `engine_commit` against the commit that generated
   it.** It was stale (`61d30e3`) for part of this phase until the fixture regeneration reset it to
   `61d681a`; checklist item 4 covers the release, and a `js/test` assertion would stop it drifting
   again without anyone noticing.
3. **No workflow runs `scripts/verify_onnx.py`.** The ONNX graphs are only checked against PyTorch
   by hand at export time; CI verifies the committed hashes but not that the graph still
   reproduces the model. It needs torch plus onnxruntime, so it belongs with `model_eval.yml`'s
   heavier machinery rather than in `js.yml`.
4. **The parity job compares nothing.** By design — the surfaces live in the app repository — but
   it does mean this repository's CI cannot notice that the app has drifted. The app's own CI is
   where that failure has to appear (its workflow was written in this phase, in that repository);
   a scheduled cross-repository run would close the loop if drift ever becomes routine.
5. **`UPSTREAM.md` is a list, not a plan.** Every item is open. The ML team owns them, and the two
   with lead time — the BUSTED checkpoint (S2) and publishing the codon vocabulary as a contract
   (W1) — should start before the rest, because both need a re-export and a fixture regeneration
   after the fix.
6. **`CODEOWNERS`'s placeholder team** (pre-flight item 3) is still a placeholder.
