# Wave signs: the `canonical` convention, what it changes, and why it is the library's default

This is D28 in `PLAN-TEMPORAL.md §9`, and it is `MDS_SIGN.md`'s argument applied to a second class of
object. Read that file first if you have not: the reasoning is identical and is not repeated here.

## The problem in one paragraph

`run_temporal_surveillance` (`hyphaeon/temporal.py`) decomposes the standardised velocity
trajectories of the confirmed sweeps into four collective **wave modes** — the right singular vectors
of `Z_fpca` (`temporal.py:731`) — and projects every codon onto them to get the four
`Wave_k_loading` columns (`:740-741`). A singular vector is defined only up to sign, and every solver
picks its own: LAPACK `gesdd` (what numpy calls), and the JavaScript library's tred2/tql2 on the
time-side Gram, disagree with each other, and neither is stable under a change of BLAS, thread count
or library version. Unlike the MDS coordinates, **no number downstream of a wave depends on its
sign** — `fpca_wave_variance_pct`, `r2_fpca`, `p_perm`, both classification columns and every
per-site metric are sign-free. What depends on it is what a *reader* sees: `_waves.csv` is plotted as
a curve, so a flip turns a burst into a trough; the loading column is read as "this site moves *with*
the wave" versus "*against* it"; and the reference's own figure colours panel D's bars blue for a
positive loading and red for a negative one (`temporal.py:1073-1079`). On the acceptance run
(`hyphaeon temporal -a examples/H1N1_2009_pandemic.fasta -t examples/H1N1_2009_pandemic.nwk -B 100
--time-points 60 --cpu`) **three of the four waves have their largest-magnitude entry negative**, so
`gesdd`'s raw signs already draw wave 1 as a deep trough and colour every site on a coin flip.

`temporal.py` has **no** convention and writes the solver's signs as they come.

## The rule

`waveSign="canonical"`: for each **retained** right singular vector `v_j` (`j = 1 … min(4, rows, T)`),
find the index of its largest-magnitude entry with `np.argmax(np.abs(v))` — the **first** index on an
exact tie, strict `>` on the JavaScript side — and multiply the whole vector by −1 when that entry is
negative. An all-zero vector is left unchanged. The flip is applied **before anything is derived from
the vector**: before `waves` is written and before the loadings are computed, so a wave and its
loading column flip together and the wave-to-loading relative sign stays correct by construction.
Never flip a loading column independently of its wave.

A sign flip is exact in floating point, so canonical waves are bitwise the solver's waves up to sign
and nothing else about the decomposition changes.

`waveSign="lapack"` leaves the solver's signs alone and reproduces pre-convention output. As in
`MDS_SIGN.md`, the name is the reference's; on the JavaScript side it means "whatever `tql2`
returned", which is not LAPACK's.

## Why this rule and not the two obvious alternatives

**Signed area (`Σ_t v_j[t] > 0`) is mathematically degenerate and must not be used.** The rows of `Z`
are mean-centred over time (`temporal.py:668-671`, `:728-730`), so `Z·1_T = 0` exactly, `1_T` lies in
the null space of `Z`, and **every** right singular vector with `σ > 0` is orthogonal to `1_T` and
sums to zero. Measured on the reference's four waves: `Σ_t v_j[t] =` −1.03e-15, +6.11e-16, +6.11e-16,
−1.06e-15. The rule would be pure roundoff. It is recorded here because it is the first rule anyone
reaches for.

**Correlation with the time axis** ("positive means rising") is meaningful but has a much weaker
margin. Measured `|ρ(v_j, t)|` on the four waves: 0.178, 0.824, 0.414, **0.122**. A symmetric single
pulse — the exact shape an episodic sweep produces — has `ρ ≈ 0`, and wave 4 is already there.

**Largest-magnitude margin.** Define `m_j = (max_t v_j[t] − max_t(−v_j[t])) / max_t |v_j[t]| ∈ [−1, 1]`;
`|m_j|` is how far the rule is from a coin flip. Measured: **0.774, 0.561, 0.575, 0.589**. The worst
case beats time-correlation's worst case by 4.6×, on every wave. It is also the rule this codebase
already adopted for the MDS eigenvectors, and a second, different convention for the same class of
object would be a maintenance trap.

## Measured effect

Landing `canonical` changes **exactly eight columns in two files and nothing else**. On the
acceptance run, `wave_1`, `wave_3` and `wave_4` in `_waves.csv` and `Wave_1_loading`,
`Wave_3_loading` and `Wave_4_loading` in `_sites_summary.csv` flip sign (wave 2's pivot is already
positive). `fpca_wave_variance_pct`, every `r2_fpca`, every `p_perm` / `q_perm`, every
classification, every per-site metric and both stage counts are **bit-identical**, because none of
them reads a sign (`js/src/numeric/svd.js` has the invariance table).

That is a far smaller blast radius than `MDS_SIGN.md`'s, which moved every LRT on every example. Said
plainly: **no number changes; three curves are drawn the other way up, and the bars that were red are
blue.**

## Where it is set, and where it is not set yet

| Surface | How | Default |
|---|---|---|
| `@veg/hyphaeon-js` `dominantTimeModes(Z, rows, T, k, { waveSign })` | option | `'canonical'` |
| `@veg/hyphaeon-js` `fpcaShapeGate(..., { waveSign })`, `temporalWaveDecomposition({ waveSign })` | option, passed through | `'canonical'` |

**Not yet on the Python side.** `hyphaeon temporal` has no `--wave-sign` flag, `HYPHAEON_WAVE_SIGN`
is read by nothing, and `run_temporal_surveillance` takes no `wave_sign` keyword. Landing them —
mirroring `--mds-sign` exactly, so the engine has one idiom and not two — is an engine change with a
fixture regeneration behind it, and it is deliberately **not** part of the library port. Until it
lands:

* `fixtures/temporal/thin_svd.json` records **both** `v_raw` (the solver's own signs, which is what
  the reference writes) and `v_canonical`, so a replay can pin either side;
* `js/test/numeric-svd.test.js` compares the canonical vectors with **no sign allowance** against
  `v_canonical`, and separately compares the raw vectors against `v_raw` while **recording the flip
  vector** rather than absorbing it. A silently absorbed flip is exactly what hid a real disagreement
  for two phases in the MDS work;
* `js/test/temporal.test.js` compares wave curves up to sign and loadings by magnitude, and says so.

Measured against the reference's own `_waves.csv`, restricted to its own 18 confirmed sweeps, the
expected flip vector is **(−1, +1, −1, −1)**; once applied, the four wave curves agree to **1.3e-15**
and the singular values come out at σ = 20.6989 / 18.6961 / 12.2629 / 10.0253 against the
reference's recovered 20.699 / 18.696 / 12.263 / 10.025. Once `--wave-sign` lands on both sides, the
comparison becomes column by column at the graph class with no sign-flip allowance, and **a sign
disagreement becomes a defect**.

## What no sign convention can fix

When `σ_j ≈ σ_{j+1}` the two vectors are individually ill-conditioned: by Wedin's theorem the
rotation between the computed and the true vectors is bounded by `≈ ε‖Z‖ / gap`, and at `gap = 0` the
pair is arbitrary within its 2-plane. The ambiguity is a continuous rotation group, not a discrete
±1, so the convention does nothing about it — it would faithfully canonicalise a rotated basis and
produce a stable-looking, meaningless number. This is not only a numerical statement: a 10 % gap also
means a 1 % perturbation of the *input* — one dropped taxon, a different `--time-points`, a different
downsampling — can rotate the pair by a visible angle.

`dominantTimeModes` therefore reports the adjacent relative gaps `g_j = (σ_j − σ_{j+1}) / σ_1` for
`j = 1 … k`, **including the gap between the last retained mode and the first discarded one** (that
one moves the projector, hence `r2_fpca`, hence the stage-two gate, hence the confirmed-sweep set),
and flags a pair as near-degenerate at `g_j < 0.05`. **The constant is set by measurement, not by
taste**: the worst gap among the four retained modes on the acceptance run is 0.097 — twice the
threshold — so the reference's headline example sits clear of the flag and the flag is not
decorative. Below 0.05 the individual axes rotate under perturbations the data cannot distinguish,
while the measured pivot margins above (worst 0.561) would remain perfectly stable, which is exactly
the point: **a stable sign does not imply a stable vector.**

A mode is separately flagged `rankDeficient` when `σ_j / σ_1 < √ε ≈ 1.5e-8`. numpy writes arbitrary
null-space vectors for such modes and `_waves.csv` carries them; the library reports the flag and
leaves the suppression to the application, because that is result semantics.

One consequence of the Gram route, stated rather than smoothed over: a singular value is recovered as
`sqrt(λ)`, so a numerically zero mode comes out at `~√ε·σ_1` rather than numpy's `~ε·σ_1`. Measured on
`fixtures/temporal/thin_svd.json`'s rank-2 case, numpy reports `σ_3 = 1.2e-15` and this route reports
`9.7e-8`. Both mean "zero"; neither is a property of the data; and the rank flag is what a consumer
should read instead of either number.
