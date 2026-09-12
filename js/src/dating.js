/**
 * WHY THIS FILE EXISTS
 *
 * The model-free half of the ChronAeon dating pillar: least-squares dating on tree-free distances,
 * ported function by function from `hyphaeon/dating.py` (branch feat/date-parsers, d998cab). Nothing
 * here loads the transformer — the acceptance path
 * `hyphaeon dating -a <fa> --root-taxon <t> --no-tree --method ols` never does — so this file is
 * arithmetic over times and divergences and nothing else. Every line reference below is that file.
 *
 *   compute_tree_free_divergences        624-698   -> computeTreeFreeDivergences
 *   compute_fieller_mrca_interval        839-887   -> computeFiellerMrcaInterval
 *   run_ols_dating                      1170-1296  -> runOlsDating   (+ computeDeltaMrcaInterval,
 *                                                     lifted out of 1231-1237 so both intervals are
 *                                                     testable side by side)
 *   compute_rcs_basis                   1774-1806  -> computeRcsBasis
 *   run_restricted_spline_clock_dating  1809-1968  -> runRestrictedSplineClockDating
 *   the per-taxon block of run_mrca_dating 3000-3042 -> clockFittedAndPredicted, residualScale
 *   the ensembler's algebra              2895-2921 -> precisionWeightedEnsemble
 *
 * RECORD KEYS ARE THE REFERENCE'S, snake_case, in the reference's own insertion order, as
 * phenotype.js already does: the JSON export at dating.py:3168 is
 * `{k: v for k, v in ols_res.items() if k not in ['residuals','fitted','times']}`, so that key set
 * in that order IS the download contract and a port that renamed anything would break it silently.
 *
 * ================================ WHAT IS DELIBERATELY NOT HERE ================================
 *
 *   - The POWER-LAW clock (1564-1772). Under `clock_model="auto"` — the product's only setting —
 *     the reference NEVER fits it: 2841's `if clock_model in ["auto","spline"]` … 2857's
 *     `elif clock_model == "power"`. Porting it would add a box-constrained L-BFGS-B optimiser on a
 *     7x4 restart grid to serve a branch nothing reaches (PLAN-TEMPORAL D33).
 *   - LOOCV / jackknife (970-1168), opt-in upstream behind `--loocv`, O(N²) as written.
 *   - Everything the transformer feeds: PGLS (1299-1474), REML Pagel λ (1476-1548), the latent
 *     convex-hull root (701-838) and the neural covariance kernel (78-118). PHASE 4 PORTED THEM,
 *     into src/datingModel.js — they needed a taxon-by-taxon attention matrix and per-taxon
 *     embeddings averaged inside the graph, i.e. a new ONNX contract, a new manifest hash and new
 *     fixtures (D29), and those exist now. They are a separate file because the split is real: this
 *     one is reachable with no model at all, which is what makes the date-review page cost no model
 *     byte, and datingModel.js is not.
 *   - The Poisson (890-925) and wild-residual (927-968) intervals, and the spline bootstrap's
 *     resampling. Each needs a bit-compatible mirror of numpy's PCG64 (`rng.poisson`, `rng.choice`)
 *     to reproduce. Skipping all three means THIS PILLAR SHIPS WITH NO RNG AND NO SEED CONTRACT AT
 *     ALL, which is the largest single reduction in parity risk available here — and, for the spline
 *     bootstrap, it is what the reference itself produces (B1 below).
 *   - Tree re-rooting (494-621): D34 declines it rather than approximating it.
 *   - The root POLICY, the coverage/holdout rule, the outlier flag, the download writers and every
 *     sentence a reader sees. Those know about a reader's data, not about arithmetic, and belong to
 *     the application (PLAN.md §5.5). `root_description` is the exception that proves it: it is the
 *     reference's own provenance token, returned verbatim, not prose this library wrote.
 *
 * ============================== UPSTREAM BUGS REPLICATED AND FLAGGED ==============================
 *
 *   B1 (1917) `la.lstsq(b_X, b_d, rcond=None)` where `la` is `scipy.linalg`, whose keyword is
 *      `cond` — `rcond` is numpy's. VERIFIED on scipy 1.16.2: every replicate raises
 *      `TypeError: lstsq() got an unexpected keyword argument 'rcond'. Did you mean 'cond'?`, the
 *      bare `except Exception: pass` at 1924 swallows it, `len(boot_t0) == 0 < 20`, and all FOUR
 *      spline intervals collapse to their point estimates. MEASURED on the acceptance case:
 *      `spline.ci_mrca == [1938.7746674292187, 1938.7746674292187]`, a zero-width 95% interval for
 *      a quantity whose real interval (with `cond=`) is about 55 years wide. `nBoot` therefore
 *      DEFAULTS TO 0 here and the four CIs are the point estimates, which is what the reference
 *      produces; a non-zero `nBoot` is refused rather than guessed at.
 *   B2 (1884) `dB[-1, 0]` reads the derivative at the LAST ROW IN INPUT ORDER, not at t_max, so
 *      `rate_recent`, `rate_ratio` and therefore `is_nonlinear_preferred` depend on the order of the
 *      FASTA. On korber the last training row happens to be t_max, so the published number is right
 *      by luck.
 *   B5 (1269) `p_value = 1.0 - stats.f.cdf(...)` where the sibling test at 1877 correctly uses
 *      `stats.f.sf`. The cancellation floors the reported p at one ulp of 1 (1.11e-16) for F ≳ 100.
 *      `runOlsDating` calls `1 - fCdf(...)`, deliberately; `fSf` exists for the other call site.
 *   B6 (1246, 1285) `ci_bootstrap` is initialised to `None` and never assigned; it ships in the
 *      JSON as a permanent null. Replicated as `null`.
 *   B7 (880-886) the UNBOUNDED branch uses `t_ref` — a DATE — as the upper bound when `min_time` is
 *      absent, and in that same case ignores the candidate root it just computed.
 *   B8 (1857-1866) the spline solves UNCENTRED normal equations on a calendar axis. MEASURED:
 *      `cond(X_spᵀX_sp) = 9.307e12` on the acceptance data, so only about five significant figures
 *      of the spline's `t_mrca` survive. See the tolerance note on `runRestrictedSplineClockDating`.
 *   B10 (2916-2919) the ensembler converts each interval to an SE by `(hi − lo)/(2·1.96)` — a
 *      symmetric-normal assumption applied to a deliberately skewed Fieller interval, with a 1.96
 *      that does not match the t-based bounds it is reading.
 *
 * ============================ MEASUREMENTS BEHIND THE NON-OBVIOUS CHOICES ============================
 *
 *   - THE 2x2 NORMAL EQUATIONS, WITH THE `Σx` TERM KEPT. Centring makes `XᵀX` nearly, but NOT
 *     exactly, diagonal: MEASURED on the acceptance case `Σx = -1.59e-11`, `Σx² = 1941.5460992907801`
 *     and `cov_beta[0,1] = 3.687e-21`. Against scipy's `lstsq` (gelsd/SVD): solving the full 2x2 by
 *     adjugate costs Δmu = −6.5e-19 relative and Δt_mrca = −2.3e-13 years, while ASSUMING `Σx = 0`
 *     (`mu = Σxd/Σx²`, `d0 = Σd/n`) costs −9.4e-16 and −7.8e-11. The full solve is ported.
 *   - Every reduction that numpy performs (`np.mean`, `np.sum`, `np.corrcoef`'s inner products) goes
 *     through `numpyPairwiseSum` in float64, so the summation ORDER is numpy's and not a left fold.
 *   - The `1e-15` floors under `se_mu`/`se_d0` (1210-1211), the `1e-12` in `f_stat` (1268), the
 *     `999.0` sentinel (1268) and every `1e-6`/`1e-9`/`1e-12` guard below are the reference's own
 *     numbers and are reproduced rather than rationalised.
 *
 * ONE CONSEQUENCE OF PHASE 4 THAT LANDS ON THIS FILE, recorded here because it changes a number this
 * file produces (fixtures/manifest.json, DATING Q10): `run_restricted_spline_clock_dating` becomes a
 * GLS spline the moment the model runs. dating.py:2842-2844 passes `cov_train` as `spline_cov`
 * whenever the neural path ran, so on IDENTICAL divergences the spline's `beta_0` moves from
 * -4.386825916889575 to -1.7112000894725579, its `t_mrca` from 1938.7746674292187 to 1864.5477949,
 * and the clock selection flips from Restricted Spline to Linear PGLS. `runRestrictedSplineClockDating`
 * here takes no covariance and is therefore the MODEL-FREE spline — the right answer when no model
 * ran, and the wrong one to show beside a PGLS fit. An application that runs both must say which
 * spline it drew.
 *
 * FULL-CHAIN VERIFICATION (measured in this session, tn93 binary off PATH, `*` rewritten to `-`,
 * `verify_coding_alignment`'s `L mod 3` trim applied): the chain parse -> parse_header_timestamp ->
 * computeTreeFreeDivergences -> coverage holdout -> runOlsDating / runRestrictedSplineClockDating
 * reproduces `hyphaeon dating`'s published record BIT FOR BIT on examples/korber_env_gp160.fasta
 * (n = 141 of 142 dated, t_MRCA 1893.91095511759, Fieller [1850.9002455209902, 1916.7928463014516])
 * and on examples/H1N1_2009_pandemic.fasta (n = 95, t_MRCA 2009.0426391755814, the time-decay
 * consensus root at γ = 3.0030). test/dating.test.js replays both.
 */

import { fCdf, fSf, tPpf } from './numeric/special.js';
import { numpyPairwiseSum, percentile } from './numeric/reduce.js';
import { brentq } from './numeric/optimize.js';
import { consensusSequence, timeDecayConsensusSequence } from './preprocess/consensus.js';
import { tn93CrossDistanceMatrix, tn93DistanceMatrix } from './preprocess/tn93.js';

const identity = (/** @type {number} */ v) => v;

/** `float(np.sum(a))` in float64: numpy's pairwise order, not a left fold. */
function sum64(a, n = a.length) {
	return numpyPairwiseSum(a, 0, n, identity);
}

/** `float(np.mean(a))`. */
function mean64(a, n = a.length) {
	return sum64(a, n) / n;
}

/** `float(np.std(a))`, ddof = 0 (the reference never passes ddof). */
function std64(a) {
	const n = a.length;
	if (n === 0) return NaN;
	const m = mean64(a);
	const sq = new Float64Array(n);
	for (let i = 0; i < n; i++) sq[i] = (a[i] - m) * (a[i] - m);
	return Math.sqrt(sum64(sq) / n);
}

/** A Float64Array view of any ArrayLike, copying only when it is not already one. */
function f64(a) {
	return a instanceof Float64Array ? a : Float64Array.from(a);
}

/** The three `status` values `run_ols_dating` can return (1215-1231). */
export const DATING_STATUS = Object.freeze({
	OK: 'OK',
	NON_POSITIVE_RATE: 'NON_POSITIVE_RATE',
	MRCA_AFTER_EARLIEST_SAMPLE: 'MRCA_AFTER_EARLIEST_SAMPLE',
});

/** The three `status` values `compute_fieller_mrca_interval` can return (856, 877, 887). */
export const FIELLER_STATUS = Object.freeze({
	BOUNDED: 'BOUNDED',
	UNBOUNDED_ANTIQUITY: 'UNBOUNDED_ANTIQUITY',
	NON_POSITIVE_RATE: 'NON_POSITIVE_RATE',
});

/** `ci_method` strings `run_ols_dating` dispatches on and this port implements (1248-1258). */
export const CI_METHODS_PORTED = Object.freeze(['fieller', 'delta', 'linear']);
/**
 * The three `ci_method` strings the reference implements and this port does not, each because it
 * needs numpy's PCG64 stream to reproduce. `runOlsDating` REFUSES them rather than silently taking
 * the reference's `else` branch into Fieller, which is what an unrecognised string does there.
 */
export const CI_METHODS_UNPORTED = Object.freeze(['poisson', 'residual-boot', 'wild', 'jackknife', 'jack', 'loocv']);

// =================================================================================================
// intervals
// =================================================================================================

/**
 * `compute_fieller_mrca_interval(mu, d0, cov_beta, t_ref, df, alpha, min_time)`, dating.py:839-887.
 *
 * Parameterise by θ = t_ref − t₀. The null is H₀: d₀ − μθ = 0, and accepting it at level α is
 * `(d0 − μθ)² ≤ t_crit²·Var(d0 − μθ)`, which is the quadratic `Aθ² + Bθ + C ≤ 0` below. Fieller's
 * `g = t_crit²·var_mu/mu²` is the same statement as `A > 0`: the rate is bounded away from zero at
 * that level. When it is not, the interval is genuinely half-infinite and `-Infinity` is returned,
 * not a number (the ensembler at 2897 skips such candidates explicitly, and a page must be able to
 * render it).
 *
 * Note the coordinate flip in the BOUNDED branch: the LARGER root in θ is the EARLIER date. Only
 * the upper bound is clamped to `minTime`; the lower bound is never clamped.
 *
 * @param {number} mu
 * @param {number} d0
 * @param {ArrayLike<number>} covBeta row-major 2x2 `[c00, c01, c10, c11]`
 * @param {number} tRef
 * @param {number} df
 * @param {{alpha?: number, minTime?: number|null}} [options]
 * @returns {{ci: [number, number], g: number, status: string}}
 */
export function computeFiellerMrcaInterval(mu, d0, covBeta, tRef, df, options = {}) {
	const alpha = options.alpha ?? 0.05;
	const minTime = options.minTime ?? null;
	if (mu <= 1e-12) return { ci: [NaN, NaN], g: NaN, status: FIELLER_STATUS.NON_POSITIVE_RATE };

	const tCrit = tPpf(1.0 - alpha / 2.0, Math.max(1, df));
	const varMu = covBeta[0];
	const varD0 = covBeta[3];
	const covMuD0 = covBeta[1];

	const g = (tCrit ** 2 * varMu) / mu ** 2;
	const A = mu ** 2 - tCrit ** 2 * varMu;
	const B = -2.0 * (mu * d0 - tCrit ** 2 * covMuD0);
	const C = d0 ** 2 - tCrit ** 2 * varD0;
	const disc = B ** 2 - 4.0 * A * C;

	if (A > 0 && disc >= 0) {
		const th1 = (-B - Math.sqrt(disc)) / (2.0 * A);
		const th2 = (-B + Math.sqrt(disc)) / (2.0 * A);
		const tLow = tRef - th2;
		let tHigh = tRef - th1;
		if (minTime !== null) tHigh = Math.min(minTime, tHigh);
		return { ci: [tLow, tHigh], g, status: FIELLER_STATUS.BOUNDED };
	}
	// g >= 1: the rate is not bounded away from zero, so antiquity is unbounded. B7: with no
	// min_time the reference uses t_ref — a date — as the bound, and ignores `cand` entirely.
	let tHigh = minTime !== null ? minTime : tRef;
	if (disc >= 0 && Math.abs(A) > 1e-12) {
		const th1 = (-B - Math.sqrt(disc)) / (2.0 * A);
		const cand = tRef - th1;
		if (minTime !== null) tHigh = Math.min(minTime, cand);
	}
	return { ci: [-Infinity, tHigh], g, status: FIELLER_STATUS.UNBOUNDED_ANTIQUITY };
}

/**
 * The delta-method interval of dating.py:1233-1237, lifted out of `run_ols_dating` so it and
 * Fieller can be tested side by side on the same `cov_beta`.
 *
 * `grad = [d0/mu², −1/mu]`, `var = gradᵀ·cov·grad`, and `t_crit = t.ppf(0.975, max(1, df))`. The
 * UPPER bound is clamped to `minTime` and the lower is not — the asymmetry is the reference's.
 *
 * MEASURED on the acceptance case, for the record: delta [1864.04, 1923.78] against Fieller
 * [1850.90, 1916.79] at g = 0.0934 — Fieller is 13 years wider and shifted earlier.
 *
 * @param {number} mu
 * @param {number} d0
 * @param {ArrayLike<number>} covBeta row-major 2x2
 * @param {number} tRef
 * @param {number} df
 * @param {{alpha?: number, minTime?: number|null}} [options] `alpha` defaults to 0.05, i.e. t.ppf(0.975)
 * @returns {{ci: [number, number], seMrca: number, tCrit: number}}
 */
export function computeDeltaMrcaInterval(mu, d0, covBeta, tRef, df, options = {}) {
	const alpha = options.alpha ?? 0.05;
	const minTime = options.minTime ?? null;
	const tMrca = tRef - d0 / mu;
	const g0 = d0 / mu ** 2;
	const g1 = -1.0 / mu;
	// gradᵀ · cov · grad, written out: the 2x2 is symmetric, but the reference multiplies the full
	// matrix, so both off-diagonal terms appear.
	const varMrca = g0 * (covBeta[0] * g0 + covBeta[1] * g1) + g1 * (covBeta[2] * g0 + covBeta[3] * g1);
	const seMrca = Math.sqrt(Math.max(0.0, varMrca));
	const tCrit = tPpf(1.0 - alpha / 2.0, Math.max(1, df));
	const lo = tMrca - tCrit * seMrca;
	let hi = tMrca + tCrit * seMrca;
	if (minTime !== null) hi = Math.min(minTime, hi);
	return { ci: [lo, hi], seMrca, tCrit };
}

// =================================================================================================
// the centred OLS fit
// =================================================================================================

/**
 * `run_ols_dating(times, dists, t_ref, ci_method, ...)`, dating.py:1170-1296: the centred
 * root-to-tip least-squares clock `d_i = mu·(t_i − t_ref) + d_0 + ε_i` and its ancestor date
 * `t_MRCA = t_ref − d_0/mu`.
 *
 * The returned record is the reference's dict verbatim, including `ci_bootstrap: null` (B6) and
 * `ci_method` echoed back UN-NORMALISED, and including the three arrays the JSON export strips.
 *
 * THE THREE STATUSES ARE NOT LABELS. `NON_POSITIVE_RATE` (mu ≤ 1e-12) and
 * `MRCA_AFTER_EARLIEST_SAMPLE` (t_MRCA ≥ min(times)) both NaN out `t_mrca` and `se_mrca`: the
 * reference DISCARDS the estimate in those cases, it does not merely flag it, and both intervals
 * stay `[NaN, NaN]`. Note that `min_time` is the earliest TRAINING time, not the earliest sample in
 * the dataset — on the acceptance case that is 1983.5, not the 1959.5 the printed timespan shows,
 * because the 1959 isolate is a coverage holdout.
 *
 * @param {ArrayLike<number>} times sampling times of the FIT subset
 * @param {ArrayLike<number>} dists root divergences of the same subset, same order
 * @param {{tRef?: number|null, ciMethod?: string, alpha?: number}} [options]
 * @returns {Record<string, any>} the reference's key set, in the reference's order
 * @throws {RangeError} n < 3 (dating.py:1190), or an unported `ci_method`
 */
export function runOlsDating(times, dists, options = {}) {
	const t = f64(times);
	const d = f64(dists);
	const n = t.length;
	if (n < 3) throw new RangeError(`At least 3 observations are required for OLS dating (got N=${n}).`);
	if (d.length !== n) throw new RangeError(`times and dists differ in length (${n} vs ${d.length}).`);

	const ciMethod = options.ciMethod ?? 'fieller';
	const ciLower = String(ciMethod).toLowerCase();
	if (CI_METHODS_UNPORTED.includes(ciLower)) {
		throw new RangeError(
			`ci_method '${ciMethod}' is not ported: the Poisson, wild-residual and jackknife intervals each need ` +
				`numpy's PCG64 stream to reproduce (dating.py:890-968, 970-1168). The reference's own fall-through would ` +
				`silently return the Fieller interval instead, which is why this refuses rather than answering.`
		);
	}

	const tRef = options.tRef ?? mean64(t);
	const x = new Float64Array(n);
	for (let i = 0; i < n; i++) x[i] = t[i] - tRef;

	// The full 2x2 normal equations. XᵀX = [[Σx², Σx], [Σx, n]] — the Σx term is kept; see header.
	const xx = new Float64Array(n);
	const xd = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		xx[i] = x[i] * x[i];
		xd[i] = x[i] * d[i];
	}
	const sX = sum64(x);
	const sXX = sum64(xx);
	const sXD = sum64(xd);
	const sD = sum64(d);
	const det = sXX * n - sX * sX;
	// `la.inv(XᵀX)` at dating.py:1208 RAISES `numpy.linalg.LinAlgError: singular matrix` when the
	// design is degenerate — every sampling time identical is the way a reader reaches it — and it
	// raises before any status branch is taken. Refusing here is the same answer, not a new one;
	// returning an infinite rate would be a different one. Pinned by
	// fixtures/dating/run_ols_dating.json's `zero_variance_times_raises` case.
	if (det === 0 || !Number.isFinite(det)) {
		throw new RangeError('singular matrix: the OLS design has no variance in `times` (dating.py:1208 raises LinAlgError here)');
	}
	const mu = (n * sXD - sX * sD) / det;
	const d0 = (sXX * sD - sX * sXD) / det;

	let minTime = t[0];
	for (let i = 1; i < n; i++) if (t[i] < minTime) minTime = t[i];

	const res = new Float64Array(n);
	const fitted = new Float64Array(n);
	const res2 = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		fitted[i] = x[i] * mu + d0;
		res[i] = d[i] - fitted[i];
		res2[i] = res[i] * res[i];
	}
	const sigma2 = sum64(res2) / Math.max(1, n - 2);
	// cov_beta = sigma2 * inv(XᵀX); inv by adjugate: [[n, −Σx], [−Σx, Σx²]] / det.
	const covBeta = new Float64Array([(sigma2 * n) / det, (sigma2 * -sX) / det, (sigma2 * -sX) / det, (sigma2 * sXX) / det]);
	const seMu = Math.sqrt(Math.max(1e-15, covBeta[0]));
	const seD0 = Math.sqrt(Math.max(1e-15, covBeta[3]));

	let ciFieller = [NaN, NaN];
	let fiellerG = NaN;
	let ciDelta = [NaN, NaN];
	const ciBootstrap = null; // B6: initialised and never assigned, upstream
	let ciMrca = [NaN, NaN];
	let tMrca;
	let seMrca;
	let status;

	if (mu <= 1e-12) {
		tMrca = NaN;
		seMrca = NaN;
		status = DATING_STATUS.NON_POSITIVE_RATE;
	} else {
		tMrca = tRef - d0 / mu;
		if (tMrca >= minTime) {
			tMrca = NaN;
			seMrca = NaN;
			status = DATING_STATUS.MRCA_AFTER_EARLIEST_SAMPLE;
		} else {
			status = DATING_STATUS.OK;
			const delta = computeDeltaMrcaInterval(mu, d0, covBeta, tRef, Math.max(1, n - 2), {
				alpha: options.alpha ?? 0.05,
				minTime,
			});
			seMrca = delta.seMrca;
			ciDelta = delta.ci;
			const fieller = computeFiellerMrcaInterval(mu, d0, covBeta, tRef, Math.max(1, n - 2), {
				alpha: options.alpha ?? 0.05,
				minTime,
			});
			ciFieller = fieller.ci;
			fiellerG = fieller.g;
			ciMrca = ciLower === 'delta' || ciLower === 'linear' ? ciDelta : ciFieller;
		}
	}

	// np.corrcoef(times, dists)[0, 1], with np.cov's ddof = 1 (which cancels) and its divide order.
	const stdT = std64(t);
	const stdD = std64(d);
	let r = 0.0;
	if (stdT > 1e-8 && stdD > 1e-8) {
		const mt = mean64(t);
		const md = mean64(d);
		const ct = new Float64Array(n);
		const cd = new Float64Array(n);
		for (let i = 0; i < n; i++) {
			ct[i] = t[i] - mt;
			cd[i] = d[i] - md;
		}
		const p00 = new Float64Array(n);
		const p01 = new Float64Array(n);
		const p11 = new Float64Array(n);
		for (let i = 0; i < n; i++) {
			p00[i] = ct[i] * ct[i];
			p01[i] = ct[i] * cd[i];
			p11[i] = cd[i] * cd[i];
		}
		const fact = 1 / (n - 1);
		const c00 = sum64(p00) * fact;
		const c01 = sum64(p01) * fact;
		const c11 = sum64(p11) * fact;
		const s0 = Math.sqrt(c00);
		const s1 = Math.sqrt(c11);
		r = c01 / s0 / s1;
		if (r < -1) r = -1;
		if (r > 1) r = 1;
	}
	const r2 = r ** 2;
	const fStat = r2 < 1.0 ? (r2 / (1.0 - r2 + 1e-12)) * (n - 2) : 999.0;
	// B5: the reference spells this `1 - f.cdf`, not `f.sf`, and the cancellation floors it at 1.11e-16.
	const pValue = 1.0 - fCdf(fStat, 1, Math.max(1, n - 2));

	return {
		method: 'OLS',
		status,
		mu,
		d0,
		t_ref: tRef,
		t_mrca: tMrca,
		se_mu: seMu,
		se_d0: seD0,
		se_mrca: seMrca,
		ci_fieller: ciFieller,
		fieller_g: fiellerG,
		ci_delta: ciDelta,
		ci_bootstrap: ciBootstrap,
		ci_mrca: ciMrca,
		ci_method: ciMethod,
		r,
		r2,
		p_value: pValue,
		sigma2,
		rmse: Math.sqrt(mean64(res2)), // mean over n, NOT over n − 2 (1287)
		residuals: res,
		fitted,
		times: t,
		n,
		cov_beta: covBeta, // not a reference key; the intervals need it and the app must not refit
	};
}

// =================================================================================================
// the restricted cubic spline clock
// =================================================================================================

/**
 * `compute_rcs_basis(x, knots)`, dating.py:1774-1806: Harrell's restricted cubic spline basis and
 * its first derivative, `K − 2` non-linear columns for `K` knots.
 *
 * Because `knots[0] = t_min` in every call the reference makes, every column and every derivative
 * is identically zero for `x ≤ t_min` — which is the whole point: extrapolation back to the
 * ancestor is strictly linear, with no boundary blow-up.
 *
 * @param {ArrayLike<number>} x
 * @param {ArrayLike<number>} knots strictly increasing, at least 3
 * @returns {{basis: Float64Array, derivative: Float64Array, ncols: number, nrows: number}}
 *   row-major `x.length x ncols`
 * @throws {RangeError} fewer than 3 knots (dating.py:1789)
 */
export function computeRcsBasis(x, knots) {
	const xs = f64(x);
	const k = f64(knots);
	const K = k.length;
	if (K < 3) throw new RangeError('At least 3 knots are required for restricted cubic splines.');
	const t1 = k[0];
	const tk1 = k[K - 2];
	const tk = k[K - 1];
	const denom = (tk - t1) ** 2;
	const ncols = K - 2;
	const nrows = xs.length;
	const basis = new Float64Array(nrows * ncols);
	const derivative = new Float64Array(nrows * ncols);
	const pos = (/** @type {number} */ v) => (v > 0 ? v : 0); // np.maximum(0.0, ·)
	for (let j = 0; j < ncols; j++) {
		const tj = k[j];
		const w1 = (tk - tj) / (tk - tk1);
		const w2 = (tk1 - tj) / (tk - tk1);
		for (let i = 0; i < nrows; i++) {
			const a = pos(xs[i] - tj);
			const b = pos(xs[i] - tk1);
			const c = pos(xs[i] - tk);
			basis[i * ncols + j] = (a ** 3 - w1 * b ** 3 + w2 * c ** 3) / denom;
			derivative[i * ncols + j] = (3.0 * a ** 2 - w1 * (3.0 * b ** 2) + w2 * (3.0 * c ** 2)) / denom;
		}
	}
	return { basis, derivative, ncols, nrows };
}

/**
 * Gaussian elimination with partial pivoting on a small dense system, row-major `A` of size p.
 * This is what `scipy.linalg.solve` (LAPACK `gesv`) does; the ORDER of the eliminations differs, so
 * the two are not bit-identical on an ill-conditioned system. See the tolerance note below.
 */
function solveDense(A, b, p) {
	const M = Float64Array.from(A);
	const y = Float64Array.from(b);
	for (let col = 0; col < p; col++) {
		let piv = col;
		let best = Math.abs(M[col * p + col]);
		for (let r = col + 1; r < p; r++) {
			const v = Math.abs(M[r * p + col]);
			if (v > best) {
				best = v;
				piv = r;
			}
		}
		if (piv !== col) {
			for (let c = 0; c < p; c++) {
				const tmp = M[col * p + c];
				M[col * p + c] = M[piv * p + c];
				M[piv * p + c] = tmp;
			}
			const tb = y[col];
			y[col] = y[piv];
			y[piv] = tb;
		}
		const d = M[col * p + col];
		if (d === 0) throw new RangeError('singular matrix in the spline clock normal equations');
		for (let r = col + 1; r < p; r++) {
			const f = M[r * p + col] / d;
			if (f === 0) continue;
			for (let c = col; c < p; c++) M[r * p + c] -= f * M[col * p + c];
			y[r] -= f * y[col];
		}
	}
	const out = new Float64Array(p);
	for (let r = p - 1; r >= 0; r--) {
		let s = y[r];
		for (let c = r + 1; c < p; c++) s -= M[r * p + c] * out[c];
		out[r] = s / M[r * p + r];
	}
	return out;
}

/** `Xᵀ X` and `Xᵀ y` for a row-major n x p design, every inner product summed in numpy's order. */
function normalEquations(X, y, n, p) {
	const A = new Float64Array(p * p);
	const rhs = new Float64Array(p);
	const buf = new Float64Array(n);
	for (let a = 0; a < p; a++) {
		for (let b = a; b < p; b++) {
			for (let i = 0; i < n; i++) buf[i] = X[i * p + a] * X[i * p + b];
			const v = sum64(buf);
			A[a * p + b] = v;
			A[b * p + a] = v;
		}
		for (let i = 0; i < n; i++) buf[i] = X[i * p + a] * y[i];
		rhs[a] = sum64(buf);
	}
	return { A, rhs };
}

/**
 * `run_restricted_spline_clock_dating(times, dists, cov_matrix, ridge, n_boot, seed)`,
 * dating.py:1809-1968: the three-knot restricted cubic spline clock, its nested F test against the
 * straight line, and the automatic preference rule.
 *
 * KNOTS are `[min(times), median(times), percentile(times, 90)]` at numpy's `'linear'` method; the
 * knot is on the deterministic critical path, so `percentile` here is held to bit-equality with
 * numpy rather than to a tolerance. MEASURED on the acceptance case: [1983.5, 1992.5, 1995.5].
 *
 * THE PREFERENCE RULE HAS FOUR CONJUNCTS, not three (1894-1897): `p < 0.05`, `ΔAIC ≥ 2`,
 * `mu_ancestral > 0` AND `|rate_ratio − 1| ≥ 0.15`. The positive-ancestral-rate clause is easy to
 * miss and is what stops a backwards clock from being "preferred".
 *
 * TOLERANCE, AND WHY IT IS LOOSER THAN EVERYTHING ELSE HERE (B8). The reference solves the
 * UNCENTRED normal equations on a calendar axis: MEASURED `cond(X_spᵀX_sp) = 9.307e12` on the
 * acceptance data. Partial-pivot Gaussian elimination in float64 reproduces LAPACK's `t_mrca` to
 * 1.1e-8 years, and merely REASSOCIATING the reference's own products (multiplying by the identity
 * `C_inv` first, as it does) moves the answer by ~3.5e-10 years. Centring would change the answer
 * at the 1e-8 level and is "improving during a port", so it is not done, and the spline's
 * `t_mrca` is pinned at 1e-5 years absolute while the OLS fit is pinned at 1e-9.
 *
 * `cov_matrix` is always None on the phase-3 path (PGLS is phase 4), so `C_inv` is the identity and
 * multiplying by it is exact; the ridge is unused. `nBoot` defaults to 0 — see B1 in the header.
 *
 * @param {ArrayLike<number>} times
 * @param {ArrayLike<number>} dists
 * @param {{nBoot?: number}} [options]
 * @returns {Record<string, any>} the reference's key set, in the reference's order
 * @throws {RangeError} n < 5 (dating.py:1837), or nBoot !== 0
 */
export function runRestrictedSplineClockDating(times, dists, options = {}) {
	const t = f64(times);
	const d = f64(dists);
	const n = t.length;
	if (n < 5) throw new RangeError(`At least 5 observations are required for restricted spline dating (got N=${n}).`);
	if (d.length !== n) throw new RangeError(`times and dists differ in length (${n} vs ${d.length}).`);
	const nBoot = options.nBoot ?? 0;
	if (nBoot !== 0) {
		throw new RangeError(
			'nBoot must be 0: the reference bootstrap at dating.py:1900-1941 raises TypeError on every replicate ' +
				'(la.lstsq(..., rcond=...) on scipy.linalg) and its four intervals therefore collapse to their point ' +
				'estimates, which is what this reproduces. Running the loop as intended would need numpy PCG64 resampling ' +
				'and would match neither the reference nor this port.'
		);
	}

	let tMin = t[0];
	for (let i = 1; i < n; i++) if (t[i] < tMin) tMin = t[i];
	const knots = Float64Array.from([tMin, percentile(t, 50), percentile(t, 90)]);
	const { basis: B, derivative: dB, ncols } = computeRcsBasis(t, knots);

	// 1. the linear null: X_lin = [1, times] (UNCENTRED, B8)
	const pLin = 2;
	const Xlin = new Float64Array(n * pLin);
	for (let i = 0; i < n; i++) {
		Xlin[i * pLin] = 1;
		Xlin[i * pLin + 1] = t[i];
	}
	const eqLin = normalEquations(Xlin, d, n, pLin);
	const betaLin = solveDense(eqLin.A, eqLin.rhs, pLin);
	const resLin = new Float64Array(n);
	const resLin2 = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		resLin[i] = d[i] - (betaLin[0] + betaLin[1] * t[i]);
		resLin2[i] = resLin[i] * resLin[i];
	}
	const rssLin = sum64(resLin2);
	const aicLin = n * Math.log(Math.max(1e-12, rssLin / n)) + 2 * 2;

	// 2. the spline: X_sp = [1, times, B]
	const pSp = 2 + ncols;
	const Xsp = new Float64Array(n * pSp);
	for (let i = 0; i < n; i++) {
		Xsp[i * pSp] = 1;
		Xsp[i * pSp + 1] = t[i];
		for (let j = 0; j < ncols; j++) Xsp[i * pSp + 2 + j] = B[i * ncols + j];
	}
	const eqSp = normalEquations(Xsp, d, n, pSp);
	const betaSp = solveDense(eqSp.A, eqSp.rhs, pSp);
	const fittedSp = new Float64Array(n);
	const resSp = new Float64Array(n);
	const resSp2 = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		let v = 0;
		for (let j = 0; j < pSp; j++) v += Xsp[i * pSp + j] * betaSp[j];
		fittedSp[i] = v;
		resSp[i] = d[i] - v;
		resSp2[i] = resSp[i] * resSp[i];
	}
	const rssSp = sum64(resSp2);
	const aicSp = n * Math.log(Math.max(1e-12, rssSp / n)) + 2 * 3;
	const deltaAic = aicLin - aicSp;

	// 3. the nested F test — `stats.f.sf` here, unlike run_ols_dating's `1 - cdf` (B5)
	const dfSp = n - 3;
	const diffRss = Math.max(0.0, rssLin - rssSp);
	const fStat = diffRss / 1.0 / Math.max(1e-12, rssSp / Math.max(1, dfSp));
	const pFTest = fSf(fStat, 1, Math.max(1, dfSp));

	const muAncestral = betaSp[1];
	let t0Sp;
	if (muAncestral > 1e-9) t0Sp = -betaSp[0] / muAncestral;
	else if (betaLin[1] > 1e-9) t0Sp = -betaLin[0] / Math.max(1e-12, betaLin[1]);
	else t0Sp = NaN;

	// B2: the derivative at the LAST ROW IN INPUT ORDER, not at t_max.
	const dBMax = dB.length > 0 ? dB[(n - 1) * ncols] : 0.0;
	const muRecent = betaSp[1] + betaSp[2] * dBMax;
	const rateRatio = muAncestral > 1e-9 ? muRecent / muAncestral : 1.0;

	const isNonlinearPreferred = pFTest < 0.05 && deltaAic >= 2.0 && muAncestral > 0 && Math.abs(rateRatio - 1.0) >= 0.15;

	// B1: the bootstrap is dead upstream, so the four intervals ARE the point estimates.
	const ciT0 = [t0Sp, t0Sp];
	const ciMuAnc = [muAncestral, muAncestral];
	const ciMuRec = [muRecent, muRecent];
	const ciBeta2 = [betaSp[2], betaSp[2]];

	// generalised R², with C_inv the identity
	const weightedMean = sum64(d) / Math.max(1e-12, n);
	const totRes2 = new Float64Array(n);
	for (let i = 0; i < n; i++) totRes2[i] = (d[i] - weightedMean) * (d[i] - weightedMean);
	const ssTot = sum64(totRes2);
	const r2Sp = Math.max(0.0, 1.0 - rssSp / Math.max(1e-12, ssTot));

	return {
		method: 'RESTRICTED_SPLINE',
		t_mrca: t0Sp,
		ci_mrca: ciT0,
		rate_ancestral: muAncestral,
		ci_rate_ancestral: ciMuAnc,
		rate_recent: muRecent,
		ci_rate_recent: ciMuRec,
		rate_ratio: rateRatio,
		beta_0: betaSp[0],
		beta_1: betaSp[1],
		beta_2: betaSp[2],
		beta: Array.from(betaSp.slice(0, 3)),
		ci_beta_2: ciBeta2,
		knots: Array.from(knots),
		rss: rssSp,
		aic: aicSp,
		rss_linear: rssLin,
		aic_linear: aicLin,
		delta_aic: deltaAic,
		f_stat: fStat,
		p_f_test: pFTest,
		is_nonlinear_preferred: isNonlinearPreferred,
		r2: r2Sp,
		rmse: Math.sqrt(mean64(resSp2)),
		fitted: fittedSp,
		residuals: resSp,
		n,
	};
}

// =================================================================================================
// the per-taxon block
// =================================================================================================

/**
 * The fitted divergences and inverted dates of `run_mrca_dating`'s per-taxon block,
 * dating.py:3000-3042, over ALL taxa — training rows and holdouts alike, under the SELECTED model.
 *
 * Under the spline the knots come from the model (fitted on the training subset) and the basis is
 * re-evaluated at every taxon's time, so a holdout earlier than `knots[0]` sits on the strictly
 * linear arm by construction. Three arms, in the reference's order:
 *   - `b1 ≤ 1e-6` -> every predicted date is NaN;
 *   - `d ≤ d_kn0` or `|b2| < 1e-12` -> the linear inverse `(d − b0)/b1`;
 *   - otherwise `brentq` on `[knots[0], max(times) + 100]` — AND ITS FAILURE IS PART OF THE ANSWER.
 *     A decelerating spline is bounded above, so taxa whose divergence exceeds the fit at the
 *     bracket's right end have no root there, `brentq` raises, and dating.py:3026's bare `except`
 *     substitutes the ANCESTRAL LINEAR inverse. MEASURED on the acceptance case: 12 of 142 rows
 *     take the linear arm, 118 are solved, and 12 more are refused and silently switched to a
 *     different model with nothing in the JSON or CSV saying which rows they are. `method` is
 *     reported per taxon here so a page CAN say so; the reference emits no such field.
 *
 * `max(times)` is over every taxon, not the fit (3023).
 *
 * @param {Record<string, any>} model an OLS record (`mu`, `d0`, `t_ref`) or a spline record
 * @param {ArrayLike<number>} times every taxon's sampling time, in the record's order
 * @param {ArrayLike<number>} dists every taxon's root divergence, same order
 * @returns {{fitted: Float64Array, predicted: Float64Array, methods: string[], dKnot0: number|null}}
 *   `methods[i]` is 'spline' | 'linear_fallback' | 'linear_arm' | 'ols' | 'none'
 */
export function clockFittedAndPredicted(model, times, dists) {
	const t = f64(times);
	const d = f64(dists);
	const n = t.length;
	const fitted = new Float64Array(n);
	const predicted = new Float64Array(n);
	const methods = new Array(n);

	if (model.method === 'POWER_LAW') {
		throw new RangeError('the POWER_LAW clock (dating.py:1564-1772, 3029-3034) is not ported; see the file header');
	}

	if (model.method === 'RESTRICTED_SPLINE') {
		const b0 = model.beta_0 ?? model.beta[0];
		const b1 = model.beta_1 ?? model.beta[1];
		const b2 = model.beta_2 ?? model.beta[2];
		const knots = f64(model.knots);
		const { basis: Ball, ncols } = computeRcsBasis(t, knots);
		for (let i = 0; i < n; i++) fitted[i] = b0 + b1 * t[i] + b2 * Ball[i * ncols];
		const tKn0 = knots[0];
		const { basis: Bk } = computeRcsBasis([tKn0], knots);
		const dKn0 = b0 + b1 * tKn0 + b2 * Bk[0];
		let tMax = t[0];
		for (let i = 1; i < n; i++) if (t[i] > tMax) tMax = t[i];
		const right = tMax + 100.0;
		for (let i = 0; i < n; i++) {
			const dv = d[i];
			if (b1 <= 1e-6) {
				predicted[i] = NaN;
				methods[i] = 'none';
			} else if (dv <= dKn0 || Math.abs(b2) < 1e-12) {
				predicted[i] = (dv - b0) / b1;
				methods[i] = 'linear_arm';
			} else {
				const f = (/** @type {number} */ tc) => {
					const { basis } = computeRcsBasis([tc], knots);
					return b0 + b1 * tc + b2 * basis[0] - dv;
				};
				try {
					predicted[i] = brentq(f, tKn0, right).x;
					methods[i] = 'spline';
				} catch {
					predicted[i] = (dv - b0) / b1;
					methods[i] = 'linear_fallback';
				}
			}
		}
		return { fitted, predicted, methods, dKnot0: dKn0 };
	}

	// OLS (and, in the reference, PGLS): the plain centred inverse.
	const mu = model.mu;
	const d0 = model.d0;
	const tRef = model.t_ref;
	for (let i = 0; i < n; i++) fitted[i] = d0 + mu * (t[i] - tRef);
	if (mu > 1e-6) {
		for (let i = 0; i < n; i++) {
			predicted[i] = tRef + (d[i] - d0) / mu;
			methods[i] = 'ols';
		}
	} else {
		predicted.fill(NaN);
		methods.fill('none');
	}
	return { fitted, predicted, methods, dKnot0: null };
}

/**
 * `std_res` of dating.py:3041-3042: the POPULATION standard deviation (ddof = 0) of the TRAINING
 * residuals, with both fallbacks — the whole set when fewer than three training rows survive or
 * their spread is degenerate, and a flat 1.0 when that too is degenerate.
 *
 * It is a plain standard deviation, not a robust scale, so several bad dates mask one another by
 * inflating their own denominator; the reference's outlier threshold sits at |z| ≥ 2.5 on top of
 * it. Both are replicated on purpose (runtime/src/clockRegression.js says the same thing about its
 * own independent copy).
 *
 * @param {ArrayLike<number>} residuals over ALL taxa
 * @param {ArrayLike<number>} trainIndices indices into `residuals`
 * @returns {number}
 */
export function residualScale(residuals, trainIndices) {
	const all = f64(residuals);
	const train = Float64Array.from(trainIndices, (i) => all[i]);
	if (train.length >= 3) {
		const s = std64(train);
		if (s > 1e-12) return s;
	}
	const sAll = std64(all);
	return sAll > 1e-12 ? sAll : 1.0;
}

/**
 * The precision-weighted ensemble ALGEBRA of dating.py:2895-2921 — and only the algebra. Which
 * models are admitted (bounded Fieller, clade attenuation, `is_nonlinear_preferred`) is a policy
 * the application owns, because it depends on which estimators that surface ran.
 *
 * Weights are `1/width²`; the total variance is Burnham & Anderson eq. 4.9, within-model plus
 * between-model; the interval is `t ± 1.96·SE` with the upper bound clamped to the earliest sample.
 * B10: the per-model SE is `(hi − lo)/(2·1.96)`, a symmetric-normal read of a deliberately skewed
 * Fieller interval, and the 1.96 does not match the t-based bounds it is reading. A candidate whose
 * interval is not finite and positive-width is dropped, which is why a spline whose bootstrap died
 * (B1) is silently ignored even when it is the SELECTED model — on the acceptance case
 * `selected_clock` is the spline and `ensemble.weights` is `{ols: 1.0}` at the same time.
 *
 * @param {Array<{name: string, t_mrca: number, ci_mrca: ArrayLike<number>|null|undefined}>} candidates
 *   in the order the reference would have inserted them (ols, pgls, spline)
 * @param {number} minSampleTime `min(times)` over every taxon
 * @returns {{t_mrca: number, ci: [number, number], weights: Record<string, number>}|null} null when
 *   no candidate is admissible (the reference then falls back to OLS alone — app policy, not here)
 */
export function precisionWeightedEnsemble(candidates, minSampleTime) {
	const precisions = [];
	for (const c of candidates) {
		const ci = c.ci_mrca;
		const w = ci && ci[0] !== -Infinity && ci[1] !== Infinity ? ci[1] - ci[0] : NaN;
		if (!Number.isNaN(w) && w > 0) precisions.push({ name: c.name, p: 1.0 / w ** 2, model: c });
	}
	if (precisions.length === 0) return null;
	let totPrec = 0;
	for (const e of precisions) totPrec += e.p;
	const weights = {};
	for (const e of precisions) weights[e.name] = e.p / Math.max(1e-12, totPrec);
	let tEns = 0;
	for (const e of precisions) tEns += weights[e.name] * e.model.t_mrca;
	let totVar = 0.0;
	for (const e of precisions) {
		const ci = e.model.ci_mrca;
		const se = (ci[1] - ci[0]) / (2.0 * 1.96);
		totVar += weights[e.name] * (se ** 2 + (e.model.t_mrca - tEns) ** 2);
	}
	const seEns = Math.sqrt(Math.max(1e-12, totVar));
	return { t_mrca: tEns, ci: [tEns - 1.96 * seEns, Math.min(minSampleTime, tEns + 1.96 * seEns)], weights };
}

// =================================================================================================
// tree-free root-to-tip divergences
// =================================================================================================

/** The two synthetic keys the reference writes into a COPY of the alignment (dating.py:653, 694). */
export const SYNTHETIC_CONSENSUS_KEY = '__SYNTHETIC_CONSENSUS__';
export const TIME_DECAY_ROOT_KEY = '__TIME_DECAY_ROOT__';
/** The magic `root_taxon` strings of cases 2 and 3, matched case-INSENSITIVELY (650, 659). */
export const UNWEIGHTED_CONSENSUS_ALIASES = Object.freeze(['unweighted_consensus', 'flat_consensus', 'modal_consensus']);
export const EARLIEST_ALIASES = Object.freeze(['earliest', 'earliest_taxon', 'earliest_cohort']);

/** `f"{v:.4f}"` for the γ the root description prints (dating.py:698). */
function fixed4(v) {
	return v.toFixed(4);
}

/**
 * `compute_tree_free_divergences(seq_dict, dated_taxa, dates_map, root_taxon, decay_gamma,
 * decay_half_life)`, dating.py:624-698: root-to-tip divergence with no tree at all, as the TN93
 * distance from every dated taxon to a root anchor.
 *
 * FOUR CASES, TESTED IN THIS ORDER — the order is itself a quirk:
 *   1. `root_taxon` is a LITERAL KEY of the alignment (643). Tested FIRST and case-sensitively, so a
 *      sequence actually named `earliest` beats the magic string below.
 *   2. `root_taxon.lower()` in UNWEIGHTED_CONSENSUS_ALIASES (650): the modal consensus of the dated
 *      taxa, inserted under `__SYNTHETIC_CONSENSUS__` in a COPY of the alignment — a key that would
 *      collide with a real sequence of that name.
 *   3. `root_taxon.lower()` in EARLIEST_ALIASES (659): the earliest dated taxon, or the mean over a
 *      cohort within 1e-4 time units (≈ 53 minutes on a calendar axis) of it. Two sub-branches that
 *      differ in more than cost: the rectangular one imputes against the maximum of ONE COLUMN and
 *      returns float64, the square one against the whole matrix and stays float32. NOTE that this
 *      case keeps the root IN the regression, at divergence exactly 0 — unlike case 1, where the
 *      root was already removed from `dated_taxa` upstream.
 *   4. otherwise (689): the time-decay weighted consensus, the reference's recommended default.
 *
 * The returned `root_description` is the reference's own provenance token, verbatim, including the
 * non-ASCII `γ` and the `.4f` formatting; it is not a sentence for a reader.
 *
 * THE `*` CONVENTION IS NOT DECIDED HERE — see the note on `tn93CrossDistanceMatrix`. Hand this
 * function the sequences the pillar's reference-of-record used.
 *
 * @param {Map<string, string>|Record<string, string>} sequences the WHOLE alignment, in file order
 * @param {string[]} datedTaxa the dated taxa, in alignment order, root already removed (2497)
 * @param {Map<string, number>|Record<string, number>} dates
 * @param {{rootTaxon?: string|null, decayGamma?: number|null, decayHalfLife?: number|null,
 *   matchMode?: string, maxAmbigFraction?: number, ignoreGaps?: boolean, threshold?: number}} [options]
 * @returns {{divergences: Float64Array, root_description: string, taxa: string[], case: number,
 *   root_taxa: string[]|null, gamma: number|null, root_sequence: string|null}}
 */
export function computeTreeFreeDivergences(sequences, datedTaxa, dates, options = {}) {
	const rootTaxon = options.rootTaxon ?? null;
	const asMap = sequences instanceof Map ? sequences : new Map(Object.entries(sequences));
	const dateOf = dates instanceof Map ? (/** @type {string} */ t) => dates.get(t) : (/** @type {string} */ t) => dates[t];
	const hasDate = dates instanceof Map ? (/** @type {string} */ t) => dates.has(t) : (/** @type {string} */ t) => Object.prototype.hasOwnProperty.call(dates, t);
	const col0 = (/** @type {Float32Array} */ mat, /** @type {number} */ m) => {
		const out = new Float64Array(mat.length / m);
		for (let i = 0; i < out.length; i++) out[i] = mat[i * m];
		return out;
	};

	// Case 1: an explicit taxon of the alignment.
	if (rootTaxon && asMap.has(rootTaxon)) {
		const evalTaxa = datedTaxa.filter((t) => t !== rootTaxon);
		const cross = tn93CrossDistanceMatrix(asMap, evalTaxa, [rootTaxon], options);
		return {
			divergences: col0(cross, 1),
			root_description: `explicit_root_${rootTaxon}`,
			taxa: evalTaxa,
			case: 1,
			root_taxa: [rootTaxon],
			gamma: null,
			root_sequence: asMap.get(rootTaxon),
		};
	}

	// Case 2: the unweighted modal consensus.
	if (rootTaxon && UNWEIGHTED_CONSENSUS_ALIASES.includes(rootTaxon.toLowerCase())) {
		const conSeq = consensusSequence(asMap, datedTaxa);
		const aug = new Map(asMap);
		aug.set(SYNTHETIC_CONSENSUS_KEY, conSeq);
		const cross = tn93CrossDistanceMatrix(aug, datedTaxa, [SYNTHETIC_CONSENSUS_KEY], options);
		return {
			divergences: col0(cross, 1),
			root_description: 'unweighted_modal_consensus_root',
			taxa: datedTaxa.slice(),
			case: 2,
			root_taxa: null,
			gamma: null,
			root_sequence: conSeq,
		};
	}

	// Case 3: the earliest taxon, or the earliest cohort.
	if (rootTaxon && EARLIEST_ALIASES.includes(rootTaxon.toLowerCase())) {
		const valid = datedTaxa.filter((t) => hasDate(t) && !Number.isNaN(dateOf(t))).map((t) => [t, dateOf(t)]);
		// Python's sort is stable and the key is the DATE only, so alignment order survives a tie.
		valid.sort((a, b) => (a[1] < b[1] ? -1 : a[1] > b[1] ? 1 : 0));
		const minDate = valid[0][1];
		const earliest = valid.filter(([, dd]) => Math.abs(dd - minDate) < 1e-4).map(([t]) => t);

		if (datedTaxa.length > 2500 || earliest.length <= 10) {
			const cross = tn93CrossDistanceMatrix(asMap, datedTaxa, earliest, options);
			const m = earliest.length;
			let divergences;
			let desc;
			if (m === 1) {
				divergences = col0(cross, 1);
				desc = `earliest_taxon_${earliest[0]}`;
			} else {
				divergences = new Float64Array(datedTaxa.length);
				const row = new Float64Array(m);
				for (let i = 0; i < datedTaxa.length; i++) {
					for (let j = 0; j < m; j++) row[j] = cross[i * m + j];
					divergences[i] = mean64(row);
				}
				desc = `earliest_cohort_n${m}`;
			}
			return { divergences, root_description: desc, taxa: datedTaxa.slice(), case: 3, root_taxa: earliest, gamma: null, root_sequence: null };
		}
		// The square sub-branch: a DIFFERENT imputation maximum, and float32 arithmetic throughout.
		const nAll = datedTaxa.length;
		const square = tn93DistanceMatrix(asMap, datedTaxa, options);
		const idx = new Map(datedTaxa.map((t, i) => [t, i]));
		const rows = earliest.map((t) => idx.get(t));
		let divergences;
		let desc;
		if (rows.length === 1) {
			divergences = new Float64Array(nAll);
			for (let j = 0; j < nAll; j++) divergences[j] = square[rows[0] * nAll + j];
			desc = `earliest_taxon_${datedTaxa[rows[0]]}`;
		} else {
			divergences = new Float64Array(nAll);
			const colBuf = new Float32Array(rows.length);
			for (let j = 0; j < nAll; j++) {
				for (let k = 0; k < rows.length; k++) colBuf[k] = square[rows[k] * nAll + j];
				// np.mean over a float32 array: a float32 pairwise sum and a float32 divide.
				divergences[j] = Math.fround(numpyPairwiseSum(colBuf, 0, colBuf.length, Math.fround) / colBuf.length);
			}
			desc = `earliest_cohort_n${rows.length}`;
		}
		return { divergences, root_description: desc, taxa: datedTaxa.slice(), case: 3, root_taxa: earliest, gamma: null, root_sequence: null };
	}

	// Case 4: the time-decay weighted consensus (the default).
	const { sequence: decaySeq, gamma: effGamma } = timeDecayConsensusSequence(asMap, dates, datedTaxa, {
		gamma: options.decayGamma ?? null,
		halfLife: options.decayHalfLife ?? null,
	});
	const aug = new Map(asMap);
	aug.set(TIME_DECAY_ROOT_KEY, decaySeq);
	const cross = tn93CrossDistanceMatrix(aug, datedTaxa, [TIME_DECAY_ROOT_KEY], options);
	return {
		divergences: col0(cross, 1),
		root_description: `time_decay_consensus_root (γ=${fixed4(effGamma)})`,
		taxa: datedTaxa.slice(),
		case: 4,
		root_taxa: null,
		gamma: effGamma,
		root_sequence: decaySeq,
	};
}
