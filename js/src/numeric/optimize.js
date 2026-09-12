/**
 * WHY THIS FILE EXISTS
 *
 * `scipy.optimize.brentq`, which hyphaeon/dating.py:3023 calls once per taxon to invert the fitted
 * restricted-cubic-spline clock for a sampling date. On the flagship example
 * (examples/korber_env_gp160.fasta) the spline is the SELECTED model, so the per-taxon table every
 * reader acts on is built through this root finder: MEASURED over its 142 taxa, 118 rows are
 * solved here, 12 take the `d ≤ d_kn0` linear arm and 12 more have their bracket REFUSED and fall
 * back to the ancestral linear inverse.
 *
 * PORTED FROM scipy's own `scipy/optimize/Zeros/brentq.c` (scipy 1.16.2), NOT from Numerical
 * Recipes' `zbrent`. The two differ in the stopping rule — scipy tests `|sbis| < delta` with
 * `delta = (xtol + rtol·|xcur|)/2`, NR tests the bisection half-step against `tol1` — and in the
 * acceptance test for a secant/inverse-quadratic step. Matching the reference's stopping point is
 * free here, so it is done.
 *
 * MEASUREMENT BEHIND THE DEFAULTS. `xtol = 2e-12`, `rtol = 4·eps = 8.881784197001252e-16` and
 * `maxiter = 100` are scipy's own defaults, which is what the reference call uses (it passes none
 * of them). MEASURED: this transcription, run over all 142 korber taxa through the reference's own
 * fitted spline, reproduced the reference's branch counts exactly (118 / 12 / 12) and its
 * `predicted_date` values to a worst 2.5e-11 years — 0.8 milliseconds. A tighter claim than that
 * would be theatre: the root is only located to `xtol`.
 *
 * THE THROW IS LOAD-BEARING. scipy raises `ValueError` when `f(a)` and `f(b)` do not straddle zero,
 * and dating.py:3024's bare `except Exception` catches it and silently switches those rows to a
 * different model. A port that returned NaN, or widened the bracket "to help", would change 12 of
 * korber's 142 dates. This throws, and the caller catches, exactly as the Python does.
 *
 * ------------------------------------------------------------------------------------------------
 * `scipy.optimize.minimize_scalar(..., bounds=..., method='bounded')` joined it in phase 4.
 * hyphaeon/dating.py:1539 calls it once per PGLS fit to profile Pagel's lambda by REML, and it is
 * the ONE numeric primitive the two model-based dating estimators needed that this library did not
 * already have. `brentq` cannot stand in: it is a ROOT finder on a bracketing interval, and this is
 * a MINIMISER on a box — different algorithm, different stopping rule, no shared code.
 *
 * PORTED FROM scipy's own `_minimize_scalar_bounded`, scipy/optimize/_optimize.py:2289-2436
 * (scipy 1.16.2, the version fixtures/manifest.json records), line for line: golden section with
 * successive parabolic interpolation, `xatol = 1e-5`, `maxiter = 500`.
 *
 * WHY A FAITHFUL TRANSCRIPTION RATHER THAN "a bounded minimiser". `xatol = 1e-5` is loose enough
 * that the reference's answer is not the objective's argmin but the particular point this
 * algorithm's bracket lands on, and MEASURED on the acceptance case (korber, tn93 divergences) the
 * REML profile is flat to ~1e-9 across a bracket 1e-5 wide: the objective at the reference's
 * lambda* = 0.6984254721729402 and at the grid argmin differ in the tenth digit. Substituting plain
 * golden section, or converging harder, therefore returns a DIFFERENT lambda inside the reference's
 * own tolerance — and dating.py:1353 turns lambda straight into the covariance, so the difference
 * propagates to t_MRCA at roughly 0.37 years per 1e-3 of lambda. Ninety lines of transcription buy
 * lambda to ~1e-8 instead of ~1e-5, which is why they are here.
 *
 * MEASURED at this commit, on both acceptance cases (korber under latent and under tn93 divergences,
 * n = 141): this transcription takes scipy's OWN function-call count, 12 evaluations, in both, and
 * lands on lambda* = 0.8591318193673835 against scipy's 0.8591318190927182 (2.75e-10) and
 * 0.6984254593646657 against 0.6984254721729402 (1.28e-8). The residual is NOT the minimiser's: on
 * fixtures/numeric/fminbound.json, where both sides evaluate a closed-form objective and therefore
 * see bit-identical function values, the two agree exactly. It is the REML objective itself, whose
 * float64 reassociation differs from numpy's at ~1e-13 relative — and near the optimum the profile
 * is flat enough (obj(scipy's lambda) = -1408.418707102613 against obj(this lambda) =
 * -1408.4187071025901 on the tn93 case) that 1e-13 in the objective is 1e-8 in the argument. A
 * tighter claim would be a claim about numpy's summation order, not about this file.
 *
 * Two transcription traps, both replicated: `sqrt_eps` is `sqrt(2.2e-16)`, a HARD-CODED constant in
 * scipy that is NOT `sqrt(Number.EPSILON)` (2.2e-16 vs 2.220446049250313e-16 — they differ in the
 * 4th digit of the square root); and `si = np.sign(rat) + (rat == 0)` makes the step direction +1
 * when `rat` is exactly zero, where a bare `Math.sign` would give 0 and stall the search.
 *
 * No I/O, no method semantics: a scalar root finder and a scalar minimiser over a caller's callback.
 */

/** scipy's default `rtol` for the bracketing solvers: 4 × DBL_EPSILON. */
export const BRENTQ_RTOL = 8.881784197001252e-16;
/** scipy's default `xtol`. */
export const BRENTQ_XTOL = 2e-12;
/** scipy's default `maxiter`. */
export const BRENTQ_MAXITER = 100;

/** C's `signbit`, which is what brentq.c compares (so −0 and +0 differ; both return early anyway). */
function signbit(v) {
	return v < 0 || Object.is(v, -0);
}

/**
 * Brent's method on a bracketing interval — `scipy.optimize.brentq(f, a, b, xtol, rtol, maxiter)`.
 *
 * @param {(x: number) => number} f
 * @param {number} a bracket lower end
 * @param {number} b bracket upper end
 * @param {{xtol?: number, rtol?: number, maxIter?: number}} [options] scipy's defaults
 * @returns {{x: number, iterations: number, converged: boolean, functionCalls: number}}
 * @throws {RangeError} when f(a) and f(b) have the same sign (scipy raises ValueError)
 */
export function brentq(f, a, b, options = {}) {
	const xtol = options.xtol ?? BRENTQ_XTOL;
	const rtol = options.rtol ?? BRENTQ_RTOL;
	const maxIter = options.maxIter ?? BRENTQ_MAXITER;

	let xpre = a;
	let xcur = b;
	let xblk = 0;
	let fblk = 0;
	let spre = 0;
	let scur = 0;
	let calls = 2;
	let fpre = f(xpre);
	let fcur = f(xcur);
	if (fpre === 0) return { x: xpre, iterations: 0, converged: true, functionCalls: calls };
	if (fcur === 0) return { x: xcur, iterations: 0, converged: true, functionCalls: calls };
	if (signbit(fpre) === signbit(fcur)) {
		throw new RangeError(`brentq: f(a) and f(b) must have different signs (f(${a}) = ${fpre}, f(${b}) = ${fcur})`);
	}

	let iterations = 0;
	for (let i = 0; i < maxIter; i++) {
		iterations++;
		if (fpre !== 0 && fcur !== 0 && signbit(fpre) !== signbit(fcur)) {
			xblk = xpre;
			fblk = fpre;
			spre = xcur - xpre;
			scur = spre;
		}
		if (Math.abs(fblk) < Math.abs(fcur)) {
			xpre = xcur;
			xcur = xblk;
			xblk = xpre;
			fpre = fcur;
			fcur = fblk;
			fblk = fpre;
		}
		const delta = (xtol + rtol * Math.abs(xcur)) / 2;
		const sbis = (xblk - xcur) / 2;
		if (fcur === 0 || Math.abs(sbis) < delta) {
			return { x: xcur, iterations, converged: true, functionCalls: calls };
		}
		if (Math.abs(spre) > delta && Math.abs(fcur) < Math.abs(fpre)) {
			let stry;
			if (xpre === xblk) {
				// interpolate (secant)
				stry = (-fcur * (xcur - xpre)) / (fcur - fpre);
			} else {
				// extrapolate (inverse quadratic)
				const dpre = (fpre - fcur) / (xpre - xcur);
				const dblk = (fblk - fcur) / (xblk - xcur);
				stry = (-fcur * (fblk * dblk - fpre * dpre)) / (dblk * dpre * (fblk - fpre));
			}
			if (2 * Math.abs(stry) < Math.min(Math.abs(spre), 3 * Math.abs(sbis) - delta)) {
				spre = scur;
				scur = stry;
			} else {
				spre = sbis;
				scur = sbis;
			}
		} else {
			spre = sbis;
			scur = sbis;
		}
		xpre = xcur;
		fpre = fcur;
		if (Math.abs(scur) > delta) xcur += scur;
		else xcur += sbis > 0 ? delta : -delta;
		fcur = f(xcur);
		calls++;
	}
	// scipy sets CONVERR and still returns xcur; the caller decides what that is worth.
	return { x: xcur, iterations, converged: false, functionCalls: calls };
}

// =================================================================================================
// bounded scalar minimisation
// =================================================================================================

/** scipy's default `xatol` for `method='bounded'`. */
export const FMINBOUND_XATOL = 1e-5;
/** scipy's default `maxiter`, which is also its `maxfun`: the loop counts FUNCTION CALLS. */
export const FMINBOUND_MAXITER = 500;
/** `sqrt(2.2e-16)`, scipy's own hard-coded constant — NOT `sqrt(Number.EPSILON)`. */
const SQRT_EPS = Math.sqrt(2.2e-16);
/** `0.5 * (3 - sqrt(5))`, the golden-section fraction. */
const GOLDEN_MEAN = 0.5 * (3.0 - Math.sqrt(5.0));

/** `np.sign(v) + (v == 0)`: +1 at zero, where `Math.sign` alone would give 0 and stall the step. */
function signOrPositive(v) {
	return v > 0 ? 1 : v < 0 ? -1 : 1;
}

/**
 * Bounded scalar minimisation — `scipy.optimize.minimize_scalar(f, bounds=[a, b], method='bounded')`,
 * `scipy/optimize/_optimize.py:2289-2436`.
 *
 * Golden section with successive parabolic interpolation. `status`/`success` mirror scipy's flags:
 * 0 solution found, 1 the function-call budget ran out, 2 a NaN was seen.
 *
 * @param {(x: number) => number} f
 * @param {number} a lower bound
 * @param {number} b upper bound
 * @param {{xatol?: number, maxIter?: number}} [options] scipy's defaults
 * @returns {{x: number, fun: number, nfev: number, status: number, success: boolean}}
 * @throws {RangeError} when the bounds are not finite or are the wrong way round (scipy raises ValueError)
 */
export function fminbound(f, a, b, options = {}) {
	const xatol = options.xatol ?? FMINBOUND_XATOL;
	const maxfun = options.maxIter ?? FMINBOUND_MAXITER;
	if (!Number.isFinite(a) || !Number.isFinite(b)) throw new RangeError('fminbound: bounds must be finite scalars.');
	if (a > b) throw new RangeError('fminbound: the lower bound exceeds the upper bound.');

	let flag = 0;
	let lo = a;
	let hi = b;
	let fulc = lo + GOLDEN_MEAN * (hi - lo);
	let nfc = fulc;
	let xf = fulc;
	let rat = 0.0;
	let e = 0.0;
	let x = xf;
	let fx = f(x);
	let num = 1;
	let fu = Infinity;

	let ffulc = fx;
	let fnfc = fx;
	let xm = 0.5 * (lo + hi);
	let tol1 = SQRT_EPS * Math.abs(xf) + xatol / 3.0;
	let tol2 = 2.0 * tol1;

	while (Math.abs(xf - xm) > tol2 - 0.5 * (hi - lo)) {
		let golden = 1;
		if (Math.abs(e) > tol1) {
			golden = 0;
			let r = (xf - nfc) * (fx - ffulc);
			let q = (xf - fulc) * (fx - fnfc);
			let p = (xf - fulc) * q - (xf - nfc) * r;
			q = 2.0 * (q - r);
			if (q > 0.0) p = -p;
			q = Math.abs(q);
			r = e;
			e = rat;

			if (Math.abs(p) < Math.abs(0.5 * q * r) && p > q * (lo - xf) && p < q * (hi - xf)) {
				rat = (p + 0.0) / q;
				x = xf + rat;
				if (x - lo < tol2 || hi - x < tol2) {
					rat = tol1 * signOrPositive(xm - xf);
				}
			} else {
				golden = 1;
			}
		}

		if (golden) {
			e = xf >= xm ? lo - xf : hi - xf;
			rat = GOLDEN_MEAN * e;
		}

		x = xf + signOrPositive(rat) * Math.max(Math.abs(rat), tol1);
		fu = f(x);
		num += 1;

		if (fu <= fx) {
			if (x >= xf) lo = xf;
			else hi = xf;
			fulc = nfc;
			ffulc = fnfc;
			nfc = xf;
			fnfc = fx;
			xf = x;
			fx = fu;
		} else {
			if (x < xf) lo = x;
			else hi = x;
			if (fu <= fnfc || nfc === xf) {
				fulc = nfc;
				ffulc = fnfc;
				nfc = x;
				fnfc = fu;
			} else if (fu <= ffulc || fulc === xf || fulc === nfc) {
				fulc = x;
				ffulc = fu;
			}
		}

		xm = 0.5 * (lo + hi);
		tol1 = SQRT_EPS * Math.abs(xf) + xatol / 3.0;
		tol2 = 2.0 * tol1;

		if (num >= maxfun) {
			flag = 1;
			break;
		}
	}

	if (Number.isNaN(xf) || Number.isNaN(fx) || Number.isNaN(fu)) flag = 2;
	return { x: xf, fun: fx, nfev: num, status: flag, success: flag === 0 };
}
