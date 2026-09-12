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
 * No I/O, no method semantics: a scalar root finder over a caller's callback.
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
