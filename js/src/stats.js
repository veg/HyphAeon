/**
 * WHY THIS FILE EXISTS
 *
 * Mirrors hyphaeon/stats.py (at reconcile/phase-5a), the four shared statistical reductions every CLI command
 * and method module calls: LRT -> p-value under the MEME mixture null (stats.py:16-32), LRT ->
 * p-value under the Self & Liang mixture null (stats.py:34-48), Benjamini-Hochberg q-values
 * (stats.py:50-70) and the Cauchy Combination Test (stats.py:72-84). The last two live in the
 * numeric kernel (src/numeric/bh.js, src/numeric/cauchy.js) because filter.py, epistasis.py and
 * phenotype.py call them too; this module re-exports them under the Python names so a port reading
 * stats.py finds every function of that file here.
 *
 * WHAT IT MIRRORS, LINE FOR LINE.
 *
 *   pvals_from_lrt_meme (stats.py:16-32):
 *       pvals = np.full(len(lrts), 2.0 / 3.0, dtype=np.float64)
 *       pos = lrts > 0.0
 *       pvals[pos] = (2.0 / 3.0) * (0.45 * chi2.sf(lrts[pos], df=1) + 0.55 * chi2.sf(lrts[pos], df=2))
 *
 *     The MEME null is 1/3 * delta(0) + 2/3 * (0.45 chi2_1 + 0.55 chi2_2). The atom at zero is
 *     handled by the default fill: every LRT that is NOT strictly positive — zero, negative (the
 *     model clamps at zero but the fixture pins the negative branch), and NaN (a NaN comparison is
 *     False in numpy and in JS alike) — keeps p = 2/3 exactly. Only lrt > 0 evaluates the
 *     continuous tail, in the same operation order as the numpy expression.
 *
 *   pvals_from_lrt_self_liang (stats.py:34-48):
 *       pvals = np.ones(len(lrts)); pos = lrts > 0.0; pvals[pos] = 0.5 * chi2.sf(lrts[pos], df=1)
 *
 *     Null 0.5 * delta(0) + 0.5 * chi2_1; LRT <= 0 (and NaN) -> 1.0.
 *
 *   Both return float64 whatever the input dtype: scipy's chi2.sf promotes a float32 array to
 *   float64 and the result array is allocated float64 (fixture case `float32_model_like`, whose
 *   notes say "p computed in float64"). A Float32Array input is therefore read as the exact
 *   float64 value of each float32 element and the output is a Float64Array.
 *
 * PRECISION NOTE FOR THE RUNTIME (cli.py cmd_meme:99-100, documented in scripts/parity.py:34-37):
 *   the reference CLI casts BOTH statistics to float32 before writing them —
 *       pvals = pvals_from_lrt_meme(lrts).astype(np.float32)
 *       qvals = benjamini_hochberg(pvals).astype(np.float32)
 *   — and BH runs ON THE FLOAT32 p (float32 arithmetic inside, see bh.js). This library returns
 *   float64 from `pvalsFromLrtMeme` as stats.py does; the app's runtime performs the rounding.
 *   `memeSitePq` below is that exact two-line cast sequence, provided so the runtime does not have
 *   to rediscover that the cast happens BEFORE BH rather than after. cmd_busted (cli.py:437) does
 *   not cast: Self-Liang p-values stay float64 there.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: no argument validation beyond what numpy does (a non-numeric
 * element becomes NaN and takes the point-mass branch, as `np.asarray` of garbage would not — this
 * is not reachable from the runtime, which hands over Float32Array LRTs). No float32 output.
 *
 * Tolerance class (fixtures/stats/*.json): 1e-9. The only numerical primitive is chi2Sf from
 * src/numeric/special.js, which is the regularized upper incomplete gamma Q(df/2, x/2).
 */

import { chi2Sf } from './numeric/special.js';
import { benjaminiHochberg } from './numeric/bh.js';
import { cauchyCombination } from './numeric/cauchy.js';

export { benjaminiHochberg };
/** Cauchy Combination Test — hyphaeon/stats.py:72-84 `cauchy_combination_p`; see numeric/cauchy.js. */
export { cauchyCombination as cauchyCombinationP };

const TWO_THIRDS = 2.0 / 3.0;

/**
 * MEME asymptotic mixture p-values — hyphaeon/stats.py:16-32 `pvals_from_lrt_meme`.
 *
 * @param {ArrayLike<number>} lrts site LRTs (Float32Array as the model emits them, or any array)
 * @returns {Float64Array} p-values; 2/3 wherever lrt is not > 0
 */
export function pvalsFromLrtMeme(lrts) {
	const n = lrts.length;
	const pvals = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		const x = lrts[i];
		if (x > 0.0) {
			pvals[i] = TWO_THIRDS * (0.45 * chi2Sf(x, 1) + 0.55 * chi2Sf(x, 2));
		} else {
			pvals[i] = TWO_THIRDS;
		}
	}
	return pvals;
}

/**
 * Self & Liang (1987) mixture p-values — hyphaeon/stats.py:34-48 `pvals_from_lrt_self_liang`.
 *
 * @param {ArrayLike<number>} lrts
 * @returns {Float64Array} p-values; 1.0 wherever lrt is not > 0
 */
export function pvalsFromLrtSelfLiang(lrts) {
	const n = lrts.length;
	const pvals = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		const x = lrts[i];
		pvals[i] = x > 0.0 ? 0.5 * chi2Sf(x, 1) : 1.0;
	}
	return pvals;
}

/**
 * The cmd_meme cast sequence (cli.py:102-103): p = float32(pvals_from_lrt_meme(lrt)), then
 * q = float32(benjamini_hochberg(p)) with BH evaluated on the float32 p (float32 arithmetic, per
 * numpy 2 promotion — see numeric/bh.js). This is what `hyphaeon meme` writes to JSON and CSV.
 *
 * @param {ArrayLike<number>} lrts
 * @returns {{pvals: Float32Array, qvals: Float32Array}}
 */
export function memeSitePq(lrts) {
	const pvals = Float32Array.from(pvalsFromLrtMeme(lrts));
	const qvals = Float32Array.from(benjaminiHochberg(pvals));
	return { pvals, qvals };
}
