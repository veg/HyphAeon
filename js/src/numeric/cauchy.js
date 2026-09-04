/**
 * WHY THIS FILE EXISTS
 *
 * Mirrors `cauchy_combination_p` in hyphaeon/stats.py:72-84 (at 267f5cf), line for line:
 *
 *     if len(pvals) == 0: return 1.0
 *     p_clipped = np.clip(pvals, 1e-15, 1.0 - 1e-15)
 *     t = np.mean(np.tan((0.5 - p_clipped) * np.pi))
 *     p_cct = 0.5 - (np.arctan(t) / np.pi)
 *     return float(np.clip(p_cct, 1e-15, 1.0))
 *
 * The Cauchy Combination Test (Liu & Xie 2020) is the gene-level omnibus used by cmd_busted and
 * cmd_meme; it is in the kernel because it is a pure reduction over p-values with no method
 * semantics of its own.
 *
 * DTYPE FOLLOWS THE INPUT, as in bh.js. With a float32 array every step above stays float32 under
 * numpy 2 promotion (Python float scalars are "weak"): the clip bounds round to float32(1e-15) and
 * to exactly 1.0, `tan`, `mean`, `arctan` and the divisions all run in float32. The fixture case
 * `float32_input` differs from a float64 replay by 9e-8 and matches a float32 replay (measured with
 * the reference Python), so a Float32Array selects Math.fround at every step. Two details of that
 * path matter at the 1e-9 class:
 *   - `np.mean` is a pairwise sum (numpy `pairwise_sum`: 8-way unrolled blocks of ≤ 128, then
 *     recursive halving), with a float32 accumulator, then a float32 divide; `mean` below is that
 *     algorithm with the rounding function injected, so float64 inputs also get numpy's summation
 *     order.
 *   - float32 `tan` in numpy is the platform / SIMD `tanf`; here it is Math.tan on the float32
 *     value rounded with Math.fround, which is within 1 float32 ulp of numpy and agreed exactly on
 *     the fixture.
 */

import { numpyPairwiseSum as pairwiseSum } from './reduce.js';

/**
 * Cauchy combination p-value — hyphaeon/stats.py:72-84. Empty input gives 1.0.
 *
 * @param {ArrayLike<number>} pvals a Float32Array selects float32 arithmetic (see header)
 * @returns {number}
 */
export function cauchyCombination(pvals) {
	const n = pvals.length;
	if (n === 0) return 1.0;
	const f32 = pvals instanceof Float32Array;
	const round = f32 ? Math.fround : (/** @type {number} */ v) => v;
	const lower = round(1e-15);
	const upper = round(1.0 - 1e-15);
	const pi = round(Math.PI);
	const tans = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		let p = pvals[i];
		if (p < lower) p = lower;
		else if (p > upper) p = upper;
		tans[i] = round(Math.tan(round(round(0.5 - p) * pi)));
	}
	const t = round(pairwiseSum(tans, 0, n, round) / n);
	let pCct = round(0.5 - round(round(Math.atan(t)) / pi));
	if (pCct < lower) pCct = lower;
	else if (pCct > 1.0) pCct = 1.0;
	return pCct;
}
