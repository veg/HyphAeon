/**
 * WHY THIS FILE EXISTS
 *
 * The rank and correlation primitives hyphaeon/evaluation.py takes from scipy.stats:
 * `rankdata(scores, method="average")` (evaluation.py:280), `pearsonr` and `spearmanr`
 * (evaluation.py:270-271), and the Mann–Whitney ROC-AUC that `_roc_auc` builds from those ranks
 * (evaluation.py:275-283). The evaluation port composes these; the kernel only supplies the
 * arithmetic, in float64, to the 1e-9 tolerance class.
 *
 * WHAT IT DELIBERATELY DOES NOT DO. It does not port scipy's input validation or warnings. Where
 * scipy raises (`pearsonr` on n < 2) or warns and returns nan (constant input), these return NaN
 * and leave the decision to the caller — `_correlations` guards both cases itself
 * (evaluation.py:266-267) before calling scipy, so the ported guard runs first and the NaN path is
 * never reached in practice. `rocAuc` returns NaN where `_roc_auc` returns None (no positives or no
 * negatives); JSON.stringify turns that NaN into null, which is what the Python's None serialises
 * to.
 *
 * NUMERICS. `pearson` follows scipy 1.16's `pearsonr`: centre both vectors, normalise each by its
 * Euclidean norm, dot, clip to [−1, 1]. `spearman` is `spearmanr`'s path: average ranks then
 * `np.corrcoef`, i.e. cov / sqrt(var·var) on the ranks; the two formulas differ by rounding only.
 * `rankdata` sorts a stable index permutation so ties take the mean of the ranks they would occupy.
 */

/**
 * Average ranks (1-based), ties share the mean rank — scipy.stats.rankdata(method="average").
 *
 * @param {ArrayLike<number>} values
 * @returns {Float64Array}
 */
export function rankdata(values) {
	const n = values.length;
	const idx = new Array(n);
	for (let i = 0; i < n; i++) idx[i] = i;
	idx.sort((a, b) => values[a] - values[b] || a - b);
	const ranks = new Float64Array(n);
	let i = 0;
	while (i < n) {
		let j = i;
		while (j + 1 < n && values[idx[j + 1]] === values[idx[i]]) j++;
		const avg = (i + j + 2) / 2; // mean of ranks i+1 .. j+1
		for (let k = i; k <= j; k++) ranks[idx[k]] = avg;
		i = j + 1;
	}
	return ranks;
}

/**
 * Pearson correlation — scipy.stats.pearsonr(x, y).statistic. NaN for n < 2 or a constant input.
 *
 * @param {ArrayLike<number>} x
 * @param {ArrayLike<number>} y
 * @returns {number}
 */
export function pearson(x, y) {
	const n = x.length;
	if (n !== y.length) throw new RangeError(`pearson: length mismatch ${n} vs ${y.length}`);
	if (n < 2) return NaN;
	let mx = 0;
	let my = 0;
	for (let i = 0; i < n; i++) {
		mx += x[i];
		my += y[i];
	}
	mx /= n;
	my /= n;
	let nx = 0;
	let ny = 0;
	for (let i = 0; i < n; i++) {
		const dx = x[i] - mx;
		const dy = y[i] - my;
		nx += dx * dx;
		ny += dy * dy;
	}
	nx = Math.sqrt(nx);
	ny = Math.sqrt(ny);
	if (nx === 0 || ny === 0) return NaN;
	let r = 0;
	for (let i = 0; i < n; i++) r += ((x[i] - mx) / nx) * ((y[i] - my) / ny);
	return Math.max(-1, Math.min(1, r));
}

/**
 * Spearman rank correlation — scipy.stats.spearmanr(x, y).statistic: Pearson on average ranks,
 * evaluated as np.corrcoef does (cov / sqrt(var·var)). NaN for n < 2 or a constant input.
 *
 * @param {ArrayLike<number>} x
 * @param {ArrayLike<number>} y
 * @returns {number}
 */
export function spearman(x, y) {
	const n = x.length;
	if (n !== y.length) throw new RangeError(`spearman: length mismatch ${n} vs ${y.length}`);
	if (n < 2) return NaN;
	const rx = rankdata(x);
	const ry = rankdata(y);
	let mx = 0;
	let my = 0;
	for (let i = 0; i < n; i++) {
		mx += rx[i];
		my += ry[i];
	}
	mx /= n;
	my /= n;
	let sxx = 0;
	let syy = 0;
	let sxy = 0;
	for (let i = 0; i < n; i++) {
		const dx = rx[i] - mx;
		const dy = ry[i] - my;
		sxx += dx * dx;
		syy += dy * dy;
		sxy += dx * dy;
	}
	if (sxx === 0 || syy === 0) return NaN;
	const r = sxy / Math.sqrt(sxx * syy);
	return Math.max(-1, Math.min(1, r));
}

/**
 * ROC-AUC by the Mann–Whitney rank-sum, ties counted ½ — hyphaeon/evaluation.py:275-283 `_roc_auc`:
 *
 *     ranks = rankdata(scores, method="average")
 *     auc = (ranks[labels].sum() − P(P+1)/2) / (P·N)
 *
 * @param {ArrayLike<boolean|number>} labels truthy = positive
 * @param {ArrayLike<number>} scores
 * @returns {number} AUC, or NaN when there are no positives or no negatives (Python returns None)
 */
export function rocAuc(labels, scores) {
	const n = labels.length;
	if (n !== scores.length) throw new RangeError(`rocAuc: length mismatch ${n} vs ${scores.length}`);
	let positives = 0;
	for (let i = 0; i < n; i++) if (labels[i]) positives++;
	const negatives = n - positives;
	if (positives === 0 || negatives === 0) return NaN;
	const ranks = rankdata(scores);
	let rankSum = 0;
	for (let i = 0; i < n; i++) if (labels[i]) rankSum += ranks[i];
	return (rankSum - (positives * (positives + 1)) / 2) / (positives * negatives);
}
