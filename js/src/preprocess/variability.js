/**
 * WHY THIS FILE EXISTS
 *
 * Ported verbatim from datamonkey3 (main@fac1330) src/lib/services/axomeme/postprocess.js. Function
 * bodies unchanged: `isSiteVariable` and `siteVariability` are lifted with their documentation and
 * not one character of their logic, and the `./tokenizer.js` import resolves to the sibling module
 * here as it did there.
 *
 * ONLY THOSE TWO FUNCTIONS CAME ACROSS, and the split is not arbitrary. They are ALIGNMENT
 * functions: given aligned codon strings, is this site variable? That is preprocessing — it decides
 * what the model is even asked about, and it mirrors the invariable-site rule of
 * `hyphaeon/dataset.py` (dataset.py:718-723), which is why it belongs in a library that mirrors
 * `hyphaeon/*.py`. Everything else in datamonkey3's postprocess.js — `buildPredictions`, the tier
 * gates, z-scores, percentile ranks and `callModes.js` — is RESULT SEMANTICS: what to show a
 * researcher and where to draw a line on a number the model produced. That is a product decision,
 * it changes without the methods changing, and it lives in `hyphaeon-app`'s `runtime/`. Splitting
 * the file is how the ownership rule in src/README.md stays enforceable rather than aspirational.
 *
 * THE REFERENCE IT MIRRORS, in full, so the divergences below can be checked without opening it:
 *
 *     is_aa_invariable = np.zeros(L, dtype=bool)
 *     for site in range(L):
 *         aa_col = a_all[site, :, 0]
 *         valid_aa = aa_col[aa_col < 20]
 *         if len(np.unique(valid_aa)) <= 1:
 *             is_aa_invariable[site] = True
 *
 * TWO DIVERGENCES AGAINST v1.0.0 dataset.py. Both are recorded rather than resolved, for the reason
 * given in tokenizer.js: this port was verified against real data and dataset.py has not been, so an
 * inspection-time "fix" would trade a measured behaviour for a guess. The fixture harness decides.
 *
 *   1. THE SERINE RULE IS NOT IN dataset.py. `isSiteVariable` calls a site variable when every
 *      sequence codes serine but reaches it through both codon families (TCN and AGY) — the
 *      selection-relevant case documented below. dataset.py compares amino-acid TOKENS only, so
 *      serine is serine and such a site is marked invariable, which zeroes it. This is the more
 *      consequential of the two: it changes which sites are scored at all, always in the direction
 *      of this port scoring MORE sites than the Python.
 *   2. STOP CODONS. dataset.py's filter is `aa_col < 20`, and `get_aa_token` maps a stop to exactly
 *      20, so a stop contributes nothing to the comparison. This port keeps `aaToken(c) <= 20`, so a
 *      stop is a real observation (`'*'` is a token in `AA_LIST`) and a site mixing Met with a stop
 *      is variable here and invariable there. In-frame stops are not rare in real submissions —
 *      dataset.py:715-716 counts and reports them — so this is reachable, not theoretical.
 *
 * The empty case agrees by both routes and is worth stating so it is not mistaken for a third
 * divergence: a site where nothing is usable has zero distinct residues, `len(unique) <= 1` is true
 * in Python and `aas.size === 0` returns false here, and both call it invariable.
 */

import { GENETIC_CODE, aaToken } from './tokenizer.js';

/**
 * Is this site variable, in the reference's sense?
 *
 * Two conditions, and the second is easy to miss: a site is also variable if every sequence codes
 * SERINE but reaches it through both codon families (TCN and AGY). Serine is the one residue whose
 * codons occupy two disjoint blocks of the genetic code, so a TCN<->AGY switch requires multiple
 * substitutions while remaining synonymous — selection-relevant despite the amino acid never
 * changing. Dropping this condition silently marks those sites invariant and zeroes them.
 *
 * @param {string[]} codons observed codons at this site, gaps/ambiguity already excluded
 * @returns {boolean}
 */
export function isSiteVariable(codons) {
	if (!codons || codons.length === 0) return false;
	const aas = new Set();
	for (const c of codons) {
		const aa = GENETIC_CODE.get(c.toUpperCase());
		if (aa && aa !== '?') aas.add(aa);
	}
	if (aas.size === 0) return false;
	if (aas.size > 1) return true;
	if (aas.size === 1 && aas.has('S')) {
		const upper = codons.map((c) => c.toUpperCase());
		const hasTCN = upper.some((c) => c === 'TCA' || c === 'TCC' || c === 'TCG' || c === 'TCT');
		const hasAGY = upper.some((c) => c === 'AGC' || c === 'AGT');
		if (hasTCN && hasAGY) return true;
	}
	return false;
}

/**
 * Variability flags for every site of an alignment, from the aligned sequences.
 *
 * Mirrors the reference's collection loop: only codons that are complete, ungapped and unambiguous
 * are considered, and a sequence shorter than the reference simply contributes nothing at the sites
 * it does not reach.
 *
 * @param {string[]} sequences aligned nucleotide sequences, same frame
 * @param {number} totalCodons
 * @returns {boolean[]}
 */
export function siteVariability(sequences, totalCodons) {
	const flags = new Array(totalCodons).fill(false);
	for (let s = 0; s < totalCodons; s++) {
		const codons = [];
		for (const seq of sequences) {
			const start = s * 3;
			if (start + 3 > seq.length) continue;
			const c = seq.slice(start, start + 3).toUpperCase();
			if (c.includes('-') || c.includes('N') || c.includes('?')) continue;
			// aaToken rejects anything the genetic code cannot translate, which is the same filter the
			// reference applies before collecting a codon.
			if (aaToken(c) > 20) continue;
			codons.push(c);
		}
		flags[s] = isSiteVariable(codons);
	}
	return flags;
}
