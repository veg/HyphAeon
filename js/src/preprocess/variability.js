/**
 * WHY THIS FILE EXISTS
 *
 * Mirrors the invariable-site rule of `hyphaeon/dataset.py:718-723` at veg/HyphAeon 267f5cf:
 *
 *     is_aa_invariable = np.zeros(L, dtype=bool)
 *     for site in range(L):
 *         aa_col = a_all[site, :, 0]
 *         valid_aa = aa_col[aa_col < 20]
 *         if len(np.unique(valid_aa)) <= 1:
 *             is_aa_invariable[site] = True
 *
 * A site is invariable iff at most one distinct amino-acid TOKEN below 20 appears in its column.
 * Stops, gaps and unknowns are all token 20 (`get_aa_token`, dataset.py:55-57) and so are not
 * observations; a column with nothing usable is invariable. `invariableMask` is the exact loop over
 * a [L, N, 1] token array; `isSiteVariable` / `siteVariability` are the same rule phrased over codon
 * strings for callers that have not tokenised yet.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: it does not decide what to do with an invariable site. That is
 * result semantics (cli.py zeroes the LRT) and belongs to the runtime.
 *
 * DIVERGENCE FROM THE DATAMONKEY3 PORT (main@fac1330 src/lib/services/axomeme/postprocess.js
 * `isSiteVariable`/`siteVariability`), which this file replaces:
 *   1. THE SERINE RULE IS GONE. DM3 called a site variable when every sequence coded serine but
 *      through both codon families (TCN and AGY). dataset.py compares amino-acid tokens only, so
 *      such a site is invariable here.
 *   2. STOPS ARE NOT OBSERVATIONS. DM3 kept a stop ('*', token 20 in its 23-letter list) as a real
 *      residue, so Met + stop was variable. dataset.py filters `aa_col < 20` and a stop is 20.
 *   3. AMBIGUITY IS DECIDED BY THE TOKEN, not by scanning for '-', 'N' or '?': anything
 *      `get_aa_token` cannot translate is 20 and drops out, which is the same set plus every other
 *      untranslatable string.
 * Both DM3 rules scored MORE sites than the Python; the fixture replay (`is_aa_invariable` for Smc6
 * and bat_oas1 in fixtures/dataset/load_alignment_and_tree.json) pins the Python.
 */

import { aaToken } from './tokenizer.js';
import { AA_VALID_BELOW } from './modelContract.js';

/**
 * dataset.py:719-723 for one column of amino-acid tokens.
 *
 * @param {ArrayLike<number>} aaColumn amino-acid tokens of every taxon at one site
 * @returns {boolean} true when `len(np.unique(aa_col[aa_col < 20])) <= 1`
 */
export function isAaInvariable(aaColumn) {
	let first = -1;
	for (let i = 0; i < aaColumn.length; i++) {
		const t = aaColumn[i];
		if (t < AA_VALID_BELOW) {
			if (first < 0) first = t;
			else if (t !== first) return false;
		}
	}
	return true;
}

/**
 * The whole mask of dataset.py:718-723 over a [L, N, 1] amino-acid token array (row-major, site
 * major: element `site * N + taxon`).
 *
 * @param {ArrayLike<number>} aTokens L * N amino-acid tokens
 * @param {number} L
 * @param {number} N
 * @returns {Uint8Array} length L, 1 where the site is invariable
 */
export function invariableMask(aTokens, L, N) {
	const out = new Uint8Array(L);
	const col = new Array(N);
	for (let site = 0; site < L; site++) {
		for (let i = 0; i < N; i++) col[i] = aTokens[site * N + i];
		out[site] = isAaInvariable(col) ? 1 : 0;
	}
	return out;
}

/**
 * Is this site variable — the negation of dataset.py's rule, phrased over codon strings.
 *
 * @param {string[]|null|undefined} codons observed codons at this site (any strings; whatever
 *   `get_aa_token` cannot translate is token 20 and ignored, exactly as in the reference)
 * @returns {boolean} true iff more than one distinct amino-acid token below 20 is present
 */
export function isSiteVariable(codons) {
	if (!codons || codons.length === 0) return false;
	return !isAaInvariable(codons.map((c) => aaToken(c)));
}

/**
 * Variability flags for every site of an alignment, from aligned nucleotide strings, mirroring the
 * per-site codon slice of dataset.py:699 (`seq[site*3:(site+1)*3]`, which past a short sequence's
 * end is shorter than 3 characters and tokenises to 20).
 *
 * @param {string[]} sequences aligned nucleotide sequences, same frame
 * @param {number} totalCodons L
 * @returns {boolean[]} true where the site is variable (NOT invariable)
 */
export function siteVariability(sequences, totalCodons) {
	const flags = new Array(totalCodons).fill(false);
	for (let s = 0; s < totalCodons; s++) {
		const codons = [];
		for (const seq of sequences) codons.push(String(seq).slice(s * 3, s * 3 + 3));
		flags[s] = isSiteVariable(codons);
	}
	return flags;
}
