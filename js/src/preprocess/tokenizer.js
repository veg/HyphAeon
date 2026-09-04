/**
 * WHY THIS FILE EXISTS
 *
 * Mirrors the three lookup tables and the two token functions of `hyphaeon/dataset.py` at
 * veg/HyphAeon 267f5cf:
 *   - `GENETIC_CODE`  dataset.py:25-34   61 sense codons -> 0..60 in TCAG table order, stops -> 64
 *   - `AA_MAP`        dataset.py:36-39   'A'..'Y' alphabetical -> 0..19
 *   - `CODON_TO_AA`   dataset.py:41-50   the standard genetic code, stops as '*'
 *   - `get_codon_token(codon)`  dataset.py:52-53   `GENETIC_CODE.get(codon.upper(), 64)`
 *   - `get_aa_token(codon)`     dataset.py:55-57   `AA_MAP.get(CODON_TO_AA.get(codon.upper(), '-'), 20)`
 *
 * The tables are generated from the TCAG codon list and the one-letter translation string rather
 * than transcribed as 64-entry literals; `js/test/fixtures.test.js` compares the generated tables
 * against the verbatim dump in `fixtures/dataset/tokenizer.json` ("tables" case), so a generator
 * mistake fails there, entry by entry.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: no 'U' -> 'T' replacement. dataset.py does that once, in
 * `parse_alignment_sequences` (lines 80, 106, 138, 145, 157), so `get_codon_token('AUG')` is 64
 * in the reference (pinned by the fixture's `gaps_ambiguity_lowercase_U` case) and is 64 here.
 *
 * DIVERGENCE FROM THE DATAMONKEY3 PORT (main@fac1330 src/lib/services/axomeme/tokenizer.js), which
 * this file replaces: DM3 implemented the AxoMEME 2.0 training vocabulary — all 64 codons in TCAG
 * order (ATG 35, GGG 63, TAA a real token 10), gap 64, unknown 65, and amino-acid sentinels stop 20 /
 * gap 21 / unknown 22, with a "gap anywhere wins" rule. Measured at Phase 0 against the fixture:
 * 54 of 64 codon tokens differed (every codon from TAA onward shifted by the number of stops before
 * it), the 20 residues agreed, and 17 of 21 codon sentinels and 18 of 21 amino-acid sentinels
 * differed. The shipped checkpoint wants dataset.py's tables (the app repository's Phase 0 notes on
 * bat_oas1: Spearman vs `hyphaeon meme` 0.13 -> 0.94 with them), so the DM3 bodies are gone.
 * `CODON_LIST`
 * (the 64 codons in TCAG order) is kept because `GENETIC_CODE` is built from it;
 * `CODON_TO_IDX` and `AA_TO_IDX` (DM3's 64-codon and 23-letter maps) are removed.
 */

import { CODON_ORDER, CODON_UNKNOWN, AA_UNKNOWN, AA_LIST } from './modelContract.js';

/** The 64 codons in TCAG table order (TTT first, GGG last), the row order of dataset.py:25-33. */
export const CODON_LIST = (() => {
	const out = [];
	for (const a of CODON_ORDER)
		for (const b of CODON_ORDER) for (const c of CODON_ORDER) out.push(a + b + c);
	return out;
})();

/** One amino-acid letter per codon of CODON_LIST; '*' for a stop. dataset.py:41-50 by rows. */
const TRANSLATION =
	'FFLLSSSSYY**CC*W' + // TTx TCx TAx TGx
	'LLLLPPPPHHQQRRRR' + // CTx CCx CAx CGx
	'IIIMTTTTNNKKSSRR' + // ATx ACx AAx AGx
	'VVVVAAAADDEEGGGG'; //  GTx GCx GAx GGx

/** dataset.py:41-50 `CODON_TO_AA`: codon -> one-letter amino acid, stops as '*'. */
export const CODON_TO_AA = new Map(CODON_LIST.map((c, i) => [c, TRANSLATION[i]]));

/**
 * dataset.py:25-33 `GENETIC_CODE`: the 61 sense codons numbered consecutively in TCAG order (a
 * stop does not consume a number), and TAA/TAG/TGA -> 64.
 */
export const GENETIC_CODE = (() => {
	const map = new Map();
	let next = 0;
	for (const c of CODON_LIST) {
		if (CODON_TO_AA.get(c) === '*') map.set(c, CODON_UNKNOWN);
		else map.set(c, next++);
	}
	return map;
})();

/** dataset.py:36-39 `AA_MAP`: 'A'..'Y' alphabetical -> 0..19. */
export const AA_MAP = new Map([...AA_LIST].map((a, i) => [a, i]));

/**
 * `get_codon_token`, dataset.py:52-53.
 *
 * @param {string} codon
 * @returns {number} 0..60 for a sense codon; 64 for a stop, a gap, ambiguity, 'U', or any other
 *   string of any length
 */
export function codonToken(codon) {
	const t = GENETIC_CODE.get(String(codon).toUpperCase());
	return t === undefined ? CODON_UNKNOWN : t;
}

/**
 * `get_aa_token`, dataset.py:55-57.
 *
 * @param {string} codon
 * @returns {number} 0..19 for a translated residue; 20 for a stop or anything untranslatable
 */
export function aaToken(codon) {
	const aa = CODON_TO_AA.get(String(codon).toUpperCase());
	if (aa === undefined) return AA_UNKNOWN;
	const t = AA_MAP.get(aa);
	return t === undefined ? AA_UNKNOWN : t;
}

/**
 * Tokenise one sequence into per-codon (codon, aa) tokens for the `len(seq) // 3` whole codons —
 * the per-site loop of dataset.py:696-708 for a single sequence whose own length sets L. A trailing
 * partial codon is dropped, matching `L = raw_len // 3` (dataset.py:666).
 *
 * @param {string} seq nucleotides, gaps allowed
 * @returns {{codons: Uint8Array, aas: Uint8Array}}
 */
export function tokenizeSequence(seq) {
	const s = String(seq);
	const n = Math.floor(s.length / 3);
	const codons = new Uint8Array(n);
	const aas = new Uint8Array(n);
	for (let i = 0; i < n; i++) {
		const c = s.slice(i * 3, i * 3 + 3);
		codons[i] = codonToken(c);
		aas[i] = aaToken(c);
	}
	return { codons, aas };
}
