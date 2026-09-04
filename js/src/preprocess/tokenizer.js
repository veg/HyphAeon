/**
 * WHY THIS FILE EXISTS
 *
 * Ported verbatim from datamonkey3 (main@fac1330) src/lib/services/axomeme/tokenizer.js. Function
 * bodies unchanged; its `./modelContract.js` import resolves to the sibling module here.
 *
 * It mirrors `get_codon_token` / `get_aa_token` and the three tables they read — `GENETIC_CODE`,
 * `AA_MAP`, `CODON_TO_AA` — in `hyphaeon/dataset.py` (dataset.py:25-57).
 *
 * ============================================================================================
 * UNRESOLVED FOR PHASE 0, AND THE SINGLE MOST CONSEQUENTIAL OPEN QUESTION IN THIS PACKAGE:
 * THIS FILE AND dataset.py DO NOT AGREE ON WHAT A CODON TOKEN IS.
 * ============================================================================================
 *
 * This port implements the 2.0 TRAINING tokenizer: all 64 codons in TCAG order, stops as real
 * tokens, gap 64, unknown 65 (the long note in modelContract.js is the measurement that settled it
 * against the 2.0 inference driver, which disagreed with training on 63 of 64 codons).
 *
 * `hyphaeon/dataset.py` at v1.0.0 does something different again. Its `GENETIC_CODE` is a
 * SIXTY-ONE-entry map of sense codons only, numbered 0..60 in TCAG order WITH THE STOPS SKIPPED, and
 * the three stops are mapped to 64 — the same slot `get_codon_token` returns for anything it does
 * not recognise, gaps included. Checked value by value against dataset.py:25-33:
 *
 *     codon    this file (2.0 training)    dataset.py v1.0.0
 *     TTT      0                           0
 *     ATG      35                          32
 *     AAA      42                          39
 *     GGG      63                          60
 *     TAA      10  (a real codon)          64  (indistinguishable from a gap)
 *
 * The amino-acid stream is much closer: dataset.py's `AA_MAP` is the same alphabetical A..Y = 0..19
 * that `AA_LIST` opens with, so every one of the twenty residues agrees. It diverges only at the
 * sentinels — `get_aa_token` sends stops, gaps, ambiguity and anything untranslatable alike to 20,
 * where this file separates stop 20, gap 21 and unknown 22, and where `model.py`'s
 * `nn.Embedding(23, ...)` has room for all three.
 *
 * WHY NOTHING WAS CHANGED HERE. Both tokenizers cannot be right for one set of weights, and which is
 * right is decided by what the SHIPPED CHECKPOINT was trained with — a fixture question
 * (`scripts/gen_fixtures.py` replayed under `js/test`), not something to settle by reading two files
 * and preferring one. Editing this file on inspection would destroy a port that was verified against
 * 270 real trees and replace it with an unverified guess. So it stays, loudly, until a fixture says
 * otherwise; the app's runtime must not ship a scoring path on the losing vocabulary, because
 * neither tokenizer FAILS on the other's input — they just quietly mean different codons.
 */

/**
 * tokenizer.js — codon and amino-acid tokenisation for AxoMEME 2.0.
 *
 * THIS IMPLEMENTS THE TRAINING TOKENIZER (train_transformer_selection.py:74-97), NOT THE ONE IN THE
 * HANDOFF'S INFERENCE DRIVER. That is a deliberate, measured choice, and it is the single most
 * consequential decision in this file — see the long note in modelContract.js. In short:
 * predict_regression_nexus.py defines its own 60-codon ALPHABETICAL vocabulary at lines 49-55, then
 * redefines `get_codon_token` without redefining `CODON_TO_IDX`, so 63 of 64 codons come out with a
 * different token at inference than the model was trained on, and TTA plus the three stop codons
 * vanish into "unknown". A model trained on TCAG-64 must be served TCAG-64.
 *
 * If a future handoff fixes the driver, this file does not change — it already matches training.
 * If someone "aligns" this file to the driver, the model silently degrades and nothing fails.
 * src/test/axomeme-tokenizer.test.js pins both directions.
 */

import {
	CODON_ORDER,
	CODON_GAP,
	CODON_UNKNOWN,
	AA_LIST,
	AA_GAP,
	AA_UNKNOWN
} from './modelContract.js';

/**
 * The 64 codons in TCAG order — the standard genetic-code table order, so TTT is 0 and GGG is 63.
 * Built rather than transcribed: a 64-entry literal is a transcription error waiting to happen, and
 * the generator IS the reference's definition.
 */
export const CODON_LIST = (() => {
	const out = [];
	for (const a of CODON_ORDER)
		for (const b of CODON_ORDER) for (const c of CODON_ORDER) out.push(a + b + c);
	return out;
})();

/** codon string -> token, for the 64 sense+stop codons. */
export const CODON_TO_IDX = new Map(CODON_LIST.map((c, i) => [c, i]));

/**
 * The standard genetic code, with stops as '*' to match AA_LIST.
 *
 * Derived from the codon table rather than transcribed, for the same reason as CODON_LIST. The four
 * blocks below are the standard code's structure; every codon is covered, so there is no default.
 *
 * ONE THING WORTH FLAGGING UPSTREAM: this is the UNIVERSAL code, hard-coded. DM3 lets a user pick a
 * genetic code (mitochondrial, mycoplasma, several others) and HyPhy honours that choice, so an
 * AxoMEME prediction on a non-universal alignment translates codons the user did not ask for. That
 * is a model-side limitation, not something this file can fix — recorded here so it is not
 * rediscovered as a mystery.
 */
export const GENETIC_CODE = (() => {
	// TCAG-ordered amino acids, one per codon, in the same order CODON_LIST generates.
	const table =
		'FFLLSSSSYY**CC*W' + // TTx TCx TAx TGx
		'LLLLPPPPHHQQRRRR' + // CTx CCx CAx CGx
		'IIIMTTTTNNKKSSRR' + // ATx ACx AAx AGx
		'VVVVAAAADDEEGGGG'; //  GTx GCx GAx GGx
	const map = new Map();
	CODON_LIST.forEach((c, i) => map.set(c, table[i]));
	return map;
})();

/** amino acid character -> token. */
export const AA_TO_IDX = new Map([...AA_LIST].map((a, i) => [a, i]));

/**
 * Token for a codon string.
 *
 * Order of tests matters and matches the reference exactly: a gap ANYWHERE wins over everything
 * else, so `A-T` is a gap rather than an unknown. `'-' in codon` is a substring test in Python, not
 * an equality test, which is why a partial gap counts.
 *
 * @param {string} codon
 * @returns {number} 0..63 for a real codon, CODON_GAP for a gap, CODON_UNKNOWN otherwise
 */
export function codonToken(codon) {
	const c = String(codon).toUpperCase();
	if (c.includes('-')) return CODON_GAP;
	if (c.length !== 3 || c.includes('N')) return CODON_UNKNOWN;
	const t = CODON_TO_IDX.get(c);
	return t === undefined ? CODON_UNKNOWN : t;
}

/**
 * Token for the amino acid a codon translates to.
 *
 * Note the asymmetry with codonToken, which is in the reference and is not a typo here: this one
 * also rejects '?', codonToken does not. A '?' codon reaches codonToken's length/N test and, being
 * length 3 without an N, falls through to the map lookup and returns CODON_UNKNOWN anyway — same
 * answer by a different route.
 *
 * @param {string} codon
 * @returns {number} 0..20 for a translated residue (20 = stop), AA_GAP or AA_UNKNOWN
 */
export function aaToken(codon) {
	const c = String(codon).toUpperCase();
	if (c.includes('-')) return AA_GAP;
	if (c.length !== 3 || c.includes('N') || c.includes('?')) return AA_UNKNOWN;
	const aa = GENETIC_CODE.get(c);
	if (aa === undefined) return AA_UNKNOWN;
	const t = AA_TO_IDX.get(aa);
	return t === undefined ? AA_UNKNOWN : t;
}

/**
 * Tokenise one sequence into per-codon (codon, aa) token pairs.
 *
 * A trailing partial codon is DROPPED, matching `total_codons = len(ref_seq) // 3`. Sites beyond a
 * sequence's own length are the caller's problem: the reference leaves those at the pad value it
 * pre-filled the tensor with, which is why this returns only what the sequence actually covers.
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
