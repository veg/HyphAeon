/**
 * WHY THIS FILE EXISTS
 *
 * The two synthetic root anchors the tree-free dating path builds when the reader names no root
 * sequence of their own: `generate_consensus_sequence` (hyphaeon/dating.py:225-245) and
 * `generate_time_decay_consensus_sequence` (dating.py:248-310). They are pure functions of
 * (sequences, taxa[, dates, γ]) returning a string, they mirror named reference functions line for
 * line, and they carry two quirks that only a fixture can pin — so they are library code, and they
 * live beside the other dataset-shaped string arithmetic rather than in dating.js, which consumes
 * them.
 *
 * QUIRKS REPLICATED ON PURPOSE (each has a fixture case; see fixtures/manifest.json DATING Q5/Q6):
 *
 *   - The skip set is EXACTLY `-`, `?`, `N` (dating.py:239, 302). `*` and every IUPAC ambiguity
 *     code are counted, so `R` can win a column outright and a stop-codon `*` can become the root's
 *     character. The reference does not treat them as missing anywhere in these two functions.
 *   - Ties go to the FIRST character encountered in the caller's taxon order. Python's `max` over
 *     `dict.items()` returns the first maximum in insertion order, and insertion order is the order
 *     characters were first seen while iterating `taxa`. The port iterates in the same order and
 *     compares with a strict `>`, which is the same rule; a `Map` gives the same insertion order a
 *     CPython dict does.
 *   - The alignment length is read from the FIRST taxon of the list handed in (dating.py:233, 316),
 *     not checked across the rest. A ragged alignment silently truncates or reads past the end of
 *     the shorter sequences — in Python that is an IndexError; here `s[pos]` is `undefined` and
 *     would be counted as its own character, so this port raises instead of inventing a column.
 *     That is a refusal, not a different answer: the reference cannot return anything there either.
 *   - `--decay-half-life` exists in `generate_time_decay_consensus_sequence` and has NO CLI flag,
 *     so the half-life arm is unreachable from `hyphaeon dating`. It is ported anyway, because the
 *     function is the contract, not the CLI.
 *
 * THE γ LADDER (dating.py:278-287) is the non-obvious constant: half-life wins if positive, else an
 * explicit γ if positive, else `0.05` when `0.05·Δt ≥ 1` (i.e. Δt ≥ 20 time units) and `2/Δt`
 * otherwise, and `0.0` when the span is zero. MEASURED on examples/H1N1_2009_pandemic.fasta, whose
 * span is 0.666 years: the second arm fires and γ = 3.0030, which is the value the reference's
 * `root_description` prints.
 *
 * No I/O, no model, no messages for a reader: the caller decides what to call the root.
 */

import { numpyPairwiseSum } from '../numeric/reduce.js';

/** The three characters both builders abstain on (dating.py:239, 302). */
export const CONSENSUS_SKIP = Object.freeze(['-', '?', 'N']);
/** The column answer when every taxon abstains (dating.py:242, 306). */
export const CONSENSUS_EMPTY_CHAR = '-';

const identity = (/** @type {number} */ v) => v;

/** `seq_dict[t]` for a Map or a plain object, with the reference's KeyError made explicit. */
function sequenceReader(sequences) {
	const get = sequences instanceof Map ? (/** @type {string} */ t) => sequences.get(t) : (/** @type {string} */ t) => sequences[t];
	return (/** @type {string} */ t) => {
		const s = get(t);
		if (typeof s !== 'string') throw new Error(`consensus: no sequence for taxon '${t}'`);
		return s;
	};
}

/**
 * `generate_consensus_sequence(seq_dict, taxa)`, dating.py:225-245: the majority-rule nucleotide
 * consensus over `taxa`, skipping only `-`, `?` and `N`, ties to the first character seen.
 *
 * @param {Map<string, string>|Record<string, string>} sequences taxon -> aligned sequence
 * @param {string[]} [taxa] defaults to every key, in insertion order (the reference's default)
 * @returns {string}
 */
export function consensusSequence(sequences, taxa) {
	const names = taxa ?? (sequences instanceof Map ? [...sequences.keys()] : Object.keys(sequences));
	if (names.length === 0) throw new RangeError('Cannot compute consensus of 0 sequences.');
	const read = sequenceReader(sequences);
	const seqs = names.map(read);
	const seqLen = seqs[0].length;
	const out = new Array(seqLen);
	const counts = new Map();
	for (let pos = 0; pos < seqLen; pos++) {
		counts.clear();
		for (let i = 0; i < seqs.length; i++) {
			const s = seqs[i];
			if (pos >= s.length) throw new RangeError(`consensus: sequence '${names[i]}' is shorter than '${names[0]}'`);
			const ch = s[pos].toUpperCase();
			if (ch === '-' || ch === '?' || ch === 'N') continue;
			counts.set(ch, (counts.get(ch) ?? 0) + 1);
		}
		let best = CONSENSUS_EMPTY_CHAR;
		let bestCount = -1;
		for (const [ch, c] of counts) {
			if (c > bestCount) {
				bestCount = c;
				best = ch;
			}
		}
		out[pos] = best;
	}
	return out.join('');
}

/**
 * `generate_time_decay_consensus_sequence(seq_dict, dates_map, taxa, gamma, half_life)`,
 * dating.py:248-310: the weight-max consensus under `w_i ∝ exp(−γ(t_i − t_min))`.
 *
 * Taxa with no date, or a NaN date, are dropped first (dating.py:268); if NONE survive the
 * reference falls back to the unweighted consensus over the ORIGINAL taxa list with γ = 0.0, which
 * is replicated. The weights are normalised by their `np.sum` (numpy's pairwise order, reproduced
 * here) and fall back to uniform when that sum is not positive.
 *
 * @param {Map<string, string>|Record<string, string>} sequences
 * @param {Map<string, number>|Record<string, number>} dates taxon -> decimal time
 * @param {string[]} [taxa] defaults to every sequence key
 * @param {{gamma?: number|null, halfLife?: number|null}} [options]
 * @returns {{sequence: string, gamma: number}} `gamma` is the reference's `eff_gamma`
 */
export function timeDecayConsensusSequence(sequences, dates, taxa, options = {}) {
	const names = taxa ?? (sequences instanceof Map ? [...sequences.keys()] : Object.keys(sequences));
	const hasDate = dates instanceof Map ? (/** @type {string} */ t) => dates.has(t) : (/** @type {string} */ t) => Object.prototype.hasOwnProperty.call(dates, t);
	const dateOf = dates instanceof Map ? (/** @type {string} */ t) => dates.get(t) : (/** @type {string} */ t) => dates[t];
	const valid = names.filter((t) => hasDate(t) && !Number.isNaN(dateOf(t)));
	if (valid.length === 0) return { sequence: consensusSequence(sequences, names), gamma: 0.0 };

	const times = Float64Array.from(valid, (t) => dateOf(t));
	let tMin = times[0];
	let tMax = times[0];
	for (let i = 1; i < times.length; i++) {
		if (times[i] < tMin) tMin = times[i];
		if (times[i] > tMax) tMax = times[i];
	}
	const deltaT = tMax - tMin;

	const gamma = options.gamma ?? null;
	const halfLife = options.halfLife ?? null;
	let effGamma;
	if (halfLife !== null && halfLife > 0) effGamma = Math.log(2.0) / halfLife;
	else if (gamma !== null && gamma > 0) effGamma = gamma;
	else if (deltaT > 0) effGamma = 0.05 * deltaT >= 1.0 ? 0.05 : 2.0 / deltaT;
	else effGamma = 0.0;

	const weights = new Float64Array(times.length);
	for (let i = 0; i < times.length; i++) weights[i] = Math.exp(-effGamma * (times[i] - tMin));
	const wSum = numpyPairwiseSum(weights, 0, weights.length, identity);
	if (wSum > 0) for (let i = 0; i < weights.length; i++) weights[i] /= wSum;
	else weights.fill(1 / weights.length);

	const read = sequenceReader(sequences);
	const seqs = valid.map(read);
	const seqLen = seqs[0].length;
	const out = new Array(seqLen);
	const charWeights = new Map();
	for (let pos = 0; pos < seqLen; pos++) {
		charWeights.clear();
		for (let i = 0; i < seqs.length; i++) {
			const s = seqs[i];
			if (pos >= s.length) throw new RangeError(`consensus: sequence '${valid[i]}' is shorter than '${valid[0]}'`);
			const ch = s[pos].toUpperCase();
			if (ch === '-' || ch === '?' || ch === 'N') continue;
			charWeights.set(ch, (charWeights.get(ch) ?? 0) + weights[i]);
		}
		let best = CONSENSUS_EMPTY_CHAR;
		let bestW = -1;
		for (const [ch, wv] of charWeights) {
			if (wv > bestW) {
				bestW = wv;
				best = ch;
			}
		}
		out[pos] = best;
	}
	return { sequence: out.join(''), gamma: effGamma };
}
