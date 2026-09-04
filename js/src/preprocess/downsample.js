/**
 * WHY THIS FILE EXISTS
 *
 * Mirrors the two taxon-reduction steps of `hyphaeon/dataset.py` at veg/HyphAeon 267f5cf:
 *   - `prune_identical_sequences(seq_dict, taxa)` (dataset.py:420-441): byte-identical duplicates
 *     collapse onto the FIRST taxon (in the given order) carrying that sequence; returns the unique
 *     taxa in first-seen order, `dup_map` representative -> [duplicates], and the count pruned.
 *     Byte-identical only: 'atgaaa' != 'ATGAAA' at this level (parsing upper-cased earlier) and a
 *     gapped variant is distinct.
 *   - `downsample_taxa_faith_pd(dist_mat, taxa, max_species)` (dataset.py:396-418): farthest-point
 *     traversal seeded with the most distant PAIR — `np.unravel_index(np.argmax(dist_mat))`, the
 *     first maximum in row-major order — then repeatedly the argmax of the running min-distance
 *     vector (first index on ties). Returns `(sub_dist_mat, selected_taxa)` in SELECTION order, or
 *     the inputs unchanged when `max_species >= n` or `max_species <= 0`.
 *   - the stride pre-selection of `load_alignment_and_tree` (dataset.py:670-674): before the
 *     distance matrix is built, `taxa = taxa[::max(1, n // (2 * max_species))][:2 * max_species]`
 *     — for fewer than 4 * max_species taxa this simply keeps the FIRST 2 * max_species in tree
 *     order (pinned by the camelid case of fixtures/dataset/downsample_taxa_faith_pd.json).
 *
 * QUIRKS REPLICATED ON PURPOSE:
 *   - `max_species == 1` returns TWO taxa: the seed pair is placed before the loop
 *     `for _ in range(2, max_species)` runs (dataset.py:405-411).
 *   - An all-zero distance matrix (or one whose remaining entries are all zero) re-selects index 0
 *     — selected taxa can REPEAT, because a selected index has min-distance 0 and argmax of all
 *     zeros is 0. Not deduplicated; the reference feeds the duplicates to the model.
 *   - `max_species == 0` makes the stride expression divide by zero (ZeroDivisionError in Python);
 *     a negative `max_species` slices with a negative stop (Python semantics reproduced).
 *
 * DIVERGENCE FROM THE DATAMONKEY3 PORT (main@fac1330 src/lib/services/axomeme/patristic.js
 * `maxPdSelect`): DM3 seeded at index 0 (the AxoMEME 2.0 driver) and had no stride pre-selection
 * and no duplicate pruning. Not identical, so replaced.
 */

/** `seq_dict[t]` on a Map or a plain object; KeyError becomes a thrown Error. */
function seqOf(seqDict, name) {
	const v =
		seqDict instanceof Map
			? seqDict.get(name)
			: Object.prototype.hasOwnProperty.call(seqDict, name)
				? seqDict[name]
				: undefined;
	if (v === undefined) throw new Error(`prune_identical_sequences: KeyError: ${JSON.stringify(name)}`);
	return String(v);
}

/**
 * `prune_identical_sequences`, dataset.py:420-441.
 *
 * @param {Map<string, string>|Record<string, string>} seqDict
 * @param {string[]} taxa
 * @returns {{uniqueTaxa: string[], dupMap: Map<string, string[]>, numPruned: number}}
 */
export function pruneIdenticalSequences(seqDict, taxa) {
	const seenSeqs = new Map();
	const uniqueTaxa = [];
	const dupMap = new Map();
	for (const t of taxa) {
		const seq = seqOf(seqDict, t);
		if (!seenSeqs.has(seq)) {
			seenSeqs.set(seq, t);
			uniqueTaxa.push(t);
			dupMap.set(t, []);
		} else {
			dupMap.get(seenSeqs.get(seq)).push(t);
		}
	}
	return { uniqueTaxa, dupMap, numPruned: taxa.length - uniqueTaxa.length };
}

/** `np.argmax` over a flat array: first index of the maximum. */
function argmax(a) {
	let best = 0;
	for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
	return best;
}

/**
 * `downsample_taxa_faith_pd`, dataset.py:396-418.
 *
 * @param {Float32Array} distMat row-major n x n, float32 (the reference's dtype; comparisons and
 *   minima are taken on float32 values)
 * @param {string[]} taxa
 * @param {number} maxSpecies
 * @returns {{distMat: Float32Array, taxa: string[], selectedIndices: number[]}} the sub-matrix and
 *   taxa in selection order; the inputs themselves when nothing is done
 */
export function downsampleTaxaFaithPd(distMat, taxa, maxSpecies) {
	const n = taxa.length;
	if (maxSpecies >= n || maxSpecies <= 0) {
		return { distMat, taxa, selectedIndices: taxa.map((_, i) => i) };
	}
	if (distMat.length !== n * n) {
		throw new Error(`downsampleTaxaFaithPd: matrix has ${distMat.length} entries for ${n} taxa`);
	}

	// Step 1: the most distant pair (first maximum in row-major order).
	const flat = argmax(distMat);
	const idx1 = Math.floor(flat / n);
	const idx2 = flat % n;
	const selected = [idx1, idx2];
	const minDists = new Float32Array(n);
	for (let k = 0; k < n; k++) minDists[k] = Math.min(distMat[idx1 * n + k], distMat[idx2 * n + k]);

	// Step 2: greedily add the taxon farthest from the current set.
	for (let step = 2; step < maxSpecies; step++) {
		const next = argmax(minDists);
		selected.push(next);
		for (let k = 0; k < n; k++) {
			const v = distMat[next * n + k];
			if (v < minDists[k]) minDists[k] = v;
		}
	}

	const m = selected.length;
	const sub = new Float32Array(m * m);
	for (let i = 0; i < m; i++)
		for (let j = 0; j < m; j++) sub[i * m + j] = distMat[selected[i] * n + selected[j]];
	return { distMat: sub, taxa: selected.map((i) => taxa[i]), selectedIndices: selected };
}

/**
 * `taxa[::stride][:max_species * 2]` with `stride = max(1, len(taxa) // (max_species * 2))`,
 * dataset.py:672-673. Only called by the reference when `len(taxa) > max_species`.
 *
 * @param {string[]} taxa
 * @param {number} maxSpecies
 * @returns {string[]}
 */
export function stridePreselect(taxa, maxSpecies) {
	const twice = maxSpecies * 2;
	if (twice === 0) {
		throw new Error('stridePreselect: ZeroDivisionError: integer division or modulo by zero (max_species = 0)');
	}
	const stride = Math.max(1, Math.floor(taxa.length / twice));
	const strided = [];
	for (let i = 0; i < taxa.length; i += stride) strided.push(taxa[i]);
	// Python slice `[:stop]` semantics, including a negative stop.
	const stop = twice < 0 ? Math.max(0, strided.length + twice) : twice;
	return strided.slice(0, stop);
}
