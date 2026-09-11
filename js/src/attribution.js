/**
 * WHY THIS FILE EXISTS
 *
 * Mirrors `hyphaeon/attribution.py` at veg/HyphAeon reconcile/phase-5a:
 *
 *   INV_GENETIC_CODE            attribution.py:16   the token -> codon inverse of GENETIC_CODE
 *   attributeSelection          attribute_selection, attribution.py:19-175
 *   attributionSiteFields       the per-site decoration cmd_meme derives from a record
 *                               (cli.py:287-294: evolutionary_epoch, adaptation_mode, top_driver,
 *                               top_mutation, attribution_details)
 *   attributionsOneIndexed      `{str(k+1): v}` as cmd_meme writes them (cli.py:310)
 *
 * attribute_selection, step by step (line numbers of attribution.py):
 *    1. focal sites: the explicit list; else `where(base_lrts >= min_lrt)`; else every site      55-61
 *    2. mean_dist_to_others = d.mean(axis=1) (float32 row means); max_tree_dist = max(d) or 1  68-70
 *    3. per focal site: site_lrt = base_lrts[site], or one clamped forward pass                80-86
 *       skip when site_lrt < min_lrt                                                           88-89
 *    4. consensus codon token = np.unique + argmax (the LOWEST token among the most frequent)   91-93
 *       -> codon string via INV_GENETIC_CODE ('NNN' if absent) -> amino acid -> AA token      94-96
 *    5. every taxon whose codon token differs, in index order: revert that one taxon to the     99-130
 *       consensus codon and AA tokens, one forward pass, clamp; delta = site_lrt - mod_lrt;
 *       pct = max(0, delta / site_lrt * 100) (0 when site_lrt is 0)
 *    6. sort drivers by delta descending (stable)                                              133
 *    7. positive drivers -> weighted patristic depth -> depth ratio -> epoch, mode              136-156
 *    8. the record                                                                              158-172
 *
 * QUIRKS REPLICATED (all pinned by tests):
 *   - INV_GENETIC_CODE inverts a many-to-one dict: the three stops all map to 64 and the last
 *     wins, so token 64 (a stop, gap, ambiguity or unknown codon) reads back as 'TGA' and its
 *     amino acid as '*'; AA_MAP.get('*', 20) makes the consensus AA token 20. A site whose most
 *     frequent codon token is 64 therefore has consensus codon 'TGA' / '*'.
 *   - `inv` is a parameter attribute_selection never reads; with no base_lrts every focal site
 *     is predicted, invariable or not. Same here.
 *   - delta_lrt is float64 arithmetic on two float32-valued floats; pct uses the same numbers.
 *   - The epoch thresholds are `>= 0.60` tip, `>= 0.35` intermediate, else deep (143-148); the
 *     mode is "Recurrent / Multi-Lineage Adaptation" with >= 2 drivers at pct >= 10.0
 *     (150-151); no positive driver gives "Diffuse / Unresolved" / "Diffuse Background
 *     Variation" with depth and ratio 0.0 (152-156).
 *
 * THE PREDICTION CALLBACK is the one filter.js defines (`predict(c, a, meta)`, tokens
 * [batch, N, 1], raw `lrt` back; the clamp and float32 storage are applied here). The reference
 * runs one forward pass per counterfactual; here the counterfactuals of a site are batched
 * `batchSize` at a time — the graph scores batch elements independently, so the values agree to
 * the graph's 1e-5 class, and it is the difference between N ONNX calls and one.
 *
 * FLOAT PRECISION: `d.mean(axis=1)` on the float32 matrix is numpy's pairwise float32 sum per
 * row and a float32 divide (numpyPairwiseSum from numeric/reduce.js); `np.sum(weights * depths)` and
 * `np.sum(weights)` are float64 pairwise sums; `mean_patristic_depth` is the float32 row mean
 * as a JS number.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: tree caches, devices, progress bars, printing. No
 * datamonkey3 body is replaced by this file; DM3 had no port of attribution.py.
 */

import { GENETIC_CODE, CODON_TO_AA, AA_MAP } from './preprocess/tokenizer.js';
import { numpyPairwiseSum } from './numeric/reduce.js';
import { predictSiteLrts } from './filter.js';

/**
 * @typedef {import('./preprocess/assemble.js').LoadedAlignment} LoadedAlignment
 * @typedef {import('./filter.js').PredictFn} PredictFn
 */

/**
 * attribution.py:16 `{v: k for k, v in GENETIC_CODE.items()}` — Map insertion order is the
 * dataset.py literal order, so for 64 the last stop (TGA) wins.
 * @type {Map<number, string>}
 */
export const INV_GENETIC_CODE = (() => {
	/** @type {Map<number, string>} */
	const m = new Map();
	for (const [codon, tok] of GENETIC_CODE) m.set(tok, codon);
	return m;
})();

/**
 * @typedef {{
 *   taxon: string, taxon_index: number, observed_codon: string, observed_aa: string,
 *   consensus_codon: string, consensus_aa: string, delta_lrt: number,
 *   pct_signal_explained: number, mean_patristic_depth: number
 * }} DrivingSpecies
 */

/**
 * @typedef {{
 *   site_0indexed: number, site_1indexed: number, predicted_lrt: number,
 *   consensus_codon: string, consensus_aa: string, num_mutated_taxa: number,
 *   driving_species: DrivingSpecies[],
 *   when_selection_occurred: {evolutionary_epoch: string, mode_of_adaptation: string,
 *     weighted_patristic_depth: number, tree_depth_ratio: number}
 * }} AttributionRecord
 */

const identity = (/** @type {number} */ v) => v;

/**
 * `attribute_selection(model, c, a, d, z, inv, taxa, focal_sites, min_lrt, base_lrts, cache)`.
 * `loaded` supplies c, a (Int32Array [L, N, 1]), d (Float32Array [N*N]), z, N, L and taxa;
 * `options.taxa` overrides the names; `null` is the reference's `taxa=None` (Taxon_001..,
 * attribution.py:64-65) and an omitted option uses `loaded.taxa`.
 *
 * @param {LoadedAlignment} loaded
 * @param {PredictFn} predict
 * @param {{focalSites?: ArrayLike<number>|null, minLrt?: number, baseLrts?: ArrayLike<number>|null,
 *   taxa?: string[]|null, batchSize?: number,
 *   onProgress?: (p: {phase: string, done: number, total: number}) => void}} [options]
 * @returns {Promise<Map<number, AttributionRecord>>} site (0-indexed) -> record, in focal order
 */
export async function attributeSelection(loaded, predict, options = {}) {
	const { minLrt = 3.84, batchSize = 64, onProgress } = options;
	const baseLrts = options.baseLrts ?? null;
	const { c, a, d, z, L, N } = loaded;
	// attribution.py:64-65: `taxa=None` -> Taxon_001..; here `options.taxa === null` is that None,
	// while an omitted option takes the names the loader produced.
	const taxa =
		(options.taxa === undefined ? loaded.taxa : options.taxa) ??
		Array.from({ length: N }, (_, i) => `Taxon_${String(i + 1).padStart(3, '0')}`);

	// 1. Focal sites.
	/** @type {number[]} */
	let focal;
	if (options.focalSites != null) {
		focal = Array.from(options.focalSites, (s) => Number(s));
	} else if (baseLrts !== null) {
		focal = [];
		for (let s = 0; s < L; s++) if (baseLrts[s] >= minLrt) focal.push(s);
	} else {
		focal = Array.from({ length: L }, (_, s) => s);
	}

	// 2. Patristic depths.
	const meanDistToOthers = new Float64Array(N);
	let maxD = -Infinity;
	for (let i = 0; i < N; i++) {
		meanDistToOthers[i] = Math.fround(numpyPairwiseSum(d, i * N, N, Math.fround) / N);
		for (let j = 0; j < N; j++) if (d[i * N + j] > maxD) maxD = d[i * N + j];
	}
	const maxTreeDist = maxD > 0 ? maxD : 1.0;

	// 3. Site LRTs when the caller has none: one pass per focal site (attribution.py:83-86).
	/** @type {ArrayLike<number>} */
	let siteLrts;
	if (baseLrts !== null) {
		siteLrts = baseLrts;
	} else {
		siteLrts = await predictSiteLrts(loaded, predict, { siteIndices: focal, batchSize, phase: 'attribution-base', onProgress });
	}

	/** @type {Map<number, AttributionRecord>} */
	const attributions = new Map();
	let done = 0;
	for (const site of focal) {
		done++;
		const siteLrt = Number(siteLrts[site]);
		if (siteLrt < minLrt) continue;

		// 4. Consensus token: np.unique (sorted) + argmax (first max) = lowest token among the modes.
		const siteCodons = c.subarray(site * N, (site + 1) * N);
		const counts = new Map();
		for (let i = 0; i < N; i++) counts.set(siteCodons[i], (counts.get(siteCodons[i]) ?? 0) + 1);
		let consTok = -1;
		let consCount = -1;
		for (const [tok, n] of counts) {
			if (n > consCount || (n === consCount && tok < consTok)) {
				consTok = tok;
				consCount = n;
			}
		}
		const consCodon = INV_GENETIC_CODE.get(consTok) ?? 'NNN';
		const consAa = CODON_TO_AA.get(consCodon) ?? '-';
		const consAaTok = AA_MAP.get(consAa) ?? 20;

		// 5. One counterfactual per non-consensus taxon.
		/** @type {number[]} */
		const nonCons = [];
		for (let i = 0; i < N; i++) if (siteCodons[i] !== consTok) nonCons.push(i);
		const modLrts = new Float32Array(nonCons.length);
		for (let b0 = 0; b0 < nonCons.length; b0 += Math.max(1, Math.floor(batchSize))) {
			const b1 = Math.min(b0 + Math.max(1, Math.floor(batchSize)), nonCons.length);
			const batch = b1 - b0;
			const cMod = new Int32Array(batch * N);
			const aMod = new Int32Array(batch * N);
			for (let k = 0; k < batch; k++) {
				cMod.set(siteCodons, k * N);
				aMod.set(a.subarray(site * N, (site + 1) * N), k * N);
				cMod[k * N + nonCons[b0 + k]] = consTok;
				aMod[k * N + nonCons[b0 + k]] = consAaTok;
			}
			const y = await predict(cMod, aMod, {
				batch,
				N,
				d,
				z,
				siteIndices: new Int32Array(batch).fill(site),
				phase: 'attribution',
				loaded,
				taxonIndices: Int32Array.from(nonCons.slice(b0, b1))
			});
			if (y.length !== batch) throw new Error(`predict returned ${y.length} values for ${batch} counterfactuals`);
			for (let k = 0; k < batch; k++) modLrts[b0 + k] = Math.fround(Math.max(0, y[k]));
		}

		/** @type {DrivingSpecies[]} */
		const drivers = [];
		for (let k = 0; k < nonCons.length; k++) {
			const tIdx = nonCons[k];
			const origTok = siteCodons[tIdx];
			const origCodon = INV_GENETIC_CODE.get(origTok) ?? 'NNN';
			const origAa = CODON_TO_AA.get(origCodon) ?? '-';
			const modLrt = modLrts[k];
			const delta = siteLrt - modLrt;
			const pct = siteLrt > 0 ? Math.max(0.0, (delta / siteLrt) * 100.0) : 0.0;
			drivers.push({
				taxon: taxa[tIdx],
				taxon_index: tIdx,
				observed_codon: origCodon,
				observed_aa: origAa,
				consensus_codon: consCodon,
				consensus_aa: consAa,
				delta_lrt: delta,
				pct_signal_explained: pct,
				mean_patristic_depth: meanDistToOthers[tIdx]
			});
		}

		// 6. Stable sort by delta, descending.
		drivers.sort((x, y) => y.delta_lrt - x.delta_lrt);

		// 7. When.
		const pos = drivers.filter((sp) => sp.delta_lrt > 0);
		let weightedDepth;
		let depthRatio;
		let epoch;
		let mode;
		if (pos.length > 0) {
			const weights = pos.map((sp) => sp.delta_lrt);
			const products = pos.map((sp) => sp.delta_lrt * sp.mean_patristic_depth);
			weightedDepth = numpyPairwiseSum(products, 0, products.length, identity) / (numpyPairwiseSum(weights, 0, weights.length, identity) + 1e-8);
			depthRatio = weightedDepth / maxTreeDist;
			if (depthRatio >= 0.6) epoch = 'Recent Terminal / Tip Sweep';
			else if (depthRatio >= 0.35) epoch = 'Intermediate Subclade Burst';
			else epoch = 'Deep Ancestral / Basal Divergence';
			let major = 0;
			for (const sp of pos) if (sp.pct_signal_explained >= 10.0) major++;
			mode = major >= 2 ? 'Recurrent / Multi-Lineage Adaptation' : 'Single-Lineage Clade Sweep';
		} else {
			weightedDepth = 0.0;
			depthRatio = 0.0;
			epoch = 'Diffuse / Unresolved';
			mode = 'Diffuse Background Variation';
		}

		// 8. Record.
		attributions.set(site, {
			site_0indexed: site,
			site_1indexed: site + 1,
			predicted_lrt: siteLrt,
			consensus_codon: consCodon,
			consensus_aa: consAa,
			num_mutated_taxa: nonCons.length,
			driving_species: drivers,
			when_selection_occurred: {
				evolutionary_epoch: epoch,
				mode_of_adaptation: mode,
				weighted_patristic_depth: weightedDepth,
				tree_depth_ratio: depthRatio
			}
		});
		if (onProgress) onProgress({ phase: 'attribution', done, total: focal.length });
	}
	return attributions;
}

/**
 * The fields cmd_meme adds to a site entry from its attribution record (cli.py:287-294):
 * `top_driver` / `top_mutation` are the first driver (highest delta) or null when the site has no
 * non-consensus taxon.
 *
 * @param {AttributionRecord} rec
 * @returns {{evolutionary_epoch: string, adaptation_mode: string, top_driver: string|null,
 *   top_mutation: string|null, attribution_details: AttributionRecord}}
 */
export function attributionSiteFields(rec) {
	const top = rec.driving_species.length > 0 ? rec.driving_species[0] : null;
	return {
		evolutionary_epoch: rec.when_selection_occurred.evolutionary_epoch,
		adaptation_mode: rec.when_selection_occurred.mode_of_adaptation,
		top_driver: top ? top.taxon : null,
		top_mutation: top ? `${rec.consensus_aa}->${top.observed_aa}` : null,
		attribution_details: rec
	};
}

/**
 * `{str(k + 1): v for k, v in attributions.items()}` (cli.py:310): the records keyed by the
 * 1-indexed site as a string, in insertion order.
 *
 * @param {Map<number, AttributionRecord>} attributions
 * @returns {Record<string, AttributionRecord>}
 */
export function attributionsOneIndexed(attributions) {
	/** @type {Record<string, AttributionRecord>} */
	const out = {};
	for (const [site, rec] of attributions) out[String(site + 1)] = rec;
	return out;
}
