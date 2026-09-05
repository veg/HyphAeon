/**
 * WHY THIS FILE EXISTS
 *
 * Mirrors the digital Deep Mutational Scanning half of `hyphaeon/epistasis.py` at veg/HyphAeon
 * cf838ab — the part of that file that sweeps single amino-acid substitutions through the network
 * rather than mining the attribution graph. `epistasis.js` owns the graph half (attributions,
 * co-selection network); the split follows the two entry points the CLI exposes (`hyphaeon
 * epistasis` and `hyphaeon dms`), so neither module has to import the other.
 *
 *   CANONICAL_AA_TO_CODON      epistasis.py:43-49    one sense codon per residue
 *   runInsilicoSelectionDms    run_insilico_selection_dms, epistasis.py:446-628
 *   digitalDmsRecord           the result dict of run_digital_dms_analysis, epistasis.py:759-768
 *   runDigitalDmsAnalysis      run_digital_dms_analysis, epistasis.py:724-768, steps 4-5 only
 *   resolveFocalTaxon          epistasis.py:479-486
 *   dmsTargetSites             epistasis.py:467-471
 *   dmsWildTypeAa              epistasis.py:531-536 (and its verbatim second copy, 550-555)
 *
 * run_insilico_selection_dms, step by step (line numbers of epistasis.py):
 *    1. sites to sweep: `target_sites` int-cast, range-filtered, de-duplicated, sorted;      467-471
 *       every site 0..L-1 when absent. An empty result returns [] BEFORE any forward pass.
 *    2. chunking from the adaptive safe batch size: `sites_per_chunk = max(1, safe // 19)`   476-477
 *       for the mutants, `max(1, min(64, safe))` sites per baseline pass.                    493
 *    3. focal taxon: index 0 unless `focal_taxon` case-insensitively occurs in a name;       479-486
 *       the FIRST such taxon wins and a miss is silent.
 *    4. baseline LRT for EVERY site (not only the variable ones, unlike cmd_meme), clamped   490-500
 *       at 0 and held float32.
 *    5. per site: the wild-type residue of the focal taxon, the 19 OTHER standard residues,  530-543
 *       each written into a copy of the site's tokens as CANONICAL_AA_TO_CODON's codon —
 *       both the codon token and the amino-acid token change, every other taxon untouched.
 *    6. delta = mutant - baseline (mutant clamped at 0 first), in float32;                   547-565
 *       intrinsic_plasticity = mean |delta| over the 19; mean / max / min delta; the
 *       baseline's Self-Liang p-value; the record at 567-577.
 *
 * THE SIGN OF `delta`. `delta_lrts = s_mut_lrts - baseline_lrts[s]` (epistasis.py:560) is
 * MUTANT MINUS BASELINE, the opposite of attribution.py's `delta = site_lrt - mod_lrt`. A positive
 * delta here means the substitution INCREASES the selection signal. Checked against
 * fixtures/dms/run_insilico_selection_dms.json, whose mean_delta_lrt is negative at five of six
 * bat_oas1 sites while the deltas straddle zero.
 *
 * QUIRKS REPLICATED (all pinned by tests):
 *   - CANONICAL_AA_TO_CODON is a hand-written table, NOT "the first codon for that residue in
 *     table order": it disagrees with that rule for 18 of the 20 residues (L is CTG, not TTA;
 *     S is AGC, not TCT; only M and W, which have one codon each, coincide). Transcribed
 *     verbatim from epistasis.py:45-48; a generated table would silently change every mutant.
 *   - A `focal_taxon` that matches no taxon leaves the focal index at 0 and reports taxa[0]
 *     (the defaults are assigned before the search loop and the miss is never reported). The
 *     empty string is falsy in Python, so it skips the search rather than matching everything.
 *   - `run_digital_dms_analysis` reports the CALLER'S `focal_taxon` string in its result, not the
 *     taxon that was actually swept: `focal_taxon="beta"` on a taxon named `Beta` records
 *     `"beta"` (epistasis.py:764). `digitalDmsRecord` does the same.
 *   - `total_mutations` is `19 * codon_count` — every site of the alignment, whatever was swept.
 *     The standalone analysis passes no `target_sites`, so the two agree there; a caller that
 *     builds the record around a SUBSET (as `hyphaeon epistasis` does for its sector sites) still
 *     gets 19 * L. Mirrored, not corrected.
 *   - The wild-type residue and the candidate list are computed TWICE per site from the same
 *     inputs — once to build the tensors (531-536), once to read the results (550-555). The two
 *     copies are identical, so this is cost, not behaviour; one call site here, noted so a reader
 *     of the Python does not go looking for the difference.
 *   - `plasticity` and `selection_dms_plasticity` are the SAME array (Python binds one list to
 *     both keys), so a writer serialises it twice and a mutation through one key shows in the other.
 *   - `max(0.0, baseline)` before the p-value is Python's two-argument `max`, which returns the
 *     FIRST argument when the comparison is false — so a NaN baseline gives 0.0 and p = 1.0, where
 *     `Math.max(0, NaN)` would give NaN. Written as `base > 0 ? base : 0` to match. Unreachable
 *     from a clamped model output; mirrored because the clamp is the callback's to honour.
 *
 * FLOAT PRECISION (PLAN.md §5.3 rule 4). The reference holds baseline and mutant LRTs in float32
 * numpy arrays, so `delta` is a float32 subtraction of two float32 values and the four reductions
 * are numpy's float32 reductions: `np.mean` is a pairwise float32 sum and a float32 divide
 * (numpyMeanFloat32 from numeric/reduce.js), `np.max` / `np.min` are exact. Measured on
 * fixtures/dms/run_insilico_selection_dms.json: reconstructing the four fields from the recorded
 * per-mutant deltas in float32 reproduces all 32 of them EXACTLY (max |Δ| 0.0), while the same
 * reductions in float64 miss `intrinsic_plasticity` and `mean_delta_lrt` by up to 2.9e-7 — above
 * the 1e-9 class those fields would otherwise sit at, which is why the float32 path is not
 * optional. `p_value` is float64 throughout (stats.js), as `pvals_from_lrt_self_liang` is.
 *
 * BATCHING IS NOT SEMANTICS. The reference scores 19 * sites_per_chunk mutants in one forward pass
 * and the progress bar reports mut/s; the graph scores batch elements independently, so `batchSize`
 * changes only how many mutants ride in each `predict` call. Every record is computed from its own
 * site's 19 mutant LRTs and that site's baseline, so results are identical at any batch size —
 * asserted directly in dms.test.js (batch 64 vs 19 vs 1, deep-equal records, different call shapes).
 *
 * WHAT IT DELIBERATELY DOES NOT DO. No model, no device, no tree cache, no `torch.mps.empty_cache`
 * (epistasis.py:499-500, 613-614). No printing and no clock: the reference's ESSM bar computes
 * `mut/s` and an ETA from `time.time()` (epistasis.py:584-608), which a pure function cannot read —
 * `options.progress` receives the counts the bar is built from and the runtime times it. No
 * `compute_adaptive_safe_batch_size`: it probes device memory (inference.py:22-57), which PLAN.md
 * §5.1 keeps in the app's runtime; `batchSize` IS that already-computed safe batch size, and its
 * default of 64 is the reference's own (`run_digital_dms_analysis(batch_size=64)`, epistasis.py:731,
 * which the CPU branch passes through unchanged for any N the app caps at). Steps 1-3 of
 * run_digital_dms_analysis (device, `load_alignment_and_tree`, `load_model`) are the runtime's;
 * `runDigitalDmsAnalysis` takes the loaded bundle and a predict callback and does steps 4-5.
 *
 * REV_AA_MAP (epistasis.py:41) and `standard_aas` (epistasis.py:505) are NOT exported. The second
 * is character for character the `AA_LIST` that preprocess/modelContract.js already exports, and
 * both are also defined at the top of epistasis.py, phenotype.py and disease.py — four copies in
 * the reference, and exporting a fifth from here would collide with `epistasis.js` in the barrel
 * (index.js re-exports whole modules; two different bindings under one name are dropped silently).
 */

import { GENETIC_CODE, AA_MAP, CODON_TO_AA } from './preprocess/tokenizer.js';
import { AA_LIST, AA_UNKNOWN, CODON_UNKNOWN } from './preprocess/modelContract.js';
import { numpyMeanFloat32 } from './numeric/reduce.js';
import { pvalsFromLrtSelfLiang } from './stats.js';
import { predictSiteLrts } from './filter.js';

/**
 * @typedef {import('./preprocess/assemble.js').LoadedAlignment} LoadedAlignment
 * @typedef {import('./filter.js').PredictFn} PredictFn
 */

/**
 * @typedef {{
 *   site: number, wt_aa: string, baseline_lrt: number, p_value: number,
 *   intrinsic_plasticity: number, mean_delta_lrt: number, max_delta_lrt: number,
 *   min_delta_lrt: number, mutant_deltas: Record<string, number>
 * }} PlasticityRecord
 */

/**
 * `CANONICAL_AA_TO_CODON`, epistasis.py:44-49, transcribed verbatim: the codon each substitution
 * is written as. Not derivable from the codon table (see the header) — 18 of the 20 entries are
 * not the residue's first codon in TCAG order.
 * @type {Map<string, string>}
 */
export const CANONICAL_AA_TO_CODON = new Map([
	['A', 'GCC'], ['C', 'TGC'], ['D', 'GAC'], ['E', 'GAG'], ['F', 'TTC'],
	['G', 'GGC'], ['H', 'CAC'], ['I', 'ATC'], ['K', 'AAG'], ['L', 'CTG'],
	['M', 'ATG'], ['N', 'AAC'], ['P', 'CCC'], ['Q', 'CAG'], ['R', 'CGC'],
	['S', 'AGC'], ['T', 'ACC'], ['V', 'GTG'], ['W', 'TGG'], ['Y', 'TAC']
]);

/** The 19 alternatives swept at every site (20 standard residues less the wild type). */
export const DMS_MUTANTS_PER_SITE = 19;

/** epistasis.py:41 `REV_AA_MAP` — token 0..19 back to its letter. Private; see the header. */
const REV_AA_MAP = new Map([...AA_LIST].map((aa, i) => [i, aa]));

/**
 * The focal taxon, epistasis.py:479-486: index 0 and `taxa[0]` ("consensus" when there are no
 * taxa) unless `focalTaxon` is a non-empty string that occurs, case-insensitively, in a taxon
 * name — then the FIRST such taxon. A miss keeps the defaults and is not reported.
 *
 * @param {ArrayLike<string>} taxa
 * @param {string|null} [focalTaxon]
 * @returns {{index: number, name: string}}
 */
export function resolveFocalTaxon(taxa, focalTaxon = null) {
	const n = taxa.length;
	let index = 0;
	let name = n > 0 ? taxa[0] : 'consensus';
	if (focalTaxon) {
		const needle = String(focalTaxon).toLowerCase();
		for (let i = 0; i < n; i++) {
			if (String(taxa[i]).toLowerCase().includes(needle)) {
				index = i;
				name = taxa[i];
				break;
			}
		}
	}
	return { index, name };
}

/**
 * The sites to sweep, epistasis.py:467-470: `sorted(set(int(s) for s in target_sites if 0 <= s < L))`
 * when a subset is given, else every site. The range test runs on the RAW value and the cast
 * truncates toward zero, so 7.9 with L = 8 is kept as site 7 and -0.5 is dropped.
 *
 * @param {ArrayLike<number>|null|undefined} targetSites null/undefined for "every site"
 * @param {number} L
 * @returns {number[]} ascending, de-duplicated
 */
export function dmsTargetSites(targetSites, L) {
	if (targetSites == null) return Array.from({ length: L }, (_, s) => s);
	/** @type {Set<number>} */
	const keep = new Set();
	for (let i = 0; i < targetSites.length; i++) {
		const raw = targetSites[i];
		if (raw >= 0 && raw < L) keep.add(Math.trunc(raw));
	}
	return [...keep].sort((x, y) => x - y);
}

/**
 * The wild-type residue at one site, epistasis.py:531-536. The focal taxon's amino-acid token
 * when it is a residue (< 20); otherwise — a gap, an in-frame stop, an ambiguity — the site's
 * MAJORITY residue over the taxa that have one, `np.bincount(...).argmax()`, which takes the
 * LOWEST token among tied modes; 'A' when no taxon carries a residue at all.
 *
 * @param {ArrayLike<number>} a amino-acid tokens, [L, N, 1] layout
 * @param {number} site
 * @param {number} focalIndex
 * @param {number} N
 * @returns {string} one of the 20 letters of AA_LIST
 */
export function dmsWildTypeAa(a, site, focalIndex, N) {
	const base = site * N;
	const focalTok = a[base + focalIndex];
	if (focalTok < AA_UNKNOWN) return REV_AA_MAP.get(focalTok) ?? '-';
	// np.bincount + argmax over the residues present at this site.
	const counts = new Int32Array(AA_UNKNOWN);
	let seen = 0;
	for (let i = 0; i < N; i++) {
		const t = a[base + i];
		if (t < AA_UNKNOWN) {
			counts[t]++;
			seen++;
		}
	}
	if (seen === 0) return REV_AA_MAP.get(0) ?? 'A'; // np.bincount never runs: wt_tok stays 0 -> 'A'
	let best = 0;
	for (let t = 1; t < AA_UNKNOWN; t++) if (counts[t] > counts[best]) best = t; // strict >: first max wins
	return REV_AA_MAP.get(best) ?? 'A';
}

/**
 * `run_insilico_selection_dms(model, c, a, tree_cache, taxa, device, focal_taxon, batch_size,
 * progress, target_sites)` — epistasis.py:446-628.
 *
 * `loaded` supplies the codon and amino-acid tokens (`c`, `a`, Int32Array in [L, N, 1] layout),
 * the distance matrix and MDS coordinates the callback needs (`d`, `z`), `L`, `N` and `taxa`.
 * `options.taxa` overrides the names.
 *
 * @param {LoadedAlignment} loaded
 * @param {PredictFn} predict async `(c, a, meta) -> ArrayLike<number>`; the raw graph `lrt`, one
 *   per batch element. The clamp at 0 and the float32 storage are applied here, as the reference
 *   applies them to `forward_cached`'s output.
 * @param {{focalTaxon?: string|null, batchSize?: number, siteSubset?: ArrayLike<number>|null,
 *   targetSites?: ArrayLike<number>|null, taxa?: ArrayLike<string>|null,
 *   progress?: (p: {phase: string, done: number, total: number, mutantsDone: number,
 *     totalMutants: number, meanPlasticity: number}) => void}} [options]
 *   `siteSubset` is the reference's `target_sites`; `targetSites` is accepted under the Python
 *   name and used only when `siteSubset` is absent. `batchSize` is the already-computed safe
 *   batch size in MUTANTS (see the header); it changes call shapes, never results.
 * @returns {Promise<PlasticityRecord[]>} one record per swept site, in ascending site order
 */
export async function runInsilicoSelectionDms(loaded, predict, options = {}) {
	const { c, a, d, z, L, N } = loaded;
	const taxa = (options.taxa === undefined ? loaded.taxa : options.taxa) ?? [];
	const batchSize = Math.max(1, Math.floor(options.batchSize ?? 64));
	const progress = typeof options.progress === 'function' ? options.progress : null;
	const subset = options.siteSubset !== undefined ? options.siteSubset : options.targetSites;

	// 1. Sites to sweep (467-474). The early return precedes every forward pass.
	const sites = dmsTargetSites(subset, L);
	if (sites.length === 0) return [];

	// 2. Chunking (477, 493).
	const sitesPerChunk = Math.max(1, Math.floor(batchSize / DMS_MUTANTS_PER_SITE));
	const baselineChunk = Math.max(1, Math.min(64, batchSize));

	// 3. Focal taxon (479-486).
	const { index: focalIdx } = resolveFocalTaxon(taxa, options.focalTaxon ?? null);

	// 4. Baseline over EVERY site, clamped at 0, float32 (490-500).
	const totalMutants = DMS_MUTANTS_PER_SITE * sites.length;
	const baseline = await predictSiteLrts(loaded, predict, {
		siteIndices: Array.from({ length: L }, (_, s) => s),
		batchSize: baselineChunk,
		phase: 'dms-baseline',
		onProgress: progress
			? (p) =>
					progress({
						phase: p.phase,
						done: p.done,
						total: p.total,
						mutantsDone: 0,
						totalMutants,
						meanPlasticity: 0
					})
			: undefined
	});

	/** @type {PlasticityRecord[]} */
	const results = [];
	let plasticitySum = 0; // for the bar's running mean Phi (596)
	let mutantsDone = 0;

	for (let chunkStart = 0; chunkStart < sites.length; chunkStart += sitesPerChunk) {
		const chunk = sites.slice(chunkStart, Math.min(chunkStart + sitesPerChunk, sites.length));
		const nMut = DMS_MUTANTS_PER_SITE * chunk.length;

		// 5. `repeat_interleave(19, dim=0)` then one in-place substitution per row (526-543).
		// Row i * 19 + m is site chunk[i] with the focal taxon carrying candidate m.
		const cMut = new Int32Array(nMut * N);
		const aMut = new Int32Array(nMut * N);
		const siteIndices = new Int32Array(nMut);
		/** @type {string[]} */
		const mutantAas = new Array(nMut);
		/** @type {string[][]} */
		const chunkMutAas = [];
		for (let i = 0; i < chunk.length; i++) {
			const s = chunk[i];
			const wtAa = dmsWildTypeAa(a, s, focalIdx, N);
			const cand = [...AA_LIST].filter((aa) => aa !== wtAa);
			// Unreachable: dmsWildTypeAa always returns one of the 20 letters, so exactly 19 remain.
			// The reference would write past row i * 19 + 18 here (and then `zip` would drop the
			// extra), so a guard is safer than emulating a latent out-of-bounds write.
			if (cand.length !== DMS_MUTANTS_PER_SITE) {
				throw new Error(`site ${s}: wild-type residue ${JSON.stringify(wtAa)} left ${cand.length} candidates, expected ${DMS_MUTANTS_PER_SITE}`);
			}
			chunkMutAas.push(cand);
			for (let m = 0; m < cand.length; m++) {
				const row = i * DMS_MUTANTS_PER_SITE + m;
				cMut.set(c.subarray(s * N, (s + 1) * N), row * N);
				aMut.set(a.subarray(s * N, (s + 1) * N), row * N);
				const codon = CANONICAL_AA_TO_CODON.get(cand[m]);
				// `GENETIC_CODE.get(c_str, 64)` / `AA_MAP.get(CODON_TO_AA.get(c_str, '-'), 20)` (542-543).
				cMut[row * N + focalIdx] = (codon !== undefined ? GENETIC_CODE.get(codon) : undefined) ?? CODON_UNKNOWN;
				const aaOfCodon = codon !== undefined ? CODON_TO_AA.get(codon) : undefined;
				aMut[row * N + focalIdx] = (aaOfCodon !== undefined ? AA_MAP.get(aaOfCodon) : undefined) ?? AA_UNKNOWN;
				siteIndices[row] = s;
				mutantAas[row] = cand[m];
			}
		}

		// One forward pass for the chunk (545-547).
		const y = await predict(cMut, aMut, {
			batch: nMut,
			N,
			d,
			z,
			siteIndices,
			phase: 'dms',
			loaded,
			mutantAas,
			focalIndex: focalIdx
		});
		if (y.length !== nMut) {
			throw new Error(`predict returned ${y.length} values for ${nMut} mutants`);
		}
		const mutLrts = new Float32Array(nMut);
		for (let k = 0; k < nMut; k++) mutLrts[k] = Math.fround(Math.max(0, y[k]));

		// 6. The per-site records (549-577).
		for (let i = 0; i < chunk.length; i++) {
			const s = chunk[i];
			const wtAa = dmsWildTypeAa(a, s, focalIdx, N); // recomputed, as the reference does (550-555)
			const cand = chunkMutAas[i];
			const base = baseline[s];

			const deltas = new Float32Array(DMS_MUTANTS_PER_SITE);
			const absDeltas = new Float32Array(DMS_MUTANTS_PER_SITE);
			/** @type {Record<string, number>} */
			const mutantDeltas = {};
			let maxDelta = -Infinity;
			let minDelta = Infinity;
			for (let m = 0; m < DMS_MUTANTS_PER_SITE; m++) {
				const delta = Math.fround(mutLrts[i * DMS_MUTANTS_PER_SITE + m] - base);
				deltas[m] = delta;
				absDeltas[m] = Math.abs(delta);
				mutantDeltas[cand[m]] = delta;
				if (delta > maxDelta) maxDelta = delta;
				if (delta < minDelta) minDelta = delta;
			}

			// Python's `max(0.0, base)` returns 0.0 for a NaN baseline; Math.max would not.
			const clampedBase = base > 0 ? base : 0;
			results.push({
				site: s + 1,
				wt_aa: wtAa,
				baseline_lrt: base,
				p_value: pvalsFromLrtSelfLiang([clampedBase])[0],
				intrinsic_plasticity: numpyMeanFloat32(absDeltas),
				mean_delta_lrt: numpyMeanFloat32(deltas),
				max_delta_lrt: maxDelta,
				min_delta_lrt: minDelta,
				mutant_deltas: mutantDeltas
			});
			plasticitySum += results[results.length - 1].intrinsic_plasticity;
		}

		mutantsDone += nMut;
		if (progress) {
			progress({
				phase: 'dms',
				done: Math.min(chunkStart + sitesPerChunk, sites.length),
				total: sites.length,
				mutantsDone,
				totalMutants,
				meanPlasticity: results.length > 0 ? plasticitySum / results.length : 0
			});
		}
	}

	return results;
}

/**
 * The result dict of `run_digital_dms_analysis`, epistasis.py:759-768, in its key order.
 *
 * `focalTaxon` is recorded as the CALLER passed it, not as the taxon that was swept, and
 * `total_mutations` is 19 * L whatever subset produced `plasticity` — both are the reference's
 * behaviour (see the header). `plasticity` and `selection_dms_plasticity` are the same array.
 *
 * @param {LoadedAlignment} loaded
 * @param {PlasticityRecord[]} plasticity
 * @param {{alignment?: string|null, tree?: string|null, focalTaxon?: string|null,
 *   taxa?: ArrayLike<string>|null}} [options] `alignment` / `tree` are the reference's paths; the
 *   runtime passes whatever identifies its inputs (a file name, a URL, or null).
 * @returns {{alignment: string|null, tree: string|null, taxa_count: number, codon_count: number,
 *   focal_taxon: string, total_mutations: number, plasticity: PlasticityRecord[],
 *   selection_dms_plasticity: PlasticityRecord[]}}
 */
export function digitalDmsRecord(loaded, plasticity, options = {}) {
	const taxa = (options.taxa === undefined ? loaded.taxa : options.taxa) ?? [];
	const focalTaxon = options.focalTaxon || (taxa.length > 0 ? taxa[0] : 'consensus');
	return {
		alignment: options.alignment ?? null,
		tree: options.tree ?? null,
		// epistasis.py:746 `N = len(taxa)`, reported verbatim: names, not loaded.N. The reference's
		// taxa always come from load_alignment_and_tree, so the two agree on every real input.
		taxa_count: taxa.length,
		codon_count: loaded.L,
		focal_taxon: focalTaxon,
		total_mutations: DMS_MUTANTS_PER_SITE * loaded.L,
		plasticity,
		selection_dms_plasticity: plasticity
	};
}

/**
 * `run_digital_dms_analysis(alignment_path, tree_path, ...)` — epistasis.py:724-768, steps 4 and 5.
 * Steps 1-3 (device selection, `load_alignment_and_tree`, `load_model`) are the app runtime's, so
 * this takes the loaded bundle and the predict callback and returns the same record.
 *
 * @param {LoadedAlignment} loaded
 * @param {PredictFn} predict
 * @param {Parameters<typeof runInsilicoSelectionDms>[2] &
 *   {alignment?: string|null, tree?: string|null}} [options]
 * @returns {Promise<ReturnType<typeof digitalDmsRecord>>}
 */
export async function runDigitalDmsAnalysis(loaded, predict, options = {}) {
	// The reference passes no target_sites here: the standalone analysis sweeps every codon.
	const plasticity = await runInsilicoSelectionDms(loaded, predict, {
		...options,
		siteSubset: null,
		targetSites: null
	});
	return digitalDmsRecord(loaded, plasticity, options);
}
