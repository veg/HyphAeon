/**
 * WHY THIS FILE EXISTS
 *
 * Mirrors the statistics of `cmd_busted` in `hyphaeon/cli.py:424-508` at veg/HyphAeon 267f5cf —
 * the alignment-wide omnibus ("BUSTED emulation") that the CLI computes per alignment once the
 * backbone has produced a site LRT and a pooled `root_repr` for every variable site, plus the
 * neural-head fields the same block derives. Line by line, in the reference's order:
 *
 *   variable_indices = np.where(~inv)[0]                                                   cli.py:424
 *   batch_size = min(adaptive(num_species), max(1, num_variable))                          429-430
 *   lrts = zeros(L, float32); lrts[variable] = clamp(y_soft, min=0)                        433, 439-446
 *   hidden_all = zeros(1, L, embed_dim); hidden_all[0, variable] = root_repr               434, 447
 *   neural_out = busted_head(hidden_all)   (no key-padding mask)                           458
 *   pred_prob_pos = cls_prob; pred_neural_lrt = pred_lrt (else sqrt_lrt ** 2, else 0)      459-460
 *   pred_syn_var = syn_var; pred_w3 = pred_omega3 (else omega_vals[2], else 1.0)          461-462
 *   pred_prop = omega_prop.squeeze(); pred_omega = [0.10, 1.00, pred_w3]                    463-464
 *   pvals = pvals_from_lrt_self_liang(lrts)                                                 469
 *   var_p = pvals[variable_indices] if num_variable > 0 else pvals; p_acat = CCT(var_p)    472-473
 *   p_simes = min((L / ranks) * sort(pvals)), clipped to [1e-15, 1]                         475-478
 *   sig_sites_05/10 = sum(pvals < 0.05 / 0.10)                                              480-481
 *   total_selection_energy = sum(lrts)                                                      482
 *   omnibus_lrt = sum(max(0, lrts - 3.841))                                                 483
 *   is_significant = p_acat < 0.05 or pred_prob_pos > 0.50                                  484
 *   record = {...}                                                                          486-505
 *
 * PRECISION IS PART OF THE CONTRACT (PLAN.md §5.3 rule 4). `lrts` is a float32 array in the
 * reference, so `np.sum(lrts)` and `np.sum(np.maximum(0.0, lrts - 3.841))` are float32 reductions
 * with numpy's pairwise algorithm, and `lrts - 3.841` is float32 minus float32(3.841) under numpy 2
 * promotion (fixtures were generated with numpy 2.3.3). A float64 sum of the Smc6 LRTs is off from
 * the fixture's `total_selection_energy` by 6.8e-6 and HIV1_RT's by 5.8e-6 — inside the e2e class
 * but not equal; the float32 pairwise sum (`float32Sum`, numeric/reduce.js) reproduces both bit for bit (measured with the
 * reference Python on fixtures/e2e/meme_*.json's per-site LRTs). The p-values are float64: scipy
 * promotes the float32 LRT to float64 before `chi2.sf` (`rv_continuous.sf`, `promote_types(x.dtype,
 * float64)`), the CCT runs on the float64 `pvals`, and Simes is float64 `(L / rank) * p` in that
 * order.
 *
 * THE NEURAL HEAD IS NOT DETERMINISTIC UPSTREAM (fixtures/manifest.json `known_quirks`,
 * PHASE0.md gap 7): `model.safetensors` lacks 11 `BustedMultiTaskHead` parameters and cmd_busted
 * loads with `strict=False`, unseeded, so `selection_probability`, `predicted_gene_lrt`,
 * `synonymous_rate_variation`, `omega_3`, `proportion_*` and the verdict differ on every reference
 * run. `busted_head.onnx` is one seeded draw of that (export.py BUSTED_HEAD_INIT_SEED = 0). The
 * e2e fixtures null those fields; this module reproduces the ARITHMETIC that turns head outputs
 * into record fields (`bustedHeadFields`) and does not claim the numbers mean anything until the
 * checkpoint is completed upstream.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: no model. `runBusted` is the per-alignment loop of cmd_busted
 * with the two forward passes handed to async callbacks the runtime supplies (onnxruntime lives in
 * the app's runtime/, PLAN.md §5.5). No files, no printing (the console report of cli.py:511-540
 * is presentation), no batch-directory handling (cli.py:365-408), no CSV (cli.py:548-560), no
 * `elapsed_seconds` (timing is the runtime's; the field is emitted as null, as the fixtures hold it).
 * The site LRTs themselves come from the backbone graph; `pvals_from_lrt_self_liang` is ported in
 * ./stats.js and imported here.
 *
 * Constants, with their source lines in cli.py:
 *   BUSTED_OMEGA_1 = 0.10, BUSTED_OMEGA_2 = 1.00       pred_omega = [0.10, 1.00, pred_w3]   (464)
 *   BUSTED_LRT_THRESHOLD = 3.841                       omnibus_lrt = sum(max(0, lrt - 3.841)) (483);
 *                                                      chi2(1) 95% quantile, 3.8415 rounded
 *   BUSTED_ACAT_ALPHA = 0.05, BUSTED_PROB_THRESHOLD = 0.50   is_significant (484)
 *   BUSTED_SIMES_FLOOR = 1e-15                         max(1e-15, min(1.0, p_simes)) (478)
 *   BUSTED_EMBED_DIM = 384                             model_config.json embed_dim; hidden_all's
 *                                                      last axis (cli.py:434)
 *
 * There was no DataMonkey 3 counterpart: DM3 has no BUSTED surface.
 */

import { pvalsFromLrtSelfLiang } from './stats.js';
import { cauchyCombination } from './numeric/cauchy.js';
import { float32Sum } from './numeric/reduce.js';
import { batchSizeFor } from './preprocess/assemble.js';
import { MDS_COMPONENTS } from './preprocess/modelContract.js';

/** cli.py:464 — the two fixed omega classes of the reported mixture. */
export const BUSTED_OMEGA_1 = 0.1;
export const BUSTED_OMEGA_2 = 1.0;
/** cli.py:483 — the per-site LRT excess that accumulates into `omnibus_lrt`. */
export const BUSTED_LRT_THRESHOLD = 3.841;
/** cli.py:484 — the verdict rule `p_acat < 0.05 or pred_prob_pos > 0.50`. */
export const BUSTED_ACAT_ALPHA = 0.05;
export const BUSTED_PROB_THRESHOLD = 0.5;
/** cli.py:478 — `p_simes = max(1e-15, min(1.0, p_simes))`. */
export const BUSTED_SIMES_FLOOR = 1e-15;
/** model_config.json `embed_dim`: the width of `root_repr` and of `hidden_all` (cli.py:434). */
export const BUSTED_EMBED_DIM = 384;

/** export.py BUSTED_INPUT_NAMES / BUSTED_OUTPUT_NAMES: what busted_head.onnx takes and returns. */
export const BUSTED_HEAD_INPUT_NAMES = Object.freeze(['root_repr', 'mask']);
export const BUSTED_HEAD_OUTPUT_NAMES = Object.freeze([
	'cls_prob',
	'pred_gene_lrt',
	'omega_prop',
	'syn_var',
	'pred_omega3',
	'pred_logp'
]);

/**
 * `np.where(~inv)[0]`, cli.py:424.
 *
 * @param {ArrayLike<number|boolean>} invariable 1/true = invariable, length L
 * @returns {Int32Array} ascending site indices of the variable sites
 */
export function variableSiteIndices(invariable) {
	const out = [];
	for (let i = 0; i < invariable.length; i++) if (!invariable[i]) out.push(i);
	return Int32Array.from(out);
}

/**
 * `float(np.sum(lrts))`, cli.py:482.
 *
 * @param {ArrayLike<number>} lrts float32 site LRTs, length L
 * @returns {number}
 */
export function totalSelectionEnergy(lrts) {
	return float32Sum(lrts);
}

/**
 * `float(np.sum(np.maximum(0.0, lrts - 3.841)))`, cli.py:483, in float32 throughout: the scalar is
 * rounded to float32 first (numpy 2 promotion), each difference is a float32 subtraction, and the
 * reduction is the pairwise float32 sum.
 *
 * @param {ArrayLike<number>} lrts float32 site LRTs, length L
 * @param {number} [threshold] default BUSTED_LRT_THRESHOLD
 * @returns {number}
 */
export function omnibusLrt(lrts, threshold = BUSTED_LRT_THRESHOLD) {
	const t = Math.fround(threshold);
	const excess = new Float32Array(lrts.length);
	for (let i = 0; i < lrts.length; i++) {
		const d = Math.fround(Math.fround(lrts[i]) - t);
		excess[i] = d > 0 ? d : 0;
	}
	return float32Sum(excess);
}

/**
 * Simes' combination, cli.py:475-478:
 *
 *     sorted_p = np.sort(pvals); ranks = np.arange(1, L + 1)
 *     p_simes = float(np.min((L / ranks) * sorted_p))
 *     p_simes = max(1e-15, min(1.0, p_simes))
 *
 * `L` is the number of p-values (all sites, including invariable ones at p = 1). Note the
 * operation order `(L / rank) * p`, kept for bitwise agreement with the reference.
 *
 * @param {ArrayLike<number>} pvals float64 site p-values
 * @returns {number}
 */
export function simesP(pvals) {
	const L = pvals.length;
	if (L === 0) {
		// `np.min` of an empty array raises ValueError in the reference.
		throw new Error('simesP: zero-size array to reduction operation minimum which has no identity');
	}
	const sorted = Float64Array.from(pvals).sort();
	let min = Infinity;
	for (let i = 0; i < L; i++) {
		const v = (L / (i + 1)) * sorted[i];
		if (v < min || Number.isNaN(v)) min = v; // np.min propagates NaN
		if (Number.isNaN(min)) break;
	}
	return Math.max(BUSTED_SIMES_FLOOR, Math.min(1.0, min));
}

/**
 * @typedef {{
 *   L: number,
 *   numVariable: number,
 *   variableIndices: Int32Array,
 *   pvals: Float64Array,
 *   pAcat: number,
 *   pSimes: number,
 *   omnibusLrt: number,
 *   totalSelectionEnergy: number,
 *   sigSitesP05: number,
 *   sigSitesP10: number
 * }} BustedStatistics
 */

/**
 * The statistical bridge of cmd_busted (cli.py:469-483) from the site LRT vector and the
 * invariable mask: Self–Liang site p-values, ACAT over the VARIABLE sites, Simes over ALL sites,
 * significant-site counts, total selection energy and the omnibus LRT.
 *
 * @param {ArrayLike<number>} lrts site LRTs, length L (float32 in the reference: zeros at
 *   invariable sites, clamp(y, min=0) elsewhere)
 * @param {ArrayLike<number|boolean>} invariable 1/true = invariable, length L
 * @returns {BustedStatistics}
 */
export function bustedStatistics(lrts, invariable) {
	const L = lrts.length;
	if (invariable.length !== L) {
		throw new Error(`bustedStatistics: ${L} LRTs but ${invariable.length} invariable flags`);
	}
	const f32 = lrts instanceof Float32Array ? lrts : Float32Array.from(lrts);
	const variableIndices = variableSiteIndices(invariable);
	const numVariable = variableIndices.length;

	// 4. Asymptotic mixture p-values (cli.py:469).
	const pvals = pvalsFromLrtSelfLiang(f32);

	// 5. ACAT over variable sites, Simes over all (cli.py:472-478).
	let varP = pvals;
	if (numVariable > 0) {
		varP = new Float64Array(numVariable);
		for (let k = 0; k < numVariable; k++) varP[k] = pvals[variableIndices[k]];
	}
	const pAcat = cauchyCombination(varP);
	const pSimes = simesP(pvals);

	let sig05 = 0;
	let sig10 = 0;
	for (let i = 0; i < L; i++) {
		if (pvals[i] < 0.05) sig05++;
		if (pvals[i] < 0.1) sig10++;
	}

	return {
		L,
		numVariable,
		variableIndices,
		pvals,
		pAcat,
		pSimes,
		omnibusLrt: omnibusLrt(f32),
		totalSelectionEnergy: totalSelectionEnergy(f32),
		sigSitesP05: sig05,
		sigSitesP10: sig10
	};
}

/**
 * `.item()` on a 0-d / 1-element tensor output, or a plain number.
 * @param {ArrayLike<number>|number|undefined} v
 * @returns {number|undefined}
 */
function item(v) {
	if (v === undefined || v === null) return undefined;
	if (typeof v === 'number') return v;
	if (typeof v === 'bigint') return Number(v);
	return v.length === undefined ? undefined : Number(v[0]);
}

/**
 * @typedef {{
 *   cls_prob: ArrayLike<number>|number,
 *   pred_gene_lrt?: ArrayLike<number>|number,
 *   pred_lrt?: ArrayLike<number>|number,
 *   sqrt_lrt?: ArrayLike<number>|number,
 *   syn_var: ArrayLike<number>|number,
 *   pred_omega3?: ArrayLike<number>|number,
 *   omega_vals?: ArrayLike<number>,
 *   omega_prop: ArrayLike<number>,
 *   pred_logp?: ArrayLike<number>|number
 * }} BustedHeadOutputs
 */

/**
 * @typedef {{
 *   selection_probability: number,
 *   predicted_gene_lrt: number,
 *   synonymous_rate_variation: number,
 *   omega: [number, number, number],
 *   proportions: [number, number, number],
 *   rate_distributions: {
 *     omega_1: number, proportion_1: number,
 *     omega_2: number, proportion_2: number,
 *     omega_3: number, proportion_3: number
 *   }
 * }} BustedHeadFields
 */

/**
 * cli.py:459-464 and 497-500: the record fields derived from `BustedMultiTaskHead` outputs, given
 * either the ONNX names (`pred_gene_lrt`, export.py) or the module's (`pred_lrt`). Each output may
 * be a number or a 1-element typed array (`.item()`); `omega_prop` is the 3-vector after
 * `.squeeze()`. The dead fallbacks of the reference are kept: `sqrt_lrt ** 2` (else 0.0) when
 * neither LRT key is present, `omega_vals[2]` (else 1.0) when `pred_omega3` is absent.
 *
 *     pred_omega = [0.10, 1.00, pred_w3]
 *     rate_distributions = {omega_i: float(pred_omega[i-1]), proportion_i: float(pred_prop[i-1])}
 *
 * @param {BustedHeadOutputs} head
 * @returns {BustedHeadFields}
 */
export function bustedHeadFields(head) {
	const prob = item(head.cls_prob);
	if (prob === undefined) throw new Error('bustedHeadFields: head output `cls_prob` is missing');
	let lrt = item(head.pred_gene_lrt ?? head.pred_lrt);
	if (lrt === undefined) {
		const s = item(head.sqrt_lrt) ?? 0.0;
		lrt = s * s;
	}
	const synVar = item(head.syn_var);
	if (synVar === undefined) throw new Error('bustedHeadFields: head output `syn_var` is missing');
	let w3 = item(head.pred_omega3);
	if (w3 === undefined) {
		w3 = head.omega_vals !== undefined && head.omega_vals.length > 2 ? Number(head.omega_vals[2]) : 1.0;
	}
	const prop = head.omega_prop;
	if (prop === undefined || prop.length < 3) {
		throw new Error('bustedHeadFields: head output `omega_prop` must hold three proportions');
	}
	const p = /** @type {[number, number, number]} */ ([Number(prop[0]), Number(prop[1]), Number(prop[2])]);
	const omega = /** @type {[number, number, number]} */ ([BUSTED_OMEGA_1, BUSTED_OMEGA_2, w3]);
	return {
		selection_probability: prob,
		predicted_gene_lrt: lrt,
		synonymous_rate_variation: synVar,
		omega,
		proportions: p,
		rate_distributions: {
			omega_1: omega[0],
			proportion_1: p[0],
			omega_2: omega[1],
			proportion_2: p[1],
			omega_3: omega[2],
			proportion_3: p[2]
		}
	};
}

/**
 * `is_significant = bool(p_acat < 0.05 or pred_prob_pos > 0.50)`, cli.py:484. With no head
 * probability (null), the verdict is `true` when the ACAT half decides and `null` otherwise — the
 * value the e2e fixtures hold for the un-reproducible head.
 *
 * @param {number} pAcat
 * @param {number|null} [selectionProbability]
 * @returns {boolean|null}
 */
export function bustedVerdict(pAcat, selectionProbability = null) {
	if (pAcat < BUSTED_ACAT_ALPHA) return true;
	if (selectionProbability === null || selectionProbability === undefined) return null;
	return selectionProbability > BUSTED_PROB_THRESHOLD;
}

/**
 * @typedef {{
 *   alignment: string|null,
 *   gene: string|null,
 *   taxa: number,
 *   sites: number,
 *   p_value_acat: number,
 *   p_value_simes: number,
 *   omnibus_lrt: number,
 *   predicted_gene_lrt: number|null,
 *   selection_probability: number|null,
 *   synonymous_rate_variation: number|null,
 *   total_selection_energy: number,
 *   sig_sites_p05: number,
 *   sig_sites_p10: number,
 *   rate_distributions: {
 *     omega_1: number, proportion_1: number|null,
 *     omega_2: number, proportion_2: number|null,
 *     omega_3: number|null, proportion_3: number|null
 *   },
 *   positive_selection_detected: boolean|null,
 *   elapsed_seconds: null
 * }} BustedRecord
 */

/**
 * The per-alignment record of cmd_busted (cli.py:486-505), key for key and in the same order as
 * the CLI's JSON, from the site LRTs, the invariable mask and (optionally) the head outputs. With
 * `head` null the neural fields are null, as fixtures/e2e/busted_*.json hold them.
 *
 * @param {{
 *   lrts: ArrayLike<number>,
 *   invariable: ArrayLike<number|boolean>,
 *   numTaxa: number,
 *   head?: BustedHeadOutputs|null,
 *   gene?: string|null,
 *   alignment?: string|null
 * }} args
 * @returns {BustedRecord & {statistics: BustedStatistics}}
 */
export function bustedRecord({ lrts, invariable, numTaxa, head = null, gene = null, alignment = null }) {
	const stats = bustedStatistics(lrts, invariable);
	const fields = head ? bustedHeadFields(head) : null;
	const record = {
		alignment,
		gene,
		taxa: numTaxa,
		sites: stats.L,
		p_value_acat: stats.pAcat,
		p_value_simes: stats.pSimes,
		omnibus_lrt: stats.omnibusLrt,
		predicted_gene_lrt: fields ? fields.predicted_gene_lrt : null,
		selection_probability: fields ? fields.selection_probability : null,
		synonymous_rate_variation: fields ? fields.synonymous_rate_variation : null,
		total_selection_energy: stats.totalSelectionEnergy,
		sig_sites_p05: stats.sigSitesP05,
		sig_sites_p10: stats.sigSitesP10,
		rate_distributions: fields
			? fields.rate_distributions
			: {
					omega_1: BUSTED_OMEGA_1,
					proportion_1: null,
					omega_2: BUSTED_OMEGA_2,
					proportion_2: null,
					omega_3: null,
					proportion_3: null
				},
		positive_selection_detected: bustedVerdict(stats.pAcat, fields ? fields.selection_probability : null),
		elapsed_seconds: null
	};
	return Object.assign(record, { statistics: stats });
}

/**
 * The graph bundle for an ARBITRARY set of sites (`c[batch_site_idx]`, cli.py:441-443), in the
 * shape `siteBatch` (assemble.js) gives for a contiguous range: `msa_codons` / `msa_aas` as
 * BigInt64Array [b, N, 1], `dist_matrix` [b, N, N] and `mds_coords` [b, N, 4] repeated per site.
 * cmd_busted batches the variable sites, which are not contiguous.
 *
 * @param {import('./preprocess/assemble.js').LoadedAlignment} loaded
 * @param {ArrayLike<number>} siteIndices
 * @returns {Record<string, {data: BigInt64Array|Float32Array, dims: number[]}>}
 */
export function gatherSiteBatch(loaded, siteIndices) {
	const { c, a, d, z, N } = loaded;
	const b = siteIndices.length;
	const codons = new BigInt64Array(b * N);
	const aas = new BigInt64Array(b * N);
	for (let k = 0; k < b; k++) {
		const site = siteIndices[k];
		for (let i = 0; i < N; i++) {
			codons[k * N + i] = BigInt(c[site * N + i]);
			aas[k * N + i] = BigInt(a[site * N + i]);
		}
	}
	const distData = new Float32Array(b * N * N);
	const mdsData = new Float32Array(b * N * MDS_COMPONENTS);
	for (let k = 0; k < b; k++) {
		distData.set(d, k * N * N);
		mdsData.set(z, k * N * MDS_COMPONENTS);
	}
	return {
		msa_codons: { data: codons, dims: [b, N, 1] },
		msa_aas: { data: aas, dims: [b, N, 1] },
		dist_matrix: { data: distData, dims: [b, N, N] },
		mds_coords: { data: mdsData, dims: [b, N, MDS_COMPONENTS] }
	};
}

/**
 * @typedef {{
 *   predictSites: (bundle: ReturnType<typeof gatherSiteBatch>, info: {siteIndices: Int32Array, count: number, N: number})
 *     => Promise<{lrt: ArrayLike<number>, root_repr?: ArrayLike<number>, rootRepr?: ArrayLike<number>}>,
 *   predictHead?: ((input: {root_repr: Float32Array, mask: Uint8Array, dims: [number, number, number], L: number, embedDim: number})
 *     => Promise<BustedHeadOutputs>) | null
 * }} BustedModel
 */

/**
 * The per-alignment loop of cmd_busted (cli.py:424-484) over callbacks the runtime supplies:
 *
 *   - `predictSites(bundle, {siteIndices, count, N})` runs the backbone on a batch of VARIABLE
 *     sites and resolves `{lrt: [b], root_repr: [b * embedDim]}` (the graph's `lrt` and
 *     `root_repr` outputs). LRTs are clamped at 0 and stored as float32 (cli.py:445-446);
 *     `root_repr` rows are scattered into `hidden_all` [1, L, embedDim], which stays zero at
 *     invariable sites (cli.py:434, 447).
 *   - `predictHead({root_repr, mask, dims, L, embedDim})` runs busted_head.onnx on the whole
 *     `hidden_all` with an all-false key-padding mask — cmd_busted calls the head with
 *     `mask=None` (cli.py:458), and the exported graph's `mask` input is that `None` spelled as
 *     a [1, L] boolean tensor of zeros. Pass `null` to skip the head (statistics only).
 *
 * Batches are consecutive slices of the ascending variable-site index list of at most
 * `batchSize` sites (cli.py:439-443); the reference sizes the batch by device memory
 * (`compute_adaptive_safe_batch_size`), here `batchSizeFor(N)` from assemble.js unless given.
 * Batch composition changes nothing but float noise: every site is independent in the graph.
 *
 * @param {import('./preprocess/assemble.js').LoadedAlignment} loaded from `loadAlignmentAndTree`
 * @param {BustedModel} model
 * @param {{batchSize?: number, embedDim?: number, gene?: string|null, alignment?: string|null,
 *   progress?: (done: number, total: number) => void}} [options]
 * @returns {Promise<BustedRecord & {statistics: BustedStatistics, lrts: Float32Array,
 *   root_repr: Float32Array|null, head: BustedHeadOutputs|null}>}
 */
export async function runBusted(loaded, model, options = {}) {
	const { L, N, invariable, taxa } = loaded;
	const { gene = null, alignment = null, progress } = options;
	const variableIndices = variableSiteIndices(invariable);
	const numVariable = variableIndices.length;
	const batchSize = Math.max(1, Math.min(Math.floor(options.batchSize ?? batchSizeFor(N)), Math.max(1, numVariable)));

	const lrts = new Float32Array(L);
	let embedDim = options.embedDim ?? BUSTED_EMBED_DIM;
	/** @type {Float32Array|null} */
	let hiddenAll = null;

	for (let start = 0; start < numVariable; start += batchSize) {
		const end = Math.min(start + batchSize, numVariable);
		const batchSites = variableIndices.subarray(start, end);
		const bundle = gatherSiteBatch(loaded, batchSites);
		const out = await model.predictSites(bundle, { siteIndices: batchSites, count: end - start, N });
		if (!out || out.lrt === undefined || out.lrt.length < end - start) {
			throw new Error(`runBusted: predictSites returned ${out && out.lrt ? out.lrt.length : 0} LRTs for ${end - start} sites`);
		}
		const repr = out.root_repr ?? out.rootRepr;
		if (repr !== undefined) {
			const width = repr.length / (end - start);
			if (!Number.isInteger(width) || width <= 0) {
				throw new Error(`runBusted: root_repr has ${repr.length} values for ${end - start} sites`);
			}
			if (hiddenAll === null) {
				embedDim = width;
				hiddenAll = new Float32Array(L * embedDim);
			} else if (width !== embedDim) {
				throw new Error(`runBusted: root_repr width changed from ${embedDim} to ${width}`);
			}
		}
		for (let k = 0; k < end - start; k++) {
			const site = batchSites[k];
			const y = Math.fround(Number(out.lrt[k]));
			// torch.clamp(y, min=0.0): NaN propagates, as in torch.
			lrts[site] = Number.isNaN(y) ? NaN : y > 0 ? y : 0;
			if (repr !== undefined && hiddenAll !== null) {
				for (let e = 0; e < embedDim; e++) hiddenAll[site * embedDim + e] = Math.fround(Number(repr[k * embedDim + e]));
			}
		}
		if (progress) progress(end, numVariable);
	}

	// 3. Neural head on the whole alignment (cli.py:457-464), mask=None.
	/** @type {BustedHeadOutputs|null} */
	let head = null;
	if (model.predictHead) {
		if (hiddenAll === null) hiddenAll = new Float32Array(L * embedDim);
		head = await model.predictHead({
			root_repr: hiddenAll,
			mask: new Uint8Array(L),
			dims: [1, L, embedDim],
			L,
			embedDim
		});
	}

	const record = bustedRecord({ lrts, invariable, numTaxa: taxa.length, head, gene, alignment });
	return Object.assign(record, { lrts, root_repr: hiddenAll, head });
}
