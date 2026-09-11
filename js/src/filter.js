/**
 * WHY THIS FILE EXISTS
 *
 * Mirrors `hyphaeon/filter.py` at veg/HyphAeon reconcile/phase-5a as pure functions:
 *
 *   scanHypergeometricPatches   scan_hypergeometric_patches   filter.py:52-100
 *   predictSiteLrts             the variable-site inference loop of filter.py:155-165 / 308-316,
 *                               which is `inference.predict_site_lrts` (inference.py:162-192)
 *   consensusCodons             the per-site consensus of filter.py:204-209 (cli.py:131-136)
 *   auditPatch                  the OCI attribution of one patch, filter.py:223-253 (cli.py:140-169)
 *   maskCodonSpan               the NNN masking of filter.py:272-275 (cli.py:179-181)
 *   runAlignmentFilter          run_alignment_filter, filter.py:102-399, both forward-pass phases
 *
 * plus, behind `{cliVariant: true}`, the SECOND copy of the OCI screen that `cli.py cmd_meme
 * --filter` carries (cli.py:114-221 at reconcile/phase-5a). The two copies differ, and the fixture notes
 * (fixtures/e2e/meme_bat_oas1_attribute_filter.json, manifest `known_quirks`) pin the differences:
 *
 *   1. consensus codons: cmd_meme drops only codons containing '-' or 'N' (cli.py:135);
 *      run_alignment_filter also drops '?' and anything not of length 3 (filter.py:208);
 *   2. the artifact rule reads `min_patch_consec` (cli.py:138, 166) — an attribute no argparse flag
 *      sets, so it is always 3 — and hard-codes min_oci = 0.25, alpha 0.05, min_k 3, max_span 35;
 *      only `--filter-p-thresh` (p_local) is configurable;
 *   3. the p-values it scans are float32 (`pvals_from_lrt_meme(lrts).astype(np.float32)`,
 *      cli.py:105), so `site_pvals <= 0.05` compares against float32(0.05);
 *   4. the cleaned FASTA holds only the matched `taxa` in taxa order (cli.py:189-192), where
 *      run_alignment_filter writes EVERY parsed sequence in file order (filter.py:293-294);
 *   5. the cleaned re-load is given `args.tree` as is (cli.py:195): with an embedded tree and at
 *      least one artifact the cleaned FASTA has no tree and the loader raises — the command
 *      crashes. run_alignment_filter passes the alignment path instead (filter.py:297).
 *   6. the cleaned re-score reuses the ORIGINAL tree cache (cli.py:198-200), not the cleaned
 *      distances; the two are identical unless masking changed the taxon set, in which case the
 *      model fails on a shape mismatch.
 *   7. cmd_meme does not compute cct / mean-LRT metrics; its outputs are `artifacts_masked`
 *      records {start, end, span, outlier_taxon, consecutive_mismatches, oci} and the per-site
 *      {hyphaeon_lrt, p_value, q_value, is_invariable} of the CLEANED run.
 *
 * All seven are replicated here, not reconciled; the upstream issue is recorded in the report.
 *
 * THE PREDICTION CALLBACK. Everything that needs the network goes through one async callback,
 * `predict(c, a, meta)`, so the library stays free of onnxruntime. `c` and `a` are Int32Array
 * token tensors in [batch, N, 1] row-major layout (dataset.py vocabularies, 0..64 / 0..20); `meta`
 * carries the batch size, N, the float32 `d` [N*N] and `z` [N*4] the tree cache should be built
 * from, the site indices of the batch and a phase label. The callback returns the graph's raw
 * `lrt` output (one number per batch element; a Promise is fine). The clamp `torch.clamp(y, min=0)`
 * and the float32 storage (`np.zeros(L, dtype=np.float32)`) are applied HERE, as filter.py:165
 * does, so the callback returns what the graph returns.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: files (the cleaned FASTA is returned as text and as a
 * Map, the audit CSV as the `artifacts` records), printing / progress bars (an optional
 * `onProgress` hook replaces filter.py:168-175 and 318-325), device management, adaptive batch
 * sizing (filter.py:140-144 depends on the accelerator; `batchSize` is an option, and batching
 * does not change results), timing beyond `elapsed_seconds`, and the MEME p-value conversion,
 * which is stats.js's `pvalsFromLrtMeme` (stats.py:16-31), imported.
 *
 * FLOAT PRECISION, following the reference (PLAN.md §5.3 rule 4): LRT arrays are float32; the
 * two `np.mean` calls on float32 arrays (filter.py:269, 385, 391) are numpy's pairwise float32
 * summation followed by a float32 divide, reproduced by numeric/reduce.js's `numpyMeanFloat32`;
 * p-values, q-values,
 * CCT and OCI are float64 in run_alignment_filter and float32 in the cmd_meme variant.
 *
 * No datamonkey3 body is replaced by this file; DM3 had no port of filter.py.
 */

import { hypergeomCdf, benjaminiHochberg, cauchyCombination, numpyMeanFloat32 } from './numeric/index.js';
import { pvalsFromLrtMeme } from './stats.js';
import { parseAlignmentSequences } from './preprocess/parse.js';
import { loadAlignmentAndTree } from './preprocess/assemble.js';
import { CODON_TO_AA } from './preprocess/tokenizer.js';

/**
 * @typedef {import('./preprocess/assemble.js').LoadedAlignment} LoadedAlignment
 */

/**
 * @typedef {{
 *   batch: number, N: number, d: Float32Array, z: Float32Array,
 *   siteIndices: Int32Array, phase: string, loaded: LoadedAlignment,
 *   taxonIndices?: Int32Array
 * }} PredictMeta
 */

/**
 * The model callback. `c`/`a` are [batch, N, 1] Int32Array tokens; return the graph's raw `lrt`
 * output, one number per batch element (the library clamps at 0 and stores float32).
 * @callback PredictFn
 * @param {Int32Array} c
 * @param {Int32Array} a
 * @param {PredictMeta} meta
 * @returns {Promise<ArrayLike<number>>|ArrayLike<number>}
 */

/**
 * @typedef {{start: number, end: number, k: number, d: number, p_local: number}} Patch
 */

// ---------------------------------------------------------------------------------------------
// numpy reductions
// ---------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------
// scan_hypergeometric_patches  (filter.py:52-100)
// ---------------------------------------------------------------------------------------------

/**
 * `scan_hypergeometric_patches(site_pvals, alpha_site=0.05, min_k=3, max_span=35,
 * p_local_thresh=0.01)`, filter.py:52-100. Candidate windows are every pair of significant sites
 * (i, j >= i + min_k - 1) spanning at most max_span codons with
 * p_local = 1 - hypergeom.cdf(k - 1, L, K, d) <= p_local_thresh (L = len(site_pvals),
 * K = #significant, d = span, k = significant sites in the window); overlapping candidates
 * (start <= current end) are merged with the minimum p_local and k recounted over the merged span.
 *
 * `1 - cdf` is computed literally (not the survival function) so the float64 value matches the
 * reference expression. A Float32Array input compares against float32(alpha_site), as numpy does
 * for a float32 array against a Python float — the cmd_meme path (cli.py:105, 119).
 *
 * @param {ArrayLike<number>} sitePvals
 * @param {{alphaSite?: number, minK?: number, maxSpan?: number, pLocalThresh?: number}} [options]
 * @returns {Patch[]}
 */
export function scanHypergeometricPatches(sitePvals, options = {}) {
	const { alphaSite = 0.05, minK = 3, maxSpan = 35, pLocalThresh = 0.01 } = options;
	const L = sitePvals.length;
	const alpha = sitePvals instanceof Float32Array ? Math.fround(alphaSite) : alphaSite;
	const sel = [];
	for (let i = 0; i < L; i++) if (sitePvals[i] <= alpha) sel.push(i);
	const K = sel.length;
	if (K < minK) return [];

	/** @type {Patch[]} */
	const candidates = [];
	for (let idxI = 0; idxI < K; idxI++) {
		const posI = sel[idxI];
		for (let idxJ = idxI + minK - 1; idxJ < K; idxJ++) {
			const posJ = sel[idxJ];
			const d = posJ - posI + 1;
			if (d > maxSpan) break;
			const k = idxJ - idxI + 1;
			const pLocal = 1.0 - hypergeomCdf(k - 1, L, K, d);
			if (pLocal <= pLocalThresh) candidates.push({ start: posI, end: posJ, k, d, p_local: pLocal });
		}
	}
	if (candidates.length === 0) return [];

	// list.sort(key=start) is stable; the candidates are already generated in start order.
	candidates.sort((x, y) => x.start - y.start);
	const countIn = (/** @type {number} */ s, /** @type {number} */ e) => {
		let n = 0;
		for (const p of sel) if (p >= s && p <= e) n++;
		return n;
	};
	/** @type {Patch[]} */
	const merged = [];
	let curr = { ...candidates[0] };
	for (let i = 1; i < candidates.length; i++) {
		const c = candidates[i];
		if (c.start <= curr.end) {
			curr.end = Math.max(curr.end, c.end);
			curr.p_local = Math.min(curr.p_local, c.p_local);
		} else {
			curr.d = curr.end - curr.start + 1;
			curr.k = countIn(curr.start, curr.end);
			merged.push(curr);
			curr = { ...c };
		}
	}
	curr.d = curr.end - curr.start + 1;
	curr.k = countIn(curr.start, curr.end);
	merged.push(curr);
	return merged;
}

// ---------------------------------------------------------------------------------------------
// The inference loop  (filter.py:155-165, 305-316; inference.py:162-192)
// ---------------------------------------------------------------------------------------------

/**
 * `predict_site_lrts`: run `predict` over the given sites (default: the variable sites, in
 * order) in batches of `batchSize`, clamp at 0, and return a float32 array of length L with 0 for
 * every site not predicted. Tokens are gathered from `loaded.c` / `loaded.a` unless `c` / `a`
 * override them (same [L, N, 1] layout).
 *
 * @param {LoadedAlignment} loaded
 * @param {PredictFn} predict
 * @param {{siteIndices?: ArrayLike<number>|null, batchSize?: number, phase?: string,
 *   d?: Float32Array, z?: Float32Array, c?: Int32Array, a?: Int32Array,
 *   onProgress?: (p: {phase: string, done: number, total: number}) => void}} [options]
 * @returns {Promise<Float32Array>}
 */
export async function predictSiteLrts(loaded, predict, options = {}) {
	const { L, N } = loaded;
	const c = options.c ?? loaded.c;
	const a = options.a ?? loaded.a;
	const d = options.d ?? loaded.d;
	const z = options.z ?? loaded.z;
	const phase = options.phase ?? 'baseline';
	const batchSize = Math.max(1, Math.floor(options.batchSize ?? 64));
	const lrts = new Float32Array(L);

	let sites;
	if (options.siteIndices != null) {
		sites = Int32Array.from(options.siteIndices);
	} else {
		const v = [];
		for (let s = 0; s < L; s++) if (!loaded.invariable[s]) v.push(s);
		sites = Int32Array.from(v);
	}
	const total = sites.length;
	for (let bStart = 0; bStart < total; bStart += batchSize) {
		const bEnd = Math.min(bStart + batchSize, total);
		const batch = bEnd - bStart;
		const idx = sites.subarray(bStart, bEnd);
		const cb = new Int32Array(batch * N);
		const ab = new Int32Array(batch * N);
		for (let k = 0; k < batch; k++) {
			const s = idx[k];
			cb.set(c.subarray(s * N, (s + 1) * N), k * N);
			ab.set(a.subarray(s * N, (s + 1) * N), k * N);
		}
		const y = await predict(cb, ab, { batch, N, d, z, siteIndices: idx, phase, loaded });
		if (y.length !== batch) {
			throw new Error(`predict returned ${y.length} values for a batch of ${batch} sites`);
		}
		for (let k = 0; k < batch; k++) lrts[idx[k]] = Math.fround(Math.max(0, y[k]));
		if (options.onProgress) options.onProgress({ phase, done: bEnd, total });
	}
	return lrts;
}

// ---------------------------------------------------------------------------------------------
// Consensus, OCI audit, masking  (filter.py:204-275; cli.py:131-181)
// ---------------------------------------------------------------------------------------------

/**
 * The per-site consensus codon, filter.py:204-209: over the taxa (in order) whose raw sequence
 * reaches codon s, the upper-cased codon; drop those containing '-' or 'N' (and, unless
 * `cliVariant`, those containing '?' or not of length 3); the mode with pandas' tie-break
 * (`Series.mode()` is sorted, so the lexicographically smallest codon wins); 'NNN' when nothing
 * is valid.
 *
 * @param {Map<string, string>} rawSeqs
 * @param {string[]} taxa
 * @param {number} L
 * @param {{cliVariant?: boolean}} [options]
 * @returns {string[]}
 */
export function consensusCodons(rawSeqs, taxa, L, options = {}) {
	const cli = options.cliVariant === true;
	const out = new Array(L);
	for (let s = 0; s < L; s++) {
		/** @type {Map<string, number>} */
		const counts = new Map();
		for (const t of taxa) {
			const seq = rawSeqs.get(t);
			if (seq === undefined || seq.length < (s + 1) * 3) continue;
			const cd = seq.slice(s * 3, (s + 1) * 3).toUpperCase();
			if (cd.includes('-') || cd.includes('N')) continue;
			if (!cli && (cd.includes('?') || cd.length !== 3)) continue;
			counts.set(cd, (counts.get(cd) ?? 0) + 1);
		}
		let best = null;
		let bestCount = -1;
		for (const [cd, n] of counts) {
			if (n > bestCount || (n === bestCount && best !== null && cd < best)) {
				best = cd;
				bestCount = n;
			}
		}
		out[s] = best === null ? 'NNN' : best;
	}
	return out;
}

/**
 * @typedef {{
 *   topTaxon: string|null, topRun: number, topMuts: number, totalPatchMuts: number, oci: number,
 *   isArtifact: boolean, maxRun: Map<string, number>, totalMuts: Map<string, number>
 * }} PatchAudit
 */

/**
 * The Outlier Contamination Index audit of one patch, filter.py:223-253 (cli.py:140-169). For
 * every taxon: the longest run of consecutive sites and the total count where the observed amino
 * acid and the consensus amino acid are both residues-or-stops (neither '-' nor '?';
 * `CODON_TO_AA.get(codon, '-')` so a stop '*' counts) and differ. The top taxon is
 * `max(taxa, key=(run, count))` — the FIRST taxon on ties; OCI = its count / (total + 1e-8);
 * artifact iff (run >= minRunLength and OCI >= minOci) or run >= 4.
 *
 * @param {Map<string, string>} rawSeqs
 * @param {string[]} taxa
 * @param {string[]} consensus
 * @param {number} start 0-indexed inclusive
 * @param {number} end 0-indexed inclusive
 * @param {{minOci?: number, minRunLength?: number}} [options]
 * @returns {PatchAudit}
 */
export function auditPatch(rawSeqs, taxa, consensus, start, end, options = {}) {
	const { minOci = 0.25, minRunLength = 3 } = options;
	/** @type {Map<string, number>} */
	const maxRun = new Map();
	/** @type {Map<string, number>} */
	const totalMuts = new Map();
	let totalPatchMuts = 0;
	for (const t of taxa) {
		const seq = rawSeqs.get(t);
		if (seq === undefined) continue;
		let run = 0;
		let maxR = 0;
		let mCnt = 0;
		for (let s = start; s <= end; s++) {
			const obsCd = seq.slice(s * 3, (s + 1) * 3).toUpperCase();
			const conCd = consensus[s];
			const obsAa = CODON_TO_AA.get(obsCd) ?? '-';
			const conAa = CODON_TO_AA.get(conCd) ?? '-';
			if (!'-?'.includes(obsAa) && !'-?'.includes(conAa) && obsAa !== conAa) {
				mCnt += 1;
				run += 1;
				maxR = Math.max(maxR, run);
			} else {
				run = 0;
			}
		}
		maxRun.set(t, maxR);
		totalMuts.set(t, mCnt);
		totalPatchMuts += mCnt;
	}

	let topTaxon = null;
	let topRun = 0;
	let topMuts = 0;
	for (const t of taxa) {
		const r = maxRun.get(t) ?? 0;
		const m = totalMuts.get(t) ?? 0;
		if (topTaxon === null || r > topRun || (r === topRun && m > topMuts)) {
			topTaxon = t;
			topRun = r;
			topMuts = m;
		}
	}
	const oci = topMuts / (totalPatchMuts + 1e-8);
	const isArtifact = (topRun >= minRunLength && oci >= minOci) || topRun >= 4;
	return { topTaxon, topRun, topMuts, totalPatchMuts, oci, isArtifact, maxRun, totalMuts };
}

/**
 * `cleaned_seqs[t][s*3:(s+1)*3] = ['N','N','N']` for s in start..end (filter.py:273-274): a
 * Python list slice assignment, so a codon beyond the end of a short sequence APPENDS (the slice
 * clamps to the current length and the three 'N's are inserted there), growing the sequence.
 * Mutates and returns `chars`.
 *
 * @param {string[]} chars
 * @param {number} start 0-indexed inclusive codon
 * @param {number} end 0-indexed inclusive codon
 * @returns {string[]}
 */
export function maskCodonSpan(chars, start, end) {
	for (let s = start; s <= end; s++) {
		const lo = Math.min(s * 3, chars.length);
		const hi = Math.min((s + 1) * 3, chars.length);
		chars.splice(lo, hi - lo, 'N', 'N', 'N');
	}
	return chars;
}

/**
 * `">{t}\n{seq}\n"` per entry, filter.py:293-294 / cli.py:189-192.
 * @param {Iterable<[string, string]>} entries
 * @returns {string}
 */
export function fastaText(entries) {
	let out = '';
	for (const [name, seq] of entries) out += `>${name}\n${seq}\n`;
	return out;
}

// ---------------------------------------------------------------------------------------------
// run_alignment_filter  (filter.py:102-399) and cmd_meme --filter (cli.py:114-221)
// ---------------------------------------------------------------------------------------------

/**
 * @typedef {{
 *   patch_start_1idx: number, patch_end_1idx: number, span_codons: number,
 *   significant_sites_k: number, p_hypergeom: number, outlier_taxon: string,
 *   consecutive_mismatches: number, outlier_contamination_index: number,
 *   outlier_aa_sequence: string, consensus_aa_sequence: string, mean_raw_patch_lrt: number
 * }} ArtifactRecord
 */

/**
 * @typedef {{
 *   start: number, end: number, span: number, outlier_taxon: string,
 *   consecutive_mismatches: number, oci: number
 * }} CliArtifactRecord
 */

/**
 * @typedef {{cct_p_value: number, sig_sites_p05: number, sig_sites_q10: number, mean_lrt: number}} FilterMetrics
 */

/**
 * @typedef {{
 *   loaded: LoadedAlignment, lrts: Float32Array,
 *   pvals: Float64Array|Float32Array, qvals: Float64Array|Float32Array
 * }} FilterPass
 */

/**
 * @typedef {{
 *   num_taxa: number, num_codons: number, num_patches_detected: number,
 *   num_artifacts_masked: number, masked_codons_count: number,
 *   raw_metrics: FilterMetrics, cleaned_metrics: FilterMetrics,
 *   suppressed_spurious_sites: number, artifacts: ArtifactRecord[], elapsed_seconds: number,
 *   output_alignment: null, audit_csv: null,
 *   patches: Patch[], artifacts_masked: CliArtifactRecord[],
 *   masked_codon_ranges_1idx_by_taxon: Record<string, number[][]>,
 *   sites: {site: number, hyphaeon_lrt: number, p_value: number, q_value: number,
 *     is_invariable: boolean, raw_lrt: number, raw_p_value: number, raw_q_value: number}[],
 *   raw: FilterPass,
 *   cleaned: (FilterPass & {sequences: Map<string, string>, fastaText: string})|null,
 *   notices: {cliVariant: boolean, baseline: LoadedAlignment['notices'],
 *     cleaned: LoadedAlignment['notices']|null, baseLrtsSupplied: boolean, treeSource: string}
 * }} FilterResult
 */

/**
 * `run_alignment_filter` (filter.py:102-399) on strings, with `{cliVariant: true}` selecting the
 * cmd_meme copy (see header). Phases, with the reference lines:
 *
 *   1. load_alignment_and_tree(alignment, tree, max_species, prune_duplicates=True)   133-135
 *      — or `input.loaded` if the runtime already has it;
 *   2. baseline inference over the variable sites (predict), float32, clamped          155-165
 *      — or `input.baseLrts` if the runtime already has them (cmd_meme reuses its own);
 *   3. MEME p, BH q, CCT, counts                                                        185-189
 *   4. scan_hypergeometric_patches                                                     196-198
 *   5. consensus codons; OCI audit per patch; artifact records; NNN masking            204-275
 *   6. cleaned FASTA -> load_alignment_and_tree again (tree = treeText, else the        286-338
 *      alignment text itself) -> inference -> p, q, CCT, counts; skipped with 0 artifacts
 *   7. the result dict                                                                  375-399
 *
 * `metrics.sig_sites_*` and `suppressed_spurious_sites` follow filter.py even in the CLI
 * variant (whose own printout uses q <= 0.05/0.10 and p <= 0.05/0.10 counts; those are
 * recoverable from `sites`).
 *
 * @param {{alignmentText: string, treeText?: string|null, loaded?: LoadedAlignment|null,
 *   baseLrts?: ArrayLike<number>|null}} input
 * @param {PredictFn} predict
 * @param {{alphaSite?: number, minK?: number, maxSpan?: number, pLocalThresh?: number,
 *   minOci?: number, minRunLength?: number, maxSpecies?: number|null, pruneDuplicates?: boolean,
 *   batchSize?: number, cliVariant?: boolean,
 *   onProgress?: (p: {phase: string, done: number, total: number}) => void}} [options]
 * @returns {Promise<FilterResult>}
 */
export async function runAlignmentFilter(input, predict, options = {}) {
	const {
		alphaSite = 0.05,
		minK = 3,
		maxSpan = 35,
		pLocalThresh = 0.01,
		minOci = 0.25,
		minRunLength = 3,
		maxSpecies = null,
		pruneDuplicates = true,
		batchSize = 64,
		cliVariant = false,
		onProgress
	} = options;
	const t0 = Date.now();
	const alignmentText = input.alignmentText;
	const treeText = input.treeText ?? null;

	// 1-2. Alignment, raw sequences, baseline LRTs.
	const loaded =
		input.loaded ?? loadAlignmentAndTree(alignmentText, treeText, { maxSpecies, pruneDuplicates });
	const rawSeqs = parseAlignmentSequences(alignmentText);
	const { taxa, L } = loaded;
	const N = taxa.length;

	/** @type {Float32Array} */
	let lrtsRaw;
	if (input.baseLrts != null) {
		if (input.baseLrts.length !== L) {
			throw new Error(`baseLrts has ${input.baseLrts.length} entries for ${L} codons`);
		}
		lrtsRaw = Float32Array.from(input.baseLrts);
	} else {
		lrtsRaw = await predictSiteLrts(loaded, predict, { batchSize, phase: 'baseline', onProgress });
	}

	// 3. Statistics. The cmd_meme copy casts p and q to float32 (cli.py:105-106).
	const stats = (/** @type {Float32Array} */ lrts) => {
		const p64 = pvalsFromLrtMeme(lrts);
		const pvals = cliVariant ? Float32Array.from(p64) : p64;
		const qvals = benjaminiHochberg(pvals);
		const p05 = cliVariant ? Math.fround(0.05) : 0.05;
		const q10 = cliVariant ? Math.fround(0.1) : 0.1;
		let sigP05 = 0;
		let sigQ10 = 0;
		for (let i = 0; i < L; i++) {
			if (pvals[i] <= p05) sigP05++;
			if (qvals[i] <= q10) sigQ10++;
		}
		return {
			pvals,
			qvals,
			metrics: {
				cct_p_value: cauchyCombination(pvals),
				sig_sites_p05: sigP05,
				sig_sites_q10: sigQ10,
				mean_lrt: numpyMeanFloat32(lrts)
			}
		};
	};
	const raw = stats(lrtsRaw);

	// 4. Spatial scan. cmd_meme fixes alpha 0.05 / min_k 3 / max_span 35 (cli.py:122-124).
	const patches = scanHypergeometricPatches(
		raw.pvals,
		cliVariant
			? { alphaSite: 0.05, minK: 3, maxSpan: 35, pLocalThresh }
			: { alphaSite, minK, maxSpan, pLocalThresh: 0.01 }
	);

	// 5. Consensus, OCI audit, masking.
	/** @type {ArtifactRecord[]} */
	const artifacts = [];
	/** @type {CliArtifactRecord[]} */
	const artifactsMasked = [];
	/** @type {Record<string, number[][]>} */
	const maskedRanges = {};
	let maskedCodonsTotal = 0;
	/** @type {Map<string, string[]>} */
	const cleanedSeqs = new Map();
	if (cliVariant) {
		// cli.py:130: only the matched taxa, in taxa order; computed only when there are patches.
		if (patches.length > 0) {
			for (const t of taxa) {
				const seq = rawSeqs.get(t);
				if (seq !== undefined) cleanedSeqs.set(t, Array.from(seq));
			}
		}
	} else {
		for (const [t, seq] of rawSeqs) cleanedSeqs.set(t, Array.from(seq));
	}
	const consensus = patches.length > 0 || !cliVariant ? consensusCodons(rawSeqs, taxa, L, { cliVariant }) : [];
	const effMinOci = cliVariant ? 0.25 : minOci;
	const effMinRun = cliVariant ? 3 : minRunLength; // cli.py:138 `min_patch_consec` has no flag

	for (const p of patches) {
		if (taxa.length === 0) continue;
		const audit = auditPatch(rawSeqs, taxa, consensus, p.start, p.end, {
			minOci: effMinOci,
			minRunLength: effMinRun
		});
		if (!audit.isArtifact || audit.topTaxon === null) continue;
		const topTax = audit.topTaxon;
		const topSeq = rawSeqs.get(topTax) ?? '';
		let obs = '';
		let con = '';
		for (let s = p.start; s <= p.end; s++) {
			obs += CODON_TO_AA.get(topSeq.slice(s * 3, (s + 1) * 3).toUpperCase()) ?? '-';
			con += CODON_TO_AA.get(consensus[s]) ?? '-';
		}
		artifacts.push({
			patch_start_1idx: p.start + 1,
			patch_end_1idx: p.end + 1,
			span_codons: p.d,
			significant_sites_k: p.k,
			p_hypergeom: p.p_local,
			outlier_taxon: topTax,
			consecutive_mismatches: audit.topRun,
			outlier_contamination_index: audit.oci,
			outlier_aa_sequence: obs,
			consensus_aa_sequence: con,
			mean_raw_patch_lrt: numpyMeanFloat32(lrtsRaw, p.start, p.end - p.start + 1)
		});
		artifactsMasked.push({
			start: p.start + 1,
			end: p.end + 1,
			span: p.d,
			outlier_taxon: topTax,
			consecutive_mismatches: audit.topRun,
			oci: audit.oci
		});
		(maskedRanges[topTax] ??= []).push([p.start + 1, p.end + 1]);
		const chars = cleanedSeqs.get(topTax);
		if (chars === undefined) {
			// Unreachable in the reference too (a taxon with mismatches is in raw_seqs); a KeyError there.
			throw new Error(`outlier taxon ${topTax} has no raw sequence`);
		}
		maskCodonSpan(chars, p.start, p.end);
		maskedCodonsTotal += p.end - p.start + 1;
	}

	// 6. Cleaned re-evaluation.
	const numArtifacts = artifacts.length;
	/** @type {FilterResult['cleaned']} */
	let cleaned = null;
	let cleanedStats = raw;
	let cleanedLrts = lrtsRaw;
	const treeSource = cliVariant ? (treeText === null ? 'none' : 'tree') : (treeText === null ? 'alignment' : 'tree');
	if (numArtifacts > 0) {
		/** @type {Map<string, string>} */
		const sequences = new Map();
		for (const [t, chars] of cleanedSeqs) sequences.set(t, chars.join(''));
		const text = fastaText(sequences);
		// filter.py:297 falls back to the alignment (its embedded tree); cli.py:195 passes args.tree
		// as is, so a null tree raises in the loader — the cmd_meme crash, reproduced.
		const effectiveTree = cliVariant ? treeText : (treeText ?? alignmentText);
		const loadedCl = loadAlignmentAndTree(text, effectiveTree, { maxSpecies, pruneDuplicates });
		if (loadedCl.L !== L) {
			throw new Error(`cleaned alignment has ${loadedCl.L} codons, expected ${L}`);
		}
		// cli.py:198-200 re-scores with the ORIGINAL tree cache.
		const dz = cliVariant ? { d: loaded.d, z: loaded.z } : {};
		if (cliVariant && loadedCl.N !== loaded.N) {
			throw new Error(
				`cleaned alignment has ${loadedCl.N} taxa but the baseline tree cache has ${loaded.N} (cmd_meme reuses the baseline cache; the reference fails here)`
			);
		}
		cleanedLrts = await predictSiteLrts(loadedCl, predict, { batchSize, phase: 'cleaned', onProgress, ...dz });
		cleanedStats = stats(cleanedLrts);
		cleaned = {
			loaded: loadedCl,
			lrts: cleanedLrts,
			pvals: cleanedStats.pvals,
			qvals: cleanedStats.qvals,
			sequences,
			fastaText: text
		};
	}

	// 7. Result.
	const finalLoaded = cleaned ? cleaned.loaded : loaded;
	const sites = new Array(L);
	for (let i = 0; i < L; i++) {
		sites[i] = {
			site: i + 1,
			hyphaeon_lrt: cleanedLrts[i],
			p_value: cleanedStats.pvals[i],
			q_value: cleanedStats.qvals[i],
			is_invariable: finalLoaded.invariable[i] === 1,
			raw_lrt: lrtsRaw[i],
			raw_p_value: raw.pvals[i],
			raw_q_value: raw.qvals[i]
		};
	}

	return {
		num_taxa: N,
		num_codons: L,
		num_patches_detected: patches.length,
		num_artifacts_masked: numArtifacts,
		masked_codons_count: maskedCodonsTotal,
		raw_metrics: raw.metrics,
		cleaned_metrics: { ...cleanedStats.metrics },
		suppressed_spurious_sites: Math.max(0, raw.metrics.sig_sites_p05 - cleanedStats.metrics.sig_sites_p05),
		artifacts,
		elapsed_seconds: (Date.now() - t0) / 1000,
		output_alignment: null,
		audit_csv: null,
		patches,
		artifacts_masked: artifactsMasked,
		masked_codon_ranges_1idx_by_taxon: maskedRanges,
		sites,
		raw: { loaded, lrts: lrtsRaw, pvals: raw.pvals, qvals: raw.qvals },
		cleaned,
		notices: {
			cliVariant,
			baseline: loaded.notices,
			cleaned: cleaned ? cleaned.loaded.notices : null,
			baseLrtsSupplied: input.baseLrts != null,
			treeSource
		}
	};
}
