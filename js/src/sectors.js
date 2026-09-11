/**
 * WHY THIS FILE EXISTS
 *
 * Mirrors the sector-mining half of `hyphaeon/epistasis.py` at veg/HyphAeon cf838ab:
 *
 *   REV_AA_MAP                     epistasis.py:41       `{v: k for k, v in AA_MAP.items()}`
 *   computeSectorPermutationTest   compute_sector_permutation_test, epistasis.py:224-307
 *   extractEpistaticSectorsTse     extract_epistatic_sectors_tse,   epistasis.py:309-448
 *
 * The co-selection graph itself (`compute_branch_coselection_network`, epistasis.py:120-222) is
 * another module's; this file takes the graph the fixtures record — `{nodes: [[id, attrs]],
 * edges: [[u, v, {weight, ...}]]}` in insertion order — or an adjacency Map, or an object carrying
 * one as `adj` (epistasis.js's CoselectionGraph), and the float32 attribution matrix [L, N] as
 * rows or as `{data, rows, cols}`.
 *
 * extract_epistatic_sectors_tse, step by step (line numbers of epistasis.py):
 *    1. no edges -> []                                                                      327-328
 *    2. sub_nodes = nodes with degree > 0, in node order                                    330
 *       fewer than min_clique_size -> connected components of G.subgraph(sub_nodes), size >= 2  331-332
 *       else greedy_modularity_communities(G.subgraph(sub_nodes), weight='weight'), size >= 2;
 *       any exception -> the components instead                                             333-339
 *       (numeric/graph.js supplies both with networkx 3.6.1 ordering, including the subgraph
 *       view's set-order iteration; see its header)
 *    3. per community, members sorted ascending, 0-indexed; sub_A = attributions[sites]      343-345
 *    4. spectral coherence: Gram = sub_A @ sub_A.T (float32), eigh, clamp >= 0,
 *       C = lambda_max / sum(lambda) (0 when the sum is 0); |top eigenvector| >= 0.10 keeps a
 *       site; with >= 2 kept, recompute C on the kept rows and adopt them as the sector; a
 *       community of < 2 rows or an all-zero sub_A has C = 1.0                              348-364
 *    5. drop when C < min_coherence or fewer than 2 sites                                    367-368
 *    6. permutation test on the (pruned) sites with active_only=True                        371-379
 *    7. drop when max_perm_p is set and p_perm exceeds it                                     382-383
 *    8. shared_taxa = columns where every UNPRUNED community row is > 0 (sub_A is not
 *       reassigned after pruning — replicated); mean_lrt = float32 mean over the pruned sites 385-386
 *    9. pars_signature "[ A3 - B7 - C11 ]" over the first 10 sorted sites                    388-390
 *   10. focal taxon: first taxon whose lower-cased name contains the lower-cased query;
 *       token < 20 -> REV_AA_MAP (else '-'), otherwise the consensus residue; a differing,
 *       non-gap residue gets "*" and a "C3->F" entry                                          393-413
 *   11. the record; sector_id counts accepted sectors in community order                     415-435
 *   12. sort by (spectral_coherence, size) descending, stable                                 437
 *
 * compute_sector_permutation_test (224-304): pool = sites whose float32 L2 norm > 1e-9 (or all
 * sites); pool < K, K < 2 or B <= 0 -> the degenerate record; else B draws of K distinct pool sites,
 * each scored C = max(lambda_max, 0) / trace of the float32 Gram (1/K when trace <= 1e-9);
 * p_perm = #(C_null >= C_obs - 1e-7) / B; null mean, population std (ddof 0) and 95th percentile
 * (numpy linear interpolation); isotropic_baseline 1/K.
 *
 * QUIRKS REPLICATED (all pinned by tests or noted in the header of test/sectors.test.js):
 *   - `max_overlap` is accepted and never read: epistasis.py:315 declares it, nothing in 306-445
 *     uses it (no Jaccard suppression exists in the reference). Same here.
 *   - shared_taxa / shared_branches count over the unpruned community rows (step 8).
 *   - `focal_idx` defaults to 0 but the focal fields appear only when a name matched.
 *   - numpy 2 weak-scalar comparisons: `v_dom >= 0.10`, `traces > 1e-9` and
 *     `coherences >= observed - 1e-7` compare float32 arrays with Python floats, which NEP 50 casts
 *     to float32 first — so the thresholds here are Math.fround(0.10), Math.fround(1e-9) and
 *     Math.fround(observed - 1e-7).
 *   - `lrts` and `attributions` are treated as float32 (the dtype the pipeline produces,
 *     epistasis.py:74-112): mean_lrt is numpy's float32 pairwise mean; the Gram, eigenvalues,
 *     traces and coherences are float32 values.
 *
 * FLOAT PRECISION. The reference forms `sub_A @ sub_A.T` with float32 BLAS and takes float32
 * eigenvalues (ssyevd). Here each Gram entry is a float64 dot product rounded to float32, the
 * eigenproblem runs in float64 (symmetricEigen / largestEigenvalue) and the eigenvalues are rounded
 * to float32 before the clamp, the float32 sum and the float32 divide, so a coherence is a
 * float32-representable number like the reference's `float(np.float32)`. Measured on every sector
 * of fixtures/epistasis/extract_epistatic_sectors_tse.json the two sides agree EXACTLY (max |dC|
 * 0.0, the test prints it); the class stays the fixture's 1e-6 because BLAS accumulation order is
 * not a contract. Degenerate top eigenvalues make the pruning eigenvector basis-dependent on both
 * sides; no convention can fix that and none is attempted.
 *
 * RANDOMNESS. numpy's PCG64 stream cannot be reproduced; the draws come from Xoshiro256 seeded with
 * `seed` (default 42, the reference's rng_seed default) and `choiceWithoutReplacement`, and the
 * outputs are compared statistically (PLAN.md §5.4: |dp| <= 3·sqrt(p(1-p)/B), null moments within
 * 2 %). Measured at B = 200,000 on the fixture's planted_sector_3_7_11 and planted_pair_21_22 pools
 * (three seeds each side): null mean 0.6085 vs numpy 0.6086 and 0.7235 vs 0.7232, std 0.0910 vs
 * 0.0909 and 0.1012 vs 0.1014, 95th percentile 0.7646 vs 0.7646 and 0.8789 vs 0.8789 — the two
 * generators estimate the same null; the fixtures' B = 5,000 values differ from these by up to
 * 1.9 % because C(38, 3) = 8,436 distinct subsets make the null discrete and its upper quantiles
 * step. A null or non-integer seed throws — `rng_seed=None` means fresh entropy in numpy and the
 * library has no unseeded generator (numeric/prng.js).
 *
 * WHAT IT DELIBERATELY DOES NOT DO. No batching of the null (the reference batches <= 25,000 draws
 * for numpy's sake; here each draw forms one K x K Gram and one eigenvalue call, ~22 µs per matrix
 * per numeric/linalg.js); no networkx graph object; no model, tree or I/O.
 */

import { AA_MAP } from './preprocess/tokenizer.js';
import { symmetricEigen } from './preprocess/symmetricEigen.js';
import { largestEigenvalue } from './numeric/linalg.js';
import { Xoshiro256 } from './numeric/prng.js';
import { numpyPairwiseSum, float32Sum, numpyMeanFloat32 } from './numeric/reduce.js';
import {
	adjacencyFromGraph,
	inducedSubgraph,
	connectedComponents,
	greedyModularityCommunities,
	numberOfEdges,
	degree
} from './numeric/graph.js';

/**
 * epistasis.py:41 `REV_AA_MAP = {v: k for k, v in AA_MAP.items()}`: token 0..19 -> residue.
 * @type {Map<number, string>}
 */
export const REV_AA_MAP = (() => {
	/** @type {Map<number, string>} */
	const m = new Map();
	for (const [aa, tok] of AA_MAP) m.set(tok, aa);
	return m;
})();

/**
 * @typedef {ArrayLike<ArrayLike<number>> | ArrayLike<number> | {data: ArrayLike<number>, rows: number, cols: number}} Matrix
 *   a matrix as rows (arrays or typed arrays), as a flat row-major buffer with its shape, or as a
 *   bare flat row-major buffer when the caller passes `N` (the third form is what `epistasis.js`
 *   hands over: `leafAttributions` is a Float32Array [L*N] and `computeBranchCoselectionNetwork`
 *   takes the same three shapes, so the two ports chain with no wrapper — see the note below)
 */

/**
 * The float32 [L, N] matrix behind any of the three input shapes.
 *
 * WHY THE FLAT FORM TAKES A HINT. `computeTransformerAttributions` returns `leafAttributions` as a
 * flat Float32Array because that is how the reference stores it and how a runtime receives it from
 * the graph; a flat buffer carries no shape, and `[L*N]` and `[N*L]` are the same length. The
 * caller's `N` (`options.N`, the taxon count it already has) settles it. Without a hint a flat
 * buffer is ambiguous and is rejected by name rather than read as a list of rows, which is what
 * produced `expected NaN*undefined` when the two ports were first chained.
 *
 * @param {Matrix} m
 * @param {string} name
 * @param {number} [cols] N, when `m` is a bare flat buffer
 * @returns {{data: Float32Array, L: number, N: number}}
 */
function asFloat32Matrix(m, name, cols) {
	if (m && typeof m === 'object' && 'data' in m && 'rows' in m && 'cols' in m) {
		const L = m.rows;
		const N = m.cols;
		if (m.data.length !== L * N) throw new RangeError(`${name}: data has ${m.data.length} entries, expected ${L}*${N}`);
		return { data: m.data instanceof Float32Array ? m.data : Float32Array.from(m.data), L, N };
	}
	if (isFlat(m)) {
		const flat = /** @type {ArrayLike<number>} */ (m);
		if (!Number.isInteger(cols) || /** @type {number} */ (cols) <= 0) {
			throw new RangeError(`${name}: a flat buffer of ${flat.length} values needs its shape — pass options.N, or {data, rows, cols}`);
		}
		const N = /** @type {number} */ (cols);
		if (flat.length % N !== 0) throw new RangeError(`${name}: ${flat.length} values is not a whole number of rows of ${N}`);
		return { data: flat instanceof Float32Array ? flat : Float32Array.from(flat), L: flat.length / N, N };
	}
	const rows = /** @type {ArrayLike<ArrayLike<number>>} */ (m);
	const L = rows.length;
	const N = L > 0 ? rows[0].length : 0;
	const data = new Float32Array(L * N);
	for (let i = 0; i < L; i++) {
		const r = rows[i];
		if (r.length !== N) throw new RangeError(`${name}: row ${i} has ${r.length} entries, expected ${N}`);
		for (let j = 0; j < N; j++) data[i * N + j] = r[j];
	}
	return { data, L, N };
}

/**
 * True when `m` is a flat buffer of numbers rather than a list of rows. An empty array-like is
 * flat by this test; `asFloat32Matrix` then needs a hint, and `asIntMatrix` yields an empty matrix
 * either way.
 * @param {Matrix} m
 * @returns {boolean}
 */
function isFlat(m) {
	if (ArrayBuffer.isView(m)) return true;
	const a = /** @type {ArrayLike<unknown>} */ (m);
	return a.length === 0 || typeof a[0] === 'number';
}

/**
 * Integer token matrix [L, N] (a_np) behind any of the three input shapes.
 * @param {Matrix} m
 * @param {number} [cols] N, when `m` is a bare flat buffer (the `a` tokens of a LoadedAlignment)
 * @returns {{data: Int32Array, L: number, N: number}}
 */
function asIntMatrix(m, cols) {
	if (m && typeof m === 'object' && 'data' in m && 'rows' in m && 'cols' in m) {
		return { data: m.data instanceof Int32Array ? m.data : Int32Array.from(m.data), L: m.rows, N: m.cols };
	}
	if (isFlat(m)) {
		const flat = /** @type {ArrayLike<number>} */ (m);
		if (!Number.isInteger(cols) || /** @type {number} */ (cols) <= 0) {
			throw new RangeError(`a_np: a flat buffer of ${flat.length} values needs its shape — pass options.N, or {data, rows, cols}`);
		}
		const N = /** @type {number} */ (cols);
		if (flat.length % N !== 0) throw new RangeError(`a_np: ${flat.length} values is not a whole number of rows of ${N}`);
		return { data: flat instanceof Int32Array ? flat : Int32Array.from(flat), L: flat.length / N, N };
	}
	const rows = /** @type {ArrayLike<ArrayLike<number>>} */ (m);
	const L = rows.length;
	const N = L > 0 ? rows[0].length : 0;
	const data = new Int32Array(L * N);
	for (let i = 0; i < L; i++) for (let j = 0; j < N; j++) data[i * N + j] = rows[i][j];
	return { data, L, N };
}

/**
 * The float32 Gram matrix of a set of rows — `sub_A @ sub_A.T` on a float32 array: each entry a
 * float64 dot product rounded to float32 (within a float32 ulp of BLAS sgemm's own accumulation).
 *
 * @param {Float32Array} A row-major [L, N]
 * @param {number} N
 * @param {ArrayLike<number>} sites row indices
 * @returns {Float64Array} row-major K x K, float32-valued
 */
function float32Gram(A, N, sites) {
	const K = sites.length;
	const G = new Float64Array(K * K);
	for (let i = 0; i < K; i++) {
		const oi = sites[i] * N;
		for (let j = 0; j <= i; j++) {
			const oj = sites[j] * N;
			let s = 0;
			for (let n = 0; n < N; n++) s += A[oi + n] * A[oj + n];
			const v = Math.fround(s);
			G[i * K + j] = v;
			G[j * K + i] = v;
		}
	}
	return G;
}

/** `np.linalg.norm(row)` in float32: float32 squares, pairwise float32 sum, float32 sqrt. */
function float32RowNorm(A, N, site) {
	const sq = new Float32Array(N);
	const o = site * N;
	for (let n = 0; n < N; n++) sq[n] = A[o + n] * A[o + n];
	return Math.fround(Math.sqrt(numpyPairwiseSum(sq, 0, N, Math.fround)));
}

/**
 * `float(np.percentile(x, q))` for a float32 array, method 'linear' (numpy 2.3
 * `_quantile`: virtual index `n*q + (1 - q) - 1`, `_lerp` with the `t >= 0.5` branch); the result
 * is rounded to float32 as numpy's is for float32 input.
 *
 * @param {Float32Array} x
 * @param {number} q in percent
 * @returns {number}
 */
export function float32Percentile(x, q) {
	const n = x.length;
	if (n === 0) return NaN;
	const sorted = Float32Array.from(x).sort();
	const quantile = q / 100;
	const virtual = n * quantile + (1 - quantile) - 1;
	let prev = Math.floor(virtual);
	if (prev < 0) prev = 0;
	if (prev > n - 1) prev = n - 1;
	let next = prev + 1;
	if (next > n - 1) next = n - 1;
	const gamma = virtual - prev;
	const a = sorted[prev];
	const b = sorted[next];
	const diff = Math.fround(b - a);
	let r = Math.fround(a + Math.fround(diff * gamma));
	if (gamma >= 0.5) r = Math.fround(b - Math.fround(diff * (1 - gamma)));
	return r;
}

/**
 * `float(np.std(x))` for a float32 array: float32 mean, float32 deviations and squares, float32
 * pairwise sum, population divisor.
 * @param {Float32Array} x
 * @returns {number}
 */
function float32Std(x) {
	const n = x.length;
	const mean = Math.fround(numpyPairwiseSum(x, 0, n, Math.fround) / n);
	const sq = new Float32Array(n);
	for (let i = 0; i < n; i++) {
		const d = Math.fround(x[i] - mean);
		sq[i] = d * d;
	}
	return Math.fround(Math.sqrt(Math.fround(numpyPairwiseSum(sq, 0, n, Math.fround) / n)));
}

/**
 * @typedef {{
 *   p_perm: number, null_coherence_mean: number, null_coherence_std: number,
 *   null_coherence_95: number, isotropic_baseline: number
 * }} PermutationStats
 */

/**
 * `compute_sector_permutation_test(attributions, site_indices, observed_coherence, n_permutations=10000,
 * active_only=True, rng_seed=42)` (epistasis.py:224-307).
 *
 * @param {Matrix} attributions float32 [L, N]
 * @param {ArrayLike<number>} siteIndices 0-indexed sites of the sector (K of them)
 * @param {number} observedCoherence C(S_obs)
 * @param {{nPermutations?: number, activeOnly?: boolean, seed?: number | bigint, N?: number}} [options]
 *   nPermutations default 10000; activeOnly default true (pool = sites with norm > 1e-9);
 *   seed default 42 — the Xoshiro256 seed standing in for numpy's rng_seed; N the taxon count,
 *   required only when `attributions` is a bare flat buffer
 * @returns {PermutationStats}
 */
export function computeSectorPermutationTest(attributions, siteIndices, observedCoherence, options = {}) {
	const nPermutations = options.nPermutations ?? 10000;
	const activeOnly = options.activeOnly ?? true;
	const seed = options.seed === undefined ? 42 : options.seed;
	if (seed === null || (typeof seed !== 'bigint' && !Number.isSafeInteger(seed))) {
		throw new TypeError(`computeSectorPermutationTest: seed must be an integer (numpy's rng_seed=None has no counterpart), got ${String(seed)}`);
	}
	const { data: A, L, N } = asFloat32Matrix(attributions, 'attributions', options.N);
	const K = siteIndices.length;

	/** @type {number[]} */
	const pool = [];
	if (activeOnly) {
		const threshold = Math.fround(1e-9);
		for (let s = 0; s < L; s++) if (float32RowNorm(A, N, s) > threshold) pool.push(s);
	} else {
		for (let s = 0; s < L; s++) pool.push(s);
	}

	if (pool.length < K || K < 2 || nPermutations <= 0) {
		return {
			p_perm: 1.0,
			null_coherence_mean: observedCoherence,
			null_coherence_std: 0.0,
			null_coherence_95: observedCoherence,
			isotropic_baseline: K > 0 ? 1.0 / K : 1.0
		};
	}

	const rng = new Xoshiro256(seed);
	const isotropic32 = Math.fround(1.0 / K);
	const traceThreshold = Math.fround(1e-9);
	const cutoff = Math.fround(observedCoherence - 1e-7);
	const nullCoherences = new Float32Array(nPermutations);
	let greaterEqual = 0;
	const sites = new Int32Array(K);
	for (let b = 0; b < nPermutations; b++) {
		const draw = rng.choiceWithoutReplacement(pool.length, K);
		for (let k = 0; k < K; k++) sites[k] = pool[draw[k]];
		const G = float32Gram(A, N, sites);
		const diag = new Float32Array(K);
		for (let k = 0; k < K; k++) diag[k] = G[k * K + k];
		const trace = float32Sum(diag);
		let coherence;
		if (trace > traceThreshold) {
			const eig = Math.max(Math.fround(largestEigenvalue(G, K)), 0);
			coherence = Math.fround(eig / trace);
		} else {
			coherence = isotropic32;
		}
		nullCoherences[b] = coherence;
		if (coherence >= cutoff) greaterEqual++;
	}

	return {
		p_perm: greaterEqual / nPermutations,
		null_coherence_mean: numpyMeanFloat32(nullCoherences),
		null_coherence_std: float32Std(nullCoherences),
		null_coherence_95: float32Percentile(nullCoherences, 95),
		isotropic_baseline: 1.0 / K
	};
}

/**
 * Spectral coherence of a set of rows (epistasis.py:352-367, one pass): the float32 eigenvalues of
 * the Gram, clamped, C = lambda_max / sum, plus |top eigenvector| for the pruning step.
 *
 * @param {Float32Array} A
 * @param {number} N
 * @param {number[]} sites
 * @returns {{coherence: number, vDom: Float64Array}}
 */
function spectralCoherence(A, N, sites) {
	const K = sites.length;
	const G = float32Gram(A, N, sites);
	const { values, vectors } = symmetricEigen(G, K);
	const eig32 = new Float32Array(K);
	for (let k = 0; k < K; k++) eig32[k] = Math.max(Math.fround(values[k]), 0);
	const tot = float32Sum(eig32);
	const coherence = tot > 0 ? Math.fround(eig32[K - 1] / tot) : 0.0;
	const vDom = new Float64Array(K);
	for (let i = 0; i < K; i++) vDom[i] = Math.fround(Math.abs(vectors[i * K + (K - 1)]));
	return { coherence, vDom };
}

/**
 * @typedef {{
 *   sector_id: number, size: number, sites: number[], spectral_coherence: number, p_perm: number,
 *   null_coherence_mean: number, null_coherence_std: number, null_coherence_95: number,
 *   isotropic_baseline: number, shared_taxa: number, shared_branches: number, mean_lrt: number,
 *   pars_signature: string, consensus_signature: string,
 *   focal_taxon?: string, focal_signature?: string, focal_mutations?: string[], focal_mutations_count?: number
 * }} Sector
 */

/**
 * `extract_epistatic_sectors_tse(G, attributions, lrts, consensus_aas, min_clique_size=3,
 * max_overlap=0.50, min_coherence=0.50, focal_taxon=None, a_np=None, taxa=None, n_permutations=10000,
 * max_perm_p=None, rng_seed=42)` (epistasis.py:309-448).
 *
 * @param {import('./numeric/graph.js').GraphJson | import('./numeric/graph.js').Adjacency | {adj: import('./numeric/graph.js').Adjacency}} graph
 *   the co-selection graph: nodes are 1-indexed sites, edges carry `weight`; the fixtures' JSON
 *   shape, an adjacency Map, or any object with an `adj` Map (epistasis.js's CoselectionGraph)
 * @param {Matrix} attributions float32 [L, N]
 * @param {ArrayLike<number>} lrts float32 [L]
 * @param {ArrayLike<string>} consensusAas [L] one residue per site
 * @param {{
 *   minCliqueSize?: number, maxOverlap?: number, minCoherence?: number,
 *   focalTaxon?: string | null, aNp?: Matrix | null, taxa?: ArrayLike<string> | null,
 *   nPermutations?: number, maxPermP?: number | null, seed?: number | bigint, N?: number
 * }} [options] the Python keyword arguments; `maxOverlap` is accepted and unused, as in the
 *   reference. `N` is the taxon count, required only when `attributions` (or `aNp`) is a bare flat
 *   buffer — which is what `computeTransformerAttributions` and a `LoadedAlignment`'s `a` are, so
 *   the whole pillar chains as
 *   `extractEpistaticSectorsTse(graph, attr.leafAttributions, attr.lrts, attr.consensusAas,
 *   {aNp: loaded.a, taxa, N: loaded.N})`
 * @returns {Sector[]}
 */
export function extractEpistaticSectorsTse(graph, attributions, lrts, consensusAas, options = {}) {
	const minCliqueSize = options.minCliqueSize ?? 3;
	const minCoherence = options.minCoherence ?? 0.5;
	const focalTaxon = options.focalTaxon ?? null;
	const taxa = options.taxa ?? null;
	const nPermutations = options.nPermutations ?? 10000;
	const maxPermP = options.maxPermP ?? null;
	const seed = options.seed ?? 42;
	void options.maxOverlap; // epistasis.py:315 declares max_overlap and never reads it

	const adj = adjacencyFromGraph(graph);
	if (numberOfEdges(adj) === 0) return [];

	const { data: A, N } = asFloat32Matrix(attributions, 'attributions', options.N);
	const lrts32 = lrts instanceof Float32Array ? lrts : Float32Array.from(lrts);
	const aNp = options.aNp ? asIntMatrix(options.aNp, options.N ?? N) : null;

	/** @type {import('./numeric/graph.js').NodeId[]} */
	const subNodes = [];
	for (const n of adj.keys()) if (degree(adj, n) > 0) subNodes.push(n);
	const gSub = inducedSubgraph(adj, subNodes);

	/** @type {import('./numeric/graph.js').NodeId[][]} */
	let components;
	if (subNodes.length < minCliqueSize) {
		components = connectedComponents(gSub).filter((c) => c.length >= 2);
	} else {
		try {
			components = greedyModularityCommunities(gSub, { weight: 'weight' }).filter((c) => c.length >= 2);
		} catch {
			components = connectedComponents(gSub).filter((c) => c.length >= 2);
		}
	}

	/** @type {Sector[]} */
	const sectors = [];
	let sectorId = 1;
	for (const members of components) {
		const memberSites = members
			.map((n) => Number(n) - 1)
			.sort((x, y) => x - y);
		let siteIndices = memberSites;

		let coherence;
		let anyNonzero = false;
		for (const s of memberSites) {
			for (let n = 0; n < N && !anyNonzero; n++) if (A[s * N + n] !== 0) anyNonzero = true;
			if (anyNonzero) break;
		}
		if (memberSites.length >= 2 && anyNonzero) {
			const first = spectralCoherence(A, N, memberSites);
			coherence = first.coherence;
			const keep = Math.fround(0.1);
			const retained = memberSites.filter((_, i) => first.vDom[i] >= keep);
			if (retained.length >= 2) {
				coherence = spectralCoherence(A, N, retained).coherence;
				siteIndices = retained;
			}
		} else {
			coherence = 1.0;
		}

		if (coherence < minCoherence || siteIndices.length < 2) continue;

		const permStats = computeSectorPermutationTest(
			{ data: A, rows: A.length / N, cols: N },
			siteIndices,
			coherence,
			{ nPermutations, activeOnly: true, seed }
		);
		const pPerm = permStats.p_perm;
		if (maxPermP !== null && pPerm > maxPermP) continue;

		// np.sum(np.all(sub_A > 0, axis=0)) over the UNPRUNED community rows
		let sharedTaxa = 0;
		for (let n = 0; n < N; n++) {
			let all = true;
			for (const s of memberSites) {
				if (!(A[s * N + n] > 0)) {
					all = false;
					break;
				}
			}
			if (all) sharedTaxa++;
		}
		const lrtRows = new Float32Array(siteIndices.length);
		for (let i = 0; i < siteIndices.length; i++) lrtRows[i] = lrts32[siteIndices[i]];
		const meanLrt = numpyMeanFloat32(lrtRows);

		const sortedSites = siteIndices.slice().sort((x, y) => x - y);
		const sigTokens = sortedSites.map((s) => `${consensusAas[s]}${s + 1}`);
		const parsSig = `[ ${sigTokens.slice(0, 10).join(' - ')} ]`;

		let focalName = null;
		let focalSig = null;
		/** @type {string[]} */
		const focalDiffs = [];
		if (focalTaxon && taxa !== null && aNp !== null) {
			let focalIdx = 0;
			const needle = focalTaxon.toLowerCase();
			for (let t = 0; t < taxa.length; t++) {
				if (taxa[t].toLowerCase().includes(needle)) {
					focalIdx = t;
					focalName = taxa[t];
					break;
				}
			}
			if (focalName) {
				/** @type {string[]} */
				const focalTokens = [];
				for (const s of sortedSites) {
					const cAa = consensusAas[s];
					const fTok = aNp.data[s * aNp.N + focalIdx];
					const fAa = fTok < 20 ? (REV_AA_MAP.get(fTok) ?? '-') : cAa;
					if (fAa !== cAa && fAa !== '-') {
						focalTokens.push(`${fAa}${s + 1}*`);
						focalDiffs.push(`${cAa}${s + 1}->${fAa}`);
					} else {
						focalTokens.push(`${fAa}${s + 1}`);
					}
				}
				focalSig = `[ ${focalTokens.slice(0, 10).join(' - ')} ]`;
			}
		}

		/** @type {Sector} */
		const sector = {
			sector_id: sectorId,
			size: sortedSites.length,
			sites: sortedSites.map((s) => s + 1),
			spectral_coherence: coherence,
			p_perm: pPerm,
			null_coherence_mean: permStats.null_coherence_mean,
			null_coherence_std: permStats.null_coherence_std,
			null_coherence_95: permStats.null_coherence_95,
			isotropic_baseline: permStats.isotropic_baseline,
			shared_taxa: sharedTaxa,
			shared_branches: sharedTaxa,
			mean_lrt: meanLrt,
			pars_signature: parsSig,
			consensus_signature: parsSig
		};
		if (focalName) {
			sector.focal_taxon = focalName;
			sector.focal_signature = /** @type {string} */ (focalSig);
			sector.focal_mutations = focalDiffs;
			sector.focal_mutations_count = focalDiffs.length;
		}
		sectors.push(sector);
		sectorId++;
	}

	// sort(key=(spectral_coherence, size), reverse=True): stable, ties keep community order
	sectors.sort((x, y) => y.spectral_coherence - x.spectral_coherence || y.size - x.size);
	return sectors;
}
