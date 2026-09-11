/**
 * WHY THIS FILE EXISTS
 *
 * Mirrors the network-construction half of `hyphaeon/epistasis.py` at veg/HyphAeon cf838ab
 * (tag phase-1a):
 *
 *   consensusDelta                       epistasis.py:72-83   the consensus amino acid per site and the
 *                                                             non-consensus indicator delta[s, n]
 *   computeTransformerAttributions       compute_transformer_attributions, epistasis.py:51-114,
 *                                                             as a PURE function of the model outputs
 *   runTransformerAttributions           the batched inference loop of the same function (85-103)
 *                                                             over an async `predict` callback
 *   computeBranchCoselectionNetwork      compute_branch_coselection_network, epistasis.py:121-222
 *   CoselectionGraph                     the `nx.Graph` the network returns, with networkx's
 *                                                             insertion-order semantics
 *
 * Sector mining (extract_epistatic_sectors_tse, compute_sector_permutation_test, epistasis.py:224-448)
 * and the DMS driver are NOT here; they are `sectors.js` / `dms.js` and consume the graph this file
 * builds (its shape is documented at CoselectionGraph below).
 *
 * compute_transformer_attributions, step by step (line numbers of epistasis.py):
 *    1. `a_np = a_tensor.squeeze(-1)` [L, N] amino-acid tokens; `valid_mask = a_np < 20`         68, 75
 *    2. per site: `np.bincount(valid_row, minlength=20)` then `np.argmax` -> the LOWEST token     76-83
 *       among the most frequent residues; `REV_AA_MAP.get(major, '-')` (always a residue for
 *       0..19); `delta[s] = valid & (a != major)` as float32. A site with no valid token gets
 *       consensus '-' and a zero delta row. Tokens >= 20 (gap, stop, unknown, all 20 in
 *       dataset.py) never vote and never count as mutated. The consensus is AMINO-ACID based;
 *       codon tokens are not read at all by this function.
 *    3. per chunk of `safe_batch_size` sites: `forward_cached(..., return_attentions=True)`      89-103
 *       -> `lrts = clamp(y_soft, min=0)` float32, `mean_attns = root_attns` float32 [L, N]
 *    4. `pvals = pvals_from_lrt_self_liang(lrts).astype(np.float32)`                            109
 *    5. `leaf_attributions = mean_attns * delta` (float32 x {0,1}: exact)                         112
 *    Returns (leaf_attributions [L, N], lrts [L], pvals [L], consensus_aas [L]).
 *    There is NO tree projection: `taxa` and `tree_cache` are passed through to the model and the
 *    attribution matrix is indexed by leaf (taxon), which is why `branch_names` downstream is the
 *    taxon list and `shared_branches == shared_taxa`.
 *
 * compute_branch_coselection_network, step by step:
 *    1. `df = max(1, N - 2)`; a graph with nodes 1..L (attrs ref=consensus_aas[s], lrt=float)   139-143
 *    2. `norms = np.linalg.norm(attributions, axis=1)` (float32); active = norms > 0;             145-150
 *       V < 2 -> ([], G with nodes only)
 *    3. `A_bin = A_active > 0`; `dot = A @ A.T`; `shared = A_bin @ A_bin.T`;                     154-161
 *       `sim = dot / max(outer(norms, norms), 1e-9)` — all float32
 *    4. upper triangle k=1 in row-major order (i < j over the ACTIVE index)                       163-167
 *    5. `t = sim * sqrt(df / max(1e-9, 1 - sim^2))` (float32); `p = t.sf(t, df)` (float64)       170-171
 *    6. `cesi = sim * sqrt(max(lrt_u, 0.1) * max(lrt_v, 0.1))` (float32)                           172
 *    7. `q = benjamini_hochberg(p).astype(float32)` over ALL V(V-1)/2 pairs                        176
 *    8. pass = q <= max_fdr & shared >= min_shared & sim >= min_sim & cesi >= min_cesi           178-185
 *              & max(lrt_u, lrt_v) >= min_lrt
 *    9. per passing pair in triangle order: the record and `G.add_edge(u, v, weight=sim,           187-219
 *       shared, cesi, fdr_q)`; then `sig_pairs.sort(key=cesi, reverse=True)` (stable)              221
 *
 * FLOAT PRECISION, deliberately matched (PLAN.md §5.3 rule 4):
 *   - `attributions` and `lrts` are float32 in the reference (they come out of the model); every
 *     intermediate of steps 2-6 and 8 is float32 under numpy 2 promotion, because the Python
 *     scalars (`1e-9`, `1.0`, `0.1`, `df`, the thresholds) are "weak" and take the array dtype.
 *     Each such operation is rounded with Math.fround here. Only `p` (scipy promotes to float64)
 *     and the BH pass (float64 input) run in double; `q` is then rounded to float32.
 *   - THRESHOLDS ARE COMPARED IN FLOAT32. `sim_arr >= min_sim` with a float32 array casts the
 *     Python float to float32 first: float32(0.35) = 0.3499999940395355 < 0.35, so a pair whose
 *     cosine is exactly float32(0.35) PASSES `min_sim=0.35` in the reference and would fail a
 *     float64 comparison. Measured with the reference (test/data/epistasis/gen.py case
 *     `threshold_float32_promotion`: edge present). All five thresholds are compared against
 *     `Math.fround(threshold)`.
 *   - `np.linalg.norm(axis=1)` on float32 is `sqrt(add.reduce(x*x))`, numpy's pairwise float32
 *     sum per row: reproduced bit for bit with numpyPairwiseSum (measured 100% at N = 5, 12, 37,
 *     128, 300, 511 on random 40%-sparse rows).
 *   - `A @ A.T` is Accelerate/OpenBLAS sgemm, whose blocked accumulation order is not a fixed
 *     function of N. Measured against the reference: neither a sequential nor a pairwise float32
 *     sum reproduces it (88-95% of entries at N = 12, 25-30% at N = 300); a float64 accumulation
 *     rounded once to float32 is the closest (91% / 31%) and the most accurate, with a maximum
 *     relative deviation of 1.2e-7 at N = 12 and 5.0e-7 at N = 511. That is what the dot uses.
 *     Consequence: `similarity`, `cesi` and (through t) `p_val`/`fdr_q` agree with the reference
 *     to the 1e-6 class the fixture declares, not to the ulp, and a pair whose cosine sits within
 *     ~5e-7 of a threshold can be classified differently. `shared` is exact (integer sums).
 *   - `cesi` uses `lrts[s]` as float32 and `0.1` as float32(0.1); `similarity`, `cesi`, `fdr_q`,
 *     `lrt_u/v` in the records are float32 values read as JS numbers, `p_val` is float64.
 *
 * QUIRKS REPLICATED (pinned by tests; fix upstream first — PLAN.md §5.3 rule 3):
 *   - `branch_names` is accepted and never read (only `attributions.shape` defines N); it is
 *     kept as a parameter here for the same reason and ignored.
 *   - `hyper_p` duplicates `p_val` and `shared_branches` duplicates `shared_taxa` (the record
 *     keeps the names of an older hypergeometric formulation); the graph edge stores `shared`
 *     once. Both duplicates are emitted.
 *   - The Student-t test is applied to a cosine of NON-NEGATIVE attribution vectors, so t >= 0
 *     in practice; the port keeps scipy's full `t.sf` (negative t gives p > 0.5) because the
 *     function is generic over its input.
 *   - `1 - sim**2` in float32 loses most of its digits near sim = 1 (the planted fixture pairs at
 *     0.998-0.9994 keep 3-4 significant digits); the p-values inherit that. Reproduced, not
 *     improved.
 *   - The float32 cosine can EXCEED 1: for two identical rows [1, 1] the norms are
 *     float32(sqrt 2) = 1.4142135 and their outer product rounds to 1.9999999, so
 *     sim = 2 / 1.9999999 = 1.0000001192. Then `1 - sim**2` is negative, `max(1e-9, .)` takes
 *     1e-9 and t = sqrt(df / 1e-9); at df = 1 that is only p = 1.0066e-5 (measured with the
 *     reference; pinned in test/epistasis.test.js). Collinear pairs therefore do not all get the
 *     same p, and `similarity` in a record may read 1.0000001.
 *   - `shared_arr` is a float32 matrix product of 0/1 indicators; exact below 2^24 (N <= 512).
 *   - The default `min_sim` is 0.35 here while `cli.py` passes 0.30 (fixtures/manifest.json
 *     known_quirks); the CLI never passes `min_cesi`, so 2.0 applies to it as well.
 *   - `compute_adaptive_safe_batch_size` (inference.py:59-98) depends on the device; the chunk
 *     size changes nothing in the outputs (each site is scored independently), so
 *     `runTransformerAttributions` takes a plain `batchSize` (default 64, the pipeline's).
 *
 * WHAT IT DELIBERATELY DOES NOT DO: no torch, no device selection, no progress printing, no
 * tree cache (the ONNX graph takes the distance matrix and MDS coordinates instead), no
 * networkx. `CoselectionGraph` implements exactly the subset of `nx.Graph` this file and the
 * sector port need, with networkx 3.6.1's ordering; it is a data structure, not a graph library.
 *
 * MEMORY: like the reference, the pairwise arrays are dense over the V(V-1)/2 active pairs
 * (M pairs x (4 + 4 + 8 + 4 + 4) bytes here versus three V x V float32 matrices plus the
 * triangle arrays in numpy). At L = 1097 (Smc6) M = 601k; at L = 5000, 12.5M.
 */

import { pvalsFromLrtSelfLiang } from './stats.js';
import { benjaminiHochberg } from './numeric/bh.js';
import { tSf } from './numeric/special.js';
import { numpyPairwiseSum } from './numeric/reduce.js';
import { AA_LIST, AA_VALID_BELOW } from './preprocess/modelContract.js';

/**
 * @typedef {import('./preprocess/assemble.js').LoadedAlignment} LoadedAlignment
 */

// ---------------------------------------------------------------------------------------------
// CoselectionGraph — the nx.Graph of compute_branch_coselection_network
// ---------------------------------------------------------------------------------------------

/**
 * @typedef {{ref: string, lrt: number}} CoselectionNodeAttrs
 * @typedef {{weight: number, shared: number, cesi: number, fdr_q: number}} CoselectionEdgeAttrs
 */

/**
 * The `nx.Graph` compute_branch_coselection_network returns, reduced to what the epistasis and
 * sector ports use, with networkx 3.6.1's ordering semantics (PLAN.md §5.3 rule 6):
 *
 *   - `nodes: Map<number, attrs>` — nodes in INSERTION order (`G._node`): 1..L as the network
 *     adds them (`G.add_node(s + 1, ref=..., lrt=...)`, epistasis.py:143). `add_edge` on a node
 *     not yet present appends it with `{}` attrs, as networkx does.
 *   - `adj: Map<number, Map<number, attrs>>` — `G._adj`: for each node, its neighbours in the
 *     order the edges were added, both directions SHARING one attrs object (networkx stores one
 *     datadict in both `_adj[u][v]` and `_adj[v][u]`). Re-adding an edge updates that object in
 *     place and keeps its position (dict.update semantics).
 *   - `edges()` — `G.edges(data=True)`: for each node in node order, each neighbour in adjacency
 *     order, skipping neighbours already visited as a source. This is the order the fixture
 *     records and the order `graph_to_json` (scripts/gen_fixtures.py:155-161) serialises.
 *   - `degree(n)` counts a self-loop twice, as networkx does; `numberOfEdges()` is
 *     `G.number_of_edges()`.
 *
 * Node ids are the 1-based site numbers (JS numbers). Nothing here is weighted-graph algebra;
 * greedy modularity and connected components belong to sectors.js.
 */
export class CoselectionGraph {
	constructor() {
		/** @type {Map<number, Record<string, any>>} */
		this.nodes = new Map();
		/** @type {Map<number, Map<number, Record<string, any>>>} */
		this.adj = new Map();
	}

	/**
	 * `G.add_node(n, **attrs)`: a new node is appended; an existing one has its attrs updated in
	 * place and keeps its position.
	 * @param {number} n
	 * @param {Record<string, any>} [attrs]
	 */
	addNode(n, attrs = {}) {
		const existing = this.nodes.get(n);
		if (existing === undefined) {
			this.nodes.set(n, { ...attrs });
			this.adj.set(n, new Map());
		} else {
			Object.assign(existing, attrs);
		}
	}

	/**
	 * `G.add_edge(u, v, **attrs)`: missing endpoints are appended as nodes with empty attrs; an
	 * existing edge's shared datadict is updated in place.
	 * @param {number} u
	 * @param {number} v
	 * @param {Record<string, any>} [attrs]
	 */
	addEdge(u, v, attrs = {}) {
		if (!this.nodes.has(u)) this.addNode(u);
		if (!this.nodes.has(v)) this.addNode(v);
		const au = /** @type {Map<number, Record<string, any>>} */ (this.adj.get(u));
		const av = /** @type {Map<number, Record<string, any>>} */ (this.adj.get(v));
		const existing = au.get(v);
		if (existing !== undefined) {
			Object.assign(existing, attrs);
			return;
		}
		const d = { ...attrs };
		au.set(v, d);
		av.set(u, d);
	}

	/** @param {number} n @returns {boolean} */
	hasNode(n) {
		return this.nodes.has(n);
	}

	/** @param {number} u @param {number} v @returns {boolean} */
	hasEdge(u, v) {
		const au = this.adj.get(u);
		return au !== undefined && au.has(v);
	}

	/**
	 * `G.get_edge_data(u, v)`: the shared attrs object, or `undefined`.
	 * @param {number} u @param {number} v
	 * @returns {Record<string, any>|undefined}
	 */
	getEdgeData(u, v) {
		const au = this.adj.get(u);
		return au === undefined ? undefined : au.get(v);
	}

	/**
	 * `G.degree(n)`: neighbour count, a self-loop counted twice.
	 * @param {number} n @returns {number}
	 */
	degree(n) {
		const an = this.adj.get(n);
		if (an === undefined) return 0;
		return an.size + (an.has(n) ? 1 : 0);
	}

	/** `G.number_of_nodes()` @returns {number} */
	numberOfNodes() {
		return this.nodes.size;
	}

	/** `G.number_of_edges()` @returns {number} */
	numberOfEdges() {
		let twice = 0;
		let loops = 0;
		for (const [n, an] of this.adj) {
			twice += an.size;
			if (an.has(n)) loops++;
		}
		// each non-loop edge is counted from both ends; a loop appears once in its own row
		return (twice - loops) / 2 + loops;
	}

	/**
	 * `G.edges(data=True)`, in networkx's iteration order (see the class comment).
	 * @returns {Array<[number, number, Record<string, any>]>}
	 */
	edges() {
		/** @type {Array<[number, number, Record<string, any>]>} */
		const out = [];
		const seen = new Set();
		for (const [n, an] of this.adj) {
			for (const [nbr, d] of an) {
				if (!seen.has(nbr)) out.push([n, nbr, d]);
			}
			seen.add(n);
		}
		return out;
	}

	/**
	 * `[(n, d) for n, d in G.nodes(data=True)]` in insertion order.
	 * @returns {Array<[number, Record<string, any>]>}
	 */
	nodeList() {
		return Array.from(this.nodes, ([n, d]) => [n, d]);
	}

	/**
	 * `graph_to_json` of scripts/gen_fixtures.py:155-161 — `{nodes: [[n, attrs]], edges: [[u, v, attrs]]}`
	 * in networkx order; the shape the epistasis fixtures record.
	 * @returns {{nodes: Array<[number, Record<string, any>]>, edges: Array<[number, number, Record<string, any>]>}}
	 */
	toJson() {
		return {
			nodes: this.nodeList().map(([n, d]) => [n, { ...d }]),
			edges: this.edges().map(([u, v, d]) => [u, v, { ...d }])
		};
	}

	/**
	 * `graph_from_json` of scripts/gen_fixtures.py:164-171: nodes then edges, in the recorded order.
	 * @param {{nodes?: Array<[number, Record<string, any>]>, edges?: Array<[number, number, Record<string, any>]>}} d
	 * @returns {CoselectionGraph}
	 */
	static fromJson(d) {
		const g = new CoselectionGraph();
		for (const [n, attrs] of d.nodes ?? []) g.addNode(n, attrs);
		for (const [u, v, attrs] of d.edges ?? []) g.addEdge(u, v, attrs);
		return g;
	}
}

// ---------------------------------------------------------------------------------------------
// compute_transformer_attributions  (epistasis.py:51-114)
// ---------------------------------------------------------------------------------------------

/**
 * Accept an [L, N] matrix as a flat typed array, a flat plain array, or an array of rows, and
 * return it as a flat Float32Array of length L*N (every element rounded to float32, as the
 * reference's arrays are float32).
 *
 * @param {ArrayLike<number>|ArrayLike<ArrayLike<number>>} m
 * @param {number} L
 * @param {number} N
 * @param {string} what
 * @returns {Float32Array}
 */
function toFloat32Matrix(m, L, N, what) {
	if (m instanceof Float32Array && m.length === L * N) return m;
	if (m.length === L && L > 0 && typeof m[0] === 'object' && m[0] !== null) {
		const out = new Float32Array(L * N);
		for (let s = 0; s < L; s++) {
			const row = /** @type {ArrayLike<number>} */ (m[s]);
			if (row.length !== N) throw new Error(`${what}: row ${s} has ${row.length} columns, expected ${N}`);
			for (let n = 0; n < N; n++) out[s * N + n] = row[n];
		}
		return out;
	}
	if (m.length !== L * N) throw new Error(`${what}: expected ${L}*${N} = ${L * N} values, got ${m.length}`);
	return Float32Array.from(/** @type {ArrayLike<number>} */ (m));
}

/**
 * Like toFloat32Matrix for integer token matrices ([L, N] or [L, N, 1] flat).
 * @param {ArrayLike<number>|ArrayLike<ArrayLike<number>>} m
 * @param {number} L
 * @param {number} N
 * @returns {Int32Array}
 */
function toInt32Matrix(m, L, N) {
	if (m instanceof Int32Array && m.length === L * N) return m;
	if (m.length === L && L > 0 && typeof m[0] === 'object' && m[0] !== null) {
		const out = new Int32Array(L * N);
		for (let s = 0; s < L; s++) {
			const row = /** @type {ArrayLike<number>} */ (m[s]);
			if (row.length !== N) throw new Error(`aaTokens: row ${s} has ${row.length} columns, expected ${N}`);
			for (let n = 0; n < N; n++) out[s * N + n] = row[n];
		}
		return out;
	}
	if (m.length !== L * N) throw new Error(`aaTokens: expected ${L}*${N} = ${L * N} tokens, got ${m.length}`);
	return Int32Array.from(/** @type {ArrayLike<number>} */ (m));
}

/**
 * epistasis.py:72-83 — the consensus amino acid per site and the non-consensus indicator.
 *
 *     valid_mask = a_np < 20
 *     for s: valid_row = a_np[s, valid_mask[s]]
 *            if len(valid_row) > 0:
 *                major = int(np.argmax(np.bincount(valid_row, minlength=20)))   # lowest on ties
 *                consensus_aas.append(REV_AA_MAP.get(major, '-'))
 *                delta[s] = (valid_mask[s] & (a_np[s] != major)).astype(np.float32)
 *            else: consensus_aas.append('-')                                       # delta row stays 0
 *
 * `majorAa[s]` is the winning token, or -1 for a site with no valid token. Tokens >= 20 never
 * vote and are never "mutated"; negative tokens do not occur in the reference (np.bincount would
 * raise) and are treated as invalid here.
 *
 * @param {ArrayLike<number>|ArrayLike<ArrayLike<number>>} aaTokens [L, N] (or [L, N, 1] flat) amino-acid tokens
 * @param {number} L
 * @param {number} N
 * @returns {{consensusAas: string[], majorAa: Int32Array, delta: Float32Array}} delta is [L*N] float32 {0, 1}
 */
export function consensusDelta(aaTokens, L, N) {
	const a = toInt32Matrix(aaTokens, L, N);
	const consensusAas = new Array(L);
	const majorAa = new Int32Array(L);
	const delta = new Float32Array(L * N);
	const counts = new Int32Array(AA_VALID_BELOW);
	for (let s = 0; s < L; s++) {
		counts.fill(0);
		let any = false;
		const base = s * N;
		for (let n = 0; n < N; n++) {
			const t = a[base + n];
			if (t >= 0 && t < AA_VALID_BELOW) {
				counts[t]++;
				any = true;
			}
		}
		if (!any) {
			consensusAas[s] = '-';
			majorAa[s] = -1;
			continue;
		}
		// np.argmax: the first (lowest) index among the maxima.
		let major = 0;
		for (let t = 1; t < AA_VALID_BELOW; t++) if (counts[t] > counts[major]) major = t;
		majorAa[s] = major;
		consensusAas[s] = AA_LIST[major];
		for (let n = 0; n < N; n++) {
			const t = a[base + n];
			delta[base + n] = t >= 0 && t < AA_VALID_BELOW && t !== major ? 1 : 0;
		}
	}
	return { consensusAas, majorAa, delta };
}

/**
 * @typedef {{
 *   leafAttributions: Float32Array, lrts: Float32Array, pvals: Float32Array, consensusAas: string[],
 *   delta: Float32Array, L: number, N: number
 * }} TransformerAttributions
 */

/**
 * `compute_transformer_attributions` (epistasis.py:51-114) as a pure function of what the model
 * produced: the root-token attention `mean_root_attns` [L, N] and the raw `y_soft` LRT
 * surrogate [L] (clamped at 0 here, as line 100 does; already-clamped values pass unchanged),
 * with the amino-acid tokens [L, N] that define the consensus and delta. Codon tokens, the tree
 * and the taxon names play no part in the arithmetic and are not taken.
 *
 * Returns the reference's four outputs (`leafAttributions` [L*N] float32 = attention x delta,
 * `lrts` float32, `pvals` float32 Self & Liang, `consensusAas`) plus `delta` and the shape.
 *
 * @param {{
 *   attention: ArrayLike<number>|ArrayLike<ArrayLike<number>>,
 *   lrts: ArrayLike<number>,
 *   aaTokens: ArrayLike<number>|ArrayLike<ArrayLike<number>>,
 *   L?: number, N?: number
 * }} input `L`/`N` default to `lrts.length` and `aaTokens.length / L` (flat) or the row length
 * @returns {TransformerAttributions}
 */
export function computeTransformerAttributions(input) {
	const L = input.L ?? input.lrts.length;
	let N = input.N;
	if (N === undefined) {
		const at = input.aaTokens;
		if (L > 0 && typeof at[0] === 'object' && at[0] !== null) N = /** @type {ArrayLike<number>} */ (at[0]).length;
		else N = L > 0 ? at.length / L : 0;
	}
	if (!Number.isInteger(N) || N < 0) throw new Error(`computeTransformerAttributions: cannot infer N from ${input.aaTokens.length} tokens and L=${L}`);
	if (input.lrts.length !== L) throw new Error(`computeTransformerAttributions: lrts has ${input.lrts.length} values, expected L=${L}`);

	const meanAttns = toFloat32Matrix(input.attention, L, N, 'attention');
	const { consensusAas, delta } = consensusDelta(input.aaTokens, L, N);

	// epistasis.py:100 — torch.clamp(y_soft, min=0.0) stored float32. NaN stays NaN (torch.clamp
	// propagates NaN; Math.max would too).
	const lrts = new Float32Array(L);
	for (let s = 0; s < L; s++) {
		const y = input.lrts[s];
		lrts[s] = y < 0 ? 0 : y;
	}

	// epistasis.py:109 — float64 Self & Liang p, cast to float32.
	const pvals = Float32Array.from(pvalsFromLrtSelfLiang(lrts));

	// epistasis.py:112 — mean_attns * delta, float32 x {0, 1}.
	const leafAttributions = new Float32Array(L * N);
	for (let i = 0; i < L * N; i++) leafAttributions[i] = Math.fround(meanAttns[i] * delta[i]);

	return { leafAttributions, lrts, pvals, consensusAas, delta, L, N };
}

/**
 * The model callback for the attention pillars. `c`/`a` are [batch, N, 1] Int32Array tokens (the
 * same layout as filter.js's PredictFn); return the graph's raw `lrt` output ([batch]) AND its
 * `mean_root_attns` output ([batch * N], row-major), both as produced (the clamp and float32
 * storage are applied here).
 *
 * @callback AttentionPredictFn
 * @param {Int32Array} c
 * @param {Int32Array} a
 * @param {{batch: number, N: number, d: Float32Array, z: Float32Array, siteIndices: Int32Array,
 *   phase: string, loaded: LoadedAlignment}} meta
 * @returns {Promise<{lrt: ArrayLike<number>, attention: ArrayLike<number>}>|{lrt: ArrayLike<number>, attention: ArrayLike<number>}}
 */

/**
 * The inference loop of `compute_transformer_attributions` (epistasis.py:85-103) over an async
 * callback: every site 0..L-1 in order, `batchSize` at a time (the reference's
 * `compute_adaptive_safe_batch_size` is device-dependent and changes no output), then the pure
 * `computeTransformerAttributions` on the gathered `lrt` and `mean_root_attns`.
 *
 * @param {LoadedAlignment} loaded `c`, `a` (Int32Array [L, N, 1]), `d`, `z`, `L`, `N`
 * @param {AttentionPredictFn} predict
 * @param {{batchSize?: number, onProgress?: (p: {phase: string, done: number, total: number}) => void}} [options]
 * @returns {Promise<TransformerAttributions>}
 */
export async function runTransformerAttributions(loaded, predict, options = {}) {
	const { L, N, c, a, d, z } = loaded;
	const batchSize = Math.max(1, Math.floor(options.batchSize ?? 64));
	const phase = 'attribution-attention';
	const attention = new Float32Array(L * N);
	const ySoft = new Float32Array(L);
	for (let start = 0; start < L; start += batchSize) {
		const end = Math.min(start + batchSize, L);
		const batch = end - start;
		const cb = c.subarray(start * N, end * N);
		const ab = a.subarray(start * N, end * N);
		const idx = new Int32Array(batch);
		for (let k = 0; k < batch; k++) idx[k] = start + k;
		const out = await predict(cb, ab, { batch, N, d, z, siteIndices: idx, phase, loaded });
		if (out.lrt.length !== batch) throw new Error(`predict returned ${out.lrt.length} lrt values for a batch of ${batch} sites`);
		if (out.attention.length !== batch * N) {
			throw new Error(`predict returned ${out.attention.length} attention values for a batch of ${batch} x ${N}`);
		}
		for (let k = 0; k < batch; k++) ySoft[start + k] = out.lrt[k];
		attention.set(out.attention, start * N);
		if (options.onProgress) options.onProgress({ phase, done: end, total: L });
	}
	return computeTransformerAttributions({ attention, lrts: ySoft, aaTokens: a, L, N });
}

// ---------------------------------------------------------------------------------------------
// compute_branch_coselection_network  (epistasis.py:121-222)
// ---------------------------------------------------------------------------------------------

/**
 * @typedef {{
 *   site_u: number, site_v: number, ref_u: string, ref_v: string, lrt_u: number, lrt_v: number,
 *   similarity: number, shared_taxa: number, shared_branches: number,
 *   p_val: number, hyper_p: number, fdr_q: number, cesi: number
 * }} CoselectionEdge
 */

/**
 * @typedef {{
 *   minSim?: number, minShared?: number, maxFdr?: number, minLrt?: number, minCesi?: number, N?: number
 * }} CoselectionOptions
 */

const F32_1E_9 = Math.fround(1e-9);
const F32_0_1 = Math.fround(0.1);

/**
 * `compute_branch_coselection_network(attributions, lrts, branch_names, consensus_aas, min_sim=0.35,
 * min_shared=2, max_fdr=0.05, min_lrt=1.0, min_cesi=2.0)` -> `(sig_pairs, G)`.
 *
 * `attributions` is the [L, N] leaf attribution matrix (flat Float32Array, flat array, or rows);
 * L is `lrts.length`, N is `options.N`, the row length, or `attributions.length / L`.
 * `branchNames` is accepted for signature parity and never read (the reference does not read it
 * either). Records carry the reference's keys verbatim; `sigPairs` is sorted by `cesi`
 * descending (stable), `graph` has nodes 1..L with `{ref, lrt}` and one edge per record with
 * `{weight: similarity, shared, cesi, fdr_q}` in triangle order.
 *
 * @param {ArrayLike<number>|ArrayLike<ArrayLike<number>>} attributions
 * @param {ArrayLike<number>} lrts float32 site LRTs (values are read as float32)
 * @param {ArrayLike<string>|null|undefined} branchNames unused, as in the reference
 * @param {ArrayLike<string>} consensusAas
 * @param {CoselectionOptions} [options]
 * @returns {{sigPairs: CoselectionEdge[], graph: CoselectionGraph}}
 */
export function computeBranchCoselectionNetwork(attributions, lrts, branchNames, consensusAas, options = {}) {
	void branchNames;
	const L = lrts.length;
	if (consensusAas.length !== L) {
		throw new Error(`computeBranchCoselectionNetwork: consensus_aas has ${consensusAas.length} entries, lrts ${L}`);
	}
	let N = options.N;
	if (N === undefined) {
		if (L > 0 && typeof attributions[0] === 'object' && attributions[0] !== null) N = /** @type {ArrayLike<number>} */ (attributions[0]).length;
		else N = L > 0 ? attributions.length / L : 0;
	}
	if (!Number.isInteger(N) || N < 0) throw new Error(`computeBranchCoselectionNetwork: cannot infer N from ${attributions.length} values and L=${L}`);
	const A = toFloat32Matrix(attributions, L, N, 'attributions');
	const lrt32 = lrts instanceof Float32Array ? lrts : Float32Array.from(lrts);

	// Thresholds as numpy compares them: the float32 array side promotes the weak Python scalar.
	const minSim = Math.fround(options.minSim ?? 0.35);
	const minShared = Math.fround(options.minShared ?? 2);
	const maxFdr = Math.fround(options.maxFdr ?? 0.05);
	const minLrt = Math.fround(options.minLrt ?? 1.0);
	const minCesi = Math.fround(options.minCesi ?? 2.0);

	const df = Math.max(1, N - 2);

	// epistasis.py:141-143
	const G = new CoselectionGraph();
	for (let s = 0; s < L; s++) G.addNode(s + 1, { ref: consensusAas[s], lrt: lrt32[s] });

	// epistasis.py:145-150 — float32 row norms (pairwise float32 sum of float32 squares, then sqrt).
	const norms = new Float32Array(L);
	const sq = new Float32Array(N);
	const validSites = [];
	for (let s = 0; s < L; s++) {
		const base = s * N;
		for (let n = 0; n < N; n++) sq[n] = Math.fround(A[base + n] * A[base + n]);
		norms[s] = Math.fround(Math.sqrt(numpyPairwiseSum(sq, 0, N, Math.fround)));
		if (norms[s] > 0) validSites.push(s);
	}
	const V = validSites.length;
	/** @type {CoselectionEdge[]} */
	const sigPairs = [];
	if (V < 2) return { sigPairs, graph: G };

	// epistasis.py:152-172 — the upper triangle (k=1) in row-major order over the active sites.
	const M = (V * (V - 1)) / 2;
	const simArr = new Float32Array(M);
	const sharedArr = new Int32Array(M);
	const pVals = new Float64Array(M);
	const cesiArr = new Float32Array(M);
	let k = 0;
	for (let i = 0; i < V - 1; i++) {
		const s1 = validSites[i];
		const b1 = s1 * N;
		const n1 = norms[s1];
		const l1 = Math.max(lrt32[s1], F32_0_1);
		for (let j = i + 1; j < V; j++) {
			const s2 = validSites[j];
			const b2 = s2 * N;
			// dot: float64 accumulation rounded once to float32 (the closest reproducible stand-in
			// for sgemm, see the header); shared: exact integer count of jointly positive taxa.
			let dot = 0;
			let shared = 0;
			for (let n = 0; n < N; n++) {
				const x = A[b1 + n];
				const y = A[b2 + n];
				dot += x * y;
				if (x > 0 && y > 0) shared++;
			}
			const denom = Math.max(Math.fround(n1 * norms[s2]), F32_1E_9);
			const sim = Math.fround(Math.fround(dot) / denom);
			simArr[k] = sim;
			sharedArr[k] = shared;
			// t = sim * sqrt(df / max(1e-9, 1 - sim^2)), float32 throughout; p = t.sf in float64.
			const oneMinus = Math.max(F32_1E_9, Math.fround(1 - Math.fround(sim * sim)));
			const t = Math.fround(sim * Math.fround(Math.sqrt(Math.fround(df / oneMinus))));
			pVals[k] = tSf(t, df);
			// cesi = sim * sqrt(max(lrt_u, 0.1) * max(lrt_v, 0.1)), float32.
			const l2 = Math.max(lrt32[s2], F32_0_1);
			cesiArr[k] = Math.fround(sim * Math.fround(Math.sqrt(Math.fround(l1 * l2))));
			k++;
		}
	}

	// epistasis.py:176 — BH in float64 over all M pairs, cast to float32.
	const q64 = benjaminiHochberg(pVals);

	// epistasis.py:178-219
	k = 0;
	for (let i = 0; i < V - 1; i++) {
		const s1 = validSites[i];
		for (let j = i + 1; j < V; j++, k++) {
			const s2 = validSites[j];
			const q = Math.fround(q64[k]);
			const sim = simArr[k];
			const shared = sharedArr[k];
			const cesi = cesiArr[k];
			const lu = lrt32[s1];
			const lv = lrt32[s2];
			const pass =
				q <= maxFdr &&
				shared >= minShared &&
				sim >= minSim &&
				cesi >= minCesi &&
				Math.max(lu, lv) >= minLrt;
			if (!pass) continue;
			sigPairs.push({
				site_u: s1 + 1,
				site_v: s2 + 1,
				ref_u: consensusAas[s1],
				ref_v: consensusAas[s2],
				lrt_u: lu,
				lrt_v: lv,
				similarity: sim,
				shared_taxa: shared,
				shared_branches: shared,
				p_val: pVals[k],
				hyper_p: pVals[k],
				fdr_q: q,
				cesi
			});
			G.addEdge(s1 + 1, s2 + 1, { weight: sim, shared, cesi, fdr_q: q });
		}
	}

	// epistasis.py:221 — Python's sort is stable; so is Array.prototype.sort.
	sigPairs.sort((x, y) => y.cesi - x.cesi);
	return { sigPairs, graph: G };
}
