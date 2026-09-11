/**
 * WHY THIS FILE EXISTS
 *
 * Mirrors `compute_fast_dist_matrix` of `hyphaeon/dataset.py:524-574` at veg/HyphAeon reconcile/phase-5a,
 * plus the `> 10.0` rescale rule of `load_alignment_and_tree` (dataset.py:1037-1040):
 *
 *   - depths accumulated from the root in float64, `branch_length if not None else 0.0`;
 *   - `terminals = {t.name.strip("'\""): t for t in tree.get_terminals() if t.name and ... in taxa}`,
 *     a dict, so a DUPLICATE tip name resolves to the last terminal in preorder;
 *   - d(i, j) = depth_i + depth_j - 2.0 * depth_lca, in that operation order, stored float32;
 *   - a taxon with no terminal of that name keeps its ZERO row and column (`if ti is None: continue`)
 *     — replicated, not repaired (task issue #9): it happens whenever tier-2/3 name matching hands
 *     this function alignment names that differ from the quote-stripped tree names;
 *   - the root's own branch length is ignored (depth 0 at the root);
 *   - `if dist_mat.max() > 10.0: dist_mat = dist_mat / L`, on the float32 matrix, AFTER
 *     `enforce_nonzero_branch_lengths`, so a path of exactly 10.0 that crosses a zero-length branch
 *     (raised to 1e-4) tips over the threshold (pinned by fixtures/dataset/rescale_rule.json).
 *
 * `rootDistances` / `patristicRow` / `patristicMatrix` are the float64 primitives; `patristicRow`
 * uses the same (a + b) - 2c arithmetic as the reference so the float64 values are identical before
 * the float32 store.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: no negative-distance clamping (dataset.py has none; after
 * `enforce_nonzero_branch_lengths` there are no negative branches anyway), no on-demand rows (the
 * reference materialises N x N and so does this), no Max-PD (that is downsample.js).
 *
 * DIVERGENCE FROM THE DATAMONKEY3 PORT (main@fac1330 src/lib/services/axomeme/patristic.js), which
 * this file replaces: DM3's `maxPdSelect` seeded Faith's PD at index 0 (AxoMEME 2.0 driver) where
 * dataset.py:628 seeds with the most distant PAIR — removed, see downsample.js. DM3 read a missing
 * branch length as 0 at parse time; here it is `null` until `enforceNonzeroBranchLengths` runs, and
 * `rootDistances` applies the `else 0.0` itself. The tree shape is tree.js's (`parent` array,
 * `children`, `root`) rather than newick.js's.
 */

import { getTerminals, stripQuotes } from './tree.js';

/**
 * Depth of every node from the root (dataset.py:534-540 `calc_depths`), float64, a missing branch
 * length contributing 0.0.
 *
 * @param {import('./tree.js').PhyloTree} tree
 * @returns {Float64Array}
 */
export function rootDistances(tree) {
	const dist = new Float64Array(tree.parent.length);
	// Explicit stack: the reference recurses, and a ladder tree deeper than Python's frame limit
	// would fail there; values are identical.
	const stack = [tree.root];
	dist[tree.root] = 0;
	while (stack.length) {
		const node = /** @type {number} */ (stack.pop());
		for (const c of tree.children[node]) {
			const bl = tree.branchLength[c];
			dist[c] = dist[node] + (bl === null ? 0.0 : bl);
			stack.push(c);
		}
	}
	return dist;
}

/**
 * A reusable ancestor marker: stamp the root path of one node with a generation number so the LCA
 * walk from another node stops at the first stamped node, with no per-pair clearing.
 */
function ancestorMarker(nodeCount) {
	const stamp = new Int32Array(nodeCount);
	let generation = 0;
	return {
		markPath(tree, node) {
			generation++;
			let cur = node;
			while (cur !== -1) {
				stamp[cur] = generation;
				cur = tree.parent[cur];
			}
		},
		isMarked(node) {
			return stamp[node] === generation;
		}
	};
}

/**
 * One row of patristic distances: from `fromNode` to each of `toNodes`, float64, computed as
 * `depth_i + depth_j - 2.0 * depth_lca` exactly like dataset.py:570.
 *
 * @param {import('./tree.js').PhyloTree} tree
 * @param {Float64Array} rootDist from rootDistances()
 * @param {number} fromNode
 * @param {number[]} toNodes
 * @param {ReturnType<typeof ancestorMarker>} [marker]
 * @returns {Float64Array}
 */
export function patristicRow(tree, rootDist, fromNode, toNodes, marker) {
	const m = marker ?? ancestorMarker(tree.parent.length);
	m.markPath(tree, fromNode);
	const out = new Float64Array(toNodes.length);
	for (let k = 0; k < toNodes.length; k++) {
		const b = toNodes[k];
		let cur = b;
		while (cur !== -1 && !m.isMarked(cur)) cur = tree.parent[cur];
		out[k] = cur === -1 ? 0 : rootDist[fromNode] + rootDist[b] - 2.0 * rootDist[cur];
	}
	return out;
}

/**
 * The full N x N float64 patristic matrix over the given nodes, row-major.
 *
 * @param {import('./tree.js').PhyloTree} tree
 * @param {number[]} nodes
 * @returns {Float64Array}
 */
export function patristicMatrix(tree, nodes) {
	const n = nodes.length;
	const rootDist = rootDistances(tree);
	const marker = ancestorMarker(tree.parent.length);
	const out = new Float64Array(n * n);
	for (let i = 0; i < n; i++) out.set(patristicRow(tree, rootDist, nodes[i], nodes, marker), i * n);
	return out;
}

/**
 * `compute_fast_dist_matrix(tree, taxa)`, dataset.py:524-574: float32 [n, n], row-major, with a
 * zero row and column for any taxon that has no terminal of that (quote-stripped) name.
 *
 * @param {import('./tree.js').PhyloTree} tree already through enforceNonzeroBranchLengths, as in
 *   the reference's call order (dataset.py:972 then 676)
 * @param {string[]} taxa alignment names, in the order the rows should come out
 * @returns {Float32Array} length taxa.length ** 2
 */
export function computeFastDistMatrix(tree, taxa) {
	const n = taxa.length;
	const dist = new Float32Array(n * n);
	const taxaSet = new Set(taxa);
	/** name -> terminal node; a later terminal with the same name overwrites (dict comprehension). */
	const terminals = new Map();
	for (const t of getTerminals(tree)) {
		const nm = tree.name[t];
		if (!nm) continue;
		const clean = stripQuotes(nm);
		if (taxaSet.has(clean)) terminals.set(clean, t);
	}

	const rootDist = rootDistances(tree);
	const marker = ancestorMarker(tree.parent.length);
	const nodes = taxa.map((t) => terminals.get(t));
	for (let i = 0; i < n; i++) {
		const ti = nodes[i];
		if (ti === undefined) continue;
		const row = patristicRow(tree, rootDist, ti, /** @type {number[]} */ (nodes.map((v) => v ?? -1)), marker);
		for (let j = i; j < n; j++) {
			if (nodes[j] === undefined) continue;
			const d = Math.fround(row[j]);
			dist[i * n + j] = d;
			dist[j * n + i] = d;
		}
	}
	return dist;
}

/**
 * The rescale rule of dataset.py:1037-1040: `if dist_mat.max() > 10.0: dist_mat = dist_mat / L`
 * on the float32 matrix. Returns a new matrix; the input is not modified.
 *
 * @param {Float32Array} dist float32 patristic matrix (any length)
 * @param {number} L codon count
 * @returns {{dist: Float32Array, rescaled: boolean, rawMax: number}}
 */
export function rescaleDistances(dist, L) {
	if (dist.length === 0) {
		// `np.max` of an empty array raises ValueError in the reference.
		throw new Error('rescaleDistances: zero-size array to reduction operation maximum which has no identity');
	}
	let rawMax = dist[0];
	for (let i = 1; i < dist.length; i++) if (dist[i] > rawMax) rawMax = dist[i];
	if (rawMax > 10.0) {
		const out = new Float32Array(dist.length);
		for (let i = 0; i < dist.length; i++) out[i] = Math.fround(dist[i] / L);
		return { dist: out, rescaled: true, rawMax };
	}
	return { dist: Float32Array.from(dist), rescaled: false, rawMax };
}
