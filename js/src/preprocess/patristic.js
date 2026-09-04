/**
 * WHY THIS FILE EXISTS
 *
 * Ported verbatim from datamonkey3 (main@fac1330) src/lib/services/axomeme/patristic.js. Function
 * bodies unchanged; the `./newick.js` import resolves to the sibling module here as it did there.
 *
 * It mirrors two functions of `hyphaeon/dataset.py`:
 *   - `compute_fast_dist_matrix` (dataset.py:302-350) — the same
 *     d(a,b) = depth(a) + depth(b) - 2*depth(lca(a,b)) over accumulated root depths, with the same
 *     `branch_length if ... is not None else 0.0` rule.
 *   - `downsample_taxa_faith_pd` (dataset.py:396-418) — farthest-point traversal on the patristic
 *     matrix, which is what `maxPdSelect` implements.
 *
 * TWO DIVERGENCES AGAINST v1.0.0 dataset.py, RECORDED AND NOT SILENTLY RESOLVED. The port follows
 * the 2.0 handoff driver, which seeds Max-PD at index 0; dataset.py:406 instead seeds with the pair
 * at `np.unravel_index(np.argmax(dist_mat), ...)` — the two most distant taxa — and only then runs
 * the same greedy loop. dataset.py also pre-strides to `2 * max_species` taxa (dataset.py:670-674)
 * before the distance matrix is built at all, so on a large upload the two implementations do not
 * even start from the same candidate set. Both change WHICH taxa reach the model, not how they are
 * scored. Which behaviour the shipped weights were fitted against is a fixture question, not an
 * inspection question, so the port is left as it is and the difference is stated here.
 *
 * The on-demand row computation described below (one row per Max-PD step instead of the full N x N)
 * is a memory optimisation with identical values, and is what makes a browser tab a possible host
 * for a 5,000-taxon submission. dataset.py materialises the whole matrix because it is not running
 * in one.
 */

/**
 * patristic.js — pairwise tree distances and Max-PD taxon selection.
 *
 * This is the first half of the AxoMEME preprocessing port, and the half that CAN be made
 * bit-identical to the Python reference. It is plain float arithmetic over a tree walk: no
 * eigendecomposition, no library-version-dependent LAPACK, nothing whose result is only unique up to
 * a sign. That is why it comes first — it establishes the parity harness on the part of the pipeline
 * where a mismatch is unambiguously a bug rather than a convention difference.
 *
 * THE REFERENCE (predict_regression_nexus.py:895-965, 1164-1188):
 *   d(a, b) = rootDist(a) + rootDist(b) - 2 * rootDist(lca(a, b))
 * computed over root distances accumulated as `current_dist + (child.branch_length or 0.0)`.
 *
 * ONE DELIBERATE DIVERGENCE, AND IT CHANGES NO RESULT. The reference materialises the full N x N
 * matrix before Max-PD (`D_full = np.zeros((N_full, N_full))`). Max-PD is a farthest-point traversal
 * that only ever reads ONE ROW per iteration, so this port computes rows on demand: `max_species + 1`
 * rows instead of N^2 cells. For a 5,000-taxon submission that is ~10 MB instead of ~100 MB, which is
 * the difference between running in a browser tab and not. The selected taxa are identical because
 * the values are identical — only the cells that are never read go uncomputed.
 *
 * NEGATIVE DISTANCES ARE PRESERVED, NOT CLAMPED. DM3's own NJ emits negative branch lengths and 5% of
 * real DM3 trees carry one at or past -0.1; the Python inference path throws on those rather than
 * degrading. Clamping here would convert a loud failure into a quiet wrong answer. See
 * src/lib/utils/treeSanitation.js for the measurement and the crash site.
 */

/**
 * Distance from the root to every node, accumulated down the tree.
 *
 * @param {import('./newick.js').default | any} tree from parseNewick
 * @returns {Float64Array}
 */
export function rootDistances(tree) {
	const dist = new Float64Array(tree.name.length);
	// Explicit stack rather than recursion: a ladder-shaped tree is as deep as it is wide, and the
	// Python reference recurses (so it caps out around 1,000 taxa on such a tree).
	const stack = [tree.root];
	dist[tree.root] = 0;
	while (stack.length) {
		const node = stack.pop();
		for (const c of tree.children[node]) {
			dist[c] = dist[node] + tree.branchLength[c];
			stack.push(c);
		}
	}
	return dist;
}

/**
 * A reusable scratch buffer for ancestor marking.
 *
 * LCA is found by marking every ancestor of `a` and walking up from `b` to the first mark. The naive
 * version clears the mark array per pair, which is O(nodes) per pair and dominates the whole
 * computation for a wide tree. Stamping with a monotonically increasing generation makes the clear
 * free.
 */
function ancestorMarker(nodeCount) {
	const stamp = new Int32Array(nodeCount);
	let generation = 0;
	return {
		/** Mark the root path of `node`, then return an `isAncestor` predicate for it. */
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
 * One row of the patristic distance matrix: distances from `fromNode` to each of `toNodes`.
 *
 * @param {any} tree
 * @param {Float64Array} rootDist from rootDistances()
 * @param {number} fromNode
 * @param {number[]} toNodes node indices, in the order the row should come out
 * @param {ReturnType<typeof ancestorMarker>} [marker] reused across rows by patristicMatrix/maxPd
 * @returns {Float64Array}
 */
export function patristicRow(tree, rootDist, fromNode, toNodes, marker) {
	const m = marker ?? ancestorMarker(tree.name.length);
	m.markPath(tree, fromNode);
	const out = new Float64Array(toNodes.length);
	for (let k = 0; k < toNodes.length; k++) {
		const b = toNodes[k];
		if (b === fromNode) {
			out[k] = 0;
			continue;
		}
		let cur = b;
		while (cur !== -1 && !m.isMarked(cur)) cur = tree.parent[cur];
		// cur === -1 cannot happen for two nodes of the same tree (the root is always marked), but a
		// caller can pass a node from a different tree, and 0 is a less destructive answer than NaN.
		out[k] = cur === -1 ? 0 : rootDist[fromNode] + rootDist[b] - 2 * rootDist[cur];
	}
	return out;
}

/**
 * The full N x N patristic matrix, row-major.
 *
 * Only for N small enough to want it whole — the model input is [max_species, max_species], so this
 * is the right call there. Max-PD deliberately does NOT use it; see maxPdSelect.
 *
 * @param {any} tree
 * @param {number[]} nodes node indices, defining row and column order
 * @returns {Float64Array} length nodes.length ** 2
 */
export function patristicMatrix(tree, nodes) {
	const n = nodes.length;
	const rootDist = rootDistances(tree);
	const marker = ancestorMarker(tree.name.length);
	const out = new Float64Array(n * n);
	for (let i = 0; i < n; i++) {
		const row = patristicRow(tree, rootDist, nodes[i], nodes, marker);
		out.set(row, i * n);
	}
	return out;
}

/**
 * Max-PD (Faith's PD) farthest-point traversal, matching predict_regression_nexus.py:1174-1183.
 *
 *   selected = [0]
 *   minDist  = D[0]
 *   repeat:  next = argmax(minDist); selected.push(next); minDist = min(minDist, D[next])
 *
 * TWO BEHAVIOURS REPRODUCED ON PURPOSE, BOTH OF WHICH LOOK LIKE BUGS:
 *
 *   1. THE SEED IS ALWAYS INDEX 0 — the first taxon in alignment order, not the most divergent one
 *      and not a deterministic function of the tree. Reordering the sequences in an upload changes
 *      which 512 taxa the model sees. That is what the model was fine-tuned against, so this port
 *      matches it; it is not a defect this layer gets to fix.
 *
 *   2. A SELECTED TAXON CAN REPEAT. Every selected index has minDist 0 (its own distance to itself
 *      enters the running minimum), so once every remaining candidate is also at 0 — an all-zero
 *      distance matrix, i.e. a tree whose branch lengths are all zero — argmax returns index 0 over
 *      and over and the same taxon fills every slot. `duplicates` reports it rather than silently
 *      deduplicating, because deduplicating would change which taxa reach a model that was trained
 *      with the duplicates present.
 *
 * @param {any} tree
 * @param {number[]} nodes candidate node indices, IN ALIGNMENT ORDER (index 0 is the seed)
 * @param {number} maxSpecies
 * @returns {{selected: number[], duplicates: number}} selected are indices INTO `nodes`
 */
export function maxPdSelect(tree, nodes, maxSpecies) {
	const n = nodes.length;
	if (n <= maxSpecies) {
		return { selected: Array.from({ length: n }, (_, i) => i), duplicates: 0 };
	}
	const rootDist = rootDistances(tree);
	const marker = ancestorMarker(tree.name.length);

	const selected = [0];
	const seen = new Set([0]);
	let duplicates = 0;
	// One row, reused. This is the whole memory argument in the header: never N x N, only N.
	const minDist = patristicRow(tree, rootDist, nodes[0], nodes, marker);

	for (let step = 1; step < maxSpecies; step++) {
		// argmax with FIRST-max-wins on ties, matching np.argmax.
		let best = 0;
		let bestVal = minDist[0];
		for (let k = 1; k < n; k++) {
			if (minDist[k] > bestVal) {
				bestVal = minDist[k];
				best = k;
			}
		}
		selected.push(best);
		if (seen.has(best)) duplicates++;
		else seen.add(best);

		const row = patristicRow(tree, rootDist, nodes[best], nodes, marker);
		for (let k = 0; k < n; k++) if (row[k] < minDist[k]) minDist[k] = row[k];
	}
	return { selected, duplicates };
}
