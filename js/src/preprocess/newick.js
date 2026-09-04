/**
 * WHY THIS FILE EXISTS
 *
 * Ported verbatim from datamonkey3 (main@fac1330) src/lib/services/axomeme/newick.js. Function
 * bodies unchanged; not one relative import needed adjusting, because every module it reaches for
 * is a sibling here exactly as it was there.
 *
 * In @veg/hyphaeon-js it is the tree-parsing half of what `hyphaeon/dataset.py` gets from Biopython:
 * `extract_tree_from_string_or_file` (dataset.py:161-212) hands `Phylo.read(..., 'newick')` a
 * cleaned string, and `compute_fast_dist_matrix` (dataset.py:302-311) then reads
 * `tree.get_terminals()`, `clade.branch_length` and `t.name.strip("'\"")` off the result. Those four
 * behaviours — terminal order, a missing branch length as 0.0, quote stripping, internal labels that
 * are not taxa — are what this file reproduces without a dependency.
 *
 * WHAT PHASE 0 HAS NOT YET SETTLED, and it applies to every seeded module in this directory: the
 * datamonkey3 port was written against the AxoMEME 2.0 handoff scripts
 * (`predict_regression_nexus.py`, `train_transformer_selection.py`), which the citations below name.
 * `hyphaeon/*.py` at v1.0.0 is the reference now, and it is NOT line-for-line the same code. The
 * divergences that matter are recorded in tokenizer.js, mds.js, patristic.js and variability.js, and
 * the fixture harness (`scripts/gen_fixtures.py`, replayed by `js/test`) is what will decide each
 * one. Nothing here has been "corrected" toward dataset.py on inspection alone: an unverified
 * rewrite of a verified port is strictly worse than a verified port with a known question against it.
 */

/**
 * newick.js — a minimal newick parser for the AxoMEME preprocessing path.
 *
 * WHY NOT phylotree, WHICH THIS REPO ALREADY DEPENDS ON. phylotree is a rendering library: it pulls
 * d3, it is built for interactive branch selection, and it is used in exactly one place
 * (BranchSelector.svelte). What is needed here is the opposite of that — a leaf module with no
 * dependencies that produces a tree whose branch lengths and names match, exactly, what Biopython's
 * `Phylo.read` hands to the Python reference implementation. Parity is the whole job, and it is much
 * easier to defend against a 100-line parser whose semantics are written down than against a
 * general-purpose library that also happens to draw SVG.
 *
 * WHAT IT DELIBERATELY MATCHES (predict_regression_nexus.py:895-905):
 *   - A MISSING BRANCH LENGTH IS 0.0, not null and not NaN. The reference walks with
 *     `current_dist + (child.branch_length or 0.0)`, so an absent length contributes nothing rather
 *     than poisoning every descendant. Note this also maps a length of exactly 0.0 to 0.0, so the
 *     `or` is harmless there — but a NEGATIVE length is preserved, which matters: DM3's own NJ emits
 *     them (see src/lib/utils/treeSanitation.js) and silently clamping here would hide that.
 *   - INTERNAL NODES KEEP THEIR LABELS but are never leaves. A bootstrap value like `)95:0.05` is a
 *     label on an internal node, and reading it as a taxon name would invent a species.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: it does not clamp, normalise, or repair anything. A tree that is
 * wrong arrives wrong, so the caller can refuse it with the user told rather than scoring a quietly
 * altered tree.
 */

/**
 * Strip a quoted or bare newick label down to the name the reference pipeline compares against.
 *
 * The reference normalises with `s.replace("'", "").replace('"', '').strip()` — Python's str.replace
 * removes EVERY occurrence, not just the first — and does it at comparison time
 * (predict_regression_nexus.py:1221-1222). Quotes are removed wherever they appear, not just at the
 * ends, so a label like `Homo_'sapiens'` and `Homo_sapiens` normalise to the same taxon.
 *
 * @param {string} name
 * @returns {string}
 */
export function normalizeTaxonName(name) {
	return String(name).replaceAll("'", '').replaceAll('"', '').trim();
}

/**
 * Parse a newick string.
 *
 * Returns a flat node array plus the root index, rather than a linked object graph. That shape is
 * chosen on purpose: every consumer here walks the tree by index (root distances, LCA, Max-PD), flat
 * typed arrays keep those walks allocation-free, and there are no parent cycles for a structured
 * clone or a test's `toEqual` to trip over.
 *
 * @param {string} text
 * @returns {{
 *   name: string[], parent: Int32Array, branchLength: Float64Array,
 *   children: number[][], depth: Int32Array, root: number, leaves: number[]
 * }}
 */
export function parseNewick(text) {
	if (typeof text !== 'string' || !text.trim()) {
		throw new Error('parseNewick: empty input');
	}

	// Strip newick comments before tokenising. `[&&NHX:...]` and `[100]` both appear in the wild and
	// neither is part of the topology; leaving them in makes a comment body look like a label.
	const src = text.replace(/\[[^\]]*\]/g, '');

	const name = [];
	const parent = [];
	const branchLength = [];
	const children = [];

	const newNode = (parentIdx) => {
		const idx = name.length;
		name.push('');
		parent.push(parentIdx);
		branchLength.push(0);
		children.push([]);
		if (parentIdx >= 0) children[parentIdx].push(idx);
		return idx;
	};

	let i = 0;
	const n = src.length;
	const skipSpace = () => {
		while (i < n && /\s/.test(src[i])) i++;
	};

	/** Read a label: quoted (quotes preserved, caller normalises) or bare up to a delimiter. */
	const readLabel = () => {
		skipSpace();
		if (i >= n) return '';
		const q = src[i];
		if (q === "'" || q === '"') {
			// A quoted label may legally contain ':' ',' '(' ')' — which is the entire reason quoting
			// exists, and the reason a naive split on ':' corrupts these names.
			let out = q;
			i++;
			while (i < n && src[i] !== q) out += src[i++];
			if (i < n) out += src[i++]; // closing quote
			return out;
		}
		let out = '';
		while (i < n && !'(),:;'.includes(src[i])) out += src[i++];
		return out.trim();
	};

	/** Read `:length` if present. Absent means 0, matching `branch_length or 0.0`. */
	const readLength = (node) => {
		skipSpace();
		if (src[i] !== ':') return;
		i++;
		skipSpace();
		let out = '';
		while (i < n && !'(),;'.includes(src[i])) out += src[i++];
		const v = parseFloat(out);
		// A malformed length is 0, not NaN. NaN would propagate silently through every descendant's
		// root distance and surface much later as an all-NaN model input.
		branchLength[node] = Number.isFinite(v) ? v : 0;
	};

	// Iterative descent. A recursive parser is simpler but a 5,000-taxon ladder tree is 5,000 frames
	// deep, and the reference implementation has exactly that latent bug (Python's traverse() is
	// recursive and caps at ~1,000 frames).
	const root = newNode(-1);
	let current = root;
	skipSpace();

	if (src[i] === '(') {
		i++;
		current = newNode(root);
		while (i < n) {
			skipSpace();
			const c = src[i];
			if (c === '(') {
				i++;
				current = newNode(current);
			} else if (c === ',') {
				i++;
				current = newNode(parent[current]);
			} else if (c === ')') {
				i++;
				current = parent[current];
				name[current] = readLabel(); // internal label / bootstrap
				readLength(current);
				if (current === root) break;
			} else if (c === ';' || i >= n) {
				break;
			} else {
				name[current] = readLabel();
				readLength(current);
			}
		}
	} else {
		// A bare single-taxon "tree".
		name[root] = readLabel();
		readLength(root);
	}

	const count = name.length;
	const depth = new Int32Array(count);
	const leaves = [];
	// PREORDER, depth-first, left to right — matching Biopython's `tree.get_terminals()`, which is
	// `find_clades(terminal=True, order='preorder')`. The order is not cosmetic: the reference builds
	// its name lookup as `{leaf.name: leaf for leaf in leaves}`, so on a tree with duplicate tip names
	// the LAST leaf in THIS order wins, and 1.1% of real DM3 trees have duplicate tips. A
	// breadth-first traversal produces the same set and a different winner.
	const stack = [root];
	depth[root] = 0;
	while (stack.length) {
		const node = stack.pop();
		if (children[node].length === 0) {
			leaves.push(node);
			continue;
		}
		// Reversed, so the leftmost child is popped first and the walk is genuinely left-to-right.
		for (let k = children[node].length - 1; k >= 0; k--) {
			const c = children[node][k];
			depth[c] = depth[node] + 1;
			stack.push(c);
		}
	}

	return {
		name,
		parent: Int32Array.from(parent),
		branchLength: Float64Array.from(branchLength),
		children,
		depth,
		root,
		leaves
	};
}

/**
 * Map normalised taxon name -> node index, for the named leaves only.
 *
 * A DUPLICATE NAME RESOLVES TO THE LAST LEAF IN PREORDER, matching the reference's
 * `{leaf.name: leaf for leaf in leaves if leaf.name}` — a dict comprehension, so later entries
 * overwrite earlier ones.
 *
 * This was originally written the other way round, keeping the first leaf, on the reasoning that a
 * tree with two identically-named tips is malformed and reproducing a dict-overwrite artifact was
 * not worth doing. Measurement settled it: 3 of 270 real DM3 trees (1.1%) have duplicate tips — one
 * of them has only 36 distinct names across 100 tips — and on every one of those the two policies
 * pick different tips and therefore compute different distances. The parity run failed on exactly
 * those three and nothing else.
 *
 * So this layer matches the reference, because its whole job is fidelity to what the model was
 * trained and served on, and `duplicates` is returned so the CALLER can refuse the tree. Refusing is
 * a product decision; quietly disagreeing with the reference is not.
 *
 * @param {ReturnType<typeof parseNewick>} tree
 * @returns {{index: Map<string, number>, duplicates: string[]}}
 */
export function leafIndex(tree) {
	const index = new Map();
	const duplicates = [];
	for (const leaf of tree.leaves) {
		const nm = normalizeTaxonName(tree.name[leaf]);
		if (!nm) continue;
		if (index.has(nm)) duplicates.push(nm);
		index.set(nm, leaf); // last wins
	}
	return { index, duplicates };
}
