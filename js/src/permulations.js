/**
 * WHY THIS FILE EXISTS
 *
 * The Brownian-motion null of the phenotype pillar, mirroring `hyphaeon/phenotype.py` at
 * veg/HyphAeon reconcile/phase-5a:
 *
 *   computePhylogeneticCovariance   compute_phylogenetic_covariance, phenotype.py:275-326
 *   generatePermulations            generate_permulations,           phenotype.py:327-374
 *
 * It is a separate module from `phenotype.js` for the reason PLAN.md §5.1 splits `epistasis.py`
 * into three: this half is the ONLY part of the pillar that needs a tree, and D22 made a tree
 * optional (no tree, or a tree without branch lengths -> tree-free TN93 distances). Keeping the
 * tree-dependent half in its own file makes "the app has no tree, so there are no permulations"
 * a fact about which module the pipeline can call, not a branch buried inside a 600-line driver.
 * `run_phenotype_association` says the same thing at phenotype.py:454 (`permulations > 0 and
 * tree_obj is not None`), and `runPhenotypeAssociation` calls into here only when it has a tree.
 *
 * WHAT IT DELIBERATELY DOES NOT DO. It does not read a newick file (`readNewick` /
 * `extractTree` in preprocess/tree.js do that and hand over a `PhyloTree`), it does not build a
 * tree from distances, and it does not decide whether a tree is usable — `hasNonzeroBranchLengths`
 * is the app's gate. There is no numpy: `V` is a flat row-major `Float64Array`, as
 * `preprocess/patristic.js` returns its matrices.
 *
 * ---------------------------------------------------------------------------------------------
 * compute_phylogenetic_covariance, step by step (phenotype.py line numbers):
 *
 *   1. `root_dists = {t.name: tree.distance(root, t) for t in tree.get_terminals()}`        283
 *      `tree.distance(root, t)` is Bio.Phylo `TreeMixin.distance(target1, target2)`: the MRCA of
 *      the root and a node is the root, so it reduces to the sum of branch lengths on the
 *      root-to-node path, a `None` length contributing 0 — exactly `patristic.rootDistances`.
 *      The comprehension is keyed by NAME, so duplicate terminal names collapse to the LAST one
 *      in preorder.
 *   2. `V[i, i] = root_dists.get(taxa[i], 1.0)`                                             291
 *      A taxon absent from the tree gets 1.0 on the diagonal and 0 everywhere off it.
 *   3. `t = tree.find_any(name=taxa[k])` for the off-diagonal entries                    286, 288
 *      and `V[i, j] = V[j, i] = tree.distance(root, common_ancestor(t1, t2))`             292-295
 *      i.e. the root-to-MRCA depth: the shared path length under Brownian motion.
 *      Missing taxa leave the entry at 0.
 *
 * THE `find_any` QUIRK, replicated. Bio.Phylo's `_attribute_matcher` (BaseTree.py, Biopython
 * 1.85 — the version the fixtures were generated with) treats a STRING attribute value as a
 * REGULAR EXPRESSION: `re.match(pattern + "$", target)`. So `find_any(name="foo.bar")` matches
 * the terminal `fooXbar` as well, `find_any(name="a|b")` matches a terminal named `a` (re.match
 * anchors only the start, and the trailing `$` binds to the last alternative), and a name with an
 * unbalanced bracket raises `re.error` out of the whole function. `findAnyByName` below
 * reproduces this, translating the pattern to a JS RegExp with a `y` flag (which is `re.match`'s
 * anchor-at-0) and appending `$` outside any group, as the reference does. A name with no regex
 * metacharacter is equality, so those go through a precomputed first-occurrence map — the same
 * answer, without M x V regex evaluations.
 *
 * MATCH ORDER. `find_any` is `next(find_elements(...))`, which is `find_clades()` order: depth
 * first PREORDER over every clade, internal nodes included. An internal node whose name matches
 * therefore wins over a terminal deeper in the tree, and the MRCA of two internal nodes is a
 * perfectly ordinary answer for this function. `findClades` in preprocess/tree.js is that order.
 *
 * ---------------------------------------------------------------------------------------------
 * generate_permulations, step by step:
 *
 *   1. `np.random.seed(seed)`; `V = compute_phylogenetic_covariance(tree, taxa)`         318-320
 *   2. `L = np.linalg.cholesky(V + 1e-7 * np.eye(M))`                                        323
 *      The 1e-7 ridge is the reference's, on the raw covariance: it is NOT scaled by the
 *      magnitude of V, so on a tree whose branch lengths are in the tens (bat_oas1) it is
 *      negligible and on a tree scaled to substitutions per site (Smc6, max depth 0.041) it is
 *      about 2.4e-6 of the diagonal. A V that is still not positive definite raises
 *      `LinAlgError` in the reference and a `RangeError` here; `run_phenotype_association`
 *      catches it and drops the permulations (phenotype.py:469-471), and so does
 *      `runPhenotypeAssociation`.
 *   3. `Z = L @ np.random.randn(M, n_perm)`                                                  326
 *   4. binary y (exactly two unique values, both in {0, 1}): `k = int(np.sum(y > 0))` and the
 *      top-k entries of each COLUMN of Z become 1                                        332-336
 *   5. otherwise: rank matching, `sorted(y)[argsort(argsort(Z[:, p]))]`                  338-341
 *
 * RANDOMNESS IS A CONTRACT, NOT A BIT PATTERN (PLAN.md §5.3 rule 5, §5.4). The reference draws
 * from numpy's LEGACY global state — `np.random.seed(42)` then `np.random.randn`, i.e. MT19937
 * with numpy's polar-method gaussian — and this port draws from Xoshiro256's Box–Muller
 * `normal()`. The two streams cannot agree element by element at any seed, so parity for
 * everything downstream of here is statistical: `fixtures/phenotype/generate_permulations.json`
 * records per-taxon foreground frequencies (binary) and per-taxon means (continuous) rather than
 * the matrix, and `test/permulations.test.js` compares those, plus the moments and the empirical
 * covariance of the draws themselves against V. What IS exact and is tested as such: every binary
 * row sums to k, and every continuous row is a permutation of `y`.
 *
 * FILL ORDER. `np.random.randn(M, n_perm)` fills C-order, so draw (i, p) is the (i * n_perm + p)th
 * variate. Nothing observable depends on it — the columns are exchangeable under any fill order —
 * but matching it keeps the two implementations describable by the same sentence, and
 * `returnDraws` hands the raw [M, n_perm] matrix back so a test can check its covariance.
 *
 * ---------------------------------------------------------------------------------------------
 * QUIRKS REPLICATED (fix upstream first — PLAN.md §5.3 rule 3):
 *   - `np.argsort(Z[:, p])[-k_foreground:]` with k_foreground == 0 is `[-0:]`, which Python
 *     evaluates as `[0:]` — the WHOLE column, so every entry would become 1. The branch is
 *     UNREACHABLE: `is_binary` requires exactly two unique values drawn from {0, 1}, which forces
 *     at least one 1 and therefore k >= 1. An all-zero y has ONE unique value and goes down the
 *     rank-matching branch instead, coming back all zeros. The slice semantics are kept as the
 *     reference wrote them so that a future y with a different binary test does not change
 *     behaviour silently; `test/permulations.test.js` pins the all-zero case's actual output.
 *   - The diagonal of V is the root-to-TIP distance while the off-diagonal is the root-to-MRCA
 *     distance, so V is a valid Brownian covariance only when every tip is equidistant from the
 *     root; on a non-ultrametric tree V[i, i] exceeds what the model implies and the null is
 *     slightly over-dispersed. That is the published RERconverge/Saputra construction and it is
 *     kept as is.
 *   - `np.unique` on a float vector makes {0.0, 1.0, 1.0000000001} non-binary and {0.0} (all
 *     background) non-binary too — an all-zero y goes down the rank-matching branch and comes
 *     back all zeros.
 *   - `np.argsort` is numpy's introsort and is NOT stable; this port sorts stably by index. Two
 *     exactly equal Gaussian draws would be ordered differently, which no float64 stream produces
 *     in practice.
 */

import { findClades, getTerminals } from './preprocess/tree.js';
import { rootDistances } from './preprocess/patristic.js';
import { cholesky } from './numeric/linalg.js';
import { Xoshiro256 } from './numeric/prng.js';

/**
 * @typedef {import('./preprocess/tree.js').PhyloTree} PhyloTree
 */

/** Characters that make a Python pattern more than a literal string. */
const RE_METACHARS = /[.^$*+?{}[\]\\|()]/;

/**
 * Python `re` source translated to JavaScript RegExp source. Only the constructs whose SYNTAX
 * differs are rewritten; the rest of the two languages' regex grammars coincide for the patterns
 * a taxon name or a foreground rule can plausibly be.
 *
 * KNOWN DIFFERENCES that are not translated, and cannot be without a full parser:
 *   - `\d`, `\w`, `\s`, `\b` are Unicode-aware in Python 3 for `str` patterns and ASCII-only in
 *     JavaScript without `u` + property escapes. A taxon name is ASCII in every bundled example.
 *   - Python's `$` also matches immediately before a trailing newline; JavaScript's does not
 *     (without the `m` flag). Taxon names do not contain newlines.
 *   - `(?#comment)`, conditional references `(?(1)a|b)`, possessive quantifiers and atomic groups
 *     `(?>...)` (Python 3.11+) have no JavaScript equivalent and raise a SyntaxError here, where
 *     Python would have compiled them. Both `findAnyByName` and the foreground matcher treat a
 *     failed compile the way the reference treats `re.error` at that point.
 * The `u` flag is deliberately NOT set: without it JavaScript accepts the lenient Annex B forms
 * (a bare `]`, `{`, or a redundant `\-`) that Python also accepts.
 *
 * @param {string} source Python regex source
 * @returns {string} JavaScript regex source
 */
export function pyRegexSource(source) {
	// (?P<name>...) -> (?<name>...) and (?P=name) -> \k<name>; both are Python-only spellings.
	return String(source)
		.replace(/\(\?P</g, '(?<')
		.replace(/\(\?P=([A-Za-z_]\w*)\)/g, '\\k<$1>');
}

/**
 * `Bio.Phylo` `tree.find_any(name=name)`: the first clade in `find_clades()` preorder whose name
 * matches `re.match(name + "$", clade.name)`.
 *
 * @param {PhyloTree} tree
 * @param {string} name the reference's attribute value — a REGEX, see the file header
 * @returns {number} node index, or -1 when nothing matches (the reference's `None`)
 * @throws {SyntaxError} when the pattern does not compile (the reference's `re.error`)
 */
export function findAnyByName(tree, name) {
	const target = String(name);
	const order = findClades(tree);
	if (!RE_METACHARS.test(target)) {
		// A literal pattern: `re.match(lit + "$", x)` is `x == lit`.
		for (const n of order) if (tree.name[n] === target) return n;
		return -1;
	}
	// `y` anchors at lastIndex 0, which is what re.match does; the `$` is appended OUTSIDE any
	// group, exactly as `pattern + "$"` does in the reference.
	const re = new RegExp(pyRegexSource(target) + '$', 'y');
	for (const n of order) {
		const nm = tree.name[n];
		if (typeof nm !== 'string') continue;
		re.lastIndex = 0;
		if (re.test(nm)) return n;
	}
	return -1;
}

/**
 * `compute_phylogenetic_covariance(tree, taxa)` (phenotype.py:275-326).
 *
 * V[i, i] is the root-to-tip distance of `taxa[i]` (1.0 when the tree has no terminal of that
 * name); V[i, j] is the root-to-MRCA distance of the two clades `find_any` returns, and 0 when
 * either is missing. Row-major float64, as `patristicMatrix` returns.
 *
 * @param {PhyloTree} tree
 * @param {ArrayLike<string>} taxa
 * @returns {Float64Array} length taxa.length ** 2
 */
export function computePhylogeneticCovariance(tree, taxa) {
	const M = taxa.length;
	const V = new Float64Array(M * M);
	const depth = rootDistances(tree);

	// phenotype.py:297 — {t.name: distance(root, t) for t in get_terminals()}; a repeated name
	// keeps the LAST terminal's depth, as the dict comprehension does.
	/** @type {Map<string|null, number>} */
	const rootDists = new Map();
	for (const t of getTerminals(tree)) rootDists.set(tree.name[t], depth[t]);

	// find_any per taxon, once (the reference calls it once per i and once per j; the result for a
	// given name is the same node every time).
	const node = new Int32Array(M);
	for (let i = 0; i < M; i++) node[i] = findAnyByName(tree, taxa[i]);

	// Root paths, for the MRCA. Bio.Phylo's common_ancestor walks the two root paths (root
	// excluded) in step and stops at the first level where they differ; the deepest shared clade
	// is the MRCA, and the root when they differ immediately.
	/** @type {(Int32Array|null)[]} */
	const paths = new Array(M);
	for (let i = 0; i < M; i++) {
		if (node[i] < 0) {
			paths[i] = null;
			continue;
		}
		const rev = [];
		for (let n = node[i]; n !== tree.root && n >= 0; n = tree.parent[n]) rev.push(n);
		rev.reverse();
		paths[i] = Int32Array.from(rev);
	}

	for (let i = 0; i < M; i++) {
		const di = rootDists.has(taxa[i]) ? /** @type {number} */ (rootDists.get(taxa[i])) : 1.0;
		V[i * M + i] = di;
		const pi = paths[i];
		if (pi === null) continue;
		for (let j = i + 1; j < M; j++) {
			const pj = paths[j];
			if (pj === null) continue;
			let mrca = tree.root;
			const lim = Math.min(pi.length, pj.length);
			for (let k = 0; k < lim; k++) {
				if (pi[k] !== pj[k]) break;
				mrca = pi[k];
			}
			const shared = depth[mrca];
			V[i * M + j] = shared;
			V[j * M + i] = shared;
		}
	}
	return V;
}

/**
 * @typedef {{
 *   permMatrix: Float64Array|null, nPerm: number, M: number, isBinary: boolean, seed: number,
 *   kForeground: number, V: Float64Array|null, draws: Float64Array|null,
 *   skipped: {reason: string, detail: string}|null
 * }} PermulationResult
 */

/**
 * `generate_permulations(original_y, tree, taxa, n_perm=1000, seed=42)` (phenotype.py:327-374).
 *
 * `permMatrix` is [n_perm, M] row-major float64. TREE-FREE MODE: the reference only calls this
 * when `tree_obj is not None` (phenotype.py:454), so a null `tree` here is not an error — it
 * returns the same record with `permMatrix: null` and `skipped: {reason: 'no-tree', detail}` so a
 * caller can say WHY a report has no permulation p-values instead of silently showing none.
 * A tree without branch lengths is the caller's decision (`hasNonzeroBranchLengths`); passed in,
 * it gives an all-zero V and a Cholesky failure, reported as `skipped.reason: 'not-positive-definite'`.
 *
 * @param {ArrayLike<number>} y the trait vector over `taxa`
 * @param {PhyloTree|null|undefined} tree
 * @param {ArrayLike<string>} taxa
 * @param {{nPerm?: number, seed?: number|bigint, returnDraws?: boolean, returnV?: boolean}} [options]
 *   `returnDraws` also returns the raw [M, n_perm] Gaussian matrix `L @ randn` (NOT a reference
 *   output; it is how test/permulations.test.js checks the covariance structure). `returnV`
 *   returns the covariance matrix that produced it.
 * @returns {PermulationResult}
 */
export function generatePermulations(y, tree, taxa, options = {}) {
	const M = taxa.length;
	const nPerm = Math.max(0, Math.floor(options.nPerm ?? 1000));
	const seedOpt = options.seed ?? 42;
	const seed = typeof seedOpt === 'bigint' ? Number(BigInt.asIntN(53, seedOpt)) : seedOpt;

	// phenotype.py:356-358 — np.unique(original_y): sorted distinct values.
	const uniq = new Set();
	for (let i = 0; i < M; i++) uniq.add(y[i]);
	let isBinary = uniq.size === 2;
	if (isBinary) for (const v of uniq) if (v !== 0 && v !== 1) isBinary = false;
	let kForeground = 0;
	for (let i = 0; i < M; i++) if (y[i] > 0) kForeground++;

	/** @param {string} reason @param {string} detail @returns {PermulationResult} */
	const skip = (reason, detail) => ({
		permMatrix: null, nPerm, M, isBinary, seed, kForeground,
		V: null, draws: null, skipped: { reason, detail }
	});

	if (tree === null || tree === undefined) {
		return skip(
			'no-tree',
			'phenotype.py:454 runs permulations only when tree_obj is not None; in tree-free mode ' +
				'(PLAN.md D22) there is no tree to draw a Brownian null from.'
		);
	}

	const V = computePhylogeneticCovariance(tree, taxa);

	// phenotype.py:351 — cholesky(V + 1e-7 * I). The ridge is added to a copy; V itself is
	// returned unmodified when the caller asked for it.
	const ridged = Float64Array.from(V);
	for (let i = 0; i < M; i++) ridged[i * M + i] += 1e-7;
	/** @type {Float64Array} */
	let L;
	try {
		L = cholesky(ridged, M);
	} catch (err) {
		return skip('not-positive-definite', /** @type {Error} */ (err).message);
	}

	// phenotype.py:354 — np.random.randn(M, n_perm), C-order, then Z = L @ R.
	const rng = new Xoshiro256(seed);
	const R = new Float64Array(M * nPerm);
	for (let i = 0; i < M * nPerm; i++) R[i] = rng.normal();
	const Z = new Float64Array(M * nPerm);
	for (let i = 0; i < M; i++) {
		for (let k = 0; k <= i; k++) {
			const l = L[i * M + k];
			if (l === 0) continue;
			for (let p = 0; p < nPerm; p++) Z[i * nPerm + p] += l * R[k * nPerm + p];
		}
	}

	const permMatrix = new Float64Array(nPerm * M);
	const col = new Float64Array(M);
	const idx = new Int32Array(M);
	if (isBinary) {
		for (let p = 0; p < nPerm; p++) {
			for (let i = 0; i < M; i++) {
				col[i] = Z[i * nPerm + p];
				idx[i] = i;
			}
			const order = Array.from(idx).sort((a, b) => col[a] - col[b] || a - b);
			// QUIRK (phenotype.py:362): `[-k:]` with k == 0 is `[0:]`, the whole column.
			const start = kForeground === 0 ? 0 : M - kForeground;
			for (let r = start; r < M; r++) permMatrix[p * M + order[r]] = 1.0;
		}
	} else {
		const sortedOrig = Float64Array.from({ length: M }, (_, i) => y[i]).sort();
		for (let p = 0; p < nPerm; p++) {
			for (let i = 0; i < M; i++) {
				col[i] = Z[i * nPerm + p];
				idx[i] = i;
			}
			const order = Array.from(idx).sort((a, b) => col[a] - col[b] || a - b);
			// argsort(argsort(x))[i] is the rank of element i.
			for (let r = 0; r < M; r++) permMatrix[p * M + order[r]] = sortedOrig[r];
		}
	}

	return {
		permMatrix, nPerm, M, isBinary, seed, kForeground,
		V: options.returnV ? V : null,
		draws: options.returnDraws ? Z : null,
		skipped: null
	};
}
