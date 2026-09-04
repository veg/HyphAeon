/**
 * WHY THIS FILE EXISTS
 *
 * The tree half of `hyphaeon/dataset.py` at veg/HyphAeon 267f5cf, which the reference gets from
 * Biopython and which this package reproduces without a dependency:
 *   - `Bio.Phylo.NewickIO.Parser` (Biopython 1.85, the version the fixtures were generated with):
 *     `parseNewickTrees` is `Parser.parse` (line-buffered on ';'), `readNewick` is `Phylo.read`
 *     (exactly one tree or ValueError). The tokenizer regex, the clade-building state machine,
 *     `process_clade`'s confidence-from-label rule and the parenthesis / text-after-semicolon
 *     errors are transcribed one for one, because dataset.py catches parse errors and falls through
 *     to the next extraction strategy, so WHICH strings fail is part of the behaviour.
 *   - `extract_tree_from_string_or_file` (dataset.py:161-212), adapted to a string: the NEXUS/HyPhy
 *     `TREE name = (...);` regex, then any line starting with '(' that has >= 2 '(', then the whole
 *     content; `{...}` (HyPhy tags) and `[...]` (NEXUS comments) stripped before parsing.
 *   - `has_nonzero_branch_lengths` (214-222), `enforce_nonzero_branch_lengths` (289-300).
 *   - the three-tier taxon matching of `load_alignment_and_tree` (614-642): exact, then
 *     quote-stripped alignment names, then case-insensitive; matched taxa in TREE TERMINAL ORDER.
 *   - `find_clades()` / `get_terminals()` preorder (Bio.Phylo.BaseTree `_preorder_traverse`).
 *
 * The tree is a flat node table (`name`, `branchLength`, `confidence`, `comment`, `children`,
 * `parent`, `root`) with `null` where Biopython has `None`: `branchLength[i] === null` is a missing
 * length, which `enforce_nonzero_branch_lengths` treats differently from 0.0 (1e-3 vs 1e-4), so the
 * distinction has to survive parsing.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: `estimate_tree_branch_lengths_hyphy` (dataset.py:224-287)
 * shells out to HyPhy. A library cannot. `needsBranchLengths(tree)` is the predicate the reference
 * uses to decide whether to call HyPhy (`not has_nonzero_branch_lengths`); the runtime can supply an
 * estimated tree and call `loadAlignmentAndTree` again with it. Without one, `loadAlignmentAndTree`
 * takes the reference's "HyPhy not found" branch (dataset.py:610-614): enforce 1e-3 / 1e-4 and go on.
 *
 * QUIRKS REPLICATED ON PURPOSE (pinned by fixtures/dataset/extract_tree_from_string_or_file.json):
 *   - `TREE x = [&R] (...)` is rejected: the rooting tag sits between '=' and '(' so the regex does
 *     not match, and the line does not start with '(' (nexus_tree_block case -> null).
 *   - A single-pair newick `(A:0.1,B:0.2);` has one '(' and is rejected (single_pair case).
 *   - A double-quoted name with a space, `"sp two"`, is split by Biopython's tokenizer into two
 *     unquoted labels and the SECOND wins: terminal name `two"` (quoted_names case). Single quotes
 *     are real quoting and keep spaces.
 *   - `enforce_nonzero_branch_lengths` raises a NEGATIVE length to 1e-4 too (`< min_len`).
 *
 * DIVERGENCE FROM THE DATAMONKEY3 PORT (main@fac1330 src/lib/services/axomeme/newick.js), which this
 * file replaces: DM3's parser treated '"' as a quote character, stripped `[...]` itself, read a
 * missing length as 0 rather than None, and normalised names with `replace` of EVERY quote character
 * (`normalizeTaxonName`) where dataset.py uses `strip("'\"")` on the ends only (`stripQuotes` here).
 * Its preorder leaf order and last-duplicate-wins lookup agree with Biopython and are kept in spirit
 * (`getTerminals`, and the `terminals` dict in patristic.js).
 */

import { PY_WS, pyStrip, pyRstrip, pyStripChars, pySplitLines, pyIsDigit } from './parse.js';

/** Bio.Phylo.NewickIO.NewickError. */
export class NewickError extends Error {}

/**
 * Bio.Phylo.NewickIO `tokens`, in order (alternation order matters: the first alternative that
 * matches at a position wins, and characters no alternative matches are skipped, as re.finditer).
 *   \(                         open parens
 *   \)                         close parens
 *   [^\s()\[\]':;,]+           unquoted node label  (note: '"' is NOT excluded)
 *   : ?[+-]?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?   edge length
 *   ,                          comma
 *   \[(\\.|[^\]])*\]           comment
 *   '(\\.|[^'])*'              quoted node label
 *   ;                          semicolon
 *   \n                         newline
 */
const TOKEN_RE = new RegExp(
	'\\(|\\)|' +
		`[^${PY_WS.slice(1, -1)}()\\[\\]':;,]+|` +
		': ?[+-]?[0-9]*\\.?[0-9]+(?:[eE][+-]?[0-9]+)?|' +
		',|' +
		'\\[(?:\\\\.|[^\\]])*\\]|' +
		"'(?:\\\\.|[^'])*'|" +
		';|' +
		'\\n',
	'gu'
);

/**
 * @typedef {{
 *   name: (string|null)[],
 *   branchLength: (number|null)[],
 *   confidence: (number|null)[],
 *   comment: (string|null)[],
 *   children: number[][],
 *   parent: Int32Array,
 *   root: number
 * }} PhyloTree
 */

/** `float(text)` for the strings Python accepts; null where Python raises ValueError. */
function pyFloat(text) {
	const t = pyStrip(text).toLowerCase().replaceAll('_', '');
	if (/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/.test(t)) return Number(t);
	if (/^[+-]?(?:inf|infinity)$/.test(t)) return t.startsWith('-') ? -Infinity : Infinity;
	if (/^[+-]?nan$/.test(t)) return NaN;
	return null;
}

/** NewickIO `_parse_confidence`: int for a digit string, float where parseable, else None. */
function parseConfidence(text) {
	if (pyIsDigit(text)) return Number.parseInt(text, 10);
	return pyFloat(text);
}

/**
 * NewickIO `Parser._parse_tree` on one buffered tree string.
 *
 * @param {string} text
 * @returns {PhyloTree}
 */
function parseTree(text) {
	/** @type {(string|null)[]} */ const name = [];
	/** @type {(number|null)[]} */ const branchLength = [];
	/** @type {(number|null)[]} */ const confidence = [];
	/** @type {(string|null)[]} */ const comment = [];
	/** @type {number[][]} */ const children = [];
	// The parser's temporary `clade.parent` reference: set by new_clade(parent), consumed (deleted)
	// by process_clade. -1 means "no parent attribute".
	/** @type {number[]} */ const pending = [];

	const newClade = (parent = -1) => {
		const idx = name.length;
		name.push(null);
		branchLength.push(null);
		confidence.push(null);
		comment.push(null);
		children.push([]);
		pending.push(parent);
		return idx;
	};

	/** process_clade: confidence-from-label, then hand the clade to its parent. Returns parent or -1. */
	const processClade = (c) => {
		const nm = name[c];
		if (nm && confidence[c] === null && children[c].length) {
			const conf = parseConfidence(nm);
			if (conf !== null) {
				confidence[c] = conf;
				name[c] = null;
			}
		}
		const p = pending[c];
		if (p === -1) return -1;
		children[p].push(c);
		pending[c] = -1;
		return p;
	};

	let rootClade = newClade();
	let current = rootClade;
	let lp = 0;
	let rp = 0;
	let sawSemicolon = false;
	let trailing = null;

	for (const m of pyStrip(text).matchAll(TOKEN_RE)) {
		const token = m[0];
		if (sawSemicolon) {
			trailing = token;
			break;
		}
		if (token.startsWith("'")) {
			// Quoted label. A second quoted token on the same clade is the escaped-quote hack.
			if (!name[current]) name[current] = token.slice(1, -1);
			else name[current] += token.slice(0, -1);
		} else if (token.startsWith('[')) {
			comment[current] = token.slice(1, -1);
		} else if (token === '(') {
			current = newClade(current);
			lp++;
		} else if (token === ',') {
			if (current === rootClade) {
				rootClade = newClade();
				pending[current] = rootClade;
			}
			const parent = processClade(current);
			current = newClade(parent);
		} else if (token === ')') {
			const parent = processClade(current);
			if (parent === -1) throw new NewickError('Parenthesis mismatch.');
			current = parent;
			rp++;
		} else if (token === ';') {
			sawSemicolon = true;
		} else if (token.startsWith(':')) {
			branchLength[current] = Number(token.slice(1).replace(/^ /, ''));
		} else if (token === '\n') {
			// pass
		} else {
			name[current] = token;
		}
	}

	if (lp !== rp) throw new NewickError(`Mismatch, ${lp} open vs ${rp} close parentheses.`);
	if (trailing !== null) throw new NewickError(`Text after semicolon in Newick tree: ${trailing}`);

	processClade(current);
	processClade(rootClade);

	const parent = new Int32Array(name.length).fill(-1);
	for (let p = 0; p < children.length; p++) for (const c of children[p]) parent[c] = p;
	return { name, branchLength, confidence, comment, children, parent, root: rootClade };
}

/**
 * NewickIO `Parser.parse`: lines are `rstrip`ped and buffered; a buffer ending in ';' is one tree;
 * a non-empty remainder is a last tree without its ';'. Lines split on '\n' only (StringIO).
 *
 * @param {string} text
 * @returns {PhyloTree[]}
 */
export function parseNewickTrees(text) {
	const trees = [];
	let buf = '';
	for (const line of String(text).split('\n')) {
		buf += pyRstrip(line);
		if (buf.endsWith(';')) {
			trees.push(parseTree(buf));
			buf = '';
		}
	}
	if (buf) trees.push(parseTree(buf));
	return trees;
}

/**
 * `Bio.Phylo.read(handle, 'newick')`: exactly one tree, else ValueError.
 *
 * @param {string} text
 * @returns {PhyloTree}
 */
export function readNewick(text) {
	const trees = parseNewickTrees(text);
	if (trees.length === 0) throw new Error('There are no trees in this file.');
	if (trees.length > 1) throw new Error('There are multiple trees in this file; use parse() instead.');
	return trees[0];
}

/** `tree.find_clades()`: every node, depth-first preorder from the root. */
export function findClades(tree) {
	const out = [];
	const stack = [tree.root];
	while (stack.length) {
		const n = /** @type {number} */ (stack.pop());
		out.push(n);
		const ch = tree.children[n];
		for (let k = ch.length - 1; k >= 0; k--) stack.push(ch[k]);
	}
	return out;
}

/** `tree.get_terminals()`: leaves (no children) in preorder. The root counts if it has no children. */
export function getTerminals(tree) {
	return findClades(tree).filter((n) => tree.children[n].length === 0);
}

/** `name.strip("'\"")`, the only name normalisation dataset.py applies (lines 233, 309, 617, 621). */
export function stripQuotes(name) {
	return pyStripChars(String(name), '\'"');
}

/**
 * `tree_taxa = [term.name.strip("'\"") for term in tree_obj.get_terminals() if term.name]`,
 * dataset.py:617. Unnamed terminals are dropped; duplicates are kept.
 *
 * @param {PhyloTree} tree
 * @returns {string[]}
 */
export function treeTaxa(tree) {
	const out = [];
	for (const t of getTerminals(tree)) {
		const nm = tree.name[t];
		if (nm) out.push(stripQuotes(nm));
	}
	return out;
}

/**
 * `has_nonzero_branch_lengths`, dataset.py:214-222: at least one non-root branch, and at least half
 * of them present and strictly positive.
 *
 * @param {PhyloTree} tree
 */
export function hasNonzeroBranchLengths(tree) {
	const branches = findClades(tree)
		.filter((n) => n !== tree.root)
		.map((n) => tree.branchLength[n]);
	if (branches.length === 0) return false;
	const pos = branches.filter((b) => b !== null && b > 0.0).length;
	return pos > 0 && pos / branches.length >= 0.5;
}

/**
 * The condition under which dataset.py:601-611 would call HyPhy to estimate branch lengths. The
 * library cannot; the runtime decides whether to supply an estimated tree.
 *
 * @param {PhyloTree} tree
 */
export function needsBranchLengths(tree) {
	return !hasNonzeroBranchLengths(tree);
}

/**
 * `enforce_nonzero_branch_lengths`, dataset.py:289-300. Mutates and returns the tree: every non-root
 * clade with `None` gets `defaultMissing`; every one below `minLen` (zero or negative included) gets
 * `minLen`. The root is untouched.
 *
 * @param {PhyloTree} tree
 * @param {number} [minLen]
 * @param {number} [defaultMissing]
 * @returns {PhyloTree}
 */
export function enforceNonzeroBranchLengths(tree, minLen = 1e-4, defaultMissing = 1e-3) {
	for (const n of findClades(tree)) {
		if (n === tree.root) continue;
		const bl = tree.branchLength[n];
		if (bl === null) tree.branchLength[n] = defaultMissing;
		else if (bl < minLen) tree.branchLength[n] = minLen;
	}
	return tree;
}

/** `re.sub(r'\{[^}]*\}', '', s)` then `re.sub(r'\[[^\]]*\]', '', s)`: HyPhy tags, NEXUS comments. */
function cleanNewick(s) {
	return s.replace(/\{[^}]*\}/g, '').replace(/\[[^\]]*\]/g, '');
}

/** `line.count('(')`. */
function countOpen(s) {
	let n = 0;
	for (let i = 0; i < s.length; i++) if (s[i] === '(') n++;
	return n;
}

const TREE_COMMAND_RE = new RegExp(`tree${PY_WS}+[^=]+=${PY_WS}*(\\([^;]+;)`, 'iu');

/**
 * `extract_tree_from_string_or_file`, dataset.py:161-212, for a string that is the tree text or an
 * alignment with an embedded tree. Returns null where the Python returns None.
 *
 * @param {string} content
 * @returns {PhyloTree|null}
 */
export function extractTree(content) {
	const text = String(content);

	// 1. Explicit NEXUS / HyPhy TREE command.
	const m = TREE_COMMAND_RE.exec(text);
	if (m) {
		try {
			return readNewick(cleanNewick(m[1]));
		} catch {
			// fall through
		}
	}

	// 2. Any line starting with '(' and holding at least two '('.
	for (const line of pySplitLines(text)) {
		let lineClean = pyStrip(line);
		if (lineClean.startsWith('(') && countOpen(lineClean) >= 2) {
			if (!lineClean.endsWith(';')) lineClean += ';';
			try {
				return readNewick(cleanNewick(lineClean));
			} catch {
				// next line
			}
		}
	}

	// 3. The whole content.
	let cleanAll = pyStrip(text);
	if (cleanAll.startsWith('(') && countOpen(cleanAll) >= 2) {
		if (!cleanAll.endsWith(';')) cleanAll += ';';
		try {
			return readNewick(cleanNewick(cleanAll));
		} catch {
			// give up
		}
	}

	return null;
}

/**
 * The three-tier taxon matching of `load_alignment_and_tree`, dataset.py:616-642.
 *
 * Tier 1 keeps tree terminal names (quote-stripped) that are alignment names exactly; tier 2, only
 * if tier 1 is empty, maps quote-stripped ALIGNMENT names back to their originals; tier 3, only if
 * tier 2 is empty, does the same case-insensitively. The result is in tree terminal order and holds
 * ALIGNMENT names (the keys of the sequence map), which is what every later step indexes by. Later
 * duplicates overwrite earlier ones in the tier-2/3 lookups, as a dict comprehension does.
 *
 * @param {string[]} treeTaxaList from `treeTaxa(tree)`
 * @param {Iterable<string>} alignmentNames the sequence map's keys, in order
 * @returns {{taxa: string[], tier: 'exact'|'quote_stripped'|'case_insensitive',
 *   droppedAlignment: number, droppedTree: number}}
 */
export function matchTaxa(treeTaxaList, alignmentNames) {
	const names = Array.from(alignmentNames);
	const nameSet = new Set(names);
	let taxa = treeTaxaList.filter((t) => nameSet.has(t));
	let tier = /** @type {'exact'|'quote_stripped'|'case_insensitive'} */ ('exact');
	if (taxa.length === 0) {
		const clean = new Map();
		for (const k of names) clean.set(stripQuotes(k), k);
		taxa = treeTaxaList.filter((t) => clean.has(t)).map((t) => /** @type {string} */ (clean.get(t)));
		tier = 'quote_stripped';
		if (taxa.length === 0) {
			const lower = new Map();
			for (const k of names) lower.set(stripQuotes(k).toLowerCase(), k);
			taxa = [];
			for (const t of treeTaxaList) {
				const tc = stripQuotes(t).toLowerCase();
				if (lower.has(tc)) taxa.push(/** @type {string} */ (lower.get(tc)));
			}
			tier = 'case_insensitive';
			if (taxa.length === 0) {
				throw new Error(
					`No matching taxa found between tree terminals (${JSON.stringify(treeTaxaList.slice(0, 5))}...) ` +
						`and alignment sequences (${JSON.stringify(names.slice(0, 5))}...).`
				);
			}
		}
	}
	return {
		taxa,
		tier,
		droppedAlignment: names.length - taxa.length,
		droppedTree: treeTaxaList.length - taxa.length
	};
}
