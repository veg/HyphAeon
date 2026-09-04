/**
 * preprocess-parse.test.js — parse.js and tree.js against Python-computed edge cases.
 *
 * WHY THIS FILE EXISTS
 *
 * fixtures/dataset covers the bundled examples and one synthetic input per format. The code paths
 * that decide FORMAT DETECTION (a byte-order mark, a PHYLIP header that does not deliver enough
 * records, a bare '>' line) and the Biopython parser's less obvious rules (a root-level comma, a
 * numeric internal label read as confidence, an escaped quote, text after the semicolon, a root
 * branch length) are not in the fixtures, so js/test/data/preprocess/edge_cases.json holds the
 * reference's answers for them, computed with hyphaeon.dataset and Bio.Phylo 1.85 by the script
 * recorded in the file's provenance comment (scratchpad script, values only). Nothing here is
 * hand-derived; every expectation is the Python's.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { parseAlignmentSequences, pyStrip, pySplit1, pySplit, pySplitLines, pyIsAlnum } from '../src/preprocess/parse.js';
import {
	extractTree,
	readNewick,
	findClades,
	getTerminals,
	hasNonzeroBranchLengths,
	needsBranchLengths,
	enforceNonzeroBranchLengths,
	treeTaxa,
	matchTaxa,
	stripQuotes,
	NewickError
} from '../src/preprocess/tree.js';

const DATA = JSON.parse(
	readFileSync(join(dirname(fileURLToPath(import.meta.url)), 'data', 'preprocess', 'edge_cases.json'), 'utf8')
);

describe('Python string semantics', () => {
	it('strip() does not remove a byte-order mark, and does remove NBSP and U+001C', () => {
		expect(pyStrip('﻿x')).toBe('﻿x');
		expect(pyStrip(' x')).toBe('x');
	});

	it('split(None, 1) keeps the remainder"s trailing whitespace', () => {
		expect(pySplit1('  a   b c  ')).toEqual(['a', 'b c  ']);
		expect(pySplit1('a')).toEqual(['a']);
		expect(pySplit1('   ')).toEqual([]);
		expect(pySplit(' a\tb ')).toEqual(['a', 'b']);
	});

	it('splitlines() drops the empty tail and splits on \\r\\n, \\v, \\f', () => {
		expect(pySplitLines('a\r\nb\vc\fd\n')).toEqual(['a', 'b', 'c', 'd']);
	});

	it('isalnum() is Unicode-aware and false for the empty string', () => {
		expect(pyIsAlnum('ab1')).toBe(true);
		expect(pyIsAlnum('a_b')).toBe(false);
		expect(pyIsAlnum('')).toBe(false);
		expect(pyIsAlnum('é1')).toBe(true);
	});
});

describe('parseAlignmentSequences edge cases (Python-computed)', () => {
	for (const c of DATA.parse) {
		it(c.name, () => {
			if (c.error) {
				expect(() => parseAlignmentSequences(c.text)).toThrow();
				return;
			}
			const seqs = parseAlignmentSequences(c.text);
			expect(Object.fromEntries(seqs)).toEqual(c.sequences);
			expect([...seqs.keys()]).toEqual(Object.keys(c.sequences));
		});
	}
});

describe('extractTree edge cases (Python-computed, Bio.Phylo 1.85)', () => {
	for (const c of DATA.tree) {
		it(c.name, () => {
			const tree = extractTree(c.source);
			if (c.raw === null) {
				expect(tree).toBeNull();
				return;
			}
			expect(tree).not.toBeNull();
			const check = (info) => {
				const clades = findClades(tree);
				const terms = getTerminals(tree);
				expect(terms.map((t) => tree.name[t])).toEqual(info.terminals);
				expect(terms.map((t) => tree.branchLength[t])).toEqual(info.terminal_branch_lengths);
				expect(clades.length).toBe(info.n_clades);
				expect(clades.map((n) => tree.name[n])).toEqual(info.clade_names_preorder);
				expect(clades.map((n) => tree.branchLength[n])).toEqual(info.clade_branch_lengths_preorder);
				expect(clades.map((n) => tree.confidence[n])).toEqual(info.clade_confidences_preorder);
				expect(hasNonzeroBranchLengths(tree)).toBe(info.has_nonzero_branch_lengths);
			};
			check(c.raw);
			expect(needsBranchLengths(tree)).toBe(c.needs_branch_lengths);
			enforceNonzeroBranchLengths(tree, 1e-4);
			check(c.after_enforce);
		});
	}

	it('readNewick raises where Phylo.read does', () => {
		expect(() => readNewick('')).toThrow(/no trees/);
		expect(() => readNewick('(A,B);\n(C,D);')).toThrow(/multiple trees/);
		expect(() => readNewick('((A,B);')).toThrow(NewickError);
		expect(() => readNewick('(A,B)); x')).toThrow(NewickError);
		expect(() => readNewick('(A,B); x')).toThrow(/Text after semicolon/);
	});

	it('stripQuotes strips the ends only, as str.strip("\'\\"") does', () => {
		expect(stripQuotes("'Homo sapiens'")).toBe('Homo sapiens');
		expect(stripQuotes('"x"')).toBe('x');
		expect(stripQuotes("Homo_'sapiens'")).toBe("Homo_'sapiens");
		expect(stripQuotes('\'"a"\'')).toBe('a');
	});

	it('treeTaxa drops unnamed terminals and keeps duplicates', () => {
		expect(treeTaxa(readNewick("((A:1,'B x':1):1,(,A:1):1);"))).toEqual(['A', 'B x', 'A']);
	});
});

describe('matchTaxa, the three tiers of dataset.py:616-642 (Python-computed)', () => {
	for (const c of DATA.match) {
		it(c.name, () => {
			if (c.error) {
				expect(() => matchTaxa(c.tree_taxa, c.alignment_names)).toThrow(/No matching taxa/);
				return;
			}
			const r = matchTaxa(c.tree_taxa, c.alignment_names);
			expect(r.taxa).toEqual(c.taxa);
			expect(r.tier).toBe(c.tier);
			expect(r.droppedAlignment).toBe(c.dropped_aln);
			expect(r.droppedTree).toBe(c.dropped_tree);
		});
	}
});
