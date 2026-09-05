/**
 * tn93.test.js — the tree-free TN93 path (PLAN.md D22): src/preprocess/tn93.js and the tree-free
 * branch of src/preprocess/assemble.js.
 *
 * WHY THIS FILE EXISTS
 *
 * PLAN.md §5.3 rule 2: the Python generates the fixtures and the JS replays them. This file replays
 * everything scripts/gen_fixtures.py writes for TN93:
 *
 *   dataset/tn93_distance.json          the pairwise distance one pair at a time, with the 4x4
 *                                       count matrix and the nucleotide frequencies as
 *                                       intermediates, over gaps, N, every IUPAC code, lower case,
 *                                       U, '?', unequal lengths, the degenerate branch, the four
 *                                       match modes, the two cases where Python RAISES, and a real
 *                                       bat_oas1 pair (1e-9)
 *   dataset/tn93_distance_matrix.json   whole matrices for bat_oas1, Smc6 and camelid, the reduced
 *                                       HIV1_RT case, and the three imputation rules (1e-9)
 *   dataset/load_alignment_and_tree_tn93.json   the whole tree-free assembly for bat_oas1 and
 *                                       camelid: taxa, L, tokens, distances, MDS, invariable mask
 *   e2e/*_tn93.json                     what the CLI's own `--use-tn93` runs report about the
 *                                       assembly (taxon and codon counts, per-site invariability);
 *                                       everything downstream of those needs the model and is not
 *                                       replayable here
 *
 * The distances are float64 in Python and float64 here, so the class is 1e-9. MEASURED at the time
 * of writing: max |Δ| = 0 on every case of tn93_distance.json and on every entry of every matrix in
 * tn93_distance_matrix.json — bit-identical, because the tn93 package rounds every distance to six
 * significant digits before returning it. MDS coordinates are compared per column EXACTLY under the
 * canonical sign convention at the 1e-5 class, the same rule fixtures.test.js applies (measured:
 * 3.7e-8 on bat_oas1, 7.8e-8 on camelid, from the float32 eigensolver).
 *
 * WHAT IT DOES NOT DO: no model. The LRTs, attributions and sectors in the e2e fixtures come from
 * the neural graph, which the library never runs; the app's runtime tests own those.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
	tn93Distance,
	tn93Counts,
	tn93NucleotideFrequency,
	tn93CalculateDistance,
	tn93DistanceMatrix,
	tn93SaturatedPairs,
	encodeSequence,
	canResolve,
	ambigFractionTooHigh,
	TN93_MATCH_MODE,
	TN93_SATURATION_SENTINEL,
	TN93_MIN_POSITIVE_DISTANCE,
	TN93_TABLES
} from '../src/preprocess/tn93.js';
import { parseAlignmentSequences } from '../src/preprocess/parse.js';
import { loadAlignmentAndTree } from '../src/preprocess/assemble.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const FIXTURES = join(ROOT, 'fixtures');
const EXAMPLES = join(ROOT, 'examples');
const loadJson = (rel) => JSON.parse(readFileSync(join(FIXTURES, rel), 'utf8'));
const readExample = (name) => readFileSync(join(EXAMPLES, name), 'utf8');
const tol = (c) => (c.tolerance === 'exact' ? 0 : Number(c.tolerance));

/** float32 machine epsilon; the reference's eigendecomposition runs in float32 (LAPACK ssyevd). */
const EPS32 = 2 ** -23;

function maxAbsDiff(a, b) {
	expect(a.length).toBe(b.length);
	let worst = 0;
	for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
	return worst;
}

/**
 * Per-column comparison of MDS coordinates, canonical signs on both sides and no sign-flip
 * allowance, with the float32 noise floor of fixtures.test.js: a component whose magnitude is below
 * sqrt(eps32 * max Gram diagonal) is rounding noise and only has to stay there.
 */
function checkMdsColumns(name, jsFlat, pyRows, n, k, tolerance) {
	let gramMax = 0;
	for (const row of pyRows) gramMax = Math.max(gramMax, row.reduce((s, v) => s + v * v, 0));
	const noiseFloor = Math.sqrt(EPS32 * gramMax);
	for (let col = 0; col < k; col++) {
		let same = 0;
		let pyMag = 0;
		let jsMag = 0;
		for (let i = 0; i < n; i++) {
			same = Math.max(same, Math.abs(jsFlat[i * k + col] - pyRows[i][col]));
			pyMag = Math.max(pyMag, Math.abs(pyRows[i][col]));
			jsMag = Math.max(jsMag, Math.abs(jsFlat[i * k + col]));
		}
		if (pyMag < noiseFloor) {
			expect(jsMag, `${name} column ${col} below the float32 noise floor`).toBeLessThan(noiseFloor);
		} else {
			expect(same, `${name} column ${col} exact (canonical sign)`).toBeLessThanOrEqual(tolerance * Math.max(1, pyMag));
		}
	}
}

/** The fixtures pin token arrays by the sha256 of their row-major values joined with ','. */
const tokenSha = (arr) => createHash('sha256').update(Array.from(arr).join(',')).digest('hex');

describe('dataset/tn93_distance.json', () => {
	const cases = loadJson('dataset/tn93_distance.json');

	it('replays every case', () => {
		expect(cases.length).toBeGreaterThan(20);
	});

	for (const c of cases) {
		it(c.name, () => {
			const { seq1, seq2, match_mode, ignore_gaps } = c.inputs;
			const options = { matchMode: match_mode, ignoreGaps: ignore_gaps === true };
			if (c.outputs.error) {
				// The Python raises and dataset.py does not catch it; so does the port.
				expect(() => tn93Distance(seq1, seq2, options)).toThrow(new RegExp(c.outputs.error));
				return;
			}
			const counts = tn93Counts(seq1, seq2, match_mode, options);
			expect(maxAbsDiff(counts.flat(), c.outputs.counts.flat())).toBeLessThanOrEqual(tol(c));
			const freq = tn93NucleotideFrequency(counts);
			expect(maxAbsDiff(freq, c.outputs.nucleotide_frequency)).toBeLessThanOrEqual(tol(c));
			expect(Math.abs(tn93CalculateDistance(counts, freq) - c.outputs.distance)).toBeLessThanOrEqual(tol(c));
			// The composition dataset.py:544-547 uses must give the same number.
			expect(Math.abs(tn93Distance(seq1, seq2, options) - c.outputs.distance)).toBeLessThanOrEqual(tol(c));
		});
	}

	it('pins the match mode the reference asks for', () => {
		expect(TN93_MATCH_MODE).toBe('resolve');
		const resolveCase = cases.find((c) => c.name === 'identical');
		expect(resolveCase.inputs.match_mode).toBe('resolve');
	});
});

describe('dataset/tn93_distance_matrix.json', () => {
	const cases = loadJson('dataset/tn93_distance_matrix.json');

	for (const c of cases) {
		const reduced = c.name.endsWith('_reduced');
		it(c.name, () => {
			let seqs;
			let taxa;
			if (c.inputs.alignment) {
				seqs = parseAlignmentSequences(readExample(c.inputs.alignment));
				taxa = c.inputs.taxa;
				// The tree-free path takes list(seq_dict.keys()); the fixture records that order.
				expect(taxa).toEqual(Array.from(seqs.keys()));
			} else {
				seqs = new Map(Object.entries(c.inputs.seq_dict));
				taxa = c.inputs.taxa;
			}
			const n = taxa.length;
			const D = tn93DistanceMatrix(seqs, taxa);
			let max = 0;
			for (const v of D) if (v > max) max = v;
			if (reduced) {
				expect(n).toBe(c.outputs.n);
				expect(maxAbsDiff(D.subarray(0, n), c.outputs.first_row)).toBeLessThanOrEqual(tol(c));
				expect(maxAbsDiff(D.subarray((n - 1) * n, n * n), c.outputs.last_row)).toBeLessThanOrEqual(tol(c));
				let minOff = Infinity;
				for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) minOff = Math.min(minOff, D[i * n + j]);
				expect(Math.abs(minOff - c.outputs.min_off_diagonal)).toBeLessThanOrEqual(tol(c));
				const sat = [];
				for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (D[i * n + j] === TN93_SATURATION_SENTINEL) sat.push([i, j]);
				expect(sat).toEqual(c.outputs.saturated_pairs_index);
			} else {
				expect(maxAbsDiff(D, c.outputs.dist_matrix.flat())).toBeLessThanOrEqual(tol(c));
			}
			expect(Math.abs(max - c.outputs.max)).toBeLessThanOrEqual(tol(c));
			if (c.outputs['saturated_pairs_at_1.0'] !== undefined) {
				expect(tn93SaturatedPairs(D, n)).toBe(c.outputs['saturated_pairs_at_1.0']);
			}
			// The matrix is symmetric with a zero diagonal, on every case.
			for (let i = 0; i < n; i++) {
				expect(D[i * n + i]).toBe(0);
				for (let j = 0; j < n; j++) expect(D[i * n + j]).toBe(D[j * n + i]);
			}
		}, 60000);
	}

	it('reproduces the reference imputation rules the synthetic cases pin', () => {
		const byName = Object.fromEntries(cases.map((c) => [c.name, c]));
		// Different strings at distance 0 -> 1e-4.
		expect(byName.imputed_min_positive.outputs.dist_matrix[0][1]).toBe(Math.fround(TN93_MIN_POSITIVE_DISTANCE));
		// Identical strings at distance 0 -> max(1.0, max_d): the reference bug, larger than every
		// real distance in the matrix.
		const bug = byName.identical_strings_get_the_maximum.outputs;
		expect(bug.dist_matrix[0][1]).toBe(TN93_SATURATION_SENTINEL);
		expect(bug.dist_matrix[0][2]).toBeLessThan(bug.dist_matrix[0][1]);
		// All identical -> the matrix never becomes non-zero, so nothing is imputed at all.
		expect(byName.all_identical_stays_zero.outputs.max).toBe(0);
	});
});

describe('dataset/load_alignment_and_tree_tn93.json', () => {
	const cases = loadJson('dataset/load_alignment_and_tree_tn93.json');

	for (const c of cases) {
		it(`${c.name}: the whole tree-free assembly`, () => {
			const r = loadAlignmentAndTree(readExample(c.inputs.alignment), null, {
				useTn93: c.inputs.use_tn93,
				maxSpecies: c.inputs.max_species,
				pruneDuplicates: c.inputs.prune_duplicates
			});
			// Exact: taxa (alignment order), L, N, the invariable mask, the token arrays.
			expect(r.taxa).toEqual(c.outputs.taxa);
			expect(r.L).toBe(c.outputs.L);
			expect(r.N).toBe(c.outputs.N);
			expect(Array.from(r.invariable)).toEqual(c.outputs.is_aa_invariable.map((v) => (v ? 1 : 0)));
			expect(Array.from(r.invariable).reduce((a, b) => a + b, 0)).toBe(c.outputs.n_invariable);
			expect(tokenSha(r.c)).toBe(c.outputs.codon_tokens_sha256);
			expect(tokenSha(r.a)).toBe(c.outputs.aa_tokens_sha256);
			if (c.outputs.codon_tokens) {
				expect(maxAbsDiff(r.c, c.outputs.codon_tokens.flat())).toBe(0);
				expect(maxAbsDiff(r.a, c.outputs.aa_tokens.flat())).toBe(0);
			}
			// Distances: 1e-9 (float32 on both sides).
			expect(maxAbsDiff(r.d, c.outputs.dist_matrix.flat())).toBeLessThanOrEqual(1e-9);
			expect(tn93SaturatedPairs(r.d, r.N)).toBe(c.outputs['saturated_pairs_at_1.0']);
			expect(r.notices.tn93SaturatedPairs).toBe(c.outputs['saturated_pairs_at_1.0']);
			// MDS at the 1e-5 class, per column, canonical signs.
			checkMdsColumns(c.name, r.z, c.outputs.mds_coords, r.N, 4, tol(c));
			if (c.outputs.mds_gram) {
				const gram = new Float64Array(r.N * r.N);
				for (let i = 0; i < r.N; i++)
					for (let j = 0; j < r.N; j++) {
						let sum = 0;
						for (let k = 0; k < 4; k++) sum += r.z[i * 4 + k] * r.z[j * 4 + k];
						gram[i * r.N + j] = sum;
					}
				expect(maxAbsDiff(gram, c.outputs.mds_gram.flat())).toBeLessThanOrEqual(tol(c) * 100);
			}
			// The path is the tree-free one, with no rescale and no taxon matching.
			expect(r.notices.treeFree).toEqual({ reason: 'requested', taxaOrder: 'alignment' });
			expect(r.notices.matchTier).toBeNull();
			expect(r.notices.distanceRescaled).toBe(false);
			expect(r.tree).toBeNull();
		}, 60000);
	}
});

describe('e2e/*_tn93.json: what the CLI reports about the tree-free assembly', () => {
	/** @type {Array<[string, string, {taxa: string, sites: string}]>} */
	const cases = [
		['meme_camelid_tn93.json', 'camelid.fasta', { taxa: 'taxa_count', sites: 'codon_count' }],
		['meme_HIV1_RT_tn93.json', 'HIV1_RT.fasta', { taxa: 'taxa_count', sites: 'codon_count' }],
		['busted_Smc6_tn93.json', 'Smc6.fasta', { taxa: 'taxa', sites: 'sites' }],
		['epistasis_Smc6_tn93.json', 'Smc6.fasta', { taxa: 'taxa_count', sites: 'codon_count' }]
	];

	for (const [file, alignment, keys] of cases) {
		it(`${file}: taxon and codon counts`, () => {
			const fx = loadJson(join('e2e', file))[0];
			expect(fx.inputs.argv).toContain('--use-tn93');
			expect(fx.inputs.argv).toContain('canonical');
			const r = loadAlignmentAndTree(readExample(alignment), null, { useTn93: true });
			expect(r.N).toBe(fx.outputs[keys.taxa]);
			expect(r.L).toBe(fx.outputs[keys.sites]);
			expect(r.notices.treeFree.reason).toBe('requested');
			if (fx.outputs.evaluated_taxa !== undefined) expect(fx.outputs.evaluated_taxa).toBe(r.N);
			// The MEME writers carry per-site invariability, which is assembly output, not model
			// output: it must match site for site.
			if (Array.isArray(fx.outputs.sites)) {
				expect(fx.outputs.sites.map((s) => (s.is_invariable ? 1 : 0))).toEqual(Array.from(r.invariable));
				expect(fx.outputs.sites.map((s) => s.site)).toEqual(Array.from({ length: r.L }, (_, i) => i + 1));
			}
		}, 120000);
	}
});

describe('the tn93 package tables and its two raising paths', () => {
	it('maps characters exactly as tn93.py:524-557 does', () => {
		const { mapCharacter, resolutions, resolutionCounts, gap } = TN93_TABLES;
		expect(mapCharacter.length).toBe(256);
		expect(gap).toBe(17);
		expect(mapCharacter['-'.charCodeAt(0)]).toBe(17);
		expect([...'ACGTU'].map((ch) => mapCharacter[ch.charCodeAt(0)])).toEqual([0, 1, 2, 3, 4]);
		expect([...'acgtu'].map((ch) => mapCharacter[ch.charCodeAt(0)])).toEqual([0, 1, 2, 3, 4]);
		expect([...'RYSWKMBDHVN'].map((ch) => mapCharacter[ch.charCodeAt(0)])).toEqual([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
		// Anything unmapped, '?' included, is 16 and resolves to any base.
		expect(mapCharacter['?'.charCodeAt(0)]).toBe(16);
		expect(mapCharacter['X'.charCodeAt(0)]).toBe(16);
		expect(resolutions[16]).toEqual([1, 1, 1, 1]);
		expect(resolutions[17]).toEqual([0, 0, 0, 0]);
		expect(resolutionCounts[17]).toBe(0);
		expect(resolutionCounts[15]).toBe(0.25);
		expect(encodeSequence('ACGT-N?')).toEqual(new Uint8Array([0, 1, 2, 3, 17, 15, 16]));
	});

	it('raises Python IndexError on a character above U+00FF', () => {
		expect(() => encodeSequence('ATGΩ')).toThrow(/IndexError/);
	});

	it('can_resolve is false for two plain bases and for anything facing a gap', () => {
		expect(canResolve(0, 1)).toBe(false); // A, C
		expect(canResolve(0, 17)).toBe(false); // A, gap
		expect(canResolve(17, 17)).toBe(false);
		expect(canResolve(0, 5)).toBe(true); // A, R = A|G
		expect(canResolve(0, 6)).toBe(false); // A, Y = C|T
		expect(canResolve(5, 6)).toBe(true); // R, Y: no shared base, but both resolve
	});

	it('ambig_fraction_too_high sends an all-resolvable pair to the average branch, ties included', () => {
		expect(ambigFractionTooHigh('RRRR', 'AAAA')).toBe(true);
		expect(ambigFractionTooHigh('ACGT', 'ACGT')).toBe(false);
		// The degenerate 0 <= 0: no overlapping non-gap position at all.
		expect(ambigFractionTooHigh('AAAA----', '----AAAA')).toBe(true);
	});

	it('throws where Python raises, and only there', () => {
		expect(() => tn93Distance('----', '----')).toThrow(/ZeroDivisionError/);
		expect(() => tn93Distance('ATGCATGCATGCATGC', 'GCTAGCTAGCTAGCTA')).toThrow(/expected a positive input/);
		expect(tn93Distance('ATGAAACCCGGGTTT', 'ATGAAACCCGGGTTT')).toBe(0);
	});

	it('counts only pairs sitting exactly at the sentinel', () => {
		const d = Float32Array.from([0, 1, 0.5, 1, 0, 1.5, 0.5, 1.5, 0]);
		expect(tn93SaturatedPairs(d, 3)).toBe(1);
		expect(tn93SaturatedPairs(d, 3, 1.5)).toBe(1);
	});

	it('returns the zero matrix for a single taxon, as dataset.py:502-503 does', () => {
		expect(Array.from(tn93DistanceMatrix(new Map([['a', 'ATGC']]), ['a']))).toEqual([0]);
		expect(Array.from(tn93DistanceMatrix({ a: 'ATGC' }, ['a']))).toEqual([0]);
	});
});

describe('the tree-free decision in loadAlignmentAndTree (D22)', () => {
	const fasta = readExample('bat_oas1.fasta');
	const tree = readExample('bat_oas1.nwk');

	it("reports 'requested' for useTn93 and for the tn93/none/skip tree modes", () => {
		for (const r of [
			loadAlignmentAndTree(fasta, tree, { useTn93: true }),
			loadAlignmentAndTree(fasta, 'tn93'),
			loadAlignmentAndTree(fasta, ' NONE '),
			loadAlignmentAndTree(fasta, 'skip')
		]) {
			expect(r.notices.treeFree.reason).toBe('requested');
			expect(r.notices.distanceRescaled).toBe(false);
		}
		// bat_oas1's tree is in Mya and would be rescaled; the TN93 matrix is not.
		expect(loadAlignmentAndTree(fasta, tree).notices.distanceRescaled).toBe(true);
	}, 30000);

	it("reports 'no_tree' when neither a tree text nor an embedded tree is there", () => {
		const noTree = loadAlignmentAndTree(fasta, null);
		expect(noTree.notices.treeFree).toEqual({ reason: 'no_tree', taxaOrder: 'alignment' });
		expect(noTree.tree).toBeNull();
		expect(noTree.notices.branchLengthsMissing).toBe(false);
		expect(noTree.taxa).toEqual(Array.from(parseAlignmentSequences(fasta).keys()));
	}, 30000);

	it('keeps a topology-only tree for display but never lets it reach the model', () => {
		const names = Array.from(parseAlignmentSequences(fasta).keys());
		// Nested: extract_tree_from_string_or_file rejects a newick with a single '(' (dataset.py quirk).
		const topology = `((${names.slice(0, 2).join(',')}),${names.slice(2).join(',')});`;
		const r = loadAlignmentAndTree(fasta, topology);
		expect(r.notices.treeFree).toEqual({ reason: 'no_branch_lengths', taxaOrder: 'alignment' });
		expect(r.notices.branchLengthsMissing).toBe(true);
		expect(r.tree).not.toBeNull();
		// Untouched by enforceNonzeroBranchLengths: the reference's 1e-3 / 1e-4 defaults never ran.
		expect(r.tree.branchLength.every((b) => b === null)).toBe(true);
		expect(r.taxa).toEqual(names);
	}, 30000);

	it('still raises on tree text that will not parse', () => {
		expect(() => loadAlignmentAndTree(fasta, 'not a tree at all')).toThrow(/Could not parse phylogenetic tree/);
	});
});
