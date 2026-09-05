/**
 * WHY THIS FILE EXISTS
 *
 * Replays the two tree-dependent phenotype fixtures and the local edge cases against
 * src/permulations.js:
 *
 *   fixtures/phenotype/compute_phylogenetic_covariance.json   3 cases (Smc6.nwk, bat_oas1.nwk and
 *       a reversed Smc6 subset), tolerance 1e-9 — the whole M x M matrix, entry for entry.
 *   fixtures/phenotype/generate_permulations.json             3 cases at B = 200. Their `V` output
 *       is a second, independent covariance check; everything else is the STATISTICAL class.
 *   test/data/phenotype/python_quirks.json `find_any`, `covariance`, `permulations`
 *       Bio.Phylo's regex `find_any`, the covariance edge cases (a missing taxon's 1.0 diagonal,
 *       a name that is a regex, an internal node, a zero and a missing branch length, a reordered
 *       subset), and generate_permulations' deterministic invariants on a 6-taxon tree.
 *
 * WHY THE PERMULATION MATRIX IS NOT COMPARED (PLAN.md §5.4, fixtures/README.md). The reference
 * draws from numpy's LEGACY global stream — `np.random.seed(42)` then `np.random.randn`, MT19937 —
 * and this port from Xoshiro256's Box-Muller normal. No seed makes the two agree element by
 * element, so the fixture records SUMMARIES instead and this file compares those:
 *
 *   exact        every binary row sums to K (the foreground count); every continuous row is a
 *                permutation of `original_y`; the shape; `V`.
 *   statistical  per-taxon foreground frequency (binary) and per-taxon mean (continuous). The
 *                REFERENCE side is the noisy one — it is a B = 200 estimate — so the port is run
 *                at B = 5,000 (25x the fixture, ~0.3 s) and each taxon is required to sit within
 *                3 * sqrt(p(1-p)/200) of the fixture's estimate, the fixture's own 3-sigma
 *                binomial band. For the continuous case the same band scaled by the trait's
 *                standard deviation.
 *   structure    the draws themselves: with `returnDraws`, E[Z] is 0 and Cov(Z) is V + 1e-7 I.
 *                At B = 20,000 the port's empirical covariance matches V to within 4 * the
 *                sampling standard error of a covariance entry, sqrt((V_ii V_jj + V_ij^2)/B) —
 *                which is what "the Brownian structure is preserved" actually means and is the
 *                one property a wrong Cholesky or a wrong fill order would break while leaving
 *                every row sum intact.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { computePhylogeneticCovariance, generatePermulations, findAnyByName } from '../src/permulations.js';
import { readNewick, findClades } from '../src/preprocess/tree.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, '..', '..', 'fixtures', 'phenotype');
const covFixture = JSON.parse(readFileSync(join(FIXTURES, 'compute_phylogenetic_covariance.json'), 'utf8'));
const permFixture = JSON.parse(readFileSync(join(FIXTURES, 'generate_permulations.json'), 'utf8'));
const local = JSON.parse(readFileSync(join(HERE, 'data', 'phenotype', 'python_quirks.json'), 'utf8'));

/** max |a - b| over a flat matrix and its nested-list reference. */
function maxAbsDiff(flat, nested, M) {
	let mx = 0;
	for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) mx = Math.max(mx, Math.abs(flat[i * M + j] - nested[i][j]));
	return mx;
}

describe('computePhylogeneticCovariance — fixtures/phenotype/compute_phylogenetic_covariance.json', () => {
	it('replays all three cases', () => {
		expect(covFixture.length).toBe(3);
	});
	for (const c of covFixture) {
		it(`${c.name} matches V to 1e-9`, () => {
			const tree = readNewick(c.inputs.newick);
			const taxa = c.inputs.taxa;
			const V = computePhylogeneticCovariance(tree, taxa);
			expect(V.length).toBe(taxa.length * taxa.length);
			expect(maxAbsDiff(V, c.outputs.V, taxa.length)).toBeLessThanOrEqual(1e-9);
		});
		it(`${c.name} is symmetric off the diagonal`, () => {
			// The reference writes V[i, j] and V[j, i] from one common_ancestor call, so any
			// asymmetry here would be a bug in the MRCA walk, not a fixture disagreement.
			const tree = readNewick(c.inputs.newick);
			const M = c.inputs.taxa.length;
			const V = computePhylogeneticCovariance(tree, c.inputs.taxa);
			for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) expect(V[i * M + j]).toBe(V[j * M + i]);
		});
	}
});

describe('computePhylogeneticCovariance — edge cases (test/data/phenotype/gen.py)', () => {
	for (const c of local.covariance) {
		it(c.name, () => {
			const tree = readNewick(c.newick);
			const V = computePhylogeneticCovariance(tree, c.taxa);
			expect(maxAbsDiff(V, c.V, c.taxa.length)).toBeLessThanOrEqual(1e-12);
		});
	}

	it('a taxon absent from the tree gets 1.0 on the diagonal and 0 off it', () => {
		const c = local.covariance.find((x) => x.name === 'missing_taxon_diagonal_one');
		const M = c.taxa.length;
		const V = computePhylogeneticCovariance(readNewick(c.newick), c.taxa);
		expect(V[(M - 1) * M + (M - 1)]).toBe(1.0);
		for (let j = 0; j < M - 1; j++) expect(V[(M - 1) * M + j]).toBe(0);
	});

	it('a name that is a regex to Biopython reaches a different clade', () => {
		// `find_any(name="a.c")` matches the terminal `aXc`, so the off-diagonal is a real MRCA
		// depth while the diagonal is the dict miss's 1.0 (the dict is keyed by exact name).
		const c = local.covariance.find((x) => x.name === 'regex_name_matches_other');
		const M = c.taxa.length;
		const V = computePhylogeneticCovariance(readNewick(c.newick), c.taxa);
		expect(V[0]).toBe(1.0);
		expect(V[1]).toBeGreaterThan(0);
		expect(V[1]).toBe(c.V[0][1]);
	});
});

describe('findAnyByName — Bio.Phylo treats the name as a regular expression', () => {
	for (const t of local.find_any) {
		it(t.name, () => {
			const tree = readNewick(t.newick);
			for (const hit of t.hits) {
				if (hit.error) {
					expect(() => findAnyByName(tree, hit.query)).toThrow();
					continue;
				}
				const n = findAnyByName(tree, hit.query);
				expect(n < 0 ? null : tree.name[n]).toBe(hit.found);
			}
		});
	}

	it('scans in find_clades() preorder, internal nodes included', () => {
		const tree = readNewick('((a:1,b:2)ab:3,(c:4,d:5)cd:6)root;');
		expect(findClades(tree).map((n) => tree.name[n])).toEqual(['root', 'ab', 'a', 'b', 'cd', 'c', 'd']);
		expect(tree.name[findAnyByName(tree, '.*')]).toBe('root');
	});
});

describe('generatePermulations — fixtures/phenotype/generate_permulations.json', () => {
	it('replays all three cases', () => {
		expect(permFixture.length).toBe(3);
	});

	for (const c of permFixture) {
		const { original_y: y, newick, taxa, n_perm: B, seed } = c.inputs;
		const M = taxa.length;

		it(`${c.name}: V agrees with the fixture to 1e-9`, () => {
			const tree = readNewick(newick);
			const V = computePhylogeneticCovariance(tree, taxa);
			expect(maxAbsDiff(V, c.outputs.V, M)).toBeLessThanOrEqual(1e-9);
		});

		it(`${c.name}: the exact invariants hold at the fixture's B`, () => {
			const tree = readNewick(newick);
			const res = generatePermulations(y, tree, taxa, { nPerm: B, seed });
			expect(res.skipped).toBeNull();
			expect(res.permMatrix.length).toBe(B * M);
			const sortedY = Float64Array.from(y).sort();
			for (let p = 0; p < B; p++) {
				const row = Array.from(res.permMatrix.subarray(p * M, (p + 1) * M));
				if (c.outputs.row_sums) {
					// binary: every row carries exactly K ones
					expect(row.reduce((a, b) => a + b, 0)).toBe(c.outputs.row_sums[p]);
					for (const v of row) expect(v === 0 || v === 1).toBe(true);
				} else {
					// continuous: every row is a permutation of original_y
					const s = row.slice().sort((a, b) => a - b);
					for (let i = 0; i < M; i++) expect(s[i]).toBeCloseTo(sortedY[i], 15);
				}
			}
		});

		it(`${c.name}: the per-taxon summary sits in the fixture's 3-sigma band`, () => {
			const tree = readNewick(newick);
			const BIG = 5000;
			const res = generatePermulations(y, tree, taxa, { nPerm: BIG, seed: 20260905 });
			const mean = new Float64Array(M);
			for (let p = 0; p < BIG; p++) for (let i = 0; i < M; i++) mean[i] += res.permMatrix[p * M + i];
			for (let i = 0; i < M; i++) mean[i] /= BIG;

			if (c.outputs.per_taxon_foreground_frequency) {
				const ref = c.outputs.per_taxon_foreground_frequency;
				for (let i = 0; i < M; i++) {
					const p = Math.max(ref[i], 1 / B);
					const band = 3 * Math.sqrt((p * (1 - p)) / B);
					expect(Math.abs(mean[i] - ref[i])).toBeLessThanOrEqual(band);
				}
			} else {
				const ref = c.outputs.per_taxon_mean;
				// The per-permulation value at a taxon is a draw from the empirical distribution of
				// original_y, so its standard error at B draws is sd(y)/sqrt(B).
				const yy = Float64Array.from(y);
				const m = yy.reduce((a, b) => a + b, 0) / M;
				const sd = Math.sqrt(yy.reduce((a, b) => a + (b - m) * (b - m), 0) / M);
				const band = 3 * (sd / Math.sqrt(B));
				for (let i = 0; i < M; i++) expect(Math.abs(mean[i] - ref[i])).toBeLessThanOrEqual(band);
			}
		});
	}

	it('the draws are mean-zero with covariance V + 1e-7 I', () => {
		const c = permFixture[0];
		const taxa = c.inputs.taxa;
		const M = taxa.length;
		const tree = readNewick(c.inputs.newick);
		const B = 20000;
		const res = generatePermulations(c.inputs.original_y, tree, taxa, { nPerm: B, seed: 12345, returnDraws: true, returnV: true });
		const Z = res.draws; // [M, B]
		expect(Z.length).toBe(M * B);

		const mean = new Float64Array(M);
		for (let i = 0; i < M; i++) {
			let s = 0;
			for (let p = 0; p < B; p++) s += Z[i * B + p];
			mean[i] = s / B;
		}
		for (let i = 0; i < M; i++) {
			const sdEntry = Math.sqrt(res.V[i * M + i] / B);
			expect(Math.abs(mean[i])).toBeLessThanOrEqual(4 * sdEntry);
		}

		for (let i = 0; i < M; i++) {
			for (let j = i; j < M; j++) {
				let s = 0;
				for (let p = 0; p < B; p++) s += Z[i * B + p] * Z[j * B + p];
				const cov = s / B;
				const target = res.V[i * M + j] + (i === j ? 1e-7 : 0);
				const vii = res.V[i * M + i] + 1e-7;
				const vjj = res.V[j * M + j] + 1e-7;
				const se = Math.sqrt((vii * vjj + target * target) / B);
				expect(Math.abs(cov - target)).toBeLessThanOrEqual(4 * se);
			}
		}
	});
});

describe('generatePermulations — deterministic invariants (test/data/phenotype/gen.py)', () => {
	const newick = local.permulations.newick;
	const taxa = local.permulations.taxa;
	const M = taxa.length;
	for (const c of local.permulations.cases) {
		it(c.name, () => {
			const tree = readNewick(newick);
			const res = generatePermulations(c.y, tree, taxa, { nPerm: c.n_perm, seed: c.seed });
			expect(res.isBinary).toBe(c.is_binary);
			expect(res.permMatrix.length).toBe(c.n_perm * M);
			const sums = new Set();
			const sortedY = c.y.slice().sort((a, b) => a - b);
			for (let p = 0; p < c.n_perm; p++) {
				const row = Array.from(res.permMatrix.subarray(p * M, (p + 1) * M));
				sums.add(row.reduce((a, b) => a + b, 0));
				const s = row.slice().sort((a, b) => a - b);
				expect(s).toEqual(sortedY);
			}
			expect([...sums].sort((a, b) => a - b)).toEqual(c.row_sums);
			expect(c.all_rows_same_multiset).toBe(true);
		});
	}

	it('an all-zero y is NOT binary and comes back all zeros', () => {
		// np.unique gives ONE value, so `len(unique) == 2` is false and the rank-matching branch
		// runs on a constant vector. The `[-0:]` slice of phenotype.py:334 is therefore never
		// reached (permulations.js header, QUIRKS).
		const tree = readNewick(newick);
		const res = generatePermulations(new Array(M).fill(0), tree, taxa, { nPerm: 8, seed: 1 });
		expect(res.isBinary).toBe(false);
		expect(res.kForeground).toBe(0);
		expect(Array.from(res.permMatrix).every((v) => v === 0)).toBe(true);
	});
});

describe('generatePermulations — tree-free mode (PLAN.md D22)', () => {
	const taxa = local.permulations.taxa;

	it('returns permMatrix null with a reason when there is no tree', () => {
		for (const tree of [null, undefined]) {
			const res = generatePermulations([1, 1, 0, 0, 0, 0], tree, taxa, { nPerm: 100, seed: 42 });
			expect(res.permMatrix).toBeNull();
			expect(res.draws).toBeNull();
			expect(res.skipped?.reason).toBe('no-tree');
			expect(res.skipped?.detail).toContain('tree_obj is not None');
			// The rest of the record is still filled in, so a caller can report the shape it asked for.
			expect(res.nPerm).toBe(100);
			expect(res.M).toBe(taxa.length);
			expect(res.isBinary).toBe(true);
			expect(res.kForeground).toBe(2);
		}
	});

	it('reports a Cholesky failure rather than throwing', () => {
		// A negative internal branch puts three of V's four leading entries at -1, so V + 1e-7 I
		// has pivot 0 = -0.9999999 and numpy would raise LinAlgError. The reference catches that
		// at phenotype.py:441 and drops the permulations; here it becomes a `skipped` reason.
		const tree = readNewick('((a:0.0,b:0.0)i:-1.0,c:1.0):0;');
		const V = computePhylogeneticCovariance(tree, ['a', 'b', 'c']);
		expect(Array.from(V)).toEqual([-1, -1, 0, -1, -1, 0, 0, 0, 1]);
		const res = generatePermulations([1, 1, 0], tree, ['a', 'b', 'c'], { nPerm: 4, seed: 42 });
		expect(res.permMatrix).toBeNull();
		expect(res.skipped?.reason).toBe('not-positive-definite');
		expect(res.skipped?.detail).toContain('not positive definite');
	});
});
