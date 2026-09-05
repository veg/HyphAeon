/**
 * WHY THIS FILE EXISTS
 *
 * Replays the two epistasis.py sector fixtures (scripts/gen_fixtures.py, fixtures/README.md) and
 * the hyphaeon-generated variations in test/data/sectors/graph.json against src/sectors.js:
 *
 *   fixtures/epistasis/extract_epistatic_sectors_tse.json     8 cases. n_permutations=0 cases are
 *       deterministic: sector ids, order, membership, shared taxa, signatures and focal fields
 *       exact, coherence and mean_lrt within the fixture's 1e-6. The B=2000 cases keep membership
 *       and order exact and compare p_perm and the null moments statistically (below).
 *   fixtures/epistasis/compute_sector_permutation_test.json   5 cases: two exact degenerate paths
 *       (B=0, pool == K) and three Monte Carlo nulls compared statistically.
 *   graph.json `sectors`   the components path with two components, a min_coherence that drops a
 *       sector, and the 1,097-node graph whose two identical triangles get their sector ids from
 *       the CPython set order (33..35 before 5..7) — exact.
 *   graph.json `percentile` np.percentile / mean / std of float32 arrays, the null reductions.
 *
 * THE STATISTICAL CLASS (PLAN.md §5.4, fixtures/README.md): the reference draws from numpy's PCG64
 * and the port from Xoshiro256, so two independent estimates of the same null are compared.
 * p_perm: |p_js - p_py| <= 3 * sqrt(p* (1 - p*) / B) with p* = max(p_py, p_js, 1/B) — the floor
 * keeps the bound meaningful when the reference saw zero exceedances. Null mean, std and 95th
 * percentile: within 2 % relative (absolute 1e-6 when the reference value is 0). At B = 200,000 the
 * two generators agree to 4 decimals on mean, std and p95 (src/sectors.js header); the fixtures'
 * B = 5,000 references sit up to 1.9 % from that limit themselves, which is why the class is 2 %.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
	extractEpistaticSectorsTse,
	computeSectorPermutationTest,
	float32Percentile,
	REV_AA_MAP
} from '../src/sectors.js';
import { numpyMeanFloat32 } from '../src/numeric/reduce.js';
import { adjacencyFromGraph } from '../src/numeric/graph.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, '..', '..', 'fixtures', 'epistasis');
const sectorsFixture = JSON.parse(readFileSync(join(FIXTURES, 'extract_epistatic_sectors_tse.json'), 'utf8'));
const permFixture = JSON.parse(readFileSync(join(FIXTURES, 'compute_sector_permutation_test.json'), 'utf8'));
const local = JSON.parse(readFileSync(join(HERE, 'data', 'sectors', 'graph.json'), 'utf8'));

const NUMERIC_FIELDS = ['spectral_coherence', 'mean_lrt', 'isotropic_baseline'];
const NULL_FIELDS = ['null_coherence_mean', 'null_coherence_std', 'null_coherence_95'];
const EXACT_FIELDS = ['sector_id', 'size', 'sites', 'shared_taxa', 'shared_branches', 'pars_signature', 'consensus_signature',
	'focal_taxon', 'focal_signature', 'focal_mutations', 'focal_mutations_count'];

/** Options object from a fixture's Python keyword inputs. */
function optionsFrom(inp) {
	return {
		minCliqueSize: inp.min_clique_size,
		maxOverlap: inp.max_overlap,
		minCoherence: inp.min_coherence,
		focalTaxon: inp.focal_taxon ?? null,
		aNp: inp.a_np ?? null,
		taxa: inp.taxa ?? null,
		nPermutations: inp.n_permutations,
		maxPermP: inp.max_perm_p ?? null,
		seed: inp.rng_seed
	};
}

function expectStatisticalP(pJs, pPy, B, label) {
	const pStar = Math.max(pPy, pJs, 1 / B);
	const bound = 3 * Math.sqrt((pStar * (1 - pStar)) / B);
	expect(Math.abs(pJs - pPy), `${label}: p_perm ${pJs} vs ${pPy}, bound ${bound}`).toBeLessThanOrEqual(bound);
}

function expectWithin2pct(got, want, label) {
	const tol = want === 0 ? 1e-6 : 0.02 * Math.abs(want);
	expect(Math.abs(got - want), `${label}: ${got} vs ${want}`).toBeLessThanOrEqual(tol);
}

/** Every sector field of `got` against `want` at the fixture's class. */
function expectSector(got, want, tolerance, B, label) {
	for (const f of EXACT_FIELDS) {
		if (f in want) expect(got[f], `${label}.${f}`).toEqual(want[f]);
		else expect(f in got, `${label}.${f} should be absent`).toBe(false);
	}
	expect(Object.keys(got).sort()).toEqual(Object.keys(want).sort());
	const numTol = tolerance === 'exact' ? 0 : Number(tolerance === 'statistical' ? '1e-6' : tolerance);
	for (const f of NUMERIC_FIELDS) expect(Math.abs(got[f] - want[f]), `${label}.${f}`).toBeLessThanOrEqual(numTol);
	if (tolerance === 'statistical') {
		expectStatisticalP(got.p_perm, want.p_perm, B, `${label}.p_perm`);
		for (const f of NULL_FIELDS) expectWithin2pct(got[f], want[f], `${label}.${f}`);
	} else {
		expect(got.p_perm).toBe(want.p_perm);
		for (const f of NULL_FIELDS) expect(Math.abs(got[f] - want[f]), `${label}.${f}`).toBeLessThanOrEqual(numTol);
	}
}

describe('extract_epistatic_sectors_tse fixture replay', () => {
	let worstCoherence = 0;
	for (const c of sectorsFixture) {
		it(`${c.name} (${c.tolerance})`, () => {
			const inp = c.inputs;
			const got = extractEpistaticSectorsTse(inp.graph, inp.attributions, inp.lrts, inp.consensus_aas, optionsFrom(inp));
			const want = c.outputs.sectors;
			expect(got.map((s) => s.sites), 'membership and order').toEqual(want.map((s) => s.sites));
			got.forEach((s, i) => {
				expectSector(s, want[i], c.tolerance, inp.n_permutations, `${c.name}[${i}]`);
				worstCoherence = Math.max(worstCoherence, Math.abs(s.spectral_coherence - want[i].spectral_coherence));
			});
		});
	}
	it('reports the worst coherence deviation (float32 BLAS + ssyevd vs float64 tred2/tql2 rounded)', () => {
		console.log(`[sectors] extract_epistatic_sectors_tse: max |dC| ${worstCoherence.toExponential(2)} over all fixture sectors`);
		expect(worstCoherence).toBeLessThanOrEqual(1e-6);
	});
});

describe('compute_sector_permutation_test fixture replay', () => {
	for (const c of permFixture) {
		it(`${c.name} (${c.tolerance})`, () => {
			const inp = c.inputs;
			const got = computeSectorPermutationTest(inp.attributions, inp.site_indices, inp.observed_coherence, {
				nPermutations: inp.n_permutations,
				activeOnly: inp.active_only,
				seed: inp.rng_seed
			});
			const want = c.outputs;
			expect(Object.keys(got).sort()).toEqual(Object.keys(want).sort());
			expect(got.isotropic_baseline).toBe(want.isotropic_baseline);
			if (c.tolerance === 'exact') {
				expect(got).toEqual(want);
				return;
			}
			expectStatisticalP(got.p_perm, want.p_perm, inp.n_permutations, c.name);
			for (const f of NULL_FIELDS) expectWithin2pct(got[f], want[f], `${c.name}.${f}`);
			console.log(
				`[sectors] ${c.name}: p_perm ${got.p_perm} (ref ${want.p_perm}); null mean ${got.null_coherence_mean.toFixed(5)} (ref ${want.null_coherence_mean.toFixed(5)}), std ${got.null_coherence_std.toFixed(5)} (ref ${want.null_coherence_std.toFixed(5)}), p95 ${got.null_coherence_95.toFixed(5)} (ref ${want.null_coherence_95.toFixed(5)})`
			);
		});
	}

	it('is reproducible from the seed and changes with it', () => {
		const inp = permFixture[0].inputs;
		const a = computeSectorPermutationTest(inp.attributions, inp.site_indices, inp.observed_coherence, { nPermutations: 500, seed: 7 });
		const b = computeSectorPermutationTest(inp.attributions, inp.site_indices, inp.observed_coherence, { nPermutations: 500, seed: 7 });
		const d = computeSectorPermutationTest(inp.attributions, inp.site_indices, inp.observed_coherence, { nPermutations: 500, seed: 8 });
		expect(a).toEqual(b);
		expect(a.null_coherence_mean).not.toBe(d.null_coherence_mean);
	});

	it('refuses an unseeded run (numpy rng_seed=None has no counterpart)', () => {
		const inp = permFixture[0].inputs;
		expect(() => computeSectorPermutationTest(inp.attributions, inp.site_indices, 0.9, { seed: null })).toThrow(TypeError);
		expect(() => computeSectorPermutationTest(inp.attributions, inp.site_indices, 0.9, { seed: 1.5 })).toThrow(TypeError);
	});

	it('accepts the flat {data, rows, cols} matrix shape', () => {
		const inp = permFixture[0].inputs;
		const rows = inp.attributions.length;
		const cols = inp.attributions[0].length;
		const flat = { data: Float32Array.from(inp.attributions.flat()), rows, cols };
		const a = computeSectorPermutationTest(inp.attributions, inp.site_indices, inp.observed_coherence, { nPermutations: 300 });
		const b = computeSectorPermutationTest(flat, inp.site_indices, inp.observed_coherence, { nPermutations: 300 });
		expect(a).toEqual(b);
	});
});

describe('hyphaeon-generated variations (graph.json sectors)', () => {
	for (const c of local.sectors) {
		it(`${c.name}`, () => {
			const kw = c.kwargs;
			const got = extractEpistaticSectorsTse(c.graph, c.attributions, c.lrts, c.consensus_aas, {
				minCliqueSize: kw.min_clique_size,
				minCoherence: kw.min_coherence,
				nPermutations: 0
			});
			expect(got.map((s) => [s.sector_id, s.sites])).toEqual(c.sectors.map((s) => [s.sector_id, s.sites]));
			got.forEach((s, i) => expectSector(s, c.sectors[i], '1e-6', 0, `${c.name}[${i}]`));
		});
	}

	it('the set-order case would come out the other way under ascending node order', () => {
		const c = local.sectors.find((x) => x.name === 'two_identical_triangles_1097_set_order');
		expect(c.sectors.map((s) => s.sites)).toEqual([[33, 34, 35], [5, 6, 7]]);
		expect(c.sectors[0].spectral_coherence).toBe(c.sectors[1].spectral_coherence);
	});
});

describe('quirks replicated', () => {
	const base = sectorsFixture.find((c) => c.name === 'planted_no_permutations').inputs;

	it('max_overlap is accepted and never read', () => {
		const a = extractEpistaticSectorsTse(base.graph, base.attributions, base.lrts, base.consensus_aas, { nPermutations: 0, maxOverlap: 0 });
		const b = extractEpistaticSectorsTse(base.graph, base.attributions, base.lrts, base.consensus_aas, { nPermutations: 0, maxOverlap: 1 });
		expect(a).toEqual(b);
		expect(a.length).toBe(2);
	});

	it('shared_taxa counts over the unpruned community rows; pruning changes sites and mean_lrt only', () => {
		// A 4-site community where the 4th row is weakly loaded on the top eigenvector: it is pruned
		// from `sites` but still constrains shared_taxa (np.all over sub_A, the unpruned rows).
		const A = [
			[0.9, 0.8, 0.7, 0.0, 0.0],
			[0.9, 0.8, 0.7, 0.0, 0.0],
			[0.9, 0.8, 0.7, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.05, 0.0]
		];
		const graph = { nodes: [1, 2, 3, 4], edges: [[1, 2, { weight: 1 }], [1, 3, { weight: 1 }], [2, 3, { weight: 1 }], [3, 4, { weight: 1 }]] };
		const got = extractEpistaticSectorsTse(graph, A, [1, 2, 3, 100], ['A', 'C', 'D', 'E'], { nPermutations: 0, minCliqueSize: 100 });
		expect(got).toHaveLength(1);
		expect(got[0].sites).toEqual([1, 2, 3]);
		expect(got[0].shared_taxa).toBe(0); // row 4 has zeros in the first three columns
		expect(got[0].mean_lrt).toBe(2); // over the pruned sites only
		expect(got[0].pars_signature).toBe('[ A1 - C2 - D3 ]');
		expect(got[0].spectral_coherence).toBe(1);
	});

	it('accepts an adjacency Map or an object carrying one as `adj` (CoselectionGraph shape)', () => {
		const want = extractEpistaticSectorsTse(base.graph, base.attributions, base.lrts, base.consensus_aas, { nPermutations: 0 });
		const adj = adjacencyFromGraph(base.graph);
		expect(extractEpistaticSectorsTse(adj, base.attributions, base.lrts, base.consensus_aas, { nPermutations: 0 })).toEqual(want);
		expect(extractEpistaticSectorsTse({ adj, nodes: new Map() }, base.attributions, base.lrts, base.consensus_aas, { nPermutations: 0 })).toEqual(want);
	});

	it('returns [] for a graph without edges and skips communities below 2 sites', () => {
		expect(extractEpistaticSectorsTse({ nodes: [1, 2], edges: [] }, [[1], [1]], [1, 1], ['A', 'A'])).toEqual([]);
	});

	it('REV_AA_MAP inverts AA_MAP', () => {
		expect(REV_AA_MAP.get(0)).toBe('A');
		expect(REV_AA_MAP.get(19)).toBe('Y');
		expect(REV_AA_MAP.size).toBe(20);
	});
});

describe('float32 reductions vs numpy (graph.json percentile)', () => {
	it('np.percentile(x, 95), np.mean and np.std on float32 arrays', () => {
		for (const c of local.percentile) {
			const x = Float32Array.from(c.values);
			const p95 = float32Percentile(x, 95);
			expect(Math.abs(p95 - c.p95), `p95 n=${x.length}`).toBeLessThanOrEqual(2e-7);
			expect(Math.abs(numpyMeanFloat32(x) - c.mean), `mean n=${x.length}`).toBeLessThanOrEqual(2e-7);
		}
	});
});

/**
 * The seam between the two halves of epistasis.py's port, added by the Phase 2a integration pass.
 * `computeTransformerAttributions` returns `leafAttributions` as a flat Float32Array [L*N] and
 * `computeBranchCoselectionNetwork` reads that shape directly; before this, the sector miner took
 * only rows or `{data, rows, cols}`, so chaining the three functions the way `run_epistatic_analysis`
 * does threw `attributions: data has 0 entries, expected NaN*undefined` — a flat buffer fell into
 * the rows branch and `rows[0].length` was undefined. A flat buffer plus `options.N` is now the
 * third accepted shape, and an unaccompanied one is refused by name rather than misread.
 */
describe('a flat [L*N] buffer with options.N (the shape epistasis.js hands over)', () => {
	const rows = [
		[1, 0, 2, 0],
		[0.5, 0, 1, 0],
		[0, 3, 0, 1],
		[0, 1.5, 0, 0.5],
		[2, 2, 2, 2]
	];
	const flat = Float32Array.from(rows.flat());
	const sites = [0, 1, 4];

	it('gives exactly what the same matrix as rows gives', () => {
		const asRows = computeSectorPermutationTest(rows, sites, 0.9, { nPermutations: 200, seed: 42 });
		const asFlat = computeSectorPermutationTest(flat, sites, 0.9, { nPermutations: 200, seed: 42, N: 4 });
		const asShaped = computeSectorPermutationTest({ data: flat, rows: 5, cols: 4 }, sites, 0.9, { nPermutations: 200, seed: 42 });
		expect(asFlat).toEqual(asRows);
		expect(asShaped).toEqual(asRows);
	});

	it('carries through extractEpistaticSectorsTse, aNp included', () => {
		const graph = { nodes: [[1, {}], [2, {}], [5, {}]], edges: [[1, 2, { weight: 0.9 }], [2, 5, { weight: 0.8 }]] };
		const lrts = Float32Array.from([4, 3, 2, 1, 5]);
		const aas = ['A', 'C', 'D', 'E', 'F'];
		const tokens = [
			[0, 1, 0, 1],
			[0, 1, 1, 1],
			[2, 2, 2, 2],
			[3, 3, 3, 3],
			[4, 4, 20, 4]
		];
		const opts = { nPermutations: 100, seed: 42, taxa: ['t0', 't1', 't2', 't3'], focalTaxon: 't1' };
		const viaRows = extractEpistaticSectorsTse(graph, rows, lrts, aas, { ...opts, aNp: tokens });
		const viaFlat = extractEpistaticSectorsTse(graph, flat, lrts, aas, { ...opts, aNp: Int32Array.from(tokens.flat()), N: 4 });
		expect(viaFlat).toEqual(viaRows);
		expect(viaRows.length).toBeGreaterThan(0);
	});

	it('refuses a flat buffer whose shape nobody supplied, by name', () => {
		expect(() => computeSectorPermutationTest(flat, sites, 0.9, { nPermutations: 10 })).toThrow(/attributions: a flat buffer of 20 values needs its shape/);
		expect(() => computeSectorPermutationTest(flat, sites, 0.9, { nPermutations: 10, N: 3 })).toThrow(/not a whole number of rows of 3/);
	});
});
