/**
 * epistasis.test.js — src/epistasis.js against hyphaeon/epistasis.py:51-222 (cf838ab).
 *
 * WHY THIS FILE EXISTS
 *
 *   1. fixtures/epistasis/compute_branch_coselection_network.json replayed case by case: the five
 *      cases (function defaults, CLI min_sim 0.30, loose filters with 453 edges, strict CESI, and
 *      the V < 2 early return). Records are compared key by key — integers, strings, edge ORDER
 *      and the sorted-by-CESI order exactly; floats at the case's class (1e-6) and, as a
 *      measurement, at the tighter bound actually observed — and the graph is compared in
 *      networkx's node and edge iteration order against `graph_to_json`.
 *   2. test/data/epistasis/coselection_network_cases.json (gen.py, the reference Python on
 *      seeded inputs): the negative-t branch, the collinear 1e-9 guard, df = 1, N = 300 (the
 *      float32 norm order), and the float32 promotion of the thresholds (an edge whose cosine is
 *      exactly float32(0.35) passes `min_sim=0.35` in the reference).
 *   3. test/data/epistasis/transformer_attributions_fake_model.json: compute_transformer_attributions
 *      driven in Python by a fake torch model that plays back recorded y_soft / attention rows,
 *      so the consensus, delta, clamp, Self & Liang cast and the product are the reference's own.
 *      Replayed through the pure function and through the async batching loop.
 *   4. CoselectionGraph against networkx 3.6.1 semantics measured directly (edge iteration
 *      order, add_edge appending nodes, in-place datadict update, self-loop degree).
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
	consensusDelta,
	computeTransformerAttributions,
	runTransformerAttributions,
	computeBranchCoselectionNetwork,
	CoselectionGraph
} from '../src/epistasis.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..', '..');
const readJson = (/** @type {string} */ p) => JSON.parse(readFileSync(p, 'utf8'));
const fixture = (/** @type {string} */ rel) => readJson(join(ROOT, 'fixtures', rel));
const local = (/** @type {string} */ name) => readJson(join(HERE, 'data', 'epistasis', name));

/** fixtures/README.md: NaN and infinities are the strings "NaN", "Infinity", "-Infinity". */
const num = (/** @type {any} */ v) => (v === 'Infinity' ? Infinity : v === '-Infinity' ? -Infinity : v === 'NaN' ? NaN : v);

const EDGE_KEYS = [
	'site_u', 'site_v', 'ref_u', 'ref_v', 'lrt_u', 'lrt_v', 'similarity', 'shared_taxa', 'shared_branches',
	'p_val', 'hyper_p', 'fdr_q', 'cesi'
];
const EXACT_KEYS = new Set(['site_u', 'site_v', 'ref_u', 'ref_v', 'shared_taxa', 'shared_branches', 'lrt_u', 'lrt_v']);
const GRAPH_EXACT = new Set(['shared']);

/**
 * Compare a replay against the reference `{sig_pairs, graph}` at tolerance `tol` (0 = exact).
 * Returns the worst absolute float difference seen, per key, so a test can pin what it measured.
 */
function compareNetwork(got, want, tol, label) {
	/** @type {Record<string, number>} */
	const worst = {};
	const bump = (k, d) => {
		worst[k] = Math.max(worst[k] ?? 0, d);
	};
	expect(got.sigPairs.length, `${label}: edge count`).toBe(want.sig_pairs.length);
	expect(
		got.sigPairs.map((e) => [e.site_u, e.site_v]),
		`${label}: edge order (sorted by cesi desc)`
	).toEqual(want.sig_pairs.map((e) => [e.site_u, e.site_v]));
	got.sigPairs.forEach((e, i) => {
		const w = want.sig_pairs[i];
		expect(Object.keys(e), `${label}: keys of edge ${i}`).toEqual(EDGE_KEYS);
		for (const k of EDGE_KEYS) {
			const wv = num(w[k]);
			if (EXACT_KEYS.has(k) || tol === 0) {
				expect(e[k], `${label}: edge ${i} ${k}`).toBe(wv);
			} else {
				const d = Math.abs(e[k] - wv);
				expect(d, `${label}: edge ${i} ${k} got ${e[k]} want ${wv}`).toBeLessThanOrEqual(tol);
				bump(k, d);
			}
		}
	});
	// graph: nodes exact (float(lrt) of a float32 is exact), edges in networkx order
	const g = got.graph.toJson();
	expect(g.nodes, `${label}: nodes`).toEqual(want.graph.nodes.map(([n, d]) => [n, { ref: d.ref, lrt: num(d.lrt) }]));
	expect(
		g.edges.map(([u, v]) => [u, v]),
		`${label}: graph edge order`
	).toEqual(want.graph.edges.map(([u, v]) => [u, v]));
	g.edges.forEach(([, , d], i) => {
		const wd = want.graph.edges[i][2];
		expect(Object.keys(d), `${label}: graph edge ${i} keys`).toEqual(['weight', 'shared', 'cesi', 'fdr_q']);
		for (const k of Object.keys(wd)) {
			const wv = num(wd[k]);
			if (GRAPH_EXACT.has(k) || tol === 0) expect(d[k], `${label}: graph edge ${i} ${k}`).toBe(wv);
			else {
				const dd = Math.abs(d[k] - wv);
				expect(dd, `${label}: graph edge ${i} ${k}`).toBeLessThanOrEqual(tol);
				bump(`graph.${k}`, dd);
			}
		}
	});
	// the record and the edge carry the same numbers
	for (const e of got.sigPairs) {
		const d = got.graph.getEdgeData(e.site_u, e.site_v);
		expect(d).toBeDefined();
		expect(d.weight).toBe(e.similarity);
		expect(d.shared).toBe(e.shared_taxa);
		expect(d.cesi).toBe(e.cesi);
		expect(d.fdr_q).toBe(e.fdr_q);
	}
	expect(got.graph.numberOfEdges()).toBe(want.graph.edges.length);
	expect(got.graph.numberOfNodes()).toBe(want.graph.nodes.length);
	return worst;
}

describe('fixtures/epistasis/compute_branch_coselection_network.json replay', () => {
	const cases = fixture('epistasis/compute_branch_coselection_network.json');

	it('has the five cases the manifest records', () => {
		expect(cases.map((c) => c.name)).toEqual([
			'planted_function_defaults',
			'planted_cli_defaults_min_sim_0.30',
			'planted_loose_filters',
			'planted_strict_cesi',
			'fewer_than_two_active_sites'
		]);
	});

	for (const c of cases) {
		it(`${c.name} (${c.tolerance})`, () => {
			const tol = c.tolerance === 'exact' ? 0 : Number(c.tolerance);
			const inp = c.inputs;
			const got = computeBranchCoselectionNetwork(
				inp.attributions,
				Float32Array.from(inp.lrts),
				inp.branch_names,
				inp.consensus_aas,
				{ minSim: inp.min_sim, minShared: inp.min_shared, maxFdr: inp.max_fdr, minLrt: inp.min_lrt, minCesi: inp.min_cesi }
			);
			const worst = compareNetwork(got, c.outputs, tol, c.name);
			if (tol > 0) {
				// Measured: the float32 pipeline agrees far inside the 1e-6 class; the residual is the
				// sgemm-vs-float64 dot (header of src/epistasis.js) — at most one float32 ulp on
				// similarity (1.2e-7 below 1), cesi and fdr_q (1.49e-7, seen on planted_loose_filters
				// where q reaches 0.5). Pinned so a drift to float64 arithmetic is visible.
				expect(worst.similarity ?? 0).toBeLessThanOrEqual(2e-7);
				expect(worst.cesi ?? 0).toBeLessThanOrEqual(1e-6);
				expect(worst.fdr_q ?? 0).toBeLessThanOrEqual(2e-7);
			}
		});
	}

	it('planted structure: sites 3,7,11 and 21,22,23 form the two triangles at the function defaults', () => {
		const c = cases[0];
		const got = computeBranchCoselectionNetwork(c.inputs.attributions, c.inputs.lrts, null, c.inputs.consensus_aas);
		const pairs = new Set(got.sigPairs.map((e) => `${e.site_u}-${e.site_v}`));
		expect(pairs).toEqual(new Set(['3-7', '3-11', '7-11', '21-22', '21-23', '22-23']));
		expect(got.graph.degree(3)).toBe(2);
		expect(got.graph.degree(31)).toBe(0); // inactive site: node present, no edges
		expect(got.graph.hasNode(31)).toBe(true);
	});

	it('accepts the matrix flat (Float32Array [L*N]) and infers N from lrts.length', () => {
		const c = cases[0];
		const rows = c.inputs.attributions;
		const flat = Float32Array.from(rows.flat());
		const a = computeBranchCoselectionNetwork(rows, c.inputs.lrts, null, c.inputs.consensus_aas);
		const b = computeBranchCoselectionNetwork(flat, c.inputs.lrts, null, c.inputs.consensus_aas);
		expect(b.sigPairs).toEqual(a.sigPairs);
		expect(b.graph.toJson()).toEqual(a.graph.toJson());
	});
});

describe('test/data/epistasis/coselection_network_cases.json (reference Python on seeded inputs)', () => {
	const cases = local('coselection_network_cases.json');

	for (const c of cases) {
		it(c.name, () => {
			const inp = c.inputs;
			const got = computeBranchCoselectionNetwork(inp.attributions, Float32Array.from(inp.lrts), null, inp.consensus_aas, {
				minSim: inp.min_sim,
				minShared: inp.min_shared,
				maxFdr: inp.max_fdr,
				minLrt: inp.min_lrt,
				minCesi: inp.min_cesi
			});
			compareNetwork(got, c.outputs, 1e-6, c.name);
		});
	}

	it('all_pairs_loose_n300 exercises every pair, the collinear guard and an inactive site', () => {
		const c = cases.find((x) => x.name === 'all_pairs_loose_n300');
		const L = c.inputs.L;
		// site 7 (0-indexed) is inactive: V = L - 1 and every active pair is an edge
		const V = L - 1;
		expect(c.outputs.sig_pairs.length).toBe((V * (V - 1)) / 2);
		const collinear = c.outputs.sig_pairs.find((e) => e.site_u === 3 && e.site_v === 9);
		expect(collinear).toBeDefined();
		expect(collinear.similarity).toBeGreaterThanOrEqual(1 - 2e-7);
	});

	it('signed_n3_df1 has negative cosines with p > 0.5 (the t.sf(t < 0) branch) and df = 1', () => {
		const c = cases.find((x) => x.name === 'signed_n3_df1');
		const neg = c.outputs.sig_pairs.filter((e) => e.similarity < 0);
		expect(neg.length).toBeGreaterThan(0);
		for (const e of neg) expect(e.p_val).toBeGreaterThan(0.5);
	});

	it('threshold_float32_promotion: a cosine exactly float32(0.35) passes min_sim=0.35 as in numpy', () => {
		const c = cases.find((x) => x.name === 'threshold_float32_promotion');
		expect(c.outputs.sig_pairs.length).toBe(1);
		expect(c.outputs.sig_pairs[0].similarity).toBe(Math.fround(0.35));
		expect(Math.fround(0.35)).toBeLessThan(0.35); // the reason a float64 comparison would drop it
		const got = computeBranchCoselectionNetwork(c.inputs.attributions, c.inputs.lrts, null, c.inputs.consensus_aas, {
			minSim: 0.35, minShared: 0, maxFdr: 1.0, minLrt: 0.0, minCesi: 0.0
		});
		expect(got.sigPairs.length).toBe(1);
		expect(got.sigPairs[0].similarity).toBe(Math.fround(0.35));
	});
});

describe('computeBranchCoselectionNetwork: unit semantics', () => {
	it('V < 2 returns no pairs and a graph with the nodes only', () => {
		const A = new Float32Array([0, 0, 1, 0, 0, 0]); // L=3, N=2, one active site
		const got = computeBranchCoselectionNetwork(A, [1, 2, 3], null, ['A', 'C', 'D'], { N: 2 });
		expect(got.sigPairs).toEqual([]);
		expect(got.graph.nodeList()).toEqual([[1, { ref: 'A', lrt: 1 }], [2, { ref: 'C', lrt: 2 }], [3, { ref: 'D', lrt: 3 }]]);
		expect(got.graph.numberOfEdges()).toBe(0);
	});

	it('L = 0 returns an empty graph', () => {
		const got = computeBranchCoselectionNetwork(new Float32Array(0), [], null, [], { N: 4 });
		expect(got.sigPairs).toEqual([]);
		expect(got.graph.numberOfNodes()).toBe(0);
	});

	it('df = max(1, N - 2): orthogonal vectors give t = 0 and p = 0.5 exactly', () => {
		const got = computeBranchCoselectionNetwork([[1, 0], [0, 1]], [2, 2], null, ['A', 'C'], {
			minSim: -1, minShared: 0, maxFdr: 1, minLrt: 0, minCesi: -1
		});
		expect(got.sigPairs.length).toBe(1);
		expect(got.sigPairs[0].similarity).toBe(0);
		expect(got.sigPairs[0].p_val).toBe(0.5);
		expect(got.sigPairs[0].fdr_q).toBe(0.5);
		expect(got.sigPairs[0].shared_taxa).toBe(0);
	});

	it('cesi floors each LRT at 0.1 in float32; identical rows have cosine 1.0000001 (float32 quirk)', () => {
		// Reference values measured with hyphaeon.epistasis.compute_branch_coselection_network on
		// np.float32 [[1,1],[1,1]], lrts [0.0, 0.3]: norms float32(sqrt 2) = 1.4142135, their outer
		// product rounds to 1.9999999, so sim = 2 / 1.9999999 = 1.0000001192092896 > 1; then
		// 1 - sim^2 < 0, the 1e-9 guard fires, t = sqrt(df / 1e-9) with df = 1 and p = 1.0066e-5.
		const got = computeBranchCoselectionNetwork([[1, 1], [1, 1]], [0.0, 0.3], null, ['A', 'C'], {
			minSim: -1, minShared: 0, maxFdr: 1, minLrt: 0, minCesi: -1
		});
		const e = got.sigPairs[0];
		expect(e.lrt_u).toBe(0);
		expect(e.lrt_v).toBe(Math.fround(0.3));
		expect(e.similarity).toBe(1.0000001192092896);
		expect(e.cesi).toBe(0.17320509254932404);
		expect(Math.abs(e.p_val - 1.0065840937937342e-5)).toBeLessThanOrEqual(1e-17);
		expect(e.fdr_q).toBe(Math.fround(1.0065840937937342e-5));
	});

	it('records are sorted by cesi descending, stably', () => {
		// three identical active rows: every pair has the same similarity; cesi orders by LRT product
		const A = [[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]];
		const got = computeBranchCoselectionNetwork(A, [1, 2, 3, 3], null, ['A', 'C', 'D', 'E'], {
			minSim: -1, minShared: 0, maxFdr: 1, minLrt: 0, minCesi: -1
		});
		const cesis = got.sigPairs.map((e) => e.cesi);
		for (let i = 1; i < cesis.length; i++) expect(cesis[i - 1]).toBeGreaterThanOrEqual(cesis[i]);
		// pairs (2,4) and (2,3) tie on cesi (lrt 2*3): triangle order puts (2,3) before (2,4)
		const tied = got.sigPairs.filter((e) => e.site_u === 2 && e.site_v >= 3);
		expect(tied.map((e) => e.site_v)).toEqual([3, 4]);
		// graph edges stay in triangle (insertion) order regardless of the record sort
		expect(got.graph.edges().map(([u, v]) => [u, v])).toEqual([[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]);
	});

	it('thresholds are compared as float32 (numpy promotes the Python scalar to the array dtype)', () => {
		// a pair whose max LRT is float32(1.1) = 1.10000002384 passes min_lrt 1.1 either way, but a cosine
		// of exactly float32(0.35) passes min_sim 0.35 only under float32 comparison
		const b = Math.sqrt(1 - 0.35 * 0.35);
		const got = computeBranchCoselectionNetwork([[1, 0], [0.35, b]], [3, 3], null, ['A', 'C'], {
			minSim: 0.35, minShared: 0, maxFdr: 1, minLrt: 0, minCesi: 0
		});
		expect(got.sigPairs.length).toBe(1);
		const stricter = computeBranchCoselectionNetwork([[1, 0], [0.35, b]], [3, 3], null, ['A', 'C'], {
			minSim: 0.35000002, minShared: 0, maxFdr: 1, minLrt: 0, minCesi: 0
		});
		expect(stricter.sigPairs.length).toBe(0);
	});

	it('rejects inconsistent shapes', () => {
		expect(() => computeBranchCoselectionNetwork([[1, 0]], [1, 2], null, ['A', 'C'])).toThrow();
		expect(() => computeBranchCoselectionNetwork([[1, 0], [0, 1]], [1, 2], null, ['A'])).toThrow();
		expect(() => computeBranchCoselectionNetwork(new Float32Array(5), [1, 2], null, ['A', 'C'])).toThrow();
	});
});

describe('compute_transformer_attributions (epistasis.py:51-114) via the fake-model references', () => {
	const cases = local('transformer_attributions_fake_model.json');

	it('has the three cases gen.py writes', () => {
		expect(cases.map((c) => c.name)).toEqual(['random_tokens_with_unknowns', 'single_valid_taxon_per_site', 'wide_n300']);
	});

	for (const c of cases) {
		it(`${c.name}: consensus, delta, clamp, Self & Liang float32 and the product`, () => {
			const { aa_tokens, y_soft, mean_root_attns, L, N } = c.inputs;
			const want = c.outputs;
			expect(want.leaf_dtype).toBe('float32');
			expect(want.lrts_dtype).toBe('float32');
			expect(want.pvals_dtype).toBe('float32');
			const got = computeTransformerAttributions({ attention: mean_root_attns, lrts: Float32Array.from(y_soft), aaTokens: aa_tokens });
			expect(got.L).toBe(L);
			expect(got.N).toBe(N);
			expect(got.consensusAas).toEqual(want.consensus_aas);
			expect(got.lrts).toBeInstanceOf(Float32Array);
			expect(Array.from(got.lrts)).toEqual(want.lrts.map(num));
			// leaf attributions: float32 x {0, 1} is exact
			expect(got.leafAttributions).toBeInstanceOf(Float32Array);
			expect(Array.from(got.leafAttributions)).toEqual(want.leaf_attributions.flat().map(num));
			// p: float64 chi2 survival agreeing to 1e-12, then one float32 rounding on each side
			expect(got.pvals).toBeInstanceOf(Float32Array);
			let worst = 0;
			want.pvals.forEach((w, s) => {
				worst = Math.max(worst, Math.abs(got.pvals[s] - num(w)));
			});
			expect(worst).toBeLessThanOrEqual(6e-8);
			// every zero-delta site has a zero attribution row; every nonzero attribution is the attention
			for (let s = 0; s < L; s++) {
				for (let n = 0; n < N; n++) {
					const i = s * N + n;
					if (got.delta[i] === 0) expect(got.leafAttributions[i]).toBe(0);
					else expect(got.leafAttributions[i]).toBe(Math.fround(mean_root_attns[s][n]));
				}
			}
		});
	}

	it('random_tokens_with_unknowns pins the tie, the all-invalid site and the clamp', () => {
		const c = cases[0];
		const { aa_tokens, y_soft, mean_root_attns } = c.inputs;
		const got = computeTransformerAttributions({ attention: mean_root_attns, lrts: y_soft, aaTokens: aa_tokens });
		expect(got.consensusAas[3]).toBe('-'); // every token 20
		expect(Array.from(got.delta.subarray(3 * got.N, 4 * got.N))).toEqual(new Array(got.N).fill(0));
		expect(got.consensusAas[4]).toBe('D'); // 6 x I(7) vs 6 x D(2): argmax -> lowest token
		expect(got.consensusAas[5]).toBe('N'); // conserved
		expect(Array.from(got.delta.subarray(5 * got.N, 6 * got.N))).toEqual(new Array(got.N).fill(0));
		expect(got.lrts[0]).toBe(0); // y_soft -2.5 clamped
		expect(got.pvals[0]).toBe(1);
		expect(got.lrts[1]).toBe(0);
		expect(got.pvals[1]).toBe(1);
	});

	it('runTransformerAttributions batches every site in order and matches the pure function', async () => {
		const c = cases[0];
		const { aa_tokens, y_soft, mean_root_attns, L, N } = c.inputs;
		const a = Int32Array.from(aa_tokens.flat());
		const loaded = /** @type {any} */ ({
			L, N, a, c: new Int32Array(L * N), d: new Float32Array(N * N), z: new Float32Array(N * 4), taxa: [], invariable: new Uint8Array(L)
		});
		const calls = [];
		const predict = (cb, ab, meta) => {
			expect(meta.phase).toBe('attribution-attention');
			expect(meta.batch).toBe(meta.siteIndices.length);
			expect(cb.length).toBe(meta.batch * N);
			expect(ab.length).toBe(meta.batch * N);
			calls.push(Array.from(meta.siteIndices));
			const lrt = new Float32Array(meta.batch);
			const attention = new Float32Array(meta.batch * N);
			for (let k = 0; k < meta.batch; k++) {
				const s = meta.siteIndices[k];
				expect(Array.from(ab.subarray(k * N, (k + 1) * N))).toEqual(aa_tokens[s]);
				lrt[k] = y_soft[s];
				attention.set(mean_root_attns[s], k * N);
			}
			return { lrt, attention };
		};
		const progress = [];
		const got = await runTransformerAttributions(loaded, predict, { batchSize: 7, onProgress: (p) => progress.push(p.done) });
		expect(calls.map((x) => x.length)).toEqual([7, 7, 7, 7, 2]); // the reference's chunking at batch_size=7
		expect(calls.flat()).toEqual(Array.from({ length: L }, (_, i) => i));
		expect(progress).toEqual([7, 14, 21, 28, 30]);
		const pure = computeTransformerAttributions({ attention: mean_root_attns, lrts: y_soft, aaTokens: aa_tokens });
		expect(got.consensusAas).toEqual(pure.consensusAas);
		expect(got.lrts).toEqual(pure.lrts);
		expect(got.pvals).toEqual(pure.pvals);
		expect(got.leafAttributions).toEqual(pure.leafAttributions);
		expect(Array.from(got.lrts)).toEqual(c.outputs.lrts.map(num));
	});

	it('runTransformerAttributions rejects a predict that returns the wrong shapes', async () => {
		const loaded = /** @type {any} */ ({ L: 2, N: 3, a: new Int32Array(6), c: new Int32Array(6), d: new Float32Array(9), z: new Float32Array(12) });
		await expect(runTransformerAttributions(loaded, () => ({ lrt: [1], attention: new Float32Array(6) }))).rejects.toThrow(/lrt/);
		await expect(runTransformerAttributions(loaded, () => ({ lrt: [1, 2], attention: new Float32Array(5) }))).rejects.toThrow(/attention/);
	});
});

describe('consensusDelta (epistasis.py:72-83)', () => {
	it('lowest token wins a tie; tokens >= 20 neither vote nor count as mutated', () => {
		const { consensusAas, majorAa, delta } = consensusDelta([[7, 2, 20, 2, 7, 22]], 1, 6);
		expect(consensusAas).toEqual(['D']);
		expect(majorAa[0]).toBe(2);
		expect(Array.from(delta)).toEqual([1, 0, 0, 0, 1, 0]);
	});

	it("a site with no valid token is '-' with a zero row and majorAa -1", () => {
		const { consensusAas, majorAa, delta } = consensusDelta(new Int32Array([20, 20, 20, 0, 0, 1]), 2, 3);
		expect(consensusAas).toEqual(['-', 'A']);
		expect(Array.from(majorAa)).toEqual([-1, 0]);
		expect(Array.from(delta)).toEqual([0, 0, 0, 0, 0, 1]);
	});

	it('maps tokens 0..19 to ACDEFGHIKLMNPQRSTVWY (REV_AA_MAP)', () => {
		const tokens = Array.from({ length: 20 }, (_, t) => [t]);
		expect(consensusDelta(tokens, 20, 1).consensusAas.join('')).toBe('ACDEFGHIKLMNPQRSTVWY');
	});
});

describe('CoselectionGraph: networkx 3.6.1 semantics', () => {
	it('iterates edges as G.edges(data=True): node order, then adjacency insertion order, each edge once', () => {
		// Measured with networkx 3.6.1 (nodes 1..7 pre-added; add_edge(3,7), (1,3), (7,1), (3,7) again, (9,2), (5,5)):
		//   edges: (1,3) (1,7) (2,9) (3,7) (5,5); nodes ..., 7, 9; degree 5 -> 2 (self-loop), 9 -> 1; 5 edges, 8 nodes
		const g = new CoselectionGraph();
		for (let n = 1; n <= 7; n++) g.addNode(n, { ref: 'A', lrt: n });
		g.addEdge(3, 7, { weight: 0.5 });
		g.addEdge(1, 3, { weight: 0.6 });
		g.addEdge(7, 1, { weight: 0.7 });
		g.addEdge(3, 7, { weight: 0.9, cesi: 2.0 }); // in-place update, position kept
		g.addEdge(9, 2, { weight: 0.1 }); // appends node 9 with empty attrs
		g.addEdge(5, 5, { weight: 0.3 }); // self-loop
		expect(g.edges()).toEqual([
			[1, 3, { weight: 0.6 }],
			[1, 7, { weight: 0.7 }],
			[2, 9, { weight: 0.1 }],
			[3, 7, { weight: 0.9, cesi: 2.0 }],
			[5, 5, { weight: 0.3 }]
		]);
		expect(g.nodeList().slice(-2)).toEqual([[7, { ref: 'A', lrt: 7 }], [9, {}]]);
		expect([1, 2, 3, 4, 5, 6, 7, 9].map((n) => g.degree(n))).toEqual([2, 1, 2, 0, 2, 0, 2, 1]);
		expect(g.numberOfEdges()).toBe(5);
		expect(g.numberOfNodes()).toBe(8);
		expect(g.getEdgeData(3, 7)).toBe(g.getEdgeData(7, 3)); // one shared datadict
		expect(g.hasEdge(7, 3)).toBe(true);
		expect(g.hasEdge(2, 3)).toBe(false);
		expect(g.degree(42)).toBe(0);
	});

	it('addNode on an existing node updates attrs in place and keeps the position', () => {
		const g = new CoselectionGraph();
		g.addNode(2, { ref: 'A' });
		g.addNode(1, { ref: 'C' });
		g.addNode(2, { lrt: 3 });
		expect(g.nodeList()).toEqual([[2, { ref: 'A', lrt: 3 }], [1, { ref: 'C' }]]);
	});

	it('toJson / fromJson round-trip the fixture shape in order', () => {
		const c = fixture('epistasis/compute_branch_coselection_network.json')[0];
		const g = CoselectionGraph.fromJson(c.outputs.graph);
		expect(g.toJson()).toEqual(c.outputs.graph);
		expect(g.edges().map(([u, v]) => [u, v])).toEqual([[3, 7], [3, 11], [7, 11], [21, 22], [21, 23], [22, 23]]);
		// (extract_epistatic_sectors_tse.json records this same graph as its input; it is not read
		// here so that scripts/fixture-coverage.mjs credits it to the sectors replay only.)
		const rebuilt = CoselectionGraph.fromJson(g.toJson());
		expect(rebuilt.toJson()).toEqual(c.outputs.graph);
		expect(rebuilt.getEdgeData(7, 3)).toEqual(c.outputs.graph.edges[0][2]);
	});
});
