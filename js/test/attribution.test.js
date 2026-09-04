/**
 * attribution.test.js — src/attribution.js against hyphaeon/attribution.py and cli.py cmd_meme.
 *
 * WHY THIS FILE EXISTS
 *
 *   1. attribute_selection parity WITHOUT a model: test/data/filter/run_alignment_filter_fake_model
 *      .json carries the reference's attributions for four synthetic alignments under a fake torch
 *      model (5 * mismatches - 0.75 * unknowns, exact in float32); the same function is the
 *      `predict` here. Covers base_lrts given, base_lrts absent with explicit focal sites (the
 *      predict-yourself branch, including a focal site below min_lrt), default taxon names, and the
 *      cmd_meme path (attribution on the ORIGINAL tokens with the CLEANED LRTs after --filter).
 *   2. test/data/attribution/attribute_selection_edge_cases.json: the 'TGA'/'*' consensus quirk
 *      for token 64, a zero-LRT site (diffuse), an all-zero distance matrix (max_tree_dist 1.0).
 *   3. fixtures/attribution/attribute_selection.json replayed on bat_oas1 with a lookup `predict`
 *      reconstructed from the fixture's own delta_lrt values (mod = site_lrt - delta): every string,
 *      integer, ordering, epoch and mode is checked exactly; depths at 1e-6 (our patristic matrix
 *      vs the reference's), deltas and pct at 1e-9.
 *   4. fixtures/e2e/meme_bat_oas1_attribute_filter.json: the 1-indexed `attributions` object and the
 *      per-site top_driver / top_mutation / epoch / mode fields cmd_meme derives.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { attributeSelection, attributionSiteFields, attributionsOneIndexed, INV_GENETIC_CODE } from '../src/attribution.js';
import { runAlignmentFilter } from '../src/filter.js';
import { loadAlignmentAndTree } from '../src/preprocess/assemble.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..', '..');
const FIXTURES = join(ROOT, 'fixtures');
const EXAMPLES = join(ROOT, 'examples');
const readJson = (p) => JSON.parse(readFileSync(p, 'utf8'));
const readExample = (name) => readFileSync(join(EXAMPLES, name), 'utf8');

/** The fake model of the local data (see filter.test.js). */
function fakePredict(c, a, meta) {
	const { batch, N } = meta;
	const out = new Float32Array(batch);
	for (let b = 0; b < batch; b++) {
		const counts = new Map();
		let u = 0;
		for (let i = 0; i < N; i++) {
			const t = a[b * N + i];
			if (t < 20) counts.set(t, (counts.get(t) ?? 0) + 1);
			else if (t === 20) u++;
		}
		let cons = -1;
		let best = -1;
		for (const [t, n] of counts) if (n > best || (n === best && t < cons)) ((cons = t), (best = n));
		let m = 0;
		for (let i = 0; i < N; i++) {
			const t = a[b * N + i];
			if (t < 20 && t !== cons) m++;
		}
		out[b] = Math.fround(Math.fround(5 * m) - Math.fround(0.75 * u));
	}
	return out;
}

/**
 * A predict that plays back a fixture's attributions: for the counterfactual (site, taxon) it
 * returns predicted_lrt - delta_lrt, i.e. the modified LRT the reference model produced.
 */
function playbackPredict(attributions) {
	const table = new Map();
	for (const rec of Object.values(attributions)) {
		for (const sp of rec.driving_species) table.set(`${rec.site_0indexed}:${sp.taxon_index}`, rec.predicted_lrt - sp.delta_lrt);
	}
	return (c, a, meta) => {
		expect(meta.phase).toBe('attribution');
		const out = new Float64Array(meta.batch);
		for (let k = 0; k < meta.batch; k++) {
			const key = `${meta.siteIndices[k]}:${meta.taxonIndices[k]}`;
			expect(table.has(key), `unexpected counterfactual ${key}`).toBe(true);
			out[k] = table.get(key);
		}
		return out;
	};
}

function expectClose(actual, expected, tol, label = '') {
	if (typeof expected === 'number' && typeof actual === 'number') {
		expect(Math.abs(actual - expected), `${label}: ${actual} vs ${expected}`).toBeLessThanOrEqual(tol);
	} else {
		expect(actual, label).toEqual(expected);
	}
}

/** Compare a Map<site, record> with the JSON `{str(site): record}`; tolerances per field class. */
function expectAttributions(got, want, { depthTol, deltaTol }) {
	expect([...got.keys()].map(String)).toEqual(Object.keys(want));
	for (const [site, w] of Object.entries(want)) {
		const g = got.get(Number(site));
		expect(g.site_0indexed).toBe(w.site_0indexed);
		expect(g.site_1indexed).toBe(w.site_1indexed);
		expectClose(g.predicted_lrt, w.predicted_lrt, deltaTol, `${site} predicted_lrt`);
		expect(g.consensus_codon, `${site} consensus_codon`).toBe(w.consensus_codon);
		expect(g.consensus_aa).toBe(w.consensus_aa);
		expect(g.num_mutated_taxa).toBe(w.num_mutated_taxa);
		expect(g.driving_species.length).toBe(w.driving_species.length);
		for (let i = 0; i < w.driving_species.length; i++) {
			const gs = g.driving_species[i];
			const ws = w.driving_species[i];
			for (const k of ['taxon', 'taxon_index', 'observed_codon', 'observed_aa', 'consensus_codon', 'consensus_aa']) {
				expect(gs[k], `${site} driver ${i} ${k}`).toBe(ws[k]);
			}
			expectClose(gs.delta_lrt, ws.delta_lrt, deltaTol, `${site} driver ${i} delta`);
			expectClose(gs.pct_signal_explained, ws.pct_signal_explained, deltaTol * 100, `${site} driver ${i} pct`);
			expectClose(gs.mean_patristic_depth, ws.mean_patristic_depth, depthTol, `${site} driver ${i} depth`);
		}
		const gw = g.when_selection_occurred;
		const ww = w.when_selection_occurred;
		expect(gw.evolutionary_epoch, `${site} epoch`).toBe(ww.evolutionary_epoch);
		expect(gw.mode_of_adaptation, `${site} mode`).toBe(ww.mode_of_adaptation);
		expectClose(gw.weighted_patristic_depth, ww.weighted_patristic_depth, depthTol, `${site} weighted depth`);
		expectClose(gw.tree_depth_ratio, ww.tree_depth_ratio, depthTol, `${site} ratio`);
	}
}

// ---------------------------------------------------------------------------------------------

describe('INV_GENETIC_CODE', () => {
	it('maps 0..60 back to the sense codons and 64 to TGA (the last stop in dict order)', () => {
		expect(INV_GENETIC_CODE.get(0)).toBe('TTT');
		expect(INV_GENETIC_CODE.get(60)).toBe('GGG');
		expect(INV_GENETIC_CODE.get(32)).toBe('ATG');
		expect(INV_GENETIC_CODE.get(64)).toBe('TGA');
		expect(INV_GENETIC_CODE.size).toBe(62);
	});
});

describe('attributeSelection against the reference with a fake model', () => {
	const cases = readJson(join(HERE, 'data', 'filter', 'run_alignment_filter_fake_model.json'));
	for (const c of cases) {
		const { alignment, tree } = c.inputs;
		const o = c.outputs;

		it(`${c.name}: base_lrts given (focal = lrt >= 3.84), records exact`, async () => {
			const loaded = loadAlignmentAndTree(alignment, tree);
			expect(loaded.taxa).toEqual(o.taxa);
			const calls = [];
			const got = await attributeSelection(loaded, (cc, aa, meta) => (calls.push(meta), fakePredict(cc, aa, meta)), {
				baseLrts: o.lrts_raw,
				minLrt: 3.84
			});
			expectAttributions(got, o.attributions, { depthTol: 1e-6, deltaTol: 0 });
			// one batched call per site holding every non-consensus taxon (all < 64 here)
			expect(calls.length).toBe(Object.keys(o.attributions).length);
			for (const m of calls) {
				expect(m.phase).toBe('attribution');
				expect(m.batch).toBe(m.taxonIndices.length);
				expect(new Set(m.siteIndices).size).toBe(1);
			}
			// the counterfactual tensors revert exactly one taxon per element
			const site = Number(Object.keys(o.attributions)[0]);
			const rec = got.get(site);
			const withTokens = [];
			await attributeSelection(loaded, (cc, aa, meta) => (withTokens.push({ cc, aa, meta }), fakePredict(cc, aa, meta)), { focalSites: [site], baseLrts: o.lrts_raw });
			const { cc, aa, meta } = withTokens[0];
			const N = loaded.N;
			for (let k = 0; k < meta.batch; k++) {
				const t = meta.taxonIndices[k];
				let diffs = 0;
				for (let i = 0; i < N; i++) if (cc[k * N + i] !== loaded.c[site * N + i]) diffs++;
				expect(diffs).toBe(1);
				expect(INV_GENETIC_CODE.get(cc[k * N + t])).toBe(rec.consensus_codon);
				expect(aa[k * N + t]).toBe(rec.consensus_aa === '*' ? 20 : 'ACDEFGHIKLMNPQRSTVWY'.indexOf(rec.consensus_aa));
			}
		});

		it(`${c.name}: no base_lrts, explicit focal sites [0, 22, 40, 45] (site 0 below min_lrt is skipped)`, async () => {
			const loaded = loadAlignmentAndTree(alignment, tree);
			const phases = [];
			const got = await attributeSelection(loaded, (cc, aa, meta) => (phases.push(meta.phase), fakePredict(cc, aa, meta)), {
				focalSites: [0, 22, 40, 45],
				minLrt: 3.84
			});
			expect(phases[0]).toBe('attribution-base');
			expectAttributions(got, o.attributions_no_base_focal_0_22_40_45, { depthTol: 1e-6, deltaTol: 0 });
			expect(got.has(0)).toBe(false);
		});

		it(`${c.name}: taxa=None gives Taxon_001.. names`, async () => {
			const loaded = loadAlignmentAndTree(alignment, tree);
			// options.taxa === null is the reference's taxa=None; an omitted option uses loaded.taxa
			const got = await attributeSelection(loaded, fakePredict, { focalSites: [22], baseLrts: o.lrts_raw, taxa: null });
			expectAttributions(got, o.attributions_default_names_focal_22, { depthTol: 1e-6, deltaTol: 0 });
			expect(got.get(22).driving_species[0].taxon).toMatch(/^Taxon_\d{3}$/);
			const named = await attributeSelection(loaded, fakePredict, { focalSites: [22], baseLrts: o.lrts_raw });
			expect(named.get(22).driving_species[0].taxon).toBe(o.attributions['22'].driving_species[0].taxon);
			const anon = await attributeSelection({ ...loaded, taxa: undefined }, fakePredict, { focalSites: [22], baseLrts: o.lrts_raw });
			expect(anon.get(22).driving_species[0].taxon).toBe(got.get(22).driving_species[0].taxon);
		});

		if (o.cmd_meme !== null) {
			it(`${c.name}: cmd_meme --filter --attribute: attribution on the original tokens with the cleaned LRTs`, async () => {
				const filt = await runAlignmentFilter({ alignmentText: alignment, treeText: tree }, fakePredict, { cliVariant: true });
				const lrts = filt.sites.map((s) => s.hyphaeon_lrt);
				const got = await attributeSelection(filt.raw.loaded, fakePredict, { baseLrts: lrts, minLrt: 3.84 });
				const want = o.cmd_meme.attributions; // 1-indexed keys
				const one = attributionsOneIndexed(got);
				expect(Object.keys(one)).toEqual(Object.keys(want));
				const zero = Object.fromEntries(Object.entries(want).map(([k, v]) => [String(Number(k) - 1), v]));
				expectAttributions(got, zero, { depthTol: 1e-6, deltaTol: 0 });
				for (const s of o.cmd_meme.sites) {
					if (!('top_driver' in s)) continue;
					const f = attributionSiteFields(got.get(s.site - 1));
					expect(f.evolutionary_epoch).toBe(s.evolutionary_epoch);
					expect(f.adaptation_mode).toBe(s.adaptation_mode);
					expect(f.top_driver).toBe(s.top_driver);
					expect(f.top_mutation).toBe(s.top_mutation);
					expect(f.attribution_details).toBe(got.get(s.site - 1));
				}
			});
		}
	}

	it('edge cases: token-64 consensus reads back as TGA/*, zero-LRT site is diffuse, zero distances use max 1.0', async () => {
		const edge = readJson(join(HERE, 'data', 'attribution', 'attribute_selection_edge_cases.json'));
		const tiny = (cRows, aRows, dRows) => {
			const N = cRows[0].length;
			const L = cRows.length;
			return {
				c: Int32Array.from(cRows.flat()),
				a: Int32Array.from(aRows.flat()),
				d: Float32Array.from(dRows.flat()),
				z: new Float32Array(N * 4),
				invariable: new Uint8Array(L),
				taxa: undefined,
				L,
				N
			};
		};
		const dist5 = [[0, 1, 2, 3, 4], [1, 0, 1, 2, 3], [2, 1, 0, 1, 2], [3, 2, 1, 0, 1], [4, 3, 2, 1, 0]];
		const g1 = await attributeSelection(tiny([[64, 64, 64, 0, 5]], [[20, 20, 20, 4, 15]], dist5), fakePredict, { focalSites: [0], minLrt: 0, baseLrts: [7.75] });
		expectAttributions(g1, edge.gap_majority_consensus, { depthTol: 0, deltaTol: 0 });
		expect(g1.get(0).consensus_codon).toBe('TGA');
		expect(g1.get(0).consensus_aa).toBe('*');
		const g2 = await attributeSelection(tiny([[0, 0, 1, 2]], [[4, 4, 4, 9]], [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]), fakePredict, { focalSites: [0], minLrt: 0, baseLrts: [0] });
		expectAttributions(g2, edge.zero_lrt_diffuse, { depthTol: 0, deltaTol: 0 });
		expect(g2.get(0).when_selection_occurred.evolutionary_epoch).toBe('Diffuse / Unresolved');
		const g3 = await attributeSelection(tiny([[0, 0, 13, 13, 29]], [[4, 4, 9, 9, 7]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]), fakePredict, { focalSites: [0], minLrt: 0, baseLrts: [10] });
		expectAttributions(g3, edge.zero_distances, { depthTol: 0, deltaTol: 0 });
	});

	it('focal sites default to every site when neither focal_sites nor base_lrts is given; batchSize chunks counterfactuals', async () => {
		const c = readJson(join(HERE, 'data', 'filter', 'run_alignment_filter_fake_model.json'))[0];
		const loaded = loadAlignmentAndTree(c.inputs.alignment, c.inputs.tree);
		const batches = [];
		const got = await attributeSelection(loaded, (cc, aa, meta) => (batches.push(meta), fakePredict(cc, aa, meta)), { batchSize: 2 });
		const base = batches.filter((m) => m.phase === 'attribution-base');
		expect(base.reduce((n, m) => n + m.batch, 0)).toBe(loaded.L); // every site, invariable included
		expect(batches.every((m) => m.batch <= 2)).toBe(true);
		const ref = await attributeSelection(loaded, fakePredict, { baseLrts: c.outputs.lrts_raw });
		expect([...got.keys()]).toEqual([...ref.keys()]);
		for (const [site, rec] of ref) expect(got.get(site)).toEqual(rec);
		await expect(attributeSelection(loaded, () => [1], { focalSites: [22], baseLrts: c.outputs.lrts_raw, batchSize: 64 })).rejects.toThrow(/counterfactuals/);
	});

	it('attributionSiteFields gives null driver fields when no taxon differs from the consensus', () => {
		const rec = {
			site_0indexed: 4,
			site_1indexed: 5,
			predicted_lrt: 5,
			consensus_codon: 'ATG',
			consensus_aa: 'M',
			num_mutated_taxa: 0,
			driving_species: [],
			when_selection_occurred: { evolutionary_epoch: 'Diffuse / Unresolved', mode_of_adaptation: 'Diffuse Background Variation', weighted_patristic_depth: 0, tree_depth_ratio: 0 }
		};
		expect(attributionSiteFields(rec)).toEqual({
			evolutionary_epoch: 'Diffuse / Unresolved',
			adaptation_mode: 'Diffuse Background Variation',
			top_driver: null,
			top_mutation: null,
			attribution_details: rec
		});
		expect(attributionsOneIndexed(new Map([[4, rec]]))).toEqual({ 5: rec });
	});
});

// ---------------------------------------------------------------------------------------------

describe('fixture replay (fixtures/attribution, fixtures/e2e) on bat_oas1', () => {
	it('attribute_selection.json: 6 sites, drivers, epochs and modes from the recorded deltas', async () => {
		const fx = readJson(join(FIXTURES, 'attribution', 'attribute_selection.json'))[0];
		const loaded = loadAlignmentAndTree(readExample(fx.inputs.alignment), readExample(fx.inputs.tree));
		expect(loaded.L).toBe(fx.outputs.base_lrts.length);
		const got = await attributeSelection(loaded, playbackPredict(fx.outputs.attributions), {
			baseLrts: fx.outputs.base_lrts,
			minLrt: fx.inputs.min_lrt
		});
		expect(got.size).toBe(fx.outputs.n_sites_attributed);
		expectAttributions(got, fx.outputs.attributions, { depthTol: 1e-6, deltaTol: 1e-9 });
	});

	it('meme_bat_oas1_attribute_filter.json: 1-indexed attributions and the per-site cmd_meme fields', async () => {
		const fx = readJson(join(FIXTURES, 'e2e', 'meme_bat_oas1_attribute_filter.json'))[0];
		const out = fx.outputs;
		const loaded = loadAlignmentAndTree(readExample(out.alignment), readExample(out.tree));
		const zeroIndexed = Object.fromEntries(Object.entries(out.attributions).map(([k, v]) => [String(Number(k) - 1), v]));
		const got = await attributeSelection(loaded, playbackPredict(zeroIndexed), {
			baseLrts: out.sites.map((s) => s.hyphaeon_lrt),
			minLrt: 3.84
		});
		expectAttributions(got, zeroIndexed, { depthTol: 1e-6, deltaTol: 1e-9 });
		expect(Object.keys(attributionsOneIndexed(got))).toEqual(Object.keys(out.attributions));
		for (const s of out.sites) {
			const rec = got.get(s.site - 1);
			if (rec === undefined) {
				expect('top_driver' in s).toBe(false);
				continue;
			}
			const f = attributionSiteFields(rec);
			expect(f.evolutionary_epoch).toBe(s.evolutionary_epoch);
			expect(f.adaptation_mode).toBe(s.adaptation_mode);
			expect(f.top_driver).toBe(s.top_driver);
			expect(f.top_mutation).toBe(s.top_mutation);
		}
	});
});
