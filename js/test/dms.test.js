/**
 * dms.test.js — src/dms.js against hyphaeon/epistasis.py's digital DMS (446-628, 724-770).
 *
 * WHY THIS FILE EXISTS
 *
 * Every number `run_insilico_selection_dms` returns is a forward pass, so there are two kinds of
 * evidence here and neither alone is enough:
 *
 *   1. test/data/dms/run_insilico_selection_dms_fake_model.json (its gen.py runs the REAL
 *      reference function against a fake model whose output is a deterministic function of the
 *      token tensors). This pins WHAT IS SWEPT and WHAT IS CALLED, EXACTLY: the wild-type
 *      fallback when the focal taxon has a gap / stop / ambiguity, both bincount ties, the
 *      all-gap column, focal-taxon resolution (substring, miss, empty string), target-site
 *      de-duplication and range filtering, the empty-subset early return, the [batch, N] shape of
 *      every forward pass at three batch sizes, and run_digital_dms_analysis's record and key
 *      order. bat_oas1's focal taxon carries a residue at every target site, so the fixture below
 *      reaches none of those branches.
 *   2. fixtures/dms/run_insilico_selection_dms.json replayed with a playback `predict` built from
 *      the fixture's own recorded outputs (the pattern attribution.test.js established): the
 *      baseline is the recorded `baseline_lrt` and each mutant is `baseline + recorded delta`.
 *      This pins the derived arithmetic — the float32 delta, the four reductions, the Self-Liang
 *      p-value, the record and mutant_deltas key order — against the real network's numbers on
 *      bat_oas1, at both focal taxa the fixture records.
 *
 * A REAL-MODEL REPLAY OF THIS FIXTURE BELONGS TO THE APP RUNTIME, not here. The library has no
 * onnxruntime by design (PLAN.md §5.5), so the only thing a test in this package can do with the
 * recorded LRTs is play them back; whether the ONNX graph reproduces them at the fixture's 1e-5
 * class is a question about the graph and the session, and `veg/hyphaeon-app`'s runtime tests are
 * where it is asked (PHASE1A.md "Known gaps" item 3 lists this fixture as exactly that).
 *
 * WHY THE FIXTURE REPLAY IS ASSERTED EXACTLY (PLAYBACK_TOL = 0). The playback reconstructs each
 * mutant LRT as float32(baseline + delta) and the port then recomputes float32(mutant - baseline).
 * That round trip is not exact in general, but it is exact for all 152 mutants this fixture
 * records (measured with numpy: max |recomputed - recorded| = 0.0), so the replay checks bit
 * equality. If a regenerated fixture ever loses that, the honest fix is to relax PLAYBACK_TOL to
 * the case's declared 1e-5 class — not to change the port.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
	runInsilicoSelectionDms,
	runDigitalDmsAnalysis,
	digitalDmsRecord,
	resolveFocalTaxon,
	dmsTargetSites,
	dmsWildTypeAa,
	CANONICAL_AA_TO_CODON,
	DMS_MUTANTS_PER_SITE
} from '../src/dms.js';
import { loadAlignmentAndTree } from '../src/preprocess/assemble.js';
import { GENETIC_CODE, AA_MAP, CODON_TO_AA, CODON_LIST } from '../src/preprocess/tokenizer.js';
import { AA_LIST } from '../src/preprocess/modelContract.js';
import { pvalsFromLrtSelfLiang } from '../src/stats.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..', '..');
const readJson = (/** @type {string} */ p) => JSON.parse(readFileSync(p, 'utf8'));
const readExample = (/** @type {string} */ name) => readFileSync(join(ROOT, 'examples', name), 'utf8');

const LOCAL = readJson(join(HERE, 'data', 'dms', 'run_insilico_selection_dms_fake_model.json'));

/** See the header: measured 0.0 over all 152 mutants of the fixture. */
const PLAYBACK_TOL = 0;

/**
 * `p_value` is the ONLY field of a record that is not bit-equal to the reference, on either the
 * fake-model or the fixture side: it is `0.5 * chi2.sf(lrt, 1)`, and numeric/special.js's
 * regularized incomplete gamma is within ~1e-13 relative of scipy's rather than identical to it.
 * That is the `1e-9` class fixtures/README.md assigns to "float64 special functions and
 * statistics: χ²/t/hypergeometric survival". Worst difference measured over all 47 records this
 * file checks: 1.9e-15 absolute (at p = 0.1179). Everything else — deltas, the four float32
 * reductions, baselines, residues, key order, call shapes — is asserted exactly.
 */
const P_VALUE_TOL = 1e-9;

/**
 * test/data/dms/gen.py's fake model, character for character:
 *   y = float32( sum_i ((a_i + 1) * 0.113 + (c_i + 1) * 0.0071) * (1 + 0.05 * i)  -  10.0 )
 * accumulated left to right in float64 over the N taxa of one batch element, rounded to float32
 * once (`torch.tensor(vals, dtype=torch.float32)`).
 */
function fakePredict(c, a, meta) {
	const { batch, N } = meta;
	const out = new Float32Array(batch);
	for (let b = 0; b < batch; b++) {
		let acc = 0.0;
		for (let i = 0; i < N; i++) {
			const ai = a[b * N + i];
			const ci = c[b * N + i];
			acc = acc + ((ai + 1) * 0.113 + (ci + 1) * 0.0071) * (1.0 + 0.05 * i);
		}
		out[b] = Math.fround(acc - LOCAL.k_offset);
	}
	return out;
}

/** fakePredict plus a log of every call's [batch, N] shape, in call order. */
function recordingFakePredict() {
	/** @type {number[][]} */
	const shapes = [];
	/** @type {{c: Int32Array, a: Int32Array, meta: any}[]} */
	const calls = [];
	const predict = (c, a, meta) => {
		shapes.push([meta.batch, meta.N]);
		calls.push({ c, a, meta });
		return fakePredict(c, a, meta);
	};
	return { predict, shapes, calls };
}

/**
 * A predict that plays a fixture case back: `dms-baseline` answers with the recorded
 * `baseline_lrt` of the swept sites (and `filler` elsewhere — no asserted field reads those, which
 * the "filler does not leak" test below proves), `dms` with `baseline + recorded delta`.
 */
function playbackPredict(plasticity, filler = 0) {
	/** @type {Map<number, number>} */
	const baseline = new Map();
	/** @type {Map<string, number>} */
	const mutants = new Map();
	for (const r of plasticity) {
		const site0 = r.site - 1;
		baseline.set(site0, r.baseline_lrt);
		for (const [aa, delta] of Object.entries(r.mutant_deltas)) {
			mutants.set(`${site0}:${aa}`, Math.fround(r.baseline_lrt + delta));
		}
	}
	return (c, a, meta) => {
		const out = new Float64Array(meta.batch);
		for (let k = 0; k < meta.batch; k++) {
			const site = meta.siteIndices[k];
			if (meta.phase === 'dms-baseline') {
				out[k] = baseline.has(site) ? baseline.get(site) : filler;
			} else {
				expect(meta.phase).toBe('dms');
				const key = `${site}:${meta.mutantAas[k]}`;
				expect(mutants.has(key), `unexpected mutant ${key}`).toBe(true);
				out[k] = mutants.get(key);
			}
		}
		return out;
	};
}

/** The loaded bundle of the synthetic 6-taxon / 8-codon alignment gen.py built. */
function loadLocal() {
	return loadAlignmentAndTree(LOCAL.alignment, LOCAL.tree, { pruneDuplicates: true });
}

/** Deep-compare a record list against the reference's, every field exact. */
function expectRecords(got, want, tol = 0) {
	expect(got.length).toBe(want.length);
	for (let i = 0; i < want.length; i++) {
		const g = got[i];
		const w = want[i];
		expect(g.site, `record ${i} site`).toBe(w.site);
		expect(g.wt_aa, `record ${i} wt_aa`).toBe(w.wt_aa);
		expect(Object.keys(g), `record ${i} key order`).toEqual(Object.keys(w));
		expect(Object.keys(g.mutant_deltas), `record ${i} mutant key order`).toEqual(Object.keys(w.mutant_deltas));
		expect(Math.abs(g.p_value - w.p_value), `record ${i} p_value: ${g.p_value} vs ${w.p_value}`).toBeLessThanOrEqual(P_VALUE_TOL);
		for (const k of ['baseline_lrt', 'intrinsic_plasticity', 'mean_delta_lrt', 'max_delta_lrt', 'min_delta_lrt']) {
			if (tol === 0) expect(g[k], `record ${i} ${k}`).toBe(w[k]);
			else expect(Math.abs(g[k] - w[k]), `record ${i} ${k}: ${g[k]} vs ${w[k]}`).toBeLessThanOrEqual(tol);
		}
		for (const [aa, d] of Object.entries(w.mutant_deltas)) {
			if (tol === 0) expect(g.mutant_deltas[aa], `record ${i} delta ${aa}`).toBe(d);
			else expect(Math.abs(g.mutant_deltas[aa] - d)).toBeLessThanOrEqual(tol);
		}
	}
}

// ---------------------------------------------------------------------------------------------
// The tables and the three decisions, in isolation
// ---------------------------------------------------------------------------------------------

describe('CANONICAL_AA_TO_CODON (epistasis.py:44-49)', () => {
	it('is a sense codon for each of the 20 residues and translates back to it', () => {
		expect(CANONICAL_AA_TO_CODON.size).toBe(20);
		expect([...CANONICAL_AA_TO_CODON.keys()].join('')).toBe(AA_LIST);
		for (const [aa, codon] of CANONICAL_AA_TO_CODON) {
			expect(CODON_TO_AA.get(codon), `${aa} -> ${codon}`).toBe(aa);
			expect(GENETIC_CODE.get(codon)).toBeLessThan(61); // a sense codon, never the 64 sentinel
		}
	});

	it('is NOT the first codon of each residue in table order — 18 of 20 differ', () => {
		/** @type {Map<string, string>} */
		const firstInOrder = new Map();
		for (const codon of CODON_LIST) {
			const aa = CODON_TO_AA.get(codon);
			if (aa !== '*' && !firstInOrder.has(aa)) firstInOrder.set(aa, codon);
		}
		const differing = [...CANONICAL_AA_TO_CODON].filter(([aa, codon]) => firstInOrder.get(aa) !== codon);
		expect(differing.length).toBe(18);
		// the two that coincide are the single-codon residues
		const same = [...CANONICAL_AA_TO_CODON].filter(([aa, codon]) => firstInOrder.get(aa) === codon).map(([aa]) => aa);
		expect(same).toEqual(['M', 'W']);
		expect(CANONICAL_AA_TO_CODON.get('L')).toBe('CTG'); // not TTA
		expect(CANONICAL_AA_TO_CODON.get('S')).toBe('AGC'); // not TCT
	});
});

describe('resolveFocalTaxon (epistasis.py:482-489)', () => {
	const taxa = ['M_lyra', 'Beta', 'Gamma', 'delta_X'];
	it('defaults to index 0, matches case-insensitive substrings, and takes the FIRST match', () => {
		expect(resolveFocalTaxon(taxa)).toEqual({ index: 0, name: 'M_lyra' });
		expect(resolveFocalTaxon(taxa, 'DELTA_x')).toEqual({ index: 3, name: 'delta_X' });
		// 'ta' occurs in both 'Beta' and 'delta_X'; the first taxon in order wins
		expect(resolveFocalTaxon(taxa, 'ta')).toEqual({ index: 1, name: 'Beta' });
		// any substring counts, so a single letter can select an unexpected taxon: 'm_lyra' ends in 'a'
		expect(resolveFocalTaxon(taxa, 'a')).toEqual({ index: 0, name: 'M_lyra' });
	});
	it('keeps index 0 silently when nothing matches, and skips the search for a falsy name', () => {
		expect(resolveFocalTaxon(taxa, 'no_such_taxon')).toEqual({ index: 0, name: 'M_lyra' });
		expect(resolveFocalTaxon(taxa, '')).toEqual({ index: 0, name: 'M_lyra' });
		expect(resolveFocalTaxon(taxa, null)).toEqual({ index: 0, name: 'M_lyra' });
	});
	it('reports "consensus" when there are no taxa at all', () => {
		expect(resolveFocalTaxon([], 'anything')).toEqual({ index: 0, name: 'consensus' });
	});
});

describe('dmsTargetSites (epistasis.py:470-474)', () => {
	it('sweeps every site when no subset is given', () => {
		expect(dmsTargetSites(null, 4)).toEqual([0, 1, 2, 3]);
		expect(dmsTargetSites(undefined, 2)).toEqual([0, 1]);
	});
	it('de-duplicates, sorts and drops sites outside [0, L)', () => {
		expect(dmsTargetSites([7, 7, 2, -1, 99, 0, 2], 8)).toEqual([0, 2, 7]);
		expect(dmsTargetSites([], 8)).toEqual([]);
		expect(dmsTargetSites([-1, 8, 100], 8)).toEqual([]);
	});
	it('range-tests the raw value and truncates toward zero, as `int(s)` does', () => {
		expect(dmsTargetSites([7.9], 8)).toEqual([7]); // 7.9 < 8, int(7.9) = 7
		expect(dmsTargetSites([-0.5], 8)).toEqual([]); // 0 <= -0.5 is false
	});
});

describe('dmsWildTypeAa (epistasis.py:532-539)', () => {
	// one site, N taxa: a[i] is the amino-acid token of taxon i
	const site = (tokens) => ({ a: Int32Array.from(tokens), N: tokens.length });
	const wt = (tokens, focal) => {
		const s = site(tokens);
		return dmsWildTypeAa(s.a, 0, focal, s.N);
	};
	it('uses the focal residue when it has one', () => {
		expect(wt([10, 0, 0], 0)).toBe('M');
		expect(wt([10, 19, 0], 1)).toBe('Y');
	});
	it('falls back to the site majority when the focal token is a gap / stop / ambiguity (>= 20)', () => {
		expect(wt([20, 0, 0, 0, 1, 1], 0)).toBe('A'); // majority token 0
		expect(wt([20, 8, 8, 12, 12, 12], 0)).toBe('P'); // majority token 12
	});
	it('breaks a tie toward the LOWEST token, as np.bincount + argmax does', () => {
		expect(wt([20, 1, 1, 2, 2, 4], 0)).toBe('C'); // C(1) and D(2) tie -> C
		expect(wt([20, 5, 5, 15, 15, 18], 0)).toBe('G'); // G(5) and S(15) tie -> G
	});
	it('returns A when no taxon carries a residue at the site', () => {
		expect(wt([20, 20, 20, 20], 0)).toBe('A');
	});
});

// ---------------------------------------------------------------------------------------------
// The reference function itself, through a fake model (test/data/dms)
// ---------------------------------------------------------------------------------------------

describe('runInsilicoSelectionDms against the reference with a fake model', () => {
	it('the loader reproduces the tokens gen.py swept', () => {
		const loaded = loadLocal();
		expect(loaded.taxa).toEqual(LOCAL.taxa);
		expect(loaded.L).toBe(LOCAL.L);
		expect(loaded.N).toBe(LOCAL.taxa.length);
		for (let s = 0; s < loaded.L; s++) {
			expect([...loaded.c.subarray(s * loaded.N, (s + 1) * loaded.N)], `codons at site ${s}`).toEqual(LOCAL.codon_tokens[s]);
			expect([...loaded.a.subarray(s * loaded.N, (s + 1) * loaded.N)], `aas at site ${s}`).toEqual(LOCAL.aa_tokens[s]);
		}
		expect([...loaded.invariable].map(Boolean)).toEqual(LOCAL.invariable);
	});

	for (const c of LOCAL.cases) {
		it(`${c.name}: records and forward-pass shapes exact`, async () => {
			const loaded = loadLocal();
			const { predict, shapes } = recordingFakePredict();
			const got = await runInsilicoSelectionDms(loaded, predict, {
				focalTaxon: c.inputs.focal_taxon,
				siteSubset: c.inputs.target_sites,
				// the reference's `safe_batch_size`; compute_adaptive_safe_batch_size is the runtime's
				batchSize: c.inputs.safe_batch_size
			});
			expectRecords(got, c.outputs.plasticity);
			expect(shapes, 'forward-pass [batch, N] shapes in call order').toEqual(c.outputs.call_shapes);
		});
	}

	it('an empty target list returns [] without calling the model at all', async () => {
		const loaded = loadLocal();
		const { predict, shapes } = recordingFakePredict();
		expect(await runInsilicoSelectionDms(loaded, predict, { siteSubset: [], batchSize: 64 })).toEqual([]);
		expect(shapes).toEqual([]);
	});

	it('results do not depend on batching, but call shapes do', async () => {
		const loaded = loadLocal();
		const runs = [];
		for (const batchSize of [64, 19, 1, 5, 1000]) {
			const { predict, shapes } = recordingFakePredict();
			runs.push({ batchSize, shapes, records: await runInsilicoSelectionDms(loaded, predict, { batchSize }) });
		}
		const reference = LOCAL.cases.find((c) => c.name === 'all_sites_focal_default').outputs.plasticity;
		for (const r of runs) expectRecords(r.records, reference);
		// ... while the reference's own two batch sizes produced different call shapes
		expect(runs[0].shapes).not.toEqual(runs[1].shapes);
		expect(runs[0].shapes).toEqual(LOCAL.cases.find((c) => c.name === 'all_sites_focal_default').outputs.call_shapes);
		expect(runs[1].shapes).toEqual(LOCAL.cases.find((c) => c.name === 'batch_size_19_one_site_per_pass').outputs.call_shapes);
	});

	it('mutates only the focal taxon, changing BOTH the codon and the amino-acid token', async () => {
		const loaded = loadLocal();
		const { N } = loaded;
		const { predict, calls } = recordingFakePredict();
		await runInsilicoSelectionDms(loaded, predict, { focalTaxon: 'delta_X', siteSubset: [5], batchSize: 64 });
		const mutantCalls = calls.filter((k) => k.meta.phase === 'dms');
		expect(mutantCalls.length).toBe(1);
		const { c, a, meta } = mutantCalls[0];
		expect(meta.batch).toBe(DMS_MUTANTS_PER_SITE);
		expect(meta.focalIndex).toBe(3);
		const wtC = loaded.c.subarray(5 * N, 6 * N);
		const wtA = loaded.a.subarray(5 * N, 6 * N);
		const seen = [];
		for (let k = 0; k < meta.batch; k++) {
			let changed = [];
			for (let i = 0; i < N; i++) {
				if (c[k * N + i] !== wtC[i] || a[k * N + i] !== wtA[i]) changed.push(i);
			}
			expect(changed, `mutant ${k} changed taxa`).toEqual([3]);
			const aa = meta.mutantAas[k];
			const codon = CANONICAL_AA_TO_CODON.get(aa);
			expect(c[k * N + 3], `mutant ${k} codon token`).toBe(GENETIC_CODE.get(codon));
			expect(a[k * N + 3], `mutant ${k} aa token`).toBe(AA_MAP.get(aa));
			expect(meta.siteIndices[k]).toBe(5);
			seen.push(aa);
		}
		// the 19 alternatives are AA_LIST minus the wild type, in AA_LIST order
		const wtAa = dmsWildTypeAa(loaded.a, 5, 3, N);
		expect(seen.join('')).toBe([...AA_LIST].filter((x) => x !== wtAa).join(''));
	});

	it('reports progress per baseline chunk and per mutant chunk', async () => {
		const loaded = loadLocal();
		const seen = [];
		await runInsilicoSelectionDms(loaded, fakePredict, { batchSize: 19, progress: (p) => seen.push(p) });
		const baseline = seen.filter((p) => p.phase === 'dms-baseline');
		const dms = seen.filter((p) => p.phase === 'dms');
		expect(baseline.length).toBeGreaterThan(0);
		expect(baseline.at(-1).done).toBe(loaded.L);
		expect(dms.length).toBe(LOCAL.L); // one site per pass at batch 19
		expect(dms.at(-1)).toMatchObject({ done: LOCAL.L, total: LOCAL.L, mutantsDone: 19 * LOCAL.L, totalMutants: 19 * LOCAL.L });
		// mean plasticity is the running mean over the records so far
		const all = LOCAL.cases.find((c) => c.name === 'all_sites_focal_default').outputs.plasticity;
		const mean = all.reduce((t, r) => t + r.intrinsic_plasticity, 0) / all.length;
		expect(Math.abs(dms.at(-1).meanPlasticity - mean)).toBeLessThan(1e-12);
		// a non-function `progress` (the reference's boolean) is ignored, not called
		await expect(runInsilicoSelectionDms(loaded, fakePredict, { batchSize: 19, progress: true, siteSubset: [0] })).resolves.toHaveLength(1);
	});

	it('rejects a predict that returns the wrong number of values, in either phase', async () => {
		const loaded = loadLocal();
		// the baseline sweep is guarded by predictSiteLrts (filter.js) ...
		await expect(runInsilicoSelectionDms(loaded, () => [1, 2], { siteSubset: [0], batchSize: 64 })).rejects.toThrow(
			/predict returned 2 values for a batch of 8 sites/
		);
		// ... and the mutant pass by this module: a correct baseline, then a short mutant answer
		const shortMutants = (c, a, meta) => (meta.phase === 'dms-baseline' ? fakePredict(c, a, meta) : [1, 2]);
		await expect(runInsilicoSelectionDms(loaded, shortMutants, { siteSubset: [0], batchSize: 64 })).rejects.toThrow(
			/predict returned 2 values for 19 mutants/
		);
	});

	it('targetSites is accepted under the Python name when siteSubset is absent', async () => {
		const loaded = loadLocal();
		const viaSubset = await runInsilicoSelectionDms(loaded, fakePredict, { siteSubset: [0, 5], batchSize: 64 });
		const viaTarget = await runInsilicoSelectionDms(loaded, fakePredict, { targetSites: [0, 5], batchSize: 64 });
		expect(viaTarget).toEqual(viaSubset);
		// siteSubset wins when both are given, including an explicit null ("every site")
		const both = await runInsilicoSelectionDms(loaded, fakePredict, { siteSubset: null, targetSites: [0], batchSize: 64 });
		expect(both).toHaveLength(LOCAL.L);
	});
});

describe('runDigitalDmsAnalysis / digitalDmsRecord (epistasis.py:727-773)', () => {
	const d = LOCAL.digital;

	it('reproduces the reference result dict and its key order', async () => {
		const loaded = loadLocal();
		const { predict, shapes } = recordingFakePredict();
		const got = await runDigitalDmsAnalysis(loaded, predict, {
			alignment: d.inputs.alignment,
			tree: d.inputs.tree,
			focalTaxon: d.inputs.focal_taxon,
			batchSize: d.inputs.batch_size
		});
		expect(Object.keys(got)).toEqual(d.outputs.key_order);
		expect(got.alignment).toBe(d.outputs.result.alignment);
		expect(got.tree).toBe(d.outputs.result.tree);
		expect(got.taxa_count).toBe(d.outputs.result.taxa_count);
		expect(got.codon_count).toBe(d.outputs.result.codon_count);
		expect(got.total_mutations).toBe(d.outputs.result.total_mutations);
		expect(got.focal_taxon).toBe(d.outputs.result.focal_taxon);
		expectRecords(got.plasticity, d.outputs.result.plasticity);
		expect(shapes).toEqual(d.outputs.call_shapes);
	});

	it('reports the caller’s focal_taxon string, not the taxon that was swept', () => {
		const loaded = loadLocal();
		const rec = digitalDmsRecord(loaded, [], { focalTaxon: 'beta' });
		expect(rec.focal_taxon).toBe('beta'); // the swept taxon is 'Beta'
		expect(resolveFocalTaxon(loaded.taxa, 'beta').name).toBe('Beta');
		expect(d.outputs.result.focal_taxon).toBe('beta');
	});

	it('falls back to taxa[0], then to "consensus", for a falsy focal_taxon', () => {
		const loaded = loadLocal();
		expect(digitalDmsRecord(loaded, [], {}).focal_taxon).toBe('M_lyra');
		expect(digitalDmsRecord(loaded, [], { focalTaxon: '' }).focal_taxon).toBe('M_lyra');
		expect(digitalDmsRecord(loaded, [], { focalTaxon: null, taxa: [] }).focal_taxon).toBe('consensus');
	});

	it('counts total_mutations as 19 * codon_count whatever was swept, and shares one array', () => {
		const loaded = loadLocal();
		const plasticity = [];
		const rec = digitalDmsRecord(loaded, plasticity, {});
		expect(rec.total_mutations).toBe(19 * loaded.L);
		expect(rec.total_mutations).toBe(d.outputs.result.total_mutations);
		// plasticity and selection_dms_plasticity are the SAME array, as the Python binds one list
		expect(rec.selection_dms_plasticity).toBe(rec.plasticity);
		expect(d.outputs.plasticity_is_selection_dms_plasticity).toBe(true);
		expect(rec.alignment).toBe(null); // no path given
		expect(rec.tree).toBe(null);
	});
});

// ---------------------------------------------------------------------------------------------
// fixtures/dms replayed on bat_oas1 (the real network's numbers, played back)
// ---------------------------------------------------------------------------------------------

describe('fixture replay (fixtures/dms/run_insilico_selection_dms.json) on bat_oas1', () => {
	const cases = readJson(join(ROOT, 'fixtures', 'dms', 'run_insilico_selection_dms.json'));

	for (const fx of cases) {
		it(`${fx.name}: ${fx.outputs.plasticity.length} sites, every derived field exact from the recorded LRTs`, async () => {
			const loaded = loadAlignmentAndTree(readExample(fx.inputs.alignment), readExample(fx.inputs.tree));
			// the focal taxon the fixture recorded, from the names the loader produced
			expect(resolveFocalTaxon(loaded.taxa, fx.inputs.focal_taxon)).toEqual({
				index: fx.outputs.focal_index,
				name: fx.outputs.focal_name
			});
			const got = await runInsilicoSelectionDms(loaded, playbackPredict(fx.outputs.plasticity), {
				focalTaxon: fx.inputs.focal_taxon,
				siteSubset: fx.inputs.target_sites,
				batchSize: fx.inputs.batch_size
			});
			expect(got.map((r) => r.site)).toEqual(fx.outputs.plasticity.map((r) => r.site));
			expectRecords(got, fx.outputs.plasticity, PLAYBACK_TOL);
		});
	}

	it('sweeps the 19 non-wild-type residues in AA_LIST order at every site', () => {
		for (const fx of cases) {
			for (const r of fx.outputs.plasticity) {
				const keys = Object.keys(r.mutant_deltas);
				expect(keys.length).toBe(DMS_MUTANTS_PER_SITE);
				expect(keys.includes(r.wt_aa), `${r.site} sweeps its own wild type`).toBe(false);
				expect(keys.join('')).toBe([...AA_LIST].filter((aa) => aa !== r.wt_aa).join(''));
			}
		}
	});

	it('p_value is the Self-Liang p of the baseline, and delta is mutant MINUS baseline', () => {
		for (const fx of cases) {
			for (const r of fx.outputs.plasticity) {
				expect(Math.abs(r.p_value - pvalsFromLrtSelfLiang([r.baseline_lrt])[0])).toBeLessThanOrEqual(P_VALUE_TOL);
				// the recorded reductions agree with the recorded per-mutant deltas
				const deltas = Float32Array.from(Object.values(r.mutant_deltas));
				expect(Math.max(...deltas)).toBe(r.max_delta_lrt);
				expect(Math.min(...deltas)).toBe(r.min_delta_lrt);
			}
		}
		// the sign: a negative mean_delta_lrt with deltas straddling zero (see the module header)
		const first = cases[0].outputs.plasticity[0];
		expect(first.mean_delta_lrt).toBeLessThan(0);
		expect(first.max_delta_lrt).toBeGreaterThan(0);
		expect(first.min_delta_lrt).toBeLessThan(0);
	});

	it('the baseline sweep covers every site, and non-swept baselines never reach a record', async () => {
		const fx = cases[0];
		const loaded = loadAlignmentAndTree(readExample(fx.inputs.alignment), readExample(fx.inputs.tree));
		/** @type {number[]} */
		const baselineSites = [];
		const spy = (c, a, meta) => {
			if (meta.phase === 'dms-baseline') baselineSites.push(...meta.siteIndices);
			return playbackPredict(fx.outputs.plasticity)(c, a, meta);
		};
		const got = await runInsilicoSelectionDms(loaded, spy, {
			siteSubset: fx.inputs.target_sites,
			batchSize: fx.inputs.batch_size
		});
		// epistasis.py:498 runs over range(0, L) — every site, variable or not
		expect(baselineSites).toEqual(Array.from({ length: loaded.L }, (_, s) => s));
		expect(baselineSites.length).toBeGreaterThan(fx.inputs.target_sites.length);
		// the same run with a wildly different filler for the non-swept baselines is identical
		const other = await runInsilicoSelectionDms(loaded, playbackPredict(fx.outputs.plasticity, 999), {
			siteSubset: fx.inputs.target_sites,
			batchSize: fx.inputs.batch_size
		});
		expect(other).toEqual(got);
	});
});
