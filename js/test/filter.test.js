/**
 * filter.test.js — src/filter.js against hyphaeon/filter.py and cli.py cmd_meme --filter.
 *
 * WHY THIS FILE EXISTS
 *
 *   1. fixtures/filter/scan_hypergeometric_patches.json replay (start/end/k/d exact, p_local 1e-9).
 *   2. run_alignment_filter parity WITHOUT a model: test/data/filter/run_alignment_filter_fake_model
 *      .json was produced by the reference Python with a fake torch model whose output is an exact
 *      float32 function of the amino-acid tokens (5 * mismatches - 0.75 * unknowns); the same
 *      function is the `predict` callback here, so the result dict, cleaned FASTA, per-site raw and
 *      cleaned arrays, and the cmd_meme --filter JSON (the `cliVariant`) are compared directly —
 *      including the '?' consensus difference and the embedded-tree crash of cmd_meme.
 *   3. Python quirks pinned by test/data/filter/python_quirks.json (list-slice masking beyond the
 *      end of a short sequence, pandas mode tie-break).
 *   4. fixtures/e2e/filter_bat_oas1.json, filter_camelid.json and meme_bat_oas1_attribute_filter
 *      .json replayed with a lookup `predict` that returns the per-site LRTs recorded in the meme
 *      e2e fixtures: the scan, consensus, OCI audit, masking, cleaned FASTA hash and the raw
 *      metrics are checked. The cleaned re-score of camelid needs the model (Phase 1b runtime).
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
	scanHypergeometricPatches,
	predictSiteLrts,
	consensusCodons,
	auditPatch,
	maskCodonSpan,
	fastaText,
	runAlignmentFilter
} from '../src/filter.js';
import { numpyPairwiseSum, numpyMeanFloat32 } from '../src/numeric/reduce.js';
import { loadAlignmentAndTree } from '../src/preprocess/assemble.js';
import { parseAlignmentSequences } from '../src/preprocess/parse.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..', '..');
const FIXTURES = join(ROOT, 'fixtures');
const EXAMPLES = join(ROOT, 'examples');
const readJson = (p) => JSON.parse(readFileSync(p, 'utf8'));
const readExample = (name) => readFileSync(join(EXAMPLES, name), 'utf8');
const sha256 = (s) => createHash('sha256').update(s).digest('hex');

/**
 * The fake model of test/data/filter/gen (see the JSON's `inputs.fake_model`): per batch element,
 * the modal AA token among tokens < 20 (lowest on ties), mismatches = tokens < 20 that differ,
 * unknowns = tokens == 20; y = 5 * mismatches - 0.75 * unknowns in float32 (exact).
 */
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
	return Promise.resolve(out);
}

/** A predict that returns recorded per-site LRTs (from a meme e2e fixture) for the sites asked. */
function lookupPredict(siteLrts) {
	return (c, a, meta) => {
		const out = new Float32Array(meta.batch);
		for (let k = 0; k < meta.batch; k++) out[k] = siteLrts[meta.siteIndices[k]];
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

function expectArrayClose(actual, expected, tol, label) {
	expect(actual.length, label).toBe(expected.length);
	for (let i = 0; i < expected.length; i++) expectClose(actual[i], expected[i], tol, `${label}[${i}]`);
}

// ---------------------------------------------------------------------------------------------

describe('scanHypergeometricPatches (fixtures/filter)', () => {
	const cases = readJson(join(FIXTURES, 'filter', 'scan_hypergeometric_patches.json'));
	for (const c of cases) {
		it(c.name, () => {
			const got = scanHypergeometricPatches(c.inputs.site_pvals, {
				alphaSite: c.inputs.alpha_site,
				minK: c.inputs.min_k,
				maxSpan: c.inputs.max_span,
				pLocalThresh: c.inputs.p_local_thresh
			});
			const want = c.outputs.patches;
			expect(got.length).toBe(want.length);
			for (let i = 0; i < want.length; i++) {
				expect([got[i].start, got[i].end, got[i].k, got[i].d]).toEqual([want[i].start, want[i].end, want[i].k, want[i].d]);
				expectClose(got[i].p_local, want[i].p_local, c.tolerance === 'exact' ? 0 : Number(c.tolerance), `${c.name} p_local`);
			}
		});
	}

	it('compares a Float32Array against float32(alpha) as numpy does', () => {
		// 0.05 rounds UP to float32 (0.0500000007...), so a float32 p just above 0.05 is selected.
		const p = new Float32Array(200).fill(0.5);
		const edge = Math.fround(0.05); // > 0.05 in float64
		for (const s of [10, 11, 12]) p[s] = edge;
		expect(scanHypergeometricPatches(p).length).toBe(1);
		expect(scanHypergeometricPatches(Array.from(p)).length).toBe(0);
	});

	it('returns [] below min_k and merges overlapping windows with the min p_local', () => {
		const p = new Array(100).fill(0.5);
		p[10] = p[11] = 0.01;
		expect(scanHypergeometricPatches(p)).toEqual([]);
		p[12] = p[13] = p[30] = 0.01;
		const out = scanHypergeometricPatches(p);
		expect(out.length).toBe(1);
		expect(out[0].start).toBe(10);
		expect(out[0].k).toBe(out[0].end === 30 ? 5 : 4);
	});
});

// ---------------------------------------------------------------------------------------------

describe('numpy reductions', () => {
	it('numpyPairwiseSum matches sequential float64 sum on small inputs and the float32 mean is float32', () => {
		const x = Float32Array.from({ length: 351 }, (_, i) => Math.fround(Math.sin(i) * 3));
		let seq = 0;
		for (const v of x) seq += v;
		expect(Math.abs(numpyPairwiseSum(x, 0, x.length, (v) => v) - seq)).toBeLessThan(1e-12);
		const m = numpyMeanFloat32(x);
		expect(Math.fround(m)).toBe(m);
		expect(Math.abs(m - seq / 351)).toBeLessThan(1e-6);
		expect(numpyMeanFloat32(x, 0, 0)).toBeNaN();
	});
});

// ---------------------------------------------------------------------------------------------

describe('Python quirks (test/data/filter/python_quirks.json)', () => {
	const q = readJson(join(HERE, 'data', 'filter', 'python_quirks.json'));
	it('maskCodonSpan extends a short sequence like a list slice assignment', () => {
		for (const key of ['mask_extends_short_sequence', 'mask_partial_last_codon']) {
			const chars = Array.from(q[key].input);
			for (const s of q[key].sites) maskCodonSpan(chars, s, s);
			expect(chars.join('')).toBe(q[key].output);
		}
	});
	it('consensus tie-break is the lexicographically smallest codon (pandas mode)', () => {
		const seqs = new Map(q.pandas_mode_ties.valid.map((cd, i) => [`t${i}`, cd]));
		expect(consensusCodons(seqs, [...seqs.keys()], 1)[0]).toBe(q.pandas_mode_ties.mode);
	});
	it("consensus drops '-', 'N', '?' and short codons; cliVariant keeps '?'", () => {
		const seqs = new Map([
			['a', '???'],
			['b', '???'],
			['c', 'ATG'],
			['d', 'A-G'],
			['e', 'ANG'],
			['f', 'AT']
		]);
		const taxa = [...seqs.keys()];
		expect(consensusCodons(seqs, taxa, 1)[0]).toBe('ATG');
		expect(consensusCodons(seqs, taxa, 1, { cliVariant: true })[0]).toBe('???');
		expect(consensusCodons(new Map([['a', '---']]), ['a'], 1)[0]).toBe('NNN');
	});
	it('auditPatch: first taxon wins ties, stops count as mismatches, run >= 4 alone is an artifact', () => {
		const seqs = new Map([
			['x', 'ATGATGATGATG'],
			['y', 'AAAAAAAAAAAA'], // 4 mismatches vs consensus ATG
			['z', 'TAGATGATGATG'] // a stop at site 0: counts as a mismatch
		]);
		const cons = ['ATG', 'ATG', 'ATG', 'ATG'];
		const au = auditPatch(seqs, ['x', 'y', 'z'], cons, 0, 3, { minOci: 0.99, minRunLength: 3 });
		expect(au.topTaxon).toBe('y');
		expect(au.topRun).toBe(4);
		expect(au.totalPatchMuts).toBe(5);
		expect(au.isArtifact).toBe(true); // run >= 4 regardless of OCI
		const au2 = auditPatch(seqs, ['x', 'y', 'z'], cons, 0, 2, { minOci: 0.99, minRunLength: 3 });
		expect(au2.topRun).toBe(3);
		expect(au2.isArtifact).toBe(false);
		// ties: same run and count -> first in taxa order
		const tie = auditPatch(seqs, ['z', 'x'], cons, 0, 0, {});
		expect(tie.topTaxon).toBe('z');
	});
});

// ---------------------------------------------------------------------------------------------

describe('runAlignmentFilter against the reference with a fake model', () => {
	const cases = readJson(join(HERE, 'data', 'filter', 'run_alignment_filter_fake_model.json'));
	for (const c of cases) {
		const { alignment, tree } = c.inputs;
		const o = c.outputs;

		it(`${c.name}: run_alignment_filter result, cleaned FASTA, per-site arrays`, async () => {
			const calls = [];
			const predict = (cc, aa, meta) => {
				calls.push({ phase: meta.phase, batch: meta.batch, N: meta.N, d: meta.d, z: meta.z });
				return fakePredict(cc, aa, meta);
			};
			const res = await runAlignmentFilter({ alignmentText: alignment, treeText: tree }, predict, { batchSize: 7 });
			const want = o.result;
			expect(res.raw.loaded.taxa).toEqual(o.taxa);
			for (const k of ['num_taxa', 'num_codons', 'num_patches_detected', 'num_artifacts_masked', 'masked_codons_count', 'suppressed_spurious_sites']) {
				expect(res[k], k).toBe(want[k]);
			}
			for (const side of ['raw_metrics', 'cleaned_metrics']) {
				expect(res[side].sig_sites_p05, side).toBe(want[side].sig_sites_p05);
				expect(res[side].sig_sites_q10, side).toBe(want[side].sig_sites_q10);
				expect(res[side].mean_lrt, `${side}.mean_lrt`).toBe(want[side].mean_lrt);
				expectClose(res[side].cct_p_value, want[side].cct_p_value, 1e-12, `${side}.cct`);
			}
			expect(res.artifacts.length).toBe(want.artifacts.length);
			for (let i = 0; i < want.artifacts.length; i++) {
				const g = res.artifacts[i];
				const w = want.artifacts[i];
				for (const k of Object.keys(w)) {
					if (k === 'p_hypergeom' || k === 'outlier_contamination_index') expectClose(g[k], w[k], 1e-12, k);
					else expect(g[k], k).toEqual(w[k]);
				}
			}
			expect(res.output_alignment).toBeNull();
			expect(res.audit_csv).toBeNull();
			// patches from the raw p-values
			expect(res.patches.map((p) => [p.start, p.end, p.k, p.d])).toEqual(o.patches.map((p) => [p.start, p.end, p.k, p.d]));
			expectArrayClose(res.patches.map((p) => p.p_local), o.patches.map((p) => p.p_local), 1e-12, 'p_local');
			// per-site arrays: LRTs exact (the fake is exact in float32), p/q at 1e-12
			expect(Array.from(res.raw.lrts)).toEqual(o.lrts_raw);
			expectArrayClose(res.raw.pvals, o.pvals_raw, 1e-12, 'pvals_raw');
			expectArrayClose(res.raw.qvals, o.qvals_raw, 1e-12, 'qvals_raw');
			expect(Array.from(res.raw.loaded.invariable)).toEqual(o.invariable_raw);
			expect(res.cleaned).not.toBeNull();
			expect(res.cleaned.fastaText).toBe(o.cleaned_fasta);
			expect(Array.from(res.cleaned.lrts)).toEqual(o.lrts_clean);
			expectArrayClose(res.cleaned.pvals, o.pvals_clean, 1e-12, 'pvals_clean');
			expectArrayClose(res.cleaned.qvals, o.qvals_clean, 1e-12, 'qvals_clean');
			expect(Array.from(res.cleaned.loaded.invariable)).toEqual(o.invariable_clean);
			// masked ranges match the artifacts; masked taxa are NNN in the cleaned sequences
			for (const art of want.artifacts) {
				expect(res.masked_codon_ranges_1idx_by_taxon[art.outlier_taxon]).toContainEqual([art.patch_start_1idx, art.patch_end_1idx]);
				const seq = res.cleaned.sequences.get(art.outlier_taxon);
				expect(seq.slice((art.patch_start_1idx - 1) * 3, art.patch_end_1idx * 3)).toBe('N'.repeat(art.span_codons * 3));
			}
			// masked codons are token 64 / 20 in the cleaned tensors
			const cl = res.cleaned.loaded;
			for (const art of want.artifacts) {
				const ti = cl.taxa.indexOf(art.outlier_taxon);
				for (let s = art.patch_start_1idx - 1; s < art.patch_end_1idx; s++) {
					expect(cl.c[s * cl.N + ti]).toBe(64);
					expect(cl.a[s * cl.N + ti]).toBe(20);
				}
			}
			// sites: final = cleaned, raw_* = baseline
			expect(res.sites.length).toBe(want.num_codons);
			expect(res.sites[0].site).toBe(1);
			expect(res.sites.map((s) => s.hyphaeon_lrt)).toEqual(o.lrts_clean);
			expect(res.sites.map((s) => s.raw_lrt)).toEqual(o.lrts_raw);
			expect(res.sites.map((s) => (s.is_invariable ? 1 : 0))).toEqual(o.invariable_clean);
			// two phases through the callback, batched by 7 variable sites, cleaned phase on the cleaned d/z
			const phases = [...new Set(calls.map((x) => x.phase))];
			expect(phases).toEqual(['baseline', 'cleaned']);
			expect(calls.every((x) => x.batch <= 7 && x.N === want.num_taxa)).toBe(true);
			const nVar = o.invariable_raw.filter((v) => v === 0).length;
			expect(calls.filter((x) => x.phase === 'baseline').reduce((n, x) => n + x.batch, 0)).toBe(nVar);
			expect(calls.find((x) => x.phase === 'cleaned').d).toBe(cl.d);
			expect(res.notices.cliVariant).toBe(false);
			expect(res.notices.treeSource).toBe(tree === null ? 'alignment' : 'tree');
			expect(res.notices.baseLrtsSupplied).toBe(false);
		});

		it(`${c.name}: baseLrts / loaded inputs skip the baseline pass and give the same result`, async () => {
			const loaded = loadAlignmentAndTree(alignment, tree);
			const phases = [];
			const res = await runAlignmentFilter(
				{ alignmentText: alignment, treeText: tree, loaded, baseLrts: o.lrts_raw },
				(cc, aa, meta) => (phases.push(meta.phase), fakePredict(cc, aa, meta))
			);
			expect(phases.every((p) => p === 'cleaned')).toBe(true);
			expect(res.notices.baseLrtsSupplied).toBe(true);
			expect(res.raw.loaded).toBe(loaded);
			expect(res.artifacts).toEqual(
				(await runAlignmentFilter({ alignmentText: alignment, treeText: tree }, fakePredict)).artifacts
			);
			expect(res.cleaned.fastaText).toBe(o.cleaned_fasta);
			await expect(
				runAlignmentFilter({ alignmentText: alignment, treeText: tree, baseLrts: [1, 2, 3] }, fakePredict)
			).rejects.toThrow(/baseLrts has 3 entries/);
		});

		it(`${c.name}: cmd_meme --filter (cliVariant) JSON`, async () => {
			const input = { alignmentText: alignment, treeText: tree };
			if (o.cmd_meme === null) {
				// cli.py:192 hands the cleaned FASTA (no embedded tree) and args.tree=None to the loader.
				expect(o.cmd_meme_error).toMatch(/No tree specified/);
				await expect(runAlignmentFilter(input, fakePredict, { cliVariant: true })).rejects.toThrow(/No tree specified/);
				return;
			}
			const res = await runAlignmentFilter(input, fakePredict, { cliVariant: true, batchSize: 7 });
			const meme = o.cmd_meme;
			expect(res.notices.cliVariant).toBe(true);
			expect(res.num_taxa).toBe(meme.taxa_count);
			expect(res.num_codons).toBe(meme.codon_count);
			expect(res.artifacts_masked.length).toBe(meme.artifacts_masked.length);
			for (let i = 0; i < meme.artifacts_masked.length; i++) {
				const g = res.artifacts_masked[i];
				const w = meme.artifacts_masked[i];
				expect([g.start, g.end, g.span, g.outlier_taxon, g.consecutive_mismatches]).toEqual([w.start, w.end, w.span, w.outlier_taxon, w.consecutive_mismatches]);
				expectClose(g.oci, w.oci, 1e-12, 'oci');
			}
			// p/q are float32 in cmd_meme
			expect(res.sites.length).toBe(meme.sites.length);
			expect(res.raw.pvals).toBeInstanceOf(Float32Array);
			for (let i = 0; i < meme.sites.length; i++) {
				const g = res.sites[i];
				const w = meme.sites[i];
				expect(g.site).toBe(w.site);
				expect(g.hyphaeon_lrt, `lrt ${i}`).toBe(w.hyphaeon_lrt);
				expectClose(g.p_value, w.p_value, 1e-9, `p ${i}`);
				expectClose(g.q_value, w.q_value, 1e-9, `q ${i}`);
				expect(g.is_invariable, `inv ${i}`).toBe(w.is_invariable);
			}
			if (meme.artifacts_masked.length > 0) {
				// the cleaned FASTA holds the matched taxa only, in taxa order, and the re-score used the
				// baseline d/z (cli.py:195-197)
				expect([...res.cleaned.sequences.keys()]).toEqual(res.raw.loaded.taxa);
				expect(res.notices.treeSource).toBe('tree');
			}
			// the two copies of the OCI screen differ where the consensus rule differs
			if (c.name === 'question_marks_cli_difference') {
				const fn = await runAlignmentFilter(input, fakePredict);
				expect(fn.artifacts[0].consecutive_mismatches).toBe(8);
				expect(res.artifacts_masked[0].consecutive_mismatches).toBe(5);
			}
		});
	}

	it('predictSiteLrts clamps at 0, stores float32, honours siteIndices and rejects a short return', async () => {
		const c = readJson(join(HERE, 'data', 'filter', 'run_alignment_filter_fake_model.json'))[0];
		const loaded = loadAlignmentAndTree(c.inputs.alignment, c.inputs.tree);
		const lrts = await predictSiteLrts(loaded, () => new Float64Array(3).fill(-2.5), { siteIndices: [0, 5, 9], batchSize: 3 });
		expect(lrts.length).toBe(loaded.L);
		expect(Array.from(lrts)).toEqual(new Array(loaded.L).fill(0));
		const lrts2 = await predictSiteLrts(loaded, () => [0.1], { siteIndices: [4], batchSize: 1 });
		expect(lrts2[4]).toBe(Math.fround(0.1));
		await expect(predictSiteLrts(loaded, () => [1, 2], { siteIndices: [1, 2, 3] })).rejects.toThrow(/returned 2 values/);
	});

	it('runAlignmentFilter with zero patches leaves cleaned null and cleaned_metrics == raw_metrics', async () => {
		const c = readJson(join(HERE, 'data', 'filter', 'run_alignment_filter_fake_model.json'))[0];
		const res = await runAlignmentFilter({ alignmentText: c.inputs.alignment, treeText: c.inputs.tree }, (cc, aa, meta) => new Float32Array(meta.batch));
		expect(res.num_patches_detected).toBe(0);
		expect(res.cleaned).toBeNull();
		expect(res.cleaned_metrics).toEqual(res.raw_metrics);
		expect(res.sites.every((s) => s.hyphaeon_lrt === 0 && s.p_value === 2 / 3)).toBe(true);
	});
});

// ---------------------------------------------------------------------------------------------

describe('e2e fixture replay with recorded per-site LRTs', () => {
	const memeLrts = (name) => readJson(join(FIXTURES, 'e2e', name))[0].outputs.sites.map((s) => s.hyphaeon_lrt);

	it('filter_bat_oas1: one candidate patch, no artifact, raw metrics, cleaned FASTA hash', async () => {
		const fx = readJson(join(FIXTURES, 'e2e', 'filter_bat_oas1.json'))[0];
		const lrts = memeLrts('meme_bat_oas1.json');
		const alignmentText = readExample(fx.inputs.alignment);
		const treeText = readExample(fx.inputs.tree);
		const res = await runAlignmentFilter({ alignmentText, treeText }, lookupPredict(lrts));
		const want = fx.outputs.result;
		expect(res.num_taxa).toBe(want.num_taxa);
		expect(res.num_codons).toBe(want.num_codons);
		expect(res.num_patches_detected).toBe(want.num_patches_detected);
		expect(res.num_artifacts_masked).toBe(0);
		expect(res.artifacts).toEqual([]);
		expect(res.masked_codon_ranges_1idx_by_taxon).toEqual({});
		for (const side of ['raw_metrics', 'cleaned_metrics']) {
			expect(res[side].sig_sites_p05).toBe(want[side].sig_sites_p05);
			expect(res[side].sig_sites_q10).toBe(want[side].sig_sites_q10);
			expectClose(res[side].mean_lrt, want[side].mean_lrt, 1e-5, `${side}.mean_lrt`);
			expectClose(res[side].cct_p_value, want[side].cct_p_value, 1e-5, `${side}.cct`);
		}
		// 0 artifacts: the reference still writes every parsed sequence unchanged
		expect(sha256(fastaText(parseAlignmentSequences(alignmentText)))).toBe(fx.outputs.cleaned_alignment_sha256);
	});

	it('filter_camelid: artifact record, masked range and cleaned FASTA hash are exact from the recorded baseline', async () => {
		const fx = readJson(join(FIXTURES, 'e2e', 'filter_camelid.json'))[0];
		const meme = readJson(join(FIXTURES, 'e2e', 'meme_camelid.json'))[0].outputs;
		const lrts = meme.sites.map((s) => s.hyphaeon_lrt);
		const alignmentText = readExample(fx.inputs.alignment);
		const treeText = readExample(fx.inputs.tree);
		const res = await runAlignmentFilter({ alignmentText, treeText }, lookupPredict(lrts));
		const want = fx.outputs.result;
		expect(res.raw.loaded.notices.branchLengthsMissing).toBe(true); // HyPhy ran in the reference
		expect(res.raw.loaded.invariable.length).toBe(meme.sites.length);
		expect(Array.from(res.raw.loaded.invariable, (v) => v === 1)).toEqual(meme.sites.map((s) => s.is_invariable));
		expect(res.num_taxa).toBe(want.num_taxa);
		expect(res.num_codons).toBe(want.num_codons);
		expect(res.num_patches_detected).toBe(want.num_patches_detected);
		expect(res.num_artifacts_masked).toBe(want.num_artifacts_masked);
		expect(res.masked_codons_count).toBe(want.masked_codons_count);
		expect(res.artifacts.length).toBe(1);
		const g = res.artifacts[0];
		const w = want.artifacts[0];
		for (const k of Object.keys(w)) {
			if (k === 'p_hypergeom' || k === 'outlier_contamination_index') expectClose(g[k], w[k], 1e-9, k);
			else if (k === 'mean_raw_patch_lrt') expectClose(g[k], w[k], 1e-5, k);
			else expect(g[k], k).toEqual(w[k]);
		}
		expect(res.masked_codon_ranges_1idx_by_taxon).toEqual(fx.outputs.masked_codon_ranges_1idx_by_taxon);
		expect(sha256(res.cleaned.fastaText)).toBe(fx.outputs.cleaned_alignment_sha256);
		expect(res.raw_metrics.sig_sites_p05).toBe(want.raw_metrics.sig_sites_p05);
		expect(res.raw_metrics.sig_sites_q10).toBe(want.raw_metrics.sig_sites_q10);
		expectClose(res.raw_metrics.mean_lrt, want.raw_metrics.mean_lrt, 1e-5, 'raw mean_lrt');
		expectClose(res.raw_metrics.cct_p_value, want.raw_metrics.cct_p_value, 1e-5, 'raw cct');
		// cleaned_metrics need the model's re-score of the masked alignment (Phase 1b runtime);
		// here the lookup returned the baseline LRTs again, so only the shape is checked.
		expect(res.cleaned.lrts.length).toBe(want.num_codons);
	});

	it('meme_bat_oas1_attribute_filter: cmd_meme --filter with float32 p/q, no artifact', async () => {
		const fx = readJson(join(FIXTURES, 'e2e', 'meme_bat_oas1_attribute_filter.json'))[0];
		const out = fx.outputs;
		const lrts = out.sites.map((s) => s.hyphaeon_lrt);
		const res = await runAlignmentFilter(
			{ alignmentText: readExample(out.alignment), treeText: readExample(out.tree), baseLrts: lrts },
			lookupPredict(lrts),
			{ cliVariant: true }
		);
		expect(res.artifacts_masked).toEqual(out.artifacts_masked);
		expect(res.num_patches_detected).toBe(1);
		expect(res.sites.length).toBe(out.sites.length);
		for (let i = 0; i < out.sites.length; i++) {
			expect(res.sites[i].hyphaeon_lrt).toBe(out.sites[i].hyphaeon_lrt);
			expectClose(res.sites[i].p_value, out.sites[i].p_value, 1e-9, `p ${i}`);
			expectClose(res.sites[i].q_value, out.sites[i].q_value, 1e-9, `q ${i}`);
			expect(res.sites[i].is_invariable).toBe(out.sites[i].is_invariable);
		}
	});
});
