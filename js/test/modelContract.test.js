/**
 * WHY THIS FILE EXISTS
 *
 * Ported verbatim from datamonkey3 (main@fac1330) src/test/axomeme-model-contract.test.js. Cases
 * unchanged, with ONE BLOCK APPENDED at the end for `OUTPUT_SPEC_V1`, the three-output export that
 * did not exist in DataMonkey 3.
 *
 * These are not shape-checking-the-shape-checker busywork: every case is a mistake a JS
 * preprocessing port actually makes, one that yields a tensor of exactly the right dtype and dims
 * meaning something the model was never trained on. The transcribed constants are pinned too,
 * because a transcription error in the codon vocabulary is invisible until it meets real fixtures,
 * which is a much later and much more expensive place to find it.
 *
 * WHAT WAS CHANGED, AND ONLY THIS: the import paths, repointed from
 * `../lib/services/axomeme/…` at DataMonkey 3's layout to `../src/preprocess/…` at this package's.
 * Not one case, expectation or comment was edited, so a failure here is a failure of the port and
 * never of a local adjustment to the test. Comments that cite DataMonkey 3 paths
 * (`src/lib/utils/treeSanitation.js`, `scripts/axomeme/verify_preprocessing.py`) are left as
 * written: they are the record of where a measurement was taken, and rewriting them would break the
 * trail without moving the evidence.
 *
 * THE FILE WAS RENAMED from `axomeme-<x>.test.js` to `<x>.test.js`. "AxoMEME" was DataMonkey 3's
 * name for the pillar this model serves; in this repository the package IS HyphAeon, so the prefix
 * distinguished nothing and the module under test is what the name should say.
 */

/**
 * Tests for the AxoMEME 2.0 ONNX input contract.
 *
 * These are not shape-checking-the-shape-checker busywork. Every case below is a mistake a JS
 * preprocessing port actually makes — one that produces a tensor of exactly the right dtype and
 * exactly the right dimensions, and means something the model was never trained on. Those are the
 * errors that cost days, because nothing crashes: the graph runs, five numbers come out per site,
 * and they are wrong in a way that looks like a bad model rather than a bad tensor.
 *
 * The constants themselves are pinned too. They are transcribed from the ML team's handoff scripts,
 * and a transcription error in, say, the codon vocabulary order is invisible until it is compared
 * against real fixtures — which is a much later and much more expensive place to find it.
 */
import { describe, it, expect } from 'vitest';
import {
	CODON_ORDER,
	CODON_GAP,
	CODON_UNKNOWN,
	NUM_CODON_TOKENS,
	AA_LIST,
	AA_GAP,
	AA_UNKNOWN,
	CODON_VALID_BELOW,
	AA_VALID_BELOW,
	MAX_SPECIES_DEFAULT,
	WINDOW_SIZE_DEFAULT,
	MDS_COMPONENTS,
	INPUT_SPEC,
	INPUT_NAMES,
	OUTPUT_SPEC,
	OUTPUT_SPEC_V1,
	OUTPUT_NAMES_V1,
	VERIFIED_MODEL_SHA256,
	validateInputBundle
} from '../src/preprocess/modelContract.js';

const BATCH = 2;
const SPECIES = 4; // index 3 is padded
const WIN = 1;

/** A bundle that satisfies the contract. Each test breaks exactly one thing about it. */
function validBundle() {
	const d = [
		[0, 0.1, 0.2, 0],
		[0.1, 0, 0.3, 0],
		[0.2, 0.3, 0, 0],
		[0, 0, 0, 0]
	];
	const distOne = d.flat();
	return {
		msa_codons: {
			data: new BigInt64Array([0n, 5n, 63n, 65n, 1n, 2n, 3n, 65n]),
			dims: [BATCH, SPECIES, WIN]
		},
		msa_aas: {
			data: new BigInt64Array([0n, 4n, 20n, 22n, 1n, 2n, 3n, 22n]),
			dims: [BATCH, SPECIES, WIN]
		},
		dist_matrix: {
			data: new Float32Array([...distOne, ...distOne]),
			dims: [BATCH, SPECIES, SPECIES]
		},
		mds_coords: {
			data: new Float32Array(BATCH * SPECIES * MDS_COMPONENTS),
			dims: [BATCH, SPECIES, MDS_COMPONENTS]
		}
		// TRUE = padded. Species 3 only.
	};
}

const check = (b) => validateInputBundle(b, { batch: BATCH, numSpecies: SPECIES, windowSize: WIN });

describe('the transcribed constants', () => {
	it('orders codons TCAG, not alphabetically', () => {
		// The single most damaging transcription error available here: an ACGT vocabulary is a valid
		// permutation of the same 64 tokens and is wrong at every site of every alignment.
		expect(CODON_ORDER).toBe('TCAG');
		const codons = [...CODON_ORDER].flatMap((a) =>
			[...CODON_ORDER].flatMap((b) => [...CODON_ORDER].map((c) => a + b + c))
		);
		expect(codons).toHaveLength(64);
		expect(codons[0]).toBe('TTT');
		expect(codons[63]).toBe('GGG');
		// If someone "fixes" the order to ACGT this is the assertion that objects.
		// Met. Under TCAG: A=2, T=0, G=3 -> 2*16 + 0*4 + 3 = 35. Under ACGT it would be 14, so this
		// single number distinguishes the two orderings.
		expect(codons.indexOf('ATG')).toBe(35);
	});

	it('keeps gap and unknown distinct, and different between the two streams', () => {
		expect(CODON_GAP).toBe(64);
		expect(CODON_UNKNOWN).toBe(65);
		expect(NUM_CODON_TOKENS).toBe(66);
		expect(AA_GAP).toBe(AA_LIST.indexOf('-'));
		expect(AA_UNKNOWN).toBe(AA_LIST.indexOf('?'));
		expect(AA_GAP).toBe(21);
		expect(AA_UNKNOWN).toBe(22);
		expect(CODON_GAP).not.toBe(AA_GAP);
	});

	it('sets the validity thresholds so that a GAP is not a valid observation', () => {
		// forward() gates on (c < 64) & (a < 21). A gap is 64 / 21, so it fails both — deliberately.
		expect(CODON_VALID_BELOW).toBe(CODON_GAP);
		expect(AA_VALID_BELOW).toBe(AA_GAP);
		expect(CODON_GAP < CODON_VALID_BELOW).toBe(false);
		expect(AA_GAP < AA_VALID_BELOW).toBe(false);
	});

	it('pins the checkpoint defaults', () => {
		expect(MAX_SPECIES_DEFAULT).toBe(512);
		expect(MDS_COMPONENTS).toBe(4);
		// window_size 1 means the central index is 0 and every window IS the site. An even window
		// would put the scored codon off-centre.
		expect(WINDOW_SIZE_DEFAULT).toBe(1);
		expect(Math.floor(WINDOW_SIZE_DEFAULT / 2)).toBe(0);
	});

	it('lists the four inputs in forward() order and marks the site-invariant ones', () => {
		expect(INPUT_NAMES).toEqual(['msa_codons', 'msa_aas', 'dist_matrix', 'mds_coords']);
		// v1-viral dropped `padding_mask`; DM3 never padded anyway, so nothing was lost but the
		// tensor. These two are computed once per alignment and expanded across sites, which is what
		// makes batching every site into a single graph run cheap.
		const invariant = INPUT_SPEC.filter((s) => s.siteInvariant).map((s) => s.name);
		expect(invariant).toEqual(['dist_matrix', 'mds_coords']);
	});

	it('names the single output the graph actually exposes', () => {
		// Read from InferenceSession.outputNames on the real artifact, not from its README. This was an
		// open question — the shipped driver reaches for the TRAIN branch to get raw ordinal logits —
		// and the answer is that the export took the EVAL branch, so `lrt` arrives already decoded.
		expect(OUTPUT_SPEC.map((o) => o.name)).toEqual(['lrt']);
		expect(OUTPUT_SPEC[0].note).toMatch(/already applied in-graph/);
	});

	it('exposes no rate heads to decode', () => {
		// The test this replaces asserted that alpha / beta_neg / beta_pos were log1p and needed
		// expm1. v1-viral does not export them, so the thing to pin is that nothing downstream can
		// find a rate head and start decoding one that is not there.
		for (const name of ['alpha', 'beta_neg', 'beta_pos', 'p_neg']) {
			expect(
				OUTPUT_SPEC.find((o) => o.name === name),
				name
			).toBeUndefined();
		}
	});

	it('pins the artifact the contract was verified against', () => {
		// A different export is not necessarily wrong, but the eval-mode conclusion was read off THIS
		// graph, so swapping the model without revisiting this file is a mistake worth failing on.
		expect(VERIFIED_MODEL_SHA256).toMatch(/^[0-9a-f]{64}$/);
	});

	it('cannot be mutated by a caller', () => {
		expect(() => {
			INPUT_SPEC.push({ name: 'nope' });
		}).toThrow();
	});
});

describe('validateInputBundle', () => {
	it('accepts a well-formed bundle', () => {
		const r = check(validBundle());
		expect(r.errors).toEqual([]);
		expect(r.ok).toBe(true);
	});

	it('reports a missing tensor by name', () => {
		const b = validBundle();
		delete b.mds_coords;
		expect(check(b).errors).toContain('mds_coords: missing');
	});

	it('catches a transposed distance matrix shape', () => {
		const b = validBundle();
		b.dist_matrix.dims = [BATCH, SPECIES, MDS_COMPONENTS + 1];
		expect(check(b).ok).toBe(false);
	});

	it('catches dims that are right but data that is short', () => {
		// The shape says one thing and the buffer says another — onnxruntime will happily read past
		// the end of the meaningful data or throw something opaque.
		const b = validBundle();
		b.mds_coords.data = new Float32Array(4);
		expect(check(b).errors.join(' ')).toMatch(/mds_coords: 4 elements/);
	});

	it('catches an out-of-vocabulary token', () => {
		const b = validBundle();
		b.msa_codons.data = new BigInt64Array([0n, 5n, 63n, 66n, 1n, 2n, 3n, 65n]);
		expect(check(b).errors.join(' ')).toMatch(/msa_codons\[3\] = 66/);
	});

	it('rejects a bundle carrying a tensor the graph does not accept', () => {
		// This replaces the flipped-padding-mask test. That check existed because an inverted mask is
		// well-formed in dtype, dims and element count while meaning the opposite — the model would
		// mask out every real taxon. v1-viral has no mask input, so that failure mode is gone with
		// it. What remains worth catching is a bundle built for the OLD graph reaching the new one.
		const b = validBundle();
		b.padding_mask = { data: new Uint8Array([0, 0, 0, 0, 0, 0, 0, 0]), dims: [2, 4] };
		const r = check(b);
		expect(r.ok, 'a stale padding_mask was accepted').toBe(false);
	});

	it('catches a negative patristic distance, which is a real DM3 tree and a Python crash', () => {
		// 5% of real DM3 trees carry a branch length <= -0.1; the Python inference path throws on
		// them at predict_regression_nexus.py:955 rather than degrading. See treeSanitation.js.
		const b = validBundle();
		b.dist_matrix.data[1] = -0.4;
		const r = check(b);
		expect(r.ok).toBe(false);
		expect(r.errors.join(' ')).toMatch(/negative patristic distance/);
	});

	it('catches NaN before it reaches the graph', () => {
		const b = validBundle();
		b.dist_matrix.data[2] = NaN;
		expect(check(b).errors.join(' ')).toMatch(/dist_matrix\[2\] is NaN/);
	});

	it('catches a nonzero self-distance, which means the matrix is not a distance matrix', () => {
		const b = validBundle();
		b.dist_matrix.data[0] = 0.5; // d(0,0)
		expect(check(b).errors.join(' ')).toMatch(/self-distance/);
	});

	it('reports every independent problem, not just the first', () => {
		// A port under development usually has several at once; stopping at the first costs a whole
		// round trip per error.
		const b = validBundle();
		delete b.msa_aas;
		b.mds_coords.dims = [BATCH, SPECIES, 3];
		expect(check(b).errors.length).toBeGreaterThanOrEqual(2);
	});
});

/**
 * APPENDED FOR @veg/hyphaeon-js — not part of the DataMonkey 3 original.
 *
 * `OUTPUT_SPEC_V1` describes the THREE-output graph this repository's `export-onnx` produces, beside
 * the single-output `OUTPUT_SPEC` of the shipped viral artifact. Two shapes of failure are worth
 * pinning. The first is an export that quietly drops a head: `mean_root_attns` and `root_repr` are
 * both float32 and both `[batch, something]`, so a two-output graph does not crash anything — it
 * removes the epistasis and busted pillars, and the only place that is visible is here. The second
 * is the old spec being edited to "update" it, which would break DataMonkey 3's live path; the two
 * specs describe two real artifacts and both have to keep working.
 */
describe('OUTPUT_SPEC_V1, the three-output export', () => {
	it('names the three outputs in graph order', () => {
		expect(OUTPUT_NAMES_V1).toEqual(['lrt', 'mean_root_attns', 'root_repr']);
	});

	it('carries the shapes each pillar reads', () => {
		// lrt [batch]; mean_root_attns [batch, num_species] for epistasis and phenotype attention;
		// root_repr [batch, embed_dim] as the input to busted_head.onnx.
		const dims = Object.fromEntries(OUTPUT_SPEC_V1.map((o) => [o.name, o.dims]));
		expect(dims.lrt).toEqual(['batch']);
		expect(dims.mean_root_attns).toEqual(['batch', 'num_species']);
		expect(dims.root_repr).toEqual(['batch', 'embed_dim']);
		for (const o of OUTPUT_SPEC_V1) expect(o.dtype, o.name).toBe('float32');
	});

	it('keeps lrt first and identical in meaning to the single-output export', () => {
		// The runtime picks a spec by the manifest, so the two must agree about the head they share:
		// `lrt` is already ordinal-decoded in-graph in both, and nothing downstream decodes it twice.
		expect(OUTPUT_SPEC_V1[0].name).toBe(OUTPUT_SPEC[0].name);
		expect(OUTPUT_SPEC_V1[0].note).toMatch(/already applied in-graph/);
	});

	it('leaves the single-output spec alone', () => {
		// The shipped viral artifact still exports one head. Replacing OUTPUT_SPEC rather than adding
		// beside it would mean the runtime expecting three tensors from a graph that has one.
		expect(OUTPUT_SPEC.map((o) => o.name)).toEqual(['lrt']);
	});

	it('cannot be mutated by a caller', () => {
		expect(() => {
			OUTPUT_SPEC_V1.push({ name: 'nope' });
		}).toThrow();
		expect(() => {
			OUTPUT_SPEC_V1[0].name = 'nope';
		}).toThrow();
	});
});
