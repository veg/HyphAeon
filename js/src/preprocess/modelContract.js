/**
 * WHY THIS FILE EXISTS
 *
 * Ported verbatim from datamonkey3 (main@fac1330) src/lib/services/axomeme/modelContract.js.
 * Function bodies unchanged, existing exports untouched; it imports nothing. ONE ADDITION, at the
 * bottom of the file: `OUTPUT_SPEC_V1`, describing the three-output graph this repository's
 * `export-onnx` produces. Nothing else was edited.
 *
 * It is the description of, and the validator for, the tensors that cross the boundary this package
 * does not cross. `@veg/hyphaeon-js` never loads a model — that is `hyphaeon-app`'s `runtime/` — so
 * this file is how a pure function can still say "the bundle you just built is not what the graph
 * accepts" before anything expensive happens. The constants pin `PhyloAxialTransformer.forward`
 * (`hyphaeon/model.py:481-583`) and `model_config.json`.
 *
 * The long note below was written against the AxoMEME 2.0 handoff scripts. Read tokenizer.js's
 * header alongside it: the vocabulary it pins is the 2.0 training vocabulary, and `hyphaeon/
 * dataset.py` at v1.0.0 numbers codons differently again. That is an open Phase 0 question, and this
 * file is a description of a contract rather than the arbiter of it.
 */

/**
 * modelContract.js — what the AxoMEME 2.0 ONNX graph accepts and returns.
 *
 * WHY THIS FILE EXISTS, AND WHY IT IS CODE RATHER THAN A README.
 *
 * The exported graph is the MODEL ONLY. It takes five already-computed tensors and returns five;
 * the entire preprocessing pipeline that produces those tensors — newick parse, patristic
 * distances, taxon subsampling, codon/AA tokenisation, and the MDS eigendecomposition — is NOT in
 * the graph and has to be rebuilt in JS. `mds_coords` is a graph INPUT, not something the graph
 * computes. (`torch.linalg.eigh` has no ONNX lowering, so this is forced rather than chosen:
 * exporting the identical module with eigh removed succeeds and with it present fails.)
 *
 * That means every number below is a place a JS port can be silently, plausibly wrong — producing
 * a well-formed tensor of the right shape that means something different from what the model was
 * trained on. Several of them are genuinely surprising, so they are constants here with citations
 * rather than assumptions in someone's head:
 *
 *   - CODON ORDER IS "TCAG", NOT ALPHABETICAL. The vocabulary is built as
 *     `[a+b+c for a in "TCAG" for b in "TCAG" for c in "TCAG"]`, the standard genetic-code table
 *     order. An ACGT-ordered vocabulary is a perfectly valid-looking permutation of the same 64
 *     tokens and would be wrong at every site. THE HANDOFF'S OWN INFERENCE DRIVER GETS THIS WRONG —
 *     see the next paragraph before "correcting" anything here to match it.
 *
 * THE DRIVER'S TOKENIZER DISAGREES WITH THE TRAINING TOKENIZER. This is measured, not suspected.
 *
 *   train_transformer_selection.py:74-85  (TRAINING)  64 codons, TCAG order, '-'->64, '?'->65
 *   predict_regression_nexus.py:49-80     (INFERENCE) 60 codons, ALPHABETICAL, everything else ->65
 *
 * The driver defines its own `CODON_LIST` / `CODON_TO_IDX` at lines 49-55, then redefines
 * `get_codon_token` at line 74 — but never redefines `CODON_TO_IDX`, so the winning function looks up
 * the 60-codon alphabetical map. It imports only `PhyloAxialTransformer`,
 * `compute_mds_coordinates` and `decode_soft_ordinal_lrt` from the training module (line 28), so the
 * training tokenizer is never in scope. Result, checked over all 64 codons:
 *
 *   63 of 64 codons receive a DIFFERENT token at inference than at training
 *     ATG: train 35 -> inference 14      TTT: train  0 -> inference 59
 *     AAA: train 42 -> inference  0      GGG: train 63 -> inference 42
 *   TTA (Leucine) and all three stop codons are ABSENT from the driver's list entirely, so they
 *   collapse to 65 "unknown" — real leucine data is discarded rather than mistokenised.
 *   The amino-acid stream is nearly intact: 3 of 64 wrong, the stops, because the driver's
 *   GENETIC_CODE writes them as '_' while AA_LIST contains '*', so they land on 22 instead of 20.
 *
 * DM3 USES THE TRAINING VOCABULARY, which is what this file pins. A model trained on TCAG-64 has to
 * be served TCAG-64; there is no reading under which the driver's map is the right one for these
 * weights. Reproducing the driver here would reproduce a bug.
 *
 * Note what this does NOT invalidate: the ML team's ONNX-vs-PyTorch parity result. That test feeds
 * the same tensors to both sides, so it proves the graph was exported faithfully and is unaffected
 * by which tokenizer produced those tensors. The bug is invisible to it by construction — which is
 * exactly why it survived to here.
 *   - GAP AND UNKNOWN ARE DIFFERENT TOKENS, and they differ between the two streams: codons use
 *     64/65, amino acids use 21/22. They are not interchangeable — `forward()` gates on
 *     `(c_cent < 64) & (a_cent < 21)`, so a gap counts as invalid at the central site.
 *   - MDS IS COMPUTED ON THE PADDED MATRIX. `compute_mds_coordinates` is handed the full
 *     [max_species, max_species] tensor, zeros included, so the padded region participates in the
 *     double-centring and the coordinates depend on max_species — not just on the real taxa. A port
 *     that runs MDS on the N real species and pads afterwards gets different numbers.
 *   - THE TWO PHYLO TENSORS ARE SITE-INVARIANT. dist_matrix and mds_coords are
 *     computed once and `expand`ed across sites. They are per-alignment, not per-site, which is what
 *     makes batching every site into one graph run cheap.
 *
 * Sources, all in the ML team's handoff (axomeme-2.0-viral-handoff/scripts/):
 *   predict_regression_nexus.py:161-187   compute_mds_coordinates
 *   predict_regression_nexus.py:1213-1229 tensor allocation, pad tokens, MDS on the padded matrix
 *   predict_regression_nexus.py:1271-1275 the site-invariant expands
 *   predict_regression_nexus.py:981-982   window_size / max_species defaults
 *   train_transformer_selection.py:73-97  codon + amino acid vocabularies
 *   forward(msa_codons, msa_aas, dist_matrix, mds_coords)  — four inputs in v1-viral
 *
 * This module is deliberately a LEAF: it imports nothing, computes nothing, and holds no model. It
 * describes and it validates. Nothing here should ever pull in onnxruntime-web — the runtime is
 * loaded on the AxoMEME path only, and the reachability guard in
 * src/test/meme-hit-likelihood.test.js exists to keep that true.
 */

/** Codon vocabulary order. NOT alphabetical — see the header. */
export const CODON_ORDER = 'TCAG';

/** 64 sense+stop codons occupy 0..63; these two are the sentinels. */
export const CODON_GAP = 64;
export const CODON_UNKNOWN = 65;

/** `num_tokens=66` is passed to the model constructor: 64 codons + gap + unknown. */
export const NUM_CODON_TOKENS = 66;

/** Amino acid vocabulary, index = position in this string. */
export const AA_LIST = 'ACDEFGHIKLMNPQRSTVWY*-?';
export const AA_GAP = 21; // AA_LIST.indexOf('-')
export const AA_UNKNOWN = 22; // AA_LIST.indexOf('?')

/**
 * `forward()` treats a species as present at the central site only if
 * `(codon < 64) & (aa < 21) & !padded`. So a gap is NOT a valid observation, and these are the
 * thresholds — not `<= 64` / `<= 21`.
 */
export const CODON_VALID_BELOW = 64;
export const AA_VALID_BELOW = 21;

/** `--max_species` default. Tensors are padded to exactly this many rows. */
export const MAX_SPECIES_DEFAULT = 512;

/**
 * `--window_size` default, and the value this checkpoint was fine-tuned at. The model reads the
 * CENTRAL column (`window_size // 2`), so an even window would shift which codon is scored.
 */
export const WINDOW_SIZE_DEFAULT = 1;

/** `compute_mds_coordinates(..., n_components=4)`. */
export const MDS_COMPONENTS = 4;

/**
 * `dims` uses the symbolic names the export declares dynamic (batch, num_species); everything else
 * is fixed by the checkpoint. int64 matters: onnxruntime-web wants a BigInt64Array for these, and
 * passing Float32Array of the same values is a type error at session.run, not a silent coercion.
 */
export const INPUT_SPEC = Object.freeze([
	Object.freeze({
		name: 'msa_codons',
		dtype: 'int64',
		dims: ['batch', 'num_species', 'window_size'],
		valueRange: [0, CODON_UNKNOWN],
		padValue: CODON_UNKNOWN,
		siteInvariant: false
	}),
	Object.freeze({
		name: 'msa_aas',
		dtype: 'int64',
		dims: ['batch', 'num_species', 'window_size'],
		valueRange: [0, AA_UNKNOWN],
		padValue: AA_UNKNOWN,
		siteInvariant: false
	}),
	Object.freeze({
		name: 'dist_matrix',
		dtype: 'float32',
		dims: ['batch', 'num_species', 'num_species'],
		siteInvariant: true
	}),
	Object.freeze({
		name: 'mds_coords',
		dtype: 'float32',
		dims: ['batch', 'num_species', 'mds_components'],
		siteInvariant: true
	})
]);

/**
 * The single graph output. RESOLVED against the artifact itself — read out of the ONNX graph on
 * axomeme_v1_viral_finetuned.onnx (sha256 de765904…ccda3), not from documentation. The ML team's
 * own dev doc lists four inputs in its table while its prose says five; the table is correct for
 * this export and the prose is left over from 2.0.
 *
 * THE EXPORT IS EVAL MODE. This was an open question worth recording, because the shipped driver
 * switches the module to `model.train()` before every forward pass purely to reach the branch that
 * returns RAW ORDINAL LOGITS, so it can apply `--prior_shift` and call `decode_soft_ordinal_lrt`
 * itself (predict_regression_nexus.py:1354-1366). The answer is that the graph took the EVAL branch:
 * five outputs, `lrt` already decoded. Two consequences for the JS side:
 *
 *   - JS does NOT implement `decode_soft_ordinal_lrt`. The graph did it.
 *   - `prior_shift` is baked to 0 and is not reachable. If a calibration shift is ever wanted, it has
 *     to be re-exported, not applied here.
 *
 * STILL OPEN, and it belongs to POSTPROCESSING rather than to this contract: the driver applies
 * `np.expm1` to alpha / beta_neg / beta_pos before writing its CSV. Those heads are `F.softplus(...)`,
 * and expm1(softplus(x)) == exp(x), so the model is predicting log1p(rate) and expm1 recovers the
 * rate — meaning the raw graph outputs are NOT rates and must not be shown as such. `p_neg` is
 * sigmoid in-graph and is already a probability. Whether `lrt` needs a further exp is unresolved: the
 * driver emits both `predicted_log_lrt` and `predicted_lrt` columns, and which one this output
 * corresponds to has to be checked against sample_predictions.csv before anything is rendered.
 */
export const OUTPUT_SPEC = Object.freeze([
	Object.freeze({
		name: 'lrt',
		note: 'MEME LRT surrogate, ordinal decode already applied in-graph (eval-mode export)'
	})
	// v1-viral exports ONLY this head. The retired 2.0 export also returned alpha, beta_neg,
	// beta_pos and p_neg, which fed the dS / dN+ / p columns. Those are gone with the model.
]);

/**
 * The artifact this contract was verified against. A different export is not necessarily wrong, but
 * it is not this one, and the eval-mode conclusion above was read off THIS graph.
 */
export const VERIFIED_MODEL_SHA256 =
	'de765904107ba436c6ad6abbecb8af54962abd8444e1b5044947bb945d8ccda3';

/** Every input name, in graph order. */
export const INPUT_NAMES = Object.freeze(INPUT_SPEC.map((s) => s.name));

/**
 * Check a bundle of prepared tensors against the contract, without running anything.
 *
 * This is the cheap half of parity: it cannot tell you the MDS coordinates are RIGHT — only real
 * fixtures from the Python pipeline can do that — but it catches the whole class of errors that
 * produce a well-formed tensor with the wrong meaning, which is the class a JS port actually
 * generates. Returns every problem it finds rather than throwing on the first, because a port under
 * development usually has several and stopping at one wastes a round trip.
 *
 * @param {Record<string, {data: ArrayLike<number|bigint|boolean>, dims: number[]}>} bundle
 * @param {{batch: number, numSpecies: number, windowSize?: number, mdsComponents?: number}} shape
 * @returns {{ok: boolean, errors: string[]}}
 */
export function validateInputBundle(bundle, shape) {
	const errors = [];
	const { batch, numSpecies } = shape;
	const windowSize = shape.windowSize ?? WINDOW_SIZE_DEFAULT;
	const mdsComponents = shape.mdsComponents ?? MDS_COMPONENTS;

	const expected = {
		msa_codons: [batch, numSpecies, windowSize],
		msa_aas: [batch, numSpecies, windowSize],
		dist_matrix: [batch, numSpecies, numSpecies],
		mds_coords: [batch, numSpecies, mdsComponents]
	};

	for (const spec of INPUT_SPEC) {
		const t = bundle[spec.name];
		if (!t) {
			errors.push(`${spec.name}: missing`);
			continue;
		}
		const want = expected[spec.name];
		const got = Array.from(t.dims ?? []);
		if (got.length !== want.length || got.some((d, i) => d !== want[i])) {
			errors.push(`${spec.name}: dims [${got}], expected [${want}]`);
			continue;
		}
		const want_n = want.reduce((a, b) => a * b, 1);
		if (t.data.length !== want_n) {
			errors.push(
				`${spec.name}: ${t.data.length} elements for dims [${want}] (expected ${want_n})`
			);
			continue;
		}
		if (spec.valueRange) {
			const [lo, hi] = spec.valueRange;
			for (let i = 0; i < t.data.length; i++) {
				const v = Number(t.data[i]);
				if (!Number.isInteger(v) || v < lo || v > hi) {
					errors.push(`${spec.name}[${i}] = ${v}, outside ${lo}..${hi}`);
					break;
				}
			}
		}
	}

	// A TENSOR THE GRAPH DOES NOT ACCEPT is an error, not something to ignore.
	//
	// The specific case this catches is a bundle built for the retired 2.0 graph reaching v1-viral —
	// it would carry `padding_mask`, which no longer exists. onnxruntime would reject it too, but
	// several layers down and with a message about an invalid input name rather than about a stale
	// caller. Failing here says which tensor and why.
	for (const key of Object.keys(bundle)) {
		if (!INPUT_NAMES.includes(key)) {
			errors.push(
				`${key}: the graph has no such input — this bundle was built for a different model ` +
					`(expected exactly: ${INPUT_NAMES.join(', ')})`
			);
		}
	}

	// Cross-tensor invariants — the ones that are individually well-formed but jointly wrong.
	//
	// The mask-polarity check that used to live here is gone with `padding_mask`: v1-viral has no
	// mask input, so there is no polarity to get backwards. What it protected against — a tensor
	// that is well-formed in dtype, dims and element count but means the opposite of what the model
	// expects — no longer has a vector here.
	const dist = bundle.dist_matrix;

	if (dist) {
		// d(i,i) = 0 by definition. A nonzero diagonal means whatever was built is not a distance
		// matrix — most often an adjacency or a similarity matrix that took the same code path.
		for (let b = 0; b < batch; b++) {
			for (let i = 0; i < numSpecies; i++) {
				const self = Number(dist.data[b * numSpecies * numSpecies + i * numSpecies + i]);
				if (self !== 0) {
					errors.push(`dist_matrix: self-distance d(${i},${i}) is ${self}, expected 0`);
					b = batch;
					break;
				}
			}
		}
	}

	if (dist) {
		for (let i = 0; i < dist.data.length; i++) {
			const v = Number(dist.data[i]);
			if (!Number.isFinite(v)) {
				errors.push(`dist_matrix[${i}] is ${v}`);
				break;
			}
			if (v < 0) {
				// Not a shape error — a real one. DM3's own NJ emits negative branch lengths and the
				// Python inference path throws on them rather than degrading; see
				// src/lib/utils/treeSanitation.js for the measurement and the crash site.
				errors.push(`dist_matrix[${i}] = ${v} — negative patristic distance`);
				break;
			}
		}
	}

	return { ok: errors.length === 0, errors };
}

/**
 * THE THREE-OUTPUT GRAPH THIS REPOSITORY EXPORTS. Added for @veg/hyphaeon-js; not present in the
 * datamonkey3 original, and deliberately alongside `OUTPUT_SPEC` rather than replacing it.
 *
 * TWO GRAPHS EXIST AND BOTH ARE VALID. `OUTPUT_SPEC` above describes the shipped
 * `axomeme_v1_viral_finetuned.onnx` (sha256 de765904…ccda3), read off the artifact itself: ONE
 * output, `lrt`. That is what DataMonkey 3 serves today and what `VERIFIED_MODEL_SHA256` pins. This
 * constant describes what `hyphaeon export-onnx` produces for this repository — the same four
 * inputs, so `INPUT_SPEC` and `validateInputBundle` are unchanged and every trap they catch still
 * applies, with two further heads exposed.
 *
 * THE RUNTIME CHOOSES BETWEEN THEM BY THE MANIFEST, NEVER BY GUESSING. `models/manifest.json`
 * carries `onnx.outputs` beside each variant's `onnx_sha256`; the app's `runtime/` verifies the hash
 * of the artifact it loaded, reads that list, and picks the matching spec. Sniffing
 * `session.outputNames` would work right up until an export changed order or added a head, and then
 * silently mis-assign whole tensors — `mean_root_attns` and `root_repr` are both float32 and both
 * `[batch, something]`, so nothing would crash. This package does not load a model and therefore
 * cannot make the choice itself; it only supplies the two descriptions to choose between.
 *
 * WHERE THE SHAPES COME FROM, in `hyphaeon/model.py`:
 *   - `lrt` — `y_lrt_soft.view(batch_size)` (model.py:582), the ordinal decode already applied
 *     in-graph. Same head, same meaning, same eval-mode conclusion as `OUTPUT_SPEC`.
 *   - `mean_root_attns` — `all_attns.mean(dim=(0, 2)).view(batch_size, num_species)`
 *     (model.py:476-477), the root token's attention over species, averaged across layers and
 *     heads. `num_species` is the graph's N, i.e. the SELECTED taxa, so an attention vector is
 *     indexed by `selectedNames` from assemble.js and not by the alignment's own order.
 *   - `root_repr` — `x_full[:, 0, central_idx, :]` (model.py:576), the [ROOT] token's embedding at
 *     the central window position. `embed_dim` is 384 for this checkpoint (model_config.json), which
 *     is a property of the weights rather than of the contract, hence the symbolic dim.
 *
 * WHAT EACH ONE UNLOCKS, so an export that drops a head is recognised as a lost feature rather than
 * a smaller file: `lrt` alone gives meme, filtering, attribution, dms and the omnibus site p-values;
 * `mean_root_attns` is what epistasis multiplies by its deltas and what phenotype splits into
 * foreground and background attention; `root_repr` is the input to the separate `busted_head.onnx`
 * graph. A two-output export is not a degraded meme run — it is an epistasis pillar that cannot run.
 */
export const OUTPUT_SPEC_V1 = Object.freeze([
	Object.freeze({
		name: 'lrt',
		dtype: 'float32',
		dims: ['batch'],
		note: 'MEME LRT surrogate, ordinal decode already applied in-graph (eval-mode export)'
	}),
	Object.freeze({
		name: 'mean_root_attns',
		dtype: 'float32',
		dims: ['batch', 'num_species'],
		note: 'root-token attention over the SELECTED species, averaged over layers and heads'
	}),
	Object.freeze({
		name: 'root_repr',
		dtype: 'float32',
		dims: ['batch', 'embed_dim'],
		note: 'root-token embedding at the central window position; feeds busted_head.onnx'
	})
]);

/** Every output name of the three-output export, in graph order. */
export const OUTPUT_NAMES_V1 = Object.freeze(OUTPUT_SPEC_V1.map((s) => s.name));
