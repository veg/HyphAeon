/**
 * WHY THIS FILE EXISTS
 *
 * The description of, and the validator for, the tensors that cross the boundary this package does
 * not cross. `@veg/hyphaeon-js` never loads a model — that is `hyphaeon-app`'s `runtime/` — so this
 * file is how a pure function can still say "the bundle you just built is not what the graph
 * accepts" before anything expensive happens.
 *
 * WHAT IT MIRRORS (all at veg/HyphAeon 267f5cf):
 *   - the token vocabularies of `hyphaeon/dataset.py:25-57` (`GENETIC_CODE`, `AA_MAP`,
 *     `get_codon_token`, `get_aa_token`): 61 sense codons 0..60, everything else 64; 20 residues
 *     0..19, everything else 20;
 *   - the embedding sizes of `hyphaeon/model.py:248-256`: `nn.Embedding(num_tokens=66, ...)` for
 *     codons and `nn.Embedding(23, ...)` for amino acids — the graph accepts 0..65 and 0..22, the
 *     tokenizer only ever produces 0..64 and 0..20;
 *   - the input signature of `PhyloAxialTransformer.forward` (`model.py:481-583`): `msa_codons`
 *     [B,N,W] int64, `msa_aas` [B,N,W] int64, `dist_matrix` [B,N,N] float32, `mds_coords` [B,N,4]
 *     float32, with `dist_matrix` and `mds_coords` per-alignment (dataset.py:727-728 unsqueezes
 *     them to [1,N,N] / [1,N,4] and the model broadcasts);
 *   - `models/manifest.json`: `taxon_cap` 512, `default_taxon_cap` 256 (also `max_species=256`
 *     in `model.py:248`), the four input names and the three output names of `export-onnx`.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: it does not load a model, and it does not decide which of the
 * two output specs applies — the runtime reads `models/manifest.json` for that.
 *
 * DIVERGENCE FROM THE DATAMONKEY3 PORT (main@fac1330 src/lib/services/axomeme/modelContract.js),
 * which this file replaces:
 *   - `CODON_UNKNOWN` was 65 and `CODON_GAP` 64 (AxoMEME 2.0 training vocabulary). dataset.py has
 *     one codon sentinel, 64, for stops, gaps and anything unrecognised. Both constants are now 64
 *     and `CODON_STOP` is added, also 64.
 *   - `AA_LIST` was the 23-character 'ACDEFGHIKLMNPQRSTVWY*-?' with stop 20 / gap 21 / unknown 22.
 *     dataset.py's `AA_MAP` has the 20 residues only and `get_aa_token` returns 20 for everything
 *     else. `AA_LIST` is now the 20 residues; `AA_STOP`, `AA_GAP` and `AA_UNKNOWN` are all 20.
 *   - `NUM_AA_TOKENS` (23) is added beside `NUM_CODON_TOKENS` (66): the embedding tables are larger
 *     than the vocabulary the tokenizer produces, and a validator that used the vocabulary as the
 *     table size would wrongly reject nothing and wrongly accept nothing — it is recorded because
 *     the gap between the two is a fact about the checkpoint worth knowing.
 *   - `CODON_VALID_BELOW` / `AA_VALID_BELOW` were the 2.0 model's `(c < 64) & (a < 21)` gate.
 *     `model.py:481-583` at 267f5cf has NO token gate at all; the only "validity" rule in v1.0.0 is
 *     dataset.py:721's `aa_col < 20` for the invariable-site mask. The constants are kept for the
 *     runtime (64 and 20) and now name that rule.
 *   - `MAX_SPECIES_DEFAULT` was 512. It is now 256 (`model.py:248` `max_species=256`,
 *     `manifest.json` `default_taxon_cap`), and `MAX_SPECIES_CAP` = 512 (`manifest.json`
 *     `taxon_cap`, `cli.py:1108` busted default) is added. dataset.py itself applies NO cap unless
 *     `max_species` is passed (`cli.py:1025` meme default None), which is what
 *     `loadAlignmentAndTree` mirrors; these two constants are for the runtime's own policy.
 *   - The long AxoMEME 2.0 handoff narrative (driver-vs-training tokenizer, MDS on the padded matrix)
 *     is gone: it described a different model and a different preprocessing; mds.js and assemble.js
 *     carry the dataset.py facts instead.
 *   - `validateInputBundle` keeps its checks (dims, element counts, value ranges, stale tensors,
 *     zero diagonal, finite and non-negative distances) with the ranges read from the new spec.
 *   - `OUTPUT_SPEC` (the single-output DM3 artifact), `VERIFIED_MODEL_SHA256`, `OUTPUT_SPEC_V1` and
 *     `OUTPUT_NAMES_V1` are unchanged.
 */

/** Codon table order of `GENETIC_CODE` in dataset.py:25-33: TCAG, third position fastest. */
export const CODON_ORDER = 'TCAG';

/**
 * dataset.py:25-33 numbers the 61 sense codons 0..60 in TCAG order with the stops skipped, maps
 * TAA/TAG/TGA to 64, and `get_codon_token` (line 52) returns 64 for anything not in the table:
 * gaps, ambiguity codes, wrong lengths, 'U'. Tokens 61, 62, 63 and 65 are never produced.
 */
export const CODON_STOP = 64;
export const CODON_GAP = 64;
export const CODON_UNKNOWN = 64;

/** `nn.Embedding(num_tokens=66, ...)`, model.py:248,255. The graph accepts 0..65. */
export const NUM_CODON_TOKENS = 66;

/**
 * dataset.py:36-39 `AA_MAP`: the 20 standard residues, alphabetical, 0..19. `get_aa_token`
 * (line 55) returns 20 for a stop ('*'), a gap, ambiguity, or anything untranslatable.
 */
export const AA_LIST = 'ACDEFGHIKLMNPQRSTVWY';
export const AA_STOP = 20;
export const AA_GAP = 20;
export const AA_UNKNOWN = 20;

/** `nn.Embedding(23, ...)`, model.py:256. The graph accepts 0..22; the tokenizer emits 0..20. */
export const NUM_AA_TOKENS = 23;

/**
 * The only validity rule in v1.0.0: dataset.py:721 `valid_aa = aa_col[aa_col < 20]` when deciding
 * whether a site is invariable. Codons have no such rule in the reference; 64 is the sense-codon
 * bound (61 sense codons occupy 0..60, and 64 is the sentinel).
 */
export const CODON_VALID_BELOW = 64;
export const AA_VALID_BELOW = 20;

/**
 * `max_species=256` in `PhyloAxialTransformer.__init__` (model.py:248) and `default_taxon_cap` in
 * models/manifest.json. dataset.py applies no cap unless asked (cli.py:1025 defaults to None).
 */
export const MAX_SPECIES_DEFAULT = 256;

/** `taxon_cap` in models/manifest.json; `cli.py:1108` busted `--max-species` default. */
export const MAX_SPECIES_CAP = 512;

/** dataset.py builds [L, N, 1] token tensors: one codon per site, no window. */
export const WINDOW_SIZE_DEFAULT = 1;

/** `compute_mds_coordinates(dist_mat, n_components=4)`, dataset.py:688. */
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
 * The single output of the DataMonkey 3 artifact `axomeme_v1_viral_finetuned.onnx` (sha256
 * de765904…ccda3), read off the graph itself: `lrt`, the ordinal decode already applied in-graph
 * (eval-mode export). Kept beside `OUTPUT_SPEC_V1` because both artifacts exist and the runtime
 * picks by manifest.
 */
export const OUTPUT_SPEC = Object.freeze([
	Object.freeze({
		name: 'lrt',
		note: 'MEME LRT surrogate, ordinal decode already applied in-graph (eval-mode export)'
	})
]);

/** The DataMonkey 3 artifact `OUTPUT_SPEC` was verified against. */
export const VERIFIED_MODEL_SHA256 =
	'de765904107ba436c6ad6abbecb8af54962abd8444e1b5044947bb945d8ccda3';

/** Every input name, in graph order. */
export const INPUT_NAMES = Object.freeze(INPUT_SPEC.map((s) => s.name));

/**
 * Check a bundle of prepared tensors against the contract, without running anything.
 *
 * It cannot tell you the MDS coordinates are RIGHT — the fixture replay does that — but it catches
 * the whole class of errors that produce a well-formed tensor with the wrong meaning. Returns every
 * problem it finds rather than throwing on the first.
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

	// A tensor the graph does not accept is an error, not something to ignore: a bundle built for
	// a different model (e.g. one carrying `padding_mask`) fails here with the tensor named, rather
	// than several layers down inside onnxruntime.
	for (const key of Object.keys(bundle)) {
		if (!INPUT_NAMES.includes(key)) {
			errors.push(
				`${key}: the graph has no such input — this bundle was built for a different model ` +
					`(expected exactly: ${INPUT_NAMES.join(', ')})`
			);
		}
	}

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
				// dataset.py cannot produce one: enforce_nonzero_branch_lengths (dataset.py:289-300)
				// raises every non-root branch to >= 1e-4 before the matrix is built. A negative
				// distance therefore means the bundle did not come through loadAlignmentAndTree.
				errors.push(`dist_matrix[${i}] = ${v} — negative patristic distance`);
				break;
			}
		}
	}

	return { ok: errors.length === 0, errors };
}

/**
 * THE THREE-OUTPUT GRAPH THIS REPOSITORY EXPORTS (`hyphaeon export-onnx`), beside `OUTPUT_SPEC`.
 * The runtime chooses between them by `models/manifest.json` `onnx.outputs`, never by sniffing
 * `session.outputNames`: `mean_root_attns` and `root_repr` are both float32 and both
 * `[batch, something]`, so a mis-assignment would not crash.
 *
 * Shapes, from `hyphaeon/model.py`:
 *   - `lrt` — `y_lrt_soft.view(batch_size)` (model.py:582), ordinal decode applied in-graph.
 *   - `mean_root_attns` — `all_attns.mean(dim=(0, 2)).view(batch_size, num_species)`
 *     (model.py:476-477): the root token's attention over species, averaged across layers and heads,
 *     indexed by the `taxa` order `loadAlignmentAndTree` returns.
 *   - `root_repr` — `x_full[:, 0, central_idx, :]` (model.py:576), embed_dim 384 for this checkpoint
 *     (model_config.json); the input to `busted_head.onnx`.
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
