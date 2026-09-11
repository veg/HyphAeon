/**
 * WHY THIS FILE EXISTS
 *
 * The phenotype pillar — directional Phenotype-Genotype Association Mapping (PhyloWAS) and the
 * Phenotype-Associated Residue Signature (PARS) — mirroring `hyphaeon/phenotype.py` at
 * veg/HyphAeon reconcile/phase-5a, PLAN.md §5.1 port 5:
 *
 *   PRESETS                     phenotype.py:48-111    verbatim: the same eight keys, the same
 *                                                      species lists, in the same order
 *   resolvePhenotypeVector      resolve_phenotype_vector, phenotype.py:121-274
 *   runPhenotypeAssociation     run_phenotype_association, phenotype.py:375-674, as a pure
 *                                                      function over an async `predict` callback
 *
 * The tree-dependent half (the Brownian covariance and the permulation draws, phenotype.py:275-374)
 * is `permulations.js`, for the reason its header gives: D22 made a tree optional, and "no tree,
 * therefore no permulations" should be a fact about which module the pipeline calls.
 *
 * WHAT IT DELIBERATELY DOES NOT DO. No torch, no device selection, no onnxruntime, no file I/O:
 * `resolvePhenotypeVector` takes the CONTENT of a metadata table rather than a path (the app has a
 * File, the MCP has a string, neither has a filesystem the reference would recognise), and
 * `runPhenotypeAssociation` takes a `LoadedAlignment` plus either the model's two outputs or an
 * async `predict`, exactly as `epistasis.js` and `dms.js` do. `compute_transformer_attributions`
 * is IMPORTED from `epistasis.js` rather than reimplemented, so the two attention pillars share
 * one definition of the attribution matrix — the reference does the same thing with its
 * `from .epistasis import compute_transformer_attributions` at phenotype.py:44, and sector mining
 * comes from `sectors.js` for the same reason (phenotype.py:639).
 *
 * ---------------------------------------------------------------------------------------------
 * resolve_phenotype_vector, step by step (phenotype.py line numbers)
 *
 * Three mutually exclusive sources, tried in this PRIORITY order — the first that is truthy wins,
 * and the others are ignored even when supplied:
 *
 *   1. the metadata table (`phenotype_file`)                                             141-215
 *      - separator: TAB when the name ends `.tsv` or `.tab`, else comma                      146
 *      - species column: the first of `species, taxon, taxa, tree_leaf_name, assembly, id,
 *        name, species_name` that some column matches case-insensitively, else column 0   149-159
 *      - trait column: the first column that is not the species column                   161-166
 *      - `trait_dict[sp] = val` AND `trait_dict[sp.lower()] = val` for every row         168-173
 *      - per taxon: exact key, then lower-cased key, then the first dictionary key k with
 *        `k in t or t in k` — a SUBSTRING match in either direction, in row order        175-186
 *      - discrete: `str(val).strip().lower()` against
 *        `["1","true","yes","case","foreground","target","positive"]` plus whatever
 *        `foreground` adds (a comma-split string, or a list)                             188-199
 *      - continuous: `float(val)`, non-numeric values silently skipped                   201-207
 *      - continuous z-scoring and the description strings                                209-215
 *   2. a curated preset (`preset`), keyed by `preset.lower().replace("-", "_").strip()`  218-236
 *      matched with `fnmatch(t.lower(), pat.lower()) or (pat.lower() in t.lower())`      229-232
 *   3. inline `foreground` patterns                                                      239-272
 *      - a string containing `|` and no `,` splits on `|`, otherwise on `,`              241-245
 *      - per pattern: `re.search(pat, t, re.IGNORECASE)` first, and only if that neither
 *        matched nor raised, `fnmatch(...) or (clean_pat in t.lower())`                  249-260
 *   4. none of the three: ValueError                                                         274
 *
 * ---------------------------------------------------------------------------------------------
 * run_phenotype_association, step by step
 *
 *   6.  `y_norm = y / ||y||` (or y unchanged when ||y|| == 0); `proj = leaf_attr @ y_norm`;
 *       `spectral_energy = ||proj||`; `frob_norm = ||leaf_attr||`;
 *       `norm_spectral_ratio = spectral_energy / frob_norm` (0 when frob_norm == 0)      412-417
 *   6b. permulations, only when `permulations > 0 and tree_obj is not None`; any exception
 *       inside drops them silently (null_rhos = None, gene_p_perm = None)                423-443
 *   7.  per site s with `||a_s|| > 0` and `#{a < 20} >= min_taxa_per_site`:               448-516
 *         rho          = a_s . y / (||a_s|| * ||y|| + 1e-15)
 *         df           = max(1, N_valid - 2);  t = rho * sqrt(df / max(1e-15, 1 - rho^2))
 *         p_parametric = t.sf(t, df)
 *         p_assoc      = the permulation p when there is one, else p_parametric
 *         score        = sqrt(max(0, lrt)) * max(0, rho)
 *         p_value      = ACAT(p_lrt, p_assoc)
 *         derived_aa   = the most frequent valid residue among the FOREGROUND taxa
 *         fg/bg_freq   = its frequency in each group, in per cent
 *       then a stable sort by score, descending                                              518
 *   8.  Benjamini-Hochberg over the sorted `p_value` column                               520-525
 *   9.  the dual-track extreme-value statistics                                           527-539
 *   10. the PARS bracket: rho >= 0.40 AND score >= 0.50, first 15, in score order         541-543
 *   11. co-selection among the significant sites and the trait sectors                    545-621
 *
 * THE EVD LENGTH ADJUSTMENT (step 9), because the formula deserves a sentence. `sigma_null =
 * 1/sqrt(max(10, N))` is the standard error of a correlation between the attribution vector and a
 * random unit trait vector in N dimensions, floored at N = 10 so a tiny alignment cannot make the
 * null implausibly tight. `z_single = max_assoc / sigma_null` is the best site's association in
 * those units and `p_single = 2 * norm.sf(|z|)` its two-sided normal p, clamped to [1e-15, 1].
 * `p_evd = -expm1(-L * p_single)` is then `1 - (1 - p_single)^L` computed stably: the probability
 * that ANY of L independent sites reaches that z, i.e. the Gumbel/extreme-value correction for
 * having taken a maximum over the gene. `-np.expm1(-L * p)` rather than `1 - exp(-L*p)` keeps the
 * small-p end accurate (at L * p ~ 1e-16 the naive form returns 0). `score_track_a` is its
 * -log10, `score_track_b` is the gene-level `norm_spectral_ratio`, and the composite takes the
 * larger of `score_track_a / 10` and `score_track_b` — the /10 puts a p_evd of 1e-10 on the same
 * 0..1 scale as a ratio of 1.
 *
 * FLOAT PRECISION, deliberately matched (PLAN.md §5.3 rule 4). `leaf_attr` and `lrts` are float32
 * (they come out of the graph); `y` is float64. numpy's promotion then decides each line:
 *   - `np.linalg.norm(x)` with no axis is `sqrt(x.ravel().dot(x.ravel()))`, i.e. BLAS **sdot** in
 *     FLOAT32 for `attribution_norm` (`norm_a`, per site) and for `frob_norm` (the whole matrix).
 *     Both are float32 values in the reference's output, and the fixture records them as such.
 *     sdot's blocked accumulation is not a fixed function of n, so neither candidate reproduces
 *     it: measured on test/data/phenotype's 14 x 10 attribution matrix against numpy 2 +
 *     Accelerate, a float64 accumulation rounded once to float32 reproduces the whole-matrix norm
 *     and 8 of the 10 non-zero rows, and numpy's own pairwise float32 sum reproduces 9 rows and
 *     NOT the matrix norm. `normSdot32` therefore takes the float64 accumulation — the same
 *     choice, for the same reason, as epistasis.js's sgemm dot — and `attribution_norm` agrees
 *     with the reference to one float32 ulp (3.7e-9 absolute, 8.4e-8 relative on that matrix),
 *     which carries into `association_rho` (7.8e-8) and its p-value (1.5e-8). That is the
 *     fixtures' 1e-6 class, not the ulp.
 *   - `np.linalg.norm(A, axis=1)` is a different code path: `sqrt(add.reduce((x*x).real, axis=1))`,
 *     numpy's PAIRWISE float32 sum, not BLAS. `norm_leaf_mat` in the permulation block goes
 *     through it and `norm_a` in the site loop does not, so the two can differ in the last float32
 *     ulp on the same row. Both are reproduced as written, with `numpyPairwiseSum` for the second.
 *   - a float32 matrix times a float64 vector or matrix (`leaf_attr @ y_norm`,
 *     `leaf_attr @ Y_perms.T`) upcasts to float64 and runs in double.
 *   - `norm_sub_A @ norm_sub_A.T` is float32 sgemm. As measured in epistasis.js's header, no
 *     float32 accumulation order reproduces BLAS's blocked one; a float64 accumulation rounded
 *     once to float32 is both the closest and the most accurate, so that is what is used, and
 *     `similarity` agrees with the reference to ~5e-7 relative rather than to the ulp.
 *   - `cesi` here is FLOAT64, unlike `epistasis.py`'s float32 cesi: `lrt_1 = float(lrts[s1])` is
 *     a Python float before it reaches `np.sqrt`, so the whole expression is double
 *     (phenotype.py:604-606). The `sim >= 0.25` and `cesi >= 1.0` gates are therefore float64
 *     comparisons and do NOT have `epistasis.py`'s float32-threshold quirk.
 *   - `np.mean(a_s[is_fg])` is a float32 pairwise mean over the gathered subvector.
 *
 * ---------------------------------------------------------------------------------------------
 * QUIRKS REPLICATED (fix upstream first — PLAN.md §5.3 rule 3):
 *   - `background` is a parameter of `resolve_phenotype_vector` and of
 *     `run_phenotype_association` and is NEVER READ. Anything not matched as foreground is
 *     background. It is accepted here for signature parity and ignored.
 *   - The metadata branch's substring fallback is symmetric and unordered: `if k in t or t in k`
 *     over the dictionary in ROW order. The row `aotTri` therefore claims the taxon `aotTri_OMK`,
 *     and a one-letter species name would claim almost everything.
 *   - `trait_dict` holds each species under BOTH its original and its lower-cased key, so two
 *     rows differing only in case silently overwrite each other, last row winning.
 *   - Continuous mode z-scores `y` over ALL N taxa (unmatched taxa contribute their 0) while
 *     deciding WHETHER to z-score from `np.std(matched_values)` over the matched ones only
 *     (phenotype.py:211). It also leaves `foreground_count` and `background_count` at 0.
 *   - Discrete mode compares `str(val)`, so a trait column pandas typed as float64 yields
 *     `"1.0"`, which is not in the target list, and every taxon lands in the background. An
 *     integer column gives `"1"` and works. A column with ANY missing cell is float64 in pandas
 *     (int64 has no NA), so one blank row silently turns a working `1`/`0` column into `1.0`/`0.0`
 *     and empties the foreground. `parsePhenotypeTable` reproduces pandas' dtype inference closely
 *     enough to keep those distinctions (see its own comment).
 *   - NOT reproduced, because `resolve_phenotype_vector` cannot reach it: `df.iterrows()` builds
 *     each row as a Series and upcasts to a COMMON dtype, so in a table whose every column is
 *     numeric an int64 trait would arrive as a float64 and render `"1.0"`. The species column is a
 *     name in every input the pillar accepts, which forces the row to `object` and preserves the
 *     integer; a table with a numeric species column would diverge here.
 *   - The preset branch's second test, `pat.lower() in t.lower()`, compares a string that still
 *     contains the glob's `*`, so for the seven presets whose patterns all end in `*` it can
 *     never fire; only `fnmatch` does any work. In `echolocation`, whose patterns are bare names
 *     like `turTru`, it fires as a substring match.
 *   - The inline-foreground branch runs `re.search` FIRST, so a "glob" is silently a regex:
 *     `pan*` is `pa` followed by zero or more `n`, which matches `papAnu`. `fixtures/phenotype/
 *     resolve_phenotype_vector.json` case `foreground_glob_Smc6` pins exactly that (6 taxa, from
 *     `pa`/`ma` substrings, not from the glob).
 *   - `clean_pat = pat.rstrip(".*").lstrip(".*")` strips the CHARACTERS `.` and `*` from both
 *     ends, not the two-character sequence, and only when the pattern starts or ends with `.*`.
 *   - `description` embeds a Python list repr: `f"User-specified foreground patterns: {fg_list}"`
 *     produces `['a', 'b']` with single quotes. `pyReprStringList` reproduces `repr(list[str])`,
 *     including the switch to double quotes for a string containing an apostrophe.
 *   - `run_phenotype_association` computes `leaf_attr @ Y_perms.T` twice (phenotype.py:460, 436).
 *     Computed once here; numpy would have returned the same values both times.
 *   - The trait co-selection block indexes `sub_indices` in SCORE order, not site order, so
 *     `coselection_pairs`' `site_u`/`site_v` are not ordered by position and `site_u > site_v` is
 *     common. The e2e fixture's first pair is (325, 83) for exactly this reason.
 *   - `valid_n = (norms > 1e-12).squeeze()` leaves zero rows in `norm_sub_A` as zeros, and the
 *     pair loop still visits them; their cosine is 0, below the 0.15 gate.
 *   - `min_taxa_per_site` counts tokens `< 20`, which includes taxa with no attribution at all.
 *   - The record's `permulations_count` is the REQUESTED count when the permulations succeeded
 *     and 0 when they did not, so it doubles as a "did they run" flag.
 */

import { AA_MAP } from './preprocess/tokenizer.js';
import { pyStrip } from './preprocess/parse.js';
import { computeTransformerAttributions, runTransformerAttributions, CoselectionGraph } from './epistasis.js';
import { extractEpistaticSectorsTse, REV_AA_MAP } from './sectors.js';
import { benjaminiHochberg } from './numeric/bh.js';
import { cauchyCombination } from './numeric/cauchy.js';
import { tSf, normSf } from './numeric/special.js';
import { numpyPairwiseSum, numpyMeanFloat32 } from './numeric/reduce.js';
import { generatePermulations, pyRegexSource } from './permulations.js';

/**
 * @typedef {import('./preprocess/assemble.js').LoadedAlignment} LoadedAlignment
 * @typedef {import('./preprocess/tree.js').PhyloTree} PhyloTree
 * @typedef {import('./epistasis.js').AttentionPredictFn} AttentionPredictFn
 * @typedef {import('./epistasis.js').TransformerAttributions} TransformerAttributions
 */

// =================================================================================================
// PRESETS — phenotype.py:48-111, copied verbatim
// =================================================================================================

/**
 * The eight curated trait presets, transcribed character for character from phenotype.py:48-111:
 * the same keys, the same `title` / `description` / `foreground` fields, the same species in the
 * same order. `echolocation` additionally carries `controls`, which the reference declares and
 * never reads (`resolve_phenotype_vector` uses `p_info["foreground"]` only) — kept so the two
 * tables are diffable.
 *
 * @type {Readonly<Record<string, {title: string, description: string, foreground: string[], controls?: string[]}>>}
 */
export const PRESETS = Object.freeze({
	echolocation: {
		title: 'Mammalian Echolocation Convergence',
		description: 'Microchiropteran bats and odontocete toothed whales.',
		foreground: [
			'rhi*', 'hip*', 'myo*', 'pte*', 'mor*', 'min*', 'emb*', 'cra*', 'meg*', 'mol*',
			'turTru', 'delLeu', 'orcOrc', 'gloMel', 'phaSin', 'graGri', 'neoPho', 'phoPho',
			'phyCat', 'kogBre', 'kogSim', 'mesBid', 'zipCav', 'plaGan', 'iniGeo', 'lipVex', 'ponBla'
		],
		controls: ['pteAle', 'pteRod', 'pteRuf', 'pteGig', 'pteVam', 'ptePse', 'bal*', 'megNov', 'eubGla', 'escRob']
	},
	marine: {
		title: 'Marine Mammal Transition & Deep Diving Hypoxia',
		description: 'Cetaceans, Pinnipeds, Sirenians, and Sea Otters.',
		foreground: [
			'enhLut*', 'pusHis*', 'pusSib*', 'halGryp*', 'phoVit*', 'phoLar*', 'eriBar*', 'neoSch*',
			'odoRos*', 'zalCal*', 'eumJub*', 'arcAus*', 'arcGaz*', 'otoFla*', 'mirLeo*', 'mirAng*',
			'lepWed*', 'hydLep*', 'lobCar*', 'ommRos*', 'triMan*', 'triSen*', 'triInu*', 'dugDug*',
			'turTru*', 'delLeu*', 'orcOrc*', 'gloMel*', 'phaSin*', 'graGri*', 'neoPho*', 'phoPho*',
			'balMys*', 'balAcu*', 'balPhy*', 'balMus*', 'megNov*', 'eubGla*', 'escRob*', 'phyCat*',
			'kogBre*', 'kogSim*', 'mesBid*', 'zipCav*', 'plaGan*', 'iniGeo*', 'lipVex*', 'ponBla*'
		]
	},
	fossorial: {
		title: 'Subterranean Fossoriality & Hypercapnic Hypoxia',
		description: 'Naked mole-rats, blind mole-rats, golden moles, star-nosed moles, pocket gophers.',
		foreground: [
			'hetGla*', 'fukDam*', 'cryAns*', 'nanGal*', 'nanEhr*', 'spaCar*', 'conCri*', 'talEur*',
			'talOcc*', 'scaMos*', 'scaAqu*', 'chrAsi*', 'chrSta*', 'uroGra*', 'geoBur*', 'thoTal*',
			'canTub*', 'ellLut*', 'ellTal*'
		]
	},
	hibernation: {
		title: 'True Hibernation & Metabolic Torpor',
		description: 'Marmots, ground squirrels, dormice, tenrecs, hedgehogs, Myotis bats.',
		foreground: [
			'ictTri*', 'uroPar*', 'speCit*', 'speDau*', 'marFla*', 'marMar*', 'marVan*', 'marMon*',
			'gliGli*', 'dryNit*', 'musAve*', 'eriEur*', 'tenEca*', 'echTel*', 'micTal*', 'myoLuc*',
			'myoDau*', 'myoMyo*', 'myoNat*', 'myoBra*', 'ursArc*'
		]
	},
	longevity: {
		title: "Extreme Longevity & Peto's Paradox Centenarians",
		description: "Bowhead whale, naked mole-rat, Brandt's bat, elephants, humans.",
		foreground: [
			'balMys*', 'hetGla*', 'myoBra*', 'loxAfr*', 'eleMax*', 'homSap*'
		]
	},
	high_altitude: {
		title: 'High-Altitude Hypoxia Adaptation',
		description: 'Yak, Tibetan antelope, snow leopard, vicuna, pikas, chinchilla.',
		foreground: [
			'bosGru*', 'bosMut*', 'panHod*', 'panUnc*', 'vicVic*', 'vicPac*', 'chiLan*', 'ochCur*',
			'ochPri*', 'ochArg*'
		]
	},
	cardenolide: {
		title: 'Insect Cardenolide Resistance (ATP1a)',
		description: 'Chrysochus, Tetraopes, Danaus (Monarch), Oncopeltus.',
		foreground: [
			'chrysochus*', 'tetraopes*', 'danaus*', 'oncopeltus*', 'chrysomela*'
		]
	},
	dim_light: {
		title: 'Low-Light & Deep-Sea Rhodopsin Vision',
		description: 'Deep-sea teleosts, cavefish, coelacanth, marine diving mammals.',
		foreground: [
			'*eel*', '*conger*', '*scabbard*', '*blackdragon*', '*viperfish*', '*loosejaw*',
			'*lampfish*', '*thornyhead*', '*cavefish*', '*dolphin*', '*coelacanth*'
		]
	}
});

/**
 * The thresholds of `run_phenotype_association` that are literals in the reference, gathered where
 * a reader can find them. Every value is the reference's; none is tunable through the CLI.
 */
export const PHENOTYPE_THRESHOLDS = Object.freeze({
	/** phenotype.py:570 — a site enters the PARS bracket at rho >= 0.40 AND score >= 0.50. */
	parsMinRho: 0.4,
	parsMinScore: 0.5,
	/** phenotype.py:570 — at most 15 sites in the bracket. */
	parsMaxSites: 15,
	/** phenotype.py:599 — a pair is recorded when its cosine exceeds 0.15. */
	pairMinSimilarity: 0.15,
	/** phenotype.py:628 — and becomes a graph edge at cosine >= 0.25 and CESI >= 1.0. */
	edgeMinSimilarity: 0.25,
	edgeMinCesi: 1.0,
	/** phenotype.py:640-642 — the sector miner runs looser here than in the epistasis pillar. */
	sectorMinCliqueSize: 2,
	sectorMinCoherence: 0.45,
	/** phenotype.py:558 — the null SE floor, 1/sqrt(max(10, N)). */
	evdMinTaxa: 10
});

/** The discrete trait values the reference treats as foreground, phenotype.py:192. */
const DEFAULT_FG_TARGETS = ['1', 'true', 'yes', 'case', 'foreground', 'target', 'positive'];

// =================================================================================================
// CPython / pandas string behaviour the reference's output depends on
// =================================================================================================

/** `s.rstrip(chars)`: strip any of `chars` from the right. */
function pyRstripChars(s, chars) {
	let end = s.length;
	while (end > 0 && chars.includes(s[end - 1])) end--;
	return s.slice(0, end);
}

/** `s.lstrip(chars)`: strip any of `chars` from the left. */
function pyLstripChars(s, chars) {
	let start = 0;
	while (start < s.length && chars.includes(s[start])) start++;
	return s.slice(start);
}

/**
 * `repr(s)` for a Python string: single quotes, unless the string contains a `'` and no `"`, in
 * which case CPython switches to double quotes. Backslashes and the quote in use are escaped.
 *
 * @param {string} s
 * @returns {string}
 */
export function pyReprString(s) {
	const useDouble = s.includes("'") && !s.includes('"');
	const q = useDouble ? '"' : "'";
	let out = q;
	for (const ch of s) {
		if (ch === '\\') out += '\\\\';
		else if (ch === q) out += '\\' + ch;
		else if (ch === '\n') out += '\\n';
		else if (ch === '\r') out += '\\r';
		else if (ch === '\t') out += '\\t';
		else out += ch;
	}
	return out + q;
}

/**
 * `repr(list_of_str)` — `['a', 'b']`. This is what f-string interpolation of a list produces, and
 * it lands verbatim in `phenotype_meta.description` (phenotype.py:270).
 *
 * @param {ArrayLike<string>} items
 * @returns {string}
 */
export function pyReprStringList(items) {
	const parts = [];
	for (let i = 0; i < items.length; i++) parts.push(pyReprString(String(items[i])));
	return '[' + parts.join(', ') + ']';
}

/**
 * `str(x)` for a Python float: the shortest decimal that round-trips, always carrying a `.` or an
 * `e`, switching to exponent form at |x| >= 1e16 or |x| < 1e-4 (JavaScript switches at 1e21 and
 * 1e-6, hence the explicit band), with a signed at-least-two-digit exponent.
 *
 * @param {number} x
 * @returns {string}
 */
export function pyFloatStr(x) {
	if (Number.isNaN(x)) return 'nan';
	if (x === Infinity) return 'inf';
	if (x === -Infinity) return '-inf';
	const a = Math.abs(x);
	if (a !== 0 && (a >= 1e16 || a < 1e-4)) {
		// Exponent form keeps the shortest mantissa, with no trailing `.0`: repr(1e-05) is '1e-05'.
		const [mant, exp] = x.toExponential().split('e');
		const sign = exp[0] === '-' ? '-' : '+';
		return `${mant}e${sign}${exp.replace(/^[+-]/, '').padStart(2, '0')}`;
	}
	const s = Object.is(x, -0) ? '-0' : String(x);
	return /[.e]/.test(s) ? s : s + '.0';
}

/**
 * `fnmatch.translate(pat)` (CPython 3.12 `fnmatch`), as a JavaScript RegExp source anchored with
 * `^` and `$` and compiled with the `s` flag — the reference's `(?s:...)\Z`.
 *
 * `*` is any run of characters (including none and including `/`), `?` is exactly one character,
 * `[seq]` a class and `[!seq]` its complement; everything else is a literal. The bracket handling
 * follows CPython's: an unterminated `[` is a literal `[`, `[]` is `(?!)` (matches nothing) and
 * `[!]` is `.`.
 *
 * @param {string} pat
 * @returns {string} JavaScript regex source, unanchored
 */
function fnmatchTranslate(pat) {
	let out = '';
	let i = 0;
	const n = pat.length;
	while (i < n) {
		const c = pat[i++];
		if (c === '*') {
			// CPython collapses runs of '*' into one '.*'.
			while (i < n && pat[i] === '*') i++;
			out += '.*';
		} else if (c === '?') {
			out += '.';
		} else if (c === '[') {
			let j = i;
			if (j < n && pat[j] === '!') j++;
			if (j < n && pat[j] === ']') j++;
			while (j < n && pat[j] !== ']') j++;
			if (j >= n) {
				out += '\\[';
			} else {
				let stuff = pat.slice(i, j);
				i = j + 1;
				if (stuff === '') out += '(?!)';
				else if (stuff === '!') out += '.';
				else {
					if (stuff[0] === '!') stuff = '^' + stuff.slice(1);
					else if (stuff[0] === '^' || stuff[0] === '[') stuff = '\\' + stuff;
					// A '\' inside a class is literal in fnmatch and an escape in regex.
					out += '[' + stuff.replace(/\\/g, '\\\\') + ']';
				}
			}
		} else {
			out += c.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
		}
	}
	return out;
}

/**
 * `fnmatch.fnmatch(name, pat)` on POSIX, where `os.path.normcase` is the identity — the reference
 * lower-cases both sides itself before calling it (phenotype.py:230, 259).
 *
 * @param {string} name
 * @param {string} pat
 * @returns {boolean}
 */
export function pyFnmatch(name, pat) {
	return new RegExp('^' + fnmatchTranslate(pat) + '$', 's').test(name);
}

/**
 * `re.search(pat, text, re.IGNORECASE)` as a boolean, with the Python -> JavaScript regex source
 * translation `permulations.js` documents. A pattern JavaScript cannot compile throws, which is
 * how the reference's `re.error` behaves at phenotype.py:252 (where it is caught and ignored).
 *
 * @param {string} pat
 * @param {string} text
 * @returns {boolean}
 */
function pyRegexSearchI(pat, text) {
	return new RegExp(pyRegexSource(pat), 'i').test(text);
}

// =================================================================================================
// The metadata table
// =================================================================================================

/**
 * pandas' default `na_values` for `read_csv`, verbatim from `pandas.io.parsers`. An empty field is
 * NA too. `str(val)` is never reached for these because `pd.isna(val)` filters them first
 * (phenotype.py:188).
 */
const PANDAS_NA_VALUES = new Set([
	'', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN',
	'<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'n/a', 'nan', 'null'
]);

const INT_RE = /^[+-]?\d+$/;
const FLOAT_RE = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;

/**
 * @typedef {{columns: string[], rows: Array<Array<string|number|null>>, floatColumns: boolean[]}} PhenotypeTable
 */

/**
 * `pd.read_csv(path, sep=sep)` reduced to what `resolve_phenotype_vector` reads out of the frame:
 * the column names, the cell values, and enough dtype inference that `str(val)` renders the same
 * text (see the quirk list in the file header — a float column gives `"1.0"` and never matches a
 * discrete foreground target, an integer column gives `"1"` and does).
 *
 * Inference, per column, over the non-NA cells: all-integer -> integer, otherwise all-numeric ->
 * float, otherwise the raw strings. A column with any NA cannot be integer in pandas (int64 has no
 * NA), so it becomes float when the rest is numeric — reproduced. Quoting is RFC 4180 (`"` doubles
 * itself); blank lines are skipped (`skip_blank_lines=True`); a trailing newline is not a row.
 *
 * NOT reproduced, because `resolve_phenotype_vector` never exercises it: `dtype=`, `header=`,
 * comment and index columns, thousands separators, date parsing, duplicate-column mangling and
 * pandas' C-parser fast paths for booleans.
 *
 * @param {string} text file content
 * @param {string} sep single-character separator
 * @returns {PhenotypeTable}
 */
export function parsePhenotypeTable(text, sep) {
	const lines = String(text).split(/\r\n|\r|\n/).filter((l) => l !== '');
	if (lines.length === 0) return { columns: [], rows: [], floatColumns: [] };

	/** @param {string} line @returns {string[]} */
	const splitLine = (line) => {
		const fields = [];
		let cur = '';
		let inQuotes = false;
		for (let i = 0; i < line.length; i++) {
			const ch = line[i];
			if (inQuotes) {
				if (ch === '"') {
					if (line[i + 1] === '"') { cur += '"'; i++; } else inQuotes = false;
				} else cur += ch;
			} else if (ch === '"' && cur === '') {
				inQuotes = true;
			} else if (ch === sep) {
				fields.push(cur);
				cur = '';
			} else cur += ch;
		}
		fields.push(cur);
		return fields;
	};

	// pandas does not strip header names (`skipinitialspace=False`), so neither does this.
	const columns = splitLine(lines[0]);
	const raw = [];
	for (let r = 1; r < lines.length; r++) {
		const f = splitLine(lines[r]);
		while (f.length < columns.length) f.push('');
		raw.push(f);
	}

	const rows = raw.map(() => new Array(columns.length).fill(null));
	const floatColumns = new Array(columns.length).fill(false);
	for (let c = 0; c < columns.length; c++) {
		let allInt = true;
		let allNum = true;
		let anyValue = false;
		for (const f of raw) {
			const v = f[c];
			if (PANDAS_NA_VALUES.has(v)) { allInt = false; continue; } // an NA forces float64
			anyValue = true;
			if (!INT_RE.test(v)) allInt = false;
			if (!FLOAT_RE.test(v)) allNum = false;
		}
		const asNumber = anyValue && allNum;
		floatColumns[c] = asNumber && !allInt;
		for (let r = 0; r < raw.length; r++) {
			const v = raw[r][c];
			if (PANDAS_NA_VALUES.has(v)) rows[r][c] = null;
			else rows[r][c] = asNumber ? Number(v) : v;
		}
	}
	return { columns, rows, floatColumns };
}

/**
 * `str(row[col])` for a cell of that frame: a string is itself, an integer prints without a
 * fraction, a float goes through `pyFloatStr`.
 */
function cellToString(value, isFloatColumn) {
	if (value === null) return 'nan'; // `str(float('nan'))`, what pandas puts in an NA cell
	if (typeof value === 'number') return isFloatColumn ? pyFloatStr(value) : String(value);
	return String(value);
}

// =================================================================================================
// resolve_phenotype_vector  (phenotype.py:121-274)
// =================================================================================================

/**
 * @typedef {{
 *   y: Float64Array, mode: 'discrete'|'continuous', fgCount: number, bgCount: number,
 *   description: string,
 *   meta: {mode: string, foreground_count: number, background_count: number, description: string}
 * }} PhenotypeVector
 */

/**
 * `resolve_phenotype_vector(taxa, preset, foreground, background, phenotype_file, trait_col,
 * species_col, continuous)` (phenotype.py:121-274).
 *
 * `phenotypeCsv` replaces the reference's `phenotype_file`: this library does no I/O, so the
 * caller passes the table's CONTENT. `phenotypeFile` still carries the name, because the
 * reference derives two things from it — the separator (TAB for `.tsv`/`.tab`, else comma) and
 * the `os.path.basename` that goes into `description` — and the fixtures record both.
 *
 * `meta` is the reference's dict, key for key; `mode`, `fgCount`, `bgCount` and `description`
 * are the same values under the library's naming.
 *
 * @param {ArrayLike<string>} taxa
 * @param {{
 *   preset?: string|null, foreground?: string|ArrayLike<string>|null,
 *   background?: string|ArrayLike<string>|null,
 *   phenotypeCsv?: string|null, phenotypeFile?: string|null, sep?: string|null,
 *   traitCol?: string|null, speciesCol?: string|null, continuous?: boolean
 * }} [options] `background` is accepted and ignored, as in the reference
 * @returns {PhenotypeVector}
 * @throws {Error} when none of the three sources is given, or the preset is unknown
 */
export function resolvePhenotypeVector(taxa, options = {}) {
	const N = taxa.length;
	const y = new Float64Array(N);
	const meta = { mode: 'discrete', foreground_count: 0, background_count: 0, description: '' };
	const continuous = options.continuous === true;
	// `if foreground:` is Python truthiness, so an EMPTY list or string is no foreground at all
	// and falls through to the final ValueError (phenotype.py:239, 274).
	const fgRaw = options.foreground ?? null;
	const foreground = fgRaw !== null && fgRaw !== undefined && fgRaw.length > 0 ? fgRaw : null;
	void options.background; // phenotype.py:125 declares it and never reads it

	/** @returns {PhenotypeVector} */
	const finish = () => ({
		y,
		mode: /** @type {'discrete'|'continuous'} */ (meta.mode),
		fgCount: meta.foreground_count,
		bgCount: meta.background_count,
		description: meta.description,
		meta
	});

	/** phenotype.py:234-235, 265-266, 213-214 — the discrete counts. */
	const countDiscrete = () => {
		meta.mode = 'discrete';
		let fg = 0;
		for (let i = 0; i < N; i++) if (y[i] > 0) fg++;
		meta.foreground_count = fg;
		meta.background_count = N - fg;
	};

	// ---- 1. metadata table --------------------------------------------------------------------
	const csv = options.phenotypeCsv ?? null;
	if (csv !== null) {
		const fileName = options.phenotypeFile ?? '';
		const base = fileName.slice(fileName.lastIndexOf('/') + 1);
		const sep = options.sep ?? (/\.(tsv|tab)$/.test(fileName) ? '\t' : ',');
		const table = parsePhenotypeTable(csv, sep);

		// phenotype.py:149-159 — species column auto-detection.
		let speciesCol = options.speciesCol ?? null;
		if (!speciesCol) {
			for (const c of ['species', 'taxon', 'taxa', 'tree_leaf_name', 'assembly', 'id', 'name', 'species_name']) {
				const match = table.columns.filter((col) => col.toLowerCase() === c);
				if (match.length > 0) { speciesCol = match[0]; break; }
			}
			if (!speciesCol) speciesCol = table.columns[0];
		}
		// phenotype.py:161-166 — the first other column is the trait.
		let traitCol = options.traitCol ?? null;
		if (!traitCol) {
			const cand = table.columns.filter((c) => c !== speciesCol);
			if (cand.length === 0) throw new Error(`No valid trait column found in ${fileName}`);
			traitCol = cand[0];
		}
		const si = table.columns.indexOf(speciesCol);
		const ti = table.columns.indexOf(traitCol);
		if (si < 0) throw new Error(`Species column '${speciesCol}' not found in ${fileName}`);
		if (ti < 0) throw new Error(`Trait column '${traitCol}' not found in ${fileName}`);

		// phenotype.py:168-173 — both the original and the lower-cased key point at the value.
		/** @type {Map<string, {value: string|number|null}>} */
		const traitDict = new Map();
		for (const row of table.rows) {
			const sp = pyStrip(cellToString(row[si], table.floatColumns[si]));
			const box = { value: row[ti] };
			traitDict.set(sp, box);
			traitDict.set(sp.toLowerCase(), box);
		}

		// phenotype.py:186-199 — the discrete target list, extended by `foreground`.
		const fgTargets = DEFAULT_FG_TARGETS.slice();
		if (foreground) {
			if (typeof foreground === 'string') {
				for (const p of foreground.split(',')) {
					const t = pyStrip(p);
					if (t) fgTargets.push(t.toLowerCase());
				}
			} else {
				for (let k = 0; k < foreground.length; k++) fgTargets.push(pyStrip(String(foreground[k])).toLowerCase());
			}
		}

		/** @type {number[]} */
		const matchedValues = [];
		for (let i = 0; i < N; i++) {
			const t = taxa[i];
			let box = traitDict.get(t) ?? traitDict.get(t.toLowerCase());
			if (box === undefined) {
				// phenotype.py:181-186 — symmetric substring scan in row order.
				for (const [k, v] of traitDict) {
					if (k.includes(t) || t.includes(k)) { box = v; break; }
				}
			}
			if (box === undefined) continue;
			const val = box.value;
			if (val === null || (typeof val === 'number' && Number.isNaN(val))) continue;

			if (!continuous) {
				const valStr = pyStrip(cellToString(val, table.floatColumns[ti])).toLowerCase();
				y[i] = fgTargets.includes(valStr) ? 1.0 : 0.0;
				matchedValues.push(y[i]);
			} else {
				const num = typeof val === 'number' ? val : (FLOAT_RE.test(pyStrip(val)) ? Number(val) : NaN);
				// float(val) raising ValueError is the reference's silent skip (phenotype.py:206-207).
				if (typeof val !== 'number' && Number.isNaN(num)) continue;
				if (!Number.isNaN(num)) { y[i] = num; matchedValues.push(num); }
			}
		}

		if (continuous) {
			meta.mode = 'continuous';
			// QUIRK (phenotype.py:210-212): the decision uses std(matched), the transform uses all N.
			if (matchedValues.length > 0 && stdPopulation(matchedValues) > 0) {
				const m = meanOf(y);
				const s = stdPopulation(y);
				for (let i = 0; i < N; i++) y[i] = (y[i] - m) / s;
			}
			meta.description = `Continuous trait '${traitCol}' from ${base}`;
		} else {
			countDiscrete();
			meta.description = `Discrete trait '${traitCol}' from ${base}`;
		}
		return finish();
	}

	// ---- 2. preset ----------------------------------------------------------------------------
	if (options.preset) {
		const presetKey = pyStrip(String(options.preset).toLowerCase().replaceAll('-', '_'));
		if (!Object.prototype.hasOwnProperty.call(PRESETS, presetKey)) {
			throw new Error(
				`Unknown preset '${options.preset}'. Available: ${pyReprStringList(Object.keys(PRESETS))}`
			);
		}
		const info = PRESETS[presetKey];
		meta.description = `${info.title} (${info.description})`;
		for (let i = 0; i < N; i++) {
			const tl = taxa[i].toLowerCase();
			for (const pat of info.foreground) {
				const pl = pat.toLowerCase();
				if (pyFnmatch(tl, pl) || tl.includes(pl)) { y[i] = 1.0; break; }
			}
		}
		countDiscrete();
		return finish();
	}

	// ---- 3. inline foreground patterns --------------------------------------------------------
	if (foreground) {
		/** @type {string[]} */
		let fgList;
		if (typeof foreground === 'string') {
			// phenotype.py:241-245 — '|' wins only when there is no ','.
			const sep = foreground.includes('|') && !foreground.includes(',') ? '|' : ',';
			fgList = foreground.split(sep).map((p) => pyStrip(p)).filter((p) => p !== '');
		} else {
			fgList = Array.from(foreground, (p) => String(p));
		}

		for (let i = 0; i < N; i++) {
			const t = taxa[i];
			for (const pat of fgList) {
				const cleanPat = pat.startsWith('.*') || pat.endsWith('.*')
					? pyLstripChars(pyRstripChars(pat, '.*'), '.*')
					: pat;
				try {
					if (pyRegexSearchI(pat, t)) { y[i] = 1.0; break; }
				} catch {
					// phenotype.py:253-254 — `except Exception: pass`, then the fnmatch fallback.
				}
				const tl = t.toLowerCase();
				if (pyFnmatch(tl, pat.toLowerCase()) || tl.includes(cleanPat.toLowerCase())) {
					y[i] = 1.0;
					break;
				}
			}
		}
		countDiscrete();
		meta.description = `User-specified foreground patterns: ${pyReprStringList(fgList)}`;
		return finish();
	}

	throw new Error('Must provide one of --preset, --phenotype-file, or --foreground.');
}

/** `np.mean` over a float64 vector. */
function meanOf(v) {
	let s = 0;
	for (let i = 0; i < v.length; i++) s += v[i];
	return v.length === 0 ? NaN : s / v.length;
}

/** `np.std` — the POPULATION standard deviation (ddof = 0), as numpy defaults. */
function stdPopulation(v) {
	const n = v.length;
	if (n === 0) return NaN;
	const m = meanOf(v);
	let s = 0;
	for (let i = 0; i < n; i++) { const d = v[i] - m; s += d * d; }
	return Math.sqrt(s / n);
}

// =================================================================================================
// Float paths that must match numpy's, byte for byte where it is possible
// =================================================================================================

/**
 * `np.linalg.norm(x)` on a contiguous float32 vector: `sqrt(x.dot(x))`, i.e. BLAS **sdot** and a
 * float32 result. BLAS's blocked accumulation order is not reproducible in JavaScript (measured in
 * epistasis.js's header); a float64 accumulation rounded ONCE to float32 is the closest and the
 * most accurate approximation, within ~5e-7 relative, which is the 1e-6 fixture class.
 */
function normSdot32(a, lo, n) {
	let s = 0;
	for (let i = 0; i < n; i++) { const v = a[lo + i]; s += v * v; }
	return Math.fround(Math.sqrt(Math.fround(s)));
}

/**
 * `np.linalg.norm(A, axis=1)` on float32: `sqrt(add.reduce((x*x).real, axis=1))`, numpy's PAIRWISE
 * float32 sum over float32 squares — a different code path from the one above, reproduced exactly.
 */
function normReduce32(a, lo, n, scratch) {
	for (let i = 0; i < n; i++) { const v = a[lo + i]; scratch[i] = Math.fround(v * v); }
	return Math.fround(Math.sqrt(numpyPairwiseSum(scratch, 0, n, Math.fround)));
}

/** `np.linalg.norm(v)` on float64. */
function norm64(v, lo, n) {
	let s = 0;
	for (let i = 0; i < n; i++) { const x = v[lo + i]; s += x * x; }
	return Math.sqrt(s);
}

// =================================================================================================
// run_phenotype_association  (phenotype.py:375-674)
// =================================================================================================

/**
 * @typedef {{
 *   site: number, ref_aa: string, derived_aa: string, hyphaeon_lrt: number, p_lrt: number,
 *   attribution_norm: number, fg_mean_attn: number, bg_mean_attn: number, association_rho: number,
 *   p_value: number, p_assoc: number, p_assoc_parametric: number, p_assoc_perm: number|null,
 *   score: number, foreground_freq_pct: number, background_freq_pct: number, q_value?: number
 * }} PhenotypeSite
 */

/**
 * @typedef {{
 *   site_u: number, site_v: number, ref_u: string, ref_v: string, lrt_u: number, lrt_v: number,
 *   similarity: number, cesi: number, p_value: number, shared_branches: number, q_value?: number
 * }} TraitCoselectionPair
 */

/**
 * @typedef {{
 *   alignment: string|null, tree: string|null, taxa_count: number, codon_count: number,
 *   phenotype_meta: {mode: string, foreground_count: number, background_count: number, description: string},
 *   spectral_energy: number, norm_spectral_ratio: number, max_assoc: number,
 *   p_evd_length_adjusted: number, score_track_a: number, score_track_b: number,
 *   dual_track_composite: number, compact_pars_signature: string, permulations_count: number,
 *   gene_p_value_perm: number|null, significant_sites_count: number,
 *   coselection_pairs_count: number, trait_sectors_count: number,
 *   coselection_pairs: TraitCoselectionPair[],
 *   trait_sectors: import('./sectors.js').Sector[], sites: PhenotypeSite[]
 * }} PhenotypeReport
 */

/**
 * `run_phenotype_association(...)` (phenotype.py:375-674), everything from step 6 on: the loading,
 * the device and the model live outside the library (PLAN.md §5.5).
 *
 * The trait vector arrives either ready (`y`) or as `resolvePhenotypeVector` options
 * (`phenotype`); the model's outputs arrive either ready (`attention` + `lrt`, or a whole
 * `attributions` record from `computeTransformerAttributions`) or through `predict`.
 *
 * @param {{
 *   loaded: LoadedAlignment,
 *   taxa?: ArrayLike<string>,
 *   attributions?: TransformerAttributions,
 *   attention?: ArrayLike<number>, lrt?: ArrayLike<number>,
 *   y?: ArrayLike<number>,
 *   phenotype?: Parameters<typeof resolvePhenotypeVector>[1],
 *   tree?: PhyloTree|null,
 *   alignment?: string|null, treePath?: string|null
 * }} input
 * @param {AttentionPredictFn|null} [predict] required only when neither `attributions` nor
 *   `attention`+`lrt` is supplied
 * @param {{
 *   permulations?: number, minTaxaPerSite?: number, alpha?: number, batchSize?: number,
 *   nPermutations?: number, maxPermP?: number|null, seed?: number, continuous?: boolean,
 *   onProgress?: (p: {phase: string, done: number, total: number}) => void
 * }} [options] the reference's keyword arguments that survive the port; `continuous` also selects
 *   the minimum-foreground check's branch, as `run_phenotype_association` does
 * @returns {Promise<PhenotypeReport>}
 */
export async function runPhenotypeAssociation(input, predict = null, options = {}) {
	const loaded = input.loaded;
	const taxa = input.taxa ?? loaded.taxa;
	const permulations = Math.max(0, Math.floor(options.permulations ?? 0));
	const minTaxaPerSite = options.minTaxaPerSite ?? 4;
	const alpha = options.alpha ?? 0.05;
	const nPermutations = options.nPermutations ?? 10000;
	const maxPermP = options.maxPermP ?? null;
	const seed = options.seed ?? 42;
	const continuous = options.continuous ?? input.phenotype?.continuous ?? false;
	const tree = input.tree ?? null;

	const N = loaded.N;
	const L = loaded.L;

	// ---- 3. the phenotype vector (phenotype.py:414-430) ----------------------------------------
	let y;
	/** @type {{mode: string, foreground_count: number, background_count: number, description: string}} */
	let meta;
	if (input.y) {
		y = input.y instanceof Float64Array ? input.y : Float64Array.from(input.y);
		meta = { mode: continuous ? 'continuous' : 'discrete', foreground_count: 0, background_count: 0, description: '' };
		let fg = 0;
		for (let i = 0; i < N; i++) if (y[i] > 0) fg++;
		if (!continuous) { meta.foreground_count = fg; meta.background_count = N - fg; }
	} else {
		const resolved = resolvePhenotypeVector(taxa, { ...(input.phenotype ?? {}), continuous });
		y = resolved.y;
		meta = resolved.meta;
	}
	if (y.length !== N) throw new Error(`runPhenotypeAssociation: y has ${y.length} entries, the alignment has ${N} taxa`);

	const isFg = new Uint8Array(N);
	let fgCount = 0;
	for (let i = 0; i < N; i++) if (y[i] > 0) { isFg[i] = 1; fgCount++; }
	if (fgCount < 2 && !continuous) {
		throw new Error(`Insufficient foreground taxa (${fgCount}) matching criteria among ${N} taxa.`);
	}

	// ---- 5. the attribution matrix (phenotype.py:437-439), shared with the epistasis pillar ----
	/** @type {TransformerAttributions} */
	let attr;
	if (input.attributions) attr = input.attributions;
	else if (input.attention && input.lrt) {
		attr = computeTransformerAttributions({ attention: input.attention, lrts: input.lrt, aaTokens: loaded.a, L, N });
	} else {
		if (!predict) throw new Error('runPhenotypeAssociation: needs `attributions`, or `attention` + `lrt`, or a predict callback');
		attr = await runTransformerAttributions(loaded, predict, { batchSize: options.batchSize, onProgress: options.onProgress });
	}
	const leafAttr = attr.leafAttributions;
	const lrts = attr.lrts;
	const pvals = attr.pvals;
	const consAas = attr.consensusAas;
	const aNp = loaded.a;

	// ---- 6. projection onto the unit hypersphere (phenotype.py:441-445) ------------------------
	const normY = norm64(y, 0, N);
	const yNorm = new Float64Array(N);
	if (normY > 0) for (let i = 0; i < N; i++) yNorm[i] = y[i] / normY;
	else yNorm.set(y);
	const proj = new Float64Array(L);
	for (let s = 0; s < L; s++) {
		let acc = 0;
		const base = s * N;
		for (let n = 0; n < N; n++) acc += leafAttr[base + n] * yNorm[n];
		proj[s] = acc;
	}
	const spectralEnergy = norm64(proj, 0, L);
	const frobNorm = normSdot32(leafAttr, 0, L * N);
	const normSpectralRatio = frobNorm > 0 ? spectralEnergy / frobNorm : 0.0;

	// ---- 6b. permulations (phenotype.py:451-471) ----------------------------------------------
	/** @type {Float64Array|null} */
	let nullRhos = null; // [L, P]
	/** @type {number|null} */
	let geneP = null;
	if (permulations > 0 && tree !== null) {
		try {
			const perms = generatePermulations(y, tree, taxa, { nPerm: permulations, seed });
			if (perms.permMatrix === null) throw new Error(perms.skipped?.detail ?? 'permulations unavailable');
			const P = permulations;
			const Y = perms.permMatrix; // [P, N]
			const normYPerms = new Float64Array(P);
			for (let p = 0; p < P; p++) normYPerms[p] = norm64(Y, p * N, N);
			// leaf_attr @ Y_perms.T, float32 x float64 -> float64. The reference computes it twice
			// (lines 432 and 436) and numpy returns the same values both times.
			const dot = new Float64Array(L * P);
			for (let s = 0; s < L; s++) {
				const ab = s * N;
				for (let p = 0; p < P; p++) {
					let acc = 0;
					const yb = p * N;
					for (let n = 0; n < N; n++) acc += leafAttr[ab + n] * Y[yb + n];
					dot[s * P + p] = acc;
				}
			}
			const scratch = new Float32Array(N);
			nullRhos = new Float64Array(L * P);
			const nullProjCol = new Float64Array(L);
			const nullSpectral = new Float64Array(P);
			for (let s = 0; s < L; s++) {
				const rowNorm = normReduce32(leafAttr, s * N, N, scratch);
				for (let p = 0; p < P; p++) nullRhos[s * P + p] = dot[s * P + p] / (rowNorm * normYPerms[p] + 1e-15);
			}
			for (let p = 0; p < P; p++) {
				for (let s = 0; s < L; s++) nullProjCol[s] = dot[s * P + p] / (normYPerms[p] + 1e-15);
				nullSpectral[p] = norm64(nullProjCol, 0, L);
			}
			let ge = 0;
			for (let p = 0; p < P; p++) if (nullSpectral[p] >= spectralEnergy) ge++;
			geneP = (1.0 + ge) / (1.0 + permulations);
		} catch {
			// phenotype.py:469-471 — `except Exception: null_rhos = None; gene_p_perm = None`.
			nullRhos = null;
			geneP = null;
		}
	}

	// ---- 7. per-site associations (phenotype.py:475-544) ---------------------------------------
	/** @type {PhenotypeSite[]} */
	const siteResults = [];
	const fgBuf = new Float32Array(N);
	const bgBuf = new Float32Array(N);
	const counts = new Int32Array(20);
	for (let s = 0; s < L; s++) {
		const base = s * N;
		const normA = normSdot32(leafAttr, base, N);
		let nValid = 0;
		for (let n = 0; n < N; n++) if (aNp[base + n] < 20) nValid++;
		if (!(normA > 0 && nValid >= minTaxaPerSite)) continue;

		let dotAy = 0;
		for (let n = 0; n < N; n++) dotAy += leafAttr[base + n] * y[n];
		const rho = dotAy / (normA * normY + 1e-15);

		const df = Math.max(1, nValid - 2);
		const tStat = rho * Math.sqrt(df / Math.max(1e-15, 1.0 - rho * rho));
		const pParametric = tSf(tStat, df);

		/** @type {number|null} */
		let pAssocPerm = null;
		let pAssoc = pParametric;
		if (nullRhos !== null) {
			const P = permulations;
			let ge = 0;
			for (let p = 0; p < P; p++) if (nullRhos[s * P + p] >= rho) ge++;
			pAssocPerm = (1.0 + ge) / (1.0 + permulations);
			pAssoc = pAssocPerm;
		}

		const lrtVal = lrts[s];
		const pLrt = pvals[s];
		const score = Math.sqrt(Math.max(0.0, lrtVal)) * Math.max(0.0, rho);
		const pCombined = cauchyCombination(Float64Array.of(pLrt, pAssoc));

		const refAa = consAas[s];
		// phenotype.py:509-515 — the modal valid residue among the FOREGROUND taxa.
		counts.fill(0);
		let nFgValid = 0;
		for (let n = 0; n < N; n++) {
			if (!isFg[n]) continue;
			const t = aNp[base + n];
			if (t < 20) { counts[t]++; nFgValid++; }
		}
		let derivedAa = refAa;
		let fgFreq = 0.0;
		if (nFgValid > 0) {
			let major = 0;
			for (let t = 1; t < 20; t++) if (counts[t] > counts[major]) major = t;
			derivedAa = REV_AA_MAP.get(major) ?? refAa;
			fgFreq = (counts[major] / nFgValid) * 100.0;
		}

		// phenotype.py:517-522 — the SAME residue's frequency in the background.
		let nBgValid = 0;
		for (let n = 0; n < N; n++) if (!isFg[n] && aNp[base + n] < 20) nBgValid++;
		let bgFreq = 0.0;
		if (nBgValid > 0 && derivedAa !== '-') {
			const derivedTok = AA_MAP.get(derivedAa) ?? 20;
			let hits = 0;
			for (let n = 0; n < N; n++) {
				if (isFg[n]) continue;
				const t = aNp[base + n];
				if (t < 20 && t === derivedTok) hits++;
			}
			bgFreq = (hits / nBgValid) * 100.0;
		}

		// phenotype.py:524-525 — float32 pairwise means over the two groups.
		let nf = 0;
		let nb = 0;
		for (let n = 0; n < N; n++) {
			if (isFg[n]) fgBuf[nf++] = leafAttr[base + n];
			else bgBuf[nb++] = leafAttr[base + n];
		}
		const fgMeanAttn = fgCount > 0 ? numpyMeanFloat32(fgBuf, 0, nf) : 0.0;
		const bgMeanAttn = N - fgCount > 0 ? numpyMeanFloat32(bgBuf, 0, nb) : 0.0;

		siteResults.push({
			site: s + 1,
			ref_aa: refAa,
			derived_aa: derivedAa,
			hyphaeon_lrt: lrtVal,
			p_lrt: pLrt,
			attribution_norm: normA,
			fg_mean_attn: fgMeanAttn,
			bg_mean_attn: bgMeanAttn,
			association_rho: rho,
			p_value: pCombined,
			p_assoc: pAssoc,
			p_assoc_parametric: pParametric,
			p_assoc_perm: pAssocPerm,
			score,
			foreground_freq_pct: fgFreq,
			background_freq_pct: bgFreq
		});
	}

	// phenotype.py:546 — `sort(key=score, reverse=True)`; CPython's sort is stable and `reverse`
	// does not reverse ties, so equal scores keep site order. Array#sort is stable too (ES2019).
	siteResults.sort((a, b) => b.score - a.score);

	// ---- 8. Benjamini-Hochberg (phenotype.py:548-553) ------------------------------------------
	if (siteResults.length > 0) {
		const q = benjaminiHochberg(Float64Array.from(siteResults, (x) => x.p_value));
		for (let i = 0; i < siteResults.length; i++) siteResults[i].q_value = q[i];
	}

	// ---- 9. dual-track extreme-value statistics (phenotype.py:555-567) -------------------------
	const maxAssoc = siteResults.length > 0 ? siteResults[0].association_rho : 0.0;
	const sigmaNull = 1.0 / Math.sqrt(Math.max(PHENOTYPE_THRESHOLDS.evdMinTaxa, N));
	const zSingle = sigmaNull > 0 ? maxAssoc / sigmaNull : 0.0;
	let pSingle = 2.0 * normSf(Math.abs(zSingle));
	pSingle = Math.max(1e-15, Math.min(1.0, pSingle));
	let pEvd = -Math.expm1(-L * pSingle);
	pEvd = Math.max(1e-15, Math.min(1.0, pEvd));
	const scoreTrackA = -Math.log10(pEvd);
	const scoreTrackB = normSpectralRatio;
	const dualTrackComposite = Math.max(scoreTrackA / 10.0, scoreTrackB);

	// ---- 10. the PARS bracket (phenotype.py:569-571) -------------------------------------------
	const topPars = [];
	for (const x of siteResults) {
		if (topPars.length >= PHENOTYPE_THRESHOLDS.parsMaxSites) break;
		if (x.association_rho >= PHENOTYPE_THRESHOLDS.parsMinRho && x.score >= PHENOTYPE_THRESHOLDS.parsMinScore) {
			topPars.push(`${x.ref_aa}${x.site}${x.derived_aa}`);
		}
	}
	const compactPars = topPars.length > 0 ? `[ ${topPars.join(' - ')} ]` : '[]';

	// ---- 11. trait co-selection and sectors (phenotype.py:573-649) -----------------------------
	const sigTraitSites = siteResults.filter((x) => (x.q_value ?? 1.0) <= alpha && x.association_rho > 0);
	const traitSiteIndices = sigTraitSites.map((x) => x.site - 1);

	/** @type {TraitCoselectionPair[]} */
	let coselectionPairs = [];
	/** @type {import('./sectors.js').Sector[]} */
	let traitSectors = [];

	if (traitSiteIndices.length >= 2) {
		const Kt = traitSiteIndices.length;
		const normsScratch = new Float32Array(N);
		const norms = new Float32Array(Kt);
		const validN = new Uint8Array(Kt);
		for (let i = 0; i < Kt; i++) {
			norms[i] = normReduce32(leafAttr, traitSiteIndices[i] * N, N, normsScratch);
			validN[i] = norms[i] > 1e-12 ? 1 : 0;
		}
		let nValidRows = 0;
		for (let i = 0; i < Kt; i++) nValidRows += validN[i];

		if (nValidRows >= 2) {
			// phenotype.py:586-589 — zeros_like, then the valid rows scaled; float32 throughout.
			const normSubA = new Float32Array(Kt * N);
			for (let i = 0; i < Kt; i++) {
				if (!validN[i]) continue;
				const src = traitSiteIndices[i] * N;
				const dst = i * N;
				for (let n = 0; n < N; n++) normSubA[dst + n] = Math.fround(leafAttr[src + n] / norms[i]);
			}

			const graph = new CoselectionGraph();
			/** @type {TraitCoselectionPair[]} */
			const pairList = [];
			const dfPair = Math.max(1, N - 2);
			for (let i = 0; i < Kt; i++) {
				const s1 = traitSiteIndices[i];
				for (let j = i + 1; j < Kt; j++) {
					const s2 = traitSiteIndices[j];
					// float32 sgemm: float64 accumulation rounded once, per epistasis.js's measurement.
					let acc = 0;
					const bi = i * N;
					const bj = j * N;
					for (let n = 0; n < N; n++) acc += normSubA[bi + n] * normSubA[bj + n];
					const sim = Math.fround(acc);
					if (!(sim > PHENOTYPE_THRESHOLDS.pairMinSimilarity)) continue;

					const lrt1 = lrts[s1];
					const lrt2 = lrts[s2];
					// float64 here, unlike epistasis.py's float32 cesi (see the header).
					const cesi = sim * Math.sqrt(Math.max(0.1, lrt1) * Math.max(0.1, lrt2));
					const tPair = sim * Math.sqrt(dfPair / Math.max(1e-15, 1.0 - sim * sim));
					const pPair = tSf(tPair, dfPair);

					const tok1 = AA_MAP.get(consAas[s1]) ?? 20;
					const tok2 = AA_MAP.get(consAas[s2]) ?? 20;
					let shared = 0;
					const b1 = s1 * N;
					const b2 = s2 * N;
					for (let n = 0; n < N; n++) {
						const a1 = aNp[b1 + n];
						const a2 = aNp[b2 + n];
						if (a1 < 20 && a1 !== tok1 && a2 < 20 && a2 !== tok2) shared++;
					}

					pairList.push({
						site_u: s1 + 1,
						site_v: s2 + 1,
						ref_u: consAas[s1],
						ref_v: consAas[s2],
						lrt_u: lrt1,
						lrt_v: lrt2,
						similarity: sim,
						cesi,
						p_value: pPair,
						shared_branches: shared
					});
					if (sim >= PHENOTYPE_THRESHOLDS.edgeMinSimilarity && cesi >= PHENOTYPE_THRESHOLDS.edgeMinCesi) {
						graph.addEdge(s1 + 1, s2 + 1, { weight: sim, cesi });
					}
				}
			}

			if (pairList.length > 0) {
				const q = benjaminiHochberg(Float64Array.from(pairList, (x) => x.p_value));
				for (let i = 0; i < pairList.length; i++) pairList[i].q_value = q[i];
				pairList.sort((a, b) => b.cesi - a.cesi);
				coselectionPairs = pairList;
			}

			if (graph.numberOfEdges() > 0) {
				traitSectors = extractEpistaticSectorsTse(graph, leafAttr, lrts, consAas, {
					minCliqueSize: PHENOTYPE_THRESHOLDS.sectorMinCliqueSize,
					minCoherence: PHENOTYPE_THRESHOLDS.sectorMinCoherence,
					aNp,
					taxa,
					nPermutations,
					maxPermP,
					seed,
					N
				});
			}
		}
	}

	// ---- the record, in phenotype.py:652-674's key order ---------------------------------------
	return {
		alignment: input.alignment ?? null,
		tree: input.treePath ?? null,
		taxa_count: N,
		codon_count: L,
		phenotype_meta: meta,
		spectral_energy: spectralEnergy,
		norm_spectral_ratio: normSpectralRatio,
		max_assoc: maxAssoc,
		p_evd_length_adjusted: pEvd,
		score_track_a: scoreTrackA,
		score_track_b: scoreTrackB,
		dual_track_composite: dualTrackComposite,
		compact_pars_signature: compactPars,
		permulations_count: nullRhos !== null ? permulations : 0,
		gene_p_value_perm: geneP,
		significant_sites_count: sigTraitSites.length,
		coselection_pairs_count: coselectionPairs.length,
		trait_sectors_count: traitSectors.length,
		coselection_pairs: coselectionPairs,
		trait_sectors: traitSectors,
		sites: siteResults
	};
}
