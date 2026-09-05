/**
 * WHY THIS FILE EXISTS
 *
 * The one implementation of the "Before you run" checks of PLAN.md §4.3, run by the browser panel
 * and by the server's `/validate`, so both produce the same `warnings[]` codes (PLAN.md §2 hard
 * truth 4: "one implementation of the checks, run before the model on every surface"). It mirrors
 * no single Python function — `hyphaeon/dataset.py` at 267f5cf PRINTS its diagnostics from inside
 * `load_alignment_and_tree` (the notices at lines 606-614, 634-638, 648-650, 656, 665, 706-712) and
 * PLAN.md §7 item 6 asks upstream for a `diagnose()` that returns them; this is that function on
 * the JS side, built on the dataset mirror (parse.js, tree.js, downsample.js, assemble.js) so the
 * numbers it reports are the numbers the model would then be given.
 *
 * EVERY THRESHOLD, WITH ITS SOURCE (DIAGNOSTIC_THRESHOLDS below):
 *   - unknown codons > 5 %                      dataset.py:706-709 (`frac_unk > 0.05`)
 *   - max patristic distance > 10.0 -> rescaled  dataset.py:678-681
 *   - taxa < 3 -> refuse                         PLAN.md §4.3 (issue #7, the 2-taxon bug); dataset.py
 *                                                itself accepts any N >= 1
 *   - taxa > cap -> PD subsampling               MAX_SPECIES_CAP = 512 (models/manifest.json
 *                                                `taxon_cap`, the CLI --max-species default for busted)
 *   - taxa > 1,000 -> refuse                     PLAN.md §3.5 caps, §4.3
 *   - codons > 30,000, work L·N² > 2.5e9          PLAN.md §3.5 caps (reported inside COST_ESTIMATE)
 *   - median patristic < 0.05 -> shallow          PLAN.md §2 item 6 / §4.3 ("shallow -> suggest viral");
 *                                                the 0.05 figure is the task's, not dataset.py's.
 *                                                MEASURED on the bundled examples (dataset.py's own
 *                                                distances, after the rescale, or the TN93 distances
 *                                                on a tree-free run): Smc6 0.024 (20
 *                                                primates — it trips this), HIV1_RT 0.061, camelid
 *                                                0.284 (HyPhy lengths), bat_oas1 0.351 (rescaled from
 *                                                Mya), RHO 0.712
 *   - median patristic >= 0.2 and N >= 100 -> deep + large   PLAN.md §2 item 3 ("36 % FPR at 100
 *                                                taxa on deep trees"). 0.2 is a calibration choice
 *                                                from the measurements above: between the within-
 *                                                species viral panel (HIV1_RT, 0.06) and every
 *                                                cross-species panel (0.28-0.71). Skipped only when
 *                                                a tree WITH no branch lengths is used anyway (the
 *                                                distances are then dataset.py's 1e-3 defaults, not
 *                                                a depth); under D22 that case goes tree-free, and
 *                                                TN93 distances ARE a depth, so they are measured.
 *   - TN93 saturation: any pair at the sentinel  tn93.js / dataset.py:566. A pair the TN93 formula
 *                                                cannot resolve is reported as exactly 1.0, and a
 *                                                pair of identical sequences is imputed to
 *                                                max(1.0, max_d); either way it is a floor, not a
 *                                                distance. Measured on the bundled examples after
 *                                                the duplicate collapse the loader does first: 0
 *                                                pairs, on all five.
 *   - unique haplotypes < 5 or mean pairwise p-distance < 0.005 -> star-like   PLAN.md §2 item 5 /
 *                                                §4.3 (issue #33); numbers from the task statement.
 *                                                Measured mean p-distances: Smc6 0.023, HIV1_RT 0.053,
 *                                                camelid 0.173, bat_oas1 0.172, RHO 0.097
 *   - ambiguity (non-gap, non-ACGT) > 5 % -> warn; > 50 % of non-gap characters -> refuse (not a
 *                                                nucleotide alignment: A/C/G/T are ~25 % of a protein).
 *                                                5 % reuses dataset.py's only fraction threshold.
 *   - internal in-frame stops: any -> warn; > 1 % of internal non-gap codons -> refuse. A random
 *                                                reading frame has 3/64 = 4.7 % stops; 1 % is a
 *                                                fifth of that. The terminal codon (last non-gap
 *                                                codon of each sequence) is not counted as internal.
 *   - frame heuristic: a sequence whose frame-0 internal stops >= 2 and exceed the stops of frame 1
 *                                                or 2 (whole-sequence frames) is frame-suspect;
 *                                                warn; refuse when more than half the sequences are.
 *                                                Mid-sequence frameshifts are not localised by this
 *                                                rule; they surface under IN_FRAME_STOPS.
 *   - COST_ESTIMATE: predictedSeconds = L·N_used² / WORK_PER_SECOND with WORK_PER_SECOND = 4.7e6,
 *                                                calibrated from PLAN.md §1's measured table:
 *                                                HIV1_RT 335 × 476² = 7.6e7 work in ~16 s of model
 *                                                time (17.3 s wall less ~1.5 s start-up) on CPU
 *                                                torch, Apple M4 Pro, fp32. ASSUMPTION: a single
 *                                                L·N² term; the per-site constant cost that
 *                                                dominates at small N (Smc6: 0.09 s predicted vs
 *                                                0.21 s measured model time) is ignored, and the
 *                                                browser WASM path will be recalibrated separately.
 *
 * TREE-FREE MODE (PLAN.md D22, resolved 2026-09-05) replaced two refusals with one info: a missing
 * tree (was TREE_MISSING, refuse) and a tree without usable branch lengths (was
 * BRANCH_LENGTHS_MISSING, warn) are now TREE_FREE_TN93 at info level, with
 * `data.reason` = 'no_tree' | 'no_branch_lengths' | 'requested' ('requested' being `useTn93` or a
 * `treeText` of 'tn93' / 'none' / 'skip', which used to be TREE_MISSING too). Both codes are gone
 * from DIAGNOSTIC_CODES; TREE_UNPARSEABLE stays a refuse, because bad tree text is a bad input
 * rather than a missing one. In tree-free mode the taxon-matching codes (TAXA_NOT_IN_TREE,
 * TIPS_NOT_IN_ALIGNMENT) are not emitted — the tree does not select the taxa — and the new
 * TN93_SATURATED_PAIRS reports how many pairs came back at the saturation sentinel (and refuses
 * when the matrix could not be computed at all, which is where the reference raises).
 *
 * WHAT IT DELIBERATELY DOES NOT DO: no tree estimation (HyPhy is gone with D22; TN93 distances are
 * computed here, through `loadAlignmentAndTree`) — no
 * MEME-hit-likelihood prescreen (the XGBoost row of §4.3 lives in the app's runtime/prescreen);
 * no I/O, no printing. It never throws on bad input: every parse failure becomes a `refuse`
 * warning. It may throw on a caller error (`maxSpecies` 0, which dataset.py's stride pre-selection
 * divides by).
 *
 * There was no DataMonkey 3 counterpart in the library; DM3's checks lived in its Svelte
 * AnalyzeTab and in the app's runtime/fastaValidation.js, which this supersedes for the codes below.
 */

import { parseAlignmentSequences, pyStrip, pySplit, pySplitLines, pyIsDigit } from './preprocess/parse.js';
import { extractTree, findClades, treeTaxa, matchTaxa, hasNonzeroBranchLengths, stripQuotes } from './preprocess/tree.js';
import { pruneIdenticalSequences } from './preprocess/downsample.js';
import { loadAlignmentAndTree } from './preprocess/assemble.js';
import { codonToken } from './preprocess/tokenizer.js';
import { MAX_SPECIES_CAP, CODON_UNKNOWN } from './preprocess/modelContract.js';
import { TN93_MATCH_MODE, TN93_SATURATION_SENTINEL } from './preprocess/tn93.js';

/** Every code `diagnose` can emit, in the order of PLAN.md §4.3's rows. */
export const DIAGNOSTIC_CODES = Object.freeze([
	'FORMAT_UNKNOWN',
	'ALPHABET_U_TO_T',
	'NON_ACGT_FRACTION',
	'LENGTH_NOT_MULTIPLE_OF_3',
	'IN_FRAME_STOPS',
	'FRAMESHIFT_SUSPECTED',
	'UNKNOWN_CODON_FRACTION',
	'UNEQUAL_LENGTHS',
	'DUPLICATE_SEQUENCES',
	'TOO_FEW_TAXA',
	'TAXA_OVER_CAP',
	'TAXA_OVER_LIMIT',
	'TREE_UNPARSEABLE',
	'TREE_FREE_TN93',
	'TAXA_NOT_IN_TREE',
	'TIPS_NOT_IN_ALIGNMENT',
	'NEGATIVE_BRANCH_LENGTHS',
	'DISTANCE_RESCALED',
	'TN93_SATURATED_PAIRS',
	'SHALLOW_TREE',
	'DEEP_LARGE_TREE',
	'STAR_LIKE',
	'COST_ESTIMATE'
]);

/** The thresholds, each documented in the header. */
export const DIAGNOSTIC_THRESHOLDS = Object.freeze({
	unknownCodonFraction: 0.05, // dataset.py:707
	distanceRescaleMax: 10.0, // dataset.py:680
	minTaxa: 3, // PLAN.md §4.3, issue #7
	taxaCap: MAX_SPECIES_CAP, // 512
	taxaLimit: 1000, // PLAN.md §3.5
	codonCap: 30000, // PLAN.md §3.5
	workCap: 2.5e9, // PLAN.md §3.5
	shallowMedianPatristic: 0.05, // PLAN.md §2 item 6 (task figure)
	deepMedianPatristic: 0.2, // measured, see header
	deepLargeTaxa: 100, // PLAN.md §2 item 3
	starLikeHaplotypes: 5, // PLAN.md §2 item 5 (task figure)
	starLikeDivergence: 0.005, // PLAN.md §2 item 5 (task figure)
	ambiguityWarnFraction: 0.05, // dataset.py's 5 %, applied to characters
	nonNucleotideFraction: 0.5, // see header
	internalStopRefuseFraction: 0.01, // see header
	frameshiftPervasiveFraction: 0.5 // see header
});

/** L·N² per second of model time on CPU torch (PLAN.md §1 table, HIV1_RT); see header. */
export const WORK_PER_SECOND = 4.7e6;

/** Sequences and nucleotide positions the pairwise-divergence sampler is bounded to. */
const DIVERGENCE_MAX_TAXA = 60;
const DIVERGENCE_MAX_POSITIONS = 9000;
/** How many names a `data` list carries before it is truncated (the count is always full). */
const LIST_CAP = 25;

/**
 * @typedef {{code: string, severity: 'info'|'warn'|'refuse', message: string, data: Record<string, any>}} Diagnostic
 */

/**
 * @typedef {{
 *   ok: boolean,
 *   warnings: Diagnostic[],
 *   summary: {
 *     format: 'phylip'|'fasta'|'nexus'|'unknown',
 *     treeSource: 'user'|'embedded'|null,
 *     taxaInAlignment: number,
 *     taxaMatched: number|null,
 *     uniqueHaplotypes: number|null,
 *     taxaUsed: number|null,
 *     codons: number|null,
 *     medianPatristic: number|null,
 *     meanPairwiseDivergence: number|null,
 *     matchTier: string|null
 *   }
 * }} Diagnosis
 */

/**
 * Format sniff in parse.js's detection order: PHYLIP (first non-empty line is two integers), FASTA
 * (text starts with '>'), NEXUS (`#NEXUS` or a MATRIX block), else unknown. A hint for the panel;
 * parse.js decides for real.
 *
 * @param {string} text
 * @returns {'phylip'|'fasta'|'nexus'|'unknown'}
 */
export function sniffAlignmentFormat(text) {
	const first = pySplitLines(text).map((l) => pyStrip(l)).find((l) => l !== '');
	if (first !== undefined) {
		const toks = pySplit(first);
		if (toks.length === 2 && pyIsDigit(toks[0]) && pyIsDigit(toks[1])) return 'phylip';
	}
	if (pyStrip(text).startsWith('>')) return 'fasta';
	if (/#nexus/iu.test(text) || /\bmatrix\b/iu.test(text)) return 'nexus';
	return 'unknown';
}

const STOPS = new Set(['TAA', 'TAG', 'TGA']);
const isAcgt = (ch) => ch === 'A' || ch === 'C' || ch === 'G' || ch === 'T';
const cap = (/** @type {string[]} */ list) => (list.length > LIST_CAP ? list.slice(0, LIST_CAP) : list);

/**
 * Count the 'U'/'u' characters the parser turned into 'T': the text is re-parsed with every U
 * replaced by 'J' (alphanumeric, not a nucleotide code, absent from every format keyword, so the
 * parse has the same structure), and the extra J's in the sequences are the U's.
 *
 * @param {string} text
 * @param {Map<string, string>} seqDict
 * @returns {number}
 */
function countUracil(text, seqDict) {
	if (!/[Uu]/.test(text)) return 0;
	const countJ = (/** @type {Iterable<string>} */ seqs) => {
		let n = 0;
		for (const s of seqs) for (let i = 0; i < s.length; i++) if (s[i] === 'J') n++;
		return n;
	};
	try {
		const alt = parseAlignmentSequences(text.replace(/[Uu]/g, 'J'));
		return Math.max(0, countJ(alt.values()) - countJ(seqDict.values()));
	} catch {
		return 0;
	}
}

/**
 * Per-sequence codon scan in frame 0 plus the stop counts of frames 1 and 2.
 * @param {string} seq upper-cased
 */
function scanCodons(seq) {
	const n = Math.floor(seq.length / 3);
	let lastReal = -1;
	for (let i = n - 1; i >= 0; i--) {
		const c = seq.slice(i * 3, i * 3 + 3);
		if (isAcgt(c[0]) || isAcgt(c[1]) || isAcgt(c[2])) {
			lastReal = i;
			break;
		}
	}
	let internalStops = 0;
	let terminalStop = 0;
	let nonGapInternal = 0;
	for (let i = 0; i < n; i++) {
		const c = seq.slice(i * 3, i * 3 + 3);
		const real = isAcgt(c[0]) || isAcgt(c[1]) || isAcgt(c[2]);
		if (!real) continue;
		if (i < lastReal) {
			nonGapInternal++;
			if (STOPS.has(c)) internalStops++;
		} else if (i === lastReal && STOPS.has(c)) terminalStop = 1;
	}
	const shifted = [1, 2].map((f) => {
		const m = Math.floor((seq.length - f) / 3);
		let stops = 0;
		for (let i = 0; i < m - 1; i++) if (STOPS.has(seq.slice(f + i * 3, f + i * 3 + 3))) stops++;
		return stops;
	});
	return { codons: n, internalStops, terminalStop, nonGapInternal, frameStops: [internalStops, shifted[0], shifted[1]] };
}

/**
 * Mean pairwise p-distance (differences over positions where both sequences are A/C/G/T) over a
 * deterministic, evenly strided sample of at most DIVERGENCE_MAX_TAXA sequences and
 * DIVERGENCE_MAX_POSITIONS positions. Null when no pair has a comparable position.
 *
 * @param {string[]} seqs
 * @returns {number|null}
 */
export function meanPairwiseDivergence(seqs) {
	const n = seqs.length;
	if (n < 2) return null;
	const picks = [];
	if (n <= DIVERGENCE_MAX_TAXA) for (let i = 0; i < n; i++) picks.push(i);
	else for (let i = 0; i < DIVERGENCE_MAX_TAXA; i++) picks.push(Math.floor((i * n) / DIVERGENCE_MAX_TAXA));
	let minLen = Infinity;
	for (const i of picks) minLen = Math.min(minLen, seqs[i].length);
	if (!Number.isFinite(minLen) || minLen === 0) return null;
	const step = Math.max(1, Math.ceil(minLen / DIVERGENCE_MAX_POSITIONS));
	let sum = 0;
	let pairs = 0;
	for (let x = 0; x < picks.length; x++) {
		const a = seqs[picks[x]];
		for (let y = x + 1; y < picks.length; y++) {
			const b = seqs[picks[y]];
			let compared = 0;
			let diff = 0;
			for (let p = 0; p < minLen; p += step) {
				const ca = a[p];
				const cb = b[p];
				if (isAcgt(ca) && isAcgt(cb)) {
					compared++;
					if (ca !== cb) diff++;
				}
			}
			if (compared > 0) {
				sum += diff / compared;
				pairs++;
			}
		}
	}
	return pairs > 0 ? sum / pairs : null;
}

/**
 * Median of the strict upper triangle of a row-major n × n matrix; null for n < 2.
 * @param {ArrayLike<number>} d
 * @param {number} n
 */
export function medianOffDiagonal(d, n) {
	if (n < 2) return null;
	const vals = new Float64Array((n * (n - 1)) / 2);
	let k = 0;
	for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) vals[k++] = d[i * n + j];
	vals.sort();
	const m = vals.length;
	return m % 2 === 1 ? vals[(m - 1) / 2] : (vals[m / 2 - 1] + vals[m / 2]) / 2;
}

/**
 * The "Before you run" diagnostics of PLAN.md §4.3 on an alignment and (optionally) a tree.
 *
 * @param {{
 *   alignmentText: string,
 *   treeText?: string|null,
 *   parsed?: import('./preprocess/assemble.js').LoadedAlignment|null,
 *   maxSpecies?: number,
 *   taxaLimit?: number,
 *   useTn93?: boolean
 * }} input `parsed` is an existing `loadAlignmentAndTree` result for the same texts, to avoid
 *   loading twice; when absent the model-level checks load with `maxSpecies` (the cap the model
 *   run will use, default MAX_SPECIES_CAP = 512). `useTn93` forces the tree-free path even when a
 *   usable tree is present, exactly as `loadAlignmentAndTree`'s option does (D22).
 * @returns {Diagnosis}
 */
export function diagnose({ alignmentText, treeText = null, parsed = null, maxSpecies = MAX_SPECIES_CAP, taxaLimit = DIAGNOSTIC_THRESHOLDS.taxaLimit, useTn93 = false }) {
	const T = DIAGNOSTIC_THRESHOLDS;
	/** @type {Diagnostic[]} */
	const warnings = [];
	const push = (/** @type {string} */ code, /** @type {'info'|'warn'|'refuse'} */ severity, /** @type {string} */ message, data = {}) =>
		warnings.push({ code, severity, message, data });
	const text = String(alignmentText ?? '');
	const summary = {
		format: sniffAlignmentFormat(text),
		treeSource: /** @type {'user'|'embedded'|null} */ (null),
		taxaInAlignment: 0,
		taxaMatched: /** @type {number|null} */ (null),
		uniqueHaplotypes: /** @type {number|null} */ (null),
		taxaUsed: /** @type {number|null} */ (null),
		codons: /** @type {number|null} */ (null),
		medianPatristic: /** @type {number|null} */ (null),
		meanPairwiseDivergence: /** @type {number|null} */ (null),
		matchTier: /** @type {string|null} */ (null)
	};
	const finish = () => {
		// Deterministic order: PLAN.md §4.3's rows (DIAGNOSTIC_CODES), stable within a code.
		const rank = new Map(DIAGNOSTIC_CODES.map((c, i) => [c, i]));
		const sorted = warnings.map((w, i) => ({ w, i })).sort((a, b) => (rank.get(a.w.code) ?? 99) - (rank.get(b.w.code) ?? 99) || a.i - b.i).map((x) => x.w);
		return { ok: !sorted.some((w) => w.severity === 'refuse'), warnings: sorted, summary };
	};

	// ---- Alignment: format and parse (dataset.py:540-542 raises on an empty parse). ----
	/** @type {Map<string, string>} */
	let seqDict;
	try {
		seqDict = parseAlignmentSequences(text);
	} catch (e) {
		push('FORMAT_UNKNOWN', 'refuse', `The alignment could not be parsed: ${e instanceof Error ? e.message : String(e)}`, {
			format: summary.format
		});
		return finish();
	}
	if (seqDict.size === 0) {
		push('FORMAT_UNKNOWN', 'refuse', 'No sequences could be parsed from the alignment (FASTA, PHYLIP and NEXUS are read).', {
			format: summary.format
		});
		return finish();
	}
	const names = Array.from(seqDict.keys());
	summary.taxaInAlignment = names.length;

	// ---- Alphabet. ----
	const uCount = countUracil(text, seqDict);
	if (uCount > 0) {
		push('ALPHABET_U_TO_T', 'info', `${uCount} U characters were read as T (RNA alphabet).`, { count: uCount });
	}
	let total = 0;
	let acgt = 0;
	let gaps = 0;
	for (const s of seqDict.values()) {
		total += s.length;
		for (let i = 0; i < s.length; i++) {
			const ch = s[i];
			if (isAcgt(ch)) acgt++;
			else if (ch === '-') gaps++;
		}
	}
	const ambiguous = total - acgt - gaps;
	const nonAcgtFraction = total > 0 ? (total - acgt) / total : 0;
	const ambiguousFraction = total > 0 ? ambiguous / total : 0;
	const nonGapAmbiguousFraction = total - gaps > 0 ? ambiguous / (total - gaps) : 0;
	if (total - acgt > 0) {
		const severity =
			nonGapAmbiguousFraction > T.nonNucleotideFraction ? 'refuse' : ambiguousFraction > T.ambiguityWarnFraction ? 'warn' : 'info';
		const message =
			severity === 'refuse'
				? `${(nonGapAmbiguousFraction * 100).toFixed(1)}% of non-gap characters are not A/C/G/T: this does not look like a nucleotide alignment.`
				: `${(nonAcgtFraction * 100).toFixed(1)}% of characters are not A/C/G/T (${gaps} gaps, ${ambiguous} ambiguity or other codes).`;
		push('NON_ACGT_FRACTION', severity, message, {
			total,
			acgt,
			gaps,
			ambiguous,
			fraction: nonAcgtFraction,
			ambiguousFraction,
			nonGapAmbiguousFraction
		});
	}

	// ---- Tree text, before the model-level load, so raw branch lengths can be inspected. ----
	/** @type {import('./preprocess/tree.js').PhyloTree|null} */
	let tree = null;
	/** D22: why the run will use TN93 distances instead of the tree, if it will. @type {'requested'|'no_tree'|'no_branch_lengths'|null} */
	let treeFreeReason = null;
	const treeMode = treeText === null || treeText === undefined ? null : String(treeText).trim().toLowerCase();
	if (useTn93) treeFreeReason = 'requested';
	if (treeMode === null) {
		tree = extractTree(text);
		if (tree) summary.treeSource = 'embedded';
		else if (treeFreeReason === null) treeFreeReason = 'no_tree';
	} else if (['tn93', 'none', 'skip'].includes(treeMode)) {
		treeFreeReason = 'requested';
	} else {
		tree = extractTree(String(treeText));
		if (tree) summary.treeSource = 'user';
		else push('TREE_UNPARSEABLE', 'refuse', 'The tree text could not be parsed as Newick or a NEXUS TREE block.', { recoverable: false });
	}

	let branchLengthsMissing = false;
	/** @type {Record<string, number|null>|null} */
	let branchStats = null;
	if (tree) {
		let missing = 0;
		let negatives = 0;
		let zeros = 0;
		let positives = 0;
		let minLen = Infinity;
		let maxLen = -Infinity;
		for (const n of findClades(tree)) {
			if (n === tree.root) continue;
			const bl = tree.branchLength[n];
			if (bl === null) missing++;
			else {
				if (bl < 0) negatives++;
				else if (bl === 0) zeros++;
				else positives++;
				minLen = Math.min(minLen, bl);
				maxLen = Math.max(maxLen, bl);
			}
		}
		const branches = missing + negatives + zeros + positives;
		branchLengthsMissing = !hasNonzeroBranchLengths(tree);
		if (branchLengthsMissing && treeFreeReason === null) treeFreeReason = 'no_branch_lengths';
		if (negatives > 0) {
			push('NEGATIVE_BRANCH_LENGTHS', 'warn', `${negatives} branch length(s) are negative (minimum ${minLen}); they are raised to 1e-4 (dataset.py:289-300).`, {
				count: negatives,
				min: minLen,
				raisedTo: 1e-4
			});
		}
		branchStats = { branches, missing, zeros, negatives, positives, max: Number.isFinite(maxLen) ? maxLen : null };
		Object.assign(summary, { branchLengths: branchStats });
	}

	if (treeFreeReason !== null) {
		const why =
			treeFreeReason === 'requested'
				? 'TN93 distances were requested'
				: treeFreeReason === 'no_tree'
					? 'no tree was given and none is embedded in the alignment'
					: 'the tree has no usable branch lengths';
		push(
			'TREE_FREE_TN93',
			'info',
			`Tree-free TN93 mode will be used: ${why}. Pairwise Tamura-Nei 93 distances (match mode '${TN93_MATCH_MODE}') feed the MDS directly and every alignment sequence is kept, in alignment order${tree ? '; the tree is used for display only' : ''}.`,
			{
				reason: treeFreeReason,
				taxaOrder: 'alignment',
				distances: 'tn93',
				matchMode: TN93_MATCH_MODE,
				treeKeptForDisplay: tree !== null,
				branchLengths: branchStats
			}
		);
	}

	// ---- Taxon matching (dataset.py:616-642) and haplotype collapse (645-650), on the light path. ----
	// Skipped in tree-free mode: the tree does not choose the taxa there, so a tip that is missing
	// from the alignment (or the other way round) is not a defect (D22).
	/** @type {string[]|null} */
	let taxa = null;
	/** @type {string[]|null} */
	let uniqueTaxa = null;
	if (tree && treeFreeReason === null) {
		const treeNames = treeTaxa(tree);
		try {
			const match = matchTaxa(treeNames, names);
			taxa = match.taxa;
			summary.matchTier = match.tier;
			const matchedSet = new Set(taxa);
			const unmatchedAlignment = names.filter((n) => !matchedSet.has(n));
			let treeKey = (/** @type {string} */ t) => t;
			let alignKeys = new Set(names);
			if (match.tier === 'quote_stripped') alignKeys = new Set(names.map((k) => stripQuotes(k)));
			else if (match.tier === 'case_insensitive') {
				alignKeys = new Set(names.map((k) => stripQuotes(k).toLowerCase()));
				treeKey = (t) => stripQuotes(t).toLowerCase();
			}
			const unmatchedTree = treeNames.filter((t) => !alignKeys.has(treeKey(t)));
			summary.taxaMatched = taxa.length;
			if (unmatchedAlignment.length > 0) {
				push(
					'TAXA_NOT_IN_TREE',
					'refuse',
					`${unmatchedAlignment.length} alignment sequence(s) have no tip in the tree and would be dropped silently (issue #9): ${cap(unmatchedAlignment).join(', ')}${unmatchedAlignment.length > LIST_CAP ? ', ...' : ''}.`,
					{ count: unmatchedAlignment.length, names: cap(unmatchedAlignment), matchTier: match.tier }
				);
			}
			if (unmatchedTree.length > 0) {
				push(
					'TIPS_NOT_IN_ALIGNMENT',
					'warn',
					`${unmatchedTree.length} tree tip(s) have no sequence and are ignored: ${cap(unmatchedTree).join(', ')}${unmatchedTree.length > LIST_CAP ? ', ...' : ''}.`,
					{ count: unmatchedTree.length, names: cap(unmatchedTree), matchTier: match.tier }
				);
			}
			if (match.tier !== 'exact') {
				Object.assign(summary, { matchNote: `names matched only after ${match.tier === 'quote_stripped' ? 'stripping quotes' : 'case folding'}` });
			}
		} catch {
			push('TAXA_NOT_IN_TREE', 'refuse', 'No alignment sequence name matches any tree tip, even after stripping quotes and ignoring case.', {
				count: names.length,
				names: cap(names),
				tips: cap(treeNames),
				matchTier: null
			});
		}
	}
	const dedupeOver = taxa ?? names;
	const pruned = pruneIdenticalSequences(seqDict, dedupeOver);
	uniqueTaxa = pruned.uniqueTaxa;
	summary.uniqueHaplotypes = uniqueTaxa.length;
	if (pruned.numPruned > 0) {
		const collapsed = [];
		for (const [keep, dups] of pruned.dupMap) if (dups.length) collapsed.push({ kept: keep, collapsed: dups });
		push('DUPLICATE_SEQUENCES', 'info', `${pruned.numPruned} identical sequence(s) collapsed onto ${collapsed.length} kept taxa (${dedupeOver.length} -> ${uniqueTaxa.length}).`, {
			collapsed: pruned.numPruned,
			groups: cap(collapsed.map((g) => g.kept)),
			duplicateMap: Object.fromEntries(collapsed.map((g) => [g.kept, g.collapsed]))
		});
	}

	// ---- Lengths and frame (dataset.py:652-667 takes L from the FIRST matched taxon). ----
	const first = /** @type {string} */ (seqDict.get((taxa ?? names)[0]));
	const rawLen = first.length;
	const lengthSet = new Set((taxa ?? names).map((n) => /** @type {string} */ (seqDict.get(n)).length));
	if (lengthSet.size > 1) {
		const lens = Array.from(lengthSet).sort((a, b) => a - b);
		push(
			'UNEQUAL_LENGTHS',
			'warn',
			`Sequences have ${lengthSet.size} different lengths (${lens[0]}..${lens[lens.length - 1]} bp); codon count follows the first taxon (${Math.floor(rawLen / 3)}), shorter sequences read as gaps beyond their end and longer ones are truncated.`,
			{ lengths: lens.slice(0, LIST_CAP), min: lens[0], max: lens[lens.length - 1], codonsFromFirst: Math.floor(rawLen / 3) }
		);
	}
	const notMultiple = (taxa ?? names).filter((n) => /** @type {string} */ (seqDict.get(n)).length % 3 !== 0);
	if (rawLen < 3) {
		push('LENGTH_NOT_MULTIPLE_OF_3', 'refuse', `The first sequence is ${rawLen} bp, shorter than one codon (dataset.py:661 raises).`, {
			length: rawLen,
			remainder: rawLen % 3,
			sequences: notMultiple.length
		});
	} else if (notMultiple.length > 0) {
		push(
			'LENGTH_NOT_MULTIPLE_OF_3',
			'warn',
			`${notMultiple.length} sequence(s) have a length that is not a multiple of 3; the first taxon's ${rawLen % 3} trailing nucleotide(s) are trimmed (dataset.py:664-667).`,
			{ sequences: notMultiple.length, names: cap(notMultiple), remainder: rawLen % 3, trimmedNucleotides: rawLen % 3 }
		);
	}

	// ---- Stops and frame heuristic, over every sequence that would be used. ----
	let internalStops = 0;
	let terminalStops = 0;
	let nonGapInternal = 0;
	let withInternal = 0;
	const frameSuspect = [];
	for (const n of uniqueTaxa) {
		const sc = scanCodons(/** @type {string} */ (seqDict.get(n)));
		internalStops += sc.internalStops;
		terminalStops += sc.terminalStop;
		nonGapInternal += sc.nonGapInternal;
		if (sc.internalStops > 0) withInternal++;
		const [f0, f1, f2] = sc.frameStops;
		if (f0 >= 2 && Math.min(f1, f2) < f0) frameSuspect.push({ name: n, frameStops: sc.frameStops });
	}
	const internalStopFraction = nonGapInternal > 0 ? internalStops / nonGapInternal : 0;
	if (internalStops > 0) {
		const severity = internalStopFraction > T.internalStopRefuseFraction ? 'refuse' : 'warn';
		push(
			'IN_FRAME_STOPS',
			severity,
			`${internalStops} internal in-frame stop codon(s) in ${withInternal} sequence(s) (${(internalStopFraction * 100).toFixed(2)}% of internal codons${severity === 'refuse' ? '; pervasive, check the reading frame' : ''}); ${terminalStops} terminal stop(s) not counted.`,
			{ internal: internalStops, terminal: terminalStops, sequencesWithInternal: withInternal, fraction: internalStopFraction, codonsScanned: nonGapInternal }
		);
	}
	if (frameSuspect.length > 0) {
		const frac = frameSuspect.length / uniqueTaxa.length;
		push(
			'FRAMESHIFT_SUSPECTED',
			frac > T.frameshiftPervasiveFraction ? 'refuse' : 'warn',
			`${frameSuspect.length} sequence(s) have fewer stop codons in another reading frame than in frame 0: ${cap(frameSuspect.map((f) => f.name)).join(', ')}${frameSuspect.length > LIST_CAP ? ', ...' : ''}.`,
			{ count: frameSuspect.length, fraction: frac, sequences: cap(frameSuspect.map((f) => f.name)), frameStops: frameSuspect.slice(0, LIST_CAP).map((f) => f.frameStops) }
		);
	}

	// ---- Taxon counts. ----
	const nUnique = uniqueTaxa.length;
	const nRaw = (taxa ?? names).length;
	if (nUnique < T.minTaxa) {
		push('TOO_FEW_TAXA', 'refuse', `${nUnique} usable taxa after matching and haplotype collapse; at least ${T.minTaxa} are required (issue #7).`, {
			taxa: nUnique,
			minimum: T.minTaxa
		});
	}
	if (nRaw > taxaLimit) {
		push('TAXA_OVER_LIMIT', 'refuse', `${nRaw} taxa exceed the limit of ${taxaLimit}.`, { taxa: nRaw, limit: taxaLimit });
	} else if (nUnique > maxSpecies) {
		push(
			'TAXA_OVER_CAP',
			'info',
			`${nUnique} unique taxa exceed the cap of ${maxSpecies}; Faith's PD subsampling keeps ${maxSpecies}${nUnique > 2 * maxSpecies ? ` after a stride pre-selection to ${2 * maxSpecies}` : ''} (dataset.py:669-685).`,
			{ taxa: nUnique, cap: maxSpecies, used: maxSpecies, stridePreselect: nUnique > 2 * maxSpecies }
		);
	}

	// ---- Model-level load: what the graph would actually be given. ----
	/** @type {import('./preprocess/assemble.js').LoadedAlignment|null} */
	let loaded = parsed ?? null;
	const canLoad = (treeFreeReason !== null || (tree !== null && taxa !== null)) && rawLen >= 3 && nRaw <= taxaLimit;
	if (loaded === null && canLoad) {
		try {
			loaded = loadAlignmentAndTree(text, treeMode === null ? null : String(treeText), { maxSpecies, pruneDuplicates: true, useTn93 });
		} catch (e) {
			// The reference dies here too: `compute_tn93_distance_matrix` lets the tn93 package's
			// ZeroDivisionError (no overlap) and math-domain ValueError (a saturated pair) propagate
			// (dataset.py:538-557). Report it instead of throwing; every other path keeps raising.
			if (treeFreeReason === null) throw e;
			push('TN93_SATURATED_PAIRS', 'refuse', `The TN93 distance matrix could not be computed: ${e instanceof Error ? e.message : String(e)}`, {
				pairs: null,
				sentinel: TN93_SATURATION_SENTINEL,
				reason: treeFreeReason,
				error: e instanceof Error ? e.message : String(e)
			});
		}
	}
	let L = Math.floor(rawLen / 3);
	let nUsed = Math.min(nUnique, maxSpecies);
	let unknownFraction = null;
	let unknownCodons = 0;
	let totalCodons = 0;
	if (loaded) {
		L = loaded.L;
		nUsed = loaded.N;
		unknownFraction = loaded.notices.unknownCodonFraction;
		unknownCodons = loaded.notices.unknownCodons;
		totalCodons = loaded.notices.totalCodons;
		summary.taxaUsed = nUsed;
		if (loaded.notices.distanceRescaled) {
			push(
				'DISTANCE_RESCALED',
				'warn',
				`The largest patristic distance is ${loaded.notices.rawDistMax.toPrecision(4)} (> ${T.distanceRescaleMax}): branch lengths look like a chronogram or mutation counts rather than substitutions per site; distances were divided by ${L} (dataset.py:678-681).`,
				{ rawMax: loaded.notices.rawDistMax, threshold: T.distanceRescaleMax, dividedBy: L }
			);
		}
		// Issue #9: a taxon with no terminal of its name gets an all-zero row (compute_fast_dist_matrix).
		// Tree-free runs have no terminals to miss, and their imputation leaves no zero row.
		if (loaded.N > 1 && loaded.notices.treeFree === null) {
			const zeroRows = [];
			for (let i = 0; i < loaded.N; i++) {
				let allZero = true;
				for (let j = 0; j < loaded.N; j++) if (loaded.d[i * loaded.N + j] !== 0) { allZero = false; break; }
				if (allZero) zeroRows.push(loaded.taxa[i]);
			}
			if (zeroRows.length > 0) {
				push(
					'TAXA_NOT_IN_TREE',
					'refuse',
					`${zeroRows.length} taxa have an all-zero distance row: their names matched the tree only after ${summary.matchTier === 'quote_stripped' ? 'quote stripping' : 'case folding'}, and compute_fast_dist_matrix finds no terminal for them (issue #9).`,
					{ count: zeroRows.length, names: cap(zeroRows), zeroDistanceRows: true, matchTier: summary.matchTier }
				);
			}
		}
		if (loaded.notices.treeFree !== null) {
			const saturated = loaded.notices.tn93SaturatedPairs ?? 0;
			const pairs = (loaded.N * (loaded.N - 1)) / 2;
			if (saturated > 0) {
				push(
					'TN93_SATURATED_PAIRS',
					'warn',
					`${saturated} of ${pairs} taxon pairs (${((saturated / Math.max(1, pairs)) * 100).toFixed(1)}%) are at the TN93 saturation sentinel ${TN93_SATURATION_SENTINEL}: their divergence could not be estimated and the sentinel was substituted, so the MDS positions of those taxa are floors, not measurements.`,
					{ pairs: saturated, totalPairs: pairs, fraction: saturated / Math.max(1, pairs), sentinel: TN93_SATURATION_SENTINEL, reason: loaded.notices.treeFree.reason }
				);
			}
		}
		// The depth regime needs real distances. Patristic distances from a tree without branch
		// lengths are 1e-3 defaults, so they are skipped; TN93 distances are real, so tree-free runs
		// (including the ones taken BECAUSE the tree had no lengths) are measured (D22).
		const median = loaded.notices.branchLengthsMissing && loaded.notices.treeFree === null ? null : medianOffDiagonal(loaded.d, loaded.N);
		summary.medianPatristic = median;
		if (median !== null) {
			if (median < T.shallowMedianPatristic) {
				push(
					'SHALLOW_TREE',
					'info',
					`Median patristic distance ${median.toPrecision(3)} substitutions/site is shallow (< ${T.shallowMedianPatristic}); the viral variant was trained on this regime.`,
					{ medianPatristic: median, threshold: T.shallowMedianPatristic, suggestVariant: 'viral' }
				);
			} else if (median >= T.deepMedianPatristic && nUsed >= T.deepLargeTaxa) {
				push(
					'DEEP_LARGE_TREE',
					'warn',
					`Deep tree (median patristic distance ${median.toPrecision(3)}) with ${nUsed} taxa: neutral simulations in this regime gave a false-positive rate near 36% at 100 taxa (PLAN.md §2); treat site calls as a ranking, not as calibrated p-values.`,
					{ medianPatristic: median, taxa: nUsed, thresholds: { median: T.deepMedianPatristic, taxa: T.deepLargeTaxa } }
				);
			}
		}
	} else {
		// No model-level load: count unknown codons the way dataset.py:690-704 does, over the taxa we have.
		for (const n of uniqueTaxa) {
			const s = /** @type {string} */ (seqDict.get(n));
			for (let site = 0; site < L; site++) {
				const codon = s.slice(site * 3, site * 3 + 3);
				if (codonToken(codon) === CODON_UNKNOWN && !STOPS.has(codon)) unknownCodons++;
			}
		}
		totalCodons = L * uniqueTaxa.length;
		unknownFraction = unknownCodons / Math.max(1, totalCodons);
	}
	summary.codons = L;
	if (unknownFraction !== null && unknownFraction > T.unknownCodonFraction) {
		push(
			'UNKNOWN_CODON_FRACTION',
			'warn',
			`${unknownCodons}/${totalCodons} (${(unknownFraction * 100).toFixed(1)}%) codons contain gaps, ambiguities or unrecognised bases (> ${T.unknownCodonFraction * 100}%, dataset.py:706-709).`,
			{ unknownCodons, totalCodons, fraction: unknownFraction, threshold: T.unknownCodonFraction }
		);
	}

	// ---- Regime: star-like panels (issue #33). ----
	const divergence = meanPairwiseDivergence(uniqueTaxa.map((n) => /** @type {string} */ (seqDict.get(n))));
	summary.meanPairwiseDivergence = divergence;
	if (nUnique < T.starLikeHaplotypes || (divergence !== null && divergence < T.starLikeDivergence)) {
		push(
			'STAR_LIKE',
			'warn',
			`${nUnique} unique haplotype(s) with mean pairwise divergence ${divergence === null ? 'n/a' : divergence.toPrecision(3)}: a star-like, single-ancestor panel is outside the regime the model was evaluated in and can return nothing (issue #33).`,
			{ uniqueHaplotypes: nUnique, meanPairwiseDivergence: divergence, thresholds: { haplotypes: T.starLikeHaplotypes, divergence: T.starLikeDivergence } }
		);
	}

	// ---- Cost. ----
	const work = L * nUsed * nUsed;
	const predictedSeconds = work / WORK_PER_SECOND;
	push('COST_ESTIMATE', 'info', `${L} codons x ${nUsed} taxa: about ${predictedSeconds < 1 ? '< 1' : Math.round(predictedSeconds)} s of model time on a laptop CPU (reference path).`, {
		L,
		N_used: nUsed,
		work,
		predictedSeconds,
		workPerSecond: WORK_PER_SECOND,
		exceedsCaps: { codons: L > T.codonCap, work: work > T.workCap },
		caps: { codons: T.codonCap, work: T.workCap },
		assumption: 'single L*N^2 term calibrated on HIV1_RT under CPU torch fp32 (PLAN.md section 1); browser WASM to be recalibrated'
	});

	return finish();
}
