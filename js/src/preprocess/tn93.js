/**
 * WHY THIS FILE EXISTS
 *
 * The tree-free distance path (PLAN.md D22): when no usable tree is available, pairwise
 * Tamura-Nei 1993 distances replace the patristic matrix and feed the MDS directly. This file
 * mirrors TWO sources, because the reference does not compute TN93 itself — it shells out:
 *
 *   A. `hyphaeon/dataset.py:493-571` (`compute_tn93_distance_matrix`) at veg/HyphAeon 61d30e3:
 *      the matrix assembly, the sentinel and the imputation rules.
 *   B. the `tn93` PyPI package, version 1.2.2, `tn93/tn93.py` (the fallback dataset.py:538-557
 *      imports as `from tn93.tn93 import TN93`): the distance itself.
 *
 * PLAN.md §5.1 cites `dataset.py:443-521, 544-580`; those were the line numbers at the commit the
 * plan was written against. At 61d30e3 the same code is at 493-571 (the function) and 598-636 (the
 * `use_tn93` branch of `load_alignment_and_tree`, mirrored in assemble.js).
 *
 * ============================ WHAT THE REFERENCE ACTUALLY CALLS ============================
 *
 * dataset.py:505 `tn93_bin = shutil.which("tn93")`. If the COMPILED tn93 binary is on PATH the
 * reference uses it (dataset.py:507-537):
 *
 *     tn93 -t 1.0 -l 1 -q -o distances.csv subset.fa
 *
 * i.e. threshold 1.0 (pairs at or above 1.0 are NOT WRITTEN), minimum overlap 1 nucleotide, quiet,
 * CSV out (the binary's default format, columns ID1,ID2,Distance) and the binary's DEFAULT ambiguity
 * strategy, `-a resolve`, with `-g 1.0`. The CSV is read with `csv.DictReader`, each row's Distance
 * is `float()`d into both `[i][j]` and `[j][i]`, and any pair the binary omitted (above the
 * threshold, or below the overlap) stays 0.0 and is caught by the imputation below.
 *
 * Only when the binary is absent (or its run raised) does the Python package run, one pair at a
 * time (dataset.py:538-557):
 *
 *     tn = TN93()                                       # verbose=0, ignore_gaps=False,
 *                                                       # max_ambig_fraction=1.0, minimum_overlap=500
 *     counts   = tn.get_counts(seq_i, seq_j, "resolve")
 *     nuc_freq = tn.get_nucleotide_frequency(counts)
 *     d        = tn.calculate_distance(counts, nuc_freq)
 *     if d is None or d == "-" or d < 0 or np.isnan(d): d = 1.0
 *
 * THE OPTIONS THIS FILE REPRODUCES ARE THE PACKAGE ONES: match mode "resolve", max_ambig_fraction
 * 1.0, ignore_gaps False, and NO minimum-overlap test — the reference calls `get_counts` /
 * `calculate_distance` directly and never goes through `tn93_distance` (tn93.py:152-179), so the
 * package's 500-nucleotide `minimum_overlap` and its `"-"` sentinel are unreachable from here (the
 * `d == "-"` guard at dataset.py:552 is dead code). MEASURED (2026-09-05, tn93 binary v1.0.15 vs
 * tn93 package 1.2.2, both through `compute_tn93_distance_matrix`): the two paths agree EXACTLY —
 * max |Δ| = 0.0 — on examples/bat_oas1.fasta and examples/HIV1_RT.fasta, because the binary writes
 * 6 significant digits (the same rounding the package applies, below) and every pair it drops at
 * the 1.0 threshold is imputed back to exactly the 1.0 the package returns for it. The fixtures are
 * generated with the binary forced off PATH so they are reproducible on any machine.
 *
 * ============================ THE DISTANCE (tn93 1.2.2, tn93/tn93.py) ============================
 *
 * `get_constants` (tn93.py:523-601) — replicated verbatim as MAP_CHARACTER / RESOLUTIONS /
 * RESOLUTION_COUNTS below: 256-entry character map (A C G T = 0..3, U = 4, the ten IUPAC codes
 * 5..15, '?' and everything unmapped = 16, '-' = 17; upper and lower case both), a 18x4 resolution
 * table and a 18-entry 1/|resolutions| weight table (GAP weight 0.0).
 *
 * `get_counts(seq1, seq2, "resolve")` (tn93.py:291-307):
 *   1. `find_terminal_gaps` (235-243) runs and stores first/last non-gap indices. RESOLVE never
 *      reads them (only GAPMM does, and only when ignore_gaps) — replicated as a no-op, EXCEPT for
 *      the Python bug it carries: on a sequence that is ALL gaps `seq1[self.last_nongap]` in its
 *      verbose branch would IndexError; verbose is 0 here, so it never fires.
 *   2. `ambig_fraction_too_high` (275-286): count positions where `can_resolve(nuc1, nuc2)`
 *      (245-273) and positions where NEITHER character is a gap; if
 *      `total_non_gap * max_ambig_fraction <= ambig_count` the pair is scored by AVERAGE instead of
 *      RESOLVE. With max_ambig_fraction = 1.0 that means "every overlapping position is a resolvable
 *      ambiguity" — including the degenerate `0 <= 0` case, so a pair with NO overlapping non-gap
 *      position also takes the average branch. `can_resolve` is False when either character is a gap
 *      and when both are unambiguous; note U (index 4) counts as "ambiguous" there.
 *   3. `get_counts_resolve` (313-381) or `get_counts_average` (383-431) over
 *      `p in range(min(len(seq1), len(seq2)))` — the LONGER sequence is simply truncated, no padding.
 *      RESOLVE: both plain -> counts[n1][n2] += 1; one gap and one not -> counted (only gap/gap is
 *      skipped, tn93.py:331-332 tests `nuc1 == 17 and nuc2 == 17`) — the gap is then treated as an
 *      ambiguity with resolutionsCount 0.0 and RESOLUTIONS all-zero, so nothing is added and the
 *      position silently contributes nothing; one plain and one ambiguous -> if the ambiguity
 *      CONTAINS the plain base, count it as a MATCH (this is what "resolve to minimise distance"
 *      means), else spread `resolutionsCount` over its resolutions; both ambiguous -> if they share
 *      k >= 1 resolutions, add 1/k to each shared DIAGONAL cell, else spread the product of the two
 *      weights over the k1 x k2 cells.
 *      AVERAGE: identical except that ANY gap (`nuc1 == 17 or nuc2 == 17`, 399) skips the position
 *      and there is no match-first shortcut.
 *
 * `get_nucleotide_frequency(counts)` (tn93.py:227-233): freq[j] += counts[j][k] and freq[k] +=
 * counts[j][k] for every (j, k) — row sums PLUS column sums, so `sum(freq)` is twice the total.
 *
 * `calculate_distance(counts, freq)` (tn93.py:181-225), float64 throughout:
 *     total_non_gap = 2 / sum(freq)                        # ZeroDivisionError when sum is 0
 *     AG = (c[0][2] + c[2][0]) * total_non_gap
 *     CT = (c[1][3] + c[3][1]) * total_non_gap
 *     matching = (c[0][0] + c[1][1] + c[2][2] + c[3][3]) * total_non_gap
 *     tv = 1 - (AG + CT + matching)
 *     if 0 in freq:                                        # some base absent from the pair
 *         AG = 1 - 2 * (AG + CT) - tv;  CT = 1 - 2 * tv
 *         dist = -0.5 ln(AG) - 0.25 ln(CT)   if AG > 0 and CT > 0   else   1.0
 *     else:
 *         nucF = freq / sum(freq);  fR = nucF0 + nucF2;  fY = nucF1 + nucF3
 *         K1 = 2 nucF0 nucF2 / fR;  K2 = 2 nucF1 nucF3 / fY
 *         K3 = 2 (fR fY - nucF0 nucF2 fY / fR - nucF1 nucF3 fR / fY)
 *         AG = 1 - AG/K1 - 0.5 tv/fR;  CT = 1 - CT/K2 - 0.5 tv/fY;  tv = 1 - 0.5 tv/fY/fR
 *         dist = round6(-K1 ln(AG) - K2 ln(CT) - K3 ln(tv))
 *     return round6(dist) if dist > -0.0 else 0.0          # `> -0.0` is `> 0.0`
 *
 * BASE FREQUENCIES ARE PER PAIR, from that pair's own counts — never global, never from the whole
 * alignment. GAPS contribute nothing (weight 0). The `0 in freq` branch is the package's own
 * degenerate fallback and is NOT the TN93 formula; it is where the SATURATION SENTINEL 1.0 comes
 * from. MEASURED over every pair of all five bundled alignments: NO pair reaches it — the 1.0
 * entries that do appear in a matrix come from the imputation below. What the examples do produce
 * is distance 0.0 between distinct sequences (66 pairs in HIV1_RT, 8 in RHO; imputed to 1e-4) and
 * between byte-identical ones (1 pair in HIV1_RT, 78 in RHO; imputed to max(1.0, max_d), which is
 * the bug below — `load_alignment_and_tree` prunes those pairs before the matrix is built).
 *
 * `round6` is `self.round` (tn93.py:309-311):
 * `np.format_float_positional(x, precision=6, unique=False, fractional=False, trim="k")` — SIX
 * SIGNIFICANT digits, then `float()`. Here: `Number(x.toPrecision(6))`. numpy rounds half to even
 * and ECMAScript's toPrecision rounds a tie to the larger magnitude; a tie needs the double's exact
 * decimal expansion to stop at the 7th significant digit, which no distance out of `log` reaches.
 * The rounding is why distances are stable across the C++ binary and the Python package.
 *
 * PRECISION: every count and every distance is float64 (Python floats), so `tn93Distance` is
 * float64 and the fixture class is 1e-9 — MEASURED max |Δ| = 0 against the reference on every
 * fixture case and every matrix entry, because that six-digit rounding removes the last bits where
 * the two languages' `log` could differ. The MATRIX is `np.zeros((n, n), dtype=np.float32)`
 * (dataset.py:501), so each stored value is rounded to float32 with Math.fround — matching the
 * dtype `compute_mds_coordinates` then squares and double-centres in float32.
 *
 * ============================ THE MATRIX (dataset.py:493-571) ============================
 *
 *   - n <= 1: the zero matrix, no distances computed (dataset.py:502-503).
 *   - off-diagonal, i < j only, mirrored into [j][i].
 *   - `if d is None or d == "-" or d < 0 or np.isnan(d): d = 1.0` — `calculate_distance` returns
 *     neither None, "-" nor NaN on this path and clamps negatives to 0.0, so this guard is dead;
 *     it is replicated anyway.
 *   - IMPUTATION (dataset.py:559-568), after every pair is scored:
 *         max_d = dist.max() if dist.max() > 0 else 0.1
 *         for i < j:
 *             if dist[i][j] <= 0 and seq_i != seq_j:      dist[i][j] = 1e-4
 *             elif dist[i][j] <= 0 and dist.max() > 0:    dist[i][j] = max(1.0, max_d)
 *     The `elif` is reachable only when the two RAW SEQUENCE STRINGS are byte-identical, so
 *     IDENTICAL SEQUENCES ARE GIVEN THE LARGEST DISTANCE IN THE MATRIX (at least 1.0) — a bug
 *     (task issue list), replicated. `load_alignment_and_tree` prunes identical sequences before
 *     calling this (dataset.py:601-606), so it only fires with `prune_duplicates=False` or when the
 *     duplicate pair survives pruning. `dist.max()` inside the loop is LIVE (it sees the 1e-4
 *     already written), while `max_d` is the value from BEFORE the loop; both are reproduced.
 *     When every pair is 0 (one sequence repeated) nothing is written and the matrix stays zero.
 *   - the diagonal is zeroed last (dataset.py:570).
 *
 * WHAT IT DELIBERATELY DOES NOT DO:
 *   - The compiled binary. There is no subprocess in the library; `tn93DistanceMatrix` is always the
 *     package path. See the measurement above for why that is the same answer.
 *   - The package's `tn93_distance` wrapper (tn93.py:152-179): its 500-nucleotide minimum-overlap
 *     test, its `"-"` sentinel, its SeqRecord/string duck-typing and its CSV/JSON row formatting are
 *     not on the reference's path. `minimum_overlap` is therefore not an option here.
 *   - The other match modes are implemented (`average` is on the reference's path via
 *     `ambig_fraction_too_high`; `gapmm` and `skip` are two dozen lines and are pinned by fixtures)
 *     but only `resolve` is what `dataset.py` asks for. `gapmm` with `ignoreGaps` uses the terminal
 *     gap indices, which is the only place they are read.
 *   - Neighbour joining. PLAN.md D22's display topology (`nj.js`) is a separate module.
 *   - Reading files, printing, or catching the reference's exceptions (below).
 *
 * PYTHON EXCEPTIONS, REPLICATED AS THROWS (dataset.py does not catch them, so a reference run dies):
 *   - `2 / sum(freq)` with an empty count matrix -> ZeroDivisionError. Happens when the two
 *     sequences share no position where both are non-gap.
 *   - `math.log(x <= 0)` -> ValueError("expected a positive input, got ...") in the `0 not in freq`
 *     branch. Happens when the pair is SATURATED (the corrected proportions go non-positive), e.g.
 *     two sequences that disagree everywhere, or two all-N sequences.
 *   Both throw here with the Python exception name in the message. Callers that need the whole
 *   matrix to survive such a pair must catch (diagnostics.js reports it as TN93_SATURATED_PAIRS).
 */

/** `map_character`, tn93.py:524-557: 256 entries, default 16 ('?', resolves to any base). */
const MAP_CHARACTER = (() => {
	const m = new Uint8Array(256).fill(16);
	m[45] = 17; // '-' GAP
	const codes = { A: 0, B: 11, C: 1, D: 12, G: 2, H: 13, K: 9, M: 10, N: 15, R: 5, S: 7, T: 3, U: 4, V: 14, W: 8, Y: 6 };
	for (const [ch, v] of Object.entries(codes)) {
		m[ch.charCodeAt(0)] = v;
		m[ch.toLowerCase().charCodeAt(0)] = v;
	}
	return m;
})();

/** `resolutions`, tn93.py:559-579: which of A, C, G, T each character code stands for. */
const RESOLUTIONS = Object.freeze([
	[1, 0, 0, 0], // A
	[0, 1, 0, 0], // C
	[0, 0, 1, 0], // G
	[0, 0, 0, 1], // T
	[0, 0, 0, 1], // U
	[1, 0, 1, 0], // R
	[0, 1, 0, 1], // Y
	[0, 1, 1, 0], // S
	[1, 0, 0, 1], // W
	[0, 0, 1, 1], // K
	[1, 1, 0, 0], // M
	[0, 1, 1, 1], // B
	[1, 0, 1, 1], // D
	[1, 1, 0, 1], // H
	[1, 1, 1, 0], // V
	[1, 1, 1, 1], // N
	[1, 1, 1, 1], // ?
	[0, 0, 0, 0] // GAP
].map((r) => Object.freeze(r)));

/** `resolutionsCount`, tn93.py:581-600: 1 / |resolutions|, and 0.0 for the gap. */
const RESOLUTION_COUNTS = Object.freeze([
	1.0, 1.0, 1.0, 1.0, 1.0, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 4, 1 / 4, 0.0
]);

/** The tn93 character code of the gap, `map_character[45]` (tn93.py:525). */
const GAP = 17;

/** The ambiguity strategy `dataset.py:544` asks for. */
export const TN93_MATCH_MODE = 'resolve';

/** `TN93(max_ambig_fraction=1.0)`, the constructor default the reference takes (tn93.py:136). */
export const TN93_MAX_AMBIG_FRACTION = 1.0;

/**
 * The value `calculate_distance` returns for a pair it cannot resolve (tn93.py:205, the `0 in
 * nucleotide_frequency` branch) and the floor `compute_tn93_distance_matrix` imputes with
 * (dataset.py:566). A distance EQUAL to it is a saturation sentinel, not a measurement.
 */
export const TN93_SATURATION_SENTINEL = 1.0;

/** dataset.py:562 — what a zero distance between two DIFFERENT sequence strings becomes. */
export const TN93_MIN_POSITIVE_DISTANCE = 1e-4;

/** dataset.py:560 — `max_d` when the whole matrix is still zero. */
export const TN93_FALLBACK_MAX = 0.1;

/**
 * `[self.map_character[ord(c)] for c in seq]`. Python indexes a 256-entry list, so a character
 * above U+00FF raises IndexError there; that is replicated. Encoding each sequence once is the only
 * departure from the reference's per-position `ord()` and changes no value: the inner loops below
 * read the same codes in the same order.
 *
 * @param {string} seq
 * @returns {Uint8Array}
 */
export function encodeSequence(seq) {
	const out = new Uint8Array(seq.length);
	for (let i = 0; i < seq.length; i++) {
		const c = seq.charCodeAt(i);
		if (c > 255) throw new Error(`tn93: IndexError: list index out of range (map_character[ord('${seq[i]}')] = map_character[${c}])`);
		out[i] = MAP_CHARACTER[c];
	}
	return out;
}

/** `TN93.round`, tn93.py:309-311: six SIGNIFICANT digits. */
function round6(x) {
	return Number(x.toPrecision(6));
}

/** `math.log`, which raises instead of returning NaN / -inf. */
function pyLog(x) {
	if (!(x > 0)) {
		throw new Error(`tn93: ValueError: expected a positive input, got ${x} (math.log, tn93.py:203/219)`);
	}
	return Math.log(x);
}

/**
 * `TN93.can_resolve(nuc1, nuc2)`, tn93.py:245-273. True when the two character codes can be
 * reconciled: never for a gap, never for two plain bases.
 *
 * @param {number} n1 character code 0..17
 * @param {number} n2 character code 0..17
 * @returns {boolean}
 */
export function canResolve(n1, n2) {
	if (n1 < 4 && n2 < 4) return false;
	if (n1 === GAP || n2 === GAP) return false;
	if (n1 < 4) {
		if (RESOLUTIONS[n2][n1]) return true;
		for (let j = 0; j < 4; j++) if (RESOLUTIONS[n2][j] && RESOLUTIONS[n1][j]) return true;
	} else if (n2 < 4) {
		if (RESOLUTIONS[n1][n2]) return true;
		for (let j = 0; j < 4; j++) if (RESOLUTIONS[n1][j] && RESOLUTIONS[n2][j]) return true;
	} else {
		const norm = RESOLUTION_COUNTS[n1] * RESOLUTION_COUNTS[n2];
		if (norm > 0.0) {
			for (let j = 0; j < 4; j++) if (RESOLUTIONS[n1][j] && RESOLUTIONS[n2][j]) return true;
		}
		for (let j = 0; j < 4; j++) {
			if (RESOLUTIONS[n1][j]) {
				for (let k = 0; k < 4; k++) if (RESOLUTIONS[n2][k]) return true;
			}
		}
	}
	return false;
}

/**
 * `TN93.ambig_fraction_too_high(seq1, seq2)`, tn93.py:275-286: `total_non_gap * max_ambig_fraction
 * <= ambig_count`, which for the reference's max_ambig_fraction of 1.0 also fires on the degenerate
 * `0 <= 0` (no overlapping non-gap position at all).
 *
 * @param {string} seq1
 * @param {string} seq2
 * @param {number} [maxAmbigFraction]
 * @returns {boolean}
 */
export function ambigFractionTooHigh(seq1, seq2, maxAmbigFraction = TN93_MAX_AMBIG_FRACTION) {
	return ambigFractionTooHighEnc(encodeSequence(seq1), encodeSequence(seq2), maxAmbigFraction);
}

/** `ambig_fraction_too_high` on encoded sequences. */
function ambigFractionTooHighEnc(e1, e2, maxAmbigFraction) {
	let ambig = 0;
	let totalNonGap = 0;
	const len = Math.min(e1.length, e2.length);
	for (let i = 0; i < len; i++) {
		const n1 = e1[i];
		const n2 = e2[i];
		if (canResolve(n1, n2)) ambig++;
		if (n1 !== GAP && n2 !== GAP) totalNonGap++;
	}
	return totalNonGap * maxAmbigFraction <= ambig;
}

/**
 * `find_terminal_gaps`, tn93.py:235-243, on encoded sequences: `re.match(r"-*").end()` is the number
 * of leading gaps and `re.search(r"-*$").start()` the index after the last non-gap; only '-' maps to
 * the GAP code, so the encoded scan sees the same positions.
 */
function terminalGaps(e1, e2) {
	const lead = (/** @type {Uint8Array} */ e) => {
		let i = 0;
		while (i < e.length && e[i] === GAP) i++;
		return i;
	};
	const trail = (/** @type {Uint8Array} */ e) => {
		let i = e.length;
		while (i > 0 && e[i - 1] === GAP) i--;
		return i;
	};
	// Python: first_nongap = max(s1, s2); last_nongap = min(e1, e2) - 1.
	return { first: Math.max(lead(e1), lead(e2)), last: Math.min(trail(e1), trail(e2)) - 1 };
}

const zeroCounts = () => [
	[0, 0, 0, 0],
	[0, 0, 0, 0],
	[0, 0, 0, 0],
	[0, 0, 0, 0]
];

/** `get_counts_resolve`, tn93.py:313-381. */
function countsResolve(e1, e2) {
	const counts = zeroCounts();
	const len = Math.min(e1.length, e2.length);
	for (let p = 0; p < len; p++) {
		const n1 = e1[p];
		const n2 = e2[p];
		if (n1 < 4 && n2 < 4) {
			counts[n1][n2] += 1;
			continue;
		}
		if (n1 === GAP && n2 === GAP) continue;
		if (n1 < 4) {
			if (RESOLUTION_COUNTS[n2] > 0) {
				if (RESOLUTIONS[n2][n1]) {
					counts[n1][n1] += 1;
					continue;
				}
				for (let j = 0; j < 4; j++) if (RESOLUTIONS[n2][j]) counts[n1][j] += RESOLUTION_COUNTS[n2];
			}
		} else if (n2 < 4) {
			if (RESOLUTION_COUNTS[n1] > 0) {
				if (RESOLUTIONS[n1][n2]) {
					counts[n2][n2] += 1;
					continue;
				}
				for (let j = 0; j < 4; j++) if (RESOLUTIONS[n1][j]) counts[j][n2] += RESOLUTION_COUNTS[n1];
			}
		} else {
			const norm = RESOLUTION_COUNTS[n1] * RESOLUTION_COUNTS[n2];
			if (norm > 0.0) {
				let matched = 0;
				const positive = [false, false, false, false];
				for (let j = 0; j < 4; j++) {
					if (RESOLUTIONS[n1][j] && RESOLUTIONS[n2][j]) {
						matched += 1;
						positive[j] = true;
					}
				}
				if (matched > 0) {
					const norm2 = 1 / matched;
					for (let j = 0; j < 4; j++) if (positive[j]) counts[j][j] += norm2;
					continue;
				}
				for (let j = 0; j < 4; j++) {
					if (RESOLUTIONS[n1][j]) {
						for (let k = 0; k < 4; k++) if (RESOLUTIONS[n2][k]) counts[j][k] += norm;
					}
				}
			}
		}
	}
	return counts;
}

/** `get_counts_average`, tn93.py:383-431. */
function countsAverage(e1, e2) {
	const counts = zeroCounts();
	const len = Math.min(e1.length, e2.length);
	for (let p = 0; p < len; p++) {
		const n1 = e1[p];
		const n2 = e2[p];
		if (n1 < 4 && n2 < 4) {
			counts[n1][n2] += 1;
			continue;
		}
		if (n1 === GAP || n2 === GAP) continue;
		if (n1 < 4) {
			if (RESOLUTION_COUNTS[n2] > 0) {
				for (let j = 0; j < 4; j++) if (RESOLUTIONS[n2][j]) counts[n1][j] += RESOLUTION_COUNTS[n2];
			}
		} else if (n2 < 4) {
			if (RESOLUTION_COUNTS[n1] > 0) {
				for (let j = 0; j < 4; j++) if (RESOLUTIONS[n1][j]) counts[j][n2] += RESOLUTION_COUNTS[n1];
			}
		} else {
			const norm = RESOLUTION_COUNTS[n1] * RESOLUTION_COUNTS[n2];
			if (norm > 0.0) {
				for (let j = 0; j < 4; j++) {
					if (RESOLUTIONS[n1][j]) {
						for (let k = 0; k < 4; k++) if (RESOLUTIONS[n2][k]) counts[j][k] += norm;
					}
				}
			}
		}
	}
	return counts;
}

/** `get_counts_gapmm`, tn93.py:433-490: a gap facing a base is treated as an N (code 15). */
function countsGapmm(e1, e2, ignoreGaps) {
	const counts = zeroCounts();
	const len = Math.min(e1.length, e2.length);
	const { first, last } = terminalGaps(e1, e2);
	const start = ignoreGaps ? first : 0;
	const end = ignoreGaps ? last : len;
	for (let p = start; p < end; p++) {
		let n1 = e1[p];
		let n2 = e2[p];
		if (n1 < 4 && n2 < 4) {
			counts[n1][n2] += 1;
			continue;
		}
		if (n1 === GAP || n2 === GAP) {
			if (n1 === GAP && n2 === GAP) continue;
			if (n1 === GAP) n1 = 15;
			else n2 = 15;
		}
		if (n1 < 4) {
			if (RESOLUTION_COUNTS[n2] > 0) {
				for (let j = 0; j < 4; j++) if (RESOLUTIONS[n2][j]) counts[n1][j] += RESOLUTION_COUNTS[n2];
			}
		} else if (n2 < 4) {
			if (RESOLUTION_COUNTS[n1] > 0) {
				for (let j = 0; j < 4; j++) if (RESOLUTIONS[n1][j]) counts[j][n2] += RESOLUTION_COUNTS[n1];
			}
		} else {
			const norm = RESOLUTION_COUNTS[n1] * RESOLUTION_COUNTS[n2];
			if (norm > 0.0) {
				for (let j = 0; j < 4; j++) {
					if (RESOLUTIONS[n1][j]) {
						for (let k = 0; k < 4; k++) if (RESOLUTIONS[n2][k]) counts[j][k] += norm;
					}
				}
			}
		}
	}
	return counts;
}

/** `get_counts_skip`, tn93.py:492-518: only positions where both characters are plain bases. */
function countsSkip(e1, e2) {
	const counts = zeroCounts();
	const len = Math.min(e1.length, e2.length);
	for (let p = 0; p < len; p++) {
		const n1 = e1[p];
		const n2 = e2[p];
		if (n1 < 4 && n2 < 4) counts[n1][n2] += 1;
	}
	return counts;
}

/**
 * `TN93.get_counts(seq1, seq2, match_mode)`, tn93.py:291-307: the 4x4 pairwise count matrix
 * [A, C, G, T] x [A, C, G, T], float64. `resolve` falls back to `average` when
 * `ambig_fraction_too_high` (see the header).
 *
 * @param {string} seq1
 * @param {string} seq2
 * @param {string} [matchMode] 'resolve' | 'average' | 'gapmm' | 'skip' (case-insensitive)
 * @param {{maxAmbigFraction?: number, ignoreGaps?: boolean}} [options]
 * @returns {number[][]} 4 x 4
 */
export function tn93Counts(seq1, seq2, matchMode = TN93_MATCH_MODE, options = {}) {
	return tn93CountsEnc(encodeSequence(seq1), encodeSequence(seq2), matchMode, options);
}

/** `get_counts` on already-encoded sequences (what `tn93DistanceMatrix` reuses across pairs). */
function tn93CountsEnc(e1, e2, matchMode, options) {
	const { maxAmbigFraction = TN93_MAX_AMBIG_FRACTION, ignoreGaps = false } = options;
	const mode = String(matchMode).toUpperCase();
	if (mode === 'RESOLVE') {
		return ambigFractionTooHighEnc(e1, e2, maxAmbigFraction) ? countsAverage(e1, e2) : countsResolve(e1, e2);
	}
	if (mode === 'AVERAGE') return countsAverage(e1, e2);
	if (mode === 'GAPMM') return countsGapmm(e1, e2, ignoreGaps);
	if (mode === 'SKIP') return countsSkip(e1, e2);
	// tn93.py:304-306 logs and calls sys.exit(1).
	throw new Error(`tn93: Match mode ${matchMode} is not recognized`);
}

/**
 * `TN93.get_nucleotide_frequency(pairwise_counts)`, tn93.py:227-233: row sums PLUS column sums, so
 * the total is counted twice.
 *
 * @param {number[][]} counts 4 x 4
 * @returns {number[]} 4 entries
 */
export function tn93NucleotideFrequency(counts) {
	const freq = [0, 0, 0, 0];
	for (let j = 0; j < 4; j++) {
		for (let k = 0; k < 4; k++) {
			freq[j] += counts[j][k];
			freq[k] += counts[j][k];
		}
	}
	return freq;
}

/**
 * `TN93.calculate_distance(pairwise_counts, nucleotide_frequency)`, tn93.py:181-225, float64.
 * Throws where Python raises (see the header): ZeroDivisionError on an empty count matrix,
 * ValueError from `math.log` on a saturated pair.
 *
 * @param {number[][]} counts 4 x 4
 * @param {number[]} freq 4 entries
 * @returns {number}
 */
export function tn93CalculateDistance(counts, freq) {
	let dist = 0;
	const freqSum = freq[0] + freq[1] + freq[2] + freq[3];
	if (freqSum === 0) {
		throw new Error('tn93: ZeroDivisionError: division by zero (tn93.py:190, the pair has no overlapping non-gap position)');
	}
	const totalNonGap = 2 / freqSum;
	let AG = (counts[0][2] + counts[2][0]) * totalNonGap;
	let CT = (counts[1][3] + counts[3][1]) * totalNonGap;
	const matching = (counts[0][0] + counts[1][1] + counts[2][2] + counts[3][3]) * totalNonGap;
	let tv = 1 - (AG + CT + matching);

	if (freq[0] === 0 || freq[1] === 0 || freq[2] === 0 || freq[3] === 0) {
		// `if 0 in nucleotide_frequency` (tn93.py:199-205): not the TN93 formula, a fallback.
		AG = 1 - 2 * (AG + CT) - tv;
		CT = 1 - 2 * tv;
		dist = AG > 0 && CT > 0 ? -0.5 * pyLog(AG) - 0.25 * pyLog(CT) : TN93_SATURATION_SENTINEL;
	} else {
		const auxd = 1 / freqSum;
		const nucF = [freq[0] * auxd, freq[1] * auxd, freq[2] * auxd, freq[3] * auxd];
		const fR = nucF[0] + nucF[2];
		const fY = nucF[1] + nucF[3];
		const K1 = (2 * nucF[0] * nucF[2]) / fR;
		const K2 = (2 * nucF[1] * nucF[3]) / fY;
		const K3 = 2 * (fR * fY - (nucF[0] * nucF[2] * fY) / fR - (nucF[1] * nucF[3] * fR) / fY);
		AG = 1 - AG / K1 - (0.5 * tv) / fR;
		CT = 1 - CT / K2 - (0.5 * tv) / fY;
		tv = 1 - 0.5 * tv / fY / fR;
		dist = round6(-K1 * pyLog(AG) - K2 * pyLog(CT) - K3 * pyLog(tv));
	}
	// tn93.py:225 `return self.round(dist) if dist > -0.0 else 0.0`.
	return dist > -0.0 ? round6(dist) : 0.0;
}

/**
 * One pairwise TN93 distance, float64, exactly as `dataset.py:544-547` composes the package:
 * `get_counts` -> `get_nucleotide_frequency` -> `calculate_distance`. The package's
 * `tn93_distance` wrapper (with its minimum-overlap test) is deliberately not used.
 *
 * @param {string} seqA aligned nucleotide string
 * @param {string} seqB aligned nucleotide string
 * @param {{matchMode?: string, maxAmbigFraction?: number, ignoreGaps?: boolean}} [options]
 * @returns {number}
 */
export function tn93Distance(seqA, seqB, options = {}) {
	return tn93DistanceEnc(encodeSequence(seqA), encodeSequence(seqB), options);
}

/** `tn93Distance` on encoded sequences. */
function tn93DistanceEnc(e1, e2, options) {
	const counts = tn93CountsEnc(e1, e2, options.matchMode ?? TN93_MATCH_MODE, options);
	return tn93CalculateDistance(counts, tn93NucleotideFrequency(counts));
}

/**
 * `compute_tn93_distance_matrix(seq_dict, taxa)`, dataset.py:493-571: the float32 [n, n] matrix,
 * row-major, with the reference's dead guard, imputation rules and zeroed diagonal.
 *
 * @param {Map<string, string>|Record<string, string>} sequences taxon -> aligned sequence
 * @param {string[]} taxa the rows/columns, in order
 * @param {{matchMode?: string, maxAmbigFraction?: number, ignoreGaps?: boolean,
 *   pairwiseDistances?: (seqs: string[], taxa: string[]) => ArrayLike<number>}} [options]
 *   `pairwiseDistances` replaces the per-pair computation with an external engine's raw numbers,
 *   read at [i * n + j]; the rounding, the sentinel and dataset.py's imputation still happen here.
 * @returns {Float32Array} length taxa.length ** 2
 */
export function tn93DistanceMatrix(sequences, taxa, options = {}) {
	const get = sequences instanceof Map ? (/** @type {string} */ t) => sequences.get(t) : (/** @type {string} */ t) => sequences[t];
	const n = taxa.length;
	const dist = new Float32Array(n * n);
	if (n <= 1) return dist;

	const seqs = taxa.map((t) => {
		const s = get(t);
		if (typeof s !== 'string') throw new Error(`tn93DistanceMatrix: no sequence for taxon '${t}'`);
		return s;
	});
	// An external provider (the compiled tn93 through WebAssembly, dataset.py:507-537's branch when
	// the binary is on PATH) may supply the RAW pairwise distances; everything downstream of them —
	// the dead guard, float32 rounding, the sentinel and dataset.py's imputation — stays here, so
	// the two engines can differ only in the numbers they compute, never in what is done with them.
	// The callback takes the sequences in `taxa` order and returns an n*n array-like read at
	// [i * n + j]; `null` or `undefined` for a pair means "not computed", the same as the package's
	// own saturation answer, TN93_SATURATION_SENTINEL.
	const provider = typeof options.pairwiseDistances === 'function' ? options.pairwiseDistances(seqs, taxa) : null;
	const encoded = provider ? null : seqs.map(encodeSequence);

	let liveMax = 0;
	for (let i = 0; i < n; i++) {
		for (let j = i + 1; j < n; j++) {
			let d = provider ? provider[i * n + j] : tn93DistanceEnc(encoded[i], encoded[j], options);
			// dataset.py:552 `if d is None or d == "-" or d < 0 or np.isnan(d): d = 1.0` (dead).
			if (d === null || d === undefined || d < 0 || Number.isNaN(d)) d = TN93_SATURATION_SENTINEL;
			const f = Math.fround(d);
			dist[i * n + j] = f;
			dist[j * n + i] = f;
			if (f > liveMax) liveMax = f;
		}
	}

	// dataset.py:560 `max_d = float(dist_mat.max()) if dist_mat.max() > 0 else 0.1`, taken ONCE.
	const maxD = liveMax > 0 ? liveMax : TN93_FALLBACK_MAX;
	for (let i = 0; i < n; i++) {
		for (let j = i + 1; j < n; j++) {
			const d = dist[i * n + j];
			if (d <= 0.0 && seqs[i] !== seqs[j]) {
				const v = Math.fround(TN93_MIN_POSITIVE_DISTANCE);
				dist[i * n + j] = v;
				dist[j * n + i] = v;
				// `dist_mat.max()` in the elif is re-read from the live matrix, so track it.
				if (v > liveMax) liveMax = v;
			} else if (d <= 0.0 && liveMax > 0) {
				const v = Math.fround(Math.max(TN93_SATURATION_SENTINEL, maxD));
				dist[i * n + j] = v;
				dist[j * n + i] = v;
				if (v > liveMax) liveMax = v;
			}
		}
	}
	for (let i = 0; i < n; i++) dist[i * n + i] = 0.0;
	return dist;
}

/**
 * How many unordered pairs sit exactly at the saturation sentinel (PLAN.md D22's report line, and
 * diagnostics.js's TN93_SATURATED_PAIRS). Equality is exact: those pairs were never measured —
 * they come from `calculate_distance`'s `0 in nucleotide_frequency` fallback, from the reference's
 * `d < 0 or isnan` guard, or from the `max(1.0, max_d)` imputation. A pair whose TN93 distance
 * happens to exceed 1.0 by measurement is NOT counted.
 *
 * @param {ArrayLike<number>} dist row-major n x n
 * @param {number} n
 * @param {number} [sentinel]
 * @returns {number}
 */
export function tn93SaturatedPairs(dist, n, sentinel = TN93_SATURATION_SENTINEL) {
	let count = 0;
	for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (dist[i * n + j] === sentinel) count++;
	return count;
}

/** The raw tables, exported so the fixtures can pin them (tn93.py:523-601). */
export const TN93_TABLES = Object.freeze({
	mapCharacter: MAP_CHARACTER,
	resolutions: RESOLUTIONS,
	resolutionCounts: RESOLUTION_COUNTS,
	gap: GAP
});
