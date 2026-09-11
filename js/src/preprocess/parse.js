/**
 * WHY THIS FILE EXISTS
 *
 * Mirrors `parse_alignment_sequences` of `hyphaeon/dataset.py:252-364` at veg/HyphAeon reconcile/phase-5a —
 * the FASTA / PHYLIP / NEXUS alignment reader — on a string the caller has already read. The
 * three format branches, in the reference's detection order:
 *   1. PHYLIP (lines 68-93): first non-empty line is exactly two integer tokens. Then, per
 *      non-empty line, "a name-only line" (<= 35 chars, no '-' or ' ', alphanumeric or containing
 *      '_'), "a name + sequence line" (two whitespace-split parts, name <= 35 chars and alphanumeric
 *      or containing '_', sequence part LONGER THAN 10 characters), else a continuation. Accepted
 *      only if at least `ntaxa` records came out; otherwise fall through.
 *   2. FASTA (lines 95-115): text starts with '>'. Name = first whitespace token after '>', with
 *      surrounding quotes stripped. Parsing STOPS at the first line starting with '(' or (case-
 *      insensitively) 'tree ' or 'begin ' — that is how an embedded tree is skipped.
 *   3. NEXUS (lines 117-159): `[...]` comments removed; TAXLABELS (quoted or bare tokens), FORMAT
 *      ... NOLABELS, MATRIX ... ; with either positional rows or `name sequence` rows (a repeated
 *      name APPENDS — interleaved matrices); without a MATRIX block, any `name sequence` line whose
 *      sequence part is longer than 20 characters (overwrite, not append).
 * Every sequence is upper-cased and then 'U' -> 'T' (so lowercase 'u' also becomes 'T'). The
 * result is an insertion-ordered Map, the JS shape of the Python dict.
 *
 * PYTHON STRING SEMANTICS ARE REPRODUCED, NOT APPROXIMATED, because format detection hangs on them:
 * `str.strip()`/`split()`/`splitlines()` treat a different whitespace set from JS `trim()`/`\s`
 * (Python: Unicode White_Space plus U+001C..U+001F, and NOT U+FEFF; JS `\s`: no U+0085 and no
 * U+001C..1F, but U+FEFF). A byte-order mark at the top of a FASTA file, for instance, means the
 * Python does NOT see a leading '>' and falls to the NEXUS branch. The `py*` helpers below are the
 * reference's semantics and are exported for tree.js, which needs the same ones.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: no file or gzip handling (dataset.py:256-259). The runtime reads
 * and decompresses; this takes text.
 *
 * QUIRKS REPLICATED ON PURPOSE (dataset.py bugs, pinned by fixtures/dataset/parse_alignment_sequences.json):
 *   - PHYLIP multi-line records parse into garbage: an alphanumeric continuation line <= 35 chars
 *     is taken as a taxon name, and a `name seq` line with <= 10 sequence characters is not a
 *     record start (synthetic_phylip_sequential / _interleaved cases).
 *   - A quoted NEXUS matrix label containing a space ('sp one') is split on whitespace: name "sp",
 *     sequence starting with "ONE'" (synthetic_nexus_labels case).
 *   - A FASTA header line that is just ">" raises IndexError in Python (`split()[0]` on an empty
 *     list); this throws the same way.
 *
 * There was no DataMonkey 3 counterpart: DM3 parsed alignments elsewhere and handed
 * `{names, sequences}` to its assemble step.
 */

/**
 * The characters `str.isspace()` is true for in CPython: Unicode White_Space plus the four
 * information separators U+001C..U+001F. U+FEFF is NOT whitespace to Python.
 */
const PY_WS_CHARS =
	'\\t\\n\\v\\f\\r \\x1c\\x1d\\x1e\\x1f\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000';

/** `[ \s ]` and `[^\s]` in Python's Unicode sense, as regex source fragments. */
export const PY_WS = `[${PY_WS_CHARS}]`;
export const PY_NOT_WS = `[^${PY_WS_CHARS}]`;

const RE_STRIP = new RegExp(`^${PY_WS}+|${PY_WS}+$`, 'gu');
const RE_RSTRIP = new RegExp(`${PY_WS}+$`, 'u');
const RE_LSTRIP = new RegExp(`^${PY_WS}+`, 'u');
const RE_WS_RUN = new RegExp(`${PY_WS}+`, 'gu');
/** `str.splitlines()` boundaries: \n \r \r\n \v \f U+001C U+001D U+001E U+0085 U+2028 U+2029. */
const RE_LINE_BREAK = /\r\n|[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]/u;

/** `s.strip()` with no argument. */
export function pyStrip(s) {
	return s.replace(RE_STRIP, '');
}

/** `s.rstrip()` with no argument. */
export function pyRstrip(s) {
	return s.replace(RE_RSTRIP, '');
}

/** `s.lstrip()` with no argument. */
export function pyLstrip(s) {
	return s.replace(RE_LSTRIP, '');
}

/**
 * `s.strip(chars)`: remove any run of the given characters from both ends.
 *
 * @param {string} s
 * @param {string} chars
 */
export function pyStripChars(s, chars) {
	let a = 0;
	let b = s.length;
	while (a < b && chars.includes(s[a])) a++;
	while (b > a && chars.includes(s[b - 1])) b--;
	return s.slice(a, b);
}

/** `s.split()` with no argument: runs of whitespace, ends ignored, empty string -> []. */
export function pySplit(s) {
	const t = pyStrip(s);
	return t === '' ? [] : t.split(RE_WS_RUN);
}

/**
 * `s.split(None, 1)`: at most one split, on the first whitespace run after leading whitespace; the
 * remainder keeps its trailing whitespace. Returns 0, 1 or 2 parts.
 *
 * @param {string} s
 * @returns {string[]}
 */
export function pySplit1(s) {
	const t = pyLstrip(s);
	if (t === '') return [];
	const m = new RegExp(`${PY_WS}+`, 'u').exec(t);
	if (!m || m.index === undefined) return [t];
	const first = t.slice(0, m.index);
	const rest = t.slice(m.index + m[0].length);
	return rest === '' ? [first] : [first, rest];
}

/** `s.splitlines()`: no trailing empty element for a final line break. */
export function pySplitLines(s) {
	const parts = s.split(RE_LINE_BREAK);
	if (parts.length && parts[parts.length - 1] === '') parts.pop();
	return parts;
}

/** `s.isalnum()`: non-empty and every character a Unicode letter or number. */
export function pyIsAlnum(s) {
	return /^[\p{L}\p{N}]+$/u.test(s);
}

/**
 * `s.isdigit()` for the practical case: Unicode decimal digits. (CPython also accepts a few
 * Numeric_Type=Digit characters such as superscripts, on which `int()` then raises; not modelled.)
 */
export function pyIsDigit(s) {
	return /^\p{Nd}+$/u.test(s);
}

/** `len(s)` in code points, not UTF-16 units. */
export function pyLen(s) {
	let n = 0;
	for (const _ of s) n++; // eslint-disable-line no-unused-vars
	return n;
}

/** `int(s)` for a string `pyIsDigit` accepted (any Unicode decimal digits). */
function pyInt(s) {
	let v = 0;
	for (const ch of s) {
		const d = Number.parseInt(ch, 10);
		v = v * 10 + (Number.isNaN(d) ? digitValue(ch) : d);
	}
	return v;
}

/** Decimal value of a non-ASCII Unicode Nd digit: the offset from its block's zero. */
function digitValue(ch) {
	const cp = ch.codePointAt(0) ?? 0;
	for (let z = cp; z > cp - 10; z--) {
		if (!/\p{Nd}/u.test(String.fromCodePoint(z - 1))) return cp - z;
	}
	return cp % 10;
}

/** `.upper().replace('U', 'T')`, the finishing step applied to every sequence. */
function finishSeq(s) {
	return s.toUpperCase().replaceAll('U', 'T');
}

/**
 * `parse_alignment_sequences`, dataset.py:252-364, on already-read text.
 *
 * @param {string} fullText the alignment file's content (already decompressed)
 * @returns {Map<string, string>} name -> upper-cased sequence with U -> T, in file order
 */
export function parseAlignmentSequences(fullText) {
	const text = String(fullText);

	// 1. PHYLIP / sequential / interleaved (starts with an 'ntaxa nsites' header) — lines 68-93.
	const linesNonEmpty = pySplitLines(text)
		.map((l) => pyStrip(l))
		.filter((l) => l !== '');
	if (linesNonEmpty.length) {
		const firstTokens = pySplit(linesNonEmpty[0]);
		if (firstTokens.length === 2 && pyIsDigit(firstTokens[0]) && pyIsDigit(firstTokens[1])) {
			const seqDict = new Map();
			let currName = null;
			let currSeq = [];
			for (const line of linesNonEmpty.slice(1)) {
				const parts = pySplit1(line);
				if (
					pyLen(line) <= 35 &&
					!(line.includes('-') || line.includes(' ')) &&
					(pyIsAlnum(line) || line.includes('_'))
				) {
					if (currName) seqDict.set(currName, finishSeq(currSeq.join('')));
					currName = line;
					currSeq = [];
				} else if (
					parts.length === 2 &&
					(pyIsAlnum(parts[0]) || parts[0].includes('_')) &&
					pyLen(parts[0]) <= 35 &&
					pyLen(parts[1]) > 10
				) {
					if (currName) seqDict.set(currName, finishSeq(currSeq.join('')));
					currName = parts[0];
					currSeq = [parts[1].replaceAll(' ', '')];
				} else {
					currSeq.push(line.replaceAll(' ', ''));
				}
			}
			if (currName) seqDict.set(currName, finishSeq(currSeq.join('')));
			if (seqDict.size && seqDict.size >= pyInt(firstTokens[0])) return seqDict;
		}
	}

	// 2. FASTA — lines 95-115.
	if (pyStrip(text).startsWith('>')) {
		const seqDict = new Map();
		let currId = null;
		let currChunks = [];
		for (const line of pySplitLines(text)) {
			const lStrip = pyStrip(line);
			if (!lStrip) continue;
			if (lStrip.startsWith('>')) {
				if (currId !== null) seqDict.set(currId, finishSeq(currChunks.join('')));
				const toks = pySplit(lStrip.slice(1));
				if (toks.length === 0) {
					// dataset.py:312 `l_strip[1:].split()[0]` on an empty list.
					throw new Error(
						"parseAlignmentSequences: FASTA header line with no name (dataset.py:312 raises IndexError: list index out of range)"
					);
				}
				currId = pyStripChars(toks[0], '\'"');
				currChunks = [];
			} else {
				const lower = lStrip.toLowerCase();
				if (lStrip.startsWith('(') || lower.startsWith('tree ') || lower.startsWith('begin ')) {
					break;
				}
				currChunks.push(lStrip.replaceAll(' ', ''));
			}
		}
		if (currId !== null) seqDict.set(currId, finishSeq(currChunks.join('')));
		return seqDict;
	}

	// 3. NEXUS — lines 117-159.
	const cleanNexusText = text.replace(/\[[^\]]*\]/g, '');
	const taxlabels = [];
	const taxMatch = new RegExp(`taxlabels${PY_WS}+([\\s\\S]*?)${PY_WS}*;`, 'iu').exec(cleanNexusText);
	if (taxMatch) {
		const tokenRe = new RegExp(`'([^']+)'|"([^"]+)"|(${PY_NOT_WS}+)`, 'gu');
		for (const m of taxMatch[1].matchAll(tokenRe)) {
			const name = m[1] || m[2] || m[3];
			if (name) taxlabels.push(pyStrip(name));
		}
	}

	const formatMatch = new RegExp(`format${PY_WS}+([\\s\\S]*?)${PY_WS}*;`, 'iu').exec(cleanNexusText);
	const isNolabels = Boolean(formatMatch && formatMatch[1].toLowerCase().includes('nolabels'));

	const matrixMatch = new RegExp(`matrix${PY_WS}+([\\s\\S]*?)${PY_WS}*;`, 'iu').exec(cleanNexusText);
	const seqDict = new Map();
	if (matrixMatch) {
		const matrixLines = pySplitLines(matrixMatch[1])
			.map((l) => pyStrip(l))
			.filter((l) => l !== '');
		if (isNolabels && taxlabels.length) {
			matrixLines.forEach((line, idx) => {
				if (idx < taxlabels.length) {
					seqDict.set(taxlabels[idx], finishSeq(line.replaceAll(' ', '').replaceAll('\t', '')));
				}
			});
		} else {
			for (const line of matrixLines) {
				const parts = pySplit1(line);
				if (parts.length === 2) {
					const name = pyStrip(parts[0].replaceAll("'", '').replaceAll('"', ''));
					const seq = finishSeq(pyStrip(parts[1].replaceAll(' ', '').replaceAll('\t', '')));
					if (seqDict.has(name)) seqDict.set(name, seqDict.get(name) + seq);
					else seqDict.set(name, seq);
				}
			}
		}
	} else {
		// Fallback for plain matrices without an explicit block wrapper (lines 151-157).
		for (const line of pySplitLines(cleanNexusText)) {
			const parts = pySplit1(pyStrip(line));
			if (parts.length === 2 && pyLen(parts[1].replaceAll(' ', '')) > 20) {
				const name = pyStrip(parts[0].replaceAll("'", '').replaceAll('"', ''));
				seqDict.set(name, finishSeq(parts[1].replaceAll(' ', '')));
			}
		}
	}

	return seqDict;
}
