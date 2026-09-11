/**
 * WHY THIS FILE EXISTS
 *
 * Mirrors hyphaeon/evaluation.py (at reconcile/phase-5a, 667 lines), the pooled evaluation of `hyphaeon meme`
 * results against HyPhy MEME: the two loaders, gene pairing, the pooled metrics and the text report.
 * PLAN.md §5.1 lists it as `evaluate`, parity class "exact to 1e-9"; the fixtures under
 * fixtures/evaluation/ replay every public function and the two error cases.
 *
 * WHAT IT MIRRORS (function -> Python lines)
 *
 *   EvaluationError          28-29
 *   loadPredictionCsv        46-108   (_finite_float, _probability, _site_number, _boolean inlined
 *                                      as pyFloat/probability/siteNumber/boolean below)
 *   loadMemeJson             111-218  (_normalized_header, _column_index, _partition_key,
 *                                      _flatten_coverage)
 *   matchGeneFiles           221-260  (_files_by_gene)
 *   correlations             263-272
 *   rocAuc                   275-283  (re-exported from the numeric kernel; see below)
 *   confusion / ppv / fpr    286-302
 *   thresholdMetrics         305-327
 *   evaluatePairs            330-449  (_evaluate_pairs)
 *   evaluateDirectories      452-483
 *   evaluateFiles            486-527  (_gene_name_from_file)
 *   formatTextReport         530-553
 *
 * THE ONE SUBSTITUTION: THE FILE SYSTEM. evaluation.py reads paths; this library reads nothing, so
 * every "path" becomes a `{name, text}` pair (text = the file's decoded content, or for MEME an
 * already-parsed object under `data`) and every directory becomes a list of such pairs. `name` is
 * what the Python's `path` would print in a message, so the error strings — which the fixtures
 * compare exactly, with absolute paths reduced to basenames — come out identical when the caller
 * passes basenames. `evaluate_directories` resolves the directory paths into the report
 * (`prediction_directory`, `meme_directory`); here the caller supplies those labels.
 *
 * PYTHON SEMANTICS REPRODUCED, because the loaders are parsers and the report is exact:
 *   - `csv.DictReader` over a file opened with newline="" and encoding utf-8-sig: the excel
 *     dialect state machine of _csv.c (quoted fields, doubled quotes, text after a closing quote
 *     appended, \r, \n and \r\n record ends, blank lines as empty records that DictReader skips,
 *     a header read that does NOT skip a leading blank line, short rows padded with None, long
 *     rows dumped under key None, a leading BOM stripped, an unterminated quote closed at EOF),
 *     with `line_number` counting yielded rows from 2 as `enumerate(reader, start=2)` does.
 *   - `float(str)`: C-isspace plus non-ASCII Unicode whitespace stripped (NOT \x1c..\x1f, which
 *     str.strip() removes — measured: float("\x1c8\x1f") raises, float("  7  ") is 7.0),
 *     underscores between digits, `.5`, `5.`, unicode decimal digits, inf/infinity/nan in any
 *     case; `float(True)` = 1.0; None and lists raise.
 *   - `int(str)` for partition keys: the same whitespace, sign, unicode digits, underscores; no
 *     dot, no exponent.
 *   - `html.unescape` inside `_normalized_header` (evaluation.py:111-114): numeric character
 *     references with the HTML5 invalid-charref and invalid-codepoint tables, named references
 *     against the full HTML5 table (2231 names, including the 106 legacy names that match without
 *     a semicolon) with Python's longest-prefix rule for an unterminated name. The replacement
 *     text matters only through what survives `re.sub(r"[^a-z0-9]+", "", label.lower())`, and the
 *     header-normaliser here keeps exactly that: all but two named entities leave nothing
 *     (`&fjlig;` -> "fj", `&Idot;` -> "i" — measured over the table), numeric references decode to
 *     the code point and go through the same lower-case + strip. That is why the table below is
 *     names only.
 *   - Python `repr`/`str` in messages (`{value!r}`, `{parsed}`) via writers.js pyRepr/pyFloatRepr.
 *
 * WHAT IT DELIBERATELY DOES NOT DO. No argparse (`configure_parser`, `command`, `main`,
 * evaluation.py:556-667): the app's CLI is not this library's. No JSON writing: writers.js
 * `evaluateJson` is `json.dumps(result, indent=2, allow_nan=False)`. It does not reproduce
 * `json.load`'s acceptance of NaN/Infinity literals (JSON.parse rejects them; a MEME file with
 * such a literal fails with "Could not read MEME JSON" whose trailing text is the JS parser's, not
 * Python's). It cannot tell a JSON `2.0` from `2` after JSON.parse, so a message that reprs such a
 * value prints `2` where Python prints `2.0`, and `str(5.0)` in a header normalises to "5" here
 * against "50" in Python; neither is reachable from HyPhy output. Python's `dict` keeps
 * MLE.content partitions in document order, JS objects move integer-like keys first — the sort by
 * `_partition_key` removes the difference except between keys that parse to the same int
 * ("1" and "01"). A MEME document whose top level is not an object raises EvaluationError here
 * where the Python raises AttributeError.
 *
 * FIELD NAMES. Site records keep the dataclass field names (`lrt`, `p_value`, `is_invariable`;
 * `lrt`, `p_value`, `lrt_was_clamped`, evaluation.py:32-43) and the report keeps the Python's
 * keys, because they are an output schema (PLAN.md Appendix B), not an API.
 *
 * Numerical primitives: pearson, spearman, rocAuc from src/numeric/ranks.js (1e-9 class).
 */

import { pearson, spearman, rocAuc } from './numeric/ranks.js';
import { pyStrip } from './preprocess/parse.js';
import { pyRepr, pyStr, pyFloatRepr, pyFormatFixed, pyFormatG } from './writers.js';

export { rocAuc };

/** evaluation.py:28-29 — raised when inputs cannot be paired or safely aligned by site. */
export class EvaluationError extends Error {
	/** @param {string} message */
	constructor(message) {
		super(message);
		this.name = 'EvaluationError';
	}
}

/**
 * @typedef {object} PredictionSite   evaluation.py:32-36
 * @property {number} lrt
 * @property {number} p_value
 * @property {boolean} is_invariable
 */
/**
 * @typedef {object} MemeSite         evaluation.py:39-43
 * @property {number} lrt
 * @property {number} p_value
 * @property {boolean} lrt_was_clamped
 */
/**
 * @typedef {object} NamedText  a file's content in place of its path
 * @property {string} name      what the Python's `path` prints in messages (a basename, usually)
 * @property {string} [text]    decoded content
 * @property {unknown} [data]   MEME only: an already-parsed JSON document, used instead of `text`
 */

// ---------------------------------------------------------------------------------------------
// Python numeric parsing
// ---------------------------------------------------------------------------------------------

const ND = /\p{Nd}/u;

/**
 * Replace every Unicode decimal digit by its ASCII value (Python's int()/float() accept them).
 * Each Nd run starts at a zero and has a multiple of ten code points, so the digit value is the
 * offset into the run modulo 10.
 * @param {string} s
 * @returns {string}
 */
function asciiDigits(s) {
	if (!ND.test(s)) return s;
	let out = '';
	for (const ch of s) {
		const cp = /** @type {number} */ (ch.codePointAt(0));
		if (cp < 0x80 || !ND.test(ch)) {
			out += ch;
			continue;
		}
		let k = 0;
		while (k < 60 && ND.test(String.fromCodePoint(cp - k - 1))) k++;
		out += String(k % 10);
	}
	return out;
}

const PY_FLOAT_RE = /^[+-]?(?:\d(?:_?\d)*(?:\.(?:\d(?:_?\d)*)?)?|\.\d(?:_?\d)*)(?:[eE][+-]?\d(?:_?\d)*)?$/;
const PY_INT_RE = /^[+-]?\d(?:_?\d)*$/;

/**
 * The whitespace `float()` and `int()` strip: CPython maps non-ASCII Unicode whitespace to ' '
 * (_PyUnicode_TransformDecimalAndSpaceToASCII) and then skips only C `isspace` — so U+00A0 and
 * U+3000 are stripped but \x1c..\x1f, which `str.strip()` removes, are not.
 */
const NUMERIC_WS = '[ \\t\\n\\v\\f\\r\\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]';
const NUMERIC_TRIM_RE = new RegExp(`^${NUMERIC_WS}+|${NUMERIC_WS}+$`, 'g');

/**
 * Python `float(value)`; throws TypeError/RangeError where Python raises TypeError/ValueError.
 * @param {unknown} value
 * @returns {number}
 */
function pyFloat(value) {
	if (typeof value === 'number') return value;
	if (typeof value === 'boolean') return value ? 1.0 : 0.0;
	if (typeof value !== 'string') throw new TypeError(`float() argument must be a string or a real number, not ${typeof value}`);
	const s = asciiDigits(value.replace(NUMERIC_TRIM_RE, ''));
	if (PY_FLOAT_RE.test(s)) return parseFloat(s.replace(/_/g, ''));
	const lower = s.toLowerCase();
	if (/^[+-]?(?:inf|infinity)$/.test(lower)) return lower.startsWith('-') ? -Infinity : Infinity;
	if (/^[+-]?nan$/.test(lower)) return NaN;
	throw new RangeError(`could not convert string to float: ${pyRepr(value)}`);
}

/**
 * Python `int(str)`, or null where it raises ValueError (used by _partition_key).
 * @param {string} value
 * @returns {number|null}
 */
function pyIntOrNull(value) {
	const s = asciiDigits(value.replace(NUMERIC_TRIM_RE, ''));
	if (!PY_INT_RE.test(s)) return null;
	return parseInt(s.replace(/_/g, ''), 10);
}

// ---------------------------------------------------------------------------------------------
// _finite_float / _probability / _site_number / _boolean (evaluation.py:46-77)
// ---------------------------------------------------------------------------------------------

/**
 * @param {unknown} value
 * @param {string} label
 * @param {string} path
 * @returns {number}
 */
function finiteFloat(value, label, path) {
	let parsed;
	try {
		parsed = pyFloat(value);
	} catch (exc) {
		throw new EvaluationError(`${path}: ${label} is not numeric: ${pyRepr(value)}`);
	}
	if (!Number.isFinite(parsed)) throw new EvaluationError(`${path}: ${label} must be finite, got ${pyRepr(value)}`);
	return parsed;
}

/**
 * @param {unknown} value
 * @param {string} label
 * @param {string} path
 * @returns {number}
 */
function probability(value, label, path) {
	const parsed = finiteFloat(value, label, path);
	if (!(0.0 <= parsed && parsed <= 1.0))
		throw new EvaluationError(`${path}: ${label} must be between 0 and 1, got ${pyFloatRepr(parsed)}`);
	return parsed;
}

/**
 * @param {unknown} value
 * @param {string} path
 * @returns {number}
 */
function siteNumber(value, path) {
	const parsed = finiteFloat(value, 'site', path);
	const site = Math.trunc(parsed);
	if (parsed !== site || site < 1) throw new EvaluationError(`${path}: site must be a positive integer, got ${pyRepr(value)}`);
	return site;
}

/**
 * @param {unknown} value
 * @param {string} label
 * @param {string} path
 * @returns {boolean}
 */
function boolean(value, label, path) {
	const normalized = pyStrip(pyStr(value)).toLowerCase();
	if (normalized === 'true' || normalized === '1' || normalized === 'yes') return true;
	if (normalized === 'false' || normalized === '0' || normalized === 'no') return false;
	throw new EvaluationError(`${path}: ${label} must be true or false, got ${pyRepr(value)}`);
}

// ---------------------------------------------------------------------------------------------
// csv.reader (excel dialect) and csv.DictReader
// ---------------------------------------------------------------------------------------------

/**
 * Python's `csv.reader` over a text opened with newline="": the _csv.c state machine for the
 * excel dialect (delimiter ',', quotechar '"', doublequote, no escapechar, non-strict).
 *
 * @param {string} text
 * @returns {string[][]}
 */
export function parseCsvRows(text) {
	// The reader is fed one line at a time by the file object (newline="" splits on \r, \n and \r\n
	// and keeps the terminator), and finishes a record at the end of every line that is not inside
	// a quoted field — a blank line is therefore an empty record, and a '\r\n' pair one terminator.
	const START_RECORD = 0;
	const START_FIELD = 1;
	const IN_FIELD = 2;
	const IN_QUOTED = 3;
	const QUOTE_IN_QUOTED = 4;
	/** @type {string[][]} */
	const rows = [];
	/** @type {string[]} */
	let fields = [];
	let field = '';
	let state = START_RECORD;
	const saveField = () => {
		fields.push(field);
		field = '';
	};
	const endRecord = () => {
		rows.push(fields);
		fields = [];
	};
	const n = text.length;
	for (let i = 0; i < n; i++) {
		const c = text[i];
		if ((c === '\n' || c === '\r') && state !== IN_QUOTED) {
			if (state !== START_RECORD) saveField();
			if (c === '\r' && text[i + 1] === '\n') i++;
			endRecord();
			state = START_RECORD;
			continue;
		}
		switch (state) {
			case START_RECORD:
				state = START_FIELD;
			// fall through
			case START_FIELD:
				if (c === '"') state = IN_QUOTED;
				else if (c === ',') saveField();
				else {
					field += c;
					state = IN_FIELD;
				}
				break;
			case IN_FIELD:
				if (c === ',') {
					saveField();
					state = START_FIELD;
				} else field += c;
				break;
			case IN_QUOTED:
				if (c === '"') state = QUOTE_IN_QUOTED;
				else field += c;
				break;
			case QUOTE_IN_QUOTED:
				if (c === '"') {
					field += c;
					state = IN_QUOTED;
				} else if (c === ',') {
					saveField();
					state = START_FIELD;
				} else {
					// non-strict: text after a closing quote is appended
					field += c;
					state = IN_FIELD;
				}
				break;
			default:
				throw new Error('unreachable');
		}
	}
	// End of data without a terminator: the open field (a quoted one too, non-strict) is saved.
	if (state !== START_RECORD) {
		saveField();
		endRecord();
	}
	return rows;
}

/**
 * `csv.DictReader(handle)` rows for a file opened with encoding="utf-8-sig": `fieldnames` is the
 * first row exactly (a leading blank line gives []), blank rows are skipped afterwards, and each
 * yielded row is a Map from field name to value with `null` for the fields a short row lacks.
 * Row numbers count yielded rows from `start`.
 *
 * @param {string} text
 * @param {number} [start=2]
 * @returns {{fieldnames: string[]|null, rows: Array<[number, Map<string|null, string|null|string[]>]>}}
 */
export function dictReaderRows(text, start = 2) {
	if (text.startsWith('\uFEFF')) text = text.slice(1);
	const raw = parseCsvRows(text);
	if (raw.length === 0) return { fieldnames: null, rows: [] };
	const fieldnames = raw[0];
	/** @type {Array<[number, Map<string|null, string|null|string[]>]>} */
	const rows = [];
	let number = start;
	for (let r = 1; r < raw.length; r++) {
		const row = raw[r];
		if (row.length === 0) continue;
		/** @type {Map<string|null, string|null|string[]>} */
		const d = new Map();
		const lf = fieldnames.length;
		const lr = row.length;
		for (let i = 0; i < Math.min(lf, lr); i++) d.set(fieldnames[i], row[i]);
		if (lf < lr) d.set(null, row.slice(lf));
		else for (let i = lr; i < lf; i++) d.set(fieldnames[i], null);
		rows.push([number++, d]);
	}
	return { fieldnames, rows };
}

// ---------------------------------------------------------------------------------------------
// load_prediction_csv (evaluation.py:80-108)
// ---------------------------------------------------------------------------------------------

const PREDICTION_REQUIRED = ['site', 'hyphaeon_lrt', 'p_value', 'is_invariable'];

/**
 * Load site-indexed results written by `hyphaeon meme` — evaluation.py:80-108.
 *
 * @param {string} text  the CSV content
 * @param {string} [path='prediction.csv']  the name used in messages
 * @returns {Map<number, PredictionSite>}
 */
export function loadPredictionCsv(text, path = 'prediction.csv') {
	const { fieldnames, rows } = dictReaderRows(text);
	const have = new Set(fieldnames ?? []);
	const missing = PREDICTION_REQUIRED.filter((k) => !have.has(k)).sort();
	if (missing.length) throw new EvaluationError(`${path}: missing required CSV column(s): ${missing.join(', ')}`);
	/** @type {Map<number, PredictionSite>} */
	const sites = new Map();
	for (const [lineNumber, row] of rows) {
		const site = siteNumber(row.get('site'), path);
		if (sites.has(site)) throw new EvaluationError(`${path}:${lineNumber}: duplicate site ${site}`);
		const lrt = finiteFloat(row.get('hyphaeon_lrt'), 'hyphaeon_lrt', path);
		if (lrt < 0.0) throw new EvaluationError(`${path}:${lineNumber}: hyphaeon_lrt cannot be negative`);
		sites.set(site, {
			lrt,
			p_value: probability(row.get('p_value'), 'p_value', path),
			is_invariable: boolean(row.get('is_invariable'), 'is_invariable', path),
		});
	}
	if (sites.size === 0) throw new EvaluationError(`${path}: prediction CSV contains no sites`);
	return sites;
}

// ---------------------------------------------------------------------------------------------
// html.unescape + _normalized_header (evaluation.py:111-114)
// ---------------------------------------------------------------------------------------------

/** HTML5 named character references (Python html.entities.html5 keys without their ';'). */
const HTML5_ENTITY_NAMES = 'AElig AMP Aacute Abreve Acirc Acy Afr Agrave Alpha Amacr And Aogon Aopf ApplyFunction Aring Ascr Assign Atilde Auml Backslash Barv Barwed Bcy Because Bernoullis Beta Bfr Bopf Breve Bscr Bumpeq CHcy COPY Cacute Cap CapitalDifferentialD Cayleys Ccaron Ccedil Ccirc Cconint Cdot Cedilla CenterDot Cfr Chi CircleDot CircleMinus CirclePlus CircleTimes ClockwiseContourIntegral CloseCurlyDoubleQuote CloseCurlyQuote Colon Colone Congruent Conint ContourIntegral Copf Coproduct CounterClockwiseContourIntegral Cross Cscr Cup CupCap DD DDotrahd DJcy DScy DZcy Dagger Darr Dashv Dcaron Dcy Del Delta Dfr DiacriticalAcute DiacriticalDot DiacriticalDoubleAcute DiacriticalGrave DiacriticalTilde Diamond DifferentialD Dopf Dot DotDot DotEqual DoubleContourIntegral DoubleDot DoubleDownArrow DoubleLeftArrow DoubleLeftRightArrow DoubleLeftTee DoubleLongLeftArrow DoubleLongLeftRightArrow DoubleLongRightArrow DoubleRightArrow DoubleRightTee DoubleUpArrow DoubleUpDownArrow DoubleVerticalBar DownArrow DownArrowBar DownArrowUpArrow DownBreve DownLeftRightVector DownLeftTeeVector DownLeftVector DownLeftVectorBar DownRightTeeVector DownRightVector DownRightVectorBar DownTee DownTeeArrow Downarrow Dscr Dstrok ENG ETH Eacute Ecaron Ecirc Ecy Edot Efr Egrave Element Emacr EmptySmallSquare EmptyVerySmallSquare Eogon Eopf Epsilon Equal EqualTilde Equilibrium Escr Esim Eta Euml Exists ExponentialE Fcy Ffr FilledSmallSquare FilledVerySmallSquare Fopf ForAll Fouriertrf Fscr GJcy GT Gamma Gammad Gbreve Gcedil Gcirc Gcy Gdot Gfr Gg Gopf GreaterEqual GreaterEqualLess GreaterFullEqual GreaterGreater GreaterLess GreaterSlantEqual GreaterTilde Gscr Gt HARDcy Hacek Hat Hcirc Hfr HilbertSpace Hopf HorizontalLine Hscr Hstrok HumpDownHump HumpEqual IEcy IJlig IOcy Iacute Icirc Icy Idot Ifr Igrave Im Imacr ImaginaryI Implies Int Integral Intersection InvisibleComma InvisibleTimes Iogon Iopf Iota Iscr Itilde Iukcy Iuml Jcirc Jcy Jfr Jopf Jscr Jsercy Jukcy KHcy KJcy Kappa Kcedil Kcy Kfr Kopf Kscr LJcy LT Lacute Lambda Lang Laplacetrf Larr Lcaron Lcedil Lcy LeftAngleBracket LeftArrow LeftArrowBar LeftArrowRightArrow LeftCeiling LeftDoubleBracket LeftDownTeeVector LeftDownVector LeftDownVectorBar LeftFloor LeftRightArrow LeftRightVector LeftTee LeftTeeArrow LeftTeeVector LeftTriangle LeftTriangleBar LeftTriangleEqual LeftUpDownVector LeftUpTeeVector LeftUpVector LeftUpVectorBar LeftVector LeftVectorBar Leftarrow Leftrightarrow LessEqualGreater LessFullEqual LessGreater LessLess LessSlantEqual LessTilde Lfr Ll Lleftarrow Lmidot LongLeftArrow LongLeftRightArrow LongRightArrow Longleftarrow Longleftrightarrow Longrightarrow Lopf LowerLeftArrow LowerRightArrow Lscr Lsh Lstrok Lt Map Mcy MediumSpace Mellintrf Mfr MinusPlus Mopf Mscr Mu NJcy Nacute Ncaron Ncedil Ncy NegativeMediumSpace NegativeThickSpace NegativeThinSpace NegativeVeryThinSpace NestedGreaterGreater NestedLessLess NewLine Nfr NoBreak NonBreakingSpace Nopf Not NotCongruent NotCupCap NotDoubleVerticalBar NotElement NotEqual NotEqualTilde NotExists NotGreater NotGreaterEqual NotGreaterFullEqual NotGreaterGreater NotGreaterLess NotGreaterSlantEqual NotGreaterTilde NotHumpDownHump NotHumpEqual NotLeftTriangle NotLeftTriangleBar NotLeftTriangleEqual NotLess NotLessEqual NotLessGreater NotLessLess NotLessSlantEqual NotLessTilde NotNestedGreaterGreater NotNestedLessLess NotPrecedes NotPrecedesEqual NotPrecedesSlantEqual NotReverseElement NotRightTriangle NotRightTriangleBar NotRightTriangleEqual NotSquareSubset NotSquareSubsetEqual NotSquareSuperset NotSquareSupersetEqual NotSubset NotSubsetEqual NotSucceeds NotSucceedsEqual NotSucceedsSlantEqual NotSucceedsTilde NotSuperset NotSupersetEqual NotTilde NotTildeEqual NotTildeFullEqual NotTildeTilde NotVerticalBar Nscr Ntilde Nu OElig Oacute Ocirc Ocy Odblac Ofr Ograve Omacr Omega Omicron Oopf OpenCurlyDoubleQuote OpenCurlyQuote Or Oscr Oslash Otilde Otimes Ouml OverBar OverBrace OverBracket OverParenthesis PartialD Pcy Pfr Phi Pi PlusMinus Poincareplane Popf Pr Precedes PrecedesEqual PrecedesSlantEqual PrecedesTilde Prime Product Proportion Proportional Pscr Psi QUOT Qfr Qopf Qscr RBarr REG Racute Rang Rarr Rarrtl Rcaron Rcedil Rcy Re ReverseElement ReverseEquilibrium ReverseUpEquilibrium Rfr Rho RightAngleBracket RightArrow RightArrowBar RightArrowLeftArrow RightCeiling RightDoubleBracket RightDownTeeVector RightDownVector RightDownVectorBar RightFloor RightTee RightTeeArrow RightTeeVector RightTriangle RightTriangleBar RightTriangleEqual RightUpDownVector RightUpTeeVector RightUpVector RightUpVectorBar RightVector RightVectorBar Rightarrow Ropf RoundImplies Rrightarrow Rscr Rsh RuleDelayed SHCHcy SHcy SOFTcy Sacute Sc Scaron Scedil Scirc Scy Sfr ShortDownArrow ShortLeftArrow ShortRightArrow ShortUpArrow Sigma SmallCircle Sopf Sqrt Square SquareIntersection SquareSubset SquareSubsetEqual SquareSuperset SquareSupersetEqual SquareUnion Sscr Star Sub Subset SubsetEqual Succeeds SucceedsEqual SucceedsSlantEqual SucceedsTilde SuchThat Sum Sup Superset SupersetEqual Supset THORN TRADE TSHcy TScy Tab Tau Tcaron Tcedil Tcy Tfr Therefore Theta ThickSpace ThinSpace Tilde TildeEqual TildeFullEqual TildeTilde Topf TripleDot Tscr Tstrok Uacute Uarr Uarrocir Ubrcy Ubreve Ucirc Ucy Udblac Ufr Ugrave Umacr UnderBar UnderBrace UnderBracket UnderParenthesis Union UnionPlus Uogon Uopf UpArrow UpArrowBar UpArrowDownArrow UpDownArrow UpEquilibrium UpTee UpTeeArrow Uparrow Updownarrow UpperLeftArrow UpperRightArrow Upsi Upsilon Uring Uscr Utilde Uuml VDash Vbar Vcy Vdash Vdashl Vee Verbar Vert VerticalBar VerticalLine VerticalSeparator VerticalTilde VeryThinSpace Vfr Vopf Vscr Vvdash Wcirc Wedge Wfr Wopf Wscr Xfr Xi Xopf Xscr YAcy YIcy YUcy Yacute Ycirc Ycy Yfr Yopf Yscr Yuml ZHcy Zacute Zcaron Zcy Zdot ZeroWidthSpace Zeta Zfr Zopf Zscr aacute abreve ac acE acd acirc acute acy aelig af afr agrave alefsym aleph alpha amacr amalg amp and andand andd andslope andv ang ange angle angmsd angmsdaa angmsdab angmsdac angmsdad angmsdae angmsdaf angmsdag angmsdah angrt angrtvb angrtvbd angsph angst angzarr aogon aopf ap apE apacir ape apid apos approx approxeq aring ascr ast asymp asympeq atilde auml awconint awint bNot backcong backepsilon backprime backsim backsimeq barvee barwed barwedge bbrk bbrktbrk bcong bcy bdquo becaus because bemptyv bepsi bernou beta beth between bfr bigcap bigcirc bigcup bigodot bigoplus bigotimes bigsqcup bigstar bigtriangledown bigtriangleup biguplus bigvee bigwedge bkarow blacklozenge blacksquare blacktriangle blacktriangledown blacktriangleleft blacktriangleright blank blk12 blk14 blk34 block bne bnequiv bnot bopf bot bottom bowtie boxDL boxDR boxDl boxDr boxH boxHD boxHU boxHd boxHu boxUL boxUR boxUl boxUr boxV boxVH boxVL boxVR boxVh boxVl boxVr boxbox boxdL boxdR boxdl boxdr boxh boxhD boxhU boxhd boxhu boxminus boxplus boxtimes boxuL boxuR boxul boxur boxv boxvH boxvL boxvR boxvh boxvl boxvr bprime breve brvbar bscr bsemi bsim bsime bsol bsolb bsolhsub bull bullet bump bumpE bumpe bumpeq cacute cap capand capbrcup capcap capcup capdot caps caret caron ccaps ccaron ccedil ccirc ccups ccupssm cdot cedil cemptyv cent centerdot cfr chcy check checkmark chi cir cirE circ circeq circlearrowleft circlearrowright circledR circledS circledast circledcirc circleddash cire cirfnint cirmid cirscir clubs clubsuit colon colone coloneq comma commat comp compfn complement complexes cong congdot conint copf coprod copy copysr crarr cross cscr csub csube csup csupe ctdot cudarrl cudarrr cuepr cuesc cularr cularrp cup cupbrcap cupcap cupcup cupdot cupor cups curarr curarrm curlyeqprec curlyeqsucc curlyvee curlywedge curren curvearrowleft curvearrowright cuvee cuwed cwconint cwint cylcty dArr dHar dagger daleth darr dash dashv dbkarow dblac dcaron dcy dd ddagger ddarr ddotseq deg delta demptyv dfisht dfr dharl dharr diam diamond diamondsuit diams die digamma disin div divide divideontimes divonx djcy dlcorn dlcrop dollar dopf dot doteq doteqdot dotminus dotplus dotsquare doublebarwedge downarrow downdownarrows downharpoonleft downharpoonright drbkarow drcorn drcrop dscr dscy dsol dstrok dtdot dtri dtrif duarr duhar dwangle dzcy dzigrarr eDDot eDot eacute easter ecaron ecir ecirc ecolon ecy edot ee efDot efr eg egrave egs egsdot el elinters ell els elsdot emacr empty emptyset emptyv emsp emsp13 emsp14 eng ensp eogon eopf epar eparsl eplus epsi epsilon epsiv eqcirc eqcolon eqsim eqslantgtr eqslantless equals equest equiv equivDD eqvparsl erDot erarr escr esdot esim eta eth euml euro excl exist expectation exponentiale fallingdotseq fcy female ffilig fflig ffllig ffr filig fjlig flat fllig fltns fnof fopf forall fork forkv fpartint frac12 frac13 frac14 frac15 frac16 frac18 frac23 frac25 frac34 frac35 frac38 frac45 frac56 frac58 frac78 frasl frown fscr gE gEl gacute gamma gammad gap gbreve gcirc gcy gdot ge gel geq geqq geqslant ges gescc gesdot gesdoto gesdotol gesl gesles gfr gg ggg gimel gjcy gl glE gla glj gnE gnap gnapprox gne gneq gneqq gnsim gopf grave gscr gsim gsime gsiml gt gtcc gtcir gtdot gtlPar gtquest gtrapprox gtrarr gtrdot gtreqless gtreqqless gtrless gtrsim gvertneqq gvnE hArr hairsp half hamilt hardcy harr harrcir harrw hbar hcirc hearts heartsuit hellip hercon hfr hksearow hkswarow hoarr homtht hookleftarrow hookrightarrow hopf horbar hscr hslash hstrok hybull hyphen iacute ic icirc icy iecy iexcl iff ifr igrave ii iiiint iiint iinfin iiota ijlig imacr image imagline imagpart imath imof imped in incare infin infintie inodot int intcal integers intercal intlarhk intprod iocy iogon iopf iota iprod iquest iscr isin isinE isindot isins isinsv isinv it itilde iukcy iuml jcirc jcy jfr jmath jopf jscr jsercy jukcy kappa kappav kcedil kcy kfr kgreen khcy kjcy kopf kscr lAarr lArr lAtail lBarr lE lEg lHar lacute laemptyv lagran lambda lang langd langle lap laquo larr larrb larrbfs larrfs larrhk larrlp larrpl larrsim larrtl lat latail late lates lbarr lbbrk lbrace lbrack lbrke lbrksld lbrkslu lcaron lcedil lceil lcub lcy ldca ldquo ldquor ldrdhar ldrushar ldsh le leftarrow leftarrowtail leftharpoondown leftharpoonup leftleftarrows leftrightarrow leftrightarrows leftrightharpoons leftrightsquigarrow leftthreetimes leg leq leqq leqslant les lescc lesdot lesdoto lesdotor lesg lesges lessapprox lessdot lesseqgtr lesseqqgtr lessgtr lesssim lfisht lfloor lfr lg lgE lhard lharu lharul lhblk ljcy ll llarr llcorner llhard lltri lmidot lmoust lmoustache lnE lnap lnapprox lne lneq lneqq lnsim loang loarr lobrk longleftarrow longleftrightarrow longmapsto longrightarrow looparrowleft looparrowright lopar lopf loplus lotimes lowast lowbar loz lozenge lozf lpar lparlt lrarr lrcorner lrhar lrhard lrm lrtri lsaquo lscr lsh lsim lsime lsimg lsqb lsquo lsquor lstrok lt ltcc ltcir ltdot lthree ltimes ltlarr ltquest ltrPar ltri ltrie ltrif lurdshar luruhar lvertneqq lvnE mDDot macr male malt maltese map mapsto mapstodown mapstoleft mapstoup marker mcomma mcy mdash measuredangle mfr mho micro mid midast midcir middot minus minusb minusd minusdu mlcp mldr mnplus models mopf mp mscr mstpos mu multimap mumap nGg nGt nGtv nLeftarrow nLeftrightarrow nLl nLt nLtv nRightarrow nVDash nVdash nabla nacute nang nap napE napid napos napprox natur natural naturals nbsp nbump nbumpe ncap ncaron ncedil ncong ncongdot ncup ncy ndash ne neArr nearhk nearr nearrow nedot nequiv nesear nesim nexist nexists nfr ngE nge ngeq ngeqq ngeqslant nges ngsim ngt ngtr nhArr nharr nhpar ni nis nisd niv njcy nlArr nlE nlarr nldr nle nleftarrow nleftrightarrow nleq nleqq nleqslant nles nless nlsim nlt nltri nltrie nmid nopf not notin notinE notindot notinva notinvb notinvc notni notniva notnivb notnivc npar nparallel nparsl npart npolint npr nprcue npre nprec npreceq nrArr nrarr nrarrc nrarrw nrightarrow nrtri nrtrie nsc nsccue nsce nscr nshortmid nshortparallel nsim nsime nsimeq nsmid nspar nsqsube nsqsupe nsub nsubE nsube nsubset nsubseteq nsubseteqq nsucc nsucceq nsup nsupE nsupe nsupset nsupseteq nsupseteqq ntgl ntilde ntlg ntriangleleft ntrianglelefteq ntriangleright ntrianglerighteq nu num numero numsp nvDash nvHarr nvap nvdash nvge nvgt nvinfin nvlArr nvle nvlt nvltrie nvrArr nvrtrie nvsim nwArr nwarhk nwarr nwarrow nwnear oS oacute oast ocir ocirc ocy odash odblac odiv odot odsold oelig ofcir ofr ogon ograve ogt ohbar ohm oint olarr olcir olcross oline olt omacr omega omicron omid ominus oopf opar operp oplus or orarr ord order orderof ordf ordm origof oror orslope orv oscr oslash osol otilde otimes otimesas ouml ovbar par para parallel parsim parsl part pcy percnt period permil perp pertenk pfr phi phiv phmmat phone pi pitchfork piv planck planckh plankv plus plusacir plusb pluscir plusdo plusdu pluse plusmn plussim plustwo pm pointint popf pound pr prE prap prcue pre prec precapprox preccurlyeq preceq precnapprox precneqq precnsim precsim prime primes prnE prnap prnsim prod profalar profline profsurf prop propto prsim prurel pscr psi puncsp qfr qint qopf qprime qscr quaternions quatint quest questeq quot rAarr rArr rAtail rBarr rHar race racute radic raemptyv rang rangd range rangle raquo rarr rarrap rarrb rarrbfs rarrc rarrfs rarrhk rarrlp rarrpl rarrsim rarrtl rarrw ratail ratio rationals rbarr rbbrk rbrace rbrack rbrke rbrksld rbrkslu rcaron rcedil rceil rcub rcy rdca rdldhar rdquo rdquor rdsh real realine realpart reals rect reg rfisht rfloor rfr rhard rharu rharul rho rhov rightarrow rightarrowtail rightharpoondown rightharpoonup rightleftarrows rightleftharpoons rightrightarrows rightsquigarrow rightthreetimes ring risingdotseq rlarr rlhar rlm rmoust rmoustache rnmid roang roarr robrk ropar ropf roplus rotimes rpar rpargt rppolint rrarr rsaquo rscr rsh rsqb rsquo rsquor rthree rtimes rtri rtrie rtrif rtriltri ruluhar rx sacute sbquo sc scE scap scaron sccue sce scedil scirc scnE scnap scnsim scpolint scsim scy sdot sdotb sdote seArr searhk searr searrow sect semi seswar setminus setmn sext sfr sfrown sharp shchcy shcy shortmid shortparallel shy sigma sigmaf sigmav sim simdot sime simeq simg simgE siml simlE simne simplus simrarr slarr smallsetminus smashp smeparsl smid smile smt smte smtes softcy sol solb solbar sopf spades spadesuit spar sqcap sqcaps sqcup sqcups sqsub sqsube sqsubset sqsubseteq sqsup sqsupe sqsupset sqsupseteq squ square squarf squf srarr sscr ssetmn ssmile sstarf star starf straightepsilon straightphi strns sub subE subdot sube subedot submult subnE subne subplus subrarr subset subseteq subseteqq subsetneq subsetneqq subsim subsub subsup succ succapprox succcurlyeq succeq succnapprox succneqq succnsim succsim sum sung sup sup1 sup2 sup3 supE supdot supdsub supe supedot suphsol suphsub suplarr supmult supnE supne supplus supset supseteq supseteqq supsetneq supsetneqq supsim supsub supsup swArr swarhk swarr swarrow swnwar szlig target tau tbrk tcaron tcedil tcy tdot telrec tfr there4 therefore theta thetasym thetav thickapprox thicksim thinsp thkap thksim thorn tilde times timesb timesbar timesd tint toea top topbot topcir topf topfork tosa tprime trade triangle triangledown triangleleft trianglelefteq triangleq triangleright trianglerighteq tridot trie triminus triplus trisb tritime trpezium tscr tscy tshcy tstrok twixt twoheadleftarrow twoheadrightarrow uArr uHar uacute uarr ubrcy ubreve ucirc ucy udarr udblac udhar ufisht ufr ugrave uharl uharr uhblk ulcorn ulcorner ulcrop ultri umacr uml uogon uopf uparrow updownarrow upharpoonleft upharpoonright uplus upsi upsih upsilon upuparrows urcorn urcorner urcrop uring urtri uscr utdot utilde utri utrif uuarr uuml uwangle vArr vBar vBarv vDash vangrt varepsilon varkappa varnothing varphi varpi varpropto varr varrho varsigma varsubsetneq varsubsetneqq varsupsetneq varsupsetneqq vartheta vartriangleleft vartriangleright vcy vdash vee veebar veeeq vellip verbar vert vfr vltri vnsub vnsup vopf vprop vrtri vscr vsubnE vsubne vsupnE vsupne vzigzag wcirc wedbar wedge wedgeq weierp wfr wopf wp wr wreath wscr xcap xcirc xcup xdtri xfr xhArr xharr xi xlArr xlarr xmap xnis xodot xopf xoplus xotime xrArr xrarr xscr xsqcup xuplus xutri xvee xwedge yacute yacy ycirc ycy yen yfr yicy yopf yscr yucy yuml zacute zcaron zcy zdot zeetrf zeta zfr zhcy zigrarr zopf zscr zwj zwnj';
/** The 106 legacy names html.unescape also matches WITHOUT a trailing semicolon. */
const HTML5_LEGACY_NAMES = 'AElig AMP Aacute Acirc Agrave Aring Atilde Auml COPY Ccedil ETH Eacute Ecirc Egrave Euml GT Iacute Icirc Igrave Iuml LT Ntilde Oacute Ocirc Ograve Oslash Otilde Ouml QUOT REG THORN Uacute Ucirc Ugrave Uuml Yacute aacute acirc acute aelig agrave amp aring atilde auml brvbar ccedil cedil cent copy curren deg divide eacute ecirc egrave eth euml frac12 frac14 frac34 gt iacute icirc iexcl igrave iquest iuml laquo lt macr micro middot nbsp not ntilde oacute ocirc ograve ordf ordm oslash otilde ouml para plusmn pound quot raquo reg sect shy sup1 sup2 sup3 szlig thorn times uacute ucirc ugrave uml uuml yacute yen yuml';
/** Named entities whose replacement text survives lower-casing and [^a-z0-9] stripping. */
const ENTITY_ALNUM_RESIDUE = new Map([
	['fjlig', 'fj'],
	['Idot', 'i'],
]);

/** @type {Set<string>|null} */
let entityNames = null;
/** @type {Set<string>|null} */
let legacyNames = null;

/** html._invalid_charrefs: numeric references remapped (windows-1252 range and two others). */
const INVALID_CHARREFS = new Map([
	[0x00, '�'], [0x0d, '\r'], [0x80, '€'], [0x81, '\x81'], [0x82, '‚'], [0x83, 'ƒ'],
	[0x84, '„'], [0x85, '…'], [0x86, '†'], [0x87, '‡'], [0x88, 'ˆ'], [0x89, '‰'],
	[0x8a, 'Š'], [0x8b, '‹'], [0x8c, 'Œ'], [0x8d, '\x8d'], [0x8e, 'Ž'], [0x8f, '\x8f'],
	[0x90, '\x90'], [0x91, '‘'], [0x92, '’'], [0x93, '“'], [0x94, '”'], [0x95, '•'],
	[0x96, '–'], [0x97, '—'], [0x98, '˜'], [0x99, '™'], [0x9a, 'š'], [0x9b, '›'],
	[0x9c, 'œ'], [0x9d, '\x9d'], [0x9e, 'ž'], [0x9f, 'Ÿ'],
]);

/** html._invalid_codepoints: numeric references that decode to nothing. @param {number} cp */
function isInvalidCodepoint(cp) {
	if (cp >= 0x1 && cp <= 0x8) return true;
	if (cp === 0xb) return true;
	if (cp >= 0xe && cp <= 0x1f) return true;
	if (cp >= 0x7f && cp <= 0x9f) return true;
	if (cp >= 0xfdd0 && cp <= 0xfdef) return true;
	return (cp & 0xfffe) === 0xfffe && cp <= 0x10ffff;
}

/**
 * `_replace_charref` from html.__init__ for one match `s` (the text after '&').
 * @param {string} s
 * @returns {string}
 */
function replaceCharref(s) {
	if (s[0] === '#') {
		const num = s[1] === 'x' || s[1] === 'X' ? parseInt(s.slice(2).replace(/;$/, ''), 16) : parseInt(s.slice(1).replace(/;$/, ''), 10);
		const mapped = INVALID_CHARREFS.get(num);
		if (mapped !== undefined) return mapped;
		if ((num >= 0xd800 && num <= 0xdfff) || num > 0x10ffff) return '�';
		if (isInvalidCodepoint(num)) return '';
		return String.fromCodePoint(num);
	}
	if (entityNames === null) {
		entityNames = new Set(HTML5_ENTITY_NAMES.split(' '));
		legacyNames = new Set(HTML5_LEGACY_NAMES.split(' '));
	}
	const names = /** @type {Set<string>} */ (entityNames);
	const legacy = /** @type {Set<string>} */ (legacyNames);
	/** @param {string} name  @returns {string} */
	const residue = (name) => ENTITY_ALNUM_RESIDUE.get(name) ?? '';
	if (s.endsWith(';')) {
		if (names.has(s.slice(0, -1))) return residue(s.slice(0, -1));
	} else if (legacy.has(s)) return residue(s);
	for (let x = s.length - 1; x > 1; x--) {
		const prefix = s.slice(0, x);
		if (legacy.has(prefix)) return residue(prefix) + s.slice(x);
	}
	return '&' + s;
}

const CHARREF_RE = /&(#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[^\t\n\f <&#;]{1,32};?)/g;

/**
 * Python `html.unescape`, except that a named reference is replaced by the alphanumeric residue
 * of its replacement text (see header) — identical after `_normalized_header`'s final strip.
 * @param {string} s
 * @returns {string}
 */
function htmlUnescapeForHeader(s) {
	if (!s.includes('&')) return s;
	return s.replace(CHARREF_RE, (_, m) => replaceCharref(m));
}

/**
 * evaluation.py:111-114 `_normalized_header`.
 * @param {unknown} value
 * @returns {string}
 */
export function normalizedHeader(value) {
	const label = Array.isArray(value) && value.length ? value[0] : value;
	const unescaped = htmlUnescapeForHeader(pyStr(label).replace(/<[^>]+>/g, ''));
	return unescaped.toLowerCase().replace(/[^a-z0-9]+/g, '');
}

/**
 * evaluation.py:117-125 `_column_index`.
 * @param {unknown} headers
 * @param {string} wanted
 * @param {number} fallback
 * @param {string} path
 * @returns {number}
 */
function columnIndex(headers, wanted, fallback, path) {
	if (Array.isArray(headers)) {
		const matches = [];
		headers.forEach((h, i) => {
			if (normalizedHeader(h) === wanted) matches.push(i);
		});
		if (matches.length === 1) return matches[0];
		if (matches.length > 1) throw new EvaluationError(`${path}: MEME has more than one ${pyRepr(wanted)} column`);
	}
	// Official HyPhy MEME output has LRT and p-value at zero-based columns 5 and 6.
	return fallback;
}

/**
 * evaluation.py:128-132 `_partition_key`: numeric partitions first, ascending; then the rest by
 * string order.
 * @param {string} value
 * @returns {[number, number|string]}
 */
function partitionKey(value) {
	const n = pyIntOrNull(value);
	return n === null ? [1, value] : [0, n];
}

/**
 * @param {[number, number|string]} a
 * @param {[number, number|string]} b
 */
function comparePartitionKeys(a, b) {
	if (a[0] !== b[0]) return a[0] - b[0];
	if (a[0] === 0) return /** @type {number} */ (a[1]) - /** @type {number} */ (b[1]);
	return a[1] < b[1] ? -1 : a[1] > b[1] ? 1 : 0;
}

/**
 * evaluation.py:135-155 `_flatten_coverage`.
 * @param {unknown} value
 * @param {string} path
 * @param {string} partition
 * @returns {number[]}
 */
function flattenCoverage(value, path, partition) {
	/** @type {number[]} */
	const flattened = [];
	/** @param {unknown} item */
	const visit = (item) => {
		if (Array.isArray(item)) {
			for (const child of item) visit(child);
			return;
		}
		if (typeof item !== 'number')
			throw new EvaluationError(`${path}: invalid site index in data partition ${pyRepr(partition)} coverage`);
		const index = Math.trunc(item);
		if (index !== item || index < 0)
			throw new EvaluationError(`${path}: invalid site index ${pyRepr(item)} in partition ${pyRepr(partition)} coverage`);
		flattened.push(index);
	};
	visit(value);
	return flattened;
}

/** @param {unknown} v */
const isDict = (v) => typeof v === 'object' && v !== null && !Array.isArray(v);

/**
 * Load MEME sites, using partition coverage to recover global site IDs — evaluation.py:158-218.
 *
 * @param {string|object} source  the MEME JSON text, or the parsed document
 * @param {string} [path='meme.json']  the name used in messages
 * @returns {Map<number, MemeSite>}
 */
export function loadMemeJson(source, path = 'meme.json') {
	/** @type {any} */
	let data;
	if (typeof source === 'string') {
		try {
			data = JSON.parse(source);
		} catch (exc) {
			throw new EvaluationError(`Could not read MEME JSON ${path}: ${/** @type {Error} */ (exc).message}`);
		}
	} else data = source;

	const mle = isDict(data) ? data.MLE : undefined;
	if (!isDict(mle) || !isDict(mle.content)) throw new EvaluationError(`${path}: expected an MLE.content object`);
	const content = mle.content;
	const partitions = Object.keys(content);
	if (partitions.length === 0) throw new EvaluationError(`${path}: MLE.content contains no partitions`);

	const lrtColumn = columnIndex(mle.headers, 'lrt', 5, path);
	const pColumn = columnIndex(mle.headers, 'pvalue', 6, path);
	const dataPartitions = data['data partitions'] === undefined ? {} : data['data partitions'];
	/** @type {Map<number, MemeSite>} */
	const sites = new Map();
	let nextFallbackIndex = 0;

	const ordered = partitions.map((p) => ({ p, key: partitionKey(p) })).sort((a, b) => comparePartitionKeys(a.key, b.key));
	for (const { p: partition } of ordered) {
		const rows = content[partition];
		if (!Array.isArray(rows)) throw new EvaluationError(`${path}: MLE.content[${pyRepr(partition)}] must be a list`);

		const partitionInfo = isDict(dataPartitions) ? (dataPartitions[partition] === undefined ? {} : dataPartitions[partition]) : {};
		const coverage = isDict(partitionInfo) ? partitionInfo.coverage : null;
		/** @type {number[]} */
		let siteIndices;
		if (coverage === null || coverage === undefined) {
			siteIndices = [];
			for (let i = 0; i < rows.length; i++) siteIndices.push(nextFallbackIndex + i);
		} else {
			siteIndices = flattenCoverage(coverage, path, partition);
			if (siteIndices.length !== rows.length)
				throw new EvaluationError(
					`${path}: partition ${pyRepr(partition)} has ${rows.length} MLE rows but ${siteIndices.length} covered sites`,
				);
		}
		if (siteIndices.length) nextFallbackIndex = Math.max(nextFallbackIndex, Math.max(...siteIndices) + 1);

		const count = Math.min(siteIndices.length, rows.length);
		for (let r = 0; r < count; r++) {
			const rowNumber = r + 1;
			const row = rows[r];
			if (!Array.isArray(row) || Math.max(lrtColumn, pColumn) >= row.length)
				throw new EvaluationError(`${path}: partition ${pyRepr(partition)} row ${rowNumber} lacks LRT/p-value columns`);
			const site = siteIndices[r] + 1;
			if (sites.has(site)) throw new EvaluationError(`${path}: duplicate MEME site ${site} across partitions`);
			const rawLrt = finiteFloat(row[lrtColumn], 'MEME LRT', path);
			// An LRT is non-negative by definition, but MEME can emit small negative values when
			// its two numerical likelihood fits cross.
			const lrt = Math.max(0.0, rawLrt);
			sites.set(site, {
				lrt,
				p_value: probability(row[pColumn], 'MEME p-value', path),
				lrt_was_clamped: rawLrt < 0.0,
			});
		}
	}
	if (sites.size === 0) throw new EvaluationError(`${path}: MEME JSON contains no sites`);
	return sites;
}

// ---------------------------------------------------------------------------------------------
// _files_by_gene / match_gene_files (evaluation.py:221-260)
// ---------------------------------------------------------------------------------------------

/**
 * Python's default string ordering (code points); JS default sort compares UTF-16 units, which
 * agrees on the BMP.
 * @param {string} a
 * @param {string} b
 */
const strCmp = (a, b) => (a < b ? -1 : a > b ? 1 : 0);

/**
 * evaluation.py:221-236 `_files_by_gene` over a directory listing given as {name, text} pairs.
 * @param {NamedText[]} files
 * @param {string} suffix
 * @param {string} label
 * @param {string} directory  printed where the Python prints the directory path
 * @returns {Map<string, NamedText>}
 */
function filesByGene(files, suffix, label, directory) {
	/** @type {Map<string, NamedText>} */
	const out = new Map();
	for (const file of [...files].sort((a, b) => strCmp(a.name, b.name))) {
		if (!file.name.endsWith(suffix)) continue;
		const gene = suffix ? file.name.slice(0, file.name.length - suffix.length) : file.name;
		if (!gene) throw new EvaluationError(`${file.name}: file name has no gene name before ${pyRepr(suffix)}`);
		if (out.has(gene)) throw new EvaluationError(`${label} directory has duplicate files for gene ${pyRepr(gene)}`);
		out.set(gene, file);
	}
	if (out.size === 0) throw new EvaluationError(`No *${suffix} files found in ${label} directory ${directory}`);
	return out;
}

/**
 * @typedef {object} GenePair
 * @property {string} gene
 * @property {NamedText} prediction
 * @property {NamedText} meme
 */

/**
 * evaluation.py:239-260 `match_gene_files`, over two directory listings.
 *
 * @param {NamedText[]} predictionFiles
 * @param {NamedText[]} memeFiles
 * @param {{predictionSuffix?: string, memeSuffix?: string, allowUnmatched?: boolean,
 *   predictionDirectory?: string, memeDirectory?: string}} [options]
 * @returns {{pairs: GenePair[], predictionOnly: string[], memeOnly: string[]}}
 */
export function matchGeneFiles(
	predictionFiles,
	memeFiles,
	{ predictionSuffix = '.csv', memeSuffix = '.MEME.json', allowUnmatched = false, predictionDirectory = '.', memeDirectory = '.' } = {},
) {
	const predictions = filesByGene(predictionFiles, predictionSuffix, 'prediction', predictionDirectory);
	const memeResults = filesByGene(memeFiles, memeSuffix, 'MEME', memeDirectory);
	const predictionOnly = [...predictions.keys()].filter((g) => !memeResults.has(g)).sort(strCmp);
	const memeOnly = [...memeResults.keys()].filter((g) => !predictions.has(g)).sort(strCmp);
	if ((predictionOnly.length || memeOnly.length) && !allowUnmatched) {
		const details = [];
		if (predictionOnly.length) details.push('prediction-only genes: ' + predictionOnly.join(', '));
		if (memeOnly.length) details.push('MEME-only genes: ' + memeOnly.join(', '));
		throw new EvaluationError('Unmatched input files; ' + details.join('; '));
	}
	const genes = [...predictions.keys()].filter((g) => memeResults.has(g)).sort(strCmp);
	if (genes.length === 0) throw new EvaluationError('The two directories contain no matching gene names');
	const pairs = genes.map((gene) => ({
		gene,
		prediction: /** @type {NamedText} */ (predictions.get(gene)),
		meme: /** @type {NamedText} */ (memeResults.get(gene)),
	}));
	return { pairs, predictionOnly, memeOnly };
}

// ---------------------------------------------------------------------------------------------
// metrics (evaluation.py:263-327)
// ---------------------------------------------------------------------------------------------

/** @param {number} v @returns {number|null} */
const optionalFloat = (v) => (Number.isFinite(v) ? v : null);

/** @param {ArrayLike<number>} a */
function ptp(a) {
	let lo = Infinity;
	let hi = -Infinity;
	for (let i = 0; i < a.length; i++) {
		if (a[i] < lo) lo = a[i];
		if (a[i] > hi) hi = a[i];
	}
	return hi - lo;
}

/**
 * evaluation.py:267-272 `_correlations`: (None, None) below two points or for a constant vector,
 * else scipy pearsonr / spearmanr statistics (NaN -> None).
 * @param {ArrayLike<number>} predicted
 * @param {ArrayLike<number>} observed
 * @returns {{pearson: number|null, spearman: number|null}}
 */
export function correlations(predicted, observed) {
	if (predicted.length < 2 || ptp(predicted) === 0.0 || ptp(observed) === 0.0) return { pearson: null, spearman: null };
	return { pearson: optionalFloat(pearson(predicted, observed)), spearman: optionalFloat(spearman(predicted, observed)) };
}

/**
 * evaluation.py:286-292 `_confusion`.
 * @param {ArrayLike<boolean>} truth
 * @param {ArrayLike<boolean>} predicted
 * @returns {{true_positive: number, false_positive: number, true_negative: number, false_negative: number}}
 */
export function confusion(truth, predicted) {
	let tp = 0;
	let fp = 0;
	let tn = 0;
	let fn = 0;
	for (let i = 0; i < truth.length; i++) {
		if (truth[i] && predicted[i]) tp++;
		else if (!truth[i] && predicted[i]) fp++;
		else if (!truth[i] && !predicted[i]) tn++;
		else fn++;
	}
	return { true_positive: tp, false_positive: fp, true_negative: tn, false_negative: fn };
}

/** evaluation.py:295-297. @param {{true_positive: number, false_positive: number}} c */
export function ppv(c) {
	const denominator = c.true_positive + c.false_positive;
	return denominator ? c.true_positive / denominator : null;
}

/** evaluation.py:300-302. @param {{false_positive: number, true_negative: number}} c */
export function fpr(c) {
	const denominator = c.false_positive + c.true_negative;
	return denominator ? c.false_positive / denominator : null;
}

/**
 * evaluation.py:305-327 `_threshold_metrics`.
 * @param {ArrayLike<number>} predictedLrt
 * @param {ArrayLike<number>} predictedP
 * @param {ArrayLike<number>} memeP
 * @param {number} alpha
 * @returns {Record<string, unknown>}
 */
export function thresholdMetrics(predictedLrt, predictedP, memeP, alpha) {
	const n = memeP.length;
	const inclusiveTruth = new Array(n);
	const inclusivePredictions = new Array(n);
	let truthCount = 0;
	let predCount = 0;
	for (let i = 0; i < n; i++) {
		inclusiveTruth[i] = memeP[i] <= alpha;
		inclusivePredictions[i] = predictedP[i] <= alpha;
		if (inclusiveTruth[i]) truthCount++;
		if (inclusivePredictions[i]) predCount++;
	}
	const ppvConfusion = confusion(inclusiveTruth, inclusivePredictions);
	const fprConfusion = confusion(inclusiveTruth, inclusivePredictions);
	const alphaG = pyFormatG(alpha);
	return {
		roc_auc: optionalFloat(rocAuc(inclusiveTruth, predictedLrt)),
		roc_auc_definition: `ground truth MEME p_value <= ${alphaG}; score = HyphAeon hyphaeon_lrt`,
		ppv: ppv(ppvConfusion),
		ppv_definition: `MEME and HyphAeon p_value <= ${alphaG}`,
		fpr: fpr(fprConfusion),
		fpr_definition: `MEME and HyphAeon p_value <= ${alphaG}`,
		meme_positive_sites: truthCount,
		hyphaeon_positive_sites: predCount,
		ppv_confusion_matrix: ppvConfusion,
		fpr_confusion_matrix: fprConfusion,
	};
}

// ---------------------------------------------------------------------------------------------
// _evaluate_pairs / evaluate_directories / evaluate_files (evaluation.py:330-527)
// ---------------------------------------------------------------------------------------------

/** @param {NamedText} file @returns {Map<number, PredictionSite>} */
function readPrediction(file) {
	return loadPredictionCsv(file.text ?? '', file.name);
}

/** @param {NamedText} file @returns {Map<number, MemeSite>} */
function readMeme(file) {
	return loadMemeJson(file.data !== undefined ? /** @type {object} */ (file.data) : (file.text ?? ''), file.name);
}

/** @param {Iterable<number>} a @param {Map<number, unknown>} b */
function sortedDifference(a, b) {
	return [...a].filter((k) => !b.has(k)).sort((x, y) => x - y);
}

/**
 * Pool already-paired gene files and calculate dataset-level metrics — evaluation.py:330-449
 * `_evaluate_pairs`. The report's keys and order are the Python's (PLAN.md Appendix B).
 *
 * @param {GenePair[]} pairs
 * @param {{inputMetadata?: Record<string, unknown>, predictionOnly?: string[], memeOnly?: string[],
 *   allowSiteMismatch?: boolean, variableOnly?: boolean}} [options]
 * @returns {Record<string, unknown>}
 */
export function evaluatePairs(
	pairs,
	{ inputMetadata = {}, predictionOnly = [], memeOnly = [], allowSiteMismatch = false, variableOnly = false } = {},
) {
	/** @type {number[]} */
	const pooledPredictionLrt = [];
	/** @type {number[]} */
	const pooledPredictionP = [];
	/** @type {number[]} */
	const pooledMemeLrt = [];
	/** @type {number[]} */
	const pooledMemeP = [];
	/** @type {Array<Record<string, unknown>>} */
	const perGene = [];
	let totalPredictionSites = 0;
	let totalMemeSites = 0;
	let totalMatchedSites = 0;
	let totalInvariableSites = 0;
	let totalClampedMemeLrts = 0;

	for (const { gene, prediction, meme } of pairs) {
		const predictions = readPrediction(prediction);
		const memeSites = readMeme(meme);
		totalPredictionSites += predictions.size;
		totalMemeSites += memeSites.size;

		const predictionOnlySites = sortedDifference(predictions.keys(), memeSites);
		const memeOnlySites = sortedDifference(memeSites.keys(), predictions);
		if ((predictionOnlySites.length || memeOnlySites.length) && !allowSiteMismatch) {
			const details = [];
			if (predictionOnlySites.length) details.push(`prediction-only sites ${pyRepr(predictionOnlySites.slice(0, 10), 'int')}`);
			if (memeOnlySites.length) details.push(`MEME-only sites ${pyRepr(memeOnlySites.slice(0, 10), 'int')}`);
			throw new EvaluationError(`Gene ${pyRepr(gene)} has mismatched sites: ` + details.join('; '));
		}

		const matchedSites = [...predictions.keys()].filter((s) => memeSites.has(s)).sort((x, y) => x - y);
		if (matchedSites.length === 0) throw new EvaluationError(`Gene ${pyRepr(gene)} has no sites shared by its two result files`);
		totalMatchedSites += matchedSites.length;
		for (const site of matchedSites) {
			if (/** @type {PredictionSite} */ (predictions.get(site)).is_invariable) totalInvariableSites++;
			if (/** @type {MemeSite} */ (memeSites.get(site)).lrt_was_clamped) totalClampedMemeLrts++;
		}

		const evaluatedSites = matchedSites.filter(
			(site) => !variableOnly || !(/** @type {PredictionSite} */ (predictions.get(site)).is_invariable),
		);
		for (const site of evaluatedSites) {
			const p = /** @type {PredictionSite} */ (predictions.get(site));
			const m = /** @type {MemeSite} */ (memeSites.get(site));
			pooledPredictionLrt.push(p.lrt);
			pooledPredictionP.push(p.p_value);
			pooledMemeLrt.push(m.lrt);
			pooledMemeP.push(m.p_value);
		}

		perGene.push({
			gene,
			prediction_sites: predictions.size,
			meme_sites: memeSites.size,
			matched_sites: matchedSites.length,
			evaluated_sites: evaluatedSites.length,
			prediction_only_sites: predictionOnlySites.length,
			meme_only_sites: memeOnlySites.length,
		});
	}

	if (pooledPredictionLrt.length === 0) {
		const scope = variableOnly ? 'variable' : 'matched';
		throw new EvaluationError(`No ${scope} sites are available for evaluation`);
	}

	const predictionLrt = Float64Array.from(pooledPredictionLrt);
	const predictionP = Float64Array.from(pooledPredictionP);
	const memeLrt = Float64Array.from(pooledMemeLrt);
	const memeP = Float64Array.from(pooledMemeP);
	const { pearson: pearsonR, spearman: spearmanRho } = correlations(predictionLrt, memeLrt);

	/** @type {string[]} */
	const warnings = [];
	if (predictionOnly.length) warnings.push('Ignored prediction-only genes: ' + predictionOnly.join(', '));
	if (memeOnly.length) warnings.push('Ignored MEME-only genes: ' + memeOnly.join(', '));
	if (allowSiteMismatch) {
		let droppedPrediction = 0;
		let droppedMeme = 0;
		for (const item of perGene) {
			droppedPrediction += Math.trunc(/** @type {number} */ (item.prediction_only_sites));
			droppedMeme += Math.trunc(/** @type {number} */ (item.meme_only_sites));
		}
		if (droppedPrediction || droppedMeme)
			warnings.push(`Site intersection used; dropped ${droppedPrediction} prediction-only and ${droppedMeme} MEME-only sites`);
	}
	if (totalClampedMemeLrts) warnings.push(`Clamped ${totalClampedMemeLrts} negative MEME LRT numerical artifact(s) to zero`);

	return {
		...inputMetadata,
		matched_genes: pairs.length,
		genes: pairs.map((p) => p.gene),
		total_sites: totalMatchedSites,
		total_prediction_sites: totalPredictionSites,
		total_meme_sites: totalMemeSites,
		evaluated_sites: predictionLrt.length,
		variable_sites: totalMatchedSites - totalInvariableSites,
		invariable_sites: totalInvariableSites,
		evaluation_scope: variableOnly ? 'variable sites only' : 'all matched sites',
		correlation_definition:
			'HyphAeon hyphaeon_lrt versus MEME LRT over pooled evaluated sites; ' +
			'negative MEME LRT numerical artifacts are clamped to zero',
		clamped_negative_meme_lrts: totalClampedMemeLrts,
		pearson_r: pearsonR,
		spearman_rho: spearmanRho,
		thresholds: {
			'0.05': thresholdMetrics(predictionLrt, predictionP, memeP, 0.05),
			'0.10': thresholdMetrics(predictionLrt, predictionP, memeP, 0.1),
		},
		per_gene: perGene,
		warnings,
	};
}

/**
 * Pair genes from two directory listings, pool their sites, and evaluate — evaluation.py:452-483.
 * `predictionDirectory` / `memeDirectory` are the resolved directory strings the Python puts in
 * the report (and prints in the "No *suffix files found" message).
 *
 * @param {NamedText[]} predictionFiles
 * @param {NamedText[]} memeFiles
 * @param {{predictionSuffix?: string, memeSuffix?: string, allowUnmatched?: boolean,
 *   allowSiteMismatch?: boolean, variableOnly?: boolean, predictionDirectory?: string, memeDirectory?: string}} [options]
 * @returns {Record<string, unknown>}
 */
export function evaluateDirectories(
	predictionFiles,
	memeFiles,
	{
		predictionSuffix = '.csv',
		memeSuffix = '.MEME.json',
		allowUnmatched = false,
		allowSiteMismatch = false,
		variableOnly = false,
		predictionDirectory = '.',
		memeDirectory = '.',
	} = {},
) {
	const { pairs, predictionOnly, memeOnly } = matchGeneFiles(predictionFiles, memeFiles, {
		predictionSuffix,
		memeSuffix,
		allowUnmatched,
		predictionDirectory,
		memeDirectory,
	});
	return evaluatePairs(pairs, {
		inputMetadata: { input_mode: 'directories', prediction_directory: predictionDirectory, meme_directory: memeDirectory },
		predictionOnly,
		memeOnly,
		allowSiteMismatch,
		variableOnly,
	});
}

/**
 * evaluation.py:486-494 `_gene_name_from_file` (the is_file check has no counterpart here).
 * @param {NamedText} file
 * @param {string} suffix
 * @param {string} label
 * @returns {string}
 */
function geneNameFromFile(file, suffix, label) {
	if (suffix && !file.name.endsWith(suffix)) throw new EvaluationError(`${label} file must end with ${pyRepr(suffix)}: ${file.name}`);
	const gene = suffix ? file.name.slice(0, file.name.length - suffix.length) : file.name;
	if (!gene) throw new EvaluationError(`${file.name}: file name has no gene name before ${pyRepr(suffix)}`);
	return gene;
}

/**
 * Evaluate one matched HyphAeon prediction and MEME result file — evaluation.py:497-527.
 * The report's `prediction_file` / `meme_result_file` are the given names (the Python resolves
 * them to absolute paths; the fixtures reduce those to basenames).
 *
 * @param {NamedText} predictionFile
 * @param {NamedText} memeResultFile
 * @param {{predictionSuffix?: string, memeSuffix?: string, allowSiteMismatch?: boolean, variableOnly?: boolean}} [options]
 * @returns {Record<string, unknown>}
 */
export function evaluateFiles(
	predictionFile,
	memeResultFile,
	{ predictionSuffix = '.csv', memeSuffix = '.MEME.json', allowSiteMismatch = false, variableOnly = false } = {},
) {
	const predictionGene = geneNameFromFile(predictionFile, predictionSuffix, 'prediction');
	const memeGene = geneNameFromFile(memeResultFile, memeSuffix, 'MEME result');
	if (predictionGene !== memeGene)
		throw new EvaluationError(
			'Single-gene files do not match: ' + `prediction gene ${pyRepr(predictionGene)}, MEME result gene ${pyRepr(memeGene)}`,
		);
	return evaluatePairs([{ gene: predictionGene, prediction: predictionFile, meme: memeResultFile }], {
		inputMetadata: { input_mode: 'files', prediction_file: predictionFile.name, meme_result_file: memeResultFile.name },
		allowSiteMismatch,
		variableOnly,
	});
}

// ---------------------------------------------------------------------------------------------
// format_text_report (evaluation.py:530-553)
// ---------------------------------------------------------------------------------------------

/** evaluation.py:530-531 `_display`. @param {unknown} value */
function display(value) {
	return value === null || value === undefined ? 'undefined' : pyFormatFixed(Number(value), 6);
}

/** @param {string} s @param {number} width */
const rjust = (s, width) => s.padStart(width);

/**
 * evaluation.py:534-553 `format_text_report`.
 * @param {Record<string, any>} result
 * @returns {string}
 */
export function formatTextReport(result) {
	const thresholds = result.thresholds;
	const a05 = thresholds['0.05'];
	const a10 = thresholds['0.10'];
	const lines = [
		`Matched genes: ${result.matched_genes}`,
		`Total sites: ${result.total_sites}`,
		`Evaluated sites: ${result.evaluated_sites} (${result.evaluation_scope})`,
		`Pearson r (LRT): ${display(result.pearson_r)}`,
		`Spearman rho (LRT): ${display(result.spearman_rho)}`,
		'',
		'Metric                 p <= 0.05    p <= 0.10',
		`ROC-AUC                ${rjust(display(a05.roc_auc), 10)}    ${rjust(display(a10.roc_auc), 10)}`,
		`PPV                    ${rjust(display(a05.ppv), 10)}    ${rjust(display(a10.ppv), 10)}`,
		`FPR                    ${rjust(display(a05.fpr), 10)}    ${rjust(display(a10.fpr), 10)}`,
	];
	const warnings = result.warnings ?? [];
	if (warnings.length) lines.push('', ...warnings.map((/** @type {string} */ w) => `Warning: ${w}`));
	return lines.join('\n');
}
