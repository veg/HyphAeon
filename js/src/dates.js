/**
 * WHY THIS FILE EXISTS
 *
 * The time-aware pillars (dating and temporal selection) both need one thing before they need
 * anything else: a string a human typed turned into a decimal time coordinate. That conversion is
 * the same arithmetic no matter who is asking — a browser, an MCP server, a batch CLI — so by the
 * rule in src/README.md it is the library's. Everything AROUND it (opening a file, sniffing a CSV
 * delimiter, walking an Auspice JSON, matching a metadata name to a taxon, deciding what to tell
 * the reader) knows about files, URLs and messages, and lives in the application's `runtime/`.
 *
 * WHAT IT MIRRORS, at veg/HyphAeon tag phase-5a:
 *
 *   `parse_date_to_decimal`        hyphaeon/temporal.py:73-151   -> parseDate / parseDateToDecimal
 *   `extract_date_from_string`     hyphaeon/temporal.py:196-245  -> extractDate / extractDateFromString
 *   `parse_header_timestamp`       hyphaeon/dating.py:317-364    -> parseHeaderDate / parseHeaderTimestamp
 *   `_parse_timestamp_flexible`    hyphaeon/dating.py:367-384    -> parseFlexibleDate / parseTimestampFlexible
 *
 * NOT mirrored here, on purpose: `parse_dates_from_auspice_json` (temporal.py:154), the tabular
 * branch of `parse_temporal_metadata` (temporal.py:248) and the file/BEAST dispatch of
 * `parse_sample_dates` (dating.py:387). All three open files, discover columns and decide which
 * taxon a row belongs to — application work, and the application is where their quirks have to be
 * made visible (a partial metadata table never falls back to the headers, because temporal.py:323
 * guards the fallback with `len(dates) == 0` rather than per taxon; the reference then silently
 * drops every taxon it could not date).
 *
 * THE CONTRACT THIS ADDS, and the reason the primary functions are not the Python's shape. Every
 * reference function returns a bare float and NaN for "nothing", which throws away the two facts a
 * reviewable page needs: WHICH rule fired, and whether a component of the answer was invented
 * rather than read. `parseDate`, `extractDate`, `parseHeaderDate` and `parseFlexibleDate` return a
 * `DateParse` record carrying both; `parseDateToDecimal`, `extractDateFromString`,
 * `parseHeaderTimestamp` and `parseTimestampFlexible` are one-line wrappers that return `.value`
 * and so are the exact Python signatures for parity. No arithmetic differs between the two forms.
 *
 * CALENDAR VS NON-CALENDAR. `timeUnits` selects the interpretation exactly as the reference does:
 * `'years'` gates a value into [1800, 2100] and understands ISO dates; `'generations'`, `'days'`
 * and `'arbitrary'` accept any non-negative real and read embedded generation/day tokens instead
 * (temporal.py:90-102, 204-215). An unknown string is treated as calendar, as Python's
 * `if time_units in (...)` does.
 *
 * UPSTREAM QUIRKS, REPLICATED AND FLAGGED (PLAN.md §5.3 rule 3 — never fixed during a port):
 *
 *  Q1. THE HARD-CODED 1959 ANCHOR. `parse_header_timestamp` opens with
 *          if 'Z59' in name or 'ZR59' in name or '1959' in name: return 1959.5
 *      (dating.py:330-332), before any date pattern is tried. It is a substring test on the WHOLE
 *      header, so `A/Kinshasa/1959-03-04` yields 1959.5 rather than the date that is written
 *      there, and any accession containing `Z59` — `AZ59012` — is dated to the archival ZR59
 *      isolate. That is a data-loss bug, not a convention, so here it is OPT-IN: the rule is
 *      skipped unless the caller passes `{archival1959: true}`, in which case it fires exactly as
 *      the Python does and reports `rule: 'archival_1959'`. `matchesArchival1959` exposes the
 *      predicate on its own, so an application can offer the rule rather than apply it.
 *      Q1 is the ONE place where a `...Timestamp`/`...Decimal` wrapper does not reproduce the
 *      reference by default; `parseHeaderTimestamp(name, {archival1959: true})` does.
 *
 *  Q2. A MISSING MONTH OR DAY IS IMPUTED TO MID-YEAR / MID-MONTH, SILENTLY. On the delimiter path
 *      month defaults to 6 and day to 15 (temporal.py:132, 138), so `2021-XX-XX` becomes
 *      2021.45205 and `2021-04` becomes mid-April. The reference says nothing about it; here
 *      `imputations.month` / `imputations.day` say which component was invented and `imputed` is
 *      their disjunction.
 *      The reference's OWN DOCSTRING is wrong about the commonest case: temporal.py:80 promises
 *      `"2021" -> 2021.5 (mid-year)`, but a bare year string is claimed by the `float(val_str)`
 *      attempt at temporal.py:116, passes the gate, and comes back as 2021.0 with nothing imputed.
 *      Only a year whose month is present-but-unreadable reaches the mid-year default. Replicated,
 *      not corrected: `parseDate('2021')` reports `rule: 'decimal_year'`, `imputed: false`.
 *
 *  Q3. THE DAY IS CAPPED AT 30, IN EVERY MONTH. `datetime.date(year, month, min(day, 28 if
 *      month == 2 else 30))` (temporal.py:144). So `2021-01-31` is read as 31 January and
 *      returned as 30 January, and 29 February of a leap year is unreachable. Reported as
 *      `imputations.dayClamped`.
 *
 *  Q4. AN INVALID MONTH OR DAY IS DISCARDED WITHOUT COMPLAINT. `2021-13-40` keeps the defaults and
 *      comes back as mid-2021 (temporal.py:133-142), indistinguishable from a bare `2021` except
 *      by `imputations`.
 *
 *  Q5. '.' IS A DATE DELIMITER. `clean_str = val_str.replace('/', '-').replace('.', '-')`
 *      (temporal.py:123) runs only after `float(val_str)` has been tried, so `2021.25` is a decimal
 *      year but `2021.4.15` is 15 April 2021 and `2021.5` — which `float` accepts and the gate
 *      admits — is mid-2021 by the decimal-year rule, not by the month rule.
 *
 *  Q6. THE OUT-OF-GATE VALUE VANISHES. A year outside [1800, 2100] returns NaN with no distinction
 *      from an unparseable string (temporal.py:108, 130). Kept, but separated here into
 *      `rule: 'out_of_range'` vs `rule: 'unparsed'` so a page can say which happened.
 *
 *  Q7. THE HEADER PATTERNS NEED A DELIMITER. `extract_date_from_string`'s separator class is
 *      `[\|/_\s]` — NOT '-' (temporal.py:218-239) — so `A-2021-05-14` does not match pattern 1,
 *      and pattern 4 (a trailing 4-digit year) has no `^` alternative, so a header that IS just
 *      `2021` is not matched at all. Pattern 2 requires two to four decimals, so `|2021.5` is not
 *      a decimal year here.
 *
 *  Q8. KORBER / LANL YEAR PIVOT. A two-digit year >= 30 is 19xx, below 30 is 20xx
 *      (dating.py:344, 351). Hard-coded; 2030 onwards will be read as 1930.
 *
 *  Q9. `WPI` AND `DPI` ARE CASE-SENSITIVE PAIRS. `(?:WPI|wpi)` (dating.py:355) matches neither
 *      `Wpi` nor `wPI`, while the non-calendar unit patterns use `re.IGNORECASE` throughout.
 *
 *  Q10. NON-CALENDAR ACCEPTS INFINITY. `float('inf') >= 0.0` is True, so `parse_date_to_decimal
 *      ('inf', time_units='days')` returns +Infinity (temporal.py:92-93). NaN is rejected by the
 *      same comparison.
 *
 * REGEX AND `float()` SEMANTICS. The patterns are transcribed character for character; the two
 * engines differ only where Python's `\s` and `re.IGNORECASE` are Unicode-aware over a `str` and
 * JavaScript's are not by default. `pyFloatOrNull` below is CPython's `float(str)` — underscore
 * separators, `inf`/`nan` spellings, and the whitespace `float()` strips, which is not the set
 * `str.strip()` removes — the same definition `evaluate.js` uses privately for `evaluation.py`.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: it does not read a file, does not know what a taxon is, does
 * not decide that an undated sequence should be dropped or that a clamped day deserves a warning.
 * Those are result semantics and belong to the application.
 */

import { pyStrip, pyIsDigit, pyLen } from './preprocess/parse.js';

// -------------------------------------------------------------------------------------------
// Constants, all from the reference
// -------------------------------------------------------------------------------------------

/** The calendar interpretation (`time_units='years'`, temporal.py:73). */
export const CALENDAR_TIME_UNITS = 'years';

/** The non-calendar interpretations, verbatim from temporal.py:90 and 204. */
export const NON_CALENDAR_TIME_UNITS = Object.freeze(['generations', 'days', 'arbitrary']);

/** Every `time_units` value the reference names. Anything else falls through to the calendar path. */
export const TIME_UNITS = Object.freeze([CALENDAR_TIME_UNITS, ...NON_CALENDAR_TIME_UNITS]);

/** The calendar gate, temporal.py:106, 117, 129. Both bounds inclusive. */
export const CALENDAR_YEAR_MIN = 1800;
/** @see CALENDAR_YEAR_MIN */
export const CALENDAR_YEAR_MAX = 2100;

/** The month used when none was read (temporal.py:132). June, so a bare year lands mid-year. */
export const IMPUTED_MONTH = 6;
/** The day used when none was read (temporal.py:138). */
export const IMPUTED_DAY = 15;
/** `min(day, 28 if month == 2 else 30)` (temporal.py:144) — Q3. */
export const FEBRUARY_DAY_CAP = 28;
/** @see FEBRUARY_DAY_CAP */
export const MONTH_DAY_CAP = 30;

/** Lower-cased strings temporal.py:111 treats as "no date". */
export const NULL_DATE_TOKENS = Object.freeze(['unknown', 'nan', 'none', 'na', '?']);

/** The value dating.py:332 returns for the archival ZR59 isolate — Q1, opt-in here. */
export const ARCHIVAL_1959_DECIMAL = 1959.5;
/** The substrings dating.py:331 tests for, in its order — Q1. */
export const ARCHIVAL_1959_MARKERS = Object.freeze(['Z59', 'ZR59', '1959']);

/** Two-digit years at or above this are 19xx, below it 20xx (dating.py:344, 351) — Q8. */
export const KORBER_YEAR_PIVOT = 30;

/**
 * Which rule produced a `DateParse.value`. Every parse reports exactly one of these, including the
 * failures, so a caller can say WHY a sequence has no date rather than only that it has none.
 */
export const DATE_RULES = Object.freeze({
	/** No value at all: null, undefined, NaN, the empty string, or a `NULL_DATE_TOKENS` word. */
	NONE: 'none',
	/** Non-calendar, `float(val)` succeeded and was >= 0 (temporal.py:91-94). */
	NON_CALENDAR_NUMERIC: 'non_calendar_numeric',
	/** Non-calendar, first `\d+(\.\d+)?` run of the string (temporal.py:96-101). */
	NON_CALENDAR_EMBEDDED: 'non_calendar_embedded',
	/** Calendar, the value arrived as a number inside the gate (temporal.py:104-108). */
	NUMERIC: 'numeric',
	/** Calendar, the string parsed as a float inside the gate (temporal.py:115-120). */
	DECIMAL_YEAR: 'decimal_year',
	/** Calendar, the delimiter path: four-digit year plus optional month and day (temporal.py:126-147). */
	YMD: 'ymd',
	/** A year was read and fell outside [1800, 2100] (temporal.py:108, 130) — Q6. */
	OUT_OF_RANGE: 'out_of_range',
	/** Nothing matched (temporal.py:151). */
	UNPARSED: 'unparsed',

	/** Header pattern 1, `[\|/_\s]YYYY-MM-DD` (temporal.py:218). */
	HEADER_ISO: 'header_iso',
	/** Header pattern 2, `[\|/_\s]YYYY.DDDD` with two to four decimals (temporal.py:225) — Q7. */
	HEADER_DECIMAL_YEAR: 'header_decimal_year',
	/** Header pattern 3, `[\|/_\s]YYYY-MM` (temporal.py:232). */
	HEADER_YEAR_MONTH: 'header_year_month',
	/** Header pattern 4, a delimiter and a four-digit year at the end (temporal.py:239) — Q7. */
	HEADER_TRAILING_YEAR: 'header_trailing_year',
	/** Non-calendar header, an explicit unit prefix: `gen_5000`, `|day-3` (temporal.py:205). */
	HEADER_UNIT_TOKEN: 'header_unit_token',
	/** Non-calendar header, a unit suffix: `_20000gen`, `|50d` (temporal.py:207). */
	HEADER_UNIT_SUFFIX: 'header_unit_suffix',
	/** Non-calendar header, a bare delimiter-bound number: `|5000` (temporal.py:209). */
	HEADER_BARE_NUMBER: 'header_bare_number',

	/** The opt-in archival anchor (dating.py:331-332) — Q1. */
	ARCHIVAL_1959: 'archival_1959',
	/** Korber HIV-1 isolate, `^[A-Za-z]\d\d[A-Za-z]{2}[._]` (dating.py:341) — Q8. */
	KORBER_ISOLATE: 'korber_isolate',
	/** Nextstrain / LANL `_XX_86_` year (dating.py:348) — Q8. */
	LANL_PIPE_YEAR: 'lanl_pipe_year',
	/** Weeks post infection (dating.py:355) — Q9. */
	WPI: 'wpi',
	/** Days post infection (dating.py:360) — Q9. */
	DPI: 'dpi',

	/** `_parse_timestamp_flexible`'s ungated numeric fallback (dating.py:378-381). */
	FLEXIBLE_NUMERIC: 'flexible_numeric'
});

/**
 * A parse, with its provenance.
 *
 * @typedef {Object} DateParse
 * @property {number} value the decimal time coordinate, `NaN` when nothing was read. Identical to
 *   what the corresponding reference function returns (except under Q1's opt-in).
 * @property {string} rule which rule produced it — one of `DATE_RULES`.
 * @property {boolean} imputed true when any part of `value` was supplied rather than read; the
 *   disjunction of `imputations`.
 * @property {{month: boolean, day: boolean, dayClamped: boolean}} imputations `month` / `day`: the
 *   component was defaulted (Q2) or discarded as invalid (Q4). `dayClamped`: the day the reader
 *   wrote was reduced by the 28/30 cap (Q3).
 * @property {string|null} matched the substring the rule consumed, when the rule searched inside a
 *   longer string; `null` when the whole input was the value.
 * @property {string} timeUnits the interpretation this parse used.
 */

// -------------------------------------------------------------------------------------------
// CPython `float(str)`, private (evaluate.js keeps its own copy for evaluation.py)
// -------------------------------------------------------------------------------------------

/** Unicode decimal digits, for the `float()` transform CPython applies before parsing. */
const ND = /\p{Nd}/u;

/** CPython `_PyUnicode_TransformDecimalAndSpaceToASCII`: any Nd digit becomes its ASCII value. */
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

/**
 * The whitespace `float()` strips, which is NOT the set `str.strip()` removes: CPython maps
 * non-ASCII Unicode whitespace to ' ' and then skips only C `isspace`, so U+00A0 and U+3000 go and
 * \x1c..\x1f stay.
 */
const NUMERIC_WS = '[ \\t\\n\\v\\f\\r\\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]';
const NUMERIC_TRIM_RE = new RegExp(`^${NUMERIC_WS}+|${NUMERIC_WS}+$`, 'g');

/**
 * Python `float(value)`, or `null` where Python raises `ValueError` / `TypeError` — which is what
 * every call site here does with the exception.
 *
 * @param {unknown} value
 * @returns {number|null}
 */
function pyFloatOrNull(value) {
	if (typeof value === 'number') return value;
	if (typeof value === 'boolean') return value ? 1.0 : 0.0;
	if (typeof value !== 'string') return null;
	const s = asciiDigits(value.replace(NUMERIC_TRIM_RE, ''));
	if (PY_FLOAT_RE.test(s)) return parseFloat(s.replace(/_/g, ''));
	const lower = s.toLowerCase();
	if (/^[+-]?(?:inf|infinity)$/.test(lower)) return lower.startsWith('-') ? -Infinity : Infinity;
	if (/^[+-]?nan$/.test(lower)) return NaN;
	return null;
}

/** `int(s)` for a string `pyIsDigit` accepted. */
function digitsToInt(s) {
	let v = 0;
	for (const ch of s) {
		const cp = /** @type {number} */ (ch.codePointAt(0));
		// The digit's value is how many Nd code points sit directly below it in its own block.
		let k = 0;
		while (k < 60 && ND.test(String.fromCodePoint(cp - k - 1))) k++;
		v = v * 10 + (k % 10);
	}
	return v;
}

// -------------------------------------------------------------------------------------------
// Record construction
// -------------------------------------------------------------------------------------------

/**
 * @param {number} value
 * @param {string} rule
 * @param {string} timeUnits
 * @param {{month?: boolean, day?: boolean, dayClamped?: boolean, matched?: string|null}} [extra]
 * @returns {DateParse}
 */
function parse(value, rule, timeUnits, extra = {}) {
	const month = extra.month === true;
	const day = extra.day === true;
	const dayClamped = extra.dayClamped === true;
	return {
		value,
		rule,
		imputed: month || day || dayClamped,
		imputations: { month, day, dayClamped },
		matched: extra.matched ?? null,
		timeUnits
	};
}

/**
 * `val is None or pd.isna(val)` (temporal.py:86, dating.py:373) for the value types this library
 * accepts: a string, a number, a boolean, `null` or `undefined`.
 *
 * @param {unknown} value
 */
function isMissing(value) {
	return value === null || value === undefined || (typeof value === 'number' && Number.isNaN(value));
}

/**
 * Is `timeUnits` one of the non-calendar interpretations? The reference's own membership test, so
 * an unrecognised string is calendar, as `if time_units in (...)` makes it.
 *
 * @param {string} timeUnits
 * @returns {boolean}
 */
export function isNonCalendarTimeUnits(timeUnits) {
	return NON_CALENDAR_TIME_UNITS.includes(timeUnits);
}

// -------------------------------------------------------------------------------------------
// parse_date_to_decimal (temporal.py:73-151)
// -------------------------------------------------------------------------------------------

/** Days before the first of each month in a non-leap year. */
const CUMULATIVE_DAYS = Object.freeze([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]);

/** `year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)` (temporal.py:146). */
function isLeapYear(year) {
	return year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
}

/**
 * `float(year + (dt - start_of_year).days / days_in_year)` (temporal.py:144-147) with the day
 * already capped by the caller. Same two double operations as the Python, in the same order.
 */
function decimalOfYmd(year, month, day) {
	const leap = isLeapYear(year);
	const doy = CUMULATIVE_DAYS[month - 1] + (leap && month > 2 ? 1 : 0) + day - 1;
	return year + doy / (leap ? 366 : 365);
}

/**
 * `parse_date_to_decimal` (temporal.py:73-151) with its provenance.
 *
 * Calendar (`timeUnits: 'years'`): a number or numeric string inside [1800, 2100] is the answer;
 * otherwise '/' and '.' are normalised to '-' and a four-digit leading year is read, with month
 * defaulted to June and day to the 15th (Q2, Q4) and the day capped at 28/30 (Q3).
 *
 * Non-calendar (`'generations' | 'days' | 'arbitrary'`): any non-negative real, or the first number
 * embedded in the string; no year gate (Q10).
 *
 * @param {string|number|boolean|null|undefined} value
 * @param {{timeUnits?: string}} [options]
 * @returns {DateParse}
 */
export function parseDate(value, options = {}) {
	const timeUnits = options.timeUnits ?? CALENDAR_TIME_UNITS;

	if (isMissing(value)) return parse(NaN, DATE_RULES.NONE, timeUnits);

	// Non-calendar time coordinates: accept any non-negative real (no year gate). temporal.py:90-102.
	if (isNonCalendarTimeUnits(timeUnits)) {
		const direct = pyFloatOrNull(value);
		if (direct !== null) {
			// `val_f if val_f >= 0.0 else np.nan` — NaN fails the comparison, Infinity passes (Q10).
			if (direct >= 0.0) return parse(direct, DATE_RULES.NON_CALENDAR_NUMERIC, timeUnits);
			// A negative real, or a literal "nan": converted, then rejected by the >= 0 gate. The
			// reference RETURNS here rather than falling through to the embedded-number search.
			return parse(NaN, DATE_RULES.OUT_OF_RANGE, timeUnits);
		}
		const m = /\d+(?:\.\d+)?/.exec(String(value));
		if (m) return parse(parseFloat(m[0]), DATE_RULES.NON_CALENDAR_EMBEDDED, timeUnits, { matched: m[0] });
		return parse(NaN, DATE_RULES.UNPARSED, timeUnits);
	}

	// `isinstance(val, (int, float))` — a Python bool is an int, so `True` is 1.0 and fails the gate.
	if (typeof value === 'number' || typeof value === 'boolean') {
		const valF = typeof value === 'boolean' ? (value ? 1.0 : 0.0) : value;
		if (valF >= CALENDAR_YEAR_MIN && valF <= CALENDAR_YEAR_MAX) {
			return parse(valF, DATE_RULES.NUMERIC, timeUnits);
		}
		return parse(NaN, DATE_RULES.OUT_OF_RANGE, timeUnits);
	}

	const valStr = pyStrip(String(value));
	if (valStr === '' || NULL_DATE_TOKENS.includes(valStr.toLowerCase())) {
		return parse(NaN, DATE_RULES.NONE, timeUnits);
	}

	// Try direct float conversion (e.g. "2021.25"). Out of range falls THROUGH to the delimiter
	// path rather than returning — temporal.py:115-120, and the reason "1700" reports out_of_range
	// from the year check below rather than from here.
	const direct = pyFloatOrNull(valStr);
	if (direct !== null && direct >= CALENDAR_YEAR_MIN && direct <= CALENDAR_YEAR_MAX) {
		return parse(direct, DATE_RULES.DECIMAL_YEAR, timeUnits);
	}

	// Normalise delimiters — Q5: '.' is a date separator here, but only because the float attempt
	// above already claimed every string that is a plain decimal year.
	const cleanStr = valStr.replace(/\//g, '-').replace(/\./g, '-');
	const parts = cleanStr.split('-');

	if (parts.length >= 1 && pyIsDigit(parts[0]) && pyLen(parts[0]) === 4) {
		const year = digitsToInt(parts[0]);
		if (!(year >= CALENDAR_YEAR_MIN && year <= CALENDAR_YEAR_MAX)) {
			return parse(NaN, DATE_RULES.OUT_OF_RANGE, timeUnits);
		}

		let month = IMPUTED_MONTH;
		let monthImputed = true;
		if (parts.length >= 2 && pyIsDigit(parts[1])) {
			const m = digitsToInt(parts[1]);
			if (m >= 1 && m <= 12) {
				month = m;
				monthImputed = false;
			}
		}

		let day = IMPUTED_DAY;
		let dayImputed = true;
		if (parts.length >= 3 && pyIsDigit(parts[2])) {
			const d = digitsToInt(parts[2]);
			if (d >= 1 && d <= 31) {
				day = d;
				dayImputed = false;
			}
		}

		const cap = month === 2 ? FEBRUARY_DAY_CAP : MONTH_DAY_CAP;
		const cappedDay = Math.min(day, cap);
		return parse(decimalOfYmd(year, month, cappedDay), DATE_RULES.YMD, timeUnits, {
			month: monthImputed,
			day: dayImputed,
			dayClamped: !dayImputed && cappedDay !== day
		});
	}

	return parse(NaN, DATE_RULES.UNPARSED, timeUnits);
}

/**
 * `parse_date_to_decimal` (temporal.py:73) exactly: the decimal time coordinate, `NaN` for nothing.
 *
 * @param {string|number|boolean|null|undefined} value
 * @param {{timeUnits?: string}} [options]
 * @returns {number}
 */
export function parseDateToDecimal(value, options = {}) {
	return parseDate(value, options).value;
}

// -------------------------------------------------------------------------------------------
// extract_date_from_string (temporal.py:196-245)
// -------------------------------------------------------------------------------------------

/** temporal.py:205. `gen`/`generation` before `g` matters only through backtracking; kept verbatim. */
const RE_UNIT_TOKEN = /(?:[|/_\-\s]|^)(?:gen|generation|g|day|d|t)[-_]?(\d+(?:\.\d+)?)(?:[|/_\-\s]|$)/i;
/** temporal.py:207. */
const RE_UNIT_SUFFIX = /(?:[|/_\-\s]|^)(\d+(?:\.\d+)?)(?:gen|g|d)(?:[|/_\-\s]|$)/i;
/** temporal.py:209. */
const RE_BARE_NUMBER = /(?:[|/_\-\s]|^)(\d+(?:\.\d+)?)(?:[|/_\-\s]|$)/;
/** temporal.py:218. The separator class has no '-' — Q7. */
const RE_HEADER_ISO = /(?:[|/_\s]|^)(\d{4}-\d{2}-\d{2})(?:[|/_\s]|$)/;
/** temporal.py:225. Two to four decimals, so `|2021.5` does not match — Q7. */
const RE_HEADER_DECIMAL = /(?:[|/_\s]|^)(\d{4}\.\d{2,4})(?:[|/_\s]|$)/;
/** temporal.py:232. */
const RE_HEADER_YEAR_MONTH = /(?:[|/_\s]|^)(\d{4}-\d{2})(?:[|/_\s]|$)/;
/** temporal.py:239. No `^` alternative, so a header that IS a bare year does not match — Q7. */
const RE_HEADER_TRAILING_YEAR = /(?:[|/_\s])(\d{4})$/;

const CALENDAR_HEADER_PATTERNS = Object.freeze([
	Object.freeze({ re: RE_HEADER_ISO, rule: DATE_RULES.HEADER_ISO }),
	Object.freeze({ re: RE_HEADER_DECIMAL, rule: DATE_RULES.HEADER_DECIMAL_YEAR }),
	Object.freeze({ re: RE_HEADER_YEAR_MONTH, rule: DATE_RULES.HEADER_YEAR_MONTH }),
	Object.freeze({ re: RE_HEADER_TRAILING_YEAR, rule: DATE_RULES.HEADER_TRAILING_YEAR })
]);

/**
 * `extract_date_from_string` (temporal.py:196-245) with its provenance: the time coordinate a FASTA
 * header or any other name carries, under the four calendar patterns or the three non-calendar
 * ones. The rule reported is the PATTERN that matched; `imputations` come from `parseDate`, which
 * converts whatever the pattern captured (Q2 — a year-month match imputes the day).
 *
 * @param {string|null|undefined} name
 * @param {{timeUnits?: string}} [options]
 * @returns {DateParse}
 */
export function extractDate(name, options = {}) {
	const timeUnits = options.timeUnits ?? CALENDAR_TIME_UNITS;
	if (!name) return parse(NaN, DATE_RULES.NONE, timeUnits);

	if (isNonCalendarTimeUnits(timeUnits)) {
		// Explicit unit prefix, then unit suffix, then a bare delimiter-bound number.
		for (const [re, rule] of /** @type {[RegExp, string][]} */ ([
			[RE_UNIT_TOKEN, DATE_RULES.HEADER_UNIT_TOKEN],
			[RE_UNIT_SUFFIX, DATE_RULES.HEADER_UNIT_SUFFIX],
			[RE_BARE_NUMBER, DATE_RULES.HEADER_BARE_NUMBER]
		])) {
			const m = re.exec(name);
			if (m) {
				// `float(m.group(1))` — the capture is always `\d+(\.\d+)?`, so it never raises.
				return parse(parseFloat(m[1]), rule, timeUnits, { matched: m[1] });
			}
		}
		return parse(NaN, DATE_RULES.UNPARSED, timeUnits);
	}

	for (const { re, rule } of CALENDAR_HEADER_PATTERNS) {
		const m = re.exec(name);
		if (!m) continue;
		const d = parseDate(m[1], { timeUnits });
		if (!Number.isNaN(d.value)) {
			return parse(d.value, rule, timeUnits, {
				month: d.imputations.month,
				day: d.imputations.day,
				dayClamped: d.imputations.dayClamped,
				matched: m[1]
			});
		}
		// The reference falls through to the next pattern when the capture did not convert.
	}

	return parse(NaN, DATE_RULES.UNPARSED, timeUnits);
}

/**
 * `extract_date_from_string` (temporal.py:196) exactly.
 *
 * @param {string|null|undefined} name
 * @param {{timeUnits?: string}} [options]
 * @returns {number}
 */
export function extractDateFromString(name, options = {}) {
	return extractDate(name, options).value;
}

// -------------------------------------------------------------------------------------------
// parse_header_timestamp (dating.py:317-364)
// -------------------------------------------------------------------------------------------

/** dating.py:341. '.' inside the class is literal. */
const RE_KORBER = /^[A-Za-z](\d{2})[A-Za-z]{2}[._]/;
/** dating.py:348. Uppercase country code only. */
const RE_LANL_PIPE = /_(?:[A-Z]{2})_(\d{2})_/;
/** dating.py:355 — Q9, `Wpi` does not match. */
const RE_WPI = /(\d+(?:\.\d+)?)\s*(?:WPI|wpi)/;
/** dating.py:360 — Q9. */
const RE_DPI = /(\d+(?:\.\d+)?)\s*(?:DPI|dpi)/;

/**
 * The predicate dating.py:331 applies before every other rule — Q1. Exposed on its own so an
 * application can OFFER the archival interpretation (and name the sequences it would change)
 * instead of applying it silently.
 *
 * @param {string|null|undefined} name
 * @returns {boolean}
 */
export function matchesArchival1959(name) {
	if (!name) return false;
	return ARCHIVAL_1959_MARKERS.some((marker) => name.includes(marker));
}

/**
 * `parse_header_timestamp` (dating.py:317-364) with its provenance: the calendar patterns of
 * `extractDate`, then the Korber HIV-1 isolate code, the LANL two-digit year, and weeks / days post
 * infection.
 *
 * THE 1959 ANCHOR IS OPT-IN (Q1). The reference returns 1959.5 for any header containing `Z59`,
 * `ZR59` or `1959`, before it looks at anything else — which overwrites a real `1959-03-04` and
 * mis-dates an accession like `AZ59012`. Pass `{archival1959: true}` to reproduce the reference
 * exactly; the default skips the rule and lets the date patterns run.
 *
 * This function is calendar-only, as the reference is: it calls `extract_date_from_string(name)`
 * with the default units and has no non-calendar branch. For generations or days use `extractDate`
 * with `timeUnits` instead.
 *
 * @param {string|null|undefined} name
 * @param {{archival1959?: boolean}} [options]
 * @returns {DateParse}
 */
export function parseHeaderDate(name, options = {}) {
	const units = CALENDAR_TIME_UNITS;
	if (!name) return parse(NaN, DATE_RULES.NONE, units);

	// 1. The archival 1959 ZR59 anchor — Q1, opt-in.
	if (options.archival1959 === true && matchesArchival1959(name)) {
		const marker = ARCHIVAL_1959_MARKERS.find((m) => name.includes(m)) ?? null;
		// 1959.5 is a year with the month and day invented, exactly as a bare year would be.
		return parse(ARCHIVAL_1959_DECIMAL, DATE_RULES.ARCHIVAL_1959, units, {
			month: true,
			day: true,
			matched: marker
		});
	}

	// 2. Standard ISO / decimal extraction.
	const extracted = extractDate(name, { timeUnits: units });
	if (!Number.isNaN(extracted.value)) return extracted;

	// 3. Korber HIV-1 isolate pattern: [Subtype][2-digit year][Country].[Strain].
	const korber = RE_KORBER.exec(name);
	if (korber) {
		return parse(korberYear(korber[1]), DATE_RULES.KORBER_ISOLATE, units, {
			month: true,
			day: true,
			matched: korber[1]
		});
	}

	// 4. Nextstrain / LANL format with a two-digit year: Ref_B_FR_83_HXB2.
	const lanl = RE_LANL_PIPE.exec(name);
	if (lanl) {
		return parse(korberYear(lanl[1]), DATE_RULES.LANL_PIPE_YEAR, units, {
			month: true,
			day: true,
			matched: lanl[1]
		});
	}

	// 5. Weeks post infection, and 6. days post infection. Both are elapsed times, not calendar
	// dates: the reference returns them on the same axis regardless, and so does this.
	const wpi = RE_WPI.exec(name);
	if (wpi) return parse(parseFloat(wpi[1]), DATE_RULES.WPI, units, { matched: wpi[1] });

	const dpi = RE_DPI.exec(name);
	if (dpi) return parse(parseFloat(dpi[1]), DATE_RULES.DPI, units, { matched: dpi[1] });

	return parse(NaN, DATE_RULES.UNPARSED, units);
}

/** `1900 + yr if yr >= 30 else 2000 + yr`, then `+ 0.5` (dating.py:344-345, 351-352) — Q8. */
function korberYear(twoDigits) {
	const yr = Number(twoDigits);
	const full = yr >= KORBER_YEAR_PIVOT ? 1900 + yr : 2000 + yr;
	return full + 0.5;
}

/**
 * `parse_header_timestamp` (dating.py:317). Pass `{archival1959: true}` for the reference's own
 * behaviour; see `parseHeaderDate` for why the default differs (Q1).
 *
 * @param {string|null|undefined} name
 * @param {{archival1959?: boolean}} [options]
 * @returns {number}
 */
export function parseHeaderTimestamp(name, options = {}) {
	return parseHeaderDate(name, options).value;
}

// -------------------------------------------------------------------------------------------
// _parse_timestamp_flexible (dating.py:367-384)
// -------------------------------------------------------------------------------------------

/**
 * `_parse_timestamp_flexible` (dating.py:367-384) with its provenance: the calendar parse first,
 * then any float at all, so an experimental coordinate (`0.25`, `5000`) survives the [1800, 2100]
 * gate that `parseDate` would have rejected.
 *
 * Note the asymmetry it inherits: the ungated fallback accepts `nan` and `inf` spellings through
 * `float()`, and `float('nan')` is not caught by the `np.isnan(val_f)` guard above it because that
 * guard tests the result of the SUCCESSFUL conversion — `if not np.isnan(val_f): return val_f`
 * means a literal `"nan"` string falls through and the function returns NaN anyway.
 *
 * @param {string|number|boolean|null|undefined} value
 * @returns {DateParse}
 */
export function parseFlexibleDate(value) {
	const units = CALENDAR_TIME_UNITS;
	if (isMissing(value)) return parse(NaN, DATE_RULES.NONE, units);

	const calendar = parseDate(value, { timeUnits: units });
	if (!Number.isNaN(calendar.value)) return calendar;

	const valF = pyFloatOrNull(value);
	if (valF !== null && !Number.isNaN(valF)) return parse(valF, DATE_RULES.FLEXIBLE_NUMERIC, units);

	return parse(NaN, calendar.rule === DATE_RULES.NONE ? DATE_RULES.NONE : DATE_RULES.UNPARSED, units);
}

/**
 * `_parse_timestamp_flexible` (dating.py:367) exactly.
 *
 * @param {string|number|boolean|null|undefined} value
 * @returns {number}
 */
export function parseTimestampFlexible(value) {
	return parseFlexibleDate(value).value;
}
