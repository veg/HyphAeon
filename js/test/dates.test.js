/**
 * WHY THIS FILE EXISTS
 *
 * src/dates.js is the only part of the time-aware pillars that is pure arithmetic over strings, and
 * it is the part every surface will reach for first. Two different things have to be pinned, and
 * they fail in different ways:
 *
 *   1. THE NUMBERS ARE THE REFERENCE'S. `fixtures/dates/*.json` is produced by the four reference
 *      functions themselves — lifted out of hyphaeon/temporal.py and hyphaeon/dating.py by `ast`
 *      rather than retyped (scripts/gen_fixtures.py, `load_date_reference`; regenerate with
 *      `--only dates`, which needs no weights and no torch) — so every case below is a replay, not
 *      an opinion about what the Python ought to do. Each case is asserted input by input at the
 *      `exact` class, because a single wrong delimiter class turns a whole surveillance dataset
 *      silently undated, and because the conversion is integer arithmetic plus one division: both
 *      engines round-trip IEEE-754 doubles through their decimal spellings, so there is nothing for
 *      a tolerance to absorb.
 *
 *   2. THE PROVENANCE IS THE PORT'S. `rule`, `imputed` and `imputations` have no Python counterpart
 *      — the reference returns a bare float — so those expectations are written here, against the
 *      quirks named Q1..Q10 in the module header. They are the contract the application's review
 *      page is built on: which rule fired, and whether the answer was read or invented. One test
 *      replays every fixture case through BOTH forms and asserts they never disagree, so the record
 *      cannot drift away from the number it describes.
 *
 * The 1959 anchor (Q1) is checked in BOTH directions: `{archival1959: true}` against the unmodified
 * reference table in `fixtures/dates/parse_header_timestamp.json`, and the default against
 * `test/data/dates/parsers.json`, which the generator there produces by running the reference's
 * remaining steps in the reference's own order. That table has no reference function behind it —
 * which is why it is the one date table that is NOT a fixture. A port that quietly kept the anchor
 * on and a port that quietly dropped it entirely both fail here.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
	ARCHIVAL_1959_DECIMAL,
	CALENDAR_YEAR_MAX,
	CALENDAR_YEAR_MIN,
	DATE_RULES,
	IMPUTED_DAY,
	IMPUTED_MONTH,
	KORBER_YEAR_PIVOT,
	NON_CALENDAR_TIME_UNITS,
	NULL_DATE_TOKENS,
	TIME_UNITS,
	extractDate,
	extractDateFromString,
	isNonCalendarTimeUnits,
	matchesArchival1959,
	parseDate,
	parseDateToDecimal,
	parseFlexibleDate,
	parseHeaderDate,
	parseHeaderTimestamp,
	parseTimestampFlexible
} from '../src/dates.js';

const FIXTURES = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'fixtures');
const loadFixture = (fn) => JSON.parse(readFileSync(join(FIXTURES, 'dates', `${fn}.json`), 'utf8'));
const MANIFEST = JSON.parse(readFileSync(join(FIXTURES, 'manifest.json'), 'utf8'));

const PARSE_DATE = loadFixture('parse_date_to_decimal');
const EXTRACT_DATE = loadFixture('extract_date_from_string');
const HEADER_TIMESTAMP = loadFixture('parse_header_timestamp');
const FLEXIBLE = loadFixture('_parse_timestamp_flexible');

/** The anchor-off table, the one date table with no reference function behind it (Q1). */
const NO_ANCHOR = JSON.parse(
	readFileSync(fileURLToPath(new URL('./data/dates/parsers.json', import.meta.url)), 'utf8')
);

/** The fixtures/README.md convention: non-finite floats travel as strings. */
function decode(v) {
	if (v === 'NaN') return NaN;
	if (v === 'Infinity') return Infinity;
	if (v === '-Infinity') return -Infinity;
	return v;
}

/**
 * `Object.is` up to the double rounding the two engines share. Python and JavaScript both print and
 * parse IEEE-754 doubles round-trip, so every finite reference value is exact; NaN, ±0 and ±Infinity
 * are compared by identity, which is what `Object.is` does and `toBe` uses.
 */
function expectSame(actual, expected, label) {
	if (Number.isNaN(expected)) {
		expect(actual, label).toBeNaN();
	} else {
		expect(actual, label).toBe(expected);
	}
}

/**
 * Replay one fixture file through one call, case by case.
 *
 * @param {Array<object>} cases
 * @param {(inputs: object) => number} call
 * @param {string} label
 */
function replayFixture(cases, call, label) {
	expect(cases.length, `${label} is non-empty`).toBeGreaterThan(0);
	for (const c of cases) {
		expect(c.tolerance, `${label}/${c.name} tolerance`).toBe('exact');
		expectSame(call(c.inputs), decode(c.outputs.result), `${label}/${c.name}`);
	}
}

/** The four fixture tables, each with the record form and the bare-float form of its port. */
const TABLES = [
	{
		file: 'parse_date_to_decimal',
		cases: PARSE_DATE,
		record: (i) => parseDate(i.val, { timeUnits: i.time_units }),
		value: (i) => parseDateToDecimal(i.val, { timeUnits: i.time_units })
	},
	{
		file: 'extract_date_from_string',
		cases: EXTRACT_DATE,
		record: (i) => extractDate(i.name, { timeUnits: i.time_units }),
		value: (i) => extractDateFromString(i.name, { timeUnits: i.time_units })
	},
	{
		// The reference VERBATIM, anchor included: the port reproduces it only on request (Q1).
		file: 'parse_header_timestamp',
		cases: HEADER_TIMESTAMP,
		record: (i) => parseHeaderDate(i.name, { archival1959: true }),
		value: (i) => parseHeaderTimestamp(i.name, { archival1959: true })
	},
	{
		file: '_parse_timestamp_flexible',
		cases: FLEXIBLE,
		record: (i) => parseFlexibleDate(i.val),
		value: (i) => parseTimestampFlexible(i.val)
	}
];

describe('the reference fixtures, replayed', () => {
	it('was generated from the reference functions at the line ranges it names', () => {
		// Both generators record the ast lift's line ranges, so a move upstream shows up as a diff in
		// the data rather than as an unexplained failure — and the two tables cannot drift apart.
		const expected = {
			parse_date_to_decimal: 'hyphaeon/temporal.py:73-151',
			extract_date_from_string: 'hyphaeon/temporal.py:196-245',
			parse_header_timestamp: 'hyphaeon/dating.py:317-364',
			_parse_timestamp_flexible: 'hyphaeon/dating.py:367-384'
		};
		expect(MANIFEST.date_reference_source).toEqual(expected);
		expect(NO_ANCHOR.source).toEqual(expected);
	});

	it('is the size the manifest records, so a silently shortened table fails', () => {
		expect(MANIFEST.counts.dates).toEqual({
			parse_date_to_decimal: 135,
			extract_date_from_string: 44,
			parse_header_timestamp: 26,
			_parse_timestamp_flexible: 19
		});
		for (const { file, cases } of TABLES) {
			expect(cases.length, file).toBe(MANIFEST.counts.dates[file]);
		}
	});

	it('parse_date_to_decimal: calendar, the three non-calendar units, and an unknown unit', () => {
		replayFixture(PARSE_DATE, (i) => parseDateToDecimal(i.val, { timeUnits: i.time_units }), 'parse_date_to_decimal');
		// Every unit the fixture exercises, so a table that lost its non-calendar half is visible.
		expect(new Set(PARSE_DATE.map((c) => c.inputs.time_units))).toEqual(
			new Set(['years', 'generations', 'days', 'arbitrary', 'fortnights'])
		);
	});

	it('extract_date_from_string: the header patterns, and the five H1N1 headers they lose', () => {
		replayFixture(EXTRACT_DATE, (i) => extractDateFromString(i.name, { timeUnits: i.time_units }), 'extract_date_from_string');

		// The real-data proof of Q7, measured on examples/H1N1_2009_pandemic.fasta: three headers
		// whose isolate number shadows the real decimal year, two written to one decimal place.
		const undated = EXTRACT_DATE.filter((c) => c.notes.includes('H1N1_2009_pandemic.fasta'));
		expect(undated.length).toBe(5);
		for (const c of undated) {
			expect(extractDateFromString(c.inputs.name), c.name).toBeNaN();
			// The date IS in the header; nothing but the pattern set stops it being read.
			expect(c.inputs.name).toMatch(/\|20\d\d\.\d+$/);
		}
	});

	it('parse_header_timestamp with the 1959 anchor ON is the reference verbatim (Q1)', () => {
		replayFixture(HEADER_TIMESTAMP, (i) => parseHeaderTimestamp(i.name, { archival1959: true }), 'parse_header_timestamp');
	});

	it('parse_header_timestamp with the anchor OFF is the default, and differs on four names (Q1)', () => {
		const cases = NO_ANCHOR.parse_header_timestamp_without_1959;
		expect(cases.length).toBe(HEADER_TIMESTAMP.length);
		for (const { input, output } of cases) {
			expectSame(parseHeaderTimestamp(input), decode(output), `anchor off: ${JSON.stringify(input)}`);
		}

		// The whole point of making it opt-in: these are the headers the reference mis-dates. The two
		// tables carry the same inputs in the same order, so they line up case for case.
		const changed = HEADER_TIMESTAMP.map((c, i) => [c.inputs.name, c.outputs.result, cases[i].output])
			.filter(([, a, b]) => a !== b)
			.map(([name]) => name);
		expect(changed).toEqual(['ZR59', 'AZ59012', 'A/Kinshasa/1959-03-04', 'X|1959-03-04']);
		expect(cases.map((c) => c.input)).toEqual(HEADER_TIMESTAMP.map((c) => c.inputs.name));
	});

	it('_parse_timestamp_flexible', () => {
		replayFixture(FLEXIBLE, (i) => parseTimestampFlexible(i.val), '_parse_timestamp_flexible');
	});

	it('the DateParse record and the bare-float wrapper agree on every case, in all four tables', () => {
		// The two forms share one implementation; this is what keeps it that way. A record whose
		// `value` drifted from the number the wrapper returns would make every `rule` on the page a
		// claim about a different answer.
		for (const { file, cases, record, value } of TABLES) {
			for (const c of cases) {
				const r = record(c.inputs);
				expect(Object.is(r.value, value(c.inputs)), `${file}/${c.name}`).toBe(true);
				expect(Object.values(DATE_RULES), `${file}/${c.name} rule`).toContain(r.rule);
				expect(r.imputed, `${file}/${c.name} imputed`).toBe(
					r.imputations.month || r.imputations.day || r.imputations.dayClamped
				);
				// A failure never carries a number, and a success always does.
				const failed = [DATE_RULES.NONE, DATE_RULES.OUT_OF_RANGE, DATE_RULES.UNPARSED].includes(r.rule);
				expect(Number.isNaN(r.value), `${file}/${c.name} NaN iff failed`).toBe(failed);
			}
		}
	});
});

describe('the upstream quirks, replicated and named (Q1..Q10 of src/dates.js)', () => {
	it('Q1: the 1959 anchor is a substring test that fires before every pattern, and is OPT-IN here', () => {
		// dating.py:330-332. On: the reference. Off: the default, because it is data loss.
		expect(parseHeaderTimestamp('A/Kinshasa/1959-03-04', { archival1959: true })).toBe(1959.5);
		expect(parseHeaderTimestamp('A/Kinshasa/1959-03-04')).toBe(1959 + 62 / 365); // 4 March
		expect(parseHeaderTimestamp('AZ59012', { archival1959: true })).toBe(1959.5);
		expect(parseHeaderTimestamp('AZ59012')).toBeNaN();
		expect(parseHeaderDate('X|1959-03-04', { archival1959: true }).rule).toBe(DATE_RULES.ARCHIVAL_1959);
	});

	it('Q2: a missing month is June and a missing day the 15th — and "2021" is NOT mid-year', () => {
		const masked = parseDate('2021-XX-XX'); // temporal.py:132, 138
		expect(masked.value).toBe(2021 + 165 / 365); // 15 June
		expect(masked.imputations).toEqual({ month: true, day: true, dayClamped: false });
		expect(parseDate('2021-04').imputations).toEqual({ month: false, day: true, dayClamped: false });
		// The reference's OWN DOCSTRING (temporal.py:80) promises 2021.5 here; float() claims the
		// string first (temporal.py:116) and the answer is 1 January. Replicated, not corrected.
		expect(parseDate('2021').value).toBe(2021);
		expect(parseDate('2021').rule).toBe(DATE_RULES.DECIMAL_YEAR);
		expect(parseDate('2021').imputed).toBe(false);
	});

	it('Q3: the day is capped at 28/30 in every month, so 31 January and 29 February are unreachable', () => {
		const jan31 = parseDate('2021-01-31'); // temporal.py:144
		expect(jan31.imputations).toEqual({ month: false, day: false, dayClamped: true });
		expect(jan31.value).toBe(parseDateToDecimal('2021-01-30'));
		const leap = parseDate('2020-02-29');
		expect(leap.imputations.dayClamped).toBe(true);
		expect(leap.value).toBe(parseDateToDecimal('2020-02-28'));
		expect(parseDate('2021-12-30').imputations.dayClamped).toBe(false);
	});

	it('Q4: an out-of-range month or day is discarded and the default kept, with no error', () => {
		expect(parseDate('2021-13-15').imputations).toEqual({ month: true, day: false, dayClamped: false });
		expect(parseDate('2021-04-32').imputations).toEqual({ month: false, day: true, dayClamped: false });
		// Indistinguishable from a masked date by VALUE — which is exactly why `imputations` exists.
		expect(parseDate('2021-13-40').value).toBe(parseDateToDecimal('2021-XX-XX'));
		expect(parseDate('2021-13-40').imputations).toEqual({ month: true, day: true, dayClamped: false });
	});

	it("Q5: '.' is a date delimiter, but only after float() has claimed every plain decimal year", () => {
		expect(parseDateToDecimal('2021.25')).toBe(2021.25); // temporal.py:116, then :123
		expect(parseDate('2021.25').rule).toBe(DATE_RULES.DECIMAL_YEAR);
		expect(parseDateToDecimal('2021.4.15')).toBe(parseDateToDecimal('2021-04-15'));
		expect(parseDate('2021.4.15').rule).toBe(DATE_RULES.YMD);
		// A decimal year just outside the gate is re-read as a date rather than rejected.
		expect(parseDateToDecimal('2100.5')).toBe(parseDateToDecimal('2100-05-15'));
	});

	it('Q6: an out-of-gate year returns NaN like an unparseable string — separated here by rule only', () => {
		expect(parseDateToDecimal('1700-04-15')).toBeNaN(); // temporal.py:130
		expect(parseDateToDecimal('notadate')).toBeNaN(); // temporal.py:151
		expect(parseDate('1700-04-15').rule).toBe(DATE_RULES.OUT_OF_RANGE);
		expect(parseDate(1700).rule).toBe(DATE_RULES.OUT_OF_RANGE);
		expect(parseDate('notadate').rule).toBe(DATE_RULES.UNPARSED);
		expect(parseDate('15-04-2021').rule).toBe(DATE_RULES.UNPARSED);
		expect(parseDate(null).rule).toBe(DATE_RULES.NONE);
		expect(parseDate('unknown').rule).toBe(DATE_RULES.NONE);
	});

	it("Q7: the calendar delimiter class excludes '-', pattern 2 needs two decimals, pattern 4 is $-anchored", () => {
		expect(extractDateFromString('seq_2021-05-14')).toBe(2021 + 133 / 365); // 14 May
		expect(extractDateFromString('2021-05-14')).toBe(2021 + 133 / 365);
		expect(extractDateFromString('A-2021-05-14')).toBeNaN(); // temporal.py:218, no '-'
		expect(extractDateFromString('X|2021.35')).toBe(2021.35);
		expect(extractDateFromString('X|2021.5')).toBeNaN(); // temporal.py:225, \d{2,4}
		expect(extractDateFromString('strain|2021')).toBe(2021);
		expect(extractDateFromString('2021')).toBeNaN(); // temporal.py:239, no '^' alternative
		// The non-calendar class is WIDER: it does include '-'.
		expect(extractDateFromString('sample_day-7', { timeUnits: 'days' })).toBe(7);
	});

	it('Q8: the two-digit year pivots at 30, hard-coded, and both LANL rules add a flat half year', () => {
		expect(KORBER_YEAR_PIVOT).toBe(30); // dating.py:344, 351
		expect(parseHeaderTimestamp('B29US.X')).toBe(2029.5);
		expect(parseHeaderTimestamp('B30US.X')).toBe(1930.5);
		expect(parseHeaderDate('B86US.SFMHS18').rule).toBe(DATE_RULES.KORBER_ISOLATE);
		expect(parseHeaderTimestamp('Ref_B_FR_83_HXB2')).toBe(1983.5);
		expect(parseHeaderDate('Ref_B_FR_83_HXB2').rule).toBe(DATE_RULES.LANL_PIPE_YEAR);
		// The country code must be UPPERCASE, and mid-year is invented rather than read.
		expect(parseHeaderTimestamp('Ref_C_et_86_ETH2220')).toBeNaN();
		expect(parseHeaderDate('B86US.SFMHS18').imputations).toEqual({
			month: true,
			day: true,
			dayClamped: false
		});
	});

	it('Q9: WPI and DPI are case-sensitive pairs, and come back raw on the calendar axis', () => {
		expect(parseHeaderTimestamp('patient1_16WPI')).toBe(16); // dating.py:355
		expect(parseHeaderTimestamp('patient1_16wpi')).toBe(16);
		expect(parseHeaderTimestamp('patient1_16Wpi')).toBeNaN();
		expect(parseHeaderDate('patient1_16WPI').rule).toBe(DATE_RULES.WPI);
		expect(parseHeaderDate('patient1_120.5dpi').rule).toBe(DATE_RULES.DPI);
		// 16 is the number the reader wrote, so nothing is imputed — and nothing marks it as weeks.
		expect(parseHeaderDate('patient1_16WPI').imputed).toBe(false);
		expect(parseHeaderDate('patient1_16WPI').timeUnits).toBe('years');
	});

	it('Q10: the non-calendar path accepts Infinity and rejects a negative by the same comparison', () => {
		expect(parseDateToDecimal('inf', { timeUnits: 'days' })).toBe(Infinity); // temporal.py:92-93
		expect(parseDateToDecimal(-1, { timeUnits: 'days' })).toBeNaN();
		expect(parseDateToDecimal('-inf', { timeUnits: 'days' })).toBeNaN();
		expect(parseDateToDecimal('nan', { timeUnits: 'days' })).toBeNaN();
		expect(parseDate(-1, { timeUnits: 'days' }).rule).toBe(DATE_RULES.OUT_OF_RANGE);
		// A negative RETURNS: it never falls through to the embedded-number search.
		expect(parseDateToDecimal('-5000', { timeUnits: 'generations' })).toBeNaN();
	});
});

describe('the provenance record', () => {
	it('names the rule that produced a calendar value', () => {
		expect(parseDate(2021.25).rule).toBe(DATE_RULES.NUMERIC);
		expect(parseDate('2021.25').rule).toBe(DATE_RULES.DECIMAL_YEAR);
		expect(parseDate('2021-04-15').rule).toBe(DATE_RULES.YMD);
	});

	it('reports a full date as imputing nothing', () => {
		const full = parseDate('2021-04-15');
		expect(full.imputed).toBe(false);
		expect(full.imputations).toEqual({ month: false, day: false, dayClamped: false });
	});

	it('keeps the substring a search rule consumed', () => {
		expect(extractDate('seq_2021-05-14').matched).toBe('2021-05-14');
		expect(extractDate('pop1|gen_5000', { timeUnits: 'generations' }).matched).toBe('5000');
		expect(parseDate('2021-04-15').matched).toBeNull();
	});

	it('carries the interpretation it used', () => {
		expect(parseDate('2021-04-15').timeUnits).toBe('years');
		expect(parseDate('5000', { timeUnits: 'days' }).timeUnits).toBe('days');
		expect(extractDate('x|5000', { timeUnits: 'arbitrary' }).timeUnits).toBe('arbitrary');
	});

	it('names the non-calendar pattern that matched', () => {
		expect(extractDate('pop1|gen_5000', { timeUnits: 'generations' }).rule).toBe(
			DATE_RULES.HEADER_UNIT_TOKEN
		);
		expect(extractDate('pop1_20000gen', { timeUnits: 'generations' }).rule).toBe(
			DATE_RULES.HEADER_UNIT_SUFFIX
		);
		expect(extractDate('lineage|5000', { timeUnits: 'generations' }).rule).toBe(
			DATE_RULES.HEADER_BARE_NUMBER
		);
	});

	it('falls through to the next pattern when a capture does not convert', () => {
		// "9999-05-14" matches pattern 1 and then fails the year gate; the reference keeps going.
		expect(extractDateFromString('X|9999-05-14')).toBeNaN();
		expect(extractDate('X|9999-05-14').rule).toBe(DATE_RULES.UNPARSED);
	});
});

describe('calendar versus non-calendar selection', () => {
	it('gates a calendar value into [1800, 2100] and does not gate a non-calendar one', () => {
		expect(parseDateToDecimal(5000)).toBeNaN();
		expect(parseDateToDecimal(5000, { timeUnits: 'generations' })).toBe(5000);
		expect(parseDateToDecimal(CALENDAR_YEAR_MIN)).toBe(1800);
		expect(parseDateToDecimal(CALENDAR_YEAR_MAX)).toBe(2100);
		expect(parseDateToDecimal(CALENDAR_YEAR_MIN - 0.001)).toBeNaN();
		expect(parseDateToDecimal(CALENDAR_YEAR_MAX + 0.001)).toBeNaN();
	});

	it('reads an ISO date only on the calendar path', () => {
		expect(parseDateToDecimal('2021-04-15')).toBe(2021 + 104 / 365); // 15 April
		// Non-calendar takes the first embedded number instead: the year, not the date.
		expect(parseDateToDecimal('2021-04-15', { timeUnits: 'days' })).toBe(2021);
	});

	it('exposes the membership test the reference uses', () => {
		expect(TIME_UNITS).toEqual(['years', 'generations', 'days', 'arbitrary']);
		expect(isNonCalendarTimeUnits('years')).toBe(false);
		expect(isNonCalendarTimeUnits('fortnights')).toBe(false);
		for (const u of NON_CALENDAR_TIME_UNITS) expect(isNonCalendarTimeUnits(u)).toBe(true);
	});
});

describe('the 1959 archival anchor (Q1), offered rather than applied', () => {
	it('is off by default', () => {
		expect(parseHeaderTimestamp('ZR59')).toBeNaN();
		expect(parseHeaderDate('ZR59').rule).toBe(DATE_RULES.UNPARSED);
	});

	it('fires on request, before every other rule, and says what it invented', () => {
		const r = parseHeaderDate('A/Kinshasa/1959-03-04', { archival1959: true });
		expect(r.value).toBe(ARCHIVAL_1959_DECIMAL);
		expect(r.rule).toBe(DATE_RULES.ARCHIVAL_1959);
		expect(r.imputed).toBe(true);
		expect(r.matched).toBe('1959');
	});

	it('exposes its predicate so an application can offer the rule instead of applying it', () => {
		expect(matchesArchival1959('Z59ZR.ZHU')).toBe(true);
		expect(matchesArchival1959('ZR59')).toBe(true);
		expect(matchesArchival1959('AZ59012')).toBe(true);
		expect(matchesArchival1959('X|1959-03-04')).toBe(true);
		expect(matchesArchival1959('X|2019-03-04')).toBe(false);
		expect(matchesArchival1959('')).toBe(false);
		expect(matchesArchival1959(null)).toBe(false);
	});

	it('does not cost the real ZR59 isolate its date: the Korber rule reads it anyway', () => {
		// Z59ZR.ZHU is `[Subtype][2-digit year][Country].[Strain]`, so step 3 gets 1959.5 with no hack.
		// This is what makes shipping the anchor off safe on the flagship HIV-1 example.
		expect(parseHeaderTimestamp('Z59ZR.ZHU')).toBe(1959.5);
		expect(parseHeaderDate('Z59ZR.ZHU').rule).toBe(DATE_RULES.KORBER_ISOLATE);
	});
});

describe('_parse_timestamp_flexible', () => {
	it('prefers the calendar reading, then accepts any float', () => {
		expect(parseFlexibleDate('2021-04-15').rule).toBe(DATE_RULES.YMD);
		expect(parseFlexibleDate('5000').rule).toBe(DATE_RULES.FLEXIBLE_NUMERIC);
		expect(parseFlexibleDate(0.25).rule).toBe(DATE_RULES.FLEXIBLE_NUMERIC);
		expect(parseTimestampFlexible('-3')).toBe(-3);
	});

	it('still answers NaN for the null tokens, through the second conversion as well', () => {
		expect(parseTimestampFlexible('nan')).toBeNaN();
		expect(parseFlexibleDate('nan').rule).toBe(DATE_RULES.NONE);
		expect(parseTimestampFlexible('unknown')).toBeNaN();
		expect(parseTimestampFlexible(null)).toBeNaN();
	});
});

describe('the constants are the reference’s', () => {
	it('carries the gate, the imputed components and the null tokens', () => {
		expect([CALENDAR_YEAR_MIN, CALENDAR_YEAR_MAX]).toEqual([1800, 2100]);
		expect([IMPUTED_MONTH, IMPUTED_DAY]).toEqual([6, 15]);
		expect(NULL_DATE_TOKENS).toEqual(['unknown', 'nan', 'none', 'na', '?']);
		expect(ARCHIVAL_1959_DECIMAL).toBe(1959.5);
	});

	it('freezes what it publishes, so a consumer cannot edit the reference out from under another', () => {
		expect(Object.isFrozen(DATE_RULES)).toBe(true);
		expect(Object.isFrozen(TIME_UNITS)).toBe(true);
		expect(Object.isFrozen(NULL_DATE_TOKENS)).toBe(true);
		expect(Object.isFrozen(NON_CALENDAR_TIME_UNITS)).toBe(true);
	});

	it('reports a rule id for every DATE_RULES member, and no duplicates', () => {
		const ids = Object.values(DATE_RULES);
		expect(new Set(ids).size).toBe(ids.length);
	});
});
