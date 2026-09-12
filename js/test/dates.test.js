/**
 * WHY THIS FILE EXISTS
 *
 * src/dates.js is the only part of the time-aware pillars that is pure arithmetic over strings, and
 * it is the part every surface will reach for first. Two different things have to be pinned, and
 * they fail in different ways:
 *
 *   1. THE NUMBERS ARE THE REFERENCE'S. `test/data/dates/parsers.json` is produced by the four
 *      reference functions themselves — lifted out of hyphaeon/temporal.py and hyphaeon/dating.py by
 *      `ast` rather than retyped (see the generator's header) — so every case below is a replay, not
 *      an opinion about what the Python ought to do. Each table is asserted input by input, because
 *      a single wrong delimiter class turns a whole surveillance dataset silently undated.
 *
 *   2. THE PROVENANCE IS THE PORT'S. `rule`, `imputed` and `imputations` have no Python counterpart
 *      — the reference returns a bare float — so those expectations are written here, against the
 *      quirks named Q1..Q10 in the module header. They are the contract the application's review
 *      page is built on: which rule fired, and whether the answer was read or invented.
 *
 * The 1959 anchor (Q1) is checked in BOTH directions: `{archival1959: true}` against the unmodified
 * reference table, and the default against the table the generator produces by running the
 * reference's remaining steps in the reference's own order. A port that quietly kept the anchor on
 * and a port that quietly dropped it entirely both fail here.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import {
	ARCHIVAL_1959_DECIMAL,
	CALENDAR_YEAR_MAX,
	CALENDAR_YEAR_MIN,
	DATE_RULES,
	IMPUTED_DAY,
	IMPUTED_MONTH,
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

const REFERENCE = JSON.parse(
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
 * Replay one table of the reference through one function.
 *
 * @param {string} table key in parsers.json
 * @param {(input: any) => number} fn
 */
function replay(table, fn) {
	const cases = REFERENCE[table];
	expect(cases.length, `${table} is non-empty`).toBeGreaterThan(0);
	for (const { input, output } of cases) {
		expectSame(fn(input), decode(output), `${table}: ${JSON.stringify(input)}`);
	}
}

describe('the reference tables, replayed', () => {
	it('records which reference functions and line ranges it was generated from', () => {
		// The generator writes the line range it extracted, so a move upstream shows up as a diff in
		// this file's data rather than as an unexplained failure.
		expect(REFERENCE.source).toEqual({
			parse_date_to_decimal: 'hyphaeon/temporal.py:73-151',
			extract_date_from_string: 'hyphaeon/temporal.py:196-245',
			parse_header_timestamp: 'hyphaeon/dating.py:317-364',
			_parse_timestamp_flexible: 'hyphaeon/dating.py:367-384'
		});
	});

	it('parse_date_to_decimal, calendar (72 cases)', () => {
		replay('parse_date_to_decimal_years', (v) => parseDateToDecimal(v));
	});

	it('parse_date_to_decimal, generations / days / arbitrary', () => {
		for (const timeUnits of NON_CALENDAR_TIME_UNITS) {
			replay(`parse_date_to_decimal_${timeUnits}`, (v) => parseDateToDecimal(v, { timeUnits }));
		}
	});

	it('an unrecognised time_units takes the calendar path, as `in (...)` makes it', () => {
		replay('parse_date_to_decimal_unknown_units', (v) =>
			parseDateToDecimal(v, { timeUnits: 'fortnights' })
		);
	});

	it('extract_date_from_string, calendar', () => {
		replay('extract_date_from_string_years', (v) => extractDateFromString(v));
	});

	it('extract_date_from_string, generations', () => {
		replay('extract_date_from_string_generations', (v) =>
			extractDateFromString(v, { timeUnits: 'generations' })
		);
	});

	it('parse_header_timestamp with the 1959 anchor ON is the reference verbatim (Q1)', () => {
		replay('parse_header_timestamp', (v) => parseHeaderTimestamp(v, { archival1959: true }));
	});

	it('parse_header_timestamp with the anchor OFF is the default, and differs on four names (Q1)', () => {
		replay('parse_header_timestamp_without_1959', (v) => parseHeaderTimestamp(v));

		// The whole point of making it opt-in: these are the headers the reference mis-dates.
		const withAnchor = REFERENCE.parse_header_timestamp;
		const without = REFERENCE.parse_header_timestamp_without_1959;
		const changed = withAnchor
			.map((c, i) => [c.input, c.output, without[i].output])
			.filter(([, a, b]) => a !== b)
			.map(([name]) => name);
		expect(changed).toEqual(['ZR59', 'AZ59012', 'A/Kinshasa/1959-03-04', 'X|1959-03-04']);
	});

	it('_parse_timestamp_flexible', () => {
		replay('parse_timestamp_flexible', (v) => parseTimestampFlexible(v));
	});
});

describe('the provenance record', () => {
	it('names the rule that produced a calendar value', () => {
		expect(parseDate(2021.25).rule).toBe(DATE_RULES.NUMERIC);
		expect(parseDate('2021.25').rule).toBe(DATE_RULES.DECIMAL_YEAR);
		expect(parseDate('2021-04-15').rule).toBe(DATE_RULES.YMD);
		// NOT the ymd path: `float('2021')` succeeds and the gate admits it, so a bare year string is
		// a decimal year and stays 2021.0 — Q2, where the reference's own docstring says 2021.5.
		expect(parseDate('2021').rule).toBe(DATE_RULES.DECIMAL_YEAR);
		expect(parseDate('2021').value).toBe(2021);
		expect(parseDate('2021').imputed).toBe(false);
	});

	it('names the rule that produced nothing, and separates the three ways of failing', () => {
		// The reference collapses all of these to NaN; a reader needs to know which happened.
		expect(parseDate(null).rule).toBe(DATE_RULES.NONE);
		expect(parseDate('').rule).toBe(DATE_RULES.NONE);
		expect(parseDate('unknown').rule).toBe(DATE_RULES.NONE);
		expect(parseDate('?').rule).toBe(DATE_RULES.NONE);
		expect(parseDate('1700-04-15').rule).toBe(DATE_RULES.OUT_OF_RANGE); // Q6
		expect(parseDate(1700).rule).toBe(DATE_RULES.OUT_OF_RANGE);
		expect(parseDate('notadate').rule).toBe(DATE_RULES.UNPARSED);
		expect(parseDate('15-04-2021').rule).toBe(DATE_RULES.UNPARSED);
		for (const r of ['none', 'out_of_range', 'unparsed']) {
			expect([DATE_RULES.NONE, DATE_RULES.OUT_OF_RANGE, DATE_RULES.UNPARSED]).toContain(r);
		}
	});

	it('reports a year with unreadable month and day as month- and day-imputed (Q2)', () => {
		// The delimiter path is the only one that imputes: "2021" itself never reaches it.
		const masked = parseDate('2021-XX-XX');
		expect(masked.value).toBe(2021 + 165 / 365); // 15 June
		expect(masked.imputed).toBe(true);
		expect(masked.imputations).toEqual({ month: true, day: true, dayClamped: false });
	});

	it('reports a year-month as day-imputed only (Q2)', () => {
		const ym = parseDate('2021-04');
		expect(ym.imputed).toBe(true);
		expect(ym.imputations).toEqual({ month: false, day: true, dayClamped: false });
	});

	it('reports a full date as imputing nothing', () => {
		const full = parseDate('2021-04-15');
		expect(full.imputed).toBe(false);
		expect(full.imputations).toEqual({ month: false, day: false, dayClamped: false });
	});

	it('reports the 28/30 day cap as a clamp, not as an imputation (Q3)', () => {
		const jan31 = parseDate('2021-01-31');
		expect(jan31.imputations).toEqual({ month: false, day: false, dayClamped: true });
		expect(jan31.value).toBe(parseDateToDecimal('2021-01-30'));

		// 29 February of a leap year is unreachable: the cap is 28 whatever the year.
		const leap = parseDate('2020-02-29');
		expect(leap.imputations.dayClamped).toBe(true);
		expect(leap.value).toBe(parseDateToDecimal('2020-02-28'));

		// 30 in a 31-day month is not clamped.
		expect(parseDate('2021-12-30').imputations.dayClamped).toBe(false);
	});

	it('reports an invalid month or day as imputed, because the reference discards it (Q4)', () => {
		expect(parseDate('2021-13-15').imputations).toEqual({
			month: true,
			day: false,
			dayClamped: false
		});
		expect(parseDate('2021-04-32').imputations).toEqual({
			month: false,
			day: true,
			dayClamped: false
		});
		// Indistinguishable from a bare year by value alone — which is exactly why `imputations` exists.
		expect(parseDate('2021-13-40').value).toBe(parseDateToDecimal('2021-XX-XX'));
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

	it('accepts a negative-free non-negative real, and Infinity with it (Q10)', () => {
		expect(parseDateToDecimal(-1, { timeUnits: 'days' })).toBeNaN();
		expect(parseDateToDecimal('inf', { timeUnits: 'days' })).toBe(Infinity);
		expect(parseDateToDecimal('-inf', { timeUnits: 'days' })).toBeNaN();
		expect(parseDate(-1, { timeUnits: 'days' }).rule).toBe(DATE_RULES.OUT_OF_RANGE);
	});

	it('exposes the membership test the reference uses', () => {
		expect(TIME_UNITS).toEqual(['years', 'generations', 'days', 'arbitrary']);
		expect(isNonCalendarTimeUnits('years')).toBe(false);
		expect(isNonCalendarTimeUnits('fortnights')).toBe(false);
		for (const u of NON_CALENDAR_TIME_UNITS) expect(isNonCalendarTimeUnits(u)).toBe(true);
	});
});

describe('the header patterns and the delimiters they require (Q7)', () => {
	it('needs a pipe, slash, underscore, whitespace or the string start before an ISO date', () => {
		expect(extractDateFromString('seq_2021-05-14')).toBe(2021 + 133 / 365); // 14 May
		expect(extractDateFromString('2021-05-14')).toBe(2021 + 133 / 365);
		// A hyphen is NOT in the class, so this reads as nothing at all.
		expect(extractDateFromString('A-2021-05-14')).toBeNaN();
	});

	it('needs two to four decimals for a decimal year', () => {
		expect(extractDateFromString('X|2021.35')).toBe(2021.35);
		expect(extractDateFromString('X|2021.3512')).toBe(2021.3512);
		expect(extractDateFromString('X|2021.5')).toBeNaN();
	});

	it('will not read a header that IS a bare year, because pattern 4 has no `^`', () => {
		// A trailing year needs a delimiter in front of it, and then converts as a decimal year (Q2).
		expect(extractDateFromString('strain|2021')).toBe(2021);
		expect(extractDateFromString('strain-2021')).toBeNaN();
		expect(extractDateFromString('2021')).toBeNaN();
	});

	it('falls through to the next pattern when a capture does not convert', () => {
		// "9999-05-14" matches pattern 1 and then fails the year gate; the reference keeps going.
		expect(extractDateFromString('X|9999-05-14')).toBeNaN();
		expect(extractDate('X|9999-05-14').rule).toBe(DATE_RULES.UNPARSED);
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
});

describe('the 1959 archival anchor (Q1)', () => {
	it('is off by default', () => {
		expect(parseHeaderTimestamp('ZR59')).toBeNaN();
		expect(parseHeaderDate('ZR59').rule).toBe(DATE_RULES.UNPARSED);
	});

	it('fires on request, before every other rule, and says so', () => {
		const r = parseHeaderDate('A/Kinshasa/1959-03-04', { archival1959: true });
		expect(r.value).toBe(ARCHIVAL_1959_DECIMAL);
		expect(r.rule).toBe(DATE_RULES.ARCHIVAL_1959);
		expect(r.imputed).toBe(true);
		expect(r.matched).toBe('1959');
	});

	it('overwrites a real date and mis-dates an unrelated accession, which is why it is opt-in', () => {
		expect(parseHeaderTimestamp('A/Kinshasa/1959-03-04', { archival1959: true })).toBe(1959.5);
		expect(parseHeaderTimestamp('A/Kinshasa/1959-03-04')).toBe(1959 + 62 / 365); // 4 March
		expect(parseHeaderTimestamp('AZ59012', { archival1959: true })).toBe(1959.5);
		expect(parseHeaderTimestamp('AZ59012')).toBeNaN();
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
		expect(parseHeaderTimestamp('Z59ZR.ZHU')).toBe(1959.5);
		expect(parseHeaderDate('Z59ZR.ZHU').rule).toBe(DATE_RULES.KORBER_ISOLATE);
	});
});

describe('the header rules of dating.py', () => {
	it('pivots a two-digit year at 30 (Q8)', () => {
		expect(parseHeaderTimestamp('B29US.X')).toBe(2029.5);
		expect(parseHeaderTimestamp('B30US.X')).toBe(1930.5);
		expect(parseHeaderDate('B86US.SFMHS18').rule).toBe(DATE_RULES.KORBER_ISOLATE);
		expect(parseHeaderDate('Ref_B_FR_83_HXB2').rule).toBe(DATE_RULES.LANL_PIPE_YEAR);
	});

	it('needs an UPPERCASE country code in the LANL pattern', () => {
		expect(parseHeaderTimestamp('Ref_C_ET_86_ETH2220')).toBe(1986.5);
		expect(parseHeaderTimestamp('Ref_C_et_86_ETH2220')).toBeNaN();
	});

	it('matches WPI and wpi but not Wpi (Q9)', () => {
		expect(parseHeaderTimestamp('patient1_16WPI')).toBe(16);
		expect(parseHeaderTimestamp('patient1_16wpi')).toBe(16);
		expect(parseHeaderTimestamp('patient1_16Wpi')).toBeNaN();
		expect(parseHeaderDate('patient1_16WPI').rule).toBe(DATE_RULES.WPI);
		expect(parseHeaderDate('patient1_120.5dpi').rule).toBe(DATE_RULES.DPI);
	});

	it('marks the two-digit-year rules as imputed, because mid-year is invented', () => {
		expect(parseHeaderDate('B86US.SFMHS18').imputations).toEqual({
			month: true,
			day: true,
			dayClamped: false
		});
		// An elapsed time is not: 16 weeks post infection is the number the reader wrote.
		expect(parseHeaderDate('patient1_16WPI').imputed).toBe(false);
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
