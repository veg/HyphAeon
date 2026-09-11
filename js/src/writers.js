/**
 * WHY THIS FILE EXISTS
 *
 * The output writers of hyphaeon/cli.py and hyphaeon/io.py (at reconcile/phase-5a), as pure string builders.
 * PLAN.md §5.1 lists them as the `writers` module with parity class "byte-equal after
 * canonicalisation"; this file aims one notch higher — byte-equal before canonicalisation — because
 * the Python writers are thin wrappers over three serialisers whose formatting is deterministic and
 * cheap to reproduce: `json.dump(data, f, indent=2)` (io.py:20-29), `pandas.DataFrame(records)
 * .to_csv(path, index=False)` (io.py:32-41) and `networkx.write_graphml` in its lxml form
 * (cli.py:823-835). The user-visible files are therefore identical to the reference CLI's, which is
 * what makes `scripts/parity.py` a comparison rather than an argument.
 *
 * WHAT IT MIRRORS
 *
 *   memeSiteRecords      cli.py:283-299   the `results_list` site dicts, attribution fields folded in
 *   memeResult           cli.py:301-314   the `hyphaeon meme` JSON document
 *   memeJson             cli.py:303       write_json of that document
 *   memeCsv              cli.py:316-330   flattened site rows (5 base + 4 attribution columns)
 *   bustedJson           cli.py:552-553   one record, or the list in batch mode
 *   bustedCsv            cli.py:555-571   the `df_summary` columns Gene .. Time_ms
 *   phenotypeCsv         cli.py:702-703   DataFrame(sites)
 *   epistasisCsv         cli.py:810-821   plasticity / edges / plasticity / sectors selection
 *   graphml              cli.py:823-835   nx.Graph of str(site) nodes, edge attrs weight/cesi/shared/fdr_q
 *   dmsCsv               cli.py:900-904   DataFrame(plasticity) minus `mutant_deltas`
 *   resultJson           io.py:20-29      json.dump(indent=2) for any of the result dicts
 *   evaluateJson         evaluation.py:641,644  json.dumps(result, indent=2, allow_nan=False)
 *
 *   plus the Python formatting primitives the above are made of, exported because evaluate.js and
 *   the tests need the same semantics: pyFloatRepr (float.__repr__), pyRepr (repr()), pyStr
 *   (str()), pyFormatFixed (format(x, '.Nf')), pyFormatG (format(x, 'g')), pyJsonDumps
 *   (json.dumps(indent=...)), dataFrameCsv (DataFrame(records).to_csv(index=False)).
 *
 * WHAT IT DELIBERATELY DOES NOT DO. No file I/O, no confirmation prints, no directory creation
 * (io.py:14-17, :29, :41 are the app runtime's). No statistics: the busted record fields
 * (p_ACAT, Simes, omnibus LRT) come from the omnibus port, the attribution records from the
 * attribution port; this file only lays them out. It does not reproduce pandas' `to_csv` for
 * DataFrames the CLI never builds (multi-index, datetime, categorical columns).
 *
 * THE ONE THING JAVASCRIPT CANNOT SEE: whether a number is a Python int or a Python float. The
 * reference writes `"hyphaeon_lrt": 0.0` but `"site": 1`, and pandas prints a float64 column as
 * `0.0` and an int64 column as `0`. A JS number carries no such tag, so the serialisers take a
 * key-name schema: PY_FLOAT_KEYS / PY_INT_KEYS below list every result field the Python builds
 * with `float(...)` / true division (float) or with `int(...)`, `len`, `s + 1`, a count (int),
 * read off cli.py, attribution.py:117-170, epistasis.py:196-211,419-438,569-577,760-767,
 * phenotype.py:526-543,589-605,622-644, filter.py:80-98 and evaluation.py:286-449. Numbers under
 * a key in neither set are formatted by value (integer-valued -> int). A list or dict inherits the
 * kind of the key that holds it (`sites: [244, 279]` int, `mutant_deltas: {A: 0.3}` float).
 * Parsers cannot tell `0` from `0.0`, so this only matters for byte comparison, never for values.
 *
 * FORMATTING FACTS PINNED BY THE TESTS (measured against Python 3.14 / pandas 3.0.5 / lxml 6.1 in
 * the reference venv; js/test/writers.test.js carries the strings):
 *   - float repr: shortest round-trip digits, exponent form when the decimal exponent is < -4 or
 *     >= 16 (`1e-05`, `1e+16`, `1000000000000000.0`), at least two exponent digits, `.0` appended
 *     to an integral fixed form, `-0.0`, `inf`, `nan`. JS `toExponential()` yields the same
 *     shortest digits (both pick the candidate closest to the binary value); only the layout differs.
 *   - json.dumps(indent=2): `,` + newline between items, `: ` after keys, `[]`/`{}` for empties,
 *     ensure_ascii (non-ASCII and control characters as \uXXXX, astral as surrogate pairs), NaN /
 *     Infinity literals unless allow_nan=False, no trailing newline.
 *   - to_csv(index=False): header + rows joined by `\n` with a trailing `\n`; QUOTE_MINIMAL (a
 *     field is quoted when it contains `,`, `"`, `\n` or `\r`; `"` doubled; a lone empty field is
 *     written as `""`); columns are the union of record keys in first-seen order; a column with
 *     any missing value or None is float64 (`3.0`, missing -> empty) for numbers and object for
 *     booleans (`True`/`False`/empty); bool columns print `True`/`False`; list and dict cells are
 *     their Python repr (`[244, 279]`, `{'A': 0.5}`), which is why sector and plasticity CSVs carry
 *     quoted Python literals; NaN prints empty, inf prints `inf`.
 *   - write_graphml (lxml path, networkx 3.6): `<?xml version='1.0' encoding='utf-8'?>` + newline,
 *     the graphml root opened WITHOUT a newline, keys in REVERSE order of first appearance (each
 *     `get_key` does `insert(0, ...)`: d3 fdr_q, d2 shared, d1 cesi, d0 weight), one `<key/>` per
 *     line, `<graph edgedefault="undirected">` opened without a newline, `<node id=".."/>` per line
 *     in insertion order, each `<edge>` pretty-printed with two-space-indented `<data>` children,
 *     `</graph></graphml>` with no trailing newline. Node insertion order is the order sites first
 *     appear in the edge list; edges are enumerated the way `Graph.edges` does (per node in
 *     insertion order, each neighbour in adjacency order, skipping nodes already visited), so the
 *     printed (source, target) can be the reverse of the input pair. Re-adding an edge updates its
 *     attributes in place. Attribute types are fixed by the cli.py casts: double, double, long, double.
 */

// ---------------------------------------------------------------------------------------------
// Python formatting primitives
// ---------------------------------------------------------------------------------------------

/**
 * Python `float.__repr__` (CPython `float_repr_style == 'short'`, format code 'r').
 *
 * @param {number} x
 * @returns {string}
 */
export function pyFloatRepr(x) {
	if (Number.isNaN(x)) return 'nan';
	if (x === Infinity) return 'inf';
	if (x === -Infinity) return '-inf';
	if (x === 0) return Object.is(x, -0) ? '-0.0' : '0.0';
	const neg = x < 0;
	const m = /^(\d)(?:\.(\d+))?e([+-]\d+)$/.exec(Math.abs(x).toExponential());
	if (!m) throw new RangeError(`pyFloatRepr: unexpected toExponential form for ${x}`);
	const digits = m[1] + (m[2] || '');
	const exp10 = parseInt(m[3], 10);
	const decpt = exp10 + 1; // x = 0.digits * 10^decpt
	let out;
	if (decpt <= -4 || decpt > 16) {
		out =
			digits[0] +
			(digits.length > 1 ? '.' + digits.slice(1) : '') +
			'e' +
			(exp10 < 0 ? '-' : '+') +
			String(Math.abs(exp10)).padStart(2, '0');
	} else if (decpt <= 0) {
		out = '0.' + '0'.repeat(-decpt) + digits;
	} else if (decpt >= digits.length) {
		out = digits + '0'.repeat(decpt - digits.length) + '.0';
	} else {
		out = digits.slice(0, decpt) + '.' + digits.slice(decpt);
	}
	return neg ? '-' + out : out;
}

/**
 * Python `str(int)` for an integer-valued JS number (digits beyond 2^53 come out exact, as an
 * int64 column or a Python int would print them, rather than as `1e+21`).
 * @param {number} x integer-valued
 * @returns {string}
 */
function pyIntStr(x) {
	if (Number.isSafeInteger(x)) return String(x);
	return BigInt(x).toString();
}

/**
 * Python `format(x, '.{digits}f')`: correctly rounded (round-half-even on the exact binary value,
 * so 0.0078125 -> '0.007812' where JS toFixed gives '0.007813'), sign kept on negative zero results.
 *
 * @param {number} x
 * @param {number} digits
 * @returns {string}
 */
export function pyFormatFixed(x, digits) {
	if (Number.isNaN(x)) return 'nan';
	if (x === Infinity) return 'inf';
	if (x === -Infinity) return '-inf';
	const neg = x < 0 || Object.is(x, -0);
	const view = new DataView(new ArrayBuffer(8));
	view.setFloat64(0, Math.abs(x));
	const bits = view.getBigUint64(0);
	const expBits = Number((bits >> 52n) & 0x7ffn);
	let mant = bits & ((1n << 52n) - 1n);
	let exp2;
	if (expBits === 0) {
		exp2 = -1074;
	} else {
		mant |= 1n << 52n;
		exp2 = expBits - 1075;
	}
	const num = mant * 10n ** BigInt(digits);
	let q;
	if (exp2 >= 0) {
		q = num << BigInt(exp2);
	} else {
		const den = 1n << BigInt(-exp2);
		q = num / den;
		const twice = (num % den) * 2n;
		if (twice > den || (twice === den && (q & 1n) === 1n)) q += 1n;
	}
	let s = q.toString();
	if (digits > 0) {
		s = s.padStart(digits + 1, '0');
		s = s.slice(0, -digits) + '.' + s.slice(-digits);
	}
	return (neg ? '-' : '') + s;
}

/**
 * Python `format(x, 'g')` at the default precision 6 (used for `f"{alpha:g}"` in evaluation.py:
 * 318-321): six significant digits, trailing zeros removed, exponent form below 1e-4 or at 1e6.
 * Rounding at an exact decimal tie follows JS toExponential (half-up) rather than CPython's
 * half-even; the only callers pass 0.05 and 0.10.
 *
 * @param {number} x
 * @param {number} [precision=6]
 * @returns {string}
 */
export function pyFormatG(x, precision = 6) {
	if (Number.isNaN(x)) return 'nan';
	if (x === Infinity) return 'inf';
	if (x === -Infinity) return '-inf';
	if (x === 0) return Object.is(x, -0) ? '-0' : '0';
	const p = Math.max(1, precision);
	const neg = x < 0;
	const m = /^(\d)(?:\.(\d+))?e([+-]\d+)$/.exec(Math.abs(x).toExponential(p - 1));
	if (!m) throw new RangeError(`pyFormatG: unexpected toExponential form for ${x}`);
	let digits = (m[1] + (m[2] || '')).replace(/0+$/, '');
	if (digits === '') digits = '0';
	const exp10 = parseInt(m[3], 10);
	let out;
	if (exp10 < -4 || exp10 >= p) {
		out =
			digits[0] +
			(digits.length > 1 ? '.' + digits.slice(1) : '') +
			'e' +
			(exp10 < 0 ? '-' : '+') +
			String(Math.abs(exp10)).padStart(2, '0');
	} else if (exp10 < 0) {
		out = '0.' + '0'.repeat(-exp10 - 1) + digits;
	} else if (exp10 + 1 >= digits.length) {
		out = digits + '0'.repeat(exp10 + 1 - digits.length);
	} else {
		out = digits.slice(0, exp10 + 1) + '.' + digits.slice(exp10 + 1);
	}
	return neg ? '-' + out : out;
}

/** Characters Python's `str.isprintable` rejects (Cc, Cf, Cs, Co, Cn, Zl, Zp, Zs except ASCII space). */
const NOT_PRINTABLE = /[\p{Cc}\p{Cf}\p{Cs}\p{Co}\p{Cn}\p{Zl}\p{Zp}\p{Zs}]/u;

/**
 * Python `repr(str)`: single quotes unless the string holds a single quote and no double quote;
 * backslash, the quote, \n \r \t escaped; other non-printable code points as \xNN, \uNNNN or
 * \UNNNNNNNN; printable non-ASCII kept.
 * @param {string} s
 * @returns {string}
 */
function pyStrRepr(s) {
	const quote = s.includes("'") && !s.includes('"') ? '"' : "'";
	let out = quote;
	for (const ch of s) {
		const cp = /** @type {number} */ (ch.codePointAt(0));
		if (ch === '\\') out += '\\\\';
		else if (ch === quote) out += '\\' + quote;
		else if (ch === '\n') out += '\\n';
		else if (ch === '\r') out += '\\r';
		else if (ch === '\t') out += '\\t';
		else if (ch !== ' ' && NOT_PRINTABLE.test(ch)) {
			if (cp < 0x100) out += '\\x' + cp.toString(16).padStart(2, '0');
			else if (cp < 0x10000) out += '\\u' + cp.toString(16).padStart(4, '0');
			else out += '\\U' + cp.toString(16).padStart(8, '0');
		} else out += ch;
	}
	return out + quote;
}

/**
 * @typedef {'int'|'float'|undefined} NumberKind
 */

/**
 * Format a JS number the way Python prints an int or a float, given the kind the schema assigns.
 * @param {number} x
 * @param {NumberKind} kind
 * @returns {string}
 */
function pyNumberStr(x, kind) {
	if (kind === 'int' && Number.isInteger(x)) return pyIntStr(x);
	if (kind === 'float') return pyFloatRepr(x);
	return Number.isInteger(x) ? pyIntStr(x) : pyFloatRepr(x);
}

/**
 * Python `repr()` of the JSON-ish values the CLI puts into result dicts: str, int/float (by
 * `kind`), bool, None, list (arrays and typed arrays), dict (plain objects and Maps).
 *
 * @param {unknown} value
 * @param {NumberKind} [kind]
 * @returns {string}
 */
export function pyRepr(value, kind) {
	if (value === null || value === undefined) return 'None';
	if (typeof value === 'string') return pyStrRepr(value);
	if (typeof value === 'number') return pyNumberStr(value, kind);
	if (typeof value === 'bigint') return value.toString();
	if (typeof value === 'boolean') return value ? 'True' : 'False';
	if (Array.isArray(value) || ArrayBuffer.isView(value)) {
		const items = [];
		for (const v of /** @type {ArrayLike<unknown>} */ (/** @type {unknown} */ (value)))
			items.push(pyRepr(v, kind));
		return '[' + items.join(', ') + ']';
	}
	if (value instanceof Map) {
		const items = [];
		for (const [k, v] of value) items.push(pyRepr(k) + ': ' + pyRepr(v, kind));
		return '{' + items.join(', ') + '}';
	}
	if (typeof value === 'object') {
		const items = [];
		for (const [k, v] of Object.entries(value)) items.push(pyStrRepr(k) + ': ' + pyRepr(v, kind));
		return '{' + items.join(', ') + '}';
	}
	return String(value);
}

/**
 * Python `str()`: the string itself, otherwise `repr()`.
 * @param {unknown} value
 * @param {NumberKind} [kind]
 * @returns {string}
 */
export function pyStr(value, kind) {
	return typeof value === 'string' ? value : pyRepr(value, kind);
}

// ---------------------------------------------------------------------------------------------
// The int/float schema (see header)
// ---------------------------------------------------------------------------------------------

/** Result fields the Python builds as float (float(...) casts, np.float -> float(), true division). */
export const PY_FLOAT_KEYS = new Set([
	// cmd_meme / attribution.py
	'hyphaeon_lrt', 'p_value', 'q_value', 'runtime_sec', 'oci', 'predicted_lrt', 'delta_lrt',
	'pct_signal_explained', 'mean_patristic_depth', 'weighted_patristic_depth', 'tree_depth_ratio',
	// filter.py patches / run_alignment_filter
	'p_local', 'cct_p_value', 'mean_lrt', 'elapsed_seconds',
	// cmd_busted
	'p_value_acat', 'p_value_simes', 'omnibus_lrt', 'predicted_gene_lrt', 'selection_probability',
	'synonymous_rate_variation', 'total_selection_energy', 'omega_1', 'omega_2', 'omega_3',
	'proportion_1', 'proportion_2', 'proportion_3',
	'p_ACAT', 'p_Simes', 'Selection_Prob', 'Pred_Gene_LRT', 'Omnibus_LRT', 'Omega_3', 'Prop_Positive', 'Time_ms',
	// epistasis.py edges / sectors / plasticity
	'lrt', 'lrt_u', 'lrt_v', 'similarity', 'cesi', 'fdr_q', 'p_val', 'hyper_p', 'p_hyper', 'weight',
	'spectral_coherence', 'p_perm', 'null_coherence_mean', 'null_coherence_std', 'null_coherence_95',
	'isotropic_baseline', 'baseline_lrt', 'intrinsic_plasticity', 'mean_delta_lrt', 'max_delta_lrt',
	'min_delta_lrt', 'mutant_deltas', 'max_shift',
	// phenotype.py
	'p_lrt', 'attribution_norm', 'fg_mean_attn', 'bg_mean_attn', 'association_rho', 'p_assoc',
	'p_assoc_parametric', 'p_assoc_perm', 'score', 'foreground_freq_pct', 'background_freq_pct',
	'spectral_energy', 'norm_spectral_ratio', 'max_assoc', 'p_evd_length_adjusted', 'score_track_a',
	'score_track_b', 'dual_track_composite', 'gene_p_value_perm', 'expected_shared',
	// evaluation.py
	'pearson_r', 'spearman_rho', 'roc_auc', 'ppv', 'fpr',
]);

/** Result fields the Python builds as int (int(...) casts, len(), s + 1, counts). */
export const PY_INT_KEYS = new Set([
	'site', 'site_u', 'site_v', 'site_0indexed', 'site_1indexed', 'taxon_index', 'num_mutated_taxa',
	'taxa_count', 'codon_count', 'start', 'end', 'span', 'consecutive_mismatches', 'k', 'd',
	'num_taxa', 'num_codons', 'num_patches_detected', 'num_artifacts_masked', 'masked_codons_count',
	'suppressed_spurious_sites', 'sig_sites_p05', 'sig_sites_p10', 'sig_sites_q10',
	'taxa', 'sites', 'Taxa', 'Sites', 'Sig_Sites_p05',
	'shared_taxa', 'shared_branches', 'branches_u', 'branches_v', 'shared', 'sector_id', 'size',
	'evaluated_taxa', 'coselection_edges_count', 'discovered_sectors_count', 'focal_mutations_count',
	'total_mutations', 'focal_index',
	'foreground_count', 'background_count', 'permulations_count', 'significant_sites_count',
	'coselection_pairs_count', 'trait_sectors_count', 'total_mutations', 'shared_foreground_mutations',
	'matched_genes', 'total_sites', 'total_prediction_sites', 'total_meme_sites', 'evaluated_sites',
	'variable_sites', 'invariable_sites', 'clamped_negative_meme_lrts', 'meme_positive_sites',
	'hyphaeon_positive_sites', 'true_positive', 'false_positive', 'true_negative', 'false_negative',
	'prediction_sites', 'meme_sites', 'matched_sites', 'prediction_only_sites', 'meme_only_sites',
]);

/**
 * @param {string|undefined} key
 * @param {Set<string>} floatKeys
 * @param {Set<string>} intKeys
 * @returns {NumberKind}
 */
function kindOf(key, floatKeys, intKeys) {
	if (key === undefined) return undefined;
	if (floatKeys.has(key)) return 'float';
	if (intKeys.has(key)) return 'int';
	return undefined;
}

// ---------------------------------------------------------------------------------------------
// json.dumps(value, indent=N)
// ---------------------------------------------------------------------------------------------

/**
 * Python `json.dumps` string escaping with ensure_ascii=True.
 * @param {string} s
 * @returns {string}
 */
function pyJsonString(s) {
	let out = '"';
	for (let i = 0; i < s.length; i++) {
		const ch = s[i];
		const c = s.charCodeAt(i);
		if (ch === '"') out += '\\"';
		else if (ch === '\\') out += '\\\\';
		else if (ch === '\n') out += '\\n';
		else if (ch === '\r') out += '\\r';
		else if (ch === '\t') out += '\\t';
		else if (ch === '\b') out += '\\b';
		else if (ch === '\f') out += '\\f';
		else if (c < 0x20 || c > 0x7e) out += '\\u' + c.toString(16).padStart(4, '0');
		else out += ch;
	}
	return out + '"';
}

/**
 * Python `json.dumps(value, indent=indent)` (item separator `,`, key separator `: `), with the
 * int/float schema for numbers. Maps serialise as dicts in insertion order (use a Map when keys
 * are integer-like strings: a JS object would reorder them numerically). `undefined` object
 * values are omitted, as JSON.stringify does; `undefined` list items become null.
 *
 * @param {unknown} value
 * @param {{indent?: number, floatKeys?: Set<string>, intKeys?: Set<string>, allowNan?: boolean}} [options]
 * @returns {string}
 */
export function pyJsonDumps(value, { indent = 2, floatKeys = PY_FLOAT_KEYS, intKeys = PY_INT_KEYS, allowNan = true } = {}) {
	const pad = ' '.repeat(indent);
	/**
	 * @param {unknown} v
	 * @param {number} level
	 * @param {string|undefined} key
	 * @returns {string}
	 */
	function enc(v, level, key) {
		if (v === null || v === undefined) return 'null';
		if (typeof v === 'boolean') return v ? 'true' : 'false';
		if (typeof v === 'number') {
			if (!Number.isFinite(v)) {
				if (!allowNan) throw new RangeError(`Out of range float values are not JSON compliant: ${pyFloatRepr(v)}`);
				return Number.isNaN(v) ? 'NaN' : v > 0 ? 'Infinity' : '-Infinity';
			}
			return pyNumberStr(v, kindOf(key, floatKeys, intKeys));
		}
		if (typeof v === 'bigint') return v.toString();
		if (typeof v === 'string') return pyJsonString(v);
		const inner = pad.repeat(level + 1);
		const outer = pad.repeat(level);
		if (Array.isArray(v) || ArrayBuffer.isView(v)) {
			const arr = /** @type {ArrayLike<unknown>} */ (/** @type {unknown} */ (v));
			if (arr.length === 0) return '[]';
			const items = [];
			for (let i = 0; i < arr.length; i++) items.push(inner + enc(arr[i], level + 1, key));
			return '[\n' + items.join(',\n') + '\n' + outer + ']';
		}
		/** @type {[string, unknown][]} */
		const entries = [];
		if (v instanceof Map) {
			for (const [k, item] of v) entries.push([pyStr(k), item]);
		} else if (typeof v === 'object') {
			for (const [k, item] of Object.entries(/** @type {object} */ (v))) if (item !== undefined) entries.push([k, item]);
		} else {
			return pyJsonString(String(v));
		}
		if (entries.length === 0) return '{}';
		const items = entries.map(([k, item]) => inner + pyJsonString(k) + ': ' + enc(item, level + 1, k));
		return '{\n' + items.join(',\n') + '\n' + outer + '}';
	}
	return enc(value, 0, undefined);
}

// ---------------------------------------------------------------------------------------------
// pandas.DataFrame(records).to_csv(index=False)
// ---------------------------------------------------------------------------------------------

/**
 * csv.QUOTE_MINIMAL with the excel dialect and lineterminator '\n'.
 * @param {string} cell
 * @returns {string}
 */
function csvQuote(cell) {
	return /[",\r\n]/.test(cell) ? '"' + cell.replace(/"/g, '""') + '"' : cell;
}

/**
 * @param {string[]} cells
 * @returns {string}
 */
function csvRow(cells) {
	if (cells.length === 1 && cells[0] === '') return '""';
	return cells.map(csvQuote).join(',');
}

/**
 * Missing in the pandas sense: absent key, undefined, null, or a NaN number.
 * @param {unknown} v
 * @returns {boolean}
 */
function isMissing(v) {
	return v === undefined || v === null || (typeof v === 'number' && Number.isNaN(v));
}

/**
 * pandas `DataFrame(records).to_csv(index=False)` for a list of flat record objects (Maps accepted).
 *
 * @param {Array<Record<string, unknown>|Map<string, unknown>>} records
 * @param {{columns?: string[]|null, drop?: string[], floatKeys?: Set<string>, intKeys?: Set<string>}} [options]
 *   columns: explicit column order (default: union of keys in first-seen order);
 *   drop: columns to remove (DataFrame.drop(columns=...), silently ignored when absent)
 * @returns {string}
 */
export function dataFrameCsv(records, { columns = null, drop = [], floatKeys = PY_FLOAT_KEYS, intKeys = PY_INT_KEYS } = {}) {
	/** @param {Record<string, unknown>|Map<string, unknown>} r @param {string} k */
	const get = (r, k) => (r instanceof Map ? r.get(k) : r[k]);
	/** @param {Record<string, unknown>|Map<string, unknown>} r @returns {string[]} */
	const keysOf = (r) => (r instanceof Map ? [...r.keys()].map(String) : Object.keys(r));
	/** @type {string[]} */
	let cols;
	if (columns) {
		cols = [...columns];
	} else {
		cols = [];
		const seen = new Set();
		for (const r of records)
			for (const k of keysOf(r))
				if (!seen.has(k)) {
					seen.add(k);
					cols.push(k);
				}
	}
	if (drop.length) cols = cols.filter((c) => !drop.includes(c));

	/** @type {string[][]} */
	const table = records.map(() => []);
	for (const col of cols) {
		const kindHint = kindOf(col, floatKeys, intKeys);
		const values = records.map((r) => get(r, col));
		let anyMissing = false;
		let anyObject = false;
		let allBool = true;
		let allNumber = true;
		let allIntegral = true;
		let present = 0;
		for (const v of values) {
			if (isMissing(v)) {
				anyMissing = true;
				continue;
			}
			present++;
			if (typeof v === 'boolean') {
				allNumber = false;
			} else if (typeof v === 'number') {
				allBool = false;
				if (!Number.isInteger(v)) allIntegral = false;
			} else {
				anyObject = true;
				allBool = false;
				allNumber = false;
			}
		}
		/** @type {'object'|'bool'|'int'|'float'} */
		let dtype;
		if (present === 0 || anyObject) dtype = 'object';
		else if (allBool) dtype = anyMissing ? 'object' : 'bool';
		else if (allNumber) {
			const kind = kindHint ?? (allIntegral ? 'int' : 'float');
			dtype = kind === 'int' && !anyMissing && allIntegral ? 'int' : 'float';
		} else dtype = 'object';

		values.forEach((v, i) => {
			let cell;
			if (isMissing(v)) cell = '';
			else if (dtype === 'int') cell = pyIntStr(/** @type {number} */ (v));
			else if (dtype === 'float') cell = pyFloatRepr(/** @type {number} */ (v));
			else if (dtype === 'bool') cell = v ? 'True' : 'False';
			else cell = pyStr(v, kindHint);
			table[i].push(cell);
		});
	}
	const lines = [csvRow(cols)];
	for (const row of table) lines.push(csvRow(row));
	return lines.join('\n') + '\n';
}

// ---------------------------------------------------------------------------------------------
// json.dump wrappers (io.py:20-29 write_json; evaluation.py:641/644)
// ---------------------------------------------------------------------------------------------

/**
 * `write_json(path, data)` body — `json.dump(data, f, indent=2)` — for any CLI result dict
 * (meme, busted, epistasis, dms, phenotype, filter). No trailing newline, as json.dump writes none.
 * @param {unknown} result
 * @returns {string}
 */
export function resultJson(result) {
	return pyJsonDumps(result, { indent: 2 });
}

/**
 * `json.dumps(result, indent=2, allow_nan=False)` — evaluation.py:641 (file) and :644 (stdout).
 * Throws RangeError on NaN/inf like Python's ValueError. The file form appends "\n"; the caller adds it.
 * @param {unknown} result
 * @returns {string}
 */
export function evaluateJson(result) {
	return pyJsonDumps(result, { indent: 2, allowNan: false });
}

// ---------------------------------------------------------------------------------------------
// hyphaeon meme (cli.py:283-330)
// ---------------------------------------------------------------------------------------------

/**
 * @typedef {object} DrivingSpecies
 * @property {string} taxon
 * @property {string} observed_aa
 */
/**
 * @typedef {object} AttributionRecord  attribution.py:157-172 (one site)
 * @property {string} consensus_aa
 * @property {DrivingSpecies[]} driving_species
 * @property {{evolutionary_epoch: string, mode_of_adaptation: string}} when_selection_occurred
 */

/**
 * @param {Map<number, AttributionRecord>|Record<string|number, AttributionRecord>|null|undefined} attributions
 * @returns {Map<number, AttributionRecord>}
 */
function attributionMap(attributions) {
	const out = new Map();
	if (!attributions) return out;
	if (attributions instanceof Map) {
		for (const [k, v] of attributions) out.set(Number(k), v);
	} else {
		for (const [k, v] of Object.entries(attributions)) out.set(Number(k), v);
	}
	return out;
}

/**
 * cli.py:283-299 — the `results_list` of per-site dicts. `attributions` is keyed by 0-based site
 * index, as `attribute_selection` returns it. Attribution fields are added only for attributed
 * sites, in the Python's key order, and the whole record is attached as `attribution_details`.
 *
 * @param {ArrayLike<number>} lrts        float32 LRTs, length L
 * @param {ArrayLike<number>} pvals       cmd_meme's float32 p (see stats.js memeSitePq)
 * @param {ArrayLike<number>} qvals       cmd_meme's float32 q
 * @param {ArrayLike<number|boolean>} invariable  the invariable mask (1/true = invariable)
 * @param {Map<number, AttributionRecord>|Record<string|number, AttributionRecord>|null} [attributions]
 * @returns {Array<Record<string, unknown>>}
 */
export function memeSiteRecords(lrts, pvals, qvals, invariable, attributions = null) {
	const attr = attributionMap(attributions);
	const L = lrts.length;
	const out = [];
	for (let i = 0; i < L; i++) {
		/** @type {Record<string, unknown>} */
		const entry = {
			site: i + 1,
			hyphaeon_lrt: Number(lrts[i]),
			p_value: Number(pvals[i]),
			q_value: Number(qvals[i]),
			is_invariable: Boolean(invariable[i]),
		};
		const rec = attr.get(i);
		if (rec !== undefined) {
			const drivers = rec.driving_species;
			const hasDrivers = Array.isArray(drivers) && drivers.length > 0;
			entry.evolutionary_epoch = rec.when_selection_occurred.evolutionary_epoch;
			entry.adaptation_mode = rec.when_selection_occurred.mode_of_adaptation;
			entry.top_driver = hasDrivers ? drivers[0].taxon : null;
			entry.top_mutation = hasDrivers ? `${rec.consensus_aa}->${drivers[0].observed_aa}` : null;
			entry.attribution_details = rec;
		}
		out.push(entry);
	}
	return out;
}

/**
 * cli.py:301-314 — the `hyphaeon meme` JSON document as a Map-free object whose key order is the
 * Python's; `attributions` (0-based keys) becomes a Map keyed by the 1-based site as a string,
 * exactly `{str(k+1): v for k, v in attributions.items()}` in insertion order.
 *
 * @param {{alignment: string, tree?: string|null, taxaCount: number, codonCount: number,
 *   runtimeSec: number, filterEnabled?: boolean, artifactsMasked?: unknown[],
 *   attributionEnabled?: boolean, attributions?: Map<number, AttributionRecord>|Record<string|number, AttributionRecord>|null,
 *   sites: Array<Record<string, unknown>>}} parts
 * @returns {Record<string, unknown>}
 */
export function memeResult({
	alignment,
	tree = null,
	taxaCount,
	codonCount,
	runtimeSec,
	filterEnabled = false,
	artifactsMasked = [],
	attributionEnabled = false,
	attributions = null,
	sites,
}) {
	const byOneBased = new Map();
	for (const [k, v] of attributionMap(attributions)) byOneBased.set(String(k + 1), v);
	return {
		alignment,
		tree: tree ? tree : 'embedded_in_alignment',
		taxa_count: taxaCount,
		codon_count: codonCount,
		runtime_sec: runtimeSec,
		filter_enabled: Boolean(filterEnabled),
		artifacts_masked: artifactsMasked,
		attribution_enabled: Boolean(attributionEnabled),
		attributions: byOneBased,
		sites,
	};
}

/**
 * cli.py:303 — `write_json(args.output, {...})` for the meme document.
 * @param {unknown} result  from memeResult (or any object of that shape)
 * @returns {string}
 */
export function memeJson(result) {
	return resultJson(result);
}

const MEME_CSV_BASE = ['site', 'hyphaeon_lrt', 'p_value', 'q_value', 'is_invariable'];
const MEME_CSV_ATTR = ['evolutionary_epoch', 'adaptation_mode', 'top_driver', 'top_mutation'];

/**
 * cli.py:316-330 — the flattened site CSV. A row carries the four attribution columns only when
 * its site dict has `evolutionary_epoch`; the DataFrame then has those columns (empty for other
 * rows) if any site does. `attribution` overrides that inference: true forces the columns, false
 * drops them.
 *
 * @param {Array<Record<string, unknown>>} sites  from memeSiteRecords
 * @param {{attribution?: boolean}} [options]
 * @returns {string}
 */
export function memeCsv(sites, { attribution } = {}) {
	const anyAttr = sites.some((r) => 'evolutionary_epoch' in r);
	const withAttr = attribution === undefined ? anyAttr : attribution;
	const records = sites.map((r) => {
		/** @type {Record<string, unknown>} */
		const row = {};
		for (const k of MEME_CSV_BASE) row[k] = r[k];
		if (withAttr && 'evolutionary_epoch' in r) for (const k of MEME_CSV_ATTR) row[k] = r[k];
		return row;
	});
	const columns = withAttr && records.some((r) => 'evolutionary_epoch' in r) ? [...MEME_CSV_BASE, ...MEME_CSV_ATTR] : MEME_CSV_BASE;
	return dataFrameCsv(records, { columns });
}

// ---------------------------------------------------------------------------------------------
// hyphaeon busted (cli.py:552-571)
// ---------------------------------------------------------------------------------------------

/**
 * cli.py:552-553 — `write_json(args.output, batch_results if is_batch else batch_results[0])`.
 * `is_batch` is `len(input_files) > 1`; a caller that ran one alignment through a batch directory
 * of one file gets the bare record, as the Python does.
 *
 * @param {Array<Record<string, unknown>>} records
 * @param {{batch?: boolean}} [options]
 * @returns {string}
 */
export function bustedJson(records, { batch = records.length > 1 } = {}) {
	return resultJson(batch ? records : records[0]);
}

/**
 * cli.py:555-571 — the `df_summary` CSV. `Omega_3`/`Prop_Positive` are read from
 * `rate_distributions`; `Time_ms` is `elapsed_seconds * 1000`. A null `elapsed_seconds` (the e2e
 * fixtures null every timing field) gives an empty `Time_ms` cell here, where the Python would
 * raise TypeError on `None * 1000`; a real run always has the number.
 *
 * @param {Array<Record<string, any>>} records
 * @returns {string}
 */
export function bustedCsv(records) {
	const rows = records.map((r) => ({
		Gene: r.gene,
		Taxa: r.taxa,
		Sites: r.sites,
		p_ACAT: r.p_value_acat,
		p_Simes: r.p_value_simes,
		Selection_Prob: r.selection_probability,
		Pred_Gene_LRT: r.predicted_gene_lrt,
		Omnibus_LRT: r.omnibus_lrt,
		Omega_3: r.rate_distributions?.omega_3 == null ? null : Number(r.rate_distributions.omega_3),
		Prop_Positive: r.rate_distributions?.proportion_3 == null ? null : Number(r.rate_distributions.proportion_3),
		Sig_Sites_p05: r.sig_sites_p05,
		Selected: r.positive_selection_detected,
		Time_ms: r.elapsed_seconds == null ? null : r.elapsed_seconds * 1000,
	}));
	return dataFrameCsv(rows, {
		columns: ['Gene', 'Taxa', 'Sites', 'p_ACAT', 'p_Simes', 'Selection_Prob', 'Pred_Gene_LRT', 'Omnibus_LRT', 'Omega_3', 'Prop_Positive', 'Sig_Sites_p05', 'Selected', 'Time_ms'],
	});
}

// ---------------------------------------------------------------------------------------------
// hyphaeon phenotype / epistasis / dms CSVs (cli.py:702-703, 808-819, 898-902)
// ---------------------------------------------------------------------------------------------

/**
 * cli.py:702-703 — `write_csv(args.csv, sites)` with `sites = res["sites"]`.
 * @param {Array<Record<string, unknown>>} sites
 * @returns {string}
 */
export function phenotypeCsv(sites) {
	return dataFrameCsv(sites);
}

/**
 * cli.py:810-821 — which table `hyphaeon epistasis --csv` writes: the plasticity table when the
 * invoking subcommand is dms/essm/digital-dms and there is any, else the edges when there are
 * any, else the plasticity, else the sectors (whose `sites` lists print as Python list literals).
 * Note that `mutant_deltas` is NOT dropped on this path (only cmd_dms drops it), so each
 * plasticity row carries a quoted Python dict literal.
 *
 * @param {{edges?: Array<Record<string, unknown>>, sectors?: Array<Record<string, unknown>>, plasticity?: Array<Record<string, unknown>>}} result
 * @param {{command?: string}} [options]  the argparse subcommand name (default 'epistasis')
 * @returns {string}
 */
export function epistasisCsv(result, { command = 'epistasis' } = {}) {
	const edges = result.edges ?? [];
	const sectors = result.sectors ?? [];
	const plasticity = result.plasticity ?? [];
	let table;
	if (['dms', 'essm', 'digital-dms'].includes(command) && plasticity.length) table = plasticity;
	else if (edges.length) table = edges;
	else if (plasticity.length) table = plasticity;
	else table = sectors;
	return dataFrameCsv(table);
}

/**
 * cli.py:900-904 — `DataFrame(plasticity).drop(columns=["mutant_deltas"])` when present.
 * @param {Array<Record<string, unknown>>} plasticity
 * @returns {string}
 */
export function dmsCsv(plasticity) {
	return dataFrameCsv(plasticity, { drop: ['mutant_deltas'] });
}

// ---------------------------------------------------------------------------------------------
// nx.write_graphml (cli.py:823-835), lxml layout
// ---------------------------------------------------------------------------------------------

const GRAPHML_HEADER =
	"<?xml version='1.0' encoding='utf-8'?>\n" +
	'<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">';

/** libxml2 attribute-value escaping. @param {string} s */
function xmlAttr(s) {
	return s
		.replace(/&/g, '&amp;')
		.replace(/</g, '&lt;')
		.replace(/>/g, '&gt;')
		.replace(/"/g, '&quot;')
		.replace(/\n/g, '&#10;')
		.replace(/\r/g, '&#13;')
		.replace(/\t/g, '&#9;');
}

/** libxml2 text-content escaping. @param {string} s */
function xmlText(s) {
	return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\r/g, '&#13;');
}

/**
 * cli.py:823-835 —
 *
 *     G = nx.Graph()
 *     for e in edges:
 *         G.add_edge(str(e["site_u"]), str(e["site_v"]), weight=float(e["similarity"]),
 *                    cesi=float(e["cesi"]), shared=int(e["shared_branches"]), fdr_q=float(e["fdr_q"]))
 *     nx.write_graphml(G, path)
 *
 * `nodes` is an optional list of node ids to register before the edges (as `G.add_nodes_from`
 * would); cli.py registers none, so isolated nodes never appear in its output.
 *
 * @param {Array<{site_u: number|string, site_v: number|string, similarity: number, cesi: number, shared_branches: number, fdr_q: number}>} edges
 * @param {Array<number|string>} [nodes]
 * @returns {string}
 */
export function graphml(edges, nodes = []) {
	/** @type {Map<string, Map<string, Record<string, number>>>} */
	const adj = new Map();
	/** @param {string} id */
	const addNode = (id) => {
		if (!adj.has(id)) adj.set(id, new Map());
	};
	for (const n of nodes) addNode(pyStr(n, 'int'));
	for (const e of edges) {
		const u = pyStr(e.site_u, 'int');
		const v = pyStr(e.site_v, 'int');
		addNode(u);
		addNode(v);
		const data = /** @type {Map<string, Record<string, number>>} */ (adj.get(u)).get(v) ?? {};
		data.weight = Number(e.similarity);
		data.cesi = Number(e.cesi);
		data.shared = Math.trunc(Number(e.shared_branches));
		data.fdr_q = Number(e.fdr_q);
		/** @type {Map<string, Record<string, number>>} */ (adj.get(u)).set(v, data);
		/** @type {Map<string, Record<string, number>>} */ (adj.get(v)).set(u, data);
	}
	/** Edge attribute XML types, fixed by the cli.py casts. @type {Record<string, string>} */
	const XML_TYPE = { weight: 'double', cesi: 'double', shared: 'long', fdr_q: 'double' };

	// Graph.edges enumeration order.
	/** @type {Array<[string, string, Record<string, number>]>} */
	const edgeList = [];
	const seen = new Set();
	for (const [n, nbrs] of adj) {
		for (const [nbr, data] of nbrs) if (!seen.has(nbr)) edgeList.push([n, nbr, data]);
		seen.add(n);
	}
	// Key registration: first appearance of (name, type, scope) over the edge data, insert(0, ...).
	/** @type {Map<string, string>} */
	const keyIds = new Map();
	/** @type {string[]} */
	const keyLines = [];
	for (const [, , data] of edgeList) {
		for (const name of Object.keys(data)) {
			if (keyIds.has(name)) continue;
			const id = `d${keyIds.size}`;
			keyIds.set(name, id);
			keyLines.unshift(`<key id="${id}" for="edge" attr.name="${xmlAttr(name)}" attr.type="${XML_TYPE[name]}"/>\n`);
		}
	}
	let out = GRAPHML_HEADER + keyLines.join('') + '<graph edgedefault="undirected">';
	for (const n of adj.keys()) out += `<node id="${xmlAttr(n)}"/>\n`;
	for (const [u, v, data] of edgeList) {
		const names = Object.keys(data);
		if (names.length === 0) {
			out += `<edge source="${xmlAttr(u)}" target="${xmlAttr(v)}"/>\n`;
			continue;
		}
		out += `<edge source="${xmlAttr(u)}" target="${xmlAttr(v)}">\n`;
		for (const name of names) {
			const text = name === 'shared' ? pyIntStr(data[name]) : pyFloatRepr(data[name]);
			out += `  <data key="${keyIds.get(name)}">${xmlText(text)}</data>\n`;
		}
		out += '</edge>\n';
	}
	out += '</graph></graphml>';
	return out;
}
