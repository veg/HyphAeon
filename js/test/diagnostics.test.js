/**
 * diagnostics.test.js — the "Before you run" checks of PLAN.md §4.3 (src/diagnostics.js).
 *
 * WHY THIS FILE EXISTS
 *
 * There is no Python oracle for `diagnose` (PLAN.md §7 item 6 asks upstream for one), so the tests
 * are constructive: one synthetic alignment or tree per code, built so that exactly the intended
 * rule fires, with the data fields checked against values you can count by hand. The bundled
 * examples then pin what the panel says about real inputs: bat_oas1's tree is in Mya
 * (DISTANCE_RESCALED), camelid.nwk and HIV1_RT.nwk carry no branch lengths so both go tree-free
 * (TREE_FREE_TN93, reason 'no_branch_lengths'), Smc6's 20 primates are shallow by the 0.05 rule
 * (SHALLOW_TREE — a documented consequence of the threshold, see the module header), RHO has 710
 * taxa over a deep mammalian tree (TAXA_OVER_CAP, DEEP_LARGE_TREE).
 *
 * D22 (tree-free TN93) rewrote three of these cases: TREE_MISSING and BRANCH_LENGTHS_MISSING are
 * gone, replaced by the info-level TREE_FREE_TN93, and TN93_SATURATED_PAIRS is new. Because the
 * tree-free path really computes distances, the 24-nucleotide SEQS alignment above is too short for
 * it — eight codons saturate the TN93 formula — so the tree-free cases use LONG_SEQS (30 codons,
 * hand-built so no pair saturates) and one case deliberately keeps a saturating input to pin the
 * refuse branch. Distances there are TN93, so the depth-regime codes now fire on inputs that used
 * to suppress them (camelid: DEEP_LARGE_TREE; HIV1_RT: SHALLOW_TREE).
 *
 * Every warning is checked to carry a code from DIAGNOSTIC_CODES and a valid severity, and the
 * output order is the §4.3 row order with COST_ESTIMATE last.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
	diagnose,
	DIAGNOSTIC_CODES,
	DIAGNOSTIC_THRESHOLDS,
	WORK_PER_SECOND,
	sniffAlignmentFormat,
	meanPairwiseDivergence,
	medianOffDiagonal
} from '../src/diagnostics.js';
import { loadAlignmentAndTree } from '../src/preprocess/assemble.js';
import { Xoshiro256 } from '../src/numeric/prng.js';

const EXAMPLES = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'examples');
const readExample = (name) => readFileSync(join(EXAMPLES, name), 'utf8');

/** Six taxa, eight stop-free codons, mutually distinct, divergence well above the star-like floor. */
const SEQS = {
	alpha: 'ATGGCTAAAGAATTTTGGCATCGA',
	beta: 'ATGGGTAGAGGATATTGTCACCGG',
	gamma: 'ATGGCTAGAGAATATTGGCATCGG',
	delta: 'ATGGGTAAAGGATTTTGTCACCGA',
	eps: 'ATGGCCAAGGAGTTCTGGCATCGA',
	zeta: 'ATGGGCAGGGGGTACTGTCACCGG'
};
const fasta = (seqs = SEQS) =>
	Object.entries(seqs)
		.map(([n, s]) => `>${n}\n${s}\n`)
		.join('');
const TREE6 = '((alpha:0.1,beta:0.1):0.05,(gamma:0.1,delta:0.1):0.05,(eps:0.1,zeta:0.1):0.05);';

/**
 * The same six taxa over 30 codons: TN93 needs enough overlapping, non-saturating sites to return a
 * distance at all (tn93.js), so every tree-free case uses these. Each sequence differs from the
 * first in a handful of positions only.
 */
const LONG_STEM = 'ATGGCTAAAGAATTTTGGCATCGAACCGGGTTTAAACCCGGGTTTAAACCCGGGTTTAAACCCGGGTTTAAACCCGGG';
const LONG_SEQS = {
	alpha: LONG_STEM + 'ATGGCTAAA',
	beta: LONG_STEM.replace('GCTAAA', 'GGTAGA') + 'ATGGCTAAG',
	gamma: LONG_STEM.replace('GAATTT', 'GAGTAT') + 'ATGGCCAAA',
	delta: LONG_STEM.replace('TGGCAT', 'TGTCAC') + 'ATGGCTAGA',
	eps: LONG_STEM.replace('ACCGGG', 'ACAGGA') + 'ATGGCAAAA',
	zeta: LONG_STEM.replace('TTTAAACCC', 'TTCAAGCCG') + 'ATGGCTAAT'
};

const codes = (r) => r.warnings.map((w) => w.code);
const find = (r, code) => r.warnings.find((w) => w.code === code);
const run = (alignmentText, treeText = TREE6, extra = {}) => diagnose({ alignmentText, treeText, ...extra });

function checkShape(r) {
	expect(typeof r.ok).toBe('boolean');
	for (const w of r.warnings) {
		expect(DIAGNOSTIC_CODES).toContain(w.code);
		expect(['info', 'warn', 'refuse']).toContain(w.severity);
		expect(typeof w.message).toBe('string');
		expect(w.message.length).toBeGreaterThan(0);
		expect(typeof w.data).toBe('object');
	}
	expect(r.ok).toBe(!r.warnings.some((w) => w.severity === 'refuse'));
	// §4.3 row order, COST_ESTIMATE last when present.
	const ranks = r.warnings.map((w) => DIAGNOSTIC_CODES.indexOf(w.code));
	expect(ranks).toEqual([...ranks].sort((a, b) => a - b));
	if (find(r, 'COST_ESTIMATE')) expect(r.warnings[r.warnings.length - 1].code).toBe('COST_ESTIMATE');
}

describe('the clean synthetic panel', () => {
	it('passes with only the cost estimate', () => {
		const r = run(fasta());
		checkShape(r);
		expect(r.ok).toBe(true);
		expect(codes(r)).toEqual(['COST_ESTIMATE']);
		expect(r.summary).toMatchObject({
			format: 'fasta',
			treeSource: 'user',
			taxaInAlignment: 6,
			taxaMatched: 6,
			uniqueHaplotypes: 6,
			taxaUsed: 6,
			codons: 8,
			matchTier: 'exact'
		});
		expect(r.summary.medianPatristic).toBeCloseTo(0.3, 6);
		expect(r.summary.meanPairwiseDivergence).toBeGreaterThan(0.05);
	});

	it('COST_ESTIMATE data is {L, N_used, work, predictedSeconds} with the calibrated constant', () => {
		const c = find(run(fasta()), 'COST_ESTIMATE');
		expect(c.severity).toBe('info');
		expect(c.data.L).toBe(8);
		expect(c.data.N_used).toBe(6);
		expect(c.data.work).toBe(8 * 36);
		expect(c.data.predictedSeconds).toBe((8 * 36) / WORK_PER_SECOND);
		expect(WORK_PER_SECOND).toBe(4.7e6);
		expect(c.data.exceedsCaps).toEqual({ codons: false, work: false });
		expect(c.data.caps).toEqual({ codons: 30000, work: 2.5e9 });
		expect(c.data.assumption).toMatch(/HIV1_RT/);
	});

	it('accepts an existing loadAlignmentAndTree result and reports the same', () => {
		const parsed = loadAlignmentAndTree(fasta(), TREE6);
		const a = run(fasta());
		const b = run(fasta(), TREE6, { parsed });
		expect(b.warnings).toEqual(a.warnings);
		expect(b.summary).toEqual(a.summary);
	});
});

describe('format and alphabet', () => {
	it('FORMAT_UNKNOWN refuses text that parses to no sequences, and a bare ">" header', () => {
		const r = run('hello world\n');
		checkShape(r);
		expect(r.ok).toBe(false);
		expect(codes(r)).toEqual(['FORMAT_UNKNOWN']);
		expect(find(r, 'FORMAT_UNKNOWN').data.format).toBe('unknown');
		const bare = run('>\nATG\n');
		expect(codes(bare)).toEqual(['FORMAT_UNKNOWN']);
		expect(find(bare, 'FORMAT_UNKNOWN').message).toMatch(/could not be parsed/);
	});

	it('sniffAlignmentFormat follows parse.js detection order', () => {
		expect(sniffAlignmentFormat(fasta())).toBe('fasta');
		expect(sniffAlignmentFormat('  4 12\nalpha ATGATGATGATG\n')).toBe('phylip');
		expect(sniffAlignmentFormat('#NEXUS\nBEGIN DATA;\nMATRIX\n;\n')).toBe('nexus');
		expect(sniffAlignmentFormat('nothing')).toBe('unknown');
		expect(run('#NEXUS\nBEGIN DATA;\nMATRIX\nalpha ATGATG\n;\nEND;\n', null).summary.format).toBe('nexus');
	});

	it('ALPHABET_U_TO_T counts the U characters read as T, names excluded', () => {
		const { gamma, ...rest } = SEQS;
		const seqs = { ...rest, alpha: 'AUGGCUAAAGAAUUUUGGCAUCGA', beta: SEQS.beta.replace(/T/g, 'u'), uracil_taxon_u: gamma };
		const r = run(fasta(seqs), TREE6.replace('gamma', 'uracil_taxon_u'));
		checkShape(r);
		const w = find(r, 'ALPHABET_U_TO_T');
		expect(w.severity).toBe('info');
		// alpha: AUG GCU AAA GAA UUU UGG CAU CGA -> 7 U; beta: ATG GGT AGA GGA TAT TGT CAC CGG -> 6 T -> u.
		expect(w.data.count).toBe(7 + 6);
		expect(r.ok).toBe(true);
	});

	it('NON_ACGT_FRACTION is info for gaps, warn above 5% ambiguity, refuse for a protein alphabet', () => {
		const gappy = run(fasta({ ...SEQS, alpha: 'ATGGCTAAAGAATTTTGG---CGA' }));
		const g = find(gappy, 'NON_ACGT_FRACTION');
		expect(g.severity).toBe('info');
		expect(g.data).toMatchObject({ gaps: 3, ambiguous: 0, total: 144, acgt: 141 });
		expect(g.data.fraction).toBeCloseTo(3 / 144, 12);

		const nn = run(fasta({ ...SEQS, alpha: 'ATGGCTAAAGAATTTTGGNNNNNN', beta: 'ATGGGTAGAGGATATTGTNNNNNN' }));
		const n = find(nn, 'NON_ACGT_FRACTION');
		expect(n.severity).toBe('warn');
		expect(n.data.ambiguous).toBe(12);
		expect(n.data.ambiguousFraction).toBeCloseTo(12 / 144, 12);

		const protein = run(fasta({ alpha: 'MKLVQPRSEDWHFYNIKLVQPRSE', beta: 'MKLIQPRSEDWHFYNIRLVQPRSE', gamma: 'MRLVQPKSEDWHFYNIKLVQPRSD', delta: 'MKLVEPRSEDWHFYSIKLVQPRSE', eps: 'MKLVQPRSEDYHFYNIKLVQPRSE', zeta: 'MKLVQPRSEDWHFYNIKLVQPRSQ' }));
		const p = find(protein, 'NON_ACGT_FRACTION');
		expect(p.severity).toBe('refuse');
		expect(p.data.nonGapAmbiguousFraction).toBeGreaterThan(0.5);
		expect(protein.ok).toBe(false);
	});
});

describe('length, frame and stops', () => {
	it('LENGTH_NOT_MULTIPLE_OF_3 warns with the remainder dataset.py trims from the first taxon', () => {
		const r = run(fasta({ ...SEQS, alpha: SEQS.alpha + 'AC', beta: SEQS.beta + 'A' }));
		checkShape(r);
		const w = find(r, 'LENGTH_NOT_MULTIPLE_OF_3');
		expect(w.severity).toBe('warn');
		expect(w.data).toMatchObject({ sequences: 2, remainder: 2, trimmedNucleotides: 2 });
		expect(w.data.names).toEqual(['alpha', 'beta']);
		expect(find(r, 'UNEQUAL_LENGTHS').data).toMatchObject({ min: 24, max: 26, codonsFromFirst: 8 });
		expect(r.summary.codons).toBe(8);
	});

	it('LENGTH_NOT_MULTIPLE_OF_3 refuses a first sequence shorter than a codon', () => {
		const r = run(fasta({ alpha: 'AT', beta: 'AT', gamma: 'AT', delta: 'AT', eps: 'AT', zeta: 'AT' }));
		checkShape(r);
		expect(find(r, 'LENGTH_NOT_MULTIPLE_OF_3').severity).toBe('refuse');
		expect(r.summary.taxaUsed).toBeNull();
	});

	it('IN_FRAME_STOPS warns on an internal stop and ignores a terminal one', () => {
		// One stop among 6 x 104 internal codons (0.16 %, below the 1 % refuse line).
		const seqs = longPanel([]);
		seqs.alpha = 'ATGTAA' + seqs.alpha.slice(6);
		const internal = run(fasta(seqs));
		checkShape(internal);
		const w = find(internal, 'IN_FRAME_STOPS');
		expect(w.severity).toBe('warn');
		expect(w.data).toMatchObject({ internal: 1, terminal: 0, sequencesWithInternal: 1, codonsScanned: 6 * 104 });
		expect(w.data.fraction).toBeCloseTo(1 / 624, 12);
		expect(internal.ok).toBe(true);
		expect(find(internal, 'FRAMESHIFT_SUSPECTED')).toBeUndefined();

		const term = longPanel([]);
		term.alpha = term.alpha.slice(0, -3) + 'TAA';
		term.beta = term.beta.slice(0, -3) + 'TGA' + '---';
		const terminal = run(fasta(term));
		expect(find(terminal, 'IN_FRAME_STOPS')).toBeUndefined();
		expect(find(terminal, 'FRAMESHIFT_SUSPECTED')).toBeUndefined();
	});

	it('IN_FRAME_STOPS refuses when internal stops exceed 1% of internal codons (short panel: 1 of 42 already does)', () => {
		const one = find(run(fasta({ ...SEQS, alpha: 'ATGGCTTAAGAATTTTGGCATCGA' })), 'IN_FRAME_STOPS');
		expect(one.severity).toBe('refuse');
		expect(one.data).toMatchObject({ internal: 1, codonsScanned: 42 });
		const r = run(fasta({ ...SEQS, alpha: 'ATGTAATAATAATAATAACATCGA' }));
		checkShape(r);
		const w = find(r, 'IN_FRAME_STOPS');
		expect(w.severity).toBe('refuse');
		expect(w.data.internal).toBe(5);
		expect(w.data.fraction).toBeGreaterThan(DIAGNOSTIC_THRESHOLDS.internalStopRefuseFraction);
		expect(r.ok).toBe(false);
	});

	// A stop-free ORF X: shifting it by one base ('G' + X + 'TT') puts two stops in frame 0 while
	// frame 1 (X itself) has none.
	const ORF = 'ACTAAGCCTGACCCA' + 'GCT'.repeat(100);
	const SHIFTED = 'G' + ORF + 'TT';
	const longPanel = (shiftedNames) => {
		const seqs = {};
		for (const [i, n] of Object.keys(SEQS).entries()) {
			// Distinct haplotypes: a taxon-specific synonymous change near the end.
			const tail = ['GCA', 'GCC', 'GCG', 'GCT', 'GCA', 'GCC'][i] + ['GCT', 'GCA', 'GCG', 'GCC', 'GCC', 'GCT'][i];
			const body = ORF.slice(0, ORF.length - 6) + tail;
			seqs[n] = shiftedNames.includes(n) ? 'G' + body + 'TT' : body;
		}
		return seqs;
	};

	it('FRAMESHIFT_SUSPECTED warns for a sequence with fewer stops in another frame', () => {
		expect(SHIFTED.length % 3).toBe(0);
		const r = run(fasta(longPanel(['beta'])));
		checkShape(r);
		const w = find(r, 'FRAMESHIFT_SUSPECTED');
		expect(w.severity).toBe('warn');
		expect(w.data.sequences).toEqual(['beta']);
		expect(w.data.frameStops[0][0]).toBe(2);
		expect(w.data.frameStops[0][1]).toBe(0);
		expect(find(r, 'IN_FRAME_STOPS').severity).toBe('warn');
		expect(find(r, 'DUPLICATE_SEQUENCES')).toBeUndefined();
	});

	it('FRAMESHIFT_SUSPECTED refuses when more than half the sequences are frame-suspect', () => {
		const r = run(fasta(longPanel(['alpha', 'beta', 'gamma', 'delta'])));
		checkShape(r);
		const w = find(r, 'FRAMESHIFT_SUSPECTED');
		expect(w.severity).toBe('refuse');
		expect(w.data.count).toBe(4);
		expect(w.data.fraction).toBeCloseTo(4 / 6, 12);
	});

	it('UNKNOWN_CODON_FRACTION warns above 5% of codons, from the loaded notices', () => {
		const r = run(fasta({ ...SEQS, alpha: 'NNNNNN---AAAGAATTTTGGCATCGA'.slice(0, 24) }));
		checkShape(r);
		const w = find(r, 'UNKNOWN_CODON_FRACTION');
		expect(w.severity).toBe('warn');
		expect(w.data).toMatchObject({ unknownCodons: 3, totalCodons: 48, threshold: 0.05 });
		expect(w.data.fraction).toBeCloseTo(3 / 48, 12);
	});

	it('UNKNOWN_CODON_FRACTION is also computed when the model-level load fails', () => {
		// No tree -> tree-free TN93 (D22), and eight codons of which three are NNN/gaps saturate the
		// formula, so the load raises exactly where the reference does and the count falls back to
		// the light path.
		const r = run(fasta({ ...SEQS, alpha: 'NNNNNN---AAAGAATTTTGGCATCGA'.slice(0, 24) }), null);
		expect(find(r, 'TREE_FREE_TN93').data.reason).toBe('no_tree');
		expect(find(r, 'TN93_SATURATED_PAIRS')).toMatchObject({ severity: 'refuse', data: { pairs: null } });
		expect(find(r, 'TN93_SATURATED_PAIRS').data.error).toMatch(/ZeroDivisionError|expected a positive input/);
		expect(find(r, 'UNKNOWN_CODON_FRACTION').data).toMatchObject({ unknownCodons: 3, totalCodons: 48 });
	});
});

describe('taxa', () => {
	it('UNEQUAL_LENGTHS warns and reports the lengths', () => {
		const r = run(fasta({ ...SEQS, zeta: SEQS.zeta.slice(0, 18) }));
		checkShape(r);
		const w = find(r, 'UNEQUAL_LENGTHS');
		expect(w.severity).toBe('warn');
		expect(w.data).toMatchObject({ lengths: [18, 24], min: 18, max: 24 });
	});

	it('DUPLICATE_SEQUENCES is info with the collapse map', () => {
		const r = run(fasta({ ...SEQS, zeta: SEQS.alpha, eps: SEQS.alpha }));
		checkShape(r);
		const w = find(r, 'DUPLICATE_SEQUENCES');
		expect(w.severity).toBe('info');
		expect(w.data.collapsed).toBe(2);
		expect(w.data.duplicateMap).toEqual({ alpha: ['eps', 'zeta'] });
		expect(r.summary.uniqueHaplotypes).toBe(4);
		expect(r.summary.taxaUsed).toBe(4);
		// Four haplotypes: below the star-like floor of 5.
		expect(find(r, 'STAR_LIKE').data.uniqueHaplotypes).toBe(4);
	});

	it('TOO_FEW_TAXA refuses below 3 usable taxa (after collapse)', () => {
		const r = run(fasta({ alpha: SEQS.alpha, beta: SEQS.beta, gamma: SEQS.alpha }), '((alpha:0.1,beta:0.1):0.1,gamma:0.1);');
		checkShape(r);
		const w = find(r, 'TOO_FEW_TAXA');
		expect(w.severity).toBe('refuse');
		expect(w.data).toEqual({ taxa: 2, minimum: 3 });
		expect(r.ok).toBe(false);
	});

	it('TAXA_OVER_CAP is info and the cost uses the subsampled count', () => {
		const r = run(fasta(), TREE6, { maxSpecies: 4 });
		checkShape(r);
		const w = find(r, 'TAXA_OVER_CAP');
		expect(w.severity).toBe('info');
		expect(w.data).toEqual({ taxa: 6, cap: 4, used: 4, stridePreselect: false });
		expect(r.summary.taxaUsed).toBe(4);
		expect(find(r, 'COST_ESTIMATE').data.N_used).toBe(4);
		expect(run(fasta(), TREE6, { maxSpecies: 2 }).warnings.find((x) => x.code === 'TAXA_OVER_CAP').data.stridePreselect).toBe(true);
	});

	it('TAXA_OVER_LIMIT refuses and skips the model-level load', () => {
		const r = run(fasta(), TREE6, { taxaLimit: 5 });
		checkShape(r);
		expect(find(r, 'TAXA_OVER_LIMIT')).toMatchObject({ severity: 'refuse', data: { taxa: 6, limit: 5 } });
		expect(r.summary.taxaUsed).toBeNull();
		expect(find(r, 'TAXA_OVER_CAP')).toBeUndefined();
	});
});

describe('tree', () => {
	it('TREE_FREE_TN93 informs without a tree, and for the tn93/none/skip modes and useTn93 (D22)', () => {
		const r = run(fasta(LONG_SEQS), null);
		checkShape(r);
		expect(find(r, 'TREE_FREE_TN93')).toMatchObject({
			severity: 'info',
			data: { reason: 'no_tree', taxaOrder: 'alignment', distances: 'tn93', matchMode: 'resolve', treeKeptForDisplay: false }
		});
		expect(r.ok).toBe(true);
		expect(r.summary.treeSource).toBeNull();
		expect(find(run(fasta(LONG_SEQS), 'TN93 '), 'TREE_FREE_TN93').data.reason).toBe('requested');
		expect(find(run(fasta(LONG_SEQS), 'none'), 'TREE_FREE_TN93').data.reason).toBe('requested');
		expect(find(run(fasta(LONG_SEQS), 'skip'), 'TREE_FREE_TN93').data.reason).toBe('requested');
		// A perfectly good tree, overridden.
		const forced = run(fasta(LONG_SEQS), TREE6, { useTn93: true });
		expect(find(forced, 'TREE_FREE_TN93')).toMatchObject({ severity: 'info', data: { reason: 'requested', treeKeptForDisplay: true } });
	});

	it('uses an embedded tree when no tree text is given', () => {
		const r = run(fasta() + '\n' + TREE6 + '\n', null);
		expect(find(r, 'TREE_FREE_TN93')).toBeUndefined();
		expect(r.summary.treeSource).toBe('embedded');
		expect(r.ok).toBe(true);
	});

	it('TREE_UNPARSEABLE refuses text that is not a tree', () => {
		const r = run(fasta(), 'not a tree');
		checkShape(r);
		expect(find(r, 'TREE_UNPARSEABLE').severity).toBe('refuse');
		expect(find(run(fasta(), '(alpha:0.1,beta:0.2);'), 'TREE_UNPARSEABLE')).toBeDefined(); // single pair, dataset.py quirk
	});

	it('TAXA_NOT_IN_TREE refuses alignment sequences without a tip; TIPS_NOT_IN_ALIGNMENT warns', () => {
		const r = run(fasta({ ...SEQS, extra: SEQS.alpha.replace('CGA', 'CGC') }), '((alpha:0.1,beta:0.1):0.05,(gamma:0.1,delta:0.1):0.05,(eps:0.1,zeta:0.1,ghost:0.1):0.05);');
		checkShape(r);
		expect(find(r, 'TAXA_NOT_IN_TREE')).toMatchObject({ severity: 'refuse', data: { count: 1, names: ['extra'], matchTier: 'exact' } });
		expect(find(r, 'TIPS_NOT_IN_ALIGNMENT')).toMatchObject({ severity: 'warn', data: { count: 1, names: ['ghost'] } });
		expect(r.summary.taxaMatched).toBe(6);
		expect(r.ok).toBe(false);
	});

	it('TAXA_NOT_IN_TREE refuses when nothing matches at all', () => {
		const r = run(fasta(), '((a:0.1,b:0.1):0.05,(c:0.1,d:0.1):0.05);');
		checkShape(r);
		expect(find(r, 'TAXA_NOT_IN_TREE')).toMatchObject({ severity: 'refuse', data: { count: 6, matchTier: null } });
		expect(r.summary.taxaMatched).toBeNull();
	});

	it('TAXA_NOT_IN_TREE catches the issue #9 zero rows a case-insensitive match produces', () => {
		const upper = Object.fromEntries(Object.entries(SEQS).map(([n, s]) => [n.toUpperCase(), s]));
		const r = run(fasta(upper));
		checkShape(r);
		expect(r.summary.matchTier).toBe('case_insensitive');
		const w = find(r, 'TAXA_NOT_IN_TREE');
		expect(w.severity).toBe('refuse');
		expect(w.data.zeroDistanceRows).toBe(true);
		expect(w.data.count).toBe(6);
		expect(find(r, 'TIPS_NOT_IN_ALIGNMENT')).toBeUndefined();
	});

	it('TREE_FREE_TN93 takes over a topology-only tree, keeps it for display and measures the depth', () => {
		const r = run(fasta(LONG_SEQS), '((alpha,beta),(gamma,delta),(eps,zeta));');
		checkShape(r);
		const w = find(r, 'TREE_FREE_TN93');
		expect(w.severity).toBe('info');
		expect(w.data).toMatchObject({
			reason: 'no_branch_lengths',
			treeKeptForDisplay: true,
			branchLengths: { branches: 9, missing: 9, positives: 0 }
		});
		// The distances are now real (TN93), so the depth regime is judged instead of skipped.
		expect(r.summary.medianPatristic).toBeGreaterThan(0);
		expect(find(r, 'DEEP_LARGE_TREE')).toBeUndefined(); // 6 taxa, far under the 100 rule
		expect(r.ok).toBe(true);
	});

	it('NEGATIVE_BRANCH_LENGTHS warns with the minimum', () => {
		const r = run(fasta(), '((alpha:0.1,beta:-0.02):0.05,(gamma:0.1,delta:0.1):0.05,(eps:0.1,zeta:-0.5):0.05);');
		checkShape(r);
		expect(find(r, 'NEGATIVE_BRANCH_LENGTHS')).toMatchObject({ severity: 'warn', data: { count: 2, min: -0.5, raisedTo: 1e-4 } });
		// Some branches are positive, so the tree is still usable and the run stays tree-based.
		expect(find(r, 'TREE_FREE_TN93')).toBeUndefined();
	});

	it('DISTANCE_RESCALED warns when the largest patristic distance exceeds 10', () => {
		const r = run(fasta(), '((alpha:12,beta:12):3,(gamma:12,delta:12):3,(eps:12,zeta:12):3);');
		checkShape(r);
		const w = find(r, 'DISTANCE_RESCALED');
		expect(w.severity).toBe('warn');
		expect(w.data).toEqual({ rawMax: 30, threshold: 10, dividedBy: 8 });
		expect(r.summary.medianPatristic).toBeCloseTo(30 / 8, 5);
	});
});

/** n taxa on a balanced binary tree with every edge `edge` long; sequences mutated from a base ORF. */
function syntheticPanel(n, codons, edge, seed) {
	const CODONS = ['GCT', 'GGT', 'AAA', 'GAA', 'TTT', 'TGG', 'CAT', 'CGA', 'ATG', 'CCT', 'AGC', 'GTT'];
	const rng = new Xoshiro256(seed);
	const names = Array.from({ length: n }, (_, i) => `t${i}`);
	const seqs = {};
	for (const name of names) {
		let s = '';
		for (let p = 0; p < codons; p++) {
			const base = CODONS[p % CODONS.length];
			s += rng.uniform() < 0.2 ? CODONS[(p + 1 + Math.floor(rng.uniform() * 5)) % CODONS.length] : base;
		}
		seqs[name] = s;
	}
	const build = (list) => (list.length === 1 ? `${list[0]}:${edge}` : `(${build(list.slice(0, Math.ceil(list.length / 2)))},${build(list.slice(Math.ceil(list.length / 2)))}):${edge}`);
	return { alignmentText: fasta(seqs), treeText: build(names).replace(/:[^:]*$/, ';') };
}

describe('regime', () => {
	it('SHALLOW_TREE is info below a median patristic distance of 0.05 and suggests the viral variant', () => {
		const r = run(fasta(), TREE6.replace(/0\.1/g, '0.01').replace(/0\.05/g, '0.005'));
		checkShape(r);
		const w = find(r, 'SHALLOW_TREE');
		expect(w.severity).toBe('info');
		expect(w.data.suggestVariant).toBe('viral');
		expect(w.data.medianPatristic).toBeCloseTo(0.03, 6);
		expect(r.ok).toBe(true);
	});

	it('DEEP_LARGE_TREE warns for >= 100 taxa on a deep tree', () => {
		const { alignmentText, treeText } = syntheticPanel(120, 60, 0.15, 7);
		const r = diagnose({ alignmentText, treeText });
		checkShape(r);
		expect(r.summary.uniqueHaplotypes).toBe(120);
		expect(find(r, 'DUPLICATE_SEQUENCES')).toBeUndefined();
		const w = find(r, 'DEEP_LARGE_TREE');
		expect(w.severity).toBe('warn');
		expect(w.data.taxa).toBe(120);
		expect(w.data.medianPatristic).toBeGreaterThanOrEqual(0.2);
		expect(find(r, 'STAR_LIKE')).toBeUndefined();
		expect(find(r, 'DISTANCE_RESCALED')).toBeUndefined();
		// Same depth with fewer taxa: no warning.
		const small = syntheticPanel(40, 60, 0.15, 7);
		expect(find(diagnose(small), 'DEEP_LARGE_TREE')).toBeUndefined();
	});

	it('STAR_LIKE warns for tiny divergence and for fewer than 5 haplotypes', () => {
		const near = {};
		for (const [i, n] of Object.keys(SEQS).entries()) near[n] = 'GCT'.repeat(200).slice(0, 597) + ['GCA', 'GCC', 'GCG', 'GCT', 'GCC', 'GCA'][i] + '';
		near.alpha = 'GCT'.repeat(200);
		near.beta = 'GCT'.repeat(199) + 'GCC';
		near.gamma = 'GCT'.repeat(199) + 'GCA';
		near.delta = 'GCT'.repeat(199) + 'GCG';
		near.eps = 'GCT'.repeat(198) + 'GCCGCT';
		near.zeta = 'GCT'.repeat(198) + 'GCAGCT';
		const r = run(fasta(near));
		checkShape(r);
		const w = find(r, 'STAR_LIKE');
		expect(w.severity).toBe('warn');
		expect(w.data.uniqueHaplotypes).toBe(6);
		expect(w.data.meanPairwiseDivergence).toBeLessThan(0.005);

		const few = run(fasta({ alpha: SEQS.alpha, beta: SEQS.beta, gamma: SEQS.gamma, delta: SEQS.delta }), '((alpha:0.1,beta:0.1):0.05,(gamma:0.1,delta:0.1):0.05);');
		expect(find(few, 'STAR_LIKE').data.uniqueHaplotypes).toBe(4);
		expect(find(few, 'STAR_LIKE').data.meanPairwiseDivergence).toBeGreaterThan(0.005);
	});

	it('meanPairwiseDivergence and medianOffDiagonal by hand', () => {
		expect(meanPairwiseDivergence(['ACGT', 'ACGA'])).toBe(0.25);
		expect(meanPairwiseDivergence(['ACGT', 'ACGA', 'ACGT'])).toBeCloseTo((0.25 + 0 + 0.25) / 3, 12);
		expect(meanPairwiseDivergence(['NNNN', 'ACGT'])).toBeNull();
		expect(meanPairwiseDivergence(['ACGT'])).toBeNull();
		expect(meanPairwiseDivergence(['AC-T', 'ACGA'])).toBeCloseTo(1 / 3, 12);
		expect(medianOffDiagonal([0, 1, 2, 1, 0, 4, 2, 4, 0], 3)).toBe(2);
		expect(medianOffDiagonal([0, 1, 1, 0], 2)).toBe(1);
		expect(medianOffDiagonal([0], 1)).toBeNull();
	});
});

describe('bundled examples', () => {
	it('bat_oas1: the tree is in Mya, so DISTANCE_RESCALED', () => {
		const r = diagnose({ alignmentText: readExample('bat_oas1.fasta'), treeText: readExample('bat_oas1.nwk') });
		checkShape(r);
		expect(r.ok).toBe(true);
		expect(codes(r)).toEqual(['NON_ACGT_FRACTION', 'DISTANCE_RESCALED', 'COST_ESTIMATE']);
		expect(find(r, 'DISTANCE_RESCALED').data.rawMax).toBeCloseTo(123.35, 1);
		expect(r.summary).toMatchObject({ taxaInAlignment: 18, taxaUsed: 18, codons: 351, matchTier: 'exact' });
		expect(r.summary.medianPatristic).toBeCloseTo(0.3513, 3);
	});

	it('camelid: no branch lengths, so tree-free TN93, which is deep enough to warn', () => {
		const r = diagnose({ alignmentText: readExample('camelid.fasta'), treeText: readExample('camelid.nwk') });
		checkShape(r);
		expect(r.ok).toBe(true);
		expect(codes(r)).toEqual(['NON_ACGT_FRACTION', 'TREE_FREE_TN93', 'DEEP_LARGE_TREE', 'COST_ESTIMATE']);
		expect(find(r, 'TREE_FREE_TN93').data).toMatchObject({ reason: 'no_branch_lengths', branchLengths: { branches: 421, missing: 421, positives: 0 } });
		expect(find(r, 'TN93_SATURATED_PAIRS')).toBeUndefined(); // no pair saturates on camelid
		expect(r.summary).toMatchObject({ taxaInAlignment: 212, taxaUsed: 212, codons: 96, matchTier: null });
		// TN93 distances over the whole alignment, no tree involved.
		expect(r.summary.medianPatristic).toBeCloseTo(0.20776, 4);
	});

	it('Smc6: 20 primates read as shallow by the 0.05 rule (documented consequence)', () => {
		const r = diagnose({ alignmentText: readExample('Smc6.fasta'), treeText: readExample('Smc6.nwk') });
		checkShape(r);
		expect(r.ok).toBe(true);
		expect(codes(r)).toEqual(['NON_ACGT_FRACTION', 'SHALLOW_TREE', 'COST_ESTIMATE']);
		expect(r.summary.medianPatristic).toBeCloseTo(0.02376, 4);
		expect(r.summary.meanPairwiseDivergence).toBeCloseTo(0.0232, 3);
	});

	it('HIV1_RT: no branch lengths in the bundled tree, one duplicate, 30% unknown codons', () => {
		const r = diagnose({ alignmentText: readExample('HIV1_RT.fasta'), treeText: readExample('HIV1_RT.nwk') });
		checkShape(r);
		expect(r.ok).toBe(true);
		expect(codes(r)).toEqual(['NON_ACGT_FRACTION', 'UNKNOWN_CODON_FRACTION', 'DUPLICATE_SEQUENCES', 'TREE_FREE_TN93', 'SHALLOW_TREE', 'COST_ESTIMATE']);
		expect(find(r, 'TREE_FREE_TN93').data.reason).toBe('no_branch_lengths');
		expect(r.summary.medianPatristic).toBeCloseTo(0.046269, 5);
		expect(find(r, 'DUPLICATE_SEQUENCES').data.collapsed).toBe(1);
		expect(r.summary).toMatchObject({ taxaInAlignment: 476, uniqueHaplotypes: 475, taxaUsed: 475, codons: 335 });
		expect(find(r, 'COST_ESTIMATE').data.work).toBe(335 * 475 * 475);
		expect(find(r, 'COST_ESTIMATE').data.predictedSeconds).toBeCloseTo(16.08, 1);
	}, 20000);

	it('RHO: 710 taxa in NEXUS with an embedded tree, over the cap and deep', () => {
		const r = diagnose({ alignmentText: readExample('RHO.fasta'), treeText: null });
		checkShape(r);
		expect(r.ok).toBe(true);
		expect(r.summary).toMatchObject({ format: 'nexus', treeSource: 'embedded', taxaInAlignment: 710, uniqueHaplotypes: 655, taxaUsed: 512, codons: 349 });
		expect(codes(r)).toEqual(['NON_ACGT_FRACTION', 'DUPLICATE_SEQUENCES', 'TAXA_OVER_CAP', 'DEEP_LARGE_TREE', 'COST_ESTIMATE']);
		expect(find(r, 'TAXA_OVER_CAP').data).toEqual({ taxa: 655, cap: 512, used: 512, stridePreselect: false });
		expect(find(r, 'DEEP_LARGE_TREE').data.medianPatristic).toBeCloseTo(0.7116, 3);
	}, 30000);
});
