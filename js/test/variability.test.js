/**
 * variability.test.js — the invariable-site rule of dataset.py:718-723.
 *
 * WHY THIS FILE EXISTS
 *
 * Which sites are SCORED AT ALL is decided here, so the rule deserves cases a reader can verify by
 * translating codons. Replaces the DataMonkey 3 `isSiteVariable` / `siteVariability` cases from
 * axomeme-postprocess.test.js: the serine-island case ("TCA + AGC is variable") asserted a rule that
 * is not in dataset.py and is now asserted the OTHER way; the stop-codon behaviour (a stop counted
 * as an observation) likewise. The fixture replay pins the full masks for Smc6 and bat_oas1.
 */
import { describe, it, expect } from 'vitest';
import { isAaInvariable, invariableMask, isSiteVariable, siteVariability } from '../src/preprocess/variability.js';

describe('isAaInvariable (the exact loop body)', () => {
	it('is `len(np.unique(aa_col[aa_col < 20])) <= 1`', () => {
		expect(isAaInvariable([3, 3, 3])).toBe(true);
		expect(isAaInvariable([3, 4])).toBe(false);
		expect(isAaInvariable([3])).toBe(true);
		expect(isAaInvariable([])).toBe(true);
	});

	it('excludes token 20 (stop / gap / unknown) from the comparison', () => {
		expect(isAaInvariable([20, 20, 20])).toBe(true);
		expect(isAaInvariable([3, 20, 3, 20])).toBe(true);
		expect(isAaInvariable([20, 3, 4])).toBe(false);
	});
});

describe('isSiteVariable over codon strings', () => {
	it('is true when more than one amino acid is present', () => {
		expect(isSiteVariable(['ATG', 'TTA'])).toBe(true); // M, L
	});

	it('is false when every codon codes the same amino acid', () => {
		expect(isSiteVariable(['TTA', 'TTG', 'CTA'])).toBe(false); // all Leucine
		expect(isSiteVariable(['ATG', 'ATG'])).toBe(false);
	});

	it('is FALSE for a serine island: dataset.py compares amino-acid tokens only', () => {
		// DM3 called this variable (TCN vs AGY families). dataset.py:721 sees token 15 twice.
		expect(isSiteVariable(['TCA', 'AGC'])).toBe(false);
		expect(isSiteVariable(['TCT', 'AGT'])).toBe(false);
	});

	it('does not count a stop as an observation', () => {
		expect(isSiteVariable(['ATG', 'TAA'])).toBe(false); // M + stop: one valid residue
		expect(isSiteVariable(['ATG', 'TAA', 'TTA'])).toBe(true);
		expect(isSiteVariable(['TAA', 'TGA'])).toBe(false);
	});

	it('is false for an empty, gapped or ambiguous site', () => {
		expect(isSiteVariable([])).toBe(false);
		expect(isSiteVariable(null)).toBe(false);
		expect(isSiteVariable(['---', 'NNN', 'A-G'])).toBe(false);
	});
});

describe('siteVariability and invariableMask', () => {
	it('ignores gapped and ambiguous codons when judging a site', () => {
		// Site 0: ATG / --- / ANT -> one usable residue, not variable.
		// Site 1: TTA / TTG / TTT -> Leu, Leu, Phe -> variable.
		expect(siteVariability(['ATGTTA', '---TTG', 'ANTTTT'], 2)).toEqual([false, true]);
	});

	it('lets a short sequence contribute nothing past its end (the slice is shorter than 3)', () => {
		expect(siteVariability(['ATGTTA', 'ATG'], 2)[1]).toBe(false);
		expect(siteVariability(['ATGTTA', 'ATGC'], 2)[1]).toBe(false);
	});

	it('invariableMask walks a [L, N, 1] token array site by site', () => {
		// L = 3, N = 2: site 0 (10, 10) invariable; site 1 (9, 4) variable; site 2 (15, 20) invariable.
		const a = Int32Array.from([10, 10, 9, 4, 15, 20]);
		expect(Array.from(invariableMask(a, 3, 2))).toEqual([1, 0, 1]);
	});
});
