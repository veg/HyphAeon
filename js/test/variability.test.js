/**
 * WHY THIS FILE EXISTS
 *
 * The `isSiteVariable` and `siteVariability` describe blocks of datamonkey3 (main@fac1330)
 * src/test/axomeme-postprocess.test.js, lifted with their cases and comments unchanged and their
 * import repointed at `../src/preprocess/variability.js`.
 *
 * ONLY THOSE TWO BLOCKS CAME ACROSS. The rest of that file exercises `buildPredictions` — tier
 * labels, z-scores, percentile ranks, call modes, the float32 gate boundary — which is result
 * semantics and lives in `hyphaeon-app`'s `runtime/` along with the code it tests. Those cases are
 * not lost, they are owned elsewhere; splitting them here is what keeps the library/app boundary in
 * src/README.md checkable rather than aspirational.
 *
 * WHAT THIS PINS, and why it is worth its own file: the invariable-site rule decides which sites are
 * SCORED AT ALL. A site marked invariable is zeroed before the model's output is consulted
 * (`hyphaeon/dataset.py:718-723` computes the same mask), so a disagreement here does not shift a
 * number — it changes the set of sites that have numbers. The serine case below is exactly such a
 * disagreement against dataset.py today, and variability.js's header explains why the port was left
 * alone rather than reconciled by inspection. If a fixture later settles it the other way, THIS is
 * the file that has to change first.
 */
import { describe, it, expect } from 'vitest';
import { isSiteVariable, siteVariability } from '../src/preprocess/variability.js';

describe('isSiteVariable', () => {
	it('is true when more than one amino acid is present', () => {
		expect(isSiteVariable(['ATG', 'TTA'])).toBe(true); // M, L
	});

	it('is false when every codon codes the same amino acid', () => {
		expect(isSiteVariable(['TTA', 'TTG', 'CTA'])).toBe(false); // all Leucine
		expect(isSiteVariable(['ATG', 'ATG'])).toBe(false);
	});

	it('is TRUE for a serine island — synonymous but selection-relevant', () => {
		// The condition that is easy to drop. Serine is the one residue whose codons occupy two
		// disjoint blocks (TCN and AGY), so switching between them is synonymous yet needs multiple
		// substitutions. Every codon here is Serine, so the amino-acid test alone says "invariant".
		expect(isSiteVariable(['TCA', 'AGC'])).toBe(true);
		expect(isSiteVariable(['TCT', 'AGT'])).toBe(true);
		// ...but only when BOTH families are present.
		expect(isSiteVariable(['TCA', 'TCG'])).toBe(false);
		expect(isSiteVariable(['AGC', 'AGT'])).toBe(false);
	});

	it('is false for an empty site', () => {
		expect(isSiteVariable([])).toBe(false);
		expect(isSiteVariable(null)).toBe(false);
	});
});

describe('siteVariability', () => {
	it('ignores gapped and ambiguous codons when judging a site', () => {
		// Site 0: ATG / --- / ANT -> only one usable codon, so not variable.
		// Site 1: TTA / TTG / TTT -> Leu, Leu, Phe -> variable.
		const flags = siteVariability(['ATGTTA', '---TTG', 'ANTTTT'], 2);
		expect(flags).toEqual([false, true]);
	});

	it('lets a short sequence contribute nothing past its end', () => {
		const flags = siteVariability(['ATGTTA', 'ATG'], 2);
		expect(flags[1]).toBe(false); // only one codon reaches site 1
	});
});
