/**
 * tokenizer.test.js — hand-checkable values for src/preprocess/tokenizer.js.
 *
 * WHY THIS FILE EXISTS
 *
 * The fixture replay (fixtures.test.js) is the oracle for every token; this file is the readable
 * half: values anyone can check against dataset.py:27-59 with a finger on the table, so a failure
 * there says WHICH rule broke. Replaces DataMonkey 3's axomeme-tokenizer.test.js, whose cases pinned
 * the AxoMEME 2.0 training vocabulary (ATG 35, GGG 63, TAA a real token, gap 64 / unknown 65,
 * amino-acid sentinels 20/21/22 and a "gap anywhere wins" rule). Those cases were DELETED, not
 * adapted: they asserted a vocabulary the shipped checkpoint does not use (PHASE0.md gap 1).
 */
import { describe, it, expect } from 'vitest';
import {
	CODON_LIST,
	GENETIC_CODE,
	AA_MAP,
	CODON_TO_AA,
	codonToken,
	aaToken,
	tokenizeSequence
} from '../src/preprocess/tokenizer.js';
import { CODON_UNKNOWN, CODON_STOP, AA_UNKNOWN, AA_STOP, NUM_CODON_TOKENS, NUM_AA_TOKENS } from '../src/preprocess/modelContract.js';

describe('the tables (dataset.py:27-52)', () => {
	it('lists the 64 codons in TCAG order, third position fastest', () => {
		expect(CODON_LIST).toHaveLength(64);
		expect(CODON_LIST.slice(0, 4)).toEqual(['TTT', 'TTC', 'TTA', 'TTG']);
		expect(CODON_LIST[63]).toBe('GGG');
	});

	it('GENETIC_CODE numbers the 61 sense codons 0..60 consecutively and sends the 3 stops to 64', () => {
		expect(GENETIC_CODE.size).toBe(64);
		const sense = [...GENETIC_CODE.values()].filter((v) => v !== 64);
		expect(sense).toHaveLength(61);
		expect(new Set(sense).size).toBe(61);
		expect(Math.max(...sense)).toBe(60);
		expect([...GENETIC_CODE].filter(([, v]) => v === 64).map(([k]) => k).sort()).toEqual(['TAA', 'TAG', 'TGA']);
		// Row 1 of dataset.py:28-29, read off the source: TAA and TAG do not consume a number, so
		// TGT is 10, not 12.
		expect(GENETIC_CODE.get('TAC')).toBe(9);
		expect(GENETIC_CODE.get('TGT')).toBe(10);
		expect(GENETIC_CODE.get('TGG')).toBe(12);
		expect(GENETIC_CODE.get('CTT')).toBe(13);
		expect(GENETIC_CODE.get('ATG')).toBe(32);
		expect(GENETIC_CODE.get('AAA')).toBe(39);
		expect(GENETIC_CODE.get('GGG')).toBe(60);
	});

	it('AA_MAP is A..Y alphabetical 0..19 and CODON_TO_AA is the standard code with * for stops', () => {
		expect([...AA_MAP.keys()].join('')).toBe('ACDEFGHIKLMNPQRSTVWY');
		expect(AA_MAP.get('A')).toBe(0);
		expect(AA_MAP.get('M')).toBe(10);
		expect(AA_MAP.get('Y')).toBe(19);
		expect(CODON_TO_AA.size).toBe(64);
		expect(CODON_TO_AA.get('ATG')).toBe('M');
		expect(CODON_TO_AA.get('TGG')).toBe('W');
		expect(CODON_TO_AA.get('TTA')).toBe('L');
		expect([...CODON_TO_AA].filter(([, a]) => a === '*').map(([c]) => c).sort()).toEqual(['TAA', 'TAG', 'TGA']);
	});

	it('the embedding tables are larger than the vocabulary the tokenizer produces', () => {
		expect(NUM_CODON_TOKENS).toBe(66);
		expect(NUM_AA_TOKENS).toBe(23);
		expect(Math.max(...GENETIC_CODE.values())).toBeLessThan(NUM_CODON_TOKENS);
		expect(AA_UNKNOWN).toBeLessThan(NUM_AA_TOKENS);
	});
});

describe('codonToken (dataset.py:54-55)', () => {
	it('is `GENETIC_CODE.get(codon.upper(), 64)`', () => {
		expect(codonToken('atg')).toBe(32);
		expect(codonToken('ATG')).toBe(32);
		expect(codonToken('TTA')).toBe(2);
	});

	it('sends stops, gaps, ambiguity and wrong lengths all to the one sentinel, 64', () => {
		expect(CODON_STOP).toBe(64);
		expect(CODON_UNKNOWN).toBe(64);
		for (const c of ['TAA', 'TAG', 'TGA', '---', 'A-T', 'NNN', 'ANT', '', 'AT', 'ATGC', 'XYZ', '???']) {
			expect(codonToken(c), c).toBe(64);
		}
	});

	it('does NOT translate U: that happens once, in parse_alignment_sequences', () => {
		expect(codonToken('AUG')).toBe(64);
		expect(codonToken('aug')).toBe(64);
	});
});

describe('aaToken (dataset.py:57-59)', () => {
	it('translates through CODON_TO_AA then AA_MAP', () => {
		expect(aaToken('ATG')).toBe(10); // M
		expect(aaToken('ttt')).toBe(4); // F
		expect(aaToken('GGG')).toBe(5); // G
	});

	it('sends a stop, a gap and anything untranslatable to the one sentinel, 20', () => {
		expect(AA_STOP).toBe(20);
		expect(AA_UNKNOWN).toBe(20);
		for (const c of ['TAA', 'TAG', 'TGA', '---', 'A-T', 'ANT', 'AUG', '', 'AT', '?A?']) {
			expect(aaToken(c), c).toBe(20);
		}
	});
});

describe('tokenizeSequence', () => {
	it('tokenises in frame and drops a trailing partial codon (len // 3)', () => {
		const { codons, aas } = tokenizeSequence('ATGTTTAA');
		expect(Array.from(codons)).toEqual([32, 0]);
		expect(Array.from(aas)).toEqual([10, 4]);
	});

	it('carries gaps through as the 64 / 20 sentinels', () => {
		const { codons, aas } = tokenizeSequence('ATG---');
		expect(Array.from(codons)).toEqual([32, 64]);
		expect(Array.from(aas)).toEqual([10, 20]);
	});

	it('handles an empty and a sub-codon sequence', () => {
		expect(tokenizeSequence('').codons).toHaveLength(0);
		expect(tokenizeSequence('AT').codons).toHaveLength(0);
	});
});
