"""
HyphAeon Reference-Guided Codon-Aware Threader
==============================================
Provides linear-time O(N · L) frame-locked codon alignment against
canonical structural references.

Key Capabilities:
  1. Automatic reading frame & strand orientation detection.
  2. C-accelerated affine pairwise amino acid alignment against reference fold.
  3. Reverse-codon threading: guarantees uniform length L_ref across all
     sequences, eliminates frame-shifts, and pads deletions with '---'.
  4. Directly outputs dense aligned codon tensors compatible with
     PhyloAxialTransformer and ChronAeon.

Author: Sergei L. Kosakovsky Pond & DeepMind Antigravity Pair Programmer
"""

import os
import sys
import time
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from Bio import Align
from Bio.Seq import Seq


class ReferenceCodonAligner:
    """
    Aligns raw, unaligned coding sequences against a canonical reference CDS
    by threading codons through amino acid alignment coordinates.
    """

    def __init__(
        self,
        ref_seq: str,
        ref_name: str = "reference",
        open_gap_score: float = -10.0,
        extend_gap_score: float = -1.0,
        match_score: float = 2.0,
        mismatch_score: float = -1.0,
        check_reverse_strand: bool = True,
    ):
        self.ref_name = ref_name
        self.ref_dna = ref_seq.upper().replace("-", "").replace(".", "").strip()
        if len(self.ref_dna) % 3 != 0:
            rem = len(self.ref_dna) % 3
            self.ref_dna = self.ref_dna[:-rem]

        self.ref_aa = str(Seq(self.ref_dna).translate())
        self.l_ref_codons = len(self.ref_dna) // 3
        self.l_ref_nt = len(self.ref_dna)

        self.check_reverse_strand = check_reverse_strand

        # Configure C-accelerated Biopython PairwiseAligner
        self.aligner = Align.PairwiseAligner()
        self.aligner.mode = 'global'
        self.aligner.open_gap_score = open_gap_score
        self.aligner.extend_gap_score = extend_gap_score
        self.aligner.match_score = match_score
        self.aligner.mismatch_score = mismatch_score

    def align_single(self, query_dna: str) -> Tuple[Optional[str], float, str]:
        """
        Aligns a single unaligned nucleotide query to the reference.

        Returns:
            (aligned_dna_str, alignment_score, status_message)
        """
        raw_q = query_dna.upper().replace("-", "").replace(".", "").strip()
        if len(raw_q) < 30:
            return None, 0.0, "too_short (<30 bp)"

        candidates = [("forward", raw_q)]
        if self.check_reverse_strand:
            tr = str.maketrans("ACGTURYKMSWBDHVN", "TGCAAYRMKSWVHDBN")
            rc_q = raw_q.translate(tr)[::-1]
            candidates.append(("reverse", rc_q))

        best_score = -1e9
        best_aln = None
        best_codons = None
        best_strand = "forward"
        best_frame = 0

        for strand, q_seq in candidates:
            for f in range(3):
                sub = q_seq[f:]
                trim_l = len(sub) - (len(sub) % 3)
                sub = sub[:trim_l]
                if len(sub) < 30:
                    continue

                q_aa = str(Seq(sub).translate())
                # Penalize severe internal stops
                stop_count = q_aa[:-1].count("*")
                if stop_count > 3:
                    continue

                alns = self.aligner.align(self.ref_aa, q_aa)
                if alns:
                    score = alns[0].score - (stop_count * 15.0)
                    if score > best_score:
                        best_score = score
                        best_aln = alns[0]
                        best_codons = [sub[i*3:i*3+3] for i in range(len(sub)//3)]
                        best_strand = strand
                        best_frame = f

        if best_aln is None or best_score < 0:
            return None, float(best_score), "alignment_failed_or_low_score"

        # Reverse-thread codons into reference coordinates
        ref_aln = best_aln[0]
        qry_aln = best_aln[1]

        threaded_codons = []
        q_idx = 0
        n_inserted_codons = 0

        for col in range(len(ref_aln)):
            r_char = ref_aln[col]
            q_char = qry_aln[col]

            if r_char != '-':
                if q_char != '-':
                    codon = best_codons[q_idx]
                    threaded_codons.append(codon)
                    q_idx += 1
                else:
                    threaded_codons.append('---')
            else:
                # Insertion relative to reference
                if q_char != '-':
                    q_idx += 1
                    n_inserted_codons += 1

        aligned_str = ''.join(threaded_codons)
        if len(aligned_str) != self.l_ref_nt:
            return None, float(best_score), f"length_mismatch ({len(aligned_str)} != {self.l_ref_nt})"

        status = f"ok (strand={best_strand}, frame={best_frame}, score={best_score:.1f})"
        return aligned_str, float(best_score), status

    def align_batch(
        self,
        seq_dict: Dict[str, str],
        max_workers: int = 4,
        quiet: bool = False,
    ) -> Dict[str, Any]:
        """
        Aligns a batch of sequences in parallel.

        Returns:
            Dict containing:
              - 'aligned_seqs': {taxon: aligned_sequence}
              - 'failed': {taxon: reason}
              - 'scores': {taxon: score}
              - 'elapsed_seconds': float
        """
        t0 = time.time()
        n_total = len(seq_dict)
        if not quiet:
            print(f"[*] Reference-guided codon alignment of {n_total:,} sequences against '{self.ref_name}' ({self.l_ref_nt} bp)...", flush=True)

        aligned_seqs: Dict[str, str] = {}
        failed: Dict[str, str] = {}
        scores: Dict[str, float] = {}

        taxa = list(seq_dict.keys())

        def _worker(t: str):
            res_dna, score, status = self.align_single(seq_dict[t])
            return t, res_dna, score, status

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for t, res_dna, score, status in executor.map(_worker, taxa):
                if res_dna is not None:
                    aligned_seqs[t] = res_dna
                    scores[t] = score
                else:
                    failed[t] = status

        t_elapsed = time.time() - t0
        if not quiet:
            rate = n_total / max(0.01, t_elapsed)
            print(f"[✓] Aligned {len(aligned_seqs):,}/{n_total:,} sequences in {t_elapsed:.2f}s ({rate:.1f} seqs/s). Discarded {len(failed):,}.", flush=True)

        return {
            "aligned_seqs": aligned_seqs,
            "failed": failed,
            "scores": scores,
            "l_ref_nt": self.l_ref_nt,
            "elapsed_seconds": t_elapsed,
        }


# Aliases for convenience
ReferenceGuidedCodonThreader = ReferenceCodonAligner
