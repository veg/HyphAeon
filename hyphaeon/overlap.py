"""Overlapping reading frame (dual-coding) detection for HyphAeon.

HyphAeon already ingests codon alignments; an overlapping reading frame is simply the
SAME alignment re-tokenized at a 1-nucleotide offset. This module detects, in a CDS
alignment, alternate reading frames (+1, +2, and optionally the antisense strand) that
carry a genuine dual-coding signal — an ORF that stays open across taxa AND shows a
double-constraint substitution signature — with no re-alignment. It is fast (frame-shift
re-tokenization) and model-free for the screen; per-frame HyPhy dN/dS confirms.

Why single-frame selection tools miss this: in an overlap each nucleotide sits in two
codons, so a change synonymous in frame A may be non-synonymous (constrained) in frame B.
The signature measured here (Chung et al. 2007; Szklarczyk et al. 2007, PNAS):
  * substitution density in the overlap vs the rest of the gene (double constraint -> lower),
  * position-3 (wobble) suppression,
  * syn_in_B: fraction of substitutions that stay synonymous in frame B,
  * synB_at_nonsynA: frame-B constraint at frame-A NON-synonymous sites (geometry-robust),
  * inflation_flag: extreme wobble freeze (codon-usage/RNA-structure confounder, e.g. KRAS).

Roadmap: the model-free signature is a screen; per-frame dN/dS is the adjudicator; a
dual-frame HyphAeon head (predicting per-frame selection + a frame-B stop-avoidance term
in one pass) is the fast, reliable detector this is built toward.
"""
from __future__ import annotations

import os
from collections import OrderedDict

GENETIC_CODE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L',
    'CTG': 'L', 'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M', 'GTT': 'V', 'GTC': 'V',
    'GTA': 'V', 'GTG': 'V', 'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'CCT': 'P',
    'CCC': 'P', 'CCA': 'P', 'CCG': 'P', 'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A', 'TAT': 'Y', 'TAC': 'Y', 'TAA': '*',
    'TAG': '*', 'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q', 'AAT': 'N', 'AAC': 'N',
    'AAA': 'K', 'AAG': 'K', 'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E', 'TGT': 'C',
    'TGC': 'C', 'TGA': '*', 'TGG': 'W', 'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R', 'GGT': 'G', 'GGC': 'G', 'GGA': 'G',
    'GGG': 'G'}
_RC = str.maketrans("ACGTacgtNn-", "TGCAtgcaNn-")
_COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
_BASES = set("ACGT")


# --------------------------------------------------------------------------- IO
def read_alignment(path):
    """Read FASTA or (relaxed, sequential) PHYLIP [.gz] into OrderedDict{name: seq}."""
    import gzip
    opener = gzip.open if path.endswith(".gz") else open
    seqs = OrderedDict()
    with opener(path, "rt") as fh:
        head = fh.readline()
        is_phylip = len(head.split()) == 2 and head.split()[0].isdigit()
        if not is_phylip and head.startswith(">"):
            name, buf = head[1:].strip().split()[0], []
            for line in fh:
                if line.startswith(">"):
                    seqs[name] = "".join(buf).upper()
                    name, buf = line[1:].strip().split()[0], []
                else:
                    buf.append(line.strip())
            seqs[name] = "".join(buf).upper()
        elif is_phylip:
            for line in fh:
                p = line.split(None, 1)
                if len(p) == 2:
                    seqs[p[0].split("_")[0]] = p[1].replace(" ", "").replace("\n", "").upper()
        else:  # fasta whose first line we consumed as non-header (shouldn't happen)
            pass
    ncols = max((len(s) for s in seqs.values()), default=0)
    for n in list(seqs):
        if len(seqs[n]) < ncols:
            seqs[n] += "-" * (ncols - len(seqs[n]))
    return seqs, ncols


def revcomp_alignment(seqs):
    return OrderedDict((n, s.translate(_RC)[::-1]) for n, s in seqs.items())


# ----------------------------------------------------------------- ORF discovery
def discover_alt_orfs(seqs, order, ref, ncols, allow, min_codons=25):
    """Longest stop-free alt-frame ORFs (per allowed strand+frame) in the reference,
    conservation-scored by cross-species stop fraction. `allow` = set of (strand, frame)."""
    out = []
    for strand in ("+", "-"):
        s = revcomp_alignment(seqs) if strand == "-" else seqs
        rseq = s[ref]
        for frame in (0, 1, 2):
            if (strand, frame) not in allow:
                continue
            n = max(0, (ncols - frame) // 3)
            runs, start = [], None
            for c in range(n):
                tri = rseq[frame + 3 * c:frame + 3 * c + 3]
                broken = ("-" in tri or "N" in tri or len(tri) < 3
                          or GENETIC_CODE.get(tri) == "*")
                if broken:
                    if start is not None:
                        runs.append((start, c - 1)); start = None
                elif start is None:
                    start = c
            if start is not None:
                runs.append((start, n - 1))
            for a, b in runs:
                if b - a + 1 < min_codons:
                    continue
                cols = list(range(frame + 3 * a, frame + 3 * b + 3))
                fwd = [(ncols - 1 - w) for w in cols] if strand == "-" else cols
                stops = tot = 0
                for c in range(a, b + 1):
                    for sp in order:
                        tri = s[sp][frame + 3 * c:frame + 3 * c + 3]
                        if "-" in tri or "N" in tri or len(tri) < 3:
                            continue
                        tot += 1
                        if GENETIC_CODE.get(tri) == "*":
                            stops += 1
                out.append({"strand": strand, "frame": frame, "n_codons": b - a + 1,
                            "nt_start": min(fwd) + 1, "nt_end": max(fwd) + 1,
                            "species_stop_frac": round(stops / max(1, tot), 4)})
    out.sort(key=lambda r: (-r["n_codons"], r["species_stop_frac"]))
    return out


# --------------------------------------------------------- double-constraint signature
def _syn(ref_cod, mut_cod):
    ra, ma = GENETIC_CODE.get(ref_cod), GENETIC_CODE.get(mut_cod)
    if ra is None or ma is None:
        return None, False
    return (ra == ma), (ma == "*" and ra != "*")


def signature(seqs, order, ref, ncols, region, frameB, strandB="+", min_subs=40):
    """Double-constraint metrics comparing the overlap region to the rest of the CDS.
    region = (nt_start, nt_end) 1-based forward. Returns metrics + a boolean `passes`."""
    refseq = seqs[ref]
    rc = refseq.translate(_RC)[::-1] if strandB == "-" else None
    r0, r1 = region
    ovcols = set(range(r0 - 1, r1))
    st = {b: {"subs": 0, "sites": 0, "pos": [0, 0, 0],
              "spec": {"SS": 0, "SN": 0, "NS": 0, "NN": 0}} for b in ("ov", "no")}
    for col in range(ncols):
        rb = refseq[col] if col < len(refseq) else "-"
        if rb not in _BASES:
            continue
        bucket = "ov" if col in ovcols else "no"
        posA = col % 3
        ca = 3 * (col // 3)
        ref_cA = refseq[ca:ca + 3]
        for sp in order:
            if sp == ref:
                continue
            b = seqs[sp][col] if col < len(seqs[sp]) else "-"
            if b not in _BASES or b == rb:
                continue
            st[bucket]["subs"] += 1
            st[bucket]["pos"][posA] += 1
            synA = _syn(ref_cA, ref_cA[:posA] + b + ref_cA[posA + 1:])[0] if len(ref_cA) == 3 else None
            synB = None
            if bucket == "ov":
                if strandB == "+" and col >= frameB:
                    cb = frameB + 3 * ((col - frameB) // 3)
                    pB = (col - frameB) % 3
                    cod = refseq[cb:cb + 3]
                    if len(cod) == 3:
                        synB = _syn(cod, cod[:pB] + b + cod[pB + 1:])[0]
                elif strandB == "-":
                    rcc = ncols - 1 - col
                    if rcc >= frameB:
                        cb = frameB + 3 * ((rcc - frameB) // 3)
                        pB = (rcc - frameB) % 3
                        cod = rc[cb:cb + 3]
                        if len(cod) == 3:
                            mb = _COMP.get(b, b)
                            synB = _syn(cod, cod[:pB] + mb + cod[pB + 1:])[0]
            if synA is not None and bucket == "ov" and synB is not None:
                st["ov"]["spec"][("S" if synA else "N") + ("S" if synB else "N")] += 1
        st[bucket]["sites"] += 1
    ns = len(order)

    def dens(x):
        return x["subs"] / (max(1, x["sites"]) * max(1, ns - 1))
    o, no = st["ov"], st["no"]
    p3o = o["pos"][2] / max(1, sum(o["pos"]))
    p3n = no["pos"][2] / max(1, sum(no["pos"]))
    spec = o["spec"]; tot = max(1, sum(spec.values()))
    nonsynA = spec["NS"] + spec["NN"]
    sig = {
        "density_ratio": round(dens(o) / max(1e-9, dens(no)), 3),
        "pos3_ratio": round(p3o / max(1e-9, p3n), 3),
        "syn_in_B": round((spec["SS"] + spec["NS"]) / tot, 3),
        "synB_at_nonsynA": round(spec["NS"] / nonsynA, 3) if nonsynA else None,
        "nonsynA_subs": nonsynA,
        "n_overlap_subs": o["subs"],
    }
    sig["inflation_flag"] = int(sig["pos3_ratio"] < 0.20)
    sig["passes"] = passes_signature(sig, min_subs)
    return sig


def passes_signature(sig, min_subs=40, min_syn_in_B=0.30, max_pos3_ratio=0.85,
                     max_density_ratio=0.90, min_synB_at_nonsynA=0.45, min_nonsynA=20):
    """Real overlap: frame B constrained (high syn_in_B), wobble suppressed (pos3<1),
    region more conserved than the gene (density<1), and — geometry-robust — frame B
    constrained even at frame-A non-synonymous sites. Calibrated on CDKN2A/p14ARF."""
    if not (sig["n_overlap_subs"] >= min_subs and sig["syn_in_B"] >= min_syn_in_B
            and sig["pos3_ratio"] <= max_pos3_ratio
            and sig.get("density_ratio", 1) <= max_density_ratio):
        return False
    sb = sig.get("synB_at_nonsynA")
    if sb is not None and sig.get("nonsynA_subs", 0) >= min_nonsynA and sb < min_synB_at_nonsynA:
        return False
    return True


# ------------------------------------------------------------------- public API
def detect(alignment_path, reference=None, allow=(("+", 1), ("+", 2)),
           min_codons=25, max_stop_frac=0.10, min_subs=40):
    """Detect overlapping reading frames in a CDS alignment. Returns ranked hits."""
    seqs, ncols = read_alignment(alignment_path)
    if not seqs:
        return [], None
    ref = reference or ("hg38" if "hg38" in seqs else next(iter(seqs)))
    order = [ref] + [n for n in seqs if n != ref]
    allow = set(allow)
    hits = []
    for c in discover_alt_orfs(seqs, order, ref, ncols, allow, min_codons):
        if c["species_stop_frac"] > max_stop_frac:
            continue
        sig = signature(seqs, order, ref, ncols, (c["nt_start"], c["nt_end"]),
                        c["frame"], c["strand"], min_subs)
        hits.append({**c, **sig})
    hits.sort(key=lambda h: (not h["passes"], -h["syn_in_B"]))
    return hits, ref
