"""
chromosomal_map.py
==================
Chromosomal polar coordinate engine and biophysical feature extraction.
Maps recombination breakpoints against circular bacterial chromosome architecture:
- Replication polar coordinates (distance to oriC and ter)
- Replication fork polarity (GC skew, leading vs lagging strand)
- Species-specific Chi (χ) octamer density landscapes
- Attachment sites and tRNA integration anchors
"""

import math
from typing import Dict, List, Tuple, Optional
import numpy as np


# Species-specific Chi octamer motifs (forward and reverse complement)
SPECIES_CHI_MOTIFS: Dict[str, List[str]] = {
    "Streptococcus pneumoniae": ["GAGAATGA", "TCATTCTC"],
    "Escherichia coli": ["GCTGGTGG", "CCACCAGC"],
    "Salmonella enterica": ["GCTGGTGG", "CCACCAGC"],
    "Staphylococcus aureus": ["GAAGCGG", "CCGCTTC"],
    "Bacillus subtilis": ["AGCGG", "CCGCT"],
    "Pseudomonas aeruginosa": ["GCTGGTGG", "CCACCAGC"],
    "Mycobacterium tuberculosis": ["GCTGGTGG", "CCACCAGC"]
}


class ChromosomalCoordinates:
    """Manages circular chromosome polar coordinates relative to oriC and ter."""

    def __init__(self, chromosome_length: int, ori_pos: int = 0, ter_pos: Optional[int] = None):
        self.length = chromosome_length
        self.ori_pos = ori_pos % chromosome_length
        # Terminus is typically ~180 degrees opposite the origin
        if ter_pos is None:
            self.ter_pos = (self.ori_pos + chromosome_length // 2) % chromosome_length
        else:
            self.ter_pos = ter_pos % chromosome_length

    def polar_angle_rad(self, pos: int) -> float:
        """Returns replication angle in radians [0, 2π) where oriC is 0."""
        offset = (pos - self.ori_pos) % self.length
        return 2.0 * math.pi * (offset / self.length)

    def dist_to_origin_bp(self, pos: int) -> int:
        """Returns shortest circular distance to the replication origin."""
        diff = abs(pos - self.ori_pos)
        return min(diff, self.length - diff)

    def dist_to_terminus_bp(self, pos: int) -> int:
        """Returns shortest circular distance to the replication terminus."""
        diff = abs(pos - self.ter_pos)
        return min(diff, self.length - diff)

    def is_leading_strand(self, pos: int, strand: int = +1) -> bool:
        """
        Determines whether a gene at pos on strand (+1/-1) is codirectional
        with the replication fork (leading) or head-on (lagging).
        """
        # Right arm of replication (oriC -> ter clockwise)
        if self.ori_pos <= self.ter_pos:
            in_right_arm = self.ori_pos <= pos <= self.ter_pos
        else:
            in_right_arm = (pos >= self.ori_pos) or (pos <= self.ter_pos)

        if in_right_arm:
            # Fork moves forward (+1)
            return strand == +1
        else:
            # Left arm: fork moves backward (-1)
            return strand == -1


def compute_gc_skew(
    sequence: str,
    window_bp: int = 10000,
    step_bp: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes cumulative or sliding-window GC skew (G - C) / (G + C)
    across the circular bacterial chromosome.
    
    Returns:
        (coordinates, skew_values)
    """
    seq_upper = sequence.upper()
    L = len(seq_upper)
    coords = np.arange(0, L, step_bp)
    skews = np.zeros(len(coords), dtype=float)

    for idx, c in enumerate(coords):
        half = window_bp // 2
        # Handle circular wrap-around
        if c - half < 0:
            subseq = seq_upper[c - half :] + seq_upper[: c + half]
        elif c + half > L:
            subseq = seq_upper[c - half :] + seq_upper[: (c + half) % L]
        else:
            subseq = seq_upper[c - half : c + half]

        g = subseq.count("G")
        c_count = subseq.count("C")
        total = g + c_count
        skews[idx] = (g - c_count) / total if total > 0 else 0.0

    return coords, skews


def compute_chi_density(
    sequence: str,
    species: str = "Streptococcus pneumoniae",
    bandwidth_bp: int = 25000,
    grid_step_bp: int = 5000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scans for Chi octamer instances and calculates a Gaussian kernel density
    field D_chi(s) along the circular chromosome.
    
    Returns:
        (coordinates, normalized_density_values)
    """
    seq_upper = sequence.upper()
    L = len(seq_upper)
    motifs = SPECIES_CHI_MOTIFS.get(species, ["GAGAATGA", "TCATTCTC"])

    hit_positions: List[int] = []
    for motif in motifs:
        start = 0
        while True:
            pos = seq_upper.find(motif, start)
            if pos == -1:
                break
            hit_positions.append(pos)
            start = pos + 1

    coords = np.arange(0, L, grid_step_bp)
    density = np.zeros(len(coords), dtype=float)

    if not hit_positions:
        return coords, density

    hit_arr = np.array(hit_positions)
    sigma2 = 2.0 * (bandwidth_bp ** 2)

    for idx, c in enumerate(coords):
        # Circular distances
        diffs = np.abs(hit_arr - c)
        circ_diffs = np.minimum(diffs, L - diffs)
        # Gaussian kernel sum
        density[idx] = np.sum(np.exp(-(circ_diffs ** 2) / sigma2))

    # Normalize to mean 1.0
    mean_d = np.mean(density)
    if mean_d > 0:
        density /= mean_d

    return coords, density
