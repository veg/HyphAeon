"""
_sim.py — Neutral alignment simulation via seq-gen for calibration tests.

Uses seq-gen (installed via bioconda) with an HKY nucleotide model under
neutrality (equal frequencies, no selection). This is a well-established
simulation tool — we don't need to validate the simulated data ourselves.

For null calibration, what matters is that the data has NO positive selection
signal. Neutral nucleotide evolution satisfies this by construction. The
simulated DNA is treated as codons (length always divisible by 3) by the
HyphAeon pipeline.

Trees are generated as random coalescent trees with controlled depth.
"""
import os
import subprocess
import numpy as np
from io import StringIO
from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree


def _build_random_tree(n_taxa, max_depth, rng):
    """Build a random binary tree by iterative coalescence.

    Branch lengths drawn from Exponential(max_depth * 0.3) to produce
    trees with controlled patristic depth.
    """
    taxa_names = [f"taxon_{i}" for i in range(n_taxa)]
    tips = [Clade(branch_length=float(rng.exponential(max_depth * 0.3)),
                  name=name) for name in taxa_names]
    while len(tips) > 1:
        i, j = rng.choice(len(tips), size=2, replace=False)
        a, b = tips[int(i)], tips[int(j)]
        remaining = [t for k, t in enumerate(tips) if k not in (int(i), int(j))]
        parent = Clade(
            branch_length=float(rng.exponential(max_depth * 0.3)),
            clades=[a, b],
        )
        tips = remaining + [parent]
    return Tree(root=tips[0])


def simulate_neutral_alignment(n_taxa, n_codons, tree_depth=0.1,
                               scale=1.0, kappa=2.0,
                               freqs=(0.25, 0.25, 0.25, 0.25),
                               seed=0, out_dir=None):
    """Simulate a neutral codon alignment using seq-gen.

    Parameters:
        n_taxa: number of taxa
        n_codons: number of codon sites (sequence length = 3 * n_codons)
        tree_depth: controls branch length distribution
        scale: seq-gen branch length scaling factor (-s)
        kappa: transition/transversion ratio (-t)
        freqs: (A, C, G, T) equilibrium frequencies for seq-gen -f. Default
            is uniform. Pass a biased tuple (e.g. AT-rich or GC-rich) to
            simulate under non-uniform composition — see
            test_composition_bias.py.
        seed: random seed (used for both tree generation and seq-gen -z)
        out_dir: directory for output files (default:
            /tmp/hyphaeon_sim_<n_taxa>_<n_codons>_<depth>_<seed>_<freqs>)

    Returns:
        (fasta_path, newick_path)

    The simulation uses HKY with no dN/dS structure, which is neutral by
    construction regardless of the chosen base frequencies.
    """
    rng = np.random.default_rng(seed)

    if out_dir is None:
        # Include key parameters in the directory name to avoid collisions
        # when different tests use the same seed with different params
        # (e.g. test_hyphaeon_null seed=0 n_taxa=20 vs test_composition_bias
        # seed=0 n_taxa=50 freqs=at_rich).
        freq_tag = "".join(f"{int(f*10)}" for f in freqs)
        out_dir = os.path.join(
            "/tmp", f"hyphaeon_sim_{n_taxa}_{n_codons}_{tree_depth}_{seed}_{freq_tag}")
    os.makedirs(out_dir, exist_ok=True)

    # Build random tree
    tree = _build_random_tree(n_taxa, tree_depth, rng)
    nwk_path = os.path.join(out_dir, "tree.nwk")
    out = StringIO()
    Phylo.write(tree, out, "newick")
    with open(nwk_path, "w") as f:
        f.write(out.getvalue())

    # Run seq-gen: HKY, relaxed PHYLIP to stdout
    seq_len = n_codons * 3
    freq_str = ",".join(f"{f:.4f}" for f in freqs)
    cmd = [
        "seq-gen",
        "-mHKY",
        f"-l{seq_len}",
        "-or",           # relaxed PHYLIP to stdout
        f"-s{scale}",
        f"-f{freq_str}",
        f"-t{kappa}",
        f"-z{seed}",
        "-q",             # quiet
        nwk_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"seq-gen failed: {result.stderr}")

    # Parse relaxed PHYLIP output and convert to FASTA
    lines = result.stdout.strip().splitlines()
    # First line: "N_taxa N_chars"
    if not lines or len(lines) < 2:
        raise RuntimeError(f"seq-gen produced no output: {result.stdout[:200]}")

    fasta_path = os.path.join(out_dir, "neutral.fasta")
    with open(fasta_path, "w") as f:
        for line in lines[1:]:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                name, seq = parts
                f.write(f">{name}\n{seq}\n")

    return fasta_path, nwk_path


def simulate_neutral_alignment_batch(n_alignments, n_taxa, n_codons,
                                     tree_depth=0.1, scale=1.0, seed=0):
    """Generate multiple neutral alignments with different seeds.

    Returns list of (fasta_path, newick_path) tuples.
    """
    paths = []
    for i in range(n_alignments):
        fa, nwk = simulate_neutral_alignment(
            n_taxa, n_codons,
            tree_depth=tree_depth, scale=scale,
            seed=seed * 1000 + i)
        paths.append((fa, nwk))
    return paths


def purge_stop_codons(fa_path, seed=0):
    """Replace in-frame stop codons with random sense codons.

    seq-gen simulates raw nucleotide substitution with no codon-level
    constraints, so in-frame stop codons appear at the rate expected under
    neutral nucleotide evolution. HyphAeon tolerates this (it just notes the
    count), but real biological coding sequences never have in-frame stops
    (purifying selection removes them), and HyPhy MEME hard-rejects any
    alignment containing one. This purges stops so a simulated alignment can
    be run through real MEME for concordance comparison.

    Returns a new fasta path (does not modify the input file).
    """
    from Bio import SeqIO

    STOP_CODONS = {"TAA", "TAG", "TGA"}
    SENSE_CODONS = [a + b + c
                    for a in "TCAG" for b in "TCAG" for c in "TCAG"
                    if a + b + c not in STOP_CODONS]

    rng = np.random.default_rng(seed)
    recs = list(SeqIO.parse(fa_path, "fasta"))
    seqs = {r.id: list(str(r.seq)) for r in recs}
    taxa = list(seqs.keys())

    seq_len = len(seqs[taxa[0]])
    n_purged = 0
    for site_start in range(0, seq_len - 2, 3):
        for taxon in taxa:
            codon = "".join(seqs[taxon][site_start:site_start + 3])
            if codon.upper() in STOP_CODONS:
                new_codon = SENSE_CODONS[rng.integers(0, len(SENSE_CODONS))]
                for i, base in enumerate(new_codon):
                    seqs[taxon][site_start + i] = base
                n_purged += 1

    out_dir = os.path.dirname(fa_path)
    out_path = os.path.join(out_dir, "no_stops.fasta")
    with open(out_path, "w") as f:
        for taxon in taxa:
            f.write(f">{taxon}\n{''.join(seqs[taxon])}\n")

    return out_path, n_purged


def inject_selection(fa_path, nwk_path, n_taxa, n_codons,
                     n_selected_sites, n_selected_branches, seed=0):
    """Inject episodic positive selection into a neutral alignment.

    Replaces codons at selected sites on a clade of taxa (sharing a common
    ancestor in the tree) with a radical amino acid change. This models
    episodic selection on an ancestral branch — the signal MEME's branch-site
    test is designed to detect — rather than scattering independent radical
    changes on random taxa, which does not match either tool's expected input.

    The clade is chosen as the subtree under a random internal node whose
    terminal descendant count is closest to n_selected_branches. All
    terminal taxa in that clade receive the radical substitution at each
    selected site, producing a phylogenetically correlated signal.

    Returns (modified_fasta_path, set_of_selected_site_indices,
             n_selected_taxa_actual, list_of_selected_taxa_names).
    """
    from Bio import SeqIO
    from Bio import Phylo
    import os

    rng = np.random.default_rng(seed)
    recs = list(SeqIO.parse(fa_path, "fasta"))
    seqs = {r.id: list(str(r.seq)) for r in recs}
    taxa = list(seqs.keys())
    taxa_set = set(taxa)

    # Parse the tree and find a clade with ~n_selected_branches terminals.
    tree = Phylo.read(nwk_path, "newick")
    # Collect all internal clades and their terminal descendant names.
    candidates = []
    for clade in tree.find_clades():
        if not clade.is_terminal():
            terms = [t.name.strip("'\"") for t in clade.get_terminals()
                     if t.name and t.name.strip("'\"") in taxa_set]
            if len(terms) >= 2:
                candidates.append((abs(len(terms) - n_selected_branches), terms))
    if not candidates:
        # Fallback: if the tree is a star or too shallow, use random taxa.
        # This preserves the function's contract but the signal will be
        # scattered rather than clade-concentrated.
        selected_taxa = list(rng.choice(taxa, size=min(n_selected_branches, len(taxa)),
                                         replace=False))
    else:
        candidates.sort(key=lambda x: x[0])
        selected_taxa = candidates[0][1]

    # Radical AA change: small/hydrophobic -> large aromatic.
    # TGG (Trp) is the largest amino acid; AGG (Arg) is the alternative
    # when the original is already Trp. This is a deliberately extreme
    # non-synonymous change to create an unambiguous selection signal.
    RADICAL_TARGET = "TGG"
    RADICAL_ALT = "AGG"

    selected_sites = set(rng.choice(n_codons, size=n_selected_sites, replace=False))

    for site_idx in selected_sites:
        codon_start = site_idx * 3
        for taxon in selected_taxa:
            orig = "".join(seqs[taxon][codon_start:codon_start + 3]).upper()
            new_codon = RADICAL_ALT if orig == RADICAL_TARGET else RADICAL_TARGET
            for i, base in enumerate(new_codon):
                seqs[taxon][codon_start + i] = base

    # Write modified alignment
    out_dir = os.path.dirname(fa_path)
    mod_path = os.path.join(out_dir, "with_selection.fasta")
    with open(mod_path, "w") as f:
        for taxon in taxa:
            f.write(f">{taxon}\n{''.join(seqs[taxon])}\n")

    return mod_path, selected_sites, len(selected_taxa), selected_taxa
