#!/usr/bin/env python3
"""
scripts/plot_avian_artifact_context_multipanel.py
-------------------------------------------------
Generates publication-quality Figure 2:
Multi-panel alignment context showing broken single-leaf sequence runs
against 38-species consensus and closely related phylogenetic sister taxa.
"""

import os
import sys
import zipfile
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Bio import Phylo

plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['axes.linewidth'] = 0.8

tree_path = 'benchmark/avian_shultz2019/avian_39_species_tree_clean.nwk'
tree = Phylo.read(tree_path, 'newick')

CODON_TO_AA = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*', 'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
}

SP_SHORT_NAMES = {
    'tinGut': 'T. guttatus (tinamou)',
    'cryCin': 'C. cinereus (tinamou)',
    'eudEle': 'E. elegans (tinamou)',
    'notPer': 'N. perdicarius (tinamou)',
    'galGal': 'G. gallus (chicken)',
    'melGal': 'M. gallopavo (turkey)',
    'anaPla': 'A. platyrhynchos (duck)',
    'taeGut': 'T. guttata (zebra finch)',
    'ficAlb': 'F. albicollis (flycatcher)',
    'pseHum': 'P. humilis (ground tit)',
    'corBra': 'C. brachyrhynchos (crow)',
    'calAnn': 'C. anna (hummingbird)',
    'chaPel': 'C. pelagica (swift)',
    'cucCan': 'C. canorus (cuckoo)'
}

def get_sister_taxa(tree, target_sp, max_taxa=3):
    target_node = next((t for t in tree.get_terminals() if t.name == target_sp), None)
    if not target_node: return []
    path = tree.get_path(target_node)
    collected = []
    for parent in reversed(path[:-1]):
        for leaf in parent.get_terminals():
            if leaf.name != target_sp and leaf.name not in collected:
                collected.append(leaf.name)
            if len(collected) >= max_taxa: break
        if len(collected) >= max_taxa: break
    return collected

def plot_multipanel_alignment_figure(out_path="hyphaeon_paper/figures/avian_artifact_alignment_context.pdf"):
    zf = zipfile.ZipFile('benchmark/avian_shultz2019/comparative_genomics_hog_alignments.zip', 'r')
    
    panels = [
        {'panel': 'A', 'hog': 720, 'gene': 'NCOA1', 'title': 'Nuclear Receptor Coactivator 1 (NCOA1)', 'outlier': 'tinGut', 'start': 345, 'end': 358, 'flank': 5},
        {'panel': 'B', 'hog': 9395, 'gene': 'ORC6', 'title': 'Origin Recognition Complex Subunit 6 (ORC6)', 'outlier': 'anaPla', 'start': 1, 'end': 26, 'flank': 5},
        {'panel': 'C', 'hog': 38072, 'gene': 'COIL', 'title': 'Coilin / Cajal Body Organizer (COIL)', 'outlier': 'eudEle', 'start': 43, 'end': 62, 'flank': 5},
        {'panel': 'D', 'hog': 5935, 'gene': 'ODF1', 'title': 'Outer Dense Fiber of Sperm Tails 1 (ODF1)', 'outlier': 'pseHum', 'start': 8, 'end': 39, 'flank': 5},
        {'panel': 'E', 'hog': 8541, 'gene': 'FASTKD5', 'title': 'FAST Kinase Domain-Containing 5 (FASTKD5)', 'outlier': 'galGal', 'start': 693, 'end': 721, 'flank': 5},
        {'panel': 'F', 'hog': 15914, 'gene': 'RHCE', 'title': 'Rh Blood Group Transporter (RHCE)', 'outlier': 'taeGut', 'start': 93, 'end': 106, 'flank': 5}
    ]
    
    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(13.5, 2.30 * n_panels), dpi=300)
    
    for p_idx, p_info in enumerate(panels):
        ax = axes[p_idx]
        hog = p_info['hog']
        outlier_sp = p_info['outlier']
        p_start = p_info['start'] - 1 # 0-indexed
        p_end = p_info['end'] - 1
        flank = p_info['flank']
        
        # Load alignment
        fname = f"compgen_alignments/{hog}.phy.gz"
        with zf.open(fname) as f:
            content = gzip.GzipFile(fileobj=f).read().decode('utf-8')
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        seqs = {}
        for l in lines[1:]:
            parts = l.split()
            if len(parts) >= 2:
                sp = parts[0].split('_')[0] if '_' in parts[0] else parts[0]
                seqs[sp] = ''.join(parts[1:]).upper()
                
        L_total = len(next(iter(seqs.values()))) // 3
        slice_start = max(0, p_start - flank)
        slice_end = min(L_total, p_end + flank + 1)
        slice_len = slice_end - slice_start
        
        # Compute consensus across all species
        consensus_aas = []
        for s in range(slice_start, slice_end):
            cds = [seqs[t][s*3:(s+1)*3] for t in seqs if len(seqs[t]) >= (s+1)*3]
            aas = [CODON_TO_AA.get(cd, '-') for cd in cds if '-' not in cd and 'N' not in cd and len(cd)==3]
            consensus_aas.append(pd.Series(aas).mode().iloc[0] if aas else '-')
            
        # Select taxa to show: Consensus, 3 sister taxa, Outlier
        sister_taxa = get_sister_taxa(tree, outlier_sp, max_taxa=3)
        sister_taxa = [s for s in sister_taxa if s in seqs]
        
        display_taxa = [('Consensus', '38-Sp. Consensus', consensus_aas)]
        for st in sister_taxa:
            st_aas = [CODON_TO_AA.get(seqs[st][s*3:(s+1)*3], '-') for s in range(slice_start, slice_end)]
            display_taxa.append((st, f"Sister: {SP_SHORT_NAMES.get(st, st)}", st_aas))
            
        outlier_aas = [CODON_TO_AA.get(seqs[outlier_sp][s*3:(s+1)*3], '-') for s in range(slice_start, slice_end)]
        display_taxa.append((outlier_sp, f"Outlier: {SP_SHORT_NAMES.get(outlier_sp, outlier_sp)}", outlier_aas))
        
        # Plot Setup: Left margin for right-aligned labels
        left_margin = -7.5
        ax.set_xlim(left_margin, max(slice_len, 12) + 0.8)
        ax.set_ylim(-len(display_taxa) * 0.92 + 0.1, 1.35)
        ax.axis('off')
        
        # Panel Title
        title_str = f"{p_info['panel']}. {p_info['title']} (HOG {hog}) | Codons {p_info['start']}–{p_info['end']} | Outlier: {outlier_sp}"
        ax.text(left_margin + 0.2, 0.95, title_str, fontsize=10.0, fontweight='bold', color='#1a252f')
        
        # Position Header
        for i, s in enumerate(range(slice_start, slice_end)):
            pos_num = s + 1
            if pos_num == 1 or pos_num % 5 == 0 or pos_num == p_info['start'] or pos_num == p_info['end']:
                ax.text(i + 0.5, 0.35, str(pos_num), fontsize=6.8, fontfamily='sans-serif', color='#7f8c8d', ha='center')
                ax.plot([i + 0.5, i + 0.5], [0.15, 0.22], color='#bdc3c7', lw=0.6)
                
        # Draw each taxon row
        for r_idx, (tax_id, tax_label, tax_aas) in enumerate(display_taxa):
            y_pos = -r_idx * 0.88
            is_outlier = (tax_id == outlier_sp)
            is_consensus = (tax_id == 'Consensus')
            
            # Row label (Right-aligned at x = -0.4 to avoid overlapping sequence grid)
            label_col = '#922b21' if is_outlier else '#1b4f72' if is_consensus else '#2c3e50'
            label_fontweight = 'bold' if is_outlier or is_consensus else 'normal'
            ax.text(-0.4, y_pos, tax_label, fontsize=7.8, fontfamily='monospace', color=label_col, fontweight=label_fontweight, ha='right', va='center')
            
            # Draw residues
            for c_idx in range(slice_len):
                aa = tax_aas[c_idx] if c_idx < len(tax_aas) else '-'
                con_aa = consensus_aas[c_idx] if c_idx < len(consensus_aas) else '-'
                abs_pos = slice_start + c_idx # 0-indexed codon pos
                in_patch = (p_start <= abs_pos <= p_end)
                is_mut = (aa != con_aa and aa not in '-?' and con_aa not in '-?')
                
                # Determine styling
                if is_outlier:
                    if in_patch and is_mut:
                        # Highlighted broken mutation
                        box_col = '#fadbd8'
                        text_col = '#78281f'
                        fw = 'bold'
                        rect = patches.Rectangle((c_idx + 0.1, y_pos - 0.30), 0.8, 0.60, facecolor=box_col, edgecolor='#e6b0aa', lw=0.5, alpha=0.95)
                        ax.add_patch(rect)
                    elif not is_mut:
                        text_col = '#566573'
                        fw = 'normal'
                    else:
                        text_col = '#922b21'
                        fw = 'normal'
                elif is_consensus:
                    text_col = '#1b4f72'
                    fw = 'bold'
                else: # Sister taxon
                    if is_mut:
                        text_col = '#2e4053'
                        fw = 'bold'
                        rect = patches.Rectangle((c_idx + 0.12, y_pos - 0.28), 0.76, 0.56, facecolor='#eaecee', edgecolor='none', alpha=0.7)
                        ax.add_patch(rect)
                    else:
                        text_col = '#5d6d7e'
                        fw = 'normal'
                        
                ax.text(c_idx + 0.5, y_pos, aa, fontsize=8.0, fontfamily='monospace', color=text_col, fontweight=fw, ha='center', va='center')
                
        # Separator line
        ax.plot([left_margin + 0.2, slice_len + 0.5], [-(len(display_taxa)-0.5)*0.88, -(len(display_taxa)-0.5)*0.88], color='#eaeded', lw=0.8)
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"[✓] Multi-panel alignment figure saved to: {out_path} and .png")

if __name__ == '__main__':
    plot_multipanel_alignment_figure()
