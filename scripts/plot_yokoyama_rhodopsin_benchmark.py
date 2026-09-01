import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

# Set publication style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0

# 1. Load data
yoko_df = pd.read_csv('/Users/sergei/Projects/TOGA_MEME/yoko.csv')
yoko_df = yoko_df.sort_values('site').reset_index(drop=True)

sites = yoko_df['site'].values
rhos = yoko_df['association_rho'].values

known_spectral = {
    277: ('T277C', -9, 'blue'),
    278: ('H278N', -5, 'blue'),
    292: ('A292S', -12, 'blue'),
    261: ('F261Y', 10, 'red'),
    122: ('E122I', -14, 'blue'),
    83:  ('D83N', -4, 'blue'),
    198: ('F198Y', -3, 'blue'),
    260: ('G260S', -3, 'blue'),
    164: ('A164S', -2, 'blue'),
    117: ('A117S', -2, 'blue'),
    262: ('Y262F', -2, 'blue'),
    295: ('S295A', -1, 'blue'),
    269: ('A269T', -2, 'blue')
}

novel_allosteric = {
    264: ('C264 (Rk #1)', 0.654),
    185: ('C185 (Rk #2)', 0.578),
    210: ('C210V (Rk #3)', 0.472),
    119: ('L119H (Rk #6)', 0.654),
    35:  ('A35V (Rk #7)', 0.605),
    165: ('L165 (Rk #5)', 0.397),
    136: ('W136', 0.440),
    259: ('I259', 0.614)
}

fig = plt.figure(figsize=(14, 9.5))
gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.0], hspace=0.35)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# Panel A
ax1.plot(sites, rhos, color='#94a3b8', alpha=0.75, linewidth=1.2, label='Neutral Background Attribution')
ax1.axhline(0.40, color='#dc2626', linestyle='--', linewidth=1.2, alpha=0.85, label=r'FDR Significance Threshold ($\rho \geq 0.40, q \leq 0.05$)')
ax1.axhline(0.00, color='#64748b', linestyle=':', linewidth=0.8, alpha=0.6)

for s, (label, shift, color_type) in known_spectral.items():
    if s in yoko_df['site'].values:
        r_val = yoko_df.loc[yoko_df['site'] == s, 'association_rho'].values[0]
        shift_str = f'+{shift}nm' if shift > 0 else f'{shift}nm'
        txt_label = f'{label} ({shift_str})'
        c = '#2563eb' if color_type == 'blue' else '#ea580c'
        ax1.scatter(s, r_val, s=70, facecolor=c, edgecolor='black', linewidth=1.2, zorder=5)
        y_offset = 0.06 if s not in [261, 278, 117] else -0.10
        if s in [277, 264]:
            y_offset = 0.08
        elif s == 278:
            y_offset = 0.14
        elif s == 292:
            y_offset = 0.06
        elif s == 122:
            y_offset = -0.09
        ax1.annotate(txt_label, xy=(s, r_val), xytext=(s, r_val + y_offset),
                     fontsize=8, fontweight='bold', color=c,
                     ha='center', va='bottom' if y_offset > 0 else 'top',
                     arrowprops=dict(arrowstyle='->', color=c, alpha=0.7, lw=0.8))

for s, (label, r_val) in novel_allosteric.items():
    if s in yoko_df['site'].values and s not in known_spectral:
        r_val_actual = yoko_df.loc[yoko_df['site'] == s, 'association_rho'].values[0]
        ax1.scatter(s, r_val_actual, s=75, facecolor='#d97706', edgecolor='black', linewidth=1.2, zorder=5)
        y_offset = 0.08 if s not in [210] else -0.09
        if s == 264:
            y_offset = 0.16
        elif s == 35:
            y_offset = 0.08
        ax1.annotate(label, xy=(s, r_val_actual), xytext=(s, r_val_actual + y_offset),
                     fontsize=8, fontweight='bold', color='#b45309',
                     ha='center', va='bottom' if y_offset > 0 else 'top',
                     arrowprops=dict(arrowstyle='->', color='#b45309', alpha=0.8, lw=0.8))

ax1.set_xlim(1, 330)
ax1.set_ylim(-0.25, 1.15)
ax1.set_xlabel('Rhodopsin (RH1) Codon Position (1 to 330)', fontweight='bold')
ax1.set_ylabel(r'Trait Directional Concordance ($\rho_s$)', fontweight='bold')
ax1.set_title('(A) HyphAeon Directional Attribution Track Across 38 Vertebrate Rhodopsins', fontweight='bold', loc='left', pad=12)

legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2563eb', markeredgecolor='black', markersize=8, label=r'Yokoyama et al. (2008) Resurrected Spectral Mutants (In Vitro Verified $\Delta\lambda$)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#d97706', markeredgecolor='black', markersize=8, label='HyphAeon Discovered Pressure & Thermal Allosteric Residues (Novel Discoveries)'),
    plt.Line2D([0], [0], color='#dc2626', linestyle='--', linewidth=1.2, label=r'FDR Significance Threshold ($\rho \geq 0.40, q \leq 0.05$)'),
    plt.Line2D([0], [0], color='#94a3b8', linewidth=1.2, label='Neutral Background Attribution')
]
ax1.legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=8.5)

# Panel B
ax2.set_xlim(-1, 15)
ax2.set_ylim(-1, 9)
ax2.axis('off')
ax2.set_title('(B) Discovered Epistatic Sectors Across Dim-Light Trait Sites', fontweight='bold', loc='left', pad=12)

poly_s2 = plt.Polygon([[8.5, 4.0], [13.8, 5.0], [14.0, 7.8], [9.5, 7.8], [7.5, 5.5]], 
                      closed=True, color='#fee2e2', ec='#ef4444', lw=1.5, ls='--', alpha=0.6, zorder=1)
ax2.add_patch(poly_s2)

poly_s1 = plt.Polygon([[0.5, 2.5], [3.5, 7.5], [6.5, 6.0], [7.5, 1.5], [4.5, -0.5], [1.5, 0.5]],
                      closed=True, color='#f1f5f9', ec='#334155', lw=1.5, ls='--', alpha=0.6, zorder=1)
ax2.add_patch(poly_s1)

nodes = {
    'C264': (10.0, 7.2, '#ef4444', 'white'),
    'T277': (13.2, 6.8, '#ef4444', 'white'),
    'A292': (12.2, 4.8, '#ef4444', 'white'),
    'C185': (9.2, 5.5, '#ef4444', 'white'),
    'L119': (8.2, 4.8, '#ef4444', 'white'),
    'I259': (11.0, 3.8, '#ef4444', 'white'),
    'C210': (9.5, 2.2, '#f97316', 'white'),
    'E122': (12.2, 1.2, '#f97316', 'white'),
    
    'F261': (1.8, 6.5, '#1e293b', 'white'),
    'G260': (4.2, 5.5, '#1e293b', 'white'),
    'F198': (1.5, 4.8, '#1e293b', 'white'),
    'A236': (3.0, 3.5, '#1e293b', 'white'),
    'C308': (5.0, 2.2, '#1e293b', 'white'),
    'W136': (7.0, 1.2, '#1e293b', 'white'),
    'A35':  (5.5, -0.2, '#1e293b', 'white')
}

edges = [
    ('C264', 'T277', 2.85),
    ('C264', 'C185', 3.94),
    ('C185', 'A292', 2.91),
    ('C185', 'L119', 2.98),
    ('A292', 'T277', 3.12),
    ('A292', 'I259', 2.75),
    ('C210', 'C185', 2.45),
    ('C210', 'E122', 2.25),
    
    ('F261', 'G260', 3.80),
    ('F261', 'F198', 3.20),
    ('F198', 'A236', 2.54),
    ('G260', 'A236', 2.90),
    ('A236', 'C308', 4.49),
    ('C308', 'W136', 2.95),
    ('C308', 'A35',  2.60)
]

for n1, n2, w in edges:
    x1, y1, _, _ = nodes[n1]
    x2, y2, _, _ = nodes[n2]
    ax2.plot([x1, x2], [y1, y2], color='#64748b', lw=w*0.9, alpha=0.75, zorder=2)
    mx, my = (x1 + x2)/2, (y1 + y2)/2
    ax2.text(mx, my, f'{w:.2f}', fontsize=7.5, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#cbd5e1', lw=0.6), zorder=3)

for name, (x, y, bg_c, txt_c) in nodes.items():
    circle = plt.Circle((x, y), 0.52, facecolor=bg_c, edgecolor='black', lw=1.2, zorder=4)
    ax2.add_patch(circle)
    ax2.text(x, y, name, color=txt_c, fontsize=8.5, fontweight='bold', ha='center', va='center', zorder=5)

sec_legend_elements = [
    patches.Patch(facecolor='#fee2e2', edgecolor='#ef4444', linestyle='--', label=r'Sector 2: Schiff Base Electronic Tuning Cluster ($C(\mathcal{S}) = 0.582$)'),
    patches.Patch(facecolor='#f1f5f9', edgecolor='#334155', linestyle='--', label=r'Sector 1: Extracellular Cap & Pocket Roof ($C(\mathcal{S}) = 0.630$)'),
    patches.Patch(facecolor='#f97316', edgecolor='black', label=r'Sector 4 / Bridge: Hydrophobic Pressure Clamp ($C(\mathcal{S}) = 0.655$)')
]
ax2.legend(handles=sec_legend_elements, loc='upper left', frameon=True, fontsize=8.5)

out_pdf = '/Users/sergei/Projects/TOGA_MEME/hyphaeon_paper/figures/yokoyama_rhodopsin_benchmark.pdf'
out_png = '/Users/sergei/Projects/TOGA_MEME/hyphaeon_paper/figures/yokoyama_rhodopsin_benchmark.png'

plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
plt.savefig(out_png, bbox_inches='tight', dpi=300)
plt.close()
print('Successfully saved 2-panel figure to PDF and PNG!')
