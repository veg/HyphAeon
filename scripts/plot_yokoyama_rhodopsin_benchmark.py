import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

# Set publication style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.0

# 1. Load data
yoko_df = pd.read_csv('/Users/sergei/Projects/TOGA_MEME/yoko.csv')
yoko_df = yoko_df.sort_values('site').reset_index(drop=True)

sites = yoko_df['site'].values
rhos = yoko_df['association_rho'].values
qvals = yoko_df['q_value'].values

# Sectors definition from modularity clustering on trait co-selection graph
s1_sites = {83, 260, 198, 308, 136, 35, 33, 300}
s2_sites = {292, 185, 119, 165, 144, 145, 137, 157}
s3_sites = {264, 277, 186, 100, 75, 51, 38}
s4_sites = {122, 210, 259, 213, 14, 8, 290}

# Confirmed experimental mutations altering wavelength (Yokoyama et al. 2008)
known_spectral = [
    (83,  'D83N',  '-4nm',  '#1d4ed8'),
    (122, 'E122I', '-14nm', '#1d4ed8'),
    (164, 'A164S', '-2nm',  '#1d4ed8'),
    (198, 'F198Y', '-3nm',  '#1d4ed8'),
    (260, 'G260S', '-3nm',  '#1d4ed8'),
    (261, 'F261Y', '+10nm', '#b91c1c'),
    (277, 'T277C', '-9nm',  '#1d4ed8'),
    (278, 'H278N', '-5nm',  '#1d4ed8'),
    (292, 'A292S', '-12nm', '#1d4ed8'),
]

# Create figure
fig = plt.figure(figsize=(15.5, 11.2), dpi=300)
gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 0.30, 1.15], hspace=0.22)

ax1 = fig.add_subplot(gs[0])
ax_gene = fig.add_subplot(gs[1], sharex=ax1)
ax2 = fig.add_subplot(gs[2])

# =============================================================
# Panel A: Impulse Plot of Trait Directional Concordance
# =============================================================
fdr_threshold = 0.40

# Masks
mask_s1 = np.isin(sites, list(s1_sites))
mask_s2 = np.isin(sites, list(s2_sites))
mask_s3 = np.isin(sites, list(s3_sites))
mask_s4 = np.isin(sites, list(s4_sites))
in_any_sector = mask_s1 | mask_s2 | mask_s3 | mask_s4

sig_unclustered = (qvals <= 0.05) & (rhos >= fdr_threshold) & (~in_any_sector)
background = ~in_any_sector & ~sig_unclustered

# 1. Background
ax1.vlines(sites[background], 0, rhos[background], color='#cbd5e1', lw=0.9, alpha=0.65)
ax1.scatter(sites[background], rhos[background], s=14, color='#94a3b8', alpha=0.5,
            edgecolor='none', label=f'Background Sites (q > 0.05, N = {sum(background)})')

# 2. Unclustered significant
ax1.vlines(sites[sig_unclustered], 0, rhos[sig_unclustered], color='#a855f7', lw=1.2, alpha=0.85)
ax1.scatter(sites[sig_unclustered], rhos[sig_unclustered], s=32, facecolor='#c084fc', edgecolor='#7e22ce', lw=0.8,
            zorder=4, label=f'Unclustered Significant Sites (q \u2264 0.05, N = {sum(sig_unclustered)})')

# 3. Sector 1 (Green)
ax1.vlines(sites[mask_s1], 0, rhos[mask_s1], color='#16a34a', lw=1.5, alpha=0.95)
ax1.scatter(sites[mask_s1], rhos[mask_s1], s=42, facecolor='#22c55e', edgecolor='#14532d', lw=1.0,
            zorder=5, label=f'Sector 1: Extracellular Cap (N = {sum(mask_s1)})')

# 4. Sector 2 (Blue)
ax1.vlines(sites[mask_s2], 0, rhos[mask_s2], color='#2563eb', lw=1.5, alpha=0.95)
ax1.scatter(sites[mask_s2], rhos[mask_s2], s=42, facecolor='#3b82f6', edgecolor='#1e3a8a', lw=1.0,
            zorder=5, label=f'Sector 2: Schiff Base Core (N = {sum(mask_s2)})')

# 5. Sector 3 (Red)
ax1.vlines(sites[mask_s3], 0, rhos[mask_s3], color='#dc2626', lw=1.5, alpha=0.95)
ax1.scatter(sites[mask_s3], rhos[mask_s3], s=42, facecolor='#ef4444', edgecolor='#7f1d1d', lw=1.0,
            zorder=5, label=f'Sector 3: Schiff Base Ceiling (N = {sum(mask_s3)})')

# 6. Sector 4 (Orange)
ax1.vlines(sites[mask_s4], 0, rhos[mask_s4], color='#ea580c', lw=1.5, alpha=0.95)
ax1.scatter(sites[mask_s4], rhos[mask_s4], s=42, facecolor='#f97316', edgecolor='#7c2d12', lw=1.0,
            zorder=5, label=f'Sector 4: Pressure Clamp (N = {sum(mask_s4)})')

# FDR significance cutoff
ax1.axhline(fdr_threshold, color='#dc2626', linestyle='--', linewidth=1.1, alpha=0.85,
            label=r'FDR Significance Cutoff ($\rho_s \geq 0.40, q \leq 0.05$)')
ax1.axhline(0.00, color='#64748b', linestyle='-', linewidth=0.8, alpha=0.5)

ax1.set_xlim(1, 330)
ax1.set_ylim(-0.06, 1.02)
ax1.set_ylabel(r'Trait Directional Concordance ($\rho_s$)', fontweight='bold', fontsize=10.5)
ax1.set_title('(A) Trait Directional Concordance Across 38 Vertebrate Rhodopsins (Color-Coded by Sector Membership)',
              fontweight='bold', loc='left', pad=10, fontsize=11.5)
ax1.legend(loc='upper left', frameon=True, fontsize=8.2, framealpha=0.95, ncol=2)
plt.setp(ax1.get_xticklabels(), visible=False)
ax1.tick_params(direction='out', length=4, width=1.0)

# =============================================================
# Track: Rhodopsin Gene Schematic & Experimental Annotations
# =============================================================
ax_gene.set_ylim(-1.35, 0.95)
ax_gene.set_xlim(1, 330)
ax_gene.axis('off')

bar_y = 0.50
bar_h = 0.32
ax_gene.add_patch(patches.Rectangle((1, bar_y), 329, bar_h,
                                    facecolor='#f1f5f9', edgecolor='#94a3b8', lw=0.9, zorder=2))

tm_domains = [
    (35, 64, 'TM-I'),
    (73, 99, 'TM-II'),
    (107, 139, 'TM-III'),
    (152, 173, 'TM-IV'),
    (200, 225, 'TM-V'),
    (246, 277, 'TM-VI'),
    (286, 309, 'TM-VII')
]

for start, end, label in tm_domains:
    rect = patches.Rectangle((start, bar_y), end - start, bar_h,
                             facecolor='#334155', edgecolor='#0f172a', lw=1.0, zorder=3)
    ax_gene.add_patch(rect)
    ax_gene.text((start + end) / 2, bar_y + bar_h/2, label, fontsize=6.8, fontweight='bold',
                 color='white', ha='center', va='center', zorder=4)

ax_gene.text(17, bar_y + bar_h/2, 'N-term', fontsize=6.5, color='#64748b', ha='center', va='center', zorder=3)
ax_gene.text(320, bar_y + bar_h/2, 'C-term', fontsize=6.5, color='#64748b', ha='center', va='center', zorder=3)

ax_gene.text(-4, bar_y + bar_h/2, 'Gene Model:\n(Units)', fontsize=8.0, fontweight='bold',
             color='#1e293b', ha='right', va='center')

# Stagger tiers for experimental mutation callouts (Yokoyama et al. 2008)
tiers = {
    83:  -0.32,
    122: -0.32,
    164: -0.32,
    198: -0.32,
    260: -0.25,
    261: -0.80,
    277: -0.25,
    278: -0.80,
    292: -0.32,
}

for site, aa_mut, shift, c in known_spectral:
    ax_gene.plot([site, site], [bar_y, bar_y + bar_h], color='#f59e0b', lw=2.2, zorder=5)
    ax_gene.scatter([site], [bar_y], s=25, facecolor=c, edgecolor='white', lw=0.9, zorder=6)
    
    target_y = tiers.get(site, -0.35)
    ax_gene.plot([site, site], [bar_y, target_y + 0.12], color=c, lw=0.9, ls=':', zorder=4)
    
    badge_str = f'{aa_mut}\n({shift})'
    ax_gene.text(site, target_y, badge_str, fontsize=7.2, fontweight='bold', color=c,
                 ha='center', va='top',
                 bbox=dict(boxstyle='round,pad=0.20', facecolor='white', edgecolor=c, lw=0.8, alpha=0.95),
                 zorder=7)

ticks = [1, 50, 100, 150, 200, 250, 300, 330]
for t in ticks:
    ax_gene.plot([t, t], [bar_y + bar_h, bar_y + bar_h + 0.06], color='#334155', lw=1.0)
    ax_gene.text(t, bar_y + bar_h + 0.10, str(t), fontsize=7.0, color='#334155', ha='center', va='bottom')
ax_gene.text(165, bar_y + bar_h + 0.28, 'Codon Position along Rhodopsin (RH1)', fontsize=8.8,
             fontweight='bold', color='#1e293b', ha='center', va='bottom')

# =============================================================
# Panel B: Clean 4-Sector Architecture
# =============================================================
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 100)
ax2.axis('off')
ax2.set_title(r'(B) Four Epistatic Sectors Extracted via Modularity Mining',
              fontweight='bold', loc='left', pad=12, fontsize=11.5)

sector_cards = [
    {
        'id': 'Sector 1: Extracellular Cap',
        'color': '#f0fdf4',
        'ec': '#16a34a',
        'header_c': '#166534',
        'box': [1, 4, 23, 91],
        'nodes': {
            'D83':  (7, 72, '#16a34a', 'white'),
            'G260': (18, 72, '#16a34a', 'white'),
            'F198': (7, 52, '#22c55e', 'white'),
            'C308': (18, 52, '#22c55e', 'white'),
            'W136': (7, 32, '#4ade80', 'black'),
            'A35':  (18, 32, '#4ade80', 'black'),
            'A33':  (7, 13, '#86efac', 'black'),
            'I300': (18, 13, '#86efac', 'black'),
        },
        'edges': [
            ('D83',  'G260', 2.15),
            ('F198', 'C308', 1.86),
            ('W136', 'C308', 2.95),
            ('G260', 'C308', 2.27),
            ('A35',  'W136', 1.75),
            ('A33',  'I300', 1.65)
        ]
    },
    {
        'id': 'Sector 2: Schiff Base Core',
        'color': '#eff6ff',
        'ec': '#2563eb',
        'header_c': '#1e40af',
        'box': [26, 4, 23, 91],
        'nodes': {
            'A292': (32, 72, '#2563eb', 'white'),
            'C185': (43, 72, '#1d4ed8', 'white'),
            'L119': (32, 52, '#3b82f6', 'white'),
            'L165': (43, 52, '#3b82f6', 'white'),
            'S144': (32, 32, '#60a5fa', 'white'),
            'T145': (43, 32, '#60a5fa', 'white'),
            'V137': (32, 13, '#93c5fd', 'black'),
            'L157': (43, 13, '#93c5fd', 'black'),
        },
        'edges': [
            ('A292', 'C185', 2.91),
            ('C185', 'L119', 2.98),
            ('C185', 'S144', 2.98),
            ('L119', 'L165', 2.10),
            ('S144', 'T145', 1.85),
            ('V137', 'L157', 1.70)
        ]
    },
    {
        'id': 'Sector 3: Schiff Base Ceiling',
        'color': '#fee2e2',
        'ec': '#dc2626',
        'header_c': '#991b1b',
        'box': [51, 4, 23, 91],
        'nodes': {
            'C264': (57, 72, '#b91c1c', 'white'),
            'T277': (68, 72, '#dc2626', 'white'),
            'C186': (62.5, 52, '#dc2626', 'white'),
            'I100': (57, 32, '#ef4444', 'white'),
            'F75':  (68, 32, '#ef4444', 'white'),
            'L51':  (57, 13, '#f87171', 'black'),
            'L38':  (68, 13, '#f87171', 'black'),
        },
        'edges': [
            ('C264', 'T277', 2.85),
            ('C264', 'C186', 1.85),
            ('T277', 'C186', 1.95),
            ('I100', 'F75',  1.75),
            ('I100', 'L51',  1.60),
            ('F75',  'L38',  1.55)
        ]
    },
    {
        'id': 'Sector 4: Pressure Clamp',
        'color': '#fff7ed',
        'ec': '#ea580c',
        'header_c': '#9a3412',
        'box': [76, 4, 23, 91],
        'nodes': {
            'E122': (82, 72, '#ea580c', 'white'),
            'C210': (93, 72, '#ea580c', 'white'),
            'I259': (82, 52, '#f97316', 'white'),
            'T213': (93, 52, '#f97316', 'white'),
            'S14':  (82, 32, '#fb923c', 'black'),
            'N8':   (93, 32, '#fb923c', 'black'),
            'G290': (87.5, 13, '#fdba74', 'black'),
        },
        'edges': [
            ('E122', 'C210', 2.25),
            ('C210', 'I259', 2.15),
            ('I259', 'T213', 1.80),
            ('S14',  'N8',   1.75),
            ('S14',  'G290', 1.60)
        ]
    }
]

for card in sector_cards:
    bx, by, bw, bh = card['box']
    rect = patches.FancyBboxPatch((bx, by), bw, bh, boxstyle='Round,pad=0.5,rounding_size=1.5',
                                  facecolor=card['color'], edgecolor=card['ec'], lw=1.3, zorder=1)
    ax2.add_patch(rect)
    
    ax2.text(bx + bw/2, by + bh - 5.5, card['id'], fontsize=8.6, fontweight='bold',
             color=card['header_c'], ha='center', va='center', zorder=2)
    
    for n1, n2, w in card['edges']:
        if n1 in card['nodes'] and n2 in card['nodes']:
            x1, y1, _, _ = card['nodes'][n1]
            x2, y2, _, _ = card['nodes'][n2]
            ax2.plot([x1, x2], [y1, y2], color='#64748b', lw=w*0.8, alpha=0.75, zorder=3)
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            ax2.text(mx, my, f'{w:.2f}', fontsize=6.2, ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='#cbd5e1', lw=0.5), zorder=4)
            
    for name, (x, y, bg_c, txt_c) in card['nodes'].items():
        circle = plt.Circle((x, y), 2.9, facecolor=bg_c, edgecolor='black', lw=1.1, zorder=5)
        ax2.add_patch(circle)
        ax2.text(x, y, name, color=txt_c, fontsize=7.0, fontweight='bold', ha='center', va='center', zorder=6)

# Output paths
output_paths = [
    '/Users/sergei/Projects/TOGA_MEME/hyphaeon_paper/figures/yokoyama_rhodopsin_benchmark.pdf',
    '/Users/sergei/Projects/TOGA_MEME/hyphaeon_paper/figures/yokoyama_rhodopsin_benchmark.png'
]

for p in output_paths:
    plt.savefig(p, bbox_inches='tight', dpi=300)
plt.close()
print('Successfully saved updated figure to PDF and PNG!')
