import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set publication style - clean, crisp, Tufte-style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']
plt.rcParams['axes.edgecolor'] = '#475569'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# Define full benchmark systems with reconciled MSA coordinates and rigorous validation tiers:
# tier: 1 = Directly Experimentally Validated in vitro / in vivo (★)
#       2 = Structurally / Biochemically Supported in literature (◆)
#       3 = Computational / Evolutionary Candidate (—)
benchmark_systems = [
    {
        'id': 'Prestin',
        'title': 'Prestin (SLC26A5) — Mammalian Echolocation',
        'stats': 'N = 128 taxa, L = 758 aa | 4 Shared, 9 HyphAeon-only, 0 CSUBST-only | 2 Exp. Validated, 10 Struct. Supported',
        'length': 758,
        'domains': [
            (1, 97, 'N-term', '#f1f5f9', '#94a3b8'),
            (98, 501, '14-TM Core Engine', '#e2e8f0', '#64748b'),
            (502, 720, 'STAS Acoustic Domain', '#cbd5e1', '#475569'),
            (721, 758, 'C-term', '#f1f5f9', '#94a3b8'),
        ],
        'sites': [
            # Concordant Core
            {'site': 14, 'from': 'N', 'to': 'T', 'cat': 'both', 'tier': 1, 'lit_desc': 'NLC\nV-sens', 'sector': 'S3', 'sec_color': '#c2410c', 'rho': 0.264, 'q': 0.014, 'csubst': True},
            {'site': 576, 'from': 'M', 'to': 'L', 'cat': 'both', 'tier': 2, 'lit_desc': 'STAS\nClamp', 'sector': 'S3', 'sec_color': '#c2410c', 'rho': 0.735, 'q': 1.4e-13, 'csubst': True},
            {'site': 612, 'from': 'G', 'to': 'A', 'cat': 'both', 'tier': 2, 'lit_desc': 'STAS\nMotor', 'sector': '', 'sec_color': '#94a3b8', 'rho': 0.375, 'q': 1.5e-4, 'csubst': True},
            {'site': 703, 'from': 'Q', 'to': 'R', 'cat': 'both', 'tier': 2, 'lit_desc': 'STAS\nTail', 'sector': 'S1', 'sec_color': '#4f46e5', 'rho': 0.585, 'q': 1.7e-11, 'csubst': True},
            # HyphAeon Only
            {'site': 82, 'from': 'L', 'to': 'I', 'cat': 'hyphaeon', 'tier': 2, 'lit_desc': 'TM2\nHinge', 'sector': 'S2', 'sec_color': '#0d9488', 'rho': 0.438, 'q': 4.0e-6, 'csubst': False},
            {'site': 175, 'from': 'G', 'to': 'S', 'cat': 'hyphaeon', 'tier': 2, 'lit_desc': 'TM4\nMotor', 'sector': '', 'sec_color': '#94a3b8', 'rho': 0.229, 'q': 0.043, 'csubst': False},
            {'site': 194, 'from': 'S', 'to': 'T', 'cat': 'hyphaeon', 'tier': 2, 'lit_desc': 'Loop', 'sector': '', 'sec_color': '#94a3b8', 'rho': 0.676, 'q': 1.4e-13, 'csubst': False},
            {'site': 316, 'from': 'I', 'to': 'V', 'cat': 'hyphaeon', 'tier': 2, 'lit_desc': 'TM8\nPivot', 'sector': 'S2', 'sec_color': '#0d9488', 'rho': 0.692, 'q': 1.4e-13, 'csubst': False},
            {'site': 392, 'from': 'I', 'to': 'T', 'cat': 'hyphaeon', 'tier': 1, 'lit_desc': 'NLC\nV1/2', 'sector': 'S3', 'sec_color': '#c2410c', 'rho': 0.438, 'q': 4.0e-6, 'csubst': False},
            {'site': 400, 'from': 'S', 'to': 'A', 'cat': 'hyphaeon', 'tier': 2, 'lit_desc': 'Gate', 'sector': '', 'sec_color': '#94a3b8', 'rho': 0.602, 'q': 3.2e-12, 'csubst': False},
            {'site': 431, 'from': 'L', 'to': 'M', 'cat': 'hyphaeon', 'tier': 3, 'lit_desc': '', 'sector': 'S1', 'sec_color': '#4f46e5', 'rho': 0.541, 'q': 1.4e-9, 'csubst': False},
            {'site': 505, 'from': 'V', 'to': 'I', 'cat': 'hyphaeon', 'tier': 2, 'lit_desc': 'STAS\nMotor', 'sector': 'S1', 'sec_color': '#4f46e5', 'rho': 0.592, 'q': 9.2e-12, 'csubst': False},
            {'site': 699, 'from': 'S', 'to': 'A', 'cat': 'hyphaeon', 'tier': 2, 'lit_desc': 'Clamp', 'sector': 'S3', 'sec_color': '#c2410c', 'rho': 0.298, 'q': 4.4e-3, 'csubst': False},
        ]
    },
    {
        'id': 'ATP1alpha',
        'title': r'ATP1$\alpha$ (ATP1A1) — Insect Cardenolide/Ouabain Resistance',
        'stats': 'N = 30 taxa, L = 1045 aa | 3 Shared, 4 HyphAeon-only, 2 CSUBST-only | 2 Exp. Validated, 2 Struct. Supported',
        'length': 1045,
        'domains': [
            (1, 85, 'Actuator (A)', '#f1f5f9', '#94a3b8'),
            (86, 141, 'TM1–2 (Loop 1–2)', '#cbd5e1', '#475569'),
            (142, 283, 'Actuator (A)', '#f1f5f9', '#94a3b8'),
            (284, 341, 'TM3–4', '#e2e8f0', '#64748b'),
            (342, 378, 'P-domain', '#f1f5f9', '#94a3b8'),
            (379, 589, 'N-domain', '#e2e8f0', '#64748b'),
            (590, 769, 'P-domain', '#f1f5f9', '#94a3b8'),
            (770, 1045, 'TM5–10 Core', '#cbd5e1', '#475569'),
        ],
        'sites': [
            # Concordant Core
            {'site': 140, 'from': 'Q', 'to': 'V', 'cat': 'both', 'tier': 1, 'lit_desc': 'Pocket\nKi', 'sector': 'S1', 'sec_color': '#4f46e5', 'rho': 0.665, 'q': 0.005, 'csubst': True},
            {'site': 148, 'from': 'A', 'to': 'S', 'cat': 'both', 'tier': 1, 'lit_desc': 'Allosteric\nRescue', 'sector': '', 'sec_color': '#94a3b8', 'rho': 0.274, 'q': 0.340, 'csubst': True},
            {'site': 687, 'from': 'V', 'to': 'I', 'cat': 'both', 'tier': 2, 'lit_desc': 'TM5\nPore', 'sector': '', 'sec_color': '#94a3b8', 'rho': 0.740, 'q': 0.001, 'csubst': True},
            # HyphAeon Only
            {'site': 138, 'from': 'S', 'to': 'G', 'cat': 'hyphaeon', 'tier': 2, 'lit_desc': 'Shield', 'sector': 'S1', 'sec_color': '#4f46e5', 'rho': 0.652, 'q': 0.007, 'csubst': False},
            {'site': 283, 'from': 'C', 'to': 'T', 'cat': 'hyphaeon', 'tier': 3, 'lit_desc': '', 'sector': 'S2', 'sec_color': '#0d9488', 'rho': 0.560, 'q': 0.049, 'csubst': False},
            {'site': 545, 'from': 'I', 'to': 'L', 'cat': 'hyphaeon', 'tier': 3, 'lit_desc': '', 'sector': 'S2', 'sec_color': '#0d9488', 'rho': 0.685, 'q': 0.004, 'csubst': False},
            {'site': 1034, 'from': 'N', 'to': 'C', 'cat': 'hyphaeon', 'tier': 3, 'lit_desc': '', 'sector': '', 'sec_color': '#94a3b8', 'rho': 0.586, 'q': 0.112, 'csubst': False},
            # CSUBST Only
            {'site': 457, 'from': 'P', 'to': 'S', 'cat': 'csubst', 'tier': 3, 'lit_desc': '', 'sector': '', 'sec_color': '#94a3b8', 'rho': None, 'q': None, 'csubst': True},
            {'site': 598, 'from': 'P', 'to': 'A', 'cat': 'csubst', 'tier': 3, 'lit_desc': '', 'sector': '', 'sec_color': '#94a3b8', 'rho': None, 'q': None, 'csubst': True},
        ]
    },
    {
        'id': 'PEPC2',
        'title': r'PEPC (PEPC2) — Plant C$_4$ Photosynthetic Adaptation',
        'stats': 'N = 71 taxa, L = 971 aa | 5 Shared, 2 HyphAeon-only, 0 CSUBST-only | 2 Exp. Validated, 2 Struct. Supported',
        'length': 971,
        'domains': [
            (1, 150, 'N-term Reg', '#f1f5f9', '#94a3b8'),
            (151, 650, r'Catalytic $(\beta/\alpha)_8$ Barrel Core', '#e2e8f0', '#64748b'),
            (651, 971, 'C-term Allosteric Domain', '#cbd5e1', '#475569'),
        ],
        'sites': [
            # Concordant Core
            {'site': 58, 'from': 'T', 'to': 'F', 'cat': 'both', 'tier': 3, 'lit_desc': '', 'sector': '', 'sec_color': '#94a3b8', 'rho': 0.285, 'q': 0.103, 'csubst': True},
            {'site': 159, 'from': 'V', 'to': 'I', 'cat': 'both', 'tier': 2, 'lit_desc': 'Barrel\nLid', 'sector': 'S2', 'sec_color': '#0d9488', 'rho': 0.508, 'q': 0.007, 'csubst': True},
            {'site': 575, 'from': 'A', 'to': 'S', 'cat': 'both', 'tier': 1, 'lit_desc': 'C4\nSwitch', 'sector': 'S3', 'sec_color': '#c2410c', 'rho': 0.669, 'q': 1.4e-7, 'csubst': True},
            {'site': 778, 'from': 'A', 'to': 'S', 'cat': 'both', 'tier': 1, 'lit_desc': 'Malate\nReg.', 'sector': 'S1', 'sec_color': '#4f46e5', 'rho': 0.987, 'q': 1.2e-12, 'csubst': True},
            {'site': 902, 'from': 'C', 'to': 'F', 'cat': 'both', 'tier': 2, 'lit_desc': 'Allosteric', 'sector': '', 'sec_color': '#94a3b8', 'rho': 0.600, 'q': 7.2e-5, 'csubst': True},
            # HyphAeon Only
            {'site': 54, 'from': 'D', 'to': 'H', 'cat': 'hyphaeon', 'tier': 3, 'lit_desc': '', 'sector': 'S1', 'sec_color': '#4f46e5', 'rho': 0.565, 'q': 0.004, 'csubst': False},
            {'site': 852, 'from': 'W', 'to': 'K', 'cat': 'hyphaeon', 'tier': 3, 'lit_desc': '', 'sector': '', 'sec_color': '#94a3b8', 'rho': 0.570, 'q': 0.0002, 'csubst': False},
        ]
    },
    {
        'id': 'RNASE1',
        'title': 'RNASE1 — Colobine Primate Foregut Digestive Adaptation',
        'stats': 'N = 34 taxa, L = 156 aa | 4 Shared, 3 HyphAeon-only, 0 CSUBST-only | 4 Exp. Validated (Zhang 2002, 2006)',
        'length': 156,
        'domains': [
            (1, 28, 'Signal', '#f1f5f9', '#94a3b8'),
            (29, 156, 'RNase A Catalytic Core', '#e2e8f0', '#64748b'),
        ],
        'sites': [
            # Concordant Core
            {'site': 29, 'from': 'R', 'to': 'K', 'cat': 'both', 'tier': 1, 'lit_desc': 'Acid\npI', 'sector': '', 'sec_color': '#94a3b8', 'rho': 0.334, 'q': 0.024, 'csubst': True},
            {'site': 32, 'from': 'R', 'to': 'Q', 'cat': 'both', 'tier': 1, 'lit_desc': 'Acid\npH 6.3', 'sector': 'S1', 'sec_color': '#4f46e5', 'rho': 0.378, 'q': 0.012, 'csubst': True},
            {'site': 34, 'from': 'K', 'to': 'E', 'cat': 'both', 'tier': 1, 'lit_desc': 'Charge\nSwap', 'sector': 'S1', 'sec_color': '#4f46e5', 'rho': 0.389, 'q': 0.010, 'csubst': True},
            {'site': 67, 'from': 'R', 'to': 'W', 'cat': 'both', 'tier': 1, 'lit_desc': 'Pepsin\nResist', 'sector': 'S1', 'sec_color': '#4f46e5', 'rho': 0.468, 'q': 0.003, 'csubst': True},
            # HyphAeon Only
            {'site': 50, 'from': 'N', 'to': 'S', 'cat': 'hyphaeon', 'tier': 3, 'lit_desc': '', 'sector': 'S1', 'sec_color': '#4f46e5', 'rho': 0.760, 'q': 1.2e-6, 'csubst': False},
            {'site': 106, 'from': 'S', 'to': 'R', 'cat': 'hyphaeon', 'tier': 3, 'lit_desc': '', 'sector': 'S1', 'sec_color': '#4f46e5', 'rho': 0.738, 'q': 2.4e-6, 'csubst': False},
            {'site': 150, 'from': 'A', 'to': 'D', 'cat': 'hyphaeon', 'tier': 3, 'lit_desc': '', 'sector': 'S2', 'sec_color': '#0d9488', 'rho': 0.662, 'q': 4.1e-5, 'csubst': False},
        ]
    },
    {
        'id': 'Lysozyme_c',
        'title': 'Lysozyme c — Foregut Fermenting Stomach Lysozyme',
        'stats': 'N = 28 taxa, L = 176 aa | 5 Shared, 4 HyphAeon-only, 0 CSUBST-only | 5 Struct. Supported (Stewart 1987)',
        'length': 176,
        'domains': [
            (1, 18, 'Signal', '#f1f5f9', '#94a3b8'),
            (19, 58, r'$\alpha$-Domain (N)', '#e2e8f0', '#64748b'),
            (59, 99, r'$\beta$-Domain (Cleft)', '#cbd5e1', '#475569'),
            (100, 176, r'$\alpha$-Domain (C)', '#e2e8f0', '#64748b'),
        ],
        'sites': [
            # Concordant Core
            {'site': 60, 'from': 'R', 'to': 'K', 'cat': 'both', 'tier': 2, 'lit_desc': 'Pepsin\nEvasion', 'sector': 'S2', 'sec_color': '#0d9488', 'rho': 0.224, 'q': 0.048, 'csubst': True},
            {'site': 87, 'from': 'Q', 'to': 'K', 'cat': 'both', 'tier': 2, 'lit_desc': 'Acid\nStability', 'sector': 'S1', 'sec_color': '#4f46e5', 'rho': 0.241, 'q': 0.038, 'csubst': True},
            {'site': 121, 'from': 'N', 'to': 'D', 'cat': 'both', 'tier': 2, 'lit_desc': 'Charge\nReduct.', 'sector': 'S1', 'sec_color': '#4f46e5', 'rho': 0.757, 'q': 8.3e-14, 'csubst': True},
            {'site': 133, 'from': 'S', 'to': 'N', 'cat': 'both', 'tier': 2, 'lit_desc': 'Acid\nCleft', 'sector': '', 'sec_color': '#94a3b8', 'rho': 0.334, 'q': 0.009, 'csubst': True},
            {'site': 172, 'from': 'R', 'to': 'E', 'cat': 'both', 'tier': 2, 'lit_desc': 'C-term\nTail', 'sector': '', 'sec_color': '#94a3b8', 'rho': 0.204, 'q': 0.075, 'csubst': True},
            # HyphAeon Only
            {'site': 63, 'from': 'M', 'to': 'L', 'cat': 'hyphaeon', 'tier': 3, 'lit_desc': '', 'sector': '', 'sec_color': '#94a3b8', 'rho': 0.362, 'q': 0.005, 'csubst': False},
            {'site': 75, 'from': 'V', 'to': 'I', 'cat': 'hyphaeon', 'tier': 3, 'lit_desc': '', 'sector': '', 'sec_color': '#94a3b8', 'rho': 0.339, 'q': 0.008, 'csubst': False},
            {'site': 144, 'from': 'R', 'to': 'H', 'cat': 'hyphaeon', 'tier': 3, 'lit_desc': '', 'sector': 'S1', 'sec_color': '#4f46e5', 'rho': 0.584, 'q': 1.6e-7, 'csubst': False},
            {'site': 145, 'from': 'V', 'to': 'I', 'cat': 'hyphaeon', 'tier': 3, 'lit_desc': '', 'sector': 'S1', 'sec_color': '#4f46e5', 'rho': 0.461, 'q': 4.2e-5, 'csubst': False},
        ]
    },
    {
        'id': 'RNaseT2',
        'title': 'RNase T2 — Carnivorous Plant Digestive Fluid Adaptation',
        'stats': 'N = 67 taxa, L = 274 aa | 5 Shared, 2 HyphAeon-only, 0 CSUBST-only | 5 Struct. Supported (Fukushima 2017)',
        'length': 274,
        'domains': [
            (1, 24, 'Signal', '#f1f5f9', '#94a3b8'),
            (25, 39, 'N-term', '#e2e8f0', '#64748b'),
            (40, 60, 'CAS-I Active Cleft', '#cbd5e1', '#475569'),
            (61, 99, 'Core', '#e2e8f0', '#64748b'),
            (100, 120, 'CAS-II Active Cleft', '#cbd5e1', '#475569'),
            (121, 274, 'C-term Secretory', '#f1f5f9', '#94a3b8'),
        ],
        'sites': [
            # Concordant Core
            {'site': 57, 'from': 'K', 'to': 'S', 'cat': 'both', 'tier': 2, 'lit_desc': 'CAS-I\nCleft', 'sector': 'S1', 'sec_color': '#4f46e5', 'rho': 0.421, 'q': 0.012, 'csubst': True},
            {'site': 146, 'from': 'Q', 'to': 'E', 'cat': 'both', 'tier': 2, 'lit_desc': 'CAS-II\nPocket', 'sector': '', 'sec_color': '#94a3b8', 'rho': 0.322, 'q': 0.050, 'csubst': True},
            {'site': 148, 'from': 'G', 'to': 'S', 'cat': 'both', 'tier': 2, 'lit_desc': 'CAS-II\nFlank', 'sector': 'S1', 'sec_color': '#4f46e5', 'rho': 0.377, 'q': 0.029, 'csubst': True},
            {'site': 197, 'from': 'G', 'to': 'A', 'cat': 'both', 'tier': 2, 'lit_desc': 'Core\nRigid.', 'sector': 'S2', 'sec_color': '#0d9488', 'rho': 0.485, 'q': 0.002, 'csubst': True},
            {'site': 234, 'from': 'F', 'to': 'S', 'cat': 'both', 'tier': 2, 'lit_desc': 'Disulfide', 'sector': '', 'sec_color': '#94a3b8', 'rho': 0.503, 'q': 0.001, 'csubst': True},
            # HyphAeon Only
            {'site': 151, 'from': 'Q', 'to': 'A', 'cat': 'hyphaeon', 'tier': 3, 'lit_desc': '', 'sector': 'S1', 'sec_color': '#4f46e5', 'rho': 0.411, 'q': 0.010, 'csubst': False},
            {'site': 215, 'from': 'L', 'to': 'I', 'cat': 'hyphaeon', 'tier': 3, 'lit_desc': '', 'sector': 'S2', 'sec_color': '#0d9488', 'rho': 0.460, 'q': 0.005, 'csubst': False},
        ]
    }
]

# Set up figure - optimized height & width for crystal clear readability
n_genes = len(benchmark_systems)
fig = plt.figure(figsize=(15.5, 21.5), dpi=300)

# Create 6 vertical panels
gs_outer = fig.add_gridspec(n_genes, 1, height_ratios=[1.0]*n_genes, hspace=0.40,
                            left=0.12, right=0.98, top=0.965, bottom=0.040)

N_MAX_COLS = 13  # Prestin has 13 sites

for i, gdata in enumerate(benchmark_systems):
    # Sub-grid: 1D ribbon on top, clean matrix below
    sub_gs = gs_outer[i].subgridspec(2, 1, height_ratios=[0.12, 0.88], hspace=0.14)
    
    ax_ribbon = fig.add_subplot(sub_gs[0])
    ax_matrix = fig.add_subplot(sub_gs[1])
    
    L = gdata['length']
    ax_ribbon.set_xlim(0, L)
    ax_ribbon.set_ylim(-0.1, 1.45)
    ax_ribbon.axis('off')
    
    # -----------------------------------------------------------------
    # Panel Title (Enlarged, Bold, Clean Typography)
    # -----------------------------------------------------------------
    fig.text(0.12, sub_gs[0].get_position(fig).y1 + 0.008, gdata['title'],
             fontsize=13.5, fontweight='bold', color='#0f172a', va='bottom', ha='left')
             
    # -----------------------------------------------------------------
    # 1D Domain Ribbon (Structural Architecture)
    # -----------------------------------------------------------------
    ax_ribbon.plot([0, L], [0.45, 0.45], color='#94a3b8', lw=1.5, zorder=1)
    
    for start, end, dname, fill_c, edge_c in gdata['domains']:
        w = max(end - start, 3)
        rect = patches.Rectangle(
            (start, 0.10), w, 0.70,
            facecolor=fill_c, edgecolor=edge_c, lw=0.9, zorder=2
        )
        ax_ribbon.add_patch(rect)
        if w > L * 0.08:
            ax_ribbon.text(start + w/2.0, 0.45, dname, ha='center', va='center',
                           fontsize=9.5, fontweight='bold', color='#334155', zorder=3)
                           
    # Sequence position pins on ribbon
    for sinfo in gdata['sites']:
        site_pos = sinfo['site']
        tier = sinfo['tier']
        ax_ribbon.plot([site_pos, site_pos], [0.80, 1.15], color='#0f172a', lw=1.3, zorder=4)
        if tier == 1:
            ax_ribbon.plot(site_pos, 1.25, marker='*', markersize=8.5, color='#0f172a', zorder=5)
        elif tier == 2:
            ax_ribbon.plot(site_pos, 1.25, marker='D', markersize=6.0, color='#334155', zorder=5)

    # -----------------------------------------------------------------
    # Discrete Evidence Matrix
    # -----------------------------------------------------------------
    ax_matrix.set_ylim(-0.55, 5.75)
    sites_list = gdata['sites']
    N_sites = len(sites_list)
    ax_matrix.set_xlim(-1.90, N_MAX_COLS - 0.10)
    ax_matrix.axis('off')
    
    # Left Row Labels (Right-aligned at x = -0.35)
    row_labels = [
        (4.70, 'Codon', '#0f172a', 'bold'),
        (3.90, 'Variant', '#334155', 'bold'),
        (2.95, 'Literature', '#0f172a', 'bold'),
        (1.80, 'Sector', '#334155', 'bold'),
        (0.85, r'HyphAeon $\rho_s$', '#0f172a', 'bold'),
        (-0.15, 'CSUBST', '#0f172a', 'bold'),
    ]
    for y_pos, rlabel, lcol, fweight in row_labels:
        ax_matrix.text(-0.35, y_pos, rlabel, ha='right', va='center',
                       fontsize=12.0, fontweight=fweight, color=lcol)
                       
    # Subtle vertical dividing rule between row labels and data
    ax_matrix.plot([-0.22, -0.22], [-0.45, 5.15], color='#94a3b8', lw=1.2, zorder=1)
    
    # Horizontal dividing lines across the table
    for y_line in [4.30, 3.48, 2.30, 1.32, 0.25, -0.45]:
        ax_matrix.plot([-0.22, N_MAX_COLS - 0.10], [y_line, y_line], color='#e2e8f0', lw=0.9, zorder=1)
                        
    # Group sites by category
    cat_indices = {'both': [], 'hyphaeon': [], 'csubst': []}
    for idx, s in enumerate(sites_list):
        cat_indices[s['cat']].append(idx)
        
    cat_headers = [
        ('both', 'CONCORDANT', '#0f172a', '#059669'),
        ('hyphaeon', 'HYPHAEON ONLY', '#1e293b', '#2563eb'),
        ('csubst', 'CSUBST ONLY', '#334155', '#d97706'),
    ]
    
    for ckey, cheader, text_c, bar_c in cat_headers:
        idxs = cat_indices[ckey]
        if not idxs:
            continue
        min_x = min(idxs) - 0.44
        max_x = max(idxs) + 0.44
        w_block = max_x - min_x
        
        # Clean category header line + title above
        ax_matrix.plot([min_x, max_x], [5.18, 5.18], color=bar_c, lw=2.6, zorder=3)
        ax_matrix.text(min_x + w_block/2.0, 5.34, cheader, ha='center', va='bottom',
                       fontsize=11.0, fontweight='bold', color=text_c, zorder=3)
        
        # Subtle category group boundary line
        if max_x < N_sites - 0.5:
            ax_matrix.plot([max_x + 0.08, max_x + 0.08], [-0.45, 5.18], color='#cbd5e1', lw=1.1, linestyle=':', zorder=2)
                        
    # Render Columns for each candidate site
    for col_idx, s in enumerate(sites_list):
        cx = col_idx
        
        # Alternating subtle column background
        if col_idx % 2 == 1:
            col_bg = patches.Rectangle(
                (cx - 0.46, -0.45), 0.92, 5.63,
                facecolor='#f8fafc', edgecolor='none', zorder=0
            )
            ax_matrix.add_patch(col_bg)
        
        # 1. Codon
        ax_matrix.text(cx, 4.70, f"{s['site']}", ha='center', va='center',
                       fontsize=12.5, fontweight='bold', color='#0f172a', zorder=4)
                       
        # 2. Variant
        subst_tex = r"$\mathbf{" + s['from'] + r"\rightarrow " + s['to'] + r"}$"
        ax_matrix.text(cx, 3.90, subst_tex, ha='center', va='center',
                       fontsize=11.8, fontweight='bold', color='#334155', zorder=4)
                        
        # 3. Literature Support (Tier 1 = Star, Tier 2 = Diamond, Tier 3 = em-dash)
        tier = s['tier']
        if tier == 1:
            ax_matrix.plot(cx, 3.16, marker='*', markersize=10.0, color='#0f172a', zorder=4)
            if s['lit_desc']:
                ax_matrix.text(cx, 2.68, s['lit_desc'], ha='center', va='center',
                               fontsize=9.0, fontweight='bold', color='#0f172a', zorder=4)
        elif tier == 2:
            ax_matrix.plot(cx, 3.16, marker='D', markersize=7.0, color='#334155', zorder=4)
            if s['lit_desc']:
                ax_matrix.text(cx, 2.68, s['lit_desc'], ha='center', va='center',
                               fontsize=9.0, fontweight='bold', color='#334155', zorder=4)
        else:
            ax_matrix.text(cx, 2.95, '—', ha='center', va='center',
                           fontsize=12.0, color='#94a3b8', fontweight='bold', zorder=4)
                            
        # 4. Epistatic Sector (Small color circle dot with label, or em-dash)
        if s['sector']:
            ax_matrix.plot(cx - 0.18, 1.80, marker='o', markersize=7.5, color=s['sec_color'], zorder=4)
            ax_matrix.text(cx + 0.14, 1.80, s['sector'], ha='center', va='center',
                           fontsize=11.5, fontweight='bold', color='#1e293b', zorder=4)
        else:
            ax_matrix.text(cx, 1.80, '—', ha='center', va='center',
                           fontsize=12.0, color='#94a3b8', fontweight='bold', zorder=4)
                            
        # 5. HyphAeon Association (Crisp numeric rho + compact horizontal bar, or em-dash)
        rho_val = s['rho']
        q_val = s['q']
        if rho_val is not None:
            bar_w = min(max(rho_val * 0.72, 0.04), 0.72)
            # Numeric rho value
            ax_matrix.text(cx, 0.98, f"{rho_val:.2f}", ha='center', va='center',
                           fontsize=11.5, fontweight='bold' if (q_val is not None and q_val <= 0.05) else 'normal',
                           color='#0f172a' if (q_val is not None and q_val <= 0.05) else '#64748b', zorder=4)
            # Clean progress bar
            ax_matrix.plot([cx - 0.36, cx + 0.36], [0.55, 0.55], color='#e2e8f0', lw=4.2, solid_capstyle='round', zorder=2)
            if rho_val > 0.02:
                ax_matrix.plot([cx - 0.36, cx - 0.36 + bar_w], [0.55, 0.55],
                               color='#0f172a' if (q_val is not None and q_val <= 0.05) else '#64748b',
                               lw=4.2, solid_capstyle='round', zorder=3)
        else:
            ax_matrix.text(cx, 0.85, '—', ha='center', va='center',
                           fontsize=12.0, color='#94a3b8', fontweight='bold', zorder=4)
                        
        # 6. CSUBST Convergence (Solid dot if detected, em-dash if not)
        if s['csubst']:
            ax_matrix.plot(cx, -0.15, marker='o', markersize=9.0, color='#0f172a', zorder=4)
        else:
            ax_matrix.text(cx, -0.15, '—', ha='center', va='center',
                           fontsize=12.0, color='#94a3b8', fontweight='bold', zorder=4)

# -----------------------------------------------------------------
# Clean Bottom Legend (Evenly Spaced, Non-Overlapping)
# -----------------------------------------------------------------
leg_ax = fig.add_axes([0.10, 0.010, 0.88, 0.022])
leg_ax.set_xlim(0, 1.0)
leg_ax.set_ylim(0, 1.0)
leg_ax.axis('off')

# Simple, crisp legend elements with precise non-overlapping coordinates
leg_items = [
    (0.015, 'line', '#059669', 'Shared (Concordant)'),
    (0.155, 'line', '#2563eb', 'HyphAeon Only'),
    (0.280, 'line', '#d97706', 'CSUBST Only'),
    (0.400, 'star', '#0f172a', 'Exp. Validated (★)'),
    (0.550, 'diamond', '#334155', 'Struct. Supported (◆)'),
    (0.710, 'dot', '#0f172a', 'CSUBST Hit (●)'),
    (0.835, 'sector', '#4f46e5', 'Epistatic Sectors'),
    (0.950, 'dash', '#94a3b8', 'Novel (—)'),
]

for cx, itype, icol, ilabel in leg_items:
    if itype == 'line':
        leg_ax.plot([cx - 0.018, cx - 0.005], [0.5, 0.5], color=icol, lw=2.8)
        leg_ax.text(cx, 0.5, ilabel, ha='left', va='center',
                    fontsize=10.5, fontweight='bold', color='#1e293b')
    elif itype == 'star':
        leg_ax.plot(cx - 0.010, 0.5, marker='*', markersize=8.5, color=icol)
        leg_ax.text(cx, 0.5, ilabel, ha='left', va='center',
                    fontsize=10.5, fontweight='bold', color='#1e293b')
    elif itype == 'diamond':
        leg_ax.plot(cx - 0.010, 0.5, marker='D', markersize=6.0, color=icol)
        leg_ax.text(cx, 0.5, ilabel, ha='left', va='center',
                    fontsize=10.5, fontweight='bold', color='#1e293b')
    elif itype == 'dot':
        leg_ax.plot(cx - 0.010, 0.5, marker='o', markersize=7.5, color=icol)
        leg_ax.text(cx, 0.5, ilabel, ha='left', va='center',
                    fontsize=10.5, fontweight='bold', color='#1e293b')
    elif itype == 'sector':
        leg_ax.plot(cx - 0.014, 0.5, marker='o', markersize=6.0, color='#4f46e5')
        leg_ax.plot(cx - 0.006, 0.5, marker='o', markersize=6.0, color='#0d9488')
        leg_ax.text(cx, 0.5, ilabel, ha='left', va='center',
                    fontsize=10.5, fontweight='bold', color='#1e293b')
    elif itype == 'dash':
        leg_ax.text(cx - 0.008, 0.5, '—', ha='center', va='center', fontsize=11.0, color='#94a3b8', fontweight='bold')
        leg_ax.text(cx, 0.5, ilabel, ha='left', va='center',
                    fontsize=10.5, fontweight='bold', color='#64748b')


for out_dir in ['hyphaeon_paper/figures', 'science_paper/figures']:
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, 'csubst_hyphaeon_site_concordance.pdf')
    out_png = os.path.join(out_dir, 'csubst_hyphaeon_site_concordance.png')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, bbox_inches='tight', dpi=300)
    print(f'Generated: {out_pdf} and {out_png}')
plt.close()
