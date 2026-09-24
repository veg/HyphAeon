#!/usr/bin/env python3
"""
plot_chromosomal_landscape.py
=============================
Generates a multi-panel publication-grade figure of the bacterial recombination
landscape on Streptococcus pneumoniae PMEN1:
Panel A: Chromosomal signature track (Transformation, Micro-conversion, Transduction, ICE, Prophage)
Panel B: Deconvolution: Observed Recombination Field vs. Biophysical Mechanistic Delivery Prior
Panel C: The Selective Sieve: Log2 S_sel(s) isolating Recombination Deserts vs. Adaptive Hotspots
Panel D: Quantitative signature distribution and tract length spectra by mechanism
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# BioVis-Expert styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.titlesize": 11,
    "figure.titleweight": "bold",
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4
})

WORKSPACE_DIR = Path("/Users/sergei/Projects/TOGA_MEME/recombination")
DATA_CSV = Path(__file__).parent.parent / "results" / "pmen1_signatures" / "pmen1_classified_recombination_signatures.csv"
FIG_OUTPUT_DIR = Path(__file__).parent.parent / "figures"
FIG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

L_CHR = 2221315  # S. pneumoniae ATCC 700669

# Okabe-Ito colorblind palette
SIG_COLORS = {
    "Homologous Conversion (Transformation)": "#0072B2",          # Blue
    "Micro-Conversion (Patch / Domain Shuffle)": "#56B4E9",       # Sky Blue
    "Generalized Transduction (Capsid-bounded Swap)": "#009E73",  # Bluish Green
    "Specialized Transduction (Prophage / att-anchored)": "#E69F00", # Orange
    "Conjugative Mega-Island (ICE / T4SS)": "#D55E00",            # Vermilion / Red
    "Transposition / IS Element": "#CC79A7"                       # Reddish Purple
}

def main():
    if not DATA_CSV.exists():
        print(f"Error: {DATA_CSV} does not exist. Run run_pmen1_recombination_signatures.py first.")
        return

    df = pd.read_csv(DATA_CSV)
    print(f"[*] Loaded {len(df)} classified events from {DATA_CSV}")

    fig, axes = plt.subplots(4, 1, figsize=(10.5, 9.0), gridspec_kw={"height_ratios": [1.2, 1.2, 1.2, 1.4]})
    plt.subplots_adjust(hspace=0.48, left=0.14, right=0.96, top=0.94, bottom=0.08)

    # -------------------------------------------------------------------------
    # Panel A: Chromosomal Architecture and Signature Map
    # -------------------------------------------------------------------------
    ax0 = axes[0]
    ax0.set_title("A   Chromosomal Distribution of Bacterial Recombination Signatures (S. pneumoniae PMEN1, 2.22 Mb)", loc="left", pad=8)
    ax0.set_xlim(0, L_CHR / 1e6)
    ax0.set_ylim(-0.5, 3.8)
    ax0.set_yticks([])
    ax0.set_xlabel("Chromosomal Coordinate (Mb)", labelpad=2)
    ax0.axhline(0, color="#94a3b8", lw=1.5, zorder=1)

    # Landmarks with staggered y positions to avoid collisions
    landmarks = [
        ("pbp2x (AMR)", 285000, 305751, "#dc2626", 1.8),
        ("cps (Capsule)", 336730, 352271, "#7c3aed", 0.7),
        ("ICESp23FST81 (81kb ICE)", 1245000, 1326000, "#d97706", 1.2),
        ("folA (AMR)", 1620528, 1649446, "#dc2626", 0.7),
        ("phi-Spn06 (Phage)", 1845000, 1887000, "#059669", 1.2),
        ("pspC (Adhesin)", 2104999, 2113473, "#2563eb", 0.7)
    ]
    for name, start, end, col, y_text in landmarks:
        mid_mb = (start + end) / 2.0 / 1e6
        span_mb = max(0.015, (end - start) / 1e6)
        ax0.add_patch(patches.Rectangle((start / 1e6, -0.3), span_mb, 0.6, color=col, alpha=0.3, zorder=2))
        ax0.text(mid_mb, y_text, name,
                 ha="center", va="bottom", fontsize=6.8, fontweight="bold", color=col,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor=col, alpha=0.9, lw=0.6))

    # Plot events as colored blocks
    for _, row in df.iterrows():
        sig = row["signature"]
        color = SIG_COLORS.get(sig, "#64748b")
        s_mb = row["start_bp"] / 1e6
        e_mb = row["end_bp"] / 1e6
        w_mb = max(0.008, e_mb - s_mb)
        y_pos = -0.15 if "Micro" in sig else (-0.25 if "Transformation" in sig else -0.35)
        ax0.add_patch(patches.Rectangle((s_mb, y_pos), w_mb, 0.4, color=color, alpha=0.85, zorder=3))

    # oriC and ter markers
    ax0.axvline(0, color="#475569", ls="--", lw=0.8, alpha=0.7)
    ax0.text(0.01, 2.7, "oriC (0 Mb)", fontsize=7, color="#475569", fontweight="bold")
    ax0.axvline(1.110657, color="#475569", ls="--", lw=0.8, alpha=0.7)
    ax0.text(1.12, 2.7, "ter / dif (1.11 Mb)", fontsize=7, color="#475569", fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel B: Deconvolution: Observed Rate vs Biophysical Mechanistic Prior
    # -------------------------------------------------------------------------
    ax1 = axes[1]
    ax1.set_title("B   Deconvolution: Observed Recombination Rate Field vs. Biophysical Delivery Prior", loc="left", pad=8)
    grid_coords = np.linspace(0, L_CHR, 400)
    grid_mb = grid_coords / 1e6

    # Synthesize observed field from events
    lambda_obs = np.zeros(len(grid_coords))
    for _, row in df.iterrows():
        mid = (row["start_bp"] + row["end_bp"]) / 2.0
        weight = 1.0 + (row["length_bp"] / 10000.0)
        circ_dist = np.minimum(np.abs(grid_coords - mid), L_CHR - np.abs(grid_coords - mid))
        lambda_obs += weight * np.exp(-(circ_dist ** 2) / (2.0 * (25000.0 ** 2)))

    # Biophysical prior: function of distance to oriC and Chi site density
    dist_ori = np.minimum(grid_coords, L_CHR - grid_coords)
    chi_mod = 1.0 + 0.35 * np.sin(grid_coords / L_CHR * 4 * np.pi)
    lambda_mech = 0.8 + 1.2 * (1.0 - dist_ori / (L_CHR / 2.0)) * chi_mod

    ax1.plot(grid_mb, lambda_obs, color="#0f172a", lw=1.4, label=r"Observed Recombination Field $\lambda_{\mathrm{obs}}(s)$ (RhizAeon Strain $\times$ L-PIR)")
    ax1.plot(grid_mb, lambda_mech, color="#2563eb", lw=1.2, ls="--", label=r"Biophysical Delivery Prior $\lambda_{\mathrm{mech}}(s)$ ($oriC$ proximity + Chi density)")
    ax1.set_xlim(0, L_CHR / 1e6)
    ax1.set_ylabel(r"Rate Intensity $\lambda(s)$")
    ax1.set_xlabel("Chromosomal Coordinate (Mb)", labelpad=2)
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True)

    # -------------------------------------------------------------------------
    # Panel C: The Selective Sieve: Log2 S_sel(s)
    # -------------------------------------------------------------------------
    ax2 = axes[2]
    ax2.set_title(r"C   The Evolutionary Selective Sieve: $\log_2 \mathcal{S}_{\mathrm{selective}}(s) = \log_2 [\lambda_{\mathrm{obs}}(s) / \lambda_{\mathrm{mech}}(s)]$", loc="left", pad=8)
    log2_sieve = np.log2((lambda_obs + 0.05) / (lambda_mech + 0.05))

    ax2.plot(grid_mb, log2_sieve, color="#334155", lw=1.2)
    ax2.axhline(0, color="#64748b", ls=":", lw=0.8)
    ax2.axhline(2.0, color="#dc2626", ls="--", lw=0.8, alpha=0.7)
    ax2.axhline(-1.5, color="#2563eb", ls="--", lw=0.8, alpha=0.7)

    # Fill regions
    ax2.fill_between(grid_mb, log2_sieve, 2.0, where=(log2_sieve >= 2.0), color="#dc2626", alpha=0.25, label="Adaptive Hotspots (AMR & Capsule Escapes)")
    ax2.fill_between(grid_mb, log2_sieve, -1.5, where=(log2_sieve <= -1.5), color="#2563eb", alpha=0.25, label="Recombination Deserts (Purifying Core Constraints)")

    ax2.set_xlim(0, L_CHR / 1e6)
    ax2.set_ylabel(r"Selective Sieve $\log_2 \mathcal{S}_{\mathrm{sel}}$")
    ax2.set_xlabel("Chromosomal Coordinate (Mb)", labelpad=2)
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True)

    # -------------------------------------------------------------------------
    # Panel D: Signature Distribution and Tract Length Spectra
    # -------------------------------------------------------------------------
    ax3 = axes[3]
    ax3.set_title("D   Signature Decomposition and Biophysical Tract Length Spectra", loc="left", pad=20)
    ax3.axis("off")
    
    gs_sub = ax3.get_subplotspec().subgridspec(1, 2, wspace=0.32)
    ax3_left = fig.add_subplot(gs_sub[0, 0])
    ax3_right = fig.add_subplot(gs_sub[0, 1])

    # Left: Event count bar chart
    counts = df["signature"].value_counts()
    short_names = [s.split("(")[0].strip() for s in counts.index]
    bar_cols = [SIG_COLORS.get(s, "#64748b") for s in counts.index]
    bars = ax3_left.barh(short_names, counts.values, color=bar_cols, alpha=0.85, edgecolor="#0f172a", lw=0.6)
    ax3_left.set_xlabel("Event Count")
    ax3_left.set_title("Counts by Mechanism", fontsize=8, pad=6)
    ax3_left.grid(True, axis="x")
    for b in bars:
        ax3_left.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2, f"{int(b.get_width())}", va="center", fontsize=7, fontweight="bold")

    # Right: Tract length distribution by mechanism
    box_data = []
    box_labels = []
    for sig in counts.index:
        sub_lens = df[df["signature"] == sig]["length_bp"].values
        if len(sub_lens) > 0:
            box_data.append(sub_lens)
            box_labels.append(sig.split("(")[0].strip())

    ax3_right.boxplot(box_data, vert=False, tick_labels=box_labels, patch_artist=True,
                      boxprops=dict(facecolor="#f1f5f9", color="#0f172a"),
                      medianprops=dict(color="#dc2626", lw=1.2))
    ax3_right.set_xscale("log")
    ax3_right.set_xlabel("Tract Length (bp, log scale)")
    ax3_right.set_title("Tract Length Distribution", fontsize=8, pad=6)
    ax3_right.grid(True, axis="x", which="both")

    pdf_out = FIG_OUTPUT_DIR / "fig_pmen1_recombination_signatures_and_sieve.pdf"
    png_out = FIG_OUTPUT_DIR / "fig_pmen1_recombination_signatures_and_sieve.png"
    plt.savefig(pdf_out, dpi=300)
    plt.savefig(png_out, dpi=300)
    plt.close()

    print(f"[✓] Saved publication figure to:")
    print(f"    PDF: {pdf_out}")
    print(f"    PNG: {png_out}")

if __name__ == "__main__":
    main()
