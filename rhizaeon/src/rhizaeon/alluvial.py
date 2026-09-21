"""
rhizaeon.alluvial
=================
Continuous Manifold Alluvial Genome River-Flow Visualization.

Renders genome-wide lineage trajectories along the sequence coordinates,
illustrating smooth clade corridors and sharp reticulate crossovers.
"""

from typing import List, Dict, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from rhizaeon.tensor import PrefixDistanceEngine
from rhizaeon.manifold import trace_continuous_manifold_flow


def render_alluvial_genome_river(
    engine: PrefixDistanceEngine,
    taxa: List[str],
    recombination_events: Optional[List[Dict]] = None,
    window_units: int = 25,
    step: int = 2,
    output_path: str = "alluvial_genome_river.png",
    title: str = "RhizAeon Continuous Manifold Alluvial Genome River Flow",
    highlight_taxa: Optional[List[str]] = None,
    palette_name: str = "tab10"
):
    """
    Renders an Alluvial Genome River plot:
      Horizontal axis: Genomic Position (Codon or Nucleotide Units)
      Vertical axis: Primary Continuous Manifold Coordinate y_1(s)
      Lineages: Smooth ribbons / lines flowing along the chromosome.
      Recombinants: Visibly crossover between parental corridors at validated breakpoints.
    """
    flow_res = trace_continuous_manifold_flow(
        engine,
        window_units=window_units,
        step=step,
        k_dims=3
    )

    cutpoints = flow_res["cutpoints"]
    trajectories = flow_res["trajectories"]  # [T, N, 3]
    N = len(taxa)

    # Primary manifold coordinate y_1 along the genome: [T, N]
    y1 = trajectories[:, :, 0]

    # Assign distinct colors to clades / lineages
    cmap = plt.get_cmap(palette_name)
    taxa_colors = {}
    for i, t in enumerate(taxa):
        taxa_colors[t] = cmap(i % 10)

    # Recombinant taxa to highlight
    recs_in_events = set()
    if recombination_events:
        for ev in recombination_events:
            recs_in_events.add(ev["recombinant"])

    if highlight_taxa:
        recs_in_events.update(highlight_taxa)

    # Setup publication figure
    fig, (ax_river, ax_vel) = plt.subplots(
        2, 1,
        figsize=(14, 8),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True,
        dpi=300
    )

    # 1. Plot Background Non-Recombinant Lineages
    for i, t in enumerate(taxa):
        if t in recs_in_events:
            continue
        ax_river.plot(
            cutpoints,
            y1[:, i],
            color=taxa_colors[t],
            alpha=0.35,
            linewidth=1.8,
            label=None
        )

    # 2. Plot Recombinant Lineages with Bold Emphasis
    rec_line_handles = []
    for i, t in enumerate(taxa):
        if t in recs_in_events:
            line, = ax_river.plot(
                cutpoints,
                y1[:, i],
                color='#E63946',
                alpha=0.95,
                linewidth=3.2,
                linestyle='-',
                zorder=10,
                label=f"Recombinant: {t}"
            )
            rec_line_handles.append(line)

    # 3. Plot Detected Breakpoints and Annotations
    if recombination_events:
        for ev in recombination_events:
            bp = ev["breakpoint"]
            rec = ev["recombinant"]
            p_left = ev.get("parent_left", "")
            p_right = ev.get("parent_right", "")
            pir = ev.get("l_pir", ev.get("refined_pir", 0.0))

            ax_river.axvline(
                x=bp,
                color='#1D3557',
                linestyle='--',
                linewidth=2.0,
                alpha=0.85,
                zorder=15
            )
            ax_vel.axvline(
                x=bp,
                color='#1D3557',
                linestyle='--',
                linewidth=2.0,
                alpha=0.85,
                zorder=15
            )

            # Label on river plot
            annot_text = f"BP: {bp}\n{p_left} $\\to$ {p_right}\n(PIR={pir:.2f})"
            ax_river.annotate(
                annot_text,
                xy=(bp, np.max(y1)),
                xytext=(bp + 5, np.max(y1) * 0.85),
                fontsize=9,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="#F1FAEE", ec="#1D3557", lw=1.2, alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="#1D3557", lw=1.2)
            )

    unit_name = "Codon Position" if engine.codon_aligned else "Nucleotide Position"
    ax_river.set_ylabel("Leading Manifold Coordinate ($y_1$)", fontsize=12, fontweight='bold')
    ax_river.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax_river.grid(True, linestyle=':', alpha=0.5)

    if rec_line_handles:
        ax_river.legend(handles=rec_line_handles, loc='upper right', framealpha=0.9)

    # 4. Lower Panel: Instantaneous Velocity / Phase Transition
    velocities = flow_res["velocities"]  # [T - 1, N]
    mid_points = cutpoints[:-1] if len(cutpoints) > 1 else cutpoints

    # Mean background velocity
    mean_bg_vel = np.mean(velocities, axis=1)
    ax_vel.plot(mid_points, mean_bg_vel, color='#457B9D', linewidth=1.5, label="Cohort Mean Velocity")

    # Plot velocity for recombinants
    for i, t in enumerate(taxa):
        if t in recs_in_events:
            ax_vel.plot(
                mid_points,
                velocities[:, i],
                color='#E63946',
                linewidth=2.5,
                label=f"Velocity: {t}"
            )

    ax_vel.set_xlabel(unit_name, fontsize=12, fontweight='bold')
    ax_vel.set_ylabel("Manifold Velocity", fontsize=10, fontweight='bold')
    ax_vel.grid(True, linestyle=':', alpha=0.5)
    ax_vel.legend(loc='upper right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    return output_path
