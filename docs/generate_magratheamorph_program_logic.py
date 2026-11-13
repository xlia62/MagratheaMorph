"""Generate a methods-style conceptual figure for Magratheamorph."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUT_DIR / "magratheamorph_program_logic.png"
PDF_PATH = OUT_DIR / "magratheamorph_program_logic.pdf"


ZONE_COLORS = {
    -1: "#f7f7f7",
    0: "#67a9cf",
    1: "#ef8a62",
    2: "#91cf60",
    3: "#c2a5cf",
    4: "#ffd92f",
}


def main() -> None:
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.05, 1.2, 0.95],
        width_ratios=[1.1, 1.25, 1.0],
        left=0.035,
        right=0.985,
        bottom=0.055,
        top=0.92,
        hspace=0.28,
        wspace=0.18,
    )

    fig.suptitle(
        "Magratheamorph program logic: topographic change to river-fish lineages",
        fontsize=18,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.938,
        "Inputs define habitat zones through time; zone reorganization drives allopatric/vicariance speciation.",
        ha="center",
        fontsize=10.5,
        color="0.28",
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, :2])
    ax_e = fig.add_subplot(gs[1, 2])
    ax_f = fig.add_subplot(gs[2, :])

    panel_a_input(ax_a)
    panel_b_zone_extraction(ax_b)
    panel_c_temporal_tracking(ax_c)
    panel_d_species_logic(ax_d)
    panel_e_outputs(ax_e)
    panel_f_legend(ax_f)

    add_panel_arrows(fig, [ax_a, ax_b, ax_c, ax_d, ax_e])

    fig.savefig(PNG_PATH, dpi=260, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)
    print(PNG_PATH)
    print(PDF_PATH)


def panel_title(ax, label: str, title: str) -> None:
    ax.set_axis_off()
    ax.text(
        0.0,
        1.03,
        f"{label}. {title}",
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        va="bottom",
    )


def panel_a_input(ax) -> None:
    panel_title(ax, "A", "Input topography time series")
    frames = [
        np.array([[6, 5, 4, 3], [5, 4, 3, 2], [4, 3, 2, 1], [3, 2, 1, 0]]),
        np.array([[6, 5, 5, 3], [5, 4, 3, 2], [4, 4, 2, 1], [3, 2, 1, 0]]),
        np.array([[6, 6, 5, 4], [5, 4, 4, 2], [4, 3, 2, 1], [2, 2, 1, 0]]),
    ]
    xs = [0.05, 0.36, 0.67]
    for i, (x, data) in enumerate(zip(xs, frames)):
        draw_matrix(ax, x, 0.35, 0.24, 0.34, data, cmap="terrain", vmin=0, vmax=6)
        ax.text(x + 0.12, 0.73, f"t{i}", ha="center", fontsize=10, fontweight="bold")
    ax.text(
        0.5,
        0.18,
        "ASCII DEM / matrix DEM / FastScape-ASPECT VTK\nfields: elevation, drainage area, catchment ID",
        ha="center",
        fontsize=9.5,
        linespacing=1.25,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def panel_b_zone_extraction(ax) -> None:
    panel_title(ax, "B", "Habitat mask and zone extraction")
    dem = np.array([[5, 4, 4, 3, 2], [4, 3, 3, 2, 1], [3, 3, 2, 2, 1], [2, 1, 1, 1, 0]])
    mask = np.array([[0, 1, 1, 0, 0], [1, 1, 0, 0, 1], [1, 0, 0, 1, 1], [0, 0, 1, 1, 0]])
    zones = np.array([[-1, 0, 0, -1, -1], [0, 0, -1, -1, 1], [0, -1, -1, 1, 1], [-1, -1, 1, 1, -1]])
    draw_matrix(ax, 0.02, 0.42, 0.25, 0.32, dem, cmap="terrain", vmin=0, vmax=5)
    draw_matrix(ax, 0.37, 0.42, 0.25, 0.32, mask, cmap="Greys", vmin=0, vmax=1)
    draw_zone_matrix(ax, 0.72, 0.42, 0.25, 0.32, zones)
    arrow(ax, (0.29, 0.58), (0.35, 0.58))
    arrow(ax, (0.64, 0.58), (0.70, 0.58))
    ax.text(0.145, 0.78, "DEM/VTK frame", ha="center", fontsize=9, fontweight="bold")
    ax.text(0.495, 0.78, "habitat mask", ha="center", fontsize=9, fontweight="bold")
    ax.text(0.845, 0.78, "zones", ha="center", fontsize=9, fontweight="bold")
    ax.text(
        0.5,
        0.16,
        "habitat mode: drainage, elevation, or both\nconnectivity, minimum zone size, and gap tolerance define connected zones",
        ha="center",
        fontsize=9.3,
        linespacing=1.25,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def panel_c_temporal_tracking(ax) -> None:
    panel_title(ax, "C", "Track zone identity through time")
    z0 = np.array([[-1, 0, 0, -1], [0, 0, -1, 1], [0, -1, 1, 1], [-1, -1, 1, -1]])
    z1 = np.array([[-1, 0, 0, 0], [0, 0, -1, 1], [0, -1, 1, 1], [-1, 2, 2, -1]])
    draw_zone_matrix(ax, 0.08, 0.48, 0.32, 0.32, z0)
    draw_zone_matrix(ax, 0.60, 0.48, 0.32, 0.32, z1)
    arrow(ax, (0.43, 0.64), (0.57, 0.64))
    ax.text(0.24, 0.83, "zones at t", ha="center", fontsize=9, fontweight="bold")
    ax.text(0.76, 0.83, "zones at t+1", ha="center", fontsize=9, fontweight="bold")
    ax.text(
        0.5,
        0.27,
        "overlap preserves IDs\nnew non-overlap creates new zone ID\ncontact/capture transfers species without merging lineages",
        ha="center",
        fontsize=9.3,
        linespacing=1.25,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def panel_d_species_logic(ax) -> None:
    panel_title(ax, "D", "Allopatric/vicariance species evolution")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    times = [0.10, 0.36, 0.62, 0.88]
    labels = ["t0", "t1 split", "t2 isolation", "t3 contact"]
    zone_sets = [
        [("Z0", "sp0", 0.5, "#67a9cf")],
        [("Z0a", "sp0", 0.35, "#67a9cf"), ("Z0b", "sp0", 0.65, "#67a9cf")],
        [("Z0a", "sp0", 0.35, "#67a9cf"), ("Z3", "sp1", 0.65, "#c2a5cf")],
        [("Z0 + Z3", "sp0, sp1", 0.5, "#9ecae1")],
    ]
    for x, label, zones in zip(times, labels, zone_sets):
        ax.text(x, 0.88, label, ha="center", fontsize=9.5, fontweight="bold")
        for zname, sp, y, color in zones:
            blob(ax, x, y, 0.075, color)
            ax.text(x, y + 0.005, f"{zname}\n{sp}", ha="center", va="center", fontsize=8)
    for x0, x1 in zip(times[:-1], times[1:]):
        arrow(ax, (x0 + 0.09, 0.50), (x1 - 0.09, 0.50))
    ax.text(0.49, 0.72, "isolation", color="#b2182b", fontsize=9, ha="center")
    ax.plot([0.50, 0.50], [0.57, 0.72], color="#b2182b", linestyle="--", linewidth=1.3)
    ax.text(0.75, 0.72, "secondary contact\nno species merger", color="#2166ac", fontsize=9, ha="center")

    # Mini parent-daughter tree.
    x0, x1, x2, x3 = 0.12, 0.38, 0.64, 0.90
    y0, y1 = 0.17, 0.29
    ax.plot([x0, x3], [y0, y0], color="0.1", linewidth=2)
    ax.plot([x1, x3], [y1, y1], color="0.1", linewidth=2)
    ax.plot([x1, x1], [y0, y1], color="tab:red", linestyle="--", linewidth=1.5)
    ax.plot([x1, x1], [y0, y1], "o", color="tab:red", markersize=4)
    ax.text(x3 + 0.02, y0, "sp0", va="center", fontsize=8)
    ax.text(x3 + 0.02, y1, "sp1", va="center", fontsize=8)
    ax.text(0.5, 0.06, "species follows persistent zone; isolated daughter zone can become new species", ha="center", fontsize=9)


def panel_e_outputs(ax) -> None:
    panel_title(ax, "E", "Outputs and diagnostics")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rows = [
        ("VTK fields", "magrathea_zone_id\nmagrathea_species_richness\ntarget species presence"),
        ("CSV table", "time, zone_id,\nspecies_count, species_ids"),
        ("Figures", "zone map, species map,\nrichness map, lineage tree"),
    ]
    y = 0.72
    for title, body in rows:
        rounded_box(ax, (0.12, y - 0.11), 0.76, 0.16, title, body, "#d9f0d3")
        y -= 0.24


def panel_f_legend(ax) -> None:
    panel_title(ax, "F", "How to read Magratheamorph results")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0.05, 0.18], [0.62, 0.62], color="0.1", linewidth=2.3)
    ax.text(0.20, 0.62, "solid black line = species lifetime", va="center", fontsize=9.5)
    ax.plot([0.05, 0.18], [0.42, 0.42], color="0.62", linewidth=2.3)
    ax.text(0.20, 0.42, "solid gray line = extinct species lifetime", va="center", fontsize=9.5)
    ax.plot([0.48, 0.48], [0.34, 0.70], color="tab:red", linestyle="--", linewidth=1.6)
    ax.plot([0.48, 0.48], [0.34, 0.70], "o", color="tab:red", markersize=4.5)
    ax.text(0.52, 0.52, "red dashed connector + marker = true parent-daughter speciation", va="center", fontsize=9.5)
    ax.text(
        0.05,
        0.13,
        "Important: visual crossings in compact tree mode are not speciation events unless they have the red connector/marker.",
        fontsize=9.5,
        color="#7d2e2a",
        weight="bold",
    )


def draw_matrix(ax, x, y, w, h, data, *, cmap, vmin, vmax) -> None:
    rows, cols = data.shape
    cm = plt.get_cmap(cmap)
    for r in range(rows):
        for c in range(cols):
            value = data[r, c]
            color = cm((value - vmin) / (vmax - vmin))
            ax.add_patch(
                Rectangle(
                    (x + c * w / cols, y + (rows - r - 1) * h / rows),
                    w / cols,
                    h / rows,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.6,
                )
            )


def draw_zone_matrix(ax, x, y, w, h, data) -> None:
    rows, cols = data.shape
    for r in range(rows):
        for c in range(cols):
            value = int(data[r, c])
            ax.add_patch(
                Rectangle(
                    (x + c * w / cols, y + (rows - r - 1) * h / rows),
                    w / cols,
                    h / rows,
                    facecolor=ZONE_COLORS.get(value, "#cccccc"),
                    edgecolor="white",
                    linewidth=0.7,
                )
            )
            if value >= 0:
                ax.text(
                    x + (c + 0.5) * w / cols,
                    y + (rows - r - 0.5) * h / rows,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )


def rounded_box(ax, xy, width, height, title, body, color) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=color,
        edgecolor="0.25",
        linewidth=1.0,
    )
    ax.add_patch(patch)
    x, y = xy
    ax.text(x + 0.03, y + height - 0.035, title, ha="left", va="top", fontsize=9.2, fontweight="bold")
    ax.text(x + 0.03, y + 0.035, body, ha="left", va="bottom", fontsize=8.2, linespacing=1.15)


def blob(ax, x, y, radius, color) -> None:
    circle = plt.Circle((x, y), radius, facecolor=color, edgecolor="0.25", linewidth=1.0)
    ax.add_patch(circle)


def arrow(ax, start, end, *, color="0.2") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            color=color,
            linewidth=1.5,
        )
    )


def add_panel_arrows(fig, axes) -> None:
    # Deliberately minimal: internal panel arrows carry the main sequence.
    return None


if __name__ == "__main__":
    main()
