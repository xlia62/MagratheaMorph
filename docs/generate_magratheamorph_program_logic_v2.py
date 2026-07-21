"""Generate a publication-style logic figure for Magratheamorph."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUT_DIR / "magratheamorph_program_logic.png"
PDF_PATH = OUT_DIR / "magratheamorph_program_logic.pdf"

ZONE_COLORS = {
    -1: "#f3f3f3",
    0: "#67a9cf",
    1: "#ef8a62",
    2: "#91cf60",
    3: "#c2a5cf",
}


def main() -> None:
    fig = plt.figure(figsize=(15.5, 8.8), facecolor="white")
    fig.text(
        0.5,
        0.965,
        "Magratheamorph program logic",
        ha="center",
        va="top",
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.925,
        "DEM or VTK time series -> river habitat zones -> zone history -> allopatric/vicariance species evolution -> outputs",
        ha="center",
        va="top",
        fontsize=11,
        color="0.30",
    )

    axes = {
        "A": fig.add_axes([0.045, 0.58, 0.28, 0.28]),
        "B": fig.add_axes([0.365, 0.58, 0.28, 0.28]),
        "C": fig.add_axes([0.685, 0.58, 0.27, 0.28]),
        "D": fig.add_axes([0.045, 0.14, 0.59, 0.34]),
        "E": fig.add_axes([0.685, 0.14, 0.28, 0.34]),
    }
    for ax in axes.values():
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    panel_input(axes["A"])
    panel_zone_extraction(axes["B"])
    panel_zone_tracking(axes["C"])
    panel_species_logic(axes["D"])
    panel_outputs(axes["E"])
    draw_between_panel_arrows(fig)

    fig.savefig(PNG_PATH, dpi=260, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)
    print(PNG_PATH)
    print(PDF_PATH)


def panel_header(ax, letter: str, title: str) -> None:
    ax.text(0.0, 1.02, f"{letter}. {title}", fontsize=12.5, fontweight="bold", va="bottom")


def panel_input(ax) -> None:
    panel_header(ax, "A", "Inputs: evolving topography")
    frames = [
        np.array([[6, 5, 4, 3], [5, 4, 3, 2], [4, 3, 2, 1], [3, 2, 1, 0]]),
        np.array([[6, 5, 5, 3], [5, 4, 3, 2], [4, 4, 2, 1], [3, 2, 1, 0]]),
        np.array([[6, 6, 5, 4], [5, 4, 4, 2], [4, 3, 2, 1], [2, 2, 1, 0]]),
    ]
    for i, (x, frame) in enumerate(zip([0.06, 0.37, 0.68], frames)):
        draw_matrix(ax, x, 0.44, 0.23, 0.32, frame, cmap="terrain", vmin=0, vmax=6)
        ax.text(x + 0.115, 0.80, f"t{i}", ha="center", fontsize=10, fontweight="bold")
    bullet_box(
        ax,
        0.08,
        0.11,
        0.84,
        0.22,
        ["ASCII/matrix DEM", "FastScape-ASPECT VTK", "elevation, drainage area, catchment"],
        face="#e5f2fb",
    )


def panel_zone_extraction(ax) -> None:
    panel_header(ax, "B", "Habitat mask -> zones")
    dem = np.array([[5, 4, 4, 3, 2], [4, 3, 3, 2, 1], [3, 3, 2, 2, 1], [2, 1, 1, 1, 0]])
    mask = np.array([[0, 1, 1, 0, 0], [1, 1, 0, 0, 1], [1, 0, 0, 1, 1], [0, 0, 1, 1, 0]])
    zones = np.array([[-1, 0, 0, -1, -1], [0, 0, -1, -1, 1], [0, -1, -1, 1, 1], [-1, -1, 1, 1, -1]])
    draw_matrix(ax, 0.02, 0.48, 0.24, 0.30, dem, cmap="terrain", vmin=0, vmax=5)
    draw_matrix(ax, 0.38, 0.48, 0.24, 0.30, mask, cmap="Greys", vmin=0, vmax=1)
    draw_zone_matrix(ax, 0.74, 0.48, 0.24, 0.30, zones)
    arrow(ax, (0.28, 0.63), (0.36, 0.63))
    arrow(ax, (0.64, 0.63), (0.72, 0.63))
    ax.text(0.14, 0.82, "DEM/fields", ha="center", fontsize=9, fontweight="bold")
    ax.text(0.50, 0.82, "habitat", ha="center", fontsize=9, fontweight="bold")
    ax.text(0.86, 0.82, "zone IDs", ha="center", fontsize=9, fontweight="bold")
    bullet_box(
        ax,
        0.07,
        0.10,
        0.86,
        0.25,
        ["habitat mode: drainage, elevation, or both", "4/8-neighbor connectivity", "minimum zone size and gap tolerance"],
        face="#edf8e9",
    )


def panel_zone_tracking(ax) -> None:
    panel_header(ax, "C", "Zone tracking")
    z0 = np.array([[-1, 0, 0, -1], [0, 0, -1, 1], [0, -1, 1, 1], [-1, -1, 1, -1]])
    z1 = np.array([[-1, 0, 0, 0], [0, 0, -1, 1], [0, -1, 1, 1], [-1, 2, 2, -1]])
    draw_zone_matrix(ax, 0.08, 0.51, 0.32, 0.32, z0)
    draw_zone_matrix(ax, 0.60, 0.51, 0.32, 0.32, z1)
    arrow(ax, (0.43, 0.67), (0.57, 0.67))
    ax.text(0.24, 0.86, "t", ha="center", fontsize=10, fontweight="bold")
    ax.text(0.76, 0.86, "t+1", ha="center", fontsize=10, fontweight="bold")
    bullet_box(
        ax,
        0.08,
        0.12,
        0.84,
        0.27,
        ["overlap preserves zone ID", "new non-overlap creates new ID", "capture/contact transfers species, not identity"],
        face="#fff4ce",
    )


def panel_species_logic(ax) -> None:
    panel_header(ax, "D", "Allopatric/vicariance species rules")
    times = [0.08, 0.32, 0.58, 0.82]
    headings = ["t0", "t1 split", "t2 isolated", "t3 contact"]
    states = [
        [("Z0", "sp0", 0.50, "#67a9cf")],
        [("Z0a", "sp0", 0.38, "#67a9cf"), ("Z0b", "sp0", 0.62, "#67a9cf")],
        [("Z0a", "sp0", 0.38, "#67a9cf"), ("Z3", "sp1", 0.62, "#c2a5cf")],
        [("Z0 + Z3", "sp0, sp1", 0.50, "#9ecae1")],
    ]
    for x, heading, zones in zip(times, headings, states):
        ax.text(x, 0.87, heading, ha="center", fontsize=9.5, fontweight="bold")
        for zone, species, y, color in zones:
            ax.add_patch(plt.Circle((x, y), 0.065, facecolor=color, edgecolor="0.25", linewidth=1.2))
            ax.text(x, y, f"{zone}\n{species}", ha="center", va="center", fontsize=7.5)
    for x0, x1 in zip(times[:-1], times[1:]):
        arrow(ax, (x0 + 0.075, 0.50), (x1 - 0.075, 0.50))
    ax.text(0.45, 0.76, "isolation", color="#b2182b", fontsize=9)
    ax.plot([0.45, 0.45], [0.57, 0.73], color="#b2182b", linestyle="--", linewidth=1.4)
    ax.text(0.70, 0.75, "secondary contact:\ncoexist, no merger", color="#2166ac", fontsize=8.5, ha="center")

    # Mini lineage tree.
    y0, y1 = 0.16, 0.28
    ax.plot([0.12, 0.90], [y0, y0], color="0.1", linewidth=2.0)
    ax.plot([0.36, 0.90], [y1, y1], color="0.1", linewidth=2.0)
    ax.plot([0.36, 0.36], [y0, y1], color="tab:red", linestyle="--", linewidth=1.5)
    ax.plot([0.36, 0.36], [y0, y1], "o", color="tab:red", markersize=4)
    ax.text(0.92, y0, "sp0", va="center", fontsize=8)
    ax.text(0.92, y1, "sp1", va="center", fontsize=8)
    ax.text(0.50, 0.05, "species follows persistent zone; isolated daughter zone can become a new species", ha="center", fontsize=8.8)


def panel_outputs(ax) -> None:
    panel_header(ax, "E", "Outputs and diagnostics")
    output_box(ax, 0.06, 0.70, "VTK fields", ["magrathea_zone_id", "species_richness", "target species presence"])
    output_box(ax, 0.06, 0.47, "CSV table", ["time", "zone_id", "species_count", "species_ids"])
    output_box(ax, 0.06, 0.24, "Figures", ["zone map", "species/richness map", "parent-daughter tree"])
    legend_inset(ax)


def legend_inset(ax) -> None:
    ax.plot([0.08, 0.25], [0.11, 0.11], color="0.1", linewidth=2)
    ax.text(0.29, 0.11, "species lifetime", va="center", fontsize=7.2)
    ax.plot([0.08, 0.25], [0.04, 0.04], color="0.62", linewidth=2)
    ax.text(0.29, 0.04, "extinct lifetime", va="center", fontsize=7.2)
    ax.plot([0.67, 0.67], [0.02, 0.13], color="tab:red", linestyle="--", linewidth=1.3)
    ax.plot([0.67, 0.67], [0.02, 0.13], "o", color="tab:red", markersize=3.2)
    ax.text(0.71, 0.075, "true speciation", va="center", fontsize=7.2)


def output_box(ax, x, y, title, lines) -> None:
    patch = FancyBboxPatch(
        (x, y),
        0.88,
        0.17,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor="#d9f0d3",
        edgecolor="0.25",
        linewidth=1.0,
    )
    ax.add_patch(patch)
    ax.text(x + 0.03, y + 0.135, title, ha="left", va="top", fontsize=8.4, fontweight="bold")
    ax.text(x + 0.03, y + 0.092, "\n".join(lines), ha="left", va="top", fontsize=6.9, linespacing=1.02)


def bullet_box(ax, x, y, w, h, lines, face) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=face,
        edgecolor="0.45",
        linewidth=0.9,
    )
    ax.add_patch(patch)
    ax.text(x + 0.035, y + h - 0.045, "\n".join(f"- {line}" for line in lines), ha="left", va="top", fontsize=8.2, linespacing=1.25)


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
                    linewidth=0.7,
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
                    linewidth=0.75,
                )
            )
            if value >= 0:
                ax.text(x + (c + 0.5) * w / cols, y + (rows - r - 0.5) * h / rows, str(value), ha="center", va="center", fontsize=7)


def arrow(ax, start, end) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.45,
            color="0.20",
        )
    )


def draw_between_panel_arrows(fig) -> None:
    arrows = [
        ((0.326, 0.72), (0.360, 0.72), None, 1.6, "0.20"),
        ((0.646, 0.72), (0.680, 0.72), None, 1.6, "0.20"),
        ((0.805, 0.535), (0.555, 0.535), None, 1.15, "0.35"),
        ((0.635, 0.31), (0.685, 0.31), None, 1.6, "0.20"),
    ]
    for start, end, connectionstyle, linewidth, color in arrows:
        fig.patches.append(
            FancyArrowPatch(
                start,
                end,
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=linewidth,
                color=color,
                connectionstyle=connectionstyle,
            )
        )


if __name__ == "__main__":
    main()
