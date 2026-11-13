"""Visualization helpers for Magratheamorph outputs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .lineage import build_lineage_layout
from .model import SpeciationModel
from .zones import ZoneMap


def species_by_zone_at(model: SpeciationModel, time: float) -> dict[int, list[int]]:
    """Return sorted species IDs present in each zone at ``time``."""

    species_by_zone: dict[int, set[int]] = defaultdict(set)
    for row in model.range_table():
        if row["time"] == time:
            species_by_zone[row["zone_id"]].add(row["species_id"])
    return {zone_id: sorted(species_ids) for zone_id, species_ids in species_by_zone.items()}


def plot_zone_map(ax, zone_map: ZoneMap, *, show_labels: bool = True, cmap: str = "tab20"):
    """Plot zone IDs for one extracted zone map."""

    image = _plot_label_raster(ax, zone_map.labels, reference_labels=zone_map.labels, cmap=cmap)
    if show_labels:
        _label_zones(ax, zone_map.labels, {zone_id: str(zone_id) for zone_id in zone_map.zones})
    ax.set_title(f"Zone IDs, time {zone_map.time:g}")
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def plot_species_map(
    ax,
    model: SpeciationModel,
    zone_map: ZoneMap,
    *,
    max_labels_per_zone: int = 8,
    show_labels: bool = True,
    cmap: str = "tab20",
):
    """Plot species occupancy for one zone map.

    Colors use the first species ID present in a zone. Text labels show stacked
    species IDs with a capped ``+N`` suffix for species-rich zones.
    """

    species_by_zone = species_by_zone_at(model, zone_map.time)
    species_labels = np.full(zone_map.labels.shape, -1, dtype=int)
    text_by_zone = {}
    for zone_id, species_ids in species_by_zone.items():
        if not species_ids:
            continue
        species_labels[zone_map.labels == zone_id] = species_ids[0]
        text_by_zone[zone_id] = format_species_label(species_ids, max_labels=max_labels_per_zone)

    image = _plot_label_raster(ax, species_labels, reference_labels=zone_map.labels, cmap=cmap)
    if show_labels:
        _label_zones(ax, zone_map.labels, text_by_zone)
    ax.set_title(f"Species IDs, time {zone_map.time:g}")
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def plot_richness_map(
    ax,
    model: SpeciationModel,
    zone_map: ZoneMap,
    *,
    show_labels: bool = True,
    cmap: str = "magma",
):
    """Plot species richness by zone for one time step."""

    richness = model.richness_map_at(zone_map)
    masked = np.ma.masked_where(zone_map.labels < 0, richness)
    image = ax.imshow(masked, cmap=cmap, interpolation="nearest", vmin=0)
    _shade_background(ax, zone_map.labels)
    if show_labels:
        text_by_zone = {zone_id: str(value) for zone_id, value in model.richness_by_zone_at(zone_map.time).items()}
        _label_zones(ax, zone_map.labels, text_by_zone)
    ax.set_title(f"Species richness, time {zone_map.time:g}")
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def plot_parent_daughter_tree(
    ax,
    species_rows: list[dict[str, Any]],
    *,
    end_time: float | None = None,
    involved_only: bool = True,
    label_mode: str = "all",
    max_labeled_species: int = 10000,
    style: str = "rectangular",
    display_time_scale: float = 1.0,
    time_unit: str = "year",
    label_position: str = "end",
    label_fontsize: float = 6.5,
    time_block_cmap: str | None = "Greens",
    time_block_count: int = 5,
    show_time_blocks: bool = True,
    compact_extinct: bool = False,
    show_extinct_labels: bool = True,
    extinct_label_suffix: str = "",
    split_color: str = "tab:red",
    split_marker: bool = True,
    split_marker_color: str | None = None,
    split_marker_size: float = 4.2,
    split_linewidth: float = 1.4,
    extant_color: str = "0.12",
    extinct_color: str = "0.62",
    label_extinct: bool | None = None,
):
    """Plot a time-calibrated parent-daughter species tree."""

    ax.clear()
    if label_mode not in {"terminal", "all", "none"}:
        raise ValueError("label_mode must be 'terminal', 'all', or 'none'")
    if max_labeled_species < 0:
        raise ValueError("max_labeled_species must be non-negative")
    if style not in {"rectangular", "semicircle"}:
        raise ValueError("style must be 'rectangular' or 'semicircle'")
    if label_position not in {"center", "end", "both"}:
        raise ValueError("label_position must be 'center', 'end', or 'both'")
    if label_fontsize <= 0:
        raise ValueError("label_fontsize must be positive")
    if time_block_count < 1:
        raise ValueError("time_block_count must be at least 1")
    if display_time_scale <= 0:
        raise ValueError("display_time_scale must be positive")
    if split_linewidth <= 0:
        raise ValueError("split_linewidth must be positive")
    if split_marker_size <= 0:
        raise ValueError("split_marker_size must be positive")
    if split_marker_color is None:
        split_marker_color = split_color
    if label_extinct is not None:
        show_extinct_labels = label_extinct

    layout = build_lineage_layout(
        species_rows,
        end_time=end_time,
        involved_only=involved_only,
        compact_extinct=compact_extinct,
    )
    nodes = layout["nodes"]
    branches = layout["branches"]
    if not nodes:
        ax.text(0.5, 0.5, "No daughter species generated", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return layout

    if style == "semicircle":
        _plot_semicircular_tree(
            ax,
            nodes,
            branches,
            label_mode=label_mode,
            max_labeled_species=max_labeled_species,
            display_time_scale=display_time_scale,
            time_unit=time_unit,
            label_position=label_position,
            label_fontsize=label_fontsize,
            time_block_cmap=time_block_cmap,
            time_block_count=time_block_count,
            show_time_blocks=show_time_blocks,
            show_extinct_labels=show_extinct_labels,
            extinct_label_suffix=extinct_label_suffix,
            split_color=split_color,
            split_marker=split_marker,
            split_marker_color=split_marker_color,
            split_marker_size=split_marker_size,
            split_linewidth=split_linewidth,
            extant_color=extant_color,
            extinct_color=extinct_color,
        )
    else:
        _plot_rectangular_tree(
            ax,
            nodes,
            branches,
            label_mode=label_mode,
            max_labeled_species=max_labeled_species,
            display_time_scale=display_time_scale,
            time_unit=time_unit,
            label_position=label_position,
            label_fontsize=label_fontsize,
            time_block_cmap=time_block_cmap,
            time_block_count=time_block_count,
            show_time_blocks=show_time_blocks,
            show_extinct_labels=show_extinct_labels,
            extinct_label_suffix=extinct_label_suffix,
            split_color=split_color,
            split_marker=split_marker,
            split_marker_color=split_marker_color,
            split_marker_size=split_marker_size,
            split_linewidth=split_linewidth,
            extant_color=extant_color,
            extinct_color=extinct_color,
        )
    return layout


def _plot_rectangular_tree(
    ax,
    nodes: dict[int, dict[str, Any]],
    branches: list[dict[str, Any]],
    *,
    label_mode: str,
    max_labeled_species: int,
    display_time_scale: float,
    time_unit: str,
    label_position: str,
    label_fontsize: float,
    time_block_cmap: str | None,
    time_block_count: int,
    show_time_blocks: bool,
    show_extinct_labels: bool,
    extinct_label_suffix: str,
    split_color: str,
    split_marker: bool,
    split_marker_color: str,
    split_marker_size: float,
    split_linewidth: float,
    extant_color: str,
    extinct_color: str,
) -> None:
    labeled_species = set(
        _labeled_tree_species(
            nodes,
            label_mode=label_mode,
            max_labeled_species=max_labeled_species,
            show_extinct_labels=show_extinct_labels,
        )
    )
    x0 = min(node["branch_start"] for node in nodes.values()) * display_time_scale
    x1 = max(node["branch_end"] for node in nodes.values()) * display_time_scale
    span = max(x1 - x0, 1.0)
    if show_time_blocks:
        _draw_rectangular_time_blocks(ax, x0, x1, time_block_count, time_block_cmap)

    for branch in branches:
        if branch["kind"] != "branch":
            continue
        node = nodes[branch["species_id"]]
        color = extant_color
        if node["extinction_time"] is not None:
            color = extinct_color
        ax.hlines(
            branch["y0"],
            branch["x0"] * display_time_scale,
            branch["x1"] * display_time_scale,
            color=color,
            linewidth=1.5,
            zorder=2,
        )
        if branch["species_id"] in labeled_species:
            label = f"{branch['species_id']}"
            if node["extinction_time"] is not None:
                label += extinct_label_suffix
            if label_position in {"center", "both"}:
                label_x = 0.5 * (branch["x0"] + branch["x1"]) * display_time_scale
                _draw_branch_label(
                    ax, label_x, branch["y0"] + 0.08, label, color, ha="center", fontsize=label_fontsize
                )
            if label_position in {"end", "both"}:
                end_x = branch["x1"] * display_time_scale
                _draw_branch_label(
                    ax, end_x, branch["y0"] - 0.10, label, color, ha="right", fontsize=label_fontsize
                )

    for branch in branches:
        if branch["kind"] != "split":
            continue
        x = branch["x0"] * display_time_scale
        ax.vlines(
            x,
            min(branch["y0"], branch["y1"]),
            max(branch["y0"], branch["y1"]),
            color=split_color,
            linewidth=split_linewidth,
            linestyle="--",
            zorder=3,
        )
        if split_marker:
            ax.plot(x, branch["y0"], "o", color=split_marker_color, markersize=split_marker_size, zorder=4)
            ax.plot(x, branch["y1"], "o", color=split_marker_color, markersize=split_marker_size, zorder=4)

    ax.set_title("Parent-daughter species tree")
    ax.set_xlabel(f"time ({time_unit})" if time_unit else "time")
    ax.set_yticks([])
    ax.set_xlim(x0 - 0.03 * span, x1 + 0.18 * span)
    max_y = max(node["y"] for node in nodes.values())
    ax.set_ylim(-0.8, max_y + 0.8)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def _plot_semicircular_tree(
    ax,
    nodes: dict[int, dict[str, Any]],
    branches: list[dict[str, Any]],
    *,
    label_mode: str,
    max_labeled_species: int,
    display_time_scale: float,
    time_unit: str,
    label_position: str,
    label_fontsize: float,
    time_block_cmap: str | None,
    time_block_count: int,
    show_time_blocks: bool,
    show_extinct_labels: bool,
    extinct_label_suffix: str,
    split_color: str,
    split_marker: bool,
    split_marker_color: str,
    split_marker_size: float,
    split_linewidth: float,
    extant_color: str,
    extinct_color: str,
) -> None:
    raw_x0 = min(node["branch_start"] for node in nodes.values())
    raw_x1 = max(node["branch_end"] for node in nodes.values())
    x0 = raw_x0 * display_time_scale
    x1 = raw_x1 * display_time_scale
    span = max(x1 - x0, 1.0)
    max_y = max(node["y"] for node in nodes.values())
    y_span = max(max_y, 1.0)

    def radius(time: float) -> float:
        scaled_time = time * display_time_scale
        return 0.18 + 0.82 * ((scaled_time - x0) / span)

    def theta(y: float) -> float:
        return np.pi * (1.0 - y / y_span)

    def xy(time: float, y: float) -> tuple[float, float]:
        r = radius(time)
        angle = theta(y)
        return r * np.cos(angle), r * np.sin(angle)

    if show_time_blocks:
        _draw_semicircular_time_blocks(ax, raw_x0, raw_x1, display_time_scale, radius, time_block_count, time_block_cmap)
    _draw_semicircular_time_scale(ax, raw_x0, raw_x1, display_time_scale, radius, time_unit)
    labeled_species = set(
        _labeled_tree_species(
            nodes,
            label_mode=label_mode,
            max_labeled_species=max_labeled_species,
            show_extinct_labels=show_extinct_labels,
        )
    )

    for branch in branches:
        if branch["kind"] != "branch":
            continue
        node = nodes[branch["species_id"]]
        color = extant_color
        if node["extinction_time"] is not None:
            color = extinct_color
        x_start, y_start = xy(branch["x0"], branch["y0"])
        x_end, y_end = xy(branch["x1"], branch["y1"])
        ax.plot([x_start, x_end], [y_start, y_end], color=color, linewidth=1.4, zorder=2)
        if branch["species_id"] in labeled_species:
            angle = theta(branch["y0"])
            rotation = np.degrees(angle)
            if 90 < rotation < 270:
                rotation += 180
            label = f"{branch['species_id']}"
            if node["extinction_time"] is not None:
                label += extinct_label_suffix
            if label_position in {"center", "both"}:
                label_time = 0.5 * (branch["x0"] + branch["x1"])
                x_text, y_text = xy(label_time, branch["y0"])
                _draw_branch_label(
                    ax, x_text, y_text, label, color, ha="center", rotation=rotation, fontsize=label_fontsize
                )
            if label_position in {"end", "both"}:
                x_end_label, y_end_label = xy(branch["x1"], branch["y1"])
                _draw_branch_label(
                    ax,
                    x_end_label * 1.02,
                    y_end_label * 1.02,
                    label,
                    color,
                    ha="center",
                    rotation=rotation,
                    fontsize=label_fontsize,
                )

    for branch in branches:
        if branch["kind"] != "split":
            continue
        split_radius = radius(branch["x0"])
        angle0 = theta(branch["y0"])
        angle1 = theta(branch["y1"])
        angles = np.linspace(angle0, angle1, 40)
        ax.plot(
            split_radius * np.cos(angles),
            split_radius * np.sin(angles),
            color=split_color,
            linewidth=split_linewidth,
            linestyle="--",
            zorder=3,
        )
        if split_marker:
            ax.plot(
                [split_radius * np.cos(angle0), split_radius * np.cos(angle1)],
                [split_radius * np.sin(angle0), split_radius * np.sin(angle1)],
                "o",
                color=split_marker_color,
                markersize=split_marker_size,
                zorder=4,
            )

    ax.set_title("Parent-daughter species tree")
    ax.set_aspect("equal")
    ax.set_axis_off()


def _draw_semicircular_time_scale(
    ax,
    raw_x0: float,
    raw_x1: float,
    display_time_scale: float,
    radius,
    time_unit: str,
) -> None:
    x0 = raw_x0 * display_time_scale
    x1 = raw_x1 * display_time_scale
    ticks = np.linspace(x0, x1, 5)
    angles = np.linspace(0, np.pi, 160)
    for tick in ticks:
        r = radius(tick / display_time_scale)
        ax.plot(
            r * np.cos(angles),
            r * np.sin(angles),
            color="0.86",
            linewidth=0.7,
            zorder=0,
        )
        ax.text(
            r,
            -0.035,
            f"{tick:g} {time_unit}",
            ha="center",
            va="top",
            fontsize=7,
            color="0.45",
        )


def _draw_rectangular_time_blocks(
    ax,
    x0: float,
    x1: float,
    time_block_count: int,
    cmap_name: str | None,
) -> None:
    cmap = _optional_colormap(cmap_name)
    edges = np.linspace(x0, x1, time_block_count + 1)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        color = cmap((index + 0.5) / time_block_count) if cmap is not None else "0.94"
        ax.axvspan(left, right, facecolor=color, alpha=0.22, edgecolor="none", zorder=0)


def _draw_semicircular_time_blocks(
    ax,
    raw_x0: float,
    raw_x1: float,
    display_time_scale: float,
    radius,
    time_block_count: int,
    cmap_name: str | None,
) -> None:
    cmap = _optional_colormap(cmap_name)
    raw_edges = np.linspace(raw_x0, raw_x1, time_block_count + 1)
    angles = np.linspace(0, np.pi, 220)
    for index, (inner_time, outer_time) in enumerate(zip(raw_edges[:-1], raw_edges[1:])):
        inner_radius = radius(inner_time)
        outer_radius = radius(outer_time)
        color = cmap((index + 0.5) / time_block_count) if cmap is not None else "0.94"
        x_values = np.concatenate([outer_radius * np.cos(angles), inner_radius * np.cos(angles[::-1])])
        y_values = np.concatenate([outer_radius * np.sin(angles), inner_radius * np.sin(angles[::-1])])
        ax.fill(
            x_values,
            y_values,
            color=color,
            alpha=0.22,
            edgecolor="none",
            zorder=0,
        )


def _optional_colormap(cmap_name: str | None):
    if cmap_name is None:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt.get_cmap(cmap_name)


def _draw_branch_label(
    ax,
    x: float,
    y: float,
    label: str,
    color: str,
    *,
    ha: str,
    rotation: float = 0.0,
    fontsize: float = 6.5,
) -> None:
    ax.text(
        x,
        y,
        label,
        va="center",
        ha=ha,
        fontsize=fontsize,
        color=color,
        rotation=rotation,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 0.35},
        zorder=4,
    )


def _labeled_tree_species(
    nodes: dict[int, dict[str, Any]],
    *,
    label_mode: str,
    max_labeled_species: int,
    show_extinct_labels: bool,
) -> list[int]:
    if label_mode == "none" or max_labeled_species == 0:
        return []
    if label_mode == "all":
        candidates = list(nodes)
    else:
        candidates = [
            species_id
            for species_id, node in nodes.items()
            if node["is_terminal"] or node["extinction_time"] is not None
        ]
    if not show_extinct_labels:
        candidates = [species_id for species_id in candidates if nodes[species_id]["extinction_time"] is None]
    candidates.sort(key=lambda species_id: (-nodes[species_id]["y"], species_id))
    return candidates[:max_labeled_species]


def format_species_label(species_ids: list[int], *, max_labels: int = 8) -> str:
    """Format species IDs as a stacked map label with an optional cap."""

    if max_labels < 1:
        raise ValueError("max_labels must be at least 1")
    values = sorted(species_ids)
    shown = values[:max_labels]
    label = "\n".join(str(value) for value in shown)
    hidden = len(values) - len(shown)
    if hidden > 0:
        label = f"{label}\n+{hidden}" if label else f"+{hidden}"
    return label


def _ordered_parent_daughter_rows(species_rows: list[dict[str, Any]], *, involved_only: bool) -> list[dict[str, Any]]:
    rows_by_id = {row["species_id"]: row for row in species_rows}
    daughters_by_parent: dict[int, list[int]] = defaultdict(list)
    involved = set()
    for row in species_rows:
        parent_id = row["parent_id"]
        if parent_id is None:
            continue
        daughters_by_parent[parent_id].append(row["species_id"])
        involved.add(parent_id)
        involved.add(row["species_id"])

    if not involved_only:
        involved = set(rows_by_id)

    order = []
    visited = set()

    def visit(species_id: int) -> None:
        if species_id in visited or species_id not in rows_by_id or species_id not in involved:
            return
        visited.add(species_id)
        order.append(species_id)
        daughters = sorted(
            daughters_by_parent.get(species_id, []),
            key=lambda daughter_id: (rows_by_id[daughter_id]["origin_time"], daughter_id),
        )
        for daughter_id in daughters:
            visit(daughter_id)

    root_ids = sorted(
        involved,
        key=lambda species_id: (rows_by_id[species_id]["origin_time"], species_id),
    )
    for species_id in root_ids:
        visit(species_id)

    return [rows_by_id[species_id] for species_id in order]


def _default_end_time(species_rows: list[dict[str, Any]]) -> float:
    times = []
    for row in species_rows:
        times.append(row["origin_time"])
        if row["extinction_time"] is not None:
            times.append(row["extinction_time"])
    return max(times) if times else 0


def _plot_label_raster(ax, labels: np.ndarray, *, reference_labels: np.ndarray, cmap: str):
    masked = np.ma.masked_where(reference_labels < 0, labels)
    image = ax.imshow(masked, cmap=cmap, interpolation="nearest")
    _shade_background(ax, reference_labels)
    return image


def _shade_background(ax, reference_labels: np.ndarray) -> None:
    background = reference_labels < 0
    ax.imshow(background, cmap="Greys", alpha=background * 0.25, interpolation="nearest")


def _label_zones(ax, labels: np.ndarray, text_by_zone: dict[int, str]) -> None:
    for zone_id, text in text_by_zone.items():
        if not text:
            continue
        rows, cols = np.where(labels == zone_id)
        if len(rows) == 0:
            continue
        ax.text(
            float(np.mean(cols)),
            float(np.mean(rows)),
            text,
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "black", "alpha": 0.35, "linewidth": 0},
        )
