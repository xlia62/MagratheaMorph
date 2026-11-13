"""Lineage tree helpers for Magratheamorph species output."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_lineage_layout(
    species_rows: list[dict[str, Any]],
    end_time: float | None = None,
    *,
    involved_only: bool = False,
    compact_extinct: bool = False,
) -> dict[str, Any]:
    """Build plotting coordinates for species parent-daughter relationships.

    Parameters
    ----------
    species_rows
        Rows from ``SpeciationModel.species_table()``. Each row must contain
        ``species_id``, ``parent_id``, ``origin_time``, and ``extinction_time``.
    end_time
        Time used for extant species branch endpoints. If omitted, the maximum
        recorded origin/extinction time is used.
    involved_only
        If True, include only species that participate in at least one
        parent-daughter relationship.
    compact_extinct
        If True, pack non-overlapping species lifetimes into shared plotting
        lanes. A lane is reusable only after the previous branch in that lane
        ends.

    Returns
    -------
    dict
        ``nodes`` maps species IDs to row metadata plus y-position. ``branches``
        contains horizontal branch segments and optional vertical parent-child
        connector segments.
    """

    rows = _filtered_lineage_rows(species_rows, involved_only=involved_only)
    if not rows:
        return {"nodes": {}, "branches": [], "end_time": end_time}

    species_by_id = {row["species_id"]: dict(row) for row in rows}

    if end_time is None:
        end_time = _default_end_time(rows)

    daughters_by_parent: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        parent_id = row["parent_id"]
        if parent_id in species_by_id:
            daughters_by_parent[parent_id].append(row["species_id"])
    for daughter_ids in daughters_by_parent.values():
        daughter_ids.sort(key=lambda species_id: (species_by_id[species_id]["origin_time"], species_id))

    root_ids = _lineage_roots(rows, species_by_id)
    ordered_ids = _clade_order(root_ids, daughters_by_parent, species_by_id)
    if compact_extinct:
        y_positions = _packed_interval_positions(ordered_ids, species_by_id, end_time)
    else:
        y_positions = {species_id: float(len(ordered_ids) - index - 1) for index, species_id in enumerate(ordered_ids)}

    nodes: dict[int, dict[str, Any]] = {}
    branches: list[dict[str, Any]] = []
    for species_id in ordered_ids:
        row = species_by_id[species_id]
        origin = row["origin_time"]
        final = row["extinction_time"] if row["extinction_time"] is not None else end_time
        y = y_positions[species_id]
        nodes[species_id] = {
            **row,
            "y": y,
            "branch_start": origin,
            "branch_end": final,
            "daughter_ids": list(daughters_by_parent.get(species_id, [])),
            "is_terminal": species_id not in daughters_by_parent,
        }
        branches.append(
            {
                "kind": "branch",
                "species_id": species_id,
                "x0": origin,
                "x1": final,
                "y0": y,
                "y1": y,
            }
        )
    for species_id in ordered_ids:
        row = species_by_id[species_id]
        parent_id = row["parent_id"]
        if parent_id in y_positions:
            branches.append(
                {
                    "kind": "split",
                    "parent_id": parent_id,
                    "species_id": species_id,
                    "x0": row["origin_time"],
                    "x1": row["origin_time"],
                    "y0": y_positions[parent_id],
                    "y1": y_positions[species_id],
                }
            )

    return {"nodes": nodes, "branches": branches, "end_time": end_time}


def plot_lineage_tree(ax, species_rows: list[dict[str, Any]], end_time: float | None = None, **kwargs):
    """Plot a time-calibrated parent-daughter species tree.

    The parent species branch continues after speciation, while daughter species
    branches split off at their origin time. This matches the current model rule
    where, for example, species 4 can split into persisting species 4 plus
    daughter species 13.
    """

    from .visualization import plot_parent_daughter_tree

    return plot_parent_daughter_tree(ax, species_rows, end_time=end_time, involved_only=True, **kwargs)


def _filtered_lineage_rows(species_rows: list[dict[str, Any]], *, involved_only: bool) -> list[dict[str, Any]]:
    rows_by_id = {row["species_id"]: dict(row) for row in species_rows}
    if not involved_only:
        return list(rows_by_id.values())

    involved = set()
    for row in species_rows:
        parent_id = row["parent_id"]
        if parent_id is None:
            continue
        involved.add(row["species_id"])
        if parent_id in rows_by_id:
            involved.add(parent_id)
    return [rows_by_id[species_id] for species_id in rows_by_id if species_id in involved]


def _lineage_roots(rows: list[dict[str, Any]], species_by_id: dict[int, dict[str, Any]]) -> list[int]:
    roots = [
        row["species_id"]
        for row in rows
        if row["parent_id"] is None or row["parent_id"] not in species_by_id
    ]
    return sorted(roots, key=lambda species_id: (species_by_id[species_id]["origin_time"], species_id))


def _clade_order(
    root_ids: list[int],
    daughters_by_parent: dict[int, list[int]],
    species_by_id: dict[int, dict[str, Any]],
) -> list[int]:
    ordered: list[int] = []
    visited: set[int] = set()

    def visit(species_id: int) -> None:
        if species_id in visited or species_id not in species_by_id:
            return
        visited.add(species_id)
        ordered.append(species_id)
        for daughter_id in daughters_by_parent.get(species_id, []):
            visit(daughter_id)

    for root_id in root_ids:
        visit(root_id)

    for species_id in sorted(species_by_id, key=lambda item: (species_by_id[item]["origin_time"], item)):
        visit(species_id)
    return ordered


def _packed_interval_positions(
    ordered_ids: list[int],
    species_by_id: dict[int, dict[str, Any]],
    end_time: float,
) -> dict[int, float]:
    """Assign species to reusable lanes when branch intervals do not overlap."""

    order_index = {species_id: index for index, species_id in enumerate(ordered_ids)}
    interval_ids = sorted(
        ordered_ids,
        key=lambda species_id: (
            species_by_id[species_id]["origin_time"],
            _branch_end(species_by_id[species_id], end_time),
            order_index[species_id],
        ),
    )
    lane_ends: list[float] = []
    lane_by_id: dict[int, int] = {}
    for species_id in interval_ids:
        row = species_by_id[species_id]
        start = row["origin_time"]
        final = _branch_end(row, end_time)
        lane_index = None
        for candidate_index, lane_end in enumerate(lane_ends):
            if start >= lane_end:
                lane_index = candidate_index
                break
        if lane_index is None:
            lane_index = len(lane_ends)
            lane_ends.append(final)
        else:
            lane_ends[lane_index] = final
        lane_by_id[species_id] = lane_index

    lane_count = max(lane_by_id.values(), default=-1) + 1
    return {species_id: float(lane_count - lane_by_id[species_id] - 1) for species_id in ordered_ids}


def _branch_end(row: dict[str, Any], end_time: float) -> float:
    return row["extinction_time"] if row["extinction_time"] is not None else end_time


def _default_end_time(species_rows: list[dict[str, Any]]) -> float:
    times = []
    for row in species_rows:
        times.append(row["origin_time"])
        if row["extinction_time"] is not None:
            times.append(row["extinction_time"])
    return max(times) if times else 0
