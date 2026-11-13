"""River habitat zone extraction and overlap tracking."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .hydrology import compute_d8_hydrology
from .readers import DemFrame


@dataclass(frozen=True)
class Zone:
    zone_id: int
    time: float
    mask: np.ndarray
    catchment_id: int | None = None

    @property
    def area_cells(self) -> int:
        return int(np.count_nonzero(self.mask))


@dataclass(frozen=True)
class ZoneMap:
    time: float
    labels: np.ndarray
    zones: dict[int, Zone]
    drainage_area: np.ndarray
    habitat_mask: np.ndarray


@dataclass(frozen=True)
class ZoneTransition:
    old_zone_id: int
    new_zone_id: int
    overlap_cells: int


class ZoneExtractor:
    """Extract connected river habitat zones from DEM frames."""

    def __init__(
        self,
        min_drainage_area: float,
        min_elevation: float | None = None,
        max_elevation: float | None = None,
        use_catchments: bool = True,
        connectivity: int = 8,
        min_zone_cells: int = 1,
        gap_tolerance_cells: int = 0,
        habitat_mode: str = "drainage",
        min_lake_depth: float = 0.0,
        max_lake_depth: float | None = None,
    ):
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        if min_zone_cells < 1:
            raise ValueError("min_zone_cells must be at least 1")
        if gap_tolerance_cells < 0:
            raise ValueError("gap_tolerance_cells must be non-negative")
        if habitat_mode not in {"drainage", "elevation", "drainage_and_elevation", "drainage_or_shallow_lake"}:
            raise ValueError(
                "habitat_mode must be 'drainage', 'elevation', 'drainage_and_elevation', "
                "or 'drainage_or_shallow_lake'"
            )
        if min_lake_depth < 0:
            raise ValueError("min_lake_depth must be non-negative")
        if max_lake_depth is not None and max_lake_depth < min_lake_depth:
            raise ValueError("max_lake_depth must be greater than or equal to min_lake_depth")
        self.min_drainage_area = min_drainage_area
        self.min_zone_cells = min_zone_cells
        self.gap_tolerance_cells = gap_tolerance_cells
        self.habitat_mode = habitat_mode
        self.min_lake_depth = min_lake_depth
        self.max_lake_depth = max_lake_depth
        self.min_elevation = min_elevation
        self.max_elevation = max_elevation
        self.use_catchments = use_catchments
        self.connectivity = connectivity
        self._used_zone_ids: set[int] = set()

    def extract(self, frame: DemFrame, previous: ZoneMap | None = None, preserve_ids: bool = True) -> ZoneMap:
        """Extract zones from one DEM frame.

        Parameters
        ----------
        frame
            DEM frame to process.
        previous
            Optional previous zone map. When provided, persistent zone IDs are
            carried forward from this map.
        preserve_ids
            If True and ``previous`` is provided, relabel successor zones so
            dominant continuing zones keep their previous IDs.
        """

        drainage_area = frame.drainage_area
        if drainage_area is None:
            _, drainage_area = compute_d8_hydrology(frame.elevation, frame.cell_area)

        elevation_mask = self._elevation_mask(frame)
        drainage_mask = np.asarray(drainage_area) >= self.min_drainage_area
        if self.habitat_mode == "elevation":
            habitat = elevation_mask
        elif self.habitat_mode == "drainage_and_elevation":
            habitat = drainage_mask & elevation_mask
        elif self.habitat_mode == "drainage_or_shallow_lake":
            lake_submerged_mask = self._lake_submerged_mask(frame)
            exposed_drainage = drainage_mask & ~lake_submerged_mask
            shallow_lake = self._shallow_lake_mask(frame) & ~drainage_mask
            habitat = exposed_drainage | shallow_lake
        else:
            habitat = self._exposed_drainage_mask(frame, drainage_mask) & elevation_mask

        if self.gap_tolerance_cells:
            habitat = self._bridge_small_gaps(habitat)
        if frame.water_depth is not None and self.habitat_mode in {"drainage", "drainage_or_shallow_lake"}:
            habitat &= ~(drainage_mask & self._water_submerged_mask(frame))
        catchment = (
            frame.catchment
            if self.use_catchments
            and self.habitat_mode not in {"elevation", "drainage_or_shallow_lake"}
            and not self.gap_tolerance_cells
            else None
        )
        labels, zones = self._label_components(habitat, catchment, frame.time)
        habitat = labels >= 0
        zone_map = ZoneMap(
            time=frame.time,
            labels=labels,
            zones=zones,
            drainage_area=np.asarray(drainage_area),
            habitat_mask=habitat,
        )
        if previous is not None and preserve_ids:
            zone_map = relabel_successor_zones(previous, zone_map, used_zone_ids=self._used_zone_ids)
        self._used_zone_ids.update(zone_map.zones)
        return zone_map

    def _elevation_mask(self, frame: DemFrame) -> np.ndarray:
        mask = np.isfinite(frame.elevation)
        if self.min_elevation is not None:
            mask &= frame.elevation >= self.min_elevation
        if self.max_elevation is not None:
            mask &= frame.elevation <= self.max_elevation
        return mask

    def _shallow_lake_mask(self, frame: DemFrame) -> np.ndarray:
        if self.max_lake_depth is None:
            raise ValueError("max_lake_depth is required for drainage_or_shallow_lake habitat mode")
        if frame.water_depth is None:
            raise ValueError("water_depth is required for drainage_or_shallow_lake habitat mode")
        water_depth = np.asarray(frame.water_depth)
        return (
            np.isfinite(water_depth)
            & (water_depth >= self.min_lake_depth)
            & (water_depth <= self.max_lake_depth)
        )

    def _lake_submerged_mask(self, frame: DemFrame) -> np.ndarray:
        if frame.water_depth is None:
            raise ValueError("water_depth is required for drainage_or_shallow_lake habitat mode")
        return self._water_submerged_mask(frame)

    def _water_submerged_mask(self, frame: DemFrame) -> np.ndarray:
        if frame.water_depth is None:
            return np.zeros(frame.elevation.shape, dtype=bool)
        water_depth = np.asarray(frame.water_depth)
        return np.isfinite(water_depth) & (water_depth >= 0.0)

    def _exposed_drainage_mask(self, frame: DemFrame, drainage_mask: np.ndarray) -> np.ndarray:
        """Remove submerged channels when water-depth data are available."""

        if frame.water_depth is None:
            return drainage_mask
        water_depth = np.asarray(frame.water_depth)
        submerged = np.isfinite(water_depth) & (water_depth >= 0.0)
        return drainage_mask & ~submerged

    def _label_components(
        self,
        habitat: np.ndarray,
        catchment: np.ndarray | None,
        time: float,
    ) -> tuple[np.ndarray, dict[int, Zone]]:
        labels = np.full(habitat.shape, -1, dtype=int)
        zones: dict[int, Zone] = {}
        zone_id = 0
        nrows, ncols = habitat.shape

        for row in range(nrows):
            for col in range(ncols):
                if not habitat[row, col] or labels[row, col] != -1:
                    continue

                catchment_id = _catchment_id(catchment[row, col]) if catchment is not None else None
                cells = []
                queue: deque[tuple[int, int]] = deque([(row, col)])
                labels[row, col] = zone_id

                while queue:
                    rr, cc = queue.popleft()
                    cells.append((rr, cc))
                    for nr, nc in self._neighbors(rr, cc, nrows, ncols):
                        if not habitat[nr, nc] or labels[nr, nc] != -1:
                            continue
                        if catchment is not None and _catchment_id(catchment[nr, nc]) != catchment_id:
                            continue
                        labels[nr, nc] = zone_id
                        queue.append((nr, nc))

                mask = np.zeros(habitat.shape, dtype=bool)
                rows, cols = zip(*cells)
                if len(cells) < self.min_zone_cells:
                    labels[rows, cols] = -1
                    continue
                mask[rows, cols] = True
                zones[zone_id] = Zone(
                    zone_id=zone_id,
                    time=time,
                    mask=mask,
                    catchment_id=catchment_id,
                )
                zone_id += 1

        return labels, zones

    def _neighbors(self, row: int, col: int, nrows: int, ncols: int):
        offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        if self.connectivity == 8:
            offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in offsets:
            rr, cc = row + dr, col + dc
            if 0 <= rr < nrows and 0 <= cc < ncols:
                yield rr, cc

    def _bridge_small_gaps(self, habitat: np.ndarray) -> np.ndarray:
        """Fill short non-habitat gaps between habitat components."""

        if self.gap_tolerance_cells == 0 or not np.any(habitat):
            return habitat

        component_labels, component_count = self._raw_component_labels(habitat)
        if component_count < 2:
            return habitat

        bridge_cells: set[tuple[int, int]] = set()
        nrows, ncols = habitat.shape

        for component_id in range(component_count):
            queue: deque[tuple[int, int]] = deque()
            distance = np.full(habitat.shape, -1, dtype=int)
            predecessor: dict[tuple[int, int], tuple[int, int] | None] = {}
            for row, col in zip(*np.where(component_labels == component_id)):
                queue.append((row, col))
                distance[row, col] = 0
                predecessor[(row, col)] = None

            while queue:
                row, col = queue.popleft()
                for nr, nc in self._neighbors(row, col, nrows, ncols):
                    other_component = component_labels[nr, nc]
                    if other_component >= 0:
                        if other_component != component_id:
                            bridge_cells.update(_trace_non_habitat_path((row, col), predecessor, habitat))
                        continue

                    next_distance = distance[row, col] + 1
                    if next_distance > self.gap_tolerance_cells or distance[nr, nc] != -1:
                        continue
                    distance[nr, nc] = next_distance
                    predecessor[(nr, nc)] = (row, col)
                    queue.append((nr, nc))

        if not bridge_cells:
            return habitat
        bridged = habitat.copy()
        for row, col in bridge_cells:
            bridged[row, col] = True
        return bridged

    def _raw_component_labels(self, habitat: np.ndarray) -> tuple[np.ndarray, int]:
        labels = np.full(habitat.shape, -1, dtype=int)
        component_id = 0
        nrows, ncols = habitat.shape
        for row in range(nrows):
            for col in range(ncols):
                if not habitat[row, col] or labels[row, col] != -1:
                    continue
                queue: deque[tuple[int, int]] = deque([(row, col)])
                labels[row, col] = component_id
                while queue:
                    rr, cc = queue.popleft()
                    for nr, nc in self._neighbors(rr, cc, nrows, ncols):
                        if habitat[nr, nc] and labels[nr, nc] == -1:
                            labels[nr, nc] = component_id
                            queue.append((nr, nc))
                component_id += 1
        return labels, component_id


def compute_zone_transitions(old: ZoneMap, new: ZoneMap) -> list[ZoneTransition]:
    """Return nonzero overlaps between old and new zone labels."""

    if old.labels.shape != new.labels.shape:
        raise ValueError("zone maps must have the same shape to compute overlap")

    transitions: list[ZoneTransition] = []
    old_ids = [zid for zid in old.zones]
    new_ids = [zid for zid in new.zones]
    for old_id in old_ids:
        old_mask = old.labels == old_id
        for new_id in new_ids:
            overlap = int(np.count_nonzero(old_mask & (new.labels == new_id)))
            if overlap > 0:
                transitions.append(ZoneTransition(old_id, new_id, overlap))
    return transitions


def compute_biological_transitions(
    old: ZoneMap,
    new: ZoneMap,
    min_overlap_cells: int = 1,
) -> list[ZoneTransition]:
    """Return zone transitions used for species/population inheritance.

    Any overlap at or above ``min_overlap_cells`` transfers species from the old
    zone into the new zone. This allows river capture/secondary contact, where a
    new zone can inherit species from multiple old zones and an old zone can
    transfer species to both its persistent successor and a capturing zone.
    """

    if min_overlap_cells < 1:
        raise ValueError("min_overlap_cells must be at least 1")
    return [
        transition
        for transition in compute_zone_transitions(old, new)
        if transition.overlap_cells >= min_overlap_cells
    ]


def relabel_successor_zones(old: ZoneMap, new: ZoneMap, used_zone_ids: set[int] | None = None) -> ZoneMap:
    """Return a copy of ``new`` with persistent zone IDs carried from ``old``.

    Freshly extracted zones are numbered in scan order, so a persistent river
    zone can receive a different label when its shape grows or shrinks. This
    function preserves old IDs for dominant one-to-one overlaps. Splits and
    merges still produce new IDs for the non-dominant/new components, which
    keeps zone IDs useful for visual interpretation without hiding topology
    changes from the transition table.
    """

    transitions = compute_zone_transitions(old, new)
    if not transitions:
        return new

    old_to_new: dict[int, list[ZoneTransition]] = {}
    new_to_old: dict[int, list[ZoneTransition]] = {}
    for transition in transitions:
        old_to_new.setdefault(transition.old_zone_id, []).append(transition)
        new_to_old.setdefault(transition.new_zone_id, []).append(transition)

    mapping: dict[int, int] = {}
    used_old_ids: set[int] = set()

    candidates = []
    for new_id, incoming in new_to_old.items():
        best = max(incoming, key=lambda item: item.overlap_cells)
        candidates.append((best.overlap_cells, new_id, best.old_zone_id))

    for _, new_id, best_old in sorted(candidates, reverse=True):
        if best_old not in used_old_ids:
            mapping[new_id] = best_old
            used_old_ids.add(best_old)

    next_id = 0
    reserved = set(old.zones) | set(mapping.values())
    if used_zone_ids is not None:
        reserved |= set(used_zone_ids)
    for new_id in sorted(new.zones):
        if new_id in mapping:
            continue
        while next_id in reserved:
            next_id += 1
        mapping[new_id] = next_id
        reserved.add(next_id)

    labels = np.full(new.labels.shape, -1, dtype=int)
    zones: dict[int, Zone] = {}
    for old_new_id, persistent_id in mapping.items():
        mask = new.labels == old_new_id
        labels[mask] = persistent_id
        zone = new.zones[old_new_id]
        zones[persistent_id] = Zone(
            zone_id=persistent_id,
            time=zone.time,
            mask=mask,
            catchment_id=zone.catchment_id,
        )

    return ZoneMap(
        time=new.time,
        labels=labels,
        zones=zones,
        drainage_area=new.drainage_area,
        habitat_mask=new.habitat_mask,
    )


def _catchment_id(value) -> int:
    if not np.isfinite(value):
        return -1
    return int(value)


def _trace_non_habitat_path(
    start: tuple[int, int],
    predecessor: dict[tuple[int, int], tuple[int, int] | None],
    habitat: np.ndarray,
) -> list[tuple[int, int]]:
    cells = []
    current: tuple[int, int] | None = start
    while current is not None:
        row, col = current
        if not habitat[row, col]:
            cells.append(current)
        current = predecessor[current]
    return cells
