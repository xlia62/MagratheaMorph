"""High-level helpers for running the speciation model."""

from __future__ import annotations

from collections.abc import Iterable

from .model import SpeciationModel
from .readers import DemFrame
from .zones import ZoneExtractor, ZoneMap


def run_frames(
    frames: Iterable[DemFrame],
    extractor: ZoneExtractor,
    time_to_speciation: float,
    species_per_initial_zone: int = 1,
    seed_new_zones: bool = False,
    min_species_transfer_cells: int = 1,
) -> tuple[SpeciationModel, list[ZoneMap]]:
    """Extract zones and evolve species over a sequence of DEM frames."""

    model = SpeciationModel(
        time_to_speciation=time_to_speciation,
        seed_new_zones=seed_new_zones,
        min_species_transfer_cells=min_species_transfer_cells,
    )
    zone_maps: list[ZoneMap] = []

    previous_zone_map = None
    for index, frame in enumerate(frames):
        zone_map = extractor.extract(frame, previous=previous_zone_map)
        zone_maps.append(zone_map)
        if index == 0:
            model.initialize(zone_map, species_per_zone=species_per_initial_zone)
        else:
            model.step(zone_map)
        previous_zone_map = zone_map

    return model, zone_maps
