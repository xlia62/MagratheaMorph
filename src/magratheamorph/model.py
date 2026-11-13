"""Allopatric/vicariance river fish speciation model."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .zones import ZoneMap, ZoneTransition, compute_biological_transitions


@dataclass
class Species:
    species_id: int
    parent_id: int | None
    origin_time: float
    extinction_time: float | None = None

    @property
    def extant(self) -> bool:
        return self.extinction_time is None


@dataclass
class Population:
    population_id: int
    species_id: int
    zone_id: int
    isolation_parent: int | None = None
    isolation_start_time: float | None = None


@dataclass(frozen=True)
class EvolutionEvent:
    time: float
    event_type: str
    details: dict


class SpeciationModel:
    """Evolve fish lineages from changing river zones."""

    def __init__(
        self,
        time_to_speciation: float,
        seed_new_zones: bool = False,
        min_species_transfer_cells: int = 1,
    ):
        if time_to_speciation < 0:
            raise ValueError("time_to_speciation must be non-negative")
        if min_species_transfer_cells < 1:
            raise ValueError("min_species_transfer_cells must be at least 1")
        self.time_to_speciation = time_to_speciation
        self.seed_new_zones = seed_new_zones
        self.min_species_transfer_cells = min_species_transfer_cells
        self.species: dict[int, Species] = {}
        self.populations: dict[int, Population] = {}
        self.events: list[EvolutionEvent] = []
        self.range_history: list[dict] = []
        self._next_species_id = 0
        self._next_population_id = 0
        self._last_zone_map: ZoneMap | None = None

    def initialize(self, zone_map: ZoneMap, species_per_zone: int = 1) -> None:
        """Initialize one or more species in every current zone."""

        if self._last_zone_map is not None:
            raise RuntimeError("model is already initialized")
        for zone_id in zone_map.zones:
            for _ in range(species_per_zone):
                species_id = self._create_species(parent_id=None, time=zone_map.time)
                self._create_population(species_id, zone_id)
        self._last_zone_map = zone_map
        self._record_ranges(zone_map.time)

    def step(self, zone_map: ZoneMap) -> None:
        """Advance the model to a new zone map."""

        if self._last_zone_map is None:
            self.initialize(zone_map)
            return

        transitions = compute_biological_transitions(
            self._last_zone_map,
            zone_map,
            min_overlap_cells=self.min_species_transfer_cells,
        )
        successors = _successor_lookup(transitions)
        predecessors = _predecessor_lookup(transitions)

        old_populations = list(self.populations.values())
        new_populations: dict[int, Population] = {}
        split_groups: dict[tuple[int, int], list[int]] = defaultdict(list)

        for pop in old_populations:
            next_zone_ids = successors.get(pop.zone_id, [])
            if not next_zone_ids:
                self.events.append(
                    EvolutionEvent(
                        zone_map.time,
                        "population_lost",
                        {"population_id": pop.population_id, "species_id": pop.species_id, "old_zone_id": pop.zone_id},
                    )
                )
                continue

            contact_zone_ids = [
                zone_id
                for zone_id in next_zone_ids
                if zone_id != pop.zone_id and len(predecessors.get(zone_id, [])) > 1
            ]
            isolation_zone_ids = [zone_id for zone_id in next_zone_ids if zone_id not in set(contact_zone_ids)]

            if len(isolation_zone_ids) == 1:
                new_pop = Population(
                    population_id=pop.population_id,
                    species_id=pop.species_id,
                    zone_id=isolation_zone_ids[0],
                    isolation_parent=pop.isolation_parent,
                    isolation_start_time=pop.isolation_start_time,
                )
                _add_population_if_absent(new_populations, new_pop)
            elif len(isolation_zone_ids) > 1:
                group_parent = pop.population_id
                for zone_id in isolation_zone_ids:
                    new_pop = self._clone_population_for_step(
                        pop,
                        zone_id,
                        isolation_parent=group_parent,
                        isolation_start_time=zone_map.time,
                    )
                    _add_population_if_absent(new_populations, new_pop)
                    split_groups[(pop.species_id, group_parent)].append(new_pop.population_id)
                self.events.append(
                    EvolutionEvent(
                        zone_map.time,
                        "fragmentation",
                        {"species_id": pop.species_id, "old_zone_id": pop.zone_id, "new_zone_ids": isolation_zone_ids},
                    )
                )
            else:
                for index, zone_id in enumerate(contact_zone_ids):
                    if index == 0:
                        new_pop = Population(
                            population_id=pop.population_id,
                            species_id=pop.species_id,
                            zone_id=zone_id,
                        )
                    else:
                        new_pop = self._clone_population_for_step(
                            pop,
                            zone_id,
                            isolation_parent=None,
                            isolation_start_time=None,
                        )
                    _add_population_if_absent(new_populations, new_pop)

            for zone_id in contact_zone_ids:
                if any(existing.species_id == pop.species_id and existing.zone_id == zone_id for existing in new_populations.values()):
                    continue
                new_pop = self._clone_population_for_step(
                    pop,
                    zone_id,
                    isolation_parent=None,
                    isolation_start_time=None,
                )
                _add_population_if_absent(new_populations, new_pop)
                self.events.append(
                    EvolutionEvent(
                        zone_map.time,
                        "secondary_contact",
                        {"species_id": pop.species_id, "old_zone_id": pop.zone_id, "new_zone_id": zone_id},
                    )
                )

            if not isolation_zone_ids and contact_zone_ids:
                for zone_id in contact_zone_ids:
                    new_pop = self._clone_population_for_step(
                        pop,
                        zone_id,
                        isolation_parent=None,
                        isolation_start_time=None,
                    )
                    _add_population_if_absent(new_populations, new_pop)

        self.populations = new_populations
        if self.seed_new_zones:
            inherited_zone_ids = {pop.zone_id for pop in self.populations.values()}
            for zone_id in sorted(set(zone_map.zones) - inherited_zone_ids):
                species_id = self._create_species(parent_id=None, time=zone_map.time)
                self._create_population(species_id, zone_id)
                self.events.append(
                    EvolutionEvent(
                        zone_map.time,
                        "new_zone_species",
                        {"species_id": species_id, "zone_id": zone_id},
                    )
                )
        self._cancel_reconnected_isolation(predecessors, zone_map.time)
        self._apply_speciation(zone_map.time)
        self._mark_extinctions(zone_map.time)
        self._last_zone_map = zone_map
        self._record_ranges(zone_map.time)

    def richness_by_zone(self) -> dict[int, int]:
        richness: dict[int, set[int]] = defaultdict(set)
        for pop in self.populations.values():
            if self.species[pop.species_id].extant:
                richness[pop.zone_id].add(pop.species_id)
        return {zone_id: len(species_ids) for zone_id, species_ids in richness.items()}

    def richness_by_zone_at(self, time: float) -> dict[int, int]:
        richness: dict[int, set[int]] = defaultdict(set)
        for row in self.range_history:
            if row["time"] == time:
                richness[row["zone_id"]].add(row["species_id"])
        return {zone_id: len(species_ids) for zone_id, species_ids in richness.items()}

    def species_by_zone_at(self, time: float) -> dict[int, list[int]]:
        species_by_zone: dict[int, set[int]] = defaultdict(set)
        for row in self.range_history:
            if row["time"] == time:
                species_by_zone[row["zone_id"]].add(row["species_id"])
        return {zone_id: sorted(species_ids) for zone_id, species_ids in species_by_zone.items()}

    def richness_map(self, zone_map: ZoneMap) -> np.ndarray:
        richness_by_zone = self.richness_by_zone()
        out = np.zeros(zone_map.labels.shape, dtype=int)
        for zone_id, richness in richness_by_zone.items():
            out[zone_map.labels == zone_id] = richness
        return out

    def richness_map_at(self, zone_map: ZoneMap) -> np.ndarray:
        richness_by_zone = self.richness_by_zone_at(zone_map.time)
        out = np.zeros(zone_map.labels.shape, dtype=int)
        for zone_id, richness in richness_by_zone.items():
            out[zone_map.labels == zone_id] = richness
        return out

    def species_presence_map_at(self, zone_map: ZoneMap, species_id: int) -> np.ndarray:
        species_by_zone = self.species_by_zone_at(zone_map.time)
        out = np.zeros(zone_map.labels.shape, dtype=int)
        for zone_id, species_ids in species_by_zone.items():
            if species_id in species_ids:
                out[zone_map.labels == zone_id] = 1
        return out

    def species_table(self) -> list[dict]:
        return [species.__dict__.copy() for species in self.species.values()]

    def population_table(self) -> list[dict]:
        return [pop.__dict__.copy() for pop in self.populations.values()]

    def range_table(self) -> list[dict]:
        return list(self.range_history)

    def event_table(self) -> list[dict]:
        return [
            {"time": event.time, "event_type": event.event_type, **event.details}
            for event in self.events
        ]

    def _create_species(self, parent_id: int | None, time: float) -> int:
        species_id = self._next_species_id
        self._next_species_id += 1
        self.species[species_id] = Species(species_id, parent_id, time)
        self.events.append(EvolutionEvent(time, "species_origin", {"species_id": species_id, "parent_id": parent_id}))
        return species_id

    def _create_population(self, species_id: int, zone_id: int) -> int:
        population_id = self._next_population_id
        self._next_population_id += 1
        self.populations[population_id] = Population(population_id, species_id, zone_id)
        return population_id

    def _clone_population_for_step(
        self,
        pop: Population,
        zone_id: int,
        isolation_parent: int | None,
        isolation_start_time: float | None,
    ) -> Population:
        population_id = self._next_population_id
        self._next_population_id += 1
        return Population(
            population_id=population_id,
            species_id=pop.species_id,
            zone_id=zone_id,
            isolation_parent=isolation_parent,
            isolation_start_time=isolation_start_time,
        )

    def _cancel_reconnected_isolation(self, predecessors: dict[int, list[int]], time: float) -> None:
        """Cancel isolation for conspecific populations reconnected before speciation."""

        by_species_zone: dict[tuple[int, int], list[Population]] = defaultdict(list)
        for pop in self.populations.values():
            by_species_zone[(pop.species_id, pop.zone_id)].append(pop)

        to_remove = set()
        for (_, zone_id), pops in by_species_zone.items():
            groups = {pop.isolation_parent for pop in pops if pop.isolation_parent is not None}
            if len(pops) > 1 or len(predecessors.get(zone_id, [])) > 1:
                for group in groups:
                    group_pops = [p for p in self.populations.values() if p.isolation_parent == group]
                    if group_pops and all(p.zone_id == zone_id for p in group_pops):
                        keeper = min(group_pops, key=lambda p: p.population_id)
                        keeper.isolation_parent = None
                        keeper.isolation_start_time = None
                        to_remove.update(p.population_id for p in group_pops if p.population_id != keeper.population_id)
                        self.events.append(
                            EvolutionEvent(
                                time,
                                "reconnection_before_speciation",
                                {"species_id": keeper.species_id, "zone_id": zone_id, "isolation_parent": group},
                            )
                        )

        for population_id in to_remove:
            self.populations.pop(population_id, None)

    def _apply_speciation(self, time: float) -> None:
        groups: dict[tuple[int, int], list[Population]] = defaultdict(list)
        for pop in self.populations.values():
            if pop.isolation_parent is not None and pop.isolation_start_time is not None:
                groups[(pop.species_id, pop.isolation_parent)].append(pop)

        for (species_id, isolation_parent), pops in list(groups.items()):
            if len(pops) < 2:
                continue
            start_time = min(p.isolation_start_time for p in pops if p.isolation_start_time is not None)
            if time - start_time < self.time_to_speciation:
                continue

            # Parent species remains in the largest/first population; others become daughters.
            pops_sorted = sorted(pops, key=lambda p: p.population_id)
            parent_pop = pops_sorted[0]
            parent_pop.isolation_parent = None
            parent_pop.isolation_start_time = None
            daughter_ids = []
            for pop in pops_sorted[1:]:
                daughter_species = self._create_species(parent_id=species_id, time=time)
                pop.species_id = daughter_species
                pop.isolation_parent = None
                pop.isolation_start_time = None
                daughter_ids.append(daughter_species)
            self.events.append(
                EvolutionEvent(
                    time,
                    "speciation",
                    {
                        "parent_species_id": species_id,
                        "daughter_species_ids": daughter_ids,
                        "isolation_parent": isolation_parent,
                    },
                )
            )

    def _mark_extinctions(self, time: float) -> None:
        occupied_species = {pop.species_id for pop in self.populations.values()}
        for species in self.species.values():
            if species.extant and species.species_id not in occupied_species:
                species.extinction_time = time
                self.events.append(EvolutionEvent(time, "extinction", {"species_id": species.species_id}))

    def _record_ranges(self, time: float) -> None:
        for pop in self.populations.values():
            self.range_history.append(
                {
                    "time": time,
                    "population_id": pop.population_id,
                    "species_id": pop.species_id,
                    "zone_id": pop.zone_id,
                    "isolation_parent": pop.isolation_parent,
                    "isolation_start_time": pop.isolation_start_time,
                }
            )


def _successor_lookup(transitions: list[ZoneTransition]) -> dict[int, list[int]]:
    by_old: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for transition in transitions:
        by_old[transition.old_zone_id].append((transition.new_zone_id, transition.overlap_cells))
    return {old: [new for new, _ in sorted(values)] for old, values in by_old.items()}


def _predecessor_lookup(transitions: list[ZoneTransition]) -> dict[int, list[int]]:
    by_new: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for transition in transitions:
        by_new[transition.new_zone_id].append((transition.old_zone_id, transition.overlap_cells))
    return {new: [old for old, _ in sorted(values)] for new, values in by_new.items()}


def _add_population_if_absent(populations: dict[int, Population], population: Population) -> None:
    for existing in populations.values():
        if existing.species_id == population.species_id and existing.zone_id == population.zone_id:
            return
    populations[population.population_id] = population
