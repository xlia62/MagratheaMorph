import numpy as np

from magratheamorph.model import SpeciationModel
from magratheamorph.zones import Zone, ZoneMap


def zone_map(time, labels):
    labels = np.array(labels, dtype=int)
    zones = {}
    for zid in sorted(set(labels.ravel()) - {-1}):
        zones[zid] = Zone(zid, time, labels == zid)
    return ZoneMap(
        time=time,
        labels=labels,
        zones=zones,
        drainage_area=np.ones_like(labels, dtype=float),
        habitat_mask=labels >= 0,
    )


def test_reconnection_before_speciation_cancels_isolation():
    model = SpeciationModel(time_to_speciation=10)
    model.initialize(zone_map(0, [[0, 0, 0]]))
    model.step(zone_map(1, [[0, -1, 1]]))
    model.step(zone_map(2, [[0, 0, 0]]))

    extant = [s for s in model.species.values() if s.extant]
    assert len(extant) == 1
    assert len(model.populations) == 1
    assert any(e.event_type == "reconnection_before_speciation" for e in model.events)


def test_persistent_split_creates_daughter_species():
    model = SpeciationModel(time_to_speciation=2)
    model.initialize(zone_map(0, [[0, 0, 0]]))
    model.step(zone_map(1, [[0, -1, 1]]))
    model.step(zone_map(3, [[0, -1, 1]]))

    extant = [s for s in model.species.values() if s.extant]
    assert len(extant) == 2
    assert any(e.event_type == "speciation" for e in model.events)


def test_secondary_contact_after_speciation_keeps_species_separate():
    model = SpeciationModel(time_to_speciation=1)
    model.initialize(zone_map(0, [[0, 0, 0]]))
    model.step(zone_map(1, [[0, -1, 1]]))
    model.step(zone_map(2, [[0, -1, 1]]))
    model.step(zone_map(3, [[0, 0, 0]]))

    extant = [s for s in model.species.values() if s.extant]
    assert len(extant) == 2
    assert len(model.populations) == 2
    assert set(model.richness_by_zone().values()) == {2}


def test_capture_overlap_transfers_species_without_immediate_speciation():
    model = SpeciationModel(time_to_speciation=0)
    model.initialize(zone_map(0, [[0, 0, 6, 6]]))
    model.step(zone_map(1, [[0, 0, 0, 6]]))

    species_by_zone = model.species_by_zone_at(1)

    assert species_by_zone[0] == [0, 1]
    assert species_by_zone[6] == [1]
    assert len(model.species) == 2
    assert not any(event.event_type == "speciation" for event in model.events)
    assert any(event.event_type == "secondary_contact" for event in model.events)


def test_species_presence_map_marks_zones_containing_target_species():
    model = SpeciationModel(time_to_speciation=0)
    model.initialize(zone_map(0, [[0, 0, 6, 6]]))
    model.step(zone_map(1, [[0, 0, 0, 6]]))

    presence = model.species_presence_map_at(zone_map(1, [[0, 0, 0, 6]]), species_id=1)

    assert presence.tolist() == [[1, 1, 1, 1]]


def test_species_extinct_when_all_populations_lost():
    model = SpeciationModel(time_to_speciation=10)
    model.initialize(zone_map(0, [[0, 0]]))
    model.step(zone_map(1, [[-1, -1]]))

    assert all(not s.extant for s in model.species.values())
    assert any(e.event_type == "extinction" for e in model.events)
