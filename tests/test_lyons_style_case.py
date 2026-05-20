import numpy as np

from magratheamorph.model import SpeciationModel
from magratheamorph.zones import Zone, ZoneMap, compute_zone_transitions


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


def test_lyons_figure3_like_transition():
    """One synthetic transition with persistence, split, and merger.

    This mirrors the kind of step0-to-step1 bookkeeping in the fish tectonics
    slide deck and Lyons-style stream-network reorganization examples.
    """

    step0 = zone_map(
        0,
        [
            [0, 0, -1, 1, 1, -1],
            [2, 2, -1, 3, 3, -1],
            [4, 4, -1, 5, 5, -1],
            [6, 6, -1, 7, 7, -1],
        ],
    )
    step1 = zone_map(
        1,
        [
            [0, 0, -1, 1, 1, -1],
            [2, 8, -1, 3, 3, -1],
            [4, 4, -1, 5, 5, -1],
            [10, 10, -1, 10, 10, -1],
        ],
    )

    transitions = compute_zone_transitions(step0, step1)
    by_old = {}
    for transition in transitions:
        by_old.setdefault(transition.old_zone_id, set()).add(transition.new_zone_id)

    assert by_old[0] == {0}
    assert by_old[2] == {2, 8}
    assert by_old[6] == {10}
    assert by_old[7] == {10}

    model = SpeciationModel(time_to_speciation=0)
    model.initialize(step0)
    model.step(step1)

    assert len([species for species in model.species.values() if species.extant]) == 9
    assert model.richness_by_zone()[10] == 2
    assert any(event.event_type == "fragmentation" for event in model.events)
    assert any(event.event_type == "speciation" for event in model.events)
