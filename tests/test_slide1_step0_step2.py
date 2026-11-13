import numpy as np

from magratheamorph.model import SpeciationModel
from magratheamorph.readers import DemFrame
from magratheamorph.zones import ZoneExtractor, compute_zone_transitions


# These 9x18 masks were sampled from slide 1 of 20240308_fishtectonics.pptx.
# The upper panel is step 0 and the lower panel is step 2; 1 means the dark
# river-zone color and 0 means background.
STEP0_DARK_MASK = np.array(
    [
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    ],
    dtype=float,
)

STEP2_DARK_MASK = np.array(
    [
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    ],
    dtype=float,
)


def _zone_map(time, dark_mask):
    frame = DemFrame(
        time=time,
        elevation=np.zeros_like(dark_mask),
        x=np.arange(dark_mask.shape[1]),
        y=np.arange(dark_mask.shape[0]),
        drainage_area=dark_mask,
    )
    extractor = ZoneExtractor(
        min_drainage_area=0.5,
        connectivity=8,
        use_catchments=False,
    )
    return extractor.extract(frame)


def test_slide1_step0_to_step2_zone_transition_and_species_events():
    step0 = _zone_map(0, STEP0_DARK_MASK)
    step2_raw = _zone_map(2, STEP2_DARK_MASK)
    frame = DemFrame(
        time=2,
        elevation=np.zeros_like(STEP2_DARK_MASK),
        x=np.arange(STEP2_DARK_MASK.shape[1]),
        y=np.arange(STEP2_DARK_MASK.shape[0]),
        drainage_area=STEP2_DARK_MASK,
    )
    extractor = ZoneExtractor(min_drainage_area=0.5, connectivity=8, use_catchments=False)
    step2 = extractor.extract(frame, previous=step0)

    assert len(step0.zones) == 12
    assert len(step2_raw.zones) == 10
    assert len(step2.zones) == 10
    assert 8 in step2.zones
    assert 9 in step2.zones

    transitions = compute_zone_transitions(step0, step2)
    by_old = {}
    by_new = {}
    for transition in transitions:
        by_old.setdefault(transition.old_zone_id, set()).add(transition.new_zone_id)
        by_new.setdefault(transition.new_zone_id, set()).add(transition.old_zone_id)

    # Split examples from the slide-derived masks.
    assert by_old[1] == {1, 8}
    assert by_old[4] == {4, 12}

    # Persistent shape-change examples: these keep their old IDs even though
    # raw scan-order labels changed in step2_raw.
    raw_transitions = compute_zone_transitions(step0, step2_raw)
    raw_by_old = {}
    for transition in raw_transitions:
        raw_by_old.setdefault(transition.old_zone_id, set()).add(transition.new_zone_id)
    assert raw_by_old[9] == {7}
    assert by_old[9] == {9}

    # Merger/secondary-contact examples.
    assert by_new[2] == {2, 3}
    assert by_new[6] == {5, 6, 11}

    from magratheamorph.zones import compute_biological_transitions

    bio_by_old = {}
    for transition in compute_biological_transitions(step0, step2):
        bio_by_old.setdefault(transition.old_zone_id, set()).add(transition.new_zone_id)
    assert bio_by_old[1] == {1, 8}
    assert bio_by_old[8] == {8}

    model = SpeciationModel(time_to_speciation=0)
    model.initialize(step0)
    model.step(step2)

    events = model.event_table()
    fragmentation_count = sum(event["event_type"] == "fragmentation" for event in events)
    speciation_count = sum(event["event_type"] == "speciation" for event in events)
    secondary_contact_count = sum(event["event_type"] == "secondary_contact" for event in events)
    extinction_count = sum(event["event_type"] == "extinction" for event in events)

    # The slide supplies only zone masks. Speciation/extinction values are
    # calculated from zone overlap and the model rules, not copied from the
    # slide. With time_to_speciation=0, each fragmentation immediately speciates.
    assert fragmentation_count > 0
    assert speciation_count == fragmentation_count
    assert secondary_contact_count > 0
    assert extinction_count > 0

    extant_species = sum(species.extant for species in model.species.values())
    occupied_species = {pop.species_id for pop in model.populations.values()}
    assert extant_species == len(occupied_species)

    richness = model.richness_by_zone()
    assert richness[8] == 2
    assert max(richness.values()) > 1
    assert model.richness_map_at(step2).shape == step2.labels.shape
    species_in_zone8 = {pop.species_id for pop in model.populations.values() if pop.zone_id == 8}
    assert species_in_zone8 == {1, 8}
