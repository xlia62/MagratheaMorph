import numpy as np

from magratheamorph.model import SpeciationModel
from magratheamorph.readers import DemFrame
from magratheamorph.zones import ZoneExtractor


def _frame(time, mask):
    return DemFrame(
        time=time,
        elevation=np.zeros_like(mask, dtype=float),
        x=np.arange(mask.shape[1]),
        y=np.arange(mask.shape[0]),
        drainage_area=mask.astype(float),
    )


def test_new_zone_gets_new_species_and_ids_are_not_reused():
    step0 = np.array([[1, 0, 0, 0, 0]], dtype=float)
    step1 = np.array([[1, 0, 0, 0, 1]], dtype=float)
    step2 = np.array([[0, 0, 0, 0, 1]], dtype=float)
    step3 = np.array([[1, 0, 0, 0, 1]], dtype=float)

    extractor = ZoneExtractor(min_drainage_area=0.5, connectivity=8, use_catchments=False)
    z0 = extractor.extract(_frame(0, step0))
    z1 = extractor.extract(_frame(1, step1), previous=z0)
    z2 = extractor.extract(_frame(2, step2), previous=z1)
    z3 = extractor.extract(_frame(3, step3), previous=z2)

    assert sorted(z0.zones) == [0]
    assert sorted(z1.zones) == [0, 1]
    assert sorted(z2.zones) == [1]
    assert sorted(z3.zones) == [1, 2]

    model = SpeciationModel(time_to_speciation=10, seed_new_zones=True)
    model.initialize(z0)
    model.step(z1)
    model.step(z2)
    model.step(z3)

    species_by_zone = {}
    for pop in model.populations.values():
        species_by_zone.setdefault(pop.zone_id, set()).add(pop.species_id)

    assert 1 in species_by_zone
    assert 2 in species_by_zone
    assert species_by_zone[1] != species_by_zone[2]
    assert max(model.species) == len(model.species) - 1
