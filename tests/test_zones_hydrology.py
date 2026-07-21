import numpy as np

from magratheamorph.hydrology import compute_d8_hydrology
from magratheamorph.readers import DemFrame, with_water_depth
from magratheamorph.zones import ZoneExtractor


def test_d8_hydrology_routes_down_simple_slope():
    dem = np.array([[3.0, 2.0, 1.0]])
    receiver, drainage_area = compute_d8_hydrology(dem, cell_area=1.0)

    assert receiver.shape == dem.shape
    assert drainage_area[0, 2] == 3.0


def test_zone_extractor_uses_drainage_area_and_elevation_filter():
    frame = DemFrame(
        time=0,
        elevation=np.array([[100.0, 200.0, 900.0]]),
        x=np.array([0.0, 1.0, 2.0]),
        y=np.array([0.0]),
        drainage_area=np.array([[10.0, 10.0, 10.0]]),
    )
    extractor = ZoneExtractor(min_drainage_area=5.0, max_elevation=500.0)
    zone_map = extractor.extract(frame)

    assert len(zone_map.zones) == 1
    assert zone_map.habitat_mask.tolist() == [[True, True, False]]


def test_zone_extractor_elevation_only_mode_works_without_drainage_area():
    frame = DemFrame(
        time=0,
        elevation=np.array(
            [
                [50.0, 120.0, 150.0, 500.0],
                [60.0, 130.0, 160.0, 510.0],
            ]
        ),
        x=np.arange(4),
        y=np.arange(2),
    )
    extractor = ZoneExtractor(
        min_drainage_area=999.0,
        min_elevation=100.0,
        max_elevation=200.0,
        habitat_mode="elevation",
        connectivity=4,
    )
    zone_map = extractor.extract(frame)

    assert len(zone_map.zones) == 1
    assert zone_map.habitat_mask.tolist() == [
        [False, True, True, False],
        [False, True, True, False],
    ]


def test_zone_extractor_drainage_and_elevation_mode_intersects_masks():
    frame = DemFrame(
        time=0,
        elevation=np.array([[50.0, 120.0, 150.0, 500.0]]),
        x=np.arange(4),
        y=np.arange(1),
        drainage_area=np.array([[10.0, 10.0, 1.0, 10.0]]),
    )
    extractor = ZoneExtractor(
        min_drainage_area=5.0,
        min_elevation=100.0,
        max_elevation=200.0,
        habitat_mode="drainage_and_elevation",
        connectivity=4,
    )
    zone_map = extractor.extract(frame)

    assert len(zone_map.zones) == 1
    assert zone_map.habitat_mask.tolist() == [[False, True, False, False]]


def test_with_water_depth_computes_positive_submerged_depths():
    frame = DemFrame(
        time=0,
        elevation=np.array([[90.0, 100.0, 110.0]]),
        x=np.arange(3),
        y=np.arange(1),
    )

    out = with_water_depth(frame, 100.0)

    assert out.water_depth.tolist() == [[10.0, 0.0, -10.0]]


def test_zone_extractor_drainage_or_shallow_lake_unions_masks():
    frame = DemFrame(
        time=0,
        elevation=np.zeros((1, 4)),
        x=np.arange(4),
        y=np.arange(1),
        drainage_area=np.array([[10.0, 1.0, 10.0, 1.0]]),
        water_depth=np.array([[-5.0, 10.0, 40.0, 40.0]]),
    )
    extractor = ZoneExtractor(
        min_drainage_area=5.0,
        habitat_mode="drainage_or_shallow_lake",
        min_lake_depth=0.0,
        max_lake_depth=30.0,
        connectivity=4,
    )
    zone_map = extractor.extract(frame)

    assert zone_map.habitat_mask.tolist() == [[True, True, False, False]]
    assert len(zone_map.zones) == 1


def test_zone_extractor_drainage_or_shallow_lake_removes_submerged_rivers():
    frame = DemFrame(
        time=0,
        elevation=np.zeros((1, 3)),
        x=np.arange(3),
        y=np.arange(1),
        drainage_area=np.array([[10.0, 10.0, 10.0]]),
        water_depth=np.array([[-1.0, 10.0, 40.0]]),
    )
    extractor = ZoneExtractor(
        min_drainage_area=5.0,
        habitat_mode="drainage_or_shallow_lake",
        max_lake_depth=30.0,
        connectivity=4,
    )
    zone_map = extractor.extract(frame)

    assert zone_map.habitat_mask.tolist() == [[True, False, False]]


def test_zone_extractor_drainage_mode_removes_submerged_rivers_when_depth_available():
    frame = DemFrame(
        time=0,
        elevation=np.zeros((1, 3)),
        x=np.arange(3),
        y=np.arange(1),
        drainage_area=np.array([[10.0, 10.0, 10.0]]),
        water_depth=np.array([[-1.0, 0.0, 40.0]]),
    )
    extractor = ZoneExtractor(
        min_drainage_area=5.0,
        habitat_mode="drainage",
        connectivity=4,
    )
    zone_map = extractor.extract(frame)

    assert zone_map.habitat_mask.tolist() == [[True, False, False]]


def test_zone_extractor_shallow_lake_mode_requires_depth_inputs():
    frame = DemFrame(
        time=0,
        elevation=np.zeros((1, 1)),
        x=np.arange(1),
        y=np.arange(1),
        drainage_area=np.array([[0.0]]),
    )
    extractor = ZoneExtractor(
        min_drainage_area=5.0,
        habitat_mode="drainage_or_shallow_lake",
        max_lake_depth=30.0,
    )

    try:
        extractor.extract(frame)
    except ValueError as exc:
        assert "water_depth" in str(exc)
    else:
        raise AssertionError("missing water_depth should fail")


def test_zone_extractor_shallow_lake_mode_requires_max_depth():
    frame = DemFrame(
        time=0,
        elevation=np.zeros((1, 1)),
        x=np.arange(1),
        y=np.arange(1),
        drainage_area=np.array([[0.0]]),
        water_depth=np.array([[1.0]]),
    )
    extractor = ZoneExtractor(
        min_drainage_area=5.0,
        habitat_mode="drainage_or_shallow_lake",
    )

    try:
        extractor.extract(frame)
    except ValueError as exc:
        assert "max_lake_depth" in str(exc)
    else:
        raise AssertionError("missing max_lake_depth should fail")


def test_zone_extractor_drainage_or_shallow_lake_ignores_catchment_splits():
    frame = DemFrame(
        time=0,
        elevation=np.zeros((1, 2)),
        x=np.arange(2),
        y=np.arange(1),
        drainage_area=np.array([[0.0, 0.0]]),
        catchment=np.array([[1, 2]]),
        water_depth=np.array([[10.0, 10.0]]),
    )
    extractor = ZoneExtractor(
        min_drainage_area=5.0,
        habitat_mode="drainage_or_shallow_lake",
        max_lake_depth=30.0,
        use_catchments=True,
        connectivity=4,
    )
    zone_map = extractor.extract(frame)

    assert len(zone_map.zones) == 1
    assert zone_map.zones[0].area_cells == 2


def test_zone_extractor_elevation_only_mode_ignores_catchments():
    frame = DemFrame(
        time=0,
        elevation=np.array([[120.0, 130.0]]),
        x=np.arange(2),
        y=np.arange(1),
        drainage_area=np.array([[0.0, 0.0]]),
        catchment=np.array([[1, 2]]),
    )
    extractor = ZoneExtractor(
        min_drainage_area=999.0,
        min_elevation=100.0,
        max_elevation=200.0,
        habitat_mode="elevation",
        use_catchments=True,
        connectivity=4,
    )
    zone_map = extractor.extract(frame)

    assert len(zone_map.zones) == 1
    assert zone_map.zones[0].area_cells == 2


def test_zone_extractor_rejects_invalid_habitat_mode():
    try:
        ZoneExtractor(min_drainage_area=1.0, habitat_mode="bad_mode")
    except ValueError as exc:
        assert "habitat_mode" in str(exc)
    else:
        raise AssertionError("invalid habitat_mode should fail")


def test_zone_extractor_filters_small_zones_by_cell_count():
    frame = DemFrame(
        time=0,
        elevation=np.zeros((3, 5)),
        x=np.arange(5),
        y=np.arange(3),
        drainage_area=np.array(
            [
                [10.0, 10.0, 0.0, 10.0, 0.0],
                [0.0, 0.0, 0.0, 10.0, 0.0],
                [10.0, 0.0, 0.0, 10.0, 0.0],
            ]
        ),
    )
    extractor = ZoneExtractor(min_drainage_area=5.0, min_zone_cells=3, connectivity=4)
    zone_map = extractor.extract(frame)

    assert len(zone_map.zones) == 1
    assert zone_map.zones[0].area_cells == 3
    assert zone_map.labels[0, 0] == -1
    assert zone_map.labels[2, 0] == -1
    assert zone_map.habitat_mask.tolist() == [
        [False, False, False, True, False],
        [False, False, False, True, False],
        [False, False, False, True, False],
    ]


def test_zone_extractor_bridges_one_cell_gap_with_tolerance():
    frame = DemFrame(
        time=0,
        elevation=np.zeros((1, 5)),
        x=np.arange(5),
        y=np.arange(1),
        drainage_area=np.array([[10.0, 10.0, 0.0, 10.0, 10.0]]),
    )

    without_tolerance = ZoneExtractor(min_drainage_area=5.0, connectivity=4).extract(frame)
    with_tolerance = ZoneExtractor(min_drainage_area=5.0, connectivity=4, gap_tolerance_cells=1).extract(frame)

    assert len(without_tolerance.zones) == 2
    assert len(with_tolerance.zones) == 1
    assert with_tolerance.labels.tolist() == [[0, 0, 0, 0, 0]]


def test_zone_extractor_respects_gap_tolerance_distance():
    frame = DemFrame(
        time=0,
        elevation=np.zeros((1, 6)),
        x=np.arange(6),
        y=np.arange(1),
        drainage_area=np.array([[10.0, 10.0, 0.0, 0.0, 10.0, 10.0]]),
    )

    tolerance_one = ZoneExtractor(min_drainage_area=5.0, connectivity=4, gap_tolerance_cells=1).extract(frame)
    tolerance_two = ZoneExtractor(min_drainage_area=5.0, connectivity=4, gap_tolerance_cells=2).extract(frame)

    assert len(tolerance_one.zones) == 2
    assert len(tolerance_two.zones) == 1
    assert tolerance_two.labels.tolist() == [[0, 0, 0, 0, 0, 0]]


def test_zone_extractor_gap_tolerance_uses_connectivity():
    frame = DemFrame(
        time=0,
        elevation=np.zeros((3, 3)),
        x=np.arange(3),
        y=np.arange(3),
        drainage_area=np.array(
            [
                [10.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 10.0],
            ]
        ),
    )

    four_neighbor = ZoneExtractor(min_drainage_area=5.0, connectivity=4, gap_tolerance_cells=1).extract(frame)
    eight_neighbor = ZoneExtractor(min_drainage_area=5.0, connectivity=8, gap_tolerance_cells=1).extract(frame)

    assert len(four_neighbor.zones) == 2
    assert len(eight_neighbor.zones) == 1
    assert eight_neighbor.labels[1, 1] == 0


def test_zone_extractor_applies_min_zone_cells_after_gap_bridging():
    frame = DemFrame(
        time=0,
        elevation=np.zeros((1, 5)),
        x=np.arange(5),
        y=np.arange(1),
        drainage_area=np.array([[10.0, 10.0, 0.0, 10.0, 10.0]]),
    )

    zone_map = ZoneExtractor(
        min_drainage_area=5.0,
        connectivity=4,
        gap_tolerance_cells=1,
        min_zone_cells=5,
    ).extract(frame)

    assert len(zone_map.zones) == 1
    assert zone_map.zones[0].area_cells == 5
