from magratheamorph.lineage import build_lineage_layout


def test_lineage_layout_has_parent_daughter_split():
    rows = [
        {"species_id": 4, "parent_id": None, "origin_time": 1, "extinction_time": None},
        {"species_id": 13, "parent_id": 4, "origin_time": 4, "extinction_time": None},
    ]
    layout = build_lineage_layout(rows, end_time=5)

    assert 4 in layout["nodes"]
    assert 13 in layout["nodes"]
    split_edges = [branch for branch in layout["branches"] if branch["kind"] == "split"]
    assert split_edges == [
        {
            "kind": "split",
            "parent_id": 4,
            "species_id": 13,
            "x0": 4,
            "x1": 4,
            "y0": layout["nodes"][4]["y"],
            "y1": layout["nodes"][13]["y"],
        }
    ]
    assert layout["nodes"][4]["branch_end"] == 5
    assert layout["nodes"][13]["branch_start"] == 4


def test_lineage_layout_groups_descendants_by_clade_and_origin_time():
    rows = [
        {"species_id": 0, "parent_id": None, "origin_time": 0, "extinction_time": None},
        {"species_id": 1, "parent_id": None, "origin_time": 0, "extinction_time": None},
        {"species_id": 10, "parent_id": 0, "origin_time": 3, "extinction_time": None},
        {"species_id": 11, "parent_id": 0, "origin_time": 2, "extinction_time": None},
        {"species_id": 20, "parent_id": 11, "origin_time": 4, "extinction_time": None},
    ]
    layout = build_lineage_layout(rows, end_time=5)

    ordered_ids = [
        species_id
        for species_id, _node in sorted(layout["nodes"].items(), key=lambda item: -item[1]["y"])
    ]

    assert ordered_ids == [0, 11, 20, 10, 1]
    assert layout["nodes"][0]["daughter_ids"] == [11, 10]
    assert layout["nodes"][11]["daughter_ids"] == [20]
    assert layout["nodes"][1]["is_terminal"] is True


def test_lineage_layout_can_include_only_species_with_parent_daughter_relationships():
    rows = [
        {"species_id": 0, "parent_id": None, "origin_time": 0, "extinction_time": None},
        {"species_id": 1, "parent_id": None, "origin_time": 0, "extinction_time": None},
        {"species_id": 10, "parent_id": 0, "origin_time": 3, "extinction_time": None},
    ]
    layout = build_lineage_layout(rows, end_time=5, involved_only=True)

    assert sorted(layout["nodes"]) == [0, 10]


def test_compact_lineage_layout_reuses_only_non_overlapping_lifetime_segments():
    rows = [
        {"species_id": 0, "parent_id": None, "origin_time": 0, "extinction_time": 5},
        {"species_id": 1, "parent_id": None, "origin_time": 3, "extinction_time": 7},
        {"species_id": 2, "parent_id": None, "origin_time": 5, "extinction_time": None},
    ]
    layout = build_lineage_layout(rows, end_time=10, compact_extinct=True)

    assert layout["nodes"][0]["y"] == layout["nodes"][2]["y"]
    assert layout["nodes"][0]["y"] != layout["nodes"][1]["y"]
    assert layout["nodes"][0]["branch_end"] == 5
    assert layout["nodes"][2]["branch_start"] == 5
