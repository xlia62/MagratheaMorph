import numpy as np

from magratheamorph.model import SpeciationModel
from magratheamorph.readers import DemFrame
from magratheamorph.visualization import (
    format_species_label,
    plot_parent_daughter_tree,
    plot_richness_map,
    plot_species_map,
    plot_zone_map,
    species_by_zone_at,
)
from magratheamorph.zones import ZoneExtractor


def test_format_species_label_caps_long_species_lists():
    assert format_species_label([5, 6, 11], max_labels=8) == "5\n6\n11"
    assert format_species_label(list(range(10)), max_labels=4) == "0\n1\n2\n3\n+6"


def test_species_by_zone_at_returns_sorted_species_lists():
    zone_map = _zone_map()
    model = SpeciationModel(time_to_speciation=1)
    model.initialize(zone_map, species_per_zone=3)

    assert species_by_zone_at(model, zone_map.time) == {0: [0, 1, 2]}


def test_visualization_functions_smoke_with_matplotlib_agg():
    import importlib.util

    if importlib.util.find_spec("matplotlib") is None:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zone_map = _zone_map()
    model = SpeciationModel(time_to_speciation=1)
    model.initialize(zone_map, species_per_zone=3)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    plot_zone_map(axes[0], zone_map)
    plot_species_map(axes[1], model, zone_map, max_labels_per_zone=2)
    plot_richness_map(axes[2], model, zone_map)
    layout = plot_parent_daughter_tree(
        axes[3],
        [
            {"species_id": 4, "parent_id": None, "origin_time": 1, "extinction_time": None},
            {"species_id": 13, "parent_id": 4, "origin_time": 3, "extinction_time": None},
        ],
        end_time=5,
    )
    plt.close(fig)

    assert layout["nodes"][4]["branch_end"] == 5
    assert layout["nodes"][13]["branch_start"] == 3


def test_parent_daughter_tree_labels_extinct_species_by_id_only_with_compact_layout():
    import importlib.util

    if importlib.util.find_spec("matplotlib") is None:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        {"species_id": 0, "parent_id": None, "origin_time": 0, "extinction_time": 3},
        {"species_id": 1, "parent_id": None, "origin_time": 3, "extinction_time": None},
    ]
    fig, ax = plt.subplots()
    layout = plot_parent_daughter_tree(
        ax,
        rows,
        end_time=6,
        involved_only=False,
        compact_extinct=True,
        show_extinct_labels=True,
        extinct_label_suffix="",
    )
    labels = [text.get_text() for text in ax.texts]
    plt.close(fig)

    assert layout["nodes"][0]["y"] == layout["nodes"][1]["y"]
    assert "0" in labels
    assert "0 extinct" not in labels
    assert "1" in labels


def test_parent_daughter_tree_marks_true_split_events():
    import importlib.util

    if importlib.util.find_spec("matplotlib") is None:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        {"species_id": 0, "parent_id": None, "origin_time": 0, "extinction_time": None},
        {"species_id": 1, "parent_id": 0, "origin_time": 3, "extinction_time": None},
    ]
    fig, ax = plt.subplots()
    plot_parent_daughter_tree(
        ax,
        rows,
        end_time=6,
        split_color="tab:red",
        split_marker=True,
        split_marker_color="tab:blue",
        split_marker_size=7.5,
    )
    split_markers = [
        line
        for line in ax.lines
        if line.get_color() == "tab:blue" and line.get_marker() == "o"
    ]
    plt.close(fig)

    assert split_markers
    assert split_markers[0].get_markersize() == 7.5


def _zone_map():
    frame = DemFrame(
        time=0,
        elevation=np.zeros((2, 2)),
        x=np.arange(2),
        y=np.arange(2),
        drainage_area=np.ones((2, 2)),
    )
    return ZoneExtractor(min_drainage_area=0.5, use_catchments=False).extract(frame)
