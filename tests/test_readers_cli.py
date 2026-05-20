import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from magratheamorph.model import SpeciationModel
from magratheamorph.outputs import validate_zone_species_rows, write_zone_species_by_time_csv
from magratheamorph.cli import main
from magratheamorph.readers import read_matrix_dem, read_vtk_dem
from magratheamorph.vtk_output import (
    magrathea_vtk_fields,
    raster_to_vtk_point_values,
    read_legacy_vtk_point_fields,
    write_legacy_vtk_with_point_fields,
    write_pvd_collection,
    write_vtk_series_collection,
)
from magratheamorph.zones import Zone, ZoneMap


def test_read_matrix_dem_uses_cell_size_coordinates_and_area():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "dem.txt"
        path.write_text("1 2 3\n4 5 6\n", encoding="utf-8")

        frame = read_matrix_dem(path, time=10.0, cell_size=30.0)

    assert frame.time == 10.0
    assert frame.elevation.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert frame.x.tolist() == [0.0, 30.0, 60.0]
    assert frame.y.tolist() == [30.0, 0.0]
    assert frame.cell_area == 900.0


def test_cli_requires_cell_size_for_matrix_format():
    try:
        main(
            [
                "dem.txt",
                "--format",
                "matrix",
                "--time-to-speciation",
                "1",
                "--min-drainage-area",
                "1",
            ]
        )
    except SystemExit as exc:
        assert "--cell-size is required" in str(exc)
    else:
        raise AssertionError("matrix format without --cell-size should fail")


def test_read_vtk_dem_autodetects_fastscape_fields_when_meshio_is_available():
    try:
        import meshio
    except ImportError:
        return

    points = np.array(
        [
            [0.0, 0.0, 10.0],
            [1.0, 0.0, 11.0],
            [0.0, 1.0, 12.0],
            [1.0, 1.0, 13.0],
        ]
    )
    cells = [("quad", np.array([[0, 1, 3, 2]]))]
    mesh = meshio.Mesh(
        points,
        cells,
        point_data={
            "topography": np.array([100.0, 101.0, 102.0, 103.0]),
            "drainage_area": np.array([10.0, 20.0, 30.0, 40.0]),
            "catchment": np.array([1.0, 1.0, 2.0, 2.0]),
        },
    )

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "surface.vtk"
        meshio.write(path, mesh)
        frame = read_vtk_dem(path)

    assert frame.elevation.tolist() == [[102.0, 103.0], [100.0, 101.0]]
    assert frame.drainage_area.tolist() == [[30.0, 40.0], [10.0, 20.0]]
    assert frame.catchment.tolist() == [[2, 2], [1, 1]]
    assert frame.metadata["elevation_field"] == "topography"


def test_read_legacy_binary_structured_grid_vtk_without_meshio():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=">f4",
    )
    topography = np.array([10.0, 11.0, 12.0, 13.0], dtype=">f4")
    drainage_area = np.array([100.0, 200.0, 300.0, 400.0], dtype=">f4")
    catchment = np.array([1.0, 1.0, 2.0, 2.0], dtype=">f4")

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "surface.vtk"
        with path.open("wb") as f:
            f.write(b"# vtk DataFile Version 3.0\n")
            f.write(b"FastScape\n")
            f.write(b"BINARY\n")
            f.write(b"DATASET STRUCTURED_GRID\n")
            f.write(b"DIMENSIONS 2 2 1\n")
            f.write(b"POINTS 4 float\n")
            f.write(points.tobytes())
            f.write(b"POINT_DATA 4\n")
            f.write(b"SCALARS topography float 1\n")
            f.write(b"LOOKUP_TABLE default\n")
            f.write(topography.tobytes())
            f.write(b"SCALARS drainage_area float 1\n")
            f.write(b"LOOKUP_TABLE default\n")
            f.write(drainage_area.tobytes())
            f.write(b"SCALARS catchment float 1\n")
            f.write(b"LOOKUP_TABLE default\n")
            f.write(catchment.tobytes())

        frame = read_vtk_dem(path)

    assert frame.metadata["reader"] == "legacy_vtk_structured_grid"
    assert frame.elevation.tolist() == [[12.0, 13.0], [10.0, 11.0]]
    assert frame.drainage_area.tolist() == [[300.0, 400.0], [100.0, 200.0]]
    assert frame.catchment.tolist() == [[2, 2], [1, 1]]


def test_write_legacy_vtk_with_point_fields_preserves_original_fields_and_appends_magrathea_fields():
    with TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "surface.vtk"
        output_path = Path(tmp) / "surface_magrathea.vtk"
        _write_tiny_legacy_vtk(input_path)

        labels = np.array([[2, -1], [0, 1]], dtype=int)
        richness = np.array([[3, 0], [1, 2]], dtype=int)
        zone_map = ZoneMap(
            time=0,
            labels=labels,
            zones={
                0: Zone(0, 0, labels == 0),
                1: Zone(1, 0, labels == 1),
                2: Zone(2, 0, labels == 2),
            },
            drainage_area=np.ones((2, 2)),
            habitat_mask=labels >= 0,
        )

        write_legacy_vtk_with_point_fields(input_path, output_path, magrathea_vtk_fields(zone_map, richness))
        fields = read_legacy_vtk_point_fields(output_path)

    assert "topography" in fields
    assert "drainage_area" in fields
    assert "catchment" in fields
    assert fields["topography"].tolist() == [10.0, 11.0, 12.0, 13.0]
    assert fields["magrathea_zone_id"].astype(int).tolist() == [0, 1, 2, -1]
    assert fields["magrathea_species_richness"].astype(int).tolist() == [1, 2, 3, 0]


def test_raster_to_vtk_point_values_converts_from_raster_to_original_point_order():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "surface.vtk"
        _write_tiny_legacy_vtk(path)

        raster = np.array([[30, 40], [10, 20]])
        values = raster_to_vtk_point_values(path, raster)

    assert values.tolist() == [10, 20, 30, 40]


def test_write_pvd_collection_uses_timesteps_and_relative_paths():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        vtk_dir = root / "vtk"
        vtk_dir.mkdir()
        paths = [vtk_dir / f"Topography{i:07d}_magrathea.vtk" for i in (0, 5, 10)]
        for path in paths:
            path.write_text("placeholder", encoding="utf-8")

        pvd_path = root / "magrathea_timeseries.pvd"
        write_pvd_collection(pvd_path, paths, timesteps=[0, 5, 10])
        text = pvd_path.read_text(encoding="utf-8")

    assert '<VTKFile type="Collection"' in text
    assert 'timestep="0"' in text
    assert 'timestep="5"' in text
    assert 'timestep="10"' in text
    assert 'file="vtk/Topography0000000_magrathea.vtk"' in text
    assert 'file="vtk/Topography0000005_magrathea.vtk"' in text
    assert 'file="vtk/Topography0000010_magrathea.vtk"' in text


def test_write_vtk_series_collection_uses_timesteps_and_relative_paths():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        vtk_dir = root / "vtk"
        vtk_dir.mkdir()
        paths = [vtk_dir / f"Topography{i:07d}_magrathea.vtk" for i in (0, 5, 10)]
        for path in paths:
            path.write_text("placeholder", encoding="utf-8")

        series_path = root / "magrathea_timeseries.vtk.series"
        write_vtk_series_collection(series_path, paths, timesteps=[0, 5, 10])
        text = series_path.read_text(encoding="utf-8")

    assert '"file-series-version": "1.0"' in text
    assert '"name": "vtk/Topography0000000_magrathea.vtk"' in text
    assert '"time": 0' in text
    assert '"name": "vtk/Topography0000005_magrathea.vtk"' in text
    assert '"time": 5' in text
    assert '"name": "vtk/Topography0000010_magrathea.vtk"' in text
    assert '"time": 10' in text


def test_write_zone_species_by_time_csv_uses_sorted_species_lists():
    labels = np.array([[0, 1]], dtype=int)
    zone_map = ZoneMap(
        time=0,
        labels=labels,
        zones={0: Zone(0, 0, labels == 0), 1: Zone(1, 0, labels == 1)},
        drainage_area=np.ones((1, 2)),
        habitat_mask=labels >= 0,
    )
    model = SpeciationModel(time_to_speciation=1)
    model.initialize(zone_map, species_per_zone=2)

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "zone_species_by_time.csv"
        write_zone_species_by_time_csv(path, model, [zone_map])
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        header = path.read_text(encoding="utf-8").splitlines()[0]

    assert header == "time,zone_id,species_count,species_ids"
    assert rows == [
        {"time": "0", "zone_id": "0", "species_ids": "0;1", "species_count": "2"},
        {"time": "0", "zone_id": "1", "species_ids": "2;3", "species_count": "2"},
    ]


def test_zone_species_row_validation_catches_count_mismatch():
    try:
        validate_zone_species_rows([{"time": 0, "zone_id": 1, "species_ids": "4;5", "species_count": 1}])
    except ValueError as exc:
        assert "species_count does not match" in str(exc)
    else:
        raise AssertionError("mismatched species_count should fail validation")


def test_cli_default_writes_only_zone_species_csv():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        dem0 = root / "dem0.txt"
        dem1 = root / "dem1.txt"
        dem0.write_text("4 3 2\n3 2 1\n", encoding="utf-8")
        dem1.write_text("4 3 2\n3 2 1\n", encoding="utf-8")
        output_dir = root / "out"
        output_dir.mkdir()
        (output_dir / "species.csv").write_text("stale", encoding="utf-8")
        (output_dir / "events.csv").write_text("stale", encoding="utf-8")

        main(
            [
                str(dem0),
                str(dem1),
                "--format",
                "matrix",
                "--cell-size",
                "1",
                "--time-to-speciation",
                "1",
                "--min-drainage-area",
                "1",
                "--output-dir",
                str(output_dir),
            ]
        )

        csv_names = sorted(path.name for path in output_dir.glob("*.csv"))
        npy_names = sorted(path.name for path in output_dir.glob("*.npy"))

    assert csv_names == ["zone_species_by_time.csv"]
    assert npy_names == []


def test_cli_diagnostics_and_rasters_are_opt_in():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        dem0 = root / "dem0.txt"
        dem1 = root / "dem1.txt"
        dem0.write_text("4 3 2\n3 2 1\n", encoding="utf-8")
        dem1.write_text("4 3 2\n3 2 1\n", encoding="utf-8")
        output_dir = root / "out"

        main(
            [
                str(dem0),
                str(dem1),
                "--format",
                "matrix",
                "--cell-size",
                "1",
                "--time-to-speciation",
                "1",
                "--min-drainage-area",
                "1",
                "--output-dir",
                str(output_dir),
                "--write-diagnostics",
                "--write-rasters",
            ]
        )

        csv_names = sorted(path.name for path in output_dir.glob("*.csv"))
        npy_names = sorted(path.name for path in output_dir.glob("*.npy"))

    assert csv_names == [
        "events.csv",
        "populations_final.csv",
        "ranges.csv",
        "species.csv",
        "zone_species_by_time.csv",
        "zone_transitions.csv",
        "zones.csv",
    ]
    assert npy_names == ["richness_0000.npy", "richness_0001.npy", "zones_0000.npy", "zones_0001.npy"]


def _write_tiny_legacy_vtk(path: Path) -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=">f4",
    )
    topography = np.array([10.0, 11.0, 12.0, 13.0], dtype=">f4")
    drainage_area = np.array([100.0, 200.0, 300.0, 400.0], dtype=">f4")
    catchment = np.array([1.0, 1.0, 2.0, 2.0], dtype=">f4")

    with path.open("wb") as f:
        f.write(b"# vtk DataFile Version 3.0\n")
        f.write(b"FastScape\n")
        f.write(b"BINARY\n")
        f.write(b"DATASET STRUCTURED_GRID\n")
        f.write(b"DIMENSIONS 2 2 1\n")
        f.write(b"POINTS 4 float\n")
        f.write(points.tobytes())
        f.write(b"POINT_DATA 4\n")
        f.write(b"SCALARS topography float 1\n")
        f.write(b"LOOKUP_TABLE default\n")
        f.write(topography.tobytes())
        f.write(b"SCALARS drainage_area float 1\n")
        f.write(b"LOOKUP_TABLE default\n")
        f.write(drainage_area.tobytes())
        f.write(b"SCALARS catchment float 1\n")
        f.write(b"LOOKUP_TABLE default\n")
        f.write(catchment.tobytes())
