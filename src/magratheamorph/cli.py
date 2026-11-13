"""Command-line interface for standalone DEM time-series runs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from .outputs import remove_diagnostic_csvs, write_csv, write_zone_species_by_time_csv
from .pipeline import run_frames
from .readers import read_ascii_dem, read_matrix_dem, read_vtk_dem, read_vtk_sea_level, with_water_depth
from .zones import ZoneExtractor, compute_zone_transitions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="magratheamorph",
        description="Evolve river-fish lineages from a DEM time series.",
    )
    parser.add_argument("inputs", nargs="+", help="Input DEM files in time order.")
    parser.add_argument("--format", choices=["ascii", "matrix", "vtk", "vtu"], required=True, help="Input file format.")
    parser.add_argument("--output-dir", default="magrathea_output", help="Directory for CSV/NumPy outputs.")
    parser.add_argument("--dt", type=float, default=1.0, help="Time interval between input frames.")
    parser.add_argument("--cell-size", type=float, default=None, help="Cell size for plain matrix DEM input.")
    parser.add_argument("--time-to-speciation", type=float, required=True, help="Persistent isolation time required for speciation.")
    parser.add_argument("--min-species-transfer-cells", type=int, default=1, help="Minimum old/new zone overlap required to transfer species.")
    parser.add_argument("--min-drainage-area", type=float, required=True, help="Minimum drainage area for river habitat.")
    parser.add_argument(
        "--habitat-mode",
        choices=["drainage", "elevation", "drainage_and_elevation", "drainage_or_shallow_lake"],
        default="drainage",
        help="Habitat mask source for zone extraction.",
    )
    parser.add_argument("--min-zone-cells", type=int, default=1, help="Minimum number of cells required to keep an extracted zone.")
    parser.add_argument("--gap-tolerance-cells", type=int, default=0, help="Bridge habitat zones separated by up to this many non-habitat cells.")
    parser.add_argument("--min-elevation", type=float, default=None, help="Optional minimum habitat elevation.")
    parser.add_argument("--max-elevation", type=float, default=None, help="Optional maximum habitat elevation.")
    parser.add_argument("--ignore-catchments", action="store_true", help="Do not split zones by catchment IDs.")
    parser.add_argument("--connectivity", choices=[4, 8], type=int, default=8, help="Raster connectivity for zone labeling.")
    parser.add_argument("--elevation-field", default=None, help="VTK/VTU point-data elevation field name. Defaults to topography, then elevation, then point z.")
    parser.add_argument("--drainage-area-field", default="drainage_area", help="VTK/VTU point-data drainage area field name.")
    parser.add_argument("--catchment-field", default="catchment", help="VTK/VTU point-data catchment field name.")
    parser.add_argument("--sea-level-dir", default=None, help="Directory containing SeaLevel*.vtk files matched by numeric suffix.")
    parser.add_argument("--sea-level-field", default="SL", help="VTK/VTU point-data sea-level field name.")
    parser.add_argument("--min-lake-depth", type=float, default=0.0, help="Minimum water depth for shallow lake habitat.")
    parser.add_argument("--max-lake-depth", type=float, default=None, help="Maximum water depth for shallow lake habitat.")
    parser.add_argument("--write-diagnostics", action="store_true", help="Write detailed diagnostic CSV tables.")
    parser.add_argument("--write-rasters", action="store_true", help="Write NumPy zone and richness rasters.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.format == "matrix" and args.cell_size is None:
        raise SystemExit("--cell-size is required when --format matrix is used")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    sea_level_dir = Path(args.sea_level_dir) if args.sea_level_dir is not None else None
    for index, input_path in enumerate(args.inputs):
        time = index * args.dt
        if args.format == "ascii":
            frame = read_ascii_dem(input_path, time=time)
        elif args.format == "matrix":
            frame = read_matrix_dem(input_path, time=time, cell_size=args.cell_size)
        else:
            frame = read_vtk_dem(
                input_path,
                time=time,
                elevation_field=args.elevation_field,
                drainage_area_field=args.drainage_area_field,
                catchment_field=args.catchment_field,
            )
            if sea_level_dir is not None:
                sea_level_path = _matching_sea_level_path(input_path, sea_level_dir)
                sea_level = read_vtk_sea_level(sea_level_path, sea_level_field=args.sea_level_field)
                frame = with_water_depth(frame, sea_level)
        frames.append(frame)

    extractor = ZoneExtractor(
        min_drainage_area=args.min_drainage_area,
        min_elevation=args.min_elevation,
        max_elevation=args.max_elevation,
        use_catchments=not args.ignore_catchments,
        connectivity=args.connectivity,
        min_zone_cells=args.min_zone_cells,
        gap_tolerance_cells=args.gap_tolerance_cells,
        habitat_mode=args.habitat_mode,
        min_lake_depth=args.min_lake_depth,
        max_lake_depth=args.max_lake_depth,
    )
    model, zone_maps = run_frames(
        frames,
        extractor,
        time_to_speciation=args.time_to_speciation,
        min_species_transfer_cells=args.min_species_transfer_cells,
    )

    write_zone_species_by_time_csv(output_dir / "zone_species_by_time.csv", model, zone_maps)

    if args.write_diagnostics:
        write_csv(output_dir / "species.csv", model.species_table())
        write_csv(output_dir / "populations_final.csv", model.population_table())
        write_csv(output_dir / "ranges.csv", model.range_table())
        write_csv(output_dir / "events.csv", model.event_table())
        _write_zone_summary(output_dir / "zones.csv", zone_maps)
        _write_transitions(output_dir / "zone_transitions.csv", zone_maps)
    else:
        remove_diagnostic_csvs(output_dir)

    if args.write_rasters:
        for index, zone_map in enumerate(zone_maps):
            np.save(output_dir / f"zones_{index:04d}.npy", zone_map.labels)
            np.save(output_dir / f"richness_{index:04d}.npy", model.richness_map_at(zone_map))

    print(f"Wrote outputs to {output_dir}")


def _write_zone_summary(path: Path, zone_maps) -> None:
    rows = []
    for zone_map in zone_maps:
        for zone in zone_map.zones.values():
            rows.append(
                {
                    "time": zone_map.time,
                    "zone_id": zone.zone_id,
                    "catchment_id": zone.catchment_id,
                    "area_cells": zone.area_cells,
                }
            )
    write_csv(path, rows)


def _write_transitions(path: Path, zone_maps) -> None:
    rows = []
    for old, new in zip(zone_maps[:-1], zone_maps[1:]):
        for transition in compute_zone_transitions(old, new):
            rows.append(
                {
                    "time_old": old.time,
                    "time_new": new.time,
                    "old_zone_id": transition.old_zone_id,
                    "new_zone_id": transition.new_zone_id,
                    "overlap_cells": transition.overlap_cells,
                }
            )
    write_csv(path, rows)


def _matching_sea_level_path(input_path: str | Path, sea_level_dir: Path) -> Path:
    input_path = Path(input_path)
    match = re.search(r"(\d+)$", input_path.stem)
    if match is None:
        raise ValueError(f"could not infer numeric timestep suffix from {input_path.name!r}")
    suffix = match.group(1)
    candidates = [
        sea_level_dir / f"SeaLevel{suffix}{input_path.suffix}",
        sea_level_dir / f"SeaLevel{suffix}.vtk",
        sea_level_dir / f"SeaLevel{suffix}.vtu",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no SeaLevel file with suffix {suffix} found in {sea_level_dir}")
