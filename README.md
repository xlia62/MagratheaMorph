# Magratheamorph

Standalone tools for modeling river-fish lineage evolution from DEM time series.

The current implementation uses an allopatric/vicariance model:

1. read DEM frames from ASCII grids or ASPECT/FastScape VTK/VTU output,
2. extract river habitat zones from drainage area and optional elevation bounds,
3. track zone overlap through time,
4. split populations when river habitat fragments,
5. create daughter species only after isolation persists longer than a configured threshold.

## Install

```bash
python3 -m pip install -e .
```

Legacy FastScape `.vtk` structured-grid files can be read directly. Other VTK/VTU variants require the optional `meshio` dependency:

```bash
python3 -m pip install -e ".[vtk]"
```

## Command-Line Use

Plain 2-D matrix DEM time series:

```bash
magratheamorph dem_000.txt dem_001.txt dem_002.txt \
  --format matrix \
  --cell-size 1000 \
  --dt 1000 \
  --time-to-speciation 10000 \
  --min-drainage-area 100000 \
  --min-zone-cells 25 \
  --max-elevation 1500 \
  --output-dir output/speciation
```

ESRI ASCII DEM time series:

```bash
magratheamorph dem_000.asc dem_001.asc dem_002.asc \
  --format ascii \
  --dt 1000 \
  --time-to-speciation 10000 \
  --min-drainage-area 100000 \
  --min-zone-cells 25 \
  --max-elevation 1500 \
  --output-dir output/speciation
```

ASPECT/FastScape VTK/VTU time series:

```bash
magratheamorph Topography0000250.vtk Topography0000500.vtk Topography0000750.vtk \
  --format vtk \
  --dt 1000 \
  --time-to-speciation 10000 \
  --min-drainage-area 100000 \
  --min-zone-cells 25 \
  --output-dir output/speciation
```

ASPECT/FastScape river plus shallow-lake habitat:

```bash
magratheamorph Topography0000000.vtk Topography0000005.vtk Topography0000010.vtk \
  --format vtk \
  --dt 1000 \
  --time-to-speciation 10000 \
  --min-drainage-area 100000 \
  --habitat-mode drainage_or_shallow_lake \
  --sea-level-dir /path/to/fastscape \
  --sea-level-field SL \
  --max-lake-depth 30 \
  --min-zone-cells 25 \
  --output-dir output/river_lake_speciation
```

In `drainage_or_shallow_lake` mode, habitat is the union of exposed river cells
(`drainage_area >= min_drainage_area` and `sea_level - elevation < 0`) and
shallow submerged lake cells that are not drainage/channel cells
(`0 <= sea_level - elevation <= max_lake_depth` and
`drainage_area < min_drainage_area`). Submerged channels are removed entirely,
so rivers beneath lake water do not create river-related or lake-related zones.
Sea-level files are matched to topography files by numeric suffix, for example
`Topography0000120.vtk` uses `SeaLevel0000120.vtk`.

For ASPECT/FastScape VTK/VTU files, Magratheamorph reads point-data fields through `meshio`. Elevation is auto-detected from `topography`, then `elevation`, then point z. `drainage_area` and `catchment` are used automatically when present.

Outputs include CSV tables for species, final populations, range history, events, zones, and zone transitions, plus NumPy arrays for zone labels and richness maps.

`--min-zone-cells` removes small extracted habitat components before species inheritance is calculated. This is useful for large DEMs where small noisy channel fragments should not become independent fish zones.

## Visualization

Notebook plotting helpers are available from `magratheamorph.visualization`:

- `plot_zone_map(ax, zone_map)` for zone IDs.
- `plot_species_map(ax, model, zone_map)` for stacked species labels, capped with `+N` in species-rich zones.
- `plot_richness_map(ax, model, zone_map)` for scalable species richness plots.
- `plot_parent_daughter_tree(ax, model.species_table(), end_time=...)` for the parent-daughter species tree.

## Notebook Example

The notebook [notebooks/five_step_allopatric_vicariance_scenarios.ipynb](notebooks/five_step_allopatric_vicariance_scenarios.ipynb) runs a richer five-step river-zone scenario and uses the package visualization helpers. [notebooks/parent_daughter_species_tree_test.ipynb](notebooks/parent_daughter_species_tree_test.ipynb) focuses on the parent-daughter species tree.

## Notes

- If VTK/VTU files contain drainage area and catchment fields, those are used directly.
- If only elevation is available, the package computes a minimal D8 drainage area internally.
- Elevation filters define fish habitat bounds in v1; elevation does not yet drive adaptive speciation.
- Shallow-lake depth filtering requires a sea-level raster or scalar on the same grid and vertical datum as topography.
