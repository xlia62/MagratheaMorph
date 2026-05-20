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
