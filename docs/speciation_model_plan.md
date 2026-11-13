# DEM Time-Series River Fish Speciation Model

## Summary

Build a standalone Python program that models river-fish diversification from DEM time series using **allopatric/vicariance speciation**. A landscape model, such as ASPECT-FastScape, produces topographic outputs; this program reads those outputs afterward, extracts river habitat zones, tracks how zones split/merge/disappear through time, and evolves fish lineages from those connectivity changes.

The v1 theory is connectivity-based: geographic isolation can lead to speciation, reconnection before speciation cancels isolation, and reconnection after speciation creates secondary contact rather than automatic species fusion.

## Key Design

- Primary inputs are ASPECT/FastScape VTU output; ASCII DEM time series and in-memory arrays are supported as fallback/prototyping paths.
- If input contains `drainage_area` and `catchment`, use those directly.
- If only elevation is available, compute D8 flow direction and drainage area from the DEM.
- River habitat is `drainage_area >= min_drainage_area`, optionally filtered by elevation bounds.
- A zone is connected river habitat, optionally separated by catchment ID.
- Zones are timestep-specific; species/populations are transferred to successor zones through spatial overlap.
- Fragmentation creates isolated populations first; persistent isolation beyond `time_to_speciation` creates daughter species.
- Reconnection before speciation cancels isolation; reconnection after speciation creates secondary contact.
- Extinction occurs when all populations of a species are lost.

## Outputs

- Zone maps per timestep.
- Zone transition/event table.
- Species range table through time.
- Speciation/extinction event table.
- Richness map/table.
- Lineage table suitable for phylogenetic reconstruction.
