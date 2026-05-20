"""Magratheamorph: DEM time-series river fish speciation tools."""

from .hydrology import compute_d8_hydrology
from .lineage import build_lineage_layout, plot_lineage_tree
from .model import SpeciationModel
from .outputs import (
    ZONE_SPECIES_FIELDNAMES,
    remove_diagnostic_csvs,
    validate_zone_species_rows,
    write_csv,
    write_zone_species_by_time_csv,
    zone_species_by_time_rows,
)
from .readers import DemFrame, read_ascii_dem, read_matrix_dem, read_vtk_dem, read_vtu_dem
from .visualization import (
    format_species_label,
    plot_parent_daughter_tree,
    plot_richness_map,
    plot_species_map,
    plot_zone_map,
    species_by_zone_at,
)
from .vtk_output import (
    magrathea_vtk_fields,
    raster_to_vtk_point_values,
    read_legacy_vtk_point_fields,
    richness_to_vtk_points,
    write_legacy_vtk_with_point_fields,
    write_pvd_collection,
    write_vtk_series_collection,
    zone_labels_to_vtk_points,
)
from .zones import ZoneExtractor

__all__ = [
    "DemFrame",
    "SpeciationModel",
    "ZONE_SPECIES_FIELDNAMES",
    "ZoneExtractor",
    "build_lineage_layout",
    "compute_d8_hydrology",
    "format_species_label",
    "plot_lineage_tree",
    "plot_parent_daughter_tree",
    "plot_richness_map",
    "plot_species_map",
    "plot_zone_map",
    "read_ascii_dem",
    "read_matrix_dem",
    "read_vtk_dem",
    "read_vtu_dem",
    "remove_diagnostic_csvs",
    "validate_zone_species_rows",
    "magrathea_vtk_fields",
    "raster_to_vtk_point_values",
    "read_legacy_vtk_point_fields",
    "richness_to_vtk_points",
    "species_by_zone_at",
    "write_legacy_vtk_with_point_fields",
    "write_csv",
    "write_pvd_collection",
    "write_zone_species_by_time_csv",
    "write_vtk_series_collection",
    "zone_labels_to_vtk_points",
    "zone_species_by_time_rows",
]
