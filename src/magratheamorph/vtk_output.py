"""Legacy VTK output helpers for Magratheamorph derived fields."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np

from .zones import ZoneMap


def write_legacy_vtk_with_point_fields(
    input_path: str | Path,
    output_path: str | Path,
    fields: dict[str, np.ndarray],
    preserve_existing_point_data: bool = True,
) -> None:
    """Copy a legacy structured-grid VTK file and append point-data fields.

    By default, the original file bytes are preserved exactly and new scalar
    fields are appended to the existing ``POINT_DATA`` section as big-endian
    binary VTK arrays. If ``preserve_existing_point_data`` is False, original
    point-data fields are stripped and only the supplied fields are written.
    Field values may be provided either as 1-D arrays in original VTK point
    order, or as 2-D Magratheamorph rasters with shape ``(ny, nx)``.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    info = inspect_legacy_structured_grid_vtk(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    original = input_path.read_bytes()
    with output_path.open("wb") as f:
        if preserve_existing_point_data:
            f.write(original)
        else:
            f.write(original[: info["points_end_offset"]])
            if not original[: info["points_end_offset"]].endswith(b"\n"):
                f.write(b"\n")
            f.write(f"POINT_DATA       {info['npoints']}\n".encode("ascii"))
        if not original.endswith(b"\n"):
            f.write(b"\n")
        for name, values in fields.items():
            point_values = raster_to_vtk_point_values(input_path, values, vtk_info=info)
            dtype_name, encoded = _encode_vtk_scalar_values(point_values)
            f.write(f"SCALARS {name} {dtype_name} 1\n".encode("ascii"))
            f.write(b"LOOKUP_TABLE default\n")
            f.write(encoded)
            f.write(b"\n")


def write_pvd_collection(
    output_path: str | Path,
    vtk_paths: list[str | Path],
    timesteps: list[float] | None = None,
) -> None:
    """Write a ParaView ``.pvd`` collection for a VTK time series."""

    output_path = Path(output_path)
    if timesteps is None:
        timesteps = [float(index) for index in range(len(vtk_paths))]
    if len(timesteps) != len(vtk_paths):
        raise ValueError("timesteps must have the same length as vtk_paths")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>',
    ]
    for timestep, vtk_path in zip(timesteps, vtk_paths):
        vtk_path = Path(vtk_path)
        try:
            file_value = vtk_path.resolve().relative_to(output_path.parent.resolve())
        except ValueError:
            file_value = vtk_path
        lines.append(
            f'    <DataSet timestep="{timestep:g}" group="" part="0" file="{escape(str(file_value))}"/>'
        )
    lines.extend(
        [
            "  </Collection>",
            "</VTKFile>",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_vtk_series_collection(
    output_path: str | Path,
    vtk_paths: list[str | Path],
    timesteps: list[float] | None = None,
) -> None:
    """Write a ParaView ``.vtk.series`` file for a legacy VTK time series."""

    output_path = Path(output_path)
    if timesteps is None:
        timesteps = [float(index) for index in range(len(vtk_paths))]
    if len(timesteps) != len(vtk_paths):
        raise ValueError("timesteps must have the same length as vtk_paths")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for timestep, vtk_path in zip(timesteps, vtk_paths):
        vtk_path = Path(vtk_path)
        try:
            name = str(vtk_path.resolve().relative_to(output_path.parent.resolve()))
        except ValueError:
            name = str(vtk_path)
        files.append({"name": name, "time": timestep})

    output = {
        "file-series-version": "1.0",
        "files": files,
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


def zone_labels_to_vtk_points(input_path: str | Path, labels: np.ndarray) -> np.ndarray:
    """Convert a zone-label raster to original VTK point order."""

    return raster_to_vtk_point_values(input_path, labels)


def richness_to_vtk_points(input_path: str | Path, richness: np.ndarray) -> np.ndarray:
    """Convert a species-richness raster to original VTK point order."""

    return raster_to_vtk_point_values(input_path, richness)


def raster_to_vtk_point_values(
    input_path: str | Path,
    values: np.ndarray,
    *,
    vtk_info: dict[str, Any] | None = None,
) -> np.ndarray:
    """Return values in the original point order of a legacy VTK file."""

    info = vtk_info or inspect_legacy_structured_grid_vtk(input_path)
    array = np.asarray(values)
    npoints = info["npoints"]
    if array.ndim == 1:
        if array.shape[0] != npoints:
            raise ValueError(f"1-D field has {array.shape[0]} values, expected {npoints}")
        return array
    if array.ndim != 2:
        raise ValueError("field values must be either 1-D point values or a 2-D raster")

    nx, ny, nz = info["dimensions"]
    if nz != 1:
        raise ValueError("2-D raster conversion only supports legacy VTK grids with nz=1")
    if array.shape != (ny, nx):
        raise ValueError(f"raster field shape {array.shape} does not match VTK grid shape {(ny, nx)}")

    points = info["points"]
    x_unique = np.unique(points[:, 0])
    y_desc = np.unique(points[:, 1])[::-1]
    x_order = {value: index for index, value in enumerate(x_unique)}
    y_order = {value: index for index, value in enumerate(y_desc)}
    out = np.empty(npoints, dtype=array.dtype)
    for point_index, point in enumerate(points):
        out[point_index] = array[y_order[point[1]], x_order[point[0]]]
    return out


def magrathea_vtk_fields(zone_map: ZoneMap, richness_map: np.ndarray) -> dict[str, np.ndarray]:
    """Return the standard Magratheamorph VTK output fields."""

    if richness_map.shape != zone_map.labels.shape:
        raise ValueError("richness_map must match zone_map labels shape")
    richness = np.where(zone_map.labels >= 0, richness_map, 0)
    return {
        "magrathea_zone_id": zone_map.labels.astype(np.int32, copy=False),
        "magrathea_species_richness": richness.astype(np.int32, copy=False),
    }


def inspect_legacy_structured_grid_vtk(path: str | Path) -> dict[str, Any]:
    """Inspect a binary legacy ``DATASET STRUCTURED_GRID`` VTK file."""

    path = Path(path)
    data = path.read_bytes()
    pos = 0
    _version, pos = _read_vtk_line(data, pos)
    title, pos = _read_vtk_line(data, pos)
    mode, pos = _read_vtk_line(data, pos)
    if mode.upper() != "BINARY":
        raise ValueError("only BINARY legacy VTK files are supported")

    dataset_line, pos = _read_vtk_line(data, pos)
    dataset_parts = dataset_line.split()
    if len(dataset_parts) < 2 or dataset_parts[0].upper() != "DATASET" or dataset_parts[1].upper() != "STRUCTURED_GRID":
        raise ValueError("only legacy VTK DATASET STRUCTURED_GRID files are supported")

    dimensions_line, pos = _read_vtk_line(data, pos)
    dimensions_parts = dimensions_line.split()
    if len(dimensions_parts) != 4 or dimensions_parts[0].upper() != "DIMENSIONS":
        raise ValueError("legacy VTK STRUCTURED_GRID missing DIMENSIONS")
    nx, ny, nz = (int(value) for value in dimensions_parts[1:])

    points_line, pos = _read_vtk_line(data, pos)
    points_parts = points_line.split()
    if len(points_parts) != 3 or points_parts[0].upper() != "POINTS":
        raise ValueError("legacy VTK STRUCTURED_GRID missing POINTS")
    npoints = int(points_parts[1])
    if npoints != nx * ny * nz:
        raise ValueError("legacy VTK POINTS count does not match DIMENSIONS")

    point_dtype = _legacy_vtk_dtype(points_parts[2])
    point_count = npoints * 3
    point_bytes = point_count * point_dtype.itemsize
    if pos + point_bytes > len(data):
        raise ValueError("legacy VTK POINTS data is truncated")
    points = np.frombuffer(data, dtype=point_dtype, count=point_count, offset=pos).astype(float).reshape(npoints, 3)
    points_end_offset = pos + point_bytes

    return {
        "path": str(path),
        "title": title,
        "dimensions": (nx, ny, nz),
        "npoints": npoints,
        "point_count": npoints,
        "points": points,
        "points_end_offset": points_end_offset,
    }


def read_legacy_vtk_point_fields(
    path: str | Path,
    field_names: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict[str, np.ndarray]:
    """Read scalar point-data fields from a binary legacy structured-grid VTK file."""

    data = Path(path).read_bytes()
    info = inspect_legacy_structured_grid_vtk(path)
    npoints = info["npoints"]
    point_data_pos = data.find(b"POINT_DATA")
    if point_data_pos < 0:
        return {}

    wanted = set(field_names) if field_names is not None else None
    fields: dict[str, np.ndarray] = {}
    pos = point_data_pos
    while pos < len(data):
        line, pos = _read_vtk_line(data, pos)
        if not line:
            continue
        parts = line.split()
        keyword = parts[0].upper()
        if keyword == "POINT_DATA":
            continue
        if keyword != "SCALARS":
            raise ValueError(f"unsupported VTK point-data section {keyword!r}")
        name = parts[1]
        dtype = _legacy_vtk_dtype(parts[2])
        components = int(parts[3]) if len(parts) > 3 else 1
        lookup_line, pos = _read_vtk_line(data, pos)
        if not lookup_line.upper().startswith("LOOKUP_TABLE"):
            raise ValueError(f"SCALARS field {name!r} missing LOOKUP_TABLE")
        count = npoints * components
        nbytes = count * dtype.itemsize
        values = np.frombuffer(data, dtype=dtype, count=count, offset=pos).astype(float)
        pos += nbytes
        if wanted is None or name in wanted:
            fields[name] = values if components == 1 else values.reshape(npoints, components)
    return fields


def _encode_vtk_scalar_values(values: np.ndarray) -> tuple[str, bytes]:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.integer):
        return "int", array.astype(">i4", copy=False).tobytes()
    return "float", array.astype(">f4", copy=False).tobytes()


def _read_vtk_line(data: bytes, pos: int) -> tuple[str, int]:
    while pos < len(data) and data[pos] in b" \t\r\n":
        pos += 1
    end = data.find(b"\n", pos)
    if end < 0:
        end = len(data)
    line = data[pos:end].decode("ascii", errors="strict").strip()
    return line, end + 1


def _legacy_vtk_dtype(name: str) -> np.dtype:
    mapping = {
        "float": ">f4",
        "double": ">f8",
        "int": ">i4",
        "unsigned_int": ">u4",
        "short": ">i2",
        "unsigned_short": ">u2",
        "char": ">i1",
        "unsigned_char": ">u1",
    }
    normalized = name.lower()
    if normalized not in mapping:
        raise ValueError(f"unsupported legacy VTK data type {name!r}")
    return np.dtype(mapping[normalized])
