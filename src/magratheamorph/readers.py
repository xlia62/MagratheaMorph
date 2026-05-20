"""Input readers and common DEM frame representation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DemFrame:
    """Topographic data for one model timestep.

    Parameters
    ----------
    time
        Model time associated with the frame.
    elevation
        2-D elevation raster.
    x, y
        1-D or 2-D coordinates. If 1-D, they are interpreted as raster axes.
    drainage_area
        Optional 2-D drainage area raster.
    catchment
        Optional 2-D catchment ID raster.
    metadata
        Optional input-specific metadata.
    """

    time: float
    elevation: np.ndarray
    x: np.ndarray
    y: np.ndarray
    drainage_area: np.ndarray | None = None
    catchment: np.ndarray | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        elevation = np.asarray(self.elevation)
        if elevation.ndim != 2:
            raise ValueError("elevation must be a 2-D array")

        for name in ("drainage_area", "catchment"):
            value = getattr(self, name)
            if value is not None and np.asarray(value).shape != elevation.shape:
                raise ValueError(f"{name} must match elevation shape")

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation.shape

    @property
    def cell_area(self) -> float:
        """Return raster cell area from regular x/y coordinates."""

        x = np.asarray(self.x)
        y = np.asarray(self.y)
        if x.ndim == 2:
            dx_values = np.diff(x, axis=1)
            dx = float(np.nanmedian(np.abs(dx_values[dx_values != 0])))
        else:
            dx = float(np.nanmedian(np.abs(np.diff(x)))) if x.size > 1 else 1.0

        if y.ndim == 2:
            dy_values = np.diff(y, axis=0)
            dy = float(np.nanmedian(np.abs(dy_values[dy_values != 0])))
        else:
            dy = float(np.nanmedian(np.abs(np.diff(y)))) if y.size > 1 else dx

        return dx * dy


def read_ascii_dem(path: str | Path, time: float = 0.0) -> DemFrame:
    """Read an ESRI ASCII grid DEM.

    The reader supports standard headers with `ncols`, `nrows`, `xllcorner` or
    `xllcenter`, `yllcorner` or `yllcenter`, `cellsize`, and optional
    `NODATA_value`.
    """

    path = Path(path)
    header: dict[str, float] = {}
    header_lines = []

    with path.open("r", encoding="utf-8") as f:
        for _ in range(6):
            line = f.readline()
            if not line:
                raise ValueError(f"{path} ended before ASCII grid header completed")
            key, value = line.split()[:2]
            header[key.lower()] = float(value)
            header_lines.append(line)

    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    cellsize = float(header["cellsize"])
    nodata = header.get("nodata_value")

    data = np.loadtxt(path, skiprows=len(header_lines))
    if data.shape != (nrows, ncols):
        raise ValueError(f"DEM data shape {data.shape} does not match header {(nrows, ncols)}")
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)

    x0 = header.get("xllcenter", header.get("xllcorner", 0.0))
    y0 = header.get("yllcenter", header.get("yllcorner", 0.0))
    if "xllcorner" in header:
        x0 += 0.5 * cellsize
    if "yllcorner" in header:
        y0 += 0.5 * cellsize

    x = x0 + np.arange(ncols) * cellsize
    # ASCII grids are written north-to-south. Keep array order and assign y descending.
    y = y0 + (nrows - 1 - np.arange(nrows)) * cellsize

    return DemFrame(time=time, elevation=data, x=x, y=y, metadata={"path": str(path)})


def read_matrix_dem(path: str | Path, time: float = 0.0, cell_size: float = 1.0) -> DemFrame:
    """Read a plain whitespace-separated 2-D DEM matrix.

    Unlike ESRI ASCII grids, this format has no header. Coordinates are created
    from ``cell_size`` with y descending so array row order is treated like a
    north-to-south raster.
    """

    if cell_size <= 0:
        raise ValueError("cell_size must be positive")

    path = Path(path)
    data = np.loadtxt(path)
    if data.ndim != 2:
        raise ValueError("matrix DEM must be a 2-D numeric array")

    nrows, ncols = data.shape
    x = np.arange(ncols) * cell_size
    y = (nrows - 1 - np.arange(nrows)) * cell_size
    return DemFrame(time=time, elevation=data, x=x, y=y, metadata={"path": str(path), "format": "matrix"})


def read_vtk_dem(
    path: str | Path,
    time: float = 0.0,
    elevation_field: str | None = None,
    drainage_area_field: str | None = "drainage_area",
    catchment_field: str | None = "catchment",
) -> DemFrame:
    """Read a regular FastScape/ASPECT VTK/VTU surface using optional `meshio`.

    This reader expects point data that can be reshaped onto a regular x-y
    grid. Field names can be adjusted to match a particular ASPECT/FastScape
    output setup.
    """

    path = Path(path)
    if path.suffix.lower() == ".vtk":
        try:
            return _read_legacy_structured_grid_vtk(
                path,
                time=time,
                elevation_field=elevation_field,
                drainage_area_field=drainage_area_field,
                catchment_field=catchment_field,
            )
        except ValueError:
            pass

    try:
        import meshio
    except ImportError as exc:
        raise ImportError("read_vtk_dem requires the optional dependency `meshio`") from exc

    mesh = meshio.read(path)
    points = np.asarray(mesh.points)
    return _frame_from_points_and_point_data(
        path=path,
        time=time,
        points=points,
        point_data=mesh.point_data,
        elevation_field=elevation_field,
        drainage_area_field=drainage_area_field,
        catchment_field=catchment_field,
        reader="meshio",
    )


def _frame_from_points_and_point_data(
    path: Path,
    time: float,
    points: np.ndarray,
    point_data: dict[str, Any],
    elevation_field: str | None,
    drainage_area_field: str | None,
    catchment_field: str | None,
    reader: str,
) -> DemFrame:

    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("VTK points must contain at least x and y coordinates")

    elevation_name = _resolve_elevation_field(point_data, elevation_field)
    if elevation_name is None:
        if points.shape[1] < 3:
            raise ValueError("no elevation field found and VTK points have no z coordinate")
        elevation_values = points[:, 2]
        elevation_source = "point_z"
    else:
        elevation_values = _point_values(point_data[elevation_name], len(points), elevation_name)
        elevation_source = elevation_name

    x_unique = np.unique(points[:, 0])
    y_unique = np.unique(points[:, 1])
    nrows, ncols = len(y_unique), len(x_unique)
    if nrows * ncols != len(points):
        raise ValueError("VTK points do not form a complete regular raster grid")

    x_order = {value: i for i, value in enumerate(x_unique)}
    y_desc = y_unique[::-1]
    y_order = {value: i for i, value in enumerate(y_desc)}
    elevation = np.full((nrows, ncols), np.nan)
    drainage_values = None
    catchment_values = None
    if drainage_area_field is not None and drainage_area_field in point_data:
        drainage_values = _point_values(point_data[drainage_area_field], len(points), drainage_area_field)
    if catchment_field is not None and catchment_field in point_data:
        catchment_values = _point_values(point_data[catchment_field], len(points), catchment_field)
    drainage_area = np.full((nrows, ncols), np.nan) if drainage_values is not None else None
    catchment = np.full((nrows, ncols), -1, dtype=int) if catchment_values is not None else None

    for point_index, point in enumerate(points):
        row = y_order[point[1]]
        col = x_order[point[0]]
        elevation[row, col] = elevation_values[point_index]
        if drainage_area is not None:
            drainage_area[row, col] = drainage_values[point_index]
        if catchment is not None:
            catchment[row, col] = int(catchment_values[point_index])

    return DemFrame(
        time=time,
        elevation=elevation,
        x=x_unique,
        y=y_desc,
        drainage_area=drainage_area,
        catchment=catchment,
        metadata={
            "path": str(path),
            "reader": reader,
            "elevation_field": elevation_source,
            "drainage_area_field": drainage_area_field if drainage_values is not None else None,
            "catchment_field": catchment_field if catchment_values is not None else None,
        },
    )


def read_vtu_dem(
    path: str | Path,
    time: float = 0.0,
    elevation_field: str | None = None,
    drainage_area_field: str | None = "drainage_area",
    catchment_field: str | None = "catchment",
) -> DemFrame:
    """Backward-compatible alias for :func:`read_vtk_dem`."""

    return read_vtk_dem(
        path,
        time=time,
        elevation_field=elevation_field,
        drainage_area_field=drainage_area_field,
        catchment_field=catchment_field,
    )


def _resolve_elevation_field(point_data: dict[str, Any], requested: str | None) -> str | None:
    if requested is not None:
        if requested not in point_data:
            raise ValueError(f"elevation field {requested!r} not found in VTK point data")
        return requested
    for candidate in ("topography", "elevation"):
        if candidate in point_data:
            return candidate
    return None


def _point_values(values: Any, npoints: int, field_name: str) -> np.ndarray:
    array = np.asarray(values).squeeze()
    if array.ndim != 1 or array.shape[0] != npoints:
        raise ValueError(f"point-data field {field_name!r} must have one value per point")
    return array


def _read_legacy_structured_grid_vtk(
    path: Path,
    time: float,
    elevation_field: str | None,
    drainage_area_field: str | None,
    catchment_field: str | None,
) -> DemFrame:
    data = path.read_bytes()
    pos = 0
    _version, pos = _read_vtk_line(data, pos)
    _title, pos = _read_vtk_line(data, pos)
    mode, pos = _read_vtk_line(data, pos)
    if mode.upper() != "BINARY":
        raise ValueError("legacy VTK fallback currently supports BINARY files")

    dataset_line, pos = _read_vtk_line(data, pos)
    dataset_parts = dataset_line.split()
    if len(dataset_parts) < 2 or dataset_parts[0].upper() != "DATASET" or dataset_parts[1].upper() != "STRUCTURED_GRID":
        raise ValueError("legacy VTK fallback only supports DATASET STRUCTURED_GRID")

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
    pos += point_bytes

    point_data: dict[str, np.ndarray] = {}
    while pos < len(data):
        line, pos = _read_vtk_line(data, pos)
        if not line:
            continue
        parts = line.split()
        keyword = parts[0].upper()
        if keyword == "POINT_DATA":
            if len(parts) != 2:
                raise ValueError("legacy VTK POINT_DATA line is malformed")
            point_data_count = int(parts[1])
            if point_data_count != npoints:
                raise ValueError("legacy VTK POINT_DATA count does not match POINTS")
        elif keyword == "SCALARS":
            if len(parts) < 3:
                raise ValueError("legacy VTK SCALARS line is malformed")
            name = parts[1]
            scalar_dtype = _legacy_vtk_dtype(parts[2])
            components = int(parts[3]) if len(parts) > 3 else 1
            lookup_line, pos = _read_vtk_line(data, pos)
            if not lookup_line.upper().startswith("LOOKUP_TABLE"):
                raise ValueError("legacy VTK SCALARS missing LOOKUP_TABLE")
            value_count = npoints * components
            value_bytes = value_count * scalar_dtype.itemsize
            if pos + value_bytes > len(data):
                raise ValueError(f"legacy VTK SCALARS field {name!r} is truncated")
            values = np.frombuffer(data, dtype=scalar_dtype, count=value_count, offset=pos).astype(float)
            pos += value_bytes
            point_data[name] = values if components == 1 else values.reshape(npoints, components)
        else:
            raise ValueError(f"legacy VTK fallback does not support section {keyword!r}")

    return _frame_from_points_and_point_data(
        path=path,
        time=time,
        points=points,
        point_data=point_data,
        elevation_field=elevation_field,
        drainage_area_field=drainage_area_field,
        catchment_field=catchment_field,
        reader="legacy_vtk_structured_grid",
    )


def _read_vtk_line(data: bytes, pos: int) -> tuple[str, int]:
    while pos < len(data) and data[pos] in b" \t\r\n":
        pos += 1
    end = data.find(b"\n", pos)
    if end < 0:
        end = len(data)
    line = data[pos:end].decode("ascii", errors="strict").strip()
    return line, end + 1


def _legacy_vtk_dtype(name: str) -> np.dtype:
    normalized = name.lower()
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
    if normalized not in mapping:
        raise ValueError(f"unsupported legacy VTK data type {name!r}")
    return np.dtype(mapping[normalized])
