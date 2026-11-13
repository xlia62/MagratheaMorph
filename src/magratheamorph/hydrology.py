"""Minimal D8 hydrology for DEM-only inputs."""

from __future__ import annotations

import numpy as np

_D8_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def compute_d8_hydrology(elevation: np.ndarray, cell_area: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute D8 receiver and drainage area from a DEM.

    Sinks and flats are treated as local outlets. This is intentionally small
    and deterministic; production workflows can later swap in Landlab or
    pysheds for sink filling and more sophisticated routing.
    """

    z = np.asarray(elevation, dtype=float)
    if z.ndim != 2:
        raise ValueError("elevation must be 2-D")

    nrows, ncols = z.shape
    receiver = np.full(z.shape, -1, dtype=int)

    for row in range(nrows):
        for col in range(ncols):
            if not np.isfinite(z[row, col]):
                continue

            best_slope = 0.0
            best_index = row * ncols + col
            for dr, dc in _D8_OFFSETS:
                rr, cc = row + dr, col + dc
                if rr < 0 or rr >= nrows or cc < 0 or cc >= ncols:
                    continue
                if not np.isfinite(z[rr, cc]):
                    continue
                distance = np.sqrt(2.0) if dr and dc else 1.0
                slope = (z[row, col] - z[rr, cc]) / distance
                if slope > best_slope:
                    best_slope = slope
                    best_index = rr * ncols + cc

            receiver[row, col] = best_index

    drainage_area = np.where(np.isfinite(z), cell_area, 0.0).ravel()
    order = np.argsort(np.where(np.isfinite(z.ravel()), z.ravel(), -np.inf))[::-1]
    flat_receiver = receiver.ravel()

    for node in order:
        rec = flat_receiver[node]
        if rec >= 0 and rec != node:
            drainage_area[rec] += drainage_area[node]

    return receiver, drainage_area.reshape(z.shape)

