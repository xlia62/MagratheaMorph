"""Tabular output helpers for Magratheamorph."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


ZONE_SPECIES_FIELDNAMES = ["time", "zone_id", "species_count", "species_ids"]
DIAGNOSTIC_CSV_NAMES = [
    "events.csv",
    "populations_final.csv",
    "ranges.csv",
    "species.csv",
    "zone_transitions.csv",
    "zones.csv",
]


def zone_species_by_time_rows(model, zone_maps) -> list[dict]:
    """Return species occupancy rows for each zone through time."""

    range_rows = model.range_table()
    rows = []
    for zone_map in zone_maps:
        species_by_zone: dict[int, set[int]] = {}
        for row in range_rows:
            if row["time"] == zone_map.time:
                species_by_zone.setdefault(row["zone_id"], set()).add(row["species_id"])
        for zone_id in sorted(zone_map.zones):
            species_ids = sorted(species_by_zone.get(zone_id, []))
            rows.append(
                {
                    "time": zone_map.time,
                    "zone_id": zone_id,
                    "species_ids": ";".join(str(species_id) for species_id in species_ids),
                    "species_count": len(species_ids),
                }
            )
    validate_zone_species_rows(rows)
    return rows


def write_zone_species_by_time_csv(path: str | Path, model, zone_maps) -> None:
    """Write the main zone-species occupancy CSV."""

    write_csv(
        path,
        zone_species_by_time_rows(model, zone_maps),
        fieldnames=ZONE_SPECIES_FIELDNAMES,
    )


def remove_diagnostic_csvs(output_dir: str | Path) -> None:
    """Remove known diagnostic CSVs from an output directory."""

    output_dir = Path(output_dir)
    for name in DIAGNOSTIC_CSV_NAMES:
        path = output_dir / name
        if path.exists():
            path.unlink()


def validate_zone_species_rows(rows: Iterable[dict]) -> None:
    """Validate that species list length matches species_count."""

    for row in rows:
        species_ids = str(row["species_ids"])
        parsed_ids = [value for value in species_ids.split(";") if value]
        if len(parsed_ids) != int(row["species_count"]):
            raise ValueError(
                f"species_count does not match species_ids for time={row['time']} zone_id={row['zone_id']}"
            )


def write_csv(path: str | Path, rows: Iterable[dict], fieldnames: list[str] | None = None) -> None:
    """Write dictionaries to CSV with stable field order."""

    path = Path(path)
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
