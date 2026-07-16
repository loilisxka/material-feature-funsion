"""ASE DB readers and validation helpers.

The project databases keep labels in ``row.data``. Keeping this logic in one
module prevents individual training and preprocessing scripts from silently
assuming standard ASE calculator fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from ase.db import connect

from . import keys


@dataclass(frozen=True)
class DatabaseSummary:
    """Small, serializable summary of an ASE SQLite database."""

    path: str
    rows: int
    elements: tuple[int, ...]
    has_energy: bool
    has_forces: bool
    has_stress: bool


def iter_rows(path: str | Path) -> Iterator[Any]:
    """Yield ASE database rows without loading the whole database in memory."""

    with connect(str(path)) as db:
        yield from db.select()


def row_property(row: Any, name: str, required: bool = True) -> Any:
    """Read a property from ``row.data`` and validate its presence."""

    data = row.data or {}
    value = data.get(name)
    if value is None and required:
        raise KeyError(f"Row {row.id} does not contain row.data[{name!r}]")
    return value


def validate_row(row: Any, require_stress: bool = False) -> None:
    """Validate labels and shapes used by the training pipeline."""

    natoms = len(row.numbers)
    energy = np.asarray(row_property(row, keys.ENERGY))
    forces = np.asarray(row_property(row, keys.FORCES))
    if energy.size != 1:
        raise ValueError(f"Row {row.id}: energy must contain one scalar")
    if forces.shape != (natoms, 3):
        raise ValueError(
            f"Row {row.id}: forces has shape {forces.shape}, expected {(natoms, 3)}"
        )
    if require_stress:
        stress = np.asarray(row_property(row, keys.STRESS))
        if stress.shape not in {(3, 3), (1, 3, 3), (6,)}:
            raise ValueError(
                f"Row {row.id}: unsupported stress shape {stress.shape}; "
                "use (3, 3), (1, 3, 3), or ASE Voigt (6,)"
            )


def summarize_database(path: str | Path, limit: int | None = None) -> DatabaseSummary:
    """Inspect labels and elements without materializing structures."""

    count = 0
    elements: set[int] = set()
    has_energy = has_forces = has_stress = True
    with connect(str(path)) as db:
        for row in db.select():
            count += 1
            elements.update(int(z) for z in row.numbers)
            data = row.data or {}
            has_energy &= keys.ENERGY in data
            has_forces &= keys.FORCES in data
            has_stress &= keys.STRESS in data
            if limit is not None and count >= limit:
                break
    return DatabaseSummary(
        path=str(path),
        rows=count,
        elements=tuple(sorted(elements)),
        has_energy=has_energy,
        has_forces=has_forces,
        has_stress=has_stress,
    )
