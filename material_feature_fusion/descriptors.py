"""Descriptor generation for ASE structures.

DScribe descriptors are intentionally generated outside the differentiable
training graph. The resulting per-atom arrays are stored in ``row.data`` and
are treated as fixed auxiliary inputs by the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from . import keys


@dataclass(frozen=True)
class DescriptorConfig:
    """Parameters for the first descriptor set."""

    cutoff: float = 5.0
    acsf_g2: tuple[tuple[float, float], ...] = (
        (0.5, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (4.0, 0.0),
    )
    acsf_g4: tuple[tuple[float, float, float], ...] = (
        (0.005, 1.0, 1.0),
        (0.005, 1.0, 2.0),
        (0.005, -1.0, 1.0),
        (0.005, -1.0, 2.0),
    )
    soap_n_max: int = 6
    soap_l_max: int = 4
    soap_sigma: float = 0.5
    local_coulomb_neighbors: int = 16


def _species(atoms: Atoms, species: tuple[str, ...] | None = None) -> list[str]:
    """Return a stable species vocabulary for a dataset or one structure."""

    return list(species) if species is not None else sorted(
        {str(symbol) for symbol in atoms.get_chemical_symbols()}
    )


def acsf_descriptor(
    atoms: Atoms,
    config: DescriptorConfig,
    species: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Generate a species-resolved ACSF array with shape ``(natoms, dim)``."""

    from dscribe.descriptors import ACSF

    descriptor = ACSF(
        species=_species(atoms, species),
        r_cut=config.cutoff,
        g2_params=list(config.acsf_g2),
        g4_params=list(config.acsf_g4),
    )
    return np.asarray(descriptor.create(atoms), dtype=np.float32)


def soap_descriptor(
    atoms: Atoms,
    config: DescriptorConfig,
    species: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Generate a species-resolved SOAP array with shape ``(natoms, dim)``."""

    from dscribe.descriptors import SOAP

    descriptor = SOAP(
        species=_species(atoms, species),
        r_cut=config.cutoff,
        n_max=config.soap_n_max,
        l_max=config.soap_l_max,
        sigma=config.soap_sigma,
        periodic=bool(np.any(atoms.pbc)),
        sparse=False,
    )
    return np.asarray(descriptor.create(atoms), dtype=np.float32)


def local_coulomb_descriptor(atoms: Atoms, config: DescriptorConfig) -> np.ndarray:
    """Return a fixed-width, permutation-invariant local Coulomb descriptor.

    For each atom, neighbor contributions ``Zi*Zj/r`` are sorted by distance
    and truncated or zero-padded to a fixed width. Periodic image neighbors
    are included when the ASE cell and PBC flags are available.
    """

    n_atoms = len(atoms)
    width = config.local_coulomb_neighbors
    output = np.zeros((n_atoms, width), dtype=np.float32)
    if n_atoms == 0:
        return output

    centers, neighbors, distances, offsets = neighbor_list(
        "ijdS", atoms, config.cutoff
    )
    z = np.asarray(atoms.numbers, dtype=np.float32)
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(n_atoms)]
    for center, neighbor, distance, offset in zip(
        centers, neighbors, distances, offsets
    ):
        if distance <= 1e-8:
            continue
        # Exclude the central atom itself, but retain periodic self-images.
        if center == neighbor and not np.any(offset):
            continue
        value = float(z[center] * z[neighbor] / distance)
        buckets[int(center)].append((float(distance), value))

    for center, values in enumerate(buckets):
        values.sort(key=lambda item: item[0])
        output[center, : min(width, len(values))] = [
            value for _, value in values[:width]
        ]
    return output


class DescriptorBuilder:
    """Reuse DScribe descriptor objects across all structures in a dataset."""

    def __init__(
        self, config: DescriptorConfig, species: tuple[str, ...]
    ) -> None:
        self.config = config
        self.species = species
        self._acsf = None
        self._soap_by_periodicity: dict[bool, object] = {}

    def acsf(self, atoms: Atoms) -> np.ndarray:
        if self._acsf is None:
            from dscribe.descriptors import ACSF

            self._acsf = ACSF(
                species=list(self.species),
                r_cut=self.config.cutoff,
                g2_params=list(self.config.acsf_g2),
                g4_params=list(self.config.acsf_g4),
            )
        return np.asarray(self._acsf.create(atoms), dtype=np.float32)

    def soap(self, atoms: Atoms) -> np.ndarray:
        periodic = bool(np.any(atoms.pbc))
        if periodic not in self._soap_by_periodicity:
            from dscribe.descriptors import SOAP

            self._soap_by_periodicity[periodic] = SOAP(
                species=list(self.species),
                r_cut=self.config.cutoff,
                n_max=self.config.soap_n_max,
                l_max=self.config.soap_l_max,
                sigma=self.config.soap_sigma,
                periodic=periodic,
                sparse=False,
            )
        descriptor = self._soap_by_periodicity[periodic]
        return np.asarray(descriptor.create(atoms), dtype=np.float32)

    def build(
        self, atoms: Atoms, names: Iterable[str]
    ) -> dict[str, np.ndarray]:
        builders = {
            keys.ACSF: self.acsf,
            keys.SOAP: self.soap,
            keys.LOCAL_COULOMB: lambda value: local_coulomb_descriptor(
                value, self.config
            ),
        }
        result: dict[str, np.ndarray] = {}
        for name in names:
            if name not in builders:
                raise ValueError(f"Unsupported descriptor: {name}")
            result[name] = builders[name](atoms)
        return result


def build_descriptors(
    atoms: Atoms,
    config: DescriptorConfig,
    names: Iterable[str] = (keys.ACSF, keys.SOAP, keys.LOCAL_COULOMB),
    species: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Build the requested per-atom descriptors for one structure."""

    if species is not None:
        return DescriptorBuilder(config, species).build(atoms, names)

    builders = {
        keys.ACSF: acsf_descriptor,
        keys.SOAP: soap_descriptor,
        keys.LOCAL_COULOMB: local_coulomb_descriptor,
    }
    result: dict[str, np.ndarray] = {}
    for name in names:
        if name not in builders:
            raise ValueError(f"Unsupported descriptor: {name}")
        if name in {keys.ACSF, keys.SOAP}:
            result[name] = builders[name](atoms, config, species)
        else:
            result[name] = builders[name](atoms, config)
    return result
