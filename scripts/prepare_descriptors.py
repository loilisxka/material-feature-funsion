#!/usr/bin/env python
"""Create a descriptor-enriched copy of an ASE SQLite database."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from ase.data import chemical_symbols
from ase.db import connect
from tqdm import tqdm

from material_feature_fusion import keys
from material_feature_fusion.data import validate_row
from material_feature_fusion.descriptors import (
    DescriptorBuilder,
    DescriptorConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_db", type=Path)
    parser.add_argument("output_db", type=Path)
    parser.add_argument(
        "--descriptors",
        nargs="+",
        choices=keys.DESCRIPTOR_KEYS,
        default=list(keys.DESCRIPTOR_KEYS),
    )
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--soap-n-max", type=int, default=6)
    parser.add_argument("--soap-l-max", type=int, default=4)
    parser.add_argument("--local-coulomb-neighbors", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_db.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {args.output_db}; pass --overwrite to replace it"
        )
    args.output_db.parent.mkdir(parents=True, exist_ok=True)
    config = DescriptorConfig(
        cutoff=args.cutoff,
        soap_n_max=args.soap_n_max,
        soap_l_max=args.soap_l_max,
        local_coulomb_neighbors=args.local_coulomb_neighbors,
    )

    with connect(str(args.input_db)) as source, connect(str(args.output_db)) as target:
        species = tuple(
            sorted(
                {
                    chemical_symbols[int(z)]
                    for row in source.select()
                    for z in row.numbers
                }
            )
        )
        builder = DescriptorBuilder(config, species)
        rows = source.select()
        for row in tqdm(rows, total=source.count(), desc="Generating descriptors"):
            validate_row(row)
            atoms = row.toatoms()
            data = dict(row.data or {})
            data[keys.ENERGY] = np.asarray(
                data[keys.ENERGY], dtype=np.float64
            ).reshape(1)
            data[keys.FORCES] = np.asarray(data[keys.FORCES], dtype=np.float64)
            descriptors = builder.build(atoms, args.descriptors)
            data.update(
                {name: np.asarray(value) for name, value in descriptors.items()}
            )
            target.write(atoms, key_value_pairs=dict(row.key_value_pairs), data=data)

        metadata = dict(source.metadata or {})
        property_units = dict(metadata.get("_property_unit_dict", {}))
        property_units.update(
            {
                keys.ENERGY: "eV",
                keys.FORCES: "eV/Angstrom",
                keys.ACSF: "dimensionless",
                keys.SOAP: "dimensionless",
                keys.LOCAL_COULOMB: "dimensionless",
            }
        )
        metadata["_property_unit_dict"] = property_units
        metadata["_distance_unit"] = metadata.get("_distance_unit", "Angstrom")
        metadata["descriptor_config"] = {
            "cutoff": config.cutoff,
            "soap_n_max": config.soap_n_max,
            "soap_l_max": config.soap_l_max,
            "local_coulomb_neighbors": config.local_coulomb_neighbors,
            "species": list(species),
        }
        metadata["descriptor_names"] = list(args.descriptors)
        metadata["source_database"] = str(args.input_db)
        target.metadata = metadata
    print(f"Wrote descriptor database: {args.output_db}")


if __name__ == "__main__":
    main()
