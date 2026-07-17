#!/usr/bin/env python
"""Unified command-line entry point for the material feature fusion project.

Examples::

    python painn.py inspect data/raw/example.db
    python painn.py prepare data/raw/example.db data/processed/example.db \
        --features acsf soap
    python painn.py train data/processed/example.db \
        --feature-mode dataset --features acsf soap --fusion gated_sum

The training command keeps PaiNN as the default backbone, while ``--architecture
schnet`` selects the project's replaceable-feature SchNet representation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Material feature fusion command-line interface.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Validate and summarize an ASE SQLite database."
    )
    inspect_parser.add_argument("datapath", type=Path)
    inspect_parser.add_argument("--limit", type=int, default=None)

    prepare_parser = subparsers.add_parser(
        "prepare", help="Create a descriptor-enriched database copy."
    )
    prepare_parser.add_argument("input_db", type=Path)
    prepare_parser.add_argument("output_db", type=Path)
    _add_descriptor_arguments(prepare_parser)
    prepare_parser.add_argument("--overwrite", action="store_true")

    train_parser = subparsers.add_parser(
        "train", help="Train a model with replaceable atom features."
    )
    train_parser.add_argument("datapath", type=Path)
    train_parser.add_argument(
        "--architecture",
        choices=("painn", "schnet"),
        default="painn",
        help="Atomistic interaction backbone.",
    )
    train_parser.add_argument(
        "--feature-mode",
        choices=("atomic_numbers", "dataset", "realtime"),
        default="atomic_numbers",
        help="Source of the initial per-atom representation.",
    )
    _add_descriptor_arguments(train_parser, include_features=True)
    train_parser.add_argument(
        "--fusion",
        choices=("concat", "gated_sum"),
        default="concat",
        help="How multiple descriptor branches are processed.",
    )
    train_parser.add_argument("--num-train", type=float, default=0.8)
    train_parser.add_argument("--num-val", type=float, default=0.1)
    train_parser.add_argument("--num-test", type=float, default=0.1)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--n-atom-basis", type=int, default=64)
    train_parser.add_argument("--n-interactions", type=int, default=6)
    train_parser.add_argument("--n-rbf", type=int, default=20)
    train_parser.add_argument("--max-epochs", type=int, default=100)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--energy-weight", type=float, default=0.01)
    train_parser.add_argument("--forces-weight", type=float, default=0.99)
    train_parser.add_argument("--stress-weight", type=float, default=0.0)
    train_parser.add_argument("--seed", type=int, default=2026)
    train_parser.add_argument(
        "--device", choices=("cpu", "cuda", "mps"), default=None
    )
    train_parser.add_argument("--output-dir", type=Path, default=None)
    train_parser.add_argument(
        "--run-test",
        action="store_true",
        help="Run SchNetPack test after training (force prediction needs gradients).",
    )
    train_parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Copy at most this many structures before training.",
    )

    return parser


def _add_descriptor_arguments(
    parser: argparse.ArgumentParser, include_features: bool = False
) -> None:
    if include_features:
        parser.add_argument(
            "--features",
            nargs="+",
            choices=("acsf", "soap", "local_coulomb"),
            default=["acsf"],
            help="One or more replaceable per-atom feature branches.",
        )
        parser.add_argument(
            "--descriptor-key",
            dest="descriptor_key",
            default=None,
            help="Compatibility alias for a single feature name.",
        )
    else:
        parser.add_argument(
            "--features",
            nargs="+",
            choices=("acsf", "soap", "local_coulomb"),
            default=["acsf", "soap", "local_coulomb"],
        )
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--soap-n-max", type=int, default=6)
    parser.add_argument("--soap-l-max", type=int, default=4)
    parser.add_argument("--soap-sigma", type=float, default=0.5)
    parser.add_argument("--local-coulomb-neighbors", type=int, default=16)
    parser.add_argument(
        "--acsf-g2",
        nargs=2,
        action="append",
        type=float,
        metavar=("ETA", "RS"),
        help="ACSF radial parameter; may be repeated.",
    )
    parser.add_argument(
        "--acsf-g4",
        nargs=3,
        action="append",
        type=float,
        metavar=("ETA", "ZETA", "LAMBDA"),
        help="ACSF angular parameter; may be repeated.",
    )
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments, including the historical ``painn.py DB`` form."""

    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] not in {"inspect", "prepare", "train", "-h", "--help"}:
        values.insert(0, "train")
    return _parser().parse_args(values)


def _descriptor_config(args: argparse.Namespace):
    from material_feature_fusion.descriptors import DescriptorConfig

    defaults = DescriptorConfig()
    return DescriptorConfig(
        cutoff=args.cutoff,
        acsf_g2=tuple(tuple(value) for value in (args.acsf_g2 or defaults.acsf_g2)),
        acsf_g4=tuple(tuple(value) for value in (args.acsf_g4 or defaults.acsf_g4)),
        soap_n_max=args.soap_n_max,
        soap_l_max=args.soap_l_max,
        soap_sigma=args.soap_sigma,
        local_coulomb_neighbors=args.local_coulomb_neighbors,
    )


def _feature_names(args: argparse.Namespace) -> tuple[str, ...]:
    if getattr(args, "descriptor_key", None):
        return (args.descriptor_key,)
    return tuple(args.features)


def _slug(value: str) -> str:
    """Make a readable, filesystem-safe part of a run directory name."""

    return re.sub(r"[^A-Za-z0-9_.+-]+", "-", value).strip("-") or "run"


def _run_directory_name(
    args: argparse.Namespace, started_at: datetime | None = None
) -> str:
    """Build a unique name containing the main experimental dimensions."""

    started_at = started_at or datetime.now().astimezone()
    timestamp = started_at.strftime("%Y%m%d_%H%M%S_%f")
    if args.feature_mode == "atomic_numbers":
        feature_label = "z_embedding"
        fusion_label = "none"
    else:
        feature_label = "+".join(_feature_names(args))
        fusion_label = args.fusion
    parts = (
        timestamp,
        args.datapath.stem,
        args.architecture,
        feature_label,
        args.feature_mode,
        fusion_label,
    )
    return _slug("_".join(parts))


def _training_output_dir(args: argparse.Namespace) -> Path:
    """Return an absolute output directory for this training invocation."""

    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    return (Path("training_runs") / _run_directory_name(args)).resolve()


@contextmanager
def _in_directory(path: Path):
    """Temporarily set cwd for SchNetPack's relative splitting lock."""

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _apply_ase_compat() -> None:
    """Patch ASE versions whose SQLite metadata property is not readable."""

    import ase.db.sqlite as sqlite

    def metadata_getter(self):
        if self._metadata is None:
            with self.managed_connection() as connection:
                self._initialize(connection)
        return self._metadata.copy()

    sqlite.SQLite3Database.metadata = property(
        metadata_getter, sqlite.SQLite3Database.metadata.fset
    )


def _prepare(args: argparse.Namespace) -> None:
    _apply_ase_compat()
    import numpy as np
    from ase.data import chemical_symbols
    from ase.db import connect

    from material_feature_fusion import keys
    from material_feature_fusion.data import validate_row
    from material_feature_fusion.descriptors import DescriptorBuilder

    if args.output_db.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {args.output_db}; pass --overwrite to replace it"
        )
    if args.input_db.resolve() == args.output_db.resolve():
        raise ValueError("output_db must differ from input_db")
    if args.output_db.exists():
        args.output_db.unlink()
    args.output_db.parent.mkdir(parents=True, exist_ok=True)
    names = _feature_names(args)
    config = _descriptor_config(args)
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
        for row in source.select():
            validate_row(row)
            data = dict(row.data or {})
            data[keys.ENERGY] = np.asarray(data[keys.ENERGY]).reshape(1)
            data[keys.FORCES] = np.asarray(data[keys.FORCES])
            data.update(builder.build(row.toatoms(), names))
            target.write(
                row.toatoms(),
                key_value_pairs=dict(row.key_value_pairs),
                data=data,
            )

        metadata = dict(source.metadata or {})
        units = dict(metadata.get("_property_unit_dict", {}))
        units.update({name: "dimensionless" for name in names})
        units.update({keys.ENERGY: "eV", keys.FORCES: "eV/Angstrom"})
        metadata["_property_unit_dict"] = units
        metadata.setdefault("_distance_unit", "Angstrom")
        metadata["descriptor_config"] = {
            "cutoff": config.cutoff,
            "soap_n_max": config.soap_n_max,
            "soap_l_max": config.soap_l_max,
            "soap_sigma": config.soap_sigma,
            "local_coulomb_neighbors": config.local_coulomb_neighbors,
            "acsf_g2": [list(value) for value in config.acsf_g2],
            "acsf_g4": [list(value) for value in config.acsf_g4],
            "species": list(species),
        }
        metadata["descriptor_names"] = list(names)
        metadata["source_database"] = str(args.input_db)
        target.metadata = metadata
    print(f"Wrote descriptor database: {args.output_db}")


def _inspect(args: argparse.Namespace) -> None:
    _apply_ase_compat()
    from material_feature_fusion.data import iter_rows, summarize_database, validate_row

    checked = 0
    for row in iter_rows(args.datapath):
        validate_row(row)
        checked += 1
        if args.limit is not None and checked >= args.limit:
            break

    print(summarize_database(args.datapath, limit=args.limit))


def _limit_database(input_path: Path, output_path: Path, max_rows: int) -> Path:
    """Create a small training copy without loading the whole database."""

    from ase.db import connect

    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    if output_path.exists():
        output_path.unlink()
    with connect(str(input_path)) as source, connect(str(output_path)) as target:
        for index, row in enumerate(source.select()):
            if index >= max_rows:
                break
            target.write(
                row.toatoms(),
                key_value_pairs=dict(row.key_value_pairs),
                data=dict(row.data or {}),
            )
        target.metadata = dict(source.metadata or {})
    return output_path


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_atomrefs(datapath: Path, zmax: int = 100):
    """Fit element reference energies with least squares."""

    import numpy as np
    import torch
    from ase.data import chemical_symbols
    from ase.db import connect

    energies = []
    compositions = []
    with connect(str(datapath)) as database:
        for row in database.select():
            value = (row.data or {}).get("energy")
            if value is None:
                continue
            energies.append(float(np.asarray(value).flat[0]))
            counts = np.zeros(zmax, dtype=np.float64)
            for number in row.numbers:
                if 0 < number < zmax:
                    counts[number] += 1
            compositions.append(counts)
    if not energies:
        raise RuntimeError("Database does not contain row.data['energy']")
    matrix = np.asarray(compositions)
    active = matrix.sum(axis=0) > 0
    result, *_ = np.linalg.lstsq(matrix[:, active], np.asarray(energies), rcond=None)
    atomrefs = np.zeros(zmax, dtype=np.float64)
    atomrefs[active] = result
    symbols = [chemical_symbols[z] for z in np.where(active)[0]]
    print(f"Fitted atomrefs for elements: {symbols}")
    return torch.tensor(atomrefs, dtype=torch.float64)


def _feature_dimensions(data, names: tuple[str, ...]) -> dict[str, int]:
    sample = data.dataset[0]
    return {name: int(sample[name].shape[-1]) for name in names}


def _split_size(value: float) -> int | float:
    """Keep fractions as floats and convert absolute counts to integers."""

    if value > 1 and value.is_integer():
        return int(value)
    return value


def _train(args: argparse.Namespace) -> None:
    _apply_ase_compat()
    _seed_everything(args.seed)
    import pytorch_lightning as pl
    import schnetpack as spk
    import schnetpack.properties as properties
    import schnetpack.transform as trn
    import torch
    from schnetpack.data.datamodule import AtomsDataModule
    from schnetpack.datasets import MD17
    from torchmetrics import MeanAbsoluteError

    from material_feature_fusion.data import prepare_schnetpack_database
    from material_feature_fusion.descriptors import (
        DescriptorBuilder,
        RuntimeDescriptorModule,
    )
    from material_feature_fusion.fusion import DescriptorFusionInput
    from material_feature_fusion.schnet import FeatureSchNet

    args.datapath = args.datapath.expanduser().resolve()
    names = _feature_names(args)
    output_dir = _training_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    hyperparameters = _hyperparameters(args, output_dir)
    (output_dir / "hyperparameters.json").write_text(
        json.dumps(hyperparameters, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    train_path = args.datapath
    if args.max_rows is not None:
        train_path = _limit_database(
            args.datapath, output_dir / "training_subset.db", args.max_rows
        )
    if args.feature_mode == "realtime":
        realtime_source = train_path
        train_path = prepare_schnetpack_database(
            realtime_source, output_dir / "schnetpack_input.db"
        )

    load_properties = [MD17.energy, MD17.forces]
    if args.stress_weight > 0:
        load_properties.append(properties.stress)
    if args.feature_mode == "dataset":
        load_properties.extend(names)

    split_file = output_dir / "split.npz"
    data = AtomsDataModule(
        datapath=str(train_path),
        batch_size=args.batch_size,
        num_train=_split_size(args.num_train),
        num_val=_split_size(args.num_val),
        num_test=_split_size(args.num_test),
        split_file=str(split_file),
        load_properties=load_properties,
        transforms=[trn.ASENeighborList(cutoff=args.cutoff), trn.CastTo32()],
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    data.prepare_data()
    with _in_directory(output_dir):
        data.setup(stage="fit")

    descriptor_config = _descriptor_config(args)
    feature_key = "features"
    feature_dim = None
    input_modules = []
    dims: dict[str, int] = {}
    if args.feature_mode == "dataset":
        dims = _feature_dimensions(data, names)
    elif args.feature_mode == "realtime":
        from ase.data import chemical_symbols
        from ase.db import connect

        with connect(str(train_path)) as database:
            species = tuple(
                sorted(
                    {
                        chemical_symbols[int(z)]
                        for row in database.select()
                        for z in row.numbers
                    }
                )
            )
            sample_atoms = database.get(1).toatoms()
        builder = DescriptorBuilder(descriptor_config, species)
        dims = {
            name: int(builder.build(sample_atoms, (name,))[name].shape[-1])
            for name in names
        }
        input_modules.append(
            RuntimeDescriptorModule(
                config=descriptor_config,
                species=species,
                descriptor_name=names,
                output_key="features" if len(names) == 1 else None,
            )
        )

    if args.feature_mode != "atomic_numbers":
        if len(names) == 1:
            feature_key = names[0] if args.feature_mode == "dataset" else "features"
            feature_dim = dims[names[0]]
        else:
            input_modules.append(
                DescriptorFusionInput(
                    feature_keys=names,
                    input_dims=dims,
                    output_key=feature_key,
                    output_dim=args.n_atom_basis,
                    mode=args.fusion,
                )
            )
            feature_dim = args.n_atom_basis

    pairwise_distance = spk.atomistic.PairwiseDistances()
    radial_basis = spk.nn.GaussianRBF(n_rbf=args.n_rbf, cutoff=args.cutoff)
    if args.architecture == "painn":
        representation = spk.representation.PaiNN(
            n_atom_basis=args.n_atom_basis,
            n_interactions=args.n_interactions,
            radial_basis=radial_basis,
            cutoff_fn=spk.nn.CosineCutoff(args.cutoff),
        )
    else:
        representation = FeatureSchNet(
            n_atom_basis=args.n_atom_basis,
            n_interactions=args.n_interactions,
            radial_basis=radial_basis,
            cutoff_fn=spk.nn.CosineCutoff(args.cutoff),
            feature_mode=(
                "atomic_numbers"
                if args.feature_mode == "atomic_numbers"
                else "external"
            ),
            feature_key=feature_key,
            feature_dim=feature_dim,
        )

    if args.architecture == "painn" and args.feature_mode != "atomic_numbers":
        # PaiNN's interaction blocks remain unchanged; only its initial
        # integer-Z embedding is replaced by a projection of the chosen input.
        from torch import nn

        class ExternalPaiNNInput(nn.Module):
            def __init__(self, key: str) -> None:
                super().__init__()
                self.key = key

            def forward(self, inputs):
                inputs[properties.Z] = inputs.pop(self.key)
                return inputs

        representation.embedding = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, args.n_atom_basis)
        )
        input_modules.append(ExternalPaiNNInput(feature_key))

    energy_key = MD17.energy
    pred_energy = spk.atomistic.Atomwise(
        n_in=args.n_atom_basis, output_key=energy_key
    )
    pred_forces = spk.atomistic.Forces(
        energy_key=energy_key,
        force_key=MD17.forces,
        calc_stress=args.stress_weight > 0,
    )
    input_modules.insert(0, pairwise_distance)
    if args.stress_weight > 0:
        input_modules.insert(0, spk.atomistic.Strain())
    model = spk.model.NeuralNetworkPotential(
        representation=representation,
        input_modules=input_modules,
        output_modules=[pred_energy, pred_forces],
        postprocessors=[trn.CastTo64()],
    )
    outputs = [
        spk.task.ModelOutput(
            name=energy_key,
            loss_fn=torch.nn.MSELoss(),
            loss_weight=args.energy_weight,
            metrics={"MAE": MeanAbsoluteError()},
        ),
        spk.task.ModelOutput(
            name=MD17.forces,
            loss_fn=torch.nn.MSELoss(),
            loss_weight=args.forces_weight,
            metrics={"MAE": MeanAbsoluteError()},
        ),
    ]
    if args.stress_weight > 0:
        outputs.append(
            spk.task.ModelOutput(
                name=properties.stress,
                loss_fn=torch.nn.MSELoss(),
                loss_weight=args.stress_weight,
                metrics={},
            )
        )
    task = spk.task.AtomisticTask(
        model=model,
        outputs=outputs,
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": args.lr},
    )
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=str(output_dir / "best_model"),
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )
    ]
    accelerator = args.device or "auto"
    trainer = pl.Trainer(
        callbacks=callbacks,
        default_root_dir=str(output_dir),
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=1,
        logger=True,
        inference_mode=not args.run_test,
    )
    with _in_directory(output_dir):
        trainer.fit(task, datamodule=data)
    if args.run_test:
        with _in_directory(output_dir):
            trainer.test(task, datamodule=data, ckpt_path="best")


def _jsonable(values: dict) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in values.items()
    }


def _hyperparameters(args: argparse.Namespace, output_dir: Path) -> dict:
    """Serialize CLI parameters plus derived run metadata for comparison."""

    values = _jsonable(vars(args))
    values.update(
        {
            "command": "train",
            "feature_names": list(_feature_names(args)),
            "output_dir": str(output_dir),
            "started_at": datetime.now().astimezone().isoformat(),
            "split_file": str(output_dir / "split.npz"),
            "splitting_lock": str(output_dir / "splitting.lock"),
        }
    )
    return values


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "inspect":
        _inspect(args)
    elif args.command == "prepare":
        _prepare(args)
    else:
        _train(args)


if __name__ == "__main__":
    main()
