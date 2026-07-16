#!/usr/bin/env python
"""Train a SchNetPack SchNet baseline on an ASE DB."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for repeatable baseline runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datapath", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--feature-mode",
        choices=("atomic_numbers", "dataset", "realtime"),
        default="atomic_numbers",
        help="Source of the initial per-atom representation.",
    )
    parser.add_argument(
        "--descriptor-key",
        default="acsf",
        help="Descriptor name/key used by dataset or realtime mode.",
    )
    parser.add_argument("--num-train", type=float, default=0.8)
    parser.add_argument("--num-val", type=float, default=0.1)
    parser.add_argument("--num-test", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--n-atom-basis", type=int, default=64)
    parser.add_argument("--n-interactions", type=int, default=6)
    parser.add_argument("--n-rbf", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--energy-weight", type=float, default=0.01)
    parser.add_argument("--forces-weight", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = args.output_dir or Path("training_runs") / args.datapath.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    from material_feature_fusion.data import prepare_schnetpack_database

    train_datapath = args.datapath
    if args.feature_mode == "realtime":
        train_datapath = prepare_schnetpack_database(
            args.datapath,
            output_dir / "schnetpack_input.db",
        )

    import pytorch_lightning as pl
    import schnetpack as spk
    import schnetpack.transform as trn
    from schnetpack.data.datamodule import AtomsDataModule
    from schnetpack.datasets import MD17
    from schnetpack.task import AtomisticTask, ModelOutput
    from torchmetrics import MeanAbsoluteError

    load_properties = [MD17.energy, MD17.forces]
    if args.feature_mode == "dataset":
        load_properties.append(args.descriptor_key)
    data = AtomsDataModule(
        datapath=str(train_datapath),
        batch_size=args.batch_size,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        split_file=str(output_dir / "split.npz"),
        load_properties=load_properties,
        transforms=[trn.ASENeighborList(cutoff=args.cutoff), trn.CastTo32()],
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    data.prepare_data()
    data.setup(stage="fit")

    feature_key = args.descriptor_key
    feature_dim = None
    runtime_module = None
    if args.feature_mode == "dataset":
        feature_dim = int(data.dataset[0][feature_key].shape[-1])
    elif args.feature_mode == "realtime":
        from ase.data import chemical_symbols
        from ase.db import connect

        from material_feature_fusion.data import summarize_database
        from material_feature_fusion.descriptors import (
            DescriptorBuilder,
            DescriptorConfig,
            RuntimeDescriptorModule,
        )

        summary = summarize_database(args.datapath)
        species = tuple(chemical_symbols[z] for z in summary.elements)
        descriptor_config = DescriptorConfig(cutoff=args.cutoff)
        with connect(str(args.datapath)) as db:
            sample_atoms = db.get(1).toatoms()
        feature_dim = int(
            DescriptorBuilder(species=species, config=descriptor_config)
            .build(sample_atoms, (args.descriptor_key,))[args.descriptor_key]
            .shape[-1]
        )
        runtime_module = RuntimeDescriptorModule(
            config=descriptor_config,
            species=species,
            descriptor_name=args.descriptor_key,
            output_key=feature_key,
        )

    pairwise_distance = spk.atomistic.PairwiseDistances()
    radial_basis = spk.nn.GaussianRBF(n_rbf=args.n_rbf, cutoff=args.cutoff)
    from material_feature_fusion.schnet import FeatureSchNet

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
    pred_energy = spk.atomistic.Atomwise(
        n_in=args.n_atom_basis, output_key=MD17.energy
    )
    pred_forces = spk.atomistic.Forces(
        energy_key=MD17.energy, force_key=MD17.forces
    )
    input_modules = [pairwise_distance]
    if runtime_module is not None:
        input_modules.append(runtime_module)
    model = spk.model.NeuralNetworkPotential(
        representation=representation,
        input_modules=input_modules,
        output_modules=[pred_energy, pred_forces],
        postprocessors=[trn.CastTo64()],
    )
    task = AtomisticTask(
        model=model,
        outputs=[
            ModelOutput(
                name=MD17.energy,
                loss_fn=torch.nn.MSELoss(),
                loss_weight=args.energy_weight,
                metrics={"MAE": MeanAbsoluteError()},
            ),
            ModelOutput(
                name=MD17.forces,
                loss_fn=torch.nn.MSELoss(),
                loss_weight=args.forces_weight,
                metrics={"MAE": MeanAbsoluteError()},
            ),
        ],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": args.lr},
    )
    accelerator = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=1,
        logger=True,
    )
    trainer.fit(task, datamodule=data)
    trainer.test(task, datamodule=data)


if __name__ == "__main__":
    main()
