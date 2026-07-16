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

    import pytorch_lightning as pl
    import schnetpack as spk
    import schnetpack.transform as trn
    from schnetpack.data.datamodule import AtomsDataModule
    from schnetpack.datasets import MD17
    from schnetpack.task import AtomisticTask, ModelOutput
    from torchmetrics import MeanAbsoluteError

    load_properties = [MD17.energy, MD17.forces]
    data = AtomsDataModule(
        datapath=str(args.datapath),
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

    pairwise_distance = spk.atomistic.PairwiseDistances()
    radial_basis = spk.nn.GaussianRBF(n_rbf=args.n_rbf, cutoff=args.cutoff)
    representation = spk.representation.SchNet(
        n_atom_basis=args.n_atom_basis,
        n_interactions=args.n_interactions,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(args.cutoff),
    )
    pred_energy = spk.atomistic.Atomwise(
        n_in=args.n_atom_basis, output_key=MD17.energy
    )
    pred_forces = spk.atomistic.Forces(
        energy_key=MD17.energy, force_key=MD17.forces
    )
    model = spk.model.NeuralNetworkPotential(
        representation=representation,
        input_modules=[pairwise_distance],
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
