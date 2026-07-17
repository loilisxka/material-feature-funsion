from __future__ import annotations

from pathlib import Path

from painn import _feature_names, _hyperparameters, _run_directory_name, parse_args


def test_train_cli_parses_multiple_features_and_smoke_limit() -> None:
    args = parse_args(
        [
            "train",
            "ethanol.db",
            "--feature-mode",
            "realtime",
            "--features",
            "acsf",
            "soap",
            "--fusion",
            "gated_sum",
            "--max-rows",
            "12",
        ]
    )
    assert args.datapath == Path("ethanol.db")
    assert _feature_names(args) == ("acsf", "soap")
    assert args.max_rows == 12


def test_legacy_train_form_still_parses() -> None:
    args = parse_args(["ethanol.db", "--max-epochs", "5"])
    assert args.command == "train"
    assert args.datapath == Path("ethanol.db")
    assert args.max_rows is None


def test_run_directory_and_hyperparameters_record_experiment_dimensions() -> None:
    args = parse_args(
        [
            "train",
            "ethanol.db",
            "--architecture",
            "schnet",
            "--feature-mode",
            "realtime",
            "--features",
            "acsf",
            "soap",
            "--fusion",
            "gated_sum",
        ]
    )
    name = _run_directory_name(args)
    assert "ethanol" in name
    assert "schnet" in name
    assert "acsf+soap" in name
    assert "realtime" in name
    assert "gated_sum" in name

    values = _hyperparameters(args, Path("training_runs") / name)
    assert values["feature_names"] == ["acsf", "soap"]
    assert values["splitting_lock"].endswith("/splitting.lock")


def test_atomic_number_run_directory_identifies_z_embedding() -> None:
    args = parse_args(["train", "ethanol.db"])
    name = _run_directory_name(args)
    assert "z_embedding" in name
    assert "_none" in name
