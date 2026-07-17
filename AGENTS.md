# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `material_feature_fusion/`: `data.py` validates ASE DB rows, `descriptors.py` generates ACSF/SOAP/local Coulomb features, `schnet.py` provides the replaceable-feature SchNet representation, and `fusion.py` contains descriptor fusion modules. CLI workflows are in `scripts/`; tests are in `tests/`. `README.md` and `ASE_DB_FORMAT.md` document the research workflow and database schema. Generated databases and training outputs should stay outside source directories (for example, under `data/processed/` and `training_runs/`).

## Build, Test, and Development Commands

Use the project Conda environment (managed by the Miniconda installation at
`/Users/huzheyu/program/miniconda3`):

```bash
conda activate material-feature-fusion
which python  # /Users/huzheyu/program/miniconda3/envs/material-feature-fusion/bin/python
python -m compileall material_feature_fusion scripts tests
python -m pytest -q
ruff check material_feature_fusion scripts tests
```

The unified project entry point is `python painn.py`. Inspect a database with
`python painn.py inspect path/to/data.db`, create cached descriptors with
`python painn.py prepare input.db output.db`, and train with
`python painn.py train output.db --max-epochs 100`. Use
`--feature-mode realtime` to exercise runtime descriptor generation and
`--max-rows N` for small smoke tests. The older scripts remain available for
compatibility.

## Coding Style & Naming Conventions

Target Python 3.11, use four-space indentation, type hints, and concise docstrings for public functions and modules. Keep lines at 88 characters; Ruff enforces `E`, `F`, and `I` checks. Use `snake_case` for functions, variables, and CLI options; `PascalCase` for classes; and uppercase constants in `keys.py`. Preserve the project’s `row.data` schema and explicit unit metadata when changing data handling.

## Testing Guidelines

Tests use `pytest` and follow `tests/test_*.py` naming. Add focused tests for new data validation, descriptor shapes, feature modes, and model interfaces. For model changes, test both atomic-number and external-feature paths, and include a small batch-level smoke test where practical. Run the full test and Ruff commands before submitting changes.

## Commit & Pull Request Guidelines

Use short Conventional Commit-style subjects, such as `feat: add descriptor cache` or `fix: validate force shapes`. Keep commits focused. Pull requests should explain the motivation, affected feature mode or data schema, validation performed, and any performance or scientific assumptions. Include reproducible commands and configuration details for training changes; do not commit large datasets, generated databases, checkpoints, or secrets.

## Architecture Notes

`FeatureSchNet` keeps SchNetPack’s interaction backbone while replacing its initial atomic embedding with dataset-loaded or runtime-generated per-atom features. External descriptors are fixed NumPy/DScribe inputs and are not differentiated through coordinates; document this limitation when reporting force results.
