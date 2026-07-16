from __future__ import annotations

import numpy as np
import schnetpack as spk
import torch

from material_feature_fusion.data import validate_row
from material_feature_fusion.fusion import DescriptorFusion
from material_feature_fusion.schnet import FeatureSchNet


class FakeRow:
    id = 1
    numbers = np.array([1, 8, 1])
    data = {
        "energy": -1.0,
        "forces": np.zeros((3, 3)),
    }


def test_validate_row_accepts_project_schema() -> None:
    validate_row(FakeRow())


def test_gated_fusion_returns_group_weights() -> None:
    module = DescriptorFusion(
        {"element": 4, "acsf": 6}, output_dim=8, mode="gated_sum"
    )
    output, weights = module(
        {
            "element": torch.randn(2, 3, 4),
            "acsf": torch.randn(2, 3, 6),
        }
    )
    assert output.shape == (2, 3, 8)
    assert set(weights) == {"element", "acsf"}
    total = sum(value for value in weights.values())
    assert torch.allclose(total, torch.ones_like(total), atol=1e-6)


def _schnet_inputs(feature_dim: int | None = None) -> dict[str, torch.Tensor]:
    inputs = {
        "_atomic_numbers": torch.tensor([1, 8, 1], dtype=torch.long),
        "_Rij": torch.tensor([[0.0, 0.0, 0.74], [0.0, 0.0, -0.74]]),
        "_idx_i": torch.tensor([0, 1], dtype=torch.long),
        "_idx_j": torch.tensor([1, 0], dtype=torch.long),
    }
    if feature_dim is not None:
        inputs["features"] = torch.randn(3, feature_dim)
    return inputs


def _representation(feature_mode: str) -> FeatureSchNet:
    return FeatureSchNet(
        n_atom_basis=8,
        n_interactions=1,
        radial_basis=spk.nn.GaussianRBF(n_rbf=4, cutoff=5.0),
        cutoff_fn=spk.nn.CosineCutoff(5.0),
        feature_mode=feature_mode,
        feature_dim=5 if feature_mode == "external" else None,
    )


def test_schnet_supports_atomic_number_and_external_features() -> None:
    atomic_output = _representation("atomic_numbers")(_schnet_inputs())
    external_output = _representation("external")(_schnet_inputs(5))
    assert atomic_output["scalar_representation"].shape == (3, 8)
    assert external_output["scalar_representation"].shape == (3, 8)
