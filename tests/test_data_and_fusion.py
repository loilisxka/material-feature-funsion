from __future__ import annotations

import numpy as np
import torch

from material_feature_fusion.data import validate_row
from material_feature_fusion.fusion import DescriptorFusion


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
