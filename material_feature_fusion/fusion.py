"""Feature projection and analysis modules.

The gate is deliberately exposed as an analysis component. It can report how
much each descriptor branch contributes, while the first baseline can use a
fixed concatenation without changing the training objective.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn


class DescriptorFusion(nn.Module):
    """Project descriptor groups into a common space and optionally gate them."""

    def __init__(
        self,
        input_dims: Mapping[str, int],
        output_dim: int,
        mode: str = "concat",
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"concat", "gated_sum"}:
            raise ValueError("mode must be 'concat' or 'gated_sum'")
        if not input_dims:
            raise ValueError("input_dims cannot be empty")
        self.mode = mode
        self.names = tuple(input_dims)
        hidden = hidden_dim or output_dim
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, output_dim),
                    nn.SiLU(),
                )
                for name, dim in input_dims.items()
            }
        )
        if mode == "concat":
            self.output = nn.Sequential(
                nn.Linear(output_dim * len(self.names), output_dim),
                nn.SiLU(),
            )
        else:
            self.gate = nn.Sequential(
                nn.Linear(output_dim * len(self.names), hidden),
                nn.SiLU(),
                nn.Linear(hidden, len(self.names)),
            )

    def forward(
        self, features: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        projected = []
        for name in self.names:
            if name not in features:
                raise KeyError(f"Missing descriptor feature: {name}")
            projected.append(self.projections[name](features[name]))
        stacked = torch.cat(projected, dim=-1)
        if self.mode == "concat":
            return self.output(stacked), {}
        weights = torch.softmax(self.gate(stacked), dim=-1)
        fused = sum(
            weights[..., i : i + 1] * value for i, value in enumerate(projected)
        )
        return fused, {name: weights[..., i] for i, name in enumerate(self.names)}
