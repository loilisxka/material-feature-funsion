"""Project-owned SchNet representation with replaceable atom features.

The interaction blocks follow SchNetPack 2.1.1. The only intentional extension
is the initial atom representation: it can come from the standard nuclear
embedding or from a fixed-width per-atom tensor in the input dictionary.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

import schnetpack.nn as snn
import schnetpack.properties as properties
import torch
from schnetpack.nn import Dense, scatter_add
from schnetpack.nn.activations import shifted_softplus
from torch import nn


class SchNetInteraction(nn.Module):
    """Continuous-filter convolution block used by SchNet."""

    def __init__(
        self,
        n_atom_basis: int,
        n_rbf: int,
        n_filters: int,
        activation: Callable = shifted_softplus,
    ) -> None:
        super().__init__()
        self.in2f = Dense(n_atom_basis, n_filters, bias=False, activation=None)
        self.f2out = nn.Sequential(
            Dense(n_filters, n_atom_basis, activation=activation),
            Dense(n_atom_basis, n_atom_basis, activation=None),
        )
        self.filter_network = nn.Sequential(
            Dense(n_rbf, n_filters, activation=activation),
            Dense(n_filters, n_filters),
        )

    def forward(
        self,
        x: torch.Tensor,
        f_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        rcut_ij: torch.Tensor,
    ) -> torch.Tensor:
        x = self.in2f(x)
        filters = self.filter_network(f_ij) * rcut_ij[:, None]
        messages = x[idx_j] * filters
        messages = scatter_add(messages, idx_i, dim_size=x.shape[0])
        return self.f2out(messages)


class FeatureSchNet(nn.Module):
    """SchNet representation with standard or external atom features.

    Args:
        feature_mode: ``"atomic_numbers"`` uses the usual embedding;
            ``"external"`` reads ``feature_key`` from ``inputs`` and projects
            it into ``n_atom_basis`` dimensions.
        feature_key: Input-dictionary key for external per-atom features.
        feature_dim: Last dimension of the external feature tensor.
    """

    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        radial_basis: nn.Module,
        cutoff_fn: Callable,
        n_filters: int | None = None,
        shared_interactions: bool = False,
        activation: Union[Callable, nn.Module] = shifted_softplus,
        nuclear_embedding: Optional[nn.Module] = None,
        electronic_embeddings: Optional[List[nn.Module]] = None,
        feature_mode: str = "atomic_numbers",
        feature_key: str = "features",
        feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        if feature_mode not in {"atomic_numbers", "external"}:
            raise ValueError(
                "feature_mode must be 'atomic_numbers' or 'external'"
            )
        if feature_mode == "external" and feature_dim is None:
            raise ValueError("feature_dim is required for external features")

        self.n_atom_basis = n_atom_basis
        self.n_filters = n_filters or n_atom_basis
        self.radial_basis = radial_basis
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.feature_mode = feature_mode
        self.feature_key = feature_key

        if nuclear_embedding is None:
            nuclear_embedding = nn.Embedding(100, n_atom_basis)
        self.embedding = nuclear_embedding
        self.external_embedding = (
            nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, n_atom_basis),
            )
            if feature_mode == "external"
            else None
        )
        self.electronic_embeddings = nn.ModuleList(electronic_embeddings or [])
        self.interactions = snn.replicate_module(
            lambda: SchNetInteraction(
                n_atom_basis=self.n_atom_basis,
                n_rbf=self.radial_basis.n_rbf,
                n_filters=self.n_filters,
                activation=activation,
            ),
            n_interactions,
            shared_interactions,
        )

    def _initial_features(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.feature_mode == "atomic_numbers":
            x = self.embedding(inputs[properties.Z])
        else:
            if self.feature_key not in inputs:
                raise KeyError(
                    f"External feature key {self.feature_key!r} is missing from inputs"
                )
            x = self.external_embedding(inputs[self.feature_key])
        for embedding in self.electronic_embeddings:
            x = x + embedding(x, inputs)
        return x

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        r_ij = inputs[properties.Rij]
        idx_i = inputs[properties.idx_i].long()
        idx_j = inputs[properties.idx_j].long()
        distances = torch.norm(r_ij, dim=1)
        radial_features = self.radial_basis(distances)
        cutoff_values = self.cutoff_fn(distances)

        x = self._initial_features(inputs)
        for interaction in self.interactions:
            x = x + interaction(
                x, radial_features, idx_i, idx_j, cutoff_values
            )
        inputs["scalar_representation"] = x
        return inputs
