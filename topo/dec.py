from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import torch

from topo.complex import CellComplex


def _to_sparse_tensor(array: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
    coo = np.nonzero(array)
    if len(coo[0]) == 0:
        indices = torch.zeros((2, 0), dtype=torch.long, device=device)
        values = torch.empty((0,), dtype=torch.float32, device=device)
    else:
        indices = torch.tensor(np.vstack(coo), dtype=torch.long, device=device)
        values = torch.tensor(array[coo], dtype=torch.float32, device=device)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparse invariant checks are implicitly disabled.*",
            category=UserWarning,
        )
        return torch.sparse_coo_tensor(
            indices,
            values,
            array.shape,
            device=device,
            check_invariants=True,
        ).coalesce()


def sparse_mm(matrix: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply a sparse matrix to tensors shaped [..., source_cells, channels]."""
    if x.ndim < 2:
        raise ValueError("x must have shape [..., cells, channels]")
    leading = x.shape[:-2]
    source_cells = x.shape[-2]
    channels = x.shape[-1]
    flat = x.reshape(-1, source_cells, channels).permute(1, 0, 2)
    flat = flat.reshape(source_cells, -1)
    out = torch.sparse.mm(matrix, flat)
    target_cells = out.shape[0]
    out = out.reshape(target_cells, -1, channels).permute(1, 0, 2)
    return out.reshape(*leading, target_cells, channels)


@dataclass
class DECOperators:
    """Rigid DEC routes with identity diagonal Hodge stars."""

    b1: torch.Tensor
    b2: torch.Tensor

    @classmethod
    def from_complex(
        cls, complex_: CellComplex, device: torch.device | str | None = None
    ) -> "DECOperators":
        torch_device = torch.device(device) if device is not None else None
        return cls(
            b1=_to_sparse_tensor(complex_.b1, torch_device),
            b2=_to_sparse_tensor(complex_.b2, torch_device),
        )

    def to(self, device: torch.device | str) -> "DECOperators":
        return DECOperators(self.b1.to(device), self.b2.to(device))

    def d0(self, h0: torch.Tensor) -> torch.Tensor:
        return sparse_mm(self.b1.transpose(0, 1), h0)

    def d1(self, h1: torch.Tensor) -> torch.Tensor:
        return sparse_mm(self.b2.transpose(0, 1), h1)

    def delta1(self, h1: torch.Tensor) -> torch.Tensor:
        return sparse_mm(self.b1, h1)

    def delta2(self, h2: torch.Tensor) -> torch.Tensor:
        return sparse_mm(self.b2, h2)

    def lap0(self, h0: torch.Tensor) -> torch.Tensor:
        return self.delta1(self.d0(h0))

    def lap1_down(self, h1: torch.Tensor) -> torch.Tensor:
        return self.d0(self.delta1(h1))

    def lap1_up(self, h1: torch.Tensor) -> torch.Tensor:
        return self.delta2(self.d1(h1))

    def lap2(self, h2: torch.Tensor) -> torch.Tensor:
        return self.d1(self.delta2(h2))
