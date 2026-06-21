from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch.utils.data import Dataset

from topo.complex import CellComplex, edge_geometry, face_geometry


@dataclass(frozen=True)
class PoissonSample:
    x0: torch.Tensor
    x1: torch.Tensor
    x2: torch.Tensor
    y0: torch.Tensor


class PoissonDataset(Dataset[PoissonSample]):
    """Fixed-mesh Poisson solution operator with zero boundary values."""

    def __init__(self, complex_: CellComplex, samples: int, seed: int = 0) -> None:
        self.complex = complex_
        self.samples = samples
        self.rng = np.random.default_rng(seed)
        self.edge_features = edge_geometry(complex_)
        self.face_features = face_geometry(complex_)
        self.vertex_xy = complex_.vertices.astype(np.float32)
        self.boundary = complex_.boundary_vertices.astype(np.float32)[:, None]
        self.interior = ~complex_.boundary_vertices
        self.operator = self._dirichlet_operator()
        self._items = [self._make_sample() for _ in range(samples)]

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> PoissonSample:
        return self._items[index]

    def _dirichlet_operator(self) -> sp.csc_matrix:
        b1 = sp.csr_matrix(self.complex.b1)
        lap = b1 @ b1.T
        return lap[self.interior][:, self.interior].tocsc()

    def _forcing(self) -> np.ndarray:
        coords = self.vertex_xy
        f = np.zeros((coords.shape[0], 1), dtype=np.float32)
        bumps = int(self.rng.integers(1, 4))
        for _ in range(bumps):
            center = self.rng.uniform(0.15, 0.85, size=(1, 2)).astype(np.float32)
            sigma = float(self.rng.uniform(0.06, 0.16))
            amplitude = float(self.rng.uniform(-2.5, 2.5))
            dist2 = np.sum((coords - center) ** 2, axis=1, keepdims=True)
            f += amplitude * np.exp(-dist2 / (2.0 * sigma**2)).astype(np.float32)
        f[self.complex.boundary_vertices] = 0.0
        return f

    def _solve(self, f: np.ndarray) -> np.ndarray:
        rhs = f[self.interior, 0]
        u = np.zeros((self.complex.num_vertices, 1), dtype=np.float32)
        u[self.interior, 0] = spla.spsolve(self.operator, rhs).astype(np.float32)
        return u

    def _make_sample(self) -> PoissonSample:
        f = self._forcing()
        u = self._solve(f)
        x0 = np.concatenate([f, self.vertex_xy, self.boundary], axis=1).astype(np.float32)
        return PoissonSample(
            x0=torch.from_numpy(x0),
            x1=torch.from_numpy(self.edge_features),
            x2=torch.from_numpy(self.face_features),
            y0=torch.from_numpy(u),
        )


def poisson_collate(batch: list[PoissonSample]) -> tuple[torch.Tensor, ...]:
    x0 = torch.stack([item.x0 for item in batch], dim=0)
    x1 = torch.stack([item.x1 for item in batch], dim=0)
    x2 = torch.stack([item.x2 for item in batch], dim=0)
    y0 = torch.stack([item.y0 for item in batch], dim=0)
    return x0, x1, x2, y0

