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


class AnisotropicDarcyDataset(Dataset[PoissonSample]):
    """Toy anisotropic Darcy operator with face-native conductivity tensors."""

    def __init__(
        self,
        complex_: CellComplex,
        samples: int,
        seed: int = 0,
        k_parallel: float = 4.0,
        k_perp: float = 1.0,
        min_solution_norm: float = 0.5,
        max_sample_attempts: int = 100,
        vertex_projection: str = "mean",
        orientation_mode: str = "iid",
        orientation_blobs: int = 4,
        orientation_sigma: float = 0.25,
    ) -> None:
        if vertex_projection not in {"mean", "none"}:
            raise ValueError("vertex_projection must be 'mean' or 'none'")
        if orientation_mode not in {"iid", "blobs"}:
            raise ValueError("orientation_mode must be 'iid' or 'blobs'")
        if orientation_blobs < 1:
            raise ValueError("orientation_blobs must be at least 1")
        self.complex = complex_
        self.samples = samples
        self.rng = np.random.default_rng(seed)
        self.k_parallel = k_parallel
        self.k_perp = k_perp
        self.min_solution_norm = min_solution_norm
        self.max_sample_attempts = max_sample_attempts
        self.vertex_projection = vertex_projection
        self.orientation_mode = orientation_mode
        self.orientation_blobs = orientation_blobs
        self.orientation_sigma = orientation_sigma
        self.vertex_xy = complex_.vertices.astype(np.float32)
        self.boundary = complex_.boundary_vertices.astype(np.float32)[:, None]
        self.interior = ~complex_.boundary_vertices
        self.base_edge_features = edge_geometry(complex_)
        self.base_face_features = face_geometry(complex_)
        self.face_to_vertices = self._face_to_vertices()
        self.face_to_edges = self._face_to_edges()
        self._items = [self._make_sample() for _ in range(samples)]

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> PoissonSample:
        return self._items[index]

    def _face_to_vertices(self) -> list[list[int]]:
        incident = [[] for _ in range(self.complex.num_vertices)]
        for fidx, face in enumerate(self.complex.faces):
            for vertex in face:
                incident[int(vertex)].append(fidx)
        return incident

    def _face_to_edges(self) -> list[list[int]]:
        incident = [[] for _ in range(self.complex.num_edges)]
        b2_abs = np.abs(self.complex.b2)
        rows, cols = np.nonzero(b2_abs)
        for edge, face in zip(rows, cols, strict=True):
            incident[int(edge)].append(int(face))
        return incident

    def _forcing(self) -> np.ndarray:
        coords = self.vertex_xy
        f = np.zeros((coords.shape[0], 1), dtype=np.float32)
        bumps = int(self.rng.integers(1, 4))
        for _ in range(bumps):
            center = self.rng.uniform(0.15, 0.85, size=(1, 2)).astype(np.float32)
            sigma = float(self.rng.uniform(0.08, 0.18))
            amplitude = float(self.rng.uniform(-2.5, 2.5))
            dist2 = np.sum((coords - center) ** 2, axis=1, keepdims=True)
            f += amplitude * np.exp(-dist2 / (2.0 * sigma**2)).astype(np.float32)
        f[self.complex.boundary_vertices] = 0.0
        return f

    def _orientation_angles(self) -> np.ndarray:
        if self.orientation_mode == "iid":
            return self.rng.uniform(0.0, np.pi, size=(self.complex.num_faces, 1)).astype(
                np.float32
            )

        centroids = self.base_face_features[:, :2]
        field = np.zeros((self.complex.num_faces, 2), dtype=np.float32)
        base_angle = float(self.rng.uniform(0.0, 2.0 * np.pi))
        field += 0.15 * np.array([np.cos(base_angle), np.sin(base_angle)], dtype=np.float32)
        for _ in range(self.orientation_blobs):
            center = self.rng.uniform(0.1, 0.9, size=(1, 2)).astype(np.float32)
            angle = float(self.rng.uniform(0.0, 2.0 * np.pi))
            amplitude = float(self.rng.uniform(0.5, 1.5))
            dist2 = np.sum((centroids - center) ** 2, axis=1, keepdims=True)
            weight = np.exp(-dist2 / (2.0 * self.orientation_sigma**2)).astype(np.float32)
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            field += amplitude * weight * direction
        return (0.5 * np.mod(np.arctan2(field[:, 1:2], field[:, 0:1]), 2.0 * np.pi)).astype(
            np.float32
        )

    def _conductivity(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        phi = self._orientation_angles()
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos2 = np.cos(2.0 * phi).astype(np.float32)
        sin2 = np.sin(2.0 * phi).astype(np.float32)
        diff = self.k_parallel - self.k_perp
        kxx = self.k_perp + diff * cos_phi**2
        kyy = self.k_perp + diff * sin_phi**2
        kxy = diff * cos_phi * sin_phi
        return cos2, sin2, kxx.astype(np.float32), kyy.astype(np.float32), kxy.astype(np.float32)

    def _assemble_full_operator(
        self,
        kxx: np.ndarray,
        kyy: np.ndarray,
        kxy: np.ndarray,
    ) -> sp.csr_matrix:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for fidx, face in enumerate(self.complex.faces):
            ids = [int(v) for v in face]
            pts = self.vertex_xy[ids]
            area = float(self.base_face_features[fidx, 2])
            x0, y0 = pts[0]
            x1, y1 = pts[1]
            x2, y2 = pts[2]
            grads = np.array(
                [
                    [y1 - y2, x2 - x1],
                    [y2 - y0, x0 - x2],
                    [y0 - y1, x1 - x0],
                ],
                dtype=np.float32,
            ) / (2.0 * area)
            tensor = np.array(
                [
                    [float(kxx[fidx, 0]), float(kxy[fidx, 0])],
                    [float(kxy[fidx, 0]), float(kyy[fidx, 0])],
                ],
                dtype=np.float32,
            )
            local = area * grads @ tensor @ grads.T
            for a in range(3):
                for b in range(3):
                    rows.append(ids[a])
                    cols.append(ids[b])
                    data.append(float(local[a, b]))
        matrix = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(self.complex.num_vertices, self.complex.num_vertices),
        ).tocsr()
        return matrix

    def _assemble_operator(
        self,
        kxx: np.ndarray,
        kyy: np.ndarray,
        kxy: np.ndarray,
    ) -> sp.csc_matrix:
        matrix = self._assemble_full_operator(kxx, kyy, kxy)
        return matrix[self.interior][:, self.interior].tocsc()

    def _project_faces_to_vertices(self, values: np.ndarray) -> np.ndarray:
        out = np.zeros((self.complex.num_vertices, values.shape[1]), dtype=np.float32)
        for vertex, faces in enumerate(self.face_to_vertices):
            if faces:
                out[vertex] = values[faces].mean(axis=0)
        return out

    def _project_faces_to_edges(self, values: np.ndarray) -> np.ndarray:
        out = np.zeros((self.complex.num_edges, values.shape[1]), dtype=np.float32)
        for edge, faces in enumerate(self.face_to_edges):
            if faces:
                out[edge] = values[faces].mean(axis=0)
        return out

    def _solve(self, operator: sp.csc_matrix, f: np.ndarray) -> np.ndarray:
        rhs = f[self.interior, 0]
        u = np.zeros((self.complex.num_vertices, 1), dtype=np.float32)
        u[self.interior, 0] = spla.spsolve(operator, rhs).astype(np.float32)
        return u

    def _make_sample(self) -> PoissonSample:
        for _ in range(self.max_sample_attempts):
            f = self._forcing()
            cos2, sin2, kxx, kyy, kxy = self._conductivity()
            operator = self._assemble_operator(kxx, kyy, kxy)
            u = self._solve(operator, f)
            if float(np.linalg.norm(u)) >= self.min_solution_norm:
                break
        else:
            raise RuntimeError(
                "Could not generate a Darcy sample with a nontrivial solution norm. "
                "Lower min_solution_norm or increase max_sample_attempts."
            )
        face_orientation = np.concatenate([cos2, sin2], axis=1)
        edge_orientation = self._project_faces_to_edges(face_orientation)
        x0_parts = [f, self.vertex_xy, self.boundary]
        if self.vertex_projection == "mean":
            x0_parts.append(self._project_faces_to_vertices(face_orientation))
        x0 = np.concatenate(x0_parts, axis=1).astype(np.float32)
        x1 = np.concatenate([self.base_edge_features, edge_orientation], axis=1).astype(np.float32)
        x2 = np.concatenate(
            [self.base_face_features, cos2, sin2, kxx, kyy, kxy],
            axis=1,
        ).astype(np.float32)
        return PoissonSample(
            x0=torch.from_numpy(x0),
            x1=torch.from_numpy(x1),
            x2=torch.from_numpy(x2),
            y0=torch.from_numpy(u),
        )


class DarcyHolesDataset(AnisotropicDarcyDataset):
    """Anisotropic Darcy on a fixed square complex with circular holes."""

    def __init__(
        self,
        complex_: CellComplex,
        samples: int,
        seed: int = 0,
        k_parallel: float = 4.0,
        k_perp: float = 1.0,
        min_solution_norm: float = 0.5,
        max_sample_attempts: int = 100,
        vertex_projection: str = "none",
        orientation_mode: str = "blobs",
        orientation_blobs: int = 4,
        orientation_sigma: float = 0.25,
        hole_value_low: float = 0.5,
        hole_value_high: float = 1.5,
    ) -> None:
        self.hole_value_low = hole_value_low
        self.hole_value_high = hole_value_high
        super().__init__(
            complex_=complex_,
            samples=samples,
            seed=seed,
            k_parallel=k_parallel,
            k_perp=k_perp,
            min_solution_norm=min_solution_norm,
            max_sample_attempts=max_sample_attempts,
            vertex_projection=vertex_projection,
            orientation_mode=orientation_mode,
            orientation_blobs=orientation_blobs,
            orientation_sigma=orientation_sigma,
        )
        if self.complex.boundary_tags is None or np.max(self.complex.boundary_tags) < 1:
            raise ValueError("DarcyHolesDataset requires a complex with tagged hole boundaries")

    def _boundary_values(self) -> np.ndarray:
        tags = self.complex.boundary_tags
        if tags is None:
            raise ValueError("boundary_tags are required")
        values = np.zeros((self.complex.num_vertices, 1), dtype=np.float32)
        for hole_tag in range(1, int(tags.max()) + 1):
            hole_value = float(self.rng.uniform(self.hole_value_low, self.hole_value_high))
            values[tags == hole_tag, 0] = hole_value
        values[tags == -1, 0] = 0.0
        return values

    def _solve_with_boundary_values(
        self,
        operator: sp.csr_matrix,
        f: np.ndarray,
        boundary_values: np.ndarray,
    ) -> np.ndarray:
        boundary = self.complex.boundary_vertices
        interior = ~boundary
        rhs = f[interior, 0] - operator[interior][:, boundary] @ boundary_values[boundary, 0]
        u = boundary_values.copy()
        u[interior, 0] = spla.spsolve(operator[interior][:, interior].tocsc(), rhs).astype(
            np.float32
        )
        return u

    def _make_sample(self) -> PoissonSample:
        for _ in range(self.max_sample_attempts):
            f = self._forcing()
            boundary_values = self._boundary_values()
            cos2, sin2, kxx, kyy, kxy = self._conductivity()
            operator = self._assemble_full_operator(kxx, kyy, kxy)
            u = self._solve_with_boundary_values(operator, f, boundary_values)
            if float(np.linalg.norm(u)) >= self.min_solution_norm:
                break
        else:
            raise RuntimeError(
                "Could not generate a Darcy-holes sample with a nontrivial solution norm. "
                "Lower min_solution_norm or increase max_sample_attempts."
            )

        face_orientation = np.concatenate([cos2, sin2], axis=1)
        edge_orientation = self._project_faces_to_edges(face_orientation)
        boundary_tag_channel = (
            self.complex.boundary_tags.astype(np.float32)[:, None]
            / max(float(np.max(np.abs(self.complex.boundary_tags))), 1.0)
        )
        x0_parts = [f, self.vertex_xy, self.boundary, boundary_values, boundary_tag_channel]
        if self.vertex_projection == "mean":
            x0_parts.append(self._project_faces_to_vertices(face_orientation))
        x0 = np.concatenate(x0_parts, axis=1).astype(np.float32)
        x1 = np.concatenate([self.base_edge_features, edge_orientation], axis=1).astype(np.float32)
        x2 = np.concatenate(
            [self.base_face_features, cos2, sin2, kxx, kyy, kxy],
            axis=1,
        ).astype(np.float32)
        return PoissonSample(
            x0=torch.from_numpy(x0),
            x1=torch.from_numpy(x1),
            x2=torch.from_numpy(x2),
            y0=torch.from_numpy(u),
        )
