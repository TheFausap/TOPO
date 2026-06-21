from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CellComplex:
    """A 2D oriented triangular cell complex."""

    vertices: np.ndarray
    edges: np.ndarray
    faces: np.ndarray
    b1: np.ndarray
    b2: np.ndarray
    boundary_vertices: np.ndarray

    @property
    def num_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edges.shape[0])

    @property
    def num_faces(self) -> int:
        return int(self.faces.shape[0])


def grid_complex(nx: int = 16, ny: int = 16) -> CellComplex:
    """Create a unit-square triangular complex with consistent orientations."""
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be at least 2")

    xs = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    vertices = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)

    def vid(i: int, j: int) -> int:
        return j * nx + i

    faces: list[tuple[int, int, int]] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v00 = vid(i, j)
            v10 = vid(i + 1, j)
            v01 = vid(i, j + 1)
            v11 = vid(i + 1, j + 1)
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))

    edge_to_index: dict[tuple[int, int], int] = {}
    face_edges: list[list[tuple[int, int]]] = []
    for face in faces:
        oriented_edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
        face_edges.append(oriented_edges)
        for a, b in oriented_edges:
            key = (a, b) if a < b else (b, a)
            if key not in edge_to_index:
                edge_to_index[key] = len(edge_to_index)

    edges = np.array(
        [edge for edge, _ in sorted(edge_to_index.items(), key=lambda item: item[1])],
        dtype=np.int64,
    )
    faces_array = np.array(faces, dtype=np.int64)

    b1 = np.zeros((vertices.shape[0], edges.shape[0]), dtype=np.float32)
    for eidx, (a, b) in enumerate(edges):
        b1[a, eidx] = -1.0
        b1[b, eidx] = 1.0

    b2 = np.zeros((edges.shape[0], faces_array.shape[0]), dtype=np.float32)
    for fidx, oriented_edges in enumerate(face_edges):
        for a, b in oriented_edges:
            key = (a, b) if a < b else (b, a)
            eidx = edge_to_index[key]
            b2[eidx, fidx] = 1.0 if (a, b) == key else -1.0

    x = vertices[:, 0]
    y = vertices[:, 1]
    boundary_vertices = (
        np.isclose(x, 0.0) | np.isclose(x, 1.0) | np.isclose(y, 0.0) | np.isclose(y, 1.0)
    )

    return CellComplex(
        vertices=vertices,
        edges=edges,
        faces=faces_array,
        b1=b1,
        b2=b2,
        boundary_vertices=boundary_vertices,
    )


def edge_geometry(complex_: CellComplex) -> np.ndarray:
    """Return edge midpoint and length channels."""
    endpoints = complex_.vertices[complex_.edges]
    mids = endpoints.mean(axis=1)
    lengths = np.linalg.norm(endpoints[:, 1] - endpoints[:, 0], axis=1, keepdims=True)
    return np.concatenate([mids, lengths], axis=1).astype(np.float32)


def face_geometry(complex_: CellComplex) -> np.ndarray:
    """Return face centroid and area channels."""
    points = complex_.vertices[complex_.faces]
    centroids = points.mean(axis=1)
    edge_a = points[:, 1] - points[:, 0]
    edge_b = points[:, 2] - points[:, 0]
    areas = 0.5 * np.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0])[:, None]
    return np.concatenate([centroids, areas], axis=1).astype(np.float32)
