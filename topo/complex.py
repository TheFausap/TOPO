from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial import Delaunay


@dataclass(frozen=True)
class CellComplex:
    """A 2D oriented triangular cell complex."""

    vertices: np.ndarray
    edges: np.ndarray
    faces: np.ndarray
    b1: np.ndarray
    b2: np.ndarray
    boundary_vertices: np.ndarray
    boundary_tags: np.ndarray | None = None

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

    faces_array = np.array(faces, dtype=np.int64)
    x = vertices[:, 0]
    y = vertices[:, 1]
    boundary_tags = np.zeros(vertices.shape[0], dtype=np.int64)
    boundary_tags[
        np.isclose(x, 0.0) | np.isclose(x, 1.0) | np.isclose(y, 0.0) | np.isclose(y, 1.0)
    ] = -1
    return build_complex(vertices=vertices, faces=faces_array, boundary_tags=boundary_tags)


def build_complex(
    vertices: np.ndarray,
    faces: np.ndarray,
    boundary_tags: np.ndarray | None = None,
) -> CellComplex:
    """Build oriented edges and incidence matrices from vertices and triangular faces."""
    edge_to_index: dict[tuple[int, int], int] = {}
    face_edges: list[list[tuple[int, int]]] = []
    for face in faces:
        face_tuple = (int(face[0]), int(face[1]), int(face[2]))
        oriented_edges = [
            (face_tuple[0], face_tuple[1]),
            (face_tuple[1], face_tuple[2]),
            (face_tuple[2], face_tuple[0]),
        ]
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

    if boundary_tags is None:
        x = vertices[:, 0]
        y = vertices[:, 1]
        boundary_tags = np.zeros(vertices.shape[0], dtype=np.int64)
        boundary_tags[
            np.isclose(x, 0.0) | np.isclose(x, 1.0) | np.isclose(y, 0.0) | np.isclose(y, 1.0)
        ] = -1
    boundary_vertices = boundary_tags != 0

    return CellComplex(
        vertices=vertices.astype(np.float32),
        edges=edges,
        faces=faces_array,
        b1=b1,
        b2=b2,
        boundary_vertices=boundary_vertices,
        boundary_tags=boundary_tags,
    )


def grid_complex_with_holes(
    nx: int = 24,
    ny: int = 24,
    holes: Sequence[tuple[float, float, float]] | None = None,
) -> CellComplex:
    """Create a fixed triangular square mesh with circular holes removed."""
    if holes is None:
        holes = ((0.35, 0.5, 0.13), (0.65, 0.5, 0.13))
    if nx < 4 or ny < 4:
        raise ValueError("nx and ny must be at least 4 for a holed grid")

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

    faces_array = np.array(faces, dtype=np.int64)
    centroids = vertices[faces_array].mean(axis=1)
    keep = np.ones(faces_array.shape[0], dtype=bool)
    for cx, cy, radius in holes:
        dist2 = (centroids[:, 0] - cx) ** 2 + (centroids[:, 1] - cy) ** 2
        keep &= dist2 > radius**2
    kept_faces = faces_array[keep]

    used = np.unique(kept_faces.reshape(-1))
    old_to_new = {int(old): new for new, old in enumerate(used)}
    new_vertices = vertices[used]
    new_faces = np.array(
        [[old_to_new[int(vertex)] for vertex in face] for face in kept_faces],
        dtype=np.int64,
    )

    provisional = build_complex(new_vertices, new_faces)
    boundary_edge_mask = np.abs(provisional.b2).sum(axis=1) == 1
    boundary_tags = np.zeros(provisional.num_vertices, dtype=np.int64)
    x = provisional.vertices[:, 0]
    y = provisional.vertices[:, 1]
    outer = np.isclose(x, 0.0) | np.isclose(x, 1.0) | np.isclose(y, 0.0) | np.isclose(y, 1.0)
    boundary_tags[outer] = -1

    for edge in provisional.edges[boundary_edge_mask]:
        midpoint = provisional.vertices[edge].mean(axis=0)
        if np.any(outer[edge]):
            boundary_tags[edge] = -1
            continue
        distances = [
            (midpoint[0] - cx) ** 2 + (midpoint[1] - cy) ** 2 for cx, cy, _radius in holes
        ]
        hole_tag = int(np.argmin(distances)) + 1
        for vertex in edge:
            if boundary_tags[int(vertex)] == 0:
                boundary_tags[int(vertex)] = hole_tag

    return build_complex(new_vertices, new_faces, boundary_tags=boundary_tags)


def triangulated_square_with_holes(
    nx: int = 24,
    ny: int = 24,
    holes: Sequence[tuple[float, float, float]] | None = None,
    seed: int = 0,
    jitter: float = 0.35,
    hole_points: int | None = None,
) -> CellComplex:
    """Create a fixed irregular triangulation of a square with circular holes."""
    if holes is None:
        holes = ((0.35, 0.5, 0.13), (0.65, 0.5, 0.13))
    if nx < 4 or ny < 4:
        raise ValueError("nx and ny must be at least 4 for a holed triangulation")
    if not 0.0 <= jitter < 0.5:
        raise ValueError("jitter must be in [0, 0.5)")

    rng = np.random.default_rng(seed)
    points: list[tuple[float, float]] = []
    tags: list[int] = []

    def add_point(x: float, y: float, tag: int) -> None:
        points.append((float(x), float(y)))
        tags.append(tag)

    for x in np.linspace(0.0, 1.0, nx):
        add_point(float(x), 0.0, -1)
        add_point(float(x), 1.0, -1)
    for y in np.linspace(0.0, 1.0, ny)[1:-1]:
        add_point(0.0, float(y), -1)
        add_point(1.0, float(y), -1)

    samples_per_hole = hole_points or max(24, 2 * max(nx, ny))
    for hole_tag, (cx, cy, radius) in enumerate(holes, start=1):
        for theta in np.linspace(0.0, 2.0 * np.pi, samples_per_hole, endpoint=False):
            add_point(cx + radius * np.cos(theta), cy + radius * np.sin(theta), hole_tag)

    dx = 1.0 / float(nx - 1)
    dy = 1.0 / float(ny - 1)
    hole_margin = 0.5 * min(dx, dy)
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            x = i * dx + float(rng.uniform(-jitter, jitter)) * dx
            y = j * dy + float(rng.uniform(-jitter, jitter)) * dy
            inside_hole = any(
                (x - cx) ** 2 + (y - cy) ** 2 <= (radius + hole_margin) ** 2
                for cx, cy, radius in holes
            )
            if not inside_hole:
                add_point(x, y, 0)

    vertices = np.array(points, dtype=np.float32)
    point_tags = np.array(tags, dtype=np.int64)
    triangles = Delaunay(vertices).simplices.astype(np.int64)
    centroids = vertices[triangles].mean(axis=1)
    keep = np.ones(triangles.shape[0], dtype=bool)
    for cx, cy, radius in holes:
        dist2 = (centroids[:, 0] - cx) ** 2 + (centroids[:, 1] - cy) ** 2
        keep &= dist2 > radius**2
    kept_faces = triangles[keep]

    used = np.unique(kept_faces.reshape(-1))
    old_to_new = {int(old): new for new, old in enumerate(used)}
    new_vertices = vertices[used]
    new_faces = np.array(
        [[old_to_new[int(vertex)] for vertex in face] for face in kept_faces],
        dtype=np.int64,
    )
    new_tags = point_tags[used]

    points_for_orientation = new_vertices[new_faces]
    signed_area2 = (
        (points_for_orientation[:, 1, 0] - points_for_orientation[:, 0, 0])
        * (points_for_orientation[:, 2, 1] - points_for_orientation[:, 0, 1])
        - (points_for_orientation[:, 1, 1] - points_for_orientation[:, 0, 1])
        * (points_for_orientation[:, 2, 0] - points_for_orientation[:, 0, 0])
    )
    clockwise = signed_area2 < 0.0
    new_faces[clockwise] = new_faces[clockwise][:, [0, 2, 1]]

    return build_complex(new_vertices, new_faces, boundary_tags=new_tags)


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
