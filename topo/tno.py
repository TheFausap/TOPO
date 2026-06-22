from __future__ import annotations

import torch
from torch import nn

from topo.dec import DECOperators


class CellMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TNOLayer(nn.Module):
    """One residual rigid-DEC TNO layer over 0-, 1-, and 2-cochains."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.0,
        ablation: str = "full",
    ) -> None:
        super().__init__()
        if ablation not in {"full", "no_face_to_edge", "no_vertex_to_edge", "no_edge_laplacian"}:
            raise ValueError(f"unknown TNO ablation: {ablation}")
        self.ablation = ablation
        self.vertex_update = CellMLP(3 * hidden_dim, hidden_dim, dropout)
        self.edge_update = CellMLP(5 * hidden_dim, hidden_dim, dropout)
        self.face_update = CellMLP(3 * hidden_dim, hidden_dim, dropout)

    def forward(
        self,
        ops: DECOperators,
        h0: torch.Tensor,
        h1: torch.Tensor,
        h2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v_msg = torch.cat([h0, ops.delta1(h1), ops.lap0(h0)], dim=-1)
        vertex_to_edge = ops.d0(h0)
        face_to_edge = ops.delta2(h2)
        edge_lap_down = ops.lap1_down(h1)
        edge_lap_up = ops.lap1_up(h1)
        if self.ablation == "no_vertex_to_edge":
            vertex_to_edge = torch.zeros_like(vertex_to_edge)
        if self.ablation == "no_face_to_edge":
            face_to_edge = torch.zeros_like(face_to_edge)
        if self.ablation == "no_edge_laplacian":
            edge_lap_down = torch.zeros_like(edge_lap_down)
            edge_lap_up = torch.zeros_like(edge_lap_up)
        e_msg = torch.cat(
            [h1, vertex_to_edge, face_to_edge, edge_lap_down, edge_lap_up],
            dim=-1,
        )
        f_msg = torch.cat([h2, ops.d1(h1), ops.lap2(h2)], dim=-1)
        return (
            h0 + self.vertex_update(v_msg),
            h1 + self.edge_update(e_msg),
            h2 + self.face_update(f_msg),
        )


class TopologicalNeuralOperator(nn.Module):
    """Small TNO that predicts a vertex field from multi-rank inputs."""

    def __init__(
        self,
        vertex_in: int,
        edge_in: int,
        face_in: int,
        hidden_dim: int = 64,
        layers: int = 4,
        dropout: float = 0.0,
        ablation: str = "full",
    ) -> None:
        super().__init__()
        self.vertex_encoder = nn.Linear(vertex_in, hidden_dim)
        self.edge_encoder = nn.Linear(edge_in, hidden_dim)
        self.face_encoder = nn.Linear(face_in, hidden_dim)
        self.layers = nn.ModuleList(
            [
                TNOLayer(hidden_dim=hidden_dim, dropout=dropout, ablation=ablation)
                for _ in range(layers)
            ]
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.edge_decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        ops: DECOperators,
        x0: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
        return_edges: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h0 = self.vertex_encoder(x0)
        h1 = self.edge_encoder(x1)
        h2 = self.face_encoder(x2)
        for layer in self.layers:
            h0, h1, h2 = layer(ops, h0, h1, h2)
        y0 = self.decoder(h0)
        if return_edges:
            return y0, self.edge_decoder(h1)
        return y0


class VertexGraphLayer(nn.Module):
    """Rank-0 graph baseline using only self and graph-Laplacian routes."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.update = CellMLP(2 * hidden_dim, hidden_dim, dropout)

    def forward(self, ops: DECOperators, h0: torch.Tensor) -> torch.Tensor:
        msg = torch.cat([h0, ops.lap0(h0)], dim=-1)
        return h0 + self.update(msg)


class VertexGraphOperator(nn.Module):
    """Vertex-only baseline for checking whether multi-rank TNO routes help."""

    def __init__(
        self,
        vertex_in: int,
        hidden_dim: int = 64,
        layers: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(vertex_in, hidden_dim)
        self.layers = nn.ModuleList(
            [VertexGraphLayer(hidden_dim=hidden_dim, dropout=dropout) for _ in range(layers)]
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.edge_decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        ops: DECOperators,
        x0: torch.Tensor,
        x1: torch.Tensor | None = None,
        x2: torch.Tensor | None = None,
        return_edges: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del x1, x2
        h0 = self.encoder(x0)
        for layer in self.layers:
            h0 = layer(ops, h0)
        y0 = self.decoder(h0)
        if return_edges:
            return y0, self.edge_decoder(ops.d0(h0))
        return y0
