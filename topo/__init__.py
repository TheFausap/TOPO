"""Small rigid-DEC Topological Neural Operator prototype."""

from topo.complex import (
    CellComplex,
    grid_complex,
    grid_complex_with_holes,
    triangulated_square_with_holes,
)
from topo.dec import DECOperators
from topo.tno import TopologicalNeuralOperator, VertexGraphOperator

__all__ = [
    "CellComplex",
    "DECOperators",
    "TopologicalNeuralOperator",
    "VertexGraphOperator",
    "grid_complex",
    "grid_complex_with_holes",
    "triangulated_square_with_holes",
]
