"""Data structures for the FEA model.

The only element type supported (for now) is a 2-node 3D truss/bar:
axial stiffness only, no bending. That keeps the whole solver a single
small stiffness-matrix formula while still being a real 3D FEA problem.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Material:
    E: float  # Young's modulus (Pa)
    A: float  # cross-sectional area (m^2)


@dataclass
class Model:
    nodes: np.ndarray      # (n_nodes, 3) float — node coordinates
    elements: np.ndarray   # (n_elements, 2) int — node index pairs
    material: Material
    supports: dict          # node_index -> (fix_x, fix_y, fix_z) bools
    loads: dict              # node_index -> (fx, fy, fz) applied force

    @property
    def n_nodes(self):
        return len(self.nodes)

    @property
    def n_elements(self):
        return len(self.elements)

    @property
    def n_dof(self):
        return self.n_nodes * 3
