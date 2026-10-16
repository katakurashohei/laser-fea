"""Data structures for the shell (plate) model."""
from dataclasses import dataclass
import numpy as np


@dataclass
class ShellMaterial:
    E: float    # Young's modulus (Pa) — isotropic approximation
    nu: float   # Poisson's ratio
    t: float    # plate thickness (m)


@dataclass
class ShellModel:
    nodes: np.ndarray          # (n_nodes, 3) float
    quads: np.ndarray          # (n_elements, 4) int — node indices, wound consistently
    materials: list             # list[ShellMaterial], indexed by element_material
    element_material: np.ndarray  # (n_elements,) int — index into `materials` per element
    supports: dict               # node_index -> (fix_ux, fix_uy, fix_uz, fix_rx, fix_ry, fix_rz)
    loads: dict                   # node_index -> (fx, fy, fz, mx, my, mz)

    @property
    def n_nodes(self):
        return len(self.nodes)

    @property
    def n_elements(self):
        return len(self.quads)

    @property
    def n_dof(self):
        return self.n_nodes * 6
