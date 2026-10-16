"""A minimal flat-shell (plate) FEA module: 4-node quad elements with
membrane + Mindlin plate bending behavior, 6 DOF per node."""

from .model import ShellMaterial, ShellModel
from .solver import ShellResult, solve
from .mesh import cube_mesh, cube_mesh_with_joints
from .visualize import plot

__all__ = ["ShellMaterial", "ShellModel", "ShellResult", "solve", "cube_mesh",
           "cube_mesh_with_joints", "plot"]
