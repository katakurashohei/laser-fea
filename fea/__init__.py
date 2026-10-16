"""A minimal 3D truss finite-element solver, driven from the shell."""

from .model import Material, Model
from .io import load_model
from .solver import Result, solve
from .visualize import plot

__all__ = ["Material", "Model", "load_model", "Result", "solve", "plot"]
