"""Linear-elastic direct-stiffness solver for 3D truss elements."""
from dataclasses import dataclass

import numpy as np

from .model import Model


@dataclass
class Result:
    displacements: np.ndarray      # (n_nodes, 3)
    element_forces: np.ndarray     # (n_elements,) axial force, + = tension
    element_stresses: np.ndarray   # (n_elements,)
    reactions: np.ndarray          # (n_nodes, 3), nonzero only at supports


def _element_stiffness(p1, p2, E, A):
    vec = p2 - p1
    L = np.linalg.norm(vec)
    if L == 0:
        raise ValueError("Zero-length element between coincident nodes")
    c = vec / L
    T = np.outer(c, c)
    k_local = (E * A / L) * np.block([[T, -T], [-T, T]])
    return k_local, L, c


def solve(model: Model) -> Result:
    ndof = model.n_dof
    K = np.zeros((ndof, ndof))

    elem_geom = []  # (n1, n2, L, direction_cosines) per element, for post-processing
    for n1, n2 in model.elements:
        p1, p2 = model.nodes[n1], model.nodes[n2]
        k_local, L, c = _element_stiffness(p1, p2, model.material.E, model.material.A)
        elem_geom.append((n1, n2, L, c))

        dofs = [n1 * 3, n1 * 3 + 1, n1 * 3 + 2, n2 * 3, n2 * 3 + 1, n2 * 3 + 2]
        K[np.ix_(dofs, dofs)] += k_local

    F = np.zeros(ndof)
    for node, (fx, fy, fz) in model.loads.items():
        F[node * 3: node * 3 + 3] += (fx, fy, fz)

    fixed = np.zeros(ndof, dtype=bool)
    for node, flags in model.supports.items():
        for i, is_fixed in enumerate(flags):
            fixed[node * 3 + i] = is_fixed
    free = ~fixed

    if not free.any():
        raise ValueError("Model is fully constrained; there is nothing to solve")

    K_ff = K[np.ix_(free, free)]
    F_f = F[free]

    if np.linalg.matrix_rank(K_ff) < K_ff.shape[0]:
        raise ValueError(
            "Stiffness matrix is singular on the free degrees of freedom — "
            "the structure is a mechanism (under-constrained), has "
            "disconnected nodes, or has colinear/coincident elements."
        )

    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K_ff, F_f)

    reactions = (K @ u - F).reshape(-1, 3)

    element_forces = np.zeros(model.n_elements)
    element_stresses = np.zeros(model.n_elements)
    for idx, (n1, n2, L, c) in enumerate(elem_geom):
        u1 = u[n1 * 3: n1 * 3 + 3]
        u2 = u[n2 * 3: n2 * 3 + 3]
        elongation = c @ (u2 - u1)
        stress = model.material.E * (elongation / L)
        element_stresses[idx] = stress
        element_forces[idx] = stress * model.material.A

    return Result(
        displacements=u.reshape(-1, 3),
        element_forces=element_forces,
        element_stresses=element_stresses,
        reactions=reactions,
    )
