"""Linear-elastic direct-stiffness solver for the flat shell model."""
from dataclasses import dataclass

import numpy as np

from .element import element_bending_curvature, quad_shell_stiffness
from .model import ShellModel


@dataclass
class ShellResult:
    displacements: np.ndarray        # (n_nodes, 6)
    reactions: np.ndarray            # (n_nodes, 6), nonzero only at supports
    element_curvature: np.ndarray    # (n_elements, 3): kappa_x, kappa_y, kappa_xy per element


def solve(model: ShellModel) -> ShellResult:
    ndof = model.n_dof
    K = np.zeros((ndof, ndof))

    for elem_idx, quad in enumerate(model.quads):
        coords = model.nodes[quad]
        mat = model.materials[model.element_material[elem_idx]]
        k_elem, _area = quad_shell_stiffness(coords, mat.E, mat.nu, mat.t)
        dofs = np.array([n * 6 + d for n in quad for d in range(6)])
        K[np.ix_(dofs, dofs)] += k_elem

    F = np.zeros(ndof)
    for node, load in model.loads.items():
        F[node * 6: node * 6 + 6] += load

    fixed = np.zeros(ndof, dtype=bool)
    for node, flags in model.supports.items():
        for i, is_fixed in enumerate(flags):
            fixed[node * 6 + i] = is_fixed
    free = ~fixed

    if not free.any():
        raise ValueError("Model is fully constrained; there is nothing to solve")

    K_ff = K[np.ix_(free, free)]
    F_f = F[free]

    u = np.zeros(ndof)
    try:
        u[free] = np.linalg.solve(K_ff, F_f)
    except np.linalg.LinAlgError as e:
        raise ValueError(
            "Stiffness matrix is singular on the free degrees of freedom — "
            "check that the model is adequately supported."
        ) from e

    reactions = (K @ u - F).reshape(-1, 6)

    element_curvature = np.zeros((model.n_elements, 3))
    for elem_idx, quad in enumerate(model.quads):
        coords = model.nodes[quad]
        u_elem = np.concatenate([u[n * 6: n * 6 + 6] for n in quad])
        element_curvature[elem_idx] = element_bending_curvature(coords, u_elem)

    return ShellResult(displacements=u.reshape(-1, 6), reactions=reactions,
                        element_curvature=element_curvature)
