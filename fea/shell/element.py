"""4-node flat shell element: bilinear membrane + Mindlin plate bending.

Each node carries 6 DOF in the GLOBAL frame: (ux, uy, uz, rx, ry, rz).
The element itself is formulated in its own local (x, y, z=0) frame —
required because the 4 nodes are assumed to lie in a plane — and then
rotated into global coordinates before assembly.

Local per-node DOF layout used internally: (u, v, w, phi_x, phi_y, theta_z)
  u, v       — in-plane (membrane) translations
  w          — out-of-plane (bending) translation
  phi_x      — rotation about local x-axis  (bending)
  phi_y      — rotation about local y-axis  (bending)
  theta_z    — drilling rotation about the local normal (no real physical
               stiffness in thin-plate theory; a small artificial stiffness
               is added purely so the global matrix isn't singular at
               interior nodes of a single flat face).

Plate kinematics (through-thickness displacement field), consistent
right-handed convention:
    u(x, y, z) =  z * phi_y(x, y)
    v(x, y, z) = -z * phi_x(x, y)
    w(x, y, z) =  w0(x, y)

which gives curvatures
    kappa_x  =  d(phi_y)/dx
    kappa_y  = -d(phi_x)/dy
    kappa_xy =  d(phi_y)/dy - d(phi_x)/dx
and transverse shear strains
    gamma_xz = dw/dx + phi_y
    gamma_yz = dw/dy - phi_x

Bending uses full 2x2 Gauss integration; transverse shear uses a single
reduced-integration point (selective reduced integration) to avoid shear
locking for thin plates, a standard, simple remedy for 4-node Mindlin
plate elements.
"""
import numpy as np

_GP2 = 1.0 / np.sqrt(3.0)
_GAUSS_2x2 = [(gx, gy, 1.0) for gx in (-_GP2, _GP2) for gy in (-_GP2, _GP2)]
_GAUSS_1x1 = [(0.0, 0.0, 4.0)]


def _shape_funcs(xi, eta):
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                          (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return N, dN_dxi, dN_deta


def _local_frame_and_coords(coords):
    """coords: (4, 3) global node positions, assumed coplanar.

    Returns R (3x3 rotation, v_local = R @ v_global) and local_xy (4, 2).
    The local normal is derived from the node ordering itself, so any
    consistently-wound quad (CW or CCW) yields a positive Jacobian.
    """
    p0, p1, p2, p3 = coords
    e1 = p1 - p0
    e1 = e1 / np.linalg.norm(e1)
    n = np.cross(e1, p3 - p0)
    n = n / np.linalg.norm(n)
    e2 = np.cross(n, e1)
    R = np.vstack([e1, e2, n])

    local_xy = np.zeros((4, 2))
    for i, p in enumerate(coords):
        d = p - p0
        local_xy[i, 0] = e1 @ d
        local_xy[i, 1] = e2 @ d
    return R, local_xy


def _quad_area(local_xy):
    x, y = local_xy[:, 0], local_xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _jacobian_and_grads(xi, eta, x, y):
    N, dN_dxi, dN_deta = _shape_funcs(xi, eta)
    J = np.array([[dN_dxi @ x, dN_dxi @ y],
                  [dN_deta @ x, dN_deta @ y]])
    detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    Jinv = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / detJ
    dN_dx = Jinv[0, 0] * dN_dxi + Jinv[0, 1] * dN_deta
    dN_dy = Jinv[1, 0] * dN_dxi + Jinv[1, 1] * dN_deta
    return N, dN_dx, dN_dy, detJ


def _membrane_B(dN_dx, dN_dy):
    B_m = np.zeros((3, 8))
    for i in range(4):
        B_m[0, 2 * i] = dN_dx[i]
        B_m[1, 2 * i + 1] = dN_dy[i]
        B_m[2, 2 * i] = dN_dy[i]
        B_m[2, 2 * i + 1] = dN_dx[i]
    return B_m


def _bending_B(dN_dx, dN_dy):
    B_b = np.zeros((3, 12))
    for i in range(4):
        B_b[0, 3 * i + 2] = dN_dx[i]
        B_b[1, 3 * i + 1] = -dN_dy[i]
        B_b[2, 3 * i + 1] = -dN_dx[i]
        B_b[2, 3 * i + 2] = dN_dy[i]
    return B_b


def _shear_B(N, dN_dx, dN_dy):
    B_s = np.zeros((2, 12))
    for i in range(4):
        B_s[0, 3 * i] = dN_dx[i]
        B_s[0, 3 * i + 2] = N[i]
        B_s[1, 3 * i] = dN_dy[i]
        B_s[1, 3 * i + 1] = -N[i]
    return B_s


def _material_matrices(E, nu, t, shear_correction):
    G = E / (2.0 * (1.0 + nu))
    D_m = (E * t / (1.0 - nu ** 2)) * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0],
    ])
    D_b = (E * t ** 3 / (12.0 * (1.0 - nu ** 2))) * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0],
    ])
    D_s = shear_correction * G * t * np.eye(2)
    return G, D_m, D_b, D_s


def quad_shell_stiffness(coords, E, nu, t, shear_correction=5.0 / 6.0,
                          drilling_beta=1.0e-3):
    """Global-orientation 24x24 stiffness matrix for one 4-node flat shell
    element (still local to this element's 4 nodes; caller scatters it into
    the global matrix using each node's global DOF indices).

    coords: (4, 3) array of global node coordinates, must be planar and
    wound consistently around the quad (not self-intersecting).
    """
    R, local_xy = _local_frame_and_coords(coords)
    x, y = local_xy[:, 0], local_xy[:, 1]
    G, D_m, D_b, D_s = _material_matrices(E, nu, t, shear_correction)

    K_m = np.zeros((8, 8))
    K_b = np.zeros((12, 12))
    K_s = np.zeros((12, 12))

    for gx, gy, w in _GAUSS_2x2:
        N, dN_dx, dN_dy, detJ = _jacobian_and_grads(gx, gy, x, y)
        B_m = _membrane_B(dN_dx, dN_dy)
        K_m += B_m.T @ D_m @ B_m * detJ * w
        B_b = _bending_B(dN_dx, dN_dy)
        K_b += B_b.T @ D_b @ B_b * detJ * w

    for gx, gy, w in _GAUSS_1x1:
        N, dN_dx, dN_dy, detJ = _jacobian_and_grads(gx, gy, x, y)
        B_s = _shear_B(N, dN_dx, dN_dy)
        K_s += B_s.T @ D_s @ B_s * detJ * w

    K_plate = K_b + K_s
    area = _quad_area(local_xy)
    k_drill = drilling_beta * G * t * area / 4.0

    K_local = np.zeros((24, 24))
    mem_map = [6 * i + d for i in range(4) for d in (0, 1)]
    bend_map = [6 * i + d for i in range(4) for d in (2, 3, 4)]
    drill_idx = [6 * i + 5 for i in range(4)]

    K_local[np.ix_(mem_map, mem_map)] += K_m
    K_local[np.ix_(bend_map, bend_map)] += K_plate
    for idx in drill_idx:
        K_local[idx, idx] += k_drill

    T6 = np.zeros((6, 6))
    T6[0:3, 0:3] = R
    T6[3:6, 3:6] = R
    T = np.kron(np.eye(4), T6)

    K_global = T.T @ K_local @ T
    return K_global, area


def element_bending_curvature(coords, u_global):
    """Post-processing: given an element's 4 global node coords (4, 3) and
    the corresponding global displacement vector (24,) for those same 4
    nodes (6 DOF each, same order as `quad_shell_stiffness`), return the
    bending curvature (kappa_x, kappa_y, kappa_xy) at the element centroid.

    This is a purely kinematic quantity (no material stiffness involved),
    so it reveals where the plate is locally bending/rotating sharply —
    e.g. a soft "hinge" region — even in places where the resulting
    bending moment (and hence stress/energy) is small precisely because
    that region is compliant.
    """
    R, local_xy = _local_frame_and_coords(coords)
    x, y = local_xy[:, 0], local_xy[:, 1]

    T6 = np.zeros((6, 6))
    T6[0:3, 0:3] = R
    T6[3:6, 3:6] = R
    T = np.kron(np.eye(4), T6)
    u_local = T @ u_global

    bend_map = [6 * i + d for i in range(4) for d in (2, 3, 4)]
    u_bend = u_local[bend_map]

    _, dN_dx, dN_dy, _ = _jacobian_and_grads(0.0, 0.0, x, y)
    B_b = _bending_B(dN_dx, dN_dy)
    return B_b @ u_bend
