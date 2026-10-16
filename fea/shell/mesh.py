"""Procedural mesh generators for the shell solver."""
import numpy as np


def _generate_face(n, origin, du_vec, dv_vec, registry, nodes, margin=0.0):
    """Grid a single planar quad face into an n x n mesh.

    registry maps a rounded (x, y, z) tuple to a global node index, so
    faces that share an edge/corner (matching coordinates) automatically
    share nodes — welding the faces together.

    `margin` insets the meshed area from the face's nominal boundary by
    this distance on all four sides (used to leave room for a joint strip
    between adjacent faces instead of meeting edge-to-edge).
    """
    size_u = np.linalg.norm(du_vec)
    size_v = np.linalg.norm(dv_vec)
    u_hat = du_vec / size_u
    v_hat = dv_vec / size_v
    origin = origin + margin * u_hat + margin * v_hat
    du_vec = du_vec * (1.0 - 2.0 * margin / size_u)
    dv_vec = dv_vec * (1.0 - 2.0 * margin / size_v)

    face_ids = np.zeros((n + 1, n + 1), dtype=int)
    for i in range(n + 1):
        for j in range(n + 1):
            p = origin + (i / n) * du_vec + (j / n) * dv_vec
            key = tuple(np.round(p, 9))
            idx = registry.get(key)
            if idx is None:
                idx = len(nodes)
                registry[key] = idx
                nodes.append(p)
            face_ids[i, j] = idx

    quads = []
    for i in range(n):
        for j in range(n):
            quads.append([face_ids[i, j], face_ids[i + 1, j],
                          face_ids[i + 1, j + 1], face_ids[i, j + 1]])
    return face_ids, quads


def grid_plate(n, size):
    """A single flat, simply-meshed square plate in the z=0 plane, centered
    at the origin, edge length `size`. Used to validate the shell element
    against analytic plate-bending solutions."""
    nodes = []
    registry = {}
    h = size / 2.0
    origin = np.array([-h, -h, 0.0])
    du = np.array([size, 0.0, 0.0])
    dv = np.array([0.0, size, 0.0])
    face_ids, quads = _generate_face(n, origin, du, dv, registry, nodes)

    edge_nodes = set(face_ids[0, :].tolist()) | set(face_ids[-1, :].tolist()) \
        | set(face_ids[:, 0].tolist()) | set(face_ids[:, -1].tolist())

    return np.array(nodes), np.array(quads, dtype=int), face_ids, edge_nodes


def cube_mesh(size, n):
    """A hollow cube of six welded flat plate faces, edge length `size`,
    each face meshed n x n.

    Returns (nodes, quads, faces) where faces is a dict face_name ->
    (n+1, n+1) array of global node indices for that face, letting the
    caller pick out e.g. the top/bottom face to apply loads/supports.
    """
    h = size / 2.0
    nodes = []
    registry = {}

    face_defs = {
        "+z": (np.array([-h, -h, h]), np.array([size, 0, 0]), np.array([0, size, 0])),
        "-z": (np.array([-h, -h, -h]), np.array([size, 0, 0]), np.array([0, size, 0])),
        "+x": (np.array([h, -h, -h]), np.array([0, size, 0]), np.array([0, 0, size])),
        "-x": (np.array([-h, -h, -h]), np.array([0, size, 0]), np.array([0, 0, size])),
        "+y": (np.array([-h, h, -h]), np.array([size, 0, 0]), np.array([0, 0, size])),
        "-y": (np.array([-h, -h, -h]), np.array([size, 0, 0]), np.array([0, 0, size])),
    }

    faces = {}
    all_quads = []
    for name, (origin, du, dv) in face_defs.items():
        face_ids, quads = _generate_face(n, origin, du, dv, registry, nodes)
        faces[name] = face_ids
        all_quads.extend(quads)

    return np.array(nodes), np.array(all_quads, dtype=int), faces


_AXES = {"x": 0, "y": 1, "z": 2}


def box_mesh_with_joints(half_extents, n, joint_width, center=None, axes=None):
    """Generalization of `cube_mesh_with_joints` to an arbitrary rectangular
    box: independent half-extents (hx, hy, hz) instead of one cube size,
    optionally recentered and rotated in world space.

    `center`: (3,) world position of the box center (default the origin).
    `axes`: (3, 3) rotation matrix whose COLUMNS are the box's local x/y/z
    axes expressed in world coordinates (default the identity, i.e.
    axis-aligned). All the mesh generation below happens in the box's own
    local frame — exactly the logic `cube_mesh_with_joints` used to have
    inline — and only the final node coordinates are transformed to world
    space, so the well-tested topology/joint logic is untouched.

    Returns (nodes, plate_quads, joint_quads, faces) — same shape as
    `cube_mesh_with_joints`, with `faces` node coordinates already in world
    space.
    """
    hx, hy, hz = half_extents
    h = {"x": hx, "y": hy, "z": hz}
    nodes = []
    registry = {}

    face_defs = {
        "+z": (np.array([-hx, -hy, hz]), np.array([2 * hx, 0, 0]), np.array([0, 2 * hy, 0])),
        "-z": (np.array([-hx, -hy, -hz]), np.array([2 * hx, 0, 0]), np.array([0, 2 * hy, 0])),
        "+x": (np.array([hx, -hy, -hz]), np.array([0, 2 * hy, 0]), np.array([0, 0, 2 * hz])),
        "-x": (np.array([-hx, -hy, -hz]), np.array([0, 2 * hy, 0]), np.array([0, 0, 2 * hz])),
        "+y": (np.array([-hx, hy, -hz]), np.array([2 * hx, 0, 0]), np.array([0, 0, 2 * hz])),
        "-y": (np.array([-hx, -hy, -hz]), np.array([2 * hx, 0, 0]), np.array([0, 0, 2 * hz])),
    }

    faces = {}
    plate_quads = []
    for name, (origin, du, dv) in face_defs.items():
        face_ids, quads = _generate_face(n, origin, du, dv, registry, nodes, margin=joint_width)
        faces[name] = face_ids
        plate_quads.extend(quads)
    nodes = np.array(nodes)

    def boundary_rows(face_ids):
        m = face_ids.shape[0] - 1
        return {"i0": face_ids[0, :], "in": face_ids[m, :],
                "j0": face_ids[:, 0], "jn": face_ids[:, m]}

    def find_boundary(face_name, axis_name, value):
        rows = boundary_rows(faces[face_name])
        axis_idx = _AXES[axis_name]
        for ids in rows.values():
            coords = nodes[ids][:, axis_idx]
            if np.allclose(coords, value, atol=1e-6):
                return ids
        raise ValueError(f"no boundary of face {face_name} at {axis_name}={value}")

    def sign_char(s):
        return "+" if s > 0 else "-"

    h_inset = {a: h[a] - joint_width for a in "xyz"}  # faces stop here, not at +/-h

    axis_names = ["x", "y", "z"]
    joint_quads = []
    seen_edges = set()
    for free_axis in axis_names:
        a1, a2 = [a for a in axis_names if a != free_axis]
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                edge_key = frozenset({(a1, s1), (a2, s2)})
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

                face1 = sign_char(s1) + a1
                face2 = sign_char(s2) + a2
                ids1 = find_boundary(face1, a2, s2 * h_inset[a2])
                ids2 = find_boundary(face2, a1, s1 * h_inset[a1])

                free_idx = _AXES[free_axis]
                ids1 = ids1[np.argsort(nodes[ids1][:, free_idx])]
                ids2 = ids2[np.argsort(nodes[ids2][:, free_idx])]

                for k in range(len(ids1) - 1):
                    joint_quads.append([int(ids1[k]), int(ids1[k + 1]),
                                         int(ids2[k + 1]), int(ids2[k])])

    if axes is not None:
        nodes = nodes @ axes.T
    if center is not None:
        nodes = nodes + np.asarray(center)

    return nodes, np.array(plate_quads, dtype=int), np.array(joint_quads, dtype=int), faces


def cube_mesh_with_joints(size, n, joint_width):
    """A hollow cube of six flat plate faces, each inset by `joint_width`
    on all sides, bridged across each of the 12 cube edges by a separate
    one-element-wide strip of "joint" quads — a crude stand-in for a
    box-joint corner, meant to carry different (typically softer, more
    compliant) material properties than the solid plate faces.

    No new nodes are introduced for the joint strips: each strip simply
    connects the existing inset-boundary nodes of the two faces it bridges,
    so it's a real, distinct band of elements rather than a material
    override of the existing plate mesh.

    A small triangular gap is left unmeshed at each of the 8 cube corners,
    where three joint strips meet but don't quite close — an accepted
    simplification (and not unlike the exposed end-grain gap at a real
    box-joint corner).

    Returns (nodes, plate_quads, joint_quads, faces) where faces is a dict
    face_name -> (n+1, n+1) array of global node indices for that face's
    (inset) mesh, for picking out e.g. the top/bottom face.
    """
    h = size / 2.0
    return box_mesh_with_joints((h, h, h), n, joint_width)
