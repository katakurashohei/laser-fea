"""Reconstruct an arbitrary axis-aligned ("rectilinear") solid — any union
of boxes sharing a common orthogonal frame, e.g. an L-shape or a concave
step — from a triangulated OBJ import, and remesh it exactly (not just its
bounding box), with box-joint bands only at the solid's real edges (where
the surface actually changes direction) — not at every underlying grid
cell seam.

Algorithm:
  1. Detect the shape's 3 mutually-orthogonal face-normal directions (its
     local axes) by clustering triangle normals — robust for a true
     rectilinear shape, where normals fall into exactly 6 directions
     (+/- 3 axes), unlike a vertex-based PCA fit (which has no reason to
     align with the shape's actual edges for an asymmetric shape).
  2. Transform all triangles into that local frame and cluster vertex
     coordinates along each axis into a set of grid lines — the union's
     box boundaries.
  3. For every candidate grid cell, ray-cast from its center against the
     original triangles to decide inside/outside (odd/even crossings).
  4. For each of the 6 directions and each "layer" (the grid index along
     that direction's fixed axis), find the exposed cells and group them
     into connected components — each component is one *logical panel*:
     a maximal flat, contiguous region, e.g. a wall spanning several grid
     cells is one panel, not one per cell.
  5. Each panel is decomposed into a small number of axis-aligned
     rectangles by greedy merging (not necessarily 1 — an L-shaped panel
     needs at least 2 — but far fewer than 1-per-grid-cell). Each
     rectangle is meshed with `n` elements per side (default 1: "just the
     face", no subdivision).
  6. Per rectangle, per side: if a rectangle's neighbor across that side
     belongs to the *same* panel, that side is welded flush (no inset, no
     joint — it's the same physical piece). Otherwise it's inset by
     `joint_width` and registered as a joint candidate. Any two inset
     edges sharing exact endpoints get bridged by a joint-band strip —
     same-cell corners, coplanar neighbor-panel continuations, and
     diagonal-cell reentrant corners are all found the same way, since
     it's pure edge-coincidence, not adjacency bookkeeping.

Greedy merging alone can produce a rectangle whose side only *partially*
matches a same-panel neighbor (e.g. a ring-shaped panel's edge bordering
the hole a smaller box sits in) — a single quad can't have two different
inset values on one side, so that side is split at the transition
(`_refine_with_global`'s local pass). Splitting one panel's rectangle for
its own reasons can fragment a side that a *different* panel was relying
on matching exactly, which can in turn force that panel to split too —
so this isn't a single pass: rectangles are re-refined against a
global "which lines need a cut, and where" table computed from every
panel's current state, repeated to a fixed point (a handful of
iterations in practice; guaranteed to terminate, since cuts only ever
add grid-aligned breakpoints, of which there are finitely many). Verified
on both example shapes by checking every node is reachable from the
support nodes through the assembled mesh, and that the resulting model
solves with the correct reaction-sum equilibrium.

Returns plate quads and joint quads separately, same shape as
`box_mesh_with_joints`, so the caller assigns materials the same way.
"""
import numpy as np


_DIRECTIONS = {
    "-x": {"fixed_axis": 0, "u_axis": 1, "v_axis": 2, "sign": -1},
    "+x": {"fixed_axis": 0, "u_axis": 1, "v_axis": 2, "sign": +1},
    "-y": {"fixed_axis": 1, "u_axis": 0, "v_axis": 2, "sign": -1},
    "+y": {"fixed_axis": 1, "u_axis": 0, "v_axis": 2, "sign": +1},
    "-z": {"fixed_axis": 2, "u_axis": 0, "v_axis": 1, "sign": -1},
    "+z": {"fixed_axis": 2, "u_axis": 0, "v_axis": 1, "sign": +1},
}


def _connected_components(cells):
    """cells: set of (u, v) int tuples. Returns a list of components, each
    a set of (u, v), grouped by 4-connectivity (shared edge, not corner)."""
    remaining = set(cells)
    components = []
    while remaining:
        start = next(iter(remaining))
        stack = [start]
        comp = set()
        while stack:
            c = stack.pop()
            if c in comp:
                continue
            comp.add(c)
            remaining.discard(c)
            u, v = c
            for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (u + du, v + dv)
                if nb in remaining:
                    stack.append(nb)
        components.append(comp)
    return components


def _greedy_rectangles(cells):
    """cells: set of (u, v). Returns a list of (u0, u1, v0, v1) rectangles
    (u1/v1 exclusive) that exactly tile `cells`, via simple greedy
    row-then-column merging (not guaranteed minimal, but simple, and
    optimal for the common cases — a single rectangle or a plain
    L-shape/frame)."""
    remaining = set(cells)
    rects = []
    while remaining:
        u0, v0 = min(remaining)
        u1 = u0 + 1
        while (u1, v0) in remaining:
            u1 += 1
        v1 = v0 + 1
        while all((uu, v1) in remaining for uu in range(u0, u1)):
            v1 += 1
        for uu in range(u0, u1):
            for vv in range(v0, v1):
                remaining.discard((uu, vv))
        rects.append((u0, u1, v0, v1))
    return rects


def _generate_face_sided(n, origin, du_vec, dv_vec, registry, nodes, margins):
    """Like mesh._generate_face, but with an independent inset per side
    instead of one uniform margin — needed because a panel-internal
    rectangle can be flush (0 margin) on the sides bordering the same
    panel and inset only on the sides that are a real edge.

    margins: (margin_u0, margin_u1, margin_v0, margin_v1)."""
    m_u0, m_u1, m_v0, m_v1 = margins
    size_u = np.linalg.norm(du_vec)
    size_v = np.linalg.norm(dv_vec)
    u_hat = du_vec / size_u
    v_hat = dv_vec / size_v
    origin = origin + m_u0 * u_hat + m_v0 * v_hat
    du_vec = u_hat * (size_u - m_u0 - m_u1)
    dv_vec = v_hat * (size_v - m_v0 - m_v1)

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


def _fold(n):
    """Canonicalize a direction's sign (first-nonzero-component positive)
    so +axis and -axis normals fall in the same cluster."""
    for c in n:
        if abs(c) > 1e-9:
            return n if c > 0 else -n
    return n


def detect_orthogonal_axes(normals, angle_tol_deg=10.0):
    """normals: (N, 3) unit face normals. Returns a (3, 3) rotation matrix
    (columns = the shape's local x/y/z axes in world coordinates), built
    from the 3 most-supported mutually-orthogonal directions.

    Raises ValueError if fewer than 3 distinct directions are found (i.e.
    the input doesn't look rectilinear)."""
    folded = np.array([_fold(n) for n in normals])
    cos_tol = np.cos(np.radians(angle_tol_deg))

    remaining = folded
    clusters = []
    while len(remaining) > 0 and len(clusters) < 3:
        best_count, best_mask = -1, None
        for ref in remaining:
            mask = (remaining @ ref) > cos_tol
            count = int(mask.sum())
            if count > best_count:
                best_count, best_mask = count, mask
        mean_dir = remaining[best_mask].mean(axis=0)
        mean_dir /= np.linalg.norm(mean_dir)
        clusters.append(mean_dir)
        remaining = remaining[~best_mask]

    if len(clusters) < 3:
        raise ValueError(
            f"only found {len(clusters)} distinct face-normal direction(s); "
            "this doesn't look like a rectilinear (axis-aligned box union) shape"
        )

    a0 = clusters[0]
    a1 = clusters[1] - np.dot(clusters[1], a0) * a0
    if np.linalg.norm(a1) < 1e-6:
        raise ValueError("detected face normals are not sufficiently orthogonal")
    a1 /= np.linalg.norm(a1)
    a2 = np.cross(a0, a1)
    return np.stack([a0, a1, a2], axis=1)


def _cluster_1d(values, tol):
    values = np.sort(np.asarray(values))
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return np.array([np.mean(c) for c in clusters])


def _ray_triangle_intersect(orig, direction, v0, v1, v2, eps=1e-9):
    e1, e2 = v1 - v0, v2 - v0
    h = np.cross(direction, e2)
    a = np.dot(e1, h)
    if -eps < a < eps:
        return None
    f = 1.0 / a
    s = orig - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return None
    q = np.cross(s, e1)
    v = f * np.dot(direction, q)
    if v < 0.0 or u + v > 1.0:
        return None
    t = f * np.dot(e2, q)
    return t if t > eps else None


def _point_inside(point, triangles, direction):
    count = 0
    for v0, v1, v2 in triangles:
        if _ray_triangle_intersect(point, direction, v0, v1, v2) is not None:
            count += 1
    return count % 2 == 1


_RAY_DIR = np.array([0.5732, 0.6421, 0.5091])
_RAY_DIR /= np.linalg.norm(_RAY_DIR)


def _boundary_rows(face_ids):
    m = face_ids.shape[0] - 1
    return {"i0": face_ids[0, :], "in": face_ids[m, :],
            "j0": face_ids[:, 0], "jn": face_ids[:, m]}


def remesh_rectilinear(triangles, n=1, joint_width=0.02):
    """triangles: (T, 3, 3) array-like of world-space triangle corners.

    Returns (nodes, plate_quads, joint_quads, bottom_ids):
      nodes: (N, 3) world coordinates.
      plate_quads, joint_quads: (M, 4) int node indices — the exterior
        panel faces, and the joint-band strips bridging every real edge.
      bottom_ids: node indices in the lowest horizontal layer (world Z),
        for clamping as a support.

    Raises ValueError if the input doesn't look rectilinear, or if no
    solid cells are found — callers should catch this and fall back to a
    simpler fit (e.g. a single oriented bounding box).
    """
    triangles = np.asarray(triangles, dtype=float)
    normals, valid = [], []
    for tri in triangles:
        e1, e2 = tri[1] - tri[0], tri[2] - tri[0]
        nrm = np.cross(e1, e2)
        length = np.linalg.norm(nrm)
        if length < 1e-12:
            continue
        normals.append(nrm / length)
        valid.append(tri)
    if len(valid) < 4:
        raise ValueError("not enough non-degenerate triangles to analyze")
    triangles = np.array(valid)

    axes = detect_orthogonal_axes(normals)

    all_pts = triangles.reshape(-1, 3)
    ref = all_pts.mean(axis=0)
    local_pts = (all_pts - ref) @ axes
    local_tris = local_pts.reshape(-1, 3, 3)

    extent = local_pts.max(axis=0) - local_pts.min(axis=0)
    coord_tol = max(float(np.max(extent)) * 1e-3, 1e-6)

    grid = [_cluster_1d(local_pts[:, ax], coord_tol) for ax in range(3)]
    gx, gy, gz = grid
    if min(len(gx), len(gy), len(gz)) < 2:
        raise ValueError("degenerate grid (shape is flat along one axis)")

    occupied = set()
    for i in range(len(gx) - 1):
        for j in range(len(gy) - 1):
            for k in range(len(gz) - 1):
                lo = np.array([gx[i], gy[j], gz[k]])
                hi = np.array([gx[i + 1], gy[j + 1], gz[k + 1]])
                center = (lo + hi) / 2.0
                if _point_inside(center, local_tris, _RAY_DIR):
                    occupied.add((i, j, k))

    if not occupied:
        raise ValueError("no solid cells found on the detected grid")

    min_cell_dim = min(np.diff(gx).min(), np.diff(gy).min(), np.diff(gz).min())
    joint_width = min(joint_width, 0.2 * min_cell_dim)
    grids = [gx, gy, gz]

    # Exposed cells per (direction, layer), in that direction's own (u, v)
    # index space.
    exposed = {}
    for (i, j, k) in occupied:
        idx3 = (i, j, k)
        for name, info in _DIRECTIONS.items():
            f = info["fixed_axis"]
            neighbor = list(idx3)
            neighbor[f] += info["sign"]
            if tuple(neighbor) in occupied:
                continue  # interior face shared with an occupied neighbor
            layer = idx3[f]
            u = idx3[info["u_axis"]]
            v = idx3[info["v_axis"]]
            exposed.setdefault((name, layer), set()).add((u, v))

    # Group into logical panels (connected components) and get each's
    # (unrefined) greedy rectangles.
    panels = []  # each: {"direction", "layer", "cells": set, "greedy_rects": [...]}
    for (direction, layer), cells in exposed.items():
        for comp in _connected_components(cells):
            panels.append({"direction": direction, "layer": layer, "cells": comp,
                            "greedy_rects": _greedy_rectangles(comp)})

    def side_margin(panel, u0, u1, v0, v1, side):
        cells = panel["cells"]
        if side == "u0":
            outside = [(u0 - 1, vv) for vv in range(v0, v1)]
        elif side == "u1":
            outside = [(u1, vv) for vv in range(v0, v1)]
        elif side == "v0":
            outside = [(uu, v0 - 1) for uu in range(u0, u1)]
        else:
            outside = [(uu, v1) for uu in range(u0, u1)]
        # Weld (no inset) if ANY of the immediately-adjacent cells belong to
        # this same panel — avoids stranding a rectangle from the rest of
        # its own panel when a neighboring rectangle's edge doesn't line up
        # exactly (a T-junction, only possible for non-rectangular panels;
        # see the module docstring).
        if any(c in cells for c in outside):
            return 0.0
        return joint_width

    def rect_line_and_range(panel, u0, u1, v0, v1, side):
        """The 3D line a rectangle side lies on (its 2 *fixed* coordinates,
        ignoring the varying one) plus its [lo, hi] range along the free
        axis — used to line up cuts between different panels that share
        an edge but were decomposed independently.

        The key is built by sorting the two (axis, value) pairs by axis
        index, so two panels that disagree on which axis is "fixed" vs.
        "u/v" for their own purposes (every panel's own fixed_axis differs)
        still produce the exact same key for the exact same physical line.
        """
        info = _DIRECTIONS[panel["direction"]]
        f, ua, va, s = info["fixed_axis"], info["u_axis"], info["v_axis"], info["sign"]
        fixed_coord = grids[f][panel["layer"] + (1 if s > 0 else 0)]
        if side in ("u0", "u1"):
            u_val = grids[ua][u0 if side == "u0" else u1]
            pair = sorted([(f, round(float(fixed_coord), 9)), (ua, round(float(u_val), 9))])
            return tuple(pair), float(grids[va][v0]), float(grids[va][v1])
        v_val = grids[va][v0 if side == "v0" else v1]
        pair = sorted([(f, round(float(fixed_coord), 9)), (va, round(float(v_val), 9))])
        return tuple(pair), float(grids[ua][u0]), float(grids[ua][u1])

    def _nearest_index(axis, value):
        return int(np.argmin(np.abs(grids[axis] - value)))

    def _refine_with_global(panel, piece, line_bps):
        """One refinement step for a single piece: cut at both its own
        same-panel transitions (see the module docstring) AND any known
        global line breakpoint that falls strictly inside one of its
        sides' ranges — a neighboring panel may need a cut here that this
        panel's own geometry alone wouldn't reveal, because splitting a
        rectangle for one side's sake also fragments its other sides as a
        byproduct, which can in turn desynchronize *its* neighbors. This
        is meant to be applied repeatedly until stable (see the fixed-point
        loop below), since one panel's split can cascade to require
        another's.
        """
        u0, u1, v0, v1 = piece
        info = _DIRECTIONS[panel["direction"]]
        ua, va = info["u_axis"], info["v_axis"]
        cells = panel["cells"]

        def status(side, coord):
            if side == "u0":
                return (u0 - 1, coord) in cells
            if side == "u1":
                return (u1, coord) in cells
            if side == "v0":
                return (coord, v0 - 1) in cells
            return (coord, v1) in cells

        u_cuts, v_cuts = set(), set()
        for side in ("v0", "v1"):
            vals = [status(side, u) for u in range(u0, u1)]
            for idx in range(len(vals) - 1):
                if vals[idx] != vals[idx + 1]:
                    u_cuts.add(u0 + idx + 1)
        for side in ("u0", "u1"):
            vals = [status(side, v) for v in range(v0, v1)]
            for idx in range(len(vals) - 1):
                if vals[idx] != vals[idx + 1]:
                    v_cuts.add(v0 + idx + 1)

        for side in ("u0", "u1"):
            key, lo, hi = rect_line_and_range(panel, u0, u1, v0, v1, side)
            for bp in line_bps.get(key, ()):
                if lo < bp < hi:
                    v_cuts.add(_nearest_index(va, bp))
        for side in ("v0", "v1"):
            key, lo, hi = rect_line_and_range(panel, u0, u1, v0, v1, side)
            for bp in line_bps.get(key, ()):
                if lo < bp < hi:
                    u_cuts.add(_nearest_index(ua, bp))

        u_pts = sorted({u0, u1} | {p for p in u_cuts if u0 < p < u1})
        v_pts = sorted({v0, v1} | {p for p in v_cuts if v0 < p < v1})
        return [(u_pts[i], u_pts[i + 1], v_pts[j], v_pts[j + 1])
                for i in range(len(u_pts) - 1) for j in range(len(v_pts) - 1)]

    # Fixed-point loop: recompute global line breakpoints from the current
    # rectangles, re-refine every panel against them, and repeat until
    # nothing changes. One panel's split can force a cut on a neighbor,
    # which can in turn force a cut on a *third* panel sharing a different
    # line with that neighbor — a single pass isn't always enough. This
    # is guaranteed to terminate: cuts only ever add grid-aligned
    # breakpoints, of which there are finitely many.
    for panel in panels:
        panel["rects"] = list(panel["greedy_rects"])

    for _iteration in range(8):
        line_breakpoints = {}
        for panel in panels:
            for (u0, u1, v0, v1) in panel["rects"]:
                for side in ("u0", "u1", "v0", "v1"):
                    if side_margin(panel, u0, u1, v0, v1, side) <= 0.0:
                        continue
                    key, lo, hi = rect_line_and_range(panel, u0, u1, v0, v1, side)
                    line_breakpoints.setdefault(key, set()).update((lo, hi))

        changed = False
        for panel in panels:
            new_rects = []
            for piece in panel["rects"]:
                new_rects.extend(_refine_with_global(panel, piece, line_breakpoints))
            if len(new_rects) != len(panel["rects"]):
                changed = True
            panel["rects"] = new_rects
        if not changed:
            break

    nodes = []
    registry = {}
    plate_quads = []
    rect_face_data = []               # parallel list: (face_ids,) per rectangle
    edge_registry = {}                # canonical edge key -> [(rect_index, boundary_name), ...]

    def edge_key(p1, p2):
        return frozenset([tuple(np.round(p1, 9)), tuple(np.round(p2, 9))])

    for panel in panels:
        info = _DIRECTIONS[panel["direction"]]
        f, ua, va, s = info["fixed_axis"], info["u_axis"], info["v_axis"], info["sign"]
        layer = panel["layer"]
        fixed_coord = grids[f][layer + (1 if s > 0 else 0)]

        for (u0, u1, v0, v1) in panel["rects"]:
            origin = np.zeros(3)
            du = np.zeros(3)
            dv = np.zeros(3)
            origin[f] = fixed_coord
            origin[ua] = grids[ua][u0]
            origin[va] = grids[va][v0]
            du[ua] = grids[ua][u1] - grids[ua][u0]
            dv[va] = grids[va][v1] - grids[va][v0]

            margins = (side_margin(panel, u0, u1, v0, v1, "u0"),
                       side_margin(panel, u0, u1, v0, v1, "u1"),
                       side_margin(panel, u0, u1, v0, v1, "v0"),
                       side_margin(panel, u0, u1, v0, v1, "v1"))

            face_ids, quads = _generate_face_sided(n, origin, du, dv, registry, nodes, margins)
            plate_quads.extend(quads)
            rect_index = len(rect_face_data)
            rect_face_data.append(face_ids)

            corners = {"i0": (origin, origin + dv), "in": (origin + du, origin + du + dv),
                       "j0": (origin, origin + du), "jn": (origin + dv, origin + du + dv)}
            margin_by_bnd = {"i0": margins[0], "in": margins[1], "j0": margins[2], "jn": margins[3]}
            for bnd, (p1, p2) in corners.items():
                if margin_by_bnd[bnd] <= 0.0:
                    continue  # welded to a same-panel neighbor — not a joint candidate
                key = edge_key(p1, p2)
                edge_registry.setdefault(key, []).append((rect_index, bnd))

    # Bridge every edge shared by exactly two inset (panel-boundary) sides
    # — same-cell corners, coplanar neighbor-panel continuations, and
    # diagonal-cell reentrant corners are all found this same way, since
    # it's pure edge-coincidence, not cell-adjacency bookkeeping.
    joint_quads = []
    for key, entries in edge_registry.items():
        if len(entries) != 2:
            continue  # a true open boundary edge shouldn't occur for a closed solid
        (r1, bnd1), (r2, bnd2) = entries
        ids1 = _boundary_rows(rect_face_data[r1])[bnd1]
        ids2 = _boundary_rows(rect_face_data[r2])[bnd2]

        p1, p2 = tuple(key)
        free_axis = int(np.argmax(np.abs(np.array(p2) - np.array(p1))))
        nodes_arr = np.array(nodes)
        # Sorting both rows by the same free-axis coordinate ascending is
        # enough to line them up correctly — same approach already proven
        # correct in cube_mesh_with_joints's edge bridging.
        ids1 = ids1[np.argsort(nodes_arr[ids1][:, free_axis])]
        ids2 = ids2[np.argsort(nodes_arr[ids2][:, free_axis])]

        for m in range(len(ids1) - 1):
            joint_quads.append([int(ids1[m]), int(ids1[m + 1]),
                                 int(ids2[m + 1]), int(ids2[m])])

    nodes = np.array(nodes)
    nodes_world = nodes @ axes.T + ref

    min_z = nodes_world[:, 2].min()
    z_tol = max(coord_tol, 1e-6)
    bottom_ids = np.where(nodes_world[:, 2] <= min_z + z_tol)[0].tolist()

    return nodes_world, np.array(plate_quads, dtype=int), np.array(joint_quads, dtype=int), bottom_ids
