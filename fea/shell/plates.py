"""Mesh a plate-based OBJ (e.g. a kyub export) directly: every OBJ
object/group is one flat plywood plate, meshed on its own, and plates are
connected to each other *only* through box-joint strips — the same
plate + joint-band model as `mesh.box_mesh_with_joints`, but for arbitrary
flat polygons meeting at arbitrary angles (splayed legs, tilted backrests).

  1. Weld coincident vertices across all plates (topology only) and split
     T-junctions, so two plates sharing part of an edge share vertices
     along the overlap. Edges used by 2+ plates are *seams* (joints).
  2. Refine by longest-edge bisection until no edge exceeds `h`. Edges are
     keyed globally, so a seam is subdivided identically on both plates.
  3. Un-weld: each plate gets its own copy of every node, and is inset
     away from each of its seams by `joint_width` (in its own plane; free
     edges stay put, corners are mitered) — clamped per plate to a quarter
     of its width so narrow plates (leg caps) aren't consumed.
  4. Split every triangle into 3 quads (centroid + edge midpoints) — our
     shell element is a 4-node quad. These are the plate elements.
  5. Bridge every seam segment with a joint-strip quad connecting the two
     plates' inset edges (the joint material). Like the box path, a small
     gap is left where 3+ plates meet at a corner.

Returns the same (nodes, plate_quads, joint_quads, bottom_ids) shape as
`remesh_rectilinear`, plus `info` (dict: per-plate element ranges, inset
widths, seam count) for reporting.
"""
import numpy as np


def _plane_residual(points):
    pts = np.asarray(points, dtype=float)
    c = pts.mean(axis=0)
    s = np.linalg.svd(pts - c, compute_uv=False)
    return s[-1] / max(np.sqrt(len(pts)), 1.0)


def looks_like_plates(groups, extent):
    """True if there are several groups and each is (nearly) planar —
    i.e. this OBJ is a set of flat panels, not one closed triangle soup."""
    if len(groups) < 2:
        return False
    tol = 1e-3 * extent
    for tris in groups:
        pts = np.asarray(tris, dtype=float).reshape(-1, 3)
        if len(pts) < 3 or _plane_residual(pts) > tol:
            return False
    return True


def _weld(points, tol):
    keys = np.round(points / tol).astype(np.int64)
    index = {}
    out_pts, remap = [], np.empty(len(points), dtype=int)
    for i, k in enumerate(map(tuple, keys)):
        j = index.get(k)
        if j is None:
            # also check neighbouring buckets so points straddling a bucket
            # boundary still weld
            for d in np.ndindex(3, 3, 3):
                kk = (k[0] + d[0] - 1, k[1] + d[1] - 1, k[2] + d[2] - 1)
                jj = index.get(kk)
                if jj is not None and np.linalg.norm(out_pts[jj] - points[i]) <= tol:
                    j = jj
                    break
        if j is None:
            j = len(out_pts)
            out_pts.append(points[i])
            index[k] = j
        remap[i] = j
    return np.array(out_pts), remap


def _degenerate(P, a, b, c, tol):
    """Zero-area / collinear triangle (height below tol)."""
    if len({a, b, c}) < 3:
        return True
    area2 = np.linalg.norm(np.cross(P[b] - P[a], P[c] - P[a]))
    longest = max(np.linalg.norm(P[b] - P[a]), np.linalg.norm(P[c] - P[b]),
                  np.linalg.norm(P[a] - P[c]))
    return area2 / longest < tol


def _split_t_junctions(P, tris, tol):
    """Split any triangle whose edge has another mesh vertex lying on it."""
    changed = True
    while changed:
        changed = False
        out = []
        for (a, b, c, plate) in tris:
            done = False
            for (p, q, r) in ((a, b, c), (b, c, a), (c, a, b)):
                e = P[q] - P[p]
                L2 = e @ e
                if L2 == 0:
                    continue
                t = (P - P[p]) @ e / L2
                d = np.linalg.norm(P - (P[p] + np.outer(t, e)), axis=1)
                hit = np.where((t > 1e-6) & (t < 1 - 1e-6) & (d < tol))[0]
                if len(hit):
                    v = int(hit[np.argmin(np.abs(t[hit] - 0.5))])
                    out.append((p, v, r, plate))
                    out.append((v, q, r, plate))
                    done = changed = True
                    break
            if not done:
                out.append((a, b, c, plate))
        tris = out
    return tris


def _plate_frame(points):
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c)
    return vt[0], vt[1], vt[2]


def _min_angle(p, q, r):
    ang = []
    for a, b, c in ((p, q, r), (q, r, p), (r, p, q)):
        u, v = b - a, c - a
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        ang.append(np.arccos(np.clip(u @ v / (nu * nv), -1, 1)) if nu * nv > 0 else 0.0)
    return min(ang)


def _segments_cross(p1, p2, q1, q2):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (orient(p1, p2, q1) * orient(p1, p2, q2) < 0 and
            orient(q1, q2, p1) * orient(q1, q2, p2) < 0)


def _lawson_flip(tris, X2):
    """Flip interior edges of one plate's triangulation to maximize the
    minimum angle (constrained Delaunay; boundary edges are never touched).
    kyub exports plates as triangle fans, full of slivers."""
    tris = [list(t) for t in tris]
    for _ in range(200):
        owner = {}
        for ti, t in enumerate(tris):
            for k in range(3):
                owner.setdefault(frozenset((t[k], t[(k + 1) % 3])), []).append(ti)
        flipped = False
        for e, ts in owner.items():
            if len(ts) != 2:
                continue
            t1, t2 = tris[ts[0]], tris[ts[1]]
            a, b = tuple(e)
            c = next(v for v in t1 if v not in e)
            d = next(v for v in t2 if v not in e)
            if not _segments_cross(X2[a], X2[b], X2[c], X2[d]):
                continue  # non-convex quad: can't flip
            old = min(_min_angle(X2[a], X2[b], X2[c]), _min_angle(X2[a], X2[b], X2[d]))
            new = min(_min_angle(X2[c], X2[d], X2[a]), _min_angle(X2[c], X2[d], X2[b]))
            if new > old + 1e-9:
                tris[ts[0]] = [c, d, a]
                tris[ts[1]] = [d, c, b]
                flipped = True
                break
        if not flipped:
            break
    return [tuple(t) for t in tris]


def _quad_quality(X2, quad):
    """Min interior angle (radians) of a 4-gon; <= 0 if not convex."""
    pts = [X2[v] for v in quad]
    angs = []
    sign = None
    for k in range(4):
        a, b, c = pts[k - 1], pts[k], pts[(k + 1) % 4]
        u, v = a - b, c - b
        cr = u[0] * v[1] - u[1] * v[0]
        if sign is None:
            sign = np.sign(cr)
        if np.sign(cr) != sign or cr == 0:
            return 0.0
        angs.append(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))
    return min(angs)


def _pair_into_quads(tris, X2, min_angle_deg=5.0):
    """Merge adjacent triangles of one plate into convex quads, leaving as
    few triangles as possible (then best worst-quad angle). Exhaustive for
    small plates, greedy otherwise. Returns (quads, leftover_triangles)."""
    owner = {}
    for ti, t in enumerate(tris):
        for k in range(3):
            owner.setdefault(frozenset((t[k], t[(k + 1) % 3])), []).append(ti)
    pair = {}
    for e, ts in owner.items():
        if len(ts) != 2:
            continue
        t1, t2 = tris[ts[0]], tris[ts[1]]
        a, b = tuple(e)
        c = next(v for v in t1 if v not in e)
        d = next(v for v in t2 if v not in e)
        quad = [a, c, b, d]
        q = _quad_quality(X2, quad)
        if q > np.radians(min_angle_deg):
            pair[(min(ts), max(ts))] = (q, quad)
    n = len(tris)
    nbrs = {i: [] for i in range(n)}
    for (i, j) in pair:
        nbrs[i].append(j)
        nbrs[j].append(i)

    if n <= 20:
        memo = {}

        def solve(mask):
            # returns (n_left, -worst_quality, chosen pairs)
            if mask in memo:
                return memo[mask]
            i = next((k for k in range(n) if not mask >> k & 1), None)
            if i is None:
                return (0, -np.inf, ())
            l, wq, ch = solve(mask | 1 << i)
            best = (l + 1, wq, ch)
            for j in nbrs[i]:
                if mask >> j & 1:
                    continue
                q, _ = pair[(min(i, j), max(i, j))]
                l2, wq2, ch2 = solve(mask | 1 << i | 1 << j)
                cand = (l2, max(wq2, -q), ch2 + ((min(i, j), max(i, j)),))
                if cand[:2] < best[:2]:
                    best = cand
            memo[mask] = best
            return best

        _, _, chosen = solve(0)
    else:
        chosen, used = [], set()
        for key in sorted(pair, key=lambda k: -pair[k][0]):
            if key[0] in used or key[1] in used:
                continue
            used.update(key)
            chosen.append(key)
    used = {i for k in chosen for i in k}
    return [pair[k][1] for k in chosen], [t for ti, t in enumerate(tris) if ti not in used]


def _boundary_loop(bnd, verts):
    """Single closed boundary loop (vertex list) or None (holes etc.)."""
    nxt = {}
    for e in bnd:
        a, b = tuple(e)
        nxt.setdefault(a, []).append(b)
        nxt.setdefault(b, []).append(a)
    if any(len(v) != 2 for v in nxt.values()) or len(nxt) != len(verts):
        return None
    loop = [min(nxt)]
    prev = None
    while True:
        cur = loop[-1]
        n = nxt[cur][0] if nxt[cur][0] != prev else nxt[cur][1]
        if n == loop[0]:
            break
        prev = cur
        loop.append(n)
        if len(loop) > len(verts):
            return None
    return loop if len(loop) == len(verts) else None


def _quadrangulate(loop, X2, min_angle_deg=5.0):
    """Split a simple polygon (even vertex count) into convex quads using
    only its own vertices, maximizing the worst quad's minimum angle
    (interval DP). Returns a list of quads or None if impossible."""
    n = len(loop)
    if n < 4 or n % 2:
        return None
    X = [X2[v] for v in loop]

    def inside_diag(i, j):
        if abs(i - j) in (1, n - 1):
            return True  # polygon edge
        p1, p2 = X[i], X[j]
        for k in range(n):
            a, b = k, (k + 1) % n
            if len({a, b, i, j}) < 4:
                continue
            if _segments_cross(p1, p2, X[a], X[b]):
                return False
        mid = (p1 + p2) / 2
        cnt = 0
        for k in range(n):
            a, b = X[k], X[(k + 1) % n]
            if (a[1] > mid[1]) != (b[1] > mid[1]):
                xin = a[0] + (mid[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                if xin > mid[0]:
                    cnt += 1
        return cnt % 2 == 1

    thr = np.radians(min_angle_deg)
    memo = {}

    def best(i, j):
        # sub-polygon loop[i..j] closed by (j, i); returns (quality, quads)
        if j - i == 1:
            return (np.inf, [])
        if (j - i + 1) % 2 or j - i < 3:
            return (-1.0, None)
        if (i, j) in memo:
            return memo[(i, j)]
        res = (-1.0, None)
        if inside_diag(i, j):
            for a in range(i + 1, j - 1):
                if (a - i + 1) % 2:
                    continue
                for b in range(a + 1, j):
                    if (b - a + 1) % 2 or (j - b + 1) % 2:
                        continue
                    if not (inside_diag(i, a) and inside_diag(a, b) and inside_diag(b, j)):
                        continue
                    quad = [loop[i], loop[a], loop[b], loop[j]]
                    q = _quad_quality(X2, quad)
                    if q < thr or q <= res[0]:
                        continue
                    parts = [best(i, a), best(a, b), best(b, j)]
                    if any(pq is None for _, pq in parts):
                        continue
                    qq = min([q] + [pq for pq, _ in parts])
                    if qq > res[0]:
                        res = (qq, [quad] + [x for _, pq in parts for x in pq])
        memo[(i, j)] = res
        return res

    q, quads = best(0, n - 1)
    return quads if quads else None


def mesh_plates(groups, h, joint_width=0.02, support_tol=None):
    """groups: list of plates, each a list of triangles [[x,y,z]x3].
    h: target element size. joint_width: inset of each plate from its seams.

    Block-structured: each plate's triangulation is cleaned (Lawson flips)
    and paired into convex 4-sided patches; every patch gets a structured
    grid, every leftover triangle is split into 3 grids around its
    centroid. Segment counts are tied (union-find) across opposite patch
    sides, triangle sides, shared patch edges and plate-plate seams, so all
    grids and joint strips match node for node."""
    all_tris = []
    for gi, tris in enumerate(groups):
        for t in tris:
            all_tris.append((np.asarray(t, dtype=float), gi))
    pts = np.array([t[0] for t in all_tris]).reshape(-1, 3)
    extent = float(np.ptp(pts, axis=0).max())
    tol = 1e-4 * extent

    P, remap = _weld(pts, tol)
    tris = []
    for k, (_, gi) in enumerate(all_tris):
        a, b, c = remap[3 * k:3 * k + 3]
        if not _degenerate(P, a, b, c, tol):
            tris.append((int(a), int(b), int(c), gi))
    tris = [t for t in _split_t_junctions(P, tris, tol) if not _degenerate(P, *t[:3], tol)]
    n_plates = len(groups)
    elen = lambda e: float(np.linalg.norm(P[list(e)[0]] - P[list(e)[1]]))

    # ---- per plate: frame, clean triangulation, patches, boundary
    plates = {}
    for g in range(n_plates):
        ptris = [t[:3] for t in tris if t[3] == g]
        if not ptris:
            continue
        verts = sorted({v for t in ptris for v in t})
        e1, e2, _ = _plate_frame(P[verts])
        X2 = {v: np.array([P[v] @ e1, P[v] @ e2]) for v in verts}
        raw_tris = list(ptris)
        ptris = _lawson_flip(ptris, X2)
        use = {}
        for t in ptris:
            for k in range(3):
                use.setdefault(frozenset((t[k], t[(k + 1) % 3])), []).append(t[(k + 2) % 3])
        bnd = {e: rs[0] for e, rs in use.items() if len(rs) == 1}
        area = sum(abs((X2[b][0] - X2[a][0]) * (X2[c][1] - X2[a][1])
                       - (X2[b][1] - X2[a][1]) * (X2[c][0] - X2[a][0])) / 2 for a, b, c in ptris)
        perim = sum(elen(e) for e in bnd)
        width = 2 * area / perim  # ~ strip width for long narrow plates
        loop = _boundary_loop(bnd, verts)
        quads = _quadrangulate(loop, X2) if loop else None
        if quads:
            left = []
        else:
            # try both kyub's own triangulation and the Delaunay-flipped one
            options = [_pair_into_quads(t, X2) for t in (ptris, raw_tris)]
            quads, left = min(options, key=lambda o: len(o[1]))
        plates[g] = {"verts": verts, "e1": e1, "e2": e2, "X2": X2, "bnd": bnd,
                     "width": width, "quads": quads, "tris": left,
                     # element size: global target, at most 3x the plate's
                     # width so narrow strips get reasonable aspect ratios
                     "h": min(h, 3 * width)}

    edge_plates = {}
    for g, pl in plates.items():
        for e in pl["bnd"]:
            edge_plates.setdefault(e, set()).add(g)
    seams = {e for e, gs in edge_plates.items() if len(gs) > 1}

    # ---- segment counts per edge class
    uf = {}

    def find(e):
        uf.setdefault(e, e)
        while uf[e] != e:
            uf[e] = uf[uf[e]]
            e = uf[e]
        return e

    def union(a, b):
        uf[find(a)] = find(b)

    edge_h = {}
    for g, pl in plates.items():
        for c in pl["quads"]:
            es = [frozenset((c[k], c[(k + 1) % 4])) for k in range(4)]
            union(es[0], es[2])
            union(es[1], es[3])
            for e in es:
                edge_h[e] = min(edge_h.get(e, np.inf), pl["h"])
        oriented = []
        for t in pl["tris"]:
            # A leftover triangle is meshed as a quad collapsed at one corner
            # (apex): only the two sides meeting at the apex are tied, the
            # base is free. Pick the apex whose two sides are closest in length.
            a, b, c = min(((t[0], t[1], t[2]), (t[1], t[2], t[0]), (t[2], t[0], t[1])),
                          key=lambda o: abs(elen(frozenset((o[1], o[2]))) - elen(frozenset((o[0], o[2])))))
            oriented.append((a, b, c))  # base a-b, apex c
            union(frozenset((b, c)), frozenset((a, c)))
            for e in (frozenset((a, b)), frozenset((b, c)), frozenset((a, c))):
                edge_h[e] = min(edge_h.get(e, np.inf), pl["h"])
        pl["tris"] = oriented
    need = {}
    for e, he in edge_h.items():
        r = find(e)
        need[r] = max(need.get(r, 1.0), elen(e) / he)
    count = {e: max(int(np.ceil(need[find(e)] - 1e-9)), 1) for e in edge_h}

    # ---- per-plate inset positions of the plate's vertices
    plate_pos, insets = {}, {}
    for g, pl in plates.items():
        w = min(joint_width, 0.25 * pl["width"])
        insets[g] = w
        at = {}
        for e, r in pl["bnd"].items():
            p, q = tuple(e)
            d2 = pl["X2"][q] - pl["X2"][p]
            nrm = np.array([-d2[1], d2[0]]) / np.linalg.norm(d2)
            if nrm @ (pl["X2"][r] - pl["X2"][p]) < 0:
                nrm = -nrm
            off = w if e in seams else 0.0
            at.setdefault(p, []).append((nrm, off))
            at.setdefault(q, []).append((nrm, off))
        for v in pl["verts"]:
            cons = at.get(v, [])
            delta = np.zeros(2)
            if cons and max(o for _, o in cons) > 0:
                A = np.array([n for n, _ in cons])
                rhs = np.array([o for _, o in cons])
                if len(cons) == 2 and abs(np.linalg.det(A)) > np.sin(np.radians(10)):
                    delta = np.linalg.solve(A, rhs)
                else:
                    k = int(np.argmax(rhs))
                    delta = A[k] * rhs[k]
                if np.linalg.norm(delta) > 4 * w:
                    delta *= 4 * w / np.linalg.norm(delta)
            plate_pos[(g, v)] = P[v] + delta[0] * pl["e1"] + delta[1] * pl["e2"]

    # ---- nodes
    nodes, registry = [], {}

    def get_node(key, pos_fn):
        i = registry.get(key)
        if i is None:
            i = len(nodes)
            nodes.append(np.asarray(pos_fn(), dtype=float))
            registry[key] = i
        return i

    def side(g, a, b):
        """Node ids along edge a->b of plate g (count[e]+1 of them)."""
        e = frozenset((a, b))
        k = count[e]
        lo, hi = min(a, b), max(a, b)
        X0, X1 = plate_pos[(g, lo)], plate_pos[(g, hi)]
        ids = []
        for j in range(k + 1):
            if j == 0:
                ids.append(get_node((g, "v", lo), lambda: X0))
            elif j == k:
                ids.append(get_node((g, "v", hi), lambda: X1))
            else:
                ids.append(get_node((g, "e", e, j), lambda j=j: X0 + (X1 - X0) * (j / k)))
        return ids if a == lo else ids[::-1]

    plate_quads, plate_of_quad = [], []

    def grid(g, bottom, right, top, left):
        """Coons-patch grid from 4 boundary node lists: bottom c0->c1,
        right c1->c2, top c3->c2, left c0->c3."""
        nu, nv = len(bottom) - 1, len(left) - 1
        assert len(top) == nu + 1 and len(right) == nv + 1
        X = lambda i: nodes[i]
        c0, c1, c2, c3 = X(bottom[0]), X(bottom[-1]), X(top[-1]), X(top[0])
        ids = np.empty((nu + 1, nv + 1), dtype=int)
        ids[:, 0], ids[:, nv], ids[0, :], ids[nu, :] = bottom, top, left, right
        for i in range(1, nu):
            for j in range(1, nv):
                s_, t_ = i / nu, j / nv
                x = ((1 - t_) * X(bottom[i]) + t_ * X(top[i]) + (1 - s_) * X(left[j]) + s_ * X(right[j])
                     - ((1 - s_) * (1 - t_) * c0 + s_ * (1 - t_) * c1 + s_ * t_ * c2 + (1 - s_) * t_ * c3))
                ids[i, j] = len(nodes)
                nodes.append(x)
        for i in range(nu):
            for j in range(nv):
                plate_quads.append([int(ids[i, j]), int(ids[i + 1, j]),
                                    int(ids[i + 1, j + 1]), int(ids[i, j + 1])])
                plate_of_quad.append(g)

    for g, pl in plates.items():
        for c in pl["quads"]:
            grid(g, side(g, c[0], c[1]), side(g, c[1], c[2]),
                 side(g, c[3], c[2]), side(g, c[0], c[3]))
        for a, b, c in pl["tris"]:
            apex_id = get_node((g, "v", c), lambda: plate_pos[(g, c)])
            apex = [apex_id] * (count[frozenset((a, b))] + 1)
            grid(g, side(g, a, b), side(g, b, c), apex, side(g, a, c))

    # ---- joint strips along every seam
    joint_quads = []
    for e in seams:
        a, b = sorted(e)
        gs = sorted(edge_plates[e])
        for g1, g2 in zip(gs, gs[1:]):
            l1, l2 = side(g1, a, b), side(g2, a, b)
            for i in range(len(l1) - 1):
                joint_quads.append([l1[i], l1[i + 1], l2[i + 1], l2[i]])

    nodes = np.array(nodes)
    z = nodes[:, 2]
    if support_tol is None:
        support_tol = max(0.005 * float(np.ptp(z)), tol)
    bottom_ids = np.where(z <= z.min() + support_tol)[0].tolist()
    info = {"n_plates": n_plates, "n_seams": len(seams),
            "n_patches": sum(len(pl["quads"]) for pl in plates.values()),
            "n_tri_patches": sum(len(pl["tris"]) for pl in plates.values()),
            "inset_mm": {g: round(1000 * w, 1) for g, w in insets.items()},
            "plate_of_quad": plate_of_quad}
    return (nodes, np.array(plate_quads, dtype=int),
            np.array(joint_quads, dtype=int).reshape(-1, 4), bottom_ids, info)
