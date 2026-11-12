"""Local web server for the interactive shell-solver GUI.

    python -m fea.shell.gui

Serves a Three.js frontend (fea/shell/gui/static/) that lets you click a
node on the plywood-cube mesh, drag to aim a force vector, and see the
result of the *actual* fea.shell solver — the same code path as the CLI —
rendered live in 3D. No physics or mesh logic is duplicated here: this
module only builds the model, calls `fea.shell.solver.solve`, and ships
the result as JSON.

You can also drag-and-drop an .obj file onto the page. If it's rectilinear
(a union of boxes sharing one orthogonal frame — an L-shape, a concave
step, etc.) it's reconstructed exactly via voxel-grid + ray-cast analysis
(`fea.shell.rectilinear.remesh_rectilinear`); otherwise it falls back to
fitting a single oriented bounding box to its vertices
(`fea.shell.obb.fit_oriented_box`) and remeshing as our own box+joint grid
at that size/position/orientation — genuinely non-rectilinear (curved,
organic) surfaces are never meshed directly. See "Importing OBJ files" in
the README.

Whichever face ends up lowest in world Z (after any fit) is always the one
rigidly clamped, and the only load is whichever single point force the
user last placed.
"""
import webbrowser
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from ..cli import JOINT_STIFFNESS_FRACTION, JOINT_WIDTH, PLYWOOD_E, PLYWOOD_NU, PLYWOOD_T
from ..mesh import box_mesh_with_joints
from ..model import ShellMaterial, ShellModel
from ..obb import fit_oriented_box
from ..plates import looks_like_plates, mesh_plates
from ..rectilinear import remesh_rectilinear
from ..solver import solve

STATIC_DIR = Path(__file__).parent / "static"
CUBE_SIZE = 1.0
MESH_N = 12             # per-face grid density for the cube/OBB-fit path
RECTILINEAR_N = 5        # elements per already-merged panel-rectangle side
                          # (each merged panel rectangle is split 5x5)
MIN_MESH_N, MAX_MESH_N = 1, 20   # range of the GUI's "Mesh resolution" slider
MAX_NODES = 1800         # the solver assembles a dense K, so memory and time
                          # grow as (6 * nodes)^2 — refuse meshes past this
PLATES_N = 1             # default resolution for plate-based imports: the
                          # sparsest mesh (target edge = largest dimension / n,
                          # capped per plate at 3x its width); slider refines
MM_DETECT_EXTENT = 10.0  # imports larger than this (in file units) are
                          # assumed to be in mm and scaled to m

app = Flask(__name__, static_folder=None)

_plate_material = ShellMaterial(E=PLYWOOD_E, nu=PLYWOOD_NU, t=PLYWOOD_T)
_joint_material = ShellMaterial(E=PLYWOOD_E * JOINT_STIFFNESS_FRACTION, nu=PLYWOOD_NU, t=PLYWOOD_T)
_materials = [_plate_material, _joint_material]

# Mutable mesh state, rebuilt by _rebuild_mesh() — initially the default cube.
_nodes = _plate_quads = _joint_quads = _quads = _element_material = None
_faces = _bottom_ids = _bottom_face_name = None
# How the current mesh was built, so /api/resolution can rebuild the same
# shape at a different density: ("box", half_extents, center, axes) or
# ("rectilinear", triangles). `_mesh_n` is the density it was built at.
_mesh_source = None
_mesh_n = MESH_N


def _check_size(nodes):
    if len(nodes) > MAX_NODES:
        raise ValueError(f"mesh would have {len(nodes)} nodes (limit {MAX_NODES}) — "
                         f"lower the resolution")


def _set_mesh_state(nodes, plate_quads, joint_quads, bottom_ids, bottom_face_name):
    global _nodes, _plate_quads, _joint_quads, _quads, _element_material
    global _faces, _bottom_ids, _bottom_face_name

    _nodes = np.asarray(nodes)
    _plate_quads = np.asarray(plate_quads, dtype=int)
    _joint_quads = np.asarray(joint_quads, dtype=int).reshape(-1, 4)
    _quads = np.vstack([_plate_quads, _joint_quads]) if len(_joint_quads) else _plate_quads
    _element_material = np.array([0] * len(_plate_quads) + [1] * len(_joint_quads))
    _faces = None
    _bottom_ids = list(bottom_ids)
    _bottom_face_name = bottom_face_name


def _rebuild_mesh(half_extents, n=MESH_N, joint_width=JOINT_WIDTH, center=None, axes=None):
    # A joint band wider than the panel it's cut from makes no sense —
    # clamp it to a safe fraction of the box's smallest half-extent so
    # small imported objects don't produce a degenerate/inverted inset.
    joint_width = min(joint_width, 0.2 * min(half_extents))

    nodes, plate_quads, joint_quads, faces = box_mesh_with_joints(
        half_extents, n, joint_width, center=center, axes=axes)
    _check_size(nodes)

    # Clamp whichever face ends up physically lowest, not just local "-z" —
    # an imported box may be arbitrarily oriented. `faces[name]` is a 2D
    # (n+1, n+1) grid of node indices, so flatten before indexing `nodes`
    # (indexing with the 2D array directly and then [:, 2] would silently
    # pick the wrong axis instead of the z-coordinate).
    bottom_face_name = min(faces, key=lambda name: nodes[faces[name].flatten()][:, 2].mean())
    bottom_ids = faces[bottom_face_name].flatten().tolist()

    global _faces, _mesh_source, _mesh_n
    _set_mesh_state(nodes, plate_quads, joint_quads, bottom_ids, bottom_face_name)
    _faces = faces  # kept around for face-name lookups (box path only)
    _mesh_source = ("box", tuple(half_extents), center, axes)
    _mesh_n = n


def _rebuild_rectilinear(triangles, n=RECTILINEAR_N):
    nodes, plate_quads, joint_quads, bottom_ids = remesh_rectilinear(
        triangles, n=n, joint_width=JOINT_WIDTH)
    _check_size(nodes)

    global _faces, _mesh_source, _mesh_n
    _set_mesh_state(nodes, plate_quads, joint_quads, bottom_ids, "auto (lowest layer)")
    _faces = None
    _mesh_source = ("rectilinear", triangles)
    _mesh_n = n
    return nodes


def _plate_h(groups, n):
    """Mesh resolution n -> target edge length: largest dimension / n."""
    pts = np.asarray([p for g in groups for t in g for p in t], dtype=float)
    return float(np.ptp(pts, axis=0).max()) / n


def _rebuild_plates(groups, n):
    nodes, plate_quads, joint_quads, bottom_ids, _ = mesh_plates(
        groups, _plate_h(groups, n), joint_width=JOINT_WIDTH)
    _check_size(nodes)

    global _faces, _mesh_source, _mesh_n
    _set_mesh_state(nodes, plate_quads, joint_quads, bottom_ids, "auto (lowest nodes)")
    _faces = None
    _mesh_source = ("plates", groups)
    _mesh_n = n
    return nodes


_rebuild_mesh((CUBE_SIZE / 2, CUBE_SIZE / 2, CUBE_SIZE / 2))


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/app.js")
def app_js():
    return send_from_directory(STATIC_DIR, "app.js")


def _mesh_json(**extra):
    return jsonify({
        "nodes": _nodes.tolist(),
        "plate_quads": _plate_quads.tolist(),
        "joint_quads": _joint_quads.tolist(),
        "bottom_ids": _bottom_ids,
        "n_plate": len(_plate_quads),
        "n_joint": len(_joint_quads),
        "bottom_face": _bottom_face_name,
        "resolution": _mesh_n,
        "min_resolution": MIN_MESH_N,
        "max_resolution": MAX_MESH_N,
        "resolution_kind": {"box": "per face", "rectilinear": "per panel",
                            "plates": "across model"}[_mesh_source[0]],
        **extra,
    })


@app.route("/api/mesh")
def api_mesh():
    return _mesh_json()


@app.route("/api/reset", methods=["POST"])
def api_reset():
    _rebuild_mesh((CUBE_SIZE / 2, CUBE_SIZE / 2, CUBE_SIZE / 2))
    return _mesh_json()


@app.route("/api/resolution", methods=["POST"])
def api_resolution():
    """Body: {"n": int}. Rebuilds the current shape (default cube, fitted
    box, or rectilinear import) at a new mesh density — divisions per face
    for a box, per merged panel rectangle for a rectilinear import. Node
    indices change, so the client has to drop any placed load/fixed nodes."""
    body = request.get_json(force=True)
    try:
        n = int(body["n"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "expected integer n"}), 400
    if not (MIN_MESH_N <= n <= MAX_MESH_N):
        return jsonify({"error": f"n must be in [{MIN_MESH_N}, {MAX_MESH_N}]"}), 400

    try:
        if _mesh_source[0] == "box":
            _, half_extents, center, axes = _mesh_source
            _rebuild_mesh(half_extents, n=n, center=center, axes=axes)
        elif _mesh_source[0] == "plates":
            _rebuild_plates(_mesh_source[1], n=n)
        else:
            _rebuild_rectilinear(_mesh_source[1], n=n)
    except ValueError as e:
        return jsonify({"error": str(e), "resolution": _mesh_n}), 400
    return _mesh_json()


@app.route("/api/materials", methods=["POST"])
def api_materials():
    """Body: {"plate_E": Pa, "joint_stiffness_fraction": 0-1}. Updates the
    (shared, mesh-independent) plate and joint materials in place — no
    mesh rebuild needed, since geometry doesn't depend on stiffness.
    `_materials` already holds references to these two objects, so the
    very next /api/solve picks up the change automatically."""
    body = request.get_json(force=True)
    try:
        plate_E = float(body["plate_E"])
        fraction = float(body["joint_stiffness_fraction"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "expected plate_E (Pa) and joint_stiffness_fraction (0-1)"}), 400
    if plate_E <= 0:
        return jsonify({"error": "plate_E must be positive"}), 400
    if not (0.0 < fraction <= 1.0):
        return jsonify({"error": "joint_stiffness_fraction must be in (0, 1]"}), 400

    _plate_material.E = plate_E
    _joint_material.E = plate_E * fraction
    return jsonify({"plate_E": _plate_material.E, "joint_E": _joint_material.E})


@app.route("/api/import_obj", methods=["POST"])
def api_import_obj():
    """Body: {"vertices": [[x,y,z],...], "triangles": [[[x,y,z]x3],...]}
    — raw OBJ geometry, already parsed client-side.

    First tries to reconstruct the shape exactly as a rectilinear solid
    (`fea.shell.rectilinear.remesh_rectilinear`, using the triangles) —
    this handles L-shapes, concave steps, anything that's a union of boxes
    sharing one orthogonal frame. If that fails (input isn't rectilinear,
    or something about it defeats the axis/grid detection), falls back to
    fitting a single oriented bounding box to the vertices, same as before.
    """
    body = request.get_json(force=True)
    vertices = body.get("vertices", [])
    triangles = body.get("triangles", [])
    groups = [g for g in body.get("groups", []) if g]

    # The solver works in SI (m, Pa, N). CAD/laser-cutting tools (e.g. kyub)
    # usually export in millimeters, which would make a chair 740 "m" tall
    # and the 12 mm plywood thickness / 20 mm joint width absurdly thin
    # relative to it. Nothing we'd plausibly analyze is > 10 m across, so
    # treat larger extents as mm and convert.
    unit_scale = 1.0
    if len(vertices):
        extent = float(np.ptp(np.asarray(vertices, dtype=float), axis=0).max())
        if extent > MM_DETECT_EXTENT:
            unit_scale = 1e-3
    if unit_scale != 1.0:
        vertices = (np.asarray(vertices, dtype=float) * unit_scale).tolist()
        triangles = (np.asarray(triangles, dtype=float) * unit_scale).tolist()
        groups = [(np.asarray(g, dtype=float) * unit_scale).tolist() for g in groups]
    units = "mm (converted to m)" if unit_scale != 1.0 else "m"

    # Tier 0: a plate-based export (kyub etc.) — one flat polygon per OBJ
    # object. Mesh those panels directly; this is exact, unlike the
    # rectilinear/box tiers, and handles tilted/splayed panels.
    if len(vertices) and looks_like_plates(
            groups, float(np.ptp(np.asarray(vertices, dtype=float), axis=0).max())):
        last_error = None
        for n in range(PLATES_N, MIN_MESH_N - 1, -1):
            try:
                _rebuild_plates(groups, n)
                break
            except ValueError as e:
                last_error = str(e)
        else:
            return jsonify({"error": last_error}), 400
        dims = (_nodes.max(axis=0) - _nodes.min(axis=0)).tolist()
        return _mesh_json(fitted_dimensions=dims, method="plate mesh",
                          n_groups=len(groups), units=units)

    if triangles:
        try:
            # Validate the shape at the lowest density first, so a mesh that's
            # merely too fine at RECTILINEAR_N isn't mistaken for "not
            # rectilinear" and silently refit as a box.
            remesh_rectilinear(triangles, n=MIN_MESH_N, joint_width=JOINT_WIDTH)
        except Exception as e:
            rectilinear_error = str(e)
        else:
            # Complex shapes (e.g. a chair: many panels) can blow past
            # MAX_NODES at the default density — step down to the finest
            # density that fits instead of refusing the import outright.
            nodes = None
            for n in range(RECTILINEAR_N, MIN_MESH_N - 1, -1):
                try:
                    nodes = _rebuild_rectilinear(triangles, n=n)
                    break
                except ValueError as e:
                    last_error = str(e)
            if nodes is None:
                return jsonify({"error": last_error}), 400
            dims = (nodes.max(axis=0) - nodes.min(axis=0)).tolist()
            return _mesh_json(fitted_dimensions=dims, method="rectilinear reconstruction",
                              units=units)
    else:
        rectilinear_error = "no triangle data sent"

    if len(vertices) < 4:
        return jsonify({"error": "need at least 4 vertices to fit a box"}), 400

    try:
        center, axes, half_extents = fit_oriented_box(vertices)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if min(half_extents) < 1e-6:
        return jsonify({"error": "fitted box is degenerate (near-zero extent on one axis)"}), 400

    try:
        _rebuild_mesh(tuple(half_extents), center=center, axes=axes)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return _mesh_json(fitted_dimensions=[float(2 * h) for h in half_extents],
                       method="oriented-bounding-box fit",
                       rectilinear_fallback_reason=rectilinear_error,
                       units=units)


@app.route("/api/solve", methods=["POST"])
def api_solve():
    body = request.get_json(force=True)
    node = int(body["node"])
    force = [float(x) for x in body["force"]]
    # Which nodes are clamped is now chosen interactively in the GUI
    # (Fix-support mode); fall back to the auto-computed lowest layer only
    # if the client omits the field entirely (e.g. a direct API call like
    # the ones in this repo's own tests) — an explicit empty list means
    # "the user cleared every fixed node," which is a real error, not a
    # request to fall back (`[] or _bottom_ids` would silently paper over
    # that distinction, since an empty list is falsy in Python).
    fixed_nodes = body["fixed_nodes"] if "fixed_nodes" in body else _bottom_ids
    fixed_nodes = [int(idx) for idx in fixed_nodes]

    if not (0 <= node < len(_nodes)):
        return jsonify({"error": f"node index {node} out of range"}), 400
    if not fixed_nodes or any(not (0 <= idx < len(_nodes)) for idx in fixed_nodes):
        return jsonify({"error": "select at least one valid fixed node"}), 400

    supports = {idx: (True,) * 6 for idx in fixed_nodes}
    loads = {node: (force[0], force[1], force[2], 0.0, 0.0, 0.0)}

    model = ShellModel(nodes=_nodes, quads=_quads, materials=_materials,
                        element_material=_element_material,
                        supports=supports, loads=loads)
    try:
        result = solve(model)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    reaction_sum = result.reactions[fixed_nodes].sum(axis=0)[:3].tolist()
    disp_mag = np.linalg.norm(result.displacements[:, :3], axis=1)
    curvature_mag = np.linalg.norm(result.element_curvature, axis=1)
    n_plate = len(_plate_quads)

    # Imports reconstructed via remesh_rectilinear() have zero joint
    # elements (that path is for arbitrary rectilinear topology, not the
    # box-joint study) — guard the max() calls against the empty slice.
    def _safe_max(arr):
        return float(arr.max()) if len(arr) else 0.0

    return jsonify({
        "displacements": result.displacements[:, :3].tolist(),
        "displacement_magnitude": disp_mag.tolist(),
        "curvature": curvature_mag.tolist(),
        "reaction_sum": reaction_sum,
        "max_displacement": float(disp_mag.max()),
        "max_curvature_plate": _safe_max(curvature_mag[:n_plate]),
        "max_curvature_joint": _safe_max(curvature_mag[n_plate:]),
    })


def main():
    host, port = "127.0.0.1", 5050
    url = f"http://{host}:{port}"
    print(f"laser-fea interactive GUI: {url}  (Ctrl+C to stop)")
    webbrowser.open(url)
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
