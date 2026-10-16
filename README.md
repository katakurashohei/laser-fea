# laser-fea

A very primitive, shell-driven finite-element solver in Python. There are two
independent solvers, sharing the same style (direct-stiffness, linear
elastic, plain JSON/matplotlib):

- **`fea`** — 2-node 3D truss/bar elements (axial stiffness only, no
  bending). Good for space-frame/lattice structures.
- **`fea.shell`** — 4-node flat shell elements (membrane + Mindlin plate
  bending, 6 DOF/node). Good for plates/panels that actually bend, like a
  box made of plywood faces under compression.

Both are meant as small, readable starting points to build a more capable
FEA codebase on top of (solid elements, thermal loads, nonlinearity,
orthotropic materials, etc.).

## Quick start — truss solver

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m fea.cli examples/tripod.json
python -m fea.cli examples/pyramid_tower.json --scale 200
```

Each run solves the model, prints nodal displacements, element axial
forces/stresses, and support reactions to the terminal, then opens a 3D plot:
the undeformed mesh in dashed grey, and the deformed shape (exaggerated by
`--scale`) colored by axial stress (red = tension, blue = compression).

Useful flags:

- `--scale N` — exaggerate displacements in the plot (real deflections are
  usually too small to see).
- `--no-show` — skip the interactive plot window (e.g. for headless/CI runs).
- `--save out.png` — save the plot to a file instead of/in addition to showing it.
- `--mesh-only` — just draw the mesh without solving (sanity-check geometry).

## Model input format

Models are plain JSON. Node and element indices are 0-based.

```jsonc
{
  "nodes": [[x, y, z], ...],        // node coordinates (meters)
  "elements": [[i, j], ...],        // each element connects node i to node j
  "material": {"E": 210e9, "A": 0.0005},  // Young's modulus (Pa), area (m^2) — applied to every element
  "supports": {"0": [true, true, true]},   // node -> which of (x, y, z) are fixed
  "loads": {"3": [0, 0, -5000]}            // node -> applied force (N) in (x, y, z)
}
```

## Quick start — shell (plate) solver

```bash
python -m fea.shell.cli
python -m fea.shell.cli --mesh 8 --pressure 5000 --scale 15
python -m fea.shell.cli --mode tension --pressure 1000
```

With no arguments this builds a 1 m plywood cube — six flat plate faces,
bottom face rigidly clamped, a uniform pressure pushing down on the top
face (`--mode compression`, the default) — solves it, prints a summary
(applied load, reaction check, max displacement), and opens a 3D plot: the
undeformed box in dashed grey, and the deformed shape colored by
displacement magnitude (the top plate bows into the classic bullseye
pattern you'd expect from a plate simply loaded and clamped at its edges).

`--mode tension` flips the load to pull the top face upward instead,
stretching the box and putting the vertical box-joint bands in tension
rather than compression — usually the more revealing load case for a
glued joint, since glue lines and interlocking fingers are typically much
weaker in tension/peel than in bearing compression.

`--load-type corner` switches from a uniform pressure to a concentrated
point load at one corner of the top face (`--corner`, one of `+x+y`,
`+x-y`, `-x+y`, `-x-y`; `--force` in N instead of `--pressure` in Pa) —
deformation then concentrates locally around that corner and decays away,
instead of the symmetric bullseye you get from uniform pressure:

```bash
python -m fea.shell.cli --load-type corner --force 500 --corner +x+y
```

Useful flags: `--size`, `--mesh` (divisions per face edge), `--E`, `--nu`,
`--thickness`, `--mode` (`compression`/`tension`), `--load-type`
(`pressure`/`corner`), `--pressure`, `--force`, `--corner`, `--scale`,
`--no-show`, `--save PATH`.

### Interactive GUI — click a node, drag to apply a force

```bash
python -m fea.shell.gui
```

Starts a local Flask server and opens a Three.js 3D view of the plywood
cube in your browser (`http://127.0.0.1:5050`). Drag empty space to orbit;
**click a node, hold, and drag** to aim a force — drag direction sets the
force direction, drag distance sets magnitude — release to solve. The
deformed shape and coloring (displacement or curvature — same `--color-by`
distinction as the CLI) update live, with a stats panel showing the
applied force, reaction-sum equilibrium check, and peak plate-vs-joint
curvature, mirroring what `fea.shell.cli` prints.

Crucially, **no physics is duplicated in the browser**: `fea/shell/gui/server.py`
builds the exact same `ShellModel` and calls the exact same
`fea.shell.solver.solve` used everywhere else in this repo — the frontend
(`fea/shell/gui/static/`) only handles rendering and the drag interaction,
then POSTs `{node, force, fixed_nodes}` to `/api/solve` and renders back
whatever comes out.

### Choosing what's fixed

Every new mesh starts with whichever face ends up lowest in world Z
auto-clamped (see "Importing OBJ files" below for why that's not always
local "-z"), same as before — but now that's just a starting point, not
fixed. Click **Fix-support mode** to switch what clicking a node does:
instead of picking a force, each click toggles that node as a fixed
support (all 6 DOF), colored blue. Click through as many nodes as you
want — a whole face, a single corner, an asymmetric set, anything — then
turn Fix-support mode back off to place a force against whatever you just
built. "Clear all fixed nodes" empties the set; "Reset fixed to auto"
restores the original lowest-layer clamp. The solve is rejected
(with a clear message, not a crash) if you try to solve with zero fixed
nodes. Mesh density and joint width still have no UI (edit
`fea/shell/gui/server.py` to change them).

### Tweaking stiffness live

The **Plate E** and **Joint stiffness** sliders update the two materials
in place on the server (`/api/materials`) — no mesh rebuild, since
geometry doesn't depend on stiffness — and if a force is already placed,
automatically re-solve so the effect shows up immediately instead of
waiting for the next drag. Slider input is debounced (200ms) so dragging
doesn't flood the server with solves. Verified the direction is physically
right in both cases: stiffening the plate (9→30 GPa) dropped a fixed
test-load deflection from 0.0044mm to 0.0013mm; then softening the joint
(40%→2%, same stiff plate) pushed it back up to 0.0089mm. These are
global settings, not per-mesh — they persist through OBJ imports and
resets until you change them again.

### Changing mesh resolution

The **Mesh resolution** slider remeshes whatever shape is loaded (default
cube, fitted box, or rectilinear import) when you release it, via
`/api/resolution`. For a box it's divisions per face edge (default 12);
for a rectilinear import it's divisions per merged panel rectangle
(default 5). Node numbering changes, so any placed load is cleared and the
fixed nodes reset to the lowest layer. The solver assembles a dense
stiffness matrix, so meshes over `MAX_NODES` (1800, in
`fea/shell/gui/server.py`) are refused and the slider snaps back. Around
that size a solve takes a few seconds.

### Importing OBJ files

Drag-and-drop a `.obj` file anywhere onto the GUI page to replace the
current mesh. Our shell element needs a clean structured quad grid, and an
arbitrary OBJ is typically an irregular triangle soup that doesn't weld
into one, so nothing ever meshes the import's raw surface directly.
Instead there are two tiers, tried in order:

1. **Rectilinear reconstruction** (`fea/shell/rectilinear.py:
   remesh_rectilinear`) — handles *any* union of boxes sharing one
   orthogonal frame: an L-shape, a concave step, anything "boxy but not a
   single box". It clusters the triangles' face normals to find the
   shape's 3 true orthogonal axes (robust even for asymmetric shapes,
   unlike fitting axes to the vertex cloud), snaps all vertices onto a
   grid along those axes, and ray-casts each candidate grid cell's center
   against the original triangles to decide solid/empty. Try
   `examples/l_shape_3cubes.obj` or `examples/concave_stack.obj`: both
   come back at their *true* dimensions (e.g. the L-shape as
   0.8 x 0.4 x 0.8 m), not a bounding box.

   The exposed cells of each of the 6 directions are grouped into
   **logical panels** (connected components — a flat wall spanning
   several grid cells is one panel, not one per cell), and each panel is
   meshed with as few quads as the underlying shape allows: a plain
   rectangular panel is a single quad; an L-shaped or ring-shaped panel
   (e.g. the frame left on a face once a smaller box is centered on it)
   is greedily split into a handful of rectangles, not one per cell.
   Box-joint bands are only inserted at a panel's *real* boundary sides
   (where it meets a different panel — same-cell corners, coplanar
   neighbor-panel continuations, and diagonal-cell reentrant corners are
   all found the same way, by matching inset edges' exact shared
   endpoints); sides bordering the same panel are welded flush, not
   jointed. Getting this right took a fixed-point pass (one panel's split
   can force a neighboring panel to split too, which can cascade further)
   — verified by checking every node is reachable from the base through
   the assembled mesh and that both example shapes solve with correct
   reaction-sum equilibrium.
2. **Oriented-bounding-box fallback** (`fea/shell/obb.py:
   fit_oriented_box`) — if step 1 fails (genuinely non-rectilinear input:
   curved, organic, or too irregular for clean axis/grid detection), fits
   a single oriented box to the vertices via PCA and remeshes to that
   size/position/orientation using `box_mesh_with_joints`, same as
   before. Exact recovery verified on a synthetic rotated/translated test
   box; for a non-box shape this is a rough approximation, and the reply
   says which method actually ran so it's never ambiguous which one you got.

`examples/test_crate.obj` is a plain box — also goes through rectilinear
reconstruction (a single box is trivially rectilinear). Imported panels are
split `RECTILINEAR_N` x `RECTILINEAR_N` (5x5), while the default cube uses
`MESH_N` x `MESH_N` (12x12) per face (both in `fea/shell/gui/server.py`). Click "Reset mesh to
default cube" to go back.

The default material (`fea/shell/cli.py`: `PLYWOOD_E/NU/T`) is an
**isotropic approximation of plywood** — E ≈ 9 GPa, ν ≈ 0.3, t = 12 mm. Real
plywood is an orthotropic layered material (face-grain vs cross-grain
stiffness differ substantially, and through-thickness "rolling shear"
stiffness is much lower); modeling that properly would mean per-ply
orthotropic stiffness, which this primitive version doesn't attempt yet.

### Box-joint corners

The six faces aren't welded directly edge-to-edge. Each face is inset from
the cube's nominal boundary by `--joint-width` (default 20 mm), and the gap
at each of the 12 cube edges is bridged by its own one-element-wide strip
of quads with different (by default softer) material properties — a crude
stand-in for a real box/finger joint's combined glue-line and interlocking-
finger compliance, without modeling the actual joint geometry. Tune it with:

- `--joint-width` — width of the joint band (m).
- `--joint-stiffness` — the joint band's Young's modulus as a fraction of
  the plate's E (default 0.4, i.e. 40% as stiff as the solid plywood).

A small triangular gap is left unmeshed at each of the 8 cube corners,
where three joint strips meet but don't quite close — not unlike the
exposed end-grain gap you'd actually see at a real box-joint corner.
Setting `--joint-stiffness 1.0` makes the joints as stiff as the plate
(recovering close to the original welded-edge behavior); lowering it
increases overall compliance, as expected for a softer connection.

### Seeing the joint deform: displacement vs. curvature coloring

`--color-by displacement` (the default) colors the deformed plot by raw
translation magnitude. This will **always** show the panel centers as the
most "deformed" spots, no matter how soft the joints are — displacement is
measured from the fixed base, and a panel's center is geometrically
farthest from any support (rigid or hinge-like), so it accumulates the most
travel regardless of where the actual compliance lives. Softening the
joint does increase the *overall* displacement (confirmed: sweeping
`--joint-stiffness` from 0.4 down to 0.0005 raised max deflection from
3.1 mm to 7.2 mm), but you won't see it show up as "the joint is the
brightest region."

`--color-by curvature` instead colors by local bending-curvature magnitude
— a purely kinematic quantity (no material stiffness involved), computed
post-solve from each element's own bending B-matrix
(`element_bending_curvature` in `element.py`) at its centroid. This lights
up wherever the shape is kinking sharply, which is exactly what a soft
joint acting as a hinge does — even in the default configuration
(40% joint stiffness), the joint band's peak curvature is already about
1.8x the plate's peak curvature, and `cli.py` prints both values so you
can check this without even looking at the plot:

```bash
python -m fea.shell.cli --color-by curvature
```

## Code layout

Truss solver (`fea/`):
- `model.py` — `Model` / `Material` data structures.
- `io.py` — load a `Model` from a JSON file.
- `solver.py` — assembles the global stiffness matrix, applies boundary
  conditions, solves `K u = F`, and recovers element forces/stresses and
  support reactions.
- `visualize.py` — 3D matplotlib plot of the mesh, and of the deformed
  shape colored by stress.
- `cli.py` — the shell entry point (`python -m fea.cli ...`).
- `examples/` — sample models (`tripod.json`, `pyramid_tower.json`), and
  test OBJs for the GUI's drag-and-drop import: `test_crate.obj` (a plain
  box), `l_shape_3cubes.obj` and `concave_stack.obj` (rectilinear,
  non-convex shapes — see "Importing OBJ files" below).

Shell/plate solver (`fea/shell/`):
- `element.py` — the 4-node flat shell element: bilinear membrane +
  Mindlin plate bending (selective reduced integration on the shear term
  to avoid locking), plus a small artificial drilling stiffness so the
  global matrix isn't singular at flat interior nodes. Validated against
  the analytic simply-supported-square-plate-under-uniform-pressure
  solution (converges to within ~1% — the residual is the expected
  Mindlin shear-flexibility correction over classical thin-plate theory,
  not solver error).
- `model.py` — `ShellModel` / `ShellMaterial`. `ShellModel` holds a list of
  materials plus a per-element material index, so different elements (e.g.
  plate vs. joint band) can carry different stiffness. 6 DOF/node: 3
  translation + 3 rotation.
- `mesh.py` — procedural mesh generators: `grid_plate` (single flat panel,
  used for validation), `cube_mesh` (six welded faces, no joints),
  `box_mesh_with_joints` (the general form: independent per-axis
  half-extents, optional world position/rotation, inset faces bridged by a
  distinct joint-band strip at each of the 12 edges), and
  `cube_mesh_with_joints` (thin cube-specific wrapper around it).
- `obb.py` — `fit_oriented_box`: PCA-based oriented-bounding-box fit used
  as the fallback for a non-rectilinear OBJ import (turns its vertices
  into a (center, orientation, dimensions) that `box_mesh_with_joints`
  can remesh).
- `rectilinear.py` — `remesh_rectilinear`: the primary OBJ-import path,
  for any union of boxes sharing one orthogonal frame. Detects the
  shape's true axes from face-normal clustering, snaps vertices to a
  grid, ray-casts to classify grid cells solid/empty, meshes only the
  exterior faces (via `mesh._generate_face`) — no bounding-box
  approximation, the actual topology comes through — and bridges every
  real edge (found by matching exposed patches' edges by shared
  endpoints, not by hardcoded cube topology) with a box-joint strip.
- `solver.py` — same direct-stiffness approach as the truss solver, scaled
  to 6 DOF/node, looking up each element's material via `element_material`;
  also recovers per-element bending curvature post-solve
  (`ShellResult.element_curvature`).
- `visualize.py` — 3D matplotlib plot using filled quads (`Poly3DCollection`),
  colorable by displacement magnitude or bending curvature (`color_by`).
- `cli.py` — builds and runs the plywood-cube (with box-joint corners) axial-load scenario.
- `gui/server.py` — Flask app: serves the frontend and exposes `/api/mesh`,
  `/api/solve`, `/api/import_obj`, `/api/reset`, and `/api/materials`,
  calling straight into `solver.solve` / `mesh.box_mesh_with_joints` /
  `obb.fit_oriented_box` (no duplicated physics or mesh logic).
  `gui/static/app.js` — Three.js scene, node-picking raycaster, click-drag
  force gizmo, click-to-toggle fixed-support picking, drag-and-drop OBJ
  import, material-stiffness sliders, and live re-coloring of the result.

## Known limitations (this is the primitive version)

Truss solver:
- Axial-only elements — no bending/shear stiffness.
- One material/section applied uniformly to all elements.
- No mesh generation — nodes/elements are hand-specified.

Shell solver:
- Isotropic material only (no orthotropic/layered plywood model yet).
- No stress/moment recovery — curvature is recovered (see above), but not
  yet converted to actual bending stress/moment (would need `D_b @ kappa`
  per element, which is straightforward to add but not done yet), so
  there's no direct "this is overstressed" signal.
- The joint band is a homogenized material zone, not real joint geometry —
  it captures "the corner is more compliant" but not finger-by-finger
  stress concentration, glue-specific failure, or bearing contact between
  fingers.
- No buckling or nonlinear (large-deflection) analysis — a thin panel
  under enough in-plane compression would buckle, which this linear
  solver can't predict, only bending under transverse/pressure loads.
- Loads/supports are wired up procedurally for the cube scenario in
  `cli.py`, not read from a general-purpose input file yet.

Interactive GUI:
- One point load at a time (a new drag replaces the previous one; it
  doesn't accumulate multiple simultaneous loads). Fixed supports, by
  contrast, do accumulate — that's the point — but only per-node, not
  per-face/edge/mid-element, and each fixed node is always fully clamped
  (all 6 DOF); there's no way to pick a partial constraint (e.g. free to
  rotate but not translate) from the UI.
- No UI yet for changing mesh density, material, or joint parameters
  (edit `gui/server.py` directly).
- OBJ import reconstructs exact topology for rectilinear (box-union)
  shapes, but still falls back to a single fitted bounding box for
  anything genuinely non-rectilinear — a curved or organic OBJ will still
  come back as a box (see "Importing OBJ files" above).
- Rectilinear reconstruction merges coplanar same-panel cells and only
  joints real panel-to-panel edges (a flat wall spanning several grid
  cells is one welded panel, not one joint per cell) — but it still can't
  distinguish "this is a real physical joint" from "the input happened to
  have a grid line here" for cells that really are on different panels;
  it joints every panel boundary it finds, which is the best it can infer
  from geometry alone.
- Rectilinear axis/grid detection assumes reasonably clean geometry
  (real orthogonal faces, no near-duplicate grid lines within its
  tolerance); a noisy or slightly-non-rectilinear export may fail to
  detect 3 axes and silently fall back to the bounding-box fit instead.
- Runs the Flask dev server (`app.run(debug=False)`), fine for local
  single-user interactive use, not meant to be exposed beyond localhost.

Both solvers: linear elastic, small-displacement theory only.
