"""Shell entry point: build a 6-plate plywood cube and load it axially.

    python -m fea.shell.cli [options]

The cube sits with its bottom face rigidly clamped (like sitting on a
rigid table). The top face is loaded either by a uniform pressure
(--load-type pressure, the default) or by a concentrated point load at
one of its four corners (--load-type corner), either pushing down
(--mode compression, the default) or pulling up (--mode tension,
stretching the box and putting the vertical joints in tension — usually
the more revealing case for a glued box joint).
"""
import argparse

import numpy as np

from .mesh import cube_mesh_with_joints
from .model import ShellMaterial, ShellModel
from .solver import solve
from .visualize import plot

# Isotropic approximation of structural plywood. Real plywood is an
# orthotropic layered material (face-grain vs cross-grain stiffness differ,
# and through-thickness/rolling shear stiffness is much lower); this single
# E is a simplification, in the spirit of the rest of this primitive solver.
PLYWOOD_E = 9.0e9    # Pa
PLYWOOD_NU = 0.3
PLYWOOD_T = 0.012    # m (12 mm)

# The joint band is not a real material — it's a stand-in for the combined
# compliance of the box-joint fingers, glue line, and local grain
# discontinuity at the corner. Defaulting to a fraction of the plate
# stiffness is a rough way to make the corners softer than the solid panels
# without modeling the actual interlocking geometry.
JOINT_STIFFNESS_FRACTION = 0.4
JOINT_WIDTH = 0.02   # m (20 mm)

# Top-face corner picks, keyed the same way as --corner.
_CORNER_INDEX = {
    "+x+y": lambda top: top[-1, -1],
    "+x-y": lambda top: top[-1, 0],
    "-x+y": lambda top: top[0, -1],
    "-x-y": lambda top: top[0, 0],
}


def build_cube(size, n, E, nu, t, joint_width, joint_stiffness_fraction, mode="compression",
               load_type="pressure", pressure=2000.0, force=500.0, corner="+x+y"):
    if mode not in ("compression", "tension"):
        raise ValueError(f"mode must be 'compression' or 'tension', got {mode!r}")
    if load_type not in ("pressure", "corner"):
        raise ValueError(f"load_type must be 'pressure' or 'corner', got {load_type!r}")
    sign = -1.0 if mode == "compression" else 1.0

    nodes, plate_quads, joint_quads, faces = cube_mesh_with_joints(size, n, joint_width)

    plate_material = ShellMaterial(E=E, nu=nu, t=t)
    joint_material = ShellMaterial(E=E * joint_stiffness_fraction, nu=nu, t=t)
    materials = [plate_material, joint_material]

    quads = np.vstack([plate_quads, joint_quads])
    element_material = np.array([0] * len(plate_quads) + [1] * len(joint_quads))

    bottom = faces["-z"]
    top = faces["+z"]

    supports = {}
    for idx in bottom.flatten():
        supports[int(idx)] = (True, True, True, True, True, True)

    loads = {}
    if load_type == "pressure":
        top_quads = []
        for i in range(n):
            for j in range(n):
                top_quads.append([top[i, j], top[i + 1, j], top[i + 1, j + 1], top[i, j + 1]])

        for quad in top_quads:
            pts = nodes[quad]
            x, y = pts[:, 0], pts[:, 1]
            area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            for node in quad:
                fz = sign * pressure * area / 4.0
                f = loads.get(int(node), [0.0] * 6)
                f[2] += fz
                loads[int(node)] = f
    else:
        if corner not in _CORNER_INDEX:
            raise ValueError(f"corner must be one of {list(_CORNER_INDEX)}, got {corner!r}")
        node = int(_CORNER_INDEX[corner](top))
        loads[node] = (0.0, 0.0, sign * force, 0.0, 0.0, 0.0)

    loads = {k: tuple(v) for k, v in loads.items()}

    model = ShellModel(nodes=nodes, quads=quads, materials=materials,
                        element_material=element_material, supports=supports, loads=loads)
    return model, faces, len(plate_quads)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fea-shell", description="Axially load a 6-plate plywood cube")
    parser.add_argument("--size", type=float, default=1.0, help="Cube edge length (m)")
    parser.add_argument("--mesh", type=int, default=12, help="Mesh divisions per face edge (NxN)")
    parser.add_argument("--E", type=float, default=PLYWOOD_E, help="Young's modulus (Pa)")
    parser.add_argument("--nu", type=float, default=PLYWOOD_NU, help="Poisson's ratio")
    parser.add_argument("--thickness", type=float, default=PLYWOOD_T, help="Plate thickness (m)")
    parser.add_argument("--mode", choices=["compression", "tension"], default="compression",
                         help="Push down on the top face (compression) or pull it up (tension)")
    parser.add_argument("--load-type", choices=["pressure", "corner"], default="pressure",
                         help="Uniform pressure over the whole top face, or a concentrated "
                              "point load at one of its corners")
    parser.add_argument("--pressure", type=float, default=2000.0,
                         help="Pressure magnitude on the top face (Pa), used with --load-type pressure")
    parser.add_argument("--force", type=float, default=500.0,
                         help="Point load magnitude (N), used with --load-type corner")
    parser.add_argument("--corner", choices=list(_CORNER_INDEX), default="+x+y",
                         help="Which corner of the top face to load, used with --load-type corner")
    parser.add_argument("--joint-width", type=float, default=JOINT_WIDTH,
                         help="Width of the box-joint band at each edge (m)")
    parser.add_argument("--joint-stiffness", type=float, default=JOINT_STIFFNESS_FRACTION,
                         help="Joint band Young's modulus as a fraction of the plate's E")
    parser.add_argument("--scale", type=float, default=20.0, help="Displacement scale for the plot")
    parser.add_argument("--color-by", choices=["displacement", "curvature"], default="displacement",
                         help="Color the deformed plot by translation magnitude (dominated by "
                              "overall panel bending) or by local bending curvature (lights up "
                              "wherever the shape kinks sharply, e.g. a soft joint acting as a hinge)")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--save", metavar="PATH")
    args = parser.parse_args(argv)

    model, faces, n_plate_quads = build_cube(args.size, args.mesh, args.E, args.nu, args.thickness,
                                              args.joint_width, args.joint_stiffness, mode=args.mode,
                                              load_type=args.load_type, pressure=args.pressure,
                                              force=args.force, corner=args.corner)
    result = solve(model)

    disp_mag = np.linalg.norm(result.displacements[:, :3], axis=1)
    top_ids = faces["+z"].flatten()
    bottom_ids = faces["-z"].flatten()

    total_load = sum(load[2] for load in model.loads.values())
    reaction_fz = result.reactions[bottom_ids, 2].sum()
    n_joint_quads = model.n_elements - n_plate_quads

    verb = "pushing down on" if args.mode == "compression" else "pulling up on"
    top_disp_z = result.displacements[top_ids, 2]
    top_disp = top_disp_z.min() if args.mode == "compression" else top_disp_z.max()

    print(f"Cube: {args.size:g} m edge, {args.mesh}x{args.mesh} mesh/face, "
          f"t={args.thickness*1000:g} mm plywood (E={args.E:.3g} Pa, nu={args.nu:g})")
    print(f"Joint band: {args.joint_width*1000:g} mm wide, E={args.E*args.joint_stiffness:.3g} Pa "
          f"({args.joint_stiffness:.0%} of plate stiffness), {n_joint_quads} elements "
          f"({n_plate_quads} plate elements)")
    if args.load_type == "pressure":
        print(f"Mode: {args.mode} — {verb} the top face at {args.pressure:.1f} Pa")
    else:
        print(f"Mode: {args.mode} — {verb} the {args.corner} corner of the top face at {args.force:.1f} N")
    print(f"Applied load magnitude: {abs(total_load):.1f} N")
    print(f"Sum of vertical reactions at base: {reaction_fz:.1f} N  "
          f"(should match applied load, sign flipped)")
    print(f"Max top-face displacement (z): {top_disp*1e3:+.4f} mm")
    print(f"Max overall displacement magnitude: {disp_mag.max()*1e3:.4f} mm")

    curvature_mag = np.linalg.norm(result.element_curvature, axis=1)
    plate_curv = curvature_mag[:n_plate_quads]
    joint_curv = curvature_mag[n_plate_quads:]
    print(f"Peak bending curvature — plate: {plate_curv.max():.4f} 1/m   "
          f"joint band: {joint_curv.max():.4f} 1/m"
          f"{'  <- joint bends more sharply' if joint_curv.max() > plate_curv.max() else ''}")

    plot(model, result=result, scale=args.scale, show=not args.no_show, save_path=args.save,
         color_by=args.color_by)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
