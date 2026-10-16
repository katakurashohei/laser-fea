"""Shell entry point: `python -m fea.cli <model.json> [options]`."""
import argparse
import sys

import numpy as np

from .io import load_model
from .solver import solve
from .visualize import plot


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fea", description="Primitive 3D truss FEA solver")
    parser.add_argument("input", help="Path to a model JSON file")
    parser.add_argument("--scale", type=float, default=1.0,
                         help="Displacement scale factor for the deformed-shape plot")
    parser.add_argument("--no-show", action="store_true",
                         help="Don't open an interactive plot window")
    parser.add_argument("--save", metavar="PATH",
                         help="Save the plot to an image file (e.g. out.png)")
    parser.add_argument("--mesh-only", action="store_true",
                         help="Just plot the mesh, skip solving")
    args = parser.parse_args(argv)

    model = load_model(args.input)

    if args.mesh_only:
        plot(model, result=None, show=not args.no_show, save_path=args.save)
        return 0

    try:
        result = solve(model)
    except ValueError as e:
        print(f"Solve failed: {e}", file=sys.stderr)
        return 1

    np.set_printoptions(precision=6, suppress=True)

    print("Nodal displacements (x, y, z) [m]:")
    for i, d in enumerate(result.displacements):
        print(f"  node {i}: {d}")

    print("\nElement axial forces [N]  (+ tension / - compression):")
    for i, f in enumerate(result.element_forces):
        n1, n2 = model.elements[i]
        print(f"  elem {i} (n{n1}-n{n2}): {f:.3f}")

    print("\nSupport reactions [N]:")
    for node in sorted(model.supports):
        print(f"  node {node}: {result.reactions[node]}")

    plot(model, result=result, scale=args.scale, show=not args.no_show, save_path=args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
