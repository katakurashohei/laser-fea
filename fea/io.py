"""Load a Model from a plain JSON file.

Expected shape:

{
  "nodes": [[x, y, z], ...],
  "elements": [[node_i, node_j], ...],
  "material": {"E": 210e9, "A": 0.0005},
  "supports": {"0": [true, true, true]},
  "loads": {"3": [0, 0, -5000]}
}
"""
import json
import numpy as np

from .model import Material, Model


def load_model(path: str) -> Model:
    with open(path) as f:
        data = json.load(f)

    nodes = np.array(data["nodes"], dtype=float)
    elements = np.array(data["elements"], dtype=int)
    material = Material(E=float(data["material"]["E"]), A=float(data["material"]["A"]))

    supports = {
        int(node): tuple(bool(x) for x in flags)
        for node, flags in data.get("supports", {}).items()
    }
    loads = {
        int(node): tuple(float(x) for x in force)
        for node, force in data.get("loads", {}).items()
    }

    return Model(nodes=nodes, elements=elements, material=material,
                 supports=supports, loads=loads)
