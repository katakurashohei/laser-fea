"""3D visualization of the FEA mesh and (optionally) results."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .model import Model
from .solver import Result


def plot(model: Model, result: Result = None, scale: float = 1.0,
          save_path: str = None, show: bool = True):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    nodes = model.nodes
    undeformed_segments = [[nodes[n1], nodes[n2]] for n1, n2 in model.elements]

    if result is not None:
        ax.add_collection3d(Line3DCollection(
            undeformed_segments, colors="0.75", linestyles="dashed", linewidths=1))

        deformed_nodes = nodes + result.displacements * scale
        stresses = result.element_stresses
        vmax = max(abs(stresses.min()), abs(stresses.max()), 1e-9)
        cmap = plt.get_cmap("coolwarm")
        norm = plt.Normalize(vmin=-vmax, vmax=vmax)

        segments = [[deformed_nodes[n1], deformed_nodes[n2]] for n1, n2 in model.elements]
        colors = [cmap(norm(s)) for s in stresses]
        ax.add_collection3d(Line3DCollection(segments, colors=colors, linewidths=3))

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.6, label="Axial stress (Pa)  [+ tension / - compression]")

        plot_nodes = deformed_nodes
        title = f"Deformed shape (x{scale:g}) — colored by axial stress"
    else:
        ax.add_collection3d(Line3DCollection(undeformed_segments, colors="steelblue", linewidths=3))
        plot_nodes = nodes
        title = "Mesh (undeformed)"

    ax.scatter(*nodes.T, color="black", s=20, depthshade=False)

    all_pts = np.vstack([nodes, plot_nodes])
    mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
    center = (mins + maxs) / 2
    radius = max((maxs - mins).max() / 2, 1e-6) * 1.2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()
    return fig
