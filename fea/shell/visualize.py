"""3D visualization of the shell mesh and (optionally) results."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from .model import ShellModel
from .solver import ShellResult


def plot(model: ShellModel, result: ShellResult = None, scale: float = 1.0,
          save_path: str = None, show: bool = True, color_by: str = "displacement"):
    """color_by: "displacement" (translation magnitude — dominated by
    overall panel bending, since the panel center is always farthest from
    any support) or "curvature" (local bending-curvature magnitude — a
    purely kinematic quantity that lights up wherever the shape is
    kinking sharply, e.g. a soft joint acting as a hinge, even where the
    resulting displacement and moment there are both modest)."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    nodes = model.nodes
    undeformed_faces = [nodes[q] for q in model.quads]

    if result is not None:
        ax.add_collection3d(Line3DCollection(
            [np.vstack([f, f[:1]]) for f in undeformed_faces],
            colors="0.75", linestyles="dashed", linewidths=0.6))

        deformed_nodes = nodes + result.displacements[:, :3] * scale

        if color_by == "displacement":
            disp_mag = np.linalg.norm(result.displacements[:, :3], axis=1)
            values = disp_mag[model.quads].mean(axis=1)
            cbar_label = "Displacement magnitude (m)"
            title_label = "displacement magnitude"
        elif color_by == "curvature":
            values = np.linalg.norm(result.element_curvature, axis=1)
            cbar_label = "Bending curvature magnitude (1/m)"
            title_label = "bending curvature"
        else:
            raise ValueError(f"color_by must be 'displacement' or 'curvature', got {color_by!r}")

        vmax = max(values.max(), 1e-12)
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(vmin=0.0, vmax=vmax)

        deformed_faces = [deformed_nodes[q] for q in model.quads]
        colors = [cmap(norm(v)) for v in values]
        poly = Poly3DCollection(deformed_faces, facecolors=colors, edgecolors="0.2", linewidths=0.3)
        ax.add_collection3d(poly)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.6, label=cbar_label)

        plot_nodes = deformed_nodes
        title = f"Deformed shape (x{scale:g}) — colored by {title_label}"
    else:
        poly = Poly3DCollection(undeformed_faces, facecolors="steelblue", alpha=0.6,
                                 edgecolors="0.2", linewidths=0.3)
        ax.add_collection3d(poly)
        plot_nodes = nodes
        title = "Mesh (undeformed)"

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
