"""Fit an oriented bounding box to a point cloud (PCA-based).

Used to turn an arbitrary imported OBJ's vertices into the (center,
orientation, dimensions) of a box, which is then remeshed using our own
validated `box_mesh_with_joints` grid — we don't attempt to preserve or
mesh the import's actual (likely triangulated, arbitrary-topology)
surface, only its rough size/position/orientation.
"""
import numpy as np


def fit_oriented_box(vertices):
    """vertices: (N, 3) array-like of points.

    Returns (center, axes, half_extents):
      center: (3,) world position of the box center.
      axes: (3, 3) rotation matrix, columns are the box's local x/y/z axes
        in world coordinates (orthonormal, right-handed).
      half_extents: (3,) half-size of the box along each local axis.

    PCA recovers the true box axes exactly for a (possibly rotated)
    rectangular box with three distinct edge lengths, since the surface
    point distribution's covariance is diagonal in the box's own frame.
    For non-box-like input this is a rough fit, which is an accepted
    limitation — see the module docstring.
    """
    pts = np.asarray(vertices, dtype=float)
    if len(pts) < 4:
        raise ValueError("need at least 4 vertices to fit a box")

    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = centered.T @ centered / len(pts)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1

    local = centered @ axes
    mins, maxs = local.min(axis=0), local.max(axis=0)
    half_extents = (maxs - mins) / 2.0
    center = centroid + axes @ ((maxs + mins) / 2.0)

    return center, axes, half_extents
