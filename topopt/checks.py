"""Post-optimisation checks on a voxel design (used before any result is shown)."""
import numpy as np
from scipy import ndimage


def connectivity(x3d: np.ndarray, level: float = 0.5):
    """Label face-connected solid regions of a (nx, ny, nz) density field.
    Returns (number of components, voxel count of each component, labels)."""
    solid = x3d >= level
    labels, n = ndimage.label(solid)                    # 6-connectivity: voxels sharing a face
    sizes = ndimage.sum(solid, labels, index=np.arange(1, n + 1)) if n else np.array([])
    return n, np.asarray(sizes, int), labels


def touches(labels: np.ndarray, voxel_index: np.ndarray, component: int) -> float:
    """Fraction of the given voxels (flat indices) that belong to the given component."""
    lab = labels.ravel()[voxel_index]
    return float(np.mean(lab == component)) if lab.size else 0.0
