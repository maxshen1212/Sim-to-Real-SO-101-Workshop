# Compatibility helpers for math utilities that moved/were deprecated across Isaac Sim versions.
#
# Isaac Sim 6.0 moved `isaacsim.core.utils.rotations` into a deprecated extension that is not on
# the default import path, so `from isaacsim.core.utils.rotations import euler_angles_to_quat`
# raises `ModuleNotFoundError`. This module provides a drop-in, Isaac-version-independent
# replacement built on scipy (already a dependency of the Isaac stack).

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def euler_angles_to_quat(
    euler_angles: np.ndarray, degrees: bool = False, extrinsic: bool = True
) -> np.ndarray:
    """Convert Euler XYZ angles to a scalar-first (w, x, y, z) quaternion.

    Drop-in replacement for the deprecated
    ``isaacsim.core.utils.rotations.euler_angles_to_quat``: same XYZ input order, same
    ``degrees`` / ``extrinsic`` semantics, and the same scalar-first (w, x, y, z) output.

    Args:
        euler_angles: Euler XYZ angles, shape (3,) or (N, 3).
        degrees: True if the input is in degrees, False if radians.
        extrinsic: True for extrinsic XYZ (scipy "xyz"), False for intrinsic (scipy "XYZ").

    Returns:
        Quaternion(s) as (w, x, y, z), shape (4,) or (N, 4).
    """
    euler_angles = np.asarray(euler_angles, dtype=float)
    seq = "xyz" if extrinsic else "XYZ"
    q = Rotation.from_euler(seq, euler_angles, degrees=degrees).as_quat()  # scipy: (x, y, z, w)
    if q.ndim == 1:
        return np.array([q[3], q[0], q[1], q[2]])
    return np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]])
