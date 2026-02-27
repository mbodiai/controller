"""Re-export manifold geometry utilities used by the controller.

All canonical implementations live in manifold.utils.geometry.
This module provides backward-compatible aliases.
"""

from manifold.utils.geometry import (
    integrate_position,
    integrate_rotation,
    rotvec_from_matrix,
    rotation_error,
)

# Backward-compatible aliases
integratePosition = integrate_position
integrateRotation = integrate_rotation
findRotVec = rotvec_from_matrix
errorRotVec = rotation_error

__all__ = [
    "integrate_position",
    "integratePosition",
    "integrate_rotation",
    "integrateRotation",
    "rotvec_from_matrix",
    "findRotVec",
    "rotation_error",
    "errorRotVec",
]
