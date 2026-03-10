import numpy as np
import matplotlib.pyplot as plt

from manifold.types.common.pose import Pose6D
from manifold.types.act.trajectory import Trajectory
from manifold.utils.geometry import rotvec_from_matrix, rotation_error

def computeMetrics(t: Trajectory, targetPose: Pose6D, slidingWindowTime: np.float64):
    """Extract position, rotation, velocity, and error metrics from a trajectory.

    Computes per-step: EE/object positions, rotation matrices, rotation vectors,
    linear/angular velocities, position error magnitudes, and rotation error magnitudes.

    Args:
        t: Planned trajectory to analyze.
        targetPose: Reserved for future use.
        slidingWindowTime: Reserved for future use.

    Returns:
        Dict with keys: time, eePositions, objPositions, ee_linear_velocities,
        object_linear_velocities, ee_angular_velocities, object_angular_velocities,
        eeRotations, objRotations, ee_rotationVectors, obj_rotationVectors,
        positionErrors, rotationErrors.
    """
    raise NotImplementedError


def plotMetrics(t: Trajectory, targetPose: Pose6D, metrics):
    """Visualize trajectory metrics in a 2x3 matplotlib figure.

    Subplots: (0,0) positions, (1,0) position error, (0,1) rotations,
    (1,1) rotation error, (0,2) linear velocities, (1,2) angular velocities.

    Args:
        t: The trajectory (unused directly, present for API consistency).
        targetPose: Reserved for future use.
        metrics: Dict returned by computeMetrics.
    """
    raise NotImplementedError
