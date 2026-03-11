import numpy as np
import matplotlib.pyplot as plt

from manifold.types.common.pose import Pose6D
from manifold.types.act.control import HandControl
from manifold.utils.geometry import rotvec_from_matrix, rotation_error

def computeMetrics(t: list[HandControl], targetPose: Pose6D, slidingWindowTime: np.float64):
    """Extract position, rotation, and velocity metrics from a trajectory.

    Since trajectory steps are HandControl (EE-only), object-side metrics are
    no longer available. Object fields in the returned dict are empty arrays.

    Args:
        t: Planned trajectory to analyze.
        targetPose: Reserved for future use.
        slidingWindowTime: Reserved for future use.

    Returns:
        Dict with keys: time, eePositions, ee_linear_velocities,
        ee_angular_velocities, eeRotations, ee_rotationVectors,
        positionErrors, rotationErrors.
    """
    raise NotImplementedError


def plotMetrics(t: list[HandControl], targetPose: Pose6D, metrics):
    """Visualize trajectory metrics in a 2x3 matplotlib figure.

    Subplots: (0,0) EE positions, (1,0) position error, (0,1) EE rotations,
    (1,1) rotation error, (0,2) linear velocities, (1,2) angular velocities.

    Args:
        t: The trajectory (unused directly, present for API consistency).
        targetPose: Reserved for future use.
        metrics: Dict returned by computeMetrics.
    """
    raise NotImplementedError
