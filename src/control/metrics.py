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
    steps = t
    time = np.array([s.time for s in steps], dtype=np.float64)
    eePositions = np.array([np.asarray(s.pose.position, dtype=np.float64) for s in steps])
    eeRotations = np.array([np.asarray(s.pose.rotation_matrix, dtype=np.float64) for s in steps])

    ee_rotationVectors = np.array([rotvec_from_matrix(r) for r in eeRotations])

    outMetrics = dict()

    outMetrics['time'] = time
    outMetrics['eePositions'] = eePositions
    outMetrics['objPositions'] = np.array([]).reshape(0, 3)

    outMetrics['ee_linear_velocities'] = np.array([np.asarray(s.twist.linear, dtype=np.float64) for s in steps])
    outMetrics['object_linear_velocities'] = np.array([]).reshape(0, 3)
    outMetrics['ee_angular_velocities'] = np.array([np.asarray(s.twist.angular, dtype=np.float64) for s in steps])
    outMetrics['object_angular_velocities'] = np.array([]).reshape(0, 3)

    outMetrics['eeRotations'] = eeRotations
    outMetrics['objRotations'] = np.array([]).reshape(0, 3, 3)

    outMetrics['ee_rotationVectors'] = ee_rotationVectors
    outMetrics['obj_rotationVectors'] = np.array([]).reshape(0, 3)

    outMetrics['positionErrors'] = np.array([])
    outMetrics['rotationErrors'] = np.array([])
    return outMetrics


def plotMetrics(t: list[HandControl], targetPose: Pose6D, metrics):
    """Visualize trajectory metrics in a 2x3 matplotlib figure.

    Subplots: (0,0) EE positions, (1,0) position error, (0,1) EE rotations,
    (1,1) rotation error, (0,2) linear velocities, (1,2) angular velocities.

    Args:
        t: The trajectory (unused directly, present for API consistency).
        targetPose: Reserved for future use.
        metrics: Dict returned by computeMetrics.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(metrics['time'], metrics['eePositions'][:, 0], color="red", label="eePosition_X")
    ax.plot(metrics['time'], metrics['eePositions'][:, 1], color="blue", label="eePosition_Y")
    ax.plot(metrics['time'], metrics['eePositions'][:, 2], color="green", label="eePosition_Z")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.set_title("EE Positions")
    ax.grid()
    ax.legend()

    ax = axes[1, 0]
    if len(metrics['positionErrors']) > 0:
        ax.plot(metrics['time'], metrics['positionErrors'], color="red", label="PositionError")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error")
    ax.set_title("Position Error vs Time")
    ax.grid()
    ax.legend()

    ax = axes[0, 1]
    ax.plot(metrics['time'], metrics['ee_rotationVectors'][:, 0], color="red", label="eeRotation_X")
    ax.plot(metrics['time'], metrics['ee_rotationVectors'][:, 1], color="blue", label="eeRotation_Y")
    ax.plot(metrics['time'], metrics['ee_rotationVectors'][:, 2], color="green", label="eeRotation_Z")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rotation")
    ax.set_title("EE Rotation")
    ax.grid()
    ax.legend()

    ax = axes[1, 1]
    if len(metrics['rotationErrors']) > 0:
        ax.plot(metrics['time'], metrics['rotationErrors'], color="red", label="rotationError")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("rotationError")
    ax.set_title("Rotation Error vs Time")
    ax.grid()
    ax.legend()

    ax = axes[0, 2]
    ax.plot(metrics['time'], metrics['ee_linear_velocities'][:, 0], color="red", label="eelinearVelocity_X")
    ax.plot(metrics['time'], metrics['ee_linear_velocities'][:, 1], color="blue", label="eelinearVelocity_Y")
    ax.plot(metrics['time'], metrics['ee_linear_velocities'][:, 2], color="green", label="eelinearVelocity_Z")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Linear Velocity (m/s)")
    ax.set_title("EE Linear Velocity")
    ax.grid()
    ax.legend()

    ax = axes[1, 2]
    ax.plot(metrics['time'], metrics['ee_angular_velocities'][:, 0], color="red", label="eeangularVelocity_X")
    ax.plot(metrics['time'], metrics['ee_angular_velocities'][:, 1], color="blue", label="eeangularVelocity_Y")
    ax.plot(metrics['time'], metrics['ee_angular_velocities'][:, 2], color="green", label="eeangularVelocity_Z")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular Velocity (rad/s)")
    ax.set_title("EE Angular Velocity")
    ax.grid()
    ax.legend()

    plt.tight_layout()
    plt.show()
