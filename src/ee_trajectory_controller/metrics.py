import numpy as np
import matplotlib.pyplot as plt

from manifold.types.common.pose import Pose6D
from manifold.types.act.trajectory import Trajectory
from manifold.utils.geometry import rotvec_from_matrix, rotation_error

from .utils import return_data


def computeMetrics(t: Trajectory, targetPose: Pose6D, slidingWindowTime: np.float64):

    data = return_data(t)

    time = data['time']

    eePositions = data['ee_positions']
    eeRotations = data['ee_rotations']

    objPositions = data['object_positions']
    objRotations = data['object_rotations']

    ee_rotationVectors = []
    obj_rotationVectors = []
    for eeRot, objRot in zip(eeRotations, objRotations):
        eeRotVec = rotvec_from_matrix(eeRot)
        objRotVec = rotvec_from_matrix(objRot)

        ee_rotationVectors.append(eeRotVec)
        obj_rotationVectors.append(objRotVec)
    ee_rotationVectors = np.asarray(ee_rotationVectors)
    obj_rotationVectors = np.asarray(obj_rotationVectors)

    positionErrors = eePositions - objPositions
    positionErrorMagnitudes = np.linalg.norm(positionErrors, axis=1)

    rotationErrorMagnitudes = []
    for eeRot, objRot in zip(eeRotations, objRotations):
        rotationErrorVec = rotation_error(eeRot, objRot)
        rotationErrorVecMagnitude = np.linalg.norm(rotationErrorVec)
        rotationErrorMagnitudes.append(rotationErrorVecMagnitude)

    outMetrics = dict()

    outMetrics['time'] = time
    outMetrics['eePositions'] = eePositions
    outMetrics['objPositions'] = objPositions

    outMetrics['ee_linear_velocities'] = data['ee_linear_velocities']
    outMetrics['object_linear_velocities'] = data['object_linear_velocities']

    outMetrics['ee_angular_velocities'] = data['ee_angular_velocities']
    outMetrics['object_angular_velocities'] = data['object_angular_velocities']

    outMetrics['eeRotations'] = eeRotations
    outMetrics['objRotations'] = objRotations

    outMetrics['ee_rotationVectors'] = ee_rotationVectors
    outMetrics['obj_rotationVectors'] = obj_rotationVectors

    outMetrics['positionErrors'] = positionErrorMagnitudes
    outMetrics['rotationErrors'] = rotationErrorMagnitudes
    return outMetrics


def plotMetrics(t: Trajectory, targetPose: Pose6D, metrics):

    fig, axes = plt.subplots(2, 3, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(metrics['time'], metrics['eePositions'][:, 0], color="red", label="eePosition_X")
    ax.plot(metrics['time'], metrics['eePositions'][:, 1], color="blue", label="eePosition_Y")
    ax.plot(metrics['time'], metrics['eePositions'][:, 2], color="green", label="eePosition_Z")

    ax.plot(metrics['time'], metrics['objPositions'][:, 0], color="yellow", label="objPosition_X")
    ax.plot(metrics['time'], metrics['objPositions'][:, 1], color="purple", label="objPosition_Y")
    ax.plot(metrics['time'], metrics['objPositions'][:, 2], color="orange", label="objPosition_Z")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.set_title("EE positions vs Obj Positions")

    ax.grid()
    ax.legend()

    ax = axes[1, 0]
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

    ax.plot(metrics['time'], metrics['obj_rotationVectors'][:, 0], color="yellow", label="objRotation_X")
    ax.plot(metrics['time'], metrics['obj_rotationVectors'][:, 1], color="purple", label="objRotation_Y")
    ax.plot(metrics['time'], metrics['obj_rotationVectors'][:, 2], color="orange", label="objRotation_Z")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rotation")
    ax.set_title("EE rotation vs Obj rotation")

    ax.grid()
    ax.legend()

    ax = axes[1, 1]
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

    ax.plot(metrics['time'], metrics['object_linear_velocities'][:, 0], color="yellow", label="objlinearVelocity_X")
    ax.plot(metrics['time'], metrics['object_linear_velocities'][:, 1], color="purple", label="objlinearVelocity_Y")
    ax.plot(metrics['time'], metrics['object_linear_velocities'][:, 2], color="orange", label="objlinearVelocity_Z")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Linear Velocity (m/s)")
    ax.set_title("EE linear velocity vs Obj linear velocity")

    ax.grid()
    ax.legend()

    ax = axes[1, 2]
    ax.plot(metrics['time'], metrics['ee_angular_velocities'][:, 0], color="red", label="eeangularVelocity_X")
    ax.plot(metrics['time'], metrics['ee_angular_velocities'][:, 1], color="blue", label="eeangularVelocity_Y")
    ax.plot(metrics['time'], metrics['ee_angular_velocities'][:, 2], color="green", label="eeangularVelocity_Z")

    ax.plot(metrics['time'], metrics['object_angular_velocities'][:, 0], color="yellow", label="objectangularVelocity_X")
    ax.plot(metrics['time'], metrics['object_angular_velocities'][:, 1], color="purple", label="objectangularVelocity_Y")
    ax.plot(metrics['time'], metrics['object_angular_velocities'][:, 2], color="orange", label="objectangularVelocity_Z")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular Velocity (rad/s)")
    ax.set_title("EE angular velocity vs Obj angular velocity")

    ax.grid()
    ax.legend()

    plt.tight_layout()
    plt.show()
