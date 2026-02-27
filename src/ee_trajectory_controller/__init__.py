"""EE Trajectory Controller — now backed by manifold types."""

from manifold.types.common.pose import Pose6D
from manifold.types.common.twist import Twist
from manifold.types.act.trajectory import (
    TrajectoryControllerConfig,
    TrajectoryStep,
    Trajectory,
)

from manifold.utils.geometry import (
    integrate_position,
    integrate_rotation,
    rotvec_from_matrix,
    rotation_error,
    project_pose,
)

from .controller import computeSingleDeltaTwist, computeDeltaTwists
from .metrics import computeMetrics, plotMetrics
from .utils import return_data

# Backward-compatible aliases
Pose = Pose6D
controllerConfig = TrajectoryControllerConfig
trajectoryStep = TrajectoryStep
trajectory = Trajectory
projectObjectState = project_pose
integratePosition = integrate_position
integrateRotation = integrate_rotation
findRotVec = rotvec_from_matrix
errorRotVec = rotation_error

__all__ = [
    # Canonical names
    "Pose6D",
    "Twist",
    "TrajectoryControllerConfig",
    "TrajectoryStep",
    "Trajectory",
    "integrate_position",
    "integrate_rotation",
    "rotvec_from_matrix",
    "rotation_error",
    "project_pose",
    "computeSingleDeltaTwist",
    "computeDeltaTwists",
    "computeMetrics",
    "plotMetrics",
    "return_data",
    # Backward-compatible aliases
    "Pose",
    "controllerConfig",
    "trajectoryStep",
    "trajectory",
    "projectObjectState",
    "integratePosition",
    "integrateRotation",
    "findRotVec",
    "errorRotVec",
]
