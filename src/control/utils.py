import numpy as np

from manifold.types.common.pose import Pose6D
from manifold.types.common.twist import Twist
from manifold.types.act.controller_config import TrajectoryControllerConfig
from manifold.types.act.control import HandControl
from manifold.types.common.list import List
from manifold.utils.geometry import (
    rotvec_from_matrix,
    integrate_position,
    integrate_rotation,
)


def computeSingleDeltaTwist(
    eePose: Pose6D,
    eeTwist: Twist,
    objPose: Pose6D,
    objTwist: Twist,
    params: TrajectoryControllerConfig,
    velocity_bias: Twist | None = None,
) -> Twist:
    """Compute a single velocity correction step (proportional + feedforward).

    Applies the control law:
        linear_delta  = (v_obj + kp_position * (p_obj - p_ee)) + velocity_bias.linear - v_ee
        angular_delta = (w_obj + kp_rotation * rotvec(R_ee^T @ R_obj)) - w_ee

    When params.linear_only is True, angular_delta is zero.

    Args:
        eePose: End effector pose.
        eeTwist: End effector twist.
        objPose: Object pose.
        objTwist: Object twist.
        params: Controller configuration (gains, flags).
        velocity_bias: Optional additive twist bias on the desired velocity.

    Returns:
        Delta twist to apply to the end effector.
    """
    raise NotImplementedError


def computeDeltaTwists(
    eePose: Pose6D,
    eeTwist: Twist,
    objPose: Pose6D,
    objTwist: Twist,
    params: TrajectoryControllerConfig,
    max_linear_velocity: float = float("inf"),
    velocity_bias: Twist | None = None,
    max_steps: int | None = None,
) -> List[HandControl]:
    """Plan a multi-step trajectory using proportional + feedforward control.

    At each step: compute delta twist, update EE velocity (clamped to
    max_linear_velocity), integrate both EE and object states forward by dt.
    Number of steps is max(3, int(horizon / dt) + 1) where horizon is the
    larger of simulation_horizon and latency, optionally capped by max_steps.

    Args:
        eePose: Initial end effector pose.
        eeTwist: Initial end effector twist.
        objPose: Initial object pose.
        objTwist: Object twist (assumed constant).
        params: Controller configuration.
        max_linear_velocity: Speed clamp for EE linear velocity.
        velocity_bias: Optional additive twist bias applied at every step.
        max_steps: Optional upper bound on trajectory length.

    Returns:
        List of HandControl waypoints; index 0 = initial state, rest = planned steps.
    """
    raise NotImplementedError


def interpolate_plan(
    plan: List[HandControl], elapsed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Linearly interpolate pose/twist from a plan at a given elapsed time.

    Used by compute_blended_start to predict where the robot should be
    based on the last plan. Clamps to first/last step if elapsed is out of range.

    Returns:
        Tuple of (position, linear_vel, rotation_rpy, angular_vel).
    """
    raise NotImplementedError
