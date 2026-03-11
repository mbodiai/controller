import numpy as np

from manifold.types.common.pose import Pose6D
from manifold.types.common.twist import Twist
from manifold.types.act.controller_config import TrajectoryControllerConfig
from manifold.types.act.control import HandControl
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
    linear_error = objPose.position - eePose.position
    desired_lin_vel = objTwist.linear + linear_error * params.kp_position
    if velocity_bias is not None:
        desired_lin_vel = desired_lin_vel + velocity_bias.linear
    lin_delta = desired_lin_vel - eeTwist.linear

    if params.linear_only:
        return Twist.from_linear_angular(lin_delta, np.zeros(3, dtype=np.float64))

    rot_error = rotvec_from_matrix(eePose.rotation_matrix.T @ objPose.rotation_matrix)
    desired_ang_vel = objTwist.angular + rot_error * params.kp_rotation
    ang_delta = desired_ang_vel - eeTwist.angular

    return Twist.from_linear_angular(lin_delta, ang_delta)


def computeDeltaTwists(
    eePose: Pose6D,
    eeTwist: Twist,
    objPose: Pose6D,
    objTwist: Twist,
    params: TrajectoryControllerConfig,
    max_linear_velocity: float = float("inf"),
    velocity_bias: Twist | None = None,
    max_steps: int | None = None,
) -> list[HandControl]:
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
    dt = params.dt
    linear_only = params.linear_only
    horizon = max(params.latency, params.simulation_horizon)
    nSteps = max(3, int(horizon / dt) + 1)
    if max_steps is not None:
        nSteps = max(3, min(nSteps, max_steps))

    ee_pose = eePose
    ee_twist = eeTwist
    obj_pose = objPose
    obj_twist = objTwist

    steps: list[HandControl] = [HandControl(
        pose=ee_pose,
        twist=ee_twist,
        time=0.0,
    )]

    for step in range(1, nSteps):
        delta = computeSingleDeltaTwist(
            ee_pose, ee_twist, obj_pose, obj_twist,
            params, velocity_bias=velocity_bias,
        )

        ee_lin = ee_twist.linear + delta.linear
        ee_ang = ee_twist.angular + delta.angular
        speed = np.linalg.norm(ee_lin)
        if speed > max_linear_velocity:
            ee_lin = ee_lin * (max_linear_velocity / speed)

        ee_pos = integrate_position(
            np.asarray(ee_pose.position, dtype=np.float64), ee_lin, dt,
        )
        ee_rot = integrate_rotation(
            np.asarray(ee_pose.rotation_matrix, dtype=np.float64), ee_ang, dt,
        ) if not linear_only else np.asarray(ee_pose.rotation_matrix, dtype=np.float64)

        obj_pos = integrate_position(
            np.asarray(obj_pose.position, dtype=np.float64),
            np.asarray(obj_twist.linear, dtype=np.float64), dt,
        )
        obj_rot = integrate_rotation(
            np.asarray(obj_pose.rotation_matrix, dtype=np.float64),
            np.asarray(obj_twist.angular, dtype=np.float64), dt,
        ) if not linear_only else np.asarray(obj_pose.rotation_matrix, dtype=np.float64)

        ee_pose = Pose6D.from_position_and_rotation_matrix(ee_pos, ee_rot)
        ee_twist = Twist.from_linear_angular(ee_lin, ee_ang)
        obj_pose = Pose6D.from_position_and_rotation_matrix(obj_pos, obj_rot)

        steps.append(HandControl(
            pose=ee_pose,
            twist=ee_twist,
            time=float(step) * dt,
        ))

    return steps


def interpolate_plan(
    plan: list[HandControl], elapsed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Linearly interpolate pose/twist from a plan at a given elapsed time.

    Used by compute_blended_start to predict where the robot should be
    based on the last plan. Clamps to first/last step if elapsed is out of range.

    Returns:
        Tuple of (position, linear_vel, rotation_rpy, angular_vel).
    """
    steps = plan

    def _extract(s: HandControl) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(s.pose.position, dtype=np.float64),
            np.asarray(s.twist.linear, dtype=np.float64),
            np.array([s.pose.roll, s.pose.pitch, s.pose.yaw], dtype=np.float64),
            np.asarray(s.twist.angular, dtype=np.float64),
        )

    if elapsed <= steps[0].time:
        return _extract(steps[0])
    if elapsed >= steps[-1].time:
        return _extract(steps[-1])

    for i in range(len(steps) - 1):
        t0, t1 = steps[i].time, steps[i + 1].time
        if t0 <= elapsed <= t1:
            alpha = (elapsed - t0) / (t1 - t0) if t1 > t0 else 0.0
            p0, lv0, r0, av0 = _extract(steps[i])
            p1, lv1, r1, av1 = _extract(steps[i + 1])
            return (
                p0 + alpha * (p1 - p0),
                lv0 + alpha * (lv1 - lv0),
                r0 + alpha * (r1 - r0),
                av0 + alpha * (av1 - av0),
            )

    return _extract(steps[-1])
