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
    linear_error = np.array([objPose.x, objPose.y, objPose.z]) - np.array([eePose.x, eePose.y, eePose.z])
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
        duration=0.0,
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
            np.array([ee_pose.x, ee_pose.y, ee_pose.z], dtype=np.float64), ee_lin, dt,
        )
        ee_rot = integrate_rotation(
            np.asarray(ee_pose.rotation_matrix, dtype=np.float64), ee_ang, dt,
        ) if not linear_only else np.asarray(ee_pose.rotation_matrix, dtype=np.float64)

        obj_pos = integrate_position(
            np.array([obj_pose.x, obj_pose.y, obj_pose.z], dtype=np.float64),
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
            duration=dt,
        ))

    return steps


def interpolate_plan(
    plan: list[HandControl], elapsed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Linearly interpolate pose/twist from a plan at a given elapsed time.

    Uses np.searchsorted for O(log n) interval lookup. Packs state into a
    single 12-element array for one vectorized lerp. Clamps to first/last
    step if elapsed is out of range.

    Returns:
        Tuple of (position, linear_vel, rotation_rpy, angular_vel).
    """
    steps = plan

    def _pack(s: HandControl) -> np.ndarray:
        return np.array([
            s.pose.x, s.pose.y, s.pose.z,
            s.twist.vx, s.twist.vy, s.twist.vz,
            s.pose.roll, s.pose.pitch, s.pose.yaw,
            s.twist.wx, s.twist.wy, s.twist.wz,
        ], dtype=np.float64)

    def _unpack(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return v[:3], v[3:6], v[6:9], v[9:12]

    times = np.cumsum([s.duration for s in steps])

    if elapsed <= times[0]:
        return _unpack(_pack(steps[0]))
    if elapsed >= times[-1]:
        return _unpack(_pack(steps[-1]))

    idx = int(np.clip(np.searchsorted(times, elapsed, side='right') - 1, 0, len(steps) - 2))
    t0, t1 = times[idx], times[idx + 1]
    alpha = (elapsed - t0) / (t1 - t0) if t1 > t0 else 0.0
    v0 = _pack(steps[idx])
    v1 = _pack(steps[idx + 1])
    return _unpack(v0 + alpha * (v1 - v0))
