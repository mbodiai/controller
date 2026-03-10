import numpy as np

from manifold.types.common.pose import Pose6D
from manifold.types.common.twist import Twist
from manifold.types.act.trajectory import (
    TrajectoryControllerConfig,
    TrajectoryStep,
    Trajectory,
)
from manifold.types.act.control import HandControl
from manifold.utils.geometry import (
    rotvec_from_matrix,
    integrate_position,
    integrate_rotation,
)


# ────────────────────────────────────────────────────────────────
# Raw-array core (no Pose6D constructed mid-loop)
# ────────────────────────────────────────────────────────────────

def _compute_single_delta_raw(
    ee_pos: np.ndarray,
    ee_rot: np.ndarray,
    ee_lin_vel: np.ndarray,
    ee_ang_vel: np.ndarray,
    obj_pos: np.ndarray,
    obj_rot: np.ndarray,
    obj_lin_vel: np.ndarray,
    obj_ang_vel: np.ndarray,
    params: TrajectoryControllerConfig,
    velocity_bias: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute delta twist as raw arrays. Returns (linear_delta, angular_delta)."""
    linear_error = obj_pos - ee_pos
    desired_lin_vel = obj_lin_vel + linear_error * params.kp_position
    if velocity_bias is not None:
        desired_lin_vel = desired_lin_vel + velocity_bias
    lin_delta = desired_lin_vel - ee_lin_vel

    if params.linear_only:
        return lin_delta, np.zeros(3, dtype=np.float64)

    rot_error = rotvec_from_matrix(ee_rot.T @ obj_rot)
    desired_ang_vel = obj_ang_vel + rot_error * params.kp_rotation
    ang_delta = desired_ang_vel - ee_ang_vel

    return lin_delta, ang_delta


def _project_state(
    pos: np.ndarray,
    rot: np.ndarray,
    lin_vel: np.ndarray,
    ang_vel: np.ndarray,
    horizon: float,
    dt: float,
    linear_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-project position and rotation using Euler integration on raw arrays."""
    n = max(1, int(horizon / dt))
    for _ in range(n):
        pos = integrate_position(pos, lin_vel, dt)
        if not linear_only:
            rot = integrate_rotation(rot, ang_vel, dt)
    return pos, rot


# ────────────────────────────────────────────────────────────────
# Public API (accepts / returns manifold types)
# ────────────────────────────────────────────────────────────────

def computeSingleDeltaTwist(
    eePose: Pose6D,
    eeTwist: Twist,
    objPose: Pose6D,
    objTwist: Twist,
    params: TrajectoryControllerConfig,
) -> Twist:
    lin_delta, ang_delta = _compute_single_delta_raw(
        np.asarray(eePose.position, dtype=np.float64),
        np.asarray(eePose.rotation_matrix, dtype=np.float64),
        np.asarray(eeTwist.linear, dtype=np.float64),
        np.asarray(eeTwist.angular, dtype=np.float64),
        np.asarray(objPose.position, dtype=np.float64),
        np.asarray(objPose.rotation_matrix, dtype=np.float64),
        np.asarray(objTwist.linear, dtype=np.float64),
        np.asarray(objTwist.angular, dtype=np.float64),
        params,
    )
    return Twist.from_linear_angular(lin_delta, ang_delta)


def computeDeltaTwists(
    eePose: Pose6D,
    eeTwist: Twist,
    objPose: Pose6D,
    objTwist: Twist,
    params: TrajectoryControllerConfig,
    max_linear_velocity: float = float("inf"),
    velocity_bias: np.ndarray | None = None,
    max_steps: int | None = None,
) -> Trajectory:

    dt = params.dt
    linear_only = params.linear_only
    nSteps = max(3, int(params.latency / dt) + 1)
    if max_steps is not None:
        nSteps = max(3, min(nSteps, max_steps))

    # Extract raw arrays from inputs
    obj_pos = np.array(objPose.position, dtype=np.float64)
    obj_rot = np.array(objPose.rotation_matrix, dtype=np.float64)
    obj_lin = np.array(objTwist.linear, dtype=np.float64)
    obj_ang = np.array(objTwist.angular, dtype=np.float64)

    ee_pos = np.array(eePose.position, dtype=np.float64)
    ee_rot = np.array(eePose.rotation_matrix, dtype=np.float64)
    ee_lin = np.array(eeTwist.linear, dtype=np.float64)
    ee_ang = np.array(eeTwist.angular, dtype=np.float64)

    steps: list[TrajectoryStep] = [TrajectoryStep(
        time=0.0,
        ee_pose=Pose6D.from_position_and_rotation_matrix(ee_pos, ee_rot),
        ee_twist=Twist.from_linear_angular(ee_lin, ee_ang),
        object_pose=Pose6D.from_position_and_rotation_matrix(obj_pos, obj_rot),
        object_twist=Twist.from_linear_angular(obj_lin, obj_ang),
        delta_twist=Twist.zero(),
    )]

    for step in range(1, nSteps):
        # Delta twist
        lin_delta, ang_delta = _compute_single_delta_raw(
            ee_pos, ee_rot, ee_lin, ee_ang,
            obj_pos, obj_rot, obj_lin, obj_ang,
            params, velocity_bias=velocity_bias,
        )

        # Update EE velocity and clamp to max
        ee_lin = ee_lin + lin_delta
        ee_ang = ee_ang + ang_delta
        speed = np.linalg.norm(ee_lin)
        if speed > max_linear_velocity:
            ee_lin = ee_lin * (max_linear_velocity / speed)

        # Integrate EE state forward
        ee_pos, ee_rot = _project_state(
            ee_pos, ee_rot, ee_lin, ee_ang, dt, dt, linear_only,
        )

        # Integrate object state forward
        obj_pos, obj_rot = _project_state(
            obj_pos, obj_rot, obj_lin, obj_ang, dt, dt, linear_only,
        )

        # Pack into manifold types only at the output boundary
        steps.append(TrajectoryStep(
            time=float(step) * dt,
            ee_pose=Pose6D.from_position_and_rotation_matrix(ee_pos, ee_rot),
            ee_twist=Twist.from_linear_angular(ee_lin, ee_ang),
            object_pose=Pose6D.from_position_and_rotation_matrix(obj_pos, obj_rot),
            object_twist=Twist.from_linear_angular(obj_lin, obj_ang),
            delta_twist=Twist.from_linear_angular(lin_delta, ang_delta),
        ))

    return Trajectory(steps=steps)


# ────────────────────────────────────────────────────────────────
# Stateful controller (drift correction + planning)
# ────────────────────────────────────────────────────────────────

class TrajectoryController:
    """Wraps the stateless planning functions with plan history for drift correction."""

    def __init__(self, config: TrajectoryControllerConfig, max_linear_velocity: float = 0.5) -> None:
        self.config = config
        self.max_linear_velocity = max_linear_velocity
        self._last_plan: Trajectory | None = None
        self._last_plan_time: float | None = None
        self._last_pushed_index: int = 0
        self.last_blend_elapsed: float = 0.0
        self.last_pos_drift_norm: float = 0.0
        self.last_rot_drift_norm: float = 0.0

    def compute_blended_start(
        self,
        measured_pose: Pose6D,
        measured_twist: Twist,
        now: float,
        last_pushed_count: int,
    ) -> tuple[Pose6D, Twist]:
        """Drift-correct using internal plan history. Returns corrected (pose, twist).

        Compares where the robot actually is (measured) against where the last
        plan predicted it would be. Blends that error into the tail of what was
        actually sent. If there is no previous plan, returns measured state unchanged.
        """
        self._last_pushed_index = max(last_pushed_count - 1, 0)

        if (
            self._last_plan is None
            or self._last_plan_time is None
            or not self._last_plan.steps
        ):
            return measured_pose, measured_twist

        dt = self.config.dt
        elapsed = now - self._last_plan_time

        pred_pos, pred_lv, pred_rpy, pred_av = _interpolate_plan(self._last_plan, elapsed)

        measured_pos = np.asarray(measured_pose.position, dtype=np.float64)
        measured_rpy = np.array([measured_pose.roll, measured_pose.pitch, measured_pose.yaw], dtype=np.float64)
        measured_lv = np.asarray(measured_twist.linear, dtype=np.float64)
        measured_av = np.asarray(measured_twist.angular, dtype=np.float64)

        tail = self._last_plan.steps[
            min(self._last_pushed_index, len(self._last_plan.steps) - 1)
        ]
        tail_pos = np.asarray(tail.ee_pose.position, dtype=np.float64)
        tail_lv = np.asarray(tail.ee_twist.linear, dtype=np.float64)
        tail_rpy = np.array([tail.ee_pose.roll, tail.ee_pose.pitch, tail.ee_pose.yaw], dtype=np.float64)
        tail_av = np.asarray(tail.ee_twist.angular, dtype=np.float64)

        pos_drift = measured_pos - pred_pos
        rot_drift = measured_rpy - pred_rpy

        self.last_blend_elapsed = elapsed
        self.last_pos_drift_norm = float(np.linalg.norm(pos_drift))
        self.last_rot_drift_norm = float(np.linalg.norm(rot_drift))

        alpha = self.config.drift_alpha
        beta = self.config.drift_beta
        inv_dt = 1.0 / max(dt, 0.001)

        blended_pos = tail_pos + alpha * pos_drift
        blended_lv = tail_lv + beta * (pos_drift * inv_dt)
        blended_rpy = tail_rpy + alpha * rot_drift
        blended_av = tail_av + beta * (rot_drift * inv_dt)

        blended_pose = Pose6D(
            x=float(blended_pos[0]),
            y=float(blended_pos[1]),
            z=float(blended_pos[2]),
            roll=float(blended_rpy[0]),
            pitch=float(blended_rpy[1]),
            yaw=float(blended_rpy[2]),
        )
        blended_twist = Twist(
            vx=float(blended_lv[0]),
            vy=float(blended_lv[1]),
            vz=float(blended_lv[2]),
            wx=float(blended_av[0]),
            wy=float(blended_av[1]),
            wz=float(blended_av[2]),
        )

        return blended_pose, blended_twist

    def compute_trajectory(
        self,
        ee_pose: Pose6D,
        ee_twist: Twist,
        obj_pose: Pose6D,
        obj_twist: Twist,
        now: float,
        velocity_bias: np.ndarray | None = None,
        max_steps: int | None = None,
    ) -> Trajectory:
        """Plan a trajectory from the given start state. Stores result for next drift correction."""
        trajectory = computeDeltaTwists(
            ee_pose, ee_twist, obj_pose, obj_twist, self.config,
            max_linear_velocity=self.max_linear_velocity,
            velocity_bias=velocity_bias,
            max_steps=max_steps,
        )

        self._last_plan = trajectory
        self._last_plan_time = now

        return trajectory

    def trajectory_to_hand_controls(
        self,
        trajectory: Trajectory,
        current_orientation: tuple[float, float, float],
        grasp: float = 0.0,
    ) -> list[HandControl]:
        """Convert trajectory steps to HandControls (speed-based or time-based per config.speed_mode)."""
        roll, pitch, yaw = current_orientation
        results = []

        for i, step in enumerate(trajectory.steps[1:], start=1):
            pose = Pose6D(x=step.ee_pose.x, y=step.ee_pose.y, z=step.ee_pose.z, roll=roll, pitch=pitch, yaw=yaw)
            if self.config.speed_mode:
                linear_speed = float(np.linalg.norm([step.ee_twist.vx, step.ee_twist.vy, step.ee_twist.vz]))
                results.append(HandControl(pose=pose, grasp=grasp, speed=max(linear_speed, 0.05)))
            else:
                dt = step.time - trajectory.steps[i - 1].time if i > 0 else step.time
                results.append(HandControl(pose=pose, grasp=grasp, time=dt))

        return results


def _interpolate_plan(
    plan: Trajectory, elapsed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Linearly interpolate pose/twist from a trajectory at a given elapsed time.

    Returns (position, linear_vel, rotation_rpy, angular_vel).
    """
    steps = plan.steps

    def _extract(s: TrajectoryStep) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(s.ee_pose.position, dtype=np.float64),
            np.asarray(s.ee_twist.linear, dtype=np.float64),
            np.array([s.ee_pose.roll, s.ee_pose.pitch, s.ee_pose.yaw], dtype=np.float64),
            np.asarray(s.ee_twist.angular, dtype=np.float64),
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
