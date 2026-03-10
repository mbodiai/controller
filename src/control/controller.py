from collections import deque

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


def _project_state(
    pos: np.ndarray,
    rot: np.ndarray,
    lin_vel: np.ndarray,
    ang_vel: np.ndarray,
    horizon: float,
    dt: float,
    linear_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-project position and rotation by integrating constant velocity.

    Performs n = max(1, int(horizon / dt)) Euler integration steps.
    Position is updated via integrate_position; rotation via integrate_rotation
    (skipped when linear_only is True).

    Args:
        pos: Starting position (3,).
        rot: Starting rotation matrix (3, 3).
        lin_vel: Linear velocity (3,).
        ang_vel: Angular velocity (3,).
        horizon: Total time to integrate over (seconds).
        dt: Integration timestep (seconds).
        linear_only: If True, skip rotation integration.

    Returns:
        Tuple of (projected_position, projected_rotation).
    """
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
    velocity_bias: np.ndarray | None = None,
) -> Twist:
    """Compute a single velocity correction step (proportional + feedforward).

    Applies the control law:
        linear_delta  = (v_obj + kp_position * (p_obj - p_ee)) + velocity_bias - v_ee
        angular_delta = (w_obj + kp_rotation * rotvec(R_ee^T @ R_obj)) - w_ee

    When params.linear_only is True, angular_delta is zero.

    Args:
        eePose: End effector pose.
        eeTwist: End effector twist.
        objPose: Object pose.
        objTwist: Object twist.
        params: Controller configuration (gains, flags).
        velocity_bias: Optional additive bias on the desired linear velocity (3,).

    Returns:
        Delta twist to apply to the end effector.
    """
    linear_error = objPose.position - eePose.position
    desired_lin_vel = objTwist.linear + linear_error * params.kp_position
    if velocity_bias is not None:
        desired_lin_vel = desired_lin_vel + velocity_bias
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
    velocity_bias: np.ndarray | None = None,
    max_steps: int | None = None,
) -> Trajectory:
    """Plan a multi-step trajectory using proportional + feedforward control.

    At each step: compute delta twist, update EE velocity (clamped to
    max_linear_velocity), integrate both EE and object states forward by dt.
    Number of steps is max(3, int(latency / dt) + 1), optionally capped by max_steps.

    Args:
        eePose: Initial end effector pose.
        eeTwist: Initial end effector twist.
        objPose: Initial object pose.
        objTwist: Object twist (assumed constant).
        params: Controller configuration.
        max_linear_velocity: Speed clamp for EE linear velocity.
        velocity_bias: Optional additive velocity bias applied at every step.
        max_steps: Optional upper bound on trajectory length.

    Returns:
        Trajectory with step 0 = initial state and subsequent planned steps.
    """
    dt = params.dt
    linear_only = params.linear_only
    nSteps = max(3, int(params.latency / dt) + 1)
    if max_steps is not None:
        nSteps = max(3, min(nSteps, max_steps))

    ee_pose = eePose
    ee_twist = eeTwist
    obj_pose = objPose
    obj_twist = objTwist

    steps: list[TrajectoryStep] = [TrajectoryStep(
        time=0.0,
        ee_pose=ee_pose,
        ee_twist=ee_twist,
        object_pose=obj_pose,
        object_twist=obj_twist,
        delta_twist=Twist.zero(),
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

        ee_pos, ee_rot = _project_state(
            np.asarray(ee_pose.position, dtype=np.float64),
            np.asarray(ee_pose.rotation_matrix, dtype=np.float64),
            ee_lin, ee_ang, dt, dt, linear_only,
        )
        obj_pos, obj_rot = _project_state(
            np.asarray(obj_pose.position, dtype=np.float64),
            np.asarray(obj_pose.rotation_matrix, dtype=np.float64),
            np.asarray(obj_twist.linear, dtype=np.float64),
            np.asarray(obj_twist.angular, dtype=np.float64),
            dt, dt, linear_only,
        )

        ee_pose = Pose6D.from_position_and_rotation_matrix(ee_pos, ee_rot)
        ee_twist = Twist.from_linear_angular(ee_lin, ee_ang)
        obj_pose = Pose6D.from_position_and_rotation_matrix(obj_pos, obj_rot)

        steps.append(TrajectoryStep(
            time=float(step) * dt,
            ee_pose=ee_pose,
            ee_twist=ee_twist,
            object_pose=obj_pose,
            object_twist=obj_twist,
            delta_twist=delta,
        ))

    return Trajectory(steps=steps)


# ────────────────────────────────────────────────────────────────
# Stateful controller (drift correction + planning)
# ────────────────────────────────────────────────────────────────

class TrajectoryController:
    """Wraps the stateless planning functions with state for drift correction,
    consumed-position tracking, and start-state selection."""

    def __init__(self, config: TrajectoryControllerConfig, max_linear_velocity: float = 0.5) -> None:
        self.config = config
        self.max_linear_velocity = max_linear_velocity
        self._last_plan: Trajectory | None = None
        self._last_plan_time: float | None = None
        self._last_pushed_index: int = 0
        self.last_blend_elapsed: float = 0.0
        self.last_pos_drift_norm: float = 0.0
        self.last_rot_drift_norm: float = 0.0
        self._last_pushed_pose: Pose6D | None = None
        self._last_pushed_twist: Twist | None = None
        self._consumed_positions: deque[tuple[int, np.ndarray]] = deque(maxlen=200)
        self._last_seen_consumed: int = 0

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

        Uses drift_alpha (position blend) and drift_beta (velocity blend from
        position error / dt) from config. Note: rotation blending uses RPY
        subtraction, which has gimbal lock issues near pitch=±90°.
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

    def project_object(self, obj_pose: Pose6D, obj_twist: Twist, buffer_depth: int) -> Pose6D:
        """Forward-project object pose to compensate for buffer delay.

        Computes delay = buffer_depth * dt and advances position by twist * delay.
        Rotation is left unchanged (linear-only projection).

        Args:
            obj_pose: Current object pose.
            obj_twist: Current object twist.
            buffer_depth: Number of waypoints currently queued in the robot buffer.

        Returns:
            Projected object pose.
        """
        delay = buffer_depth * self.config.dt
        return Pose6D(
            x=obj_pose.x + obj_twist.vx * delay,
            y=obj_pose.y + obj_twist.vy * delay,
            z=obj_pose.z + obj_twist.vz * delay,
            roll=obj_pose.roll, pitch=obj_pose.pitch, yaw=obj_pose.yaw,
        )

    def record_push_result(
        self, trajectory: Trajectory, pushed_count: int,
        depth_before: int, total_consumed: int,
    ) -> None:
        """Record pushed waypoints for consumed-position drift tracking.

        Updates _last_pushed_pose/twist from the last pushed step, and appends
        each pushed position to _consumed_positions with its predicted consume
        counter value (total_consumed + depth_before + i + 1).

        Args:
            trajectory: The planned trajectory that waypoints were taken from.
            pushed_count: How many waypoints were successfully pushed.
            depth_before: Buffer depth before this push.
            total_consumed: Robot's total_consumed counter at time of push.
        """
        if pushed_count > 0:
            last_step = trajectory.steps[min(pushed_count, len(trajectory.steps) - 1)]
            self._last_pushed_pose = last_step.ee_pose
            self._last_pushed_twist = last_step.ee_twist
        for i in range(pushed_count):
            step = trajectory.steps[i + 1]
            pos = np.array([step.ee_pose.x, step.ee_pose.y, step.ee_pose.z])
            self._consumed_positions.append((total_consumed + depth_before + i + 1, pos))

    def clear_last_pushed(self) -> None:
        """Clear last-pushed state when buffer drains completely."""
        self._last_pushed_pose = None
        self._last_pushed_twist = None

    def compute_drift_correction(
        self, measured_pose: Pose6D, total_consumed: int,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Compute velocity bias from consumed-position tracking error.

        Looks up the commanded position for the robot's current total_consumed
        counter, compares to measured_pose, and returns kp_drift * error as
        velocity_bias.

        Returns:
            Tuple of (velocity_bias, tracking_error). velocity_bias is None
            when there is insufficient history.
        """
        tracking_error = np.zeros(3, dtype=np.float64)
        if not self._consumed_positions:
            return None, tracking_error
        if total_consumed > self._last_seen_consumed:
            self._last_seen_consumed = total_consumed
        expected = self._lookup_consumed_position(total_consumed)
        if expected is None:
            return None, tracking_error
        measured = np.array([measured_pose.x, measured_pose.y, measured_pose.z])
        tracking_error = expected - measured
        return self.config.kp_drift * tracking_error, tracking_error

    def get_start_state(
        self, measured_pose: Pose6D, measured_twist: Twist,
        now: float, last_pushed_count: int, disable_blend: bool = False,
    ) -> tuple[Pose6D, Twist]:
        """Determine planning start state: last-pushed tail, blended, or measured.

        Priority: (1) last pushed pose/twist if available, (2) blended start
        via compute_blended_start unless disable_blend, (3) raw measured state.

        Args:
            measured_pose: Current measured robot pose.
            measured_twist: Current measured robot twist.
            now: Current wall-clock time.
            last_pushed_count: Number of waypoints pushed in the last cycle.
            disable_blend: If True, skip blending and use measured state directly.

        Returns:
            Tuple of (start_pose, start_twist) for trajectory planning.
        """
        if self._last_pushed_pose is not None:
            return self._last_pushed_pose, self._last_pushed_twist
        if not disable_blend:
            return self.compute_blended_start(measured_pose, measured_twist, now, last_pushed_count)
        return measured_pose, measured_twist

    def _lookup_consumed_position(self, consumed_count: int) -> np.ndarray | None:
        """Find the commanded position for a given consumed count.

        Evicts entries older than consumed_count. Returns the position for the
        matching or closest earlier entry, or None if no history.
        """
        while len(self._consumed_positions) > 1 and self._consumed_positions[0][0] < consumed_count:
            self._consumed_positions.popleft()
        if not self._consumed_positions:
            return None
        best = None
        for count, pos in self._consumed_positions:
            if count <= consumed_count:
                best = pos
            elif count > consumed_count:
                break
        return best.copy() if best is not None else None

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
        """ Plan a trajectory of timestamped EE waypoints to track an object

        Delegates to computeDeltaTwists. Also stores the result internally
        for compute_blended_start (starvation recovery fallback; not used
        during normal operation where last-pushed state provides continuity).

        Args:
            ee_pose: Planning start pose (from get_start_state).
            ee_twist: Planning start twist.
            obj_pose: Object pose (typically forward-projected).
            obj_twist: Object twist.
            now: Current wall-clock time.
            velocity_bias: Optional drift correction bias.
            max_steps: Optional cap on trajectory length.

        Returns:
            Planned trajectory.
        """
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
        """Convert trajectory steps to HandControls for the robot streaming buffer.

        Uses config.speed_mode to choose between speed-based (HandControl.speed)
        and time-based (HandControl.time) waypoints. Orientation is held constant
        at current_orientation for all waypoints.

        Args:
            trajectory: Planned trajectory (steps[1:] are converted).
            current_orientation: (roll, pitch, yaw) to hold for all waypoints.
            grasp: Gripper value for all waypoints.

        Returns:
            List of HandControl waypoints.
        """
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

    Used by compute_blended_start to predict where the robot should be
    based on the last plan. Clamps to first/last step if elapsed is out of range.

    Returns:
        Tuple of (position, linear_vel, rotation_rpy, angular_vel).
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
