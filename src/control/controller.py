from collections import deque
from dataclasses import dataclass, field

import numpy as np

from manifold.types.common.pose import Pose6D
from manifold.types.common.twist import Twist
from manifold.types.act.controller_config import TrajectoryControllerConfig
from manifold.types.act.control import HandControl
from manifold.types.common.list import List
from control.utils import computeDeltaTwists, interpolate_plan


@dataclass
class TrajectoryController:
    """Wraps the stateless planning functions with state for drift correction,
    consumed-position tracking, and start-state selection."""

    config: TrajectoryControllerConfig
    max_linear_velocity: float = 0.5
    _last_plan: List[HandControl] | None = field(default=None, init=False)
    _last_plan_time: float | None = field(default=None, init=False)
    _last_pushed_index: int = field(default=0, init=False)
    last_blend_elapsed: float = field(default=0.0, init=False)
    last_pos_drift_norm: float = field(default=0.0, init=False)
    last_rot_drift_norm: float = field(default=0.0, init=False)
    _last_pushed_pose: Pose6D | None = field(default=None, init=False)
    _last_pushed_twist: Twist | None = field(default=None, init=False)
    _consumed_positions: deque[tuple[int, np.ndarray]] = field(default_factory=lambda: deque(maxlen=200), init=False)
    _last_seen_consumed: int = field(default=0, init=False)

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
            or not self._last_plan
        ):
            return measured_pose, measured_twist

        dt = self.config.dt
        elapsed = now - self._last_plan_time

        pred_pos, pred_lv, pred_rpy, pred_av = interpolate_plan(self._last_plan, elapsed)

        measured_pos = np.asarray(measured_pose.position, dtype=np.float64)
        measured_rpy = np.array([measured_pose.roll, measured_pose.pitch, measured_pose.yaw], dtype=np.float64)
        measured_lv = np.asarray(measured_twist.linear, dtype=np.float64)
        measured_av = np.asarray(measured_twist.angular, dtype=np.float64)

        tail = self._last_plan[
            min(self._last_pushed_index, len(self._last_plan) - 1)
        ]
        tail_pos = np.asarray(tail.pose.position, dtype=np.float64)
        tail_lv = np.asarray(tail.twist.linear, dtype=np.float64)
        tail_rpy = np.array([tail.pose.roll, tail.pose.pitch, tail.pose.yaw], dtype=np.float64)
        tail_av = np.asarray(tail.twist.angular, dtype=np.float64)

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

    def record_push_result(
        self, trajectory: List[HandControl], pushed_count: int,
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
            last_step = trajectory[min(pushed_count, len(trajectory) - 1)]
            self._last_pushed_pose = last_step.pose
            self._last_pushed_twist = last_step.twist
        for i in range(pushed_count):
            step = trajectory[i + 1]
            pos = np.array([step.pose.x, step.pose.y, step.pose.z])
            self._consumed_positions.append((total_consumed + depth_before + i + 1, pos))

    def clear_last_pushed(self) -> None:
        """Clear last-pushed state when buffer drains completely."""
        self._last_pushed_pose = None
        self._last_pushed_twist = None

    def compute_drift_correction(
        self, measured_pose: Pose6D, total_consumed: int,
    ) -> tuple[Twist | None, Twist]:
        """Compute velocity bias from consumed-position tracking error.

        Looks up the commanded position for the robot's current total_consumed
        counter, compares to measured_pose, and returns kp_drift * error as
        velocity_bias.

        Returns:
            Tuple of (velocity_bias, tracking_error). velocity_bias is None
            when there is insufficient history.
        """
        zero = Twist.from_linear_angular(np.zeros(3), np.zeros(3))
        if not self._consumed_positions:
            return None, zero
        if total_consumed > self._last_seen_consumed:
            self._last_seen_consumed = total_consumed
        expected = self._lookup_consumed_position(total_consumed)
        if expected is None:
            return None, zero
        measured = np.array([measured_pose.x, measured_pose.y, measured_pose.z])
        error = expected - measured
        bias = self.config.kp_drift * error
        return (
            Twist.from_linear_angular(bias, np.zeros(3)),
            Twist.from_linear_angular(error, np.zeros(3)),
        )

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
        velocity_bias: Twist | None = None,
        max_steps: int | None = None,
    ) -> List[HandControl]:
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
            Planned trajectory as a list of HandControl waypoints.
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
