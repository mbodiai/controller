from collections import deque
from dataclasses import dataclass, field

import numpy as np

from manifold.types.common.pose import Pose6D
from manifold.types.common.twist import Twist
from manifold.types.act.controller_config import TrajectoryControllerConfig
from manifold.types.act.control import HandControl
from control.utils import computeDeltaTwists, interpolate_plan


@dataclass
class TrajectoryController:
    """Wraps the stateless planning functions with state for drift correction,
    consumed-position tracking, and start-state selection."""

    config: TrajectoryControllerConfig
    max_linear_velocity: float = 0.5
    _last_plan: list[HandControl] | None = field(default=None, init=False)
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
        raise NotImplementedError

    def record_push_result(
        self, trajectory: list[HandControl], pushed_count: int,
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
        raise NotImplementedError

    def clear_last_pushed(self) -> None:
        """Clear last-pushed state when buffer drains completely."""
        raise NotImplementedError

    def compute_drift_correction(
        self, measured_pose: Pose6D, total_consumed: int,
    ) -> tuple[Twist | None, Twist]:
        """Compute velocity bias from consumed-position tracking error.

        Looks up the commanded position for the robot's current total_consumed
        counter, compares to measured_pose, and returns kp_drift * error as
        velocity_bias.

        Returns:
            Tuple of (velocity_bias, tracking_error) as Twist objects.
            velocity_bias is None when there is insufficient history.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def _lookup_consumed_position(self, consumed_count: int) -> np.ndarray | None:
        """Find the commanded position for a given consumed count.

        Evicts entries older than consumed_count. Returns the position for the
        matching or closest earlier entry, or None if no history.
        """
        raise NotImplementedError

    def compute_trajectory(
        self,
        ee_pose: Pose6D,
        ee_twist: Twist,
        obj_pose: Pose6D,
        obj_twist: Twist,
        now: float,
        velocity_bias: Twist | None = None,
        max_steps: int | None = None,
    ) -> list[HandControl]:
        """Plan a trajectory of timestamped EE waypoints to track an object.

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
        raise NotImplementedError
