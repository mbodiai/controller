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
    """Compute a single velocity correction step on raw numpy arrays.

    Applies the control law:
        linear_delta  = (v_obj + kp_position * (p_obj - p_ee)) + velocity_bias - v_ee
        angular_delta = (w_obj + kp_rotation * rotvec(R_ee^T @ R_obj)) - w_ee

    When params.linear_only is True, angular_delta is returned as zeros.

    Args:
        ee_pos: End effector position (3,).
        ee_rot: End effector rotation matrix (3, 3).
        ee_lin_vel: End effector linear velocity (3,).
        ee_ang_vel: End effector angular velocity (3,).
        obj_pos: Object position (3,).
        obj_rot: Object rotation matrix (3, 3).
        obj_lin_vel: Object linear velocity (3,).
        obj_ang_vel: Object angular velocity (3,).
        params: Controller configuration (gains, flags).
        velocity_bias: Optional additive bias on the desired linear velocity (3,).

    Returns:
        Tuple of (linear_delta, angular_delta), each shape (3,).
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    """Compute a single delta twist from manifold types.

    Convenience wrapper around _compute_single_delta_raw that unpacks
    Pose6D/Twist into numpy arrays and repacks the result.

    Args:
        eePose: End effector pose.
        eeTwist: End effector twist.
        objPose: Object pose.
        objTwist: Object twist.
        params: Controller configuration.

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
    raise NotImplementedError


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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def clear_last_pushed(self) -> None:
        """Clear last-pushed state when buffer drains completely."""
        raise NotImplementedError

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
        velocity_bias: np.ndarray | None = None,
        max_steps: int | None = None,
    ) -> Trajectory:
        """Plan a trajectory from the given start state.

        Delegates to computeDeltaTwists and stores the result as _last_plan
        for subsequent drift correction via compute_blended_start.

        Args:
            ee_pose: Planning start pose (from get_start_state).
            ee_twist: Planning start twist.
            obj_pose: Object pose (typically forward-projected).
            obj_twist: Object twist.
            now: Current wall-clock time (stored for plan interpolation).
            velocity_bias: Optional drift correction bias.
            max_steps: Optional cap on trajectory length.

        Returns:
            Planned trajectory.
        """
        raise NotImplementedError

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
        raise NotImplementedError


def _interpolate_plan(
    plan: Trajectory, elapsed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Linearly interpolate pose/twist from a trajectory at a given elapsed time.

    Used by compute_blended_start to predict where the robot should be
    based on the last plan. Clamps to first/last step if elapsed is out of range.

    Returns:
        Tuple of (position, linear_vel, rotation_rpy, angular_vel).
    """
    raise NotImplementedError
