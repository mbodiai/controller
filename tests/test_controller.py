import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from manifold.types.common.pose import Pose6D
from manifold.types.common.twist import Twist
from manifold.types.act.controller_config import TrajectoryControllerConfig
from manifold.types.act.control import HandControl
from control.controller import TrajectoryController


def _make_config(**overrides) -> TrajectoryControllerConfig:
    defaults = dict(kp_position=2.0, kp_rotation=1.0, latency=5.0, simulation_horizon=5.0, dt=0.1)
    defaults.update(overrides)
    return TrajectoryControllerConfig(**defaults)


def _zero_pose() -> Pose6D:
    return Pose6D.from_position_and_rotation_matrix(np.zeros(3), np.eye(3))


def _zero_twist() -> Twist:
    return Twist.from_linear_angular(np.zeros(3), np.zeros(3))


class TestTrajectoryController:

    def test_stationary_target_convergence(self):
        """Position error should decrease when tracking a stationary target."""
        tc = TrajectoryController(config=_make_config(kp_position=5.0))
        obj_pose = Pose6D.from_position_and_rotation_matrix(np.array([1.0, 0.0, 0.0]), np.eye(3))
        obj_twist = _zero_twist()

        ee_pose = _zero_pose()
        ee_twist = _zero_twist()
        errors = []

        for step in range(100):
            now = step * tc.config.dt
            start_pose, start_twist = tc.get_start_state(ee_pose, ee_twist, now, 0)
            traj = tc.compute_trajectory(start_pose, start_twist, obj_pose, obj_twist, now)
            tc.record_push_result(traj, min(5, len(traj) - 1), 0, step * 5)

            ee_pose = traj[1].pose
            ee_twist = traj[1].twist
            errors.append(np.linalg.norm(np.asarray(ee_pose.position) - np.asarray(obj_pose.position)))

        assert errors[-1] < errors[0], "Position error should decrease over time"
        first_quarter = np.mean(errors[:25])
        last_quarter = np.mean(errors[-25:])
        assert last_quarter < first_quarter, "Average error in last quarter should be less than first quarter"

    def test_constant_velocity_tracking(self):
        """EE should follow a moving target — error should stabilize, not diverge."""
        tc = TrajectoryController(config=_make_config())
        obj_velocity = np.array([1.0, 0.0, 0.0])
        obj_twist = Twist.from_linear_angular(obj_velocity, np.zeros(3))

        ee_pose = _zero_pose()
        ee_twist = _zero_twist()
        position_errors = []

        for step in range(200):
            now = step * tc.config.dt
            obj_pos = obj_velocity * now
            obj_pose = Pose6D.from_position_and_rotation_matrix(obj_pos, np.eye(3))

            start_pose, start_twist = tc.get_start_state(ee_pose, ee_twist, now, 0)
            traj = tc.compute_trajectory(start_pose, start_twist, obj_pose, obj_twist, now)
            tc.record_push_result(traj, min(5, len(traj) - 1), 0, step * 5)

            ee_pose = traj[1].pose
            ee_twist = traj[1].twist
            position_errors.append(np.linalg.norm(np.asarray(ee_pose.position) - obj_pos))

        avg_last_20 = np.mean(position_errors[-20:])
        peak_error = max(position_errors)
        assert avg_last_20 < peak_error, "Steady-state error should be below the peak transient"
        # Error should not grow unboundedly — the EE should keep up
        assert avg_last_20 < 2.0, f"Steady-state error {avg_last_20:.4f} should remain bounded"

    def test_rotation_convergence(self):
        """Rotation error should decrease when tracking a rotated stationary target."""
        tc = TrajectoryController(config=_make_config(linear_only=False))
        target_rot = Rotation.from_rotvec([0.0, 0.0, np.pi / 2]).as_matrix()
        obj_pose = Pose6D.from_position_and_rotation_matrix(np.zeros(3), target_rot)
        obj_twist = _zero_twist()

        ee_pose = _zero_pose()
        ee_twist = _zero_twist()
        rot_errors = []

        for step in range(100):
            now = step * tc.config.dt
            start_pose, start_twist = tc.get_start_state(ee_pose, ee_twist, now, 0)
            traj = tc.compute_trajectory(start_pose, start_twist, obj_pose, obj_twist, now)
            tc.record_push_result(traj, min(5, len(traj) - 1), 0, step * 5)

            ee_pose = traj[1].pose
            ee_twist = traj[1].twist
            rel_rot = np.asarray(ee_pose.rotation_matrix).T @ target_rot
            rot_errors.append(np.linalg.norm(Rotation.from_matrix(rel_rot).as_rotvec()))

        assert rot_errors[-1] < rot_errors[0], "Rotation error should decrease over time"
        first_quarter = np.mean(rot_errors[:25])
        last_quarter = np.mean(rot_errors[-25:])
        assert last_quarter < first_quarter, "Average rotation error should decrease over time"

    def test_drift_correction_reduces_error(self):
        """With systematic bias in measurements, drift correction should help."""
        obj_twist = _zero_twist()
        obj_pose = Pose6D.from_position_and_rotation_matrix(np.array([1.0, 0.0, 0.0]), np.eye(3))
        n_steps = 60
        systematic_bias = np.array([0.02, 0.0, 0.0])

        def _run(kp_drift: float) -> list[float]:
            tc = TrajectoryController(config=_make_config(kp_drift=kp_drift))
            ee_pose = _zero_pose()
            ee_twist = _zero_twist()
            errors = []
            total_consumed = 0

            for step in range(n_steps):
                now = step * tc.config.dt

                measured_pos = np.asarray(ee_pose.position) - systematic_bias
                measured_pose = Pose6D.from_position_and_rotation_matrix(
                    measured_pos, np.asarray(ee_pose.rotation_matrix),
                )

                bias, _ = tc.compute_drift_correction(measured_pose, total_consumed)
                start_pose, start_twist = tc.get_start_state(measured_pose, ee_twist, now, 0)
                traj = tc.compute_trajectory(
                    start_pose, start_twist, obj_pose, obj_twist, now, velocity_bias=bias,
                )

                pushed = min(5, len(traj) - 1)
                tc.record_push_result(traj, pushed, 0, total_consumed)
                total_consumed += pushed

                ee_pose = traj[1].pose
                ee_twist = traj[1].twist
                errors.append(np.linalg.norm(measured_pos - np.asarray(obj_pose.position)))

            return errors

        errors_with = _run(kp_drift=2.0)
        errors_none = _run(kp_drift=0.0)

        assert errors_with[-1] < errors_with[0], "Error with drift correction should decrease"
        assert errors_none[-1] < errors_none[0], "Error without drift correction should decrease"
