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


EPS = 1e-10


class TestTrajectoryController:

    def test_stationary_target_convergence(self):
        """Position error should strictly decrease every step for a stationary target."""
        tc = TrajectoryController(config=_make_config())
        obj_pose = Pose6D.from_position_and_rotation_matrix(np.array([1.0, 0.0, 0.0]), np.eye(3))
        obj_twist = _zero_twist()

        ee_pose = _zero_pose()
        ee_twist = _zero_twist()
        errors = []

        for step in range(50):
            now = step * tc.config.dt
            start_pose, start_twist = tc.get_start_state(ee_pose, ee_twist, now, 0)
            traj = tc.compute_trajectory(start_pose, start_twist, obj_pose, obj_twist, now)
            tc.record_push_result(traj, 1, 0, step)

            ee_pose = traj[1].pose
            ee_twist = traj[1].twist
            errors.append(np.linalg.norm(np.asarray(ee_pose.position) - np.asarray(obj_pose.position)))

        for i in range(len(errors) - 1):
            assert errors[i + 1] <= errors[i] + EPS, f"Error must not increase: step {i} ({errors[i]:.6f}) -> step {i+1} ({errors[i+1]:.6f})"
        assert errors[-1] < errors[0], "Error must converge"

    def test_constant_velocity_tracking(self):
        """EE should follow a moving target — error should stabilize, not diverge."""
        tc = TrajectoryController(config=_make_config())
        obj_velocity = np.array([0.3, 0.0, 0.0])
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
            tc.record_push_result(traj, 1, 0, step)

            ee_pose = traj[1].pose
            ee_twist = traj[1].twist
            obj_pos_next = obj_velocity * (now + tc.config.dt)
            position_errors.append(np.linalg.norm(np.asarray(ee_pose.position) - obj_pos_next))

        peak_idx = int(np.argmax(position_errors))
        for i in range(peak_idx, len(position_errors) - 1):
            assert position_errors[i + 1] <= position_errors[i] + EPS, (
                f"Error must not increase after peak: step {i} ({position_errors[i]:.6f}) -> step {i+1} ({position_errors[i+1]:.6f})"
            )
        assert position_errors[-1] < position_errors[peak_idx], "Error must converge after peak"

    def test_rotation_convergence(self):
        """Rotation error should strictly decrease every step for a rotated stationary target."""
        tc = TrajectoryController(config=_make_config(linear_only=False))
        target_rot = Rotation.from_rotvec([0.0, 0.0, np.pi / 2]).as_matrix()
        obj_pose = Pose6D.from_position_and_rotation_matrix(np.zeros(3), target_rot)
        obj_twist = _zero_twist()

        ee_pose = _zero_pose()
        ee_twist = _zero_twist()
        rot_errors = []

        for step in range(50):
            now = step * tc.config.dt
            start_pose, start_twist = tc.get_start_state(ee_pose, ee_twist, now, 0)
            traj = tc.compute_trajectory(start_pose, start_twist, obj_pose, obj_twist, now)
            tc.record_push_result(traj, 1, 0, step)

            ee_pose = traj[1].pose
            ee_twist = traj[1].twist
            rel_rot = np.asarray(ee_pose.rotation_matrix).T @ target_rot
            rot_errors.append(np.linalg.norm(Rotation.from_matrix(rel_rot).as_rotvec()))

        for i in range(len(rot_errors) - 1):
            assert rot_errors[i + 1] <= rot_errors[i] + EPS, f"Rotation error must not increase: step {i} ({rot_errors[i]:.6f}) -> step {i+1} ({rot_errors[i+1]:.6f})"
        assert rot_errors[-1] < rot_errors[0], "Rotation error must converge"

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

                tc.record_push_result(traj, 1, 0, total_consumed)
                total_consumed += 1

                ee_pose = traj[1].pose
                ee_twist = traj[1].twist
                errors.append(np.linalg.norm(measured_pos - np.asarray(obj_pose.position)))

            return errors

        errors_with = _run(kp_drift=2.0)
        errors_none = _run(kp_drift=0.0)

        for errors, label in [(errors_with, "with drift"), (errors_none, "without drift")]:
            for i in range(len(errors) - 1):
                assert errors[i + 1] <= errors[i] + EPS, (
                    f"Error must not increase ({label}): step {i} ({errors[i]:.6f}) -> step {i+1} ({errors[i+1]:.6f})"
                )
            assert errors[-1] < errors[0], f"Error must converge ({label})"
        assert errors_with[-1] < errors_none[-1], "Drift correction should reduce final error"

    def test_blended_start_corrects_drift(self):
        """Blended start should move toward measured pose by drift_alpha."""
        alpha = 0.5
        tc = TrajectoryController(config=_make_config(drift_alpha=alpha, drift_beta=0.0))
        obj_pose = Pose6D.from_position_and_rotation_matrix(np.array([1.0, 0.0, 0.0]), np.eye(3))
        obj_twist = _zero_twist()

        traj = tc.compute_trajectory(_zero_pose(), _zero_twist(), obj_pose, obj_twist, 0.0)
        tc.record_push_result(traj, 1, 0, 0)
        tc.clear_last_pushed()

        offset = np.array([0.1, 0.0, 0.0])
        measured_pose = Pose6D.from_position_and_rotation_matrix(
            np.asarray(traj[1].pose.position) + offset, np.eye(3),
        )
        blended_pose, _ = tc.get_start_state(measured_pose, _zero_twist(), tc.config.dt, 0)

        tail_x = traj[0].pose.x
        blended_x = blended_pose.x
        assert abs(blended_x - tail_x) > 0.01, "Blend should shift away from tail toward measurement"
        assert blended_x != measured_pose.x, "Blend should not equal raw measurement"

    def test_get_start_state_priority(self):
        """get_start_state should follow priority: last-pushed > blended > measured."""
        tc = TrajectoryController(config=_make_config())
        measured_pose = Pose6D.from_position_and_rotation_matrix(np.array([0.5, 0.0, 0.0]), np.eye(3))
        measured_twist = _zero_twist()
        obj_pose = Pose6D.from_position_and_rotation_matrix(np.array([1.0, 0.0, 0.0]), np.eye(3))

        # No plan yet — should return measured
        pose, _ = tc.get_start_state(measured_pose, measured_twist, 0.0, 0)
        assert np.allclose(np.asarray(pose.position), np.asarray(measured_pose.position))

        # After planning + push — should return last-pushed
        traj = tc.compute_trajectory(_zero_pose(), _zero_twist(), obj_pose, _zero_twist(), 0.0)
        tc.record_push_result(traj, 1, 0, 0)
        pose, _ = tc.get_start_state(measured_pose, measured_twist, tc.config.dt, 0)
        assert np.allclose(np.asarray(pose.position), np.asarray(traj[1].pose.position))

        # After clear — should blend (not equal to measured)
        tc.clear_last_pushed()
        pose, _ = tc.get_start_state(measured_pose, measured_twist, tc.config.dt, 0)
        assert not np.allclose(np.asarray(pose.position), np.asarray(measured_pose.position))

        # With disable_blend — should return measured
        pose, _ = tc.get_start_state(measured_pose, measured_twist, tc.config.dt, 0, disable_blend=True)
        assert np.allclose(np.asarray(pose.position), np.asarray(measured_pose.position))
