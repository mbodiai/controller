from ee_trajectory_controller import Pose6D, Twist, project_pose

import numpy as np
from scipy.spatial.transform import Rotation

import pytest


class TestProjectObjectState:
    def test_zero_velocity(self):
        position = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        rotation = np.eye(3, dtype=np.float64)
        pose = Pose6D.from_position_and_rotation_matrix(
            position=position, rotation_matrix=rotation,
        )
        twist = Twist.from_linear_angular(
            linear=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            angular=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        )
        time_horizon = 0.1
        time_step = 0.1

        projected_pose = project_pose(pose, twist, time_horizon, time_step)

        expectedPosition = position

        assert np.allclose(np.asarray(projected_pose.position), expectedPosition, atol=1e-6)
        assert np.allclose(np.asarray(projected_pose.rotation_matrix), rotation, atol=1e-6)

    def test_constant_linear_velocity(self):
        position = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        rotation = np.eye(3, dtype=np.float64)
        pose = Pose6D.from_position_and_rotation_matrix(
            position=position, rotation_matrix=rotation,
        )
        twist = Twist.from_linear_angular(
            linear=np.array([1.0, 1.0, 1.0], dtype=np.float64),
            angular=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        )

        time_horizon = 0.1
        time_step = 0.1

        projected_pose = project_pose(pose, twist, time_horizon, time_step)

        expectedPosition = np.array([1.1, 2.1, 3.1], dtype=np.float64)

        assert np.allclose(np.asarray(projected_pose.position), expectedPosition, atol=1e-6)
        assert np.allclose(np.asarray(projected_pose.rotation_matrix), rotation, atol=1e-6)

    def test_constant_angular_velocity(self):
        position = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        rotation = np.eye(3, dtype=np.float64)
        pose = Pose6D.from_position_and_rotation_matrix(
            position=position, rotation_matrix=rotation,
        )
        twist = Twist.from_linear_angular(
            linear=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            angular=np.array([0.0, 0.0, 0.1], dtype=np.float64),
        )

        time_horizon = 0.2
        time_step = 0.1

        projected_pose = project_pose(pose, twist, time_horizon, time_step)

        expectedPosition = position
        expectedRotation = Rotation.from_rotvec([0.0, 0.0, 0.02]).as_matrix()

        assert np.allclose(np.asarray(projected_pose.position), expectedPosition, atol=1e-6)
        assert np.allclose(np.asarray(projected_pose.rotation_matrix), expectedRotation, atol=1e-6)
