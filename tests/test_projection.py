from ee_trajectory_controller import Pose, Twist, trajectory, trajectoryStep, controllerConfig

from ee_trajectory_controller import projectObjectState

import numpy as np

from numpy.typing import NDArray

from scipy.spatial.transform import Rotation

import pytest




class TestProjectObjectState:
    def test_zero_velocity(self):
        position = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        rotation = np.eye(3, dtype=np.float64)
        pose = Pose(position=position, rotation=rotation)
        twist = Twist(
            linearVelocity= np.array([0.0, 0.0, 0.0], dtype=np.float64),
            angularVelocity=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        )
        time_horizon = 0.1
        time_step = 0.1  # Time step for multi-step integration

        projected_pose = projectObjectState(pose, twist, time_horizon, time_step)

        expectedPosition = position

        assert np.allclose(projected_pose.position, expectedPosition, atol=1e-6)
        assert np.allclose(projected_pose.rotation, rotation, atol=1e-6)

    def test_constant_linear_velocity(self): 
        position = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        rotation = np.eye(3, dtype=np.float64)
        pose = Pose(position=position, rotation=rotation)
        twist = Twist(
            linearVelocity= np.array([1.0, 1.0, 1.0], dtype=np.float64),
            angularVelocity=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        )

        time_horizon = 0.1
        time_step = 0.1  # Time step for multi-step integration

        projected_pose = projectObjectState(pose, twist, time_horizon, time_step)

        expectedPosition = np.array([1.1, 2.1, 3.1], dtype=np.float64)

        assert np.allclose(projected_pose.position, expectedPosition, atol=1e-6)
        assert np.allclose(projected_pose.rotation, rotation, atol=1e-6)
        
    def test_constant_angular_velocity(self): 
        position = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        rotation = np.eye(3, dtype=np.float64)
        pose = Pose(position=position, rotation=rotation)
        twist = Twist(
            linearVelocity= np.array([0.0, 0.0, 0.0], dtype=np.float64),
            angularVelocity=np.array([0.0, 0.0, 0.1], dtype=np.float64),
        )

        time_horizon = 0.2
        time_step = 0.1  # Time step for multi-step integration

        projected_pose = projectObjectState(pose, twist, time_horizon, time_step)

        expectedPosition = position
        expectedRotation = Rotation.from_rotvec([0.0, 0.0, 0.02]).as_matrix()

        assert np.allclose(projected_pose.position, expectedPosition, atol=1e-6)
        assert np.allclose(projected_pose.rotation, expectedRotation, atol=1e-6)
        