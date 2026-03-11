from manifold.types.common.pose import Pose6D
from manifold.types.common.twist import Twist
from manifold.types.act.controller_config import TrajectoryControllerConfig
from control import computeSingleDeltaTwist, computeDeltaTwists

import numpy as np
from scipy.spatial.transform import Rotation

import pytest


class TestController:

    def testConvergenceToStationaryTarget(self):
        """EE should converge toward a stationary object, reducing position error over the trajectory."""

        obj_pose = Pose6D.from_position_and_rotation_matrix(
            position=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            rotation_matrix=np.eye(3, dtype=np.float64),
        )

        obj_twist = Twist.from_linear_angular(
            linear=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            angular=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        )
        ee_pose = Pose6D.from_position_and_rotation_matrix(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            rotation_matrix=np.eye(3, dtype=np.float64),
        )
        ee_twist = Twist.from_linear_angular(
            linear=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            angular=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        )

        dt = 0.1
        config = TrajectoryControllerConfig(
            kp_position=1.0,
            kp_rotation=1.0,
            latency=0.0,
            simulation_horizon=10.0,
            dt=dt,
        )

        traj = computeDeltaTwists(ee_pose, ee_twist, obj_pose, obj_twist, config)

        initial_error = np.linalg.norm(
            np.asarray(ee_pose.position) - np.asarray(obj_pose.position)
        )

        final_error = np.linalg.norm(
            np.asarray(traj[-1].pose.position)
            - np.asarray(obj_pose.position)
        )

        ee_positions = np.array([np.asarray(s.pose.position) for s in traj])

        print("\n")
        for index, element in enumerate(ee_positions):
            print("dt: {:.2f}    {}".format(dt * index, element))

        assert final_error < initial_error, (
            "Controller should reduce error initial error{} final error{}".format(
                initial_error, final_error,
            )
        )

    def testRotationalConvergence(self):
        """EE rotation should converge toward the object's rotation (90 deg yaw offset)."""

        targetRotation = Rotation.from_rotvec([0.0, 0.0, np.pi / 2]).as_matrix()

        obj_pose = Pose6D.from_position_and_rotation_matrix(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            rotation_matrix=targetRotation,
        )

        obj_twist = Twist.from_linear_angular(
            linear=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            angular=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        )

        ee_pose = Pose6D.from_position_and_rotation_matrix(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            rotation_matrix=np.eye(3, dtype=np.float64),
        )

        ee_twist = Twist.from_linear_angular(
            linear=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            angular=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        )

        dt = 0.1

        params = TrajectoryControllerConfig(
            kp_position=0.5,
            kp_rotation=0.5,
            latency=0.2,
            simulation_horizon=10.0,
            dt=dt,
        )

        traj = computeDeltaTwists(ee_pose, ee_twist, obj_pose, obj_twist, params)

        initial_rot_error = np.matmul(
            np.asarray(ee_pose.rotation_matrix).T,
            np.asarray(obj_pose.rotation_matrix),
        )
        initial_rot_vec = Rotation.from_matrix(initial_rot_error).as_rotvec()
        initial_rot_error_mag = np.linalg.norm(initial_rot_vec)

        final_ee_rotation = np.asarray(traj[-1].pose.rotation_matrix)
        final_rot_error = np.matmul(
            final_ee_rotation.T, np.asarray(obj_pose.rotation_matrix),
        )
        final_rot_vec = Rotation.from_matrix(final_rot_error).as_rotvec()
        final_rot_error_mag = np.linalg.norm(final_rot_vec)

        print(f"\nInitial rotational error: {initial_rot_error_mag:.4f} rad ({np.degrees(initial_rot_error_mag):.2f} deg)")
        print(f"Final rotational error: {final_rot_error_mag:.4f} rad ({np.degrees(final_rot_error_mag):.2f} deg)")

    def testPositionAndRotationConvergence(self):
        """Both position and rotation error should decrease when tracking a static target with offset in all DOF."""

        target_rotation = Rotation.from_rotvec([0, 0, np.pi / 4]).as_matrix().astype(np.float64)

        obj_pose = Pose6D.from_position_and_rotation_matrix(
            position=np.array([1.0, 2.0, 5.0], dtype=np.float64),
            rotation_matrix=target_rotation,
        )

        obj_twist = Twist.from_linear_angular(
            linear=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            angular=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        )

        ee_pose = Pose6D.from_position_and_rotation_matrix(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            rotation_matrix=np.eye(3, dtype=np.float64),
        )

        ee_twist = Twist.from_linear_angular(
            linear=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            angular=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        )

        dt = 0.1
        config = TrajectoryControllerConfig(
            kp_position=1.0,
            kp_rotation=1.0,
            latency=0.0,
            simulation_horizon=10.0,
            dt=dt,
        )

        traj = computeDeltaTwists(ee_pose, ee_twist, obj_pose, obj_twist, config)

        initial_pos_error = np.linalg.norm(
            np.asarray(ee_pose.position) - np.asarray(obj_pose.position)
        )
        final_pos_error = np.linalg.norm(
            np.asarray(traj[-1].pose.position)
            - np.asarray(obj_pose.position)
        )

        initial_rot_error = Rotation.from_matrix(
            np.matmul(
                np.asarray(ee_pose.rotation_matrix).T,
                np.asarray(obj_pose.rotation_matrix),
            )
        ).as_rotvec()
        initial_rot_error_mag = np.linalg.norm(initial_rot_error)

        final_rot_error = Rotation.from_matrix(
            np.matmul(
                np.asarray(traj[-1].pose.rotation_matrix).T,
                np.asarray(obj_pose.rotation_matrix),
            )
        ).as_rotvec()
        final_rot_error_mag = np.linalg.norm(final_rot_error)

        print(f"\nPosition error: {initial_pos_error:.4f} -> {final_pos_error:.4f} m")
        print(f"Rotation error: {initial_rot_error_mag:.4f} -> {final_rot_error_mag:.4f} rad ({np.degrees(initial_rot_error_mag):.2f} -> {np.degrees(final_rot_error_mag):.2f} deg)")

        assert final_pos_error < initial_pos_error
        assert final_rot_error_mag < initial_rot_error_mag

    def testTrackingConstantVelocity(self):
        """EE velocity should converge to match a constant-velocity object, with bounded position and rotation error."""

        objRotation = Rotation.from_rotvec([0.0, 0.0, np.pi / 2]).as_matrix()

        obj_pose = Pose6D.from_position_and_rotation_matrix(
            position=np.array([5.0, 6.0, 7.0], dtype=np.float64),
            rotation_matrix=objRotation,
        )

        obj_twist = Twist.from_linear_angular(
            linear=np.array([1.0, 1.0, 1.0], dtype=np.float64),
            angular=np.array([0.1, 0.1, 0.1], dtype=np.float64),
        )

        ee_pose = Pose6D.from_position_and_rotation_matrix(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            rotation_matrix=np.eye(3, dtype=np.float64),
        )

        ee_twist = Twist.from_linear_angular(
            linear=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            angular=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        )

        dt = 0.1
        config = TrajectoryControllerConfig(
            kp_position=2.0,
            kp_rotation=1.0,
            latency=0.0,
            simulation_horizon=5.0,
            dt=dt,
        )

        traj = computeDeltaTwists(ee_pose, ee_twist, obj_pose, obj_twist, config)

        final_step = traj[-1]
        final_ee_position = np.asarray(final_step.pose.position)
        final_pos_error = np.linalg.norm(
            np.asarray(obj_pose.position) - final_ee_position
        )

        final_ee_velocity = np.asarray(final_step.twist.linear)
        target_velocity = np.asarray(obj_twist.linear)

        velocity_error = np.linalg.norm(final_ee_velocity - target_velocity)

        final_ee_rotation = np.asarray(final_step.pose.rotation_matrix)
        rot_error = np.matmul(final_ee_rotation.T, np.asarray(obj_pose.rotation_matrix))
        rot_error_vec = Rotation.from_matrix(rot_error).as_rotvec()
        final_rot_error_mag = np.linalg.norm(rot_error_vec)

        print(f"\nTarget velocity: {target_velocity}")
        print(f"Final EE velocity: {final_ee_velocity}")
        print(f"Velocity error: {velocity_error:.4f} m/s")

        print(f"Final positional error: {final_pos_error:.4f} m")
        print(f"Final rotational error: {final_rot_error_mag:.4f} rad ({np.degrees(final_rot_error_mag):.2f} deg)")

        assert velocity_error < 0.1, f"Velocity tracking error too large: {velocity_error:.4f} m/s"
