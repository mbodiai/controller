from ee_trajectory_controller import Pose, Twist, trajectory, trajectoryStep, controllerConfig
from ee_trajectory_controller import projectObjectState
from ee_trajectory_controller import computeSingleDeltaTwist, computeDeltaTwists
from ee_trajectory_controller import computeMetrics, plotMetrics

import numpy as np
from ee_trajectory_controller import integratePosition, integrateRotation, findRotVec
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

import pytest


class TestController:

    def testConvergenceToStationaryTarget(self):

        obj_pose = Pose(
        position=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        rotation=np.eye(3, dtype=np.float64)
        )

        obj_twist = Twist(
            linearVelocity=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            angularVelocity=np.array([0.0, 0.0, 0.0], dtype=np.float64)
        )
        ee_pose = Pose(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64)
        )
        ee_twist = Twist(
            linearVelocity=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            angularVelocity=np.array([0.0, 0.0, 0.0], dtype=np.float64)
        )
        
        dt = 0.1
        config = controllerConfig(
            kPposition=1.0,
            kProtation=1.0,
            latency=0.0,
            simulationHorizon=10.0,
            dt=dt
        ) 
        
        trajectory = computeDeltaTwists(ee_pose, ee_twist, obj_pose, obj_twist, config)

        initial_error = np.linalg.norm(ee_pose.position - obj_pose.position)

        final_error = np.linalg.norm(
            trajectory.steps[-1].eePose.position - 
            trajectory.steps[-1].objectPose.position
        )
        
        #ee_positions = np.array([step.eePose.position for step in trajectory.steps])
        trajectoryData = trajectory.returnData()
        ee_positions = trajectoryData['ee_positions']
        

        print("\n")

        for index, element in enumerate(ee_positions):
            print("dt: {:.2f}    {}".format(dt * index, element))

        metrics = computeMetrics(trajectory, obj_pose, 0.5 )       
        plotMetrics(trajectory, obj_pose, metrics)

        assert final_error < initial_error, "Controller should reduce error initial error{} final error{}".format(initial_error, final_error)
        

    def testRotationalConvergence(self):

        targetRotation = Rotation.from_rotvec([0.0,0.0,np.pi/2]).as_matrix()

        obj_pose = Pose(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),  # Same position
            rotation=targetRotation
        )
        
        obj_twist = Twist(
            linearVelocity=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            angularVelocity=np.array([0.0, 0.0, 0.0], dtype=np.float64)
        )

        # EE starts at identity rotation (different from target)
        ee_pose = Pose(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64)  # Identity = no rotation
        )
        
        ee_twist = Twist(
            linearVelocity=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            angularVelocity=np.array([0.0, 0.0, 0.0], dtype=np.float64)
        )
        

        dt = 0.1
        
        params = controllerConfig(
            kPposition= 05.,
            kProtation= 0.5,
            latency = 0.2,
            simulationHorizon= 10.0,
            dt = dt
        )
        
        traj = computeDeltaTwists(ee_pose, ee_twist, obj_pose, obj_twist, params)
        
        initial_rot_error = np.matmul(ee_pose.rotation.T, obj_pose.rotation)
        initial_rot_vec = Rotation.from_matrix(initial_rot_error).as_rotvec()
        initial_rot_error_mag = np.linalg.norm(initial_rot_vec)

        # Compute final rotational error
        final_ee_rotation = traj.steps[-1].eePose.rotation
        final_rot_error = np.matmul(final_ee_rotation.T, obj_pose.rotation)
        final_rot_vec = Rotation.from_matrix(final_rot_error).as_rotvec()
        final_rot_error_mag = np.linalg.norm(final_rot_vec)


        print(f"\nInitial rotational error: {initial_rot_error_mag:.4f} rad ({np.degrees(initial_rot_error_mag):.2f} deg)")
        print(f"Final rotational error: {final_rot_error_mag:.4f} rad ({np.degrees(final_rot_error_mag):.2f} deg)")

        metrics = computeMetrics(traj, obj_pose, 0.5)
        plotMetrics(traj, obj_pose, metrics)
        
    def testPositionAndRotationConvergence(self):
        
        # Target: 1m away in x, rotated 45 degrees around z-axis
        target_rotation = Rotation.from_rotvec([0, 0, np.pi/4]).as_matrix().astype(np.float64)
        
        obj_pose = Pose(
            position=np.array([1.0, 2.0, 5.0], dtype=np.float64),
            rotation=target_rotation
        )
        
        obj_twist = Twist(
            linearVelocity=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            angularVelocity=np.array([0.0, 0.0, 0.0], dtype=np.float64)
        )
        
        # EE starts at origin with identity rotation
        ee_pose = Pose(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64)
        )
        
        ee_twist = Twist(
            linearVelocity=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            angularVelocity=np.array([0.0, 0.0, 0.0], dtype=np.float64)
        )
        
        dt = 0.1
        config = controllerConfig(
            kPposition=1.0,
            kProtation=1.0,
            latency=0.0,
            simulationHorizon=10.0,
            dt=dt
        )
        
        trajectory = computeDeltaTwists(ee_pose, ee_twist, obj_pose, obj_twist, config)
        
        # Check both position and rotation errors
        initial_pos_error = np.linalg.norm(ee_pose.position - obj_pose.position)
        final_pos_error = np.linalg.norm(
            trajectory.steps[-1].eePose.position - 
            trajectory.steps[-1].objectPose.position
        )
        
        initial_rot_error = Rotation.from_matrix(
            np.matmul(ee_pose.rotation.T, obj_pose.rotation)
        ).as_rotvec()
        initial_rot_error_mag = np.linalg.norm(initial_rot_error)
        
        final_rot_error = Rotation.from_matrix(
            np.matmul(trajectory.steps[-1].eePose.rotation.T, obj_pose.rotation)
        ).as_rotvec()
        final_rot_error_mag = np.linalg.norm(final_rot_error)
        
        print(f"\nPosition error: {initial_pos_error:.4f} -> {final_pos_error:.4f} m")
        print(f"Rotation error: {initial_rot_error_mag:.4f} -> {final_rot_error_mag:.4f} rad ({np.degrees(initial_rot_error_mag):.2f} -> {np.degrees(final_rot_error_mag):.2f} deg)")
        
        metrics = computeMetrics(trajectory, obj_pose, 0.5)
        plotMetrics(trajectory, obj_pose, metrics)
        
        assert final_pos_error < initial_pos_error
        assert final_rot_error_mag < initial_rot_error_mag

    def testTrackingConstantVelocity(self):
        """Test tracking an object moving at constant velocity"""
        
        objRotation = Rotation.from_rotvec([0.0, 0.0, np.pi/2]).as_matrix()

        
        # Object moving at 1 m/s in x-direction
        obj_pose = Pose(
            position=np.array([5.0, 6.0, 7.0], dtype=np.float64),
            rotation= objRotation
        )
        
        obj_twist = Twist(
            linearVelocity=np.array([1.0, 1.0, 1.0], dtype=np.float64),  # 1 m/s in x
            angularVelocity=np.array([0.1, 0.1, 0.1], dtype=np.float64)
        )
        
        # EE starts at origin
        ee_pose = Pose(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64)
        )
        
        ee_twist = Twist(
            linearVelocity=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            angularVelocity=np.array([0.0, 0.0, 0.0], dtype=np.float64)
        )
        
        dt = 0.1
        config = controllerConfig(
            kPposition=2.0,
            kProtation=1.0,
            latency=0.0,
            simulationHorizon=5.0,
            dt=dt
        )
        
        trajectory = computeDeltaTwists(ee_pose, ee_twist, obj_pose, obj_twist, config)
        
        data = trajectory.returnData()
        time = data['time']
        ee_velocities = data['ee_linear_velocities']
        

        # Compute final positional error
        final_ee_position = data['ee_positions'][-1]
        final_obj_position = data['object_positions'][-1]
        final_pos_error = np.linalg.norm(final_obj_position - final_ee_position)

        
        # Check if EE velocity converges to object velocity
        final_ee_velocity = ee_velocities[-1]
        target_velocity = obj_twist.linearVelocity
        
        velocity_error = np.linalg.norm(final_ee_velocity - target_velocity)

        # Compute final rotational error
        final_ee_rotation = data['ee_rotations'][-1]
        final_obj_rotation = data['object_rotations'][-1]
        rot_error = np.matmul(final_ee_rotation.T, final_obj_rotation)
        rot_error_vec = Rotation.from_matrix(rot_error).as_rotvec()
        final_rot_error_mag = np.linalg.norm(rot_error_vec)

        
        print(f"\nTarget velocity: {target_velocity}")
        print(f"Final EE velocity: {final_ee_velocity}")
        print(f"Velocity error: {velocity_error:.4f} m/s")

        print(f"Final positional error: {final_pos_error:.4f} m")
        print(f"Final rotational error: {final_rot_error_mag:.4f} rad ({np.degrees(final_rot_error_mag):.2f} deg)")
        
        metrics = computeMetrics(trajectory, obj_pose, 0.5)
        plotMetrics(trajectory, obj_pose, metrics)
        
        # EE should eventually match object velocity
        assert velocity_error < 0.1, f"Velocity tracking error too large: {velocity_error:.4f} m/s"



#class TestMetrics:
#    def metricstest1(self):
#        return
