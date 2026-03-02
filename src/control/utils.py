"""Utility functions for trajectory data extraction."""

from __future__ import annotations

import numpy as np

from manifold.types.act.trajectory import Trajectory


def return_data(traj: Trajectory) -> dict[str, np.ndarray]:
    """Transpose a Trajectory's step list into arrays keyed by field.

    Returns a dict with keys:
        time, ee_positions, ee_rotations, object_positions, object_rotations,
        ee_linear_velocities, ee_angular_velocities,
        object_linear_velocities, object_angular_velocities
    """
    steps = traj.steps

    time = np.array([s.time for s in steps], dtype=np.float64)

    ee_positions = np.array(
        [np.asarray(s.ee_pose.position, dtype=np.float64) for s in steps],
    )
    ee_rotations = np.array(
        [np.asarray(s.ee_pose.rotation_matrix, dtype=np.float64) for s in steps],
    )

    object_positions = np.array(
        [np.asarray(s.object_pose.position, dtype=np.float64) for s in steps],
    )
    object_rotations = np.array(
        [np.asarray(s.object_pose.rotation_matrix, dtype=np.float64) for s in steps],
    )

    ee_linear_velocities = np.array(
        [np.asarray(s.ee_twist.linear, dtype=np.float64) for s in steps],
    )
    ee_angular_velocities = np.array(
        [np.asarray(s.ee_twist.angular, dtype=np.float64) for s in steps],
    )

    object_linear_velocities = np.array(
        [np.asarray(s.object_twist.linear, dtype=np.float64) for s in steps],
    )
    object_angular_velocities = np.array(
        [np.asarray(s.object_twist.angular, dtype=np.float64) for s in steps],
    )

    return {
        "time": time,
        "ee_positions": ee_positions,
        "ee_rotations": ee_rotations,
        "object_positions": object_positions,
        "object_rotations": object_rotations,
        "ee_linear_velocities": ee_linear_velocities,
        "ee_angular_velocities": ee_angular_velocities,
        "object_linear_velocities": object_linear_velocities,
        "object_angular_velocities": object_angular_velocities,
    }
