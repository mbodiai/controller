"""Re-export manifold types used by the controller.

All canonical types live in manifold. This module provides backward-compatible
aliases so existing imports continue to work.
"""

import numpy as np

from manifold.types.common.pose import Pose6D
from manifold.types.common.twist import Twist
from manifold.types.act.trajectory import (
    TrajectoryControllerConfig,
    TrajectoryStep,
    Trajectory,
)

# Backward-compatible aliases
Pose = Pose6D
controllerConfig = TrajectoryControllerConfig
trajectoryStep = TrajectoryStep
trajectory = Trajectory

__all__ = [
    "Pose6D",
    "Pose",
    "Twist",
    "TrajectoryControllerConfig",
    "controllerConfig",
    "TrajectoryStep",
    "trajectoryStep",
    "Trajectory",
    "trajectory",
]
