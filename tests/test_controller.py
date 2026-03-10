from manifold.types.common.pose import Pose6D
from manifold.types.common.twist import Twist
from manifold.types.act.trajectory import TrajectoryControllerConfig
from control import computeSingleDeltaTwist, computeDeltaTwists, computeMetrics, plotMetrics

import numpy as np
from scipy.spatial.transform import Rotation

import pytest


class TestController:

    def testConvergenceToStationaryTarget(self):
        """EE should converge toward a stationary object, reducing position error over the trajectory."""
        pytest.skip("stub")

    def testRotationalConvergence(self):
        """EE rotation should converge toward the object's rotation (90 deg yaw offset)."""
        pytest.skip("stub")

    def testPositionAndRotationConvergence(self):
        """Both position and rotation error should decrease when tracking a static target with offset in all DOF."""
        pytest.skip("stub")

    def testTrackingConstantVelocity(self):
        """EE velocity should converge to match a constant-velocity object, with bounded position and rotation error."""
        pytest.skip("stub")
