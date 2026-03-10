from manifold.types.common.pose import Pose6D
from manifold.types.common.twist import Twist
from manifold.utils.geometry import integrate_position, integrate_rotation, rotvec_from_matrix

import numpy as np
from scipy.spatial.transform import Rotation

import pytest


class TestIntegration:
    def testPositionIntegration(self):
        """integrate_position should advance position by velocity * dt."""
        pytest.skip("stub")

    def testAngularIntegration(self):
        """integrate_rotation should rotate by angular_velocity * dt via rodrigues."""
        pytest.skip("stub")


class TestRotationVector:
    def testfindrotvec(self):
        """rotvec_from_matrix on identity should return zero vector."""
        pytest.skip("stub")
