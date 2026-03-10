from manifold.types.common.pose import Pose6D
from manifold.types.common.twist import Twist
from manifold.utils.geometry import project_pose

import numpy as np
from scipy.spatial.transform import Rotation

import pytest


class TestProjectObjectState:
    def test_zero_velocity(self):
        """Pose should be unchanged when twist is zero."""
        pytest.skip("stub")

    def test_constant_linear_velocity(self):
        """Position should advance by velocity * horizon with zero angular velocity."""
        pytest.skip("stub")

    def test_constant_angular_velocity(self):
        """Rotation should advance by angular_velocity * horizon with zero linear velocity."""
        pytest.skip("stub")
