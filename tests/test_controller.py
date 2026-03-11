import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from manifold.types.common.pose import Pose6D
from manifold.types.common.twist import Twist
from manifold.types.act.controller_config import TrajectoryControllerConfig
from manifold.types.act.control import HandControl
from control.controller import TrajectoryController


class TestTrajectoryController:

    def test_stationary_target_convergence(self):
        """Position error should decrease when tracking a stationary target."""
        pytest.skip("stub")

    def test_constant_velocity_tracking(self):
        """EE should follow a moving target — error should stabilize, not diverge."""
        pytest.skip("stub")

    def test_rotation_convergence(self):
        """Rotation error should decrease when tracking a rotated stationary target."""
        pytest.skip("stub")

    def test_drift_correction_reduces_error(self):
        """With systematic bias in measurements, drift correction should help."""
        pytest.skip("stub")
