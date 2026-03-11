import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from manifold.types.common.pose import Pose6D
from manifold.types.common.twist import Twist
from manifold.types.act.controller_config import TrajectoryControllerConfig
from manifold.types.act.control import HandControl
from control.controller import TrajectoryController

EPS = 1e-10

def _make_config(**overrides) -> TrajectoryControllerConfig:
    defaults = dict(kp_position=2.0, kp_rotation=1.0, latency=5.0, simulation_horizon=5.0, dt=0.1)
    defaults.update(overrides)
    return TrajectoryControllerConfig(**defaults)


def _zero_pose() -> Pose6D:
    return Pose6D.from_position_and_rotation_matrix(np.zeros(3), np.eye(3))


def _zero_twist() -> Twist:
    return Twist.from_linear_angular(np.zeros(3), np.zeros(3))




class TestTrajectoryController:

    def test_stationary_target_convergence(self):
        """Position error should strictly decrease every step for a stationary target."""
        pytest.skip("stub")

    def test_constant_velocity_tracking(self):
        """EE should follow a moving target — error should stabilize, not diverge."""
        pytest.skip("stub")

    def test_rotation_convergence(self):
        """Rotation error should strictly decrease every step for a rotated stationary target."""
        pytest.skip("stub")

    def test_drift_correction_reduces_error(self):
        """With systematic bias in measurements, drift correction should help."""
        pytest.skip("stub")

    def test_blended_start_corrects_drift(self):
        """Blended start should move toward measured pose by drift_alpha."""
        pytest.skip("stub")

    def test_get_start_state_priority(self):
        """get_start_state should follow priority: last-pushed > blended > measured."""
        pytest.skip("stub")
