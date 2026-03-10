from manifold.types.common.pose import Pose6D
from manifold.types.common.twist import Twist
from manifold.utils.geometry import integrate_position, integrate_rotation, rotvec_from_matrix

import numpy as np
from scipy.spatial.transform import Rotation

import pytest


class TestIntegration:
    def testPositionIntegration(self):

        eePosition = np.array([0.0, 0.0, 0.0])

        eeLinVelocity = np.array([1.0, 1.0, 1.0])

        dt = 0.1

        futurePosition = integrate_position(eePosition, eeLinVelocity, dt)

        expectedPosition = np.array([0.1, 0.1, 0.1])

        assert np.allclose(futurePosition, expectedPosition)

    def testAngularIntegration(self):

        eeRotation = np.eye(3)
        eeAngVelocity = np.array([1.0, 1.0, 1.0])

        dt = 0.1

        futureRotation = integrate_rotation(eeRotation, eeAngVelocity, dt)

        expectedRotation = Rotation.from_rotvec(np.asarray([.1, .1, .1])).as_matrix()

        assert np.allclose(futureRotation, expectedRotation)


class TestRotationVector:
    def testfindrotvec(self):
        rotationMatrix = np.eye(3)

        out = rotvec_from_matrix(rotationMatrix)

        expectedRotVec = np.array([0.0, 0.0, 0.0])

        assert np.allclose(out, expectedRotVec)
