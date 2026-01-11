
from ee_trajectory_controller import Pose, Twist, trajectory, trajectoryStep, controllerConfig

from ee_trajectory_controller import projectObjectState

import numpy as np

from ee_trajectory_controller import integratePosition, integrateRotation, findRotVec

from numpy.typing import NDArray

from scipy.spatial.transform import Rotation

import pytest


class TestIntegration:
    def testPositionIntegration(self):
        
        eePosition = np.array([0.0, 0.0, 0.0])
        eeRotation = np.eye(3)
        
        currentPose = Pose( eePosition, eeRotation)
        
        eeLinVelocity = np.array([1.0, 1.0, 1.0])
        
        dt = 0.1
        
        futurePosition = integratePosition(eePosition, eeLinVelocity, dt)
        
        expectedPosition = np.array([0.1, 0.1, 0.1])
        
        assert np.allclose(futurePosition, expectedPosition)
        

        
    def testAngularIntegration(self):
        
        
        eeRotation = np.eye(3)
        eeAngVelocity = np.array([1.0, 1.0, 1.0])
        
        dt = 0.1
        
        futureRotation = integrateRotation(eeRotation, eeAngVelocity, dt)
        
        expectedRotation = Rotation.from_rotvec(np.asarray([.1, .1, .1])).as_matrix()
        

        assert np.allclose(futureRotation, expectedRotation)
        

class TestRotationVector:
    def testfindrotvec(self):
       rotationMatrix = np.eye(3)     
       
       out = findRotVec(rotationMatrix)
       
       expectedRotVec = np.array([0.0,0.0,0.0])
       
       assert np.allclose(out, expectedRotVec)
       
       
       

        
        



        

    