import numpy as np
from numpy.typing import NDArray

from dataclasses import dataclass


@dataclass
class Pose:
    position: NDArray[np.float64]
    rotation: NDArray[np.float64]
    
@dataclass
class Twist:
    linearVelocity: NDArray
    angularVelocity: NDArray

@dataclass
class controllerConfig:
    kPposition : float
    kProtation : float
    latency : float #seconds
    simulationHorizon : float #seconds
    dt : float #seconds, timestep
    
@dataclass    
class trajectoryStep:
    dt: float
    eePose: Pose
    eeTwist: Twist
    objectPose: Pose
    objectTwist: Twist
    deltaTwist: Twist
    
@dataclass
class trajectory:
    steps: list[trajectoryStep]
    
    def returnData(self):
        return {
            'time': np.array([step.dt for step in self.steps]),
            'ee_positions': np.array([step.eePose.position for step in self.steps]),
            'ee_rotations': np.array([step.eePose.rotation for step in self.steps]),
            'ee_linear_velocities': np.array([step.eeTwist.linearVelocity for step in self.steps]),
            'ee_angular_velocities': np.array([step.eeTwist.angularVelocity for step in self.steps]),
            'object_positions': np.array([step.objectPose.position for step in self.steps]),
            'object_rotations': np.array([step.objectPose.rotation for step in self.steps]),
            'object_linear_velocities': np.array([step.objectTwist.linearVelocity for step in self.steps]),
            'object_angular_velocities': np.array([step.objectTwist.angularVelocity for step in self.steps]),
            'delta_linear_velocities': np.array([step.deltaTwist.linearVelocity for step in self.steps]),
            'delta_angular_velocities': np.array([step.deltaTwist.angularVelocity for step in self.steps])
        }