from numpy.typing import NDArray
import numpy as np

from .datatypes import Pose, Twist 
from .geometry import integratePosition, integrateRotation 


def projectObjectState(currentPose: Pose, currentTwist: Twist, horizon: float, dt: float ) -> Pose:
    
    if (horizon < dt):
        a = 5
        #do something here
        
    nSteps = int(horizon/dt)
    position = currentPose.position.copy()    
    rotation = currentPose.rotation.copy()    
    
    linearVelocity = currentTwist.linearVelocity.copy()
    angularVelocity = currentTwist.angularVelocity.copy()
    
    for step in range(nSteps):
        position = integratePosition(position, linearVelocity, dt )
        rotation = integrateRotation(rotation, angularVelocity, dt)

    return Pose(position = position, rotation = rotation)        