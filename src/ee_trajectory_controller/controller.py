from numpy.typing import NDArray
import numpy as np

from .datatypes import Pose, Twist, trajectory, trajectoryStep, controllerConfig
from .geometry import integratePosition, integrateRotation, findRotVec
from .projection import projectObjectState


def computeSingleDeltaTwist(eePose: Pose, eeTwist: Twist, objPose: Pose, objTwist: Twist, params: controllerConfig) -> Twist:
    
    
    currentObjPose = Pose(objPose.position.copy(), objPose.rotation.copy() )
    currentObjTwist = Twist(objTwist.linearVelocity.copy(), objTwist.angularVelocity.copy())
    
    currentEEpose = Pose(eePose.position.copy(), eePose.rotation.copy() )
    currentEETwist = Twist(eeTwist.linearVelocity.copy(), eeTwist.angularVelocity.copy())

    #positional error
    linearError =   currentObjPose.position - currentEEpose.position

    #rotationalError
    rotationalError = np.matmul(currentEEpose.rotation.T, currentObjPose.rotation)
    rotationalErrorVector = findRotVec(rotationalError)
    

    desiredLinearVelocity = objTwist.linearVelocity + linearError * params.kPposition
    desiredAngularVelocity = objTwist.angularVelocity + rotationalErrorVector * params.kProtation
    
    linearVelocityDelta = desiredLinearVelocity - currentEETwist.linearVelocity
    angularVelocityDelta = desiredAngularVelocity - currentEETwist.angularVelocity
    
    return Twist(linearVelocity= linearVelocityDelta, angularVelocity=angularVelocityDelta)


def computeDeltaTwists(eePose: Pose, eeTwist: Twist, objPose: Pose, objTwist: Twist, params: controllerConfig) -> trajectory:
    
    simulationHorizon = params.simulationHorizon
    dt = params.dt
    latency = params.latency
    
    
    trajectoryLog = []

    if (simulationHorizon < dt):
        a = 5
        #do something here

    nSteps = int(simulationHorizon/dt)
    

    currentObjPose = Pose(objPose.position.copy(), objPose.rotation.copy() )
    currentObjTwist = Twist(objTwist.linearVelocity.copy(), objTwist.angularVelocity.copy())
    
    currentEEpose = Pose(eePose.position.copy(), eePose.rotation.copy() )
    currentEETwist = Twist(eeTwist.linearVelocity.copy(), eeTwist.angularVelocity.copy())
    
    
    for step in range(1,nSteps):
        
        latencyCompensatedFutureObjectPose = projectObjectState(currentObjPose, currentObjTwist, latency, dt)
        
        deltaTwist = computeSingleDeltaTwist(currentEEpose, currentEETwist, latencyCompensatedFutureObjectPose, currentObjTwist, params)
        
        newDesiredLinearVelocity = currentEETwist.linearVelocity + deltaTwist.linearVelocity
        newDesiredAngularVelocity = currentEETwist.angularVelocity + deltaTwist.angularVelocity
        
        currentEETwist = Twist(linearVelocity= newDesiredLinearVelocity, angularVelocity= newDesiredAngularVelocity) 
        currentEEpose = projectObjectState(currentEEpose, currentEETwist, dt, dt)
        currentObjPose = projectObjectState(currentObjPose, currentObjTwist, dt, dt)
        
        trajStep = trajectoryStep(float(step) * dt, currentEEpose, currentEETwist, currentObjPose, currentObjTwist, deltaTwist) 
        trajectoryLog.append(trajStep)

        
        
    return trajectory(steps = trajectoryLog)        


        
        
        
        

        


    
    return