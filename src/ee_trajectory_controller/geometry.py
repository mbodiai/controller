import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from .datatypes import Pose, Twist, controllerConfig




def integratePosition(position: NDArray[np.float64], linearVelocity: NDArray[np.float64], dt: np.float64 ) -> NDArray[np.float64]:
    return position + linearVelocity * dt


def integrateRotation(rotation: NDArray[np.float64], angularVelocity: NDArray[np.float64], dt: np.float64 ) -> NDArray[np.float64]:
    omega_dt = angularVelocity * dt
    
    rotationMatrix = Rotation.from_rotvec(omega_dt)
    rotationMatrix = rotationMatrix.as_matrix()
    
    return np.matmul(rotation, rotationMatrix)
    

def findRotVec(rotationMatrix: NDArray[np.float64]) -> NDArray:
    
    rotationMatrix = Rotation.from_matrix(rotationMatrix)
    asRotvec = rotationMatrix.as_rotvec()
    
    return np.asarray(asRotvec, dtype=np.float64)
    
def errorRotVec(rotA: NDArray[np.float64], rotB: NDArray[np.float64]) -> NDArray:
    errRot = np.matmul(rotA.T, rotB)
    return findRotVec(errRot)
