import pybullet as p
import numpy as np
import pybullet_data

class PLANE:
    def __init__(self, client, friction_coeff):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = p.loadURDF("plane.urdf", physicsClientId = client)

        p.resetBasePositionAndOrientation(self.plane, [0, 0, 0], [0, 0, 0, 1], physicsClientId = client)
        self.changeFriction(friction_coeff,client)
    
    def changeFriction(self, friction_coeff, client):
        p.resetBasePositionAndOrientation(self.plane, [0, 0, 0], [0, 0, 0, 1], physicsClientId = client)
        p.changeDynamics(self.plane, -1, lateralFriction = friction_coeff)