import pybullet as p
import os 
import sys
import numpy as np

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from calcs import calcKinematics

class walkingQUAD:
    def __init__(self, client, max_torque = 33.5, add_mass: float = 0,
                 Kp: float = 500, Kd: float = 10, Kp_damping: float = 0):

                
        # Robot Information - Constants
        # All information is from URDF or Unitree A1 Specifications
        self.NUM_JOINTS = 12
        self.NUM_LEGS = 4
        
        # Joint are collected as follows: 
        # [hip-x-axis revolute; hip-y-axis revolute; thigh calf joint revolute]
        # Joints are [front right, front left, back right, back left]
        # front right = [0, 1, 3]   front left = [4, 5, 7]
        # back right = [8, 9, 11]   back left = [12, 13, 15]
        self.JOINT_INDICES = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16]
        self.END_EFFECTOR_INDICES = [5, 9, 13, 17] # same foot joint number as foot link number 
        self.HIP_INDICES = [2, 6, 10, 14]
        self.JOINT_MAX_VEL_URDF = [52.4, 28.6, 28.6, 52.4, 28.6, 28.6, 52.4, 28.6, 28.6, 52.4, 28.6, 28.6] # from urdf file
        self.JOINT_MAX_VEL_LIM = [21]*12 # limited to half of 21 rad/s
        self.LINK_LENGTHS = [0.0838, 0.2, 0.2]
        self.LEG_ID = [1, 2, 3, 4]
        self.X_LIMITS = [-0.2, 0.2] # X-axis range of motion for foot in terms of hip co-ords
        self.Y_LIMITS = [0, 0.1] # Y-axis range of motion for foot in terms of hip co-ords
        self.Z_LIMITS = [-0.33, -0.15] # Z-axis range of motion for foot in terms of hip co-ords
        self.BODY_MASS = 4.713 # Robot totl mass is 12, however according to pybullet the trunk mass is 4.713 
        self.DEFAULT_FOOT_POS_RIGHT = [0, -0.05, -0.28]
        self.DEFAULT_FOOT_POS_LEFT = [0, 0.05, -0.28]
        
        # Variables that can change each run
        self.client = client
        self.max_torque = max_torque # 33.5 N.m (from unitree A1 datasheet) 
        self.add_mass = add_mass     # Additional mass for domain randomisation 
        self.Kp = Kp
        self.Kd = Kd
        self.Kp_damping = Kp_damping
        
        # Variables for storing information
        self.torque = [None]*self.NUM_JOINTS
        self.default_link_pos = [None]*self.NUM_JOINTS
        self.current_action = [0]*self.NUM_JOINTS
        self.previous_action = [0]*self.NUM_JOINTS
        
        # Initialise kinematic calcs 
        self.kinematcs = calcKinematics(linkLengths = self.LINK_LENGTHS, Kp=self.Kp, Kd=self.Kd, max_torque=self.max_torque, max_joint_velocity=self.JOINT_MAX_VEL_LIM[0])
        

        # Load Asset 
        fn = os.path.join(os.path.dirname(__file__), "uintreeA1/urdf/a1.urdf")
        self.quadId = p.loadURDF(fileName = fn, basePosition = (0, 0, 0.281),
                                 flags = p.URDF_USE_SELF_COLLISION | p.URDF_MAINTAIN_LINK_ORDER,
                                 physicsClientId = self.client)

        # set default joint states
        self.resetPos()
    
    def resetParams(self, max_torque, add_mass):

        self.max_torque = max_torque # 33.5 N.m (from unitree A1 datasheet) 
        self.add_mass = add_mass 
        self.kinematcs.updateParams(Kp = self.Kp, Kd = self.Kd, max_torque = self.max_torque)


    def resetPos(self, uneven_terrain = False, foot_height = None): 
        """
        Reset robot position to default position 
        This must be done before any dyanmics are changed
        This ensures that the default position is set correctly each time 
        """
        
        self.previous_action = [self.DEFAULT_FOOT_POS_RIGHT[0], self.DEFAULT_FOOT_POS_RIGHT[1], self.DEFAULT_FOOT_POS_RIGHT[2],
                                self.DEFAULT_FOOT_POS_LEFT[0], self.DEFAULT_FOOT_POS_LEFT[1], self.DEFAULT_FOOT_POS_LEFT[2],
                                self.DEFAULT_FOOT_POS_RIGHT[0], self.DEFAULT_FOOT_POS_RIGHT[1], self.DEFAULT_FOOT_POS_RIGHT[2],
                                self.DEFAULT_FOOT_POS_LEFT[0], self.DEFAULT_FOOT_POS_LEFT[1], self.DEFAULT_FOOT_POS_LEFT[2]]
        if uneven_terrain:
            self.previous_action = [self.DEFAULT_FOOT_POS_RIGHT[0], self.DEFAULT_FOOT_POS_RIGHT[1], foot_height[0],
                                self.DEFAULT_FOOT_POS_LEFT[0], self.DEFAULT_FOOT_POS_LEFT[1], foot_height[1],
                                self.DEFAULT_FOOT_POS_RIGHT[0], self.DEFAULT_FOOT_POS_RIGHT[1], foot_height[2],
                                self.DEFAULT_FOOT_POS_LEFT[0], self.DEFAULT_FOOT_POS_LEFT[1], foot_height[3]]
        
        # Calculate the default leg angles based on the defualt foot positions 
        for i in range(4):

            default_pos = self.previous_action[i*3:i*3+3]
            
            ik_angles = self.kinematcs.inverseKinematics(default_pos, self.LEG_ID[i])
            
            for j in range(3):
                self.default_link_pos[j + i*3] = ik_angles[j]
        
        self.previous_action = self.kinematcs.normaliseLimits(self.previous_action, self.X_LIMITS, self.Y_LIMITS, self.Z_LIMITS, self.LEG_ID)
        
        # reset robot to base position and orientation 
        p.resetBasePositionAndOrientation(self.quadId, [0, 0, 0.281], [0, 0, 0, 1], physicsClientId = self.client)
        p.resetBaseVelocity(self.quadId, [0, 0, 0], [0, 0, 0], physicsClientId = self.client)

        # Get joints to starting position
        self.resetJoints(self.default_link_pos)
        
        # Apply dynamic changes 
        self.changeDynamics()
        
        # Get ready for torque control
        self.applyVelocity([0]*self.NUM_JOINTS)
        # wait to stabilise
        #for i in range(50):
        #    p.stepSimulation(physicsClientId = self.client)


    def applyAction(self, action, iteration):
        """
        This function applies the action chosen by the RL algorithm to the robot 
        The action is first scaled so that it is no longer in a range of [-1, 1]
        The torque is then calculated using the kinematics class functions 
        Finally the torque is offset using the velocity of the joints and applied to the robot 
        """
        
        # Ensuring the correct storing of the previous action
        if iteration == 0:
            self.previous_action = [i for i in self.current_action]

        scaled_action = self.scaleAction(action)

        # Altering Action into matrix form
        matrix_action = np.zeros((self.NUM_LEGS, 3))
        for i in range(self.NUM_LEGS):
            for j in range(3):
                matrix_action[i, j] = scaled_action[j + i*3]
            
        # Calulate Torque  
        for i in range(self.NUM_LEGS):
            # Get link actual pos and velcoity 
            foot_pos, foot_vel, _, hip_pos, hip_vel, _,  = self.getlegInfo(self.END_EFFECTOR_INDICES[i], self.HIP_INDICES[i])
            
            # Get desired foot position and calculate torue
            _, jacobian, _ = self.kinematcs.returnJacobian(matrix_action[i], self.LEG_ID[i])
            calculated_joint_torques = self.kinematcs.Torque(hip_pos, foot_pos, hip_vel, foot_vel, jacobian, matrix_action[i])
            
            for j in range(3):
                self.torque[j + i*3] = calculated_joint_torques[j][0]
        
        # Scaling torque based on joint velocity
        #'''
        _, joint_vel = self.getJointPosAndVel()
        
        for i in range(self.NUM_JOINTS):
            self.torque[i] = self.torque[i] - joint_vel[i]*self.Kp_damping
        #'''    
            
        # set torque 
        scaled_torque = self.kinematcs.scaleTorque(self.torque, joint_vel)
        self.applyTorque(scaled_torque)

        # Ensuring the correct storing of the previous action
        if iteration == 0:
            self.current_action = [i for i in action]

        return self.torque


    def getObservations(self):
        """
        -- The following values make up the observation space ---
         
        1) - The X, Y and Z POSITION of the Body (3) 
        2) - The X, Y, z and W Orientation of the body (4) 
        3) - The X, Y and Z linear VELOCITY of the body (3) 
        4) - Angular velocity of body (3)
        5) - Angular POSITION of knee and Hip, calf joints (12)
        6) - Angular VELOCITY of knee and hip, calf joints (12)
        7) - Cartesian POSITION of the feet (4*3 = 12)
        8) - Cartesain VELOCITY of the feet (4*3 = 12)
        9) - The Action from the previous step (12)
        --- Total No. of obs is therefore 73 ---
        """

        body_pos, body_ori,  body_lin_vel, body_ang_vel  = self.getBasePosAndVel()
        #body_ori = p.getEulerFromQuaternion(body_ori)
    
        # Getting the Joint Info (Hip, Knee, Calf)
        joint_pos, joint_velocity  = self.getJointPosAndVel()

        # Getting the foot positions and velocities 
        foot_pos = []
        foot_velocity = []
        for i in range(self.NUM_LEGS):
            _foot_pos, _foot_velocity, _, _, _, _ = self.getlegInfo(self.END_EFFECTOR_INDICES[i], self.HIP_INDICES[i])
            foot_pos.append(_foot_pos)
            foot_velocity.append(_foot_velocity)
            
        #foot_contact_bool = self.getContact()
        
        return body_pos, body_ori, body_lin_vel, body_ang_vel, joint_pos, joint_velocity, foot_pos, foot_velocity, self.previous_action
    
    def scaleAction(self, action):
        
        scaled_action = self.kinematcs.scaleLimits(action, self.X_LIMITS, self.Y_LIMITS, self.Z_LIMITS, self.LEG_ID) 
        
        return scaled_action

    def getlegInfo(self, foot_number, hip_number): 
        # Get information for calculating torque
        # This is the hip position and veloctiy as well as the foot position and velocity
        
        temp = p.getLinkState(self.quadId, foot_number, computeLinkVelocity = 1, physicsClientId = self.client)
        foot_pos = temp[4]   
        foot_vel = temp[6]
        foot_ang_vel = temp[7]

        temp = p.getLinkState(self.quadId, hip_number, computeLinkVelocity = 1, physicsClientId = self.client)
        hip_pos = temp[4] 
        hip_vel = temp[6]
        hip_ang_vel = temp[7]
        
        return foot_pos, foot_vel, foot_ang_vel, hip_pos, hip_vel, hip_ang_vel
    
    
    def getJointPosAndVel(self):
        # Get joint velocities that can be used to scale torque 
        
        temp = p.getJointStates(self.quadId, jointIndices = self.JOINT_INDICES, physicsClientId = self.client)
        joint_pos = [j[0] for j in temp] 
        joint_vel = [j[1] for j in temp]
        
        return joint_pos, joint_vel
    

    def applyPosition(self, position): 
        # Adjust motor position using position control 

        for i in range(self.NUM_JOINTS):
                p.setJointMotorControl2(self.quadId, jointIndex=self.JOINT_INDICES[i], controlMode=p.POSITION_CONTROL, targetPosition = position[i], physicsClientId = self.client)   
    
    
    def applyVelocity(self, velocity): 
        # Adjust motor position using position control 

        for i in range(self.NUM_JOINTS):
                p.setJointMotorControl2(self.quadId, jointIndex=self.JOINT_INDICES[i], controlMode=p.VELOCITY_CONTROL, force = velocity[i], physicsClientId = self.client)   
                

    def applyTorque(self, torque): 
        # Adjust motor position using torque control

        for i in range(self.NUM_JOINTS):
            p.setJointMotorControl2(self.quadId, jointIndex=self.JOINT_INDICES[i], controlMode=p.TORQUE_CONTROL, force = torque[i], physicsClientId = self.client)
            
            
    def resetJoints(self, joint_positions):
        # Reset joint positions to a specific position
        
        for i in range(self.NUM_JOINTS):
            p.resetJointState(self.quadId, self.JOINT_INDICES[i], joint_positions[i], physicsClientId = self.client)


    def getBasePosAndVel(self):

        _, _, _, _, body_pos, body_ori,  body_lin_vel, body_ang_vel  = p.getLinkState(self.quadId, 0, computeLinkVelocity = 1, physicsClientId = self.client)
        
        return body_pos, body_ori, body_lin_vel, body_ang_vel 
    
    
    def getRollPitchYaw(self, orientation):

        roll_pitch_yaw = p.getEulerFromQuaternion(orientation)
        roll, pitch, yaw = roll_pitch_yaw[0], roll_pitch_yaw[1], roll_pitch_yaw[2]
        
        return roll, pitch, yaw

    def getFeetLinearVelocity(self):
        
        foot_velocity = []
        
        for i in range(self.NUM_LEGS):
            _, _foot_velocity, _, _, _, _ = self.getlegInfo(self.END_EFFECTOR_INDICES[i], self.HIP_INDICES[i])
            foot_velocity.append(_foot_velocity)
            
        return foot_velocity
    
    def getContact(self):

        contact = [0]* len(self.END_EFFECTOR_INDICES)
        normalForce = [0]* len(self.END_EFFECTOR_INDICES)

        for i in range(len(self.END_EFFECTOR_INDICES)):
            temp = p.getContactPoints(bodyA = self.quadId, linkIndexA = self.END_EFFECTOR_INDICES[i], 
                                    physicsClientId = self.client)
            
            if temp !=  ():
                contact[i] = 1
                normalForce[i] = temp[0][9]
            else: 
                contact[i] = 0
                normalForce[i] = 0 # foot not in contact with ground
        
        return contact, normalForce


    def changeDynamics(self):
        """
        Change robot dynmaics, specifically the CoM, adding an additional mass 
        and ensuring the velcity constraints are correctly applied 
        """

        # adding additional mass and possibly a CoM offset 
        p.changeDynamics(self.quadId, 0,  physicsClientId = self.client, mass = (self.add_mass + self.BODY_MASS))

        # ensure max velocity of each joint is the same as the urdf 
        for i in range(self.NUM_JOINTS):
            p.changeDynamics(self.quadId, self.JOINT_INDICES[i],  physicsClientId = self.client, maxJointVelocity = self.JOINT_MAX_VEL_LIM[i])


    def identifyJoints_andLinks(self): 
        """
        Function retrieves joint information, such as joint name and corresponding number 
        and prints the information to the terminal to aid in debugging 
        """
        
        JNum = p.getNumJoints(self.quadId, physicsClientId = self.client)

        print("Num of Joints is " + str(JNum))
        print("Joints are as follows: ")
        for i in range(JNum):
            print(p.getJointInfo(self.quadId, i, physicsClientId = self.client))

        print("Links are as follows: ")
        link_name_to_index = {p.getBodyInfo(self.quadId)[0].decode('UTF-8'):-1,}
        
        for _id in range(p.getNumJoints(self.quadId)):
            _name = p.getJointInfo(self.quadId, _id, physicsClientId = self.client)[12].decode('UTF-8')
            link_name_to_index[_name] = _id
            print(_id, link_name_to_index)
            print(p.getLinkState(self.quadId, _id,  physicsClientId = self.client))

    def getClient(self):
        return self.quadId