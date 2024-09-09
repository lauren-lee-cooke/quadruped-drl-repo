"""
This defines a class calcKinematics that implements functionalities for an A1 quadruped robot's inverse kinematics, Jacobian calculation, torque computation, and more. 
It calculates a symbolic representations of the Denavit-Hartenberg parameters, based on the given class inputs.
These D-H parameters are used to calculate the IK and the Jacobian, which includes the angles of the joints as well as the actual foot positon in terms of hip co-ordinate frame.
The IK results are then used to calculated the torque using PD control based on this formula t = J*(Kp(desired foot pos - actual foot pos) - Kd(actual foot velocity))
The class also includes transformations between world and hip coordinates in order to simply the torque calculation 
Finally the code includes a function to scale normalized input to between achievable foot positions in each axis.
"""

import numpy as np 
from math import atan, pi, sqrt, acos
from sympy import sin, cos, Matrix, symbols
from sympy.utilities.lambdify import lambdify
from motor_model import LinearMotorModel

class calcKinematics():

    def __init__(self, linkLengths = [0.0838, 0.2, 0.2], Kp=500, Kd=10, max_torque = 33.5, max_joint_velocity = 21.0):

        self.L1, self.L2, self.L3 = linkLengths[0], linkLengths[1], linkLengths[2] 

        self.Kp = Kp
        self.Kd = Kd
        self.max_torque = max_torque
        self.max_joint_velocity = max_joint_velocity
        
        # Initialse linear motor for torque calcs: 
        self.motor_model = LinearMotorModel(tau_max = self.max_torque, omega_max = self.max_joint_velocity)

        # The rest of this function just symbolically represents the Jacobian and End Effector positions for the left and right legs 

        #      Symbolic determination of Jacobian using D-H parameters
        # |------|-------------|------------|-------------|-------------|
        # | Link | link length | link twist | link offset | joint angle |
        # |------|-------------|------------|-------------|-------------|
        # |      |     a       |    alpha   |      d      |     theta   |
        # |------|-------------|------------|-------------|-------------|
        # | 0-1  |  0          | -pi/2      | 0           | pi/2        |
        # |------|-------------|------------|-------------|-------------|
        # | 1-2  | L1 (+/-)    | 0          | 0           | theta1      |
        # |------|-------------|------------|-------------|-------------|
        # | 2-3  |  0          | -pi/2      | 0           | -pi/2       |
        # |------|-------------|------------|-------------|-------------|
        # | 3-4  |  L2         | 0          | 0           | theta2      |
        # |------|-------------|------------|-------------|-------------|
        # | 4-5  |  L3         | 0          | 0           | theta3      |
        # |------|-------------|------------|-------------|-------------|

        self.a, self.alpha, self.d, self.theta = symbols("a, alpha, d, theta")
        self.a1, self.a2, self.a3 = symbols("a1, a2, a3") # a1, a2, a3 represent theta1, theta2, theta3 respectively

        # Now get the co-ordinate frames 
        # Because of method of assigning co-ord frames, need two separate ones for left and right
        t_matrix = Matrix([[cos(self.theta),    -sin(self.theta)*cos(self.alpha),   sin(self.theta)*sin(self.alpha),    self.a*cos(self.theta)], 
                           [sin(self.theta),    cos(self.theta)*cos(self.alpha),    -cos(self.theta)*sin(self.alpha),   self.a*sin(self.theta)], 
                           [0,                  sin(self.alpha),                    cos(self.alpha),                    self.d],
                           [0,                  0,                                  0,                                  1]])

        t_01 = t_matrix.subs([(self.a, 0), (self.alpha, pi/2), (self.d, 0), (self.theta, pi/2)])

        t_12_r = t_matrix.subs([(self.a, -self.L1), (self.alpha, 0), (self.d, 0), (self.theta, self.a1)])
        t_12_l = t_matrix.subs([(self.a, self.L1), (self.alpha, 0), (self.d, 0), (self.theta, self.a1)])

        t_23 = t_matrix.subs([(self.a, 0), (self.alpha, -pi/2), (self.d, 0), (self.theta, -pi/2)])
        t_34 = t_matrix.subs([(self.a, self.L2), (self.alpha, 0), (self.d, 0), (self.theta, self.a2)])
        t_45 = t_matrix.subs([(self.a, self.L3), (self.alpha, 0), (self.d, 0), (self.theta, self.a3)]) 
    
        t_05_r = t_01*t_12_r*t_23*t_34*t_45
        t_05_l = t_01*t_12_l*t_23*t_34*t_45

        # === Right Side ===
        # Split the T_05 into Rotation Matrix and End-Effector Position 
        self.Rot_matrix_right = Matrix([[t_05_r[0], t_05_r[1], t_05_r[2]],
                             [t_05_r[4], t_05_r[5], t_05_r[6]],
                             [t_05_r[8], t_05_r[9], t_05_r[10]]])

        self.T_05_right = Matrix([t_05_r[3], t_05_r[7], t_05_r[11]])

        Q = Matrix([self.a1, self.a2, self.a3])
        self.Jacobian_right = self.T_05_right.jacobian(Q) # Symbolic final jacobian

        # Changing the format of the functions to make it less computationally expensive
        #self.Rot_matrix_function_right = lambdify((self.a1, self.a2, self.a3), self.Rot_matrix_right)
        self.T_05_right_function = lambdify((self.a1, self.a2, self.a3), self.T_05_right, cse=True)
        self.Jacobian_right_function = lambdify((self.a1, self.a2, self.a3), self.Jacobian_right, cse=True)

        # === Left Side ===
        # Split the T_05 into Rotation Matrix and End-Effector Position 
        self.Rot_matrix_left = Matrix([[t_05_l[0], t_05_l[1], t_05_l[2]],
                             [t_05_l[4], t_05_l[5], t_05_l[6]],
                             [t_05_l[8], t_05_l[9], t_05_l[10]]])

        self.T_05_left = Matrix([t_05_l[3], t_05_l[7], t_05_l[11]])
        
        Q = Matrix([self.a1, self.a2, self.a3])
        self.Jacobian_left = self.T_05_left.jacobian(Q) # Symbolic final jacobian
        
        # Changing the format of the functions to make it less computationally expensive
        # self.Rot_matrix_function_left = lambdify((self.a1, self.a2, self.a3), self.Rot_matrix_left, cse=True)
        self.T_05_left_function = lambdify((self.a1, self.a2, self.a3), self.T_05_left, cse=True)
        self.Jacobian_left_function = lambdify((self.a1, self.a2, self.a3), self.Jacobian_left, cse=True)

    def updateParams(self, Kp, Kd, max_torque):
        
        self.Kp = Kp
        self.Kd = Kd
        self.max_torque = max_torque

    def inverseKinematics(self, end_effector_pos=[0, 0, -0.33], leg_ID = 1):
        """
        This function determines the inverse kinematics for the robot
        It takes in the end effector position and a variable that dictates whether the 
        IK are being determine for the left (2, 4) or right leg (1, 3)
        It return the angles of each of the links/motors for a specified end effector position
        AKA as the angles theta1, theta2, theta3
        """

        x, y, z = end_effector_pos[0], end_effector_pos[1], end_effector_pos[2]
        
        # length of vector projected on the YZ plane
        len_b = sqrt(y**2 + z**2)

        if len_b < self.L1: # Checking for math error 
            theta1 = 0
        else:
            if leg_ID in [1, 3]:
                theta1 =  pi/2 - atan(y/z) - acos(self.L1/len_b)
            else:
                theta1 =  - pi/2 - atan(y/z) + acos(self.L1/len_b)

       # Axis rotation and position change for ease of calc
        axis_rot = - theta1

        j2 = np.array([0, self.L1*cos(theta1), self.L1*sin(theta1)]) 
        j4 = np.array(end_effector_pos)
        j42 = j4 - j2

        rot_matrix = np.matrix([[1, 0,              0],
                                [0, cos(axis_rot),  -sin(axis_rot)],
                                [0, sin(axis_rot),    cos(axis_rot)]])
        
        j42_ = rot_matrix*(np.reshape(j42,[3,1]))

        # foot co-ord in terms of new axis 
        x_, y_, z_ = j42_[0], j42_[1], j42_[2]

        # length of vector projected on the Z'X' plane
        len_c = sqrt(x_**2 + z_**2)
    
        theta3 = - pi + acos((self.L2**2 + self.L3**2 - len_c**2)/(2*self.L2 *self.L3)) 
        theta2 =  acos((len_c**2 + self.L2**2 - self.L3**2)/(2*self.L2*len_c)) + atan(x_/z_) 

        ik_angles = theta1, theta2, theta3
    
        return ik_angles


    def calcJacobian(self, angles, leg_ID):
        """
        This function takes in the angles calculated from the IK (theta1, theta2, theta3)
        returns the Jacobian, End Effector position and rotation matrix in hip co-ords 
        """

        theta1, theta2, theta3 = angles[0], angles[1], angles[2]
        
        # substitute in correct angles from ik
        if leg_ID in [1, 3]:
            Foot_pos = self.T_05_right_function(theta1, theta2, theta3) 
            Jacobian = self.Jacobian_right_function(theta1, theta2, theta3) 
        else:
            Foot_pos = self.T_05_left_function(theta1, theta2, theta3) 
            Jacobian = self.Jacobian_left_function(theta1, theta2, theta3) 
        
        return Jacobian, Foot_pos
    
    
    def transformWorldtoHip_foot(self, hw_vector_pos, coord_to_transform):
        """"
        This fucntion transforms the specified foot positions into positions that can be compared with the world co-ord foot positions 
        It takes in the hip-world rotation matrix, the hip-world vector and the actual end-effector position
        It returns the actual end-effector (foot) position in hip-coords
        """

        foot_pos = np.array(coord_to_transform) - hw_vector_pos 

        return foot_pos
    
    def transformWorldtoHip_velocity(self, hw_vector_vel, vel_to_transform):
        """
        This fucntion transforms the specified foot velocities into velocities that can be compared with the world co-ord foot velocity 
        It takes in the hip-world rotation matrix, the hip-world vector and the actual end-effector velocity
        It returns the actual end-effector velocity in hip-coords
        """

        foot_vel = np.array(vel_to_transform) - hw_vector_vel 

        return foot_vel
    


    def calcTorque(self, Jacobian, desired_foot_pos, actual_foot_pos, actual_foot_vel):
        """
        This function implements PD control to determine motor torque
        It takes in the Jacobian, desired foot position (RL), actual foot position and actual foot velocity
        The desired foot positions are from RL and the actual ones are from the simulation environment 
        it returns the required motor torque
        Note: The actual and desired foot position, and the foot velocity must be in the same co-ord frame as the Jacobian 
        """

        Kp_matrix = np.matrix([[self.Kp, 0 , 0], [0, self.Kp, 0], [0, 0 , self.Kp]])
        Kd_matrix = np.matrix([[self.Kd, 0 , 0], [0, self.Kd, 0], [0, 0 , self.Kd]])

        pos_error = np.reshape(np.matrix([desired_foot_pos]), (3,1)) - np.reshape(np.matrix([actual_foot_pos]), (3,1))
        torque = (np.matrix.transpose(Jacobian))*(Kp_matrix*(pos_error) - Kd_matrix*(np.reshape(np.matrix([actual_foot_vel]), (3,1))))

        # Get torque to more usable format
        torque = np.array(torque)
        for j in range(3):
            torque[j] = torque[j]  
        
        return torque


    def returnJacobian(self, end_effector_pos=[0, 0, -0.33], leg_ID = 1):
        """
        This function simplifies the usage of the various functions.
        It calculates the IK and then uses those values to get the Jacbian and EE pos 
        This function creates inputs for the Torque() function
        """ 

        angles = self.inverseKinematics(end_effector_pos, leg_ID)
        Jacobian, ee_foot_pos = self.calcJacobian(angles, leg_ID)

        return angles, Jacobian, ee_foot_pos
    
        
    def Torque(self, actual_hip_pos, actual_foot_pos, actual_hip_vel, actual_foot_vel, Jacobian, ee_foot_pos):
        """
        This function simplifies the usage of the various functions
        It transforms the actaul foot pos into hip co-ords. 
        PD control is then implemented to determine the torque  s
        """ 

        actualFootPos_hipCoords = self.transformWorldtoHip_foot(actual_hip_pos, actual_foot_pos)
        actualFootVel_hipCoords = self.transformWorldtoHip_velocity(actual_hip_vel, actual_foot_vel)

        joint_torque = self.calcTorque(Jacobian, ee_foot_pos, actualFootPos_hipCoords, actualFootVel_hipCoords)

        return joint_torque
    
    
    def scaleTorque(self, torque, joint_velocity):
        """
        This function scales the torque so it does not exceed the max value
        It takes in the torque and returns the scaled torque 
        Uses a linear motor model to determine the torque limits for a given angular velocity
        the torque and velocity should include all the torque and velocity for one leg 
        """
        
        for i in range(len(torque)):
            lower_limit, upper_limit = self.motor_model.compute_torque_limits(joint_velocity[i])
            
            if torque[i] > upper_limit: 
                torque[i] = upper_limit
            elif torque[i] < lower_limit:
                torque[i] = lower_limit

        return torque


    def scaleLimits(self, action, x_lim, y_lim, z_lim, leg_ID):
        """
        This function is used to scale a normalised input that is between -1 and 1 
        to a value between the limits of each achievable foot position
        achievable foot positions are in terms of the hip co-ord system 
        """

        x_min, x_max = x_lim[0], x_lim[1]
        z_min, z_max = z_lim[0], z_lim[1]
        scaled_action = [0]*len(action)

        for i in range(4):

            scaled_action[i*3] =  (((action[i*3] + 1)*(x_max - x_min))/2) + x_min 

            if leg_ID[i] in [1, 3]:
                y_max, y_min = y_lim[0], -y_lim[1]
                scaled_action[(i*3) + 1] = (((action[(i*3)+ 1] + 1)*(y_max - y_min))/2) + y_min

            elif leg_ID[i] in [2, 4]:
                y_max, y_min = y_lim[1], y_lim[0]
                scaled_action[(i*3) + 1] = (((action[(i*3) + 1] + 1)*(y_max - y_min))/2) + y_min

            scaled_action[(i*3) + 2] = (((action[(i*3) + 2] + 1)*(z_max - z_min))/2) + z_min

        
        return scaled_action


    def normaliseLimits(self, action, x_lim, y_lim, z_lim, leg_ID):
        """
        This function is used to scale an input in the form [xr, yr, zr, xl, yl, zl, xr, yr, zr, xl, yl, zl]
        to a normalised input that is between -1 and 1  
        """
        
        x_min, x_max = x_lim[0], x_lim[1]
        z_min, z_max = z_lim[0], z_lim[1]
        scaled_action = [0]*len(action)

        for i in range(4):

            scaled_action[i*3] =  (((action[i*3] - x_min)*2)/(x_max - x_min)) - 1 

            if leg_ID[i] in [1, 3]:
                y_max, y_min = y_lim[0], -y_lim[1]
                scaled_action[(i*3) + 1] = (((action[(i*3) + 1] - y_min)*2)/(y_max - y_min)) - 1 

            elif leg_ID[i] in [2, 4]:
                y_max, y_min = y_lim[1], y_lim[0]
                scaled_action[(i*3) + 1] = (((action[(i*3) + 1] - y_min)*2)/(y_max - y_min)) - 1 

            scaled_action[(i*3) + 2] = (((action[(i*3) + 2] - z_min)*2)/(z_max - z_min)) - 1 
        
        return scaled_action

    