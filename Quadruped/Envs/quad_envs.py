"""
Environment based on OpenAI, Gymnasium conventions 
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p

from Quadruped.Resources.plane import PLANE
from Quadruped.Resources.uneven_terrain import unevenPLANE
from Quadruped.Resources.BoxyTerrain import RandomStepstoneScene
from Quadruped.Resources.quadruped import walkingQUAD

from utils import writingCSVs
from math import exp
from random import uniform

class QuadEnv(gym.Env):
    metadata =  {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: str = "rgb_array", render_fps: int = 60, plot_eval_data: bool = False,
                 target_lin_vel_range:  list = [1, 1], friction_range:  list = [0.5, 1], uneven_terrain: bool = True, boxy_terrain = True,
                 r_velocity = 1, r_energy = 0.05, r_alive = 1, r_height = 1, r_straight = 20, r_orientation = 10, r_foot_contact = 0.2,
                 Kp = 500, Kd = 10, Kp_damping = 1, seed = None, make_gif = False, render = False, turning = False):

        # Action limits(normalised): 
        # Includes the actual positions of the feet in terms of the hip co-ord frame (12)
        a_limits = np.array([1.0]*12, dtype=np.float32)

        # Observation limits:
        # 1) - The X, Y and Z POSITION of the Body (3) 
        # 2) - The X, Y, z and W Orientation of the body (4) 
        # 3) - The X, Y and Z linear VELOCITY of the body (3) 
        # 4) - Angular velocity of body (3)
        # 5) - Angular POSITION of knee and Hip, calf joints (12)
        # 6) - Angular VELOCITY of knee and hip, calf joints (12)
        # 7) - Cartesian POSITION of the feet (12)
        # 8) - Cartesain VELOCITY of the feet (12)
        # 9) - The Action from the previous step (12)
            
        self.Z_Pos = [0.45]
        self.XYZ_Ori = [1]*4
        #self.XYZ_Ori = [10]*2 # removed yaw as it drifts quickly and does not add to the task 
        self.XYZ_Vel = [21]*3 
        self.XYZ_Ang_Vel = [100]*3 
        self.joint_pos_lim = [10]*12
        self.joint_vel_lim = [52.4]*12
        self.foot_pos_lim = [0.4]*12
        self.foot_vel_lim = [21]*12
        #self.foot_contact_bool = [1]*4
        self.prev_action_lim = [1]*12
        #self.current_action_lim = [1]*12 # Current Action to add or not to add? 

        o_limits = (self.Z_Pos, self.XYZ_Ori, self.XYZ_Vel, self.XYZ_Ang_Vel, self.joint_pos_lim, self.joint_vel_lim, self.foot_pos_lim, self.foot_vel_lim, self.prev_action_lim)
        o_limits = np.concatenate(o_limits)

        # Action Space
        self.action_space = gym.spaces.box.Box(-a_limits, a_limits)

        # Observation Space
        self.observation_space = gym.spaces.Box(-o_limits, o_limits, dtype=np.float32)

        # Initialise useful variables
        self.quadId = None
        self.planeId = None
        self.done = None
        self.truncated = False
        self.foot_contact_counter = [0]*4
        self.foot_contact_counter_storage = [0]*4

        # Train turning or not 
        self.turning = turning
        
        # Reward 
        self.r_velocity = r_velocity
        self.r_energy = r_energy 
        self.r_alive = r_alive
        self.r_height = r_height
        self.r_straight = r_straight
        self.r_orientation = r_orientation
        self.r_foot_contact = r_foot_contact
        self.total_reward_per_episode = 0 

        # Limits and targets
        self.Kp = Kp
        self.Kd = Kd
        self.Kp_damping = Kp_damping
        self.target_lin_vel_range= target_lin_vel_range # desired linear velocity in paper was between 0.7 - 4 m/s
        self.friction_range = friction_range
        self.currentSimTime  = 0 
        self.totalSimTime = 0 

        # Rendering 
        self.rendering = render
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.rendered_img = None
        self.frame_no = None
        self.make_gif = make_gif
        self.uneven_terrain = uneven_terrain
        self.boxy_terrain = boxy_terrain

        # For graphing 
        self.writingCSV = writingCSVs()
        self.write_to_cvs = []
        self.csvHeadings = ["r1 - vel", "r2 - ori", "r3 - energy", "r4 - height", "r5 - alive", "reward", "done"]
        
        # For graphing evaluation data 
        self.plot_eval_data = plot_eval_data
        self.evaluationCSV = writingCSVs()
        self.write_to_eval_csv = []
        
        # Sim platform parameters:
        self.num_action_repeat = 10
        self.num_bullet_solver_iterations = int(300 /self.num_action_repeat)
        self.sim_timestep = 0.001
        self.max_episode_len = 10 / (self.sim_timestep * self.num_action_repeat) # 10 because we want it to walk for 10 seconds 

        self.client = p.connect(p.DIRECT)
        self.hardReset()
        self.reset(seed)


    def step(self, action):
        """
        Run one timestep of the environments dynamics using the agent actions.
        One timestep runs for a total of 0.0125 seconds or 125Hz. 
        """

        # --- Apply Action ---
        # actions are aranged [x, y, z] [RF, LF, RB, LB]
        # Action scaling is applied in the quadruped.py 
        
        for m in range(self.num_action_repeat):
            
            current_torque = self.quadId.applyAction(action, m)             
            p.stepSimulation(physicsClientId = self.client) 

        # --- Determine Observations ---
        body_pos_full, body_ori, body_lin_vel, body_ang_vel, joint_pos, joint_velocity, foot_pos, foot_vel, self.previous_action = self.quadId.getObservations()
        body_pos = body_pos_full[-1]
        
        #roll, pitch, yaw = body_ori[0], body_ori[1], body_ori[2]
        roll_pitch_yaw = p.getEulerFromQuaternion(body_ori)
        roll, pitch, yaw = roll_pitch_yaw[0], roll_pitch_yaw[1], roll_pitch_yaw[2]
        body_ori_full = body_ori
        #body_ori = body_ori[:2] # remove yaw as it drifts quickly and does not add to the task

        # Contact Booleans for Eval Plotting 
        if self.plot_eval_data:

            contact_booleans, normal_force = self.quadId.getContact()

            for i in range(len(contact_booleans)):
                if contact_booleans[i] == 1: # contact has been made
                    self.foot_contact_counter_storage[i] = self.foot_contact_counter[i]*(self.sim_timestep * self.num_action_repeat)
                    self.foot_contact_counter[i] = 0
                else:
                    self.foot_contact_counter[i] += 1
                    
        # Get contact information 
        contact_booleans, normal_force = self.quadId.getContact() 
        
        
        # --- Reward ---
        # Rewards implemented in paper are as follows: 
        # reward = r(velocity) + r(energy) + r(alive)
            
        # moving reward
        r1 = -(abs(body_lin_vel[0] - self.target_lin_vel))*self.r_velocity 
        #r2 = - self.r_orientation*(abs(body_lin_vel[1]) + abs(body_ang_vel[2]))
        r2 = -np.linalg.norm(body_ori - np.array([0, 0, 0, 1]))*self.r_orientation
        r3 = -np.sum((np.abs(np.reshape(current_torque, (1,12))[0]*joint_velocity))**2)*self.r_energy
        r4 = -(abs(body_pos - 0.28))*self.r_height
        r5 = self.target_lin_vel*self.r_alive
        
        if self.turning and self.currentSimTime == 1500:
            self.BoxID.remove_walls() # remove walls when turing so that can turn nicely. 
            
        if self.turning and self.currentSimTime > 1500:
            
            if self.random_yaw_degrees < 0: 
                self.random_yaw_degrees = self.random_yaw_degrees + 360 # convert to positive angle

            delta_angle = abs(yaw - np.radians(self.random_yaw_degrees))

            if delta_angle == 0:
                delta_angle = 1
            
            r1 = -(abs(body_lin_vel[0] - self.target_lin_vel))*(self.r_velocity*delta_angle) 
            r2 = -np.linalg.norm(body_ori - np.array(self.new_target_ori))*self.r_orientation
            r5 = self.target_lin_vel*self.r_alive*delta_angle

        reward = r1 + r2 + r3 + r4 + r5 
        self.total_reward_per_episode = self.total_reward_per_episode + reward
        
        # === writing to csv file - only done during rendering ===
        if self.rendered_img is not None: # Miss first step of iteration but that is fine
            reward_info = np.array([r1, r2, r3, r4, r5, reward, self.done])
            self.write_to_cvs = self.writingCSV.arrayStorage(reward_info)

        # --- Done Conditions ---
        # The episode should terminate when the following conditions are met:
        # 1) quad is pitching too much 
        # 2) quad is too low 
        # 3) total number of epsiodes has been reached 

        # The pitch must be in the range [-0.4, 0.4]
        # The roll must be in the range [-0.2, 0.2]

        if (pitch < -0.6) or (pitch > 0.6): 
            self.done = True
            reward = reward - 20
            done_cond = "Falling Orientation - p"

        elif (roll < -0.5) or (roll > 0.5): 
            self.done = True
            reward = reward - 20
            done_cond = "Falling Orientation - r"
        
        if self.currentSimTime >= self.max_episode_len:
            self.done = True
            reward = reward # 100 is waaaay too much 
            self.truncated = True
            done_cond = "Episodes Max"

        # Making observations in correct form for Env
        obs = self.concatenateObservations(body_pos, body_ori, body_lin_vel, body_ang_vel, joint_pos, joint_velocity, foot_pos, foot_vel, self.previous_action)
        
        # Increment counters
        self.currentSimTime += 1
        self.totalSimTime += 1 
        
        # Print for easier progress tracking of epsisode
        # if self.done:
        #     print("Done: {}     no. episodes: {}      total reward: {}".format(done_cond, self.currentSimTime, self.total_reward_per_episode))
            
        # Plotting data to be used for evaluating quality of results  
        if self.plot_eval_data:
            eval_info = np.concatenate([np.array([self.currentSimTime*(self.sim_timestep * self.num_action_repeat)]), body_pos_full, roll_pitch_yaw, body_lin_vel, body_ang_vel, joint_pos, joint_velocity, current_torque, contact_booleans, normal_force, np.reshape(np.array(foot_pos, dtype=np.float32),(1, 12))[0], np.reshape(np.array(self.quadId.getFeetLinearVelocity(), dtype=np.float32),(1, 12))[0]])
            
            self.write_to_eval_csv = self.evaluationCSV.arrayStorage(eval_info)
           
            
        return obs, reward, self.done, self.truncated, dict() 


    def seed(self, seed=None):
        self.seed_generator, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    
    def reset(self, seed = None, hard_reset = False):
        """
        Resets the environment to an initial internal state, returning an initial observation and info.
        Has randomness to ensure that the agent explores the state space and learns a generalised policy about the environment.
        This randomness can be controlled with the seed parameter otherwise if the environment already has a random number generator 
        and reset() is called with seed=None, the RNG is not reset.

        MUST CALL HARD RESET BEFORE RESET IF WANT TO CHANGE RENDER_MODE
        """
        # super().reset(seed=seed)
        
        # Change Connection with Sim Platform
        # This functionality is mainly useful when rendering 
        # The client should not be switched to GUI during training 
        # as it will slow it down significantly

        if hard_reset: 
            self.hardReset()
        
        # choose random friction between 0.5 and 1.2
        self.friction = uniform(self.friction_range[0], self.friction_range[1])
        self.target_lin_vel = uniform(self.target_lin_vel_range[0], self.target_lin_vel_range[1])
        
        self.max_motor_strenght = uniform(28.5, 33.5) 
        self.add_mass = uniform(0, 0.5)

        # Reset plane and Quad robot
        if self.uneven_terrain and self.planeId is not None:
            
            if self.boxy_terrain:
                self.BoxID.reset(self.client) 
                self.planeId.changeFriction(self.friction, self.client)
                self.quadId.resetPos()
                self.quadId.resetParams(max_torque = self.max_motor_strenght, add_mass = self.add_mass)
            else: 
                if self.currentSimTime < 300 and not self.rendering: 
                    self.planeId.removeTerrain(self.client)
                    self.planeId = unevenPLANE(self.client, friction_coeff = self.friction, terrain_size = [20, 40])
                elif self.rendering:
                    self.planeId.removeTerrain(self.client)
                    self.planeId = unevenPLANE(self.client, friction_coeff = self.friction, terrain_size = [20, 80])
                else: 
                    self.planeId.removeTerrain(self.client)
                    self.planeId = unevenPLANE(self.client, friction_coeff = self.friction, terrain_size = [20, 80])
               
                foot_heights = self.planeId.getFootHeights()
                self.quadId.resetParams(max_torque = self.max_motor_strenght, add_mass = self.add_mass)
                self.quadId.resetPos(uneven_terrain=True, foot_height = foot_heights)
        else: 
            self.quadId.resetParams( max_torque = self.max_motor_strenght, add_mass = self.add_mass)
            self.quadId.resetPos()
            self.planeId.changeFriction(self.friction, self.client)

        # Reset Variables 
        self.done = False
        self.truncated = False
        self.currentSimTime = 0
        self.frame_no = 0
        self.total_reward_per_episode = 0 

        if self.turning:
            self.random_yaw_degrees = uniform(-180, 180)
            self.new_target_ori = self.euler_to_quaternion(self.random_yaw_degrees, 0, 0)

        # Get Observations
        body_pos_full, body_ori, body_lin_vel, body_ang_vel, joint_pos, joint_velocity, foot_pos, foot_vel, self.previous_action = self.quadId.getObservations()
        
        body_pos = body_pos_full[-1]
        roll_pitch_yaw = p.getEulerFromQuaternion(body_ori)
        body_ori_full = body_ori
        
        #body_ori = body_ori[:2]
        
        # Store in array for evaluation
        if self.plot_eval_data:
            contact_booleans, normal_force = self.quadId.getContact()
            
            eval_info = np.concatenate([np.array([self.currentSimTime]), body_pos_full, roll_pitch_yaw, body_lin_vel, body_ang_vel, joint_pos, joint_velocity, 
                                       [0]*12, contact_booleans, normal_force, np.reshape(np.array(foot_pos, dtype=np.float32), (1, 12))[0], np.reshape(np.array(self.quadId.getFeetLinearVelocity(), dtype=np.float32),(1, 12))[0]])
            
            self.write_to_eval_csv = self.evaluationCSV.arrayStorage(eval_info)
        
        # Making observations in correct form for Env
        obs = self.concatenateObservations(body_pos, body_ori, body_lin_vel, body_ang_vel, joint_pos, joint_velocity, foot_pos, foot_vel, self.previous_action)
        
        return obs, dict()
    
    def hardReset(self):
        """
        Resets the environment to an initial internal state, returning an initial observation and info.
        Key difference between this and reset is that this function reloads all assests to do a full rest of the environment 
        Has randomness to ensure that the agent explores the state space and learns a generalised policy about the environment.
        This randomness can be controlled with the seed parameter otherwise if the environment already has a random number generator 
        and reset() is called with seed=None, the RNG is not reset.
        """
        # super().reset(seed=seed)
        
        # Change Connection with Sim Platform
        # This functionality is mainly useful when rendering 
        # The client should not be switched to GUI during training 
        # as it will slow it down significantly
        if self.render_mode == "human": 
            p.disconnect()
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
           
        # Reset Sim Software
        p.resetSimulation(self.client)
        p.setPhysicsEngineParameter(numSolverIterations=self.num_bullet_solver_iterations)
        p.setTimeStep(self.sim_timestep, self.client) 
        p.setGravity(0,0,-9.81)
        
        # choose random friction between 0.5 and 1.2
        self.friction = uniform(self.friction_range[0], self.friction_range[1])
        self.target_lin_vel = uniform(self.target_lin_vel_range[0], self.target_lin_vel_range[1])
        
        self.max_motor_strenght = uniform(28.5/2, 33.5/2) # divide by 2 to reduce the strength of the motors 
        self.add_mass = uniform(-0.9426, 0.9426)

        # Reload plane and Quad robot
        self.quadId = walkingQUAD(self.client, max_torque = self.max_motor_strenght, add_mass = self.add_mass,
                                Kp = self.Kp, Kd = self.Kd, Kp_damping = self.Kp_damping)
        if self.uneven_terrain and not self.boxy_terrain:
            self.planeId = unevenPLANE(self.client, self.friction)
        elif not self.uneven_terrain and not self.boxy_terrain:
            self.planeId = PLANE(self.client, self.friction)
        else: 
            self.planeId = PLANE(self.client, self.friction)
            self.BoxID = RandomStepstoneScene(num_stones=40, stone_height=0.0, stone_width_lower_bound=0.1, 
                                                stone_width_upper_bound=1.0, stone_length_lower_bound=0.1, stone_length_upper_bound=1.0, 
                                                gap_length_lower_bound=0.0, gap_length_upper_bound=0.0, height_offset_lower_bound=0.005, 
                                                height_offset_upper_bound=0.013, floor_height_lower_bound=0.0, floor_height_upper_bound=0.0, 
                                                platform_length_lower_bound=0.4, platform_length_upper_bound=0.5, total_obstacle_length = 10, 
                                                total_obstacle_width = 1, rebuild_scene_during_reset=True)

    def render(self):
        """
        Compute the render frames as specified by render_mode during the initialization of the environment.
        if render mode is human then a higher quality rendering will be produced however, if the render mode
        is anything else, it will default to the rgb_array rendering method which is lower quality and pixilated 
        """
        
        if self.render_mode == "human": 
            self.rendered_img = True # set to anything other than none so that can plot array 
            base_pos, _, _, _= self.quadId.getBasePosAndVel()
            
            p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw = 30, cameraPitch = -25, cameraTargetPosition =  base_pos)
            rgb_array = []
            
            pos = np.array(base_pos)
            pos[-1] = 0.3 
            pos.tolist()
            
            if self.make_gif: 
                view_matrix = p.computeViewMatrixFromYawPitchRoll(
                                cameraTargetPosition=pos, 
                                distance=2, 
                                yaw=30, 
                                pitch=-25,
                                roll=0, upAxisIndex=2)
                
                proj_matrix = p.computeProjectionMatrixFOV(
                                fov=60,
                                aspect=float(1920) / 1080, 
                                nearVal=0.1, 
                                farVal=10.0)
                
                (_, _, px, _, _) = p.getCameraImage( 
                                    width=1920, 
                                    height=1080,
                                    renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                    viewMatrix=view_matrix,
                                    projectionMatrix=proj_matrix)
                
                frame = p.getCameraImage(1000, 1000, view_matrix, proj_matrix)[2]
                rgb_array = frame[:, :, :3]
            
            return rgb_array, self.write_to_cvs, self.csvHeadings, self.write_to_eval_csv
        
        else:
            # Display image
            if self.rendered_img is None:
                self.rendered_img = plt.imshow(np.zeros((1000, 1000, 4)))

            # This code sets the correct view
            view_matrix = p.computeViewMatrix(
                        cameraEyePosition=[3, -2, 1],
                        cameraTargetPosition=[1, 0, 0],
                        cameraUpVector=[0, 0, 1])
        
            projection_matrix = p.computeProjectionMatrixFOV(
                        fov=50.0,
                        aspect=1,
                        nearVal=0.1,
                        farVal=10.1)

            # for slow renering use 1000
            frame = p.getCameraImage(1000, 1000, view_matrix, projection_matrix)[2]
            frame = np.reshape(frame, (1000, 1000, 4))

            # Actually drawing the position 
            self.rendered_img.set_data(frame)
            plt.title("Frame Number: " + str(self.frame_no))
            plt.draw()
            plt.pause(1/10000) 

            self.frame_no += 1
            rgb_array = frame[:, :, :3]

            return rgb_array, self.write_to_cvs, self.csvHeadings, self.write_to_eval_csv


    def close(self):
        p.disconnect(physicsClientId = self.client)

    
    def concatenateObservations(self, body_pos, body_ori, body_lin_vel, body_ang_vel, joint_pos, joint_velocity, foot_pos, foot_vel, previous_action):
        # Neaten up rest of code by having one function the performs concatenation of observations 

        body_pos = np.array([body_pos], dtype=np.float32)
        body_ori = np.array(body_ori, dtype=np.float32)
        body_lin_vel = np.array(body_lin_vel, dtype=np.float32)
        body_ang_vel = np.array(body_ang_vel, dtype=np.float32)
        joint_pos = np.array(joint_pos, dtype=np.float32)
        joint_velocity = np.array(joint_velocity, dtype=np.float32)
        foot_pos = np.reshape(np.array(foot_pos, dtype=np.float32), (1, 12))[0]
        foot_vel = np.reshape(np.array(foot_vel, dtype=np.float32), (1, 12))[0]
        #foot_contact_bool = np.array(foot_contact_bool, dtype=np.float32)
        previous_action = np.reshape(np.array(previous_action, dtype=np.float32), (1, 12))[0]

        formatted_obs = np.concatenate((body_pos, body_ori, body_lin_vel, body_ang_vel, joint_pos, joint_velocity, foot_pos, foot_vel, previous_action))

        return formatted_obs

    def euler_to_quaternion(self, yaw, pitch, roll):
        # Convert angles from degrees to radians
        yaw = np.radians(yaw)
        pitch = np.radians(pitch)
        roll = np.radians(roll)

        # Compute half angles
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        # Compute quaternion components
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        w = cr * cp * cy + sr * sp * sy

        quaternion = np.array([w, x, y, z])
    
        # Normalize the quaternion
        magnitude = np.linalg.norm(quaternion)
        normalized_quaternion = quaternion / magnitude
        
        return normalized_quaternion
        # example usuage: 
        # yaw = 0  # Variable yaw angle in degrees
        # pitch = 0  # Flat pitch
        # roll = 0   # Flat roll

        # quaternion = euler_to_quaternion(yaw, pitch, roll)
        # print(f"Quaternion: (x, y, z, w) = {quaternion}")