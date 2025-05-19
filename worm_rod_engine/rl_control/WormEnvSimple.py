import numpy as np
import os
from scipy.spatial.transform import Rotation as R

# from scipy.io import loadmat
# from scipy.interpolate import splprep, splev

import gymnasium as gym
from gymnasium.spaces import Box

from envs.mujoco.WormEnv import WormEnv
#from functions.functions_worm import fit_plane_to_point_cloud, get_principal_plane_rotation, interpolate_line
from worm_rod_engine.rl_control.natural_frame_1 import (NaturalFrame, distance_ab, align_complex_vectors)

# Import the simple-worm simulator here
from worm_rod_engine.worm import Worm
from worm_rod_engine.parameter.output_parameter import output_parameter_parser
from worm_rod_engine.parameter.numerical_parameter import numerical_argument_parser
from worm_rod_engine.parameter.dimensionless_parameter import dimensionless_parameter_parser
from worm_rod_engine.frame import Frame

class WormEnvSimple(gym.Env, WormEnv):
    
    def __init__(
        self,
        P=None,
        record=False,
        **kwargs,
    ):
        
        self.P = P
        self.record = record
        
        observation_space = Box(low=self.P["obs"]["range"][0], high=self.P["obs"]["range"][1], shape=(self.P["obs_space"],), dtype=np.float64)
        
        # Load reference midlines
        if(self.P["ref"] and (self.P["gait"] == "infinity" or self.P["gait"] == "coiling")):
            gait = self.P["gait"] + '_' + self.P["chirality"]
            self.ref_data = loadmat(self.P["reward"]["ref_data"][gait]) # 3d array of 3d coordinates [T, N, 3].

        # Initialize the worm simulator
        output_param = output_parameter_parser.parse_args(['--k', str(True)])
        numerical_param = numerical_argument_parser.parse_args(['--dt', str(self.P["dt"]), '--N', '128']) # , '--dt_report', '1e-2', '--N_report', '128'
        dimensionless_param = dimensionless_parameter_parser.parse_args()
        self.worm = Worm(numerical_param=numerical_param, dimensionless_param=dimensionless_param, output_param=output_param)

    def step(self, action):
        
        # action.shape = 3*N # twisting, bending DV, bending LR.

        self.increment_time() # This increments self.t, self.frame (the frame of reference experimental data) and self.phase (the phase of synthetically generated reference data).
        
        # self.update_midlines()

        k0 = np.array(action).reshape((3, self.worm.N))
        self.worm.update_state(k0=k0, assemble=True) # Update the state of the worm.
        
        observation = self._get_obs(self.worm.assembler.output)
        
        reward, terminated = self.get_reward()
        
        info = {"sim_midline": self.worm.assembler.output["r"], "ref_midline": self.ref_midline, "ref_frame": self.frame} # TODO.
        
        return observation, reward, terminated, False, info
    
    def reset_model(self):
        
        self.t = 0
        # self.phase = 0
        self.frame = np.random.randint(1, self.P["reward"]["ref_data"]["ref_random_pose_num"]) # Select a random ref frame.
        self.frame_step = 0

        # Initialize the simulated midline to match a reference midline in the sequence
        ref_midline = self.ref_data["XYZ"][self.frame]
        # ref_midline_previous = self.ref_data["XYZ"][self.frame-1]
        ref_nf = NaturalFrame(ref_midline)
        # ref_prev_nf = NaturalFrame(ref_midline_previous)

        # velocity = (ref_midline - ref_midline_previous) / self.P["dt_exp"]
        # ref_midline_xxx = ref_midline - (velocity * self.P["dt"]) # Interpolated previous midline using the physics time step.
        '''
        ref_nf.M1
        ref_nf.M2
        ref_nf.psi
        '''

        # Initialize the frame with the reference midline
        F0 = Frame(
            r=ref_midline.T,
            d1=ref_nf.M1,
            d2=ref_nf.M2,
            d3=ref_nf.T,
            t=(self.frame * self.P["dt_exp"]) # self.t/self.P["frame_skip"]
        )
        
        F0.euler_angles_from_body_frame()

        # F1 = None # create a frame for the previous midline.
        # F = None # create a frame sequence with F1 and F0.
        
        self.worm.initialise(F0=F0) # , F_arr_past=F

        # self.worm.assembler.output["r"] # midline.
        # self.worm.assembler.output["r_t"] # time derivative of r.
        
        observation = self._get_obs(self.worm.assembler.output)
        
        # self.update_midlines() # Initialize simulated and reference midlines [Np x 3] and their principal plane normal
        # self.plane_normal_prev = fit_plane_to_point_cloud(self.sim_midline)
        # self.plane_normal_ref_prev = fit_plane_to_point_cloud(self.ref_midline)
        
        # TODO:
        # Override the initial simulated midline. Match its shape with the shape of the initial reference midline (self.frame).
        # Make sure to adjust the initial frame such that DV takes most of the curvature.
        
        return observation
    
    def _get_obs(self, model_output):
        
        obs = np.concatenate((model_output["r"], model_output["r_t"])) # Midline and midline velocity.
        # obs = np.concatenate((model_output["k"], model_output["k"])) # Curvature and change in curvature.
        # obs = (self.frame) # Frame number.
        
        return obs
    
    def update_midlines(self):
        # Initialize simulated and reference midlines [Np x 3] and their principal plane normal
        
        if(self.P["ref"] and (self.P["gait"] == "infinity" or self.P["gait"] == "coiling")):
            self.ref_midline = interpolate_line(self.ref_data["XYZ"][self.frame], self.P["Np"]+1)
            self.ref_midline = np.array(self.ref_midline).T
        elif(self.P["ref"] and (self.P["gait"] == "2d_sine" or self.P["gait"] == "3d_sine")):
            y_dv, y_lr, x = self.get_ref_midline() # Np+1 points.
            self.ref_midline = np.array([x, y_dv, y_lr]).T
        
        # self.sim_midline = ??? # TODO.

    def get_reward(self, model_output):
        reward = (np.sum(model_output["r"] - self.ref_data["XYZ"][self.frame]) ** 2)**0.5 # Shape difference.

        return reward