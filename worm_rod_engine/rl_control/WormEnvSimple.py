import numpy as np
# import os
# from scipy.spatial.transform import Rotation as R

# from scipy.io import loadmat
# from scipy.interpolate import splprep, splev

import gymnasium as gym
from gymnasium.spaces import Box

from worm_rod_engine.rl_control.worm_functions import fit_plane_to_point_cloud, get_principal_plane_rotation, interpolate_line
from worm_rod_engine.rl_control.natural_frame_1 import (NaturalFrame, distance_ab, align_complex_vectors)

# Import the simple-worm simulator here
from worm_rod_engine.worm import Worm
from worm_rod_engine.parameter.output_parameter import output_parameter_parser
from worm_rod_engine.parameter.numerical_parameter import numerical_argument_parser
from worm_rod_engine.parameter.dimensionless_parameter import dimensionless_parameter_parser
from worm_rod_engine.frame import Frame

class WormEnvSimple(gym.Env):
    
    def __init__(
        self,
        P=None,
        record=False,
        **kwargs,
    ):

        self.P = P
        self.record = record
        self.episode_length = 50

        N = 50  # number of points along each midline
        action_size = 3 * N

        # Define action space as a continuous Box
        self.action_space = gym.spaces.Box(
            low=--10,  # Minimum value for each action
            high=+10,  # Maximum value for each action
            shape=(action_size,),  # Flattened shape (will be reshaped to (3, N) in step)
            dtype=np.float32  # Data type for actions
        )

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.P["obs_space"],), dtype=np.float64)
        
        # Load reference midlines
        # if(self.P["ref"] and (self.P["gait"] == "infinity" or self.P["gait"] == "coiling")):
        #     gait = self.P["gait"] + '_' + self.P["chirality"]
        #     self.ref_data = loadmat(self.P["reward"]["ref_data"][gait]) # 3d array of 3d coordinates [T, N, 3].
        if True:
            # t = np.linspace(0, 2 * np.pi, 128)
            #
            # x = t
            # y = np.sin(t)
            # z = np.zeros_like(t)
            # self.ref_data = np.stack((x, y, z), axis=-1) # 3D coordinates of a 2D sine function.

            # parameters
            L = 1.0  # length of midline in X
            A = 0.1  # sine amplitude (in Y)

            lam = 1.0
            q = 2 * np.pi / lam
            freq = 1.0

            dt = 0.04
            T = 5
            t_arr = np.arange(0, T+0.1*dt, dt)

            self.frame_count = len(t_arr)

            # sample points along the midline (head-to-tail)
            s = np.linspace(0, L, N)  # shape (N,)

            # prepare empty array: (frames, coords, points)
            midlines = np.zeros((len(t_arr), 3, N))

            for f, t in enumerate(t_arr):
                phase = 2 * np.pi * freq * t

                # compute time‐dependent phase for this frame
                for i in range(N):
                    xi = s[i]
                    yi = A * np.sin(q * xi + phase)
                    zi = 0.0  # flat in Z; change if you want a 3D wiggle
                    # store into the array
                    midlines[f, 0, i] = xi  # X
                    midlines[f, 1, i] = yi  # Y
                    midlines[f, 2, i] = zi  # Z

                diffs = np.diff(midlines[f], axis=-1)  # vectors between points
                seg_lengths = np.sqrt((diffs ** 2).sum(axis=0))  # length of each segment
                total_len = seg_lengths.sum()  # sum of all segments
                midlines[f] /= total_len
                t += dt

        self.ref_data = midlines

        # Initialize the worm simulator
        output_param = output_parameter_parser.parse_args(['--k', str(True), '--k_t', str(True)])
        numerical_param = numerical_argument_parser.parse_args(['--dt', str(dt), '--N', str(N)]) # , '--dt_report', '1e-2', '--N_report', '128' str(self.P["dt"])
        dimensionless_param = dimensionless_parameter_parser.parse_args()
        self.worm = Worm(numerical_param=numerical_param, dimensionless_param=dimensionless_param, output_param=output_param)

        action_size = 3*N

    def step(self, action):

        # self.increment_time() # This increments self.t, self.frame (the frame of reference experimental data) and self.phase (the phase of synthetically generated reference data).
        
        # self.update_midlines()

        k0 = np.array(action).reshape((3, self.worm.N))
        self.worm.update_state(k0=k0, assemble=True) # Update the state of the worm.
        
        observation = self._get_obs(self.worm.assembler.output)
        
        reward, terminated = self.get_reward()
        
        info = {"sim_midline": self.worm.assembler.output["r"], "ref_midline": self.ref_data[self.frame], "ref_frame": self.frame} # TODO.
        
        return observation, reward, terminated, False, info
    
    def reset_model(self):
        
        self.t = 0
        # self.phase = 0
        #self.frame = np.random.randint(1, self.P["reward"]["ref_data"]["ref_random_pose_num"]) # Select a random ref frame.

        self.frame = np.random.randint(0, self.frame_count - self.episode_length)

        # self.frame_step = 0

        # Initialize the simulated midline to match a reference midline in the sequence
        #ref_midline = self.ref_data["XYZ"][self.frame]

        # ref_midline_previous = self.ref_data["XYZ"][self.frame-1]
        ref_nf = NaturalFrame(self.ref_data[self.frame])


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
            r=ref_nf.X.T,
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
        
        #obs = np.concatenate((model_output["r"], model_output["r_t"])) # Midline and midline velocity.
        obs = np.concatenate((model_output["k"], model_output["k_t"])) # Curvature and change in curvature.
        # obs = (self.frame) # Frame number.
        
        return obs
    
    def update_midlines(self):
        # Initialize and interpolate simulated and reference midlines [Np x 3] and their principal plane normal
        
        if(self.P["ref"] and (self.P["gait"] == "infinity" or self.P["gait"] == "coiling")):
            self.ref_midline = interpolate_line(self.ref_data["XYZ"][self.frame], self.P["Np"]+1)
            self.ref_midline = np.array(self.ref_midline).T

    def get_reward(self, model_output):
        
        terminated = False
        
        reward = 0.1

        # reward += -1 * self.get_3d_midline_reward(self.ref_data["XYZ"][self.frame], model_output["r"])
        reward += -1 * self.get_natural_frame_reward(self.ref_data["XYZ"][self.frame], model_output["r"])
        # reward += -1 * self.get_principal_plane_reward(self.ref_data["XYZ"][self.frame], model_output["r"])
        
        # Early termination
        if(0):
            terminated = True

        return reward, terminated
    
    def get_natural_frame_reward(self, ref_midline, sim_midline):
        # Calculate shape difference
        n = self.P["reward"]["ref_data"]["n_skip"]
        ref_nf = NaturalFrame(ref_midline[n:-n]) # [n:-n].
        sim_nf = NaturalFrame(sim_midline[n:-n]) # [n:-n].
        shape_error = distance_ab(ref_nf.mc, sim_nf.mc)

        return shape_error
    
    def get_3d_midline_reward(self, ref_midline, sim_midline):
        return (np.sum(sim_midline - ref_midline) ** 2)**0.5
    
    def get_principal_plane_reward(self, ref_midline, sim_midline):
        # Calculate principal plane rotation error
        pp_angle_diff, self.plane_normal = get_principal_plane_rotation(sim_midline, self.plane_normal_prev)
        pp_angle_diff_ref, self.plane_normal_ref = get_principal_plane_rotation(ref_midline, self.plane_normal_ref_prev)

        self.plane_normal_prev = self.plane_normal.copy()
        self.plane_normal_ref_prev = self.plane_normal_ref.copy()
        
        return abs(pp_angle_diff - pp_angle_diff_ref) # Add a penalty for the diff between sim and ref plane rotation.

if __name__ == '__main__':

    env = WormEnvSimple()
    print(env.frame_count)

    # import matplotlib.pyplot as plt
    #
    # ax = plt.subplot(111)
    # for midline in env.ref_data:
    #     ax.plot(midline[0, :], midline[1, :], '-o')
    #
    # plt.show()
    #




