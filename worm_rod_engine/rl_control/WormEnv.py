import numpy as np
from functions.frames.natural_frame_1 import NaturalFrame, distance_ab # , align_complex_vectors
from functions.functions_worm import get_principal_plane_rotation

class WormEnv():
    def __init__():
        A0 = 0.6 # [*0.6*, 1]. TODO: move to params.
        self.F = lambda A, w, x, phase, offset: (A * np.sin((w * x) + phase)) + offset
        self.S = lambda x: 1 / (1 + np.exp(0.5 * np.pi * (x - 0.35 * 2 * np.pi)))
    
    def increment_time(self):
        self.t += self.P["dt_opt"]
        self.phase -= self.P["dt_opt"] * (2*np.pi) / self.P["T"] # [0,T] => [0,2π]. The phase only affects synthetically-generated references midlines.
        
        if(self.P["frame_skip_data"] > 1):
            self.dt_frame += 1
            if(self.t > 0 and self.dt_frame == self.P["frame_skip_data"]): # 0.04 = 4 * 0.01(=dt_opt) is the time step of the reference data.
                self.frame += 1 # Increment reference frame.
                self.dt_frame = 0 # Reset the sub-sequence frame.
        else:
            self.frame += 1 # Increment reference frame.
        
    def get_reward(self, sim_midline, ref_midline): # , sim_plane_normal, sim_plane_normal_prev, ref_plane_normal, ref_plane_normal_prev
        
        s = 1 # (32 / self.P["Np"])
        step_reward = 0.1 # A constant step reward to make the termination conditions effective.
        shape_diff_weight = 0.01 # [infinity=0.02, coiling=0.02,  2d_sine=0.005].
        pp_angle_diff_weight = 0.3 # infinity=0.3.
        # action_smoothness_weight = 1
        shape_diff_threshold = 20 # [infinity: 12, 2d_sine=15, 3d_sine=25].
        
        # Initialize reward
        reward = step_reward
        terminated = False
        
        # Compute shape and principal plane rewards only when the reference frame changes
        if(self.dt_frame == (self.P["frame_skip_data"]-1)): # Happens only on the last frame in the sub-sequence (every frame_skip_data frames).
            
            # Calculate shape difference
            n = self.P["reward"]["ref_data"]["n_skip"]
            ref_nf = NaturalFrame(ref_midline[n:-n]) # [n:-n].
            sim_nf = NaturalFrame(sim_midline[n:-n]) # [n:-n].
            diff = distance_ab(ref_nf.mc, sim_nf.mc)
            
            # Calculate principal plane rotation
            pp_angle_diff, self.plane_normal = get_principal_plane_rotation(sim_midline, self.plane_normal_prev)
            pp_angle_diff_ref, self.plane_normal_ref = get_principal_plane_rotation(ref_midline, self.plane_normal_ref_prev)
            
            reward -= (shape_diff_weight*diff/s)
            reward -= pp_angle_diff_weight * abs(pp_angle_diff - pp_angle_diff_ref) # Add a penalty for the diff between sim and ref plane rotation. A problem with this reward is that it includes both rotation due to change in shape, and due to rotation of the frame, which might not always go together.
            # reward -= action_smoothness_weight * np.mean(np.abs(np.diff(action)))
            
            # print(pp_angle_diff, pp_angle_diff_ref, abs(pp_angle_diff - pp_angle_diff_ref))
            self.plane_normal_prev = self.plane_normal.copy()
            self.plane_normal_ref_prev = self.plane_normal_ref.copy()
        
            # Shape-differnce early-termination (happens only every frame_skip_data time steps)
            if(diff > (shape_diff_threshold)): # *s.
                terminated = True

        if(0):
            # Early termination conditions that happen every time step
            # ********************************************************
            sim_angles_dv = self.data.qpos.flat.copy()[7::2]
            sim_angles_lr = self.data.qpos.flat.copy()[8::2]
            # dv_mean_abs_angle = np.mean(np.abs(sim_angles_dv))
            # lr_mean_abs_angle = np.mean(np.abs(sim_angles_lr))
            # TODO: change to 0.15 and 0.25
            if(np.max(np.abs(sim_angles_lr)) > 0.15*s or np.max(np.abs(sim_angles_dv)) > 0.25*s):
                terminated = True
                if(np.max(np.abs(np.diff(action))) > 0.1):
                    terminated = True
            
            if(np.max(np.abs(np.diff(sim_angles_lr, int(1/s)))) > 0.2 or np.max(np.abs(np.diff(sim_angles_dv, int(1/s)))) > 0.2):
                terminated = True
        
        return reward, terminated
    
    def get_ref_midline(self):
        # Generate reference midline points
        f = 1 # Frequency.
        
        x = np.linspace(0, 2*np.pi, num=(self.P["Np"]+1))
        y_dv = self.F(A0, f, x, self.phase, 0)
        
        if(self.P["gait"] == "2d_sine"):
            y_lr = 0*x
        elif(self.P["gait"] == "3d_sine"):
            y_lr = self.S(x) * self.F(A0/2, 2*f, x, self.phase + (np.pi/2), 0)
        else:
            y_lr = 0*x
        
        return y_dv, y_lr, x