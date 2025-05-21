import numpy as np

from training_params import training_params

def parameters(gait=None):
    train = True
    
    T = 0.8 # Period [s]. [*0.8*].
    dt = 0.04 # Physics time step.

    scale = 0.01 # [1, 0.5, 0.1, 0.05, 0.01].
    Np = 32 # Number of body segments. [*32*, 128].
    # total_length = 0.1 # [cm].
    
    frame_skip = int(1) # [*0.01*] Optimisation frame skipping - used to compute the optimisation time step (= dt * frame_skip).
    # frame_skip = int(0.01 / dt) # Optimisation frame skipping - used to compute the optimisation time step (= dt * frame_skip).
    dt_opt = dt*frame_skip
    
    ref = True
    gait = "coiling" # [2d_sine, 3d_sine, coiling, infinity].
    chirality = 'cw' # +1=ccw. -1=cw.
    direction = 1 # +1=forward. -1=backward.
    
    P = {
        "Np": Np, # Number of midline points. [36].
        "dt": dt, # [s]. Must be equal to frametime(=0.01) * frame_skip.
        "dt_opt": dt_opt, # [s].
        "dt_exp": 0.04, # [s]. True value is 0.04s.
        "T": T, # Period [s].
        "frame_skip": frame_skip, # Optimisation frame skipping - used to compute the optimisation time step (= dt * frame_skip)., # [s].
        "frame_skip_data": 4, # 4. int(0.04 / (dt*frame_skip)),
        "time_rounding": 2, # Number of decimal digits.
        "animal": "worm",
        "scale": scale,
        # "animal_model": "worm.xml",
        "ref": ref,
        "gait": gait,
        "chirality": chirality if gait == "coiling" else "",
        "direction": direction, # +1=forward. -1=backward.
        "env": "worm-env",
        
        "obs_space": 7 + 6, # Position, orientation and velocity of the torso. Position: 7=[x,y,z, θx,θy,θz,θw]. Velocity: 6=[dx,dy,dz,dθx,dθy,dθz].
        "action_space": 0,
        
        "exclude_xyz_from_obs": 7, # If > 0, 3=[x,y,z] or 7=[x,y,z,θx,θy,θz,θw] are removed from the observation space.
        "include_shape_diff_in_obs": False,
        "include_ref_frame_in_obs": True,
        "include_next_ref_frame_in_obs": True,
        "include_ref_shape_in_obs": False,
        
        "mechanics": {
            "solver": "RK4", # ["Euler", "Euler"].
            # "init_pos": [0, 0, 0.4*scale], # [cm].
        },
        
        "physical_model": {
            "file_name": "worm",
        },
        
        "control": {
            "infinity_trajectory": 0, # 8-shape in the x-z plane.
            "max_angle_diff_dv": 0.30, # [0.35].
            "max_angle_diff_lr": 0.20, # [0.20, 0.25].
            "max_angle_diff_pp_plane": 10 * (np.pi / 180) * (dt_opt / 0.01), # Maximum rotation [deg->rad] of the principal body plane in one step.
        },
        
        "reward": {
            "principal_plane_step": 0.1,
            "material_frame_step": -0.05,
            "body_smoothness_lr": -0.04,
            "ref_data": {
                "infinity": "./resources/ref_data/infinity/382_20190711_trial03_F1=[10450,10850]_XYZ_Inf.mat",
                "coiling_cw": "./resources/ref_data/coiling/XYZ_20180809_trial05_F1=[6355,6755]_CW.mat",
                "coiling_ccw": "./resources/ref_data/coiling/233_20180809_trial05_F1=[5450,5850]_XYZ_CCW.mat",
                "curvature_scaling": 0.41 * 200, # TODO: find the curvature scale of sim and ref, and scale ref curvature accordingly.
                "n_eval": 64, # Number of points to resample the ref lines.
                "n_skip": int(round(0.05*Np)), # [0.05]. Number of points to skip from the extremes when comparing m1 and m2 between sim and ref.
                "ref_pose_num": 200, # [cw=*200*, ccw=400].
                "ref_random_pose_num": 150, # [100, infinity=150, cw=*150*, 50, ccw=*350*].
                "opt_max_iter": 500,
            }
        },
        
        "rendering": {
            "camera_mode": "track", # ["track", "free"].
            "camera_position": str(0*scale) + " " + str(-1*scale) + " " + str(2*scale), # *[0,-1,1]*.
            "camera_distance": 0.3, # Distance of camera from the object.
            "render_fps": 1000, # Number of frames to render per simulation second.
            "texrepeat": str(int(1/scale)) + " " + str(int(1/scale)),
        },
    }
    
    # P["mechanics"]["head_size"] = str(P["mechanics"]["geom"]["segment_width"]) + " " + str(P["mechanics"]["geom"]["segment_length"]) + " " + str(P["mechanics"]["geom"]["segment_width"])
    
    P["training"] = training_params(use_gpu=train, T=P["T"], dt=P["dt_opt"], animal=P['animal'])
    
    if(P["exclude_xyz_from_obs"]):
        P["obs_space"] -= P["exclude_xyz_from_obs"] # [3,7].
    
    if(P["include_shape_diff_in_obs"]):
        P["obs_space"] += 1*(P["Np"]+1) # Real and imaginary diffs of the complex representation of m1 and m2.
        # P["obs_space"] += 2*(P["Np"]+1) # Real and imaginary diffs of the complex representation of m1 and m2.
    
    if(P["include_ref_frame_in_obs"]):
        P["obs_space"] += 1
    
    if(P["include_next_ref_frame_in_obs"]):
        P["obs_space"] += 1
    
    if(P["include_ref_shape_in_obs"]):
        P["obs_space"] += 1*(P["Np"]+1)
    
    P["obs_space"] += 4*(P["Np"]-1) # Np segments are connected via (Np-1) joints (x2). Each joint is associated with an angle and angular velocity.
    P["obs_minmax"] = np.inf # P["mechanics"]["max_joint_angle"] * (np.pi/180)
    
    # P["action_space"] = 4*(P["Np"]-1)
    
    # print("Scale =", P["scale"])
    # print("Joint damping =", P["mechanics"]["joint_damping"])
    
    return P