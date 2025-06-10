import numpy as np
import os
import pickle
from scipy.io import loadmat

import gymnasium as gym
import mujoco

from output.plot_trajectory import plot_trajectory
from output.plot_kymograms import plot_kymograms
from output.principal_plane_angle import principal_plane_angle
from output.record_video import record_video

def save_results(model=None, P=None, model_path=None, model_file=None, ep=0):
    
    print("Preparing video...")
    
    print(P)
    
    model_path += '/data/'
    direction = "forward" if P["direction"] == 1 else "backward"
    
    if(P["ref"] and (P["gait"] == "infinity" or P["gait"] == "coiling")):
        steps_per_episode = 4 / P["dt_opt"]
    else:
        steps_per_episode = 16 / P["dt_opt"]
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    mujoco_model_path = os.path.join(main_dir, 'envs', 'mujoco', 'assets', P["physical_model"]["file_name"] + ".xml")
    
    # Visualise trained agent
    for i in range(5):
        
        # np.random.seed(i)
        
        env = gym.make(P["env"], P=P, record=True, xml_file=mujoco_model_path, max_episode_steps=steps_per_episode, render_mode="rgb_array", camera_name=P["rendering"]["camera_mode"])
        
        obs, info = env.reset()
        
        env = gym.wrappers.RecordVideo(env, video_folder=model_path, name_prefix=f"{P['animal']}_{direction}_{P['gait']}_ep={ep}_{i}_cam0")
        
        env.render()
        
        env.unwrapped.mujoco_renderer.viewer.vopt.frame = 1 # [1,7].
        # print(env.unwrapped.mujoco_renderer.viewer.vopt)
        
        terminated = 0
        truncated = 0
        
        actions = np.array([])
        qpos = np.array([])
        qvel = np.array([])
        geom_xpos = np.array([])
        geom_xmat = np.array([])
        xquat = np.array([])
        # plane_normals = np.array([])
        midlines = [] # np.array([])
        ref_frames = [] # np.array([])
        
        s = 0
        while(not terminated and not truncated):
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            terminated = False
            
            actions = np.append(actions, action)
            qpos = np.append(qpos, info["qpos"])
            qvel = np.append(qvel, info["qvel"])
            # geom_xpos = np.append(geom_xpos, info["geom_xpos"])
            # geom_xmat = np.append(geom_xmat, info["geom_xmat"])
            # xquat = np.append(xquat, info["xquat"])
            # plane_normals = np.append(plane_normals, info["plane_normal"])
            
            midlines.append(np.array([info["midline"]]))
            
            if(s == 0):
                ref_frames = np.append(ref_frames, info["first_ref_frame"])
            
            ref_frames = np.append(ref_frames, info["ref_frame"])
            
            s += 1
        
        env.close()
        
        actions = actions.reshape(-1, action.shape[0])
        qpos = qpos.reshape(-1, info["qpos"].shape[0])
        # qvel = qvel.reshape(-1, info["qvel"].shape[0])
        # geom_xpos = geom_xpos.reshape(-1, info["geom_xpos"].shape[0])
        # geom_xmat = geom_xmat.reshape(-1, info["geom_xmat"].shape[0])
        # xquat = xquat.reshape(-1, info["xquat"].shape[0])
        # plane_normals = plane_normals.reshape(-1, info["plane_normal"].shape[0])
        midlines = np.vstack(midlines)
        
        '''
        # np.savetxt(os.path.join(model_path,f"actions_ep={ep}.csv"), actions, delimiter=",")
        np.savetxt(os.path.join(model_path,f"qpos_ep={ep}.csv"), qpos, delimiter=",")
        np.savetxt(os.path.join(model_path,f"qvel_ep={ep}.csv"), qvel, delimiter=",")
        np.savetxt(os.path.join(model_path,f"geom_xpos_ep={ep}.csv"), geom_xpos, delimiter=",")
        np.savetxt(os.path.join(model_path,f"geom_xmat_ep={ep}.csv"), geom_xmat, delimiter=",")
        np.savetxt(os.path.join(model_path,f"xquat_ep={ep}.csv"), xquat, delimiter=",")
        # np.savetxt(os.path.join(model_path,f"plane_normals_ep={ep}.csv"), plane_normals, delimiter=",")
        '''
        
        # print(geom_xpos.shape)
        if(P["gait"] == "infinity"):
            gait = P["gait"]
        else:
            gait = P["gait"] + '_' + P["chirality"]
        ref_data = loadmat(P["reward"]["ref_data"][gait])
        
        plot_trajectory(P, midlines, model_path, i, ref_data["XYZ"], ref_frames) # Trajectory plot.
        
        plot_kymograms(P, midlines, qpos, actions, model_path, i, ref_data["XYZ"], ref_frames) # Angles and joint torques kymograms.
        
        principal_plane_angle(P, midlines, model_path, i, ref_data["XYZ"], ref_frames) # Principal plane vectors.
        
        record_video(P, midlines, model_path, i, ref_data["XYZ"], ref_frames) # Record a video of the midline, principal plane and point trajectory.