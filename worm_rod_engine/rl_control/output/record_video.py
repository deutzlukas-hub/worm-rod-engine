import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from worm_rod_engine.rl_control.worm_functions import fit_plane_to_point_cloud, get_mean_tangent, interpolate_line
from worm_rod_engine.rl_control.natural_frame_1 import NaturalFrame, distance_ab

def record_video(P, midlines, save_dir, ep, ref_data, ref_frames):
    
    if(P["gait"] == "coiling"):
        azim_sim=-160
        elev_sim=30
        azim_ref=-240
        elev_ref=10
    else:
        azim_sim=-100
        elev_sim=30
        azim_ref=-60
        elev_ref=10
    
    fig = plt.figure()
    fig.tight_layout()
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.view_init(azim=azim_sim, elev=elev_sim)
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.view_init(azim=azim_ref, elev=elev_ref)
    
    midpoint = round(P["Np"] / 2)
    point = np.array([0,0,0])
    px, py = np.meshgrid(np.linspace(-0.0025,0.0025), np.linspace(-0.0025,0.0025))
    
    ax1.set_xlim(midlines[0][midpoint,0] + np.array([-0.0025, 0.0025]))
    ax1.set_ylim(midlines[0][midpoint,1] + np.array([-0.0025, 0.0025]))
    ax1.set_zlim(midlines[0][midpoint,2] + np.array([-0.0025, 0.0025]))
    
    midpoint_ref = round(ref_data[0].shape[1] / 2)
    # point = np.array([0,0,0])
    px_ref, py_ref = np.meshgrid(np.linspace(-0.5,0.5), np.linspace(-0.5,0.5))
    
    global plane_1, plane_2
    
    midline_1, = ax1.plot(0,0,0, linewidth=3, color='black')
    # normal_1 = ax1.plot([0,1], [0, 1], [0,1], linewidth=2, color='red')[0])
    plane_1 = ax1.plot_surface(np.array([[0, 1], [0, 1]]), np.array([[0, 0], [1, 1]]), np.array([[1, 1], [1, 1]]), alpha=0.3, color='0.75')
    tangent_1, = ax1.plot([0,0], [0,0], [0,0], linewidth=1, color='red')
    
    midline = ref_data[int(ref_frames[0])].T
    midline_2, = ax2.plot(midline[:,0], midline[:,1], midline[:,2], linewidth=3, color='black')
    # normal_2 = ax2.plot([0,1], [0, 1], [0,1], linewidth=2, color='red')[0])
    plane_2 = ax2.plot_surface(np.array([[0, 1], [0, 1]]), np.array([[0, 0], [1, 1]]), np.array([[1, 1], [1, 1]]), alpha=0.3, color='0.75')
    tangent_2, = ax2.plot([0,0], [0,0], [0,0], linewidth=1, color='red')
    
    ax2.set_xlim(midline[midpoint_ref,0] + np.array([-0.85, 0.85]))
    ax2.set_ylim(midline[midpoint_ref,1] + np.array([-0.85, 0.85]))
    ax2.set_zlim(midline[midpoint_ref,2] + np.array([-0.85, 0.85]))
    
    def update(i):
        # ax1.cla()
        global plane_1, plane_2
        
        midline = midlines[i]
        tangent = get_mean_tangent(midline)
        tangent *= 0.005
        normal = fit_plane_to_point_cloud(midlines[i])
        
        normal = (normal / np.linalg.norm(normal)) * 0.005
        d = -point.dot(normal)
        pz = (-normal[0] * px - normal[1] * py - d) * 1. / normal[2]
        
        midline_1.set_data(midline[:,0], midline[:,1])
        midline_1.set_3d_properties(midline[:,2])
        
        tangent_1.set_data([midline[-1,0],midline[-1,0]+tangent[0]], [midline[-1,1],midline[-1,1]+tangent[1]])
        tangent_1.set_3d_properties([midline[-1,2],midline[-1,2]+tangent[2]])
        
        plane_1.remove()
        plane_1 = ax1.plot_surface(px+midline[midpoint,0], py+midline[midpoint,1], pz+midline[midpoint,2], alpha=0.2, color='0.75')
        
        # Reference data
        midline_ref = ref_data[int(ref_frames[i])].T
        tangent_ref = get_mean_tangent(midline_ref)
        tangent_ref *= 1
        normal_ref = fit_plane_to_point_cloud(midline_ref)
        
        normal_ref = (normal_ref / np.linalg.norm(normal_ref)) * 0.001
        d = -point.dot(normal_ref)
        pz = (-normal_ref[0] * px_ref - normal_ref[1] * py_ref - d) * 1. / normal_ref[2]
        
        midline_2.set_data(midline_ref[:,0], midline_ref[:,1])
        midline_2.set_3d_properties(midline_ref[:,2])
        
        tangent_2.set_data([midline_ref[-1,0],midline_ref[-1,0]+tangent_ref[0]], [midline_ref[-1,1],midline_ref[-1,1]+tangent_ref[1]])
        tangent_2.set_3d_properties([midline_ref[-1,2],midline_ref[-1,2]+tangent_ref[2]])
        
        plane_2.remove()
        plane_2 = ax2.plot_surface(px_ref+midline_ref[midpoint_ref,0], py_ref+midline_ref[midpoint_ref,1], pz+midline_ref[midpoint_ref,2], alpha=0.2, color='0.75')
        
        midline_ref = interpolate_line(midline_ref.T, midline.shape[0])
        midline_ref = np.array(midline_ref).T
        
        ref_nf = NaturalFrame(midline_ref)
        sim_nf = NaturalFrame(midline)
        diff = distance_ab(ref_nf.mc, sim_nf.mc)
        
        ax1.set_title(f'{i} (e={round(diff, 2)})')
        ax2.set_title(int(ref_frames[i]))
        
       	return midline_1, midline_2
       	# return midline_1, normal_1, plane_1, title_1, midline_2, normal_2, plane_2, title_2
    
    ani = manimation.FuncAnimation(fig, update, frames=midlines.shape[0])
    # ani = manimation.ArtistAnimation(fig=fig, artists=artists, interval=100)
    metadata = dict(title=save_dir, artist='WormLab Leeds')
    ani.save(os.path.join(save_dir,f"summary_ep={ep}.mp4"), writer='ffmpeg', fps=50, metadata=metadata)
    plt.close(fig)