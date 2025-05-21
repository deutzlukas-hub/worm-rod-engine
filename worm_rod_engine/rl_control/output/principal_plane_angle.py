import numpy as np
import os
import matplotlib.pyplot as plt

from functions.functions_worm import fit_plane_to_point_cloud, get_principal_plane_rotation

def principal_plane_angle(P, midlines, save_dir, ep, ref_data, ref_frames):
	
	# subtract initial value and overlay.
	
    # print(midlines[0])
    prev_normal = fit_plane_to_point_cloud(midlines[0])
    prev_normal_ref = fit_plane_to_point_cloud(ref_data[int(ref_frames[0])].T)
    angle_diffs = np.empty(midlines.shape[0]-1)
    angle_diffs_ref = np.empty(midlines.shape[0]-1)
    
    for i in range(1, midlines.shape[0]):
    	angle_diffs[i-1], prev_normal = get_principal_plane_rotation(midlines[i], prev_normal)
    	angle_diffs_ref[i-1], prev_normal_ref = get_principal_plane_rotation(ref_data[int(ref_frames[i])].T, prev_normal_ref)
    
    angle_cumsum = get_angles(angle_diffs)
    angle_cumsum_ref = get_angles(angle_diffs_ref)
    angle_cumsum_ref -= angle_cumsum_ref[0]
    
    fig = plt.figure()
    fig.tight_layout()
    
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    t = np.arange(0,angle_cumsum.shape[0]*P["dt_opt"],P["dt_opt"])
    
    ax1.plot(t,angle_cumsum, zorder=-1)
    # ax1.scatter(t,angle_cumsum, s=1, c='r', zorder=1)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Angle [°]')
    ax1.set_title('Model')
    
    ax2.plot(t,angle_cumsum_ref, zorder=-1)
    ax2.plot(t,angle_cumsum, zorder=-1, lw=0.5)
    # ax2.scatter(t,angle_cumsum_ref, s=1, c='r', zorder=1)
    ax2.set_xlabel('Time')
    # ax2.set_ylabel('Angle [°]')
    ax2.set_title('Data')
    
    plt.savefig(os.path.join(save_dir,f"principal_plane_angle_ep={ep}.png"), bbox_inches='tight')
    
    plt.close(fig)
    
def get_angles(angle_diffs):
    return np.unwrap(np.cumsum(angle_diffs)) * 180 / np.pi