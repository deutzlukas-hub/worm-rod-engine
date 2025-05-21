import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def plot_trajectory(P, midlines, save_dir, ep, ref_data, ref_frames):
    
    fig1 = plt.figure()
    
    ax1 = fig1.add_subplot(121, projection='3d')
    ax2 = fig1.add_subplot(122, projection='3d')
    ax1.view_init(azim=-70, elev=30)
    ax2.view_init(azim=160, elev=30)
    
    # floor = 0,1,2; 3,4,5 ; 6,7,8; 9,10,11; 12,13,14; 15,16,17; 18,19,20; 21,22,23; 24,25,26, 27,28,29.
    '''
    c = 30
    for j in range(3):
        ax1.plot(geom_xpos[:,3+j*c], geom_xpos[:,4+j*c], geom_xpos[:,5+j*c], label='3D Line Plot')
    '''
    
    t = range(0, midlines.shape[0])
    
    for j in np.array([7]): # [5,15,25]
        x = midlines[:,j,0].flatten()
        y = midlines[:,j,1].flatten()
        z = midlines[:,j,2].flatten()
        # ax1.plot(x, y, z)
        make_color_gradient_line(ax1, t, x, y, z)
    
    ref_frames = ref_frames.astype(int)
    x_ref = ref_data[ref_frames, 0, 31]
    y_ref = ref_data[ref_frames, 1, 31]
    z_ref = ref_data[ref_frames, 2, 31]
    make_color_gradient_line(ax2, t, x_ref, y_ref, z_ref)
    
    # print(t, midlines.shape, x.shape)
    
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.set_zlabel('')
    
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])
    
    # Set axis limits to be tight around the data
    midpoint = np.mean(midlines[:, 7, :], axis=0)
    d1 = 0.0005
    ax1.set_xlim(midpoint[0] + np.array([-d1, d1]))
    ax1.set_ylim(midpoint[1] + np.array([-d1, d1]))
    ax1.set_zlim(midpoint[2] + np.array([-d1, d1]))
    
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax2.set_zlabel('')
    
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_zticklabels([])
    # ax2.set_xlim([np.min(x_ref), np.max(x_ref)])
    # ax2.set_ylim([np.min(y_ref), np.max(y_ref)])
    # ax2.set_zlim([np.min(z_ref), np.max(z_ref)])
    midpoint_ref = np.mean(ref_data[ref_frames, :, 31], axis=0)
    d2 = 0.125
    ax2.set_xlim(midpoint_ref[0] + np.array([-d2, d2]))
    ax2.set_ylim(midpoint_ref[1] + np.array([-d2, d2]))
    ax2.set_zlim(midpoint_ref[2] + np.array([-d2, d2]))
    
    # plt.title('3D Line Plot Example')
    # plt.legend()
    plt.savefig(os.path.join(save_dir, f"trajectory_ep={ep}.png"), bbox_inches='tight', dpi=300)
    
    # plt.show()
    
    plt.close(fig1)
    
def make_color_gradient_line(ax, t, x, y, z):
    # Normalize the color data to [0, 1] for the colormap
    norm = plt.Normalize(min(t), max(t))
    colors = plt.cm.turbo(norm(t))
    
    # Create a set of line segments so that we can color them individually
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # print(points.shape, segments.shape)
    
    # Create a 3D LineCollection
    lc = Line3DCollection(segments, cmap='turbo', norm=norm, linewidth=0.5)
    lc.set_array(np.asarray(t))
    lc.set_color(colors)  # Explicitly set the colors
    
    ax.add_collection3d(lc) # , zs=z, zdir='z')