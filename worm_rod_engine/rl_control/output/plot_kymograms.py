import numpy as np
import os
from scipy.io import savemat
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from functions.frames.TNB import TNB

def plot_kymograms(P, midlines, qpos, actions, save_dir, ep, ref_data, ref_frames):
    
    # Angles kymogram
    angles_dv = np.transpose(qpos[:,7::2])
    angles_lr = np.transpose(qpos[:,8::2])
    angles_dv_lr = np.sign(angles_dv) * (angles_dv**2 + angles_lr**2)**0.5
    # print(angles_dv.shape)
    
    sigma = 2.0 # [*2.0*].
    angles_dv = gaussian_filter(angles_dv, sigma=sigma)
    angles_lr = gaussian_filter(angles_lr, sigma=sigma)
    angles_dv_lr = gaussian_filter(angles_dv_lr, sigma=sigma)
    
    kappa = np.empty([midlines.shape[0], midlines.shape[1]])
    kappa_ref = np.empty([midlines.shape[0], ref_data.shape[2]])
    for i in range(midlines.shape[0]):
        _, _, _, kappa[i,:], _ = TNB(midlines[i][:,0], midlines[i][:,1], midlines[i][:,2])
        midline_ref = ref_data[int(ref_frames[i])].T
        _, _, _, kappa_ref[i,:], _ = TNB(midline_ref[:,0], midline_ref[:,1], midline_ref[:,2])

    fig1 = plt.figure()
    ax11 = fig1.add_subplot(321)
    ax12 = fig1.add_subplot(323)
    ax13 = fig1.add_subplot(325)
    ax14 = fig1.add_subplot(322)
    ax15 = fig1.add_subplot(324)
    ax16 = fig1.add_subplot(326)
    
   	# Joint angles
    c1 = ax11.pcolor(angles_dv_lr, cmap='turbo')
    c2 = ax12.pcolor(angles_dv, cmap='turbo')
    c3 = ax13.pcolor(angles_lr, cmap='turbo')
    
    c4 = ax14.pcolor(kappa.T, cmap='turbo')
    c5 = ax15.pcolor(np.clip(kappa.T, 0, 2000), cmap='turbo')
    c6 = ax16.pcolor(kappa_ref.T, cmap='turbo')
    
    ax11.set_title('$sign(\\alpha_{DV}) * \\sqrt{\\alpha_{DV}^2 + \\alpha_{LR}^2}$')
    ax12.set_title('$\\alpha_{DV}$')
    ax13.set_title('$\\alpha_{LR}$')
    ax14.set_title('$\\kappa_{sim}$')
    ax15.set_title('$\\kappa_{sim} (clipped)$')
    ax16.set_title('$\\kappa_{ref}$')
    
    # ax21.xticks(c1.xticks()[0] * P["dt_opt"], c1.xticks()[1])
    ax11.set_xticklabels(ax11.get_xticks() * P["dt_opt"])
    ax12.set_xticklabels(ax12.get_xticks() * P["dt_opt"])
    ax13.set_xticklabels(ax13.get_xticks() * P["dt_opt"])
    ax14.set_xticklabels(ax14.get_xticks() * P["dt_opt"])
    ax15.set_xticklabels(ax15.get_xticks() * P["dt_opt"])
    ax16.set_xticklabels(ax16.get_xticks() * P["dt_opt"])
    
    ax13.set_xlabel('Time [s]')
    ax16.set_xlabel('Time [s]')
    
    fig1.colorbar(c1, ax=ax11)
    fig1.colorbar(c2, ax=ax12)
    fig1.colorbar(c3, ax=ax13)
    fig1.colorbar(c4, ax=ax14)
    fig1.colorbar(c5, ax=ax15)
    fig1.colorbar(c6, ax=ax16)
    
    fig1.tight_layout()
    
    plt.savefig(os.path.join(save_dir, f"joint_angles_kymogram_ep={ep}.png"), bbox_inches='tight')
    # savemat(os.path.join(save_dir, f"joint_angles_ep={ep}.mat"), {'angles_dv_lr': angles_dv_lr, 'angles_dv': angles_dv, 'angles_lr': angles_lr})
    
    # ---------------------------------------------------------------------
    
    # Joint torques kymogram
    angles_dv = np.transpose(actions[:,0::2])
    angles_lr = np.transpose(actions[:,1::2])
    angles_dv_lr = np.sign(angles_dv) * (angles_dv**2 + angles_lr**2)**0.5
    # print(angles_dv.shape)
    
    sigma = 5.0 # [*10.0*].
    angles_dv = gaussian_filter(angles_dv, sigma=sigma)
    angles_lr = gaussian_filter(angles_lr, sigma=sigma)
    angles_dv_lr = gaussian_filter(angles_dv_lr, sigma=sigma)
    
    fig2 = plt.figure()
    ax31 = fig2.add_subplot(311)
    ax32 = fig2.add_subplot(312)
    ax33 = fig2.add_subplot(313)
    
    c1 = ax31.pcolor(angles_dv_lr, cmap='turbo')
    c2 = ax32.pcolor(angles_dv, cmap='turbo')
    c3 = ax33.pcolor(angles_lr, cmap='turbo')
    
    ax31.set_title('$sign(\\tau_{DV}) * \\sqrt{\\tau_{DV}^2 + \\tau_{LR}^2}$')
    ax32.set_title('$\\tau_{DV}$')
    ax33.set_title('$\\tau_{LR}$')
    
    # ax21.xticks(c1.xticks()[0] * P["dt_opt"], c1.xticks()[1])
    ax31.set_xticklabels(ax31.get_xticks() * P["dt_opt"])
    ax32.set_xticklabels(ax32.get_xticks() * P["dt_opt"])
    ax33.set_xticklabels(ax33.get_xticks() * P["dt_opt"])
    
    ax33.set_xlabel('Time [s]')
    
    fig2.colorbar(c1, ax=ax31)
    fig2.colorbar(c2, ax=ax32)
    fig2.colorbar(c3, ax=ax33)
    
    fig2.tight_layout()
    
    plt.savefig(os.path.join(save_dir,f"joint_torques_kymogram_ep={ep}_sig={sigma}.png"), bbox_inches='tight')
    # savemat(os.path.join(save_dir, f"joint_torques_ep={ep}.mat"), {'angles_dv_lr': angles_dv_lr, 'angles_dv': angles_dv, 'angles_lr': angles_lr})
    
    plt.close(fig1)
    plt.close(fig2)