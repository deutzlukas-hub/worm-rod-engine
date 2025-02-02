
# From built-in
from typing import Optional, List
# From third-party
import numpy as np
import h5py
from notebook.extensions import RED_X


class Frame():
    def __init__(self, **kwargs):

        self.frame_keys = list(kwargs.keys())

        for key, value in kwargs.items():
            setattr(self, key, value)

    def body_frame_euler_angles(self):

        """Returns an array of shape (3, 3, N) with rotation matrices for each set of angles."""

        Q_arr = np.zeros((3, 3, self.theta.shape[-1]))

        for i in range(self.theta.shape[-1]):
            # Extract angles for this iteration
            a, b, g = self.theta[0, i], self.theta[1, i], self.theta[2, i]

            # Compute individual rotation matrices
            R_x = np.array([
                [1, 0, 0],
                [0, np.cos(g), -np.sin(g)],
                [0, np.sin(g), np.cos(g)]
            ])

            R_y = np.array([
                [np.cos(b), 0, np.sin(b)],
                [0, 1, 0],
                [-np.sin(b), 0, np.cos(b)]
            ])

            R_z = np.array([
                [np.cos(a), -np.sin(a), 0],
                [np.sin(a), np.cos(a), 0],
                [0, 0, 1]
            ])

            Q_arr[:, :, i] = R_z @ R_y @ R_x


        self.d1 = Q_arr[0, ...]
        self.d2 = Q_arr[1, ...]
        self.d3 = Q_arr[2, ...]

    def euler_angles_from_body_frame(self):

        Q = np.stack([self.d1, self.d2, self.d3], axis=1)

        alpha = np.arctan2(Q[1, 0, :], Q[0, 0 , :])
        beta = np.arcsin(-Q[2, 1, :])
        gama = np.arctan2(Q[2, 1, :], Q[2, 2, :])

        self.theta = np.row_stack([alpha, beta, gama])

class FrameSequence():

    def __init__(self, frames: Optional[List[Frame]] = None, **kwargs) :

        if frames is not None:
            assert not kwargs , "When a list of frames is provided, 'kwargs' must be empty."
            self.frame_keys = frames[0].frame_keys
            self.init_from_frames(frames)
        else:
            assert kwargs, "When no list of frames is provided, 'kwargs' can't be empty."
            self.frame_keys = kwargs.keys()
            for key, value in kwargs.items():
                setattr(self, key, value)

    def init_from_frames(self, frames):

        n_t_step = len(frames)

        # Allocate numpy arrays for outputs
        for key in self.frame_keys:
            v = getattr(frames[0], key)
            if isinstance(v, float):
                setattr(self, key, np.zeros(n_t_step))
            elif isinstance(v, np.ndarray):
                setattr(self, key, np.zeros((n_t_step,) + v.shape))
        # Fill arrays
        for i, frame in enumerate(frames):
            for key in self.frame_keys:
                v = getattr(frames[i], key)
                if isinstance(v, float):
                    getattr(self, key)[i] = v
                if isinstance(v, np.ndarray):
                    getattr(self, key)[i, :] = v

    def add_to_h5(self, h5: h5py.File, group_name: Optional[str] = None):

        if group_name is None:
            group = h5.create_group('FS')
        else:
            group = h5.create_group(group_name)

        for key in self.frame_keys:
            if getattr(self, key) is not None:
                group.create_dataset(key, getattr(self, key))

