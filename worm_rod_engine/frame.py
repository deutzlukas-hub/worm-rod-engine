
# From built-in
from typing import Optional, List
# From third-party
import numpy as np
import h5py

class Frame():
    def __init__(self, **kwargs):

        self.frame_keys = list(kwargs.keys())

        for key, value in kwargs.items():
            setattr(self, key, value)

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

