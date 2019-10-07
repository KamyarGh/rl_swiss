import numpy as np

meta_irl_env_task_params_samplers = {
    'dmcs_simple_meta_reacher': {
        'train': SimpleMetaReacherMetaTrainTaskParamsSampler,
        'test': SimpleMetaReacherMetaTestTaskParamsSampler
    }
}


class SimpleMetaReacherMetaTrainTaskParamsSampler():
    def __init__(self):
        idx = 2*np.arange(16)
        angles = 2 * np.pi * idx / 32
        x = np.sin(angles)
        y = np.cos(angles)
        self.p = 0.2 * np.stack([x,y], axis=1)
        self.ptr = 0
    
    def sample(self):
        x, y = self.p[self.ptr, 0], self.p[self.ptr, 1]
        self.ptr = (self.ptr + 1) % 16
        return {'x': x, 'y': y}, np.array([x,y])


class SimpleMetaReacherMetaTestTaskParamsSampler():
    def __init__(self):
        idx = 2*np.arange(16) + 1
        angles = 2 * np.pi * idx / 32
        x = np.sin(angles)
        y = np.cos(angles)
        self.p = 0.2 * np.stack([x,y], axis=1)
        self.ptr = 0
    
    def sample(self):
        x, y = self.p[self.ptr, 0], self.p[self.ptr, 1]
        self.ptr = (self.ptr + 1) % 16
        return {'x': x, 'y': y}, np.array([x,y])
