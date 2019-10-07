import numpy as np
from collections import OrderedDict
from gym import utils
from gym import spaces
import os

# from rlkit.envs.mujoco_env import MujocoEnv
from gym.envs.mujoco.mujoco_env import MujocoEnv

from rlkit.core.vistools import plot_seaborn_heatmap, plot_scatter


class PickupTaskPointMassEnv():
    def __init__(self, env_bound=7.0, init_pos=np.array([0.0, 0.0]), episode_len=119):
        self.cur_pos = np.zeros([2])
        self.init_pos = init_pos.copy()

        self.max_action_magnitude = 1.0
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype='float32')
        self.observation_space = spaces.Box(-env_bound, env_bound, shape=(8,), dtype='float32')
        self.env_bound = env_bound

        self.episode_len = float(episode_len)

        self.num_items = 2
        self.min_angle_between_items = np.pi / 6.0
        self.radius = 10.0
        self.accept_radius = 1.0

    def seed(self, num):
        pass

    def step(self, a):
        a = np.clip(a, self.action_space.low, self.action_space.high)

        reward = 0.0 # we don't need a reward for what we want to do with this env

        self.cur_pos += a

        # if we want noisy dynamics
        # self.cur_pos += np.random.normal(loc=0.0, scale=0.2, size=2)

        # clipping to env bound
        # self.cur_pos = np.clip(self.cur_pos, -self.env_bound, self.env_bound)

        self.timestep += 1.0

        if self.timestep == self.episode_len:
            done = True
        else:
            done = False

        # check if you are in one of the target regions
        for i in range(self.obj_poses.shape[0]):
            if np.linalg.norm(self.cur_pos - self.obj_poses[i]) < self.accept_radius:
                self.visited[i] = 1.0
        
        # check success
        success = 1.0 if (np.mean(self.visited) == 1.0) else 0.0
        
        # np.array([self.cur_pos[0], self.cur_pos[1], self.timestep/self.episode_len])
        return self._get_obs() , reward, done, dict(
            xy_pos=self.cur_pos.copy(),
            timestep=self.timestep,
            success=success
        )

    def angles_check(self, prev_as, new_a):
        if len(prev_as) == 0:
            return True
        for a in prev_as:
            if abs(a - new_a) < self.min_angle_between_items:
                return False
        return True

    def reset(self):
        self.timestep = 0.0
        self.cur_pos = self.init_pos + np.random.normal(loc=0.0, scale=0.1, size=2)
        # return np.array([self.cur_pos[0], self.cur_pos[1], 0.0])

        # reset the object poses
        angles = []
        for _ in range(self.num_items):
            new_a = np.random.uniform(high=np.pi)
            while not self.angles_check(angles, new_a):
                new_a = np.random.uniform(high=np.pi)
            angles.append(new_a)
        
        angles = np.array(angles)
        self.obj_poses = np.stack([self.radius*np.cos(angles), self.radius*np.sin(angles)], axis=1)
        self.flat_obj_poses = self.obj_poses.flatten()

        # reset the visitations
        self.visited = np.zeros(self.num_items)

        return self._get_obs()
    
    def _get_obs(self):
        obs = np.concatenate((self.cur_pos, self.flat_obj_poses, self.visited)).copy()
        return obs

    def log_statistics(self, paths):
        is_success = [sum([d['success'] for d in path["env_infos"]]) > 0 for path in paths]
        return {
            'Success Rate': np.mean(is_success)
        }
