import numpy as np
from collections import OrderedDict
from gym import utils
from gym import spaces
import os

# from rlkit.envs.mujoco_env import MujocoEnv
from gym.envs.mujoco.mujoco_env import MujocoEnv

from rlkit.core.vistools import plot_seaborn_heatmap, plot_scatter


class StateMatchingPointMassEnv():
    def __init__(self, env_bound=7.0, init_pos=np.array([0.0, 0.0]), episode_len=479, obs_with_time=True):
        self.cur_pos = np.zeros([2])
        self.init_pos = init_pos.copy()

        self.max_action_magnitude = 1.0
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype='float32')

        if obs_with_time:
            self.observation_space = spaces.Box(-env_bound, env_bound, shape=(3,), dtype='float32')
        else:
            self.observation_space = spaces.Box(-env_bound, env_bound, shape=(2,), dtype='float32')
        self.env_bound = env_bound
        self.obs_with_time = obs_with_time
        self.episode_len = float(episode_len)

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
        
        obs = self._get_obs()
        # return self.cur_pos.copy() , reward, done, dict(
        #     xy_pos=self.cur_pos.copy(),
        #     timestep=self.timestep
        # )
        return obs.copy() , reward, done, dict(
            xy_pos=self.cur_pos.copy(),
            timestep=self.timestep
        )
    

    def _get_obs(self):
        if self.obs_with_time:
            obs = np.array([self.cur_pos[0], self.cur_pos[1], self.timestep/self.episode_len])
        else:
            obs = np.array([self.cur_pos[0], self.cur_pos[1]])
        return obs


    def reset(self):
        self.timestep = 0.0

        self.cur_pos = self.init_pos + np.random.normal(loc=0.0, scale=0.1, size=2)
        # return np.array([self.cur_pos[0], self.cur_pos[1], 0.0])
        
        return self._get_obs().copy()


    def log_visuals(self, paths, epoch, log_dir):
        # turn the xy pos into arrays you can work with
        # N_paths x path_len x 2
        # xy_pos = np.array([[d['xy_pos'] for d in path["env_info"]] for path in paths])

        # plot using seaborn heatmap
        xy_pos = [np.array([d['xy_pos'] for d in path["env_info"]]) for path in paths]
        xy_pos = np.array([d['xy_pos'] for path in paths for d in path['env_info']])
        
        PLOT_BOUND = int(self.env_bound * 1.25)

        plot_seaborn_heatmap(
            xy_pos[:,0],
            xy_pos[:,1],
            30,
            'Point-Mass Heatmap Epoch %d'%epoch,
            os.path.join(log_dir, 'heatmap_epoch_%d.png'%epoch),
            [[-PLOT_BOUND,PLOT_BOUND], [-PLOT_BOUND,PLOT_BOUND]]
        )
        plot_scatter(
            xy_pos[:,0],
            xy_pos[:,1],
            30,
            'Point-Mass Scatter Epoch %d'%epoch,
            os.path.join(log_dir, 'scatter_epoch_%d.png'%epoch),
            [[-PLOT_BOUND,PLOT_BOUND], [-PLOT_BOUND,PLOT_BOUND]]
        )

        return {}

        # xy_pos = [np.array([d['xy_pos'] for d in path["env_info"]]) for path in paths]
        # max_len = max([a.shape[0] for a in xy_pos])
        # full_xy_pos = np.zeros((len(xy_pos), max_len, 2))
        # for i in range(len(xy_pos)):
        #     full_xy_pos[i][:xy_pos[i].shape[0]] = xy_pos[i]
        # xy_pos = full_xy_pos

        # # N_paths x path_len x 1 x 2
        # xy_pos = np.expand_dims(xy_pos, 2)

        # # compute for each path which target it gets closest to and how close
        # d = np.linalg.norm(xy_pos - self.valid_targets, axis=-1)
        # within_traj_min = np.min(d, axis=1)
        # min_ind = np.argmin(within_traj_min, axis=1)
        # min_val = np.min(within_traj_min, axis=1)

        # return_dict = OrderedDict()
        # for i in range(self.valid_targets.shape[-2]):
        #     return_dict['Target %d Perc'%i] = np.mean(min_ind == i)
        
        # for i in range(self.valid_targets.shape[-2]):
        #     min_dist_for_target_i = min_val[min_ind == i]
        #     if len(min_dist_for_target_i) == 0:
        #         return_dict['Target %d Dist Mean'%i] = np.mean(-1)
        #         return_dict['Target %d Dist Std'%i] = np.std(-1)
        #     else:
        #         return_dict['Target %d Dist Mean'%i] = np.mean(min_dist_for_target_i)
        #         return_dict['Target %d Dist Std'%i] = np.std(min_dist_for_target_i)

        # return return_dict
