import numpy as np
from collections import OrderedDict
from gym import utils
import os

# from rlkit.envs.mujoco_env import MujocoEnv
from gym.envs.mujoco.mujoco_env import MujocoEnv

from rlkit.core.vistools import plot_seaborn_heatmap, plot_scatter


class StateMatchingAntEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, episode_len=499):
        # # 1 x 1 x 8 x 2
        # self.valid_targets = np.array(
        #     [[[
        #         [8.0, 0.0],
        #         [0.0, 8.0],
        #         [-8.0, 0.0],
        #         [0.0, -8.0],
        #     ]]]
        # )

        self.timestep = 0.0
        self.episode_len = episode_len
        # self.init_xy = init_pos.copy()

        xml_path = os.path.join(os.path.dirname(__file__), "assets", 'low_gear_ratio_ant.xml')
        # xml_path = os.path.join(os.path.dirname(__file__), "assets", 's_maze.xml')
        MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)



    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")
        reward = 0.0 # we don't need a reward for what we want to do with this env
        ob = self._get_obs()

        self.timestep += 1.0
        if self.timestep == self.episode_len:
            done = True
        else:
            done = False
        
        return ob, reward, done, dict(
            xy_pos=xposafter[:2].copy(),
            timestep=self.timestep
        )


    def _get_obs(self):
        xy_pos = self.get_body_com("torso").flat[:2].copy()
        # obs = np.concatenate([
        #     xy_pos,
        #     self.sim.data.qpos.flat,
        #     self.sim.data.qvel.flat
        # ])
        obs = np.concatenate([
            xy_pos,
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])
        return obs.copy()


    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()


    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


    def reset(self):
        self.timestep = 0.0
        obs = super().reset()
        return obs


    def log_new_ant_multi_statistics(self, paths, epoch, log_dir):
        # turn the xy pos into arrays you can work with
        # N_paths x path_len x 2
        # xy_pos = np.array([[d['xy_pos'] for d in path["env_infos"]] for path in paths])

        # plot using seaborn heatmap
        xy_pos = [np.array([d['xy_pos'] for d in path["env_infos"]]) for path in paths]
        xy_pos = np.array([d['xy_pos'] for path in paths for d in path['env_infos']])
        
        PLOT_BOUND = 6

        plot_seaborn_heatmap(
            xy_pos[:,0],
            xy_pos[:,1],
            30,
            'Ant Heatmap Epoch %d'%epoch,
            os.path.join(log_dir, 'heatmap_epoch_%d.png'%epoch),
            [[-PLOT_BOUND,PLOT_BOUND], [-PLOT_BOUND,PLOT_BOUND]]
        )
        plot_scatter(
            xy_pos[:,0],
            xy_pos[:,1],
            30,
            'Ant Heatmap Epoch %d'%epoch,
            os.path.join(log_dir, 'scatter_epoch_%d.png'%epoch),
            [[-PLOT_BOUND,PLOT_BOUND], [-PLOT_BOUND,PLOT_BOUND]]
        )

        return {}

        # xy_pos = [np.array([d['xy_pos'] for d in path["env_infos"]]) for path in paths]
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
