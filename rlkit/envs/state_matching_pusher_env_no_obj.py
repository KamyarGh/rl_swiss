import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import mujoco_py

from rlkit.core.vistools import plot_seaborn_heatmap, plot_scatter

class PusherTraceEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, obs_with_time=True, episode_len=500):
        self.timestep = 0.0
        self.episode_len = episode_len
        self.obs_with_time = obs_with_time
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(
            self,
            os.path.join(os.path.dirname(__file__), "assets", 'state_matching_pusher_no_obj.xml'),
            5
        )

    def step(self, a):
        reward = 0.0
        self.timestep += 1.0

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False

        xyz = self.get_body_com("tips_arm")
        return ob, reward, done, dict(
            timestep=self.timestep,
            xyz=xyz.copy(),
            a=np.arctan2(xyz[1] + 0.6, xyz[0]) # the +0.6 is because the center is actually at (0,-0.6)
        )

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset(self):
        obs = super().reset()
        self.timestep = 0.0
        return obs

    def reset_model(self):
        qpos = self.init_qpos

        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                high=0.005, size=self.model.nv)
        qvel[-4:] = 0

        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        if self.obs_with_time:
            return np.concatenate([
                self.get_body_com("tips_arm"),
                self.sim.data.qpos.flat[:7],
                self.sim.data.qvel.flat[:7],
                np.array([self.timestep/self.episode_len])
            ])
        else:
            return np.concatenate([
                self.get_body_com("tips_arm"),
                self.sim.data.qpos.flat[:7],
                self.sim.data.qvel.flat[:7],
            ])

    def log_visuals(self, paths, epoch, log_dir):
        xyz = np.array([d['xyz'] for path in paths for d in path['env_info']])
        a = np.array([d['a'] for path in paths for d in path['env_info']])
        
        plot_scatter(
            xyz[:,0],
            xyz[:,1],
            30,
            'Top-Down Epoch %d'%epoch,
            os.path.join(log_dir, 'top_down_epoch_%d.png'%epoch),
            [[-1,1], [-1.6,0.4]]
        )
        plot_scatter(
            a.flatten(),
            xyz[:,2],
            30,
            'AZ Epoch %d'%epoch,
            os.path.join(log_dir, 'a_z_epoch%d.png'%epoch),
            [[-np.pi,np.pi], [-1,1]]
        )

        return {}
