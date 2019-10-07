import os
from collections import OrderedDict

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import mujoco_py

from rlkit.core.vistools import plot_seaborn_heatmap, plot_scatter

class PusherSMMEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, obs_with_time=True, episode_len=200):
        self.timestep = 0.0
        self.episode_len = episode_len
        self.obs_with_time = obs_with_time
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(
            self,
            os.path.join(os.path.dirname(__file__), "assets", 'state_matching_pusher.xml'),
            5
        )

    def step(self, a):
        self.timestep += 1.0

        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        reward_near = - np.linalg.norm(vec_1)
        reward_dist = - np.linalg.norm(vec_2)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        if np.linalg.norm(vec_2) < 0.08:
            is_success = 1.0
        else:
            is_success = 0.0

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(
            timestep=self.timestep,
            reward_near=reward_near,
            reward_dist=reward_dist,
            reward_ctrl=reward_ctrl,
            arm_xyz=self.get_body_com("tips_arm").copy(),
            obj_xyz=self.get_body_com("object").copy(),
            success=is_success
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

        self.goal_pos = np.asarray([0, 0])
        while True:
            self.cylinder_pos = np.concatenate([
                    self.np_random.uniform(low=-0.3, high=0, size=1),
                    self.np_random.uniform(low=-0.2, high=0.2, size=1)])
            if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        if self.obs_with_time:
            return np.concatenate([
                self.get_body_com("tips_arm"),
                self.get_body_com("object"),
                self.get_body_com("goal"),
                self.sim.data.qpos.flat[:7],
                self.sim.data.qvel.flat[:7],
                np.array([self.timestep/self.episode_len])
            ]).copy()
        else:
            return np.concatenate([
                self.get_body_com("tips_arm"),
                self.get_body_com("object"),
                self.get_body_com("goal"),
                self.sim.data.qpos.flat[:7],
                self.sim.data.qvel.flat[:7],
            ]).copy()

    def log_visuals(self, paths, epoch, log_dir):
        arm_xyz = np.array([d['arm_xyz'] for path in paths for d in path['env_info']])
        obj_xyz = np.array([d['obj_xyz'] for path in paths for d in path['env_info']])

        min_arm_to_obj = np.array([min(-d['reward_near'] for d in path['env_info']) for path in paths])
        min_obj_to_goal = np.array([min(-d['reward_dist'] for d in path['env_info']) for path in paths])

        successes = [np.sum([d['success'] for d in path['env_info']]) > 0 for path in paths]
        success_rate = float(np.sum(successes)) / float(len(successes))
        
        plot_scatter(
            arm_xyz[:,0],
            arm_xyz[:,1],
            30,
            'Pusher Arm X-Y Epoch %d'%epoch,
            os.path.join(log_dir, 'arm_x_y_epoch_%d.png'%epoch),
            [[-0.5,1], [-1,0.5]]
        )
        plot_scatter(
            arm_xyz[:,1],
            arm_xyz[:,2],
            30,
            'Pusher Arm Y-Z Epoch %d'%epoch,
            os.path.join(log_dir, 'arm_y_z_epoch_%d.png'%epoch),
            [[-1,0.5], [-0.5,0.5]]
        )
        plot_scatter(
            obj_xyz[:,0],
            obj_xyz[:,1],
            30,
            'Obj X-Y Epoch %d'%epoch,
            os.path.join(log_dir, 'obj_x_y_epoch_%d.png'%epoch),
            [[-0.5,1], [-1,0.5]]
        )

        return {}
        
        # return_dict = OrderedDict()
        # return_dict['Success Rate'] = success_rate

        # return_dict['AvgClosestArm2Obj'] = np.mean(min_arm_to_obj)
        # return_dict['MaxClosestArm2Obj'] = np.max(min_arm_to_obj)
        # return_dict['MinClosestArm2Obj'] = np.min(min_arm_to_obj)
        # return_dict['StdClosestArm2Obj'] = np.std(min_arm_to_obj)

        # return_dict['AvgClosestObj2Goal'] = np.mean(min_obj_to_goal)
        # return_dict['MaxClosestObj2Goal'] = np.max(min_obj_to_goal)
        # return_dict['MinClosestObj2Goal'] = np.min(min_obj_to_goal)
        # return_dict['StdClosestObj2Goal'] = np.std(min_obj_to_goal)

        # return return_dict
