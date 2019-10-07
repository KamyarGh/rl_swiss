'''
Derived from: https://github.com/jonasrothfuss/ProMP/blob/master/meta_policy_search/envs/mujoco_envs/ant_rand_direc_2d.py
'''
import numpy as np
from collections import OrderedDict
from gym import utils
from rlkit.envs.meta_mujoco_env import MetaMujocoEnv
from rlkit.envs.meta_task_params_sampler import MetaTaskParamsSampler
from scipy.spatial.distance import cosine as cos_dist


class _BaseParamsSampler(MetaTaskParamsSampler):
    def __init__(self, goals, random=7823):
        super().__init__()
        if not isinstance(random, np.random.RandomState):
          random = np.random.RandomState(random)
        self._random = random
        self.goals = goals
        self._ptr = 0

    def sample(self):
        p = self.goals[self._random.choice(self.goals.shape[0])]
        return {'goal_direction': p}, p

    def sample_unique(self, num):
        idxs = self._random.choice(self.goals.shape[0], size=num, replace=False)
        p_samples = self.goals[idxs]
        return list(
            map(
                lambda p: ({'goal_direction': p}, p),
                p_samples
            )
        )

    def __iter__(self):
        # dangerous
        self._ptr = 0
        return self

    def __next__(self):
        if self._ptr == self.goals.shape[0]:
            self._ptr = 0
            raise StopIteration
        p = self.goals[self._ptr]
        self._ptr += 1
        return {'goal_direction': p}, p


class _Expert180DegreesParamsSampler(_BaseParamsSampler):
    def __init__(self, random=88374):
        a = np.linspace(0.0, np.pi, num=30, endpoint=True)
        r = 1.0
        goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        super().__init__(goals, random=random)


class _DebugParamsSamplerV1(_BaseParamsSampler):
    def __init__(self, random=88374):
        a = np.linspace(np.pi/12.0, np.pi*11.0/12.0, num=6, endpoint=True)
        r = 1.0
        goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        super().__init__(goals, random=random)


class AntRandDirec2DEnv(MetaMujocoEnv, utils.EzPickle):
    def __init__(self):
        self.goal_direction = np.array([1.0, 0.0])
        # MetaMujocoEnv.__init__(self, 'ant.xml', 5)
        MetaMujocoEnv.__init__(self, 'low_gear_ratio_ant.xml', 5)
        # MetaMujocoEnv.__init__(self, 'low_gear_ratio_ant.xml', 5)
        utils.EzPickle.__init__(self)

    def sample_tasks(self, n_tasks):
        raise NotImplementedError()
        # # for fwd/bwd env, goal direc is backwards if - 1.0, forwards if + 1.0
        # directions = np.random.normal(size=(n_tasks, 2))
        # directions /= np.linalg.norm(directions, axis=1)[..., np.newaxis]
        # return directions

    def step(self, a):
        posbefore = np.copy(self.get_body_com("torso")[:2])
        self.do_simulation(a, self.frame_skip)
        posafter = self.get_body_com("torso")[:2]

        self.ant_pos_before = posbefore.copy()
        self.ant_pos_after = posafter.copy()

        # original
        # forward_reward = np.sum(self.goal_direction * (posafter - posbefore))/self.dt
        # ctrl_cost = .5 * np.square(a).sum()
        # contact_cost = 0.5 * 1e-3 * np.sum(
        #     np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        # survive_reward = 1.0
        
        # # new try v1: cosine similarlity
        # forward_reward = 1.0 - cos_dist(self.goal_direction, posafter - posbefore)
        # ctrl_cost = 0.0
        # contact_cost = 0.0
        # survive_reward = 0.0

        # new try v2: just region clipped forward cost
        forward_reward = np.sum(self.goal_direction * (posafter - posbefore))/self.dt
        ctrl_cost = 0.0
        contact_cost = 0.0
        survive_reward = 0.0

        # clipping based on region
        # for this we also have to clip the rewards from below by 0 so that the agent won't try to do weird stuff
        # clip min by zero
        forward_reward = max(forward_reward, 0.0)
        # clip by region
        if 1.0 - cos_dist(self.goal_direction, posafter) < 0.96:
            forward_reward = 0.0

        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and 1.0 >= state[2] >= 0.
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            goal_direction=self.goal_direction.copy(),
            projected_dist=np.sum(posafter * self.goal_direction),
            debug_target_dist=np.linalg.norm(posafter - self.goal_direction)
        )

    def _get_obs(self):
        # original obs
        # obs = np.concatenate([
        #     self.sim.data.qpos.flat[2:],
        #     self.sim.data.qvel.flat,
        #     np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        # ])
        # a version of obs from
        # https://github.com/tensorflow/models/blob/master/research/efficient-hrl/environments/ant.py
        # obs = np.concatenate([
        #     self.sim.data.qpos.flat[2:],
        #     self.sim.data.qvel.flat,
        # ])
        # EASY OBS
        obs = np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            (self.ant_pos_after - self.ant_pos_before).copy(),
            self.ant_pos_after.copy()
        ])
        return {
            'obs': obs.copy(),
            'obs_task_params': self.goal_direction.copy()
        }

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)

        self.ant_pos_before = np.copy(self.get_body_com("torso")[:2])
        self.ant_pos_after = self.ant_pos_before.copy()

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
    
    def reset(self, task_params=None, obs_task_params=None):
        if task_params is None:
            self.goal_direction = self.sample_tasks(1)[0]
        else:
            self.goal_direction = task_params['goal_direction']
        obs = super().reset()
        return obs
    
    @property
    def task_identifier(self):
        return tuple(self.goal_direction)
    
    def task_id_to_obs_task_params(self, task_id):
        return np.array(task_id)

    def log_statistics(self, paths):    
        progs = [np.mean([d["reward_forward"] for d in path["env_infos"]]) for path in paths]
        ctrl_cost = [-np.mean([d["reward_ctrl"] for d in path["env_infos"]]) for path in paths]
        farthest_projected_dist = [np.max([d["projected_dist"] for d in path["env_infos"]]) for path in paths]
        min_dist_to_debug_target = [np.min([d["debug_target_dist"] for d in path["env_infos"]]) for path in paths]

        return_dict = OrderedDict()
        return_dict['AvgProjDist'] = np.mean(farthest_projected_dist)
        return_dict['MaxProjDist'] = np.max(farthest_projected_dist)
        return_dict['MinProjDist'] = np.min(farthest_projected_dist)
        return_dict['StdProjDist'] = np.std(farthest_projected_dist)

        return_dict['AvgDebugTargetDist'] = np.mean(min_dist_to_debug_target)
        return_dict['MaxDebugTargetDist'] = np.max(min_dist_to_debug_target)
        return_dict['MinDebugTargetDist'] = np.min(min_dist_to_debug_target)
        return_dict['StdDebugTargetDist'] = np.std(min_dist_to_debug_target)

        return_dict['AverageForwardReturn'] = np.mean(progs)
        return_dict['MaxForwardReturn'] = np.max(progs)
        return_dict['MinForwardReturn'] = np.min(progs)
        return_dict['StdForwardReturn'] = np.std(progs)

        return_dict['AverageCtrlCost'] = np.mean(ctrl_cost)
        return return_dict


if __name__ == "__main__":
    p = _Expert180DegreesParamsSampler()
    print(p.goals)

    env = AntRandDirec2DEnv()
    for i in range(10):
        task_params, obs_task_params = p.sample()
        obs = env.reset(task_params=task_params, obs_task_params=obs_task_params)
        print('------')
        print(task_params, obs_task_params)
        print(env.goal_direction)
