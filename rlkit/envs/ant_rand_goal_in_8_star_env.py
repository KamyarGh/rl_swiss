import numpy as np
from collections import OrderedDict
from gym import utils
from rlkit.envs.meta_mujoco_env import MetaMujocoEnv
from rlkit.envs.meta_task_params_sampler import MetaTaskParamsSampler


class ParamsSampler(MetaTaskParamsSampler):
    def __init__(self, random=8392):
        super().__init__()
        if not isinstance(random, np.random.RandomState):
          random = np.random.RandomState(random)
        self._random = random
        
        angles = np.linspace(0.0, 2*np.pi, num=8, endpoint=False)
        r = 25.0
        X = np.cos(angles) * r
        Y = np.sin(angles) * r
        self.goals = np.stack([X, Y], axis=1)

        self._sample_ptr = 0
        self._itr_ptr = 0

    def sample(self):
        p = self.goals[self._sample_ptr]
        self._sample_ptr = (self._sample_ptr + 1) % self.goals.shape[0]
        return {'goal_pos': p}, p

    def sample_unique(self, num):
        idxs = self._random.choice(self.goals.shape[0], size=num, replace=False)
        p_samples = self.goals[idxs]
        return list(
            map(
                lambda p: ({'goal_pos': p}, p),
                p_samples
            )
        )

    def __iter__(self):
        # dangerous
        self._itr_ptr = 0
        return self

    def __next__(self):
        if self._itr_ptr == self.goals.shape[0]:
            self._itr_ptr = 0
            raise StopIteration
        p = self.goals[self._itr_ptr]
        self._itr_ptr += 1
        return {'goal_pos': p}, p


class AntRandGoalIn8StarEnv(MetaMujocoEnv, utils.EzPickle):
    def __init__(self):
        self.goal_pos = np.array([14.1, 14.1])
        self.target_distance = 25.0

        MetaMujocoEnv.__init__(self, 'low_gear_ratio_ant_8_star.xml', 5)
        utils.EzPickle.__init__(self)

    def sample_tasks(self, n_tasks):
        raise NotImplementedError()
        # a = np.random.random(n_tasks) * 2 * np.pi
        # r = 3 * np.random.random(n_tasks) ** 0.5
        # return np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")

        l1_dist = np.sum(np.abs(xposafter[:2] - self.goal_pos))
        l2_dist = np.sqrt(np.sum(np.square(xposafter[:2] - self.goal_pos)))

        # goal_reward = -np.sum(np.abs(xposafter[:2] - self.goal_pos))  # make it happy, not suicidal
        # goal_reward = -np.sum(np.square(xposafter[:2] - self.goal_pos))  # make it happy, not suicidal
        # goal_reward = -1.0 * l2_dist # make it happy, not suicidal
        # goal_reward = -1.0 * l1_dist
        # goal_reward = -1.0 * (l2_dist**2)
        goal_reward = -1.0 * l2_dist

        # ctrl_cost = .1 * np.square(a).sum()
        # ctrl_cost = 0.5 * 1e-2 * np.square(a).sum()
        ctrl_cost = 0.0
        # contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        contact_cost = 0.0
        # survive_reward = 1.0
        survive_reward = 0.0
        # survive_reward = 4.0
        # reward = goal_reward - ctrl_cost - contact_cost + survive_reward

        # I add that so that rewards won't always be negative
        reward = goal_reward + self.target_distance / 2.0
        state = self.state_vector()
        # notdone = np.isfinite(state).all() and 1.0 >= state[2] >= 0.
        # done = not notdone
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            l1_dist=l1_dist,
            l2_dist=l2_dist,
            reward_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        xy_pos = self.get_body_com("torso").flat[:2].copy()
        obs = np.concatenate([
            xy_pos,
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            # np.clip(self.sim.data.cfrc_ext[:14,:], -1, 1).flat,
        ])
        # obs = np.concatenate([
        #     self.sim.data.qpos.flat,
        #     self.sim.data.qvel.flat,
        #     # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        #     self.get_body_com("torso").flat
        # ])
        # print('---')
        # print(self.get_body_com("torso"))
        # print(np.array(self.get_body_com("torso").flat))
        # print(np.concatenate([self.get_body_com("torso").flat]))
        # print(obs)
        return {
            'obs': obs.copy(),
            'obs_task_params': self.goal_pos.copy()
        }

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
    
    def reset(self, task_params=None, obs_task_params=None):
        if task_params is None:
            self.goal_pos = self.sample_tasks(1)[0]
        else:
            self.goal_pos = task_params['goal_pos']
        obs = super().reset()
        return obs
    
    @property
    def task_identifier(self):
        return tuple(self.goal_pos)

    def task_id_to_obs_task_params(self, task_id):
        return np.array(task_id)

    def log_statistics(self, paths):
        # this is run so rarely that it doesn't matter if it's a little inefficient
        progs = [np.mean([d["l1_dist"] for d in path["env_infos"]]) for path in paths]
        last_100_dists = [np.mean([d["l1_dist"] for d in path["env_infos"]][-100:]) for path in paths]
        min_dists = [np.min([d["l1_dist"] for d in path["env_infos"]]) for path in paths]
        ctrl_cost = [-np.mean([d["reward_ctrl"] for d in path["env_infos"]]) for path in paths]
        contact_cost = [-np.mean([d["reward_contact"] for d in path["env_infos"]]) for path in paths]

        l2_min_dists = [np.min([d["l2_dist"] for d in path["env_infos"]]) for path in paths]
        l2_last_100_dists = [np.mean([d["l2_dist"] for d in path["env_infos"]][-100:]) for path in paths]

        return_dict = OrderedDict()
        return_dict['AverageClosest'] = np.mean(min_dists)
        return_dict['MaxClosest'] = np.max(min_dists)
        return_dict['MinClosest'] = np.min(min_dists)
        return_dict['StdClosest'] = np.std(min_dists)

        return_dict['AverageLast100'] = np.mean(last_100_dists)
        return_dict['MaxLast100'] = np.max(last_100_dists)
        return_dict['MinLast100'] = np.min(last_100_dists)
        return_dict['StdLast100'] = np.std(last_100_dists)

        return_dict['AverageForwardReturn'] = np.mean(progs)
        return_dict['MaxForwardReturn'] = np.max(progs)
        return_dict['MinForwardReturn'] = np.min(progs)
        return_dict['StdForwardReturn'] = np.std(progs)

        return_dict['AverageCtrlCost'] = np.mean(ctrl_cost)
        return_dict['AverageContactCost'] = np.mean(contact_cost)

        return_dict['L2AverageClosest'] = np.mean(l2_min_dists)
        return_dict['L2MaxClosest'] = np.max(l2_min_dists)
        return_dict['L2MinClosest'] = np.min(l2_min_dists)
        return_dict['L2StdClosest'] = np.std(l2_min_dists)

        return_dict['L2AverageLast100'] = np.mean(l2_last_100_dists)
        return_dict['L2MaxLast100'] = np.max(l2_last_100_dists)
        return_dict['L2MinLast100'] = np.min(l2_last_100_dists)
        return_dict['L2StdLast100'] = np.std(l2_last_100_dists)
        return return_dict


if __name__ == "__main__":
    # e = _ExpertTrainParamsSampler(num_samples=200)
    e = _ExpertTestParamsSampler(num_samples=200)
    print(e.goals)
    print(e.sample())
    print(e.sample())
    print(e.sample())
    print(e.sample())
    p1 = e.sample()[1]
    p2 = e.sample()[1]
    print(np.linalg.norm(p1 - p2))
    p1 = e.sample()[1]
    p2 = e.sample()[1]
    print(np.linalg.norm(p1 - p2))
    # print(e.sample_unique(10))

    # env = AntRandGoalEnv()
    # while True:
    #     env.reset()
    #     print(env.goal_pos)
    #     for _ in range(100):
    #         # env.render()
    #         obs, reward, _, _ = env.step(env.action_space.sample())  # take a random action
