import numpy as np
from collections import OrderedDict
from gym import utils
from rlkit.envs.meta_mujoco_env import MetaMujocoEnv
from rlkit.envs.meta_task_params_sampler import MetaTaskParamsSampler


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
        return {'goal_dir': p}, p

    def sample_unique(self, num):
        idxs = self._random.choice(self.goals.shape[0], size=num, replace=False)
        p_samples = self.goals[idxs]
        return list(
            map(
                lambda p: ({'goal_dir': p}, p),
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
        return {'goal_dir': p}, p


class ParamsSampler45to135(_BaseParamsSampler):
    def __init__(self, random=8819, num_samples=41):
        a = np.linspace(np.pi/4.0, 3*np.pi/4.0, num=num_samples, endpoint=True)
        goals = np.stack((np.cos(a), np.sin(a)), axis=-1)
        super().__init__(goals, random=random)


class AntRandDirectionEnv(MetaMujocoEnv, utils.EzPickle):
    def __init__(self):
        self.goal_dir = np.array([1.41, 1.41])
        # MetaMujocoEnv.__init__(self, 'ant.xml', 5)
        MetaMujocoEnv.__init__(self, 'low_gear_ratio_ant.xml', 5)
        utils.EzPickle.__init__(self)

    def sample_tasks(self, n_tasks):
        raise NotImplementedError()
        # a = np.random.random(n_tasks) * 2 * np.pi
        # r = 3 * np.random.random(n_tasks) ** 0.5
        # return np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

    def step(self, a):
        xyposbefore = self.get_body_com("torso")[:2].copy()
        self.do_simulation(a, self.frame_skip)
        xyposafter = self.get_body_com("torso")[:2].copy()

        reward = np.dot(xyposafter, self.goal_dir)
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            xyposbefore=xyposbefore,
            xyposafter=xyposafter,
            travel=np.dot(xyposafter, self.goal_dir)
        )

    def _get_obs(self):
        obs = np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            self.get_body_com("torso")[:2].flat
        ])
        return {
            'obs': obs.copy(),
            'obs_task_params': self.goal_dir.copy()
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
            self.goal_dir = self.sample_tasks(1)[0]
        else:
            self.goal_dir = task_params['goal_dir']
        obs = super().reset()
        return obs
    
    @property
    def task_identifier(self):
        return tuple(self.goal_dir)

    def task_id_to_obs_task_params(self, task_id):
        return np.array(task_id)

    def log_statistics(self, paths):
        # this is run so rarely that it doesn't matter if it's a little inefficient
        travel = [np.max([d["travel"] for d in path["env_infos"]]) for path in paths]

        return_dict = OrderedDict()
        return_dict['AverageTravel'] = np.mean(travel)
        return_dict['MaxTravel'] = np.max(travel)
        return_dict['MinTravel'] = np.min(travel)
        return_dict['StdTravel'] = np.std(travel)
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
