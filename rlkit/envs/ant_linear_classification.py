'''
Derived from: https://github.com/jonasrothfuss/ProMP/blob/master/meta_policy_search/envs/mujoco_envs/ant_rand_goal.py
However, the task is actually different. We are asking the ant to navigate to points on the perimeter
of the circle, not inside the circle.
'''
import numpy as np
from collections import OrderedDict
from gym import utils
from rlkit.envs.meta_mujoco_env import MetaMujocoEnv
from rlkit.envs.meta_task_params_sampler import MetaTaskParamsSampler


class _BaseParamsSampler(MetaTaskParamsSampler):
    def __init__(self, tasks, random=7823):
        super().__init__()
        if not isinstance(random, np.random.RandomState):
            random = np.random.RandomState(random)
        self._random = random
        self.tasks = tasks
        self._ptr = 0

    def sample(self):
        p = self.tasks[self._random.choice(self.tasks.shape[0])]
        return {'task_hyperplane': p}, p

    def sample_unique(self, num):
        idxs = self._random.choice(self.tasks.shape[0], size=num, replace=False)
        p_samples = self.tasks[idxs]
        return list(
            map(
                lambda p: ({'task_hyperplane': p}, p),
                p_samples
            )
        )

    def __iter__(self):
        # dangerous
        self._ptr = 0
        return self

    def __next__(self):
        if self._ptr == self.tasks.shape[0]:
            self._ptr = 0
            raise StopIteration
        p = self.tasks[self._ptr]
        self._ptr += 1
        return {'task_hyperplane': p}, p


class _ExpertTrainParamsSampler(_BaseParamsSampler):
    def __init__(self, random=77129, num_tasks=64):
        if not isinstance(random, np.random.RandomState):
            random = np.random.RandomState(random)
        tasks = random.normal(size=(num_tasks, 4))
        tasks /= np.linalg.norm(tasks, axis=-1, keepdims=True)
        super().__init__(tasks, random=random)


class _ExpertTestParamsSampler(_ExpertTrainParamsSampler):
    def __init__(self, random=66082, num_tasks=64):
        super().__init__(random=random, num_tasks=num_tasks)


class _ExpertEvalParamsSampler(_ExpertTrainParamsSampler):
    def __init__(self, random=10101, num_tasks=128):
        super().__init__(random=random, num_tasks=num_tasks)


class AntLinearClassifierEnv(MetaMujocoEnv, utils.EzPickle):
    def __init__(self, use_relative_pos=False):
        self.ACCEPT_RADIUS = 0.5
        self.true_label = 0
        self.task_hyperplane = np.ones(4) / 2.0
        self.first_sample = np.random.uniform(size=4)
        self.second_sample = np.random.uniform(size=4)
        self.targets = np.array(
            [
                [1.41, 1.41],
                [-1.41, 1.41]
            ]
        )
        self.use_relative_pos = use_relative_pos
        MetaMujocoEnv.__init__(self, 'low_gear_ratio_ant.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")

        dist_to_correct = np.linalg.norm(xposafter[:2] - self.targets[self.true_label])
        dist_to_incorrect = np.linalg.norm(xposafter[:2] - self.targets[1-self.true_label])

        within_radius_of_correct = dist_to_correct < self.ACCEPT_RADIUS
        within_radius_of_incorrect = dist_to_incorrect < self.ACCEPT_RADIUS

        reward = float(within_radius_of_correct)
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            dist_to_correct=dist_to_correct,
            dist_to_incorrect=dist_to_incorrect,
            true_label=self.true_label,
            within_radius_of_correct=within_radius_of_correct,
            within_radius_of_incorrect=within_radius_of_incorrect
        )

    def _get_obs(self):
        if self.use_relative_pos:
            xy_pos = self.get_body_com("torso").flat[:2]
            obs = np.concatenate([
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
                self.targets[0] - xy_pos,
                self.targets[1] - xy_pos,
                self.first_sample,
                self.second_sample
            ])
        else:
            obs = np.concatenate([
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
                self.get_body_com("torso").flat,
                self.targets[0],
                self.targets[1],
                self.first_sample,
                self.second_sample
            ])
        return {
            'obs': obs.copy(),
            'obs_task_params': self.task_hyperplane.copy()
        }
    
    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
    
    def sample_points_for_cur_task(self):
        x = 2*np.random.uniform(size=4) - 1
        while np.dot(self.task_hyperplane, x) < 0:
            x = 2*np.random.uniform(size=4) - 1
        # print(np.dot(self.task_hyperplane, x))
        pos_sample = x.copy()
        # print(pos_sample)
        
        x = 2*np.random.uniform(size=4) - 1
        while np.dot(self.task_hyperplane, x) > 0:
            x = 2*np.random.uniform(size=4) - 1
        # print(np.dot(self.task_hyperplane, x))
        neg_sample = x.copy()
        # print(neg_sample)

        if np.random.uniform() > 0.5:
            self.true_label = 1
            self.first_sample = pos_sample
            self.second_sample = neg_sample
        else:
            self.true_label = 0
            self.first_sample = neg_sample
            self.second_sample = pos_sample

    def reset(self, task_params=None, obs_task_params=None):
        if task_params is None:
            raise NotImplementedError()
        else:
            self.task_hyperplane = task_params['task_hyperplane']
        self.sample_points_for_cur_task()
        obs = super().reset()
        return obs
    
    @property
    def task_identifier(self):
        return tuple(self.task_hyperplane)

    def task_id_to_obs_task_params(self, task_id):
        return np.array(task_id)

    def log_statistics(self, paths):
        success = [np.sum([d["within_radius_of_correct"] for d in path["env_infos"]]) > 0 for path in paths]
        went_to_incorrect = [np.sum([d["within_radius_of_incorrect"] for d in path["env_infos"]]) > 0 for path in paths]
        no_op = [np.sum([d["within_radius_of_correct"] or d["within_radius_of_incorrect"] for d in path["env_infos"]]) == 0 for path in paths]
        min_dist_to_correct = [np.min([d["dist_to_correct"] for d in path["env_infos"]]) for path in paths]
        min_dist_to_incorrect = [np.min([d["dist_to_incorrect"] for d in path["env_infos"]]) for path in paths]

        num_paths = float(len(paths))
        perc_success = np.sum(success) / num_paths
        perc_incorrect = np.sum(went_to_incorrect) / num_paths
        perc_no_op = np.sum(no_op) / num_paths

        return_dict = OrderedDict()
        return_dict['Perc Success'] = perc_success
        return_dict['Perc Incorrect'] = perc_incorrect
        return_dict['Perc No-Op'] = perc_no_op

        return_dict['AverageClosest Correct'] = np.mean(min_dist_to_correct)
        return_dict['MaxClosest Correct'] = np.max(min_dist_to_correct)
        return_dict['MinClosest Correct'] = np.min(min_dist_to_correct)
        return_dict['StdClosest Correct'] = np.std(min_dist_to_correct)

        return_dict['AverageClosest Incorrect'] = np.mean(min_dist_to_incorrect)
        return_dict['MaxClosest Incorrect'] = np.max(min_dist_to_incorrect)
        return_dict['MinClosest Incorrect'] = np.min(min_dist_to_incorrect)
        return_dict['StdClosest Incorrect'] = np.std(min_dist_to_incorrect)
        return return_dict


if __name__ == "__main__":
    params_sampler = _ExpertTrainParamsSampler()
    env = AntLinearClassifierEnv()

    task_params, obs_task_params = params_sampler.sample()
    obs = env.reset(task_params=task_params, obs_task_params=obs_task_params)
    for _ in range(20):
        env.step(env.action_space.sample())
    
    for _ in range(10):
        print('\n\n\n--------------------------------')
        task_params, obs_task_params = params_sampler.sample()
        print(obs_task_params)
        for i in range(12):
            print('-----')
            obs = env.reset(task_params=task_params, obs_task_params=obs_task_params)

            print('\n')
            print(env.first_sample, env.second_sample)
            obs, _, _, env_info = env.step(env.action_space.sample())
            print(env.first_sample, env.second_sample)

            print('\n')
            obs = obs['obs']
            print(np.dot(obs[-8:-4], env.task_hyperplane))
            print(np.dot(obs[-4:], env.task_hyperplane))
            print(env.true_label)
