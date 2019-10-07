import numpy as np
from collections import OrderedDict
from gym import utils
from rlkit.envs.meta_mujoco_env import MetaMujocoEnv

from rlkit.envs.meta_task_params_sampler import MetaTaskParamsSampler


# These random seeds are literally me typing random numbers
# (I didn't tune my random seed for demo generation LOL)
def get_some_task_params_iterator(train_env=True, num=50):
    # when you fix this make sure really all the colors are correct
    if train_env:
        return _BaseParamsSampler(random=2497, num_colors=num)
    else:
        return _BaseParamsSampler(random=8384, num_colors=num)


class _BaseParamsSampler(MetaTaskParamsSampler):
    def __init__(self, random=None, num_colors=50):
        super().__init__()
        if not isinstance(random, np.random.RandomState):
          random = np.random.RandomState(random)
        self._random = random

        # sample the goal color centers
        self.num_colors = num_colors
        self.goal_color_centers = self._random.uniform(-1.0, 1.0, size=(self.num_colors,3))

    def sample(self):
        idx = self._random.randint(0, self.num_colors)
        color = self.goal_color_centers[idx]
        return {'goal_color_center': color}, color
    
    def sample_unique(self, num):
        idxs = self._random.choice(self.num_colors, size=num, replace=False)
        return list(
            map(lambda color: ({'goal_color_center': color}, color), (self.goal_color_centers[idx] for idx in idxs))
        )

    def __iter__(self):
        # dangerous
        self.itr_ptr = 0
        return self
    
    def __next__(self):
        if self.itr_ptr == self.num_colors: raise StopIteration
        color = self.goal_color_centers[self.itr_ptr]
        self.itr_ptr += 1
        return {'goal_color_center': color}, color


class FewShotAntFetchEnv(MetaMujocoEnv, utils.EzPickle):
    def __init__(self, random=None):
        self.distance_threshold = 1.0
        self.same_color_radius = 0.5
        self.targets = np.array(
            [
                [3.4, 3.4],
                [-3.4, 3.4]
            ]
        )
        self.correct_idx = 0
        self.goal_color_center = self.np_random.uniform(-1.0, 1.0, size=3)
        self.goal_specific_color = self._sample_color_within_radius(self.goal_color_center, self.same_color_radius)
        other_center = self._sample_color_with_min_dist(self.goal_color_center, self.same_color_radius)
        self.other_specific_color = other_center
        MetaMujocoEnv.__init__(self, 'low_gear_ratio_ant.xml', 5)
        utils.EzPickle.__init__(self)


    def _sample_color_within_radius(self, center, radius):
        x = self.np_random.normal(size=3)
        x /= np.linalg.norm(x, axis=-1)
        r = radius
        u = self.np_random.uniform()
        sampled_color = r * (u**(1.0/3.0)) * x + center
        return np.clip(sampled_color, -1.0, 1.0)
    

    def _sample_color_with_min_dist(self, color, min_dist):
        new_color = self.np_random.uniform(-1.0, 1.0, size=3)
        while np.linalg.norm(new_color - color, axis=-1) < min_dist:
            new_color = self.np_random.uniform(-1.0, 1.0, size=3)
        return new_color
    

    def reset(self, task_params=None, obs_task_params=None):
        '''
        task_params = {
            'goal_color_center': np.array...
            'other_color_center': np.array... (optional)
            'specific_color_from_goal_radius': np.array (optional)
            'specific_color_from_other_radius': np.array (optional)
        }

        obs_task_params = np.array([r,g,b]) describing the goal_color_center
        '''
        if task_params is None:
            self.goal_color_center = self.np_random.uniform(-1.0, 1.0, size=3)
            self.goal_specific_color = self._sample_color_within_radius(self.goal_color_center, self.same_color_radius)
            other_center = self._sample_color_with_min_dist(self.goal_color_center, self.same_color_radius)
            self.other_specific_color = other_center
        else:
            # handle the goal color
            self.goal_color_center = task_params['goal_color_center']
            if 'specific_color_from_goal_radius' in task_params:
                self.goal_specific_color = task_params['specific_color_from_goal_radius']
                assert np.linalg.norm(self.goal_color_center - self.goal_specific_color, axis=-1) < self.same_color_radius
            else:
                self.goal_specific_color = self._sample_color_within_radius(self.goal_color_center, self.same_color_radius)
            
            # handle the other color
            if 'specific_color_from_other_radius' in task_params:
                self.other_specific_color = task_params['specific_color_from_other_radius']
            else:
                if 'other_color_center' in task_params:
                    self.other_specific_color = self._sample_color_within_radius(task_params['other_color_center'], self.same_color_radius)
                else:
                    other_center = self._sample_color_with_min_dist(self.goal_color_center, self.same_color_radius)
                    self.other_specific_color = other_center
        
        self.correct_idx = self.np_random.randint(0, 2)
        if self.correct_idx == 0:
            self.object0_color = self.goal_specific_color
            self.object1_color = self.other_specific_color
        else:
            self.object0_color = self.other_specific_color
            self.object1_color = self.goal_specific_color

        obs = super().reset()   
        return obs


    @property
    def task_identifier(self):
        return tuple(self.goal_color_center)


    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")

        correct_pos = self.targets[self.correct_idx]

        l1_dist = np.sum(np.abs(xposafter[:2] - correct_pos))
        l2_dist = np.sqrt(np.sum(np.square(xposafter[:2] - correct_pos)))

        goal_reward = -1.0 * l2_dist

        ctrl_cost = 0.5 * 1e-2 * np.square(a).sum()
        contact_cost = 0.0
        survive_reward = 0.0
        locomotion_reward = goal_reward - ctrl_cost - contact_cost + survive_reward

        is_success = (l2_dist < self.distance_threshold).astype(np.float32)
        fetching_reward = -(l2_dist > self.distance_threshold).astype(np.float32)
        
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        return ob, fetching_reward, done, dict(
            l1_dist=l1_dist,
            l2_dist=l2_dist,
            reward_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            locomotion_reward=locomotion_reward,
            fetching_reward=fetching_reward,
            is_success=is_success,
            correct_idx=self.correct_idx,
            ant_xy_pos=xposafter[:2].copy(),
            correct_target=self.targets[self.correct_idx].copy(),
            incorrect_target=self.targets[1 - self.cor_idx].copy()
        )


    def _get_obs(self):
        obs = np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            self.targets[0],
            self.targets[1]
        ])
        return {
            'obs': obs.copy(),
            'obs_task_params': self.goal_color_center.copy()
        }


    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()


    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
    

    def task_id_to_obs_task_params(self, task_id):
        return np.array(task_id)


    def log_statistics(self, paths):
        l2_min_dists = [np.min([d["l2_dist"] for d in path["env_infos"]]) for path in paths]
        l2_last_100_dists = [np.mean([d["l2_dist"] for d in path["env_infos"]][-100:]) for path in paths]

        # compute proportion of episodes that were fully solved
        successes = []
        for path in test_paths:
            successes.append(np.sum([e_info['is_success'] for e_info in path['env_infos']]) > 0)
        percent_solved = np.sum(successes) / float(len(successes))

        # compute proportion of episodes that the arm reached for the right
        # object but was not necessarily able to pick it up
        # the way I am computing this is a proxy for this, but it will probably
        # be good enough. I check for which one it is closest to in the last 50 timesteps.
        all_reached_for_correct = []
        all_correct_sum_dist = []
        for path in test_paths:
            correct_target = path['env_infos'][0]['correct_target'][None]
            incorrect_target = path['env_infos'][0]['correct_target'][None]
            ant_pos = np.array([d['ant_xy_pos'] for d in path['env_infos']][-50:])

            correct_sum_dist = np.linalg.norm(ant_pos - correct_target, axis=-1).sum()
            incorrect_sum_dist = np.linalg.norm(ant_pos - incorrect_target, axis=-1).sum()
            
            all_reached_for_correct.append((correct_sum_dist < incorrect_sum_dist).astype(np.float32))
            all_correct_sum_dist.append(correct_sum_dist)
        percent_good_reach = np.sum(all_reached_for_correct) / float(len(all_reached_for_correct))

        return_dict = OrderedDict()
        return_dict['Percent_Good_Reach'] = percent_good_reach
        return_dict['Percent_Solved'] = percent_solved
        return_dict['Avg Sum Dist to Cor'] = np.mean(all_correct_sum_dist)
        return_dict['Std Sum Dist to Cor'] = np.std(all_correct_sum_dist)

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
