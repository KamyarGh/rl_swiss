'''
RLKIT and my extensions do not support OpenAI GoalEnv interface
Here we wrap any GoalEnv we want to use
'''
from gym.envs.robotics import FetchPickAndPlaceEnv
from gym.envs.robotics.fetch.rotated_fetch_anywhere_reach import RotatedFetchAnywhereReachEnv
from gym import spaces
from gym.spaces import Box

import numpy as np

# Sorry, this is very monkey-patchy
class WrappedFetchPickAndPlaceEnv(FetchPickAndPlaceEnv):
    def __init__(self, *args, **kwargs):
        super(WrappedFetchPickAndPlaceEnv, self).__init__(*args, **kwargs)
        fetch_obs_space = self.observation_space
        new_obs_space = spaces.Dict(
            {
                'obs': fetch_obs_space.spaces['observation'],
                'obs_task_params': fetch_obs_space.spaces['desired_goal']
            }
        )
        self.observation_space = new_obs_space


    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        new_obs = {
            'obs': obs['observation'],
            'obs_task_params': obs['desired_goal']
        }
        return new_obs


    def step(self, *args, **kwargs):
        next_ob, raw_reward, terminal, env_info = super().step(*args, **kwargs)
        new_next = {
            'obs': next_ob['observation'],
            'obs_task_params': next_ob['desired_goal']
        }
        env_info['achieved_goal'] = next_ob['achieved_goal']
        return new_next, raw_reward, terminal, env_info


    def log_statistics(self, test_paths):
        rets = []
        path_lens = []
        for path in test_paths:
            rets.append(np.sum(path['rewards']))
            path_lens.append(path['rewards'].shape[0])
        solved = [t[0] > -1.0*t[1] for t in zip(rets, path_lens)]
        percent_solved = np.sum(solved) / float(len(solved))
        return {'Percent_Solved': percent_solved}


class DebugReachFetchPickAndPlaceEnv(FetchPickAndPlaceEnv):
    '''
        This environment is made in a very monkey-patchy way.
        The whole point of this environment is to make simpler
        version of the pick and place where you just have to reach
        for just above the block you have to pick up.

        The desired goal / observed task params is the position just
        above the cube.

        Please try not to use this! It's very hacky. I'm just using
        it for seeing why my models isn't working.
    '''
    def __init__(self, *args, **kwargs):
        super(DebugReachFetchPickAndPlaceEnv, self).__init__(*args, **kwargs)
        fetch_obs_space = self.observation_space
        new_obs_space = spaces.Dict(
            {
                'obs': fetch_obs_space.spaces['observation'],
                'obs_task_params': Box(-np.inf, np.inf, shape=(3,), dtype='float32')
            }
        )
        self.observation_space = new_obs_space


    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        goal = obs['observation'][3:6].copy()
        goal[2] += 0.03
        new_obs = {
            'obs': obs['observation'],
            'obs_task_params': goal
        }
        return new_obs


    def step(self, *args, **kwargs):
        next_ob, raw_reward, terminal, env_info = super().step(*args, **kwargs)
        goal = next_ob['observation'][3:6].copy()
        goal[2] += 0.03
        new_next = {
            'obs': next_ob['observation'],
            'obs_task_params': goal
        }
        dist = np.linalg.norm(next_ob['observation'][:3] - goal)
        if dist < 0.05:
            raw_reward = -dist
        else:
            raw_reward = -1.0

        return new_next, raw_reward, terminal, {}


    def log_statistics(self, test_paths):
        rets = []
        path_lens = []
        for path in test_paths:
            rets.append(np.sum(path['rewards']))
            path_lens.append(path['rewards'].shape[0])
        solved = [t[0] > -1.0*t[1] for t in zip(rets, path_lens)]
        percent_solved = np.sum(solved) / float(len(solved))
        return {'Percent_Solved': percent_solved}


class DebugFetchReachAndLiftEnv(FetchPickAndPlaceEnv):
    '''
        This environment is made in a very monkey-patchy way.
        The whole point of this environment is to make simpler
        version of the pick and place where you just have to reach
        for and lift the block straight up.kiKI

        The desired goal / observed task params is the position 0.2
        above the cube

        Please try not to use this! It's very hacky. I'm just using
        it for seeing why my models isn't working.
    '''
    def __init__(self, *args, **kwargs):
        super(DebugFetchReachAndLiftEnv, self).__init__(*args, **kwargs)
        fetch_obs_space = self.observation_space
        new_obs_space = spaces.Dict(
            {
                'obs': fetch_obs_space.spaces['observation'],
                'obs_task_params': Box(-np.inf, np.inf, shape=(3,), dtype='float32')
            }
        )
        self.observation_space = new_obs_space


    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        self.debug_goal = obs['observation'][3:6].copy()
        self.debug_goal[2] += 0.2
        new_obs = {
            'obs': obs['observation'],
            'obs_task_params': self.debug_goal
        }
        return new_obs


    def step(self, *args, **kwargs):
        next_ob, raw_reward, terminal, env_info = super().step(*args, **kwargs)
        new_next = {
            'obs': next_ob['observation'],
            'obs_task_params': self.debug_goal
        }
        # distance of the cube from the debug goal
        dist = np.linalg.norm(next_ob['observation'][3:6] - self.debug_goal)
        if dist < 0.05:
            raw_reward = -dist
        else:
            raw_reward = -1.0

        return new_next, raw_reward, terminal, {}


    def log_statistics(self, test_paths):
        rets = []
        path_lens = []
        for path in test_paths:
            rets.append(np.sum(path['rewards']))
            path_lens.append(path['rewards'].shape[0])
        solved = [t[0] > -1.0*t[1] for t in zip(rets, path_lens)]
        percent_solved = np.sum(solved) / float(len(solved))
        return {'Percent_Solved': percent_solved}


class WrappedRotatedFetchReachAnywhereEnv(RotatedFetchAnywhereReachEnv):
    '''
        Wrapped
        Also changed the reward and shaped it reward function
    '''
    def __init__(self, *args, **kwargs):
        super(WrappedRotatedFetchReachAnywhereEnv, self).__init__(*args, **kwargs)
        fetch_obs_space = self.observation_space
        new_obs_space = spaces.Dict(
            {
                'obs': fetch_obs_space.spaces['observation'],
                'obs_task_params': fetch_obs_space.spaces['desired_goal']
            }
        )
        self.observation_space = new_obs_space


    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        new_obs = {
            'obs': obs['observation'],
            'obs_task_params': obs['desired_goal']
        }
        self.prev_dist = np.linalg.norm(obs['desired_goal'] - obs['achieved_goal'], axis=-1)
        return new_obs


    def step(self, *args, **kwargs):
        next_ob, raw_reward, terminal, env_info = super().step(*args, **kwargs)
        new_next = {
            'obs': next_ob['observation'],
            'obs_task_params': next_ob['desired_goal']
        }

        if env_info['is_success']:
            raw_reward = 1.0
        else:
            raw_reward = 0.0

        cur_dist = np.linalg.norm(next_ob['desired_goal'] - next_ob['achieved_goal'], axis=-1)
        shaping = self.prev_dist - cur_dist
        self.prev_dist = cur_dist

        return new_next, raw_reward + 10*shaping, terminal, env_info


    def log_statistics(self, test_paths):
        successes = []
        for path in test_paths:
            successes.append(np.sum([e_info['is_success'] for e_info in path['env_infos']]) > 0)
        percent_solved = np.sum(successes) / float(len(successes))
        return {'Percent_Solved': percent_solved}
