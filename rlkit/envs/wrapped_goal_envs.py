'''
RLKIT and my extensions do not support OpenAI GoalEnv interface
Here we wrap any GoalEnv we want to use
'''
from gym.envs.robotics import FetchPickAndPlaceEnv
# from gym.envs.robotics.fetch.rotated_fetch_anywhere_reach import RotatedFetchAnywhereReachEnv
from rlkit.envs.easy_fetch_env import EasyFetchPickAndPlaceEnv, SuperEasyFetchPickAndPlaceEnv
from rlkit.envs.easy_fetch_env import TargetOnlyInAirFetchPickAndPlaceEnv
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


# class WrappedRotatedFetchReachAnywhereEnv(RotatedFetchAnywhereReachEnv):
#     '''
#         Wrapped
#         Also changed the reward and shaped it reward function
#     '''
#     def __init__(self, *args, **kwargs):
#         super(WrappedRotatedFetchReachAnywhereEnv, self).__init__(*args, **kwargs)
#         fetch_obs_space = self.observation_space
#         new_obs_space = spaces.Dict(
#             {
#                 'obs': fetch_obs_space.spaces['observation'],
#                 'obs_task_params': fetch_obs_space.spaces['desired_goal']
#             }
#         )
#         self.observation_space = new_obs_space


#     def reset(self, *args, **kwargs):
#         obs = super().reset(*args, **kwargs)
#         new_obs = {
#             'obs': obs['observation'],
#             'obs_task_params': obs['desired_goal']
#         }
#         self.prev_dist = np.linalg.norm(obs['desired_goal'] - obs['achieved_goal'], axis=-1)
#         return new_obs


#     def step(self, *args, **kwargs):
#         next_ob, raw_reward, terminal, env_info = super().step(*args, **kwargs)
#         new_next = {
#             'obs': next_ob['observation'],
#             'obs_task_params': next_ob['desired_goal']
#         }

#         if env_info['is_success']:
#             raw_reward = 1.0
#         else:
#             raw_reward = 0.0

#         cur_dist = np.linalg.norm(next_ob['desired_goal'] - next_ob['achieved_goal'], axis=-1)
#         shaping = self.prev_dist - cur_dist
#         self.prev_dist = cur_dist

#         return new_next, raw_reward + 10*shaping, terminal, env_info


#     def log_statistics(self, test_paths):
#         successes = []
#         for path in test_paths:
#             successes.append(np.sum([e_info['is_success'] for e_info in path['env_infos']]) > 0)
#         percent_solved = np.sum(successes) / float(len(successes))
#         return {'Percent_Solved': percent_solved}


class WrappedEasyFetchPickAndPlaceEnv(EasyFetchPickAndPlaceEnv):
    '''
        Wrapped
        Also changed the reward and shaped it reward function
    '''
    def __init__(self, *args, **kwargs):
        super(WrappedEasyFetchPickAndPlaceEnv, self).__init__(*args, **kwargs)
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
        return new_next, raw_reward, terminal, env_info


    def log_statistics(self, test_paths):
        successes = []
        for path in test_paths:
            successes.append(np.sum([e_info['is_success'] for e_info in path['env_infos']]) > 0)
        percent_solved = np.sum(successes) / float(len(successes))
        return {'Percent_Solved': percent_solved}


class ScaledWrappedEasyFetchPickAndPlaceEnv(EasyFetchPickAndPlaceEnv):
    '''
        Wrapped
    '''
    def __init__(self, acts_min, acts_max, obs_min, obs_max, goal_min, goal_max, SCALE, *args, **kwargs):
        super(ScaledWrappedEasyFetchPickAndPlaceEnv, self).__init__(*args, **kwargs)
        fetch_obs_space = self.observation_space
        new_obs_space = spaces.Dict(
            {
                'obs': fetch_obs_space.spaces['observation'],
                'obs_task_params': fetch_obs_space.spaces['desired_goal']
            }
        )
        self.observation_space = new_obs_space
        self.acts_min, self.acts_max = acts_min, acts_max
        self.obs_min, self.obs_max = obs_min, obs_max
        self.goal_min, self.goal_max = goal_min, goal_max
        self.SCALE = SCALE


    def _normalize_obs(self, observation):
        observation = (observation - self.obs_min) / (self.obs_max - self.obs_min)
        observation *= 2 * self.SCALE
        observation -= self.SCALE
        return observation
    

    def _normalize_goal(self, goal):
        goal = (goal - self.goal_min) / (self.goal_max - self.goal_min)
        goal *= 2 * self.SCALE
        goal -= self.SCALE
        return goal
    

    def _unnormalize_act(self, act):
        return self.acts_min + (act + self.SCALE)*(self.acts_max - self.acts_min) / (2 * self.SCALE)


    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)

        observation = obs['observation']
        desired_goal = obs['desired_goal']

        # normalize them
        observation = self._normalize_obs(observation)
        desired_goal = self._normalize_goal(desired_goal)

        new_obs = {
            'obs': observation,
            'obs_task_params': desired_goal
        }

        return new_obs


    def step(self, action):
        action = self._unnormalize_act(action)
        next_ob, raw_reward, terminal, env_info = super().step(action)

        observation = next_ob['observation']
        desired_goal = next_ob['desired_goal']

        # normalize them
        observation = self._normalize_obs(observation)
        desired_goal = self._normalize_goal(desired_goal)

        new_next = {
            'obs': observation,
            'obs_task_params': desired_goal
        }
        return new_next, raw_reward, terminal, env_info


    def log_statistics(self, test_paths):
        successes = []
        for path in test_paths:
            successes.append(np.sum([e_info['is_success'] for e_info in path['env_infos']]) > 0)
        percent_solved = np.sum(successes) / float(len(successes))
        return {'Percent_Solved': percent_solved}


class ScaledWrappedSuperEasyFetchPickAndPlaceEnv(SuperEasyFetchPickAndPlaceEnv):
    '''
        Wrapped
        Scaled

        Also observation is my custon defined observation not the original fetch observation
    '''
    def __init__(self, acts_min, acts_max, obs_min, obs_max, SCALE, *args, **kwargs):
        super(ScaledWrappedSuperEasyFetchPickAndPlaceEnv, self).__init__(*args, **kwargs)
        # grip rel to obj, obj rel to goal, gripper state, gripper vel (could probably remove this one even)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3+3+2+2,), dtype='float32')
        self.acts_min, self.acts_max = acts_min, acts_max
        self.obs_min, self.obs_max = obs_min, obs_max
        self.SCALE = SCALE


    def _normalize_obs(self, observation):
        observation = (observation - self.obs_min) / (self.obs_max - self.obs_min)
        observation *= 2 * self.SCALE
        observation -= self.SCALE
        return observation
    

    def _unnormalize_act(self, act):
        return self.acts_min + (act + self.SCALE)*(self.acts_max - self.acts_min) / (2 * self.SCALE)


    def _convert_obs_to_custom_obs(self, obs):
        observation = obs['observation']
        goal = obs['desired_goal']

        objectPos = observation[3:6]
        gripperPos = observation[:3]
        gripperState = observation[9:11]
        gripperVel = observation[-2:]

        new_obs = np.concatenate(
            (
                objectPos - gripperPos,
                goal - objectPos,
                gripperState,
                gripperVel
            ),
            axis=0
        )
        return new_obs.copy()


    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        conv_obs = self._convert_obs_to_custom_obs(obs)
        return self._normalize_obs(conv_obs)


    def step(self, action):
        action = self._unnormalize_act(action)
        next_ob, raw_reward, terminal, env_info = super().step(action)
        conv_next = self._convert_obs_to_custom_obs(next_ob)
        new_next = self._normalize_obs(conv_next)
        return new_next, raw_reward, terminal, env_info


    def log_statistics(self, test_paths):
        successes = []
        for path in test_paths:
            successes.append(np.sum([e_info['is_success'] for e_info in path['env_infos']]) > 0)
        percent_solved = np.sum(successes) / float(len(successes))
        return {'Percent_Solved': percent_solved}


class ScaledWrappedFetchPickAndPlaceEnv(FetchPickAndPlaceEnv):
    '''
        Wrapped
        Scaled

        Also observation is my custon defined observation not the original fetch observation
    '''
    def __init__(self, acts_min, acts_max, obs_min, obs_max, SCALE, *args, **kwargs):
        super(ScaledWrappedFetchPickAndPlaceEnv, self).__init__(*args, **kwargs)
        # grip rel to obj, obj rel to goal, gripper state, gripper vel (could probably remove this one even)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3+3+2+2,), dtype='float32')
        self.acts_min, self.acts_max = acts_min, acts_max
        self.obs_min, self.obs_max = obs_min, obs_max
        self.SCALE = SCALE

        self._max_episode_steps = 100


    def _normalize_obs(self, observation):
        observation = (observation - self.obs_min) / (self.obs_max - self.obs_min)
        observation *= 2 * self.SCALE
        observation -= self.SCALE
        return observation
    
    
    def _unnormalize_act(self, act):
        return self.acts_min + (act + self.SCALE)*(self.acts_max - self.acts_min) / (2 * self.SCALE)


    def _convert_obs_to_custom_obs(self, obs):
        observation = obs['observation']
        goal = obs['desired_goal']

        objectPos = observation[3:6]
        gripperPos = observation[:3]
        gripperState = observation[9:11]
        gripperVel = observation[-2:]

        new_obs = np.concatenate(
            (
                objectPos - gripperPos,
                goal - objectPos,
                gripperState,
                gripperVel
            ),
            axis=0
        )
        return new_obs.copy()


    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        conv_obs = self._convert_obs_to_custom_obs(obs)
        return self._normalize_obs(conv_obs)


    def step(self, action):
        action = self._unnormalize_act(action)
        next_ob, raw_reward, terminal, env_info = super().step(action)
        conv_next = self._convert_obs_to_custom_obs(next_ob)
        new_next = self._normalize_obs(conv_next)
        return new_next, raw_reward, terminal, env_info


    def log_statistics(self, test_paths):
        successes = []
        for path in test_paths:
            successes.append(np.sum([e_info['is_success'] for e_info in path['env_infos']]) > 0)
        percent_solved = np.sum(successes) / float(len(successes))
        return {'Percent_Solved': percent_solved}


class ScaledWrappedTargetOnlyInAirFetchPickAndPlaceEnv(TargetOnlyInAirFetchPickAndPlaceEnv):
    '''
        Wrapped
        Scaled

        Also observation is my custon defined observation not the original fetch observation
    '''
    def __init__(self, acts_min, acts_max, obs_min, obs_max, SCALE, *args, **kwargs):
        super(ScaledWrappedTargetOnlyInAirFetchPickAndPlaceEnv, self).__init__(*args, **kwargs)
        # grip rel to obj, obj rel to goal, gripper state, gripper vel (could probably remove this one even)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3+3+2+2,), dtype='float32')
        self.acts_min, self.acts_max = acts_min, acts_max
        self.obs_min, self.obs_max = obs_min, obs_max
        self.SCALE = SCALE

        self._max_episode_steps = 65
        # self._max_episode_steps = 70
        # self._max_episode_steps = 55


    def _normalize_obs(self, observation):
        observation = (observation - self.obs_min) / (self.obs_max - self.obs_min)
        observation *= 2 * self.SCALE
        observation -= self.SCALE
        return observation
    
    
    def _unnormalize_act(self, act):
        return self.acts_min + (act + self.SCALE)*(self.acts_max - self.acts_min) / (2 * self.SCALE)


    def _convert_obs_to_custom_obs(self, obs):
        observation = obs['observation']
        goal = obs['desired_goal']

        objectPos = observation[3:6]
        gripperPos = observation[:3]
        gripperState = observation[9:11]
        gripperVel = observation[-2:]

        new_obs = np.concatenate(
            (
                objectPos - gripperPos,
                goal - objectPos,
                gripperState,
                gripperVel
            ),
            axis=0
        )
        return new_obs.copy()


    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        conv_obs = self._convert_obs_to_custom_obs(obs)
        return self._normalize_obs(conv_obs)


    def step(self, action):
        action = self._unnormalize_act(action)
        next_ob, raw_reward, terminal, env_info = super().step(action)
        conv_next = self._convert_obs_to_custom_obs(next_ob)
        new_next = self._normalize_obs(conv_next)
        return new_next, raw_reward, terminal, env_info


    def log_statistics(self, test_paths):
        successes = []
        for path in test_paths:
            successes.append(np.sum([e_info['is_success'] for e_info in path['env_infos']]) > 0)
        percent_solved = np.sum(successes) / float(len(successes))
        return {'Percent_Solved': percent_solved}
