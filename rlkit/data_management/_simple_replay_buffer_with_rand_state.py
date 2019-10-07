from collections import defaultdict
import random as python_random
from random import sample
from itertools import starmap
from functools import partial

import numpy as np

from rlkit.data_management.replay_buffer import ReplayBuffer

class SimpleReplayBuffer(ReplayBuffer):
    '''
        THE MAX LENGTH OF AN EPISODE SHOULD BE STRICTLY SMALLER THAN THE _size
        OTHERWISE THERE IS A BUG IN TERMINATE_EPISODE
    '''
    def __init__(
        self, max_replay_buffer_size, observation_dim, action_dim,
        discrete_action_dim=False, policy_uses_pixels=False,
        policy_uses_task_params=False, concat_task_params_to_policy_obs=False,
        random_seed=1995
    ):
        prev_np_rand_state = np.random.get_state()
        np.random.seed(random_seed)
        self._np_rand_state = np.random.get_state()
        np.random.set_state(prev_np_rand_state)

        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self.discrete_action_dim = discrete_action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self.policy_uses_pixels = policy_uses_pixels
        self.policy_uses_task_params = policy_uses_task_params
        self.concat_task_params_to_policy_obs = concat_task_params_to_policy_obs
        if isinstance(observation_dim, tuple):
            dims = [d for d in observation_dim]
            dims = [max_replay_buffer_size] + dims
            dims = tuple(dims)
            self._observations = np.zeros(dims)
            self._next_obs = np.zeros(dims)
        elif isinstance(observation_dim, dict):
            # assuming that this is a one-level dictionary
            self._observations = {}
            self._next_obs = {}

            for key, dims in observation_dim.items():
                if isinstance(dims, tuple):
                    dims = tuple([max_replay_buffer_size] + list(dims))
                else:
                    dims = (max_replay_buffer_size, dims)
                self._observations[key] = np.zeros(dims)
                self._next_obs[key] = np.zeros(dims)
            
            # DO NOT USE THEM SORTED
            # HOWEVER YOU USE THEM SHOULD MATCH HOWEVER IT IS USED IN RL_ALGORITHM TO PASS TO THE POLICY
            # self.batching_keys = [k for k in sorted(self._observations.keys()) if k not in ['pixels', 'obs_task_params']]
            # if self.policy_uses_pixels:
            #     self.batching_keys.append('pixels')
            # if self.policy_uses_task_params:
            #     self.batching_keys.append('obs_task_params')
        else:
            self._observations = np.zeros((max_replay_buffer_size, observation_dim))
            self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        # absorbing[0] is if obs was absorbing, absorbing[1] is if next_obs was absorbing
        self._absorbing = np.zeros((max_replay_buffer_size, 2))
        self._top = 0
        self._size = 0

        # keeping track of trajectory boundaries
        # assumption is trajectory lengths are AT MOST the lenght of the replay buffer
        self._cur_start = 0
        self._traj_endpoints = {} # start->end means [start, end)


    def _np_randint(self, *args, **kwargs):
        prev_np_state = np.random.get_state()
        np.random.set_state(self._np_rand_state)
        rets = np.random.randint(*args, **kwargs)
        self._np_rand_state = np.random.get_state()
        np.random.set_state(prev_np_state)
        return rets
    

    def _np_choice(self, *args, **kwargs):
        prev_np_state = np.random.get_state()
        np.random.set_state(self._np_rand_state)
        rets = np.random.choice(*args, **kwargs)
        self._np_rand_state = np.random.get_state()
        np.random.set_state(prev_np_state)
        return rets


    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        # if self.discrete_action_dim:
            # action = np.eye(self._action_dim)[action]            
            # action = np.eye(self._action_space.n)[action]
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        if 'absorbing' in kwargs:
            self._absorbing[self._top] = kwargs['absorbing']

        if terminal:
            temp = (self._top + 1) % self._max_replay_buffer_size
            self._traj_endpoints[self._cur_start] = temp
            self._cur_start = temp

        if isinstance(self._observations, dict):
            for key, obs in observation.items():
                self._observations[key][self._top] = obs
            for key, obs in next_observation.items():
                self._next_obs[key][self._top] = obs
        else:
            self._observations[self._top] = observation
            self._next_obs[self._top] = next_observation
        self._advance()


    def terminate_episode(self):
        if self._cur_start != self._top:
            # if they are equal it means that the previous state was terminal
            # and was handled so there is no need to handle it again
            # THERE WILL BE A BUG HERE IS _size IS NOT STRICTLY LARGER THAN
            # MAX EPISODE LENGTH
            self._traj_endpoints[self._cur_start] = self._top
            self._cur_start = self._top


    def _advance(self):
        if self._top in self._traj_endpoints:
            del self._traj_endpoints[self._top]
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1


    def random_batch(self, batch_size):
        indices = self._np_randint(0, self._size, batch_size)
        return self._get_batch_using_indices(indices)
    
    
    def _get_batch_using_indices(self, indices):
        if isinstance(self._observations, dict):
            if self.policy_uses_task_params:
                if self.concat_task_params_to_policy_obs:
                    obs_to_return = np.concatenate((self._observations['obs'][indices], self._observations['obs_task_params'][indices]), -1)
                    next_obs_to_return = np.concatenate((self._next_obs['obs'][indices], self._next_obs['obs_task_params'][indices]), -1)
                else:
                    raise NotImplementedError()
            else:
                obs_to_return = self._observations['obs'][indices]
                next_obs_to_return = self._next_obs['obs'][indices]
        else:
            obs_to_return = self._observations[indices]
            next_obs_to_return = self._next_obs[indices]
        
        if self.policy_uses_pixels:
            obs_to_return = {
                'obs': obs_to_return,
                'pixels': self._observations['pixels'][indices]
            }
            next_obs_to_return = {
                'obs': next_obs_to_return,
                'pixels': self._next_obs['pixels'][indices]
            }

        return dict(
            observations=obs_to_return,
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=next_obs_to_return,
            absorbing=self._absorbing[indices]
        )
    

    def _get_cont_segment(self, start, end, keys=None):
        if isinstance(self._observations, dict):
            if self.policy_uses_task_params:
                if self.concat_task_params_to_policy_obs:
                    obs_to_return = np.concatenate((self._observations['obs'][start:end], self._observations['obs_task_params'][start:end]), -1)
                    if keys is None or 'next_observations' in keys:
                        next_obs_to_return = np.concatenate((self._next_obs['obs'][start:end], self._next_obs['obs_task_params'][start:end]), -1)
                else:
                    raise NotImplementedError()
            else:
                obs_to_return = self._observations['obs'][start:end]
                if keys is None or 'next_observations' in keys:
                    next_obs_to_return = self._next_obs['obs'][start:end]
        else:
            obs_to_return = self._observations[start:end]
            if keys is None or 'next_observations' in keys:
                next_obs_to_return = self._next_obs[start:end]
        
        if self.policy_uses_pixels:
            obs_to_return = {
                'obs': obs_to_return,
                'pixels': self._observations['pixels'][start:end]
            }
            if keys is None or 'next_observations' in keys:
                next_obs_to_return = {
                    'obs': next_obs_to_return,
                    'pixels': self._next_obs['pixels'][start:end]
                }

        if keys is not None:        
            return_dict = {}
            if 'observations' in keys: return_dict['observations'] = obs_to_return
            if 'actions' in keys: return_dict['actions'] = self._actions[start:end]
            if 'rewards' in keys: return_dict['rewards'] = self._rewards[start:end]
            if 'terminal' in keys: return_dict['terminal'] = self._terminals[start:end]
            if 'next_observations' in keys: return_dict['next_observations'] = next_obs_to_return
            if 'absorbing' in keys: return_dict['absorbing'] = self._absorbing[start:end]
            return return_dict
        return dict(
            observations=obs_to_return,
            actions=self._actions[start:end],
            rewards=self._rewards[start:end],
            terminals=self._terminals[start:end],
            next_observations=next_obs_to_return,
            absorbing=self._absorbing[start:end]
        )
    

    def _get_segment(self, start, end, keys=None):
        if start < end or end == 0:
            if end == 0: end = self._max_replay_buffer_size
            return self._get_cont_segment(start, end, keys)
        
        first_part = self._get_cont_segment(start, self._max_replay_buffer_size, keys)
        sec_part = self._get_cont_segment(0, end, keys)
        # concat them now
        return concat_nested_dicts(first_part, sec_part)
    

    def _get_samples_from_traj(self, start, end, samples_per_traj, keys=None):
        if start < end or end == 0:
            if end == 0: end = self._max_replay_buffer_size
            idxs = np.arange(start, end)
        else:
            idxs = np.concatenate(
                (
                    np.arange(start, self._max_replay_buffer_size),
                    np.arange(0, end)
                ),
                -1
            )
        idxs = self._np_choice(idxs, size=samples_per_traj, replace=idxs.shape[0]<samples_per_traj)
        return self._get_batch_using_indices(idxs)


    def sample_trajs(self, num_trajs, keys=None, samples_per_traj=None):
        # samples_per_traj of None mean use all of the samples
        # old version: starts = sample(self._traj_endpoints.keys(), num_trajs)
        keys_list = list(self._traj_endpoints.keys())
        starts = self._np_choice(keys_list, size=num_trajs, replace=len(keys_list)<num_trajs)
        ends = map(lambda k: self._traj_endpoints[k], starts)

        if samples_per_traj is None:
            return list(
                starmap(lambda s,e: self._get_segment(s,e,keys), zip(starts, ends))
            )
        else:
            return list(
                starmap(lambda s,e: self._get_samples_from_traj(s,e,samples_per_traj,keys), zip(starts, ends))
            )


    def num_steps_can_sample(self):
        return self._size


    def sample_and_remove(self, batch_size):
        assert not isinstance(self._observations, dict), 'not implemented'
        # This function was made for separating out a validation/test set
        # sets the top to the new self._size
        indices = self._np_randint(0, self._size, batch_size)
        samples = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )

        self._observations = np.delete(self._observations, indices, 0)
        self._actions = np.delete(self._actions, indices, 0)
        self._rewards = np.delete(self._rewards, indices, 0)
        self._terminals = np.delete(self._terminals, indices, 0)
        self._next_obs = np.delete(self._next_obs, indices, 0)

        self._size -= batch_size
        self._top = self._size % self._max_replay_buffer_size

        return samples


    def set_buffer_from_dict(self, batch_dict):
        raise NotImplementedError('Not handling absorbing')
        assert not isinstance(self._observations, dict), 'not implemented'
        self._max_replay_buffer_size = max(self._max_replay_buffer_size, batch_dict['observations'].shape[0])
        self._observations = batch_dict['observations']
        self._next_obs = batch_dict['next_observations']
        self._actions = batch_dict['actions']
        self._rewards = batch_dict['rewards']
        self._terminals = batch_dict['terminals']
        self._top = batch_dict['observations'].shape[0] % self._max_replay_buffer_size
        self._size = batch_dict['observations'].shape[0]


    def change_max_size_to_cur_size(self):
        raise NotImplementedError('Not handling absorbing')
        assert not isinstance(self._observations, dict), 'not implemented'
        self._max_replay_buffer_size = self._size
        self._observations = self._observations[:self._size]
        self._next_obs = self._next_obs[:self._size]
        self._actions = self._actions[:self._size]
        self._rewards = self._rewards[:self._size]
        self._terminals = self._terminals[:self._size]
        self._top = min(self._top, self._size) % self._size


class MetaSimpleReplayBuffer():
    def __init__(
            self, max_rb_size_per_task, observation_dim, action_dim,
            discrete_action_dim=False, policy_uses_pixels=False,
            policy_uses_task_params=False, concat_task_params_to_policy_obs=False,
            random_seed=2001
        ):
        prev_py_rand_state = python_random.getstate()
        python_random.seed(random_seed)
        self._py_rand_state = python_random.getstate()
        python_random.setstate(prev_py_rand_state)

        self._obs_dim = observation_dim
        self._act_dim = action_dim
        self._max_rb_size_per_task = max_rb_size_per_task
        self._disc_act_dim = discrete_action_dim
        self._policy_uses_pixels = policy_uses_pixels
        self._policy_uses_task_params = policy_uses_task_params
        self._concat_task_params_to_policy_obs = concat_task_params_to_policy_obs
        p = self._get_partial()
        self.task_replay_buffers = defaultdict(p)


    def _get_partial(self):
        return partial(
            SimpleReplayBuffer,
            self._max_rb_size_per_task,
            self._obs_dim,
            self._act_dim,
            discrete_action_dim=self._disc_act_dim,
            policy_uses_pixels=self._policy_uses_pixels,
            policy_uses_task_params=self._policy_uses_task_params,
            concat_task_params_to_policy_obs=self._concat_task_params_to_policy_obs
        )
    
    
    @property
    def policy_uses_pixels(self):
        return self._policy_uses_pixels
    

    @policy_uses_pixels.setter
    def policy_uses_pixels(self, value):
        if value == self._policy_uses_pixels: return
        
        for srb in self.task_replay_buffers.values():
            srb.policy_uses_pixels = value

        self._policy_uses_pixels = value
        p = self._get_partial()
        self.task_replay_buffers.default_factory = p
    

    @property
    def policy_uses_task_params(self):
        return self._policy_uses_task_params
    

    @policy_uses_pixels.setter
    def policy_uses_task_params(self, value):
        if value == self._policy_uses_task_params: return
            
        for srb in self.task_replay_buffers.values():
            srb.policy_uses_task_params = value

        self._policy_uses_task_params = value
        p = self._get_partial()
        self.task_replay_buffers.default_factory = p
    

    @property
    def concat_task_params_to_policy_obs(self):
        return self._concat_task_params_to_policy_obs
    

    @policy_uses_pixels.setter
    def concat_task_params_to_policy_obs(self, value):
        if value == self._concat_task_params_to_policy_obs: return
        
        for srb in self.task_replay_buffers.values():
            srb.concat_task_params_to_policy_obs = value

        self._concat_task_params_to_policy_obs = value
        p = self._get_partial()
        self.task_replay_buffers.default_factory = p


    def add_path(self, path, task_identifier):
        '''
            task_identifier must be hashable
        '''
        self.task_replay_buffers[task_identifier].add_path(path)
    

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, task_identifier, **kwargs):
        self.task_replay_buffers[task_identifier].add_sample(
            observation, action, reward, terminal, next_observation,
            **kwargs
        )
    

    def terminate_episode(self, task_identifier):
        self.task_replay_buffers[task_identifier].terminate_episode()
    

    def sample_trajs(self, num_trajs_per_task, num_tasks=1, task_identifiers=None, keys=None, samples_per_traj=None):
        if task_identifiers is None:
            sample_params = list(sample(self.task_replay_buffers.keys(), num_tasks))
        else:
            sample_params = task_identifiers
        batch_list = [
            self.task_replay_buffers[p].sample_trajs(num_trajs_per_task, keys=keys, samples_per_traj=samples_per_traj) \
            for p in sample_params
        ]
        return batch_list, sample_params
    

    def sample_trajs_from_task(self, task_identifier, num_trajs, samples_per_traj=None):
        return self.task_replay_buffers[task_identifier].sample_trajs(num_trajs, samples_per_traj=samples_per_traj)
    

    def random_batch(self, *args, **kwargs):
        return self.sample_random_batch(*args, **kwargs)
    def sample_random_batch(self, batch_size_per_task, num_task_params=1, task_identifiers_list=None):
        if task_identifiers_list is None:
            sample_params = list(sample(self.task_replay_buffers.keys(), num_task_params))
        else:
            sample_params = task_identifiers_list
        batch_list = [
            self.task_replay_buffers[p].random_batch(batch_size_per_task) \
            for p in sample_params
        ]
        return batch_list, sample_params


    def num_steps_can_sample(self):
        return sum(map(lambda rb: rb.num_steps_can_sample(), self.task_replay_buffers.values()))


def concat_nested_dicts(d1, d2):
    # two dicts that have the exact same nesting structure
    # and contain leaf values that are numpy arrays of the same
    # shape except for the first dimensions
    return {
        k: np.concatenate((d1[k], d2[k]), axis=0) if not isinstance(d1[k], dict) \
        else concat_nested_dicts(d1[k], d2[k]) \
        for k in d1
    }
