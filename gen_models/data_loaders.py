import numpy as np
from numpy.random import choice

import torch
from torch.autograd import Variable


class RandomDataLoader():
    def __init__(self, next_obs_array, act_array, use_gpu=False):
        self.obs_array = next_obs_array
        self.act_array = act_array

        self.use_gpu = use_gpu


    def get_next_batch(self, batch_size):
        idxs = choice(self.obs_array.shape[0], size=batch_size, replace=False)
        obs_batch = Variable(torch.FloatTensor(self.obs_array[idxs]))
        act_batch = Variable(torch.FloatTensor(self.act_array[idxs]))
        if self.use_gpu:
            obs_batch = obs_batch.cuda()
            act_batch = act_batch.cuda()
        
        return obs_batch, act_batch


class BasicDataLoader():
    def __init__(self, next_obs_array, act_array, episode_length, batch_size, use_gpu=False):
        self.obs_array = next_obs_array
        self.act_array = act_array
        self.episode_length = episode_length
        self.batch_size = batch_size

        self.num_episodes = int(next_obs_array.shape[0] / episode_length)
        self.replace = batch_size > self.num_episodes
        self.ep_idxs = choice(np.arange(self.num_episodes), size=batch_size, replace=self.replace)
        self.cur_t = 0

        self.use_gpu = use_gpu
        self.reset()


    def reset(self):
        self.cur_t = 0
        # self.ep_idxs = choice(np.arange(self.num_episodes), size=self.batch_size, replace=self.replace)
        self.idxs = choice(self.obs_array.shape[0], size=self.batch_size, replace=self.replace)
        self.idxs -= self.idxs % self.episode_length


    def get_next_batch(self):
        if self.cur_t == self.episode_length:
            self.reset()
        
        # idxs = self.ep_idxs * self.episode_length + self.cur_t

        obs_batch = Variable(torch.FloatTensor(self.obs_array[self.idxs]))
        act_batch = Variable(torch.FloatTensor(self.act_array[self.idxs]))
        if self.cur_t == 0: act_batch.zero_()
        if self.use_gpu:
            obs_batch = obs_batch.cuda()
            act_batch = act_batch.cuda()

        self.idxs += 1
        self.cur_t += 1
        return obs_batch, act_batch


class VerySpecificOnTheFLyDataLoader():
    def __init__(self, maze_env_constructor, episode_length, batch_size, use_gpu=False):
        self.maze_env_constructor = maze_env_constructor
        self.episode_length = episode_length
        self.batch_size = batch_size

        self.cur_t = 0
        self.envs = [maze_env_constructor() for _ in range(batch_size)]

        self.use_gpu = use_gpu
        self.reset()


    def reset(self):
        self.cur_t = 0
        self.cur_obs = np.array([env.reset() for env in self.envs])
        self.prev_acts = np.zeros((self.batch_size, 4))


    def get_next_batch(self):
        if self.cur_t == self.episode_length:
            self.reset()
        
        obs_batch = self.cur_obs
        obs_batch = Variable(torch.FloatTensor(obs_batch))
        act_batch = self.prev_acts
        act_batch_pytorch = Variable(torch.FloatTensor(act_batch))
        if self.cur_t == 0: act_batch_pytorch.zero_()
        if self.use_gpu:
            obs_batch = obs_batch.cuda()
            act_batch_pytorch = act_batch_pytorch.cuda()

        # acts = np.random.randint(0, high=4, size=self.batch_size)
        acts = [env.get_good_action() for env in self.envs]
        act_batch = np.zeros((self.batch_size, 4))
        act_batch[np.arange(self.batch_size), acts] = 1.
        self.prev_acts = act_batch
        self.cur_obs = [
            env.step(a)[0] for env, a in zip(self.envs, acts)
        ]
        self.cur_t += 1
        return obs_batch, act_batch_pytorch


if __name__ == '__main__':
    import joblib

    replay_dict = joblib.load('/ais/gobi6/kamyar/oorl_rlkit/maze_trans_data/trivial_grid_rand_policy_4')
    next_obs = replay_dict['next_observations']
    acts = replay_dict['actions']
    dl = BasicDataLoader(next_obs, acts, 50, 2)

    for i in range(10):
        obs, act = dl.get_next_batch()
        print(obs)
        print(act)
