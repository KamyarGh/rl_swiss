import joblib

import numpy as np
from numpy.random import choice, randint

from rlkit.data_management.env_replay_buffer import get_dim as gym_get_dim
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from rlkit.envs.maze_envs.trivial_grid import TrivialGrid
from rlkit.envs.maze_envs.pogrid import PartiallyObservedGrid
from rlkit.envs.maze_envs.mem_map_grid import MemoryGrid


class RandomGridPolicy():
    def __init__(self, max_num_consecutive):
        self.max_num_consecutive = max_num_consecutive
        self.cur_act = randint(4)
        self.num_left = randint(max_num_consecutive) + 1


    def get_action(self, obs, *args):
        if self.num_left == 0:
            self.num_left = randint(self.max_num_consecutive) + 1
            self.cur_act = randint(4)
        self.num_left -= 1
        return self.cur_act

    def reset(self):
        pass


class ListPolicy():
    def __init__(self, act_list):
        self.act_list = act_list
        self.ptr = 0

    def get_action(self, obs, *args):
        a = self.act_list[self.ptr]
        self.ptr = (self.ptr + 1) % len(self.act_list)
        return a

    def reset(self):
        self.ptr = 0


def generate_transitions(policy, env, num_timesteps_total, max_steps_per_episode, save_path): 
    buff = SimpleReplayBuffer(
        num_timesteps_total, env.observation_space.shape,
        gym_get_dim(env.action_space), discrete_action_dim=True
    )

    cur_total = 0
    steps_left_in_episode = 0
    while cur_total != num_timesteps_total:
        if steps_left_in_episode == 0:
            steps_left_in_episode = max_steps_per_episode
            obs = env.reset()
        
        act = policy.get_action(obs)
        next_obs, rew, done, _ = env.step(act)
        buff.add_sample(obs, act, rew, done, next_obs)

        obs = next_obs
        cur_total += 1
        steps_left_in_episode -= 1
    
    save_dict = dict(
        observations=buff._observations,
        actions=buff._actions,
        rewards=buff._rewards,
        terminals=buff._terminals,
        next_observations=buff._next_obs,
    )
    joblib.dump(save_dict, save_path)

    # debug
    from scipy.misc import imsave
    actions = buff._actions
    observations = buff._observations
    for i in range(1000):
        a = actions[i]
        obs = observations[i]
        print(a)
        imsave('junk_vis/tiny/mem_grid_{}.png'.format(i), np.transpose(obs, (1,2,0)))

    # for i in range(90, 110):
    #     a = actions[i]
    #     obs = observations[i]
    #     print(a)
    #     imsave('junk_vis/maze_{}.png'.format(i), np.transpose(obs, (1,2,0)))

    # for i in range(70, 90):
    #     a = actions[i]
    #     obs = observations[i]
    #     print(a)
    #     imsave('junk_vis/maze_{}.png'.format(i), np.transpose(obs, (1,2,0)))
    
    # for i in range(110, 130):
    #     a = actions[i]
    #     obs = observations[i]
    #     print(a)
    #     imsave('junk_vis/maze_{}.png'.format(i), np.transpose(obs, (1,2,0)))


if __name__ == '__main__':
    # env_specs = {
    #     'flat_repr': False,
    #     'one_hot_repr': False,
    #     'maze_h': 4,
    #     'maze_w': 4,
    #     'scale': 1,
    # }
    # env = TrivialGrid(env_specs)
    # policy = RandomGridPolicy(1)


    # env_specs = {
    #     'flat_repr': False,
    #     'one_hot_repr': False,
    #     'maze_h': 9,
    #     'maze_w': 9,
    #     'obs_h': 5,
    #     'obs_w': 5,
    #     'scale': 4,
    #     'num_objs': 10 
    # }
    # act_list = [1, 0, 3, 2]
    # env = MemoryGrid(env_specs)
    # policy = ListPolicy(act_list)

    env_specs = {
        'flat_repr': False,
        'one_hot_repr': False,
        'maze_h': 9,
        'maze_w': 9,
        'obs_h': 5,
        'obs_w': 5,
        'scale': 4,
        'num_objs': 10 
    }
    act_list = [1, 0, 3, 2]
    env = PartiallyObservedGrid(env_specs)
    policy = RandomGridPolicy(1)

    generate_transitions(policy, env, 50000, 8, '/ais/gobi6/kamyar/oorl_rlkit/maze_trans_data/pogrid_len_8_scale_4')



# 32, 128, 512 for 3x3
# 32, 512, 2048 for 5x5