from copy import deepcopy
from time import sleep

import numpy as np
from numpy.random import choice, randint

import gym
from gym.spaces import Discrete, Box
from time import sleep

debug = False

# REMEMBER TO ADD TEXTURE TO THE BACKGROUND LATER


def random_free(grid):
    h, w = grid.shape[1], grid.shape[2]
    x, y = randint(0,h), randint(0, w)

    while any(grid[:, x, y]):
        x, y = randint(0,h), randint(0, w)
    return x, y


class Maze(gym.Env):
    def __init__(self, env_specs):
        self.env_specs = env_specs
        self.num_object_types = self.env_specs['num_object_types']
        self.num_per_object = self.env_specs['num_per_object']
        self.maze_h = self.env_specs['maze_h']
        self.maze_w = self.env_specs['maze_w']
        self.flat_repr = self.env_specs['flat_repr']
        self.one_hot_repr = self.env_specs['one_hot_repr']
        self.chosen_rewards = self.env_specs['chosen_rewards']
        self.chosen_colors = self.env_specs['chosen_colors']
        self.scale = self.env_specs['scale']
        self.timestep_cost = self.env_specs['timestep_cost']
        self.add_noise = self.env_specs['add_noise']
        if self.add_noise:
            self.noise_scale = self.env_specs['noise_scale']

        if self.flat_repr:
            if self.one_hot_repr:
                s = self.maze_h*self.maze_w*(self.num_object_types + 2)
            else:
                s = self.maze_h*self.maze_w*3*self.scale*self.scale
            self.observation_space = Box(low=0, high=1, shape=(s,), dtype=np.float32)
        else:
            if self.one_hot_repr:
                s = (self.num_object_types+2, self.maze_h, self.maze_w)
            else:
                s = (3, self.maze_h*self.scale, self.maze_w*self.scale)
            self.observation_space = Box(low=0, high=1, shape=s, dtype=np.float32)
        self.action_space = Discrete(4)

        self.channel_num = np.arange(1, env_specs['num_object_types']+3)[:,None, None]        

        # for rendering
        self.timestep = 0
        self.non_zero_timesteps=0
        self.total_rewards = 0
        self.last_reward = 0

    def reset(self):
        if debug: print('--------')
        
        self._one_hot = np.zeros((self.num_object_types + 2, self.maze_h, self.maze_w))
        if not self.one_hot_repr:
            self.maze = np.zeros((3, self.maze_h, self.maze_w))
            self.bkgd = np.zeros((3, self.maze_h, self.maze_w))

        # add the objects
        for i in range(self.num_object_types):
            c = self.chosen_colors[i]
            for _ in range(self.num_per_object):
                x, y = random_free(self._one_hot)
                self._one_hot[i,x,y] = 1.0
                if not self.one_hot_repr:
                    self.maze[:,x,y] = c

        # add the termination point
        x, y = random_free(self._one_hot)
        self.cur_pos = [x,y]
        self._one_hot[-2,x,y] = 1
        if not self.one_hot_repr:
            self.maze[:,x,y] = self.chosen_colors[-2]
        
        # add the agent
        x, y = random_free(self._one_hot)
        self.cur_pos = [x,y]
        self._one_hot[-1,x,y] = 1
        if not self.one_hot_repr:
            self.maze[:,x,y] = self.chosen_colors[-1]

        if self.flat_repr:
            if self.one_hot_repr:
                return self._one_hot.flatten()
            return self.maze.flatten()
        else:
            if self.one_hot_repr:
                return self._one_hot
            elif self.scale > 1:
                return np.kron(self.maze, np.ones((1,self.scale,self.scale)))
            return self.maze
    
        self.timestep = 0
        self.non_zero_timesteps=0
        self.total_rewards = 0
        self.last_reward = 0
    

    def render(self):
        if self.timestep == 0:
            print('\n<<<<<<<<<<<<<<<<<<<<<<<<< New Env >>>>>>>>>>>>>>>>>>>>>>>>>')
        obs = self._one_hot
        obs = np.sum(obs * self.channel_num, 0)
        print('\n')
        print(obs)
        print('Got Reward: %.2f' % self.last_reward)
        print('Total Reward: %.2f' % self.total_rewards)
        print('Timestep: %d' % self.timestep)            
        print('Nonzero Timesteps: %d' % self.non_zero_timesteps)
        if self.timestep > 0:
            print('Ratio %.2f' % (self.non_zero_timesteps / self.timestep))
        sleep(0.1)


    def step(self, action):
        if action == 0:
            dx, dy = 1, 0
        elif action == 1:
            dx, dy = 0, 1
        elif action == 2:
            dx, dy = -1, 0
        elif action == 3:
            dx, dy = 0, -1
        
        done = 0.
        if (0 <= self.cur_pos[0] + dx <= self.maze_w-1) and (0 <= self.cur_pos[1] + dy <= self.maze_h-1):
            x = self.cur_pos[0]
            y = self.cur_pos[1]
            self._one_hot[:,x,y] = 0
            if not self.one_hot_repr:
                self.maze[:,x,y] = self.bkgd[:,x,y]

            x += dx
            y += dy

            reward = sum(self._one_hot[:-2,x,y] * self.chosen_rewards)
            if self._one_hot[-2,x,y] == 1: done = 1.
            self._one_hot[:,x,y] = 0
            self._one_hot[-1,x,y] = 1
            if not self.one_hot_repr:
                self.maze[:,x,y] = self.chosen_colors[-1]

            self.cur_pos = [x, y]
        else:
            reward = 0.
        
        # if reward != 0:
        if debug:
            print(reward)
        
        if self.flat_repr:
            if self.one_hot_repr:
                feats = self._one_hot.flatten()
            else:
                feats = self.maze.flatten()
        else:
            if self.one_hot_repr:
                feats = self._one_hot
            else:
                feats = self.maze
                if self.scale > 1:
                    feats = np.kron(feats, np.ones((1,self.scale,self.scale)))
        
        reward -= self.timestep_cost
        self.timestep += 1
        if reward != -1.0 * self.timestep_cost: self.non_zero_timesteps += 1
        self.total_rewards += reward
        self.last_reward = reward

        if self.add_noise: feats = feats + np.random.normal(scale=self.noise_scale)

        return feats, reward, done, {}


class MazeSampler():
    def __init__(self, env_specs):
        self.env_specs = env_specs
        self.colors = [
            [76, 144, 255],
            [255, 180, 76],
            [18, 173, 10],
            [140, 10, 173],
            [13, 17, 147],
            [209, 14, 192],
        ]
        self.colors = np.array(self.colors) / 255.0
        self.shuffle = env_specs['shuffle']
        self.num_object_types = env_specs['num_object_types']
        self.possible_reward_values = np.array(env_specs['possible_reward_values'])

        if not self.shuffle:
            self.chosen_colors = self.colors[choice(self.colors.shape[0], size=self.num_object_types+2, replace=False)]
            self.chosen_rewards = self.possible_reward_values[choice(self.possible_reward_values.shape[0], size=self.num_object_types, replace=False)]
    

    def gen_random_specs(self):
        if self.shuffle:
            chosen_colors = self.colors[choice(self.colors.shape[0], size=self.num_object_types+2, replace=False)]
            chosen_rewards = self.possible_reward_values[choice(self.possible_reward_values.shape[0], size=self.num_object_types, replace=False)]
        else:
            chosen_colors = self.chosen_colors
            chosen_rewards = self.chosen_rewards
        new_dict = deepcopy(self.env_specs)
        new_dict['chosen_colors'] = chosen_colors
        new_dict['chosen_rewards'] = chosen_rewards
        return new_dict

    
    def __call__(self, specs=None):
        if specs is not None:
            return Maze(specs), specs
        specs = self.gen_random_specs()
        return Maze(specs), specs


if __name__ == '__main__':
    from scipy.misc import imsave
    from numpy.random import randint

    env_specs = {
        'base_env_name': 'meta_maze',
        'flat_repr': False,
        'one_hot_repr': True,
        'num_object_types': 1,
        'num_per_object': 3,
        'maze_h': 3,
        'maze_w': 3,
        'possible_reward_values': [1],
        'shuffle': False,
        'scale': 1,
        'timestep_cost': 0.02
    }

    sampler = MazeSampler(env_specs)
    channel_num = np.arange(1, env_specs['num_object_types']+3)[:,None, None]
    random_actions = True
    time_limit = 50

    while True:
        print('\n<<<<<<<<<<<<<<<<<<<<<<<<< New Env >>>>>>>>>>>>>>>>>>>>>>>>>')
        env, specs = sampler()
        obs = env.reset()
        obs = np.sum(obs * channel_num, 0)
        total_reward = 0
        print(obs)
        print('Total Reward: %.2f' % total_reward)
        timestep = 0
        print('Timestep: %d' % timestep)
        non_zero_timesteps = 0
        c = ''
        while  c != 'q':
            if random_actions:
                a = randint(4)
            else:
                c = input('--> ')
                try:
                    a = {'a': 3, 's': 0, 'd': 1, 'w': 2}[c]
                except:
                    continue
            obs, r, done, _ = env.step(a)
            obs = np.sum(obs * channel_num, 0)            
            print('\n')
            print(obs)
            print('Got Reward: %.2f' % r)
            total_reward += r
            print('Total Reward: %.2f' % total_reward)
            timestep += 1
            print('Timestep: %d' % timestep)            
            if r != -1*env_specs['timestep_cost']: non_zero_timesteps += 1
            print('Nonzero Timesteps: %d' % non_zero_timesteps)
            print('Ratio %.2f' % (non_zero_timesteps / timestep))
            if timestep > time_limit: break
            if random_actions: sleep(1)
            if done: break


    # for i in range(5):
    #     print('\nsample %d'%i)
    #     env, specs = sampler()
    #     obs = env.reset()
    #     print(obs)
    #     print(env.chosen_colors)
    #     print(env.chosen_rewards)
    #     env, specs = sampler(specs)        
    #     obs = env.reset()
    #     print(obs)
    #     print(env.chosen_colors)
    #     print(env.chosen_rewards)

    # obs = maze.reset()
    # obs = np.kron(obs, np.ones((1,10,10)))
    # obs = np.transpose(obs, [1,2,0])
    # imsave('plots/debug_maze_env/obs_0.png', obs)
    # for i in range(1,101):
    #     obs, r, d, _ = maze.step(randint(4))
    #     obs = np.kron(obs, np.ones((1,10,10)))
    #     obs = np.transpose(obs, [1,2,0])
    #     imsave('plots/debug_maze_env/obs_%d.png' % i, obs)
    #     print(r, d)

    # print('-----')

    # obs = maze.reset()
    # obs = np.kron(obs, np.ones((1,10,10)))
    # obs = np.transpose(obs, [1,2,0])
    # imsave('plots/debug_maze_env/obs_2_0.png', obs)
    # for i in range(1,21):
    #     obs, r, d, _ = maze.step(randint(4))
    #     obs = np.kron(obs, np.ones((1,10,10)))
    #     obs = np.transpose(obs, [1,2,0])
    #     imsave('plots/debug_maze_env/obs_2_%d.png' % i, obs)
    #     print(r, d)
