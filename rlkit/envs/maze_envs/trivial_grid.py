from copy import deepcopy
from time import sleep

import numpy as np
from numpy.random import choice, randint

import gym
from gym.spaces import Discrete, Box
from time import sleep

debug = False

# REMEMBER TO ADD TEXTURE TO THE BACKGROUND LATER

def get_color():
    c = np.random.beta(0.5, 0.5, size=3)
    c = np.clip(c, 0.1, 0.9)
    return c

def random_free(grid):
    h, w = grid.shape[1], grid.shape[2]
    x, y = randint(0,h), randint(0, w)

    while any(grid[:, x, y]):
        x, y = randint(0,h), randint(0, w)
    return x, y


class TrivialGrid(gym.Env):
    def __init__(self, env_specs):
        self.env_specs = env_specs
        self.maze_h = self.env_specs['maze_h']
        self.maze_w = self.env_specs['maze_w']
        self.flat_repr = self.env_specs['flat_repr']
        self.one_hot_repr = self.env_specs['one_hot_repr']
        self.scale = self.env_specs['scale']

        if self.flat_repr:
            if self.one_hot_repr:
                s = self.maze_h*self.maze_w
            else:
                s = self.maze_h*self.maze_w*3*self.scale*self.scale
            self.observation_space = Box(low=0, high=1, shape=(s,), dtype=np.float32)
        else:
            if self.one_hot_repr:
                s = (1, self.maze_h, self.maze_w)
            else:
                s = (3, self.maze_h*self.scale, self.maze_w*self.scale)
            self.observation_space = Box(low=0, high=1, shape=s, dtype=np.float32)
        self.action_space = Discrete(4)

        # for rendering
        self.timestep = 0
        self.non_zero_timesteps = 0


    def reset(self):
        self.agent_color = get_color()    
        # self.agent_color = 0.95 * np.ones(3)
        
        self._one_hot = np.zeros((1, self.maze_h, self.maze_w))
        if not self.one_hot_repr:
            self.bkgd = np.ones((3, self.maze_h, self.maze_w)) * 0.5
            self.maze = np.ones((3, self.maze_h, self.maze_w)) * self.bkgd
        
        self.timestep = 0
        self.non_zero_timesteps=0

        # add the agent
        x, y = random_free(self._one_hot)
        self.cur_pos = [x,y]
        self._one_hot[-1,x,y] = 1
        if not self.one_hot_repr:
            self.maze[:,x,y] = self.agent_color

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


    def render(self):
        if self.timestep == 0:
            print('\n<<<<<<<<<<<<<<<<<<<<<<<<< New Env >>>>>>>>>>>>>>>>>>>>>>>>>')
        obs = self._one_hot
        obs = np.sum(obs, 0)
        print('\n')
        print(obs)
        print('Timestep: %d' % self.timestep)            
        # sleep(0.1)


    def step(self, action):
        if action == 0:
            dx, dy = 1, 0
        elif action == 1:
            dx, dy = 0, 1
        elif action == 2:
            dx, dy = -1, 0
        elif action == 3:
            dx, dy = 0, -1
        
        if (0 <= self.cur_pos[0] + dx <= self.maze_w-1) and (0 <= self.cur_pos[1] + dy <= self.maze_h-1):
            x = self.cur_pos[0]
            y = self.cur_pos[1]
            self._one_hot[:,x,y] = 0
            if not self.one_hot_repr:
                self.maze[:,x,y] = self.bkgd[:,x,y]

            x += dx
            y += dy

            self._one_hot[:,x,y] = 0
            self._one_hot[-1,x,y] = 1
            if not self.one_hot_repr:
                self.maze[:,x,y] = self.agent_color

            self.cur_pos = [x, y]
        
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
        
        self.timestep += 1

        return feats, 0., 0, {}
