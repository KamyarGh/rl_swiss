from copy import deepcopy
from time import sleep

import numpy as np
from numpy.random import choice, randint

import gym
from gym.spaces import Discrete, Box
from time import sleep

debug = False

# colors = np.array(
#     # [[  230.,  25.,  75.],
#     # [ 245.,  130.,   48.],
#     [[  60.,  180.,   70.],
#     [ 72.,   240.,  240.]]
#     # [  145.,   30.,  180.],
#     # [ 240.,   50.,  230.]]
# )
# colors = colors / 255.0

# REMEMBER TO ADD TEXTURE TO THE BACKGROUND LATER
def random_free(grid, pad_h, pad_w):
    h, w = grid.shape[1], grid.shape[2]
    x, y = randint(pad_h,h-pad_h), randint(pad_w, w-pad_w)
    # print(grid[-2,-5:,-5:])
    while any(grid[:, x, y]):
        x, y = randint(pad_h,h-pad_h), randint(pad_w, w-pad_w)
    return x, y


def get_color():
    # idx = choice(range(colors.shape[0]))
    # c = colors[idx]

    c = np.random.beta(0.5, 0.5, size=3)

    c = np.clip(c, 0.1, 0.9)
    return c


class MemoryGrid(gym.Env):
    def __init__(self, env_specs):
        self.env_specs = env_specs
        self.maze_h = self.env_specs['maze_h']
        self.maze_w = self.env_specs['maze_w']
        self.obs_h = self.env_specs['obs_h']
        self.obs_w = self.env_specs['obs_w']
        assert self.obs_h % 2 == self.obs_w % 2 == 1
        self.flat_repr = self.env_specs['flat_repr']
        self.one_hot_repr = self.env_specs['one_hot_repr']
        self.scale = self.env_specs['scale']
        self.num_objs = self.env_specs['num_objs']

        self.pad_h = self.obs_h // 2
        self.pad_w = self.obs_w // 2
        self.maze_h += self.obs_h // 2
        self.maze_w += self.obs_w // 2

        if self.flat_repr:
            if self.one_hot_repr:
                s = self.obs_h*self.obs_w*(num_objs+1)
            else:
                s = self.obs_h*self.obs_w*3*self.scale*self.scale
            self.observation_space = Box(low=0, high=1, shape=(s,), dtype=np.float32)
        else:
            if self.one_hot_repr:
                s = (num_objs+1, self.obs_h, self.obs_w)
            else:
                s = (3, self.obs_h*self.scale, self.obs_w*self.scale)
            self.observation_space = Box(low=0, high=1, shape=s, dtype=np.float32)
        self.action_space = Discrete(4)

        # for rendering
        self.timestep = 0
        self.non_zero_timesteps = 0

        self.wall_idx = -2


    def reset(self):
        self.wall_color = get_color()
        self._one_hot = np.zeros((self.num_objs+1, self.maze_h, self.maze_w))
        if not self.one_hot_repr:
            self.bkgd = np.ones((3, self.maze_h, self.maze_w)) * 0.5
            self.maze = np.ones((3, self.maze_h, self.maze_w)) * self.bkgd
        
        # add the walls
        self._one_hot[self.wall_idx, self.pad_h-1:-self.pad_h, self.pad_w-1] = 1.0
        self._one_hot[self.wall_idx, self.pad_h-1:-self.pad_h, -self.pad_w] = 1.0
        self._one_hot[self.wall_idx, self.pad_h-1, self.pad_w-1:-self.pad_w] = 1.0
        self._one_hot[self.wall_idx, -self.pad_h, self.pad_w-1:-self.pad_w] = 1.0

        self.maze[:, self.pad_h-1:-self.pad_h, self.pad_w-1] = self.wall_color[:,None]
        self.maze[:, self.pad_h-1:-self.pad_h, -self.pad_w] = self.wall_color[:,None]
        self.maze[:, self.pad_h-1, self.pad_w-1:-self.pad_w] = self.wall_color[:,None]
        self.maze[:, -self.pad_h, self.pad_w-1:-self.pad_w] = self.wall_color[:,None]

        self.obj_colors = []
        for i in range(self.num_objs):
            self.obj_colors.append(get_color())
            x, y = random_free(self._one_hot, self.pad_h, self.pad_w)
            self._one_hot[i,x,y] = 1.
            self.maze[:,x,y] = self.obj_colors[-1]
        
        self.timestep = 0
        self.cur_pos = [self.pad_h+2, self.pad_w+2]

        return self.get_obs()
    

    def get_obs(self):
        x, y = self.cur_pos[0], self.cur_pos[1]

        # create the observation
        if self.flat_repr:
            if self.one_hot_repr:
                return self._one_hot[:, x-self.pad_h:x+self.pad_h+1, y-self.pad_w:y+self.pad_w+1].flatten()
            else:
                return self.maze[:, x-self.pad_h:x+self.pad_h+1, y-self.pad_w:y+self.pad_w+1].flatten()
        else:
            if self.one_hot_repr:
                return self._one_hot[:, x-self.pad_h:x+self.pad_h+1, y-self.pad_w:y+self.pad_w+1]
            else:
                return np.kron(
                    self.maze[:, x-self.pad_h:x+self.pad_h+1, y-self.pad_w:y+self.pad_w+1],
                    np.ones((1,self.scale,self.scale))
                )


    def render(self):
        raise NotImplementedError()
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
        
        x, y = self.cur_pos[0], self.cur_pos[1]
        self.cur_pos = [x + dx, y + dy]

        return self.get_obs(), 0., 0, {}
