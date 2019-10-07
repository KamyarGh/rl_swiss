'''
Modified a bit from
https://github.com/martinseilair/dm_control2gym/blob/master/dm_control2gym/wrapper.py
'''

from gym import core, spaces
from gym.spaces import Dict as DictSpace
from dm_control import suite
from dm_control.rl import specs
from gym.utils import seeding
import gym

from rlkit.envs.dmcs_envs.meta_env import MetaEnvironment

# from dm_control2gym.viewer import DmControlViewer
import numpy as np
import sys


class DmcDiscrete(gym.spaces.Discrete):
    def __init__(self, _minimum, _maximum):
        super().__init__(_maximum - _minimum)
        self.offset = _minimum


def compute_min_max(spec, clip_inf=False):
    _min = spec.minimum
    _max = spec.maximum
    if clip_inf:
        _min = np.clip(spec.minimum, -sys.float_info.max, sys.float_info.max)
        _max = np.clip(spec.maximum, -sys.float_info.max, sys.float_info.max)

    return _min, _max


def convertSpec2Space(spec, clip_inf=False):
    if spec.dtype == np.int:
        assert False, 'I am not sure this is how it should be, but not important for now'
        # Discrete
        return DmcDiscrete(spec.minimum, spec.maximum)
    else:
        # Box
        if type(spec) is specs.ArraySpec:
            return spaces.Box(-np.inf, np.inf, shape=spec.shape)
        elif type(spec) is specs.BoundedArraySpec:

            _min, _max = compute_min_max(spec, clip_inf=clip_inf)

            if np.isscalar(_min) and np.isscalar(_max):
                # same min and max for every element
                return spaces.Box(_min, _max, shape=spec.shape)
            else:
                # different min and max for every element
                return spaces.Box(_min + np.zeros(spec.shape),
                                  _max + np.zeros(spec.shape))
        else:
            raise ValueError('Unknown spec!')


def convertOrderedDict2Space(odict, has_pixels=False, has_task_params=False):
    '''
        For now just using -inf and inf for the bounds of the Box
    '''
    if len(odict.keys()) == 1:
        # no concatenation
        return convertSpec2Space(list(odict.values())[0])
    else:
        # len keys is more than 1
        _min, _max = -np.Inf, np.Inf
        numdim = 0
        for key in odict:
            if key not in ['pixels', 'obs_task_params']:
                numdim += np.prod(odict[key].shape)
                # cur_min, cur_max = compute_min_max(spec)
                # _min = min(_min, cur_min)
                # _max = max(_max, cur_max)

        dict_thus_far = {
            'obs': spaces.Box(-np.inf, np.inf, shape=(numdim,))
        }
        if has_pixels:
            dict_thus_far.update(
                {'pixels': spaces.Box(low=0, high=1, shape=odict['pixels'].shape)}
            )
        if has_task_params:
            dict_thus_far.update(
                {'obs_task_params': spaces.Box(low=0, high=1, shape=odict['obs_task_params'].shape)}
            )         
        if len(dict_thus_far.keys()) == 1:
            # we just have obs so we will make it a Box
            # concatentation
            # numdim = sum([np.int(np.prod(odict[key].shape)) for key in odict])
            gym_space = spaces.Box(-np.inf, np.inf, shape=(numdim,))
        else:
            gym_space = DictSpace(dict_thus_far)
        
        return gym_space


def convertObservation(spec_obs, has_pixels=False, has_task_params=False):
    if len(spec_obs.keys()) == 1:
        # no concatenation
        return list(spec_obs.values())[0]
    else:
        # len keys is more than 1
        # concatentation

        # numdim = sum([np.int(np.prod(spec_obs[key].shape)) for key in spec_obs if key not in ['pixels', 'obs_task_params']])
        # space_obs = np.zeros((numdim,))
        # i = 0
        # for key in spec_obs:
        #     if key not in ['pixels', 'obs_task_params']:
        #         print(type(spec_obs[key]))
        #         1/0
        #         space_obs[i:i+np.prod(spec_obs[key].shape)] = spec_obs[key].ravel()
        #         i += np.prod(spec_obs[key].shape)
        # ---------------
        space_obs = np.concatenate(
            [spec_obs[key] for key in spec_obs if key not in ['pixels', 'obs_task_params']],
            -1
        )
        
        obs_dict = {'obs': space_obs}
        if has_pixels:
            obs_dict.update({'pixels': spec_obs['pixels']})
        if has_task_params:
            obs_dict.update({'obs_task_params': spec_obs['obs_task_params']})
        
        if len(obs_dict.keys()) > 1:
            return obs_dict
        return space_obs


class DmControlWrapper(core.Env):
    def __init__(self, dmcenv, render_mode_list=None):
        self.dmcenv = dmcenv
        self.is_meta_env = isinstance(dmcenv, MetaEnvironment)

        # convert spec to space
        self.action_space = convertSpec2Space(self.dmcenv.action_spec(), clip_inf=True)
        dmcenv_obs_spec = self.dmcenv.observation_spec()
        self.conversion_kwargs = {
            'has_pixels': 'pixels' in dmcenv_obs_spec.keys(),
            'has_task_params': 'obs_task_params' in dmcenv_obs_spec.keys()
        }
        self.observation_space = convertOrderedDict2Space(dmcenv_obs_spec, **self.conversion_kwargs)

        if render_mode_list is not None:
            self.metadata['render.modes'] = list(render_mode_list.keys())
            self.viewer = {key:None for key in render_mode_list.keys()}
        else:
            self.metadata['render.modes'] = []

        self.render_mode_list = render_mode_list


        # set seed
        # self._seed()

    def getObservation(self):
        return convertObservation(self.timestep.observation, **self.conversion_kwargs)

    # def _seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def reset(self, **kwargs):
        self.timestep = self.dmcenv.reset(**kwargs)
        return self.getObservation()


    def step(self, a):

        if type(self.action_space) == DmcDiscrete:
            a += self.action_space.offset
        self.timestep = self.dmcenv.step(a)

        # print(self.getObservation())
        if self.is_meta_env:
            info_dict = {'task_identifier': self.dmcenv.task_identifier}
        else:
            info_dict = {}

        return self.getObservation(), self.timestep.reward, self.timestep.last(), info_dict


    @property
    def task_identifier(self):
        return self.dmcenv.task_identifier


    def _render(self, mode='human', close=False):
        self.pixels = self.dmcenv.physics.render(**self.render_mode_list[mode]['render_kwargs'])
        if close:
            if self.viewer[mode] is not None:
                self._get_viewer(mode).close()
                self.viewer[mode] = None
            return
        elif self.render_mode_list[mode]['show']:
            self._get_viewer(mode).update(self.pixels)



        if self.render_mode_list[mode]['return_pixel']:

            return self.pixels

    def _get_viewer(self, mode):
        raise NotImplementedError()
        if self.viewer[mode] is None:
            self.viewer[mode] = DmControlViewer(self.pixels.shape[1], self.pixels.shape[0], self.render_mode_list[mode]['render_kwargs']['depth'])
        return self.viewer[mode]
