"""My Simple MetaReacher domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from rlkit.envs.dmcs_envs.meta_env import MetaEnvironment
from rlkit.envs.dmcs_envs.meta_task import MetaTask
from rlkit.envs.meta_task_params_sampler import MetaTaskParamsSampler

# Internal dependencies.

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards

import numpy as np


_DEFAULT_TIME_LIMIT = 100
_TARGET_SIZE = 0.025


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  with open('/u/kamyar/oorl_rlkit/rlkit/envs/dmcs_envs/simple_reacher.xml', mode='rb') as f:
    xml_data = f.read()
  return xml_data, common.ASSETS


def build_meta_simple_meta_reacher(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, train_env=True):
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = MetaSimpleMetaReacher(random=random)
  if train_env:
    task_params_sampler = TrainTaskParamsSampler(random=random)
  else:
    task_params_sampler = TestTaskParamsSampler(random=random)
  environment_kwargs = environment_kwargs or {}
  return MetaEnvironment(
      physics, task, task_params_sampler, concat_task_params_to_obs=True,
      time_limit=time_limit, **environment_kwargs)


def get_params_iterators(train_env=True, random=None):
    if train_env:
        return TrainTaskParamsSampler(random=random)
    return TestTaskParamsSampler(random=random)


class _BaseParamsSampler(MetaTaskParamsSampler):
    def __init__(self, idx_offset, random=None):
        super().__init__()
        if not isinstance(random, np.random.RandomState):
          random = np.random.RandomState(random)
        self._random = random

        idx = 2*np.arange(16) + idx_offset
        angles = 2 * np.pi * idx / 32
        x = np.sin(angles)
        y = np.cos(angles)
        self.p = 0.2 * np.stack([x,y], axis=1)
        self.ptr = 0
        self.itr_ptr = 0
    

    def sample(self):
        x, y = self.p[self.ptr, 0], self.p[self.ptr, 1]
        self.ptr = (self.ptr + 1) % 16
        return {'x': x, 'y': y}, np.array([x,y])


    def __iter__(self):
        self.itr_ptr = 0
        return self
    

    def __next__(self):
        if self.itr_ptr == 16: raise StopIteration
        x, y = self.p[self.itr_ptr, 0], self.p[self.itr_ptr, 1]
        self.itr_ptr += 1
        return {'x': x, 'y': y}, np.array([x,y])


class TrainTaskParamsSampler(_BaseParamsSampler):
    def __init__(self, random=None):
        super(TrainTaskParamsSampler, self).__init__(0, random=random)


class TestTaskParamsSampler(_BaseParamsSampler):
    def __init__(self, random=None):
        super(TestTaskParamsSampler, self).__init__(1, random=random)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Reacher domain."""

  def finger_to_target(self):
    """Returns the vector from target to finger in global coordinate."""
    return (self.named.data.geom_xpos['target'] -
            self.named.data.geom_xpos['finger'])

  def finger_to_target_dist(self):
    """Returns the signed distance between the finger and target surface."""
    return np.linalg.norm(self.finger_to_target())


class MetaSimpleMetaReacher(MetaTask):
  """A reacher `Task` to reach the target.
  It has a shaped version of the reward function provided by Deepmind Control Suite

  """

  def __init__(self, random=None):
    """Initialize an instance of `Reacher`.

    Args:
      target_size: A `float`, tolerance to determine whether finger reached the
          target.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    super(MetaSimpleMetaReacher, self).__init__(random=random)


  def initialize_episode(self, physics, task_params):
    """Sets the state of the environment at the start of each episode."""
    physics.named.model.geom_size['target', 0] = _TARGET_SIZE
    randomizers.randomize_limited_and_rotational_joints(physics, self.random)

    # randomize target position
    x, y = task_params['x'], task_params['y']
    physics.named.model.geom_pos['target', 'x'] = x
    physics.named.model.geom_pos['target', 'y'] = y

    self.prev_dist = physics.finger_to_target_dist()
    self.shaping_rew = 0.0


  def after_step(self, physics):
    super().after_step(physics)
    new_dist = physics.finger_to_target_dist()
    self.shaping_rew = self.prev_dist - new_dist
    # print(self.prev_dist, new_dist, self.shaping_rew)
    self.prev_dist = new_dist


  def get_observation(self, physics):
    """Returns an observation of the state and the target position."""
    obs = collections.OrderedDict()
    # obs['position'] = physics.position()
    phys_pos = physics.position()
    obs['position'] = np.concatenate([np.cos(phys_pos), np.sin(phys_pos)], -1)
    # consider adding self.named.data.geom_xpos['finger']
    # obs['to_target'] = physics.finger_to_target()
    obs['finger_pos'] = physics.named.data.geom_xpos['finger']
    obs['velocity'] = physics.velocity()
    return obs


  def get_reward(self, physics):
    # from dmcs
    radii = physics.named.model.geom_size[['target', 'finger'], 0].sum()
    sparse_reward = rewards.tolerance(physics.finger_to_target_dist(), (0, radii))

    # print(sparse_reward)
    # print(sparse_reward + self.shaping_rew)

    # c_rew = np.square(physics.control()).sum()
    return sparse_reward + self.shaping_rew


  def get_task_identifier(self, physics):
    x = physics.named.model.geom_pos['target', 'x']
    y = physics.named.model.geom_pos['target', 'y']
    return (x,y)
