"""My Simple MetaReacher domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from rlkit.envs.dmcs_envs.meta_env import MetaEnvironment
from rlkit.envs.dmcs_envs.meta_task import MetaTask

# Internal dependencies.

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards

import numpy as np


_DEFAULT_TIME_LIMIT = 20
_TARGET_SIZE = 0.025


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  with open('/u/kamyar/oorl_rlkit/rlkit/envs/dmcs_envs/simple_reacher.xml', mode='rb') as f:
    xml_data = f.read()
  return xml_data, common.ASSETS


def build_simple_meta_reacher(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, train_env=True):
  """Returns reacher with sparse reward with 5e-2 tol and randomized target."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = SimpleMetaReacher(random=random)
  if train_env:
    task_params_sampler = SimpleMetaReacherTrainTaskParamsSampler(random=random)
  else:
    task_params_sampler = SimpleMetaReacherTestTaskParamsSampler(random=random)
  environment_kwargs = environment_kwargs or {}
  return MetaEnvironment(
      physics, task, task_params_sampler, concat_task_params_to_obs=True,
      time_limit=time_limit, **environment_kwargs)


class SimpleMetaReacherTrainTaskParamsSampler():
    def __init__(self, random=None):
        if not isinstance(random, np.random.RandomState):
          random = np.random.RandomState(random)
        self._random = random
    
    def sample(self):
        angle = self._random.uniform(0, 2 * np.pi)
        x, y = 0.20 * np.sin(angle), 0.20 * np.cos(angle)
        return {'x': x, 'y': y}, np.array([x,y])


class SimpleMetaReacherTestTaskParamsSampler():
    def __init__(self):
        self.angles = np.linspace(0, 2*np.pi, num=11, endpoint=False)
        self.ptr = 0
    
    def sample(self):
        # doesn't matter that it's inefficient
        # remember Amdahl's Law
        angle = self.angles[self.ptr]
        x, y = 0.20 * np.sin(angle), 0.20 * np.cos(angle)
        self.ptr = (self.ptr + 1) % len(self.angles)
        return {'x': x, 'y': y}, np.array([x,y])


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Reacher domain."""

  def finger_to_target(self):
    """Returns the vector from target to finger in global coordinate."""
    return (self.named.data.geom_xpos['target'] -
            self.named.data.geom_xpos['finger'])

  def finger_to_target_dist(self):
    """Returns the signed distance between the finger and target surface."""
    return np.linalg.norm(self.finger_to_target())


class SimpleMetaReacher(MetaTask):
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
    super(SimpleMetaReacher, self).__init__(random=random)


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
