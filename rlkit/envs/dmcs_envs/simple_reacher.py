"""My Reacher domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

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


def build_simple_reacher(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns reacher with sparse reward with 5e-2 tol and randomized target."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Reacher(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Reacher domain."""

  def finger_to_target(self):
    """Returns the vector from target to finger in global coordinate."""
    return (self.named.data.geom_xpos['target'] -
            self.named.data.geom_xpos['finger'])

  def finger_to_target_dist(self):
    """Returns the signed distance between the finger and target surface."""
    return np.linalg.norm(self.finger_to_target())


class Reacher(base.Task):
  """A reacher `Task` to reach the target."""

  def __init__(self, random=None):
    """Initialize an instance of `Reacher`.

    Args:
      target_size: A `float`, tolerance to determine whether finger reached the
          target.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    super(Reacher, self).__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode."""
    physics.named.model.geom_size['target', 0] = _TARGET_SIZE
    randomizers.randomize_limited_and_rotational_joints(physics, self.random)

    # randomize target position
    angle = self.random.uniform(0, 2 * np.pi)
    radius = self.random.uniform(.05, .20)
    physics.named.model.geom_pos['target', 'x'] = radius * np.sin(angle)
    physics.named.model.geom_pos['target', 'y'] = radius * np.cos(angle)

  def get_observation(self, physics):
    """Returns an observation of the state and the target position."""
    obs = collections.OrderedDict()
    obs['position'] = physics.position()
    obs['to_target'] = physics.finger_to_target()
    obs['velocity'] = physics.velocity()
    return obs

  def get_reward(self, physics):
    d_rew = physics.finger_to_target_dist()
    c_rew = np.square(physics.control()).sum()
    return -1.0 * d_rew
