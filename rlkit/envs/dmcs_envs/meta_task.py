"""Base class for tasks in the Control Suite.
Pretty much a direct copy of this:
https://github.com/deepmind/dm_control/blob/master/dm_control/suite/base.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_control import mujoco
from rlkit.envs.dmcs_envs import meta_env

import numpy as np


class MetaTask(meta_env.MetaTask):
  """Base class for tasks in the Control Suite.

  Actions are mapped directly to the states of MuJoCo actuators: each element of
  the action array is used to set the control input for a single actuator. The
  ordering of the actuators is the same as in the corresponding MJCF XML file.

  Attributes:
    random: A `numpy.random.RandomState` instance. This should be used to
      generate all random variables associated with the task, such as random
      starting states, observation noise* etc.

  *If sensor noise is enabled in the MuJoCo model then this will be generated
  using MuJoCo's internal RNG, which has its own independent state.
  """

  def __init__(self, random=None):
    """Initializes a new continuous control task.

    Args:
      random: Optional, either a `numpy.random.RandomState` instance, an integer
        seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    if not isinstance(random, np.random.RandomState):
      random = np.random.RandomState(random)
    self._random = random
    self._visualize_reward = False

  @property
  def random(self):
    """Task-specific `numpy.random.RandomState` instance."""
    return self._random

  def action_spec(self, physics):
    """Returns a `BoundedArraySpec` matching the `physics` actuators."""
    return mujoco.action_spec(physics)

  def before_step(self, action, physics):
    """Sets the control signal for the actuators to values in `action`."""
    # Support legacy internal code.
    try:
      physics.set_control(action.continuous_actions)
    except AttributeError:
      physics.set_control(action)

    # Reset any reward visualisation at the start of a new episode.
    if self._visualize_reward and physics.time() == 0.0:
      _set_reward_colors(physics, reward=0.0)

  def after_step(self, physics):
    """Modifies colors according to the reward."""
    if self._visualize_reward:
      reward = np.clip(self.get_reward(physics), 0.0, 1.0)
      _set_reward_colors(physics, reward)

  @property
  def visualize_reward(self):
    return self._visualize_reward

  @visualize_reward.setter
  def visualize_reward(self, value):
    if not isinstance(value, bool):
      raise ValueError("Expected a boolean, got {}.".format(type(value)))
    self._visualize_reward = value

  def get_task_identifier(self, physics):
    raise NotImplementedError()


def _set_reward_colors(physics, reward):
  """Sets the highlight, effector and target colors according to the reward."""
  assert 0.0 <= reward <= 1.0

  colors = physics.named.model.mat_rgba

  def blend(color1, color2):
    return reward * colors[color1] + (1.0 - reward) * colors[color2]

  colors["self"] = blend("self_highlight", "self_default")
  colors["effector"] = blend("effector_highlight", "effector_default")
  colors["target"] = blend("target_highlight", "target_default")