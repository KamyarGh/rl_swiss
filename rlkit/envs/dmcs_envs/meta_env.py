"""
An environment.Base subclass for control-specific environments.
Almost Identical to the one found in dmcontrol, just meta
I changed this a tiiiiiiny bit, so I can use it for meta-learning.
- Kamyar
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import contextlib

import numpy as np
import six
from six.moves import range

from dm_control.rl import environment
from dm_control.rl import specs
from dm_control.suite import base

FLAT_OBSERVATION_KEY = 'observations'


class MetaEnvironment(environment.Base):
  """Class for physics-based reinforcement learning environments."""

  def __init__(self,
               physics,
               task,
               task_params_sampler,
               concat_task_params_to_obs=True,
               time_limit=float('inf'),
               control_timestep=None,
               n_sub_steps=None,
               flat_observation=False):
    """Initializes a new `Environment`.

    Args:
      physics: Instance of `Physics`.
      task: Instance of `Task`.
      task_params_sampler: function with a .sample() function
      time_limit: Optional `int`, maximum time for each episode in seconds. By
        default this is set to infinite.
      control_timestep: Optional control time-step, in seconds.
      n_sub_steps: Optional number of physical time-steps in one control
        time-step, aka "action repeats". Can only be supplied if
        `control_timestep` is not specified.
      flat_observation: If True, observations will be flattened and concatenated
        into a single numpy array.

    Raises:
      ValueError: If both `n_sub_steps` and `control_timestep` are supplied.
    """
    self._task = task
    self.task_params_sampler = task_params_sampler
    self._physics = physics
    self._flat_observation = flat_observation
    self.concat_task_params_to_obs = concat_task_params_to_obs
    # doing this so that when observation_spec is called we can fill it in
    self.task_params, self.obs_task_params = self.task_params_sampler.sample()

    if n_sub_steps is not None and control_timestep is not None:
      raise ValueError('Both n_sub_steps and control_timestep were supplied.')
    elif n_sub_steps is not None:
      self._n_sub_steps = n_sub_steps
    elif control_timestep is not None:
      self._n_sub_steps = compute_n_steps(control_timestep,
                                          self._physics.timestep())
    else:
      self._n_sub_steps = 1

    if time_limit == float('inf'):
      self._step_limit = float('inf')
    else:
      self._step_limit = time_limit / (
          self._physics.timestep() * self._n_sub_steps)
    self._step_count = 0
    self._reset_next_step = True


  def reset(self, task_params=None, obs_task_params=None):
    """Starts a new episode and returns the first `TimeStep`.
    task_params: a dict of task params to pass to the task
    obs_task_params: a flat numpy array of task parameters that you'd give to your policy
    """
    self._reset_next_step = False
    self._step_count = 0
    if task_params is None:
        self.task_params, self.obs_task_params = self.task_params_sampler.sample()
    else:
        self.task_params, self.obs_task_params = task_params, obs_task_params
    with self._physics.reset_context():
      self._task.initialize_episode(self._physics, self.task_params)

    observation = self._task.get_observation(self._physics)
    if self.concat_task_params_to_obs:
        observation['obs_task_params'] = self.obs_task_params
    if self._flat_observation:
      observation = flatten_observation(observation)
    
    return environment.TimeStep(
        step_type=environment.StepType.FIRST,
        reward=None,
        discount=None,
        observation=observation
    )


  def step(self, action):
    """Updates the environment using the action and returns a `TimeStep`."""

    if self._reset_next_step:
      return self.reset()

    self._task.before_step(action, self._physics)
    for _ in range(self._n_sub_steps):
      self._physics.step()
    self._task.after_step(self._physics)

    reward = self._task.get_reward(self._physics)
    observation = self._task.get_observation(self._physics)
    if self.concat_task_params_to_obs:
        observation['obs_task_params'] = self.obs_task_params
    if self._flat_observation:
      observation = flatten_observation(observation)

    self._step_count += 1
    if self._step_count >= self._step_limit:
      discount = 1.0
    else:
      discount = self._task.get_termination(self._physics)

    episode_over = discount is not None

    if episode_over:
      self._reset_next_step = True
      return environment.TimeStep(
          environment.StepType.LAST, reward, discount, observation)
    else:
      return environment.TimeStep(
          environment.StepType.MID, reward, 1.0, observation)


  def action_spec(self):
    """Returns the action specification for this environment."""
    return self._task.action_spec(self._physics)


  def step_spec(self):
    """May return a specification for the values returned by `step`."""
    return self._task.step_spec(self._physics)


  def observation_spec(self):
    """Returns the observation specification for this environment.

    Infers the spec from the observation, unless the Task implements the
    `observation_spec` method.

    Returns:
      An dict mapping observation name to `ArraySpec` containing observation
      shape and dtype.
    """
    try:
      return self._task.observation_spec(self._physics)
    except NotImplementedError:
      observation = self._task.get_observation(self._physics)
      if self.concat_task_params_to_obs:
        observation['obs_task_params'] = self.obs_task_params
      if self._flat_observation:
        observation = flatten_observation(observation)
      print(observation)
      return _spec_from_observation(observation)


  @property
  def physics(self):
    return self._physics


  @property
  def task(self):
    return self._task
  

  @property
  def task_identifier(self):
    return self._task.get_task_identifier(self._physics)


  def control_timestep(self):
    """Returns the interval between agent actions in seconds."""
    return self.physics.timestep() * self._n_sub_steps


def compute_n_steps(control_timestep, physics_timestep, tolerance=1e-8):
  """Returns the number of physics timesteps in a single control timestep.

  Args:
    control_timestep: Control time-step, should be an integer multiple of the
      physics timestep.
    physics_timestep: The time-step of the physics simulation.
    tolerance: Optional tolerance value for checking if `physics_timestep`
      divides `control_timestep`.

  Returns:
    The number of physics timesteps in a single control timestep.

  Raises:
    ValueError: If `control_timestep` is smaller than `physics_timestep` or if
      `control_timestep` is not an integer multiple of `physics_timestep`.
  """
  if control_timestep < physics_timestep:
    raise ValueError(
        'Control timestep ({}) cannot be smaller than physics timestep ({}).'.
        format(control_timestep, physics_timestep))
  if abs((control_timestep / physics_timestep - round(
      control_timestep / physics_timestep))) > tolerance:
    raise ValueError(
        'Control timestep ({}) must be an integer multiple of physics timestep '
        '({})'.format(control_timestep, physics_timestep))
  return int(round(control_timestep / physics_timestep))


def _spec_from_observation(observation):
  result = collections.OrderedDict()
  for key, value in six.iteritems(observation):
    result[key] = specs.ArraySpec(value.shape, value.dtype, name=key)
  return result


@six.add_metaclass(abc.ABCMeta)
class MetaTask(object):
  """Defines a task in a `control.Environment`."""

  @abc.abstractmethod
  def initialize_episode(self, physics, task_params):
    """Sets the state of the environment at the start of each episode.

    Called by `control.Environment` at the start of each episode *within*
    `physics.reset_context()` (see the documentation for `base.Physics`).

    Args:
      physics: Instance of `Physics`.
    """


  @abc.abstractmethod
  def before_step(self, action, physics):
    """Updates the task from the provided action.

    Called by `control.Environment` before stepping the physics engine.

    Args:
      action: numpy array or array-like action values, or a nested structure of
        such arrays. Should conform to the specification returned by
        `self.action_spec(physics)`.
      physics: Instance of `Physics`.
    """


  def after_step(self, physics):
    """Optional method to update the task after the physics engine has stepped.

    Called by `control.Environment` after stepping the physics engine and before
    `control.Environment` calls `get_observation, `get_reward` and
    `get_termination`.

    The default implementation is a no-op.

    Args:
      physics: Instance of `Physics`.
    """


  @abc.abstractmethod
  def action_spec(self, physics):
    """Returns a specification describing the valid actions for this task.

    Args:
      physics: Instance of `Physics`.

    Returns:
      A `BoundedArraySpec`, or a nested structure containing `BoundedArraySpec`s
      that describe the shapes, dtypes and elementwise lower and upper bounds
      for the action array(s) passed to `self.step`.
    """


  def step_spec(self, physics):
    """Returns a specification describing the time_step for this task.

    Args:
      physics: Instance of `Physics`.

    Returns:
      A `BoundedArraySpec`, or a nested structure containing `BoundedArraySpec`s
      that describe the shapes, dtypes and elementwise lower and upper bounds
      for the array(s) returned by `self.step`.
    """
    raise NotImplementedError()


  @abc.abstractmethod
  def get_observation(self, physics):
    """Returns an observation from the environment.

    Args:
      physics: Instance of `Physics`.
    """


  @abc.abstractmethod
  def get_reward(self, physics):
    """Returns a reward from the environment.

    Args:
      physics: Instance of `Physics`.
    """
  

  @abc.abstractmethod
  def get_task_identifier(self, physics):
    """Returns a hashable task identifier

    Args:
      physics: Instance of `Physics`.
    """


  def get_termination(self, physics):
    """If the episode should end, returns a final discount, otherwise None."""


  def observation_spec(self, physics):
    """Optional method that returns the observation spec.

    If not implemented, the Environment infers the spec from the observation.

    Args:
      physics: Instance of `Physics`.

    Returns:
      A dict mapping observation name to `ArraySpec` containing observation
      shape and dtype.
    """
    raise NotImplementedError()


def flatten_observation(observation, output_key=FLAT_OBSERVATION_KEY):
  """Flattens multiple observation arrays into a single numpy array.

  Args:
    observation: A mutable mapping from observation names to numpy arrays.
    output_key: The key for the flattened observation array in the output.

  Returns:
    A mutable mapping of the same type as `observation`. This will contain a
    single key-value pair consisting of `output_key` and the flattened
    and concatenated observation array.

  Raises:
    ValueError: If `observation` is not a `collections.MutableMapping`.
  """
  if not isinstance(observation, collections.MutableMapping):
    raise ValueError('Can only flatten dict-like observations.')

  if isinstance(observation, collections.OrderedDict):
    keys = six.iterkeys(observation)
  else:
    # Keep a consistent ordering for other mappings.
    keys = sorted(six.iterkeys(observation))

  observation_arrays = [observation[key].ravel() for key in keys]
  return type(observation)([(output_key, np.concatenate(observation_arrays))])
