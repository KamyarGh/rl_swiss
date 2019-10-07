import numpy as np
from collections import OrderedDict
from gym import utils
from rlkit.envs.meta_mujoco_env import MetaMujocoEnv

from rlkit.envs.meta_task_params_sampler import MetaTaskParamsSampler

class _BaseParamsSampler(MetaTaskParamsSampler):
    def __init__(self, random=8032, num_tasks=40):
        super().__init__()
        if not isinstance(random, np.random.RandomState):
          random = np.random.RandomState(random)
        self._random = random
        self._ptr = 0
        self.num_tasks = num_tasks

        self.log_scale_limit = 3.0
        # body mass
        body_mass_shape = (5,)
        self._all_body_mass_mults = [
            np.array(1.5) ** self._random.uniform(-self.log_scale_limit, self.log_scale_limit,  size=body_mass_shape)
            for _ in range(num_tasks)
        ]
        # body inertia
        body_inertia_shape = (5, 3)
        self._all_body_inertia_mults = [
            np.array(1.5) ** self._random.uniform(-self.log_scale_limit, self.log_scale_limit,  size=body_inertia_shape)
            for _ in range(num_tasks)
        ]
        # damping
        dof_damping_shape = (6,)
        self._all_dof_damping_mults = [
            np.array(1.3) ** self._random.uniform(-self.log_scale_limit, self.log_scale_limit, size=dof_damping_shape)
            for _ in range(num_tasks)
        ]
        # geom friction
        geom_friction_shape = (5,3)
        self._all_geom_friction_mults = [
            np.array(1.5) ** self._random.uniform(-self.log_scale_limit, self.log_scale_limit, size=geom_friction_shape)
            for _ in range(num_tasks)
        ]
    
    def get_task(self, idx):
        return {
            'body_mass_multiplier': self._all_body_mass_mults[idx],
            'body_inertia_multiplier': self._all_body_inertia_mults[idx],
            'dof_damping_multiplier': self._all_dof_damping_mults[idx],
            'geom_friction_multiplier': self._all_geom_friction_mults[idx]
        }
    
    def get_obs_task_params(self, task):
        concat_list = [
            task['body_mass_multiplier'].flatten(),
            task['body_inertia_multiplier'].flatten(),
            task['dof_damping_multiplier'].flatten(),
            task['geom_friction_multiplier'].flatten(),
        ]
        return np.concatenate(concat_list).copy()

    def sample(self):
        idx = self._random.choice(self.num_tasks)
        task = self.get_task(idx)
        obs_task_params = self.get_obs_task_params(task)
        return task, obs_task_params

    def sample_unique(self, num):
        idx_samples = self._random.choice(self.num_tasks, size=num, replace=False)
        all_samples = []
        for idx in idx_samples:
            task = self.get_task(idx)
            obs_task_params = self.get_obs_task_params(task)
            all_samples.append((task, obs_task_params))
        return all_samples

    def __iter__(self):
        # dangerous
        self._ptr = 0
        return self

    def __next__(self):
        if self._ptr == self.num_tasks:
            self._ptr = 0
            raise StopIteration
        task = self.get_task(self._ptr)
        obs_task_params = self.get_obs_task_params(task)
        self._ptr += 1
        return task, obs_task_params


class _MetaExpertTrainParamsSampler(_BaseParamsSampler):
    def __init__(self, random=9827, num_tasks=50):
        super().__init__(random=random, num_tasks=num_tasks)

class _MetaExpertTestParamsSampler(_BaseParamsSampler):
    def __init__(self, random=5382, num_tasks=25):
        super().__init__(random=random, num_tasks=num_tasks)


class HopperRandomDynamicsEnv(MetaMujocoEnv, utils.EzPickle):
    def __init__(self):
        self.multipliers = {
            'body_mass_multiplier': np.ones(5),
            'body_inertia_multiplier': np.ones((5,3)),
            'dof_damping_multiplier': np.ones(6),
            'geom_friction_multiplier': np.ones((5,3))
        }
        MetaMujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)

        self.original_params = {
            'body_mass': self.model.body_mass.copy(),
            'body_inertia': self.model.body_inertia.copy(),
            'dof_damping': self.model.dof_damping.copy(),
            'geom_friction': self.model.geom_friction.copy()
        }

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

    def get_flat_dynamics_params(self):
        concat_list = [
            self.multipliers['body_mass_multiplier'].flatten(),
            self.multipliers['body_inertia_multiplier'].flatten(),
            self.multipliers['dof_damping_multiplier'].flatten(),
            self.multipliers['geom_friction_multiplier'].flatten(),
        ]
        return np.concatenate(concat_list).copy()

    def _get_obs(self):
        obs = np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])
        return {
            'obs': obs.copy(),
            'obs_task_params': self.get_flat_dynamics_params()
        }

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
    
    def reset(self, task_params=None, obs_task_params=None):
        if task_params is None:
            raise NotImplementedError()
        else:
            self.multipliers = {k: task_params[k].copy() for k in task_params}
            self.model.body_mass[...] = task_params['body_mass_multiplier'] * self.original_params['body_mass']
            self.model.body_inertia[...] = task_params['body_inertia_multiplier'] * self.original_params['body_inertia']
            self.model.dof_damping[...] = np.multiply(task_params['dof_damping_multiplier'], self.original_params['dof_damping'])
            self.model.geom_friction[...] = np.multiply(task_params['geom_friction_multiplier'], self.original_params['geom_friction'])

        obs = super().reset()
        return obs

    @property
    def task_identifier(self):
        return tuple(self.get_flat_dynamics_params())

    def task_id_to_obs_task_params(self, task_id):
        return np.array(task_id)

    def log_statistics(self, paths):
        return {}
