'''
Adapted from a combination of OpenAI gym halfcheetah and Chelsea Finn's rand vel HalfCheetah,
and made to fit into the interface I've set up for doing meta-rl/irl
https://github.com/cbfinn/maml_rl/blob/master/rllab/envs/mujoco/half_cheetah_env_rand.py
'''
import numpy as np
from collections import OrderedDict
from gym import utils
from rlkit.envs.meta_mujoco_env import MetaMujocoEnv

from rlkit.envs.meta_task_params_sampler import MetaTaskParamsSampler

class _TrainParamsSampler(MetaTaskParamsSampler):
    def __init__(self, random=8032, num_samples=91):
        super().__init__()
        if not isinstance(random, np.random.RandomState):
          random = np.random.RandomState(random)
        self._random = random
        self.vels = np.linspace(0.0, 3.0, num=num_samples, endpoint=True)
        self._ptr = 0

    def sample(self):
        v = self._random.choice(self.vels)
        v = np.array([v])
        return {'target_velocity': v}, v

    def sample_unique(self, num):
        vel_samples = self._random.choice(self.vels, size=num, replace=False)
        return list(
            map(
                lambda vel: ({'target_velocity': np.array([vel])}, np.array([vel])),
                vel_samples
            )
        )

    def __iter__(self):
        # dangerous
        self._ptr = 0
        return self

    def __next__(self):
        if self._ptr == len(self.vels):
            self._ptr = 0
            raise StopIteration
        vel = np.array([self.vels[self._ptr]])
        self._ptr += 1
        return {'target_velocity': vel}, vel


class _TestParamsSampler(_TrainParamsSampler):
    def __init__(self, random=2340, num_samples=25):
        super().__init__(random, num_samples=num_samples)
        self.vels = self._random.uniform(low=0.0, high=3.0, size=25)


class _MetaTrainParamsSampler(_TrainParamsSampler):
    def __init__(self, random=9827):
        super().__init__(random, num_samples=1)
        self.vels = np.linspace(0.0, 3.0, num=31, endpoint=True)


class _MetaTestParamsSampler(_TrainParamsSampler):
    def __init__(self, random=5382):
        super().__init__(random, num_samples=1)
        self.vels = np.linspace(0.05, 2.95, num=30, endpoint=True)


class _DebugMetaTrainSampler(_TrainParamsSampler):
    def __init__(self, random=9827):
        super().__init__(random, num_samples=1)
        # self.vels = np.linspace(0.0, 3.0, num=31, endpoint=True)
        # self.vels = np.array([1.0])
        # self.vels = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        # self.vels = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
        self.vels = np.array([1.0, 1.25, 1.5, 1.75, 2.0])

class _DebugMetaTestSampler(_TrainParamsSampler):
    def __init__(self, random=9827):
        super().__init__(random, num_samples=1)
        # self.vels = np.linspace(0.0, 3.0, num=31, endpoint=True)
        # self.vels = np.array([1.05])
        # self.vels = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        # self.vels = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
        self.vels = np.array([1.0, 1.25, 1.5, 1.75, 2.0])


class HalfCheetahRandVelEnv(MetaMujocoEnv, utils.EzPickle):
    def __init__(self):
        self.target_velocity = np.array([0.0])
        MetaMujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)
    
    def get_body_comvel(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.body_comvels[idx]

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        cur_vel = (xposafter - xposbefore)/self.dt
        ob = self._get_obs()
        ctrl_cost = 0.1 * 0.5 * np.square(action).sum()
        # run_cost = 1.*np.abs(self.get_body_comvel("torso")[0] - self.target_velocity)
        run_cost = 1.*np.abs(cur_vel - self.target_velocity[0])
        cost = ctrl_cost + run_cost
        # cost = run_cost
        reward = -cost
        done = False
        return ob, reward, done, dict(ctrl_cost=ctrl_cost, run_cost=run_cost, vel=cur_vel)
        # return ob, reward, done, dict(run_cost=run_cost, vel=cur_vel)

    def _get_obs(self):
        obs = np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
        ])
        return {
            'obs': obs.copy(),
            'obs_task_params': self.target_velocity
        }

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def reset(self, task_params=None, obs_task_params=None):
        if task_params is None:
            self.target_velocity = np.array([self.np_random.uniform(0.0, 3.0)])
        else:
            self.target_velocity = task_params['target_velocity']
        obs = super().reset()
        return obs

    @property
    def task_identifier(self):
        return self.target_velocity[0]

    def task_id_to_obs_task_params(self, task_id):
        return np.array([task_id])

    def log_statistics(self, paths):
        progs = [
            path['observations'][-1]['obs'][-3] - path['observations'][0]['obs'][-3]
            for path in paths
        ]
        return_dict = OrderedDict()
        return_dict['Avg Forward Progress'] = np.mean(progs)
        return_dict['Max Forward Progress'] = np.max(progs)
        return_dict['Min Forward Progress'] = np.min(progs)
        return_dict['Std Forward Progress'] = np.std(progs)

        run_costs = []
        ctrl_costs = []
        for path in paths:
            run_costs.append(np.sum([e_info['run_cost'] for e_info in path['env_infos']]))
            ctrl_costs.append(np.sum([e_info['ctrl_cost'] for e_info in path['env_infos']]))
        
        return_dict['Avg Run Rew'] = -1.0*np.mean(run_costs)
        return_dict['Max Run Rew'] = -1.0*np.min(run_costs)
        return_dict['Min Run Rew'] = -1.0*np.max(run_costs)
        return_dict['Std Run Rew'] = np.std(run_costs)

        return_dict['Avg Ctrl Rew'] = -1.0*np.mean(ctrl_costs)
        return_dict['Max Ctrl Rew'] = -1.0*np.min(ctrl_costs)
        return_dict['Min Ctrl Rew'] = -1.0*np.max(ctrl_costs)
        return_dict['Std Ctrl Rew'] = np.std(ctrl_costs)

        return return_dict
