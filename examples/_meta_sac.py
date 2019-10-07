"""
Run PyTorch Soft Actor Critic on HalfCheetahEnv.

NOTE: You need PyTorch 0.3 or more (to have torch.distributions)
"""
import numpy as np
from gym.envs.mujoco import HalfCheetahEnv, InvertedPendulumEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.networks import FlattenMlp

from rlkit.envs import get_meta_env


def experiment(variant):
    # env = NormalizedBoxEnv(HalfCheetahEnv())
    # env = NormalizedBoxEnv(InvertedPendulumEnv())
    env = NormalizedBoxEnv(get_meta_env(variant['env_specs']))
    training_env = NormalizedBoxEnv(get_meta_env(variant['env_specs']))
    
    # Or for a specific version:
    # import gym
    # env = NormalizedBoxEnv(gym.make('HalfCheetah-v1'))

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    net_size = variant['net_size']
    qf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    algorithm = SoftActorCritic(
        env=env,
        training_env=training_env,
        policy=policy,
        qf=qf,
        vf=vf,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    
    variant = dict(
        algo_params=dict(
            num_epochs=100,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            num_updates_per_env_step=4,
            batch_size=128,
            max_path_length=999,
            discount=0.99,
            reward_scale=1.0,

            soft_target_tau=0.01,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,

            render=False,

            save_replay_buffer=True,
        ),

        net_size=128,

        env_specs=dict(
            base_env_name='gravity_gear_inverted_pendulum',
            gravity=-9.81,
            gear=100
        )
    )
    setup_logger('pendulum_gravity_debug', exp_id=9, variant=variant)
    experiment(variant)
