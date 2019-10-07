"""
Run PyTorch Soft Actor Critic on HalfCheetahEnv.

NOTE: You need PyTorch 0.3 or more (to have torch.distributions)
"""
import numpy as np
from gym.envs.mujoco import HalfCheetahEnv, InvertedPendulumEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger, build_nested_variant_generator
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.networks import FlattenMlp

from rlkit.envs import get_meta_env

import yaml
import argparse
import importlib


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

    return 1


def pool_function(arg):
    variant = arg[0]
    exp_id = arg[1]
    exp_prefix = variant['meta_data']['exp_name']
    setup_logger(exp_prefix=exp_prefix=, exp_id=exp_id, variant=variant)
    return experiment(variant)


if __name__ == "__main__":
    # noinspection PyTypeChecker
    from multiprocessing import Pool

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)
    
    vg_fn = build_nested_variant_generator(exp_specs)
    all_exp_args = []
    for i, variant in enumerate(vg_fn()):
        all_exp_args.append([variant, i])
    
    num_total = len(all_exp_args)
    num_workers = min(exp_specs['meta_data']['num_workers'], num_total)
    p = Pool(num_workers)

    print(
        '\n\n\n\n{}/{} experiments ran successfully!'.format(
            sum(p.map(pool_function, all_exp_args)),
            num_total
        )
    )
