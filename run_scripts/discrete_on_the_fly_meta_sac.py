import numpy as np
from gym.envs.mujoco import HalfCheetahEnv, InvertedPendulumEnv, ReacherEnv
from gym.spaces import Discrete

from rllab.misc.instrument import VariantGenerator
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.envs.meta_maze import Maze, MazeSampler
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.sac.policies import TanhGaussianPolicy, DiscretePolicy
from rlkit.torch.sac.sac import MetaSoftActorCritic, NewMetaSoftActorCritic
from rlkit.torch.networks import FlattenMlp

from rlkit.envs import OnTheFlyEnvSampler

import yaml
import argparse
import importlib
import psutil
import os
import argparse


def experiment(variant):
    # we have to generate the combinations for the env_specs
    env_specs = variant['env_specs']
    env_sampler = MazeSampler(env_specs)
    sample_env, _ = env_sampler()
    meta_params_dim = 0
    
    obs_dim = int(np.prod(sample_env.observation_space.shape))
    if isinstance(sample_env.action_space, Discrete):
        action_dim = int(sample_env.action_space.n)
    else:
        action_dim = int(np.prod(sample_env.action_space.shape))

    net_size = variant['net_size']
    qf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + meta_params_dim,
        output_size=action_dim,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + meta_params_dim,
        output_size=1,
    )
    policy = DiscretePolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim + meta_params_dim,
        action_dim=action_dim,
    )
    algorithm = MetaSoftActorCritic(
        env_sampler=env_sampler,
        policy=policy,
        qf=qf,
        vf=vf,
        **variant['algo_params']
    )
    # assert False, "Have not added new sac yet!"
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()

    return 1


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)
    
    exp_id = exp_specs['exp_id']
    exp_prefix = exp_specs['exp_name']
    seed = exp_specs['seed']
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    experiment(exp_specs)
