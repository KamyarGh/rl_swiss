"""
Run DQN on grid world.
"""

import gym
import numpy as np
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.dqn.dqn import MetaDQN
from rlkit.torch.networks import Mlp, ConvNet
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.envs.meta_maze import MazeSampler

import argparse
import yaml


def experiment(variant):
    env_sampler = MazeSampler(variant['env_specs'])
    env, _ = env_sampler()

    if variant['conv_input']:
        qf = ConvNet(
            kernel_sizes=variant['kernel_sizes'], num_channels=variant['num_channels'],
            strides=variant['strides'], paddings=variant['paddings'],
            hidden_sizes=variant['hidden_sizes'], input_size=env.observation_space.shape,
            output_size=env.action_space.n
        )
    else:
        qf = Mlp(
            hidden_sizes=[variant['net_size'] for _ in range(variant['num_layers'])],
            input_size=int(np.prod(env.observation_space.shape)),
            output_size=env.action_space.n,
        )
    qf_criterion = nn.MSELoss()
    # Use this to switch to DoubleDQN
    # algorithm = DoubleDQN(
    print('WTF is going on!')
    print(env_sampler)
    algorithm = MetaDQN(
        env_sampler=env_sampler,
        qf=qf,
        qf_criterion=qf_criterion,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
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
