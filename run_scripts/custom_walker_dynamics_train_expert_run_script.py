'''
This is a hack script for now until I figure things out
'''
import numpy as np
from gym.envs.mujoco import HalfCheetahEnv, InvertedPendulumEnv, ReacherEnv
from gym.spaces import Dict

import torch
import torch.nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.sac.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.sac.policies import AntRandGoalCustomReparamTanhMultivariateGaussianPolicy
from rlkit.torch.sac.policies import AntCustomGatingV1ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.sac.sac import NewSoftActorCritic, MetaNewSoftActorCritic
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.networks import AntRandGoalCustomQFunc, AntRandGoalCustomVFunc
from rlkit.torch.networks import AntCustomGatingVFuncV1, AntCustomGatingQFuncV1

from rlkit.envs import get_env, get_meta_env, get_meta_env_params_iters

import yaml
import argparse
import importlib
import psutil
import os
import argparse

from rlkit.envs.walker_random_dynamics import SingleTaskWalkerEnv
from rlkit.envs.walker_random_dynamics import _MetaExpertTrainParamsSampler as WalkerTrainParamsSampler
from rlkit.envs.walker_random_dynamics import _MetaExpertTestParamsSampler as WalkerTestParamsSampler

def experiment(variant):
    task_mode = variant['task_mode'] # train, test, eval
    task_idx = variant['task_idx']

    if task_mode == 'train':
        task_sampler = WalkerTrainParamsSampler()
    elif task_mode == 'test':
        task_sampler = WalkerTestParamsSampler()
    else:
        raise NotImplementedError()
    task_params = task_sampler.get_task(task_idx)
    obs_task_params = task_sampler.get_obs_task_params(task_params)
    env = SingleTaskWalkerEnv(task_params, obs_task_params)
    training_env = SingleTaskWalkerEnv(task_params, obs_task_params)

    print(env.observation_space)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    net_size = variant['net_size']
    hidden_sizes = [net_size] * variant['num_hidden_layers']
    print('Using simple model')
    qf1 = FlattenMlp(
        hidden_sizes=hidden_sizes,
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=hidden_sizes,
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=hidden_sizes,
        input_size=obs_dim,
        output_size=1,
    )
    policy = ReparamTanhMultivariateGaussianPolicy(
        hidden_sizes=hidden_sizes,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    algorithm = NewSoftActorCritic(
        env=env,
        training_env=training_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        **variant['algo_params']
    )
    
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
    
    if exp_specs['use_gpu']:
        print('\n\nUSING GPU\n\n')
        ptu.set_gpu_mode(True)
    exp_id = exp_specs['exp_id']
    exp_prefix = exp_specs['exp_name']
    seed = exp_specs['seed']
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    experiment(exp_specs)
