import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from copy import deepcopy

from gym.spaces import Dict

from rllab.misc.instrument import VariantGenerator
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger, set_seed

from rlkit.envs import get_meta_env, get_meta_env_params_iters
from rlkit.envs.wrappers import ScaledMetaEnv

from rlkit.torch.sac.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.networks import FlattenMlp

from rlkit.torch.irl.meta_dagger import MetaDagger
from rlkit.torch.irl.encoders.mlp_encoder import TimestepBasedEncoder, WeightShareTimestepBasedEncoder
from rlkit.torch.irl.encoders.conv_seq_encoder import ConvTrajEncoder, R2ZMap, Dc2RMap, NPEncoder

import yaml
import argparse
import importlib
import psutil
import os
from os import path
import argparse
import joblib
from time import sleep


EXPERT_LISTING_YAML_PATH = '/h/kamyar/oorl_rlkit/rlkit/torch/irl/experts.yaml'


def experiment(variant):
    with open(EXPERT_LISTING_YAML_PATH, 'r') as f:
        listings = yaml.load(f.read())
    expert_dir = listings[variant['expert_name']]['exp_dir']
    specific_run = listings[variant['expert_name']]['seed_runs'][variant['expert_seed_run_idx']]
    file_to_load = path.join(expert_dir, specific_run, 'extra_data.pkl')
    extra_data = joblib.load(file_to_load)

    # this script is for the non-meta-learning airl
    train_context_buffer, train_test_buffer = extra_data['meta_train']['context'], extra_data['meta_train']['test']
    test_context_buffer, test_test_buffer = extra_data['meta_test']['context'], extra_data['meta_test']['test']

    # load the expert
    expert_policy = joblib.load(variant['expert_policy'])['algorithm']
    expert_policy.replay_buffer = None

    # set up the envs
    env_specs = variant['env_specs']
    meta_train_env, meta_test_env = get_meta_env(env_specs)
    meta_train_env.seed(variant['seed'])
    meta_test_env.seed(variant['seed'])

    if variant['scale_env_with_given_demo_stats']:
        assert not env_specs['normalized']
        meta_train_env = ScaledMetaEnv(
            meta_train_env,
            obs_mean=extra_data['obs_mean'],
            obs_std=extra_data['obs_std'],
            acts_mean=extra_data['acts_mean'],
            acts_std=extra_data['acts_std'],
        )
        meta_test_env = ScaledMetaEnv(
            meta_test_env,
            obs_mean=extra_data['obs_mean'],
            obs_std=extra_data['obs_std'],
            acts_mean=extra_data['acts_mean'],
            acts_std=extra_data['acts_std'],
        )
    print(meta_train_env)
    print(meta_test_env)

    # set up the policy and training algorithm
    if isinstance(meta_train_env.observation_space, Dict):
        if variant['algo_params']['policy_uses_pixels']:
            raise NotImplementedError('Not implemented pixel version of things!')
        else:
            obs_dim = int(np.prod(meta_train_env.observation_space.spaces['obs'].shape))
    else:
        obs_dim = int(np.prod(meta_train_env.observation_space.shape))
    action_dim = int(np.prod(meta_train_env.action_space.shape))

    print('obs dim: %d' % obs_dim)
    print('act dim: %d' % action_dim)
    sleep(3)

    # make the disc model
    z_dim = variant['algo_params']['z_dim']
    policy_net_size = variant['policy_net_size']
    hidden_sizes = [policy_net_size] * variant['num_hidden_layers']

    policy = ReparamTanhMultivariateGaussianPolicy(
        hidden_sizes=hidden_sizes,
        obs_dim=obs_dim + z_dim,
        action_dim=action_dim,
    )

    # Make the encoder
    encoder = TimestepBasedEncoder(
        2*obs_dim + action_dim, #(s,a,s')
        variant['algo_params']['r_dim'],
        variant['algo_params']['z_dim'],
        variant['algo_params']['enc_hid_dim'],
        variant['algo_params']['r2z_hid_dim'],
        variant['algo_params']['num_enc_layer_blocks'],
        hid_act='relu',
        use_bn=True,
        within_traj_agg=variant['algo_params']['within_traj_agg']
    )
    
    train_task_params_sampler, test_task_params_sampler = get_meta_env_params_iters(env_specs)

    algorithm = MetaDagger(
        meta_test_env, # env is the test env, training_env is the training env (following rlkit original setup)
        
        policy,
        expert_policy,

        train_context_buffer,
        train_test_buffer,
        test_context_buffer,
        test_test_buffer,

        encoder,

        training_env=meta_train_env, # the env used for generating trajectories
        train_task_params_sampler=train_task_params_sampler,
        test_task_params_sampler=test_task_params_sampler,

        **variant['algo_params']
    )

    for task_id in train_context_buffer.task_replay_buffers:
        erb = train_context_buffer.task_replay_buffers[task_id]
        rb = algorithm.replay_buffer.task_replay_buffers[task_id]
        erb_size = erb._size
        print(erb_size)
        for k in erb._observations:
            rb._observations[k][:erb_size] = erb._observations[k][:erb_size]
            rb._next_obs[k][:erb_size] = erb._next_obs[k][:erb_size]
        rb._actions[:erb_size] = erb._actions[:erb_size]
        rb._rewards[:erb_size] = erb._rewards[:erb_size]
        rb._terminals[:erb_size] = erb._terminals[:erb_size]
        rb._absorbing[:erb_size] = erb._absorbing[:erb_size]
        rb._size = erb_size
        rb._top = erb_size
    
    # print('\n\n')
    # for task_id in algorithm.replay_buffer.task_replay_buffers:
    #     rb = algorithm.replay_buffer.task_replay_buffers[task_id]
    #     print(rb._size)
    #     print(rb._top)
    #     print(rb._max_replay_buffer_size)

    if ptu.gpu_enabled():
        expert_policy.cuda()
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
    print('\n\nSET SEED TO {}\n\n'.format(seed))
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    experiment(exp_specs)
