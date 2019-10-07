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
from rlkit.torch.networks import Mlp

from rlkit.torch.irl.few_shot_LfD_upper_bound import UpperBound
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

    # set up the envs
    env_specs = variant['env_specs']
    meta_train_env, meta_test_env = get_meta_env(env_specs)
    
    # set up the policy and training algorithm
    obs_dim = int(np.prod(meta_train_env.observation_space.spaces['obs'].shape))
    action_dim = int(np.prod(meta_train_env.action_space.shape))

    print('obs dim: %d' % obs_dim)
    print('act dim: %d' % action_dim)
    sleep(3)

    # make the disc model
    z_dim = variant['algo_params']['z_dim']

    # make the MLP
    hidden_sizes = [variant['algo_params']['mlp_hid_dim']] * variant['algo_params']['mlp_layers']
    obs_task_params_dim = int(np.prod(meta_train_env.observation_space.spaces['obs_task_params'].shape))
    mlp = Mlp(
        hidden_sizes,
        output_size=obs_task_params_dim if variant['algo_params']['training_regression'] else 1,
        input_size=z_dim if variant['algo_params']['training_regression'] else z_dim + 2*obs_task_params_dim,
        batch_norm=variant['algo_params']['mlp_use_bn']
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
    # ---------------
    # encoder = WeightShareTimestepBasedEncoder(
    #     obs_dim,
    #     action_dim,
    #     64,
    #     variant['algo_params']['r_dim'],
    #     variant['algo_params']['z_dim'],
    #     variant['algo_params']['enc_hid_dim'],
    #     variant['algo_params']['r2z_hid_dim'],
    #     variant['algo_params']['num_enc_layer_blocks'],
    #     hid_act='relu',
    #     use_bn=True,
    #     within_traj_agg=variant['algo_params']['within_traj_agg']
    # )
    # ---------------
    # traj_enc = ConvTrajEncoder(
    #     variant['algo_params']['np_params']['traj_enc_params']['num_conv_layers'],
    #     # obs_dim + action_dim,
    #     obs_dim + action_dim + obs_dim,
    #     variant['algo_params']['np_params']['traj_enc_params']['channels'],
    #     variant['algo_params']['np_params']['traj_enc_params']['kernel'],
    #     variant['algo_params']['np_params']['traj_enc_params']['stride'],
    # )
    # Dc2R_map = Dc2RMap(
    #     variant['algo_params']['np_params']['Dc2r_params']['agg_type'],
    #     traj_enc,
    #     state_only=False
    # )
    # r2z_map = R2ZMap(
    #     variant['algo_params']['np_params']['r2z_params']['num_layers'],
    #     variant['algo_params']['np_params']['traj_enc_params']['channels'],
    #     variant['algo_params']['np_params']['r2z_params']['hid_dim'],
    #     variant['algo_params']['z_dim']
    # )
    # encoder = NPEncoder(
    #     Dc2R_map,
    #     r2z_map,
    # )

    
    train_task_params_sampler, test_task_params_sampler = get_meta_env_params_iters(env_specs)

    algorithm = UpperBound(
        meta_train_env,
        
        train_context_buffer,
        train_test_buffer,
        test_context_buffer,
        test_test_buffer,

        mlp,
        encoder,

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
