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
from rlkit.envs.mil_pusher_env import ParamsSampler as PusherParamsSampler
from rlkit.envs.mil_pusher_env import build_pusher_getter

from rlkit.data_management.pusher_mil_pytorch_data_loader import build_train_val_datasets

# from rlkit.torch.sac.policies import PusherTaskReparamTanhMultivariateGaussianPolicy, BaselineContextualPolicy
from rlkit.torch.sac.policies import PusherTaskReparamMultivariateGaussianPolicy, BaselineContextualPolicy
from rlkit.torch.sac.policies import YetAnotherPusherTaskReparamMultivariateGaussianPolicy
# from rlkit.torch.networks import PusherTaskQFunc, PusherTaskVFunc
from rlkit.torch.irl.encoders.pusher_video_encoder import PusherVideoEncoder, PusherLastTimestepEncoder, PusherAggTimestepEncoder
from rlkit.torch.irl.pusher_mil_np_bc import PusherSpecificNeuralProcessBC
from rlkit.torch.irl.disc_models.pusher_disc import ImageProcessor

import yaml
import argparse
import importlib
import psutil
import os
from os import path
import argparse
import joblib
from time import sleep


def experiment(variant, log_dir):
    # load the data and get train-val idxs
    train_ds, val_ds, train_idx, val_idx, state_dim, action_dim, state_mean, state_std = build_train_val_datasets(
        variant['algo_params']['num_context_trajs_for_training'],
        variant['algo_params']['num_test_trajs_for_training'],
    )

    # the meshes for these envs have bugs for some reason
    train_idx = [ind for ind in train_idx if ind not in [507,618,288,564,508,477,100,5,287,476,617,565,101]]
    val_idx = [ind for ind in val_idx if ind not in [765,766]]

    # for debugging
    print('\n\n\n\n\nNOT DEBUG RUN\n\n\n\n\n')
    # print('\n\n\n\n\nDEBUG RUN!!!!!!!!\n\n\n\n\n')
    # train_idx = train_idx[:16]
    # val_idx = val_idx[:16]

    # build the train and test params samplers
    train_task_names = [
        'train_%d'%ind for ind in train_idx
    ]
    train_params_sampler = PusherParamsSampler(train_task_names)

    val_task_names = [
        'val_%d'%ind for ind in val_idx
    ]
    test_params_sampler = PusherParamsSampler(val_task_names)

    # set up the models
    # z_dims = variant['algo_params']['z_dims']
    # encoder
    # encoder = PusherVideoEncoder(z_dims)
    if variant['algo_params']['easy_context']:
        encoder = PusherLastTimestepEncoder()
    elif variant['algo_params']['using_all_context']:
        encoder = PusherAggTimestepEncoder(state_dim, action_dim)
    else:
        encoder = PusherVideoEncoder()
    # image processor
    # image_processor = ImageProcessor(z_dims[0])
    image_processor = ImageProcessor()
    # policy
    policy_net_size = variant['policy_net_size']
    hidden_sizes = [policy_net_size] * variant['num_hidden_layers']
    if variant['algo_params']['use_basic_contextual_policy']:
        policy = BaselineContextualPolicy(action_dim)
    elif variant['algo_params']['using_all_context']:
        policy = YetAnotherPusherTaskReparamMultivariateGaussianPolicy(
            image_processor=image_processor,
            image_only=variant['algo_params']['image_only'],
            train_img_processor=True,
            hidden_sizes=hidden_sizes,
            obs_dim=state_dim + image_processor.output_dim + 64, # 64 for extra latents
            action_dim=action_dim,
        )
    else:
        # policy = PusherTaskReparamTanhMultivariateGaussianPolicy(
        #     image_processor=image_processor,
        #     image_only=variant['algo_params']['image_only'],
        #     train_img_processor=True,
        #     hidden_sizes=hidden_sizes,
        #     obs_dim=state_dim + image_processor.output_dim,
        #     action_dim=action_dim,
        # )
        policy = PusherTaskReparamMultivariateGaussianPolicy(
            image_processor=image_processor,
            image_only=variant['algo_params']['image_only'],
            train_img_processor=True,
            hidden_sizes=hidden_sizes,
            obs_dim=image_processor.output_dim if variant['algo_params']['image_only'] else state_dim + image_processor.output_dim,
            action_dim=action_dim,
        )

    # set up the environment
    training_env_getter = build_pusher_getter(
        train_idx,
        distractors=True,
        mode='train',
        state_mean=state_mean[0],
        state_std=state_std[0]
    )
    val_env_getter = build_pusher_getter(
        val_idx,
        distractors=True,
        mode='val',
        state_mean=state_mean[0],
        state_std=state_std[0]
    )

    # set up the algorithm
    algorithm = PusherSpecificNeuralProcessBC(
        policy,
        encoder,

        train_ds,
        val_ds,

        train_task_params_sampler=train_params_sampler,
        test_task_params_sampler=test_params_sampler,

        use_env_getter=True,
        training_env_getter=training_env_getter,
        test_env_getter=val_env_getter,
        
        get_full_obs_dict=True,
        log_dir=log_dir,
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
    log_dir = setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    experiment(exp_specs, log_dir)
