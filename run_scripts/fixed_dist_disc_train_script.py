import yaml
import argparse
import importlib
import psutil
import os
from os import path
import argparse
import joblib
from time import sleep

import numpy as np
from gym.spaces import Dict

from rlkit.envs import get_env
import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.sac.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.irl.policy_optimizers.sac import NewSoftActorCritic
from rlkit.torch.networks import FlattenMlp, Mlp
from rlkit.torch.irl.disc_models.airl_disc import StandardAIRLDisc, ResNetAIRLDisc, ThreeWayResNetAIRLDisc
from rlkit.envs.wrappers import ScaledEnv
from rlkit.state_matching_algs.fixed_dist_disc_train_alg import FixedDistDiscTrainAlg
from rlkit.state_matching_algs.three_way_fixed_dist_disc_train_alg import ThreeWayFixedDistDiscTrainAlg


def experiment(variant):
    expert_buffer = joblib.load(variant['exp_xy_data_path'])['xy_data']
    policy_buffer = joblib.load(variant['pol_xy_data_path'])['xy_data']

    # set up the discriminator models
    if variant['threeway']:
        disc_model_class = ThreeWayResNetAIRLDisc
    else:
        if variant['use_resnet_disc']:
            disc_model_class = ResNetAIRLDisc
        else:
            disc_model_class = StandardAIRLDisc
    disc_model = disc_model_class(
        2, # obs is just x-y pos
        num_layer_blocks=variant['disc_num_blocks'],
        hid_dim=variant['disc_hid_dim'],
        hid_act=variant['disc_hid_act'],
        use_bn=variant['disc_use_bn'],
        clamp_magnitude=variant['disc_clamp_magnitude']
    )
    print(disc_model)
    print(disc_model.clamp_magnitude)

    # set up the AIRL algorithm
    alg_class = ThreeWayFixedDistDiscTrainAlg if variant['threeway'] else FixedDistDiscTrainAlg
    algorithm = alg_class(
        disc_model,
        expert_buffer,
        policy_buffer,
        **variant['algo_params']
    )
    print(algorithm.disc_optimizer.defaults['lr'])

    # train
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
