import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from gym.spaces import Dict

from rllab.misc.instrument import VariantGenerator
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger, set_seed

from rlkit.envs import get_meta_env, get_meta_env_params_iters

from rlkit.torch.irl.disc_models.airl_disc import ThirdVersionSingleColorFetchCustomDisc
from rlkit.torch.irl.disc_models.airl_disc import TransferVersionSingleColorFetchCustomDisc
from rlkit.torch.irl.policy_optimizers.sac import NewSoftActorCritic
from rlkit.torch.sac.policies import WithZObsPreprocessedReparamTanhMultivariateGaussianPolicy
from rlkit.torch.networks import FlattenMlp, ObsPreprocessedQFunc, ObsPreprocessedVFunc

from rlkit.torch.irl.np_airl import NeuralProcessAIRL
from rlkit.torch.irl.meta_fairl import MetaFAIRL
from rlkit.torch.irl.encoders.conv_trivial_encoder import TrivialTrajEncoder, TrivialR2ZMap, TrivialNPEncoder, TrivialContextEncoder
# from rlkit.torch.irl.encoders.trivial_encoder import TrivialTrajEncoder, TrivialR2ZMap, TrivialNPEncoder

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

    # this script is for the non-meta-learning AIRL
    train_context_buffer, train_test_buffer = extra_data['meta_train']['context'], extra_data['meta_train']['test']
    test_context_buffer, test_test_buffer = extra_data['meta_test']['context'], extra_data['meta_test']['test']

    # set up the envs
    env_specs = variant['env_specs']
    meta_train_env, meta_test_env = get_meta_env(env_specs)
    meta_train_env.seed(variant['seed'])
    meta_test_env.seed(variant['seed'])

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
    if variant['algo_params']['state_only']: print('\n\nUSING STATE ONLY DISC\n\n')
    if variant['algo_params']['transfer_version']:
        disc_model = TransferVersionSingleColorFetchCustomDisc(
            clamp_magnitude=variant['disc_clamp_magnitude'],
            z_dim=variant['algo_params']['np_params']['z_dim'],
            gamma=0.99
        )
    else:
        disc_model = ThirdVersionSingleColorFetchCustomDisc(
            clamp_magnitude=variant['disc_clamp_magnitude'],
            state_only=variant['algo_params']['state_only'],
            wrap_absorbing=variant['algo_params']['wrap_absorbing'],
            z_dim=variant['algo_params']['np_params']['z_dim']
        )
    if variant['algo_params']['use_target_disc']:
        target_disc = disc_model.copy()
    else:
        target_disc = None
    print(disc_model)
    print(disc_model.clamp_magnitude)

    z_dim = variant['algo_params']['np_params']['z_dim']
    policy_net_size = variant['policy_net_size']
    hidden_sizes = [policy_net_size] * variant['num_hidden_layers']
    qf1 = ObsPreprocessedQFunc(
        target_disc.obs_processor if target_disc is not None else disc_model.obs_processor,
        z_dim,
        hidden_sizes=hidden_sizes,
        input_size=6 + 4 + 4 + 1*variant['algo_params']['wrap_absorbing'],
        output_size=1,
        wrap_absorbing=variant['algo_params']['wrap_absorbing']
    )
    qf2 = ObsPreprocessedQFunc(
        target_disc.obs_processor if target_disc is not None else disc_model.obs_processor,
        z_dim,
        hidden_sizes=hidden_sizes,
        input_size=6 + 4 + 4 + 1*variant['algo_params']['wrap_absorbing'],
        output_size=1,
        wrap_absorbing=variant['algo_params']['wrap_absorbing']
    )
    vf = ObsPreprocessedVFunc(
        target_disc.obs_processor if target_disc is not None else disc_model.obs_processor,
        z_dim,
        hidden_sizes=hidden_sizes,
        input_size=6 + 4 + 1*variant['algo_params']['wrap_absorbing'],
        output_size=1,
        wrap_absorbing=variant['algo_params']['wrap_absorbing']
    )
    policy = WithZObsPreprocessedReparamTanhMultivariateGaussianPolicy(
        target_disc.obs_processor if target_disc is not None else disc_model.obs_processor,
        z_dim,
        hidden_sizes=hidden_sizes,
        obs_dim=6 + 4,
        action_dim=4
    )

    policy_optimizer = NewSoftActorCritic(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        wrap_absorbing=variant['algo_params']['wrap_absorbing'],
        **variant['policy_params']
    )

    # Make the neural process
    traj_enc = TrivialTrajEncoder(state_only=variant['algo_params']['state_only'])
    context_enc = TrivialContextEncoder(
        variant['algo_params']['np_params']['agg_type'],
        traj_enc,
        state_only=variant['algo_params']['state_only']
    )
    r2z_map = TrivialR2ZMap(z_dim)
    
    np_enc = TrivialNPEncoder(
        context_enc,
        r2z_map,
        state_only=variant['algo_params']['state_only']
    )
    
    train_task_params_sampler, test_task_params_sampler = get_meta_env_params_iters(env_specs)

    if variant['meta_fairl']:
        print('\n\nUSING META-FAIRL\n\n')
        algorithm_class = MetaFAIRL
    else:
        print('\n\nUSING META-AIRL\n\n')
        algorithm_class = NeuralProcessAIRL
    
    algorithm = algorithm_class(
        meta_test_env, # env is the test env, training_env is the training env (following rlkit original setup)
        
        policy,
        disc_model,

        train_context_buffer,
        train_test_buffer,
        test_context_buffer,
        test_test_buffer,

        np_enc,

        policy_optimizer,

        training_env=meta_train_env, # the env used for generating trajectories
        train_task_params_sampler=train_task_params_sampler,
        test_task_params_sampler=test_task_params_sampler,

        target_disc=target_disc,
        **variant['algo_params']
    )

    if ptu.gpu_enabled():
        algorithm.cuda()
        # print(target_disc)
        # print(next(algorithm.discriminator.obs_processor.parameters()).is_cuda)
        # print(next(algorithm.main_policy.preprocess_model.parameters()).is_cuda)
        # print(algorithm.main_policy.preprocess_model is algorithm.main_policy.copy().preprocess_model)
        # print(algorithm.main_policy.preprocess_model is algorithm.main_policy.preprocess_model.copy())
        # 1/0
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
