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

# from rlkit.torch.irl.disc_models.gail_disc import Model as GAILDiscModel
from rlkit.torch.irl.disc_models.gail_disc import MlpGAILDisc
from rlkit.torch.irl.policy_optimizers.sac import NewSoftActorCritic
from rlkit.torch.sac.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.networks import FlattenMlp

from rlkit.torch.irl.np_airl import NeuralProcessAIRL
from rlkit.torch.irl.encoders.trivial_encoder import TrivialTrajEncoder, TrivialR2ZMap, TrivialNPEncoder

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

    # this script is for the non-meta-learning GAIL
    train_context_buffer, train_test_buffer = extra_data['meta_train']['context'], extra_data['meta_train']['test']
    test_context_buffer, test_test_buffer = extra_data['meta_test']['context'], extra_data['meta_test']['test']

    # set up the envs
    env_specs = variant['env_specs']
    meta_train_env, meta_test_env = get_meta_env(env_specs)

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

    # make the policy and policy optimizer
    hidden_sizes = [variant['algo_params']['policy_net_size']] * variant['algo_params']['policy_num_layers']
    z_dim = variant['algo_params']['np_params']['z_dim']
    qf1 = FlattenMlp(
        hidden_sizes=hidden_sizes,
        input_size=obs_dim + z_dim + action_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=hidden_sizes,
        input_size=obs_dim + z_dim + action_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=hidden_sizes,
        input_size=obs_dim + z_dim,
        output_size=1,
    )
    policy = ReparamTanhMultivariateGaussianPolicy(
        hidden_sizes=hidden_sizes,
        obs_dim=obs_dim + z_dim,
        action_dim=action_dim,
    )
    
    # disc_model = GAILDiscModel(obs_dim + action_dim + z_dim, hid_dim=variant['algo_params']['disc_net_size'])
    disc_model = MlpGAILDisc(
        hidden_sizes=variant['disc_hidden_sizes'],
        output_size=1,
        input_size=obs_dim + action_dim + z_dim,
        hidden_activation=torch.nn.functional.tanh,
        layer_norm=variant['disc_uses_layer_norm']
        # output_activation=identity,
    )

    policy_optimizer = NewSoftActorCritic(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        **variant['algo_params']['policy_params']
    )

    # Make the neural process
    # in the initial version we are assuming all trajectories have the same length
    timestep_enc_params = variant['algo_params']['np_params']['traj_enc_params']['timestep_enc_params']
    traj_enc_params = variant['algo_params']['np_params']['traj_enc_params']['traj_enc_params']
    timestep_enc_params['input_size'] = obs_dim + action_dim
    
    traj_samples, _ = train_context_buffer.sample_trajs(1, num_tasks=1)
    # len_context_traj = traj_samples[0][0]['observations'].shape[0]
    len_context_traj = 5
    traj_enc_params['input_size'] = timestep_enc_params['output_size'] * len_context_traj

    traj_enc = TrivialTrajEncoder(
        timestep_enc_params,
        traj_enc_params
    )

    trunk_params = variant['algo_params']['np_params']['r2z_map_params']['trunk_params']
    trunk_params['input_size'] = traj_enc.output_size
    
    split_params = variant['algo_params']['np_params']['r2z_map_params']['split_heads_params']
    split_params['input_size'] = trunk_params['output_size']
    split_params['output_size'] = variant['algo_params']['np_params']['z_dim']
    
    r2z_map = TrivialR2ZMap(
        trunk_params,
        split_params
    )
    
    np_enc = TrivialNPEncoder(
        variant['algo_params']['np_params']['np_enc_params']['agg_type'],
        traj_enc,
        r2z_map
    )
    
    # class StupidDistFormat():
    #     def __init__(self, var):
    #         self.mean = var
    # class ZeroModule(nn.Module):
    #     def __init__(self, z_dim):
    #         super().__init__()
    #         self.z_dim = z_dim
    #         self.fc = nn.Linear(10,10)
        
    #     def forward(self, context):
    #         c_len = len(context)
    #         return StupidDistFormat(Variable(torch.zeros(c_len, self.z_dim), requires_grad=False))
    # np_enc = ZeroModule(variant['algo_params']['np_params']['z_dim'])


    train_task_params_sampler, test_task_params_sampler = get_meta_env_params_iters(env_specs)
    algorithm = NeuralProcessAIRL(
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
    
    exp_id = exp_specs['exp_id']
    exp_prefix = exp_specs['exp_name']
    seed = exp_specs['seed']
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    experiment(exp_specs)
