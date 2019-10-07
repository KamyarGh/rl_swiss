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
from rlkit.torch.irl.adv_bc import AdvBC
from rlkit.torch.networks import FlattenMlp, Mlp
from rlkit.torch.irl.disc_models.airl_disc import StandardAIRLDisc
from rlkit.envs.wrappers import ScaledEnv

EXPERT_LISTING_YAML_PATH = '/h/kamyar/oorl_rlkit/rlkit/torch/irl/experts.yaml'


def experiment(variant):
    with open(EXPERT_LISTING_YAML_PATH, 'r') as f:
        listings = yaml.load(f.read())
    expert_dir = listings[variant['expert_name']]['exp_dir']
    specific_run = listings[variant['expert_name']]['seed_runs'][variant['expert_seed_run_idx']]
    file_to_load = path.join(expert_dir, specific_run, 'extra_data.pkl')
    extra_data = joblib.load(file_to_load)

    # this script is for the non-meta-learning GAIL
    expert_buffer = extra_data['train']

    # set up the env
    env_specs = variant['env_specs']
    if env_specs['train_test_env']:
        env, training_env = get_env(env_specs)
    else:
        env, _ = get_env(env_specs)
        training_env, _ = get_env(env_specs)
    env.seed(variant['seed'])
    training_env.seed(variant['seed'])
    print(env.observation_space)

    if variant['scale_env_with_given_demo_stats']:
        assert not env_specs['normalized']
        env = ScaledEnv(
            env,
            obs_mean=extra_data['obs_mean'],
            obs_std=extra_data['obs_std'],
            acts_mean=extra_data['acts_mean'],
            acts_std=extra_data['acts_std'],
        )
        training_env = ScaledEnv(
            training_env,
            obs_mean=extra_data['obs_mean'],
            obs_std=extra_data['obs_std'],
            acts_mean=extra_data['acts_mean'],
            acts_std=extra_data['acts_std'],
        )

    # compute obs_dim and action_dim
    if isinstance(env.observation_space, Dict):
        if not variant['algo_params']['policy_uses_pixels']:
            obs_dim = int(np.prod(env.observation_space.spaces['obs'].shape))
            if variant['algo_params']['policy_uses_task_params']:
                if variant['algo_params']['concat_task_params_to_policy_obs']:
                    obs_dim += int(np.prod(env.observation_space.spaces['obs_task_params'].shape))
                else:
                    raise NotImplementedError()
        else:
            raise NotImplementedError()
    else:
        obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    print(obs_dim, action_dim)
    
    sleep(3)

    # set up the policy models
    policy_net_size = variant['policy_net_size']
    hidden_sizes = [policy_net_size] * variant['policy_num_hidden_layers']
    policy = ReparamTanhMultivariateGaussianPolicy(
        hidden_sizes=hidden_sizes,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    # set up the discriminator models
    disc_model = StandardAIRLDisc(
        obs_dim + action_dim,
        num_layer_blocks=variant['disc_num_blocks'],
        hid_dim=variant['disc_hid_dim'],
        hid_act=variant['disc_hid_act'],
        use_bn=variant['disc_use_bn'],
        clamp_magnitude=variant['disc_clamp_magnitude']
    )
    print(disc_model)
    print(disc_model.clamp_magnitude)

    # set up the AdvBC algorithm
    algorithm = AdvBC(
        env,
        policy,
        disc_model,
        expert_buffer,
        training_env=training_env,
        wrap_absorbing=variant['wrap_absorbing_state'],
        **variant['algo_params']
    )
    print(algorithm.use_target_disc)
    print(algorithm.soft_target_disc_tau)
    print(algorithm.exploration_policy)
    print(algorithm.eval_policy)
    print(algorithm.disc_optimizer.defaults['lr'])
    print(algorithm.policy_optimizer.defaults['lr'])

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
