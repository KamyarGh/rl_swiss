import numpy as np
from gym.envs.mujoco import HalfCheetahEnv, InvertedPendulumEnv, ReacherEnv
from gym.spaces import Dict

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.gen_exp_traj_algorithm import ExpertTrajGeneratorAlgorithm
from rlkit.data_management.expert_replay_buffer import ExpertReplayBuffer

from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.sac.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.sac.policies import MakeDeterministic

from rlkit.envs import get_env

import yaml
import argparse
import importlib
import psutil
import os
from os import path
import argparse
import joblib
import json


def experiment(specs):    
    with open(path.join(specs['specific_exp_dir'], 'variant.json'), 'r') as f:
        variant = json.load(f)
    variant['algo_params']['do_not_train'] = True
    variant['seed'] = specs['seed']
    policy = joblib.load(path.join(specs['specific_exp_dir'], 'params.pkl'))['exploration_policy']

    assert False, 'Do you really wanna make it deterministic?'
    policy = MakeDeterministic(policy)

    env_specs = variant['env_specs']
    env, _ = get_env(env_specs)
    training_env, _ = get_env(env_specs)

    variant['algo_params']['replay_buffer_size'] = int(np.floor(
        specs['num_episodes'] * variant['algo_params']['max_path_length'] / specs['subsampling']
    ))
    # Hack until I figure out how things are gonna be in general then I'll clean it up
    if 'policy_uses_pixels' not in variant['algo_params']: variant['algo_params']['policy_uses_pixels'] = False
    if 'policy_uses_task_params' not in variant['algo_params']: variant['algo_params']['policy_uses_task_params'] = False
    if 'concat_task_params_to_policy_obs' not in variant['algo_params']: variant['algo_params']['concat_task_params_to_policy_obs'] = False
    replay_buffer = ExpertReplayBuffer(
        variant['algo_params']['replay_buffer_size'],
        env,
        subsampling=specs['subsampling'],
        policy_uses_pixels=variant['algo_params']['policy_uses_pixels'],
        policy_uses_task_params=variant['algo_params']['policy_uses_task_params'],
        concat_task_params_to_policy_obs=variant['algo_params']['concat_task_params_to_policy_obs'],
    )
    variant['algo_params']['freq_saving'] = 1

    algorithm = ExpertTrajGeneratorAlgorithm(
        env=env,
        training_env=training_env,
        exploration_policy=policy,
        replay_buffer=replay_buffer,
        max_num_episodes=specs['num_episodes'],
        **variant['algo_params']
    )

    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()

    return 1


def exp_fn(variant):
    exp_id = variant['exp_id']

    print(variant.keys())
    exp_prefix = variant['exp_name']
    set_seed(exp_specs['seed'])
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=variant)

    # run the experiment
    exp_return = experiment(variant)
    return exp_return


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)
    
    exp_fn(exp_specs)
