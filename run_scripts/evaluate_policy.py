import yaml
import argparse
import joblib
import numpy as np

from gym.spaces import Dict
from rlkit.envs import get_env

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.core import eval_util

from rlkit.envs.wrappers import ScaledEnv
from rlkit.samplers import PathSampler
from rlkit.torch.sac.policies import MakeDeterministic


def experiment(variant):
    env_specs = variant['env_specs']
    env = get_env(env_specs)
    env.seed(env_specs['eval_env_seed'])

    print('\n\nEnv: {}'.format(env_specs['env_name']))
    print('kwargs: {}'.format(env_specs['env_kwargs']))
    print('Obs Space: {}'.format(env.observation_space))
    print('Act Space: {}\n\n'.format(env.action_space))
    
    if variant['scale_env_with_demo_stats']:
        with open('expert_demos_listing.yaml', 'r') as f:
            listings = yaml.load(f.read())
        expert_demos_path = listings[variant['expert_name']]['file_paths'][variant['expert_idx']]
        buffer_save_dict = joblib.load(expert_demos_path)
        env = ScaledEnv(
            env,
            obs_mean=buffer_save_dict['obs_mean'],
            obs_std=buffer_save_dict['obs_std'],
            acts_mean=buffer_save_dict['acts_mean'],
            acts_std=buffer_save_dict['acts_std'],
        )

    policy = joblib.load(variant['policy_checkpoint'])['exploration_policy']
    if variant['eval_deterministic']:
        policy = MakeDeterministic(policy)
    policy.to(ptu.device)

    eval_sampler = PathSampler(
        env,
        policy,
        variant['num_eval_steps'],
        variant['max_path_length'],
        no_terminal=variant['no_terminal'],
        render=variant['render'],
        render_kwargs=variant['render_kwargs']
    )
    test_paths = eval_sampler.obtain_samples()
    average_returns = eval_util.get_average_returns(test_paths)
    print(average_returns)

    return 1


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)

    if exp_specs['num_gpu_per_worker'] > 0:
        print('\n\nUSING GPU\n\n')
        ptu.set_gpu_mode(True)
    exp_id = exp_specs['exp_id']
    exp_prefix = exp_specs['exp_name']
    seed = exp_specs['seed']
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    experiment(exp_specs)
