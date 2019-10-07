import numpy as np
from gym.envs.mujoco import HalfCheetahEnv, InvertedPendulumEnv, ReacherEnv

from rllab.misc.instrument import VariantGenerator
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.sac.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.sac.sac import NewMetaSoftActorCritic
from rlkit.torch.networks import FlattenMlp

from rlkit.envs import OnTheFlyEnvSampler

import yaml
import argparse
import importlib
import psutil
import os
import argparse


def experiment(variant):
    print('RUNNING')
    # we have to generate the combinations for the env_specs
    env_specs = variant['env_specs']
    env_sampler = OnTheFlyEnvSampler(env_specs)

    # Build the normalizer for the env params
    env_spec_ranges = {}
    for k, v in env_specs.items():
        if isinstance(v, list):
            env_spec_ranges[k] = v
    
    mean = []
    half_diff = []
    for k in sorted(env_spec_ranges.keys()):
        r = env_spec_ranges[k]
        mean.append((r[0]+r[1]) / 2.0)
        half_diff.append((r[1]-r[0]) / 2.0)
    mean = np.array(mean)
    half_diff = np.array(half_diff)

    def env_params_normalizer(params):
        return (params - mean) / half_diff
    
    variant['algo_params']['env_params_normalizer'] = env_params_normalizer

    # set up similar to non-meta version
    sample_env, _ = env_sampler()
    if variant['algo_params']['concat_env_params_to_obs']:
        meta_params_dim = sample_env.env_meta_params.shape[0]
    else:
        meta_params_dim = 0
    obs_dim = int(np.prod(sample_env.observation_space.shape))
    action_dim = int(np.prod(sample_env.action_space.shape))

    net_size = variant['net_size']
    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim + meta_params_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim + meta_params_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + meta_params_dim,
        output_size=1,
    )
    policy = ReparamTanhMultivariateGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim + meta_params_dim,
        action_dim=action_dim,
    )
    algorithm = NewMetaSoftActorCritic(
        env_sampler=env_sampler,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        **variant['algo_params']
    )
    # assert False, "Have not added new sac yet!"
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()

    return 1


if __name__ == '__main__':
    print('fucker')
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
