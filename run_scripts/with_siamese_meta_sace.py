import numpy as np
from gym.envs.mujoco import HalfCheetahEnv, InvertedPendulumEnv, ReacherEnv

from rllab.misc.instrument import VariantGenerator
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import MetaSoftActorCritic
from rlkit.torch.networks import FlattenMlp

from rlkit.envs import EnvSampler

import yaml
import argparse
import importlib
import psutil
import os
import argparse


def experiment(variant):
    # we have to generate the combinations for the env_specs
    env_specs = variant['env_specs']
    env_specs_vg = VariantGenerator()
    env_spec_constants = {}
    for k, v in env_specs.items():
        if isinstance(v, list):
            env_specs_vg.add(k, v)
        else:
            env_spec_constants[k] = v
    
    env_specs_list = []
    for es in env_specs_vg.variants():
        del es['_hidden_keys']
        es.update(env_spec_constants)
        env_specs_list.append(es)
    print(env_specs_list)
    
    print(env_specs_list[0])
    env_sampler = EnvSampler(env_specs_list)

    # set up similar to non-meta version
    sample_env, _ = env_sampler()
    if variant['algo_params']['concat_env_params_to_obs']:
        meta_params_dim = sample_env.env_meta_params.shape[0]
    else:
        meta_params_dim = 0
    obs_dim = int(np.prod(sample_env.observation_space.shape))
    action_dim = int(np.prod(sample_env.action_space.shape))

    net_size = variant['net_size']
    qf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim + meta_params_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + meta_params_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim + meta_params_dim,
        action_dim=action_dim,
    )
    algorithm = MetaSoftActorCritic(
        env_sampler=env_sampler,
        policy=policy,
        qf=qf,
        vf=vf,
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
