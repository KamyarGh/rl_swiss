import numpy as np
from gym.envs.mujoco import HalfCheetahEnv, InvertedPendulumEnv, ReacherEnv

from rllab.misc.instrument import VariantGenerator
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import NPMetaSoftActorCritic
from rlkit.torch.networks import FlattenMlp

from neural_processes.npv3 import NeuralProcessV3

from rlkit.envs import EnvSampler, OnTheFlyEnvSampler

import yaml
import argparse
import importlib
import psutil
import os
import argparse
import joblib


def experiment(variant):
    # we have to generate the combinations for the env_specs
    if variant['on_the_fly']:
        # we have to generate the combinations for the env_specs
        env_specs = variant['env_specs']
        env_sampler = OnTheFlyEnvSampler(env_specs)
    else:
        env_specs = variant['env_specs']
        env_specs_vg = VariantGenerator()
        env_spec_constants = {}
        env_spec_ranges = {}
        for k, v in env_specs.items():
            if isinstance(v, list):
                env_specs_vg.add(k, v)
                env_spec_ranges[k] = v
            else:
                env_spec_constants[k] = v
        
        env_specs_list = []
        for es in env_specs_vg.variants():
            del es['_hidden_keys']
            es.update(env_spec_constants)
            env_specs_list.append(es)
        
        env_sampler = EnvSampler(env_specs_list)

    # set up the neural process
    np_path = exp_specs['neural_process_load_path']
    if np_path == '':
        raise NotImplementedError()
    else:
        neural_process = joblib.load(np_path)['neural_process']

    # set up similar to non-meta version
    sample_env, _ = env_sampler()
    obs_dim = int(np.prod(sample_env.observation_space.shape))
    action_dim = int(np.prod(sample_env.action_space.shape))

    if variant['algo_params']['latent_repr_mode'] == 'concat_params':
        extra_obs_dim = 2 * neural_process.z_dim
    else: # concat samples
        extra_obs_dim = variant['algo_params']['num_latent_samples'] * neural_process.z_dim

    net_size = variant['net_size']
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + extra_obs_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim + extra_obs_dim,
        action_dim=action_dim,
    )
    qf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim + extra_obs_dim,
        output_size=1,
    )
    algorithm = NPMetaSoftActorCritic(
        env_sampler=env_sampler,
        neural_process=neural_process,
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
