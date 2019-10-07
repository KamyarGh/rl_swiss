import numpy as np
from gym.envs.mujoco import HalfCheetahEnv, InvertedPendulumEnv, ReacherEnv
from gym.spaces import Dict

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger, build_nested_variant_generator
from rlkit.torch.sac.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.sac.sac import NewSoftActorCritic
from rlkit.torch.networks import FlattenMlp

from rlkit.envs import get_env

import yaml
import argparse
import importlib
import psutil
import os
import argparse
import joblib


def experiment(variant):
    # env = NormalizedBoxEnv(HalfCheetahEnv())
    # env = NormalizedBoxEnv(InvertedPendulumEnv())
    # ---------
    # env = NormalizedBoxEnv(get_meta_env(variant['env_specs']))
    # training_env = NormalizedBoxEnv(get_meta_env(variant['env_specs']))

    # env = ReacherEnv()
    # training_env = ReacherEnv()

    # env = NormalizedBoxEnv(ReacherEnv())
    # training_env = NormalizedBoxEnv(ReacherEnv())
    
    # Or for a specific version:
    # import gym
    # env = NormalizedBoxEnv(gym.make('HalfCheetah-v1'))

    # we have to generate the combinations for the env_specs
    env_specs = variant['env_specs']
    env, _ = get_env(env_specs)
    training_env, _ = get_env(env_specs)

    print(env.observation_space)

    obs_space = env.observation_space
    if isinstance(env.observation_space, Dict):
        # possible keys: pixel, obs, obs_task_params
        if not variant['algo_params']['policy_uses_pixels']:
            obs_dim = int(np.prod(obs_space.spaces['obs'].shape))
        else:
            raise NotImplementedError()

        if variant['algo_params']['policy_uses_task_params']:
            if variant['algo_params']['concat_task_params_to_policy_obs']:
                obs_dim += int(np.prod(obs_space.spaces['obs_task_params'].shape))
            else:
                raise NotImplementedError
    else:
        # OpenAI Gym Env or DMCS Env with only one obs
        obs_dim = int(np.prod(env.observation_space.shape))
    
    action_dim = int(np.prod(env.action_space.shape))


    # if variant['reload_policy_from'] != '':
        # params = joblib.load(variant['reload_policy_from'])
        # qf1, qf2, vf, policy = params['qf1'], params['qf2'], params['vf'], params['policy']
    # else:
    net_size = variant['net_size']
    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim,
        output_size=1,
    )
    policy = ReparamTanhMultivariateGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    
    algorithm = NewSoftActorCritic(
        env=env,
        training_env=training_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()

    return 1


def exp_fn(variant):
    exp_id = variant['exp_id']
    # affinity_Q = arg[2]

    # # set the affinity
    # affinity = affinity_Q.get()
    # psutil.Process().cpu_affinity(affinity)
    # print('Affinity set to {}\n'.format(psutil.Process().cpu_affinity()))

    # os.system("taskset -p -c 0,1 %d" % os.getpid())

    print(variant.keys())
    exp_prefix = variant['exp_name']
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=variant)

    # run the experiment
    exp_return = experiment(variant)
    
    # # release the affinity
    # affinity_Q.put(psutil.Process().cpu_affinity())
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
