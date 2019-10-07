import numpy as np
from gym.envs.mujoco import HalfCheetahEnv, InvertedPendulumEnv, ReacherEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger, build_nested_variant_generator
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.networks import FlattenMlp

from rlkit.envs import get_meta_env

import yaml
import argparse
import importlib
import psutil
import os


def experiment(variant):
    # env = NormalizedBoxEnv(HalfCheetahEnv())
    # env = NormalizedBoxEnv(InvertedPendulumEnv())
    # ---------
    # env = NormalizedBoxEnv(get_meta_env(variant['env_specs']))
    # training_env = NormalizedBoxEnv(get_meta_env(variant['env_specs']))

    env = ReacherEnv()
    training_env = ReacherEnv()
    
    # Or for a specific version:
    # import gym
    # env = NormalizedBoxEnv(gym.make('HalfCheetah-v1'))

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    net_size = variant['net_size']
    qf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    algorithm = SoftActorCritic(
        env=env,
        training_env=training_env,
        policy=policy,
        qf=qf,
        vf=vf,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()

    return 1


def exp_fn(arg):
    variant = arg[0]
    exp_id = arg[1]
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
