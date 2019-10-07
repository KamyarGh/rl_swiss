import yaml
import json
import joblib
import argparse
import os
import os.path as osp
import time
from collections import defaultdict

import numpy as np

from gym.envs.mujoco.ant import AntEnv
from rlkit.envs.wrappers import ScaledEnv

from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.samplers.in_place import InPlacePathSampler
from rlkit.core.eval_util import get_average_returns

def eval_alg(policy, env, num_eval_rollouts, eval_deterministic=False, max_path_length=1000):
    if eval_deterministic:
        policy = MakeDeterministic(policy)
    
    eval_sampler = InPlacePathSampler(
        env=env,
        policy=policy,
        max_samples=max_path_length * (num_eval_rollouts + 1),
        max_path_length=max_path_length, policy_uses_pixels=False,
        policy_uses_task_params=False,
        concat_task_params_to_policy_obs=False
    )
    test_paths = eval_sampler.obtain_samples()
    average_returns = get_average_returns(test_paths)
    return average_returns


if __name__ == '__main__':
    EVAL_DETERMINISTIC = False
    N_ROLLOUTS = 10
    print('EVAL_DETERMINISTIC: {}\n'.format(EVAL_DETERMINISTIC))

    # exp_path = '/scratch/hdd001/home/kamyar/output/super-hype-search-fairl-ant-4-demos'
    exp_path = '/scratch/hdd001/home/kamyar/output/super-hype-search-airl-ant-32-demos/'
    print(exp_path)
    
    env = AntEnv()
    extra_data = joblib.load('/scratch/hdd001/home/kamyar/expert_demos/norm_ant_32_demos_20_subsampling/extra_data.pkl')
    env = ScaledEnv(
        env,
        obs_mean=extra_data['obs_mean'],
        obs_std=extra_data['obs_std'],
        acts_mean=extra_data['acts_mean'],
        acts_std=extra_data['acts_std'],
    )
    
    all_returns = defaultdict(list)
    last_time = time.time()
    for sub_exp in os.listdir(exp_path):
        try:
            policy = joblib.load(osp.join(exp_path, sub_exp, 'params.pkl'))['policy']
            with open(osp.join(exp_path, sub_exp, 'variant.json'), 'r') as f:
                sub_exp_specs = json.load(f)
        except:
            continue
        
        rew_scale = sub_exp_specs['policy_params']['reward_scale']
        gp = sub_exp_specs['algo_params']['grad_pen_weight']
        average_returns = eval_alg(
            policy,
            env,
            N_ROLLOUTS,
            EVAL_DETERMINISTIC,
            1000
        )
        all_returns[(rew_scale, gp)].append(average_returns)

        new_time = time.time()
        print('Rew {}, GP {}, Return {}'.format(rew_scale, gp, average_returns))
        print(new_time - last_time)
        last_time = new_time
    
    joblib.dump(
        all_returns,
        osp.join(exp_path, 'deterministic_{}_'.format(EVAL_DETERMINISTIC) + '_all_returns.pkl'),
        compress=3
    )
