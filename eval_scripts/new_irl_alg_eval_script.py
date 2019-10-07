import yaml
import json
import joblib
import argparse
import os
import os.path as osp
import time
from collections import defaultdict

import numpy as np

from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.samplers.in_place import InPlacePathSampler
from rlkit.core.eval_util import get_average_returns


# they were 1000 for everything we tried
def eval_alg(alg, num_eval_rollouts, eval_deterministic=True, max_path_length=1000):
    policy = alg.exploration_policy
    policy.cpu()
    env = alg.env
    if eval_deterministic:
        policy = MakeDeterministic(policy)
    
    eval_sampler = InPlacePathSampler(
        env=env,
        policy=policy,
        max_samples=max_path_length * (num_eval_rollouts + 1),
        max_path_length=max_path_length,
        policy_uses_pixels=False,
        policy_uses_task_params=False,
        concat_task_params_to_policy_obs=False
    )
    test_paths = eval_sampler.obtain_samples()
    average_returns = get_average_returns(test_paths)
    return average_returns


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)
    exp_path = exp_specs['exp_path']

    all_det_returns = defaultdict(list)
    all_stoch_returns = defaultdict(list)

    last_time = time.time()
    print('\n')
    print(exp_path)
    print('\n')
    for sub_exp in os.listdir(exp_path):
        try:
            alg = joblib.load(osp.join(exp_path, sub_exp, 'best_test.pkl'))['algorithm']
            with open(osp.join(exp_path, sub_exp, 'variant.json'), 'r') as f:
                sub_exp_specs = json.load(f)
        except:
            continue
        
        expert_name = sub_exp_specs['expert_name']
        seed = sub_exp_specs['seed']
        if 'policy_params' in sub_exp_specs:
            rew_scale = sub_exp_specs['policy_params']['reward_scale']
            gp = sub_exp_specs['algo_params']['grad_pen_weight']
            use_exp_rewards = sub_exp_specs['algo_params']['use_exp_rewards']
        else:
            rew_scale = -1.0
            gp = -1.0
            use_exp_rewards = 0.0

        average_returns = eval_alg(
            alg,
            exp_specs['num_eval_rollouts'],
            eval_deterministic=True,
            max_path_length=exp_specs['max_path_length']
        )
        all_det_returns[(expert_name, rew_scale, gp, seed, use_exp_rewards)].append(average_returns)

        average_returns = eval_alg(
            alg,
            exp_specs['num_eval_rollouts'],
            eval_deterministic=False,
            max_path_length=exp_specs['max_path_length']
        )
        all_stoch_returns[(expert_name, rew_scale, gp, seed, use_exp_rewards)].append(average_returns)

        new_time = time.time()
        print(new_time - last_time)
        last_time = new_time
    
    joblib.dump(
        {
            'all_det_returns': all_det_returns,
            'all_stoch_returns': all_stoch_returns
        },
        osp.join(exp_path, 'all_eval_stats.pkl'),
        compress=3
    )
