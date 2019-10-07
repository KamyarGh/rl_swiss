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

N_ROLLOUTS = 50
ENV_SEED = 12344321
EVAL_DETERMINISTIC = False

'''
results  = {
    rew_scale: {
        expert: []
    }
}
'''
def dd0():
    return defaultdict(list)

def eval_alg(alg, num_eval_rollouts, env_seed, eval_deterministic=False):
    policy = alg.exploration_policy
    if eval_deterministic:
        policy = MakeDeterministic(policy)
    
    env = alg.env
    env.seed(env_seed)

    eval_sampler = InPlacePathSampler(
        env=env,
        policy=policy,
        max_samples=alg.max_path_length * (num_eval_rollouts + 1),
        max_path_length=alg.max_path_length, policy_uses_pixels=alg.policy_uses_pixels,
        policy_uses_task_params=alg.policy_uses_task_params,
        concat_task_params_to_policy_obs=alg.concat_task_params_to_policy_obs
    )
    test_paths = eval_sampler.obtain_samples()
    average_returns = get_average_returns(test_paths)
    return average_returns


if __name__ == '__main__':
    print('EVAL_DETERMINISTIC: {}\n'.format(EVAL_DETERMINISTIC))
    # # Arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-e', '--experiment', help='experiment specification file')
    # args = parser.parse_args()
    # with open(args.experiment, 'r') as spec_file:
    #     spec_string = spec_file.read()
    #     eval_specs = yaml.load(spec_string)
    # exp_path = eval_specs['exp_dir']

    # exp_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/totally-absolutely-final-normalized-halfcheetah-rev-KL-airl-disc-state-action'
    # exp_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/totally-absolutely-final-normalized-halfcheetah-rev-KL-airl-disc-final-state-only'
    # exp_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/totally-absolutely-final-normalized-halfcheetah-forw-KL-airl-disc'
    # exp_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/totally-absolutely-final-normalized-halfcheetah-forw-KL-airl-disc-state-only'
    # exp_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/normalized-finally-totally-absolute-final-bc-on-varying-data-amounts'
    # exp_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/totally-workshop-last-call-halfcheetah-bc-forw-KL-true'

    # exp_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/what-matters-ant-forw-KL-with-larger-disc'
    # exp_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/what-matters-ant-rev-KL-with-larger-disc'
    # exp_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/what-matters-ant-BC-model-256'

    # exp_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/what-matters-halfcheetah-BC-model-256'
    # exp_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/what-matters-halfcheetah-forw-KL-with-128-disc-50-rew'
    # exp_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/what-matters-halfcheetah-forw-KL-with-128-disc'


    # exp_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/what-matters-halfcheetah-rev-KL-with-128-disc'


    all_returns = defaultdict(dd0)
    last_time = time.time()
    for sub_exp in os.listdir(exp_path):
        try:
            d = joblib.load(osp.join(exp_path, sub_exp, 'best_test.pkl'))
            with open(osp.join(exp_path, sub_exp, 'variant.json'), 'r') as f:
                sub_exp_specs = json.load(f)
        except:
            continue
        
        expert_name = sub_exp_specs['expert_name']
        net_size = sub_exp_specs['policy_net_size']
        # use_exp_rewards = sub_exp_specs['algo_params']['use_exp_rewards']
        rew_scale = sub_exp_specs['policy_params']['reward_scale']
        # print('Evaluating forward KL {} expert {} rew_scale {}...'.format(use_exp_rewards, expert_name, rew_scale))
        average_returns = eval_alg(
            d['algorithm'],
            N_ROLLOUTS,
            ENV_SEED,
            EVAL_DETERMINISTIC
        )
        all_returns[rew_scale][expert_name].append(average_returns)
        # all_returns[net_size][expert_name].append(average_returns)

        new_time = time.time()
        print(new_time - last_time)
        last_time = new_time
    
    joblib.dump(
        {
            'all_returns': all_returns,
            # 'forward_KL_mode': use_exp_rewards
        },
        osp.join(exp_path, 'deterministic_{}_'.format(EVAL_DETERMINISTIC) + 'converged_all_returns.pkl'),
        compress=3
    )
