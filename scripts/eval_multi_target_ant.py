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
from rlkit.envs.ant_multi_target import AntMultiTargetEnv

MAX_PATH_LENGTH = 100
N_ROLLOUTS = 120
ENV_SEED = 12344321
EVAL_DETERMINISTIC = False


def eval_alg(policy, env, max_path_length, num_eval_rollouts, env_seed, eval_deterministic=False):
    if eval_deterministic:
        policy = MakeDeterministic(policy)
    
    env.seed(env_seed)

    eval_sampler = InPlacePathSampler(
        env=env,
        policy=policy,
        max_samples=max_path_length * (num_eval_rollouts + 1),
        max_path_length=max_path_length, policy_uses_pixels=False,
        policy_uses_task_params=False,
        concat_task_params_to_policy_obs=False
    )
    test_paths = eval_sampler.obtain_samples()
    path_trajs = [np.array([d['xy_pos'] for d in path["env_infos"]]) for path in test_paths]
    return {'path_trajs': path_trajs}


if __name__ == '__main__':
    print('EVAL_DETERMINISTIC: {}\n'.format(EVAL_DETERMINISTIC))

    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-fairl-rew-search/'
    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-airl-rew-search/'

    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-airl-rew-search-32-det-demos-per-task/'
    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-fairl-rew-search-32-det-demos-per-task/'

    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-fairl-rew-search-32-det-demos-per-task-grad-pen-search/'

    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-fairl-rew-search-32-det-demos-per-task-even-lower-grad-pen-search/'
    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-airl-rew-search-32-det-demos-per-task-even-lower-grad-pen-search/'

    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-fairl-32-det-demos-per-task-low-grad-pen-and-high-rew-scale-hype-search-0'

    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-4-directions-fairl-32-det-demos-per-task-hype-search-0-rb-size-3200-correct-final'
    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-4-directions-fairl-32-det-demos-per-task-hype-search-1-rb-size-3200-correct-final'

    # 4 distance
    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-4-directions-4-distance-fairl-32-det-demos-per-task-hype-search-1-rb-size-3200-correct-final'

    # 4 distance rel pos
    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-rel-pos-4-directions-4-distance-fairl-32-det-demos-per-task-hype-search-1-rb-size-3200-correct-final-disc-512-3-relu'
    
    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-rel-pos-4-directions-4-distance-fairl-32-det-demos-per-task-hype-search-1-rb-size-3200-correct-final-disc-512-3-relu-high-rew-search'

    # tiny models
    exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-rel-pos-with-termination-small-models-airl-correct-disc-only-sees-rel-pos'
    
    # env = AntMultiTargetEnv(use_rel_pos_obs=True)
    env = AntMultiTargetEnv(use_rel_pos_obs=True, terminate_near_target=True)

    all_eval_dicts = {}
    last_time = time.time()
    for sub_exp in os.listdir(exp_path):
        print(sub_exp)
        try:
            policy = joblib.load(osp.join(exp_path, sub_exp, 'params.pkl'))['training_policy']
        except:
            print('FAILED ON %s' % sub_exp)
            continue
        
        eval_dict = eval_alg(
            policy,
            env,
            MAX_PATH_LENGTH,
            N_ROLLOUTS,
            ENV_SEED,
            EVAL_DETERMINISTIC
        )
        all_eval_dicts[sub_exp] = eval_dict

        new_time = time.time()
        print(new_time - last_time)
        last_time = new_time
    
    joblib.dump(
        all_eval_dicts,
        osp.join(exp_path, 'det_eval_{}'.format(EVAL_DETERMINISTIC) + '.pkl'),
        compress=3
    )
