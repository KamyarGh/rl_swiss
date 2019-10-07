import numpy as np
import joblib
import os
import os.path as osp

import json

from rlkit.torch.networks import WalkerDynamicsAggregateExpert

from rlkit.envs.walker_random_dynamics import _MetaExpertTrainParamsSampler as WalkerTrainParamsSampler
from rlkit.envs.walker_random_dynamics import _MetaExpertTestParamsSampler as WalkerTestParamsSampler

max_path_length = 1000
# exp_dir = '/scratch/hdd001/home/kamyar/output/walker-dynamics-experts-train-correct/'
# save_dir = '/scratch/hdd001/home/kamyar/expert_demos/walker_agg_dyn_expert_for_train_tasks'

exp_dir = '/scratch/hdd001/home/kamyar/output/walker-dynamics-experts-test-correct'
save_dir = '/scratch/hdd001/home/kamyar/expert_demos/walker_agg_dyn_expert_for_test_tasks'

train_sampler = WalkerTrainParamsSampler()
test_sampler = WalkerTestParamsSampler()

e_dict = {}
task_inds = []
for sub_exp in os.listdir(exp_dir):
    full_sub_exp = osp.join(exp_dir, sub_exp)
    if osp.isdir(full_sub_exp):
        with open(osp.join(full_sub_exp, 'variant.json'), 'r') as f:
            d = json.load(f)
        task_idx = d['task_idx']
        task_mode = d['task_mode']
        if task_mode == 'train':
            task_params = train_sampler.get_task(task_idx)
            obs_task_params = train_sampler.get_obs_task_params(task_params)
        elif task_mode == 'test':
            task_params = test_sampler.get_task(task_idx)
            obs_task_params = test_sampler.get_obs_task_params(task_params)
        
        policy = joblib.load(osp.join(full_sub_exp, 'params.pkl'))['policy']
        e_dict[tuple(obs_task_params)] = policy
        task_inds.append(task_idx)

got_all = True
for i in range(25):
    if i not in task_inds:
        got_all = False
        print('Did not get %d'%i)
print('Got all: {}'.format(got_all))

aggregate_walker = WalkerDynamicsAggregateExpert(
    e_dict,
    max_path_length,
    policy_uses_pixels=False,
    policy_uses_task_params=False,
    no_terminal=False
)

os.makedirs(save_dir, exist_ok=True)
joblib.dump({'algorithm': aggregate_walker}, osp.join(save_dir, 'extra_data.pkl'), compress=3)
print(osp.join(save_dir, 'extra_data.pkl'))
