import os
from os import path as osp
import joblib
import yaml
from collections import defaultdict
import numpy as np

exp_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/absolute-final-bc-on-varying-data-amounts'

'''
results  = {
    lr: {
        expert: {
            net_size: [different seeds]
        }
    }
}
'''
# cause of dumb picklng problems
def dd1():
    return defaultdict(list)
def dd0():
    return defaultdict(dd1)
results = defaultdict(dd0)

for sub_dir in os.listdir(exp_path):
    sub_exp_path = osp.join(exp_path, sub_dir)
    with open(osp.join(sub_exp_path, 'variant.yaml'), 'r') as f:
        variant = yaml.load(f.read())
    lr = variant['algo_params']['policy_lr']
    expert = variant['expert_name']
    net_size = variant['policy_net_size']

    returns = joblib.load(osp.join(sub_exp_path, 'best_test.pkl'))['average_returns']
    results[lr][expert][net_size].append(returns)

print(results)
