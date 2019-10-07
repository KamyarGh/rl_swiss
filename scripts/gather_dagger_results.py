import os
from os import path as osp
import joblib
import numpy as np

base_path = '/scratch/hdd001/home/kamyar/output'
for exp in ['dagger-halfcheetah', 'dagger-ant', 'dagger-hopper', 'dagger-walker']:
    rets = {}
    exp_path = osp.join(base_path, exp)
    rets['exp'] = []
    rets['test'] = []
    for sub in os.listdir(exp_path):
        sub_path = osp.join(exp_path, sub)
        d = joblib.load(osp.join(sub_path, 'best.pkl'))
        rets['exp'].append(
            d['statistics']['Exploration Returns Mean']
        )
        rets['test'].append(
            d['statistics']['Test Returns Mean']
        )
    
    print('\n{}'.format(exp))
    print('{} +/- {}'.format(np.mean(rets['exp']), np.std(rets['exp'])))
    print('{} +/- {}'.format(np.mean(rets['test']), np.std(rets['test'])))
