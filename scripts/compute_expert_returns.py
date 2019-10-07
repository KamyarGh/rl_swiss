import joblib
import numpy as np
import yaml
from os import path as osp

experts = [
# 'norm_halfcheetah_256_demos_20_subsampling',
# 'norm_halfcheetah_128_demos_20_subsampling',
# 'norm_halfcheetah_64_demos_20_subsampling',
# 'norm_halfcheetah_32_demos_20_subsampling',
# 'norm_halfcheetah_16_demos_20_subsampling',
# 'norm_halfcheetah_8_demos_20_subsampling',
# 'norm_halfcheetah_4_demos_20_subsampling',

'norm_ant_256_demos_20_subsampling',
'norm_ant_128_demos_20_subsampling',
'norm_ant_64_demos_20_subsampling',
'norm_ant_32_demos_20_subsampling',
'norm_ant_16_demos_20_subsampling',
'norm_ant_8_demos_20_subsampling',
'norm_ant_4_demos_20_subsampling',
]

EXPERT_LISTING_YAML_PATH = '/h/kamyar/oorl_rlkit/rlkit/torch/irl/experts.yaml'
with open(EXPERT_LISTING_YAML_PATH, 'r') as f:
        listings = yaml.load(f.read())

for expert in experts:
    exp_dir = listings[expert]['exp_dir']
    sub_dir = listings[expert]['seed_runs'][0]
    d = joblib.load(osp.join(exp_dir, sub_dir, 'extra_data.pkl'))
    rb = d['train']
    print('\n-------------')
    print(expert)
    print(rb._top/50)
    print(np.mean(rb._rewards[:rb._top]) * 1000)
    print('\n')
