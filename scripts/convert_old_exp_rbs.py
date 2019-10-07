import os
import os.path as osp
import yaml
import joblib
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer

def copy_over(old, new):
    for i in range(old._size):
        new.add_sample(
            old._observations[i],
            old._actions[i],
            old._rewards[i],
            old._terminals[i],
            old._next_obs[i]
        )

with open('rlkit/torch/irl/old_experts.yaml') as f:
    old_listings = yaml.load(f.read())

demos_to_convert = [
    'norm_halfcheetah_32_demos_20_subsampling',
    'norm_halfcheetah_16_demos_20_subsampling',
    'norm_halfcheetah_4_demos_20_subsampling',

    'norm_ant_32_demos_20_subsampling',
    'norm_ant_16_demos_20_subsampling',
    'norm_ant_4_demos_20_subsampling',
    
    'norm_walker_32_demos_20_subsampling',
    'norm_walker_16_demos_20_subsampling',
    'norm_walker_4_demos_20_subsampling',
    
    'norm_hopper_32_demos_20_subsampling',
    'norm_hopper_16_demos_20_subsampling',
    'norm_hopper_4_demos_20_subsampling'
]

for exp_name in demos_to_convert:
    d = joblib.load(
        osp.join(
            old_listings[exp_name]['exp_dir'],
            old_listings[exp_name]['seed_runs'][0],
            'extra_data.pkl'
        )
    )
    size = d['train']._size
    obs_dim = d['train']._observation_dim
    act_dim = d['train']._action_dim
    
    train_rb = SimpleReplayBuffer(size+1, obs_dim, act_dim)
    copy_over(d['train'], train_rb)
    d['train'] = train_rb

    test_rb = SimpleReplayBuffer(size+1, obs_dim, act_dim)
    copy_over(d['test'], test_rb)
    d['test'] = test_rb

    joblib.dump(
        d,
        osp.join(
            '/scratch/hdd001/home/kamyar/output/fmax_demos',
            exp_name + '.pkl'
        ),
        compress=3
    )
