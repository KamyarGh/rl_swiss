'''
The normalization I'm using here is different than the one for the meta version
'''
import numpy as np
import joblib
import yaml
import os
from os import path as osp

from rlkit.core.vistools import plot_histogram

EXPERT_LISTING_YAML_PATH = '/h/kamyar/oorl_rlkit/rlkit/torch/irl/experts.yaml'

NORMALIZE_OBS = True
NORMALIZE_ACTS = False

def get_stats(data):
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    # check for a pathology where some axes are constant
    std = np.where(std == 0, np.ones(std.shape), std)
    return mean, std


def do_the_thing(data_path, save_path, plot_obs_histogram=False):
    # normalization values are computed from the meta_train set
    d = joblib.load(data_path)
    d['obs_mean'] = None
    d['obs_std'] = None
    d['acts_mean'] = None
    d['acts_std'] = None
    if NORMALIZE_OBS:
        # print(d['train']._size)
        # print(d['train']._top)
        # print(np.max(d['train']._observations[:d['train']._size]))
        # print(np.min(d['train']._observations[:d['train']._size]))
        # print(d['train']._observations.shape)

        # first concatenate all the obs from all the tasks
        print(d.keys())
        con_rb = d['meta_train']['context']
        all_obs = np.concatenate(
            [rb._observations['obs'][:rb._size] for rb in con_rb.task_replay_buffers.values()]
            +
            [rb._next_obs['obs'][:rb._size] for rb in con_rb.task_replay_buffers.values()]
        )
        mean, std = get_stats(all_obs)
        print(all_obs.shape)
        print('\nBefore -----------------------')
        print('Mean:')
        print(mean)
        print('Std:')
        print(std)
        print('Max:')
        print(np.max(all_obs, axis=0))
        print('Min:')
        print(np.min(all_obs, axis=0))

        meta_rbs_to_normalize = [
            d['meta_train']['context'],
            d['meta_train']['test'],
            d['meta_test']['context'],
            d['meta_test']['test']
        ]
        for meta_rb in meta_rbs_to_normalize:
            for task_rb in meta_rb.task_replay_buffers.values():
                task_rb._observations['obs'] = (task_rb._observations['obs'] - mean) / std
                task_rb._next_obs['obs'] = (task_rb._next_obs['obs'] - mean) / std
        d['obs_mean'] = mean
        d['obs_std'] = std
        # ---------
        con_rb = d['meta_train']['context']
        all_obs = np.concatenate(
            [rb._observations['obs'][:rb._size] for rb in con_rb.task_replay_buffers.values()]
            +
            [rb._next_obs['obs'][:rb._size] for rb in con_rb.task_replay_buffers.values()]
        )
        mean, std = get_stats(all_obs)
        print(all_obs.shape)
        print('\nAfter -----------------------')
        print('Mean:')
        print(mean)
        print('Std:')
        print(std)
        print('Max:')
        print(np.max(all_obs, axis=0))
        print('Min:')
        print(np.min(all_obs, axis=0))
    if NORMALIZE_ACTS:
        raise NotImplementedError('Must take into account d[\'train\']._size')
        # d['train']._actions, mean, std = get_normalized(d['train']._actions, return_stats=True)
        # d['test']._actions = get_normalized(d['test']._actions, mean=mean, std=std)
        # d['acts_mean'] = mean
        # d['acts_std'] = std
        # print('\nActions:')
        # print('Mean:')
        # print(mean)
        # print('Std:')
        # print(std)

    print(save_path)
    joblib.dump(d, osp.join(save_path), compress=3)


# data_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/correct-halfcheetah-demos-250-no-subsampling/correct_halfcheetah_demos_250_no_subsampling_2019_02_16_18_01_27_0000--s-0/extra_data.pkl'
# save_path = '/scratch/gobi2/kamyar/oorl_rlkit/expert_demos/norm_HC_250_demos_no_subsampling'
with open(EXPERT_LISTING_YAML_PATH, 'r') as f:
    listings = yaml.load(f.read())

for i, expert in enumerate([
    # 'halfcheetah_rand_vel_expert',
    # 'deterministic_hc_rand_vel_expert_demos_0p125_separated_64_demos_sub_20'
    # 'hc_rand_vel_expert_demos_0p125_separated_16_demos_sub_20'
    # 'hc_rand_vel_expert_demos_0p1_separated_16_demos_sub_20'
    # 'hc_rand_vel_expert_demos_0p1_separated_64_demos_sub_20'
    # 'hc_rand_vel_expert_demos_0p1_separated_64_demos_sub_1'
    # 'hc_rand_vel_expert_demos_0p1_separated_64_demos_sub_1_only_task_1'
    # 'hc_rand_vel_expert_demos_0p1_separated_256_demos_sub_1_only_task_1'
    # 'hc_rand_vel_expert_debug_1_to_2_values'
    # 'hc_rand_vel_expert_demos_0p1_separated_4_demos_sub_20'

    # ANT
    # 'ant_five_points_64_demos_sub_1'
    # 'ant_two_points_64_demos_sub_1'
    # 'ant_opposite_points_64_demos_sub_1'
    # 'ant_eight_points_64_demos_sub_1'
    # 'ant_sixteen_points_16_demos_sub_1'
    # 'ant_32_points_16_demos_sub_1'
    # 'ant_32_points_4_demos_sub_1'
    # 'ant_32_points_64_demos_sub_1'
    # 'ant_test_tasks_for_32_points'

    # 'ant_lin_class_64_tasks_16_demos_each_sub_1'
    # 'rel_pos_ant_lin_class_64_tasks_16_demos_each_sub_1'

    # 'fetch_linear_classification_demos_64_tasks_16_demos_per_task_no_sub'

    # 'walker_meta_dyn_32_det_demos_per_task_20_sub'
    # 'test_tasks_walker_meta_dyn_32_det_demos_per_task_20_sub'

    'hc_rand_vel_bc_debugging_2_to_3'
  ]):
  data_path = osp.join(listings[expert]['exp_dir'], listings[expert]['seed_runs'][0], 'extra_data.pkl')
  save_dir = '/scratch/hdd001/home/kamyar/expert_demos/norm_'+expert
  os.makedirs(save_dir, exist_ok=True)
  save_path = osp.join(save_dir, 'extra_data.pkl')
  do_the_thing(data_path, save_path, False)
