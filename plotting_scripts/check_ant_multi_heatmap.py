'''
check the heatmap for ant multi
'''
import joblib
import os
from os import path as osp
import numpy as np
from rlkit.core.vistools import plot_2dhistogram


def plot_expert_heatmap(data_path, target0, num_targets, num_bins, title, save_path, rel_pos_version=True, ax_lims=None):
    assert rel_pos_version
    d = joblib.load(data_path)
    mean, std = d['obs_mean'][:,-2*num_targets:-2*num_targets+2], d['obs_std'][:,-2*num_targets:-2*num_targets+2]
    buffer = d['train']
    print(buffer._size)
    xy_pos = buffer._observations[:buffer._size][:,-2*num_targets:-2*num_targets+2]
    xy_pos = xy_pos * std + mean
    xy_pos = target0 - xy_pos
    plot_2dhistogram(xy_pos[:,0], xy_pos[:,1], num_bins, title, save_path, ax_lims=ax_lims)


if __name__ == '__main__':
    save_dir = 'plots/junk_vis/heat_maps_for_ant_multi/'
    # data_path = '/scratch/hdd001/home/kamyar/expert_demos/norm_rel_pos_obs_ant_multi_4_directions_4_distance_32_det_demos_per_task_no_sub_path_len_75'
    data_path = '/scratch/hdd001/home/kamyar/expert_demos/norm_rel_pos_obs_ant_multi_4_directions_4_distance_32_det_demos_per_task_no_sub_path_terminates_within_0p5_of_target'
    data_path = osp.join(data_path, 'extra_data.pkl')

    plot_expert_heatmap(
        data_path,
        np.array([4.0, 0.0]),
        4,
        40,
        data_path,
        osp.join(save_dir, osp.split(data_path)[-1]+'.png'),
        rel_pos_version=True,
        ax_lims=[[-4.5,4.5],[-4.5,4.5]]
    )
