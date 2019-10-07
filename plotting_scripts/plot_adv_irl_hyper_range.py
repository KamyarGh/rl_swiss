'''
check the heatmap for ant airl and fairl hyperparameters
'''
import joblib
import os
from os import path as osp
import numpy as np
from rlkit.core.vistools import plot_2dhistogram

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import seaborn as sns; sns.set()

import json

rews = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
gps = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
rew_ind = {}
for i in range(len(rews)):
    rew_ind[rews[i]] = i
gp_ind = {}
for i in range(len(gps)):
    gp_ind[gps[i]] = i


def make_heatmap(grid, save_path, title):
    ax = sns.heatmap(grid, vmin=0, vmax=6500, cmap="YlGnBu")
    ax.set(xlabel='Gradient Penalty', ylabel='Reward Scale', xticklabels=gps, yticklabels=rews, title=title)
    ax.figure.savefig(save_path)
    plt.close()


def extract_info(exp_path):
    grid = np.zeros((len(rews), len(gps)))
    for d in os.listdir(exp_path):
        sub_path = osp.join(exp_path, d)
        with open(osp.join(sub_path, 'variant.json')) as f:
            json_dict = json.load(f)
            rew_scale = json_dict['sac_params']['reward_scale']
            gp_weight = json_dict['adv_irl_params']['grad_pen_weight']
        test_ret = joblib.load(
            osp.join(sub_path, 'best.pkl')
        )['statistics']['Test Returns Mean']
        grid[rew_ind[rew_scale], gp_ind[gp_weight]] = test_ret
    return grid

if __name__ == '__main__':
    # extract the info for airl
    exp_path = '/scratch/hdd001/home/kamyar/output/airl-ant-hype-search'
    airl_grid = extract_info(exp_path)
    make_heatmap(airl_grid, 'plots/junk_vis/airl_hype_grid.png', '')

    # extract the info for fairl
    exp_path = '/scratch/hdd001/home/kamyar/output/fairl-ant-hype-search'
    fairl_grid = extract_info(exp_path)
    make_heatmap(fairl_grid, 'plots/junk_vis/fairl_hype_grid.png', '')
