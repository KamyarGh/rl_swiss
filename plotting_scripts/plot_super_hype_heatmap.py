'''
check the heatmap for ant multi
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

rews = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
gps = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
rew_ind = {}
for i in range(len(rews)):
    rew_ind[rews[i]] = i
gp_ind = {}
for i in range(len(gps)):
    gp_ind[gps[i]] = i


def make_heatmap(p, save_path, title):
    d = joblib.load(p)
    grid = np.zeros((len(rews), len(gps)))
    for k in d:
        grid[rew_ind[k[0]], gp_ind[k[1]]] = d[k][0]
    
    # fig, ax = plt.subplots()
    # im = ax.imshow(grid)
    # ax.set_xticks(np.arange(len(gps)))
    # ax.set_yticks(np.arange(len(rews)))
    # # for i in range(len(rews)):
    # #     for j in range(len(gps)):
    # #         text = ax.text(j, i, grid[i, j], ha="center", va="center", color="w")
    # # ax.set_title(p)
    # ax.set_title(title)
    # ax.set_xticklabels(gps)
    # ax.set_yticklabels(rews)
    # ax.set_xlabel('Gradient Penalty')
    # ax.set_ylabel('Reward Scale')
    # fig.tight_layout()
    # plt.savefig(save_path)
    # plt.close()

    ax = sns.heatmap(grid, vmin=0, vmax=6500, cmap="YlGnBu")
    ax.set(xlabel='Gradient Penalty', ylabel='Reward Scale', xticklabels=gps, yticklabels=rews, title=title)
    ax.figure.savefig(save_path)


if __name__ == '__main__':
    save_dir = 'plots/junk_vis/heat_maps_for_super_hype/'

    make_heatmap(
        '/scratch/hdd001/home/kamyar/output/super-hype-search-airl-ant-4-demos/deterministic_False__all_returns.pkl',
        osp.join(save_dir, 'airl_4_demos_stoch.png'),
        'AIRL 4 Demos'
    )
