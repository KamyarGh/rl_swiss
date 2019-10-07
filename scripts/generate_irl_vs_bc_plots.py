import joblib
from collections import defaultdict
from os import path as osp

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEMO_SIZES = [4, 8, 16, 32]
expert_name_format = 'norm_halfcheetah_{}_demos_20_subsampling'
# expert_name_format = 'norm_ant_{}_demos_20_subsampling'

def dd0():
    return defaultdict(list)

def get_stats(d, sizes=None):
    # print(d)
    means, stds = [], []
    s_to_use = sizes if sizes is not None else DEMO_SIZES
    for n in s_to_use:
        # print('\n')
        # print(n)
        # print(d[expert.format(n)])
        # print('----------------------')
        # print(d)
        l = d[expert_name_format.format(n)]
        # print(l)
        # l = sorted(l)[-2:]
        means.append(np.mean(l))
        stds.append(np.std(l))
    # print(means)
    # print(stds)
    return np.array(means), np.array(stds)

irl_dirs = [
    # '/scratch/gobi2/kamyar/oorl_rlkit/output/totally-absolutely-final-normalized-halfcheetah-rev-KL-airl-disc-state-action',
    # '/scratch/gobi2/kamyar/oorl_rlkit/output/totally-absolutely-final-normalized-halfcheetah-rev-KL-airl-disc-final-state-only',
    # '/scratch/gobi2/kamyar/oorl_rlkit/output/totally-absolutely-final-normalized-halfcheetah-forw-KL-airl-disc',
    # '/scratch/gobi2/kamyar/oorl_rlkit/output/totally-absolutely-final-normalized-halfcheetah-forw-KL-airl-disc-state-only',

    # '/scratch/gobi2/kamyar/oorl_rlkit/output/what-matters-ant-rev-KL-with-larger-disc',
    # '/scratch/gobi2/kamyar/oorl_rlkit/output/what-matters-ant-forw-KL-with-larger-disc'

    '/scratch/gobi2/kamyar/oorl_rlkit/output/what-matters-halfcheetah-rev-KL-with-128-disc',
    '/scratch/gobi2/kamyar/oorl_rlkit/output/what-matters-halfcheetah-forw-KL-with-128-disc-50-rew',
]
# bc_dir = '/scratch/gobi2/kamyar/oorl_rlkit/output/totally-workshop-last-call-halfcheetah-bc-forw-KL-true'
# bc_dir = '/scratch/gobi2/kamyar/oorl_rlkit/output/what-matters-ant-BC-model-256'
bc_dir = '/scratch/gobi2/kamyar/oorl_rlkit/output/what-matters-halfcheetah-BC-model-256'

# expert performances are
# ant
# expert_returns = np.array([6276.2, 6334.2, 6241.1, 6249.8, 6241.2, 6245.1, 6217.0])
# halfcheetah
expert_returns = np.array([7451.1, 7713.8, 7674.4, 7599.7, 7598.1, 7568.4, 7612.0])

# S,A
for det in [True, False]:
    fig, ax = plt.subplots(1)

    sa_rev_rets = joblib.load(osp.join(irl_dirs[0], 'deterministic_{}_converged_all_returns.pkl'.format(det)))['all_returns'][8.0]
    sa_forw_rets = joblib.load(osp.join(irl_dirs[1], 'deterministic_{}_converged_all_returns.pkl'.format(det)))['all_returns'][50.0]
    bc_rets = joblib.load(osp.join(bc_dir, 'deterministic_{}_converged_all_returns.pkl'.format(det)))['all_returns'][256]

    means, stds = get_stats(sa_rev_rets)
    # print(means)
    # print(stds)
    ax.plot(np.arange(1,len(DEMO_SIZES)+1), means, color='purple', label='AIRL')
    # ax.fill_between(np.arange(1,len(DEMO_SIZES)+1), means+stds, means-stds, facecolor='purple', alpha=0.5)

    means, stds = get_stats(sa_forw_rets)
    ax.plot(np.arange(1,len(DEMO_SIZES)+1), means, color='cyan', label='FAIRL')
    # ax.fill_between(np.arange(1,len(DEMO_SIZES)+1), means+stds, means-stds, facecolor='cyan', alpha=0.5)

    # means, stds = get_stats(bc_rets)
    # means, stds = get_stats(bc_rets, sizes=[4, 8, 16, 32, 64, 128, 256])
    # ax.plot(np.arange(1,8), means, color='orange', label='BC')
    means, stds = get_stats(bc_rets)
    ax.plot(np.arange(1,len(DEMO_SIZES)+1), means, color='orange', label='BC')
    # ax.fill_between(np.arange(1,8,1), means+stds, means-stds, facecolor='orange', alpha=0.5)

    # ax.plot(np.arange(1,8,1), expert_returns, color='green', label='Expert', linestyle='dashed')
    ax.plot(np.arange(1,len(DEMO_SIZES)+1), expert_returns[:len(DEMO_SIZES)], color='green', label='Expert', linestyle='dashed')

    # ax.set_xlim([1,7])
    ax.set_xlim([1,4])
    ax.set_ylim([0, 8500])
    ax.set_xlabel('Number of Demonstration Trajectories')
    ax.set_ylabel('Return')
    # plt.axhline(0, color='grey')
    # plt.axvline(0, color='grey')
    lgd = ax.legend(loc='lower right', shadow=False)
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(['4', '8', '16', '32'])
    # ax.set_xticks(range(1, 8))
    # ax.set_xticklabels(['4', '8', '16', '32', '64', '128', '256'])
    plt.savefig('plots/junk_vis/halfcheetah_det{}saplot.png'.format(det), bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
    plt.close()

# # S
# fig, ax = plt.subplots(1)

# means, stds = get_stats(s_rev_rets[8.0])
# print(means)
# print(stds)
# ax.plot(np.arange(1,8,1), means, color='purple', label='AIRL 8.0')
# ax.fill_between(np.arange(1,8,1), means+stds, means-stds, facecolor='purple', alpha=0.5)

# # means, stds = get_stats(s_rev_rets[12.0])
# # ax.plot(np.arange(1,8,1), means, color='green', label='AIRL 12.0')
# # ax.fill_between(np.arange(1,8,1), means+stds, means-stds, facecolor='green', alpha=0.5)

# means, stds = get_stats(s_forw_rets[50.0])
# ax.plot(np.arange(1,8,1), means, color='cyan', label='Forw KL')
# ax.fill_between(np.arange(1,8,1), means+stds, means-stds, facecolor='cyan', alpha=0.5)

# ax.plot(np.arange(1,8,1), expert_returns, color='green', label='Expert', linestyle='dashed')



# ax.set_xlim([1,7])
# ax.set_ylim([0, 8000])
# ax.set_xlabel('Number of Demonstration Trajectories')
# ax.set_ylabel('Return')
# # plt.axhline(0, color='grey')
# # plt.axvline(0, color='grey')
# lgd = ax.legend(loc='lower right', shadow=False)
# ax.set_xticklabels(['4', '8', '16', '32', '64', '128', '256'])
# plt.savefig('plots/junk_vis/s_plot.png', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
# plt.close()
