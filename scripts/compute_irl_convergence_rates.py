import yaml
import json
import joblib
import argparse
import os
import os.path as osp
import time
from collections import defaultdict
import numpy as np
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.samplers.in_place import InPlacePathSampler
from rlkit.core.eval_util import get_average_returns

N_ROLLOUTS = 50
ENV_SEED = 12344321
EVAL_DETERMINISTIC = True
STD = 5.0 # std for smoothing
# TARGETS = [3000, 4000, 5000, 6000, 7000, 8000]
TARGETS = [3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000]

def find_firsts(curve):
    curve = gaussian_filter1d(curve, STD)

    # fig, ax = plt.subplots(1)
    # ax.plot(curve)
    # ax.plot(curve)
    # ax.set_ylim([0, 8000])
    # plt.savefig('plots/junk_vis/test_smoothing.png', bbox_inches='tight', dpi=300)
    # plt.close()

    firsts = []
    for target in TARGETS:
        idxs = np.sort(np.asarray(curve > target).nonzero()[0])
        if idxs.size == 0:
            firsts.append(-1)
        else:
            firsts.append(idxs[0])
    return firsts


'''
{
    det or non det: {
        expert_name: {
            forward or reverse: [
                first_list for each seed
            ]
        }
    }
}
'''
def dd1():
    return defaultdict(list)
def dd0():
    return defaultdict(dd1)
values = defaultdict(dd0)
if __name__ == '__main__':
    exp_paths = [
        '/scratch/gobi2/kamyar/oorl_rlkit/output/totally-absolutely-final-normalized-halfcheetah-rev-KL-airl-disc-state-action',
        # '/scratch/gobi2/kamyar/oorl_rlkit/output/totally-absolutely-final-normalized-halfcheetah-rev-KL-airl-disc-final-state-only',
        '/scratch/gobi2/kamyar/oorl_rlkit/output/totally-absolutely-final-normalized-halfcheetah-forw-KL-airl-disc',
        # '/scratch/gobi2/kamyar/oorl_rlkit/output/totally-absolutely-final-normalized-halfcheetah-forw-KL-airl-disc-state-only'
    ]

    for exp_path in exp_paths:
        for sub_exp in os.listdir(exp_path):
            try:
                csv_full_path = osp.join(exp_path, sub_exp, 'progress.csv')
                csv_file = np.genfromtxt(csv_full_path, skip_header=0, delimiter=',', names=True)
                non_det_curve = csv_file['Exploration_Returns_Mean']
                det_curve = csv_file['AverageReturn']
                with open(osp.join(exp_path, sub_exp, 'variant.json'), 'r') as f:
                    sub_exp_specs = json.load(f)
            except:
                continue
            
            expert_name = sub_exp_specs['expert_name']
            use_exp_rewards = sub_exp_specs['algo_params']['use_exp_rewards']
            KL_mode = 'forward' if use_exp_rewards else 'reverse'
            rew_scale = sub_exp_specs['policy_params']['reward_scale']

            firsts = find_firsts(det_curve)
            values[True][expert_name][KL_mode].append(firsts)
            firsts = find_firsts(non_det_curve)
            values[False][expert_name][KL_mode].append(firsts)
        
    joblib.dump(
        {
            'all_firsts': values,
        },
        osp.join(exp_path, 'all_firsts.pkl'),
        compress=3
    )
    print(values)

    # Now plot them
    for det in [True, False]:
        for n in [4, 8, 16, 32, 64, 128, 256]:
            expert = 'norm_halfcheetah_%d_demos_20_subsampling' % n
            fig, ax = plt.subplots(1)
            forw_firsts = np.array(values[det][expert]['forward'])
            rev_firsts = np.array(values[det][expert]['reverse'])
            f_means, f_stds = np.zeros(len(TARGETS)), np.zeros(len(TARGETS))
            r_means, r_stds = np.zeros(len(TARGETS)), np.zeros(len(TARGETS))
            for i in range(len(TARGETS)):
                f = forw_firsts[:,i]
                r = rev_firsts[:,i]
                f = np.extract(f != -1, f)
                r = np.extract(r != -1, r)
                f_means[i] = np.mean(f)
                f_stds[i] = np.std(f)
                r_means[i] = np.mean(r)
                r_stds[i] = np.std(r)
            X = TARGETS
            print('\n')
            print(X)
            print(r_means)
            print(r_stds)
            print(f_means)
            print(f_stds)
            ax.plot(X, r_means, color='purple', label='AIRL')
            ax.fill_between(X, r_means+r_stds, r_means-r_stds, facecolor='purple', alpha=0.5)
            ax.plot(X, f_means, color='cyan', label='FAIRL')
            ax.fill_between(X, f_means+f_stds, f_means-f_stds, facecolor='cyan', alpha=0.5)

            ax.set_xlim([TARGETS[0], TARGETS[-1]])
            ax.set_ylim([0, 3000])
            ax.set_xlabel('Return Milestone')
            ax.set_ylabel('Epoch Achieved')

            lgd = ax.legend(loc='lower right', shadow=False)
            ax.set_xticks(TARGETS)
            ax.set_xticklabels(list(map(str, TARGETS)))
            plt.savefig('plots/junk_vis/det{}numdemos{}.png'.format(det, n), bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
            plt.close()
