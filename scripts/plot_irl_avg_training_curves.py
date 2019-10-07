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

DEMO_SIZES = [4, 8, 16, 32]
expert_format = 'norm_halfcheetah_%d_demos_20_subsampling'
# expert_format = 'norm_ant_%d_demos_20_subsampling'

def dd0():
    return defaultdict(list)

def get_curves(exp_path):
    curves_dict = defaultdict(dd0)
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
        curves_dict['det'][expert_name].append(det_curve)
        curves_dict['non_det'][expert_name].append(non_det_curve)
    
    return curves_dict


def get_mean_curves(curves):
    mean_curves = defaultdict(dict)
    for det in curves:
        for exp in curves[det]:
            min_len = min(map(lambda x: x.shape[0], curves[det][exp]))
            mean = np.mean(np.array([c[:min_len] for c in curves[det][exp]]), axis=0)
            mean_curves[det][exp] = mean
    return mean_curves


if __name__ == '__main__':
    # f_KL_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/what-matters-ant-forw-KL-with-larger-disc'
    # r_KL_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/what-matters-ant-rev-KL-with-larger-disc'
    
    # f_KL_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/what-matters-halfcheetah-forw-KL-with-128-disc-50-rew'
    # r_KL_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/what-matters-halfcheetah-rev-KL-with-128-disc'

    gail_path = '/scratch/hdd001/home/kamyar/output/halfcheetah-gail-rew-4'
    f_KL_path = '/scratch/hdd001/home/kamyar/output/what-matters-halfcheetah-forw-KL-with-128-disc-50-rew'
    r_KL_path = '/scratch/hdd001/home/kamyar/output/what-matters-halfcheetah-rev-KL-with-128-disc'

    gail_curves = get_curves(gail_path)
    f_curves = get_curves(f_KL_path)
    r_curves = get_curves(r_KL_path)

    gail_means = get_mean_curves(gail_curves)
    f_means = get_mean_curves(f_curves)
    r_means = get_mean_curves(r_curves)

    # Now plot them
    for det in ['det', 'non_det']:
        for n in DEMO_SIZES:
            expert = expert_format % n

            fig, ax = plt.subplots(1)
            
            # ax.plot(r_means[det][expert], color='purple', label='AIRL')
            ax.plot(r_curves[det][expert][0], color='purple', label='AIRL')
            ax.plot(r_curves[det][expert][1], color='purple', label='AIRL')
            # ax.fill_between(X, r_means+r_stds, r_means-r_stds, facecolor='purple', alpha=0.5)

            # ax.plot(gail_means[det][expert], color='salmon', label='GAIL')
            ax.plot(gail_curves[det][expert][0], color='salmon', label='GAIL')
            ax.plot(gail_curves[det][expert][1], color='salmon', label='GAIL')

            # ax.plot(f_means[det][expert], color='cyan', label='FAIRL')
            ax.plot(f_curves[det][expert][0], color='cyan', label='FAIRL')
            ax.plot(f_curves[det][expert][1], color='cyan', label='FAIRL')
            # ax.fill_between(X, f_means+f_stds, f_means-f_stds, facecolor='cyan', alpha=0.5)

            ax.set_xlim([0, 150])
            ax.set_ylim([0, 8500])
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Average Return')

            lgd = ax.legend(loc='lower right', shadow=False)
            plt.savefig('plots/junk_vis/halfcheetah_curves_det{}numdemos{}.png'.format(det, n), bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
            plt.close()
