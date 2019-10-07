import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os.path as osp
import joblib


def plot_figure(few_shot, prior, label, save_path):
    x_axis = np.arange(1, 9)
    fig, ax = plt.subplots(1)
    # ax.errorbar(list(range(1,7)), good_reaches_means, good_reaches_stds)
    ax.errorbar(x_axis, few_shot['mean'], few_shot['std'],
        elinewidth=2.0, capsize=4.0, barsabove=True, linewidth=2.0, label=label
    )
    ax.plot(x_axis, [prior['mean'][0] for _ in range(8)], label='Prior', linestyle='--', color='magenta')
    ax.set_ylim([0.0, 1.0])
    ax.set_xlim([0.75, 8.25])
    ax.set_xlabel('Context Size')
    ax.set_ylabel('Mean Avg. Performance Across Seeds')
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.725, 0.1), shadow=False, ncol=3)
    plt.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


np_bc_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/final-better-np-bc-new-hype-search-KL-0p01-weight-share-enc-32-dim-pol-dim-64-gate-dim-32-z-dim-16'
np_airl_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/less-epochs-final-better-np-airl-KL-0p01-weight-share-enc-32-dim-pol-dim-64-gate-dim-32-z-dim-16'

np_bc_few_shot = joblib.load(osp.join(np_bc_path, 'formatted_few_shot.pkl'))['solved']
prior_np_bc_few_shot = joblib.load(osp.join(np_bc_path, 'prior_sampled_formatted_few_shot.pkl'))['solved']

np_airl_few_shot = joblib.load(osp.join(np_airl_path, 'formatted_few_shot.pkl'))['solved']
prior_np_airl_few_shot = joblib.load(osp.join(np_airl_path, 'prior_sampled_formatted_few_shot.pkl'))['solved']

plot_figure(
    np_bc_few_shot,
    prior_np_bc_few_shot,
    'Meta-BC',
    'plots/final_plots/npbcfewshot.png'
)
plot_figure(
    np_airl_few_shot,
    prior_np_airl_few_shot,
    'Meta-AIRL',
    'plots/final_plots/npairlfewshot.png'
)
