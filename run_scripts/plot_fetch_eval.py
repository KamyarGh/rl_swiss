import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os.path as osp
import joblib

MAIN_PATH = '/scratch/gobi2/kamyar/oorl_rlkit/output'
WHAT_TO_PLOT = 'faster_all_eval_stats.pkl'
# WHAT_TO_PLOT = 'faster_all_eval_stats.pkl'
# WHAT_TO_PLOT = 'faster_all_eval_stats.pkl'

data_dirs = {
    'np_airl': {
        0.2: 'correct-saving-np-airl-KL-0p2-disc-512-dim-rew-2-NO-TARGET-ANYTHING-over-10-epochs',
        0.15: 'correct-saving-np-airl-KL-0p15-disc-512-dim-rew-2-NO-TARGET-ANYTHING-over-10-epochs',
        0.1: 'correct-saving-np-airl-KL-0p1-disc-512-dim-rew-2-NO-TARGET-ANYTHING-over-10-epochs',
        0.05: 'correct-saving-np-airl-KL-0p05-disc-512-dim-rew-2-NO-TARGET-ANYTHING-over-10-epochs',
        0.0: 'correct-saving-np-airl-KL-0-disc-512-dim-rew-2-NO-TARGET-ANYTHING-over-10-epochs'
    },
    'np_bc': {
        0.2: 'np-bc-KL-0p2-FINAL-WITHOUT-TARGETS',
        0.15: 'np-bc-KL-0p15-FINAL-WITHOUT-TARGETS',
        0.1: 'np-bc-KL-0p1-FINAL-WITHOUT-TARGETS',
        0.05: 'np-bc-KL-0p05-FINAL-WITHOUT-TARGETS',
        0.0: 'np-bc-KL-0-FINAL-WITHOUT-TARGETS'
    }
}

# fig, ax = plt.subplots(1, 5)
for i, beta in enumerate([0.0, 0.05, 0.1, 0.15, 0.2]):
    fig, ax = plt.subplots(1)
    ax.set_xlabel('$\\beta = %.2f$' % beta)

    # np_airl
    all_stats = joblib.load(osp.join(MAIN_PATH, data_dirs['np_airl'][beta], WHAT_TO_PLOT))['faster_all_eval_stats']
    good_reaches_means = []
    good_reaches_stds = []
    solves_means = []
    solves_stds = []
    for c_size in range(1,7):
        good_reaches = []
        solves = []
        for d in all_stats:
            good_reaches.append(d[c_size]['Percent_Good_Reach'])
            solves.append(d[c_size]['Percent_Solved'])
        good_reaches_means.append(np.mean(good_reaches))
        good_reaches_stds.append(np.std(good_reaches))
        solves_means.append(np.mean(solves))
        solves_stds.append(np.std(solves))
    
    # ax.errorbar(list(range(1,7)), good_reaches_means, good_reaches_stds)
    ax.errorbar(np.array(list(range(1,7))) + 0.1, solves_means, solves_stds,
        elinewidth=2.0, capsize=4.0, barsabove=True, linewidth=2.0, label='Meta-AIRL'
    )

    # np_bc
    all_stats = joblib.load(osp.join(MAIN_PATH, data_dirs['np_bc'][beta], WHAT_TO_PLOT))['faster_all_eval_stats']
    good_reaches_means = []
    good_reaches_stds = []
    solves_means = []
    solves_stds = []
    for c_size in range(1,7):
        good_reaches = []
        solves = []
        for d in all_stats:
            good_reaches.append(d[c_size]['Percent_Good_Reach'])
            solves.append(d[c_size]['Percent_Solved'])
        good_reaches_means.append(np.mean(good_reaches))
        good_reaches_stds.append(np.std(good_reaches))
        solves_means.append(np.mean(solves))
        solves_stds.append(np.std(solves))
    
    # ax.errorbar(list(range(1,7)), good_reaches_means, good_reaches_stds)
    ax.errorbar(np.array(list(range(1,7))) - 0.1, solves_means, solves_stds,
        elinewidth=2.0, capsize=4.0, barsabove=True, linewidth=2.0, label='Meta-BC'
    )

    ax.set_ylim([0.3, 1.0])
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.725, 0.1), shadow=False, ncol=3)
    plt.savefig('plots/abc/faster_test_%d.png'%i, bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.savefig('plots/abc/test_%d.png'%i)
    plt.close()
