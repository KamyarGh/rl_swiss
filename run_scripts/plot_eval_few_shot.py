import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os.path as osp
import joblib

MAIN_PATH = '/scratch/gobi2/kamyar/oorl_rlkit/output'
WHAT_TO_PLOT = 'all_few_shot_eval_stats.pkl'
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
# ax.set_xlabel('$\\beta = %.2f$' % beta)

all_solves_means_np_airl = []
all_solves_stds_np_airl = []
all_solves_means_np_bc = []
all_solves_stds_np_bc = []
for i, beta in enumerate([0.0, 0.05, 0.1, 0.15, 0.2]):
    # np_airl
    all_stats = joblib.load(osp.join(MAIN_PATH, data_dirs['np_airl'][beta], WHAT_TO_PLOT))['all_few_shot_eval_stats']
    all_solves = []
    for d in all_stats:
        all_solves.extend(d['algorithm_all_percent_solved'])
    all_solves_means_np_airl.append(np.mean(all_solves))
    all_solves_stds_np_airl.append(np.std(all_solves))

    # np_bc
    all_stats = joblib.load(osp.join(MAIN_PATH, data_dirs['np_bc'][beta], WHAT_TO_PLOT))['all_few_shot_eval_stats']
    all_solves = []
    for d in all_stats:
        all_solves.extend(d['algorithm_all_percent_solved'])
    all_solves_means_np_bc.append(np.mean(all_solves))
    all_solves_stds_np_bc.append(np.std(all_solves))


fig, ax = plt.subplots(1)
# ax.errorbar(list(range(1,7)), good_reaches_means, good_reaches_stds)
ax.errorbar(np.array([0.0, 0.05, 0.1, 0.15, 0.2]) + 0.01, all_solves_means_np_airl, all_solves_stds_np_airl,
    elinewidth=2.0, capsize=4.0, barsabove=True, linewidth=2.0, label='Meta-AIRL'
)
# ax.errorbar(list(range(1,7)), good_reaches_means, good_reaches_stds)
ax.errorbar(np.array([0.0, 0.05, 0.1, 0.15, 0.2]) - 0.01, all_solves_means_np_bc, all_solves_stds_np_bc,
    elinewidth=2.0, capsize=4.0, barsabove=True, linewidth=2.0, label='Meta-BC'
)
ax.set_ylim([0.0, 1.0])
lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.725, 0.1), shadow=False, ncol=3)
plt.savefig('plots/abc/few_shot_test.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
# plt.savefig('plots/abc/test_%d.png'%i)
plt.close()
