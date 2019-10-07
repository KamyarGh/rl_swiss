import joblib
from collections import defaultdict
from os import path as osp

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# expert_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/test/test_2019_03_29_00_19_09_0000--s-0'
# expert_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/hc-2-layer-vel-check/hc_2_layer_vel_check_2019_04_05_15_22_14_0000--s-0/'
# expert_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/hc-actual-2-layer-vel-check/hc_actual_2_layer_vel_check_2019_04_05_15_24_46_0000--s-0/'
# expert_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/hc-3-layer-rew-4-vel-check/hc_3_layer_rew_4_vel_check_2019_04_05_15_31_06_0000--s-0/'
# expert_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/hc-2-layer-rew-80-vel-check/hc_2_layer_rew_80_vel_check_2019_04_05_15_38_45_0000--s-0/'
# expert_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/test-stochastic/test_stochastic_2019_04_05_16_32_09_0000--s-0/'
# expert_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/test-200-limit/test_200_limit_2019_04_07_17_15_22_0000--s-0/'
# expert_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/test-no-ctrl-cost-1/test_no_ctrl_cost_1_2019_04_07_21_53_19_0000--s-0/'
# expert_path = '/scratch/hdd001/home/kamyar/output/test-meta-gen-1/test_meta_gen_1_2019_04_13_19_37_13_0000--s-0'
expert_path = '/scratch/hdd001/home/kamyar/output/test-meta-gen-2-stochastic/test_meta_gen_2_stochastic_2019_04_13_19_47_16_0000--s-0'

rb = joblib.load(osp.join(expert_path, 'extra_data.pkl'))['meta_train']['context']
X, means, stds = [], [], []
exp_dist = []
for t in sorted(list(rb.task_replay_buffers.keys())):
    print(t)
    s = rb.task_replay_buffers[t]
    # -----------
    splits = np.split(s._rewards[:s._size], 4)
    # ret_means = np.mean(list(map(lambda p: np.sum(p[:50]), splits)))
    # ret_stds = np.std(list(map(lambda p: np.sum(p[:50]), splits)))
    ret_means = np.mean(list(map(lambda p: np.mean(p[:50]), splits)))
    ret_stds = np.mean(list(map(lambda p: np.std(p[:50]), splits)))
    # -----------
    ret_means = np.mean(s._rewards[:s._size])
    ret_stds = np.std(s._rewards[:s._size])
    exp_dist.append(np.abs(s._rewards[:s._size] - t))
    # -----------
    X.append(t)
    means.append(ret_means)
    stds.append(ret_stds)
X = np.array(X)
means = np.array(means)
stds = np.array(stds)

print(np.mean(np.abs(means - X)))
exp_dist = np.concatenate(exp_dist)
print(np.mean(exp_dist))
print(np.std(exp_dist))

fig, ax = plt.subplots(1)
ax.plot(X, means)
ax.plot(X, means + stds)
ax.plot(X, means - stds)
# avg = np.mean(means)
# ax.plot(X, [avg for _ in range(len(means))])
ax.plot(X,X)
ax.set_ylim([-0.1,3.1])
# plt.savefig('plots/junk_vis/3_layer_vels_for_hc_rand_vel_expert_rew_1.png', bbox_inches='tight', dpi=300)
# plt.savefig('plots/junk_vis/2_layer_vels_for_hc_rand_vel_expert.png', bbox_inches='tight', dpi=300)
# plt.savefig('plots/junk_vis/3_layer_vels_for_hc_rand_vel_expert_rew_4.png', bbox_inches='tight', dpi=300)
# plt.savefig('plots/junk_vis/2_layer_vels_for_hc_rand_vel_expert_rew_80.png', bbox_inches='tight', dpi=300)
# plt.savefig('plots/junk_vis/3_layer_vels_for_hc_rand_vel_expert_rew_4_stochastic.png', bbox_inches='tight', dpi=300)
# plt.savefig('plots/junk_vis/3_layer_vels_for_hc_rand_vel_expert_rew_4_det_200_limit.png', bbox_inches='tight', dpi=300)
# plt.savefig('plots/junk_vis/3_layer_vels_for_hc_rand_vel_expert_rew_4_det_no_ctrl.png', bbox_inches='tight', dpi=300)
# plt.savefig('plots/junk_vis/hc_rand_vel_expert_rew_scale_150_3_layer_policy.png', bbox_inches='tight', dpi=300)
plt.savefig('plots/junk_vis/hc_rand_vel_expert_rew_scale_150_3_layer_policy_stochastic.png', bbox_inches='tight', dpi=300)
plt.close()
