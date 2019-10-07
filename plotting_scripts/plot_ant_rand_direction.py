import joblib
from os import path as osp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# demo_path = '/scratch/hdd001/home/kamyar/output/check-ant-rand-direc/check_ant_rand_direc_2019_06_30_17_22_41_0000--s-0/extra_data.pkl'
# demo_path = '/scratch/hdd001/home/kamyar/output/check-ant-rand-direc-better-reward/check_ant_rand_direc_better_reward_2019_07_01_10_53_14_0000--s-0/extra_data.pkl'
# demo_path = '/scratch/hdd001/home/kamyar/output/check-ant-rand-direc-better-reward-0/check_ant_rand_direc_better_reward_0_2019_07_01_11_05_46_0000--s-0/extra_data.pkl'
# demo_path = '/scratch/hdd001/home/kamyar/output/check-ant-rand-direc-better-reward-1/check_ant_rand_direc_better_reward_1_2019_07_01_11_14_38_0000--s-0/extra_data.pkl'
# demo_path = '/scratch/hdd001/home/kamyar/output/check-ant-rand-goal-45-to-90/check_ant_rand_goal_45_to_90_2019_07_02_11_40_23_0000--s-0/extra_data.pkl'
# demo_path = '/scratch/hdd001/home/kamyar/output/multi-dir-point-mass-301-demos-ep-len-50/multi_dir_point_mass_301_demos_ep_len_50_2019_07_03_02_33_32_0000--s-0/extra_data.pkl'
# demo_path = '/scratch/hdd001/home/kamyar/output/test-star-point-mass-demos-ep-len-25/test_star_point_mass_demos_ep_len_25_2019_07_06_13_42_45_0000--s-0/extra_data.pkl'
# demo_path = '/scratch/hdd001/home/kamyar/output/test-star-point-mass-demos-ep-len-25/test_star_point_mass_demos_ep_len_25_2019_07_06_13_45_24_0000--s-0/extra_data.pkl'
# demo_path = '/scratch/hdd001/home/kamyar/output/test-star-point-mass-demos-ep-len-25/test_star_point_mass_demos_ep_len_25_2019_07_06_13_46_33_0000--s-0/extra_data.pkl'
demo_path = '/scratch/hdd001/home/kamyar/output/star-8-dir-point-mass-demos-ep-len-25-128-demos/star_8_dir_point_mass_demos_ep_len_25_128_demos_2019_07_06_13_49_29_0000--s-0/extra_data.pkl'
save_path = 'plots/junk_vis/ant_rand_dir_expert.png'

# meta_rb = joblib.load(demo_path)['meta_train']['context']
rb = joblib.load(demo_path)['train']


fig, ax = plt.subplots(1)
# ax.set_ylim([-2.5, 2.5])
ax.set_xlim([-60, 60])
ax.set_ylim([-60, 60])
ax.set_aspect('equal')
ax.set_title('')

# for rb in meta_rb.task_replay_buffers.values():
trajs = rb.sample_trajs(300, keys='observations')
for traj in trajs:
    # xy_path = traj['observations'][:,-2:]
    xy_path = traj['observations']
    plt.plot(xy_path[:,0], xy_path[:,1])

plt.savefig(save_path, bbox_inches='tight')
plt.close()
