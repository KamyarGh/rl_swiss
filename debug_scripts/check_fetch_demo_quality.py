import os
import numpy as np
from rlkit.core.vistools import plot_histogram

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

# her_demos_path = '/u/kamyar/baselines/baselines/her/data_fetch_random_100.npz'
# her_demos_path = '/ais/gobi6/kamyar/oorl_rlkit/expert_demos/her_fetch_pick_and_place/fetch_reach_and_lift/data_for_fetch_reach_and_lift_random_100.npz'
# her_demos_path = '/ais/gobi6/kamyar/oorl_rlkit/expert_demos/her_fetch_pick_and_place/easy_0p01_range_1_goal_high/data_easy_0p01_range_goal_high_prob_1_fetch_pick_and_place_random_100.npz'
# her_demos_path = '/ais/gobi6/kamyar/oorl_rlkit/expert_demos/her_fetch_pick_and_place/easy_0p01_range_1_goal_high/1000_demos/data_easy_0p01_range_goal_high_prob_1_fetch_pick_and_place_random_1000.npz'
her_demos_path = '/ais/gobi6/kamyar/oorl_rlkit/expert_demos/her_fetch_pick_and_place/easy_0p01_range_1_goal_high/1000_clipped_demos/clipped_acts_data_easy_0p01_range_goal_high_prob_1_fetch_pick_and_place_random_1000.npz'
d = np.load(her_demos_path)

rews = []
path_lens = []
for path in d['obs']:
    path_rew = 0
    # for step in path:
    for i in range(50):
        step = path[i]
        ag = step['achieved_goal']
        dg = step['desired_goal']
        dist = goal_distance(ag, dg)
        if dist > 0.05:
            path_rew += -1.0
        else:
            path_rew += -1.0*dist
    rews.append(path_rew)
    # path_lens.append(len(path))
    path_lens.append(50)

zipped = list(zip(rews, path_lens))
print(zipped)
solved = [t[0] > -1.0*t[1] for t in zipped]
print(solved)
print('%.4f +/- %.4f' % (np.mean(rews), np.std(rews)))
print(sum(solved))
print(sum(solved) / float(len(solved)))

# compute action stats
all_acts = np.array([
    a for path in d['acs'] for a in path
])
print('\n Acts Stats ------------------\n')
print(all_acts.shape)
print(repr(np.mean(all_acts, axis=0)))
# [-0.00097687 -0.00931541  0.00991785  0.01412615]
all_acts_std = np.std(all_acts, axis=0)
print(repr(all_acts_std))
# [0.27218603 0.25925623 0.32518755 0.02619406]
print((all_acts - np.mean(all_acts, axis=0)) / np.std(all_acts, axis=0))

abs_acts = np.abs(all_acts)
print(np.mean(abs_acts, axis=0))
# [0.14009708 0.13060458 0.13870633 0.02064866]
print(np.std(abs_acts, axis=0))
# [0.23336452 0.22414953 0.29428874 0.0214315 ]
max_acts = np.max(all_acts, axis=0)
print(max_acts)
min_acts = np.min(all_acts, axis=0)
print(min_acts)


print(np.sum(abs_acts>1.0, axis=0))
print(np.sum(abs_acts>1.0, axis=0) / abs_acts.shape[0])


print('\n Obs Stats -------------\n')
all_obs = np.array([
    o['observation'] for path in d['obs'] for o in path
])
print(repr(np.mean(all_obs, axis=0)))
all_obs_std = np.std(all_obs, axis=0)
print(repr(all_obs_std))
max_obs = np.max(all_obs, axis=0)
print(repr(max_obs))
min_obs = np.min(all_obs, axis=0)
print(repr(min_obs))

print('\n Goal Stats ---------------\n')
all_achieved = np.array([
    o['achieved_goal'] for path in d['obs'] for o in path
])
all_desired = np.array([
    o['desired_goal'] for path in d['obs'] for o in path
])
all_goals = np.concatenate((all_achieved, all_desired), axis=0)
print(repr(np.mean(all_goals, axis=0)))
print(repr(np.std(all_goals, axis=0)))
print(repr(np.max(all_goals, axis=0)))
print(repr(np.min(all_goals, axis=0)))

# print('\n---------------\n')
# all_achieved = np.array([
#     o['achieved_goal'] for path in d['obs'] for o in path
# ])
# print(repr(np.mean(all_achieved, axis=0)))
# all_achieved_std = np.std(all_achieved, axis=0)
# print(repr(all_achieved_std))
# max_achieved = np.max(all_achieved, axis=0)
# print(repr(max_achieved))
# min_achieved = np.min(all_achieved, axis=0)
# print(repr(min_achieved))

# print('\n---------------\n')
# all_desired = np.array([
#     o['desired_goal'] for path in d['obs'] for o in path
# ])
# print(repr(np.mean(all_desired, axis=0)))
# all_desired_std = np.std(all_desired, axis=0)
# print(repr(all_desired_std))
# max_desired = np.max(all_desired, axis=0)
# print(repr(max_desired))
# min_desired = np.min(all_desired, axis=0)
# print(repr(min_desired))

# scale the obs and actions to -0.99 and 0.99 range
SCALE = 0.99
all_obs = (all_obs - min_obs[None,:]) / (max_obs[None,:] - min_obs[None,:])
all_obs *= 2 * SCALE
all_obs -= SCALE

all_acts = (all_acts - min_acts[None,:]) / (max_acts[None,:] - min_acts[None,:])
all_acts *= 2 * SCALE
all_acts -= SCALE

# I want to plot the histogram of observation and action dims
hist_save_path = '/u/kamyar/oorl_rlkit/plots/easy_fetch_hists'
for i, std in enumerate(all_acts_std):
    plot_histogram(all_acts[:,i], 100, 'std %.4f'%std, os.path.join(hist_save_path, 'acts_dim_%d.png'%i))
for i, std in enumerate(all_obs_std):
    plot_histogram(all_obs[:,i], 100, 'std %.4f'%std, os.path.join(hist_save_path, 'obs_dim_%d.png'%i))
