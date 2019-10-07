import os
import joblib
import numpy as np
from rlkit.core.vistools import plot_histogram

# path_to_expert_rb = '/scratch/gobi2/kamyar/oorl_rlkit/output/single-task-1000-total-demos-few-shot-larger-object-range-expert/single_task_1000_total_demos_few_shot_larger_object_range_expert_2018_12_28_19_03_11_0000--s-0/extra_data.pkl'
# path_to_expert_rb = '/scratch/gobi2/kamyar/oorl_rlkit/output/linear-demos-zero-few-shot-fetch-traj-gen/linear_demos_zero_few_shot_fetch_traj_gen_2019_01_04_18_22_15_0000--s-0/extra_data.pkl'
# path_to_expert_rb = '/scratch/gobi2/kamyar/oorl_rlkit/expert_demos/scale_0p9_linear_10K_demos_zero_fetch_traj_gen/extra_data.pkl'
# path_to_expert_rb = '/scratch/gobi2/kamyar/oorl_rlkit/output/final-correct-1K-linear-demos-zero-few-shot-reach-traj-gen/final_correct_1K_linear_demos_zero_few_shot_reach_traj_gen_2019_01_09_20_56_48_0000--s-0/extra_data.pkl'
# path_to_expert_rb = '/scratch/gobi2/kamyar/oorl_rlkit/output/final-10K-wrap-absorbing-linear-demos-zero-few-shot-fetch-traj-gen/final_10K_wrap_absorbing_linear_demos_zero_few_shot_fetch_traj_gen_2019_01_13_23_15_41_0000--s-0/extra_data.pkl'
# path_to_expert_rb = '/scratch/gobi2/kamyar/oorl_rlkit/output/new-zero-fetch-linear-demos-10K/new_zero_fetch_linear_demos_10K_2019_01_14_21_34_58_0000--s-0/extra_data.pkl'
# path_to_expert_rb = '/scratch/gobi2/kamyar/oorl_rlkit/output/fetch-linear-demos-50-tasks-25-each/fetch_linear_demos_50_tasks_25_each_2019_01_14_17_38_54_0000--s-0/extra_data.pkl'
# path_to_expert_rb = '/scratch/gobi2/kamyar/oorl_rlkit/output/fixed-colors-final-correct-another-seed-correct-new-gen-few-shot-fetch-linear-demos-32-tasks-16-total-each-subsample-1/fixed_colors_final_correct_another_seed_correct_new_gen_few_shot_fetch_linear_demos_32_tasks_16_total_each_subsample_1_2019_01_22_06_38_48_0000--s-0/extra_data.pkl'
path_to_expert_rb = '/scratch/gobi2/kamyar/oorl_rlkit/output/fixed-colors-0p5-radius-final-correct-another-seed-correct-new-gen-few-shot-fetch-linear-demos-32-tasks-16-total-each-subsample-1/fixed_colors_0p5_radius_final_correct_another_seed_correct_new_gen_few_shot_fetch_linear_demos_32_tasks_16_total_each_subsample_1_2019_01_22_07_53_12_0000--s-0/extra_data.pkl'
print(path_to_expert_rb)
# path_to_expert_rb = '/scratch/gobi2/kamyar/oorl_rlkit/expert_demos/normalized_basic_few_shot_fetch_demos/extra_data.pkl'
# path_to_expert_rb = '/scratch/gobi2/kamyar/oorl_rlkit/output/correct-basic-few-shot-fetch-traj-gen/correct_basic_few_shot_fetch_traj_gen_2018_12_19_08_57_56_0000--s-0/extra_data.pkl'
# plot_dir = 'plots/expert_demos_stats/basic_few_shot_fetch'
# plot_dir = 'plots/expert_demos_stats/scaled_0p9_linear_10K_stats'

# we should only use the train set to find the scaling factors
meta_train_rb = joblib.load(path_to_expert_rb)['meta_train']
context_rb = meta_train_rb['context']
test_rb = meta_train_rb['test']

obs = [rb._observations['obs'] for rb in context_rb.task_replay_buffers.values()] + [rb._observations['obs'] for rb in test_rb.task_replay_buffers.values()]
obs = np.concatenate(obs, 0)
acts = [rb._actions for rb in context_rb.task_replay_buffers.values()] + [rb._actions for rb in test_rb.task_replay_buffers.values()]
acts = np.concatenate(acts, 0)

print('\n----------\nObs:')
obj2goal_max = np.max(np.concatenate([obs[:,:3], obs[:,3:6]], 0), axis=0)
objrel_max = np.max(np.concatenate([obs[:,6:9], obs[:,9:12]], 0), axis=0)
objcolor_max = np.max(np.concatenate([obs[:,12:15], obs[:,15:18]], 0), axis=0)
gripper_state_max = np.max(obs[:,18:20], axis=0)
gripper_vel_max = np.max(obs[:,20:22], axis=0)

obj2goal_min = np.min(np.concatenate([obs[:,:3], obs[:,3:6]], 0), axis=0)
objrel_min = np.min(np.concatenate([obs[:,6:9], obs[:,9:12]], 0), axis=0)
objcolor_min = np.min(np.concatenate([obs[:,12:15], obs[:,15:18]], 0), axis=0)
gripper_state_min = np.min(obs[:,18:20], axis=0)
gripper_vel_min = np.min(obs[:,20:22], axis=0)

obs_max = np.concatenate(
    [
        obj2goal_max,
        obj2goal_max,
        objrel_max,
        objrel_max,
        objcolor_max,
        objcolor_max,
        gripper_state_max,
        gripper_vel_max
    ],
    axis=-1
)
obs_min = np.concatenate(
    [
        obj2goal_min,
        obj2goal_min,
        objrel_min,
        objrel_min,
        objcolor_min,
        objcolor_min,
        gripper_state_min,
        gripper_vel_min
    ],
    axis=-1
)

print(repr(obs_max))
print(repr(obs_min))

print('\n----------\nActs:')
print(repr(np.max(acts, axis=0)))
print(repr(np.min(acts, axis=0)))

# print('obs')
# print(repr(np.mean(obs, axis=0)))
# print(repr(np.std(obs, axis=0)))
# print(repr(np.max(obs, axis=0)))
# print(repr(np.min(obs, axis=0)))

# print('\nacts')
# print(repr(np.mean(acts, axis=0)))
# print(repr(np.std(acts, axis=0)))
# print(repr(np.max(acts, axis=0)))
# print(repr(np.min(acts, axis=0)))

# SCALE = 0.99
# norm_obs = (obs - np.min(obs, axis=0)) / (np.max(obs, axis=0) - np.min(obs, axis=0))
# norm_obs *= 2 * SCALE
# norm_obs -= SCALE

# norm_acts = (acts - np.min(acts, axis=0)) / (np.max(acts, axis=0) - np.min(acts, axis=0))
# norm_acts *= 2 * SCALE
# norm_acts -= SCALE

# for i in range(obs.shape[1]):
#     plot_histogram(obs[:,i], 100, 'obs dim %d' % i, os.path.join(plot_dir, 'obs_%d.png'%i))
#     plot_histogram(norm_obs[:,i], 100, 'norm obs dim %d' % i, os.path.join(plot_dir, 'norm_obs_%d.png'%i))
# for i in range(acts.shape[1]):
#     plot_histogram(acts[:,i], 100, 'acts dim %d' % i, os.path.join(plot_dir, 'acts_%d.png'%i))
#     plot_histogram(norm_acts[:,i], 100, 'norm acts dim %d' % i, os.path.join(plot_dir, 'norm_acts_%d.png'%i))


def print_extra(array, name):
    print('\n\nExtra Stats for %s ----------------------' % name)
    print('\nMean')
    print(np.mean(array, axis=0))
    print('\nStd')
    print(np.std(array, axis=0))
    print('\nMax')
    print(np.max(array, axis=0))
    print('\nMin')
    print(np.min(array, axis=0))

print_extra(obs, 'Obs')
print_extra(acts, 'Acts')
