import os
import joblib
import numpy as np
from rlkit.core.vistools import plot_histogram

path_to_expert_rb = '/scratch/gobi2/kamyar/oorl_rlkit/expert_demos/scale_0p9_linear_10K_demos_zero_fetch_traj_gen/extra_data.pkl'
# path_to_expert_rb = '/scratch/gobi2/kamyar/oorl_rlkit/expert_demos/larger_object_range_fetch_pick_and_place/extra_data.pkl'
plot_dir = 'plots/expert_demos_stats/scaled_0p9_linear_10K_stats_plus_normal_0.055'
os.makedirs(plot_dir, exist_ok=True)
# plot_dir = 'plots/expert_demos_stats/target_in_air_easy_larger_object_range'
print(path_to_expert_rb)
rb = joblib.load(path_to_expert_rb)['replay_buffer']
obs = rb._observations['obs']
obs += np.random.normal(scale=0.055, size=obs.shape)
# obs = rb._observations
acts = rb._actions
acts += np.random.normal(scale=0.055, size=acts.shape)

print('obs')
print(repr(np.mean(obs, axis=0)))
print(repr(np.std(obs, axis=0)))
print(repr(np.max(obs, axis=0)))
print(repr(np.min(obs, axis=0)))

print('\nacts')
print(repr(np.mean(acts, axis=0)))
print(repr(np.std(acts, axis=0)))
print(repr(np.max(acts, axis=0)))
print(repr(np.min(acts, axis=0)))

SCALE = 0.99
norm_obs = (obs - np.min(obs, axis=0)) / (np.max(obs, axis=0) - np.min(obs, axis=0))
norm_obs *= 2 * SCALE
norm_obs -= SCALE

norm_acts = (acts - np.min(acts, axis=0)) / (np.max(acts, axis=0) - np.min(acts, axis=0))
norm_acts *= 2 * SCALE
norm_acts -= SCALE

for i in range(obs.shape[1]):
    plot_histogram(obs[:,i], 100, 'obs dim %d' % i, os.path.join(plot_dir, 'obs_%d.png'%i))
    plot_histogram(norm_obs[:,i], 100, 'norm obs dim %d' % i, os.path.join(plot_dir, 'norm_obs_%d.png'%i))
for i in range(acts.shape[1]):
    plot_histogram(acts[:,i], 100, 'acts dim %d' % i, os.path.join(plot_dir, 'acts_%d.png'%i))
    plot_histogram(norm_acts[:,i], 100, 'norm acts dim %d' % i, os.path.join(plot_dir, 'norm_acts_%d.png'%i))
