import joblib
import numpy as np

# p = '/ais/gobi6/kamyar/oorl_rlkit/output/expert-dmcs-simple-meta-reacher-subsample-1/expert_dmcs_simple_meta_reacher_subsample_1_2018_11_04_19_24_02_0003--s-0/extra_data.pkl'
# p = '/ais/gobi6/kamyar/oorl_rlkit/output/test-reacher-expert/test_reacher_expert_2018_11_05_12_36_04_0000--s-0/extra_data.pkl'
# p = '/ais/gobi6/kamyar/oorl_rlkit/output/temp-reacher-expert-with-finger-pos/temp_reacher_expert_with_finger_pos_2018_11_07_19_47_48_0003--s-0/extra_data.pkl'
# p = '/ais/gobi6/kamyar/oorl_rlkit/output/good-halfcheetah-expert-100-traj-subsampling-20-stochastic/good_halfcheetah_expert_100_traj_subsampling_20_stochastic_2018_11_08_20_25_56_0002--s-0/extra_data.pkl'
p = '/ais/gobi6/kamyar/oorl_rlkit/output/final-good-halfcheetah-expert-100-traj-subsampling-1-stochastic/final_good_halfcheetah_expert_100_traj_subsampling_1_stochastic_2018_11_08_20_09_51_0001--s-0/extra_data.pkl'
# p = '/ais/gobi6/kamyar/oorl_rlkit/output/ugh-very-final-dmcs-simple-meta-reacher-sin-cos-with-finger-pos-100-traj-20-subsampling/ugh_very_final_dmcs_simple_meta_reacher_sin_cos_with_finger_pos_100_traj_20_subsampling_2018_11_09_03_39_04_0004--s-0/extra_data.pkl'
# p = '/ais/gobi6/kamyar/oorl_rlkit/output/ugh-very-final-dmcs-simple-meta-reacher-sin-cos-with-finger-pos-100-traj-20-subsampling/ugh_very_final_dmcs_simple_meta_reacher_sin_cos_with_finger_pos_100_traj_20_subsampling_2018_11_09_03_39_04_0004--s-0/extra_data.pkl'
rb = joblib.load(p)['replay_buffer']

print(rb.traj_starts)

reward_scale = 5.0
subsampling = 1.0

returns = []
start_pos = []
target_pos = []

for i in range(len(rb.traj_starts) - 1):
    start = rb.traj_starts[i]
    end = rb.traj_starts[i+1]

    total_rews = sum(rb._rewards[start:end])/reward_scale
    # print(rb._rewards[start:end])
    num_on_target = sum(abs(rb._rewards[start:end] - reward_scale)/reward_scale < 0.05)
    
    # rb._observations
    # t_x, t_y = rb._observations['obs_task_params'][start][0], rb._observations['obs_task_params'][start][1]
    # x, y = rb._observations['obs'][start][0], rb._observations['obs'][start][1]
    # init_dist = ((t_x - x)**2 + (t_y - y)**2)**0.5

    # start_pos.append((x, y))
    # target_pos.append((t_x, t_y))

    # print('\n(%.1f,%.1f)\t\t(%.1f,%.1f)' % (x,y, t_x,t_y))

    # for dmcs simple meta reacher
    # print('%d\t\t%d\t\t%.4f' % (i, num_on_target[0], total_rews[0]))

    returns.append(total_rews[0])

# print('\n')
print('%.4f +/- %.4f\n' % (np.mean(returns) * subsampling, np.std(returns) * (subsampling)))
