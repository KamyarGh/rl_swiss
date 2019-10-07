import numpy as np
import joblib

# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/final-zero-few-shot-fetch-traj-gen/final_zero_few_shot_fetch_traj_gen_2018_12_31_22_14_57_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/linear-demos-zero-few-shot-fetch-traj-gen/linear_demos_zero_few_shot_fetch_traj_gen_2019_01_04_18_22_15_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/10K-linear-demos-zero-few-shot-fetch-traj-gen/10K_linear_demos_zero_few_shot_fetch_traj_gen_2019_01_06_02_07_59_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/final-correct-1K-linear-demos-zero-few-shot-reach-traj-gen/final_correct_1K_linear_demos_zero_few_shot_reach_traj_gen_2019_01_09_20_56_48_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/final-10K-wrap-absorbing-linear-demos-zero-few-shot-fetch-traj-gen/final_10K_wrap_absorbing_linear_demos_zero_few_shot_fetch_traj_gen_2019_01_13_23_15_41_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/new-zero-fetch-linear-demos-10K/new_zero_fetch_linear_demos_10K_2019_01_14_21_34_58_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/fetch-linear-demos-50-tasks-25-each/fetch_linear_demos_50_tasks_25_each_2019_01_14_17_38_54_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/few-shot-fetch-linear-demos-50-tasks-5-each/few_shot_fetch_linear_demos_50_tasks_5_each_2019_01_17_00_22_32_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/final-correct-few-shot-fetch-linear-demos-50-tasks-5-each-subsample-4/final_correct_few_shot_fetch_linear_demos_50_tasks_5_each_subsample_4_2019_01_17_01_11_04_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/few-shot-fetch-linear-demos-16-tasks-5-each-subsample-4/few_shot_fetch_linear_demos_16_tasks_5_each_subsample_4_2019_01_17_02_29_32_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/few-shot-fetch-linear-demos-8-tasks-5-each-subsample-4/few_shot_fetch_linear_demos_8_tasks_5_each_subsample_4_2019_01_17_17_56_37_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/few-shot-fetch-linear-demos-32-tasks-10-each-subsample-8/few_shot_fetch_linear_demos_32_tasks_10_each_subsample_8_2019_01_17_19_19_20_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/correct-few-shot-fetch-linear-demos-32-tasks-7-each-subsample-8/correct_few_shot_fetch_linear_demos_32_tasks_7_each_subsample_8_2019_01_17_19_47_55_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/few-shot-fetch-linear-demos-16-tasks-10-each-subsample-8/few_shot_fetch_linear_demos_16_tasks_10_each_subsample_8_2019_01_17_19_51_14_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/another-few-shot-fetch-linear-demos-16-tasks-5-each-subsample-8/another_few_shot_fetch_linear_demos_16_tasks_5_each_subsample_8_2019_01_17_19_59_51_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/another-few-shot-fetch-linear-demos-32-tasks-5-each-subsample-8/another_few_shot_fetch_linear_demos_32_tasks_5_each_subsample_8_2019_01_17_20_08_11_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/few-shot-fetch-linear-demos-12-tasks-5-each-subsample-8/few_shot_fetch_linear_demos_12_tasks_5_each_subsample_8_2019_01_17_20_31_05_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/new-gen-few-shot-fetch-linear-demos-12-tasks-10-total-each-subsample-8/new_gen_few_shot_fetch_linear_demos_12_tasks_10_total_each_subsample_8_2019_01_17_20_56_23_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/new-gen-few-shot-fetch-linear-demos-16-tasks-10-total-each-subsample-8/new_gen_few_shot_fetch_linear_demos_16_tasks_10_total_each_subsample_8_2019_01_18_21_45_48_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/new-gen-few-shot-fetch-linear-demos-16-tasks-20-total-each-subsample-8/new_gen_few_shot_fetch_linear_demos_16_tasks_20_total_each_subsample_8_2019_01_18_23_02_05_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/new-gen-few-shot-fetch-linear-demos-16-tasks-15-total-each-subsample-8/new_gen_few_shot_fetch_linear_demos_16_tasks_15_total_each_subsample_8_2019_01_18_23_32_35_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/new-gen-few-shot-fetch-linear-demos-16-tasks-16-total-each-subsample-8/new_gen_few_shot_fetch_linear_demos_16_tasks_16_total_each_subsample_8_2019_01_19_00_53_52_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/correct-new-gen-few-shot-fetch-linear-demos-32-tasks-16-total-each-subsample-8/correct_new_gen_few_shot_fetch_linear_demos_32_tasks_16_total_each_subsample_8_2019_01_19_01_23_09_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/another-seed-correct-new-gen-few-shot-fetch-linear-demos-32-tasks-16-total-each-subsample-8/another_seed_correct_new_gen_few_shot_fetch_linear_demos_32_tasks_16_total_each_subsample_8_2019_01_19_02_02_47_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/another-seed-correct-new-gen-few-shot-fetch-linear-demos-24-tasks-16-total-each-subsample-8/another_seed_correct_new_gen_few_shot_fetch_linear_demos_24_tasks_16_total_each_subsample_8_2019_01_19_12_03_03_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/another-seed-correct-new-gen-few-shot-fetch-linear-demos-16-tasks-16-total-each-subsample-8/another_seed_correct_new_gen_few_shot_fetch_linear_demos_16_tasks_16_total_each_subsample_8_2019_01_19_12_44_18_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/another-seed-correct-new-gen-few-shot-fetch-linear-demos-20-tasks-16-total-each-subsample-8/another_seed_correct_new_gen_few_shot_fetch_linear_demos_20_tasks_16_total_each_subsample_8_2019_01_19_13_17_45_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/final-correct-another-seed-correct-new-gen-few-shot-fetch-linear-demos-32-tasks-16-total-each-subsample-1/final_correct_another_seed_correct_new_gen_few_shot_fetch_linear_demos_32_tasks_16_total_each_subsample_1_2019_01_20_04_51_13_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/fixed-colors-final-correct-another-seed-correct-new-gen-few-shot-fetch-linear-demos-32-tasks-16-total-each-subsample-1/fixed_colors_final_correct_another_seed_correct_new_gen_few_shot_fetch_linear_demos_32_tasks_16_total_each_subsample_1_2019_01_22_06_38_48_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/fixed-colors-0p5-radius-final-correct-another-seed-correct-new-gen-few-shot-fetch-linear-demos-32-tasks-16-total-each-subsample-1/fixed_colors_0p5_radius_final_correct_another_seed_correct_new_gen_few_shot_fetch_linear_demos_32_tasks_16_total_each_subsample_1_2019_01_22_07_53_12_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/final-few-shot-fetch-eval-expert-trajs/final_few_shot_fetch_eval_expert_trajs_2019_01_23_02_22_12_0000--s-0/extra_data.pkl'
# path_to_extra_data = '/scratch/gobi2/kamyar/oorl_rlkit/output/faster-final-few-shot-fetch-eval-expert-trajs/faster_final_few_shot_fetch_eval_expert_trajs_2019_01_23_13_32_04_0000--s-0/extra_data.pkl'
path_to_extra_data = '/scratch/hdd001/home/kamyar/output/hc-rand-vel-very-good-expert-demos-0p1-separated-16-demos-per-task-20-subsample/hc_rand_vel_very_good_expert_demos_0p1_separated_16_demos_per_task_20_subsample_2019_04_13_20_31_33_0000--s-0/extra_data.pkl'
save_path = '/scratch/gobi2/kamyar/oorl_rlkit/expert_demos/faster_few_shot_fetch_eval_expert_trajs/extra_data.pkl'

# SCALE = 0.99
SCALE = 0.9

# obs_max = np.array([0.19673975, 0.19944288, 0.20234512, 0.19673975, 0.19944288,
#     0.20234512, 0.28635685, 0.29541265, 0.00469703, 0.28635685,
#     0.29541265, 0.00469703, 1.3, 1.3, 1.3,
#     1.3, 1.3, 1.3, 0.05095022, 0.05092848,
#     0.01019219, 0.01034121])
# obs_min = np.array([-1.94986926e-01, -1.97374503e-01, -3.04622497e-03, -1.94986926e-01,
#     -1.97374503e-01, -3.04622497e-03, -3.00136632e-01, -2.82639213e-01,
#     -2.17494754e-01, -3.00136632e-01, -2.82639213e-01, -2.17494754e-01,
#     -1.3, -1.3, -1.3, -1.3,
#     -1.3, -1.3, 2.55108763e-06, -8.67902630e-08,
#     -9.42624227e-03, -9.39642018e-03])
# acts_max = np.array([0.24999889, 0.2499995 , 0.2499997 , 0.01499927])
# acts_min = np.array([-0.24999355, -0.24999517, -0.24999965, -0.01499985])

# obs_max = np.array([0.22051651, 0.22935722, 0.20480309, 0.22051651, 0.22935722,
#     0.20480309, 0.30151219, 0.29303502, 0.00444365, 0.30151219,
#     0.29303502, 0.00444365, 1.3, 1.3, 1.3,
#     1.3, 1.3, 1.3, 0.05099135, 0.05091496,
#     0.01034575, 0.0103919 ])
# obs_min = np.array([-1.98124936e-01, -2.04234846e-01, -8.51241789e-03, -1.98124936e-01,
#     -2.04234846e-01, -8.51241789e-03, -3.03874692e-01, -3.00712133e-01,
#     -2.30561716e-01, -3.03874692e-01, -3.00712133e-01, -2.30561716e-01,
#     -1.3, -1.3, -1.3, -1.3,
#     -1.3, -1.3,  2.55108763e-06, -8.67902630e-08,
#     -1.20198677e-02, -9.60486720e-03])
# acts_max = np.array([0.3667496 , 0.3676551 , 0.37420813, 0.015])
# acts_min = np.array([-0.27095875, -0.26862562, -0.27479879, -0.015])




# obs_max = np.array([0.20873973, 0.21238721, 0.20497428, 0.20873973, 0.21238721,
#     0.20497428, 0.29729787, 0.29597882, 0.00660929, 0.29729787,
#     0.29597882, 0.00660929, 1.0, 1.0, 1.0,
#     1.0, 1.0, 1.0, 0.05099425, 0.05097209,
#     0.01045247, 0.01020353])
# obs_min = np.array([-2.07733303e-01, -2.22872196e-01, -6.20862381e-03, -2.07733303e-01,
#     -2.22872196e-01, -6.20862381e-03, -3.02834854e-01, -3.18478521e-01,
#     -2.35453885e-01, -3.02834854e-01, -3.18478521e-01, -2.35453885e-01,
#     -1.0, -1.0, -1.0, -1.0,
#     -1.0, -1.0,  2.55108763e-06, -8.67902630e-08,
#     -1.12767104e-02, -1.15187468e-02])
# acts_max = np.array([0.36385158, 0.36506858, 0.37287046, 0.015])
# acts_min = np.array([-0.27378214, -0.27318582, -0.27457426, -0.015])


# obs_max = np.array([0.19732151, 0.19501755, 0.2032467 , 0.19732151, 0.19501755,
#     0.2032467 , 0.28952909, 0.27034638, 0.00461512, 0.28952909,
#     0.27034638, 0.00461512, 1.        , 1.        , 1.        ,
#     1.        , 1.        , 1.        , 0.05084346, 0.05089836,
#     0.01020451, 0.01024073])
# obs_min = np.array([-1.94163008e-01, -2.06672946e-01, -4.34817497e-03, -1.94163008e-01,
#     -2.06672946e-01, -4.34817497e-03, -2.57836261e-01, -3.02357607e-01,
#     -2.26000082e-01, -2.57836261e-01, -3.02357607e-01, -2.26000082e-01,
#     -1., -1., -1., -1.,
#     -1., -1.,  2.55108763e-06, -8.67902630e-08,
#     -9.79891841e-03, -9.23147216e-03])
# acts_max = np.array([0.36071754, 0.35800805, 0.37175567, 0.015])
# acts_min = np.array([-0.26463221, -0.26663373, -0.27413371, -0.015])



# THESE ARE THE MOST RECENT NORMALIZATION FOR THE FEW SHOT FETCH
obs_max = np.array([0.20061923, 0.19781174, 0.20549539, 0.20061923, 0.19781174,
    0.20549539, 0.29141252, 0.28891717, 0.00129714, 0.29141252,
    0.28891717, 0.00129714, 1.0        , 1.0        , 1.0        ,
    1.0        , 1.0        , 1.0        , 0.05096386, 0.05090749,
    0.01046458, 0.01028522])
obs_min = np.array([-1.83014661e-01, -2.07445100e-01, -4.79934195e-03, -1.83014661e-01,
    -2.07445100e-01, -4.79934195e-03, -2.89125464e-01, -2.96987424e-01,
    -2.30655094e-01, -2.89125464e-01, -2.96987424e-01, -2.30655094e-01,
    -1.0, -1.0, -1.0, -1.0,
    -1.0, -1.0,  2.55108763e-06, -8.67902630e-08,
    -1.11994283e-02, -9.10341004e-03])
acts_max = np.array([0.36051396, 0.36032055, 0.37415428, 0.015])
acts_min = np.array([-0.2696256 , -0.27399028, -0.27453274, -0.015])



def normalize_obs(observation):
    observation = (observation - obs_min) / (obs_max - obs_min)
    observation *= 2 * SCALE
    observation -= SCALE
    return observation

def normalize_acts(action):
    action = (action - acts_min) / (acts_max - acts_min)
    action *= 2 * SCALE
    action -= SCALE
    return action

d = joblib.load(path_to_extra_data)
for meta_split in ['meta_train', 'meta_test']:
    for sub_split in ['context', 'test']:
        for rb in d[meta_split][sub_split].task_replay_buffers.values():
            rb._observations['obs'] = normalize_obs(rb._observations['obs'])
            rb._next_obs['obs'] = normalize_obs(rb._next_obs['obs'])
            rb._actions = normalize_acts(rb._actions)
print('SAVING TO %s' % save_path)
joblib.dump(d, save_path, compress=3)
