import numpy as np
import joblib
import os
import os.path as osp

from rlkit.torch.networks import AntAggregateExpert

max_path_length = 100
# max_path_length = 50
expert_paths = {
    # multi-ant 4 directions distance 4
    (4.0, 0.0): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-4-0-no-survive/ant_large_expert_one_direction_4_0_no_survive_2019_05_15_19_58_03_0000--s-0/extra_data.pkl',
    (0.0, 4.0): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-0-4-no-survive/ant_large_expert_one_direction_0_4_no_survive_2019_05_15_19_59_23_0000--s-0/extra_data.pkl',
    (0.0, -4.0): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-0-neg-4-no-survive/ant_large_expert_one_direction_0_neg_4_no_survive_2019_05_15_20_02_50_0000--s-0/extra_data.pkl',
    (-4.0, 0.0): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-neg-4-0-no-survive/ant_large_expert_one_direction_neg_4_0_no_survive_2019_05_15_20_03_45_0000--s-0/extra_data.pkl',

    # TRAIN ONES
    # (2.0, 0.0): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-2-0-no-survive/ant_large_expert_one_direction_2_0_no_survive_2019_04_23_19_52_16_0000--s-0/extra_data.pkl',
    # (1.41, 1.41): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-1p41-1p41-no-survive/ant_large_expert_one_direction_1p41_1p41_no_survive_2019_04_23_19_53_28_0000--s-0/extra_data.pkl',
    # (0.0, 2.0): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-0-2-no-survive/ant_large_expert_one_direction_0_2_no_survive_2019_04_23_19_56_25_0000--s-0/extra_data.pkl',
    # (-1.41, 1.41): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-neg-1p41-1p41-no-survive/ant_large_expert_one_direction_neg_1p41_1p41_no_survive_2019_04_23_19_57_55_0001--s-0/extra_data.pkl',
    # (-2.0, 0.0): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-neg-2-0-no-survive/ant_large_expert_one_direction_neg_2_0_no_survive_2019_04_23_19_55_30_0000--s-0/extra_data.pkl',
    # (-1.41, -1.41): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-neg-1p41-neg-1p41-no-survive/ant_large_expert_one_direction_neg_1p41_neg_1p41_no_survive_2019_04_24_14_00_16_0000--s-0/extra_data.pkl',
    # (0.0, -2.0): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-0-neg-2-no-survive/ant_large_expert_one_direction_0_neg_2_no_survive_2019_04_24_14_06_34_0000--s-0/extra_data.pkl',
    # (1.41, -1.41): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-1p41-neg-1p41-no-survive/ant_large_expert_one_direction_1p41_neg_1p41_no_survive_2019_04_24_14_10_53_0000--s-0/extra_data.pkl',

    # (1.85, 0.77): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-1p85-0p77-no-survive/ant_large_expert_one_direction_1p85_0p77_no_survive_2019_04_26_00_42_49_0000--s-0/extra_data.pkl',
    # (0.77, 1.85): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-0p77-1p85-no-survive/ant_large_expert_one_direction_0p77_1p85_no_survive_2019_04_26_00_43_34_0000--s-0/extra_data.pkl',
    # (-0.77, 1.85): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-neg-0p77-1p85-no-survive/ant_large_expert_one_direction_neg_0p77_1p85_no_survive_2019_04_26_00_44_02_0000--s-0/extra_data.pkl',
    # (-1.85, 0.77): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-neg-1p85-0p77-no-survive/ant_large_expert_one_direction_neg_1p85_0p77_no_survive_2019_04_26_00_45_45_0000--s-0/extra_data.pkl',
    # (-1.85, -0.77): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-neg-1p85-neg-0p77-no-survive/ant_large_expert_one_direction_neg_1p85_neg_0p77_no_survive_2019_04_26_00_47_13_0000--s-0/extra_data.pkl',
    # (-0.77, -1.85): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-neg-0p77-neg-1p85-no-survive/ant_large_expert_one_direction_neg_0p77_neg_1p85_no_survive_2019_04_26_01_01_37_0000--s-0/extra_data.pkl',
    # (0.77, -1.85): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-0p77-neg-1p85-no-survive/ant_large_expert_one_direction_0p77_neg_1p85_no_survive_2019_04_26_00_48_54_0000--s-0/extra_data.pkl',
    # (1.85, -0.77): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-1p85-neg-0p77-no-survive/ant_large_expert_one_direction_1p85_neg_0p77_no_survive_2019_04_26_10_47_25_0000--s-0/extra_data.pkl',

    # ( 1.96,  0.39): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-test-0-no-survive/ant_large_expert_one_direction_test_0_no_survive_2019_04_26_11_41_01_0000--s-0/extra_data.pkl',
    # ( 1.66,  1.11): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-test-1-no-survive/ant_large_expert_one_direction_test_1_no_survive_2019_04_26_11_41_59_0000--s-0/extra_data.pkl',
    # ( 1.11,  1.66): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-test-2-no-survive/ant_large_expert_one_direction_test_2_no_survive_2019_04_26_11_42_23_0000--s-0/extra_data.pkl',
    # ( 0.39,  1.96): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-test-3-no-survive/ant_large_expert_one_direction_test_3_no_survive_2019_04_26_11_43_06_0000--s-0/extra_data.pkl',
    # (-0.39,  1.96): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-test-4-no-survive/ant_large_expert_one_direction_test_4_no_survive_2019_04_26_11_43_35_0000--s-0/extra_data.pkl',
    # (-1.11,  1.66): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-test-5-no-survive/ant_large_expert_one_direction_test_5_no_survive_2019_04_26_11_44_11_0000--s-0/extra_data.pkl',
    # (-1.66,  1.11): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-test-6-no-survive/ant_large_expert_one_direction_test_6_no_survive_2019_04_26_11_44_36_0000--s-0/extra_data.pkl',
    # (-1.96,  0.39): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-test-7-no-survive/ant_large_expert_one_direction_test_7_no_survive_2019_04_26_11_45_04_0000--s-0/extra_data.pkl',
    # (-1.96, -0.39): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-test-8-no-survive/ant_large_expert_one_direction_test_8_no_survive_2019_04_26_14_22_02_0000--s-0/extra_data.pkl',
    # (-1.66, -1.11): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-test-9-no-survive/ant_large_expert_one_direction_test_9_no_survive_2019_04_26_14_24_04_0000--s-0/extra_data.pkl',
    # (-1.11, -1.66): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-test-10-no-survive/ant_large_expert_one_direction_test_10_no_survive_2019_04_26_14_25_00_0000--s-0/extra_data.pkl',
    # (-0.39, -1.96): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-test-11-no-survive/ant_large_expert_one_direction_test_11_no_survive_2019_04_26_14_26_36_0000--s-0/extra_data.pkl',
    # ( 0.39, -1.96): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-test-12-no-survive/ant_large_expert_one_direction_test_12_no_survive_2019_04_26_14_27_04_0000--s-0/extra_data.pkl',
    # ( 1.11, -1.66): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-test-13-no-survive/ant_large_expert_one_direction_test_13_no_survive_2019_04_26_14_27_30_0000--s-0/extra_data.pkl',
    # ( 1.66, -1.11): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-test-14-no-survive/ant_large_expert_one_direction_test_14_no_survive_2019_04_26_14_28_01_0000--s-0/extra_data.pkl',
    # ( 1.96, -0.39): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-test-15-no-survive/ant_large_expert_one_direction_test_15_no_survive_2019_04_26_14_28_30_0000--s-0/extra_data.pkl'
}
# expert_paths = {
#     ( 1.99,  0.2 ): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-pi-over-32-16-points-test-0-no-survive/ant_large_expert_one_direction_pi_over_32_16_points_test_0_no_survive_2019_04_27_23_47_03_0000--s-0/extra_data.pkl',
#     ( 1.76,  0.94): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-pi-over-32-16-points-test-1-no-survive/ant_large_expert_one_direction_pi_over_32_16_points_test_1_no_survive_2019_04_27_23_48_20_0000--s-0/extra_data.pkl',
#     ( 1.27,  1.55): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-pi-over-32-16-points-test-2-no-survive/ant_large_expert_one_direction_pi_over_32_16_points_test_2_no_survive_2019_04_27_23_48_50_0000--s-0/extra_data.pkl',
#     ( 0.58,  1.91): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-pi-over-32-16-points-test-3-no-survive/ant_large_expert_one_direction_pi_over_32_16_points_test_3_no_survive_2019_04_27_23_49_25_0000--s-0/extra_data.pkl',
#     (-0.2 ,  1.99): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-pi-over-32-16-points-test-4-no-survive/ant_large_expert_one_direction_pi_over_32_16_points_test_4_no_survive_2019_04_27_23_49_55_0000--s-0/extra_data.pkl',
#     (-0.94,  1.76): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-pi-over-32-16-points-test-5-no-survive/ant_large_expert_one_direction_pi_over_32_16_points_test_5_no_survive_2019_04_27_23_50_46_0000--s-0/extra_data.pkl',
#     (-1.55,  1.27): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-pi-over-32-16-points-test-6-no-survive/ant_large_expert_one_direction_pi_over_32_16_points_test_6_no_survive_2019_04_27_23_51_12_0000--s-0/extra_data.pkl',
#     (-1.91,  0.58): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-pi-over-32-16-points-test-7-no-survive/ant_large_expert_one_direction_pi_over_32_16_points_test_7_no_survive_2019_04_27_23_51_32_0000--s-0/extra_data.pkl',
#     (-1.99, -0.2 ): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-pi-over-32-16-points-test-8-no-survive/ant_large_expert_one_direction_pi_over_32_16_points_test_8_no_survive_2019_04_27_23_51_54_0000--s-0/extra_data.pkl',
#     (-1.76, -0.94): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-pi-over-32-16-points-test-9-no-survive/ant_large_expert_one_direction_pi_over_32_16_points_test_9_no_survive_2019_04_27_23_52_30_0000--s-0/extra_data.pkl',
#     (-1.27, -1.55): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-pi-over-32-16-points-test-10-no-survive/ant_large_expert_one_direction_pi_over_32_16_points_test_10_no_survive_2019_04_27_23_53_56_0000--s-0/extra_data.pkl',
#     (-0.58, -1.91): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-pi-over-32-16-points-test-11-no-survive/ant_large_expert_one_direction_pi_over_32_16_points_test_11_no_survive_2019_04_27_23_54_24_0000--s-0/extra_data.pkl',
#     ( 0.2 , -1.99): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-pi-over-32-16-points-test-12-no-survive/ant_large_expert_one_direction_pi_over_32_16_points_test_12_no_survive_2019_04_27_23_54_58_0000--s-0/extra_data.pkl',
#     ( 0.94, -1.76): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-pi-over-32-16-points-test-13-no-survive/ant_large_expert_one_direction_pi_over_32_16_points_test_13_no_survive_2019_04_27_23_55_23_0000--s-0/extra_data.pkl',
#     ( 1.55, -1.27): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-pi-over-32-16-points-test-14-no-survive/ant_large_expert_one_direction_pi_over_32_16_points_test_14_no_survive_2019_04_27_23_56_21_0000--s-0/extra_data.pkl',
#     ( 1.91, -0.58): '/scratch/hdd001/home/kamyar/output/ant-large-expert-one-direction-pi-over-32-16-points-test-15-no-survive/ant_large_expert_one_direction_pi_over_32_16_points_test_15_no_survive_2019_04_27_23_56_47_0000--s-0/extra_data.pkl',
# }
# save_path = '/scratch/hdd001/home/kamyar/expert_demos/pi_over_32_16_points_test_ant_agg'
# save_path = '/scratch/hdd001/home/kamyar/expert_demos/ant_multi_target_experiment_expert' # the experiment where you want to see if fairl is better than airl
# save_path = '/scratch/hdd001/home/kamyar/expert_demos/ant_linear_classification_expert' # the experiment where you want to see if fairl is better than airl
# save_path = '/scratch/hdd001/home/kamyar/expert_demos/ant_multi_target_experiment_expert_4_directions' # the experiment where you want to see if fairl is better than airl
save_path = '/scratch/hdd001/home/kamyar/expert_demos/ant_multi_target_experiment_expert_4_directions_4_distance' # the experiment where you want to see if fairl is better than airl

def make_e_dict(paths):
    new_dict = {}
    for k, p in paths.items():
        print(p)
        new_dict[k] = joblib.load(p)['algorithm'].get_exploration_policy(np.array(k))
    return new_dict

if __name__ == '__main__':
    os.makedirs(save_path, exist_ok=True)

    e_dict = make_e_dict(expert_paths)
    agg = AntAggregateExpert(e_dict, max_path_length)
    joblib.dump({'algorithm': agg}, osp.join(save_path, 'extra_data.pkl'), compress=3)

    print('\n')
    print(max_path_length)
    print(osp.join(save_path, 'extra_data.pkl'))
