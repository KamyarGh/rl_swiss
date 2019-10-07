'''
For the table results aggregate results across context sizes
But then generate a plot for each of the dataset size that shows
performance for these tasks does not depend much on the context size
'''
import numpy as np
import joblib
from collections import defaultdict
import os.path as osp

context_sizes = [1,2,3,4]
# amounts = [(64,1), (64,20), (16,20), (4,20)]
amounts = [(64,1)]

sa_irl_results = {
    'name': 'Meta-IRL (state-action)',
    (64,1): [
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-64-demos-sub-1-state-action-rew-search-normalized-fixed/hc_rand_vel_np_airl_64_demos_sub_1_state_action_rew_search_normalized_fixed_2019_04_18_17_20_04_0000--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-64-demos-sub-1-state-action-rew-search-normalized-fixed/hc_rand_vel_np_airl_64_demos_sub_1_state_action_rew_search_normalized_fixed_2019_04_18_17_22_04_0001--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-64-demos-sub-1-state-action-rew-search-normalized-fixed/hc_rand_vel_np_airl_64_demos_sub_1_state_action_rew_search_normalized_fixed_2019_04_18_17_26_05_0002--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-64-demos-sub-1-state-action-rew-search-normalized-fixed/hc_rand_vel_np_airl_64_demos_sub_1_state_action_rew_search_normalized_fixed_2019_04_18_17_26_35_0003--s-0',
    ],
    (64,20): [
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-64-demos-rew-search-with-saving-more-rew-search/hc_rand_vel_np_airl_64_demos_rew_search_with_saving_more_rew_search_2019_04_14_22_27_53_0001--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-64-demos-rew-search-with-saving-more-rew-search/hc_rand_vel_np_airl_64_demos_rew_search_with_saving_more_rew_search_2019_04_14_22_27_54_0004--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-64-demos-rew-search-with-saving-more-rew-search/hc_rand_vel_np_airl_64_demos_rew_search_with_saving_more_rew_search_2019_04_14_22_27_54_0007--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-64-demos-rew-search-with-saving-more-rew-search/hc_rand_vel_np_airl_64_demos_rew_search_with_saving_more_rew_search_2019_04_14_22_27_53_0010--s-0',
    ],
    (16,20): [
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-16-demos-rew-search-with-saving-more-rew-search/hc_rand_vel_np_airl_16_demos_rew_search_with_saving_more_rew_search_2019_04_15_16_03_04_0001--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-16-demos-rew-search-with-saving-more-rew-search/hc_rand_vel_np_airl_16_demos_rew_search_with_saving_more_rew_search_2019_04_15_16_06_43_0004--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-16-demos-rew-search-with-saving-more-rew-search/hc_rand_vel_np_airl_16_demos_rew_search_with_saving_more_rew_search_2019_04_15_16_33_52_0007--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-16-demos-rew-search-with-saving-more-rew-search/hc_rand_vel_np_airl_16_demos_rew_search_with_saving_more_rew_search_2019_04_15_16_57_06_0010--s-0',
    ],
    (4,20): [
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-4-demos-rew-search-with-saving-more-rew-search/hc_rand_vel_np_airl_4_demos_rew_search_with_saving_more_rew_search_2019_04_20_13_22_53_0000--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-4-demos-rew-search-with-saving-more-rew-search/hc_rand_vel_np_airl_4_demos_rew_search_with_saving_more_rew_search_2019_04_20_13_22_53_0001--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-4-demos-rew-search-with-saving-more-rew-search/hc_rand_vel_np_airl_4_demos_rew_search_with_saving_more_rew_search_2019_04_20_13_22_54_0002--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-4-demos-rew-search-with-saving-more-rew-search/hc_rand_vel_np_airl_4_demos_rew_search_with_saving_more_rew_search_2019_04_20_13_22_54_0003--s-0',
    ]
}
s_irl_results = {
    'name': 'Meta-IRL (state-only)',
    (64,1): [
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-64-demos-sub-1-state-only-rew-search-normalized-correct/hc_rand_vel_np_airl_64_demos_sub_1_state_only_rew_search_normalized_correct_2019_04_20_00_00_12_0001--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-64-demos-sub-1-state-only-rew-search-normalized-correct/hc_rand_vel_np_airl_64_demos_sub_1_state_only_rew_search_normalized_correct_2019_04_20_00_00_12_0002--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-64-demos-sub-1-state-only-rew-search-normalized-correct/hc_rand_vel_np_airl_64_demos_sub_1_state_only_rew_search_normalized_correct_2019_04_20_00_00_13_0000--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-64-demos-sub-1-state-only-rew-search-normalized-correct/hc_rand_vel_np_airl_64_demos_sub_1_state_only_rew_search_normalized_correct_2019_04_20_00_01_42_0003--s-0',
    ],
    (64,20): [
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-64-demos-sub-20-state-only-rew-search-normalized-correct/hc_rand_vel_np_airl_64_demos_sub_20_state_only_rew_search_normalized_correct_2019_04_20_00_02_42_0000--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-64-demos-sub-20-state-only-rew-search-normalized-correct/hc_rand_vel_np_airl_64_demos_sub_20_state_only_rew_search_normalized_correct_2019_04_20_00_22_13_0001--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-64-demos-sub-20-state-only-rew-search-normalized-correct/hc_rand_vel_np_airl_64_demos_sub_20_state_only_rew_search_normalized_correct_2019_04_20_00_25_13_0002--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-64-demos-sub-20-state-only-rew-search-normalized-correct/hc_rand_vel_np_airl_64_demos_sub_20_state_only_rew_search_normalized_correct_2019_04_20_00_26_13_0003--s-0',
    ],
    (16,20): [
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-16-demos-sub-20-state-only-rew-search-normalized-correct/hc_rand_vel_np_airl_16_demos_sub_20_state_only_rew_search_normalized_correct_2019_04_20_00_26_13_0000--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-16-demos-sub-20-state-only-rew-search-normalized-correct/hc_rand_vel_np_airl_16_demos_sub_20_state_only_rew_search_normalized_correct_2019_04_20_00_35_13_0001--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-16-demos-sub-20-state-only-rew-search-normalized-correct/hc_rand_vel_np_airl_16_demos_sub_20_state_only_rew_search_normalized_correct_2019_04_20_00_46_43_0002--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-16-demos-sub-20-state-only-rew-search-normalized-correct/hc_rand_vel_np_airl_16_demos_sub_20_state_only_rew_search_normalized_correct_2019_04_20_00_47_43_0003--s-0',
    ],
    (4,20): [
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-4-demos-sub-20-state-only-rew-search-normalized-correct/hc_rand_vel_np_airl_4_demos_sub_20_state_only_rew_search_normalized_correct_2019_04_20_13_21_24_0000--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-4-demos-sub-20-state-only-rew-search-normalized-correct/hc_rand_vel_np_airl_4_demos_sub_20_state_only_rew_search_normalized_correct_2019_04_20_13_21_25_0001--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-4-demos-sub-20-state-only-rew-search-normalized-correct/hc_rand_vel_np_airl_4_demos_sub_20_state_only_rew_search_normalized_correct_2019_04_20_13_21_25_0002--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-np-airl-4-demos-sub-20-state-only-rew-search-normalized-correct/hc_rand_vel_np_airl_4_demos_sub_20_state_only_rew_search_normalized_correct_2019_04_20_13_21_26_0003--s-0',
    ]
}
# MLE version
# bc_results = {
#     'name': 'Meta-BC',
#     (64,1): [
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-64-demos-sub-1-no-norm-with-saving/hc_rand_vel_64_demos_sub_1_no_norm_with_saving_2019_04_19_21_36_41_0000--s-0',
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-64-demos-sub-1-no-norm-with-saving/hc_rand_vel_64_demos_sub_1_no_norm_with_saving_2019_04_19_21_36_41_0001--s-0',
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-64-demos-sub-1-no-norm-with-saving/hc_rand_vel_64_demos_sub_1_no_norm_with_saving_2019_04_19_21_36_41_0002--s-0',
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-64-demos-sub-1-no-norm-with-saving/hc_rand_vel_64_demos_sub_1_no_norm_with_saving_2019_04_19_21_36_41_0003--s-0',
#     ],
#     (64,20): [
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-64-demos-sub-20-no-norm-with-saving/hc_rand_vel_64_demos_sub_20_no_norm_with_saving_2019_04_19_21_41_10_0002--s-0',
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-64-demos-sub-20-no-norm-with-saving/hc_rand_vel_64_demos_sub_20_no_norm_with_saving_2019_04_19_21_41_11_0000--s-0',
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-64-demos-sub-20-no-norm-with-saving/hc_rand_vel_64_demos_sub_20_no_norm_with_saving_2019_04_19_21_41_11_0001--s-0',
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-64-demos-sub-20-no-norm-with-saving/hc_rand_vel_64_demos_sub_20_no_norm_with_saving_2019_04_19_21_41_11_0003--s-0',
#     ],
#     (16,20): [
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-16-demos-sub-20-no-norm-with-saving/hc_rand_vel_16_demos_sub_20_no_norm_with_saving_2019_04_19_21_38_40_0003--s-0',
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-16-demos-sub-20-no-norm-with-saving/hc_rand_vel_16_demos_sub_20_no_norm_with_saving_2019_04_19_21_38_41_0000--s-0',
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-16-demos-sub-20-no-norm-with-saving/hc_rand_vel_16_demos_sub_20_no_norm_with_saving_2019_04_19_21_38_41_0001--s-0',
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-16-demos-sub-20-no-norm-with-saving/hc_rand_vel_16_demos_sub_20_no_norm_with_saving_2019_04_19_21_38_41_0002--s-0',
#     ],
#     (4,20): [
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-4-demos-sub-20-no-norm-with-saving/hc_rand_vel_4_demos_sub_20_no_norm_with_saving_2019_04_19_22_22_41_0001--s-0',
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-4-demos-sub-20-no-norm-with-saving/hc_rand_vel_4_demos_sub_20_no_norm_with_saving_2019_04_19_22_22_41_0002--s-0',
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-4-demos-sub-20-no-norm-with-saving/hc_rand_vel_4_demos_sub_20_no_norm_with_saving_2019_04_19_22_22_41_0003--s-0',
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-4-demos-sub-20-no-norm-with-saving/hc_rand_vel_4_demos_sub_20_no_norm_with_saving_2019_04_19_22_22_43_0000--s-0',
#     ]
# }

# MSE version
bc_results = {
    'name': 'Meta-BC',
    (64,1): [
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-mse-64-demos-sub-1-paper-version/hc_rand_vel_mse_64_demos_sub_1_paper_version_2019_05_25_15_00_00_0000--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-mse-64-demos-sub-1-paper-version/hc_rand_vel_mse_64_demos_sub_1_paper_version_2019_05_25_15_00_01_0001--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-mse-64-demos-sub-1-paper-version/hc_rand_vel_mse_64_demos_sub_1_paper_version_2019_05_25_15_00_01_0002--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-mse-64-demos-sub-1-paper-version/hc_rand_vel_mse_64_demos_sub_1_paper_version_2019_05_25_15_00_02_0003--s-0',
    ],
    (64,20): [
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-mse-64-demos-sub-20-paper-version/hc_rand_vel_mse_64_demos_sub_20_paper_version_2019_05_25_14_59_10_0000--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-mse-64-demos-sub-20-paper-version/hc_rand_vel_mse_64_demos_sub_20_paper_version_2019_05_25_14_59_12_0001--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-mse-64-demos-sub-20-paper-version/hc_rand_vel_mse_64_demos_sub_20_paper_version_2019_05_25_14_59_12_0002--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-mse-64-demos-sub-20-paper-version/hc_rand_vel_mse_64_demos_sub_20_paper_version_2019_05_25_14_59_13_0003--s-0',
    ],
    (16,20): [
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-mse-16-demos-sub-20-paper-version/hc_rand_vel_mse_16_demos_sub_20_paper_version_2019_05_25_14_58_30_0000--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-mse-16-demos-sub-20-paper-version/hc_rand_vel_mse_16_demos_sub_20_paper_version_2019_05_25_14_58_31_0001--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-mse-16-demos-sub-20-paper-version/hc_rand_vel_mse_16_demos_sub_20_paper_version_2019_05_25_14_58_31_0002--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-mse-16-demos-sub-20-paper-version/hc_rand_vel_mse_16_demos_sub_20_paper_version_2019_05_25_14_58_33_0003--s-0',
    ],
    (4,20): [
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-mse-4-demos-sub-20-paper-version/hc_rand_vel_mse_4_demos_sub_20_paper_version_2019_05_25_14_57_46_0000--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-mse-4-demos-sub-20-paper-version/hc_rand_vel_mse_4_demos_sub_20_paper_version_2019_05_25_14_57_46_0001--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-mse-4-demos-sub-20-paper-version/hc_rand_vel_mse_4_demos_sub_20_paper_version_2019_05_25_14_57_47_0002--s-0',
        '/scratch/hdd001/home/kamyar/output/hc-rand-vel-mse-4-demos-sub-20-paper-version/hc_rand_vel_mse_4_demos_sub_20_paper_version_2019_05_25_14_57_47_0003--s-0',
    ]
}

# Dagger MSE version
dagger_results = {
    'name': 'Meta-Dagger',
    (64,1): [
        '/scratch/hdd001/home/kamyar/output/correct-hc-rand-vel-meta-dagger-use-z-sample-det-expert-MSE-64-demos-sub-1-100-updates-per-call/correct_hc_rand_vel_meta_dagger_use_z_sample_det_expert_MSE_64_demos_sub_1_100_updates_per_call_2019_07_28_16_29_06_0000--s-0',
        '/scratch/hdd001/home/kamyar/output/correct-hc-rand-vel-meta-dagger-use-z-sample-det-expert-MSE-64-demos-sub-1-100-updates-per-call/correct_hc_rand_vel_meta_dagger_use_z_sample_det_expert_MSE_64_demos_sub_1_100_updates_per_call_2019_07_28_16_29_07_0001--s-0',
        '/scratch/hdd001/home/kamyar/output/correct-hc-rand-vel-meta-dagger-use-z-sample-det-expert-MSE-64-demos-sub-1-100-updates-per-call/correct_hc_rand_vel_meta_dagger_use_z_sample_det_expert_MSE_64_demos_sub_1_100_updates_per_call_2019_07_28_16_29_09_0002--s-0',
    ]
}

# MLE version
# bc_results = {
#     'name': 'Meta-BC',
#     (64,1): [
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-MLE-64-demos-sub-1-paper-version/hc_rand_vel_MLE_64_demos_sub_1_paper_version_2019_05_25_16_47_11_0000--s-0',
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-MLE-64-demos-sub-1-paper-version/hc_rand_vel_MLE_64_demos_sub_1_paper_version_2019_05_25_16_47_11_0001--s-0',
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-MLE-64-demos-sub-1-paper-version/hc_rand_vel_MLE_64_demos_sub_1_paper_version_2019_05_25_16_47_11_0002--s-0',
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-MLE-64-demos-sub-1-paper-version/hc_rand_vel_MLE_64_demos_sub_1_paper_version_2019_05_25_16_47_12_0003--s-0',
#     ],
#     (64,20): [
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-MLE-64-demos-sub-20-paper-version/hc_rand_vel_MLE_64_demos_sub_20_paper_version_2019_05_25_16_47_49_0000--s-0',
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-MLE-64-demos-sub-20-paper-version/hc_rand_vel_MLE_64_demos_sub_20_paper_version_2019_05_25_16_47_50_0001--s-0',
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-MLE-64-demos-sub-20-paper-version/hc_rand_vel_MLE_64_demos_sub_20_paper_version_2019_05_25_16_47_50_0002--s-0',
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-MLE-64-demos-sub-20-paper-version/hc_rand_vel_MLE_64_demos_sub_20_paper_version_2019_05_25_16_47_51_0003--s-0',
#     ],
#     (16,20): [
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-MLE-16-demos-sub-20-paper-version/hc_rand_vel_MLE_16_demos_sub_20_paper_version_2019_05_25_16_48_27_0000--s-0',
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-MLE-16-demos-sub-20-paper-version/hc_rand_vel_MLE_16_demos_sub_20_paper_version_2019_05_25_16_48_28_0001--s-0',
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-MLE-16-demos-sub-20-paper-version/hc_rand_vel_MLE_16_demos_sub_20_paper_version_2019_05_25_16_48_28_0002--s-0',
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-MLE-16-demos-sub-20-paper-version/hc_rand_vel_MLE_16_demos_sub_20_paper_version_2019_05_25_16_48_29_0003--s-0',
#     ],
#     (4,20): [
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-MLE-4-demos-sub-20-paper-version/hc_rand_vel_MLE_4_demos_sub_20_paper_version_2019_05_25_16_48_59_0000--s-0',
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-MLE-4-demos-sub-20-paper-version/hc_rand_vel_MLE_4_demos_sub_20_paper_version_2019_05_25_16_49_00_0001--s-0',
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-MLE-4-demos-sub-20-paper-version/hc_rand_vel_MLE_4_demos_sub_20_paper_version_2019_05_25_16_49_00_0002--s-0',
#         '/scratch/hdd001/home/kamyar/output/hc-rand-vel-MLE-4-demos-sub-20-paper-version/hc_rand_vel_MLE_4_demos_sub_20_paper_version_2019_05_25_16_49_01_0003--s-0',
#     ]
# }

# for method in [sa_irl_results, s_irl_results, bc_results]:
#     print('\n{}'.format(method['name']))
#     for amount in [(64,1), (64,20), (16,20), (4,20)]:
#         d = joblib.load()

def gather_run_costs_for_context_size(d, c_size):
    l = []
    for task_d in d.values():
        l.extend(task_d[c_size]['run_costs'])
    return l


# gather the results
def gather_results(method_paths):
    new_dict = {}
    for data_amount in amounts:
        print('\t{}'.format(data_amount))
        amount_dict = defaultdict(list)
        for path in method_paths[data_amount]:
            print('\t\tGathering: {}'.format(path))
            try:
                d = joblib.load(osp.join(path, 'all_eval_stats.pkl'))['all_eval_stats'][0]
            except:
                print('FAILED {}'.format(path))
                continue
            for c in context_sizes:
                l = gather_run_costs_for_context_size(d, c)
                amount_dict[c].extend(l)
        new_dict[data_amount] = amount_dict
    return new_dict

# # IF YOU WANT TO REGATHER RESULTS RUN THIS
# save_dict = {}
# # for method in [sa_irl_results, s_irl_results, bc_results]:
# # for method in [bc_results]:
# for method in [dagger_results]:
#     print(method['name'])
#     save_dict[method['name']] = gather_results(method)
# joblib.dump(save_dict, 'hc_rand_vel_save_dict_with_mse_dagger.pkl', compress=3)

# ELSE
# save_dict = joblib.load('hc_rand_vel_save_dict.pkl')
save_dict = joblib.load('hc_rand_vel_save_dict_with_mse_dagger.pkl')
print(save_dict.keys())
# save_dict.update(joblib.load('hc_rand_vel_save_dict_mse_bc.pkl'))
# save_dict = joblib.load('hc_rand_vel_save_dict_with_mse_bc.pkl')
for name, method_d in save_dict.items():
    print('\n')
    print(name)
    for amount, amount_d in method_d.items():
        print('\t{}'.format(amount))
        all_run_costs = []
        for context_size, costs in amount_d.items():
            all_run_costs.extend(costs)
        print('\t\tDelta: %.3f +/- %.3f' % (np.mean(all_run_costs)/1000, np.std(all_run_costs)/1000))


# plot some things
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data_amount_to_plot = (64,1)
context_means = defaultdict(list)
context_stds = defaultdict(list)
for name, method_d in save_dict.items():
    # print(name)
    # print(method_d)
    for context_size in context_sizes:
        context_means[name].append(
            np.mean(
                method_d[data_amount_to_plot][context_size]
            ) / 1000.0
        )
        context_stds[name].append(
            np.std(
                method_d[data_amount_to_plot][context_size]
            ) / 1000.0
        )
# print(context_means)
# print(context_stds)

fig, ax = plt.subplots(1)
ax.set_xlabel('Number of Context Trajectories')
ax.set_ylabel('Delta from Target Velocity')
# ax.set_ylim([0.0, 1.6])
ax.set_ylim([0.0, 0.75])

print(context_means.keys())
# print(len(context_means['bc']))

# ax.errorbar(
#     np.array(list(range(1,5))), context_means['Meta-BC'], context_stds['Meta-BC'],
#     elinewidth=2.0, capsize=4.0, barsabove=True, linewidth=2.0, label='Meta-BC'
# )
# ax.errorbar(
#     np.array(list(range(1,5))) + 0.03, context_means['Meta-IRL (state-action)'], context_stds['Meta-IRL (state-action)'],
#     elinewidth=2.0, capsize=4.0, barsabove=True, linewidth=2.0, label='Meta-IRL (state-action)'
# )
# ax.errorbar(
#     np.array(list(range(1,5))) + 0.06, context_means['Meta-IRL (state-action)'], context_stds['Meta-IRL (state-action)'],
#     elinewidth=2.0, capsize=4.0, barsabove=True, linewidth=2.0, label='Meta-IRL (state-only)'
# )
ax.errorbar(
    np.array(list(range(1,5))) + 0.06, context_means['Meta-Dagger'], context_stds['Meta-Dagger'],
    elinewidth=2.0, capsize=4.0, barsabove=True, linewidth=2.0, label='Meta-Dagger'
)

# lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.725, 0.1), shadow=False, ncol=3)
lgd = ax.legend(loc='upper right', shadow=False, ncol=1)
# plt.savefig('hc_context_size_plot.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig('hc_rand_vel_dagger_context_size_plot.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close()
