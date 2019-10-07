import joblib
from os import path as osp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

s_a_paths = {
    # 4 demos
    4: [
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-airl-4-demos-each-1-context-only-state-action/paper_version_ant_rand_goal_np_airl_4_demos_each_1_context_only_state_action_2019_04_29_02_49_31_0000--s-0',
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-airl-4-demos-each-1-context-only-state-action/paper_version_ant_rand_goal_np_airl_4_demos_each_1_context_only_state_action_2019_04_29_02_49_32_0001--s-0',
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-airl-4-demos-each-1-context-only-state-action/paper_version_ant_rand_goal_np_airl_4_demos_each_1_context_only_state_action_2019_04_29_02_49_32_0002--s-0',
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-airl-4-demos-each-1-context-only-state-action/paper_version_ant_rand_goal_np_airl_4_demos_each_1_context_only_state_action_2019_04_29_02_49_33_0003--s-0',
    ],
    # 16 demos
    16: [
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-airl-16-demos-each-1-context-only-state-action/paper_version_ant_rand_goal_np_airl_16_demos_each_1_context_only_state_action_2019_04_29_02_49_57_0000--s-0',
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-airl-16-demos-each-1-context-only-state-action/paper_version_ant_rand_goal_np_airl_16_demos_each_1_context_only_state_action_2019_04_29_02_49_58_0001--s-0',
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-airl-16-demos-each-1-context-only-state-action/paper_version_ant_rand_goal_np_airl_16_demos_each_1_context_only_state_action_2019_04_29_02_49_59_0002--s-0',
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-airl-16-demos-each-1-context-only-state-action/paper_version_ant_rand_goal_np_airl_16_demos_each_1_context_only_state_action_2019_04_29_02_49_59_0003--s-0',
    ],
    # 64 demos
    64: [
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-airl-64-demos-each-1-context-only-state-action/paper_version_ant_rand_goal_np_airl_64_demos_each_1_context_only_state_action_2019_04_29_02_50_20_0000--s-0',
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-airl-64-demos-each-1-context-only-state-action/paper_version_ant_rand_goal_np_airl_64_demos_each_1_context_only_state_action_2019_04_29_02_50_20_0001--s-0',
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-airl-64-demos-each-1-context-only-state-action/paper_version_ant_rand_goal_np_airl_64_demos_each_1_context_only_state_action_2019_04_29_02_50_21_0002--s-0',
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-airl-64-demos-each-1-context-only-state-action/paper_version_ant_rand_goal_np_airl_64_demos_each_1_context_only_state_action_2019_04_29_02_50_21_0003--s-0',
    ]
}

s_paths = {
    # 4 demos
    4: [
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-airl-4-demos-each-1-context-only-state-only/paper_version_ant_rand_goal_np_airl_4_demos_each_1_context_only_state_only_2019_04_29_02_49_01_0000--s-0',
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-airl-4-demos-each-1-context-only-state-only/paper_version_ant_rand_goal_np_airl_4_demos_each_1_context_only_state_only_2019_04_29_02_49_02_0001--s-0',
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-airl-4-demos-each-1-context-only-state-only/paper_version_ant_rand_goal_np_airl_4_demos_each_1_context_only_state_only_2019_04_29_02_49_02_0002--s-0',
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-airl-4-demos-each-1-context-only-state-only/paper_version_ant_rand_goal_np_airl_4_demos_each_1_context_only_state_only_2019_04_29_02_49_03_0003--s-0',
    ],
    # 16 demos
    16: [
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-airl-16-demos-each-1-context-only-state-only/paper_version_ant_rand_goal_np_airl_16_demos_each_1_context_only_state_only_2019_04_29_02_48_39_0000--s-0',
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-airl-16-demos-each-1-context-only-state-only/paper_version_ant_rand_goal_np_airl_16_demos_each_1_context_only_state_only_2019_04_29_02_48_39_0001--s-0',
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-airl-16-demos-each-1-context-only-state-only/paper_version_ant_rand_goal_np_airl_16_demos_each_1_context_only_state_only_2019_04_29_02_48_40_0002--s-0',
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-airl-16-demos-each-1-context-only-state-only/paper_version_ant_rand_goal_np_airl_16_demos_each_1_context_only_state_only_2019_04_29_02_48_40_0003--s-0',
    ],
    # 64 demos
    64: [
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-airl-64-demos-each-1-context-only-state-only/paper_version_ant_rand_goal_np_airl_64_demos_each_1_context_only_state_only_2019_04_29_02_47_59_0000--s-0',
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-airl-64-demos-each-1-context-only-state-only/paper_version_ant_rand_goal_np_airl_64_demos_each_1_context_only_state_only_2019_04_29_02_47_59_0001--s-0',
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-airl-64-demos-each-1-context-only-state-only/paper_version_ant_rand_goal_np_airl_64_demos_each_1_context_only_state_only_2019_04_29_02_48_00_0002--s-0',
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-airl-64-demos-each-1-context-only-state-only/paper_version_ant_rand_goal_np_airl_64_demos_each_1_context_only_state_only_2019_04_29_02_48_00_0003--s-0',
    ]
}

# bc_paths = {
#     # 4 demos
#     4: [
#         '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-bc-4-demos-each-1-context-only-with-saving/paper_version_ant_rand_goal_np_bc_4_demos_each_1_context_only_with_saving_2019_04_29_13_12_59_0000--s-0',
#         '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-bc-4-demos-each-1-context-only-with-saving/paper_version_ant_rand_goal_np_bc_4_demos_each_1_context_only_with_saving_2019_04_29_13_13_00_0001--s-0',
#         '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-bc-4-demos-each-1-context-only-with-saving/paper_version_ant_rand_goal_np_bc_4_demos_each_1_context_only_with_saving_2019_04_29_13_13_00_0002--s-0',
#         '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-bc-4-demos-each-1-context-only-with-saving/paper_version_ant_rand_goal_np_bc_4_demos_each_1_context_only_with_saving_2019_04_29_13_13_01_0003--s-0',
#     ],
#     # 16 demos
#     16: [
#         '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-bc-16-demos-each-1-context-only-with-saving/paper_version_ant_rand_goal_np_bc_16_demos_each_1_context_only_with_saving_2019_04_29_13_12_25_0000--s-0',
#         '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-bc-16-demos-each-1-context-only-with-saving/paper_version_ant_rand_goal_np_bc_16_demos_each_1_context_only_with_saving_2019_04_29_13_12_26_0001--s-0',
#         '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-bc-16-demos-each-1-context-only-with-saving/paper_version_ant_rand_goal_np_bc_16_demos_each_1_context_only_with_saving_2019_04_29_13_12_26_0002--s-0',
#         '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-bc-16-demos-each-1-context-only-with-saving/paper_version_ant_rand_goal_np_bc_16_demos_each_1_context_only_with_saving_2019_04_29_13_12_27_0003--s-0',
#     ],
#     # 64 demos
#     64: [
#         '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-bc-64-demos-each-1-context-only-with-saving/paper_version_ant_rand_goal_np_bc_64_demos_each_1_context_only_with_saving_2019_04_29_13_13_37_0000--s-0',
#         '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-bc-64-demos-each-1-context-only-with-saving/paper_version_ant_rand_goal_np_bc_64_demos_each_1_context_only_with_saving_2019_04_29_13_13_37_0001--s-0',
#         '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-bc-64-demos-each-1-context-only-with-saving/paper_version_ant_rand_goal_np_bc_64_demos_each_1_context_only_with_saving_2019_04_29_13_13_38_0002--s-0',
#         '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-bc-64-demos-each-1-context-only-with-saving/paper_version_ant_rand_goal_np_bc_64_demos_each_1_context_only_with_saving_2019_04_29_13_13_38_0003--s-0'
#     ]
# }

# MSE version
bc_paths = {
    # 4 demos
    4: [
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-bc-4-demos-each-1-context-only-with-saving-mse-version-correct/paper_version_ant_rand_goal_np_bc_4_demos_each_1_context_only_with_saving_mse_version_correct_2019_05_25_15_07_24_0000--s-0',
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-bc-4-demos-each-1-context-only-with-saving-mse-version-correct/paper_version_ant_rand_goal_np_bc_4_demos_each_1_context_only_with_saving_mse_version_correct_2019_05_25_15_07_25_0001--s-0',
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-bc-4-demos-each-1-context-only-with-saving-mse-version-correct/paper_version_ant_rand_goal_np_bc_4_demos_each_1_context_only_with_saving_mse_version_correct_2019_05_25_15_07_25_0002--s-0',
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-bc-4-demos-each-1-context-only-with-saving-mse-version-correct/paper_version_ant_rand_goal_np_bc_4_demos_each_1_context_only_with_saving_mse_version_correct_2019_05_25_15_07_26_0003--s-0',
    ],

    # 16 demos
    16: [
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-bc-16-demos-each-1-context-only-with-saving-mse-version-correct/paper_version_ant_rand_goal_np_bc_16_demos_each_1_context_only_with_saving_mse_version_correct_2019_05_25_15_07_56_0000--s-0',
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-bc-16-demos-each-1-context-only-with-saving-mse-version-correct/paper_version_ant_rand_goal_np_bc_16_demos_each_1_context_only_with_saving_mse_version_correct_2019_05_25_15_07_57_0001--s-0',
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-bc-16-demos-each-1-context-only-with-saving-mse-version-correct/paper_version_ant_rand_goal_np_bc_16_demos_each_1_context_only_with_saving_mse_version_correct_2019_05_25_15_07_57_0002--s-0',
        '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-bc-16-demos-each-1-context-only-with-saving-mse-version-correct/paper_version_ant_rand_goal_np_bc_16_demos_each_1_context_only_with_saving_mse_version_correct_2019_05_25_15_07_58_0003--s-0',
    ],

    # 64 demos
    64: [
    '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-bc-64-demos-each-1-context-only-with-saving-mse-version-correct/paper_version_ant_rand_goal_np_bc_64_demos_each_1_context_only_with_saving_mse_version_correct_2019_05_25_15_08_17_0000--s-0',
    '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-bc-64-demos-each-1-context-only-with-saving-mse-version-correct/paper_version_ant_rand_goal_np_bc_64_demos_each_1_context_only_with_saving_mse_version_correct_2019_05_25_15_08_17_0001--s-0',
    '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-bc-64-demos-each-1-context-only-with-saving-mse-version-correct/paper_version_ant_rand_goal_np_bc_64_demos_each_1_context_only_with_saving_mse_version_correct_2019_05_25_15_08_18_0002--s-0',
    '/scratch/hdd001/home/kamyar/output/paper-version-ant-rand-goal-np-bc-64-demos-each-1-context-only-with-saving-mse-version-correct/paper_version_ant_rand_goal_np_bc_64_demos_each_1_context_only_with_saving_mse_version_correct_2019_05_25_15_08_18_0003--s-0',
    ]
}

# Dagger MSE version
dagger_paths = {
    64: [
        '/scratch/hdd001/home/kamyar/output/correct-ant-rand-goal-meta-dagger-use-z-sample-det-expert-MSE-64-demos-100-updates-per-call/correct_ant_rand_goal_meta_dagger_use_z_sample_det_expert_MSE_64_demos_100_updates_per_call_2019_07_28_17_03_34_0000--s-0',
        '/scratch/hdd001/home/kamyar/output/correct-ant-rand-goal-meta-dagger-use-z-sample-det-expert-MSE-64-demos-100-updates-per-call/correct_ant_rand_goal_meta_dagger_use_z_sample_det_expert_MSE_64_demos_100_updates_per_call_2019_07_28_17_03_49_0001--s-0',
        '/scratch/hdd001/home/kamyar/output/correct-ant-rand-goal-meta-dagger-use-z-sample-det-expert-MSE-64-demos-100-updates-per-call/correct_ant_rand_goal_meta_dagger_use_z_sample_det_expert_MSE_64_demos_100_updates_per_call_2019_07_28_17_03_49_0002--s-0',
        '/scratch/hdd001/home/kamyar/output/correct-ant-rand-goal-meta-dagger-use-z-sample-det-expert-MSE-64-demos-100-updates-per-call/correct_ant_rand_goal_meta_dagger_use_z_sample_det_expert_MSE_64_demos_100_updates_per_call_2019_07_28_17_03_49_0003--s-0',
    ]
}

sparse_rew_rl_paths = {
    64: [
        '/scratch/hdd001/home/kamyar/output/ant-rand-goal-sparse-rewards/ant_rand_goal_sparse_rewards_2019_07_31_02_42_25_0000--s-0',
        '/scratch/hdd001/home/kamyar/output/ant-rand-goal-sparse-rewards/ant_rand_goal_sparse_rewards_2019_07_31_02_49_30_0001--s-0',
        '/scratch/hdd001/home/kamyar/output/ant-rand-goal-sparse-rewards/ant_rand_goal_sparse_rewards_2019_07_31_02_49_30_0002--s-0',
        '/scratch/hdd001/home/kamyar/output/ant-rand-goal-sparse-rewards/ant_rand_goal_sparse_rewards_2019_07_31_02_50_29_0003--s-0',    
    ]
}


tasks = [
    [ 1.99,  0.2 ],
    [ 1.76,  0.94],
    [ 1.27,  1.55],
    [ 0.58,  1.91],
    [-0.2 ,  1.99],
    [-0.94,  1.76],
    [-1.55,  1.27],
    [-1.91,  0.58],
    [-1.99, -0.2 ],
    [-1.76, -0.94],
    [-1.27, -1.55],
    [-0.58, -1.91],
    [ 0.2 , -1.99],
    [ 0.94, -1.76],
    [ 1.55, -1.27],
    [ 1.91, -0.58]
]
X = np.linspace(np.pi/32, 2*np.pi + np.pi/32, num=16, endpoint=False)
# demo_sizes = [4, 16, 64]
demo_sizes = [64]

def gather_method_results(method_dict):
    results = {}
    for d_size in demo_sizes:
        paths = method_dict[d_size]
        seed_dicts = [joblib.load(osp.join(path, 'all_eval_stats.pkl')) for path in paths]
        task_means, task_stds = [], []
        for task in tasks:
            min_dists = []
            for s_dict in seed_dicts:
                min_dists.extend(
                    s_dict['all_eval_stats'][0][tuple(task)][1]['min_dists']
                )
            task_means.append(np.mean(min_dists))
            task_stds.append(np.std(min_dists))
        d_size_result = {
            'means': task_means,
            'stds': task_stds
        }
        results[d_size] = d_size_result
    return results

def get_aggregate_result(method_dict):
    results = {}
    for d_size in demo_sizes:
        paths = method_dict[d_size]
        seed_dicts = [joblib.load(osp.join(path, 'all_eval_stats.pkl')) for path in paths]
        task_results = []
        for task in tasks:
            min_dists = []
            for s_dict in seed_dicts:
                min_dists.append(
                    np.mean(s_dict['all_eval_stats'][0][tuple(task)][1]['min_dists'])
                )
            task_results.append(min_dists)
        task_results = np.array(task_results)
        task_results = np.mean(task_results, axis=0)
        print('%.2f +/- %.2f' % (np.mean(task_results), np.std(task_results)))
    return results

s_a_results = get_aggregate_result(s_a_paths)
s_results = get_aggregate_result(s_paths)
bc_results = get_aggregate_result(bc_paths)
dagger_results = get_aggregate_result(dagger_paths)

# s_a_results = gather_method_results(s_a_paths)
# s_results = gather_method_results(s_paths)
# bc_results = gather_method_results(bc_paths)
# dagger_results = gather_method_results(dagger_paths)
# sparse_rew_rl_results = gather_method_results(sparse_rew_rl_paths)

# joblib.dump(
#     {
#         # 's_a_results': s_a_results,
#         # 's_results': s_results,
#         # 'bc_results': bc_results,
#         # 'dagger_results': dagger_results,
#         'sparese_rew_rl_results': sparse_rew_rl_results,
#     },
#     'plots/ant_rand_goal_sparse_rew_rl_results_save.pkl',
#     compress=3
# )

plt.rcParams["figure.figsize"] = [6,1]
# plt.rcParams["figure.figsize"] = [6,2]
# for d_size in [4, 16, 64]:
# for d_size in [64]:
    # state-action plot
    # fig, ax = plt.subplots(1)
    # ax.errorbar(
    #     X,
    #     s_a_results[d_size]['means'],
    #     s_a_results[d_size]['stds'],
    #     elinewidth=2.0, capsize=4.0, barsabove=True, linewidth=2.0, label='Meta-AIRL (state-action)',
    #     color='royalblue'
    # )
    # ax.set_ylim([0.0, 2.25])
    # ax.set_xticks([0.0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    # ax.set_xticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    # ax.set_yticks([0.0, 1.0, 2.0])
    # # lgd = ax.legend(loc='upper right', bbox_to_anchor=(0.725, 0.1), shadow=False, ncol=3)
    # # plt.savefig('plots/paper_results/ant_rand_goal/s_a_demo_%d.png'%d_size, bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.savefig('plots/paper_results/ant_rand_goal/s_a_demo_%d.png'%d_size, bbox_inches='tight')
    # plt.close()

    # state plot
    # fig, ax = plt.subplots(1)
    # ax.errorbar(
    #     X,
    #     s_results[d_size]['means'],
    #     s_results[d_size]['stds'],
    #     elinewidth=2.0, capsize=4.0, barsabove=True, linewidth=2.0, label='Meta-AIRL (state-only)',
    #     color='forestgreen'
    # )
    # ax.set_ylim([0.0, 2.25])
    # ax.set_xticks([0.0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    # ax.set_xticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    # ax.set_yticks([0.0, 1.0, 2.0])
    # # lgd = ax.legend(loc='upper right', bbox_to_anchor=(0.725, 0.1), shadow=False, ncol=3)
    # # plt.savefig('plots/paper_results/ant_rand_goal/s_demo_%d.png'%d_size, bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.savefig('plots/paper_results/ant_rand_goal/s_demo_%d.png'%d_size, bbox_inches='tight')
    # plt.close()

    # bc plot
    # fig, ax = plt.subplots(1)
    # ax.errorbar(
    #     X,
    #     bc_results[d_size]['means'],
    #     bc_results[d_size]['stds'],
    #     elinewidth=2.0, capsize=4.0, barsabove=True, linewidth=2.0, label='Meta-BC',
    #     color='tomato'
    # )
    # ax.set_ylim([0.0, 2.25])
    # ax.set_xticks([0.0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    # ax.set_xticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    # ax.set_yticks([0.0, 1.0, 2.0])
    # # lgd = ax.legend(loc='upper right', bbox_to_anchor=(0.725, 0.1), shadow=False, ncol=3)
    # # plt.savefig('plots/paper_results/ant_rand_goal/bc_demo_%d.png'%d_size, bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.savefig('plots/paper_results/ant_rand_goal/bc_demo_%d.png'%d_size, bbox_inches='tight')
    # plt.close()

    # dagger plot
    # fig, ax = plt.subplots(1)
    # ax.errorbar(
    #     X,
    #     dagger_results[d_size]['means'],
    #     dagger_results[d_size]['stds'],
    #     elinewidth=2.0, capsize=4.0, barsabove=True, linewidth=2.0, label='Meta-Dagger',
    #     color='purple'
    # )
    # ax.set_ylim([0.0, 2.25])
    # ax.set_xticks([0.0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    # ax.set_xticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    # ax.set_yticks([0.0, 1.0, 2.0])
    # # lgd = ax.legend(loc='upper right', bbox_to_anchor=(0.725, 0.1), shadow=False, ncol=3)
    # # plt.savefig('plots/paper_results/ant_rand_goal/bc_demo_%d.png'%d_size, bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.savefig('plots/paper_results/ant_rand_goal/dagger_demo_%d.png'%d_size, bbox_inches='tight')
    # plt.close()

    # sparse reward rl
    # fig, ax = plt.subplots(1)
    # ax.errorbar(
    #     X,
    #     sparse_rew_rl_results[d_size]['means'],
    #     sparse_rew_rl_results[d_size]['stds'],
    #     elinewidth=2.0, capsize=4.0, barsabove=True, linewidth=2.0, label='Sparse-Rew-RL',
    #     color='red'
    # )
    # ax.set_ylim([0.0, 2.25])
    # ax.set_xticks([0.0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    # ax.set_xticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    # ax.set_yticks([0.0, 1.0, 2.0])
    # # lgd = ax.legend(loc='upper right', bbox_to_anchor=(0.725, 0.1), shadow=False, ncol=3)
    # # plt.savefig('plots/paper_results/ant_rand_goal/bc_demo_%d.png'%d_size, bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.savefig('plots/paper_results/ant_rand_goal/sparse_rew_rl.png', bbox_inches='tight')
    # plt.close()
