import numpy as np
import joblib
import os.path as osp

def load_stats(path):
    return joblib.load(osp.join(path, 'all_eval_stats.pkl'))['all_eval_stats'][0]

def compute_metrics(model_stats, base_stats, expert_stats):
    raw_perf = []
    custom_metric = []
    for task in model_stats:
        model_perf = np.mean(model_stats[task])
        base_perf = np.mean(base_stats[task])
        expert_perf = np.mean(expert_stats[task])
        if expert_perf < base_perf:
            print('shit')
            continue

        raw_perf.append(model_perf)
        cm = (model_perf - base_perf) / (expert_perf - base_perf)
        custom_metric.append(cm)
        # if cm > 1:
        #     print('fuck')
        #     print(model_perf)
        #     print(base_perf)
        #     print(expert_perf)
        #     print(model_perf - base_perf)
        #     print(expert_perf - base_perf)
        #     print('\n')
    # print(custom_metric)
    return np.mean(raw_perf), np.mean(custom_metric)

def compute_model_type_results(model_paths, base_stats, expert_stats):
    raw = []
    custom = []
    for p in model_paths:
        stats = load_stats(p)
        r, c = compute_metrics(stats, base_stats, expert_stats)
        raw.append(r)
        custom.append(c)
    return np.mean(raw), np.std(raw), np.mean(custom), np.std(custom)

expert_path = '/scratch/hdd001/home/kamyar/expert_demos/walker_agg_dyn_expert_for_test_tasks'

base_path = '/scratch/hdd001/home/kamyar/output/base-dynamics-walker/base_dynamics_walker_2019_05_20_15_13_56_0000--s-0/'

sa_paths = [
    '/scratch/hdd001/home/kamyar/output/walker-rand-dyn-np-airl-first-run-higher-rews-disc-512-no-terminal-no-save-larger-policy/walker_rand_dyn_np_airl_first_run_higher_rews_disc_512_no_terminal_no_save_larger_policy_2019_05_20_05_56_16_0002--s-0/',
    '/scratch/hdd001/home/kamyar/output/walker-rand-dyn-np-airl-first-run-higher-rews-disc-512-no-terminal-no-save-larger-policy/walker_rand_dyn_np_airl_first_run_higher_rews_disc_512_no_terminal_no_save_larger_policy_2019_05_20_05_56_18_0006--s-0/',
    '/scratch/hdd001/home/kamyar/output/walker-rand-dyn-np-airl-first-run-higher-rews-disc-512-no-terminal-no-save-larger-policy-another-seed/walker_rand_dyn_np_airl_first_run_higher_rews_disc_512_no_terminal_no_save_larger_policy_another_seed_2019_05_21_01_35_01_0000--s-0/',
]

s_paths = [
    '/scratch/hdd001/home/kamyar/output/walker-rand-dyn-np-airl-first-run-higher-rews-disc-512-no-terminal-no-save-larger-policy-state-only/walker_rand_dyn_np_airl_first_run_higher_rews_disc_512_no_terminal_no_save_larger_policy_state_only_2019_05_21_01_32_54_0000--s-0',
    '/scratch/hdd001/home/kamyar/output/walker-rand-dyn-np-airl-first-run-higher-rews-disc-512-no-terminal-no-save-larger-policy-state-only/walker_rand_dyn_np_airl_first_run_higher_rews_disc_512_no_terminal_no_save_larger_policy_state_only_2019_05_21_01_32_54_0001--s-0',
    '/scratch/hdd001/home/kamyar/output/walker-rand-dyn-np-airl-first-run-higher-rews-disc-512-no-terminal-no-save-larger-policy-state-only/walker_rand_dyn_np_airl_first_run_higher_rews_disc_512_no_terminal_no_save_larger_policy_state_only_2019_05_21_01_32_54_0002--s-0',
]

bc_paths = [
    '/scratch/hdd001/home/kamyar/output/correct-saving-paper-version-walker-np-bc-policy-same-as-np-airl/correct_saving_paper_version_walker_np_bc_policy_same_as_np_airl_2019_05_22_06_31_48_0000--s-0',
    '/scratch/hdd001/home/kamyar/output/correct-saving-paper-version-walker-np-bc-policy-same-as-np-airl/correct_saving_paper_version_walker_np_bc_policy_same_as_np_airl_2019_05_22_06_31_48_0001--s-0',
    '/scratch/hdd001/home/kamyar/output/correct-saving-paper-version-walker-np-bc-policy-same-as-np-airl/correct_saving_paper_version_walker_np_bc_policy_same_as_np_airl_2019_05_22_06_31_49_0002--s-0',
]

dagger_paths = [
    '/scratch/hdd001/home/kamyar/output/longer-run-correct-walker-rand-dyn-meta-dagger-use-z-sample-det-expert-MSE-64-demos-100-updates-per-call/longer_run_correct_walker_rand_dyn_meta_dagger_use_z_sample_det_expert_MSE_64_demos_100_updates_per_call_2019_07_29_04_17_45_0000--s-0',
    '/scratch/hdd001/home/kamyar/output/longer-run-correct-walker-rand-dyn-meta-dagger-use-z-sample-det-expert-MSE-64-demos-100-updates-per-call/longer_run_correct_walker_rand_dyn_meta_dagger_use_z_sample_det_expert_MSE_64_demos_100_updates_per_call_2019_07_29_04_17_47_0001--s-0',
    '/scratch/hdd001/home/kamyar/output/longer-run-correct-walker-rand-dyn-meta-dagger-use-z-sample-det-expert-MSE-64-demos-100-updates-per-call/longer_run_correct_walker_rand_dyn_meta_dagger_use_z_sample_det_expert_MSE_64_demos_100_updates_per_call_2019_07_29_04_17_47_0002--s-0',
]

base_stats = load_stats(base_path)
expert_stats = load_stats(expert_path)

# print('\nSA RESULTS:')
# r_mean, r_std, c_mean, c_std = compute_model_type_results(sa_paths, base_stats, expert_stats)
# print('Raw Perf: {} +/- {}'.format(r_mean, r_std))
# print('Custom: {} +/- {}'.format(c_mean, c_std))

# print('\nS RESULTS:')
# r_mean, r_std, c_mean, c_std = compute_model_type_results(s_paths, base_stats, expert_stats)
# print('Raw Perf: {} +/- {}'.format(r_mean, r_std))
# print('Custom: {} +/- {}'.format(c_mean, c_std))

# print('\nBC RESULTS:')
# r_mean, r_std, c_mean, c_std = compute_model_type_results(bc_paths, base_stats, expert_stats)
# print('Raw Perf: {} +/- {}'.format(r_mean, r_std))
# print('Custom: {} +/- {}'.format(c_mean, c_std))

print('\nDagger RESULTS:')
r_mean, r_std, c_mean, c_std = compute_model_type_results(dagger_paths, base_stats, expert_stats)
print('Raw Perf: {} +/- {}'.format(r_mean, r_std))
print('Custom: {} +/- {}'.format(c_mean, c_std))
