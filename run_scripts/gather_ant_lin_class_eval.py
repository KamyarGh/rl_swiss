import joblib
import os.path as osp
import numpy as np

bc_paths = [
    # bc correct
    '/scratch/hdd001/home/kamyar/output/final-paper-version-ant-lin-class-np-bc/final_paper_version_ant_lin_class_np_bc_2019_05_23_01_05_47_0000--s-0',
    '/scratch/hdd001/home/kamyar/output/final-paper-version-ant-lin-class-np-bc/final_paper_version_ant_lin_class_np_bc_2019_05_23_01_05_48_0001--s-0',
    '/scratch/hdd001/home/kamyar/output/final-paper-version-ant-lin-class-np-bc/final_paper_version_ant_lin_class_np_bc_2019_05_23_01_05_48_0002--s-0',
]

s_paths = [
    # state-only
    '/scratch/hdd001/home/kamyar/output/reproduce-best-ant-lin-class-eval-det-state-only/reproduce_best_ant_lin_class_eval_det_state_only_2019_05_21_01_58_15_0000--s-0',
    '/scratch/hdd001/home/kamyar/output/reproduce-best-ant-lin-class-eval-det-state-only/reproduce_best_ant_lin_class_eval_det_state_only_2019_05_21_01_58_15_0001--s-0',
    '/scratch/hdd001/home/kamyar/output/reproduce-best-ant-lin-class-eval-det-state-only/reproduce_best_ant_lin_class_eval_det_state_only_2019_05_21_01_58_16_0002--s-0'
]

sa_paths = [
    # state-action
    '/scratch/hdd001/home/kamyar/output/reproduce-best-ant-lin-class-eval-det-larger-policy/reproduce_best_ant_lin_class_eval_det_larger_policy_2019_05_19_16_52_44_0002--s-0',
    '/scratch/hdd001/home/kamyar/output/reproduce-best-ant-lin-class-eval-det-larger-policy/reproduce_best_ant_lin_class_eval_det_larger_policy_2019_05_19_16_52_44_0003--s-0',
    '/scratch/hdd001/home/kamyar/output/reproduce-best-ant-lin-class-eval-det-another-seed/reproduce_best_ant_lin_class_eval_det_another_seed_2019_05_21_01_59_28_0000--s-0',
]

dagger_paths = [
    '/scratch/hdd001/home/kamyar/output/correct-3-ant-lin-class-meta-dagger-use-z-sample-det-expert-MSE-64-demos-100-updates-per-call/correct_3_ant_lin_class_meta_dagger_use_z_sample_det_expert_MSE_64_demos_100_updates_per_call_2019_07_31_07_59_01_0000--s-0',
    '/scratch/hdd001/home/kamyar/output/correct-3-ant-lin-class-meta-dagger-use-z-sample-det-expert-MSE-64-demos-100-updates-per-call/correct_3_ant_lin_class_meta_dagger_use_z_sample_det_expert_MSE_64_demos_100_updates_per_call_2019_07_31_07_59_02_0001--s-0',
    '/scratch/hdd001/home/kamyar/output/correct-3-ant-lin-class-meta-dagger-use-z-sample-det-expert-MSE-64-demos-100-updates-per-call/correct_3_ant_lin_class_meta_dagger_use_z_sample_det_expert_MSE_64_demos_100_updates_per_call_2019_07_31_07_59_02_0002--s-0',
]

def load_stats(p):
    a = joblib.load(osp.join(p, 'all_eval_stats.pkl'))['all_eval_stats'][0]
    return a

def compute_success_and_no_op(stats):
    print(stats)
    all_succ = stats['all_success_transitions']
    all_no_op = stats['all_no_op_transitions']

    new_succ = [s[:8] for s in all_succ]
    new_no_op = [s[:8] for s in all_no_op]

    return np.mean(new_succ), np.mean(new_no_op)

def compute_method_results(method_paths):
    all_succs = []
    all_no_op = []
    for p in method_paths:
        succ, no_op = compute_success_and_no_op(load_stats(p))
        all_succs.append(succ)
        all_no_op.append(no_op)
    
    print('Succ: {} +/- {}'.format(np.mean(all_succs), np.std(all_succs)))
    print('No-Op: {} +/- {}'.format(np.mean(all_no_op), np.std(all_no_op)))




def compute_cool(stats):
    print(stats.keys())
    all_succ = stats['all_success_transitions']
    all_succ = np.array(all_succ)
    first = all_succ[:,0:1]
    all_succ = all_succ - first

    mean = np.mean(all_succ, axis=0)
    return mean

def compute_method_cool(method_paths):
    all_cools = []
    for p in method_paths:
        print(p)
        cool = compute_cool(load_stats(p))
        all_cools.append(cool)
        # print(cool)
        # print(cool.shape)
    print(np.mean(np.array(all_cools), axis=0))
    print(np.std(np.array(all_cools), axis=0))


# compute_method_cool(s_paths)
# compute_method_cool(sa_paths)


# print('BC')
# compute_method_results(bc_paths)

# print('S')
# compute_method_results(s_paths)

# print('SA')
# compute_method_results(sa_paths)

# dagger
compute_method_results(dagger_paths)
