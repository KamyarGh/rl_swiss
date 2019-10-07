import joblib
from os import path as osp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NUM_TRAJS_PER_PLOT = -1
# targets = np.array([
#     [2.0, 0.0],
#     # [1.41, 1.41],
#     [0.0, 2.0],
#     # [-1.41, 1.41],
#     [-2.0, 0.0],
#     # [-1.41, -1.41],
#     [0.0, -2.0],
#     # [1.41, -1.41]
# ])
targets = np.array([
    [4.0, 0.0],
    # [1.41, 1.41],
    [0.0, 4.0],
    # [-1.41, 1.41],
    [-4.0, 0.0],
    # [-1.41, -1.41],
    [0.0, -4.0],
    # [1.41, -1.41]
])
targets = np.array([
    [3.5, 0.0],
    # [1.41, 1.41],
    [0.0, 3.5],
    # [-1.41, 1.41],
    [-3.5, 0.0],
    # [-1.41, -1.41],
    [0.0, -3.5],
    # [1.41, -1.41]
])

# xy_mean = np.array([[-1.05384796e-02, -5.11796410e-03]])
# xy_std = np.array([[1.19729262, 1.21527848]])

# xy_mean = np.array([[-2.67636336e-02, 2.29878615e-03]])
# xy_std = np.array([[0.94791655, 0.96754038]])

# 4 distance
xy_mean = np.array([[-3.33176639e-02, -1.06844519e-02]])
xy_std = np.array([[1.94858432, 1.92371848]])

def make_the_plot(trajs_list, save_path, title=''):
    if NUM_TRAJS_PER_PLOT > 0:
        trajs_list = trajs_list[:NUM_TRAJS_PER_PLOT]

    fig, ax = plt.subplots(1)
    ax.scatter(targets[:,0], targets[:,1], color='royalblue', s=40)
    ax.scatter([0], [0], color='magenta', s=40)
    for traj in trajs_list:
        ax.plot(traj[:,0], traj[:,1])
    # ax.set_xlim([-2.5, 2.5])
    # ax.set_ylim([-2.5, 2.5])
    ax.set_xlim([-4.5, 4.5])
    ax.set_ylim([-4.5, 4.5])
    ax.set_aspect('equal')
    ax.set_title(title)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-fairl-rew-search/'
    # save_dir = 'plots/junk_vis/fairl_multi_plot'
    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-airl-rew-search/'
    # save_dir = 'plots/junk_vis/airl_multi_plot'

    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-fairl-rew-search-32-det-demos-per-task/'
    # save_dir = 'plots/junk_vis/multi_ant_fairl_32_det_demos_plot'
    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-airl-rew-search-32-det-demos-per-task/'
    # save_dir = 'plots/junk_vis/multi_ant_airl_32_det_demos_plot'
    import os
    import json

    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-fairl-rew-search-32-det-demos-per-task-grad-pen-search/'
    # save_dir = 'plots/junk_vis/multi_ant_fairl_32_det_demos_grad_pen_search'

    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-airl-rew-search-32-det-demos-per-task-even-lower-grad-pen-search/'
    # save_dir = 'plots/junk_vis/multi_ant_airl_32_det_demos_even_lower_grad_pen_search'
    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-fairl-rew-search-32-det-demos-per-task-even-lower-grad-pen-search/'
    # save_dir = 'plots/junk_vis/multi_ant_fairl_32_det_demos_even_lower_grad_pen_search'

    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-fairl-32-det-demos-per-task-low-grad-pen-and-high-rew-scale-hype-search-0'
    # save_dir = 'plots/junk_vis/multi_ant_fairl_32_det_demos_log_grad_high_rew_hype_search_0'

    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-4-directions-fairl-32-det-demos-per-task-hype-search-0-rb-size-3200-correct-final'
    # save_dir = 'plots/junk_vis/multi_ant_fairl_32_det_demos_4_dir_hype_search_0_rb_size_3200'

    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-4-directions-fairl-32-det-demos-per-task-hype-search-1-rb-size-3200-correct-final'
    # save_dir = 'plots/junk_vis/multi_ant_fairl_32_det_demos_4_dir_hype_search_1_rb_size_3200'

    # 4 distance
    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-4-directions-4-distance-fairl-32-det-demos-per-task-hype-search-1-rb-size-3200-correct-final'
    # save_dir = 'plots/junk_vis/multi_ant_fairl_32_det_demos_4_dir_4_distance_hype_search_1_rb_size_3200'

    # 4 distance rel pos
    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-rel-pos-4-directions-4-distance-fairl-32-det-demos-per-task-hype-search-1-rb-size-3200-correct-final-disc-512-3-relu'
    # save_dir = 'plots/junk_vis/multi_ant_fairl_32_det_demos_rel_pos_4_dir_4_distance_hype_search_1_rb_size_4800_disc_512_3_relu'

    # exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-rel-pos-4-directions-4-distance-fairl-32-det-demos-per-task-hype-search-1-rb-size-3200-correct-final-disc-512-3-relu-high-rew-search'
    # save_dir = 'plots/junk_vis/multi_ant_fairl_32_det_demos_rel_pos_4_dir_4_distance_hype_search_1_rb_size_4800_disc_512_3_relu_high_rew_search'

    # tiny models ----
    exp_path = '/scratch/hdd001/home/kamyar/output/multi-target-ant-rel-pos-with-termination-small-models-fairl-correct'
    save_dir = 'plots/junk_vis/multi_ant_tiny_fairl'

    os.makedirs(save_dir, exist_ok=True)

    all_eval_dicts = joblib.load(osp.join(exp_path, 'det_eval_False.pkl'))

    for sub_exp, eval_dict in all_eval_dicts.items():
        trajs = eval_dict['path_trajs']
        unnorm_trajs = []
        for traj in trajs:
            # unnorm_trajs.append(traj * xy_std + xy_mean)
            unnorm_trajs.append(traj)
        with open(os.path.join(exp_path, sub_exp, 'variant.json')) as f:
            variant = json.loads(f.read())
        rew = variant['policy_params']['reward_scale']
        gp = variant['algo_params']['grad_pen_weight']
        title = 'rew_%d_grad_pen_%.2f' % (rew, gp)
        # make_the_plot(unnorm_trajs, osp.join(save_dir, 'map_'+sub_exp+'.png'), title=title)
        make_the_plot(unnorm_trajs, osp.join(save_dir, 'map_'+title+'.png'), title=title)


    # plot the expert demonstration trajectories
    # import yaml
    # EXPERT_LISTING_YAML_PATH = '/h/kamyar/oorl_rlkit/rlkit/torch/irl/experts.yaml'
    # # expert_name = 'ant_multi_valid_target_demos_8_target_8_each_no_sub'
    # # expert_name = 'deterministic_ant_multi_valid_target_demos_8_target_8_each_no_sub'
    # # expert_name = 'deterministic_ant_multi_valid_target_demos_8_target_16_each_no_sub'
    # # expert_name = 'deterministic_ant_multi_valid_target_demos_8_target_24_each_no_sub'
    # # expert_name = 'deterministic_ant_multi_valid_target_demos_8_target_24_each_no_sub_path_len_50'
    # # expert_name = 'deterministic_ant_multi_valid_target_demos_8_target_16_each_no_sub_path_len_50'
    # expert_name = 'deterministic_ant_multi_valid_target_demos_8_target_32_each_no_sub_path_len_50'
    # with open(EXPERT_LISTING_YAML_PATH, 'r') as f:
    #         listings = yaml.load(f.read())
    # expert_dir = listings[expert_name]['exp_dir']
    # specific_run = listings[expert_name]['seed_runs'][0]
    # file_to_load = osp.join(expert_dir, specific_run, 'extra_data.pkl')
    
    # file_to_load = '/scratch/hdd001/home/kamyar/output/test-128-75-steps-ant-multi-4-directions-4-distance-32-det-demos-per-task-no-sub-path-len-100/test_128_75_steps_ant_multi_4_directions_4_distance_32_det_demos_per_task_no_sub_path_len_100_2019_05_16_01_04_47_0000--s-0/extra_data.pkl'
    # extra_data = joblib.load(file_to_load)
    # expert_buffer = extra_data['train']

    # # xy = expert_buffer._observations[:6400][:,-3:-1]
    # # xy = expert_buffer._observations[:19200][:,-3:-1]
    # xy = expert_buffer._observations[:9600][:,-3:-1]
    # # xy = expert_buffer._observations[:6400][:,-3:-1]
    # # xy = expert_buffer._observations[:12800][:,-3:-1]
    
    # trajs = np.split(xy, 64)
    # # make_the_plot(trajs, 'plots/junk_vis/%s.png' % expert_name)
    # make_the_plot(trajs, 'plots/junk_vis/test_4_dir_expert.png')
