import os
import json
from shutil import copyfile
from os import makedirs
import joblib


def do_the_copy(exp_log_dir, save_dir, exp_name):
    try:
        makedirs(save_dir)
    except:
        pass

    for d in os.listdir(exp_log_dir):
        path = os.path.join(exp_log_dir, d)
        print(path)
        with open(os.path.join(path, 'variant.json'), 'r') as f:
            spec_string = f.read()
            specs = json.loads(spec_string)
        
        rew = specs['policy_params']['reward_scale']
        disc_hid_sizes = specs['disc_hidden_sizes']
        grad_pen_weight = specs['grad_pen_weight']
        # rew = specs['algo_params']['reward_scale']
        # disc_blocks = specs['disc_num_blocks']
        seed = specs['seed']

        params_path = os.path.join(path, 'params.pkl')
        # try:
        policy = joblib.load(params_path)['exploration_policy']
        print(policy)
        dest_path = os.path.join(save_dir, exp_name+'_rew_%.0f_disc_hid_sizes_%d_%d_grad_pen_%d_seed_%d.pkl' % (rew, disc_hid_sizes[0], disc_hid_sizes[1], grad_pen_weight, seed))
        joblib.dump(policy, dest_path, compress=3)
        # except:
            # print('fuck')


# exp_name_without_dashes = 'fixed_fetch_anywhere_reach'
# exp_log_dir = os.path.join('/ais/gobi6/kamyar/oorl_rlkit/output/', exp_name_without_dashes.replace('_', '-'))
# save_dir = '/u/kamyar/oorl_rlkit/local_params/fixed_fetch_anywhere_reach_1x_shaping'
# exp_name = 'fixed_fetch_anywhere_reach_1x_shaping'

# log_dirs_list = [
#     '/ais/gobi6/kamyar/oorl_rlkit/output/disc-2-256-pol-2-100-easy-fetch-pick-and-place',
#     '/ais/gobi6/kamyar/oorl_rlkit/output/disc-2-256-pol-2-256-easy-fetch-pick-and-place',
#     '/ais/gobi6/kamyar/oorl_rlkit/output/batch-1-disc-2-100-pol-2-256-easy-fetch-pick-and-place',
#     '/ais/gobi6/kamyar/oorl_rlkit/output/batch-2-disc-2-100-pol-2-256-easy-fetch-pick-and-place',
#     '/ais/gobi6/kamyar/oorl_rlkit/output/disc-2-100-pol-2-100-easy-fetch-pick-and-place',    
# ]
log_dirs_list = [
    '/scratch/gobi2/kamyar/oorl_rlkit/output/more-disc-iters-larger-z-range-easy-in-the-air-fetch-dac'
]
for log_dir in log_dirs_list:
    tail = os.path.split(log_dir)[1]
    save_dir = os.path.join('/u/kamyar/oorl_rlkit/local_params/', tail)
    do_the_copy(log_dir, save_dir, '')
