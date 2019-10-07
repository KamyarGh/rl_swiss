import joblib
import numpy as np
import os
from os import path as osp
import argparse


def format_to_recall_at_k(data):
    '''
    data has this structure:
        all_seeds = [models]
        models = [tasks]
        task = [contexts]
        context = [post_samples]
        # post samples run on the same set of trajs
        post_samples = [trajs]
        trajs \in {0,1}
    4 x 16 x 4 x Max_K x 20
    '''
    print(data.shape)

    at_k_mean = []
    at_k_std = []
    for k in range(1, data.shape[-2]+1):
        at_k_data = data[...,:k,:]
        at_k_data = np.any(at_k_data, axis=-2)
        # at_k_data = np.mean(at_k_data, axis=-1)
        at_k_data = np.reshape(at_k_data, (4, -1))
        at_k_data = np.mean(at_k_data, axis=-1)
        # at_k_percent_mean.append(np.mean(at_k_data))
        # at_k_percent_std.append(np.std(at_k_data))
        at_k_mean.append(np.mean(at_k_data))
        at_k_std.append(np.std(at_k_data))
    # return at_k_percent_mean, at_k_percent_std
    return at_k_mean, at_k_std


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exppath', help='experiment path')
    parser.add_argument('--prior', help='whether it was sampled from the prior', action='store_true')
    args = parser.parse_args()
    exp_path = args.exppath

    good_reach_data = []
    solved_data = []
    for subdir in os.listdir(exp_path):
        if not osp.isdir(osp.join(exp_path, subdir)): continue
        load_name = 'all_recall_at_k_stats.pkl'
        if args.prior: load_name = 'prior_sampled_' + load_name
        sub_data = joblib.load(osp.join(exp_path, subdir, load_name))
        sub_data = sub_data['all_recall_at_k_stats'][0]
        good_reach_data.append(sub_data['algorithm_good_reach'])
        solved_data.append(sub_data['algorithm_solved'])

    good_reach_data = np.array(good_reach_data)
    solved_data = np.array(solved_data)

    good_reach_at_k_mean, good_reach_at_k_std = format_to_recall_at_k(good_reach_data)
    solved_at_k_mean, solved_at_k_std = format_to_recall_at_k(solved_data)

    save_dict = {
        'good_reach': {'mean': good_reach_at_k_mean, 'std': good_reach_at_k_std},
        'solved': {'mean': solved_at_k_mean, 'std': solved_at_k_std},
    }

    save_name = 'formatted_recall_at_k.pkl'
    if args.prior: save_name = 'prior_sampled_' + save_name
    joblib.dump(
        save_dict,
        osp.join(exp_path, save_name),
        compress=3
    )

    print(save_dict)
