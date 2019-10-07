import joblib
import numpy as np
import os
from os import path as osp
import argparse
from collections import defaultdict


def format_few_shot(data, sampled_from_prior=False):
    if not sampled_from_prior:
        max_c = 8
    else:
        max_c = 1
    percent_solved_at_c_shot = defaultdict(list)
    for seed_model_data in data:
        for c in range(1, max_c+1):
            percent_solved_at_c_shot[c].append(np.mean(seed_model_data[c]))
    means = [np.mean(percent_solved_at_c_shot[c]) for c in range(1, max_c+1)]
    stds = [np.std(percent_solved_at_c_shot[c]) for c in range(1, max_c+1)]
    return means, stds


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
        load_name = 'all_few_shot_stats.pkl'
        if args.prior: load_name = 'prior_sampled_' + load_name
        sub_data = joblib.load(osp.join(exp_path, subdir, load_name))
        sub_data = sub_data['all_few_shot_stats'][0]
        good_reach_data.append(sub_data['algorithm_good_reach'])
        solved_data.append(sub_data['algorithm_solved'])

    good_reach_at_c_mean, good_reach_at_c_std = format_few_shot(good_reach_data, sampled_from_prior=args.prior)
    solved_at_c_mean, solved_at_c_std = format_few_shot(solved_data, sampled_from_prior=args.prior)

    save_dict = {
        'good_reach': {'mean': good_reach_at_c_mean, 'std': good_reach_at_c_std},
        'solved': {'mean': solved_at_c_mean, 'std': solved_at_c_std},
    }

    save_name = 'formatted_few_shot.pkl'
    if args.prior: save_name = 'prior_sampled_' + save_name
    joblib.dump(
        save_dict,
        osp.join(exp_path, save_name),
        compress=3
    )

    print(save_dict)
