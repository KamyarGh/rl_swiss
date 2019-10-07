from rlkit.torch.gen_exp_traj_algorithm import ExpertTrajGeneratorAlgorithm

import json
import argparse
import joblib
from os import path

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment', help='specific experiment directory')
parser.add_argument('-m', '--meta', help='specific experiment directory')
args = parser.parse_args()

with open(path.join(args.experiment, 'variant.json'), 'r') as f:
    variant = json.load(f)

variant['algo_params']['do_not_train'] = True
