import argparse
import os.path as osp
from rlkit.core.eval_util import plot_experiment_returns

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment', help='experiment specification file')
parser.add_argument('-c', '--column', help='column to plot')
parser.add_argument('-t', '--title', help='title')
parser.add_argument('-s', '--save_name', help='save image name')
args = parser.parse_args()

args.experiment = args.experiment.replace('_', '-')

plot_experiment_returns(
    osp.join('/u/kamyar/oorl_rlkit/output', args.experiment),
    args.title,
    osp.join('/u/kamyar/oorl_rlkit/plots/', args.save_name + '.png'),
    column_name=args.column,
    y_axis_lims=[0, 5]
)

# python examples/plot_regression_results.py -e res_from_begin_hype_search_meta_reacher_trans_regression -t res_from_begin -c Obs_Loss -s res_from_begin
