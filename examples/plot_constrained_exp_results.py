import argparse
from rlkit.core.eval_util import plot_experiment_returns

for reward_scale in [5.0, 10.0, 25.0]:
    for gear_0 in [[50], [100], [150], [200], [250], [300]]:
        constraints = {
            'algo_params.reward_scale': reward_scale,
            'env_specs.gear_0': gear_0
        }

        try:
            print('got ', reward_scale, gear_0)
            plot_experiment_returns(
                '/u/kamyar/oorl_rlkit/output/meta-reacher-gears-search',
                'meta_reacher_gears_search',
                '/u/kamyar/oorl_rlkit/plots/rew_{}_gear_{}_meta_reacher_gears_search.png'.format(reward_scale, gear_0),
                y_axis_lims=[-300, 0],
                constraints=constraints
            )
        except:
            print('failed ', reward_scale, gear_0)
