'''
This is a crummy little script for me to debug stuff
It's not well-written and ugly
'''
import joblib
# from rlkit.envs import get_env
from rlkit.samplers.in_place import InPlacePathSampler
from wrapped_goal_envs import WrappedFetchPickAndPlaceEnv, DebugReachFetchPickAndPlaceEnv, DebugFetchReachAndLiftEnv, WrappedRotatedFetchReachAnywhereEnv
import os

# POLICY_SAVE_PATH = '/Users/kamyar/local_policies_dir/fetch_reach_and_lift_bc/fetch_reach_and_lift_bc_end_of_training.pkl'
# POLICY_SAVE_PATH = '/Users/kamyar/local_policies_dir/fetch_reach_and_lift_bc/fetch_reach_and_lift_bc_very_early_training.pkl'
# POLICY_SAVE_PATH = '/Users/kamyar/local_policies_dir/fetch_reach_and_lift_bc/fetch_reach_and_lift_bc_middle_training.pkl'
# POLICY_SAVE_PATH = '/Users/kamyar/local_policies_dir/fetch_reach_and_lift_bc/fetch_reach_and_lift_bc_later_middle_training.pkl'
# POLICY_SAVE_PATH = '/Users/kamyar/local_policies_dir/rew_100_disc_blocks_2_seed_5914.pkl'

# env_specs = {
#     'base_env_name': 'debug_fetch_reach_and_lift',
#     'normalized': False,
#     'train_test_env': False
# }

'''
For anywhere reach 10x shaping, 250 and 1000 with 9783 seed are nice
750 with both seeds is nice
'''
# POLICIES_DIR = '/Users/kamyar/local_policies_dir/local_params/fetch_reach_and_lift_dac_2_layer_pol_0_init/'
# POLICIES_DIR = '/Users/kamyar/local_policies_dir/local_params/fetch_reach_and_lift_dac_100_init/'
# POLICIES_DIR = '/Users/kamyar/local_policies_dir/local_params/fetch_reach_and_lift_dac_2_layer_pol_500_init'
# POLICIES_DIR = '/Users/kamyar/local_policies_dir/local_params/fixed_debug_fetch_reach_and_lift_dac_500_min_steps'
# POLICIES_DIR = '/Users/kamyar/local_policies_dir/local_params/fetch_reach_anywhere'
# POLICIES_DIR = '/Users/kamyar/local_policies_dir/local_params/fixed_fetch_anywhere_reach_10x_shaping'
POLICIES_DIR = '/Users/kamyar/local_policies_dir/local_params/fixed_fetch_anywhere_reach_1x_shaping'

NUM_SAMPLES = 5

for p in os.listdir(POLICIES_DIR):
    print(p)
    POLICY_SAVE_PATH = os.path.join(POLICIES_DIR, p)

    max_path_length = 50
    max_samples = NUM_SAMPLES * max_path_length
    policy_specs = {
        'policy_uses_pixels': False,
        'policy_uses_task_params': True,
        'concat_task_params_to_policy_obs': True
    }

    # set up the policy
    # policy = joblib.load(POLICY_SAVE_PATH)['exploration_policy']
    policy = joblib.load(POLICY_SAVE_PATH)

    # set up the env
    # if env_specs['train_test_env']:
    #     _, training_env = get_env(env_specs)
    # else:
    #     training_env, _ = get_env(env_specs)

    # training_env = DebugFetchReachAndLiftEnv()
    training_env = WrappedRotatedFetchReachAnywhereEnv()

    # build an eval sampler that also renders
    eval_sampler = InPlacePathSampler(
        env=training_env,
        policy=policy,
        max_samples=max_samples,
        max_path_length=max_path_length,
        policy_uses_pixels=policy_specs['policy_uses_pixels'],
        policy_uses_task_params=policy_specs['policy_uses_task_params'],
        concat_task_params_to_policy_obs=policy_specs['concat_task_params_to_policy_obs'],
        animated=True
    )
    eval_sampler.obtain_samples()

    training_env.close()
    eval_sampler = None
