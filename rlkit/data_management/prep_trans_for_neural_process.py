from collections import defaultdict
import joblib

import numpy as np
from numpy.random import choice

from rlkit.data_management.neural_process_data_sampler import NPTransDataSampler


def prep_trans(data_prep_specs):
    # Parameters ------------------
    extra_data_path = data_prep_specs['extra_data_path']
    seed = data_prep_specs['seed']
    num_train = data_prep_specs['num_train']
    num_val = data_prep_specs['num_val']
    env_timestep_limit = data_prep_specs['env_timestep_limit']
    extra_obs_dim = data_prep_specs['extra_obs_dim']
    
    # -----------------------------
    np_old_state = np.random.get_state()
    np.random.seed(seed)

    replay_buffer = joblib.load(extra_data_path)['replay_buffer']
    replay_buffer.change_max_size_to_cur_size()

    # separate the obervations into blocks that contain samples from one envrionment
    blocks_list = []

    skip_idx = 0
    # skip any initial terminals
    while replay_buffer._terminals[skip_idx] == 1:
        skip_idx += 1

    block_start_idx = skip_idx
    block_end_idx = block_start_idx + 1
    max_length = 0
    while block_start_idx < replay_buffer._terminals.shape[0]-1:
        # yes it is very ugly
        while True:
            if (block_end_idx == replay_buffer._terminals.shape[0]) \
                or \
                (replay_buffer._terminals[block_end_idx] == 1) \
                or \
                (block_end_idx - block_start_idx + 1 == env_timestep_limit):
                env_params = tuple(replay_buffer._observations[block_start_idx, :extra_obs_dim])
                new_dict = {}
                new_dict['_observations'] = \
                    replay_buffer._observations[block_start_idx:block_end_idx+1, extra_obs_dim:]
                new_dict['_next_obs'] = \
                    replay_buffer._next_obs[block_start_idx:block_end_idx+1, extra_obs_dim:]
                new_dict['_actions'] = \
                    replay_buffer._actions[block_start_idx:block_end_idx+1, :]
                new_dict['_rewards'] = \
                    replay_buffer._rewards[block_start_idx:block_end_idx+1, :]
                new_dict['_terminals'] = \
                    replay_buffer._terminals[block_start_idx:block_end_idx+1, :]
                blocks_list.append(new_dict)

                block_start_idx = block_end_idx + 1
                block_end_idx = block_start_idx + 1
                max_length = max(max_length, block_end_idx - block_start_idx + 1)
                break
            block_end_idx += 1

    # separate them into validation and train
    train_list, val_list = [], []

    num_blocks = len(blocks_list)
    num_range = np.arange(num_blocks)
    train_inds = choice(num_range, size=num_train, replace=False)
    num_range = np.delete(num_range, train_inds)
    val_inds = choice(num_range, size=num_val, replace=False)
    for i in train_inds:
        train_list.append(blocks_list[i])
    for i in val_inds:
        val_list.append(blocks_list[i])

    np.random.set_state(np_old_state)
    return train_list, val_list


# SORRY IF I HAVEN'T CLEANED THIS UP BY THE TIME YOU SEE THIS!
# -----------
# train_np_data_sampler = NPTransDataSampler(train_list)
# val_np_data_sampler = NPTransDataSampler(val_list)

# X_context, Y_context, context_mask, X_test, Y_test, test_mask = train_np_data_sampler.sample_batch(2, [1,4], 2, test_is_context=True)
# print(X_test)
# print(Y_test)
# print(test_mask)
# -----------

# for block in train_list:
#     print(block['_observations'].shape)
# print('-'*30)
# for block in val_list:
#     print(block['_observations'].shape)

# print('\nRewards: {} +/- {}'.format(
#     np.mean(train_replay_buffer._rewards),
#     np.std(train_replay_buffer._rewards)
# ))

# next_obs_mean = np.mean(train_replay_buffer._next_obs, 0)
# next_obs_std = np.std(train_replay_buffer._next_obs, 0)
# print('\nNext Obs:\n{}\n+/-\n{}'.format(
#     next_obs_mean,
#     next_obs_std
# ))

# print('\nAvg Next Obs Square Norm: {}'.format(
#     np.mean(np.linalg.norm(train_replay_buffer._next_obs, axis=1)**2)
# ))
