# Imports ---------------------------------------------------------------------
# Python
import argparse
import joblib
import yaml
from time import sleep

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd
from torch.optim import Adam

# NumPy
import numpy as np

# Model Building
from neural_processes.generic_map import GenericMap

# Data
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer

# Logging
from rlkit.core import logger
from rlkit.launchers.launcher_util import setup_logger, set_seed

def convert_numpy_dict_to_pytorch(np_dict):
    d = {
        k: torch.FloatTensor(v) for k,v in np_dict.items()
    }
    return d

def experiment(exp_specs):
    # Set up logging ----------------------------------------------------------
    exp_id = exp_specs['exp_id']
    exp_prefix = exp_specs['exp_name']
    seed = exp_specs['seed']
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    # Load the data -----------------------------------------------------------
    extra_data_path = exp_specs['extra_data_path']
    train_replay_buffer = joblib.load(extra_data_path)['replay_buffer']
    train_replay_buffer.change_max_size_to_cur_size()
    train_replay_buffer._next_obs = train_replay_buffer._next_obs[:,exp_specs['extra_obs_dim']:]
    if exp_specs['remove_env_info']:
        train_replay_buffer._observations = train_replay_buffer._observations[:,exp_specs['extra_obs_dim']:]
    else:
        if exp_specs['normalize_env_info']:
            low, high = exp_specs['env_info_range'][0], exp_specs['env_info_range'][1]
            train_replay_buffer._observations[:,:exp_specs['extra_obs_dim']] -= (low + high)/2.0
            train_replay_buffer._observations[:,:exp_specs['extra_obs_dim']] /= (high - low)/2.0

    print('\nRewards: {} +/- {}'.format(
        np.mean(train_replay_buffer._rewards),
        np.std(train_replay_buffer._rewards)
    ))

    next_obs_mean = np.mean(train_replay_buffer._next_obs, 0)
    next_obs_std = np.std(train_replay_buffer._next_obs, 0)
    print('\nNext Obs:\n{}\n+/-\n{}'.format(
        next_obs_mean,
        next_obs_std
    ))

    print('\nAvg Next Obs Square Norm: {}'.format(
        np.mean(np.linalg.norm(train_replay_buffer._next_obs, axis=1)**2)
    ))

    sample_batch = train_replay_buffer.random_batch(exp_specs['train_batch_size'])
    obs_dim = sample_batch['observations'].shape[-1]
    act_dim = sample_batch['actions'].shape[-1]

    val_replay_buffer = SimpleReplayBuffer(exp_specs['val_set_size'], obs_dim, act_dim)
    val_replay_buffer.set_buffer_from_dict(
        train_replay_buffer.sample_and_remove(exp_specs['val_set_size'])
    )
    if exp_specs['train_from_beginning_transitions']:
        trans_dict = dict(
            observations=train_replay_buffer._observations[:exp_specs['train_set_size']],
            actions=train_replay_buffer._actions[:exp_specs['train_set_size']],
            rewards=train_replay_buffer._rewards[:exp_specs['train_set_size']],
            terminals=train_replay_buffer._terminals[:exp_specs['train_set_size']],
            next_observations=train_replay_buffer._next_obs[:exp_specs['train_set_size']],
        )
        train_replay_buffer.set_buffer_from_dict(trans_dict)
    else:
        train_replay_buffer.set_buffer_from_dict(
            train_replay_buffer.sample_and_remove(exp_specs['train_set_size'])
        )

    # Model Definitions -------------------------------------------------------
    if exp_specs['remove_env_info']:
        output_dim = [obs_dim + 1]
    else:
        output_dim = [obs_dim - exp_specs['extra_obs_dim'] + 1]
    model = GenericMap(
        [obs_dim + act_dim],
        output_dim,
        siamese_input=False,
        siamese_output=False,
        num_hidden_layers=exp_specs['num_hidden_layers'],
        hidden_dim=exp_specs['hidden_dim'],
        act='relu',
        use_bn=True,
        deterministic=True
    )

    model_optim = Adam(model.parameters(), lr=float(exp_specs['lr']))

    # Train -------------------------------------------------------------------
    model.train()
    for iter_num in range(exp_specs['max_iters']):
        model_optim.zero_grad()

        batch = train_replay_buffer.random_batch(exp_specs['train_batch_size'])
        batch = convert_numpy_dict_to_pytorch(batch)
        inputs = Variable(torch.cat([batch['observations'], batch['actions']], -1))
        outputs = Variable(torch.cat([batch['next_observations'], batch['rewards']], -1))

        preds = model([inputs])[0]
        if exp_specs['residual']:
            # residual for observations
            preds = preds + Variable(
                        torch.cat(
                            [
                                batch['observations'][:,exp_specs['extra_obs_dim']:],
                                torch.zeros(exp_specs['train_batch_size'], 1)
                            ],
                        1)
                    )
        
        loss = torch.mean(torch.sum((outputs - preds)**2, -1))

        loss.backward()
        model_optim.step()

        if iter_num % exp_specs['freq_val'] == 0:
            model.eval()

            val_batch = val_replay_buffer.random_batch(exp_specs['val_batch_size'])
            val_batch = convert_numpy_dict_to_pytorch(val_batch)
            inputs = Variable(torch.cat([val_batch['observations'], val_batch['actions']], -1))
            outputs = Variable(torch.cat([val_batch['next_observations'], val_batch['rewards']], -1))

            # print(exp_specs['remove_env_info'])
            # print(inputs)
            # print(outputs)
            # sleep(5)
            
            preds = model([inputs])[0]
            if exp_specs['residual']:
                # residual for observations
                preds = preds + Variable(
                            torch.cat(
                                [
                                    val_batch['observations'][:,exp_specs['extra_obs_dim']:],
                                    torch.zeros(exp_specs['train_batch_size'], 1)
                                ],
                            1)
                        )

            loss = torch.mean(torch.sum((outputs - preds)**2, -1))
            next_obs_loss = torch.mean(torch.sum((outputs[:,:-1] - preds[:,:-1])**2, -1))
            rew_loss = torch.mean(torch.sum((outputs[:,-1:] - preds[:,-1:])**2, -1))

            print('\n')
            print('-'*20)
            logger.record_tabular('Iter', iter_num)
            logger.record_tabular('Loss', loss.data[0])
            logger.record_tabular('Obs Loss', next_obs_loss.data[0])
            logger.record_tabular('Rew Loss', rew_loss.data[0])
            logger.dump_tabular(with_prefix=False, with_timestamp=False)

            model.train()


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)
    
    experiment(exp_specs)
