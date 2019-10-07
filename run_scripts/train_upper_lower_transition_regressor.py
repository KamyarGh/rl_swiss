# Imports ---------------------------------------------------------------------
# Python
import argparse
import joblib
import yaml

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

# Vistools
from rlkit.core.vistools import save_plot, plot_returns_on_same_plot, plot_multiple_plots

def convert_numpy_dict_to_pytorch(np_dict):
    d = {
        k: torch.FloatTensor(v) for k,v in np_dict.items()
    }
    return d

def experiment(exp_specs):
    # Load the data -----------------------------------------------------------
    extra_data_path = exp_specs['extra_data_path']
    train_replay_buffer = joblib.load(extra_data_path)['replay_buffer']
    train_replay_buffer.change_max_size_to_cur_size()
    train_replay_buffer._next_obs = train_replay_buffer._next_obs[:,exp_specs['extra_obs_dim']:]

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
    train_replay_buffer.set_buffer_from_dict(
        train_replay_buffer.sample_and_remove(exp_specs['train_set_size'])
    )

    # Model Definitions -------------------------------------------------------
    model = GenericMap(
        [obs_dim + act_dim],
        [obs_dim - exp_specs['extra_obs_dim'] + 1],
        siamese_input=False,
        siamese_output=False,
        num_hidden_layers=exp_specs['num_hidden_layers'],
        hidden_dim=exp_specs['hidden_dim'],
        act='relu',
        use_bn=True,
        deterministic=True
    )

    gap_model = GenericMap(
        [obs_dim + act_dim],
        [obs_dim - exp_specs['extra_obs_dim'], obs_dim - exp_specs['extra_obs_dim']],
        siamese_input=False,
        siamese_output=True,
        num_hidden_layers=exp_specs['num_hidden_layers'],
        hidden_dim=exp_specs['hidden_dim'],
        act='relu',
        use_bn=True,
        deterministic=True
    )

    model_optim = Adam(model.parameters(), lr=float(exp_specs['lr']))
    gap_model_optim = Adam(gap_model.parameters(), lr=float(exp_specs['gap_lr']))

    # Train -------------------------------------------------------------------
    model.train()
    for iter_num in range(exp_specs['max_iters']):
        model_optim.zero_grad()
        gap_model_optim.zero_grad()

        batch = train_replay_buffer.random_batch(exp_specs['train_batch_size'])
        batch = convert_numpy_dict_to_pytorch(batch)
        inputs = Variable(torch.cat([batch['observations'], batch['actions']], -1))
        outputs = Variable(torch.cat([batch['next_observations'], batch['rewards']], -1))
        true_next_obs = Variable(batch['next_observations'])

        preds = model([inputs])[0]
        gap_preds = gap_model([inputs])
        lower, upper = gap_preds[0], gap_preds[1]
        # residual for observations
        # preds = preds + Variable(torch.cat([batch['observations'], torch.zeros(exp_specs['train_batch_size'], 1)], 1))
        
        loss = torch.mean(torch.sum((outputs - preds)**2, -1))

        lower_loss = torch.mean(torch.sum(F.relu(lower - true_next_obs), -1))
        upper_loss = torch.mean(torch.sum(F.relu(true_next_obs - upper), -1))
        upper_lower_gap_loss = torch.mean(torch.sum(torch.abs(upper - lower), -1))

        total_loss = loss + upper_loss + lower_loss + float(exp_specs['upper_lower_gap_loss_weight']) * upper_lower_gap_loss

        total_loss.backward()
        model_optim.step()
        gap_model_optim.step()

        if iter_num % exp_specs['freq_val'] == 0:
            model.eval()

            val_batch = val_replay_buffer.random_batch(exp_specs['val_batch_size'])
            val_batch = convert_numpy_dict_to_pytorch(val_batch)
            inputs = Variable(torch.cat([val_batch['observations'], val_batch['actions']], -1))
            outputs = Variable(torch.cat([val_batch['next_observations'], val_batch['rewards']], -1))
            true_next_obs = Variable(val_batch['next_observations'])
            
            preds = model([inputs])[0]
            gap_preds = gap_model([inputs])
            lower, upper = gap_preds[0], gap_preds[1]
            # residual for observations
            # pred = preds + Variable(torch.cat([val_batch['observations'], torch.zeros(exp_specs['train_batch_size'], 1)], 1))

            loss = torch.mean(torch.sum((outputs - preds)**2, -1))
            next_obs_loss = torch.mean(torch.sum((outputs[:,:-1] - preds[:,:-1])**2, -1))
            rew_loss = torch.mean(torch.sum((outputs[:,-1:] - preds[:,-1:])**2, -1))

            lower_loss = torch.mean(torch.sum(F.relu(lower - true_next_obs), -1))
            upper_loss = torch.mean(torch.sum(F.relu(true_next_obs - upper), -1))
            upper_lower_gap_loss = torch.mean(torch.sum(torch.abs(upper - lower), -1))

            pred_over_upper = torch.mean(torch.sum(F.relu(preds[:,:-1] - upper), -1))
            pred_under_lower = torch.mean(torch.sum(F.relu(lower - preds[:,:-1]), -1))

            adj_next_obs_pred = torch.max(torch.min(preds[:,:-1], upper), lower)
            adj_next_obs_loss = torch.mean(torch.sum((outputs[:,:-1] - adj_next_obs_pred)**2, -1))

            ul_mean = (upper + lower) / 2.0
            ul_mean_as_obs_loss = torch.mean(torch.sum((outputs[:,:-1] - ul_mean)**2, -1))

            print('\n')
            print('-'*20)
            print('Iter %d' % iter_num)
            print('Loss: %.4f' % loss)
            print('Obs Loss: %.4f' % next_obs_loss)
            print('Rew Loss: %.4f' % rew_loss)
            print('\nUpper Loss: %.4f' % upper_loss)
            print('Lower Loss: %.4f' % lower_loss)
            print('UL-Gap Loss: %.4f' % upper_lower_gap_loss)
            print('\nPred Over Upper: %.4f' % pred_over_upper)
            print('Pred Under Lower: %.4f' % pred_under_lower)
            print('\nAdj Obs Loss: %.4f' % adj_next_obs_loss)
            print('\nUL Mean as Obs Loss: %.4f' % ul_mean_as_obs_loss)

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
