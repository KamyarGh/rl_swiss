# Imports ---------------------------------------------------------------------
# Python
import argparse
import joblib
import yaml
import os.path as osp
from collections import defaultdict
import joblib

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd
from torch.optim import Adam

# NumPy
import numpy as np
from numpy import array
from numpy.random import choice, randint

# Model Building
from gen_models.recurrent_model import RecurrentModel
from gen_models.very_recurrent_model import VeryRecurrentModel
import rlkit.torch.pytorch_util as ptu

# Data
from gen_models.data_loaders import BasicDataLoader, RandomDataLoader, VerySpecificOnTheFLyDataLoader
from rlkit.envs.maze_envs.pogrid import PartiallyObservedGrid

# Logging
from rlkit.core import logger
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.core.vistools import generate_gif, save_pytorch_tensor_as_img

import sys

from numpy import pi
from numpy import log as np_log
log_2pi = np_log(2*pi)


def compute_diag_log_prob(recon_mean, recon_log_cov, obs):
    bs = recon_mean.size(0)
    recon_mean = recon_mean.view(bs, -1)
    recon_log_cov = recon_log_cov.view(bs, -1)
    obs = obs.view(bs, -1)

    recon_cov = torch.exp(recon_log_cov)
    log_prob = -0.5 * torch.sum(
        (recon_mean - obs)**2 / recon_cov
    )
    log_det_temp = torch.sum(recon_log_cov, 1) + log_2pi
    log_prob = log_prob - 0.5 * torch.sum(log_det_temp)

    return log_prob


def experiment(exp_specs):
    ptu.set_gpu_mode(exp_specs['use_gpu'])
    # Set up logging ----------------------------------------------------------
    exp_id = exp_specs['exp_id']
    exp_prefix = exp_specs['exp_name']
    seed = exp_specs['seed']
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    # Prep the data -----------------------------------------------------------
    env_specs = {
        'flat_repr': False,
        'one_hot_repr': False,
        'maze_h': 9,
        'maze_w': 9,
        'obs_h': 5,
        'obs_w': 5,
        'scale': 4,
        'num_objs': 10 
    }
    maze_constructor = lambda: PartiallyObservedGrid(env_specs)
    data_loader = VerySpecificOnTheFLyDataLoader(
        maze_constructor, exp_specs['episode_length'], exp_specs['batch_size'], use_gpu=ptu.gpu_enabled())
    val_data_loader = VerySpecificOnTheFLyDataLoader(
        maze_constructor, exp_specs['episode_length'], exp_specs['batch_size'], use_gpu=ptu.gpu_enabled())

    # Model Definition --------------------------------------------------------
    model = RecurrentModel()
    if ptu.gpu_enabled():
        model.cuda()
    
    # Optimizer ---------------------------------------------------------------
    model_optim = Adam(
        model.parameters(),
        lr=float(exp_specs['model_lr']),
        weight_decay=float(exp_specs['model_wd'])
    )

    # -------------------------------------------------------------------------
    freq_bptt = exp_specs['freq_bptt']
    episode_length = exp_specs['episode_length']
    losses = []
    for iter_num in range(int(float(exp_specs['max_iters']))):
        if iter_num % freq_bptt == 0:
            if iter_num > 0:
                # loss = loss / freq_bptt
                loss.backward()
                model_optim.step()
                prev_h_batch = prev_h_batch.detach()
                prev_c_batch = prev_c_batch.detach()
            loss = 0
        if iter_num % episode_length == 0:
            prev_h_batch = Variable(torch.zeros(exp_specs['batch_size'], model.lstm_dim))
            prev_c_batch = Variable(torch.zeros(exp_specs['batch_size'], model.lstm_dim))
            if ptu.gpu_enabled():
                prev_h_batch = prev_h_batch.cuda()
                prev_c_batch = prev_c_batch.cuda()
            
            train_loss_print = '\t'.join(losses)
            losses = []

        obs_batch, act_batch = data_loader.get_next_batch()
        recon, log_cov, prev_h_batch, prev_c_batch = model.forward(obs_batch, act_batch, prev_h_batch, prev_c_batch)

        losses.append('%.4f' % ((obs_batch - recon)**2).mean())
        if iter_num % episode_length != 0:
            # temp = (obs_batch - recon)**2 / 4.
            # temp[:,:,1:4,1:4] = temp[:,:,1:4,1:4] * 4.

            temp = (obs_batch - recon)**2
            loss = loss + temp.sum()/float(exp_specs['batch_size']) + model.reg_loss

            # loss = loss - compute_diag_log_prob(recon, log_cov, obs_batch)/float(exp_specs['batch_size'])

        if iter_num % (500*episode_length) in range(2*episode_length):
            save_pytorch_tensor_as_img(recon[0].data.cpu(), 'junk_vis/recurrent_deconv_stronger_2/rnn_recon_%d.png' % iter_num)
            save_pytorch_tensor_as_img(obs_batch[0].data.cpu(), 'junk_vis/recurrent_deconv_stronger_2/rnn_obs_%d.png' % iter_num)

        if iter_num % exp_specs['freq_val'] == 0:
            model.eval()
            # print(mask[0], torch.mean(mask, 1), torch.std(mask, 1), torch.min(mask, 1), torch.max(mask, 1))
            print('\nValidating Iter %d...' % iter_num)

            val_prev_h_batch = Variable(torch.zeros(exp_specs['batch_size'], model.lstm_dim))
            val_prev_c_batch = Variable(torch.zeros(exp_specs['batch_size'], model.lstm_dim))
            if ptu.gpu_enabled():
                val_prev_h_batch = val_prev_h_batch.cuda()
                val_prev_c_batch = val_prev_c_batch.cuda()

            losses = []            
            for i in range(episode_length):
                obs_batch, act_batch = val_data_loader.get_next_batch()
                
                recon, log_cov, val_prev_h_batch, val_prev_c_batch = model.forward(obs_batch, act_batch, val_prev_h_batch, val_prev_c_batch)

                # val_loss = compute_diag_log_prob(recon, log_cov, obs_batch)/float(exp_specs['batch_size'])
                losses.append('%.4f' % ((obs_batch - recon)**2).mean())

            loss_print = '\t'.join(losses)
            print('Val MSE:\t' + loss_print)
            print('Train MSE:\t' + train_loss_print)
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
