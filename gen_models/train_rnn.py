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
from gen_models.vrnn import VRNN
from gen_models.vrnn_flat_latent import VRNN as FlatLatentVRNN
from gen_models.flat_vrnn import VRNN as FlatVRNN
from gen_models.flat_net import VRNN as AE
import rlkit.torch.pytorch_util as ptu

# Data
from gen_models.data_loaders import BasicDataLoader, RandomDataLoader

# Logging
from rlkit.core import logger
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.core.vistools import generate_gif, save_pytorch_tensor_as_img

import sys

def experiment(exp_specs):
    ptu.set_gpu_mode(exp_specs['use_gpu'])
    # Set up logging ----------------------------------------------------------
    exp_id = exp_specs['exp_id']
    exp_prefix = exp_specs['exp_name']
    seed = exp_specs['seed']
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    # Prep the data -----------------------------------------------------------
    replay_dict = joblib.load(exp_specs['replay_dict_path'])
    next_obs_array = replay_dict['next_observations']
    acts_array = replay_dict['actions']
    data_loader = BasicDataLoader(
        next_obs_array[:40000], acts_array[:40000], exp_specs['episode_length'], exp_specs['batch_size'], use_gpu=ptu.gpu_enabled())
    val_data_loader = BasicDataLoader(
        next_obs_array[40000:], acts_array[40000:], exp_specs['episode_length'], exp_specs['batch_size'], use_gpu=ptu.gpu_enabled())

    # Model Definition --------------------------------------------------------
    ae_dim = 128
    encoder = nn.Sequential(
        nn.Linear(48, ae_dim, bias=False),
        nn.BatchNorm1d(ae_dim),
        nn.ReLU(),
        nn.Linear(ae_dim, ae_dim, bias=False),
        nn.BatchNorm1d(ae_dim),
        nn.ReLU()
    )
    gru = nn.GRUCell(
        ae_dim, ae_dim, bias=True
    )
    decoder = nn.Sequential(
        nn.Linear(ae_dim + 4, ae_dim, bias=False),
        nn.BatchNorm1d(ae_dim),
        nn.ReLU(),
        nn.Linear(ae_dim, ae_dim, bias=False),
        nn.BatchNorm1d(ae_dim),
        nn.ReLU(),
        nn.Linear(ae_dim, ae_dim, bias=False),
        nn.BatchNorm1d(ae_dim),
        nn.ReLU(),
        nn.Linear(ae_dim, 48),
        nn.Sigmoid()
    )
    if ptu.gpu_enabled():
        encoder.cuda()
        gru.cuda()
        decoder.cuda()

    # Optimizer ---------------------------------------------------------------
    model_optim = Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + list(gru.parameters()),
        lr=float(exp_specs['model_lr']), weight_decay=float(exp_specs['model_wd'])
    )

    # -------------------------------------------------------------------------
    freq_bptt = exp_specs['freq_bptt']
    losses = []
    for iter_num in range(int(float(exp_specs['max_iters']))):
        if iter_num % freq_bptt == 0:
            if iter_num > 0:
                # loss = loss / freq_bptt
                loss.backward()
                model_optim.step()
            loss = 0
            prev_h_batch = Variable(torch.zeros(exp_specs['batch_size'], ae_dim))
            if ptu.gpu_enabled():
                prev_h_batch = prev_h_batch.cuda()
            
            if iter_num % exp_specs['freq_val'] == 0:
                train_loss_print = '\t'.join(losses)
            losses = []

        obs_batch, act_batch = data_loader.get_next_batch()
        recon = decoder(torch.cat([prev_h_batch, act_batch], 1)).view(obs_batch.size())
        enc = encoder(obs_batch.view(obs_batch.size(0), -1))
        prev_h_batch = gru(enc, prev_h_batch)

        losses.append('%.4f' % ((obs_batch - recon)**2).mean())
        if iter_num % freq_bptt != 0:
            loss = loss + ((obs_batch - recon)**2).sum()/float(exp_specs['batch_size'])

        if iter_num % 250 in range(10):
            save_pytorch_tensor_as_img(recon[0].data.cpu(), 'junk_vis/with_wd_1e-3_ae_recon_%d.png' % iter_num)
            save_pytorch_tensor_as_img(obs_batch[0].data.cpu(), 'junk_vis/with_wd_1e-3_ae_obs_%d.png' % iter_num)

        if iter_num % exp_specs['freq_val'] == 0:
            print('\nValidating Iter %d...' % iter_num)
            list(map(lambda x: x.eval(), [encoder, decoder, gru]))

            val_prev_h_batch = Variable(torch.zeros(exp_specs['batch_size'], ae_dim))
            if ptu.gpu_enabled():
                val_prev_h_batch = val_prev_h_batch.cuda()

            losses = []            
            for i in range(freq_bptt):
                obs_batch, act_batch = val_data_loader.get_next_batch()
                recon = decoder(torch.cat([val_prev_h_batch, act_batch], 1)).view(obs_batch.size())
                enc = encoder(obs_batch.view(obs_batch.size(0), -1))
                val_prev_h_batch = gru(enc, val_prev_h_batch)
                losses.append('%.4f' % ((obs_batch - recon)**2).mean())

            loss_print = '\t'.join(losses)
            print('Val MSE:\t' + loss_print)
            print('Train MSE:\t' + train_loss_print)

            list(map(lambda x: x.train(), [encoder, decoder, gru]))            


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)
    
    experiment(exp_specs)
