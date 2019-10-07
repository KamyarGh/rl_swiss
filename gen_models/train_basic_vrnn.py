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

def compute_KL(prior_mean, prior_log_cov, post_mean, post_log_cov):
    bs = prior_mean.size(0)
    m1, lc1, m2, lc2 = post_mean.view(bs, -1), post_log_cov.view(bs, -1), prior_mean.view(bs, -1), prior_log_cov.view(bs, -1)
    KL = 0.5 * (
        torch.sum(lc2, 1) - torch.sum(lc1, 1) - m1.size(1) + 
        torch.sum(torch.exp(lc1 - lc2), 1) + torch.sum((m2 - m1)**2 / torch.exp(lc2), 1)
    )
    KL = torch.sum(KL)
    return KL/bs

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
    conv_encoder = nn.Sequential(
        nn.Conv2d(3, 32, 1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, 1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU()
    )
    ae_dim = 128
    z_dim = 128
    pre_gru = nn.Sequential(
        nn.Linear(288+z_dim+4, ae_dim, bias=False),
        nn.BatchNorm1d(ae_dim),
        nn.ReLU(),
        nn.Linear(ae_dim, ae_dim, bias=False),
        nn.BatchNorm1d(ae_dim),
        nn.ReLU()
    )
    post_fc = nn.Sequential(
        nn.Linear(ae_dim + 288 + 4, ae_dim, bias=False),
        nn.BatchNorm1d(ae_dim),
        nn.ReLU(),
        nn.Linear(ae_dim, ae_dim, bias=False),
        nn.BatchNorm1d(ae_dim),
        nn.ReLU()
    )
    post_mean_fc = nn.Linear(ae_dim, z_dim, bias=True)
    post_log_cov_fc = nn.Linear(ae_dim, z_dim, bias=True)
    prior_fc = nn.Sequential(
        nn.Linear(ae_dim + 4, ae_dim, bias=False),
        nn.BatchNorm1d(ae_dim),
        nn.ReLU(),
        nn.Linear(ae_dim, ae_dim, bias=False),
        nn.BatchNorm1d(ae_dim),
        nn.ReLU()
    )
    prior_mean_fc = nn.Linear(ae_dim, z_dim, bias=True)
    prior_log_cov_fc = nn.Linear(ae_dim, z_dim, bias=True)
    gru = nn.GRUCell(
        ae_dim, ae_dim, bias=True
    )
    fc_decoder = nn.Sequential(
        nn.Linear(ae_dim + z_dim + 4, ae_dim, bias=False),
        nn.BatchNorm1d(ae_dim),
        nn.ReLU(),
        nn.Linear(ae_dim, ae_dim, bias=False),
        nn.BatchNorm1d(ae_dim),
        nn.ReLU(),
        nn.Linear(ae_dim, 288, bias=False),
        nn.BatchNorm1d(288),
        nn.ReLU(),
    )
    conv_decoder = nn.Sequential(
        nn.ConvTranspose2d(32, 32, 1, stride=1, padding=0, output_padding=0, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 32, 1, stride=1, padding=0, output_padding=0, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 3, 1, stride=1, padding=0, bias=True),
        nn.Sigmoid()
    )
    if ptu.gpu_enabled():
        conv_encoder.cuda()
        pre_gru.cuda()
        post_fc.cuda()
        post_mean_fc.cuda()
        post_log_cov_fc.cuda()
        prior_fc.cuda()
        prior_mean_fc.cuda()
        prior_log_cov_fc.cuda()
        gru.cuda()
        fc_decoder.cuda()
        conv_decoder.cuda()

    # Optimizer ---------------------------------------------------------------
    model_optim = Adam(
        [
            item for sublist in
            map(
                lambda x: list(x.parameters()),
                [pre_gru, conv_encoder, gru, fc_decoder, conv_decoder, post_fc, post_log_cov_fc, post_mean_fc, prior_fc, prior_log_cov_fc, prior_mean_fc]
            )
            for item in sublist
        ],
        lr=float(exp_specs['model_lr']), weight_decay=float(exp_specs['model_wd'])
    )

    # -------------------------------------------------------------------------
    freq_bptt = exp_specs['freq_bptt']
    episode_length = exp_specs['episode_length']
    losses = []
    KLs = []
    for iter_num in range(int(float(exp_specs['max_iters']))):
        if iter_num % freq_bptt == 0:
            if iter_num > 0:
                # loss = loss / freq_bptt
                loss = loss + total_KL
                loss.backward()
                model_optim.step()
            loss = 0
            total_KL = 0
            prev_h_batch = Variable(torch.zeros(exp_specs['batch_size'], ae_dim))
            if ptu.gpu_enabled():
                prev_h_batch = prev_h_batch.cuda()
            
            if iter_num % exp_specs['freq_val'] == 0:
                train_loss_print = '\t'.join(losses)
                train_KLs_print = '\t'.join(KLs)
            losses = []
            KLs = []

        obs_batch, act_batch = data_loader.get_next_batch()

        enc = conv_encoder(obs_batch).view(obs_batch.size(0), -1)

        hidden = post_fc(torch.cat([prev_h_batch, enc, act_batch], 1))
        post_mean = post_mean_fc(hidden)
        post_log_cov = post_log_cov_fc(hidden)

        hidden = prior_fc(torch.cat([prev_h_batch, act_batch], 1))
        prior_mean = prior_mean_fc(hidden)
        prior_log_cov = prior_log_cov_fc(hidden)

        recon = fc_decoder(torch.cat([prev_h_batch, act_batch, post_mean], 1)).view(obs_batch.size(0), 32, 3, 3)
        recon = conv_decoder(recon)

        hidden = pre_gru(torch.cat([enc, post_mean, act_batch], 1))
        prev_h_batch = gru(hidden, prev_h_batch)

        KL = compute_KL(prior_mean, prior_log_cov, post_mean, post_log_cov)
        if iter_num % episode_length != 0:
            loss = loss + torch.sum((obs_batch.view(obs_batch.size(0), -1) - recon.view(obs_batch.size(0), -1))**2, 1).mean()
            total_KL = total_KL + KL
        losses.append('%.4f' % ((obs_batch - recon)**2).mean())
        KLs.append('%.4f' % KL)

        if iter_num % (50*exp_specs['episode_length']) in range(2*exp_specs['episode_length']):
            save_pytorch_tensor_as_img(recon[0].data.cpu(), 'junk_vis/full_KL_mem_grid_%d_recon.png' % iter_num)
            save_pytorch_tensor_as_img(obs_batch[0].data.cpu(), 'junk_vis/full_KL_mem_grid_%d_obs.png' % iter_num)

        if iter_num % exp_specs['freq_val'] == 0:
            print('\nValidating Iter %d...' % iter_num)
            list(map(lambda x: x.eval(), [pre_gru, conv_encoder, gru, fc_decoder, conv_decoder, post_fc, post_log_cov_fc, post_mean_fc, prior_fc, prior_log_cov_fc, prior_mean_fc]))

            val_prev_h_batch = Variable(torch.zeros(exp_specs['batch_size'], ae_dim))
            if ptu.gpu_enabled():
                val_prev_h_batch = val_prev_h_batch.cuda()

            val_losses = []
            val_KLs = []            
            for i in range(freq_bptt):
                obs_batch, act_batch = data_loader.get_next_batch()
                
                enc = conv_encoder(obs_batch).view(obs_batch.size(0), -1)

                hidden = post_fc(torch.cat([prev_h_batch, enc, act_batch], 1))
                post_mean = post_mean_fc(hidden)
                post_log_cov = post_log_cov_fc(hidden)

                hidden = prior_fc(torch.cat([prev_h_batch, act_batch], 1))
                prior_mean = prior_mean_fc(hidden)
                prior_log_cov = prior_log_cov_fc(hidden)

                recon = fc_decoder(torch.cat([prev_h_batch, act_batch, post_mean], 1)).view(obs_batch.size(0), 32, 3, 3)
                recon = conv_decoder(recon)

                hidden = pre_gru(torch.cat([enc, post_mean, act_batch], 1))
                prev_h_batch = gru(hidden, prev_h_batch)

                val_losses.append('%.4f' % ((obs_batch - recon)**2).mean())
                val_KL = compute_KL(prior_mean, prior_log_cov, post_mean, post_log_cov)
                val_KLs.append('%.4f' % val_KL)

            val_loss_print = '\t'.join(val_losses)
            val_KLs_print = '\t'.join(val_KLs)
            print('Val MSE:\t' + val_loss_print)
            print('Train MSE:\t' + train_loss_print)
            print('Val KL:\t\t' + val_KLs_print)
            print('Train KL:\t' + train_KLs_print)

            list(map(lambda x: x.train(), [pre_gru, conv_encoder, gru, fc_decoder, conv_decoder, post_fc, post_log_cov_fc, post_mean_fc, prior_fc, prior_log_cov_fc, prior_mean_fc]))


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)
    
    experiment(exp_specs)
