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

LOG_COV_MAX = 2
LOG_COV_MIN = -20


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
    conv_channels = 32
    conv_encoder = nn.Sequential(
        nn.Conv2d(3, conv_channels, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(conv_channels),
        nn.ReLU(),
        nn.Conv2d(conv_channels, conv_channels, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(conv_channels),
        nn.ReLU(),
        nn.Conv2d(conv_channels, conv_channels, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(conv_channels),
        nn.ReLU(),
        nn.Conv2d(conv_channels, conv_channels, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(conv_channels),
        nn.ReLU()
    )
    ae_dim = 256
    gru_dim = 512
    img_h = 5
    flat_inter_img_dim = img_h * img_h * conv_channels
    act_dim = 64
    act_proc = nn.Linear(4, act_dim, bias=True)
    fc_encoder = nn.Sequential(
        nn.Linear(flat_inter_img_dim + act_dim, ae_dim, bias=False),
        nn.BatchNorm1d(ae_dim),
        nn.ReLU(),
        # nn.Linear(ae_dim, ae_dim, bias=False),
        # nn.BatchNorm1d(ae_dim),
        # nn.ReLU(),
        # nn.Linear(ae_dim, ae_dim, bias=False),
        # nn.BatchNorm1d(ae_dim),
        # nn.ReLU(),
        # nn.Linear(ae_dim, ae_dim, bias=False),
        # nn.BatchNorm1d(ae_dim),
        # nn.ReLU()
    )
    gru = nn.LSTMCell(
        ae_dim, gru_dim, bias=True
    )
    fc_decoder = nn.Sequential(
        nn.Linear(gru_dim + act_dim, 256, bias=False),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 2*flat_inter_img_dim, bias=False),
        nn.BatchNorm1d(2*flat_inter_img_dim),
        nn.ReLU(),
        # # nn.Linear(ae_dim, ae_dim, bias=False),
        # # nn.BatchNorm1d(ae_dim),
        # # nn.ReLU(),
        # # nn.Linear(ae_dim, ae_dim, bias=False),
        # # nn.BatchNorm1d(ae_dim),
        # # nn.ReLU(),
        # nn.Linear(ae_dim, flat_inter_img_dim, bias=False),
        # nn.BatchNorm1d(flat_inter_img_dim),
        # nn.ReLU(),
    )
    conv_decoder = nn.Sequential(
        nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, output_padding=0, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, output_padding=0, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        # nn.Conv2d(conv_channels, conv_channels, 3, stride=1, padding=1, bias=False),
        # nn.BatchNorm2d(conv_channels),
        # nn.ReLU(),
    )
    mean_decoder = nn.Sequential(
        nn.Conv2d(64, 3, 1, stride=1, padding=0, bias=True),
        nn.Sigmoid()
    )
    log_cov_decoder = nn.Sequential(
        nn.Conv2d(64, 3, 1, stride=1, padding=0, bias=True),
    )
    if ptu.gpu_enabled():
        conv_encoder.cuda()
        fc_encoder.cuda()
        gru.cuda()
        fc_decoder.cuda()
        conv_decoder.cuda()
        mean_decoder.cuda()
        log_cov_decoder.cuda()
        act_proc.cuda()

    # Optimizer ---------------------------------------------------------------
    model_optim = Adam(
        [
            item for sublist in
            map(
                lambda x: list(x.parameters()),
                [fc_encoder, conv_encoder, gru, fc_decoder, conv_decoder, mean_decoder, log_cov_decoder]
            )
            for item in sublist
        ],
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
            prev_h_batch = Variable(torch.zeros(exp_specs['batch_size'], gru_dim))
            prev_c_batch = Variable(torch.zeros(exp_specs['batch_size'], gru_dim))
            if ptu.gpu_enabled():
                prev_h_batch = prev_h_batch.cuda()
                prev_c_batch = prev_c_batch.cuda()
            
            train_loss_print = '\t'.join(losses)
            losses = []

        obs_batch, act_batch = data_loader.get_next_batch()
        act_batch = act_proc(act_batch)
        
        hidden = fc_decoder(torch.cat([prev_h_batch, act_batch], 1)).view(obs_batch.size(0), 64, img_h, img_h)
        hidden = conv_decoder(hidden)
        recon = mean_decoder(hidden)
        log_cov = log_cov_decoder(hidden)
        log_cov = torch.clamp(log_cov, LOG_COV_MIN, LOG_COV_MAX)

        enc = conv_encoder(obs_batch)
        enc = enc.view(obs_batch.size(0), -1)
        enc = fc_encoder(torch.cat([enc, act_batch], 1))
        prev_h_batch, prev_c_batch = gru(enc, (prev_h_batch, prev_c_batch))

        losses.append('%.4f' % ((obs_batch - recon)**2).mean())
        if iter_num % episode_length != 0:
            loss = loss + ((obs_batch - recon)**2).sum()/float(exp_specs['batch_size'])
            # loss = loss - compute_diag_log_prob(recon, log_cov, obs_batch)/float(exp_specs['batch_size'])

        if iter_num % (500*episode_length) in range(2*episode_length):
            save_pytorch_tensor_as_img(recon[0].data.cpu(), 'junk_vis/debug_2_good_acts_on_the_fly_pogrid_len_8_scale_4/rnn_recon_%d.png' % iter_num)
            save_pytorch_tensor_as_img(obs_batch[0].data.cpu(), 'junk_vis/debug_2_good_acts_on_the_fly_pogrid_len_8_scale_4/rnn_obs_%d.png' % iter_num)

        if iter_num % exp_specs['freq_val'] == 0:
            print('\nValidating Iter %d...' % iter_num)
            list(map(lambda x: x.eval(), [fc_encoder, conv_encoder, gru, fc_decoder, conv_decoder, mean_decoder, log_cov_decoder, act_proc]))

            val_prev_h_batch = Variable(torch.zeros(exp_specs['batch_size'], gru_dim))
            val_prev_c_batch = Variable(torch.zeros(exp_specs['batch_size'], gru_dim))
            if ptu.gpu_enabled():
                val_prev_h_batch = val_prev_h_batch.cuda()
                val_prev_c_batch = val_prev_c_batch.cuda()

            losses = []            
            for i in range(episode_length):
                obs_batch, act_batch = val_data_loader.get_next_batch()
                act_batch = act_proc(act_batch)
                
                hidden = fc_decoder(torch.cat([val_prev_h_batch, act_batch], 1)).view(obs_batch.size(0), 64, img_h, img_h)
                hidden = conv_decoder(hidden)
                recon = mean_decoder(hidden)
                log_cov = log_cov_decoder(hidden)
                log_cov = torch.clamp(log_cov, LOG_COV_MIN, LOG_COV_MAX)

                enc = conv_encoder(obs_batch).view(obs_batch.size(0), -1)
                enc = fc_encoder(torch.cat([enc, act_batch], 1))
                val_prev_h_batch, val_prev_c_batch = gru(enc, (val_prev_h_batch, val_prev_c_batch))

                # val_loss = compute_diag_log_prob(recon, log_cov, obs_batch)/float(exp_specs['batch_size'])
                losses.append('%.4f' % ((obs_batch - recon)**2).mean())

            loss_print = '\t'.join(losses)
            print('Val MSE:\t' + loss_print)
            print('Train MSE:\t' + train_loss_print)

            list(map(lambda x: x.train(), [fc_encoder, conv_encoder, gru, fc_decoder, conv_decoder, mean_decoder, log_cov_decoder, act_proc]))            


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)
    
    experiment(exp_specs)
