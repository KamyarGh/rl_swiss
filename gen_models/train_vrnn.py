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
from gen_models.new_vrnn import NewVRNN
import rlkit.torch.pytorch_util as ptu

# Data
from gen_models.data_loaders import BasicDataLoader

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
    env_specs = {
        'flat_repr': False,
        'one_hot_repr': False,
        'maze_h': 9,
        'maze_w': 9,
        'obs_h': 5,
        'obs_w': 5,
        'scale': 1,
        'num_objs': 10 
    }
    maze_constructor = lambda: PartiallyObservedGrid(env_specs)
    data_loader = VerySpecificOnTheFLyDataLoader(
        maze_constructor, exp_specs['episode_length'], exp_specs['batch_size'], use_gpu=ptu.gpu_enabled())
    val_data_loader = VerySpecificOnTheFLyDataLoader(
        maze_constructor, exp_specs['episode_length'], exp_specs['batch_size'], use_gpu=ptu.gpu_enabled())

    # Model Definition --------------------------------------------------------
    model = NewVRNN(
        next_obs_array[0].shape,
        acts_array[0].shape[0],
        exp_specs['vrnn_specs']['z_dim'],
        exp_specs['vrnn_specs']['x_encoder_specs'],
        exp_specs['vrnn_specs']['lstm_dim'],
        exp_specs['vrnn_specs']['decoder_part_specs'],
    )
    if ptu.gpu_enabled(): model.cuda()

    # Optimizer ---------------------------------------------------------------
    model_optim = Adam(model.parameters(), lr=float(exp_specs['model_lr']), weight_decay=float(exp_specs['model_wd']))

    # -------------------------------------------------------------------------
    freq_bptt = exp_specs['freq_bptt']
    MSE_losses = []
    KL_losses = []
    for iter_num in range(int(float(exp_specs['max_iters']))):
        if iter_num % freq_bptt == 0:
            if iter_num > 0:
                # loss = loss / freq_bptt
                total_ELBO.backward()
                model_optim.step()
                prev_h_batch = prev_h_batch.detach()
                prev_c_batch = prev_c_batch.detach()
                total_ELBO.detach()
        if iter_num % episode_length == 0:
            total_ELBO = 0.
            total_MSE = 0.
            total_KL = 0.
            prev_h_batch = Variable(torch.zeros(exp_specs['batch_size'], model.lstm_dim))
            prev_c_batch = Variable(torch.zeros(exp_specs['batch_size'], model.lstm_dim))
            if ptu.gpu_enabled():
                prev_h_batch = prev_h_batch.cuda()
                prev_c_batch = prev_c_batch.cuda()
            
            train_mse_print = '\t'.join(MSE_losses)
            train_kl_print = '\t'.join(KL_losses)
            MSE_losses = []
            KL_losses = []

        obs_batch, act_batch = data_loader.get_next_batch()

        prior_mean, prior_log_cov, post_mean, post_log_cov, cur_z_sample, recon_mean, recon_log_cov, prev_h_batch, prev_c_batch = model(obs_batch, act_batch, prev_h_batch, prev_c_batch)
        elbo, KL = model.compute_ELBO(prior_mean, prior_log_cov, post_mean, post_log_cov, recon_mean, recon_log_cov, obs_batch, average_over_batch=True)
        mse = ((recon_mean - obs_batch)**2).mean()

        total_elbo = total_ELBO + elbo
        total_MSE = total_MSE + mse
        MSE_losses.append(mse)
        KL_losses.append(KL)

        if iter_num % exp_specs['freq_val'] == 0:
            print('\nValidating Iter %d...' % iter_num)
            model.eval()
            
            val_prev_h_batch = Variable(torch.zeros(exp_specs['batch_size'], model.lstm_dim))
            val_prev_c_batch = Variable(torch.zeros(exp_specs['batch_size'], model.lstm_dim))
            if ptu.gpu_enabled():
                val_prev_h_batch = val_prev_h_batch.cuda()
                val_prev_c_batch = val_prev_c_batch.cuda()

            val_total_ELBO = 0.
            val_total_KL = 0.
            val_total_MSE = 0.            
            val_MSE_losses = []
            val_KL_losses = []
            prior_imgs = []
            post_imgs = []
            obs_imgs = []
            while val_data_loader.cur_t != val_data_loader.episode_length:
                obs_batch, act_batch = data_loader.get_next_batch()

                prior_mean, prior_log_cov, post_mean, post_log_cov, cur_z_sample, recon_mean, recon_log_cov, val_prev_h_batch, val_prev_c_batch = model(obs_batch, act_batch, val_prev_h_batch, val_prev_c_batch)
                elbo, KL = model.compute_ELBO(prior_mean, prior_log_cov, post_mean, post_log_cov, recon_mean, recon_log_cov, obs_batch, average_over_batch=True)
                mse = ((recon_mean - obs_batch)**2).mean()

                val_total_elbo = val_total_ELBO + elbo
                val_total_MSE = val_total_MSE + mse
                val_MSE_losses.append(mse)
                val_KL_losses.append(KL)

                prior_recon_mean, _ = model.get_obs_recon_dist(prior_mean, val_prev_h_batch)
                prior_recon_mean = np.transpose(prior_recon_mean[0].data.cpu().numpy(), (1,2,0))
                prior_imgs.append(prior_recon_mean)

                post_recon_mean, _ = model.get_obs_recon_dist(post_mean, val_prev_h_batch)
                post_recon_mean = np.transpose(post_recon_mean[0].data.cpu().numpy(), (1,2,0))
                post_imgs.append(post_recon_mean)

                obs = np.transpose(obs_batch[0].data.cpu().numpy(), (1,2,0))
                obs_imgs.append(obs)

                post_prior_KL = model.compute_KL(prior_mean, prior_log_cov, post_mean, post_log_cov)
                val_elbo, val_KL = model.compute_ELBO(
                    prior_mean, prior_log_cov,
                    post_mean, post_log_cov,
                    recon_mean, recon_log_cov,
                    obs_batch,
                    average_over_batch=True
                )
                val_total_elbo += val_elbo
                val_total_KL += post_prior_KL
                val_mse = ((recon_mean - obs_batch)**2).mean()
                val_total_MSE += val_mse

                val_MSE_losses.append(val_mse)
                val_total_KL.append(val_KL)

            val_mse_print = '\t'.join(val_MSE_losses)
            val_kl_print = '\t'.join(val_KL_losses)
            print('Avg Timestep MSE:\t%.4f' % (val_total_MSE))
            print('Avg Timestep KL:\t%.4f' % (val_total_KL))
            print('MSE:\t%s' % val_mse_print)
            print('KL:\t%s' % val_kl_print)

            # generate the gifs
            generate_gif(
                [prior_imgs, post_imgs, obs_imgs],
                ['Prior', 'Posterior', 'True Obs'],
                'junk_vis/tiny_vrnn/%d.gif' % iter_num
            )
            
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
