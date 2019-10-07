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
from gen_models.vae import VAE
from gen_models.vae_with_seg import VAE as VAESeg
import rlkit.torch.pytorch_util as ptu

# Data
from gen_models.data_loaders import BasicDataLoader, RandomDataLoader

# Logging
from rlkit.core import logger
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.core.vistools import generate_gif, save_pytorch_tensor_as_img


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
    data_loader = RandomDataLoader(
        next_obs_array[:4000], acts_array[:4000], use_gpu=ptu.gpu_enabled())
    val_data_loader = RandomDataLoader(
        next_obs_array[4000:], acts_array[4000:], use_gpu=ptu.gpu_enabled())

    # Model Definition --------------------------------------------------------
    if exp_specs['use_masked_vae']:
        model = VAESeg()
    else:
        model = VAE()
    if ptu.gpu_enabled(): model.cuda()

    # Optimizer ---------------------------------------------------------------
    model_optim = Adam(model.parameters(), lr=float(exp_specs['model_lr']), weight_decay=float(exp_specs['model_wd']))

    # -------------------------------------------------------------------------
    for iter_num in range(int(float(exp_specs['max_iters']))):
        obs_batch, act_batch = data_loader.get_next_batch(exp_specs['batch_size'])
        if exp_specs['use_masked_vae']:
            recon_mean, recon_log_cov, z_mean, z_log_cov, mask = model(obs_batch)
        else:
            recon_mean, recon_log_cov, z_mean, z_log_cov = model(obs_batch)
        elbo = model.compute_ELBO(z_mean, z_log_cov, recon_mean, recon_log_cov, obs_batch)
        KL = model.compute_KL(z_mean, z_log_cov)
        neg_elbo = -1. * elbo
        neg_elbo.backward()
        model_optim.step()

        if iter_num % exp_specs['freq_val'] == 0:
            print('\nValidating Iter %d...' % iter_num)
            model.eval()

            obs_batch, act_batch = val_data_loader.get_next_batch(exp_specs['batch_size'])
            if exp_specs['use_masked_vae']:
                recon_mean, recon_log_cov, z_mean, z_log_cov, mask = model(obs_batch)
                mask = mask.repeat(1,3,1,1)
                save_pytorch_tensor_as_img(mask[0].data.cpu(), 'junk_vis/mask_vae_%d.png' % iter_num)
            else:
                recon_mean, recon_log_cov, z_mean, z_log_cov = model(obs_batch)           
            elbo = model.compute_ELBO(z_mean, z_log_cov, recon_mean, recon_log_cov, obs_batch)
            KL = model.compute_KL(z_mean, z_log_cov)

            print('\nELBO:\t%.4f' % elbo)
            print('KL:\t%.4f' % KL)
            print('MSE:\t%.4f' % ((recon_mean - obs_batch)**2).mean())
            print(obs_batch[0][0,:4,:4])
            print(recon_mean[0][0,:4,:4])
            print(recon_log_cov[0][0,:4,:4])
            print(z_mean[0,1])
            print(torch.exp(z_log_cov[0,1]))

            save_pytorch_tensor_as_img(recon_mean[0].data.cpu(), 'junk_vis/recon_vae_%d.png' % iter_num)
            save_pytorch_tensor_as_img(obs_batch[0].data.cpu(), 'junk_vis/obs_vae_%d.png' % iter_num)
            
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
