# Imports ---------------------------------------------------------------------
# Python
import argparse
import joblib
import yaml
import os.path as osp
from collections import defaultdict
import joblib
import os

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
from gen_models.attentive_vae import AttentiveVAE
import rlkit.torch.pytorch_util as ptu

# Data
from observations import multi_mnist
from torch.utils.data import DataLoader, TensorDataset

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
    path = 'junk_vis/debug_att_vae_shallower_48_64_dim_0p1_kl_stronger_seg_conv'
    (X_train, Y_train), (X_test, Y_test) = multi_mnist(path, max_digits=2, canvas_size=48, seed=42, use_max=False)
    convert_dict = {0: [0.,0.], 1: [1.,0.], 2: [1.,1.]}
    Num_train = np.array([convert_dict[a.shape[0]] for a in Y_train])
    Num_test = np.array([convert_dict[a.shape[0]] for a in Y_test])
    X_train = X_train[:,None,...]
    X_test = X_test[:,None,...]
    X_train, X_test = torch.FloatTensor(X_train)/255.0, torch.FloatTensor(X_test)/255.0
    mask_train, mask_test = torch.FloatTensor(Num_train), torch.FloatTensor(Num_test)
    train_ds = TensorDataset(X_train, Num_train)
    val_ds = TensorDataset(X_test, Num_test)

    # Model Definition --------------------------------------------------------
    model = AttentiveVAE(
        [1, 48, 48],
        exp_specs['vae_specs']['z_dim'],
        exp_specs['vae_specs']['x_encoder_specs'],
        exp_specs['vae_specs']['z_seg_conv_specs'],
        exp_specs['vae_specs']['z_seg_fc_specs'],
        exp_specs['vae_specs']['z_obj_conv_specs'],
        exp_specs['vae_specs']['z_obj_fc_specs'],
        exp_specs['vae_specs']['z_seg_recon_fc_specs'],
        exp_specs['vae_specs']['z_seg_recon_upconv_specs'],
        exp_specs['vae_specs']['z_obj_recon_fc_specs'],
        exp_specs['vae_specs']['z_obj_recon_upconv_specs'],
        exp_specs['vae_specs']['recon_upconv_part_specs']
    )
    if ptu.gpu_enabled():
        model.cuda()

    # Optimizer ---------------------------------------------------------------
    model_optim = Adam(model.parameters(), lr=float(exp_specs['model_lr']), weight_decay=float(exp_specs['model_wd']))

    # -------------------------------------------------------------------------
    global_iter = 0
    for epoch in range(exp_specs['epochs']):
        train_loader = DataLoader(train_ds, batch_size=exp_specs['batch_size'], shuffle=True, num_workers=4, pin_memory=False, drop_last=True)
        for iter_num, img_batch in enumerate(train_loader):
            img_batch, num_batch = img_batch[0], img_batch[1]
            if ptu.gpu_enabled(): img_batch = img_batch.cuda()

            what_means, what_log_covs, where_means, where_log_covs, masks, recon_mean, recon_log_cov = model(img_batch, num_batch)
            elbo, KL = model.compute_ELBO(
                what_means + where_means,
                what_log_covs + where_log_covs,
                recon_mean,
                recon_log_cov,
                img_batch,
                average_over_batch=True
            )
            loss = -1. * elbo
            loss = loss + 1. * sum([m.mean() for m in masks])
            loss.backward()
            model_optim.step()

            if global_iter % exp_specs['freq_val'] == 0:
                with torch.no_grad():
                    print('\nValidating Iter %d...' % global_iter)
                    model.eval()

                    idxs = np.random.choice(int(X_test.size(0)), size=exp_specs['batch_size'], replace=False)
                    img_batch, num_batch = X_test[idxs], Num_test[idxs]
                    if ptu.gpu_enabled(): img_batch = img_batch.cuda()
        
                    what_means, what_log_covs, where_means, where_log_covs, masks, recon_mean, recon_log_cov = model(img_batch, num_batch)
                    elbo, KL = model.compute_ELBO(
                        what_means + where_means,
                        what_log_covs + where_log_covs,
                        recon_mean,
                        recon_log_cov,
                        img_batch,
                        average_over_batch=True
                    )

                    mse = ((recon_mean - img_batch)**2).mean()
                    
                    print('ELBO:\t%.4f' % elbo)
                    print('MSE:\t%.4f' % mse)
                    print('KL:\t%.4f' % KL)

                    for i in range(1):
                        save_pytorch_tensor_as_img(img_batch[i].data.cpu(), os.path.join(path, '%d_%d_img.png'%(global_iter, i)))
                        save_pytorch_tensor_as_img(recon_mean[i].data.cpu(), os.path.join(path, '%d_%d_recon.png'%(global_iter, i)))
                        save_pytorch_tensor_as_img(masks[0][i].data.cpu(), os.path.join(path, '%d_%d_mask_0.png'%(global_iter, i)))
                        # save_pytorch_tensor_as_img(masks[1][i].data.cpu(), os.path.join(path, '%d_%d_mask_1.png'%(global_iter, i)))

                    model.train()
            
            global_iter += 1


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)
    
    experiment(exp_specs)
