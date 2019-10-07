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
from gen_models.normal_vae import NormalVAE as VAE
from gen_models.masked_normal_vae import MaskedNormalVAE as MaskedVAE
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
    img_save_path = 'junk_vis/debug_more_proper'

    # Prep the data -----------------------------------------------------------
    data_path = 'junk_vis/multi_mnist_data'
    canvas_size = 36
    (X_train, _), (X_test, _) = multi_mnist(data_path, max_digits=1, canvas_size=canvas_size, seed=42, use_max=True)
    X_train = X_train[:,None,...]
    X_test = X_test[:,None,...]
    X_train, X_test = torch.FloatTensor(X_train)/255.0, torch.FloatTensor(X_test)/255.0

    # np_imgs = np.load('/u/kamyar/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')['imgs']
    
    # np_imgs = None
    
    X_train = torch.clamp(X_train, 0.05, 0.95)
    X_test = torch.clamp(X_test, 0.05, 0.95)
    train_ds = TensorDataset(X_train)
    val_ds = TensorDataset(X_test)

    # Model Definition --------------------------------------------------------
    if exp_specs['masked']:
        model = MaskedVAE(
            [1, canvas_size, canvas_size],
            exp_specs['vae_specs']['z_dim'],
            exp_specs['vae_specs']['encoder_specs'],
            exp_specs['vae_specs']['decoder_specs'],
        )
    else:
        model = VAE(
            [1, canvas_size, canvas_size],
            exp_specs['vae_specs']['z_dim'],
            exp_specs['vae_specs']['encoder_specs'],
            exp_specs['vae_specs']['decoder_specs'],
        )
    if ptu.gpu_enabled():
        model.cuda()

    # Optimizer ---------------------------------------------------------------
    model_optim = Adam(model.parameters(), lr=float(exp_specs['model_lr']), weight_decay=float(exp_specs['model_wd']))

    # -------------------------------------------------------------------------
    global_iter = 0
    for epoch in range(exp_specs['epochs']):
        train_loader = DataLoader(train_ds, batch_size=exp_specs['batch_size'], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        for iter_num, img_batch in enumerate(train_loader):
            img_batch = img_batch[0]
            if ptu.gpu_enabled(): img_batch = img_batch.cuda()

            z_mean, z_log_cov, recon_mean, recon_log_cov, enc_mask, dec_mask = model(img_batch)
            elbo, KL = model.compute_ELBO(
                z_mean,
                z_log_cov,
                recon_mean,
                recon_log_cov,
                img_batch,
                average_over_batch=True
            )
            loss = -1. * elbo
            loss.backward()
            model_optim.step()

            if global_iter % 1000 == 0:
                mse = ((recon_mean - img_batch)**2).mean()
                print('\nTraining Iter %d...' % global_iter)
                print('ELBO:\t%.4f' % elbo)
                print('MSE:\t%.4f' % mse)
                print('KL:\t%.4f' % KL)
                save_pytorch_tensor_as_img(img_batch[0].data.cpu(), os.path.join(img_save_path, '%d_train_img.png'%(global_iter)))
                save_pytorch_tensor_as_img(recon_mean[0].data.cpu(), os.path.join(img_save_path, '%d_train_recon.png'%(global_iter)))
                if exp_specs['masked']:
                    save_pytorch_tensor_as_img(enc_mask[0].data.cpu(), os.path.join(img_save_path, '%d_train_enc_mask.png'%(global_iter)))
                    # save_pytorch_tensor_as_img(dec_mask[0].data.cpu(), os.path.join(img_save_path, '%d_train_dec_mask.png'%(global_iter)))


            if global_iter % exp_specs['freq_val'] == 0:
                with torch.no_grad():
                    print('Validating Iter %d...' % global_iter)
                    model.eval()

                    idxs = np.random.choice(int(X_test.size(0)), size=exp_specs['batch_size'], replace=False)
                    img_batch = X_test[idxs]
                    if ptu.gpu_enabled(): img_batch = img_batch.cuda()
        
                    z_mean, z_log_cov, recon_mean, recon_log_cov, enc_mask, dec_mask = model(img_batch)
                    elbo, KL = model.compute_ELBO(
                        z_mean,
                        z_log_cov,
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
                        save_pytorch_tensor_as_img(img_batch[i].data.cpu(), os.path.join(img_save_path, '%d_%d_img.png'%(global_iter, i)))
                        save_pytorch_tensor_as_img(recon_mean[i].data.cpu(), os.path.join(img_save_path, '%d_%d_recon.png'%(global_iter, i)))
                        if exp_specs['masked']:
                            save_pytorch_tensor_as_img(enc_mask[i].data.cpu(), os.path.join(img_save_path, '%d_%d_enc_mask.png'%(global_iter, i)))
                            # save_pytorch_tensor_as_img(dec_mask[i].data.cpu(), os.path.join(img_save_path, '%d_%d_dec_mask.png'%(global_iter, i)))

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
