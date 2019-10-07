import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from convgru import ConvGRUCell
from gen_models import make_conv_net, make_fc_net, make_upconv_net

import numpy as np
from numpy import pi
from numpy import log as np_log
log_2pi = np_log(2*pi)

LOG_COV_MAX = 2
LOG_COV_MIN = -20

class VRNN(nn.Module):
    def __init__(
        self,
        maze_dims,
        action_dim,
        act_proc_dim,
        z_dim,
        x_encoder_specs,
        pre_gru_specs,
        gru_specs,
        prior_part_specs,
        inference_part_specs,
        decoder_part_specs,
        masked_latent
    ):
        super().__init__()
        
        self.act_proc_dim = act_proc_dim
        self.action_fc = nn.Linear(action_dim, self.act_proc_dim, bias=True)        

        in_ch = maze_dims[0]
        in_h = maze_dims[1]
        maze_flat_dim = in_ch * in_h * in_h
        self.maze_dims = maze_dims
        self.x_encoder, out_h = make_fc_net(in_h*in_h*in_ch, x_encoder_specs)
        self.x_enc_h = out_h

        self.prior_fc_seq, hidden_dim = make_fc_net(self.act_proc_dim + gru_specs['hidden_size'], prior_part_specs)
        self.prior_mean_fc = nn.Linear(hidden_dim, z_dim, bias=True)
        self.prior_log_cov_fc = nn.Linear(hidden_dim, z_dim, bias=True)

        self.posterior_fc_seq, hidden_dim = make_fc_net(self.act_proc_dim + gru_specs['hidden_size'] + self.x_enc_h, inference_part_specs)
        self.posterior_mean_fc = nn.Linear(hidden_dim, z_dim, bias=True)
        self.posterior_log_cov_fc = nn.Linear(hidden_dim, z_dim, bias=True)

        self.pre_gru_seq, hidden_dim = make_fc_net(self.act_proc_dim + self.x_enc_h + z_dim, pre_gru_specs)

        self.gru_cell = nn.GRUCell(
            hidden_dim, gru_specs['hidden_size'], bias=True
        )
        self.h_dim = [gru_specs['hidden_size']]

        # models for the decoding/generation
        self.recon_fc_seq, out_h = make_fc_net(z_dim + self.h_dim[0], decoder_part_specs['fc_part'])
        self.recon_mean_fc = nn.Linear(out_h, maze_flat_dim, bias=True)
        self.recon_log_cov_fc = nn.Linear(out_h, maze_flat_dim, bias=True)

        ae_dim = 256
        self.autoencoder = nn.Sequential(
            nn.Linear(48, ae_dim, bias=False),
            nn.BatchNorm1d(ae_dim),
            nn.ReLU(),
            nn.Linear(ae_dim, ae_dim, bias=False),
            nn.BatchNorm1d(ae_dim),
            nn.ReLU(),
            nn.Linear(ae_dim, ae_dim, bias=False),
            nn.BatchNorm1d(ae_dim),
            nn.ReLU()
        )
        # self.autoencoder, _ = make_fc_net(
        #     maze_flat_dim,
        #     {
        #         'hidden_sizes': [128, 128, 32, 128, 128],
        #         'use_bn': True
        #     }
        # )
        self.fc = nn.Linear(ae_dim, 48, bias=True)

    # def get_masked_z(self, z_batch):
    #     if self.masked_latent:
    #         mask = self.mask_seq(z_batch)
    #         mask = mask.repeat(1, int(z_batch.size(1)/2), 1, 1)
    #         mask = torch.cat([mask, 1. - mask], 1)
    #     return z_batch * mask


    def get_obs_recon_dist(self, z_batch, prev_h_batch):
        # if self.masked_latent:
        #     z_batch = self.get_masked_z(z_batch)
        hidden = torch.cat([z_batch, prev_h_batch], 1)
        hidden = self.recon_fc_seq(hidden)
        recon_mean = self.recon_mean_fc(hidden)
        recon_mean = F.sigmoid(recon_mean)
        recon_log_cov = self.recon_log_cov_fc(hidden)
        recon_log_cov = torch.clamp(recon_log_cov, LOG_COV_MIN, LOG_COV_MAX)

        recon_mean = recon_mean.view(recon_mean.size(0), self.maze_dims[0], self.maze_dims[1], self.maze_dims[1])
        recon_log_cov = recon_log_cov.view(recon_log_cov.size(0), self.maze_dims[0], self.maze_dims[1], self.maze_dims[1])

        return recon_mean, recon_log_cov


    def forward(self, obs_batch, act_batch, prev_h_batch):
        act_enc = self.action_fc(act_batch)

        # compute the prior
        hidden = self.prior_fc_seq(torch.cat([prev_h_batch, act_enc], 1))
        prior_mean = self.prior_mean_fc(hidden)
        prior_log_cov = self.prior_log_cov_fc(hidden)
        prior_log_cov = torch.clamp(prior_log_cov, LOG_COV_MIN, LOG_COV_MAX)

        # compute posterior
        x_enc = self.x_encoder(obs_batch.view(obs_batch.size(0), -1)).view(obs_batch.size(0), -1)
        hidden = self.posterior_fc_seq(torch.cat([x_enc, act_enc, prev_h_batch], 1))
        post_mean = self.posterior_mean_fc(hidden)
        post_log_cov = self.posterior_log_cov_fc(hidden)
        post_log_cov = torch.clamp(post_log_cov, LOG_COV_MIN, LOG_COV_MAX)
        
        # sample from the posterior
        eps = Variable(torch.randn(post_mean.size()))
        if post_mean.is_cuda: eps = eps.cuda()
        cur_z_sample = post_mean + eps*torch.exp(0.5 * post_log_cov)

        # compute generation
        recon_mean, recon_log_cov = self.get_obs_recon_dist(cur_z_sample, prev_h_batch)

        # compute recurence
        hidden = torch.cat([x_enc, act_enc, cur_z_sample], 1)
        hidden = self.pre_gru_seq(hidden)
        cur_h = self.gru_cell(hidden, prev_h_batch)

        hidden = self.autoencoder(obs_batch.view(obs_batch.size(0), -1))
        recon_mean = self.fc(hidden).view(hidden.size(0), 3, 4, 4)
        recon_mean = F.sigmoid(recon_mean)
        # recon_mean = obs_batch

        return prior_mean, prior_log_cov, post_mean, post_log_cov, cur_z_sample, recon_mean, recon_log_cov, cur_h
    

    def compute_KL(self, prior_mean, prior_log_cov, post_mean, post_log_cov):
        # assert False, 'Check this KL'
        bs = prior_mean.size(0)
        m1, lc1, m2, lc2 = post_mean.view(bs, -1), post_log_cov.view(bs, -1), prior_mean.view(bs, -1), prior_log_cov.view(bs, -1)
        KL = 0.5 * (
            torch.sum(lc2, 1) - torch.sum(lc1, 1) - m1.size(1) + 
            torch.sum(torch.exp(lc1 - lc2), 1) + torch.sum((m2 - m1)**2 / torch.exp(lc2), 1)
        )
        KL = torch.sum(KL)
        # return 0.001 * KL
        return 0.


    def compute_ELBO(
        self,
        prior_mean, prior_log_cov,
        post_mean, post_log_cov,
        recon_mean, recon_log_cov,
        obs_batch,
        average_over_batch=True
    ):
        KL = self.compute_KL(prior_mean, prior_log_cov, post_mean, post_log_cov)
        
        recon_mean = recon_mean.view(recon_mean.size(0), -1)
        recon_log_cov = recon_log_cov.view(recon_log_cov.size(0), -1)
        obs_batch = obs_batch.view(obs_batch.size(0), -1)
        recon_cov = torch.exp(recon_log_cov)

        # log_prob = -0.5 * torch.sum(
        #     (recon_mean - obs_batch)**2 / recon_cov
        # )
        # log_det_temp = torch.sum(recon_log_cov, 1) + log_2pi
        # log_prob += -0.5 * torch.sum(log_det_temp)

        log_prob = -0.5 * torch.sum((recon_mean - obs_batch)**2)

        elbo = log_prob - KL
        if average_over_batch: elbo = elbo / float(obs_batch.size(0))

        return elbo, KL
