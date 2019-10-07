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
LOG_COV_MIN = -5

class NewVRNN(nn.Module):
    def __init__(
        self,
        maze_dims,
        action_proc_dim,
        z_dim,
        pre_post_gru_dim,
        x_encoder_specs,
        decoder_part_specs,
    ):
        super().__init__()

        in_ch = maze_dims[0]
        in_h = maze_dims[1]
        self.x_encoder, x_enc_ch, x_enc_h = make_conv_net(in_ch, in_h, x_encoder_specs)
        self.x_enc_ch = x_enc_ch
        self.x_enc_h = x_enc_h
        flat_inter_img_dim = x_enc_ch * x_enc_h * x_enc_h

        lstm_dim = z_dim
        self.lstm_dim = z_dim

        self.prior_action_fc = nn.Linear(4, action_proc_dim, bias=True)
        self.post_action_fc = nn.Linear(4, action_proc_dim, bias=True)
        self.recon_action_fc = nn.Linear(4, action_proc_dim, bias=True)
        self.mask_action_fc = nn.Linear(4, action_proc_dim, bias=True)

        print(self.prior_action_fc)

        self.prior_pre_gru_fc = nn.Sequential(
            nn.Linear(z_dim + action_proc_dim, flat_inter_img_dim, bias=False),
            nn.BatchNorm1d(flat_inter_img_dim),
            nn.ReLU()
        )
        self.prior_mean_gru = nn.GRUCell(flat_inter_img_dim, z_dim, bias=True)
        self.prior_log_cov_seq = nn.Sequential(
            nn.Linear(flat_inter_img_dim, z_dim, bias=False),
            nn.BatchNorm1d(z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim, bias=True)
        )
        self.post_mean_gru = nn.GRUCell(pre_post_gru_dim, z_dim, bias=True)
        self.post_log_cov_seq = nn.Sequential(
            nn.Linear(pre_post_gru_dim + z_dim, z_dim, bias=False),
            nn.BatchNorm1d(z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim, bias=True)
        )

        print(self.prior_mean_gru)
        print(self.post_mean_gru)

        self.attention_seq = nn.Sequential(
            nn.Linear(lstm_dim + action_proc_dim, lstm_dim, bias=False),
            nn.BatchNorm1d(lstm_dim),
            nn.ReLU(),
            nn.Linear(lstm_dim, lstm_dim),
            # nn.Sigmoid()
            # nn.Softmax()
        )
        print(self.attention_seq)

        self.pre_post_gru_fc = nn.Sequential(
            nn.Linear(flat_inter_img_dim + action_proc_dim, pre_post_gru_dim, bias=False),
            nn.BatchNorm1d(pre_post_gru_dim),
            nn.ReLU(),
        )
        print(self.pre_post_gru_fc)

        # models for the decoding/generation
        self.recon_fc_seq = nn.Sequential(
            nn.Linear(lstm_dim, flat_inter_img_dim, bias=False),
            nn.BatchNorm1d(flat_inter_img_dim),
            nn.ReLU(),
        )
        self.recon_upconv_seq, out_ch, out_h = make_upconv_net(x_enc_ch, x_enc_h, decoder_part_specs)
        self.recon_mean_conv = nn.Conv2d(out_ch, 3, 1, stride=1, padding=0, bias=True)
        self.recon_log_cov_conv = nn.Conv2d(out_ch, 3, 1, stride=1, padding=0, bias=True)
        assert out_h == maze_dims[1], str(out_h) + ' != ' + str(maze_dims[1])


    def get_obs_recon_dist(self, z_samples, original_act_batch):
        # mask_act_batch = self.mask_action_fc(original_act_batch)
        # hidden = torch.cat([z_samples, mask_act_batch], 1)
        # mask_logits = self.attention_seq(hidden)
        # mask = torch.sigmoid(mask_logits)
        # hidden = z_samples * mask

        hidden = z_samples

        hidden = self.recon_fc_seq(hidden).view(-1, self.x_enc_ch, self.x_enc_h, self.x_enc_h)
        hidden = self.recon_upconv_seq(hidden)
        recon_mean = self.recon_mean_conv(hidden)
        recon_mean = torch.sigmoid(recon_mean)
        recon_log_cov = self.recon_log_cov_conv(hidden)
        recon_log_cov = torch.clamp(recon_log_cov, LOG_COV_MIN, LOG_COV_MAX)

        return recon_mean, recon_log_cov


    def forward(self, obs_batch, act_batch, prev_z):
        prior_act_batch = self.prior_action_fc(act_batch)
        post_act_batch = self.post_action_fc(act_batch)
        recon_act_batch = self.recon_action_fc(act_batch)
        mask_act_batch = self.mask_action_fc(act_batch)

        # compute the prior
        hidden = torch.cat([prev_z, mask_act_batch], 1)
        mask_logits = self.attention_seq(hidden)
        mask = torch.sigmoid(mask_logits)
        hidden = prev_z * mask
        hidden = torch.cat([hidden, prior_act_batch], 1)
        hidden = self.prior_pre_gru_fc(hidden)
        prior_mean = self.prior_mean_gru(hidden, prev_z)

        # prior_log_cov = self.prior_log_cov_seq(torch.cat([hidden, prev_z], 1))
        # prior_log_cov = torch.clamp(prior_log_cov, LOG_COV_MIN, LOG_COV_MAX)
        # --------------
        prior_log_cov = self.prior_log_cov_seq(hidden)
        prior_log_cov = torch.clamp(prior_log_cov, LOG_COV_MIN, LOG_COV_MAX)

        # compute posterior
        x_enc = self.x_encoder(obs_batch).view(obs_batch.size(0), -1)
        hidden = torch.cat([x_enc, post_act_batch], 1)
        hidden = self.pre_post_gru_fc(hidden)
        post_mean = self.post_mean_gru(hidden, prev_z)
        post_log_cov = self.post_log_cov_seq(torch.cat([hidden, prev_z], 1))
        post_log_cov = torch.clamp(post_log_cov, LOG_COV_MIN, LOG_COV_MAX)

        # sample from the posterior
        # post_z_sample = post_mean
        eps = Variable(torch.randn(post_mean.size()))
        if post_mean.is_cuda: eps = eps.cuda()
        post_z_sample = post_mean + eps*torch.exp(0.5 * post_log_cov)

        # compute generation
        recon_mean, recon_log_cov = self.get_obs_recon_dist(post_z_sample, act_batch)

        return prior_mean, prior_log_cov, post_mean, post_log_cov, recon_mean, recon_log_cov, post_z_sample
    

    def compute_KL(self, prior_mean, prior_log_cov, post_mean, post_log_cov):
        # assert False, 'Check this KL'
        bs = prior_mean.size(0)
        m1, lc1, m2, lc2 = post_mean.view(bs, -1), post_log_cov.view(bs, -1), prior_mean.view(bs, -1), prior_log_cov.view(bs, -1)
        KL = 0.5 * (
            torch.sum(lc2, 1) - torch.sum(lc1, 1) - m1.size(1) + 
            torch.sum(torch.exp(lc1 - lc2), 1) + torch.sum((m2 - m1)**2 / torch.exp(lc2), 1)
        )
        KL = torch.sum(KL)
        return KL
        # return 0.


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

        elbo = log_prob - 0.0 * KL
        if average_over_batch: elbo = elbo / float(obs_batch.size(0))

        return elbo, KL
