import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from convgru import ConvGRUCell
from gen_models import make_conv_net, make_fc_net, make_upconv_net

import rlkit.torch.pytorch_util as ptu

import numpy as np
from numpy import pi
from numpy import log as np_log
log_2pi = np_log(2*pi)

LOG_COV_MAX = 2
LOG_COV_MIN = -5

class AttentiveVAE(nn.Module):
    def __init__(
        self,
        maze_dims,
        z_dim,
        x_encoder_specs,
        z_seg_conv_specs,
        z_seg_fc_specs,
        z_obj_conv_specs,
        z_obj_fc_specs,
        z_seg_recon_fc_specs,
        z_seg_recon_upconv_specs,
        z_obj_recon_fc_specs,
        z_obj_recon_upconv_specs,
        recon_upconv_part_specs
    ):
        super().__init__()

        in_ch = maze_dims[0]
        in_h = maze_dims[1]
        self.x_encoder, x_enc_ch, x_enc_h = make_conv_net(in_ch, in_h, x_encoder_specs)
        self.x_enc_ch = x_enc_ch
        self.x_enc_h = x_enc_h
        flat_inter_img_dim = x_enc_ch * x_enc_h * x_enc_h

        # self.convgru = ConvGRUCell(x_enc_ch, gru_specs['channels'], gru_specs['kernel_size'])
        # self.gru_ch = gru_specs['channels']

        self.z_seg_conv_seq, out_ch, out_h = make_conv_net(x_enc_ch+1, x_enc_h, z_seg_conv_specs)
        self.z_seg_fc_seq, out_h = make_fc_net(out_ch*out_h*out_h, z_seg_fc_specs)
        self.z_seg_mean_fc = nn.Linear(out_h, z_dim, bias=True)
        self.z_seg_log_cov_fc = nn.Linear(out_h, z_dim, bias=True)

        # self.z_obj_conv_seq, z_conv_ch, z_conv_h = make_conv_net(x_enc_ch, x_enc_h, z_obj_conv_specs)
        # flat_dim = z_conv_ch*z_conv_h*z_conv_h
        # self.z_conv_ch, self.z_conv_h = z_conv_ch, z_conv_h
        self.z_obj_fc_seq, out_h = make_fc_net(flat_inter_img_dim, z_obj_fc_specs)
        self.z_obj_mean_fc = nn.Linear(out_h, z_dim, bias=True)
        self.z_obj_log_cov_fc = nn.Linear(out_h, z_dim, bias=True)
        
        self.z_seg_mask_fc_seq, out_h = make_fc_net(z_dim, z_seg_recon_fc_specs)
        # print(out_h)
        # print(z_conv_ch, z_conv_h)
        # assert out_h == z_conv_h*z_conv_h*z_conv_ch
        self.z_seg_mask_upconv_seq, out_ch, out_h = make_upconv_net(x_enc_ch, x_enc_h, z_seg_recon_upconv_specs)
        self.z_seg_mask_conv = nn.Conv2d(out_ch, 1, 3, stride=1, padding=1, bias=True)
        print(out_h)

        self.z_obj_recon_fc_seq, z_recon_dim = make_fc_net(z_dim, z_obj_recon_fc_specs)
        # self.z_obj_recon_upconv_seq, out_ch, out_h = make_upconv_net(z_conv_ch, z_conv_h, z_obj_recon_upconv_specs)
        self.recon_upconv_seq, out_ch, out_h = make_upconv_net(x_enc_ch, x_enc_h, recon_upconv_part_specs)
        print(out_h)
        self.recon_mean_conv = nn.Conv2d(out_ch, 1, 1, stride=1, padding=0, bias=True)
        self.recon_log_cov_conv = nn.Conv2d(out_ch, 1, 1, stride=1, padding=0, bias=True)
        assert out_h == maze_dims[1], str(out_h) + ' != ' + str(maze_dims[1])


    def forward(self, img_batch, num_batch):
        num_objs = 2

        enc = self.x_encoder(img_batch)
        enc_mask_aggregate = torch.zeros(enc.size(0), 1, enc.size(2), enc.size(3)).type_as(enc)
        
        # prev_h = Variable(torch.zeros(enc.size(0), self.gru_ch, enc.size(2), enc.size(3)))
        # if ptu.gpu_enabled():
        #     prev_h = prev_h.cuda()
        
        inter_recon_tensor = 0.
        obj_means = []
        obj_log_covs = []
        seg_means = []
        seg_log_covs = []
        masks = []
        for i in range(num_objs):
            # infer
            hidden = torch.cat([enc, enc_mask_aggregate], 1)
            # hidden = enc * (1. - enc_mask_aggregate)
            # prev_h = self.convgru(hidden, prev_h)

            hidden = self.z_seg_conv_seq(hidden)
            hidden = hidden.view(hidden.size(0), -1)
            hidden = self.z_seg_fc_seq(hidden)
            z_seg_mean = self.z_seg_mean_fc(hidden)
            z_seg_log_cov = self.z_seg_log_cov_fc(hidden)

            z_seg_sample = z_seg_mean
            hidden = self.z_seg_mask_fc_seq(z_seg_sample)
            hidden = hidden.view(hidden.size(0), self.x_enc_ch, self.x_enc_h, self.x_enc_h)
            mask = self.z_seg_mask_upconv_seq(hidden)
            mask = self.z_seg_mask_conv(hidden)
            mask = torch.sigmoid(mask)

            hidden = mask*enc
            # hidden = self.z_obj_conv_seq(hidden)
            hidden = hidden.view(hidden.size(0), -1)
            hidden = self.z_obj_fc_seq(hidden)
            z_obj_mean = self.z_obj_mean_fc(hidden)
            z_obj_log_cov = self.z_obj_log_cov_fc(hidden)
            
            z_obj_sample = z_obj_mean
            hidden = self.z_obj_recon_fc_seq(z_obj_sample)
            hidden = hidden.view(hidden.size(0), self.x_enc_ch, self.x_enc_h, self.x_enc_h)
            # hidden = self.z_obj_recon_upconv_seq(hidden)
            hidden = hidden * mask
            inter_recon_tensor = inter_recon_tensor + num_batch[:,i]*hidden

            enc_mask_aggregate = torch.max(enc_mask_aggregate, mask)

            obj_means.append(z_obj_mean)
            obj_log_covs.append(z_obj_log_cov)
            seg_means.append(z_seg_mean)
            seg_log_covs.append(z_seg_log_cov)
            masks.append(mask)
        
        # recon
        hidden = self.recon_upconv_seq(inter_recon_tensor)
        recon_mean = self.recon_mean_conv(hidden)
        recon_mean = torch.sigmoid(recon_mean)
        recon_log_cov = self.recon_log_cov_conv(hidden)

        return obj_means, obj_log_covs, seg_means, seg_log_covs, masks, recon_mean, recon_log_cov
    

    def compute_KL(self, post_mean, post_log_cov):
        return -0.5 * torch.sum(
            1 + post_log_cov - post_mean**2 - torch.exp(post_log_cov)
        )


    def compute_ELBO(
        self,
        post_mean_list, post_log_cov_list,
        recon_mean, recon_log_cov,
        obs_batch,
        average_over_batch=True
    ):
        KL = 0.
        for mean, log_cov in zip(post_mean_list, post_log_cov_list):
            KL = KL + self.compute_KL(mean, log_cov)
        
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

        elbo = log_prob - 0.1 * KL
        if average_over_batch: elbo = elbo / float(obs_batch.size(0))

        return elbo, KL
