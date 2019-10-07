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

class MaskedNormalVAE(nn.Module):
    def __init__(
        self,
        maze_dims,
        z_dim,
        encoder_specs,
        decoder_specs
    ):
        super().__init__()

        in_ch = maze_dims[0]
        in_h = maze_dims[1]
        
        # make the encoder
        self.encoder_conv_seq, x_enc_ch, x_enc_h = make_conv_net(in_ch, in_h, encoder_specs['conv_part_specs'])
        self.x_enc_ch = x_enc_ch
        self.x_enc_h = x_enc_h
        flat_inter_img_dim = x_enc_ch * x_enc_h * x_enc_h

        self.enc_mask_seq, _, _ = make_conv_net(
            x_enc_ch, x_enc_h,
            {
                'kernel_sizes': [3],
                'num_channels': [64],
                'strides': [2],
                'paddings': [1],
                'use_bn': True
            }
        )
        self.enc_mask_conv = nn.Conv2d(64, 1, 1, stride=1, padding=0, bias=True)

        # meshgrid
        xv, yv = np.meshgrid(np.linspace(-1.,1.,x_enc_h), np.linspace(-1.,1.,x_enc_h))
        xv, yv = xv[None,None,...], yv[None,None,...]
        xv, yv = torch.FloatTensor(xv), torch.FloatTensor(yv)
        self.mesh = torch.cat([xv, yv], 1)
        self.mesh = Variable(self.mesh, requires_grad=False).cuda()
        
        self.encoder_fc_seq, h_dim = make_fc_net(flat_inter_img_dim, encoder_specs['fc_part_specs'])

        self.z_mean_fc = nn.Linear(h_dim, z_dim, bias=True)
        self.z_log_cov_fc = nn.Linear(h_dim, z_dim, bias=True)

        # make the decoder
        self.decoder_fc_seq, h_dim = make_fc_net(z_dim, decoder_specs['fc_part_specs'])
        # assert h_dim == flat_inter_img_dim
        self.decoder_upconv_seq, out_ch, out_h = make_upconv_net(130, x_enc_h, decoder_specs['upconv_part_specs'])

        self.recon_mean_conv = nn.Conv2d(out_ch, 1, 1, stride=1, padding=0, bias=True)
        self.recon_log_cov_conv = nn.Conv2d(out_ch, 1, 1, stride=1, padding=0, bias=True)
        assert out_h == maze_dims[1], str(out_h) + ' != ' + str(maze_dims[1])


    def forward(self, img_batch):
        enc = self.encoder_conv_seq(img_batch)
        enc_mask = self.enc_mask_seq(enc)
        enc_mask = self.enc_mask_conv(enc)
        enc_mask = torch.sigmoid(enc_mask)
        enc = enc * enc_mask
        enc = enc.view(enc.size(0), -1)
        enc = self.encoder_fc_seq(enc)

        z_mean = self.z_mean_fc(enc)
        z_log_cov = self.z_log_cov_fc(enc)
        z_log_cov = torch.clamp(z_log_cov, LOG_COV_MIN, LOG_COV_MAX)

        z_sample = z_mean
        # ----------------
        # eps = Variable(torch.randn(z_mean.size()))
        # if z_mean.is_cuda: eps = eps.cuda()
        # z_sample = z_mean + eps*torch.exp(0.5 * z_log_cov)

        # dec = self.decoder_fc_seq(z_sample)
        # dec = dec.view(dec.size(0), self.x_enc_ch, self.x_enc_h, self.x_enc_h)
        # dec = dec * enc_mask
        # dec = self.decoder_upconv_seq(dec)
        # ----------------
        dec = self.decoder_fc_seq(z_sample)
        # dec = z_sample
        dec = torch.unsqueeze(torch.unsqueeze(dec, -1), -1)
        dec = enc_mask * dec
        dec = torch.cat([dec, self.mesh.expand(dec.size(0),-1,-1,-1)], 1)
        dec = self.decoder_upconv_seq(dec)

        recon_mean = self.recon_mean_conv(dec)
        recon_mean = torch.sigmoid(recon_mean)
        recon_log_cov = self.recon_log_cov_conv(dec)
        recon_log_cov = torch.clamp(recon_log_cov, LOG_COV_MIN, LOG_COV_MAX)

        return z_mean, z_log_cov, recon_mean, recon_log_cov, enc_mask
    

    def compute_KL(self, post_mean, post_log_cov):
        return -0.5 * torch.sum(
            1 + post_log_cov - post_mean**2 - torch.exp(post_log_cov)
        )
    

    def compute_log_prob(self, recon_mean, recon_log_cov, obs_batch):
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
        return log_prob


    def compute_ELBO(
        self,
        post_mean, post_log_cov,
        recon_mean, recon_log_cov,
        obs_batch,
        average_over_batch=True
    ):
        KL = self.compute_KL(post_mean, post_log_cov)
        log_prob = self.compute_log_prob(recon_mean, recon_log_cov, obs_batch)

        elbo = log_prob - 0. * KL
        if average_over_batch: elbo = elbo / float(obs_batch.size(0))

        return elbo, KL
