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


def compute_diag_log_prob(preds_mean, preds_log_cov, true_outputs):
    '''
        Compute log prob assuming diagonal Gaussian with some mean and log cov
    '''
    # preds_cov = torch.exp(preds_log_cov)

    # log_prob = -0.5 * torch.sum(
    #     (preds_mean - true_outputs)**2 / preds_cov
    # )

    # log_det_temp = torch.sum(preds_log_cov, 1) + log_2pi
    # log_prob = -0.5*torch.sum(log_det_temp) + log_prob

    log_prob = -0.5 * torch.sum((preds_mean - true_outputs)**2)

    return log_prob


class VAE(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

        # CONV VERSION
        self.img_h = 16
        inter_h = 4
        inter_ch = 32
        z_dim = inter_h * inter_h * inter_ch
        self.inter_h = inter_h
        self.inter_ch = inter_ch

        self.encoder = make_conv_net(
            3, self.img_h, {
                'kernel_sizes': [4, 4],
                'num_channels': [inter_ch, inter_ch],
                'strides': [2, 2],
                'paddings': [1, 1],
                'use_bn': True,
            }
        )[0]

        # self.mask_net = make_conv_net(
        #     inter_ch, self.img_h, {
        #         'kernel_sizes': [3, 3],
        #         'num_channels': [4, 4],
        #         'strides': [1, 1],
        #         'paddings': [1, 1],
        #         'use_bn': True,
        #     }
        # )[0]
        self.mask_net = nn.Sequential(
            # self.mask_net,
            nn.Conv2d(inter_ch, 1, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

        # self.z_mean_fc = nn.Linear(inter_h*inter_h*inter_ch, z_dim, bias=True)
        # self.z_log_cov_fc = nn.Linear(inter_h*inter_h*inter_ch, z_dim, bias=True)
        self.z_mean_conv = nn.Conv2d(inter_ch, inter_ch, 3, stride=1, padding=1, bias=True)
        self.z_log_cov_conv = nn.Conv2d(inter_ch, inter_ch, 3, stride=1, padding=1, bias=True)        

        # self.decoder_fc = nn.Linear(z_dim, inter_h*inter_h*inter_ch)
        self.decoder = make_upconv_net(
            2*inter_ch, inter_h, {
                'kernel_sizes': [4, 4],
                'num_channels': [inter_ch, inter_ch],
                'strides': [2, 2],
                'paddings': [1, 1],
                'output_paddings': [0, 0],
                'use_bn': True
            }
        )[0]
        # self.decoder = nn.Sequential(
        #     self.decoder,
        #     nn.Conv2d(inter_ch, inter_ch, 5, stride=1, padding=2, bias=False),
        #     nn.BatchNorm2d(inter_ch),
        #     nn.ReLU()
        # )
        self.recon_mean_conv = nn.Conv2d(inter_ch, 3, 1, stride=1, padding=0, bias=True)
        self.recon_log_cov_conv = nn.Conv2d(inter_ch, 3, 1, stride=1, padding=0, bias=True)

        print(self.encoder)
        print(self.decoder)


        # # FC VERSION
        # self.img_h = 16
        # fc_dim = 256
        # z_dim = 128
        # self.encoder, _ = make_fc_net(
        #     16*16*3, {
        #         'hidden_sizes': [fc_dim, fc_dim, fc_dim],
        #         'use_bn': True
        #     }
        # )
        # self.z_mean_fc = nn.Linear(fc_dim, z_dim, bias=True)
        # self.z_log_cov_fc = nn.Linear(fc_dim, z_dim, bias=True)


        # self.decoder, _ = make_fc_net(
        #     z_dim, {
        #         'hidden_sizes': [fc_dim, fc_dim, fc_dim],
        #         'use_bn': True
        #     }
        # )
        # self.recon_mean_fc = nn.Linear(fc_dim, 16*16*3, bias=True)
        # self.recon_log_cov_fc = nn.Linear(fc_dim, 16*16*3, bias=True)


    def forward(self, obs_batch):
        # # FC VERSION (!!!!!!!!NOT SAMPLING ZZZZZZ!!!!!!!!)
        # b_size = obs_batch.size(0)
        # obs_batch = obs_batch.view(b_size, -1)
        # enc = self.encoder(obs_batch)
        # z_mean, z_log_cov = self.z_mean_fc(enc), self.z_log_cov_fc(enc)

        # # z_samples = Variable(torch.randn(z_mean.size())) * torch.exp(z_log_cov * 0.5) + z_mean
        # # print('not sampling z')
        # z_samples = z_mean
        
        # dec = self.decoder(z_samples)
        # recon_mean = self.recon_mean_fc(dec).view(b_size, 3, 16, 16)
        # recon_mean = F.sigmoid(recon_mean)
        # recon_log_cov = self.recon_log_cov_fc(dec).view(b_size, 3, 16, 16)


        # CONV VERSION
        b_size = obs_batch.size(0)
        enc = self.encoder(obs_batch)
        mask = self.mask_net(enc)
        # enc = enc.view(enc.size(0), -1)
        # z_mean, z_log_cov = self.z_mean_fc(enc), self.z_log_cov_fc(enc)
        z_mean, z_log_cov = self.z_mean_conv(enc), self.z_log_cov_conv(enc)
        z_log_cov = torch.clamp(z_log_cov, LOG_COV_MIN, LOG_COV_MAX)
        eps = Variable(torch.randn(z_mean.size()))
        if z_mean.is_cuda: eps = eps.cuda()
        z_samples = eps * torch.exp(z_log_cov * 0.5) + z_mean

        # hidden = self.decoder_fc(z_samples)
        # dec = self.decoder(hidden.view(-1, self.inter_ch, self.inter_h, self.inter_h))
        hidden = z_samples
        hidden = torch.cat([hidden*mask, hidden*(1.0 - mask)], 1)
        dec = self.decoder(hidden)
        recon_mean = self.recon_mean_conv(dec)
        recon_mean = F.sigmoid(recon_mean)
        recon_log_cov = self.recon_log_cov_conv(dec)
        recon_log_cov = torch.clamp(recon_log_cov, LOG_COV_MIN, LOG_COV_MAX)

        return recon_mean, recon_log_cov, z_mean, z_log_cov, mask
    

    def compute_KL(self, z_mean, z_log_cov):
        KL = -0.5 * torch.sum(
            1.0 + z_log_cov - z_mean**2 - torch.exp(z_log_cov)
        )
        return 0.001 * KL
        # return 0.


    def compute_ELBO(
        self,
        z_mean, z_log_cov,
        recon_mean, recon_log_cov,
        obs_batch
    ):
        batch_size = obs_batch.size(0)
        cond_log_likelihood = compute_diag_log_prob(recon_mean.view(batch_size, -1), recon_log_cov.view(batch_size, -1), obs_batch.view(batch_size, -1))
        KL = self.compute_KL(z_mean, z_log_cov)
        
        elbo = cond_log_likelihood - KL
        elbo = elbo / float(batch_size)

        return elbo
