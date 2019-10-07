import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from rlkit.torch.core import PyTorchModule
from rlkit.torch.networks import Mlp

from rlkit.torch import pytorch_util as ptu
from rlkit.torch.torch_meta_irl_algorithm import np_to_pytorch_batch
from rlkit.torch.distributions import ReparamMultivariateNormalDiag

class PusherVideoEncoder(PyTorchModule):
    def __init__(
        self,
        # z_dims, # z will take the form of a conv kernel
    ):
        self.save_init_params(locals())
        super().__init__()

        # self.z_dims = [-1] + z_dims

        CH = 32
        # kernel = (10, 10, 10) # depth, height, width
        kernel = (3, 5, 5) # depth, height, width
        stride = (2, 2, 2) # depth, height, width
        # self.conv = nn.Sequential(
        #     nn.Conv3d(3, CH, kernel, stride=stride, bias=True),
        #     nn.ReLU(),
        #     nn.Conv3d(CH, CH, kernel, stride=stride, bias=True),
        #     nn.ReLU(),
        #     nn.Conv3d(CH, CH, kernel, stride=stride, bias=True),
        #     nn.ReLU(),
        #     # nn.Conv3d(CH, CH, (1,1,1), (1,1,1), bias=True),
        # )
        self.all_convs = nn.ModuleList(
            [
                nn.Conv3d(3, CH, kernel, stride=stride, bias=True),
                nn.Conv3d(CH, CH, kernel, stride=stride, bias=True),
                nn.Conv3d(CH, CH, kernel, stride=stride, bias=True),
                nn.Conv3d(CH, CH, kernel, stride=stride, bias=True),
            ]
        )

        FLAT_SIZE = CH*1*5*5
        self.fc = nn.Sequential(
            nn.Linear(FLAT_SIZE, FLAT_SIZE),
            nn.ReLU(),
            nn.Linear(FLAT_SIZE, FLAT_SIZE),
            nn.ReLU(),
            nn.Linear(FLAT_SIZE, 3*8*5*5 + 16*8*5*5 + 16*8*5*5)
            # nn.Linear(FLAT_SIZE, 3*8*5*5 + 16*8*5*5 + 16*8*5*5 + 16*8*5*5)
        )
        # self.z1_fc = nn.Linear(FLAT_SIZE, 3*8*5*5)
        # self.z2_fc = nn.Linear(FLAT_SIZE, 16*8*5*5)
        # self.z3_fc = nn.Linear(FLAT_SIZE, 16*8*5*5)
        # self.z4_fc = nn.Linear(FLAT_SIZE, 16*8*5*5)
        # FLAT_SIZE = CH*13*13
        # OUT_SIZE = int(np.prod(z_dims))
        # self.fc = nn.Sequential(
        #     nn.Linear(FLAT_SIZE, OUT_SIZE),
        #     nn.ReLU(),
        #     nn.Linear(OUT_SIZE, OUT_SIZE),
        #     nn.ReLU(),
        #     nn.Linear(OUT_SIZE, OUT_SIZE)
        # )
    
    def forward(self, demo_batch):
        h = demo_batch
        for i, c in enumerate(self.all_convs):
            h = c(h)
            h = F.relu(h)
        output = h
        output = torch.sum(output, dim=2)
        output = output.view(
            output.size(0),
            output.size(1) * output.size(2) * output.size(3)
        )
        # z = self.fc(output).view(*self.z_dims)
        z = self.fc(output)
        return z


class PusherLastTimestepEncoder(PyTorchModule):
    def __init__(
        self,
    ):
        self.save_init_params(locals())
        super().__init__()
        
        CH = 32
        k = 5
        s = 2
        p = 2
        self.conv_part = nn.Sequential(
            nn.Conv2d(3, CH, k, stride=s, padding=p),
            # nn.BatchNorm2d(CH),
            nn.ReLU(),
            nn.Conv2d(CH, CH, k, stride=s, padding=p),
            # nn.BatchNorm2d(CH),
            nn.ReLU(),
            nn.Conv2d(CH, CH, k, stride=s, padding=p),
            # nn.BatchNorm2d(CH),
            nn.ReLU(),
        )
        flat_dim = CH * 6 * 6
        self.fc_part = nn.Sequential(
            nn.Linear(flat_dim, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Linear(256, 64)
            nn.Linear(256, 3*32)
        )

    
    def forward(self, demo_batch):
        conv_output = self.conv_part(demo_batch)
        conv_output = conv_output.view(conv_output.size(0), -1)
        fc_output = self.fc_part(conv_output)

        return fc_output



class PusherAggTimestepEncoder(PyTorchModule):
    def __init__(
        self,
        state_dim,
        action_dim
    ):
        self.save_init_params(locals())
        super().__init__()

        # make the coordinate maps of different resolutions
        self.all_xy_maps = []
        x_map_2d, y_map_2d = np.meshgrid(
            np.arange(-62.0, 63.0),
            np.arange(-62.0, 63.0),
        )
        x_map_2d = x_map_2d[None,None] / 62.0
        y_map_2d = y_map_2d[None,None] / 62.0
        xy_map_2d = np.concatenate((x_map_2d, y_map_2d), axis=1)
        self.all_xy_maps.append(Variable(ptu.from_numpy(xy_map_2d), requires_grad=False))

        x_map_2d, y_map_2d = np.meshgrid(
            np.arange(-31, 32),
            np.arange(-31, 32),
        )
        x_map_2d = x_map_2d[None,None] / 31.0
        y_map_2d = y_map_2d[None,None] / 31.0
        xy_map_2d = np.concatenate((x_map_2d, y_map_2d), axis=1)
        self.all_xy_maps.append(Variable(ptu.from_numpy(xy_map_2d), requires_grad=False))
        for d in [32.0, 16.0]:
            x_map_2d, y_map_2d = np.meshgrid(
                np.arange(-d/2, d/2),
                np.arange(-d/2, d/2),
            )
            x_map_2d = (x_map_2d[None,None] + 0.5) / (d/2)
            y_map_2d = (y_map_2d[None,None] + 0.5) / (d/2)
            xy_map_2d = np.concatenate((x_map_2d, y_map_2d), axis=1)
            self.all_xy_maps.append(Variable(ptu.from_numpy(xy_map_2d), requires_grad=False))
        
        CH = 32
        k = 5
        s = 2
        p = 2
        self.convs_list = nn.ModuleList(
            [
                nn.Conv2d(3+2, CH, k, stride=s, padding=p),
                nn.Conv2d(CH+2, CH, k, stride=s, padding=p),
                nn.Conv2d(CH+2, CH, k, stride=s, padding=p),
            ]
        )
        self.conv_bns = nn.ModuleList(
            [
                nn.BatchNorm2d(CH),
                nn.BatchNorm2d(CH),
                nn.BatchNorm2d(CH),
            ]
        )
        self.last_conv = nn.Conv2d(CH+2, 128, 1, stride=1, padding=0)

        self.just_img_fc_part = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.film_last_fcs = nn.ModuleList(
            [
                nn.Linear(128, 32),
                nn.Linear(128, 32),
                nn.Linear(128, 32),
            ]
        )

        flat_dim = 128 + state_dim + action_dim
        
        self.extra_latent_fc_part = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.extra_latent_post_agg_fc_part = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    
    def forward(self, demo_batch):
        # N_tasks x N_samples x 3 x H x W
        images = demo_batch['image']
        # N_tasks x N_samples x dim
        states = demo_batch['X']
        # N_tasks x N_samples x dim
        actions = demo_batch['U']

        N_tasks, N_samples = images.size(0), images.size(1)
        total_per_task = N_tasks * N_samples
        images = images.view(total_per_task, images.size(2), images.size(3), images.size(4))
        states = states.view(total_per_task, -1)
        actions = actions.view(total_per_task, -1)

        # process the images
        h = images
        for i in range(len(self.convs_list)):
            coords = self.all_xy_maps[i]
            coords = coords.repeat(h.size(0), 1, 1, 1)
            h = torch.cat([h, coords], dim=1)
            h = self.convs_list[i](h)
            h = self.conv_bns[i](h)
            h = F.relu(h)
        coords = self.all_xy_maps[-1]
        coords = coords.repeat(h.size(0), 1, 1, 1)
        h = torch.cat([h, coords], dim=1)
        h = self.last_conv(h)
        h = F.avg_pool2d(h, 16)
        h = h.view(h.size(0), -1)

        # get the extra latent
        extra_latent_input = torch.cat([h, states, actions], dim=-1)
        extra_latent_pre_agg = self.extra_latent_fc_part(extra_latent_input)
        extra_latent_pre_agg = extra_latent_pre_agg.view(N_tasks, N_samples, extra_latent_pre_agg.size(1))
        extra_latent_agg = torch.sum(extra_latent_pre_agg, dim=1)
        extra_latent_filters = self.extra_latent_post_agg_fc_part(extra_latent_agg)

        # get film feature
        agg_h = torch.sum(h.view(N_tasks, N_samples, h.size(1)), dim=1)
        post_fc_agg_img_feats = self.just_img_fc_part(agg_h)
        film_feats = [last_fc(post_fc_agg_img_feats) for last_fc in self.film_last_fcs]

        return film_feats, extra_latent_filters


if __name__ == '__main__':
    p = PusherVideoEncoder()
    x = Variable(torch.rand(16, 3, 100, 125, 125))
    p.cuda()
    x = x.cuda()
    y = p(x)
    print(torch.max(y), torch.min(y), y.size())
