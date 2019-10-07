import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from rlkit.torch.pytorch_util import fanin_init
from rlkit.torch.distributions import ReparamMultivariateNormalDiag


class SimpleMazeEncoder(nn.Module):
    def __init__(self, maze_dims, action_dim, r_dim, num_objects, conv_part_specs, fc_part_specs, use_bn=False):
        assert False, 'What about batch norm and init?'
        assert maze_dims[1] == maze_dims[2], 'Assuming square input maze'
        self.maze_dims = maze_dims
        self.action_dim = action_dim
        self.r_dim = r_dim
        self.num_objects = num_objects
        self.conv_part_specs = conv_part_specs
        self.fc_part_specs = fc_part_specs
        self.use_bn = use_bn

        kernel_sizes = conv_part_specs['kernel_sizes']
        num_channels = conv_part_specs['num_channels']
        strides = conv_part_specs['strides']
        paddings = conv_part_specs['paddings']
        fc_hidden_sizes = fc_part_specs['hidden_sizes']

        self.action_fc = nn.Linear(action_dim, maze_dims[1]*maze_dims[2], bias=True)
        self.reward_fc = nn.Linear(1, maze_dims[1]*maze_dims[2], bias=True)

        self.conv_list = nn.ModuleList()
        in_ch = maze_dims[0] + 2
        in_h = maze_dims[1]
        for k, ch, s, p in zip(kernel_sizes, num_channels, strides, paddings):
            seq = [nn.Conv2d(in_ch, ch, k, stride=s, padding=p, bias=not use_bn)]
            if use_bn: seq.append(nn.BatchNorm2d(ch))
            seq.append(nn.ReLU())
            self.conv_list.extend(seq)

            in_ch = ch
            in_h = int(math.floor(
                1 + (in_h + 2*p - k)/s
            ))
        self.conv_seq = nn.Sequential(*self.conv_list)

        self.fc_list = nn.ModuleList()
        in_size = in_ch * in_h * in_h
        for out_size in fc_hidden_sizes:
            seq = [nn.Linear(in_size, out_size, bias=not use_bn)]
            if use_bn: seq.append(nn.BatchNorm1d(out_size))
            seq.append(nn.ReLU())
            self.fc_list.extend(seq)
        self.fc_list.append(nn.Linear(out_size, r_dim))
        self.fc_seq = nn.Sequential(*self.fc_list)


    def forward(self, batch_list):
        '''
        Assuming the batch_list contains: [obs_batch, action_batch, next_obs_batch, rewards_batch]
        Output is [r_batch]
        '''
        pre_proc_acts = self.action_fc(batch_list[1]).view(-1, 1, self.maze_dims[1], self.maze_dims[2])
        pre_proc_rews = self.reward_fc(batch_list[3]).view(-1, 1, self.maze_dims[1], self.maze_dims[2])

        inputs = torch.cat([pre_proc_acts, pre_proc_rews, batch_list[0], batch_list[2]])
        conv_output = self.conv_seq(inputs)
        r = self.fc_seq(conv_output.view(conv_output.size(0), -1))

        return [r]


class MazeBaseMap(nn.Module):
    def __init__(self, num_objects, z_dim, maze_dims, action_dim, conv_part_specs, reward_part_specs, upconv_part_specs, use_bn=False):
        assert maze_dims[1] == maze_dims[2], 'Assuming square input maze'
        self.num_objects = num_objects
        self.z_dim = z_dim,
        self.conv_part_specs = conv_part_specs
        self.reward_part_specs = reward_part_specs
        self.upconv_part_specs = upconv_part_specs
        self.use_bn = use_bn
        self.maze_dims = maze_dims
        self.action_dim = action_dim

        # Convolutions down to a bottleneck layer
        self.conv_list = nn.ModuleList()
        conv_kernel_sizes = conv_part_specs['kernel_sizes']
        conv_num_channels = conv_part_specs['num_channels']
        conv_strides = conv_part_specs['strides']
        conv_paddings = conv_part_specs['paddings']
        in_ch = maze_dims[0]
        in_h = maze_dims[1]
        for k, ch, s, p in zip(conv_kernel_sizes, conv_num_channels, conv_strides, paddings):
            seq = [nn.Conv2d(in_ch, ch, k, stride=s, padding=p, bias=not use_bn)]
            if use_bn: seq.append(nn.BatchNorm2d(ch))
            seq.append(nn.ReLU())
            self.conv_list.extend(seq)

            in_ch = ch
            in_h = int(math.floor(
                1 + (in_h + 2*p - k)/s
            ))
        self.conv_seq = nn.Sequential(*self.conv_list)

        # From the intermediate representation we will predict the reward
        self.rew_conv_list = nn.ModuleList()
        rew_conv_kernel_sizes = rew_part_specs['conv_part']['kernel_sizes']
        rew_conv_num_channels = rew_part_specs['conv_part']['num_channels']
        rew_conv_strides = rew_part_specs['conv_part']['strides']
        rew_conv_paddings = rew_part_specs['conv_part']['paddings']
        in_ch = in_ch + z_dim + 1
        for k, ch, s, p in zip(rew_conv_kernel_sizes, rew_conv_num_channels, rew_conv_strides, rew_paddings):
            seq = [nn.Conv2d(in_ch, ch, k, stride=s, padding=p, bias=not use_bn)]
            if use_bn: seq.append(nn.BatchNorm2d(ch))
            seq.append(nn.ReLU())
            self.conv_list.extend(seq)

            in_ch = ch
            in_h = int(math.floor(
                1 + (in_h + 2*p - k)/s
            ))
        self.rew_conv_seq = nn.Sequential(*self.conv_list)

        hidden_sizes = rew_part_specs['fc_part']['hidden_sizes']
        self.reward_fc_list = nn.ModuleList()
        for next_dim in hidden_sizes:
            seq = [nn.Linear(hid_dim, next_dim, bias=not use_bn)]
            if use_bn: seq.append(nn.BatchNorm1d(next_dim))
            seq.append(nn.ReLU)
            self.reward_fc_list.extend(seq)
            hid_dim = next_dim
        self.reward_fc_list.append(nn.Linear(hid_dim, 1, bias=True))
        self.rew_fc_seq = nn.Sequential(*self.reward_fc_list)

        # Concatenate the output to the intermediate representations to pass through upconv layers
        self.action_fc = nn.Linear(action_dim, in_h*in_h, bias=True)

        # Upconv layers to generate the prediction for the next maze observation
        self.up_conv_list = nn.ModuleList()
        upconv_kernel_sizes = upconv_part_specs['kernel_sizes']
        upconv_num_channels = upconv_part_specs['num_channels']
        upconv_strides = upconv_part_specs['strides']
        upconv_paddings = upconv_part_specs['paddings']
        upconv_output_paddings = upconv_part_specs['output_paddings']
        in_ch = in_ch + z_dim + 1
        for k, ch, s, p, op in zip(conv_kernel_sizes, conv_num_channels, conv_strides, paddings, output_paddings):
            seq = [nn.ConvTranspose2d(in_ch, ch, k, stride=s, padding=p, output_padding=op, bias=not use_bn)]
            if use_bn: seq.append(nn.BatchNorm2d(ch))
            seq.append(nn.ReLU())
            self.upconv_list.extend(seq)

            in_ch = ch
            in_h = (in_h - 1)*s - 2*p + k + op
        self.upconv_list.append(nn.Conv2d(in_ch, 3, 1, stride=1, padding=0, bias=True))
        self.upconv_list.append(nn.Sigmoid())
        self.upconv_seq = nn.Sequential(*self.upconv_list)

        assert in_h == maze_dims[1], 'Output size does not match'


    def forward(self, seg_priors, z_seg_batch_list, z_prop_list, batch_list):
        '''
        Assuming:
        batch_list = [obs_batch, action_batch]
        obs_batch.shape = [N, 3, Height, Width]
        K = num_objects
        assert seg_priors.shape == [N, K]
        z_seg_batch_list = [[mu_1, log_cov_diag_1], ..., [mu_K, log_cov_diag_K]]
        z_prop_list = [z_1, ..., z_K] # samples from the posterior that has dim z_dim
        mu and log_cov_diag shapes are [N, C]
        assert C == half the dim of obs after being processed by conv part
        '''
        obs = batch_list[0]
        act = batch_list[1]
        new_z_seg_batch_list = []

        # Fix the sizes of things ---------------------------------------------
        seg_priors = torch.unsqueeze(torch.unsqueeze(mean, -1), -1) # N x K x 1 x 1

        for mean, log_cov_diag in z_seg_batch_list:
            mean = torch.unsqueeze(torch.unsqueeze(mean, -1), -1) # N x C x 1 x 1
            log_cov_diag = torch.unsqueeze(torch.unsqueeze(log_cov_diag, -1), -1) # N x C x 1 x 1
            new_z_seg_batch_list.append((mean, log_cov_diag))
        z_seg_batch_list = new_z_seg_batch_list
        
        new_z_prop_list = []
        for z in z_prop_list:
            z = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(z, -1), -1), -1) # N x z_dim x 1 x 1 x 1
            new_z_seg_batch_list.append(z)
        z_prop_list = new_z_prop_list
        z_prop_tensor = torch.cat(z_prop_list, dim=2) # N x z_dim x K x 1 x 1
        
        # Get the bottleneck activations --------------------------------------
        assert False, "The last layer of the conv_seq has to be linear without relu so that you can do MoG"
        processed_obs = self.conv_seq(obs) # N x 2C x H x H
        proc_obs_splits = processed_obs.split(processed_obs.size(1)/2, dim=1)
        seg_part = proc_obs_splits[0]
        local_state = proc_obs_splits[1]

        # Compute segmentation using Mixture of Gaussians and use it to compute the properties tensor
        p_x_given_z = []
        for mean, log_cov_diag in z_seg_batch_list:
            cov_diag = torch.exp(log_cov_diag)
            # mean: N x C x 1 x 1
            # log_cov_diag: N x C x 1 x 1
            log_prob = -0.5 * torch.sum(
                (mean - seg_part) ** 2 / cov_diag,
                1,
                keepdim=True
            )
            prob = torch.exp(log_prob)
            rest = ((1/(2*pi))**0.5) * torch.exp(-0.5 * torch.sum(log_cov_diag, 1, keepdim=True))
            prob = prob * rest
            # prob: N x 1 x H x H
        p_x_given_z = torch.cat(p_x_given_z, dim=1) # N x K x H x H
        p_x_and_z = p_x_given_z * seg_priors
        normalizer = torch.sum(p_x_and_z, dim=1, keepdim=True)
        p_z_given_x = p_x_and_z / normalizer # N x K x H x H
        p_z_given_x = torch.unsqueeze(p_z_given_x, dim=1) # N x 1 x K x H x H

        prop_tensor = p_z_given_x * z_prop_tensor # N x z_dim x K x H x H
        prop_tensor = torch.sum(prop_tensor, dim=2) # N x z_dim x H x H

        # Compute the new representation from which we predict reward and next obs
        processed_act = self.action_fc(act).view(local_state.size(0), 1, local_state.size(2), local_state.size(3))
        bottleneck_activations = torch.cat(
            [
                processed_act, prop_tensor, local_state
            ],
            dim=1
        )

        # Compute reward predictions
        rew_hid = self.rew_conv_seq(bottleneck_activations)
        rew_pred = self.rew_fc_seq(rew_hid.view(rew_hid.size(0), -1))

        # Compute next obs predictions
        next_obs_pred = self.upconv_seq(bottleneck_activations)

        return [[next_obs_pred], [rew_pred]]
