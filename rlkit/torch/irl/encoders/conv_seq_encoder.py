import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from rlkit.torch.core import PyTorchModule
from rlkit.torch.networks import Mlp

from rlkit.torch import pytorch_util as ptu
from rlkit.torch.torch_meta_irl_algorithm import np_to_pytorch_batch
from rlkit.torch.irl.encoders.aggregators import sum_aggregator_unmasked, tanh_sum_aggregator_unmasked
from rlkit.torch.irl.encoders.aggregators import sum_aggregator, tanh_sum_aggregator
from rlkit.torch.distributions import ReparamMultivariateNormalDiag


class ConvTrajEncoder(PyTorchModule):
    def __init__(
        self,
        num_conv_layers,
        input_ch, # obs or obs + act depending on state-only or not
        channels,
        kernel,
        stride
    ):
        self.save_init_params(locals())
        super().__init__()

        assert num_conv_layers > 0

        mod_list = nn.ModuleList([
            nn.Conv1d(input_ch, channels, kernel, stride=stride, padding=0, dilation=1, groups=1, bias=True),
            nn.BatchNorm1d(channels),
            nn.ReLU()
        ])
        for _ in range(num_conv_layers-2):
            mod_list.extend([
                nn.Conv1d(channels, channels, kernel, stride=stride, padding=0, dilation=1, groups=1, bias=True),
                nn.BatchNorm1d(channels),
                nn.ReLU()
            ])
        mod_list.append(
            nn.Conv1d(channels, channels, kernel, stride=stride, padding=0, dilation=1, groups=1, bias=True)
        )
        self.conv_part = nn.Sequential(*mod_list)


    def forward(self, batch):
        batch = batch.permute(0,1,3,2).contiguous()
        N_tasks, N_trajs, dim, traj_len = batch.size(0), batch.size(1), batch.size(2), batch.size(3)

        batch = batch.view(N_tasks*N_trajs, dim, traj_len)
        embeddings = self.conv_part(batch)
        embeddings = torch.mean(embeddings, dim=-1)
        embeddings = embeddings.view(N_tasks, N_trajs, -1)

        return embeddings


class R2ZMap(PyTorchModule):
    def __init__(
        self,
        num_layers,
        r_dim,
        hid_dim,
        z_dim,
        LOG_STD_SUBTRACT_VALUE=2.0
    ):
        self.save_init_params(locals())
        super().__init__()

        mod_list = nn.ModuleList([
            nn.Linear(r_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU()
        ])
        for _ in range(num_layers - 1):
            mod_list.extend([
                nn.Linear(hid_dim, hid_dim),
                nn.BatchNorm1d(hid_dim),
                nn.ReLU()
            ])
        self.trunk = nn.Sequential(*mod_list)
        self.mean_fc = nn.Linear(hid_dim, z_dim)
        self.log_sig_fc = nn.Linear(hid_dim, z_dim)

        self.LOG_STD_SUBTRACT_VALUE = LOG_STD_SUBTRACT_VALUE

        print('LOG STD SUBTRACT VALUE IS FOR APPROX POSTERIOR IS %f' % LOG_STD_SUBTRACT_VALUE)


    def forward(self, r):
        trunk_output = self.trunk(r)
        mean = self.mean_fc(trunk_output)
        log_sig = self.log_sig_fc(trunk_output) - self.LOG_STD_SUBTRACT_VALUE
        return mean, log_sig


class Dc2RMap(PyTorchModule):
    def __init__(
        self,
        agg_type,
        traj_encoder,
        state_only=False
    ):
        self.save_init_params(locals())
        super().__init__()

        self.traj_encoder = traj_encoder
        self.state_only = state_only

        if agg_type == 'sum':
            self.agg = sum_aggregator_unmasked
            self.agg_masked = sum_aggregator
        elif agg_type == 'tanh_sum':
            self.agg = tanh_sum_aggregator_unmasked
            self.agg_masked = tanh_sum_aggregator
        else:
            raise Exception('Not a valid aggregator!')
    

    def forward(self, context, mask=None):
        # first convert the list of list of dicts to a tensor of dims
        # N_tasks x N_trajs x Len x Dim

        obs = np.array([[d['observations'] for d in task_trajs] for task_trajs in context])
        acts = np.array([[d['actions'] for d in task_trajs] for task_trajs in context])
        next_obs = np.array([[d['next_observations'] for d in task_trajs] for task_trajs in context])

        if not self.state_only:
            # all_timesteps = np.concatenate([obs, acts], axis=-1)
            all_timesteps = np.concatenate([obs, acts, next_obs], axis=-1)
        else:
            # all_timesteps = obs
            all_timesteps = np.concatenate([obs, next_obs], axis=-1)
        all_timesteps = Variable(ptu.from_numpy(all_timesteps), requires_grad=False)

        traj_embeddings = self.traj_encoder(all_timesteps)

        if mask is None:
            r = self.agg(traj_embeddings)
        else:
            r = self.agg_masked(traj_embeddings, mask)
        return r


class NPEncoder(PyTorchModule):
    def __init__(
        self,
        Dc_to_r_map,
        r_to_z_map
    ):
        self.save_init_params(locals())
        super().__init__()

        self.Dc_to_r_map = Dc_to_r_map
        self.r_to_z_map = r_to_z_map


    def __call__(self, context=None, mask=None, r=None):
        if r is None:
            r = self.Dc_to_r_map(context, mask=mask)
        post_mean, post_log_sig_diag = self.r_to_z_map(r)
        return ReparamMultivariateNormalDiag(post_mean, post_log_sig_diag)
