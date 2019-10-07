import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from rlkit.torch.core import PyTorchModule
from rlkit.torch.networks import ConvNet, Mlp

from rlkit.torch import pytorch_util as ptu
from rlkit.torch.torch_meta_irl_algorithm import np_to_pytorch_batch
from rlkit.torch.irl.encoders.aggregators import sum_aggregator_unmasked, tanh_sum_aggregator_unmasked
from rlkit.torch.distributions import ReparamMultivariateNormalDiag

from rlkit.torch.irl.encoders.trivial_encoder import TrivialTrajEncoder, TrivialR2ZMap


class SingleStepProcessor(PyTorchModule):
    '''
        For when observations are pixels
    '''
    def __init__(self, conv_net_params, fc_params):
        self.save_init_params(locals())
        super().__init__()

        self.conv_part = ConvNet(**conv_net_params)
        self.fc_part = Mlp(**fc_params)
    

    def forward(self, obs, acts):
        obs = self.conv_part(obs)
        obs = F.relu(obs)
        obs = obs.view(obs.size(0), -1)

        inp = torch.cat([obs, acts], dim=1)
        out = self.fc_part(inp)
        return F.relu(out)


class TrivialConvNPEncoder(PyTorchModule):
    def __init__(
        self,
        agg_type,
        traj_encoder,
        r_to_z_map
    ):
        self.save_init_params(locals())
        super().__init__()

        self.r_to_z_map = r_to_z_map
        self.traj_encoder = traj_encoder

        if agg_type == 'sum':
            self.agg = sum_aggregator_unmasked
        elif agg_type == 'tanh_sum':
            self.agg = tanh_sum_aggregator_unmasked
        else:
            raise Exception('Not a valid aggregator!')
    

    def forward(self, context):
        '''
            For this first version of trivial encoder we are going
            to assume all tasks have the same number of trajs and
            all trajs have the same length
        '''
        # first convert the list of list of dicts to a tensor of dims

        # N_tasks x N_trajs x Channels x Height x Width
        obs = np.array([[np.transpose(d['observations']['pixels'], axes=(2,0,1)) for d in task_trajs] for task_trajs in context])
        # N_tasks x N_trajs x Len x Dim
        acts = np.array([[d['actions'] for d in task_trajs] for task_trajs in context])

        obs = Variable(ptu.from_numpy(obs), requires_grad=False)




        all_timesteps = Variable(ptu.from_numpy(all_timesteps), requires_grad=False)

        traj_embeddings = self.traj_encoder(all_timesteps)

        r = self.agg(traj_embeddings)
        post_mean, post_log_sig_diag = self.r_to_z_map(r)

        return ReparamMultivariateNormalDiag(post_mean, post_log_sig_diag)
