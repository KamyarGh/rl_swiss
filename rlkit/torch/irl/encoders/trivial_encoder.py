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
from rlkit.torch.distributions import ReparamMultivariateNormalDiag


class TrivialTrajEncoder(PyTorchModule):
    '''
        Takes N_tasks x N_trajs x N_timesteps x Dim
        and gives you N_tasks x N_trajs x Dim
    '''
    def __init__(
        self,
        # params for the mlp that encodes each timestep
        timestep_enc_params,
        # params for the mlp that 
        traj_enc_params
    ):
        self.save_init_params(locals())
        super().__init__()

        timestep_enc_params['output_activation'] = F.relu
        self.timestep_mlp = Mlp(**timestep_enc_params)
        # the relu below that has been commented out seriously hurts performance
        # traj_enc_params['output_activation'] = F.relu
        self.traj_enc_mlp = Mlp(**traj_enc_params)
        self.output_size = self.traj_enc_mlp.output_size


    def forward(self, all_timesteps):
        # DONT FORGET TO REMOVE THIS
        all_timesteps = all_timesteps[:,:,-5:,:].contiguous()

        N_tasks, N_trajs, traj_len, dim = all_timesteps.size(0), all_timesteps.size(1), all_timesteps.size(2), all_timesteps.size(3)

        all_timesteps = all_timesteps.view(-1, dim)
        all_timesteps_embeddings = self.timestep_mlp(all_timesteps)
        
        dim = all_timesteps_embeddings.size(-1)
        all_timesteps_embeddings = all_timesteps_embeddings.view(N_tasks * N_trajs, traj_len * dim)
        all_trajs_embeddings = self.traj_enc_mlp(all_timesteps_embeddings)

        all_trajs_embeddings = all_trajs_embeddings.view(N_tasks, N_trajs, -1)
        return all_trajs_embeddings


# class TrivialLSTMEncoder(PyTorchModule):
#     def __init__(
#         self,
#     )


class TrivialR2ZMap(PyTorchModule):
    def __init__(
        self,
        trunk_params,
        split_heads_params
    ):
        self.save_init_params(locals())
        super().__init__()

        trunk_params['output_activation'] = F.relu
        self.trunk = Mlp(**trunk_params)
        self.mean_mlp = Mlp(**split_heads_params)
        self.log_sig_mlp = Mlp(**split_heads_params)
    

    def forward(self, r):
        common_part = self.trunk(r)
        mean = self.mean_mlp(common_part)
        log_sig = self.log_sig_mlp(common_part)
        return mean, log_sig


class TrivialNPEncoder(PyTorchModule):
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
        # N_tasks x N_trajs x Len x Dim

        obs = np.array([[d['observations'] for d in task_trajs] for task_trajs in context])
        acts = np.array([[d['actions'] for d in task_trajs] for task_trajs in context])

        all_timesteps = np.concatenate([obs, acts], axis=-1)
        all_timesteps = Variable(ptu.from_numpy(all_timesteps), requires_grad=False)

        traj_embeddings = self.traj_encoder(all_timesteps)

        r = self.agg(traj_embeddings)
        post_mean, post_log_sig_diag = self.r_to_z_map(r)

        return ReparamMultivariateNormalDiag(post_mean, post_log_sig_diag)

        # c_len = len(context)
        # return ReparamMultivariateNormalDiag(Variable(torch.zeros(c_len, 50), requires_grad=False), Variable(torch.ones(c_len, 50), requires_grad=False))
