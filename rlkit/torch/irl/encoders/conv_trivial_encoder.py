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

DIM = 24

class TrivialTrajEncoder(PyTorchModule):
    '''
        Assuming length of trajectories is 65 and we will take [::4] so 17 timesteps
        Dimensions are hard-coded for few shot fetch env
    '''
    def __init__(
        self,
        state_only=False
    ):
        self.save_init_params(locals())
        super().__init__()

        # V0
        # self.conv_part = nn.Sequential(
        #     nn.Conv1d(26, 50, 3, stride=2, padding=0, dilation=1, groups=1, bias=True),
        #     nn.BatchNorm1d(50),
        #     nn.ReLU(),
        #     nn.Conv1d(50, 50, 3, stride=2, padding=0, dilation=1, groups=1, bias=True),
        #     nn.BatchNorm1d(50),
        #     nn.ReLU()
        # )
        # self.mlp_part = nn.Sequential(
        #     nn.Linear(150, 50)
        # )
        # self.output_size = 50

        # V1
        # self.conv_part = nn.Sequential(
        #     nn.Conv1d(26 if not state_only else 22, 128, 3, stride=2, padding=0, dilation=1, groups=1, bias=True),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Conv1d(128, 128, 3, stride=2, padding=0, dilation=1, groups=1, bias=True),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Conv1d(128, 128, 3, stride=2, padding=0, dilation=1, groups=1, bias=True),
        # )

        # Similar to V1 just sharing params for the object specific parts
        self.object_conv = nn.Conv1d(9, DIM, 3, stride=2, padding=0, dilation=1, groups=1, bias=True)
        self.rest_conv = nn.Conv1d(4 if state_only else 8, DIM, 3, stride=2, padding=0, dilation=1, groups=1, bias=True)
        self.conv_part = nn.Sequential(
            nn.BatchNorm1d(DIM),
            nn.ReLU(),
            nn.Conv1d(DIM, DIM, 3, stride=2, padding=0, dilation=1, groups=1, bias=True),
            nn.BatchNorm1d(DIM),
            nn.ReLU(),
            nn.Conv1d(DIM, DIM, 3, stride=2, padding=0, dilation=1, groups=1, bias=True),
        )


        # V1 for subsample 8
        # self.conv_part = nn.Sequential(
        #     nn.Conv1d(26, 128, 4, stride=1, padding=0, dilation=1, groups=1, bias=True),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Conv1d(128, 128, 3, stride=1, padding=0, dilation=1, groups=1, bias=True),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Conv1d(128, 128, 3, stride=1, padding=0, dilation=1, groups=1, bias=True),
        # )


    def forward(self, all_timesteps):
        # if traj len 65, subsample if by 4 so the len will be 17
        all_timesteps = all_timesteps[:,:,::4,:]

        all_timesteps = all_timesteps.permute(0,1,3,2).contiguous()

        N_tasks, N_trajs, dim, traj_len = all_timesteps.size(0), all_timesteps.size(1), all_timesteps.size(2), all_timesteps.size(3)

        # V0
        # all_timesteps = all_timesteps.view(N_tasks*N_trajs, dim, traj_len)
        # embeddings = self.conv_part(all_timesteps)
        # embeddings = embeddings.view(N_tasks*N_trajs, -1)
        # embeddings = self.mlp_part(embeddings)
        # embeddings = embeddings.view(N_tasks, N_trajs, -1)

        # V1
        # all_timesteps = all_timesteps.view(N_tasks*N_trajs, dim, traj_len)
        # embeddings = self.conv_part(all_timesteps)
        # embeddings = embeddings.view(N_tasks, N_trajs, -1)

        # V1 with weight sharing for object-specific parts
        all_timesteps = all_timesteps.view(N_tasks*N_trajs, dim, traj_len)
        all_timesteps_obj0 = torch.cat(
            [
                all_timesteps[:,:3,:],
                all_timesteps[:,6:9,:],
                all_timesteps[:,12:15,:]
            ],
            dim=1
        )
        all_timesteps_obj1 = torch.cat(
            [
                all_timesteps[:,3:6,:],
                all_timesteps[:,9:12,:],
                all_timesteps[:,15:18,:]
            ],
            dim=1
        )
        all_timesteps_rest = all_timesteps[:,18:,:]
        obj_0_emb = self.object_conv(all_timesteps_obj0)
        obj_1_emb = self.object_conv(all_timesteps_obj1)
        rest_emb = self.rest_conv(all_timesteps_rest)

        embeddings = self.conv_part(obj_0_emb + obj_1_emb + rest_emb)
        embeddings = embeddings.view(N_tasks, N_trajs, -1)

        return embeddings


class TrivialR2ZMap(PyTorchModule):
    def __init__(
        self,
        z_dim,
        LOG_STD_SUBTRACT_VALUE=2.0
    ):
        self.save_init_params(locals())
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.BatchNorm1d(DIM),
            nn.ReLU()
        )
        self.mean_fc = nn.Linear(DIM, z_dim)
        self.log_sig_fc = nn.Linear(DIM, z_dim)

        self.LOG_STD_SUBTRACT_VALUE = LOG_STD_SUBTRACT_VALUE

        print('LOG STD SUBTRACT VALUE IS FOR APPROX POSTERIOR IS %f' % LOG_STD_SUBTRACT_VALUE)


    def forward(self, r):
        trunk_output = self.trunk(r)
        mean = self.mean_fc(trunk_output)
        log_sig = self.log_sig_fc(trunk_output) - self.LOG_STD_SUBTRACT_VALUE
        return mean, log_sig


class TrivialDiscDcEncoder(PyTorchModule):
    def __init__(
        self,
        context_encoder,
        D_c_repr_dim
    ):
        self.save_init_params(locals())
        super().__init__()

        self.context_encoder = context_encoder
        self.trunk = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.BatchNorm1d(DIM),
            nn.ReLU(),
            nn.Linear(DIM, D_c_repr_dim)
        )


    def __call__(self, context, mask=None, return_r=False):
        r = self.context_encoder(context, mask=mask)
        if return_r:
            return self.trunk(r), r
        return self.trunk(r)


class TrivialContextEncoder(PyTorchModule):
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
        '''
            For this first version of trivial encoder we are going
            to assume all tasks have the same number of trajs and
            all trajs have the same length
        '''
        # first convert the list of list of dicts to a tensor of dims
        # N_tasks x N_trajs x Len x Dim

        obs = np.array([[d['observations'] for d in task_trajs] for task_trajs in context])
        acts = np.array([[d['actions'] for d in task_trajs] for task_trajs in context])

        # if traj len 16 use this instead of the above two lines
        # obs = np.array([[d['observations'][:16,...] for d in task_trajs] for task_trajs in context])
        # acts = np.array([[d['actions'][:16,...] for d in task_trajs] for task_trajs in context])

        # if traj len 8 use this instead of the above two lines
        # obs = np.array([[d['observations'][:8,...] for d in task_trajs] for task_trajs in context])
        # acts = np.array([[d['actions'][:8,...] for d in task_trajs] for task_trajs in context])

        if not self.state_only:
            all_timesteps = np.concatenate([obs, acts], axis=-1)
        else:
            all_timesteps = obs
        # print('\n'*20)
        # print(all_timesteps)
        all_timesteps = Variable(ptu.from_numpy(all_timesteps), requires_grad=False)

        traj_embeddings = self.traj_encoder(all_timesteps)

        if mask is None:
            r = self.agg(traj_embeddings)
        else:
            r = self.agg_masked(traj_embeddings, mask)
        return r


class TrivialNPEncoder(PyTorchModule):
    def __init__(
        self,
        context_encoder,
        r_to_z_map,
        train_context_encoder=True,
        state_only=False
    ):
        self.save_init_params(locals())
        super().__init__()
        self.r_to_z_map = r_to_z_map
        self.train_context_encoder = train_context_encoder
        if train_context_encoder:
            self._context_encoder = context_encoder
        else:
            self._context_encoder = [context_encoder]
    

    @property
    def context_encoder(self):
        if self.train_context_encoder:
            return self._context_encoder
        else:
            return self._context_encoder[0]
    

    def __call__(self, context=None, mask=None, r=None):
        if r is None:
            r = self.context_encoder(context, mask=mask)
        post_mean, post_log_sig_diag = self.r_to_z_map(r)
        return ReparamMultivariateNormalDiag(post_mean, post_log_sig_diag)
