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

class TrivialR2ZMap(PyTorchModule):
    def __init__(
        self,
        r_dim,
        z_dim,
        hid_dim,
        # this makes it be closer to deterministic, makes it easier to train
        # before we turn on the KL regularization
        LOG_STD_SUBTRACT_VALUE=2.0
    ):
        self.save_init_params(locals())
        super().__init__()

        # self.trunk = nn.Sequential(
        #     nn.Linear(r_dim, hid_dim),
        #     nn.BatchNorm1d(hid_dim),
        #     nn.ReLU()
        # )
        self.trunk = nn.Sequential(
            nn.Linear(r_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU()
        )
        self.mean_fc = nn.Linear(hid_dim, z_dim)
        self.log_sig_fc = nn.Linear(hid_dim, z_dim)

        self.LOG_STD_SUBTRACT_VALUE = LOG_STD_SUBTRACT_VALUE
        print('LOG STD SUBTRACT VALUE IS FOR APPROX POSTERIOR IS %f' % LOG_STD_SUBTRACT_VALUE)


    def forward(self, r):
        trunk_output = self.trunk(r)
        mean = self.mean_fc(trunk_output)
        log_sig = self.log_sig_fc(trunk_output) - self.LOG_STD_SUBTRACT_VALUE
        return mean, log_sig


class TimestepBasedEncoder(PyTorchModule):
    def __init__(
        self,
        input_dim, #(s,a,s') or (s,s') depending on state-only
        r_dim,
        z_dim,
        enc_hid_dim,
        r2z_hid_dim,
        num_enc_layer_blocks,
        hid_act='relu',
        use_bn=True,
        within_traj_agg='sum', # 'sum' or 'mean',
        state_only=False # if state-only, we only condition on the states and not actions
    ):
        self.save_init_params(locals())
        super().__init__()

        if hid_act == 'relu':
            hid_act_class = nn.ReLU
        elif hid_act == 'tanh':
            hid_act_class = nn.Tanh
        else:
            raise NotImplementedError()

        self.r_dim, self.z_dim = r_dim, z_dim
        # build the timestep encoder
        mod_list = nn.ModuleList([nn.Linear(input_dim, enc_hid_dim)])
        if use_bn: mod_list.append(nn.BatchNorm1d(enc_hid_dim))
        mod_list.append(hid_act_class())

        for i in range(num_enc_layer_blocks - 1):
            mod_list.append(nn.Linear(enc_hid_dim, enc_hid_dim))
            if use_bn: mod_list.append(nn.BatchNorm1d(enc_hid_dim))
            mod_list.append(hid_act_class())
        
        mod_list.append(nn.Linear(enc_hid_dim, r_dim))
        self.timestep_encoder = nn.Sequential(*mod_list)

        assert within_traj_agg in ['sum', 'mean']
        self.use_sum_for_traj_agg = within_traj_agg == 'sum'
        print('\nWITHIN TRAJ AGG IS SUM: {}'.format(self.use_sum_for_traj_agg))

        # aggregator
        self.agg = sum_aggregator_unmasked
        self.agg_masked = sum_aggregator

        # build the r to z map
        self.r2z_map = TrivialR2ZMap(r_dim, z_dim, r2z_hid_dim)

        self.state_only = state_only
        print('STATE-ONLY ENCODER: {}'.format(self.state_only))


    def forward(self, context=None, mask=None, r=None):
        if r is None:
            # hack for now to make things efficient
            min_context_len = min([d['observations'].shape[0] for task_trajs in context for d in task_trajs])

            obs = np.array([[d['observations'][:min_context_len] for d in task_trajs] for task_trajs in context])
            next_obs = np.array([[d['next_observations'][:min_context_len] for d in task_trajs] for task_trajs in context])
            if not self.state_only:
                acts = np.array([[d['actions'][:min_context_len] for d in task_trajs] for task_trajs in context])
                all_timesteps = np.concatenate([obs, acts, next_obs], axis=-1)
            else:
                all_timesteps = np.concatenate([obs, next_obs], axis=-1)
            
            # FOR DEBUGGING THE ENCODER
            # all_timesteps = all_timesteps[:,:,-1:,:]

            # print(all_timesteps)
            # print(all_timesteps.shape)
            # print(all_timesteps.dtype)
            # print('----')
            # print(acts.shape)
            # print(obs.shape)
            # print(next_obs.shape)
            # if acts.shape[0] == 10:
            #     print(acts)
            #     print(obs)
            #     print(next_obs)
            # if acts.shape[0]

            all_timesteps = Variable(ptu.from_numpy(all_timesteps), requires_grad=False)

            # N_tasks x N_trajs x Len x Dim
            N_tasks, N_trajs, Len, Dim = all_timesteps.size(0), all_timesteps.size(1), all_timesteps.size(2), all_timesteps.size(3)
            all_timesteps = all_timesteps.view(-1, Dim)
            embeddings = self.timestep_encoder(all_timesteps)
            embeddings = embeddings.view(N_tasks, N_trajs, Len, self.r_dim)

            if self.use_sum_for_traj_agg:
                traj_embeddings = torch.sum(embeddings, dim=2)
            else:
                traj_embeddings = torch.mean(embeddings, dim=2)

            # get r
            if mask is None:
                r = self.agg(traj_embeddings)
            else:
                r = self.agg_masked(traj_embeddings, mask)
        post_mean, post_log_sig_diag = self.r2z_map(r)
        return ReparamMultivariateNormalDiag(post_mean, post_log_sig_diag)


class WeightShareTimestepBasedEncoder(PyTorchModule):
    def __init__(
        self,
        obs_dim,
        act_dim,
        pre_enc_dim,
        r_dim,
        z_dim,
        enc_hid_dim,
        r2z_hid_dim,
        num_enc_layer_blocks,
        hid_act='relu',
        use_bn=True,
        within_traj_agg='sum' # 'sum' or 'mean'
    ):
        self.save_init_params(locals())
        super().__init__()

        if hid_act == 'relu':
            hid_act_class = nn.ReLU
        elif hid_act == 'tanh':
            hid_act_class = nn.Tanh
        else:
            raise NotImplementedError()

        self.obs_enc = nn.Sequential(
            nn.Linear(obs_dim, pre_enc_dim),
            nn.BatchNorm1d(pre_enc_dim),
            hid_act_class()
        )
        self.act_enc = nn.Sequential(
            nn.Linear(act_dim, pre_enc_dim),
            nn.BatchNorm1d(pre_enc_dim),
            hid_act_class()
        )

        self.r_dim, self.z_dim = r_dim, z_dim
        # build the timestep encoder
        mod_list = nn.ModuleList([nn.Linear(3 * pre_enc_dim, enc_hid_dim)])
        if use_bn: mod_list.append(nn.BatchNorm1d(enc_hid_dim))
        mod_list.append(hid_act_class())

        for i in range(num_enc_layer_blocks - 1):
            mod_list.append(nn.Linear(enc_hid_dim, enc_hid_dim))
            if use_bn: mod_list.append(nn.BatchNorm1d(enc_hid_dim))
            mod_list.append(hid_act_class())
        
        mod_list.append(nn.Linear(enc_hid_dim, r_dim))
        self.timestep_encoder = nn.Sequential(*mod_list)

        assert within_traj_agg in ['sum', 'mean']
        self.use_sum_for_traj_agg = within_traj_agg == 'sum'
        print('\nWITHIN TRAJ AGG IS SUM: {}'.format(self.use_sum_for_traj_agg))

        # aggregator
        self.agg = sum_aggregator_unmasked
        self.agg_masked = sum_aggregator

        # build the r to z map
        self.r2z_map = TrivialR2ZMap(r_dim, z_dim, r2z_hid_dim)


    def forward(self, context=None, mask=None, r=None):
        if r is None:
            obs = np.array([[d['observations'] for d in task_trajs] for task_trajs in context])
            acts = np.array([[d['actions'] for d in task_trajs] for task_trajs in context])
            next_obs = np.array([[d['next_observations'] for d in task_trajs] for task_trajs in context])

            obs = Variable(ptu.from_numpy(obs), requires_grad=False)
            acts = Variable(ptu.from_numpy(acts), requires_grad=False)
            next_obs = Variable(ptu.from_numpy(next_obs), requires_grad=False)

            # N_tasks x N_trajs x Len x Dim
            N_tasks, N_trajs, Len = obs.size(0), obs.size(1), obs.size(2)
            obs = obs.view(N_tasks*N_trajs*Len, -1)
            acts = acts.view(N_tasks*N_trajs*Len, -1)
            next_obs = next_obs.view(N_tasks*N_trajs*Len, -1)

            obs_embed = self.obs_enc(torch.cat([obs, next_obs], dim=0))
            acts_embed = self.act_enc(acts)
            all_timesteps = torch.cat([obs_embed[:N_tasks*N_trajs*Len], acts_embed, obs_embed[N_tasks*N_trajs*Len:]], dim=-1)

            embeddings = self.timestep_encoder(all_timesteps).view(N_tasks, N_trajs, Len, self.r_dim)

            if self.use_sum_for_traj_agg:
                traj_embeddings = torch.sum(embeddings, dim=2)
            else:
                traj_embeddings = torch.mean(embeddings, dim=2)

            # get r
            if mask is None:
                r = self.agg(traj_embeddings)
            else:
                r = self.agg_masked(traj_embeddings, mask)
        post_mean, post_log_sig_diag = self.r2z_map(r)
        return ReparamMultivariateNormalDiag(post_mean, post_log_sig_diag)
