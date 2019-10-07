from collections import OrderedDict

import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.vistools import plot_histogram
from rlkit.torch.torch_meta_irl_algorithm import TorchMetaIRLAlgorithm
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.core.train_util import linear_schedule

from rlkit.torch.sac.policies import PostCondMLPPolicyWrapper
from rlkit.data_management.path_builder import PathBuilder
from gym.spaces import Dict


def concat_trajs(trajs):
    new_dict = {}
    for k in trajs[0].keys():
        if isinstance(trajs[0][k], dict):
            new_dict[k] = concat_trajs([t[k] for t in trajs])
        else:
            new_dict[k] = np.concatenate([t[k] for t in trajs], axis=0)
    return new_dict


def subsample_traj(traj, num_samples):
    traj_len = traj['observations'].shape[0]
    idxs = np.random.choice(traj_len, size=num_samples, replace=traj_len<num_samples)
    new_traj = {k: traj[k][idxs,...] for k in traj}
    return new_traj


class NeuralProcessMetaIRL(TorchMetaIRLAlgorithm):
    def __init__(
            self,
            env,
            policy,
            discriminator,
            disc_encoder,
            # the disc context encoder will use the same optimization settings
            # as the discriminator, won't have GP, won't have Gradient Clipping for now

            train_context_expert_replay_buffer,
            train_test_expert_replay_buffer,
            test_context_expert_replay_buffer,
            test_test_expert_replay_buffer,

            q_model,

            policy_optimizer,

            state_only=False,

            num_tasks_used_per_update=5,
            num_context_trajs_for_training=3,
            num_test_trajs_for_training=5,
            disc_samples_per_traj=8,

            few_shot_version=False,
            min_context_size=1,
            max_context_size=5,

            num_context_trajs_for_exploration=3,
            
            num_tasks_per_eval=10,
            num_diff_context_per_eval_task=2,
            num_eval_trajs_per_post_sample=2,
            num_context_trajs_for_eval=3,

            # the batches for the policy optimization could be either
            # sampled randomly from the appropriate tasks or we could
            # employ a similar structure as the discriminator batching
            policy_optim_batch_mode_random=True,
            policy_optim_batch_size_per_task=1024,
            policy_optim_batch_size_per_task_from_expert=0,
            
            q_model_lr=1e-3,
            q_model_optimizer_class=optim.Adam,

            disc_lr=1e-3,
            disc_optimizer_class=optim.Adam,
            disc_Adam_beta=0.0,

            num_update_loops_per_train_call=65,
            num_disc_updates_per_loop_iter=1,
            num_policy_updates_per_loop_iter=1,

            use_grad_pen=True,
            grad_pen_weight=10,

            disc_ce_grad_clip=3.0,
            enc_ce_grad_clip=1.0,
            disc_gp_grad_clip=1.0,

            use_target_disc=False,
            target_disc=None,
            soft_target_disc_tau=0.005,

            use_target_disc_enc=False,
            target_disc_enc=None,
            soft_target_enc_tau=0.005,

            max_KL_beta = 1.0,
            KL_ramp_up_start_iter=0,
            KL_ramp_up_end_iter=100,

            plotter=None,
            render_eval_paths=False,
            eval_deterministic=False,

            only_Dc=False,
            use_rev_KL=False,
            q_uses_disc_r_getter=False,
            disc_ignores_z=False,

            **kwargs
    ):
        assert disc_lr != 1e-3, 'Just checking that this is being taken from the spec file'
        assert not (use_target_disc or use_target_disc_enc), 'Not Implemented the soft updates for them and other things probably too'
        if kwargs['policy_uses_pixels']: raise NotImplementedError('policy uses pixels')
        if kwargs['wrap_absorbing']: raise NotImplementedError('wrap absorbing')
        assert not eval_deterministic
        
        super().__init__(
            env=env,
            train_context_expert_replay_buffer=train_context_expert_replay_buffer,
            train_test_expert_replay_buffer=train_test_expert_replay_buffer,
            test_context_expert_replay_buffer=test_context_expert_replay_buffer,
            test_test_expert_replay_buffer=test_test_expert_replay_buffer,
            **kwargs
        )

        self.main_policy = policy
        self.q_model = q_model
        self.discriminator = discriminator
        self.disc_encoder = disc_encoder
        self.eval_statistics = None

        self.policy_optimizer = policy_optimizer
        self.disc_optimizer = disc_optimizer_class(
            self.discriminator.parameters(),
            lr=disc_lr,
            betas=(disc_Adam_beta, 0.999)
        )
        self.disc_enc_optimizer = disc_optimizer_class(
            self.disc_encoder.parameters(),
            lr=disc_lr,
            # betas=(disc_Adam_beta, 0.999)
            betas=(0.9, 0.999)
        )
        self.disc_Adam_beta = disc_Adam_beta
        print('\n\nDISC ADAM BETA IS %f\n\n' % disc_Adam_beta)
        print('\n\nDISC ENCODER ADAM BETA IS %f\n\n' % 0.9)
        print(self.disc_enc_optimizer)
        
        print('\n\nQ ENCODER ADAM IS 0.25\n\n')
        assert q_model_lr != 1e-3, 'just to make sure this is being set'
        self.q_optimizer = q_model_optimizer_class(
            self.q_model.parameters(),
            lr=q_model_lr,
            # betas=(0.9, 0.999)
            betas=(0.25, 0.999)
            # betas=(0.0, 0.999)
        )

        self.state_only = state_only

        self.num_tasks_used_per_update = num_tasks_used_per_update
        self.num_context_trajs_for_training = num_context_trajs_for_training
        self.num_test_trajs_for_training = num_test_trajs_for_training
        self.disc_samples_per_traj = disc_samples_per_traj

        self.num_context_trajs_for_exploration = num_context_trajs_for_exploration
        
        self.num_tasks_per_eval = num_tasks_per_eval
        self.num_diff_context_per_eval_task = num_diff_context_per_eval_task
        self.num_eval_trajs_per_post_sample = num_eval_trajs_per_post_sample
        self.num_context_trajs_for_eval = num_context_trajs_for_eval

        self.policy_optim_batch_mode_random = policy_optim_batch_mode_random
        self.policy_optim_batch_size_per_task = policy_optim_batch_size_per_task
        self.policy_optim_batch_size_per_task_from_expert = policy_optim_batch_size_per_task_from_expert

        self.bce = nn.BCEWithLogitsLoss()
        target_batch_size = self.num_tasks_used_per_update*(self.num_context_trajs_for_training + self.num_test_trajs_for_training)*self.disc_samples_per_traj
        self.bce_targets = torch.cat(
            [
                torch.ones(target_batch_size, 1),
                torch.zeros(target_batch_size, 1)
            ],
            dim=0
        )
        self.bce_targets = Variable(self.bce_targets)
        if ptu.gpu_enabled():
            self.bce.cuda()
            self.bce_targets = self.bce_targets.cuda()
        
        self.use_grad_pen = use_grad_pen
        self.grad_pen_weight = grad_pen_weight

        self.disc_ce_grad_clip = disc_ce_grad_clip
        self.enc_ce_grad_clip = enc_ce_grad_clip
        self.disc_gp_grad_clip = disc_gp_grad_clip
        self.disc_grad_buffer = {}
        self.disc_grad_buffer_is_empty = True
        self.enc_grad_buffer = {}
        self.enc_grad_buffer_is_empty = True

        self.use_target_disc = use_target_disc
        self.soft_target_disc_tau = soft_target_disc_tau

        if use_target_disc:
            if target_disc is None:
                print('\n\nMAKING TARGET DISC\n\n')
                self.target_disc = self.discriminator.copy()
            else:
                print('\n\nUSING GIVEN TARGET DISC\n\n')
                self.target_disc = target_disc
        
        self.use_target_disc_enc = use_target_disc_enc
        self.soft_target_enc_tau = soft_target_enc_tau

        if use_target_disc_enc:
            if target_disc_enc is None:
                print('\n\nMAKING TARGET ENC\n\n')
                self.target_disc_enc = self.disc_encoder.copy()
            else:
                print('\n\nUSING GIVEN TARGET ENC\n\n')
                self.target_disc_enc = target_disc_enc
        
        self.disc_ce_grad_norm = 0.0
        self.disc_ce_grad_norm_counter = 0.0
        self.max_disc_ce_grad = 0.0

        self.enc_ce_grad_norm = 0.0
        self.enc_ce_grad_norm_counter = 0.0
        self.max_enc_ce_grad = 0.0

        self.disc_gp_grad_norm = 0.0
        self.disc_gp_grad_norm_counter = 0.0
        self.max_disc_gp_grad = 0.0

        self.num_update_loops_per_train_call = num_update_loops_per_train_call
        self.num_disc_updates_per_loop_iter = num_disc_updates_per_loop_iter
        self.num_policy_updates_per_loop_iter = num_policy_updates_per_loop_iter

        self.max_KL_beta = max_KL_beta
        self.KL_ramp_up_start_iter = KL_ramp_up_start_iter
        self.KL_ramp_up_end_iter = KL_ramp_up_end_iter

        self.few_shot_version = few_shot_version
        self.max_context_size = max_context_size
        self.min_context_size = min_context_size
        assert num_context_trajs_for_training == max_context_size

        self.only_Dc = only_Dc
        self.use_rev_KL = use_rev_KL
        self.q_uses_disc_r_getter = q_uses_disc_r_getter
        self.disc_ignores_z = disc_ignores_z
    

    def get_exploration_policy(self, task_identifier):
        if self.wrap_absorbing: raise NotImplementedError('wrap absorbing')
        if self.few_shot_version:
            # no need for this if/else statement like this, make it cleaner
            this_context_size = np.random.randint(self.min_context_size, self.max_context_size+1)
            list_of_trajs = self.train_context_expert_replay_buffer.sample_trajs_from_task(
                task_identifier,
                this_context_size
            )
            mask = None
        else:
            list_of_trajs = self.train_context_expert_replay_buffer.sample_trajs_from_task(
                task_identifier,
                self.num_context_trajs_for_exploration,
            )
            mask = None

        # need the encoder for the policy, aka the q distribution
        if not self.only_Dc:
            enc_to_use = self.q_model
            mode = enc_to_use.training
            enc_to_use.eval()
            enc_to_use.context_encoder.eval()
            post_dist = enc_to_use([list_of_trajs], mask)
            enc_to_use.train(mode)
            enc_to_use.context_encoder.train(mode)

            z = post_dist.sample()
            z = z.cpu().data.numpy()[0]
        else:
            mode = self.disc_encoder.training
            self.disc_encoder.eval()
            D_c_repr = self.disc_encoder([list_of_trajs], mask)
            self.disc_encoder.train(mode)

            z = D_c_repr.cpu().data.numpy()[0]

        self.main_policy.eval()
        post_cond_policy = PostCondMLPPolicyWrapper(self.main_policy, z)
        return post_cond_policy
    

    def get_eval_policy(self, task_identifier, mode='meta_test'):
        if self.wrap_absorbing: raise NotImplementedError('wrap absorbing')
        if mode == 'meta_train':
            rb = self.train_context_expert_replay_buffer
        else:
            rb = self.test_context_expert_replay_buffer
        
        list_of_trajs = rb.sample_trajs_from_task(
            task_identifier,
            np.random.randint(self.min_context_size, self.max_context_size+1) \
                if self.few_shot_version else self.num_context_trajs_for_eval,
        )
        
        # need the encoder for the policy, aka the q distribution
        if not self.only_Dc:
            enc_to_use = self.q_model
            mode = enc_to_use.training
            enc_to_use.eval()
            enc_to_use.context_encoder.eval()
            post_dist = enc_to_use([list_of_trajs])
            enc_to_use.train(mode)
            enc_to_use.context_encoder.train(mode)

            z = post_dist.sample()
            # z = post_dist.mean
            z = z.cpu().data.numpy()[0]
        else:
            mode = self.disc_encoder.training
            self.disc_encoder.eval()
            D_c_repr = self.disc_encoder([list_of_trajs])
            self.disc_encoder.train(mode)

            z = D_c_repr.cpu().data.numpy()[0]

        self.main_policy.eval()
        post_cond_policy = PostCondMLPPolicyWrapper(self.main_policy, z)
        return post_cond_policy
    

    def _get_disc_training_batch(self):
        keys_to_get = ['observations', 'actions']
        if self.few_shot_version:
            context_batch, task_identifiers_list = self.train_context_expert_replay_buffer.sample_trajs(
                self.max_context_size,
                num_tasks=self.num_tasks_used_per_update,
                keys=keys_to_get
            )
            mask = ptu.Variable(torch.ones(self.num_tasks_used_per_update, self.max_context_size, 1))
            this_context_sizes = np.random.randint(self.min_context_size, self.max_context_size+1, size=self.num_tasks_used_per_update)
            for i, c_size in enumerate(this_context_sizes):
                mask[i,:c_size,:] = 1.0
        else:
            context_batch, task_identifiers_list = self.train_context_expert_replay_buffer.sample_trajs(
                self.num_context_trajs_for_training,
                num_tasks=self.num_tasks_used_per_update,
                keys=keys_to_get
            )
            mask = None

        # get the pred version of the context batch
        # subsample the trajs
        flat_context_batch = [subsample_traj(traj, self.disc_samples_per_traj) for task_trajs in context_batch for traj in task_trajs]
        context_pred_batch = concat_trajs(flat_context_batch)

        test_batch, _ = self.train_test_expert_replay_buffer.sample_trajs(
            self.num_test_trajs_for_training,
            task_identifiers=task_identifiers_list,
            keys=keys_to_get,
            samples_per_traj=self.disc_samples_per_traj
        )
        flat_test_batch = [traj for task_trajs in test_batch for traj in task_trajs]
        test_pred_batch = concat_trajs(flat_test_batch)

        # get the test batch for the tasks from policy buffer
        policy_test_batch_0, _ = self.replay_buffer.sample_trajs(
            self.num_context_trajs_for_training,
            task_identifiers=task_identifiers_list,
            keys=keys_to_get,
            samples_per_traj=self.disc_samples_per_traj
        )
        flat_policy_batch_0 = [traj for task_trajs in policy_test_batch_0 for traj in task_trajs]
        policy_test_pred_batch_0 = concat_trajs(flat_policy_batch_0)

        policy_test_batch_1, _ = self.replay_buffer.sample_trajs(
            self.num_test_trajs_for_training,
            task_identifiers=task_identifiers_list,
            keys=keys_to_get,
            samples_per_traj=self.disc_samples_per_traj
        )
        flat_policy_batch_1 = [traj for task_trajs in policy_test_batch_1 for traj in task_trajs]
        policy_test_pred_batch_1 = concat_trajs(flat_policy_batch_1)

        policy_test_pred_batch = {
            'observations': np.concatenate((policy_test_pred_batch_0['observations'], policy_test_pred_batch_1['observations']), axis=0),
            'actions': np.concatenate((policy_test_pred_batch_0['actions'], policy_test_pred_batch_1['actions']), axis=0)
        }

        return context_batch, context_pred_batch, test_pred_batch, policy_test_pred_batch, mask
    

    def _get_policy_training_batch(self):
        # context batch is a list of list of dicts
        if self.few_shot_version:
            context_batch, task_identifiers_list = self.train_context_expert_replay_buffer.sample_trajs(
                self.max_context_size,
                num_tasks=self.num_tasks_used_per_update,
                keys=['observations', 'actions']
            )
            mask = ptu.Variable(torch.ones(self.num_tasks_used_per_update, self.max_context_size, 1))
            this_context_sizes = np.random.randint(self.min_context_size, self.max_context_size+1, size=self.num_tasks_used_per_update)
            for i, c_size in enumerate(this_context_sizes):
                mask[i,:c_size,:] = 1.0
        else:
            context_batch, task_identifiers_list = self.train_context_expert_replay_buffer.sample_trajs(
                self.num_context_trajs_for_training,
                num_tasks=self.num_tasks_used_per_update,
                keys=['observations', 'actions']
            )
            mask = None

        if self.policy_optim_batch_mode_random:
            # get the test batch for the tasks from policy buffer
            policy_batch_from_policy, _ = self.replay_buffer.sample_random_batch(
                self.policy_optim_batch_size_per_task - self.policy_optim_batch_size_per_task_from_expert,
                task_identifiers_list=task_identifiers_list
            )
            if self.policy_optim_batch_size_per_task_from_expert > 0:
                policy_batch_from_expert, _ = self.train_test_expert_replay_buffer.sample_random_batch(
                    self.policy_optim_batch_size_per_task_from_expert,
                    task_identifiers_list=task_identifiers_list
                )
                policy_batch = []
                for task_num in range(len(policy_batch_from_policy)):
                    policy_batch.append(policy_batch_from_policy[task_num])
                    policy_batch.append(policy_batch_from_expert[task_num])
            else:
                policy_batch = policy_batch_from_policy
            policy_obs = np.concatenate([d['observations'] for d in policy_batch], axis=0) # (N_tasks * batch_size) x Dim
            policy_acts = np.concatenate([d['actions'] for d in policy_batch], axis=0) # (N_tasks * batch_size) x Dim
            policy_terminals = np.concatenate([d['terminals'] for d in policy_batch], axis=0) # (N_tasks * batch_size) x Dim
            policy_next_obs = np.concatenate([d['next_observations'] for d in policy_batch], axis=0) # (N_tasks * batch_size) x Dim
            # policy_absorbing = np.concatenate([d['absorbing'] for d in policy_batch], axis=0) # (N_tasks * batch_size) x Dim
            policy_batch = dict(
                observations=policy_obs,
                actions=policy_acts,
                terminals=policy_terminals,
                next_observations=policy_next_obs,
                # absorbing=absorbing
            )
        else:
            raise NotImplementedError()

        return context_batch, policy_batch, mask


    def _compute_expected_KL(self, post_dist):
        # computting the KL of a Gaussian posterior from the spherical prior
        z_means = post_dist.mean
        z_log_covs = post_dist.log_cov
        z_covs = post_dist.cov
        KL = torch.sum(
            1.0 + z_log_covs - z_means**2 - z_covs,
            dim=-1
        )
        KL =  -0.5 * torch.mean(KL)
        return KL
    

    def _tapered_exp(self, x):
        # linear tapering
        # C = 10.0
        # log_C = np.log(C)
        # taper_slope = 1.5 # this makes the function be 20 at x=9
        # exp = torch.exp(x)
        # taper = taper_slope*(x - log_C)
        # return torch.clamp(exp, max=C) + torch.clamp(taper, min=0.0)

        # tanh tapering
        C = 5.0
        log_C = np.log(C)
        taper_slope = 5.0
        exp = torch.exp(x)
        taper = taper_slope*torch.tanh(x - log_C)
        return torch.clamp(exp, max=C) + torch.clamp(taper, min=0.0)


    def _do_training(self, epoch):
        for t in range(self.num_update_loops_per_train_call):
            for t1 in range(self.num_disc_updates_per_loop_iter):
                self._do_reward_training(epoch, t, t1)
            for t2 in range(self.num_policy_updates_per_loop_iter):
                self._do_policy_training(epoch, t, t2)


    def _do_reward_training(self, epoch, update_loop_iter, disc_update_iter):
        '''
            Train the discriminator
        '''
        self.disc_encoder.train()
        self.discriminator.train()
        self.main_policy.train()
        self.q_model.train()

        self.disc_enc_optimizer.zero_grad()
        self.disc_optimizer.zero_grad()

        # prep the batches
        context_batch, context_pred_batch, test_pred_batch, policy_test_pred_batch, mask = self._get_disc_training_batch()

        # convert it to a pytorch tensor
        # note that our objective says we should maximize likelihood of
        # BOTH the context_batch and the test_batch
        exp_obs_batch = np.concatenate((context_pred_batch['observations'], test_pred_batch['observations']), axis=0)
        exp_obs_batch = Variable(ptu.from_numpy(exp_obs_batch), requires_grad=False)
        policy_obs_batch = Variable(ptu.from_numpy(policy_test_pred_batch['observations']), requires_grad=False)

        if not self.state_only:
            exp_acts_batch = np.concatenate((context_pred_batch['actions'], test_pred_batch['actions']), axis=0)
            exp_acts_batch = Variable(ptu.from_numpy(exp_acts_batch), requires_grad=False)
            policy_acts_batch = Variable(ptu.from_numpy(policy_test_pred_batch['actions']), requires_grad=False)

        # get the disc representation for the context batch
        D_c_repr, intermediate_r = self.disc_encoder(context_batch, mask, return_r=True)

        # get z, remember to detach it
        # I don't need to put q in eval model cause the distribution over inputs
        # is not any different than it's training scenario
        # mode = self.q_model.training
        # self.q_model.eval()
        if not self.only_Dc:
            post_dist = self.q_model(context_batch, mask)
            # post_dist = self.q_model(context_batch, mask, r=intermediate_r.detach())
            z = post_dist.sample() # N_tasks x Dim
            z = z.detach()
            # self.q_model.train(mode)

        # repeat and reshape to get the D_c_repr batch
        context_pred_D_c_repr = D_c_repr.repeat(1, self.num_context_trajs_for_training * self.disc_samples_per_traj).view(
            -1,
            D_c_repr.size(1)
        )
        test_pred_D_c_repr = D_c_repr.repeat(1, self.num_test_trajs_for_training * self.disc_samples_per_traj).view(
            -1,
            D_c_repr.size(1)
        )
        D_c_repr_batch = torch.cat([context_pred_D_c_repr, test_pred_D_c_repr], dim=0)
        repeated_D_c_repr_batch = D_c_repr_batch.repeat(2, 1)

        # repeat and reshape to get the z batch
        if not self.only_Dc:
            context_pred_z = z.repeat(1, self.num_context_trajs_for_training * self.disc_samples_per_traj).view(
                -1,
                z.size(1)
            )
            test_pred_z = z.repeat(1, self.num_test_trajs_for_training * self.disc_samples_per_traj).view(
                -1,
                z.size(1)
            )
            z_batch = torch.cat([context_pred_z, test_pred_z], dim=0)
            repeated_z_batch = z_batch.repeat(2, 1)

        # compute the loss for the discriminator
        obs_batch = torch.cat([exp_obs_batch, policy_obs_batch], dim=0)
        if self.state_only:
            acts_batch = None
        else:
            acts_batch = torch.cat([exp_acts_batch, policy_acts_batch], dim=0)

        # print(repeated_D_c_repr_batch.size())
        # print(repeated_z_batch.size())
        # print(obs_batch.size())
        # print(acts_batch.size())

        self.discriminator.eval()
        if (not self.only_Dc) and (not self.disc_ignores_z):
            eval_mode_T_outputs = self.discriminator(obs_batch, acts_batch, repeated_D_c_repr_batch, repeated_z_batch).detach()
            eval_mode_T_exp_outputs = eval_mode_T_outputs[:exp_obs_batch.size(0)]
            eval_mode_T_pol_outputs = eval_mode_T_outputs[exp_obs_batch.size(0):]

            only_exp_eval_mode_T_outputs = self.discriminator(exp_obs_batch, exp_acts_batch, D_c_repr_batch, z_batch).detach()
            only_pol_eval_mode_T_outputs = self.discriminator(policy_obs_batch, policy_acts_batch, D_c_repr_batch, z_batch).detach()
            self.discriminator.train()

            T_outputs = self.discriminator(obs_batch, acts_batch, repeated_D_c_repr_batch, repeated_z_batch)
        else:
            eval_mode_T_outputs = self.discriminator(obs_batch, acts_batch, repeated_D_c_repr_batch).detach()
            eval_mode_T_exp_outputs = eval_mode_T_outputs[:exp_obs_batch.size(0)]
            eval_mode_T_pol_outputs = eval_mode_T_outputs[exp_obs_batch.size(0):]

            only_exp_eval_mode_T_outputs = self.discriminator(exp_obs_batch, exp_acts_batch, D_c_repr_batch).detach()
            only_pol_eval_mode_T_outputs = self.discriminator(policy_obs_batch, policy_acts_batch, D_c_repr_batch).detach()
            self.discriminator.train()

            T_outputs = self.discriminator(obs_batch, acts_batch, repeated_D_c_repr_batch)

        T_preds = (T_outputs > 1.0).type(T_outputs.data.type())
        accuracy = (T_preds == self.bce_targets).type(torch.FloatTensor).mean()
        
        # compute the loss for the "discriminator"
        T_exp_outputs = T_outputs[:exp_obs_batch.size(0)]
        T_pol_outputs = T_outputs[exp_obs_batch.size(0):]

        if not self.use_rev_KL:
            # lower_bound = T_exp_outputs - torch.clamp(torch.exp(T_pol_outputs - 1.0), min=0.0, max=5.0)
            # lower_bound = T_exp_outputs - self._tapered_exp(T_pol_outputs - 1.0)
            lower_bound = T_exp_outputs - torch.exp(T_pol_outputs - 1.0)

            # lower_bound = lower_bound.sum() / ((self.num_context_trajs_for_training + self.num_test_trajs_for_training) * self.disc_samples_per_traj)
            lower_bound = lower_bound.mean()
        else:
            lower_bound = self.bce(T_outputs, self.bce_targets)
        
        if self.use_grad_pen:
            eps = Variable(torch.rand(exp_obs_batch.size(0), 1))
            if ptu.gpu_enabled(): eps = eps.cuda()
            
            interp_obs = eps*exp_obs_batch + (1-eps)*policy_obs_batch
            interp_obs = interp_obs.detach()
            interp_obs.requires_grad = True
            if self.state_only:
                if (not self.only_Dc) and (not self.disc_ignores_z):
                    gradients = autograd.grad(
                        outputs=self.discriminator(interp_obs, None, D_c_repr_batch.detach(), z_batch.detach()).sum(),
                        inputs=[interp_obs],
                        # grad_outputs=torch.ones(exp_specs['batch_size'], 1).cuda(),
                        create_graph=True, retain_graph=True, only_inputs=True
                    )
                else:
                    gradients = autograd.grad(
                        outputs=self.discriminator(interp_obs, None, D_c_repr_batch.detach()).sum(),
                        inputs=[interp_obs],
                        # grad_outputs=torch.ones(exp_specs['batch_size'], 1).cuda(),
                        create_graph=True, retain_graph=True, only_inputs=True
                    )
                total_grad = gradients[0]
            else:
                interp_actions = eps*exp_acts_batch + (1-eps)*policy_acts_batch
                interp_actions = interp_actions.detach()
                interp_actions.requires_grad = True
                if (not self.only_Dc) and (not self.disc_ignores_z):
                    gradients = autograd.grad(
                        outputs=self.discriminator(interp_obs, interp_actions, D_c_repr_batch.detach(), z_batch.detach()).sum(),
                        inputs=[interp_obs, interp_actions],
                        # grad_outputs=torch.ones(exp_specs['batch_size'], 1).cuda(),
                        create_graph=True, retain_graph=True, only_inputs=True
                    )
                else:
                    gradients = autograd.grad(
                        outputs=self.discriminator(interp_obs, interp_actions, D_c_repr_batch.detach()).sum(),
                        inputs=[interp_obs, interp_actions],
                        # grad_outputs=torch.ones(exp_specs['batch_size'], 1).cuda(),
                        create_graph=True, retain_graph=True, only_inputs=True
                    )
                total_grad = torch.cat([gradients[0], gradients[1]], dim=1)


            # GP from Gulrajani et al.
            gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()
            disc_grad_pen_loss = gradient_penalty * self.grad_pen_weight
        
        if not self.use_rev_KL:
            neg_lower_bound = -1.0 * lower_bound
            loss = neg_lower_bound + disc_grad_pen_loss
            loss.backward()
        else:
            # I didn't wanna change the name of the variable cause I'm tired
            neg_lower_bound = lower_bound
            loss = neg_lower_bound + disc_grad_pen_loss
            loss.backward()

        self.disc_optimizer.step()
        self.disc_enc_optimizer.step()

        """
        Save some statistics for eval
        """
        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
            
            if self.use_target_disc:
                if self.state_only:
                    if (not self.only_Dc) and (not self.disc_ignores_z):
                        target_T_outputs = self.target_disc(obs_batch, None, repeated_D_c_repr_batch, repeated_z_batch)
                    else:
                        target_T_outputs = self.target_disc(obs_batch, None, repeated_D_c_repr_batch)
                else:
                    if (not self.only_Dc) and (not self.disc_ignores_z):
                        target_T_outputs = self.target_disc(obs_batch, acts_batch, repeated_D_c_repr_batch, repeated_z_batch)
                    else:
                        target_T_outputs = self.target_disc(obs_batch, acts_batch, repeated_D_c_repr_batch)
                target_T_preds = (target_disc_logits > 1.0).type(target_disc_outputs.data.type())
                target_accuracy = (target_T_preds == self.bce_targets).type(torch.FloatTensor).mean()
                target_T_exp_outputs = target_T_outputs[:exp_obs_batch.size(0)]
                target_T_pol_outputs = target_T_outputs[exp_obs_batch.size(0):]
                # target_lower_bound = target_T_exp_outputs - torch.clamp(torch.exp(target_T_pol_outputs - 1.0), min=0.0, max=5.0)
                # target_lower_bound = target_T_exp_outputs - self._tapered_exp(target_T_pol_outputs - 1.0)
                target_lower_bound = target_T_exp_outputs - torch.exp(target_T_pol_outputs - 1.0)

                # target_lower_bound = target_lower_bound.sum() / ((self.num_context_trajs_for_training + self.num_test_trajs_for_training) * self.disc_samples_per_traj)
                target_lower_bound = target_lower_bound.mean()
                target_loss = -1.0 * target_lower_bound

                if self.use_grad_pen:
                    eps = Variable(torch.rand(exp_obs_batch.size(0), 1))
                    if ptu.gpu_enabled(): eps = eps.cuda()
                    
                    interp_obs = eps*exp_obs_batch + (1-eps)*policy_obs_batch
                    interp_obs = interp_obs.detach()
                    interp_obs.requires_grad = True
                    if self.state_only:
                        if self.only_Dc:
                            target_gradients = autograd.grad(
                                outputs=self.target_disc(interp_obs, None, D_c_repr_batch, z_batch).sum(),
                                inputs=[interp_obs],
                                # grad_outputs=torch.ones(exp_specs['batch_size'], 1).cuda(),
                                create_graph=True, retain_graph=True, only_inputs=True
                            )
                        else:
                            target_gradients = autograd.grad(
                                outputs=self.target_disc(interp_obs, None, D_c_repr_batch).sum(),
                                inputs=[interp_obs],
                                # grad_outputs=torch.ones(exp_specs['batch_size'], 1).cuda(),
                                create_graph=True, retain_graph=True, only_inputs=True
                            )
                        total_target_grad = target_gradients[0]
                    else:
                        interp_actions = eps*exp_acts_batch + (1-eps)*policy_acts_batch
                        interp_actios = interp_actions.detach()
                        interp_actions.requires_grad = True
                        if self.only_Dc:
                            target_gradients = autograd.grad(
                                outputs=self.target_disc(interp_obs, interp_actions, D_c_repr_batch, z_batch).sum(),
                                inputs=[interp_obs, interp_actions],
                                # grad_outputs=torch.ones(exp_specs['batch_size'], 1).cuda(),
                                create_graph=True, retain_graph=True, only_inputs=True
                            )
                        else:
                            target_gradients = autograd.grad(
                                outputs=self.target_disc(interp_obs, interp_actions, D_c_repr_batch).sum(),
                                inputs=[interp_obs, interp_actions],
                                # grad_outputs=torch.ones(exp_specs['batch_size'], 1).cuda(),
                                create_graph=True, retain_graph=True, only_inputs=True
                            )
                        total_target_grad = torch.cat([target_gradients[0], target_gradients[1]], dim=1)

                    # GP from Gulrajani et al.
                    target_gradient_penalty = ((total_target_grad.norm(2, dim=1) - 1) ** 2).mean()

                self.eval_statistics['Target Loss'] = np.mean(ptu.get_numpy(target_loss))
                self.eval_statistics['Target Acc'] = np.mean(ptu.get_numpy(target_accuracy))
                self.eval_statistics['Target Grad Pen'] = np.mean(ptu.get_numpy(target_gradient_penalty))
                self.eval_statistics['Target Grad Pen W'] = np.mean(self.grad_pen_weight)
            
            self.eval_statistics['Disc Loss'] = np.mean(ptu.get_numpy(neg_lower_bound))
            self.eval_statistics['Disc Acc'] = np.mean(ptu.get_numpy(accuracy))
            self.eval_statistics['Grad Pen'] = np.mean(ptu.get_numpy(gradient_penalty))
            self.eval_statistics['Grad Pen W'] = np.mean(self.grad_pen_weight)

            if not self.only_Dc:
                self.eval_statistics['Avg Post Mean Abs'] = np.mean(np.abs(ptu.get_numpy(post_dist.mean)))
                self.eval_statistics['Avg Post Cov Abs'] = np.mean(np.abs(ptu.get_numpy(post_dist.cov)))

            if not self.use_rev_KL:
                exp_rews = ptu.get_numpy(torch.exp(T_exp_outputs - 1.0))
                # exp_rews = ptu.get_numpy(self._tapered_exp(T_exp_outputs - 1.0))
                # exp_rews = ptu.get_numpy(T_exp_outputs - 1.0)
            else:
                # exp_rews = ptu.get_numpy(T_exp_outputs)
                exp_rews = ptu.get_numpy(T_exp_outputs * torch.exp(T_exp_outputs))
            self.eval_statistics['Avg Rew for Exp'] = np.mean(exp_rews)
            self.eval_statistics['Std Rew for Exp'] = np.std(exp_rews)
            self.eval_statistics['Max Rew for Exp'] = np.max(exp_rews)
            self.eval_statistics['Min Rew for Exp'] = np.min(exp_rews)

            if not self.use_rev_KL:
                eval_mode_exp_rews = ptu.get_numpy(torch.exp(eval_mode_T_exp_outputs - 1.0))
                # eval_mode_exp_rews = ptu.get_numpy(self._tapered_exp(eval_mode_T_exp_outputs - 1.0))
                # eval_mode_exp_rews = ptu.get_numpy(eval_mode_T_exp_outputs - 1.0)
            else:
                # eval_mode_exp_rews = ptu.get_numpy(eval_mode_T_exp_outputs)
                eval_mode_exp_rews = ptu.get_numpy(eval_mode_T_exp_outputs * torch.exp(eval_mode_T_exp_outputs))
            self.eval_statistics['Eval Mode Avg Rew for Exp'] = np.mean(eval_mode_exp_rews)
            self.eval_statistics['Eval Mode Std Rew for Exp'] = np.std(eval_mode_exp_rews)
            self.eval_statistics['Eval Mode Max Rew for Exp'] = np.max(eval_mode_exp_rews)
            self.eval_statistics['Eval Mode Min Rew for Exp'] = np.min(eval_mode_exp_rews)

            if not self.use_rev_KL:
                only_exp_eval_mode_exp_rews = ptu.get_numpy(torch.exp(only_exp_eval_mode_T_outputs - 1.0))
                # only_exp_eval_mode_exp_rews = ptu.get_numpy(self._tapered_exp(only_exp_eval_mode_T_outputs - 1.0))
                # only_exp_eval_mode_exp_rews = ptu.get_numpy(only_exp_eval_mode_T_outputs - 1.0)
            else:
                # only_exp_eval_mode_exp_rews = ptu.get_numpy(only_exp_eval_mode_T_outputs)
                only_exp_eval_mode_exp_rews = ptu.get_numpy(only_exp_eval_mode_T_outputs * torch.exp(only_exp_eval_mode_T_outputs))
            self.eval_statistics['Only Exp Eval Mode Avg Rew for Exp'] = np.mean(only_exp_eval_mode_exp_rews)
            self.eval_statistics['Only Exp Eval Mode Std Rew for Exp'] = np.std(only_exp_eval_mode_exp_rews)
            self.eval_statistics['Only Exp Eval Mode Max Rew for Exp'] = np.max(only_exp_eval_mode_exp_rews)
            self.eval_statistics['Only Exp Eval Mode Min Rew for Exp'] = np.min(only_exp_eval_mode_exp_rews)

            if not self.use_rev_KL:
                pol_rews = ptu.get_numpy(torch.exp(T_pol_outputs - 1.0))
                # pol_rews = ptu.get_numpy(self._tapered_exp(T_pol_outputs - 1.0))
                # pol_rews = ptu.get_numpy(T_pol_outputs - 1.0)
            else:
                # pol_rews = ptu.get_numpy(T_pol_outputs)
                pol_rews = ptu.get_numpy(T_pol_outputs * torch.exp(T_pol_outputs))
            self.eval_statistics['Avg Rew for Pol'] = np.mean(pol_rews)
            self.eval_statistics['Std Rew for Pol'] = np.std(pol_rews)
            self.eval_statistics['Max Rew for Pol'] = np.max(pol_rews)
            self.eval_statistics['Min Rew for Pol'] = np.min(pol_rews)

            if not self.use_rev_KL:
                eval_mode_pol_rews = ptu.get_numpy(torch.exp(eval_mode_T_pol_outputs - 1.0))
                # eval_mode_pol_rews = ptu.get_numpy(self._tapered_exp(eval_mode_T_pol_outputs - 1.0))
                # eval_mode_pol_rews = ptu.get_numpy(eval_mode_T_pol_outputs - 1.0)
            else:
                # eval_mode_pol_rews = ptu.get_numpy(eval_mode_T_pol_outputs)
                eval_mode_pol_rews = ptu.get_numpy(eval_mode_T_pol_outputs * torch.exp(eval_mode_T_pol_outputs))
            self.eval_statistics['Eval Mode Avg Rew for Pol'] = np.mean(eval_mode_pol_rews)
            self.eval_statistics['Eval Mode Std Rew for Pol'] = np.std(eval_mode_pol_rews)
            self.eval_statistics['Eval Mode Max Rew for Pol'] = np.max(eval_mode_pol_rews)
            self.eval_statistics['Eval Mode Min Rew for Pol'] = np.min(eval_mode_pol_rews)

            if not self.use_rev_KL:
                only_pol_eval_mode_pol_rews = ptu.get_numpy(torch.exp(only_pol_eval_mode_T_outputs - 1.0))
                # only_pol_eval_mode_pol_rews = ptu.get_numpy(self._tapered_exp(only_pol_eval_mode_T_outputs - 1.0))
                # only_pol_eval_mode_pol_rews = ptu.get_numpy(only_pol_eval_mode_T_outputs - 1.0)
            else:
                # only_pol_eval_mode_pol_rews = ptu.get_numpy(only_pol_eval_mode_T_outputs)
                only_pol_eval_mode_pol_rews = ptu.get_numpy(only_pol_eval_mode_T_outputs * torch.exp(only_pol_eval_mode_T_outputs))
            self.eval_statistics['Only Pol Eval Mode Avg Rew for Pol'] = np.mean(only_pol_eval_mode_pol_rews)
            self.eval_statistics['Only Pol Eval Mode Std Rew for Pol'] = np.std(only_pol_eval_mode_pol_rews)
            self.eval_statistics['Only Pol Eval Mode Max Rew for Pol'] = np.max(only_pol_eval_mode_pol_rews)
            self.eval_statistics['Only Pol Eval Mode Min Rew for Pol'] = np.min(only_pol_eval_mode_pol_rews)


    def _do_policy_training(self, epoch, update_loop_iter, pol_update_iter):
        self.disc_encoder.train()
        self.discriminator.eval()
        self.main_policy.train()
        self.q_model.train()

        self.q_optimizer.zero_grad()
        # the policy grad are zeroed inside the policy optimizer

        context_batch, policy_batch, mask = self._get_policy_training_batch()
        policy_batch = np_to_pytorch_batch(policy_batch)

        # get the disc representation for the context batch
        # mode = self.disc_encoder.training
        # self.disc_encoder.eval()
        D_c_repr, intermediate_r = self.disc_encoder(context_batch, mask, return_r=True)
        D_c_repr = D_c_repr.detach()
        # self.disc_encoder.train(mode)

        # get z, remember to detach it
        if not self.only_Dc:
            intermediate_r = intermediate_r.detach()
            post_dist = self.q_model(context_batch, mask)
            # post_dist = self.q_model(context_batch, mask, r=intermediate_r)
            z = post_dist.sample() # N_tasks x Dim

        if self.policy_optim_batch_mode_random:
            # repeat z to have the right size
            if not self.only_Dc:
                repeated_z = z.repeat(1, self.policy_optim_batch_size_per_task).view(
                    self.num_tasks_used_per_update * self.policy_optim_batch_size_per_task,
                    -1
                )
            repeated_D_c_repr = D_c_repr.repeat(1, self.policy_optim_batch_size_per_task).view(
                self.num_tasks_used_per_update * self.policy_optim_batch_size_per_task,
                -1
            ).detach()
        else:
            raise NotImplementedError()

        # compute the rewards
        if self.use_target_disc:
            disc_for_rew = self.target_disc
        else:
            disc_for_rew = self.discriminator
        
        # we have to do this cause the batch statistics for the disc are
        # different than what it is trained on in _do_reward_training
        # here it is only seeing policy samples, not policy and exp samples
        disc_for_rew.eval()
        if self.state_only:
            if (not self.only_Dc) and (not self.disc_ignores_z):
                T_outputs = disc_for_rew(policy_batch['observations'], None, repeated_D_c_repr, repeated_z).detach()
            else:
                T_outputs = disc_for_rew(policy_batch['observations'], None, repeated_D_c_repr).detach()
        else:
            disc_for_rew.eval()
            # self.discriminator.eval()
            # eval_T_outputs = self.discriminator(policy_batch['observations'], policy_batch['actions'], repeated_D_c_repr, repeated_z).detach()
            if (not self.only_Dc) and (not self.disc_ignores_z):
                eval_T_outputs = disc_for_rew(policy_batch['observations'], policy_batch['actions'], repeated_D_c_repr, repeated_z).detach()
            else:
                eval_T_outputs = disc_for_rew(policy_batch['observations'], policy_batch['actions'], repeated_D_c_repr).detach()

            disc_for_rew.train()
            # self.discriminator.train()

            # train_T_outputs = self.discriminator(policy_batch['observations'], policy_batch['actions'], repeated_D_c_repr, repeated_z).detach()
            if (not self.only_Dc) and (not self.disc_ignores_z):
                train_T_outputs = disc_for_rew(policy_batch['observations'], policy_batch['actions'], repeated_D_c_repr, repeated_z).detach()
            else:
                train_T_outputs = disc_for_rew(policy_batch['observations'], policy_batch['actions'], repeated_D_c_repr).detach()
            
        if not self.use_rev_KL:        
            # rew_to_give = torch.exp(eval_T_outputs - 1.0)
            # # maybe clip and center it
            # rew_to_give = torch.clamp(
            #     rew_to_give,
            #     min=0.0,
            #     max=5.0
            # ) - 2.5

            rew_to_give = self._tapered_exp(eval_T_outputs - 1.0) - 5.0
        else:
            # rew_to_give = eval_T_outputs
            rew_to_give = torch.exp(eval_T_outputs) * eval_T_outputs
            rew_to_give = torch.clamp(rew_to_give, min=-20.0, max=20.0) - 10.0
        
        rew_to_give = rew_to_give.detach()
        policy_batch['rewards'] = rew_to_give

        # disc_for_rew.train()

        # now augment the obs with the latent sample z
        if not self.only_Dc:
            detached_repeated_z = repeated_z.detach()
            detached_repeated_z.requires_grad = True
            policy_batch['observations'] = torch.cat([policy_batch['observations'], detached_repeated_z], dim=1)
            policy_batch['next_observations'] = torch.cat([policy_batch['next_observations'], detached_repeated_z], dim=1)
        
            # do a policy update (the zeroing of grads etc. should be handled internally)
            d_pol_loss_d_repeated_z = self.policy_optimizer.train_step(policy_batch, compute_grad_pol_loss_wrt_var=True, var_for_grad=detached_repeated_z)
            autograd.backward(
                [repeated_z],
                grad_variables=[d_pol_loss_d_repeated_z]
            )
            self.q_optimizer.step()
        else:
            policy_batch['observations'] = torch.cat([policy_batch['observations'], repeated_D_c_repr], dim=1)
            policy_batch['next_observations'] = torch.cat([policy_batch['next_observations'], repeated_D_c_repr], dim=1)
            self.policy_optimizer.train_step(policy_batch)

        if update_loop_iter == 0 and pol_update_iter == 0:
            given_rews = ptu.get_numpy(policy_batch['rewards']).flatten()
            self.eval_statistics['Disc Rew Mean'] = np.mean(given_rews)
            self.eval_statistics['Disc Rew Std'] = np.std(given_rews)
            self.eval_statistics['Disc Rew Max'] = np.max(given_rews)
            self.eval_statistics['Disc Rew Min'] = np.min(given_rews)

            if not self.use_rev_KL:
                # train_rews = ptu.get_numpy(torch.exp(train_T_outputs - 1.0))
                train_rews = ptu.get_numpy(self._tapered_exp(train_T_outputs - 1.0) - 5.0)
                # train_rews = ptu.get_numpy(train_T_outputs - 1.0)
            else:
                # train_rews = ptu.get_numpy(train_T_outputs)
                train_rews = ptu.get_numpy(torch.exp(train_T_outputs)*train_T_outputs)
            self.eval_statistics['Train Disc Rew Mean'] = np.mean(train_rews)
            self.eval_statistics['Train Disc Rew Std'] = np.std(train_rews)
            self.eval_statistics['Train Disc Rew Max'] = np.max(train_rews)
            self.eval_statistics['Train Disc Rew Min'] = np.min(train_rews)

            self.eval_statistics.update(self.policy_optimizer.eval_statistics)

            # only_Dc_exp_rews_T_clip_10_gp_1p0_rew_scale_10_repr_dim_32_disc_enc_adam_0p9_pol_128_mod_rews_clamped_disc_obj_use_disc_obs_processor
            plot_histogram(given_rews, 40, 'given_rews', 'plots/junk_vis/gp_0p5_rew_scale_5_tanh_taper_5_with_q_with_bn_z_dim_16_q_Adam_beta_0p9_disc_ignores_z_1_vs_4.png')
    

    def evaluate(self, epoch):
        super().evaluate(epoch)
        self.policy_optimizer.eval_statistics = None


    def obtain_eval_samples(self, epoch, mode='meta_train'):
        self.training_mode(False)

        if mode == 'meta_train':
            params_samples = self.train_task_params_sampler.sample_unique(self.num_tasks_per_eval)
        else:
            params_samples = self.test_task_params_sampler.sample_unique(self.num_tasks_per_eval)
        all_eval_tasks_paths = []
        for task_params, obs_task_params in params_samples:
            cur_eval_task_paths = []
            self.env.reset(task_params=task_params, obs_task_params=obs_task_params)
            task_identifier = self.env.task_identifier

            for _ in range(self.num_diff_context_per_eval_task):
                eval_policy = self.get_eval_policy(task_identifier, mode=mode)

                for _ in range(self.num_eval_trajs_per_post_sample):
                    cur_eval_path_builder = PathBuilder()
                    observation = self.env.reset(task_params=task_params, obs_task_params=obs_task_params)
                    terminal = False

                    while (not terminal) and len(cur_eval_path_builder) < self.max_path_length:
                        if isinstance(self.obs_space, Dict):
                            if self.policy_uses_pixels:
                                agent_obs = observation['pixels']
                            else:
                                agent_obs = observation['obs']
                        else:
                            agent_obs = observation
                        action, agent_info = eval_policy.get_action(agent_obs)
                        
                        next_ob, raw_reward, terminal, env_info = (self.env.step(action))
                        if self.no_terminal:
                            terminal = False
                        
                        reward = raw_reward
                        terminal = np.array([terminal])
                        reward = np.array([reward])
                        cur_eval_path_builder.add_all(
                            observations=observation,
                            actions=action,
                            rewards=reward,
                            next_observations=next_ob,
                            terminals=terminal,
                            agent_infos=agent_info,
                            env_infos=env_info,
                            task_identifiers=task_identifier
                        )
                        observation = next_ob

                    if terminal and self.wrap_absorbing:
                        raise NotImplementedError("I think they used 0 actions for this")
                        cur_eval_path_builder.add_all(
                            observations=next_ob,
                            actions=action,
                            rewards=reward,
                            next_observations=next_ob,
                            terminals=terminal,
                            agent_infos=agent_info,
                            env_infos=env_info,
                            task_identifiers=task_identifier
                        )
                    
                    if len(cur_eval_path_builder) > 0:
                        cur_eval_task_paths.append(
                            cur_eval_path_builder.get_all_stacked()
                        )
            all_eval_tasks_paths.extend(cur_eval_task_paths)
        
        # flatten the list of lists
        return all_eval_tasks_paths
    
    @property
    def networks(self):
        networks_list = [self.discriminator, self.disc_encoder, self.q_model]
        if self.use_target_disc: networks_list += [self.target_disc]
        if self.use_target_disc_enc: networks_list += [self.target_disc_enc]
        return networks_list + self.policy_optimizer.networks

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snap_update = {
            'discriminator': self.discriminator,
            'disc_encoder': self.disc_encoder,
            'q_model': self.q_model
        }
        snapshot.update(snap_update)
        if self.use_target_disc: snapshot.update(target_disc=self.target_disc)
        if self.use_target_disc_enc: snapshot.update(target_enc=self.target_enc)
        snapshot.update(self.policy_optimizer.get_snapshot())
        return snapshot


def _elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return Variable(ptu.from_numpy(elem_or_tuple).float(), requires_grad=False)


def _filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v


def np_to_pytorch_batch(np_batch):
    return {
        k: _elem_or_tuple_to_variable(x)
        for k, x in _filter_batch(np_batch)
        if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)
