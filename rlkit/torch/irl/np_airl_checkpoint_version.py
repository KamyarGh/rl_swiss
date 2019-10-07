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


class NeuralProcessAIRL(TorchMetaIRLAlgorithm):
    def __init__(
            self,
            env,
            policy,
            discriminator,

            train_context_expert_replay_buffer,
            train_test_expert_replay_buffer,
            test_context_expert_replay_buffer,
            test_test_expert_replay_buffer,

            np_encoder,

            policy_optimizer,

            state_only=False,
            transfer_version=False,
            r_V_reg_weight=0.0,

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
            
            encoder_lr=1e-3,
            encoder_optimizer_class=optim.Adam,

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

            use_target_enc=False,
            target_enc=None,
            soft_target_enc_tau=0.005,

            max_KL_beta = 1.0,
            KL_ramp_up_start_iter=0,
            KL_ramp_up_end_iter=100,

            plotter=None,
            render_eval_paths=False,
            eval_deterministic=False,

            **kwargs
    ):
        assert disc_lr != 1e-3, 'Just checking that this is being taken from the spec file'
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
        self.encoder = np_encoder
        self.discriminator = discriminator
        self.eval_statistics = None

        self.policy_optimizer = policy_optimizer
        self.disc_optimizer = disc_optimizer_class(
            self.discriminator.parameters(),
            lr=disc_lr,
            betas=(disc_Adam_beta, 0.999)
        )
        self.disc_Adam_beta = disc_Adam_beta
        print('\n\nDISC ADAM BETA IS %f\n\n' % disc_Adam_beta)
        print('\n\nENCODER ADAM IS 0.9\n\n')
        self.encoder_optimizer = encoder_optimizer_class(
            self.encoder.parameters(),
            lr=encoder_lr,
            betas=(0.9, 0.999)
            # betas=(0.0, 0.999)
        )

        self.state_only = state_only
        self.transfer_version = transfer_version
        if transfer_version: assert not state_only

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
        
        self.use_target_enc = use_target_enc
        self.soft_target_enc_tau = soft_target_enc_tau

        if use_target_enc:
            if target_enc is None:
                print('\n\nMAKING TARGET ENC\n\n')
                self.target_enc = self.encoder.copy()
            else:
                print('\n\nUSING GIVEN TARGET ENC\n\n')
                self.target_enc = target_enc
        
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

        self.r_V_reg_weight = r_V_reg_weight
        print('R_V_REG_WEIGHT is %f' % r_V_reg_weight)

        self.max_KL_beta = max_KL_beta
        self.KL_ramp_up_start_iter = KL_ramp_up_start_iter
        self.KL_ramp_up_end_iter = KL_ramp_up_end_iter

        self.few_shot_version = few_shot_version
        self.max_context_size = max_context_size
        self.min_context_size = min_context_size
        assert num_context_trajs_for_training == max_context_size
    

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

        if self.use_target_enc:
            enc_to_use = self.target_enc
        else:
            enc_to_use = self.encoder
        
        mode = enc_to_use.training
        enc_to_use.eval()
        post_dist = enc_to_use([list_of_trajs], mask)
        enc_to_use.train(mode)

        # z = post_dist.sample()
        z = post_dist.mean
        z = z.cpu().data.numpy()[0]
        return PostCondMLPPolicyWrapper(self.main_policy, z)
    

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
        
        if self.use_target_enc:
            enc_to_use = self.target_enc
        else:
            enc_to_use = self.encoder
        
        mode = enc_to_use.training
        enc_to_use.eval()
        post_dist = enc_to_use([list_of_trajs])
        enc_to_use.train(mode)

        # z = post_dist.sample()
        z = post_dist.mean
        z = z.cpu().data.numpy()[0]
        return PostCondMLPPolicyWrapper(self.main_policy, z)
    

    def _get_disc_training_batch(self):
        keys_to_get = ['observations', 'actions']
        if self.transfer_version: keys_to_get.append('next_observations')
        
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
        if self.transfer_version:
            policy_test_pred_batch['next_observations'] = np.concatenate((policy_test_pred_batch_0['next_observations'], policy_test_pred_batch_1['next_observations']), axis=0)

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


    def _compute_KL_loss(self, post_dist):
        # computting the KL of a Gaussian posterior from the spherical prior
        z_means = post_dist.mean
        z_log_covs = post_dist.log_cov
        z_covs = post_dist.cov
        KL = torch.mean(
            1.0 + z_log_covs - z_means**2 - z_covs,
            dim=-1
        )
        KL =  -0.5 * torch.sum(KL)
        return KL


    def _do_training(self, epoch):
        for t in range(self.num_update_loops_per_train_call):
            for _ in range(self.num_disc_updates_per_loop_iter):
                self._do_reward_training(epoch, t)
            for _ in range(self.num_policy_updates_per_loop_iter):
                self._do_policy_training(epoch)


    def _do_reward_training(self, epoch, loop_iter):
        '''
            Train the discriminator
        '''
        self.encoder_optimizer.zero_grad()
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

        post_dist = self.encoder(context_batch, mask)
        # z = post_dist.sample() # N_tasks x Dim
        z = post_dist.mean

        # make z's for expert samples
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
        
        if self.transfer_version:
            # make the next_obs_batch
            exp_next_obs_batch = np.concatenate((context_pred_batch['next_observations'], test_pred_batch['next_observations']), axis=0)
            exp_next_obs_batch = Variable(ptu.from_numpy(exp_next_obs_batch), requires_grad=False)
            policy_next_obs_batch = Variable(ptu.from_numpy(policy_test_pred_batch['next_observations']), requires_grad=False)
            next_obs_batch = torch.cat([exp_next_obs_batch, policy_next_obs_batch], dim=0)

            # compute the log probability of actions under the policy
            obs_input_to_policy = torch.cat([obs_batch, repeated_z_batch], dim=1)
            pol_log_prob = self.main_policy.get_log_prob(obs_input_to_policy, acts_batch).detach()
            disc_logits, r, r_shaping, V_s = self.discriminator(
                obs_batch,
                None,
                repeated_z_batch,
                pol_log_prob=pol_log_prob,
                next_obs_batch=next_obs_batch    
            )
        else:
            disc_logits = self.discriminator(obs_batch, acts_batch, repeated_z_batch)
        disc_preds = (disc_logits > 0).type(disc_logits.data.type())
        # disc_percent_policy_preds_one = disc_preds[z.size(0):].mean()
        disc_ce_loss = self.bce(disc_logits, self.bce_targets)
        accuracy = (disc_preds == self.bce_targets).type(torch.FloatTensor).mean()

        if self.transfer_version:
            # regularize the magnitude of the magnitude of r and V(s)
            if self.r_V_reg_weight > 0.0:
                # regularizing the r and V_s
                # r_L2 = torch.mean(r**2)
                # V_L2 = 0.1*torch.mean(V_s**2)
                # total_L2 = r_L2 + V_L2
                # loss_to_backprop = disc_ce_loss + self.r_V_reg_weight*total_L2
                # loss_to_backprop.backward()

                # regularizing the logit
                loss_to_backprop = disc_ce_loss + self.r_V_reg_weight*torch.mean(disc_logits**2)
            else:
                loss_to_backprop = disc_ce_loss
        else:
            loss_to_backprop = disc_ce_loss

        # add KL loss term
        cur_KL_beta = linear_schedule(
            self._n_train_steps_total*self.num_disc_updates_per_loop_iter + loop_iter - self.KL_ramp_up_start_iter,
            0.0,
            self.max_KL_beta,
            self.KL_ramp_up_end_iter - self.KL_ramp_up_start_iter
        )
        KL_loss = self._compute_KL_loss(post_dist)
        if cur_KL_beta == 0.0: KL_loss = KL_loss.detach()

        loss_to_backprop = loss_to_backprop + cur_KL_beta * KL_loss
        loss_to_backprop.backward()
        
        # compute the CE grad norms
        this_iter_disc_ce_grad_norm = 0.0
        for name, param in self.discriminator.named_parameters():
            if param.grad is not None:
                if self.disc_grad_buffer_is_empty:
                    self.disc_grad_buffer[name] = param.grad.data.clone()
                else:
                    self.disc_grad_buffer[name].copy_(param.grad.data)
                
                param_norm = param.grad.data.norm(2)
                this_iter_disc_ce_grad_norm += param_norm ** 2
        this_iter_disc_ce_grad_norm = this_iter_disc_ce_grad_norm ** 0.5
        self.disc_grad_buffer_is_empty = False

        this_iter_enc_ce_grad_norm = 0.0
        for name, param in self.encoder.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                this_iter_enc_ce_grad_norm += param_norm ** 2
        this_iter_enc_ce_grad_norm = this_iter_enc_ce_grad_norm ** 0.5
        
        # PREVIOUS VERSION OF CLIPPING
        # this_iter_total_ce_grad_norm = (this_iter_disc_ce_grad_norm**2 + this_iter_enc_ce_grad_norm**2)**0.5
        # ce_clip_coef = self.disc_ce_grad_clip / (this_iter_total_ce_grad_norm + 1e-6)
        # if ce_clip_coef < 1.:
        #     for name, grad in self.disc_grad_buffer.items():
        #         grad.mul_(ce_clip_coef)
        #     # for the encoder we directly do it since we don't have to care
        #     # about the gradients for any other loss on the encoder
        #     for param in self.encoder.parameters():
        #         if param.grad is not None: param.grad.data.mul_(ce_clip_coef)

        # if ce_clip_coef < 1.:
        #     this_iter_disc_ce_grad_norm *= ce_clip_coef
        #     this_iter_enc_ce_grad_norm *= ce_clip_coef
        #     this_iter_total_ce_grad_norm *= ce_clip_coef

        # NEW VERSION OF CLIPPING THAT HAS SEPARATE CLIPPING FOR ENCODER AND DISC MODEL
        disc_ce_clip_coef = self.disc_ce_grad_clip / (this_iter_disc_ce_grad_norm + 1e-6)
        if disc_ce_clip_coef < 1.:
            for name, grad in self.disc_grad_buffer.items():
                grad.mul_(disc_ce_clip_coef)
            this_iter_disc_ce_grad_norm *= disc_ce_clip_coef
        
        enc_ce_clip_coef = self.enc_ce_grad_clip / (this_iter_enc_ce_grad_norm + 1e-6)
        if enc_ce_clip_coef < 1.:
            for param in self.encoder.parameters():
                if param.grad is not None: param.grad.data.mul_(enc_ce_clip_coef)
            this_iter_enc_ce_grad_norm *= enc_ce_clip_coef

        self.max_disc_ce_grad = max(this_iter_disc_ce_grad_norm, self.max_disc_ce_grad)
        self.disc_ce_grad_norm += this_iter_disc_ce_grad_norm
        self.disc_ce_grad_norm_counter += 1

        self.max_enc_ce_grad = max(this_iter_enc_ce_grad_norm, self.max_enc_ce_grad)
        self.enc_ce_grad_norm += this_iter_enc_ce_grad_norm
        self.enc_ce_grad_norm_counter += 1
        
        # now to handle GP
        self.disc_optimizer.zero_grad()
        
        if self.use_grad_pen:
            eps = Variable(torch.rand(exp_obs_batch.size(0), 1))
            if ptu.gpu_enabled(): eps = eps.cuda()
            
            interp_obs = eps*exp_obs_batch + (1-eps)*policy_obs_batch
            interp_obs = interp_obs.detach()
            interp_obs.requires_grad = True
            if self.state_only:
                gradients = autograd.grad(
                    outputs=self.discriminator(interp_obs, None, z_batch.detach()).sum(),
                    inputs=[interp_obs],
                    # grad_outputs=torch.ones(exp_specs['batch_size'], 1).cuda(),
                    create_graph=True, retain_graph=True, only_inputs=True
                )
                total_grad = gradients[0]
            else:
                if self.transfer_version:
                    interp_next_obs = eps*exp_next_obs_batch + (1-eps)*policy_next_obs_batch
                    interp_next_obs = interp_next_obs.detach()
                    interp_next_obs.requires_grad = True

                    interp_policy_log_prob = eps*pol_log_prob[:interp_obs.size(0)] + (1-eps)*pol_log_prob[interp_obs.size(0):]
                    interp_policy_log_prob = interp_policy_log_prob.detach()
                    interp_policy_log_prob.requires_grad = True

                    gradients = autograd.grad(
                        outputs=self.discriminator(
                            interp_obs,
                            None,
                            z_batch.detach(),
                            pol_log_prob=interp_policy_log_prob,
                            next_obs_batch=interp_next_obs
                        )[0].sum(),
                        inputs=[interp_obs, interp_policy_log_prob],
                        # grad_outputs=torch.ones(exp_specs['batch_size'], 1).cuda(),
                        create_graph=True, retain_graph=True, only_inputs=True
                    )
                    total_grad = torch.cat([gradients[0], gradients[1]], dim=1)
                else:
                    interp_actions = eps*exp_acts_batch + (1-eps)*policy_acts_batch
                    interp_actions = interp_actions.detach()
                    interp_actions.requires_grad = True
                    gradients = autograd.grad(
                        outputs=self.discriminator(interp_obs, interp_actions, z_batch.detach()).sum(),
                        inputs=[interp_obs, interp_actions],
                        # grad_outputs=torch.ones(exp_specs['batch_size'], 1).cuda(),
                        create_graph=True, retain_graph=True, only_inputs=True
                    )
                    total_grad = torch.cat([gradients[0], gradients[1]], dim=1)


            # GP from Gulrajani et al.
            gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()
            disc_grad_pen_loss = gradient_penalty * self.grad_pen_weight

            disc_grad_pen_loss.backward()

            this_iter_gp_grad_norm = 0.0
            for p in list(filter(lambda p: p.grad is not None, self.discriminator.parameters())):
                param_norm = p.grad.data.norm(2)
                this_iter_gp_grad_norm += param_norm ** 2
            this_iter_gp_grad_norm = this_iter_gp_grad_norm ** 0.5

            gp_clip_coef = self.disc_gp_grad_clip / (this_iter_gp_grad_norm + 1e-6)
            if gp_clip_coef < 1.:
                for p in self.discriminator.parameters():
                    if p.grad is not None: p.grad.data.mul_(gp_clip_coef)
            
            if gp_clip_coef < 1.: this_iter_gp_grad_norm *= gp_clip_coef
            self.max_disc_gp_grad = max(this_iter_gp_grad_norm, self.max_disc_gp_grad)
            self.disc_gp_grad_norm += this_iter_gp_grad_norm
            self.disc_gp_grad_norm_counter += 1
        
        # now add back the gradients from the CE loss
        for name, param in self.discriminator.named_parameters():
            if param.grad is not None: param.grad.data.add_(self.disc_grad_buffer[name])

        self.disc_optimizer.step()
        self.encoder_optimizer.step()
        if self.transfer_version:
            self.discriminator._update_target_V_part()
        if self.use_target_disc:
            ptu.soft_update_from_to(self.discriminator, self.target_disc, self.soft_target_disc_tau)
        if self.use_target_enc:
            ptu.soft_update_from_to(self.encoder, self.target_enc, self.soft_target_enc_tau)

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
                    target_disc_logits = self.target_disc(obs_batch, None, repeated_z_batch)
                else:
                    if self.transfer_version:
                        target_disc_logits, target_r, target_r_shaping, target_V_s = self.target_disc(
                            obs_batch,
                            None,
                            repeated_z_batch,
                            pol_log_prob=pol_log_prob,
                            next_obs_batch=next_obs_batch
                        )
                    else:
                        target_disc_logits = self.target_disc(obs_batch, acts_batch, repeated_z_batch)
                target_disc_preds = (target_disc_logits > 0).type(target_disc_logits.data.type())
                target_disc_ce_loss = self.bce(target_disc_logits, self.bce_targets)
                target_accuracy = (target_disc_preds == self.bce_targets).type(torch.FloatTensor).mean()

                if self.use_grad_pen:
                    eps = Variable(torch.rand(exp_obs_batch.size(0), 1))
                    if ptu.gpu_enabled(): eps = eps.cuda()
                    
                    interp_obs = eps*exp_obs_batch + (1-eps)*policy_obs_batch
                    interp_obs = interp_obs.detach()
                    interp_obs.requires_grad = True
                    if self.state_only:
                        target_gradients = autograd.grad(
                            outputs=self.target_disc(interp_obs, None, z_batch).sum(),
                            inputs=[interp_obs],
                            # grad_outputs=torch.ones(exp_specs['batch_size'], 1).cuda(),
                            create_graph=True, retain_graph=True, only_inputs=True
                        )
                        total_target_grad = target_gradients[0]
                    else:
                        if self.transfer_version:
                            interp_next_obs = eps*exp_next_obs_batch + (1-eps)*policy_next_obs_batch
                            interp_next_obs = interp_next_obs.detach()
                            interp_next_obs.requires_grad = True

                            interp_policy_log_prob = eps*pol_log_prob[:interp_obs.size(0)] + (1-eps)*pol_log_prob[interp_obs.size(0):]
                            interp_policy_log_prob = interp_policy_log_prob.detach()
                            interp_policy_log_prob.requires_grad = True

                            target_gradients = autograd.grad(
                                outputs=self.target_disc(
                                    interp_obs,
                                    None,
                                    z_batch.detach(),
                                    pol_log_prob=interp_policy_log_prob,
                                    next_obs_batch=interp_next_obs
                                )[0].sum(),
                                inputs=[interp_obs, interp_policy_log_prob],
                                # grad_outputs=torch.ones(exp_specs['batch_size'], 1).cuda(),
                                create_graph=True, retain_graph=True, only_inputs=True
                            )
                            total_target_grad = torch.cat([target_gradients[0], target_gradients[1]], dim=1)
                        else:
                            interp_actions = eps*exp_acts_batch + (1-eps)*policy_acts_batch
                            interp_actios = interp_actions.detach()
                            interp_actions.requires_grad = True
                            target_gradients = autograd.grad(
                                outputs=self.target_disc(interp_obs, interp_actions, z_batch).sum(),
                                inputs=[interp_obs, interp_actions],
                                # grad_outputs=torch.ones(exp_specs['batch_size'], 1).cuda(),
                                create_graph=True, retain_graph=True, only_inputs=True
                            )
                            total_target_grad = torch.cat([target_gradients[0], target_gradients[1]], dim=1)

                    # GP from Gulrajani et al.
                    target_gradient_penalty = ((total_target_grad.norm(2, dim=1) - 1) ** 2).mean()

                self.eval_statistics['Target Disc CE Loss'] = np.mean(ptu.get_numpy(target_disc_ce_loss))
                self.eval_statistics['Target Disc Acc'] = np.mean(ptu.get_numpy(target_accuracy))
                self.eval_statistics['Target Grad Pen'] = np.mean(ptu.get_numpy(target_gradient_penalty))
                self.eval_statistics['Target Grad Pen W'] = np.mean(self.grad_pen_weight)
            
            self.eval_statistics['Disc CE Loss'] = np.mean(ptu.get_numpy(disc_ce_loss))
            self.eval_statistics['Disc Acc'] = np.mean(ptu.get_numpy(accuracy))
            self.eval_statistics['Grad Pen'] = np.mean(ptu.get_numpy(gradient_penalty))
            self.eval_statistics['Grad Pen W'] = np.mean(self.grad_pen_weight)

            self.eval_statistics['Target KL'] = np.mean(ptu.get_numpy(KL_loss))
            self.eval_statistics['Cur KL Beta'] = cur_KL_beta
            self.eval_statistics['Max KL Beta'] = self.max_KL_beta

            self.eval_statistics['Avg Post Mean Abs'] = np.mean(np.abs(ptu.get_numpy(post_dist.mean)))
            self.eval_statistics['Avg Post Cov Abs'] = np.mean(np.abs(ptu.get_numpy(post_dist.cov)))

            self.eval_statistics['Disc Avg CE Grad Norm this epoch'] = np.mean(self.disc_ce_grad_norm / self.disc_ce_grad_norm_counter)
            self.eval_statistics['Disc Max CE Grad Norm this epoch'] = np.mean(self.max_disc_ce_grad)
            self.eval_statistics['Enc Avg CE Grad Norm this epoch'] = np.mean(self.enc_ce_grad_norm / self.enc_ce_grad_norm_counter)
            self.eval_statistics['Enc Max CE Grad Norm this epoch'] = np.mean(self.max_enc_ce_grad)
            self.eval_statistics['Disc Avg GP Grad Norm this epoch'] = np.mean(self.disc_gp_grad_norm / self.disc_gp_grad_norm_counter)
            self.eval_statistics['Disc Max GP Grad Norm this epoch'] = np.mean(self.max_disc_gp_grad)

            if self.transfer_version:
                if not self.use_target_disc:
                    self.eval_statistics['Avg R'] = np.mean(ptu.get_numpy(r))
                    self.eval_statistics['Std R'] = np.std(ptu.get_numpy(r))
                    self.eval_statistics['Max R'] = np.max(ptu.get_numpy(r))
                    self.eval_statistics['Min R'] = np.min(ptu.get_numpy(r))

                    self.eval_statistics['Avg V_s'] = np.mean(ptu.get_numpy(V_s))
                    self.eval_statistics['Std V_s'] = np.std(ptu.get_numpy(V_s))
                    self.eval_statistics['Max V_s'] = np.max(ptu.get_numpy(V_s))
                    self.eval_statistics['Min V_s'] = np.min(ptu.get_numpy(V_s))

                    self.eval_statistics['Avg Shaping'] = np.mean(ptu.get_numpy(r_shaping))
                    self.eval_statistics['Std Shaping'] = np.std(ptu.get_numpy(r_shaping))
                    self.eval_statistics['Max Shaping'] = np.max(ptu.get_numpy(r_shaping))
                    self.eval_statistics['Min Shaping'] = np.min(ptu.get_numpy(r_shaping))

                    self.eval_statistics['Avg f'] = np.mean(ptu.get_numpy(r + r_shaping))
                    self.eval_statistics['Std f'] = np.std(ptu.get_numpy(r + r_shaping))
                    self.eval_statistics['Max f'] = np.max(ptu.get_numpy(r + r_shaping))
                    self.eval_statistics['Min f'] = np.min(ptu.get_numpy(r + r_shaping))
                else:
                    self.eval_statistics['Target Avg R'] = np.mean(ptu.get_numpy(target_r))
                    self.eval_statistics['Target Std R'] = np.std(ptu.get_numpy(target_r))
                    self.eval_statistics['Target Max R'] = np.max(ptu.get_numpy(target_r))
                    self.eval_statistics['Target Min R'] = np.min(ptu.get_numpy(target_r))

                    self.eval_statistics['Target Avg V_s'] = np.mean(ptu.get_numpy(target_V_s))
                    self.eval_statistics['Target Std V_s'] = np.std(ptu.get_numpy(target_V_s))
                    self.eval_statistics['Target Max V_s'] = np.max(ptu.get_numpy(target_V_s))
                    self.eval_statistics['Target Min V_s'] = np.min(ptu.get_numpy(target_V_s))

                    self.eval_statistics['Target Avg Shaping'] = np.mean(ptu.get_numpy(target_r_shaping))
                    self.eval_statistics['Target Std Shaping'] = np.std(ptu.get_numpy(target_r_shaping))
                    self.eval_statistics['Target Max Shaping'] = np.max(ptu.get_numpy(target_r_shaping))
                    self.eval_statistics['Target Min Shaping'] = np.min(ptu.get_numpy(target_r_shaping))

                    self.eval_statistics['Target Avg f'] = np.mean(ptu.get_numpy(target_r + target_r_shaping))
                    self.eval_statistics['Target Std f'] = np.std(ptu.get_numpy(target_r + target_r_shaping))
                    self.eval_statistics['Target Max f'] = np.max(ptu.get_numpy(target_r + target_r_shaping))
                    self.eval_statistics['Target Min f'] = np.min(ptu.get_numpy(target_r + target_r_shaping))

                self.eval_statistics['Avg Pol LogProb for Exp'] = np.mean(ptu.get_numpy(pol_log_prob[:interp_obs.size(0)]))
                self.eval_statistics['Std Pol LogProb for Exp'] = np.std(ptu.get_numpy(pol_log_prob[:interp_obs.size(0)]))
                self.eval_statistics['Max Pol LogProb for Exp'] = np.max(ptu.get_numpy(pol_log_prob[:interp_obs.size(0)]))
                self.eval_statistics['Min Pol LogProb for Exp'] = np.min(ptu.get_numpy(pol_log_prob[:interp_obs.size(0)]))

                self.eval_statistics['Avg Pol LogProb for Pol'] = np.mean(ptu.get_numpy(pol_log_prob[interp_obs.size(0):]))
                self.eval_statistics['Std Pol LogProb for Pol'] = np.std(ptu.get_numpy(pol_log_prob[interp_obs.size(0):]))
                self.eval_statistics['Max Pol LogProb for Pol'] = np.max(ptu.get_numpy(pol_log_prob[interp_obs.size(0):]))
                self.eval_statistics['Min Pol LogProb for Pol'] = np.min(ptu.get_numpy(pol_log_prob[interp_obs.size(0):]))


            self.max_disc_ce_grad = 0.0
            self.disc_ce_grad_norm = 0.0
            self.disc_ce_grad_norm_counter = 0.0
            self.max_enc_ce_grad = 0.0
            self.enc_ce_grad_norm = 0.0
            self.enc_ce_grad_norm_counter = 0.0
            self.max_disc_gp_grad = 0.0
            self.disc_gp_grad_norm = 0.0
            self.disc_gp_grad_norm_counter = 0.0


    def _do_policy_training(self, epoch):
        context_batch, policy_batch, mask = self._get_policy_training_batch()
        policy_batch = np_to_pytorch_batch(policy_batch)

        if self.use_target_enc:
            enc_to_use = self.target_enc
        else:
            enc_to_use = self.encoder
        
        mode = enc_to_use.training
        enc_to_use.eval()
        post_dist = enc_to_use(context_batch, mask)
        enc_to_use.train(mode)
        
        # z = post_dist.sample() # N_tasks x Dim
        z = post_dist.mean
        if self.policy_optim_batch_mode_random:
            # repeat z to have the right size
            z = z.repeat(1, self.policy_optim_batch_size_per_task).view(
                self.num_tasks_used_per_update * self.policy_optim_batch_size_per_task,
                -1
            )
        else:
            raise NotImplementedError()

        z = z.detach()
        # compute the rewards
        # If you compute log(D) - log(1-D) then you just get the logits
        if self.use_target_disc:
            disc_for_rew = self.target_disc
        else:
            disc_for_rew = self.discriminator
        
        disc_for_rew.eval()
        if self.state_only:
            policy_batch['rewards'] = disc_for_rew(policy_batch['observations'], None, z).detach()
        else:
            if self.transfer_version:
                # compute the log probability of actions under the policy
                obs_input_to_policy = torch.cat([policy_batch['observations'], z], dim=1)
                pol_log_prob = self.main_policy.get_log_prob(obs_input_to_policy, policy_batch['actions']).detach()
                policy_batch['rewards'] = disc_for_rew(
                    policy_batch['observations'],
                    None,
                    z,
                    pol_log_prob=pol_log_prob,
                    next_obs_batch=policy_batch['next_observations']   
                )[0].detach()
            else:
                policy_batch['rewards'] = disc_for_rew(policy_batch['observations'], policy_batch['actions'], z).detach()
        disc_for_rew.train()

        # now augment the obs with the latent sample z
        policy_batch['observations'] = torch.cat([policy_batch['observations'], z], dim=1)
        policy_batch['next_observations'] = torch.cat([policy_batch['next_observations'], z], dim=1)

        # do a policy update (the zeroing of grads etc. should be handled internally)
        self.policy_optimizer.train_step(policy_batch)

        self.eval_statistics['Disc Rew Mean'] = np.mean(ptu.get_numpy(policy_batch['rewards']))
        self.eval_statistics['Disc Rew Std'] = np.std(ptu.get_numpy(policy_batch['rewards']))
        self.eval_statistics['Disc Rew Max'] = np.max(ptu.get_numpy(policy_batch['rewards']))
        self.eval_statistics['Disc Rew Min'] = np.min(ptu.get_numpy(policy_batch['rewards']))
    

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
        networks_list = [self.discriminator, self.encoder]
        if self.use_target_disc: networks_list += [self.target_disc]
        if self.use_target_enc: networks_list += [self.target_enc]
        return networks_list + self.policy_optimizer.networks

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(discriminator=self.discriminator)
        snapshot.update(encoder=self.encoder)
        if self.use_target_disc: snapshot.update(target_disc=self.target_disc)
        if self.use_target_enc: snapshot.update(target_enc=self.target_enc)
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
