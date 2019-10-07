from collections import OrderedDict
from copy import deepcopy

import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_irl_algorithm import TorchIRLAlgorithm
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.core.train_util import linear_schedule


def concat_trajs(trajs):
    new_dict = {}
    for k in trajs[0].keys():
        if isinstance(trajs[0][k], dict):
            new_dict[k] = concat_trajs([t[k] for t in trajs])
        else:
            new_dict[k] = np.concatenate([t[k] for t in trajs], axis=0)
    return new_dict


class AdvBC(TorchIRLAlgorithm):
    '''
        BC trained using an adversarial setup
        Either forward or reverse KL
    '''
    def __init__(
        self,
        env,
        policy,
        discriminator,

        expert_replay_buffer,

        KL_mode='forward',
        disc_num_trajs_per_batch=128,
        disc_samples_per_traj=8,

        disc_optim_batch_size=1024,
        policy_optim_batch_size=1024,

        num_update_loops_per_train_call=1000,
        num_disc_updates_per_loop_iter=1,
        num_policy_updates_per_loop_iter=1,

        disc_lr=1e-3,
        disc_momentum=0.0,
        disc_optimizer_class=optim.Adam,

        policy_lr=1e-3,
        policy_momentum=0.0,
        policy_optimizer_class=optim.Adam,

        use_grad_pen=True,
        grad_pen_weight=10,

        disc_ce_grad_clip=0.5,
        disc_gp_grad_clip=10.0,

        use_target_disc=False,
        target_disc=None,
        soft_target_disc_tau=0.005,

        # rew_clip_min=None,
        # rew_clip_max=None,

        plotter=None,
        render_eval_paths=False,
        eval_deterministic=True,

        use_disc_input_noise=False,
        disc_input_noise_scale_start=0.1,
        disc_input_noise_scale_end=0.0,
        epochs_till_end_scale=50.0,
        **kwargs
    ):
        assert disc_lr != 1e-3, 'Just checking that this is being taken from the spec file'
        assert KL_mode in ['forward', 'reverse']
        if eval_deterministic:
            eval_policy = MakeDeterministic(policy)
        else:
            eval_policy = policy
        
        super().__init__(
            env=env,
            exploration_policy=policy,
            eval_policy=eval_policy,
            expert_replay_buffer=expert_replay_buffer,
            **kwargs
        )
        assert not self.wrap_absorbing

        self.disc_num_trajs_per_batch = disc_num_trajs_per_batch
        self.disc_samples_per_traj = disc_samples_per_traj

        self.discriminator = discriminator
        self.rewardf_eval_statistics = None
        self.disc_optimizer = disc_optimizer_class(
            self.discriminator.parameters(),
            lr=disc_lr,
            betas=(disc_momentum, 0.999)
        )
        print('\n\nDISC MOMENTUM: %f\n\n' % disc_momentum)
        
        self.policy_optimizer = policy_optimizer_class(
            self.exploration_policy.parameters(),
            lr=policy_lr,
            betas=(policy_momentum, 0.999)
        )

        self.disc_optim_batch_size = disc_optim_batch_size
        self.policy_optim_batch_size = policy_optim_batch_size

        self.bce = nn.BCEWithLogitsLoss()
        target_batch_size = self.disc_optim_batch_size
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
        self.disc_gp_grad_clip = disc_gp_grad_clip
        self.disc_grad_buffer = {}
        self.disc_grad_buffer_is_empty = True

        self.use_target_disc = use_target_disc
        self.soft_target_disc_tau = soft_target_disc_tau

        if use_target_disc:
            if target_disc is None:
                print('\n\nMAKING TARGET DISC\n\n')
                self.target_disc = deepcopy(self.discriminator)
            else:
                print('\n\nUSING GIVEN TARGET DISC\n\n')
                self.target_disc = target_disc
        
        self.disc_ce_grad_norm = 0.0
        self.disc_ce_grad_norm_counter = 0.0
        self.max_disc_ce_grad = 0.0

        self.disc_gp_grad_norm = 0.0
        self.disc_gp_grad_norm_counter = 0.0
        self.max_disc_gp_grad = 0.0

        self.use_disc_input_noise = use_disc_input_noise
        self.disc_input_noise_scale_start = disc_input_noise_scale_start
        self.disc_input_noise_scale_end = disc_input_noise_scale_end
        self.epochs_till_end_scale = epochs_till_end_scale

        self.num_update_loops_per_train_call = num_update_loops_per_train_call
        self.num_disc_updates_per_loop_iter = num_disc_updates_per_loop_iter
        self.num_policy_updates_per_loop_iter = num_policy_updates_per_loop_iter

        if KL_mode == 'forward':
            self.use_forward_KL = True
        else:
            self.use_forward_KL = False
        # self.rew_clip_min = rew_clip_min
        # self.rew_clip_max = rew_clip_max
        # self.clip_min_rews = rew_clip_min is not None
        # self.clip_max_rews = rew_clip_max is not None


    def get_batch(self, batch_size, from_expert):
        if from_expert:
            buffer = self.expert_replay_buffer
        else:
            buffer = self.replay_buffer
        batch = buffer.random_batch(batch_size)
        batch = np_to_pytorch_batch(batch)
        return batch


    def _do_training(self, epoch):
        for t in range(self.num_update_loops_per_train_call):
            for _ in range(self.num_disc_updates_per_loop_iter):
                self._do_reward_training(epoch)
            for _ in range(self.num_policy_updates_per_loop_iter):
                self._do_policy_training(epoch)


    def _do_reward_training(self, epoch):
        '''
            Train the discriminator
        '''
        self.disc_optimizer.zero_grad()

        expert_batch = self.get_batch(self.disc_optim_batch_size, True)
        policy_batch = {}
        pol_mode = self.exploration_policy.training
        self.exploration_policy.train()
        policy_batch['observations'] = expert_batch['observations']
        policy_batch['actions'] = self.exploration_policy(expert_batch['observations'])[0].detach()
        self.exploration_policy.train(pol_mode)

        expert_obs = expert_batch['observations']
        policy_obs = policy_batch['observations']
        expert_actions = expert_batch['actions']
        policy_actions = policy_batch['actions']

        # print('----------------------------')
        # print(torch.mean(expert_obs, dim=0))
        # print(torch.std(expert_obs, dim=0))
        # print(torch.mean(expert_actions, dim=0))
        # print(torch.std(expert_actions, dim=0))

        if self.use_disc_input_noise:
            noise_scale = linear_schedule(
                epoch,
                self.disc_input_noise_scale_start,
                self.disc_input_noise_scale_end,
                self.epochs_till_end_scale
            )
            if noise_scale > 0.0:
                expert_obs = expert_obs + noise_scale * Variable(torch.randn(expert_obs.size()))
                expert_actions = expert_actions + noise_scale * Variable(torch.randn(expert_actions.size()))

                policy_obs = policy_obs + noise_scale * Variable(torch.randn(policy_obs.size()))
                policy_actions = policy_actions + noise_scale * Variable(torch.randn(policy_actions.size()))
        
        obs = torch.cat([expert_obs, policy_obs], dim=0)
        actions = torch.cat([expert_actions, policy_actions], dim=0)

        disc_logits = self.discriminator(obs, actions)
        disc_preds = (disc_logits > 0).type(disc_logits.data.type())
        disc_ce_loss = self.bce(disc_logits, self.bce_targets)
        accuracy = (disc_preds == self.bce_targets).type(torch.FloatTensor).mean()

        disc_ce_loss.backward()

        ce_grad_norm = 0.0
        for name, param in self.discriminator.named_parameters():
            if param.grad is not None:
                if self.disc_grad_buffer_is_empty:
                    self.disc_grad_buffer[name] = param.grad.data.clone()
                else:
                    self.disc_grad_buffer[name].copy_(param.grad.data)
                
                param_norm = param.grad.data.norm(2)
                ce_grad_norm += param_norm ** 2
        ce_grad_norm = ce_grad_norm ** 0.5
        self.disc_grad_buffer_is_empty = False

        ce_clip_coef = self.disc_ce_grad_clip / (ce_grad_norm + 1e-6)
        if ce_clip_coef < 1.:
            for name, grad in self.disc_grad_buffer.items():
                grad.mul_(ce_clip_coef)
        
        if ce_clip_coef < 1.0: ce_grad_norm *= ce_clip_coef
        self.max_disc_ce_grad = max(ce_grad_norm, self.max_disc_ce_grad)
        self.disc_ce_grad_norm += ce_grad_norm
        self.disc_ce_grad_norm_counter += 1
        
        self.disc_optimizer.zero_grad()
        
        if self.use_grad_pen:
            eps = Variable(torch.rand(expert_obs.size(0), 1))
            if ptu.gpu_enabled(): eps = eps.cuda()
            
            interp_obs = eps*expert_obs + (1-eps)*policy_obs
            interp_obs = interp_obs.detach()
            interp_obs.requires_grad = True
            
            interp_actions = eps*expert_actions + (1-eps)*policy_actions
            interp_actions = interp_actions.detach()
            interp_actions.requires_grad = True
            gradients = autograd.grad(
                outputs=self.discriminator(interp_obs, interp_actions).sum(),
                # inputs=[interp_obs, interp_actions],
                inputs=[interp_actions],
                # grad_outputs=torch.ones(exp_specs['batch_size'], 1).cuda(),
                create_graph=True, retain_graph=True, only_inputs=True
            )
            # total_grad = torch.cat([gradients[0], gradients[1]], dim=1)
            total_grad = gradients[0]

            # GP from Gulrajani et al.
            gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()
            disc_grad_pen_loss = gradient_penalty * self.grad_pen_weight

            # # GP from Mescheder et al.
            # gradient_penalty = (total_grad.norm(2, dim=1) ** 2).mean()
            # disc_grad_pen_loss = gradient_penalty * 0.5 * self.grad_pen_weight

            disc_grad_pen_loss.backward()

            gp_grad_norm = 0.0
            for p in list(filter(lambda p: p.grad is not None, self.discriminator.parameters())):
                param_norm = p.grad.data.norm(2)
                gp_grad_norm += param_norm ** 2
            gp_grad_norm = gp_grad_norm ** 0.5

            gp_clip_coef = self.disc_gp_grad_clip / (gp_grad_norm + 1e-6)
            if gp_clip_coef < 1.:
                for p in self.discriminator.parameters():
                    p.grad.data.mul_(gp_clip_coef)
            
            if gp_clip_coef < 1.: gp_grad_norm *= gp_clip_coef
            self.max_disc_gp_grad = max(gp_grad_norm, self.max_disc_gp_grad)
            self.disc_gp_grad_norm += gp_grad_norm
            self.disc_gp_grad_norm_counter += 1
        
        # now add back the gradients from the CE loss
        for name, param in self.discriminator.named_parameters():
            param.grad.data.add_(self.disc_grad_buffer[name])

        self.disc_optimizer.step()

        if self.use_target_disc:
            ptu.soft_update_from_to(self.discriminator, self.target_disc, self.soft_target_disc_tau)

        """
        Save some statistics for eval
        """
        if self.rewardf_eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.rewardf_eval_statistics = OrderedDict()
            
            if self.use_target_disc:
                target_disc_logits = self.target_disc(obs, actions)
                target_disc_preds = (target_disc_logits > 0).type(target_disc_logits.data.type())
                target_disc_ce_loss = self.bce(target_disc_logits, self.bce_targets)
                target_accuracy = (target_disc_preds == self.bce_targets).type(torch.FloatTensor).mean()

                if self.use_grad_pen:
                    eps = Variable(torch.rand(expert_obs.size(0), 1))
                    if ptu.gpu_enabled(): eps = eps.cuda()
                    
                    interp_obs = eps*expert_obs + (1-eps)*policy_obs
                    interp_obs = interp_obs.detach()
                    interp_obs.requires_grad = True
                    
                    interp_actions = eps*expert_actions + (1-eps)*policy_actions
                    interp_actions = interp_actions.detach()
                    interp_actions.requires_grad = True
                    target_gradients = autograd.grad(
                        outputs=self.target_disc(interp_obs, interp_actions).sum(),
                        # inputs=[interp_obs, interp_actions],
                        inputs=[interp_actions],
                        # grad_outputs=torch.ones(exp_specs['batch_size'], 1).cuda(),
                        create_graph=True, retain_graph=True, only_inputs=True
                    )
                    # total_grad = torch.cat([gradients[0], gradients[1]], dim=1)
                    total_target_grad = target_gradients[0]

                    # GP from Gulrajani et al.
                    target_gradient_penalty = ((total_target_grad.norm(2, dim=1) - 1) ** 2).mean()

                    # # GP from Mescheder et al.
                    # target_gradient_penalty = (total_target_grad.norm(2, dim=1) ** 2).mean()

                self.rewardf_eval_statistics['Target Disc CE Loss'] = np.mean(ptu.get_numpy(target_disc_ce_loss))
                self.rewardf_eval_statistics['Target Disc Acc'] = np.mean(ptu.get_numpy(target_accuracy))
                self.rewardf_eval_statistics['Target Grad Pen'] = np.mean(ptu.get_numpy(target_gradient_penalty))
                self.rewardf_eval_statistics['Target Grad Pen W'] = np.mean(self.grad_pen_weight)
            
            self.rewardf_eval_statistics['Disc CE Loss'] = np.mean(ptu.get_numpy(disc_ce_loss))
            self.rewardf_eval_statistics['Disc Acc'] = np.mean(ptu.get_numpy(accuracy))
            self.rewardf_eval_statistics['Grad Pen'] = np.mean(ptu.get_numpy(gradient_penalty))
            self.rewardf_eval_statistics['Grad Pen W'] = np.mean(self.grad_pen_weight)
            self.rewardf_eval_statistics['Disc Avg CE Grad Norm this epoch'] = np.mean(self.disc_ce_grad_norm / self.disc_ce_grad_norm_counter)
            self.rewardf_eval_statistics['Disc Max CE Grad Norm this epoch'] = np.mean(self.max_disc_ce_grad)
            self.rewardf_eval_statistics['Disc Avg GP Grad Norm this epoch'] = np.mean(self.disc_gp_grad_norm / self.disc_gp_grad_norm_counter)
            self.rewardf_eval_statistics['Disc Max GP Grad Norm this epoch'] = np.mean(self.max_disc_gp_grad)
            if self.use_disc_input_noise:
                self.rewardf_eval_statistics['Disc Input Noise Scale'] = noise_scale

            self.max_disc_ce_grad = 0.0
            self.disc_ce_grad_norm = 0.0
            self.disc_ce_grad_norm_counter = 0.0
            self.max_disc_gp_grad = 0.0
            self.disc_gp_grad_norm = 0.0
            self.disc_gp_grad_norm_counter = 0.0


    def _do_policy_training(self, epoch):
        self.exploration_policy.train()
        self.policy_optimizer.zero_grad()

        batch = self.get_batch(self.policy_optim_batch_size, True)
        obs = batch['observations']
        acts = self.exploration_policy(obs)[0]
        if self.use_target_disc:
            self.target_disc.eval()
            # If you compute log(D) - log(1-D) then you just get the logits
            disc_logits = self.target_disc(obs, acts)
            self.target_disc.train()
        else:
            self.discriminator.eval()
            # If you compute log(D) - log(1-D) then you just get the logits
            disc_logits = self.discriminator(obs, acts)
            self.discriminator.train()

        if self.use_forward_KL:
            loss = (torch.exp(disc_logits)*(disc_logits)).mean()
        else:
            loss = -1.0 * disc_logits.mean()
        loss.backward()
        self.policy_optimizer.step()

        self.rewardf_eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(loss))
    
    
    @property
    def networks(self):
        return [self.exploration_policy, self.discriminator]

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(exploration_policy=self.exploration_policy)
        snapshot.update(disc=self.discriminator)
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
