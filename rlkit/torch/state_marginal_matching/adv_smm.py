import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim
from torch import nn
from torch import autograd
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.torch_base_algorithm import TorchBaseAlgorithm


class AdvSMM(TorchBaseAlgorithm):
    '''
        TODO: The implementation for this differs very little
        from adversarial irl, can be modularized

        Depending on choice of reward function and size of replay
        buffer this will be the SMM equivalent of:
            - AIRL
            - GAIL (without extra entropy term)
            - FAIRL
            - Discriminator Actor Critic
        
        I did not implement the reward-wrapping mentioned in
        https://arxiv.org/pdf/1809.02925.pdf though

        Features removed from v1.0:
            - gradient clipping
            - target disc (exponential moving average disc)
            - target policy (exponential moving average policy)
            - disc input noise
    '''
    def __init__(
        self,

        mode, # airl, gail, or fairl
        discriminator,
        policy_trainer, # e.g. SAC

        target_state_buffer,
        state_indices, # LongTensor state indices for matching marginals

        disc_optim_batch_size=1024,
        policy_optim_batch_size=1024,
        policy_optim_batch_size_from_expert=0,

        num_update_loops_per_train_call=1,
        num_disc_updates_per_loop_iter=100,
        num_policy_updates_per_loop_iter=100,

        num_initial_disc_iters=0,

        disc_lr=1e-3,
        disc_momentum=0.0,
        disc_optimizer_class=optim.Adam,

        use_grad_pen=True,
        grad_pen_weight=10,

        rew_clip_min=None,
        rew_clip_max=None,

        **kwargs
    ):
        assert mode in ['airl', 'gail', 'fairl'], 'Invalid adversarial irl algorithm!'
        if kwargs['wrap_absorbing']: raise NotImplementedError()
        super().__init__(**kwargs)

        self.mode = mode

        self.target_state_buffer = target_state_buffer
        self.state_indices = state_indices

        self.policy_trainer = policy_trainer
        self.policy_optim_batch_size = policy_optim_batch_size
        self.policy_optim_batch_size_from_expert = policy_optim_batch_size_from_expert
        
        self.discriminator = discriminator
        self.disc_optimizer = disc_optimizer_class(
            self.discriminator.parameters(),
            lr=disc_lr,
            betas=(disc_momentum, 0.999)
        )
        self.disc_optim_batch_size = disc_optim_batch_size
        print('\n\nDISC MOMENTUM: %f\n\n' % disc_momentum)

        self.bce = nn.BCEWithLogitsLoss()
        self.bce_targets = torch.cat(
            [
                torch.ones(disc_optim_batch_size, 1),
                torch.zeros(disc_optim_batch_size, 1)
            ],
            dim=0
        )
        self.bce.to(ptu.device)
        self.bce_targets = self.bce_targets.to(ptu.device)
        
        self.use_grad_pen = use_grad_pen
        self.grad_pen_weight = grad_pen_weight

        self.num_update_loops_per_train_call = num_update_loops_per_train_call
        self.num_disc_updates_per_loop_iter = num_disc_updates_per_loop_iter
        self.num_policy_updates_per_loop_iter = num_policy_updates_per_loop_iter

        self.num_initial_disc_iters = num_initial_disc_iters

        self.rew_clip_min = rew_clip_min
        self.rew_clip_max = rew_clip_max
        self.clip_min_rews = rew_clip_min is not None
        self.clip_max_rews = rew_clip_max is not None

        self.disc_eval_statistics = None
        self.not_done_initial_disc_iters = True


    def get_batch(self, batch_size, from_target_state_buffer, keys=None):
        if from_target_state_buffer:
            buffer = self.target_state_buffer
            batch = {
                'observations': buffer[np.random.choice(buffer.shape[0], size=batch_size)]
            }
        else:
            buffer = self.replay_buffer
            batch = buffer.random_batch(batch_size, keys=keys)
        batch = np_to_pytorch_batch(batch)
        return batch


    def _end_epoch(self):
        self.policy_trainer.end_epoch()
        self.disc_eval_statistics = None
        super()._end_epoch()


    def evaluate(self, epoch):
        self.eval_statistics = OrderedDict()
        self.eval_statistics.update(self.disc_eval_statistics)
        self.eval_statistics.update(self.policy_trainer.get_eval_statistics())
        super().evaluate(epoch)


    def _do_training(self, epoch):
        if self.not_done_initial_disc_iters:
            for t in range(self.num_initial_disc_iters):
                self._do_reward_training(epoch)
            self.not_done_initial_disc_iters = False
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

        keys = ['observations']
        expert_batch = self.get_batch(self.disc_optim_batch_size, True, keys)
        policy_batch = self.get_batch(self.disc_optim_batch_size, False, keys)

        if self.wrap_absorbing:
            pass
            # expert_obs = torch.cat([expert_obs, expert_batch['absorbing'][:, 0:1]], dim=-1)
            # policy_obs = torch.cat([policy_obs, policy_batch['absorbing'][:, 0:1]], dim=-1)
        
        expert_disc_input = expert_batch['observations']
        policy_disc_input = policy_batch['observations']
        policy_disc_input = torch.index_select(policy_disc_input, 1, self.state_indices)
        disc_input = torch.cat([expert_disc_input, policy_disc_input], dim=0)

        disc_logits = self.discriminator(disc_input)
        disc_preds = (disc_logits > 0).type(disc_logits.data.type())
        disc_ce_loss = self.bce(disc_logits, self.bce_targets)
        accuracy = (disc_preds == self.bce_targets).type(torch.FloatTensor).mean()

        if self.use_grad_pen:
            eps = ptu.rand(expert_disc_input.size(0), 1)
            eps.to(ptu.device)
            
            interp_obs = eps*expert_disc_input + (1-eps)*policy_disc_input
            interp_obs = interp_obs.detach()
            interp_obs.requires_grad_(True)

            gradients = autograd.grad(
                outputs=self.discriminator(interp_obs).sum(),
                inputs=[interp_obs],
                create_graph=True, retain_graph=True, only_inputs=True
            )
            total_grad = gradients[0]
            
            # GP from Gulrajani et al.
            gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()
            disc_grad_pen_loss = gradient_penalty * self.grad_pen_weight

            # # GP from Mescheder et al.
            # gradient_penalty = (total_grad.norm(2, dim=1) ** 2).mean()
            # disc_grad_pen_loss = gradient_penalty * 0.5 * self.grad_pen_weight
        else:
            disc_grad_pen_loss = 0.0

        disc_total_loss = disc_ce_loss + disc_grad_pen_loss
        disc_total_loss.backward()
        self.disc_optimizer.step()

        """
        Save some statistics for eval
        """
        if self.disc_eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.disc_eval_statistics = OrderedDict()
            
            self.disc_eval_statistics['Disc CE Loss'] = np.mean(ptu.get_numpy(disc_ce_loss))
            self.disc_eval_statistics['Disc Acc'] = np.mean(ptu.get_numpy(accuracy))
            if self.use_grad_pen:
                self.disc_eval_statistics['Grad Pen'] = np.mean(ptu.get_numpy(gradient_penalty))
                self.disc_eval_statistics['Grad Pen W'] = np.mean(self.grad_pen_weight)


    def _do_policy_training(self, epoch):
        if self.policy_optim_batch_size_from_expert > 0:
            policy_batch_from_policy_buffer = self.get_batch(
                self.policy_optim_batch_size - self.policy_optim_batch_size_from_expert, False)
            policy_batch_from_expert_buffer = self.get_batch(
                self.policy_optim_batch_size_from_expert, True)
            policy_batch = {}
            for k in policy_batch_from_policy_buffer:
                policy_batch[k] = torch.cat(
                    [
                        policy_batch_from_policy_buffer[k],
                        policy_batch_from_expert_buffer[k]
                    ],
                    dim=0
                )
        else:
            policy_batch = self.get_batch(self.policy_optim_batch_size, False)
        
        obs = policy_batch['observations']
        acts = policy_batch['actions']
        next_obs = policy_batch['next_observations']

        self.discriminator.eval()
        disc_input = torch.index_select(obs, 1, self.state_indices)
        disc_logits = self.discriminator(disc_input).detach()
        self.discriminator.train()

        # compute the reward using the algorithm
        if self.mode == 'airl':
            # If you compute log(D) - log(1-D) then you just get the logits
            policy_batch['rewards'] = disc_logits
        elif self.mode == 'gail':
            policy_batch['rewards'] = F.softplus(disc_logits, beta=-1)
        else: # fairl
            policy_batch['rewards'] = torch.exp(disc_logits)*(-1.0*disc_logits)
        
        if self.clip_max_rews:
            policy_batch['rewards'] = torch.clamp(policy_batch['rewards'], max=self.rew_clip_max)
        if self.clip_min_rews:
            policy_batch['rewards'] = torch.clamp(policy_batch['rewards'], min=self.rew_clip_min)
        
        # policy optimization step
        # import pdb; pdb.set_trace()
        self.policy_trainer.train_step(policy_batch)

        self.disc_eval_statistics['Disc Rew Mean'] = np.mean(ptu.get_numpy(policy_batch['rewards']))
        self.disc_eval_statistics['Disc Rew Std'] = np.std(ptu.get_numpy(policy_batch['rewards']))
        self.disc_eval_statistics['Disc Rew Max'] = np.max(ptu.get_numpy(policy_batch['rewards']))
        self.disc_eval_statistics['Disc Rew Min'] = np.min(ptu.get_numpy(policy_batch['rewards']))
    
    
    @property
    def networks(self):
        return [self.discriminator] + self.policy_trainer.networks


    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(disc=self.discriminator)
        snapshot.update(self.policy_trainer.get_snapshot())
        return snapshot


    def to(self, device):
        self.bce.to(ptu.device)
        self.bce_targets = self.bce_targets.to(ptu.device)
        self.state_indices = self.state_indices.to(ptu.device)
        super().to(device)
