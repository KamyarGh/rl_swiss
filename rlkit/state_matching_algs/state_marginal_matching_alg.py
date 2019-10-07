import os.path as osp
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
from rlkit.core.vistools import plot_seaborn_grid
from rlkit.core import logger


def concat_trajs(trajs):
    new_dict = {}
    for k in trajs[0].keys():
        if isinstance(trajs[0][k], dict):
            new_dict[k] = concat_trajs([t[k] for t in trajs])
        else:
            new_dict[k] = np.concatenate([t[k] for t in trajs], axis=0)
    return new_dict


class StateMarginalMatchingAlg(TorchIRLAlgorithm):
    '''
        state-marginal matching
    '''
    def __init__(
        self,
        env,
        policy,
        discriminator,

        policy_optimizer,
        expert_replay_buffer,

        disc_optim_batch_size=1024,
        policy_optim_batch_size=1024,

        num_update_loops_per_train_call=1000,
        num_disc_updates_per_loop_iter=1,
        num_policy_updates_per_loop_iter=1,

        # initial_only_disc_train_epochs=0,
        pretrain_disc=False,
        num_disc_pretrain_iters=1000,

        disc_lr=1e-3,
        disc_momentum=0.0,
        disc_optimizer_class=optim.Adam,

        use_grad_pen=True,
        grad_pen_weight=10,

        plotter=None,
        render_eval_paths=False,
        eval_deterministic=False,

        train_objective='airl',

        num_disc_input_dims=2,
        plot_reward_surface=True,

        use_survival_reward=False,
        use_ctrl_cost=False,
        ctrl_cost_weight=0.0,

        **kwargs
    ):
        assert disc_lr != 1e-3, 'Just checking that this is being taken from the spec file'
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

        self.policy_optimizer = policy_optimizer
        
        self.discriminator = discriminator
        self.rewardf_eval_statistics = None
        self.disc_optimizer = disc_optimizer_class(
            self.discriminator.parameters(),
            lr=disc_lr,
            betas=(disc_momentum, 0.999)
        )
        print('\n\nDISC MOMENTUM: %f\n\n' % disc_momentum)

        self.disc_optim_batch_size = disc_optim_batch_size
        self.policy_optim_batch_size = policy_optim_batch_size

        assert train_objective in ['airl', 'fairl', 'gail', 'w1']
        self.train_objective = train_objective

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

        self.num_update_loops_per_train_call = num_update_loops_per_train_call
        self.num_disc_updates_per_loop_iter = num_disc_updates_per_loop_iter
        self.num_policy_updates_per_loop_iter = num_policy_updates_per_loop_iter

        # self.initial_only_disc_train_epochs = initial_only_disc_train_epochs
        self.pretrain_disc = pretrain_disc
        self.did_disc_pretraining = False
        self.num_disc_pretrain_iters = num_disc_pretrain_iters
        self.cur_epoch = -1

        self.plot_reward_surface = plot_reward_surface
        if plot_reward_surface:
            d = 6
            self._d = d
            self._d_len = np.arange(-d,d+0.25,0.25).shape[0]
            self.xy_var = []
            for i in np.arange(d,-d-0.25,-0.25):
                for j in np.arange(-d,d+0.25,0.25):
                    self.xy_var.append([float(j),float(i)])
            self.xy_var = np.array(self.xy_var)
            self.xy_var = Variable(ptu.from_numpy(self.xy_var), requires_grad=False)

            # d = 20
            # self._d = d
            # # self._d_len = np.arange(0.98697072 - 0.3, 0.98697072 + 0.175, 0.02).shape[0]
            # self._d_len_rows = np.arange(0.74914774 - 0.35, 0.74914774 + 0.45, 0.01).shape[0]
            # self._d_len_cols = np.arange(0.98697072 - 0.3, 0.98697072 + 0.175, 0.01).shape[0]
            # self.xy_var = []
            # for i in np.arange(0.74914774 + 0.45, 0.74914774 - 0.35, -0.01):
            #     for j in np.arange(0.98697072 - 0.3, 0.98697072 + 0.175, 0.01):
            #         self.xy_var.append([float(j), float(i)])
            # self.xy_var = np.array(self.xy_var)
            # self.xy_var = Variable(ptu.from_numpy(self.xy_var), requires_grad=False)

        self.num_disc_input_dims = num_disc_input_dims

        self.use_survival_reward = use_survival_reward
        self.use_ctrl_cost = use_ctrl_cost
        self.ctrl_cost_weight = ctrl_cost_weight


    def get_disc_training_batch(self, batch_size, from_expert):
        if from_expert:
            buffer = self.expert_replay_buffer
            idx = np.random.choice(self.expert_replay_buffer.shape[0], size=batch_size, replace=False)
            batch = {
                'observations': buffer[idx]
            }
        else:
            buffer = self.replay_buffer
            batch = buffer.random_batch(batch_size)
            # batch['observations'] = batch['observations'][:,-2:]
            batch['observations'] = batch['observations'][:,:self.num_disc_input_dims]
        batch = np_to_pytorch_batch(batch)
        return batch
    

    def get_policy_training_batch(self, batch_size):
        buffer = self.replay_buffer
        batch = buffer.random_batch(batch_size)
        batch = np_to_pytorch_batch(batch)
        return batch


    def _do_training(self, epoch):
        if not self.did_disc_pretraining:
            for i in range(self.num_disc_pretrain_iters):
                self._do_reward_training(epoch)
                # print('pretrain iter %d' % i)
            self.did_disc_pretraining = True

        for t in range(self.num_update_loops_per_train_call):
            for _ in range(self.num_disc_updates_per_loop_iter):
                self._do_reward_training(epoch)
            for _ in range(self.num_policy_updates_per_loop_iter):
                self._do_policy_training(epoch)
            
            # if self.initial_only_disc_train_epochs < epoch:
            #     for _ in range(self.num_policy_updates_per_loop_iter):
            #         self._do_policy_training(epoch)
        
        # this is some weirdness I'll fix in refactoring but for now
        if self.plot_reward_surface:
            if self.cur_epoch != epoch:
                self.discriminator.eval()
                logits = self.discriminator(self.xy_var, None)
                rewards = self._convert_logits_to_reward(logits)
                self.discriminator.train()

                # plot the logits of the discriminator
                # logits = ptu.get_numpy(logits)
                # logits = np.reshape(logits, (int(self._d_len), int(self._d_len)))
                # plot_seaborn_grid(logits, -10, 10, 'Disc Logits Epoch %d'%epoch, osp.join(logger.get_snapshot_dir(), 'disc_logits_epoch_%d.png'%epoch))

                # plot the rewards given by the discriminator
                rewards = ptu.get_numpy(rewards)
                rewards = np.reshape(rewards, (int(self._d_len), int(self._d_len)))
                # rewards = np.reshape(rewards, (int(self._d_len_rows), int(self._d_len_cols)))
                plot_seaborn_grid(rewards, -10, 10, 'Disc Rewards Epoch %d'%epoch, osp.join(logger.get_snapshot_dir(), 'disc_rewards_epoch_%d.png'%epoch))

                self.cur_epoch = epoch


    def _do_reward_training(self, epoch):
        '''
            Train the discriminator
        '''
        self.disc_optimizer.zero_grad()

        expert_batch = self.get_disc_training_batch(self.disc_optim_batch_size, True)
        policy_batch = self.get_disc_training_batch(self.disc_optim_batch_size, False)

        expert_obs = expert_batch['observations']
        policy_obs = policy_batch['observations']

        obs = torch.cat([expert_obs, policy_obs], dim=0)
        disc_logits = self.discriminator(obs, None)

        if self.train_objective == 'w1':
            n = expert_obs.size(0)
            # not CE loss but I just got lazy about renaming things below
            disc_ce_loss = disc_logits[:n].mean() - disc_logits[n:].mean()
            total_loss = disc_ce_loss
        else: # the disc objective for all other approaches is BCE
            disc_ce_loss = self.bce(disc_logits, self.bce_targets)
            total_loss = disc_ce_loss
        # disc_preds = (disc_logits > 0).type(disc_logits.data.type())
        # accuracy = (disc_preds == self.bce_targets).type(torch.FloatTensor).mean()
        
        if self.use_grad_pen:
            eps = Variable(torch.rand(expert_obs.size(0), 1))
            if ptu.gpu_enabled(): eps = eps.cuda()
            
            interp_obs = eps*expert_obs + (1-eps)*policy_obs
            interp_obs = interp_obs.detach()
            interp_obs.requires_grad = True
            gradients = autograd.grad(
                outputs=self.discriminator(interp_obs, None).sum(),
                inputs=[interp_obs],
                # grad_outputs=torch.ones(exp_specs['batch_size'], 1).cuda(),
                create_graph=True, retain_graph=True, only_inputs=True
            )
            total_grad = gradients[0]

            # GP from Gulrajani et al.
            # gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()
            # disc_grad_pen_loss = gradient_penalty * self.grad_pen_weight

            # DIFFERENT FROM GP from Gulrajani et al.
            gradient_penalty = total_grad.norm(2, dim=1) - 1
            gradient_penalty = F.relu(gradient_penalty)
            gradient_penalty = (gradient_penalty**2).mean()
            disc_grad_pen_loss = gradient_penalty * self.grad_pen_weight

            # # GP from Mescheder et al.
            # gradient_penalty = (total_grad.norm(2, dim=1) ** 2).mean()
            # disc_grad_pen_loss = gradient_penalty * 0.5 * self.grad_pen_weight

            total_loss = total_loss + disc_grad_pen_loss

        total_loss.backward()
        self.disc_optimizer.step()

        """
        Save some statistics for eval
        """
        if self.rewardf_eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.rewardf_eval_statistics = OrderedDict()
            
            self.rewardf_eval_statistics['Disc CE Loss'] = np.mean(ptu.get_numpy(disc_ce_loss))
            # self.rewardf_eval_statistics['Disc Acc'] = np.mean(ptu.get_numpy(accuracy))
            if self.use_grad_pen:
                self.rewardf_eval_statistics['Grad Pen'] = np.mean(ptu.get_numpy(gradient_penalty))
                self.rewardf_eval_statistics['Grad Pen W'] = np.mean(self.grad_pen_weight)


    def _convert_logits_to_reward(self, logits):
        extra_reward = 0.0
        if self.use_survival_reward:
            extra_reward += 12.0
        
        if self.train_objective == 'airl':
            return logits + extra_reward
        elif self.train_objective == 'fairl':
            return torch.exp(policy_batch['rewards'])*(-1.0*policy_batch['rewards']) + extra_reward
        elif self.train_objective == 'gail':
            return F.softplus(policy_batch['rewards'], beta=-1) + extra_reward
        elif self.train_objective == 'w1':
            return -logits + extra_reward
        else:
            raise Exception()


    def _do_policy_training(self, epoch):
        policy_batch = self.get_policy_training_batch(self.policy_optim_batch_size)
        obs = policy_batch['observations']
        acts = policy_batch['actions']
        
        self.discriminator.eval()
        policy_batch['rewards'] = self.discriminator(obs[:,:self.num_disc_input_dims], None).detach()
        self.discriminator.train()

        policy_batch['rewards'] = self._convert_logits_to_reward(policy_batch['rewards'])
        if self.use_ctrl_cost:
            ctrl_cost = torch.sum(acts**2, dim=1, keepdim=True)
            policy_batch['rewards'] = policy_batch['rewards'] - ctrl_cost * self.ctrl_cost_weight

        self.policy_optimizer.train_step(policy_batch)

        self.rewardf_eval_statistics['Disc Rew Mean'] = np.mean(ptu.get_numpy(policy_batch['rewards']))
        self.rewardf_eval_statistics['Disc Rew Std'] = np.std(ptu.get_numpy(policy_batch['rewards']))
        self.rewardf_eval_statistics['Disc Rew Max'] = np.max(ptu.get_numpy(policy_batch['rewards']))
        self.rewardf_eval_statistics['Disc Rew Min'] = np.min(ptu.get_numpy(policy_batch['rewards']))
    

    def cuda(self):
        if self.plot_reward_surface: self.xy_var = self.xy_var.cuda()
        super().cuda()
    

    def cpu(self):
        if self.plot_reward_surface: self.xy_var = self.xy_var.cpu()
        super().cpu()


    @property
    def networks(self):
        return [self.discriminator] + self.policy_optimizer.networks


    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(disc=self.discriminator)
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
