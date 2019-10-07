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


class ThreewayStateMarginalMatchingAlg(TorchIRLAlgorithm):
    '''
        state-marginal matching
        expert, pol, uniform
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

        disc_lr=1e-3,
        disc_momentum=0.0,
        disc_optimizer_class=optim.Adam,

        use_grad_pen=True,
        grad_pen_weight=10,

        plotter=None,
        render_eval_paths=False,
        eval_deterministic=True,

        train_objective='airl',

        **kwargs
    ):
        assert train_objective == 'airl'
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
                torch.zeros(target_batch_size),
                torch.ones(target_batch_size),
                2*torch.ones(target_batch_size),
            ],
            dim=0
        ).type(torch.LongTensor)
        self.bce_targets = Variable(self.bce_targets)
        if ptu.gpu_enabled():
            self.bce.cuda()
            self.bce_targets = self.bce_targets.cuda()
        
        self.use_grad_pen = use_grad_pen
        self.grad_pen_weight = grad_pen_weight

        self.num_update_loops_per_train_call = num_update_loops_per_train_call
        self.num_disc_updates_per_loop_iter = num_disc_updates_per_loop_iter
        self.num_policy_updates_per_loop_iter = num_policy_updates_per_loop_iter

        d = 5.0
        self._d = d
        self._d_len = np.arange(-d,d+0.25,0.25).shape[0]
        self.xy_var = []
        for i in np.arange(d,-d-0.25,-0.25):
            for j in np.arange(-d,d+0.25,0.25):
                self.xy_var.append([float(j),float(i)])
        self.xy_var = np.array(self.xy_var)
        self.xy_var = Variable(ptu.from_numpy(self.xy_var), requires_grad=False)


    def get_disc_training_batch(self, batch_size, mode):
        if mode == 'expert':
            buffer = self.expert_replay_buffer
            idx = np.random.choice(self.expert_replay_buffer.shape[0], size=batch_size, replace=False)
            batch = {
                'observations': buffer[idx]
            }
        elif mode == 'policy':
            buffer = self.replay_buffer
            batch = buffer.random_batch(batch_size)
            batch['observations'] = batch['observations'][:,-2:]
        else:
            batch = {
                'observations': 2*self._d*np.random.uniform(size=(batch_size, 2)) - self._d
            }
        batch = np_to_pytorch_batch(batch)
        return batch
    

    def get_policy_training_batch(self, batch_size):
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
        
        self.discriminator.eval()
        logits = self.discriminator(self.xy_var, None)
        rewards = self._convert_logits_to_reward(logits)
        self.discriminator.train()

        # plot the logits of the discriminator
        # logits = ptu.get_numpy(logits)
        # logits = np.reshape(logits, (int(self._d_len), int(self._d_len)))
        # plot_seaborn_grid(logits, None, None, 'Disc Logits Epoch %d'%epoch, osp.join(logger.get_snapshot_dir(), 'disc_logits_epoch_%d.png'%epoch))

        # plot the rewards given by the discriminator
        rewards = ptu.get_numpy(rewards)
        rewards = np.reshape(rewards, (int(self._d_len), int(self._d_len)))
        plot_seaborn_grid(rewards, -10, 10, 'Disc Rewards Epoch %d'%epoch, osp.join(logger.get_snapshot_dir(), 'disc_rewards_epoch_%d.png'%epoch))


    def _do_reward_training(self, epoch):
        '''
            Train the discriminator
        '''
        self.disc_optimizer.zero_grad()

        expert_batch = self.get_disc_training_batch(self.disc_optim_batch_size, 'expert')
        policy_batch = self.get_disc_training_batch(self.disc_optim_batch_size, 'policy')
        uniform_batch = self.get_disc_training_batch(self.disc_optim_batch_size, 'uniform')

        expert_obs = expert_batch['observations']
        policy_obs = policy_batch['observations']
        uniform_obs = uniform_batch['observations']

        obs = torch.cat([expert_obs, policy_obs, uniform_obs], dim=0)
        disc_logits = self.discriminator(obs, None)

        if self.train_objective == 'w1':
            raise Exception
            n = expert_obs.size(0)
            # not CE loss but I just got lazy about renaming things below
            disc_ce_loss = disc_logits[:n].mean() - disc_logits[n:].mean()
            total_loss = disc_ce_loss
        else: # the disc objective for all other approaches is BCE
            disc_ce_loss = F.cross_entropy(disc_logits, self.bce_targets)
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
            gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()
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
        _, max_idx = torch.max(logits, dim=-1, keepdim=True)
        unif_is_max = max_idx == 2
        unif_is_max = unif_is_max.type(logits.data.type())
        log_smax = F.log_softmax(logits, dim=-1)
        # 0 if unif is max else log density ratio of exp vs. pol
        reward = (1.0 - unif_is_max) * (log_smax[:,0:1] - log_smax[:,1:2])
        return reward


    def _do_policy_training(self, epoch):
        policy_batch = self.get_policy_training_batch(self.policy_optim_batch_size)
        obs = policy_batch['observations']
        acts = policy_batch['actions']
        
        self.discriminator.eval()
        policy_batch['rewards'] = self.discriminator(obs[:,-2:], None).detach()
        self.discriminator.train()

        policy_batch['rewards'] = self._convert_logits_to_reward(policy_batch['rewards'])

        self.policy_optimizer.train_step(policy_batch)

        self.rewardf_eval_statistics['Disc Rew Mean'] = np.mean(ptu.get_numpy(policy_batch['rewards']))
        self.rewardf_eval_statistics['Disc Rew Std'] = np.std(ptu.get_numpy(policy_batch['rewards']))
        self.rewardf_eval_statistics['Disc Rew Max'] = np.max(ptu.get_numpy(policy_batch['rewards']))
        self.rewardf_eval_statistics['Disc Rew Min'] = np.min(ptu.get_numpy(policy_batch['rewards']))
    

    def cuda(self):
        self.xy_var = self.xy_var.cuda()
        super().cuda()
    

    def cpu(self):
        self.xy_var = self.xy_var.cpu()
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
