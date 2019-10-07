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
from rlkit.torch.torch_irl_algorithm import TorchIRLAlgorithm
from rlkit.torch.sac.policies import MakeDeterministic


class NPAIRL(TorchIRLAlgorithm):
    '''
        Neural Process (meta) AIRL
    '''
    def __init__(
            self,
            env,
            policy,
            discriminator,

            policy_optimizer,
            expert_replay_buffer,

            disc_optim_batch_size=32,
            policy_optim_batch_size=1000,

            disc_lr=1e-3,
            disc_optimizer_class=optim.Adam,

            use_grad_pen=True,
            grad_pen_weight=10,

            plotter=None,
            render_eval_paths=False,
            eval_deterministic=True,
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
            policy_optimizer=policy_optimizer,
            **kwargs
        )

        self.discriminator = discriminator
        self.rewardf_eval_statistics = None
        self.disc_optimizer = disc_optimizer_class(
            self.discriminator.parameters(),
            lr=disc_lr,
        )

        self.disc_optim_batch_size = disc_optim_batch_size
        self.policy_optim_batch_size = policy_optim_batch_size

        self.bce = nn.BCEWithLogitsLoss()
        self.bce_targets = torch.cat(
            [
                torch.ones(self.disc_optim_batch_size, 1),
                torch.zeros(self.disc_optim_batch_size, 1)
            ],
            dim=0
        )
        self.bce_targets = Variable(self.bce_targets)
        if ptu.gpu_enabled():
            self.bce.cuda()
            self.bce_targets = self.bce_targets.cuda()
        
        self.use_grad_pen = use_grad_pen
        self.grad_pen_weight = grad_pen_weight


    def get_expert_batch(self, batch_size):
        batch = self.expert_replay_buffer.sample_expert_random_batch(batch_size)
        if self.wrap_absorbing:
            if isinstance(batch['observations'], np.ndarray):
                obs = batch['observations']
                assert len(obs.shape) == 2
                batch['observations'] = np.concatenate((obs, np.zeros((obs.shape[0],1))), -1)
                if 'next_observations' in batch:
                    next_obs = batch['next_observations']
                    batch['next_observations'] = np.concatenate((next_obs, np.zeros((next_obs.shape[0],1))), -1)
            else:
                raise NotImplementedError()
        return np_to_pytorch_batch(batch)
    

    def get_policy_batch(self, batch_size):
        batch = self.replay_buffer.random_batch(batch_size)
        return np_to_pytorch_batch(batch)


    def _do_reward_training(self):
        '''
            Train the discriminator
        '''
        self.disc_optimizer.zero_grad()

        expert_batch = self.get_expert_batch(self.disc_optim_batch_size)
        expert_obs = expert_batch['observations']
        expert_actions = expert_batch['actions']

        policy_batch = self.get_policy_batch(self.disc_optim_batch_size)
        policy_obs = policy_batch['observations']
        policy_actions = policy_batch['actions']

        obs = torch.cat([expert_obs, policy_obs], dim=0)
        actions = torch.cat([expert_actions, policy_actions], dim=0)

        disc_logits = self.discriminator(obs, actions)
        disc_preds = (disc_logits > 0).type(torch.FloatTensor)
        disc_loss = self.bce(disc_logits, self.bce_targets)
        accuracy = (disc_preds == self.bce_targets).type(torch.FloatTensor).mean()

        if self.use_grad_pen:
            eps = Variable(torch.rand(self.disc_optim_batch_size, 1))
            if ptu.gpu_enabled(): eps = eps.cuda()
            
            interp_obs = eps*expert_obs + (1-eps)*policy_obs
            interp_obs.detach()
            interp_obs.requires_grad = True
            interp_actions = eps*expert_actions + (1-eps)*policy_actions
            interp_actions.detach()
            interp_actions.requires_grad = True
            gradients = autograd.grad(
                outputs=self.discriminator(interp_obs, interp_actions).sum(),
                inputs=[interp_obs, interp_actions],
                # grad_outputs=torch.ones(exp_specs['batch_size'], 1).cuda(),
                create_graph=True, retain_graph=True, only_inputs=True
            )
            total_grad = torch.cat([gradients[0], gradients[1]], dim=1)
            gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()

            disc_loss = disc_loss + gradient_penalty * self.grad_pen_weight

        disc_loss.backward()
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
            self.rewardf_eval_statistics['Disc Loss'] = np.mean(ptu.get_numpy(disc_loss))
            self.rewardf_eval_statistics['Disc Acc'] = np.mean(ptu.get_numpy(accuracy))


    def _do_policy_training(self):
        policy_batch = self.get_policy_batch(self.policy_optim_batch_size)
        # If you compute log(D) - log(1-D) then you just get the logits
        policy_batch['rewards'] = self.discriminator(policy_batch['observations'], policy_batch['actions'])
        self.policy_optimizer.train_step(policy_batch)


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