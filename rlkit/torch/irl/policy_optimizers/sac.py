from collections import OrderedDict

import numpy as np
import torch.optim as optim
from torch import nn as nn
import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm
from rlkit.torch.sac.policies import MakeDeterministic


class NewSoftActorCritic():
    def __init__(
            self,
            policy,
            qf1,
            qf2,
            vf,

            reward_scale=1.0,
            discount=0.99,
            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,

            beta_1=0.25,

            soft_target_tau=1e-2,
            eval_deterministic=True,

            use_policy_as_ema_policy=False,
            soft_ema_policy_exp=0.005,

            wrap_absorbing=False
    ):
        self.use_policy_as_ema_policy = use_policy_as_ema_policy
        self.soft_ema_policy_exp = soft_ema_policy_exp
        self.policy = policy
        if use_policy_as_ema_policy:
            self.training_policy = policy.copy()
        else:
            self.training_policy = self.policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.vf = vf
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight

        self.target_vf = vf.copy()
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.eval_statistics = None
        self.reward_scale = reward_scale
        self.discount = discount

        print('\n\nPOLICY OPTIMIZER BETA-1 IS %.2f\n\n' % beta_1)
        
        self.policy_optimizer = optimizer_class(
            self.training_policy.parameters(),
            lr=policy_lr,
            betas=(beta_1, 0.999)
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
            betas=(beta_1, 0.999)
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
            betas=(beta_1, 0.999)
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
            betas=(beta_1, 0.999)
        )

        self.wrap_absorbing = wrap_absorbing


    def train_step(self, batch, compute_grad_pol_loss_wrt_var=False, var_for_grad=None):
        rewards = batch['rewards'] * self.reward_scale
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        if self.wrap_absorbing:
            absorbing = batch['absorbing']

        if self.wrap_absorbing:
            obs_with_absorbing = torch.cat([obs, absorbing[:,0:1]], dim=-1)
            next_obs_with_absorbing = torch.cat([next_obs, absorbing[:,1:2]], dim=-1)
        else:
            obs_with_absorbing = obs
            next_obs_with_absorbing = next_obs

        q1_pred = self.qf1(obs_with_absorbing, actions)
        q2_pred = self.qf2(obs_with_absorbing, actions)
        v_pred = self.vf(obs_with_absorbing)
        # Make sure policy accounts for squashing functions like tanh correctly!
        policy_outputs = self.training_policy(obs, return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        """
        QF Loss
        """
        target_v_values = self.target_vf(next_obs_with_absorbing)
        if self.wrap_absorbing:
            q_target = rewards + self.discount * target_v_values
        else:
            q_target = rewards + (1. - terminals) * self.discount * target_v_values
        qf1_loss = 0.5 * torch.mean((q1_pred - q_target.detach())**2)
        qf2_loss = 0.5 * torch.mean((q2_pred - q_target.detach())**2)

        """
        VF Loss
        """
        q1_new_acts = self.qf1(obs_with_absorbing, new_actions)
        q2_new_acts = self.qf2(obs_with_absorbing, new_actions)
        q_new_actions = torch.min(q1_new_acts, q2_new_acts)
        v_target = q_new_actions - log_pi
        vf_loss = 0.5 * torch.mean((v_pred - v_target.detach())**2)

        """
        Policy Loss
        """
        policy_loss = torch.mean(log_pi - q_new_actions)

        if compute_grad_pol_loss_wrt_var:
            grad_wrt_var = autograd.grad(
                policy_loss,
                [var_for_grad],
                retain_graph=True
            )[0]

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        # pre_tanh_value = policy_outputs[-1]
        # pre_activation_reg_loss = self.policy_pre_activation_weight * (
        #     (pre_tanh_value**2).sum(dim=1).mean()
        # )
        # policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_reg_loss = mean_reg_loss + std_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        if self.use_policy_as_ema_policy:
            ptu.soft_update_from_to(self.training_policy, self.policy, self.soft_ema_policy_exp)

        self._update_target_network()

        """
        Save some statistics for eval
        """
        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
        
        if compute_grad_pol_loss_wrt_var:
            return grad_wrt_var

    @property
    def networks(self):
        networks_list = [
            self.policy,
            self.training_policy,
            self.qf1,
            self.qf2,
            self.vf,
            self.target_vf,
        ]
        if self.use_policy_as_ema_policy: networks_list += [self.training_policy]
        return networks_list

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def get_snapshot(self):
        snapshot = dict(
            qf1=self.qf1,
            qf2=self.qf2,
            policy=self.policy,
            training_policy=self.training_policy,
            vf=self.vf,
            target_vf=self.target_vf,
        )
        return snapshot
