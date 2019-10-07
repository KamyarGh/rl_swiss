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

from rlkit.torch.distributions import ReparamMultivariateNormalDiag
from rlkit.data_management.env_replay_buffer import MetaEnvReplayBuffer
from rlkit.torch.sac.policies import PostCondMLPPolicyWrapper
from rlkit.data_management.path_builder import PathBuilder
from rlkit.samplers.util import rollout
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


class PEARL(TorchMetaIRLAlgorithm):
    '''
    My implementation of PEARL, without the KL regularizer
    '''
    def __init__(
            self,
            env,

            policy,
            qf1,
            qf2,
            vf,

            encoder,
            z_dim,

            train_context_expert_replay_buffer=None,
            train_test_expert_replay_buffer=None,
            test_context_expert_replay_buffer=None,
            test_test_expert_replay_buffer=None,

            train_task_params_sampler=None,
            test_task_params_sampler=None,

            num_tasks_used_per_update=5,
            num_context_trajs_for_exploration=4,
            num_context_trajs_for_training=4,
            samples_per_traj=50,
            
            num_tasks_per_eval=10,
            num_diff_context_per_eval_task=2,
            num_eval_trajs_per_post_sample=2,

            context_buffer_size_per_task=1000,

            batch_size_per_task=128,
            
            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,

            discount=0.99,
            soft_target_tau=1e-3,
            reward_scale=1.0,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=1e-3,

            encoder_lr=1e-3,
            optimizer_class=optim.Adam,

            num_update_loops_per_train_call=65,

            max_KL_beta=1.0,

            eval_deterministic=False,
            add_context_rollouts_to_replay_buffer=False,
            **kwargs
    ):
        self.add_context_rollouts_to_replay_buffer = add_context_rollouts_to_replay_buffer
        if kwargs['policy_uses_pixels']: raise NotImplementedError('policy uses pixels')
        if kwargs['wrap_absorbing']: raise NotImplementedError('wrap absorbing')
        # assert self.policy_uses_task_params, 'Doesn\'t make sense to use this otherwise.'
        self.eval_deterministic = eval_deterministic
        
        super().__init__(
            env=env,
            train_task_params_sampler=train_task_params_sampler,
            test_task_params_sampler=test_task_params_sampler,
            
            train_context_expert_replay_buffer=train_context_expert_replay_buffer,
            train_test_expert_replay_buffer=train_test_expert_replay_buffer,
            test_context_expert_replay_buffer=test_context_expert_replay_buffer,
            test_test_expert_replay_buffer=test_test_expert_replay_buffer,
            **kwargs
        )

        self.context_buffer_size_per_task = context_buffer_size_per_task
        self.context_buffer = MetaEnvReplayBuffer(
            self.context_buffer_size_per_task,
            self.training_env,
            policy_uses_pixels=self.policy_uses_pixels,
        )

        self.max_KL_beta = max_KL_beta

        self.main_policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.vf = vf
        self.target_vf = vf.copy()
        self.encoder = encoder

        self.z_dim = z_dim
        self.prior_dist = ReparamMultivariateNormalDiag(
            Variable(ptu.from_numpy(np.zeros((1, self.z_dim))).cuda(), requires_grad=False),
            Variable(ptu.from_numpy(np.zeros((1, self.z_dim))).cuda(), requires_grad=False),
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight

        self.eval_statistics = None

        self.encoder_optimizer = optimizer_class(
            self.encoder.parameters(),
            lr=encoder_lr,
            betas=(0.9, 0.999)
            # betas=(0.0, 0.999)
        )
        self.policy_optimizer = optimizer_class(
            self.main_policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )

        self.num_tasks_used_per_update = num_tasks_used_per_update
        self.num_context_trajs_for_exploration = num_context_trajs_for_exploration
        self.num_context_trajs_for_training = num_context_trajs_for_training
        self.samples_per_traj = samples_per_traj
        
        self.num_tasks_per_eval = num_tasks_per_eval
        self.num_diff_context_per_eval_task = num_diff_context_per_eval_task
        self.num_eval_trajs_per_post_sample = num_eval_trajs_per_post_sample

        self.batch_size_per_task = batch_size_per_task

        self.num_update_loops_per_train_call = num_update_loops_per_train_call


    def pretrain(self):
        print('Generating initial contexts')

        # fill the contexts
        for task_params, obs_task_params in self.train_task_params_sampler:
            print('task')
            n_steps_total = 0
            # print(n_steps_total)
            while n_steps_total < self.context_buffer_size_per_task:
                # print('------')
                # print(n_steps_total)
                # print(self.context_buffer_size_per_task)
                # print(self.max_path_length)

                first_obs = self.training_env.reset(task_params=task_params, obs_task_params=obs_task_params)
                task_id = self.training_env.task_identifier

                z = self.prior_dist.sample()
                z = z.cpu().data.numpy()[0]
                post_cond_policy = PostCondMLPPolicyWrapper(self.main_policy, z)

                new_path = rollout(
                    self.training_env,
                    post_cond_policy,
                    max_path_length=min(self.max_path_length+1, self.context_buffer_size_per_task - n_steps_total+1),
                    do_not_reset=True,
                    first_obs=first_obs
                )
                # print(len(new_path['observations']))
                n_steps_total += len(new_path['observations'])

                if self.add_context_rollouts_to_replay_buffer:
                    self.replay_buffer.add_path(new_path, task_id)
                self.context_buffer.add_path(new_path, task_id)

        print('Generating initial replay buffer rollouts')
        super().pretrain()


    def _compute_KL_loss(self, post_dist):
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


    def get_exploration_policy(self, task_identifier):
        list_of_trajs = self.context_buffer.sample_trajs_from_task(
            task_identifier,
            self.num_context_trajs_for_exploration,
            samples_per_traj=self.samples_per_traj
        )
        mask = None

        enc_to_use = self.encoder
        mode = enc_to_use.training
        enc_to_use.eval()
        post_dist = enc_to_use([list_of_trajs], mask)
        enc_to_use.train(mode)

        z = post_dist.sample()
        z = z.cpu().data.numpy()[0]
        return PostCondMLPPolicyWrapper(self.main_policy, z)
    

    def get_eval_policy(self, task_identifier, mode='meta_test'):
        if task_identifier not in self.context_buffer.task_replay_buffers:
            # generate some rollouts with prior policy
            eval_context_buffer = MetaEnvReplayBuffer(
                self.context_buffer_size_per_task,
                self.training_env,
                policy_uses_pixels=self.policy_uses_pixels,
            )

            n_steps_total = 0
            steps_needed = self.num_context_trajs_for_exploration * self.max_path_length
            task_params = self.training_env.task_id_to_task_params(task_identifier)
            obs_task_params = self.training_env.task_id_to_obs_task_params(task_identifier)
            while n_steps_total < steps_needed:
                first_obs = self.training_env.reset(task_params=task_params, obs_task_params=obs_task_params)
                task_id = self.training_env.task_identifier

                z = self.prior_dist.sample()
                z = z.cpu().data.numpy()[0]
                post_cond_policy = PostCondMLPPolicyWrapper(self.main_policy, z)

                new_path = rollout(
                    self.training_env,
                    post_cond_policy,
                    max_path_length=min(self.max_path_length+1, steps_needed - n_steps_total+1),
                    do_not_reset=True,
                    first_obs=first_obs
                )
                n_steps_total += len(new_path['observations'])
                eval_context_buffer.add_path(new_path, task_id)
            
            list_of_trajs = eval_context_buffer.sample_trajs_from_task(
                task_identifier,
                self.num_context_trajs_for_exploration,
                samples_per_traj=self.samples_per_traj
            )
            mask = None
        else:
            list_of_trajs = self.context_buffer.sample_trajs_from_task(
                task_identifier,
                self.num_context_trajs_for_exploration,
            )
            mask = None

        enc_to_use = self.encoder
        mode = enc_to_use.training
        enc_to_use.eval()
        post_dist = enc_to_use([list_of_trajs], mask)
        enc_to_use.train(mode)

        z = post_dist.sample()
        z = z.cpu().data.numpy()[0]
        return PostCondMLPPolicyWrapper(self.main_policy, z)
    

    def _get_training_batch(self):
        # context_batch, task_identifiers_list = self.context_buffer.sample_random_batch(
        #     self.num_context_transitions_per_task,
        #     num_tasks=self.num_tasks_used_per_update
        # )
        context_batch, task_identifiers_list = self.context_buffer.sample_trajs(
            self.num_context_trajs_for_training,
            num_tasks=self.num_tasks_used_per_update,
            samples_per_traj=self.samples_per_traj
        )
        update_batch, _ = self.replay_buffer.sample_random_batch(
            self.batch_size_per_task,
            task_identifiers_list=task_identifiers_list
        )
        obs = np.concatenate([d['observations'] for d in update_batch], axis=0) # (N_tasks * batch_size) x Dim
        acts = np.concatenate([d['actions'] for d in update_batch], axis=0) # (N_tasks * batch_size) x Dim
        terminals = np.concatenate([d['terminals'] for d in update_batch], axis=0) # (N_tasks * batch_size) x Dim
        next_obs = np.concatenate([d['next_observations'] for d in update_batch], axis=0) # (N_tasks * batch_size) x Dim
        rewards = np.concatenate([d['rewards'] for d in update_batch], axis=0) # (N_tasks * batch_size) x Dim
        rewards *= self.reward_scale
        update_batch = dict(
            observations=obs,
            actions=acts,
            terminals=terminals,
            next_observations=next_obs,
            rewards=rewards
        )

        return context_batch, update_batch


    def _do_training(self, epoch):
        # sample a mini-batch of tasks
        task_batch = self.train_task_params_sampler.sample_unique(self.num_tasks_used_per_update)

        # reset the context buffer for these tasks
        for task_params, obs_task_params in task_batch:
            self.training_env.reset(task_params=task_params, obs_task_params=obs_task_params)
            task_id = self.training_env.task_identifier
            self.context_buffer.task_replay_buffers[task_id]._size = 0
            self.context_buffer.task_replay_buffers[task_id]._top = 0
        
        # generate contexts for each task in the minibatch
        for task_params, obs_task_params in task_batch:
            n_steps_total = 0
            while n_steps_total < self.context_buffer_size_per_task:
                first_obs = self.training_env.reset(task_params=task_params, obs_task_params=obs_task_params)
                task_id = self.training_env.task_identifier

                z = self.prior_dist.sample()
                z = z.cpu().data.numpy()[0]
                post_cond_policy = PostCondMLPPolicyWrapper(self.main_policy, z)

                new_path = rollout(
                    self.training_env,
                    post_cond_policy,
                    max_path_length=min(self.max_path_length+1, self.context_buffer_size_per_task - n_steps_total+1),
                    do_not_reset=True,
                    first_obs=first_obs
                )
                n_steps_total += len(new_path['observations'])

                if self.add_context_rollouts_to_replay_buffer:
                    self.replay_buffer.add_path(new_path, task_id)
                self.context_buffer.add_path(new_path, task_id)

        # # generate rollouts using the posteriors
        # for task_params, obs_task_params in task_batch:
        #     n_steps_total = 0
        #     while n_steps_total < self.num_posterior_steps_per_task:
        #         first_obs = self.training_env.reset(task_params=task_params, obs_task_params=obs_task_params)
        #         task_id = self.training_env.task_identifier

        #         post_cond_policy = self.get_posterior_policy(task_id)
        #         new_path = rollout(
        #             self.training_env,
        #             post_cond_policy,
        #             max_path_length=min(self.max_path_length, self.num_context_steps - self.max_path_length),
        #             do_not_reset=True,
        #             first_obs=first_obs
        #         )
        #         n_steps_total += len(new_path['observations'])

        #         self.replay_buffer.add_path(new_path, task_id)

        # now do some training
        for t in range(self.num_update_loops_per_train_call):
            self._do_update(epoch)


    def _do_update(self, epoch):
        self.encoder.training

        self.encoder_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        self.vf_optimizer.zero_grad()

        context_batch, update_batch = self._get_training_batch()
        update_batch = np_to_pytorch_batch(update_batch)

        enc_to_use = self.encoder
        post_dist = enc_to_use(context_batch)
        z = post_dist.sample() # N_tasks x Dim
        # repeat z to have the right size
        z = z.repeat(1, self.batch_size_per_task).view(
            self.num_tasks_used_per_update * self.batch_size_per_task,
            -1
        )

        update_batch['observations'] = torch.cat([update_batch['observations'], z], dim=1)
        update_batch['next_observations'] = torch.cat([update_batch['next_observations'], z], dim=1)

        # compute all the SAC losses
        rewards = update_batch['rewards']
        terminals = update_batch['terminals']
        obs = update_batch['observations']
        actions = update_batch['actions']
        next_obs = update_batch['next_observations']

        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        v_pred = self.vf(obs.detach())
        # Make sure policy accounts for squashing functions like tanh correctly!
        policy_outputs = self.main_policy(obs.detach(), return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        """
        QF Loss
        """
        target_v_values = self.target_vf(next_obs.detach())
        q_target = rewards + (1. - terminals) * self.discount * target_v_values
        qf1_loss = 0.5 * torch.mean((q1_pred - q_target.detach())**2)
        qf2_loss = 0.5 * torch.mean((q2_pred - q_target.detach())**2)

        """
        VF Loss
        """
        q1_new_acts = self.qf1(obs, new_actions)
        q2_new_acts = self.qf2(obs, new_actions)
        q_new_actions = torch.min(q1_new_acts, q2_new_acts)
        v_target = q_new_actions - log_pi
        vf_loss = 0.5 * torch.mean((v_pred - v_target.detach())**2)

        """
        Policy Loss
        """
        policy_loss = torch.mean(log_pi - q_new_actions)
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        policy_reg_loss = mean_reg_loss + std_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        # compute the KL loss
        KL_loss = self.max_KL_beta * self._compute_KL_loss(post_dist)

        # do update step
        total_loss = policy_loss + qf1_loss + qf2_loss + vf_loss + KL_loss
        total_loss.backward()

        self.encoder_optimizer.step()
        self.policy_optimizer.step()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.vf_optimizer.step()

        # update the target value function
        self._update_target_network()

        # update eval stats
        # self.eval_statistics['Thing'] = np.mean(ptu.get_numpy())
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
            self.eval_statistics['KL_beta'] = self.max_KL_beta
            self.eval_statistics['KL_loss'] = np.mean(ptu.get_numpy(KL_loss))
            self.eval_statistics['avg_post_z_mean'] = np.mean(ptu.get_numpy(post_dist.mean))
            self.eval_statistics['avg_post_z_cov'] = np.mean(ptu.get_numpy(post_dist.cov))
            # self.eval_statistics['Test'] = 0.0
    

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)


    def evaluate(self, epoch):
        super().evaluate(epoch)


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
                        # if self.no_terminal:
                        #     terminal = False
                        
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
        networks_list = [
            self.main_policy,
            self.qf1,
            self.qf2,
            self.vf,
            self.target_vf,
            self.encoder
        ]
        return networks_list

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(
            policy=self.main_policy,
            qf1=self.qf1,
            qf2=self.qf2,
            vf=self.vf,
            encoder=self.encoder
        )
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
