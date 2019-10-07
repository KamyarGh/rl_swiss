from collections import OrderedDict
from random import sample

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
from rlkit.samplers.util import rollout
from rlkit.torch.sac.policies import PostCondMLPPolicyWrapper
from rlkit.data_management.path_builder import PathBuilder

from gym.spaces import Dict


class NeuralProcessAIRL(TorchMetaIRLAlgorithm):
    '''
        Meta-AIRL using a neural process

        assuming the context trajectories all have the same length and flat and everything nice
    '''
    def __init__(
            self,
            env,

            # this is the main policy network that we wrap with
            # PostCondWrapperPolicy for get_exploration policy
            main_policy,
            disc,

            train_context_expert_replay_buffer,
            train_test_expert_replay_buffer,
            test_context_expert_replay_buffer,
            test_test_expert_replay_buffer,

            np_encoder,
            test_task_params_sampler,

            policy_optimizer, # the RL algorith that updates the policy

            num_policy_updates_per_epoch=1000,
            num_disc_updates_per_epoch=1000,
            num_tasks_used_per_update=4,
            num_context_trajs_for_training=3,
            test_batch_size_per_task=5,

            # for each task, for each context, infer post, for each post sample, generate some eval trajs
            num_tasks_per_eval=10,
            num_diff_context_per_eval_task=2,
            num_context_trajs_for_eval=3,
            num_eval_trajs_per_post_sample=2,

            num_context_trajs_for_exploration=3,

            encoder_lr=1e-3,
            encoder_optimizer_class=optim.Adam,

            disc_lr=1e-3,
            disc_optimizer_class=optim.Adam,

            use_grad_pen=True,
            grad_pen_weight=10,

            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            train_context_expert_replay_buffer=train_context_expert_replay_buffer,
            train_test_expert_replay_buffer=train_test_expert_replay_buffer,
            test_context_expert_replay_buffer=test_context_expert_replay_buffer,
            test_test_expert_replay_buffer=test_test_expert_replay_buffer,
            **kwargs
        )

        self.main_policy = main_policy
        self.encoder = np_encoder
        self.disc = disc
        self.rewardf_eval_statistics = None
        self.test_task_params_sampler = test_task_params_sampler
        

        self.policy_optimizer = policy_optimizer
        self.encoder_optimizer = encoder_optimizer_class(
            self.encoder.parameters(),
            lr=encoder_lr,
        )
        self.disc_optimizer = disc_optimizer_class(
            self.disc.parameters(),
            lr=disc_lr,
        )

        self.num_policy_updates_per_epoch = num_policy_updates_per_epoch
        self.num_disc_updates_per_epoch = num_disc_updates_per_epoch
        self.num_tasks_used_per_update = num_tasks_used_per_update
        self.num_context_trajs_for_training = num_context_trajs_for_training
        self.test_batch_size_per_task = test_batch_size_per_task

        self.num_tasks_per_eval = num_tasks_per_eval
        self.num_diff_context_per_eval_task = num_diff_context_per_eval_task
        self.num_context_trajs_for_eval = num_context_trajs_for_eval
        self.num_eval_trajs_per_post_sample = num_eval_trajs_per_post_sample

        self.num_context_trajs_for_exploration = num_context_trajs_for_exploration

        # things we need for computing the discriminator objective
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_targets = torch.cat(
            [
                torch.ones(self.test_batch_size_per_task * self.num_tasks_used_per_update, 1),
                torch.zeros(self.test_batch_size_per_task * self.num_tasks_used_per_update, 1)
            ],
            dim=0
        )
        self.bce_targets = Variable(self.bce_targets)
        if ptu.gpu_enabled():
            self.bce.cuda()
            self.bce_targets = self.bce_targets.cuda()
        
        self.use_grad_pen = use_grad_pen
        self.grad_pen_weight = grad_pen_weight


    def get_exploration_policy(self, task_identifier):
        list_of_trajs = self.train_context_expert_replay_buffer.sample_trajs_from_task(
            task_identifier,
            self.num_context_trajs_for_exploration,
        )
        post_dist = self.encoder([list_of_trajs])
        # z = post_dist.sample()
        z = post_dist.mean
        z = z.cpu().data.numpy()[0]
        # z = np.array([task_identifier[0], task_identifier[1]])
        return PostCondMLPPolicyWrapper(self.main_policy, z)
    

    def get_eval_policy(self, task_identifier):
        list_of_trajs = self.test_context_expert_replay_buffer.sample_trajs_from_task(
            task_identifier,
            self.num_context_trajs_for_eval,
        )
        post_dist = self.encoder([list_of_trajs])
        # z = post_dist.sample()
        z = post_dist.mean
        z = z.cpu().data.numpy()[0]
        # z = np.array([task_identifier[0], task_identifier[1]])
        return PostCondMLPPolicyWrapper(self.main_policy, z)


    def _do_training(self):
        '''
        '''
        # train the discriminator (and the encoder)
        for i in range(self.num_disc_updates_per_epoch):
            self.encoder_optimizer.zero_grad()
            self.disc_optimizer.zero_grad()

            # context batch is a list of list of dicts
            context_batch, task_identifiers_list = self.train_context_expert_replay_buffer.sample_trajs(
                self.num_context_trajs_for_training,
                num_tasks=self.num_tasks_used_per_update
            )
            post_dist = self.encoder(context_batch)
            # z = post_dist.sample() # N_tasks x Dim
            z = post_dist.mean
            # z = [[t[0], t[1]] for t in task_identifiers_list]
            # z = np.array(task_identifiers_list)
            # z = Variable(ptu.from_numpy(z), requires_grad=True)

            # z_reg_loss = 0.0001 * z.norm(2, dim=1).mean()
            z_reg_loss = 0.0

            # get the test batch for the tasks from expert buffer
            exp_test_batch, _ = self.train_test_expert_replay_buffer.sample_random_batch(
                self.test_batch_size_per_task,
                task_identifiers_list=task_identifiers_list
            )
            # test_batch is a list of dicts: each dict is a random batch from that task
            # convert it to a pytorch tensor
            exp_obs = np.concatenate([d['observations'] for d in exp_test_batch], axis=0) # (N_tasks * batch_size) x Dim
            exp_obs = Variable(ptu.from_numpy(exp_obs), requires_grad=False)
            exp_acts = np.concatenate([d['actions'] for d in exp_test_batch], axis=0) # (N_tasks * batch_size) x Dim
            exp_acts = Variable(ptu.from_numpy(exp_acts), requires_grad=False)

            # get the test batch for the tasks from policy buffer
            policy_test_batch, _ = self.replay_buffer.sample_random_batch(
                self.test_batch_size_per_task,
                task_identifiers_list=task_identifiers_list
            )
            # test_batch is a list of dicts: each dict is a random batch from that task
            # convert it to a pytorch tensor
            policy_obs = np.concatenate([d['observations'] for d in policy_test_batch], axis=0) # (N_tasks * batch_size) x Dim
            policy_obs = Variable(ptu.from_numpy(policy_obs), requires_grad=False)
            policy_acts = np.concatenate([d['actions'] for d in policy_test_batch], axis=0) # (N_tasks * batch_size) x Dim
            policy_acts = Variable(ptu.from_numpy(policy_acts), requires_grad=False)
            # policy_acts.zero_()
            # policy_obs.zero_()

            # repeat z to have the right size
            z = z.repeat(1, self.test_batch_size_per_task).view(
                z.size(0) * self.test_batch_size_per_task,
                -1
            )
            # z.zero_()

            # make the batches
            exp_obs = torch.cat([exp_obs, z], dim=1)
            policy_obs = torch.cat([policy_obs, z], dim=1)
            obs_batch = torch.cat([exp_obs, policy_obs], dim=0)
            act_batch = torch.cat([exp_acts, policy_acts], dim=0)

            # compute the loss for the discriminator
            disc_logits = self.disc(obs_batch, act_batch)
            disc_preds = (disc_logits > 0).type(torch.FloatTensor)
            # disc_percent_policy_preds_one = disc_preds[z.size(0):].mean()
            disc_loss = self.bce(disc_logits, self.bce_targets)
            accuracy = (disc_preds == self.bce_targets).type(torch.FloatTensor).mean()

            if self.use_grad_pen:
                eps = Variable(torch.rand(exp_obs.size(0), 1), requires_grad=True)
                if ptu.gpu_enabled(): eps = eps.cuda()
                
                interp_obs = eps*exp_obs + (1-eps)*policy_obs
                interp_obs.detach()
                # interp_obs.requires_grad = True
                interp_actions = eps*exp_acts + (1-eps)*policy_acts
                interp_actions.detach()
                # interp_actions.requires_grad = True
                gradients = autograd.grad(
                    outputs=self.disc(interp_obs, interp_actions).sum(),
                    inputs=[interp_obs, interp_actions],
                    # grad_outputs=torch.ones(exp_specs['batch_size'], 1).cuda(),
                    create_graph=True, retain_graph=True, only_inputs=True
                )
                total_grad = torch.cat([gradients[0], gradients[1]], dim=1)
                gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()

                disc_loss = disc_loss + gradient_penalty * self.grad_pen_weight

            total_loss = z_reg_loss + disc_loss

            total_loss.backward()
            # disc_loss.backward()
            self.disc_optimizer.step()
            self.encoder_optimizer.step()


        # train the policy
        for i in range(self.num_policy_updates_per_epoch):
            # context batch is a list of list of dicts
            context_batch, task_identifiers_list = self.train_context_expert_replay_buffer.sample_trajs(
                self.num_context_trajs_for_training,
                num_tasks=self.num_tasks_used_per_update
            )
            post_dist = self.encoder(context_batch)
            # z = post_dist.sample() # N_tasks x Dim
            z = post_dist.mean
            # z = np.array(task_identifiers_list)
            # z = Variable(ptu.from_numpy(z), requires_grad=True)

            policy_test_batch, _ = self.replay_buffer.sample_random_batch(
                self.test_batch_size_per_task,
                task_identifiers_list=task_identifiers_list
            )
            # test_batch is a list of dicts: each dict is a random batch from that task
            # convert it to a pytorch tensor
            policy_obs = np.concatenate([d['observations'] for d in policy_test_batch], axis=0) # (N_tasks * batch_size) x Dim
            policy_obs = Variable(ptu.from_numpy(policy_obs), requires_grad=False)
            policy_acts = np.concatenate([d['actions'] for d in policy_test_batch], axis=0) # (N_tasks * batch_size) x Dim
            policy_acts = Variable(ptu.from_numpy(policy_acts), requires_grad=False)
            policy_terminals = np.concatenate([d['terminals'] for d in policy_test_batch], axis=0) # (N_tasks * batch_size) x Dim
            policy_terminals = Variable(ptu.from_numpy(policy_terminals), requires_grad=False)
            policy_next_obs = np.concatenate([d['next_observations'] for d in policy_test_batch], axis=0) # (N_tasks * batch_size) x Dim
            policy_next_obs = Variable(ptu.from_numpy(policy_next_obs), requires_grad=False)

            # !!!!! For now this the workaround for the batchnorm problem !!!!!
            # get the test batch for the tasks from expert buffer
            exp_test_batch, _ = self.train_test_expert_replay_buffer.sample_random_batch(
                self.test_batch_size_per_task,
                task_identifiers_list=task_identifiers_list
            )
            # test_batch is a list of dicts: each dict is a random batch from that task
            # convert it to a pytorch tensor
            exp_obs = np.concatenate([d['observations'] for d in exp_test_batch], axis=0) # (N_tasks * batch_size) x Dim
            exp_obs = Variable(ptu.from_numpy(exp_obs), requires_grad=False)
            exp_acts = np.concatenate([d['actions'] for d in exp_test_batch], axis=0) # (N_tasks * batch_size) x Dim
            exp_acts = Variable(ptu.from_numpy(exp_acts), requires_grad=False)

            # repeat z to have the right size
            z = z.repeat(1, self.test_batch_size_per_task).view(
                z.size(0) * self.test_batch_size_per_task,
                -1
            ).detach()

            # now augment the obs with the latent sample z
            policy_obs = torch.cat([policy_obs, z], dim=1)
            policy_next_obs = torch.cat([policy_next_obs, z], dim=1)
            exp_obs = torch.cat([exp_obs, z], dim=1)
            obs_batch = torch.cat([exp_obs, policy_obs], dim=0)
            act_batch = torch.cat([exp_acts, policy_acts], dim=0)

            # compute the rewards
            # If you compute log(D) - log(1-D) then you just get the logits
            policy_rewards = self.disc(obs_batch, act_batch)[exp_obs.size(0):].detach()
            # rew_more_than_zero = (rewards > 0).type(torch.FloatTensor).mean()
            # print(rew_more_than_zero.data[0])


            # do a policy update (the zeroing of grads etc. should be handled internally)
            batch = dict(
                observations=policy_obs,
                actions=policy_acts,
                rewards=policy_rewards,
                terminals=policy_terminals,
                next_observations=policy_next_obs
            )
            self.policy_optimizer.train_step(batch)

        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
            self.eval_statistics['Disc Loss'] = np.mean(ptu.get_numpy(disc_loss))
            self.eval_statistics['Disc Acc'] = np.mean(ptu.get_numpy(accuracy))
            # self.eval_statistics['Disc Percent Policy Preds 1'] = np.mean(ptu.get_numpy(disc_percent_policy_preds_one))
            self.eval_statistics['Disc Rewards Mean'] = np.mean(ptu.get_numpy(policy_rewards))
            self.eval_statistics['Disc Rewards Std'] = np.std(ptu.get_numpy(policy_rewards))
            self.eval_statistics['Disc Rewards Max'] = np.max(ptu.get_numpy(policy_rewards))
            self.eval_statistics['Disc Rewards Min'] = np.min(ptu.get_numpy(policy_rewards))
            # self.eval_statistics['Disc Rewards GT Zero'] = np.mean(ptu.get_numpy(rew_more_than_zero))

            z_norm = z.norm(2, dim=1).mean()
            self.eval_statistics['Z Norm'] = np.mean(ptu.get_numpy(z_norm))

            if self.policy_optimizer.eval_statistics is not None:
                self.eval_statistics.update(self.policy_optimizer.eval_statistics)


    def evaluate(self, epoch):
        super().evaluate(epoch)
        self.policy_optimizer.eval_statistics = None


    def obtain_eval_samples(self, epoch):
        self.training_mode(False)

        params_samples = [self.test_task_params_sampler.sample() for _ in range(self.num_tasks_per_eval)]
        all_eval_tasks_paths = []
        for task_params, obs_task_params in params_samples:
            cur_eval_task_paths = []
            self.env.reset(task_params=task_params, obs_task_params=obs_task_params)
            task_identifier = self.env.task_identifier

            for _ in range(self.num_diff_context_per_eval_task):
                eval_policy = self.get_eval_policy(task_identifier)

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
        return [
            self.encoder,
            self.main_policy
        ] + self.policy_optimizer.networks


    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(
            encoder=self.encoder,
            main_policy=self.main_policy
        )
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
