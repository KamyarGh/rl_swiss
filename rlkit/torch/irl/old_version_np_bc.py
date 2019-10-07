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


def concat_trajs(trajs):
    new_dict = {}
    for k in trajs[0].keys():
        if isinstance(trajs[0][k], dict):
            new_dict[k] = concat_trajs([t[k] for t in trajs])
        else:
            new_dict[k] = np.concatenate([t[k] for t in trajs], axis=0)
    return new_dict


class NeuralProcessBC(TorchMetaIRLAlgorithm):
    '''
        Meta-BC using a neural process

        assuming the context trajectories all have the same length and flat and everything nice
    '''
    def __init__(
            self,
            env,

            # this is the main policy network that we wrap with
            # PostCondWrapperPolicy for get_exploration policy
            main_policy,

            train_context_expert_replay_buffer,
            train_test_expert_replay_buffer,
            test_context_expert_replay_buffer,
            test_test_expert_replay_buffer,

            np_encoder,

            num_updates_per_epoch=1000,
            num_policy_steps_per_update=0,
            num_full_model_steps_per_update=1,
            num_tasks_used_per_update=4,
            num_context_trajs_for_training=3,
            num_test_trajs_for_training=3,

            # for each task, for each context, infer post, for each post sample, generate some eval trajs
            num_tasks_per_eval=10,
            num_diff_context_per_eval_task=2,
            num_context_trajs_for_eval=3,
            num_eval_trajs_per_post_sample=2,

            num_context_trajs_for_exploration=3,

            policy_lr=1e-3,
            policy_optimizer_class=optim.Adam,

            encoder_lr=1e-3,
            encoder_optimizer_class=optim.Adam,

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
        self.rewardf_eval_statistics = None

        self.policy_optimizer = policy_optimizer_class(
            self.main_policy.parameters(),
            lr=policy_lr,
        )
        self.encoder_optimizer = encoder_optimizer_class(
            self.encoder.parameters(),
            lr=encoder_lr,
        )

        self.num_updates_per_epoch = num_updates_per_epoch
        self.num_policy_steps_per_update = num_policy_steps_per_update
        self.num_full_model_steps_per_update = num_full_model_steps_per_update
        self.num_tasks_used_per_update = num_tasks_used_per_update
        self.num_context_trajs_for_training = num_context_trajs_for_training
        self.num_test_trajs_for_training = num_test_trajs_for_training

        self.num_tasks_per_eval = num_tasks_per_eval
        self.num_diff_context_per_eval_task = num_diff_context_per_eval_task
        self.num_context_trajs_for_eval = num_context_trajs_for_eval
        self.num_eval_trajs_per_post_sample = num_eval_trajs_per_post_sample

        self.num_context_trajs_for_exploration = num_context_trajs_for_exploration


    def get_exploration_policy(self, task_identifier):
        list_of_trajs = self.train_context_expert_replay_buffer.sample_trajs_from_task(
            task_identifier,
            self.num_context_trajs_for_exploration,
        )
        post_dist = self.encoder([list_of_trajs])
        # z = post_dist.sample()
        z = post_dist.mean
        z = z.cpu().data.numpy()[0]
        return PostCondMLPPolicyWrapper(self.main_policy, z)
    

    def get_eval_policy(self, task_identifier, mode='meta_test'):
        if mode == 'meta_train':
            rb = self.train_context_expert_replay_buffer
        else:
            rb = self.test_context_expert_replay_buffer
        list_of_trajs = rb.sample_trajs_from_task(
            task_identifier,
            self.num_context_trajs_for_eval,
        )
        post_dist = self.encoder([list_of_trajs])
        # z = post_dist.sample()
        z = post_dist.mean
        z = z.cpu().data.numpy()[0]
        return PostCondMLPPolicyWrapper(self.main_policy, z)


    def _get_training_batch(self):
        # context batch is a list of list of dicts
        context_batch, task_identifiers_list = self.train_context_expert_replay_buffer.sample_trajs(
            self.num_context_trajs_for_training,
            num_tasks=self.num_tasks_used_per_update,
            keys=['observations', 'actions']
        )

        flat_context_batch = [traj for task_trajs in context_batch for traj in task_trajs]
        context_pred_batch = concat_trajs(flat_context_batch)

        test_batch, _ = self.train_test_expert_replay_buffer.sample_trajs(
            self.num_test_trajs_for_training,
            task_identifiers=task_identifiers_list,
            keys=['observations', 'actions']
        )
        flat_test_batch = [traj for task_trajs in test_batch for traj in task_trajs]
        test_pred_batch = concat_trajs(flat_test_batch)

        # if we want to handle envs with different traj lengths we need to do
        # something smarter with how we repeat z
        traj_len = flat_context_batch[0]['observations'].shape[0]
        assert all(t['observations'].shape[0] == traj_len for t in flat_context_batch), "Not handling different traj lens"
        assert all(t['observations'].shape[0] == traj_len for t in flat_test_batch), "Not handling different traj lens"

        return context_batch, context_pred_batch, test_pred_batch, traj_len
    

    def _compute_training_loss(self, training_encoder=True):
        context_batch, context_pred_batch, test_pred_batch, traj_len = self._get_training_batch()
        
        post_dist = self.encoder(context_batch)
        # z = post_dist.sample() # N_tasks x Dim
        z = post_dist.mean
        if not training_encoder: z = z.detach()
        context_pred_z = z.repeat(1, traj_len * self.num_context_trajs_for_training).view(
            -1,
            z.size(1)
        )
        test_pred_z = z.repeat(1, traj_len * self.num_test_trajs_for_training).view(
            -1,
            z.size(1)
        )
        z_batch = torch.cat([context_pred_z, test_pred_z], dim=0)

        # convert it to a pytorch tensor
        # note that our objective says we should maximize likelihood of
        # BOTH the context_batch and the test_batch
        obs_batch = np.concatenate((context_pred_batch['observations'], test_pred_batch['observations']), axis=0)
        obs_batch = Variable(ptu.from_numpy(obs_batch), requires_grad=False)
        acts_batch = np.concatenate((context_pred_batch['actions'], test_pred_batch['actions']), axis=0)
        acts_batch = Variable(ptu.from_numpy(acts_batch), requires_grad=False)

        # get action predictions
        pred_acts = self.main_policy(torch.cat([obs_batch, z_batch], dim=-1))
        loss = torch.sum((acts_batch - pred_acts)**2, dim=1).mean()

        return loss


    def _do_training(self):
        for i in range(self.num_updates_per_epoch):
            # train just the policy for a certain number of iters
            self.encoder_optimizer.zero_grad()

            for _ in range(self.num_policy_steps_per_update):
                self.policy_optimizer.zero_grad()
                loss = self._compute_training_loss(training_encoder=False)
                loss.backward()
                self.policy_optimizer.step()
            
            for _ in range(self.num_full_model_steps_per_update):
                self.encoder_optimizer.zero_grad()
                self.policy_optimizer.zero_grad()
                loss = self._compute_training_loss(training_encoder=True)
                loss.backward()
                self.encoder_optimizer.step()
                self.policy_optimizer.step()
        
        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
            self.eval_statistics['Regr MSE Loss'] = np.mean(ptu.get_numpy(loss))


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
        return [
            self.encoder,
            self.main_policy
        ]


    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(
            encoder=self.encoder,
            main_policy=self.main_policy
        )
        return snapshot
