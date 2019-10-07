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
from rlkit.torch.torch_meta_irl_algorithm import np_to_pytorch_batch
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

            policy_optimizer, # the RL algorith that updates the policy

            num_disc_updates_per_epoch=160,
            num_policy_updates_per_epoch=80,
            num_tasks_used_per_update=5,
            num_context_trajs_for_training=3,
            num_test_trajs_for_training=3,
            policy_batch_size_per_task=256,

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
        self.num_test_trajs_for_training = num_test_trajs_for_training
        self.policy_batch_size_per_task = policy_batch_size_per_task

        self.num_tasks_per_eval = num_tasks_per_eval
        self.num_diff_context_per_eval_task = num_diff_context_per_eval_task
        self.num_context_trajs_for_eval = num_context_trajs_for_eval
        self.num_eval_trajs_per_post_sample = num_eval_trajs_per_post_sample

        self.num_context_trajs_for_exploration = num_context_trajs_for_exploration

        # things we need for computing the discriminator objective
        self.bce = nn.BCEWithLogitsLoss()
        total_samples = self.max_path_length * self.num_tasks_used_per_update * (self.num_context_trajs_for_training + self.num_test_trajs_for_training)
        self.bce_targets = torch.cat(
            [
                torch.ones(total_samples, 1),
                torch.zeros(total_samples, 1)
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
    

    def _get_disc_training_batch(self):
        # k = list(self.replay_buffer.task_replay_buffers.keys())[0]
        # print('\nReplay Buffer')
        # print(len(self.replay_buffer.task_replay_buffers[k]._traj_endpoints))
        # print('\nTrain Context')
        # print(len(self.train_context_expert_replay_buffer.task_replay_buffers[k]._traj_endpoints))
        # print('\nTest Context')
        # print(len(self.train_test_expert_replay_buffer.task_replay_buffers[k]._traj_endpoints))

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

        # get the test batch for the tasks from policy buffer
        policy_test_batch_0, _ = self.replay_buffer.sample_trajs(
            self.num_context_trajs_for_training,
            task_identifiers=task_identifiers_list,
            keys=['observations', 'actions']
        )
        flat_policy_batch_0 = [traj for task_trajs in policy_test_batch_0 for traj in task_trajs]
        policy_test_pred_batch_0 = concat_trajs(flat_policy_batch_0)

        policy_test_batch_1, _ = self.replay_buffer.sample_trajs(
            self.num_test_trajs_for_training,
            task_identifiers=task_identifiers_list,
            keys=['observations', 'actions']
        )
        flat_policy_batch_1 = [traj for task_trajs in policy_test_batch_1 for traj in task_trajs]
        policy_test_pred_batch_1 = concat_trajs(flat_policy_batch_1)

        policy_test_pred_batch = {
            'observations': np.concatenate((policy_test_pred_batch_0['observations'], policy_test_pred_batch_1['observations']), axis=0),
            'actions': np.concatenate((policy_test_pred_batch_0['actions'], policy_test_pred_batch_1['actions']), axis=0)
        }

        # if we want to handle envs with different traj lengths we need to do
        # something smarter with how we repeat z
        traj_len = flat_context_batch[0]['observations'].shape[0]
        assert all(t['observations'].shape[0] == traj_len for t in flat_context_batch), "Not handling different traj lens"
        assert all(t['observations'].shape[0] == traj_len for t in flat_test_batch), "Not handling different traj lens"
        assert all(t['observations'].shape[0] == traj_len for t in flat_policy_batch_0), "Not handling different traj lens"
        assert all(t['observations'].shape[0] == traj_len for t in flat_policy_batch_1), "Not handling different traj lens"

        return context_batch, context_pred_batch, test_pred_batch, policy_test_pred_batch, traj_len
    

    def _get_policy_training_batch(self):
        # context batch is a list of list of dicts
        context_batch, task_identifiers_list = self.train_context_expert_replay_buffer.sample_trajs(
            self.num_context_trajs_for_training,
            num_tasks=self.num_tasks_used_per_update,
            keys=['observations', 'actions']
        )
        
        # get the test batch for the tasks from policy buffer
        policy_batch, _ = self.replay_buffer.sample_random_batch(
            self.policy_batch_size_per_task,
            task_identifiers_list=task_identifiers_list
        )
        policy_obs = np.concatenate([d['observations'] for d in policy_batch], axis=0) # (N_tasks * batch_size) x Dim
        policy_acts = np.concatenate([d['actions'] for d in policy_batch], axis=0) # (N_tasks * batch_size) x Dim
        policy_terminals = np.concatenate([d['terminals'] for d in policy_batch], axis=0) # (N_tasks * batch_size) x Dim
        policy_next_obs = np.concatenate([d['next_observations'] for d in policy_batch], axis=0) # (N_tasks * batch_size) x Dim
        policy_batch = dict(
            observations=policy_obs,
            actions=policy_acts,
            terminals=policy_terminals,
            next_observations=policy_next_obs
        )

        return context_batch, policy_batch


    def _do_training(self):
        '''
        '''
        # train the discriminator (and the encoder)
        # print('$$$$$$$$$')
        # print(self.num_disc_updates_per_epoch)
        for i in range(self.num_disc_updates_per_epoch):
            self.encoder_optimizer.zero_grad()
            self.disc_optimizer.zero_grad()

            context_batch, context_pred_batch, test_pred_batch, policy_test_pred_batch, traj_len = self._get_disc_training_batch()

            # convert it to a pytorch tensor
            # note that our objective says we should maximize likelihood of
            # BOTH the context_batch and the test_batch
            exp_obs_batch = np.concatenate((context_pred_batch['observations'], test_pred_batch['observations']), axis=0)
            exp_obs_batch = Variable(ptu.from_numpy(exp_obs_batch), requires_grad=False)
            exp_acts_batch = np.concatenate((context_pred_batch['actions'], test_pred_batch['actions']), axis=0)
            exp_acts_batch = Variable(ptu.from_numpy(exp_acts_batch), requires_grad=False)

            policy_obs_batch = Variable(ptu.from_numpy(policy_test_pred_batch['observations']), requires_grad=False)
            policy_acts_batch = Variable(ptu.from_numpy(policy_test_pred_batch['actions']), requires_grad=False)

            post_dist = self.encoder(context_batch)
            # z = post_dist.sample() # N_tasks x Dim
            z = post_dist.mean

            # z_reg_loss = 0.0001 * z.norm(2, dim=1).mean()
            z_reg_loss = 0.0

            # make z's for expert samples
            context_pred_z = z.repeat(1, traj_len * self.num_context_trajs_for_training).view(
                -1,
                z.size(1)
            )
            test_pred_z = z.repeat(1, traj_len * self.num_test_trajs_for_training).view(
                -1,
                z.size(1)
            )
            z_batch = torch.cat([context_pred_z, test_pred_z], dim=0)
            positive_obs_batch = torch.cat([exp_obs_batch, z_batch], dim=1)
            positive_acts_batch = exp_acts_batch

            # make z's for policy samples
            z_policy = z_batch
            negative_obs_batch = torch.cat([policy_obs_batch, z_policy], dim=1)
            negative_acts_batch = policy_acts_batch

            # compute the loss for the discriminator
            obs_batch = torch.cat([positive_obs_batch, negative_obs_batch], dim=0)
            acts_batch = torch.cat([positive_acts_batch, negative_acts_batch], dim=0)
            disc_logits = self.disc(obs_batch, acts_batch)
            disc_preds = (disc_logits > 0).type(torch.FloatTensor)
            # disc_percent_policy_preds_one = disc_preds[z.size(0):].mean()
            disc_loss = self.bce(disc_logits, self.bce_targets)
            accuracy = (disc_preds == self.bce_targets).type(torch.FloatTensor).mean()

            if self.use_grad_pen:
                eps = Variable(torch.rand(positive_obs_batch.size(0), 1), requires_grad=True)
                if ptu.gpu_enabled(): eps = eps.cuda()
                
                # old and probably has a bad weird effect on the encoder
                # difference is that before I was also taking into account norm of grad of disc
                # wrt the z
                # interp_obs = eps*positive_obs_batch + (1-eps)*negative_obs_batch

                # permute the exp_obs_batch (not just within a single traj, but overall)
                # This is actually a really tricky question how to permute the batches
                # 1) permute within each of trajectories
                # z's will be matched, colors won't be matched anyways
                # 2) permute within trajectories corresponding to a single context set
                # z's will be matched, colors will be "more unmatched"
                # 3) just shuffle everything up
                # Also, the z's need to be handled appropriately

                interp_obs = eps*exp_obs_batch + (1-eps)*policy_obs_batch
                # interp_z = z_batch.detach()
                # interp_obs = torch.cat([interp_obs, interp_z], dim=1)
                interp_obs.detach()
                # interp_obs.requires_grad = True
                interp_actions = eps*positive_acts_batch + (1-eps)*negative_acts_batch
                interp_actions.detach()
                # interp_actions.requires_grad = True
                gradients = autograd.grad(
                    outputs=self.disc(torch.cat([interp_obs, z_batch.detach()], dim=1), interp_actions).sum(),
                    inputs=[interp_obs, interp_actions],
                    # grad_outputs=torch.ones(exp_specs['batch_size'], 1).cuda(),
                    create_graph=True, retain_graph=True, only_inputs=True
                )
                # print(gradients[0].size())
                # z_norm = gradients[0][:,-50:].norm(2, dim=1)
                # print('Z grad norm: %.4f +/- %.4f' % (torch.mean(z_norm), torch.std(z_norm)))
                # print(gradients[0][:,-50:].size())
                # o_norm = gradients[0][:,:-50].norm(2, dim=1)

                # o_norm = gradients[0].norm(2, dim=1)
                # print('Obs grad norm: %.4f +/- %.4f' % (torch.mean(o_norm), torch.std(o_norm)))
                # print(gradients[0].size())
                
                # print(gradients[0][:,:50].norm(2, dim=1))
                
                total_grad = torch.cat([gradients[0], gradients[1]], dim=1)
                # print(total_grad.size())
                gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()
                
                # another form of grad pen
                # gradient_penalty = (total_grad.norm(2, dim=1) ** 2).mean()

                disc_loss = disc_loss + gradient_penalty * self.grad_pen_weight

            total_reward_loss = z_reg_loss + disc_loss

            total_reward_loss.backward()
            self.disc_optimizer.step()
            self.encoder_optimizer.step()

            # print(self.disc.fc0.bias[0])
            # print(self.encoder.traj_encoder.traj_enc_mlp.fc0.bias[0])


        # train the policy
        # print('--------')
        # print(self.num_policy_updates_per_epoch)
        for i in range(self.num_policy_updates_per_epoch):
            context_batch, policy_batch = self._get_policy_training_batch()
            policy_batch = np_to_pytorch_batch(policy_batch)

            post_dist = self.encoder(context_batch)
            # z = post_dist.sample() # N_tasks x Dim
            z = post_dist.mean
            z = z.detach()

            # repeat z to have the right size
            z = z.repeat(1, self.policy_batch_size_per_task).view(
                self.num_tasks_used_per_update * self.policy_batch_size_per_task,
                -1
            ).detach()

            # now augment the obs with the latent sample z
            policy_batch['observations'] = torch.cat([policy_batch['observations'], z], dim=1)
            policy_batch['next_observations'] = torch.cat([policy_batch['next_observations'], z], dim=1)

            # compute the rewards
            # If you compute log(D) - log(1-D) then you just get the logits
            policy_rewards = self.disc(policy_batch['observations'], policy_batch['actions']).detach()
            policy_batch['rewards'] = policy_rewards
            # rew_more_than_zero = (rewards > 0).type(torch.FloatTensor).mean()
            # print(rew_more_than_zero.data[0])

            # do a policy update (the zeroing of grads etc. should be handled internally)
            # print(policy_rewards.size())
            self.policy_optimizer.train_step(policy_batch)
            # print(self.main_policy.fc0.bias[0])

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
        ] + self.policy_optimizer.networks


    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(
            encoder=self.encoder,
            main_policy=self.main_policy
        )
        snapshot.update(self.policy_optimizer.get_snapshot())
        return snapshot
