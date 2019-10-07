from collections import OrderedDict

import numpy as np
from copy import deepcopy

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


def subsample_traj(traj, num_samples):
    traj_len = traj['observations'].shape[0]
    idxs = np.random.choice(traj_len, size=num_samples, replace=traj_len<num_samples)
    new_traj = {k: traj[k][idxs,...] for k in traj}
    return new_traj


class MetaDagger(TorchMetaIRLAlgorithm):
    '''
    The train context expert replay buffer should also be added
    to the self.replay_buffer before training
    '''
    def __init__(
            self,
            env,
            policy,
            expert_policy,

            train_context_expert_replay_buffer,
            train_test_expert_replay_buffer,
            test_context_expert_replay_buffer,
            test_test_expert_replay_buffer,

            np_encoder,

            num_tasks_used_per_update=5,
            num_context_trajs_for_training=3,
            num_test_trajs_for_training=5,
            train_samples_per_traj=8,

            num_context_trajs_for_exploration=3,

            few_shot_version=False,
            min_context_size=1,
            max_context_size=5,
            
            num_tasks_per_eval=10,
            num_diff_context_per_eval_task=2,
            num_eval_trajs_per_post_sample=2,
            num_context_trajs_for_eval=3,
            policy_optim_batch_size_per_task=64,

            policy_lr=1e-3,
            policy_optimizer_class=optim.Adam,

            encoder_lr=1e-3,
            encoder_optimizer_class=optim.Adam,

            beta_1=0.9,

            num_update_loops_per_train_call=65,

            use_target_policy=False,
            target_policy=None,
            soft_target_policy_tau=0.005,

            use_target_enc=False,
            target_enc=None,
            soft_target_enc_tau=0.005,

            objective='mse', # or coul be 'max_like'
            max_KL_beta = 1.0,
            KL_ramp_up_start_iter=0,
            KL_ramp_up_end_iter=100,

            plotter=None,
            render_eval_paths=False,
            eval_deterministic=False,

            query_det_expert=False,
            **kwargs
    ):
        if kwargs['policy_uses_pixels']: raise NotImplementedError('policy uses pixels')
        if kwargs['wrap_absorbing']: raise NotImplementedError('wrap absorbing')
        
        super().__init__(
            env=env,
            train_context_expert_replay_buffer=train_context_expert_replay_buffer,
            train_test_expert_replay_buffer=train_test_expert_replay_buffer,
            test_context_expert_replay_buffer=test_context_expert_replay_buffer,
            test_test_expert_replay_buffer=test_test_expert_replay_buffer,
            **kwargs
        )

        self.policy = policy
        self.expert_policy = expert_policy
        self.encoder = np_encoder
        self.eval_statistics = None

        self.policy_optimizer = policy_optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
            betas=(beta_1, 0.999)
        )
        self.encoder_optimizer = encoder_optimizer_class(
            self.encoder.parameters(),
            lr=encoder_lr,
            betas=(0.9, 0.999)
        )
        print('\n\nBETA-1 for POLICY IS %f\n\n' % beta_1)

        self.num_tasks_used_per_update = num_tasks_used_per_update
        self.num_context_trajs_for_training = num_context_trajs_for_training
        self.num_test_trajs_for_training = num_test_trajs_for_training
        self.train_samples_per_traj = train_samples_per_traj
        self.policy_optim_batch_size_per_task = policy_optim_batch_size_per_task

        self.num_context_trajs_for_exploration = num_context_trajs_for_exploration
        
        self.num_tasks_per_eval = num_tasks_per_eval
        self.num_diff_context_per_eval_task = num_diff_context_per_eval_task
        self.num_eval_trajs_per_post_sample = num_eval_trajs_per_post_sample
        self.num_context_trajs_for_eval = num_context_trajs_for_eval

        self.use_target_enc = use_target_enc
        self.soft_target_enc_tau = soft_target_enc_tau

        self.use_target_policy = use_target_policy
        self.soft_target_policy_tau = soft_target_policy_tau

        if use_target_enc:
            if target_enc is None:
                print('\n\nMAKING TARGET ENC\n\n')
                self.target_enc = deepcopy(self.encoder)
            else:
                print('\n\nUSING GIVEN TARGET ENC\n\n')
                self.target_enc = target_enc
        
        if use_target_policy:
            if target_policy is None:
                print('\n\nMAKING TARGET ENC\n\n')
                self.target_policy = deepcopy(self.policy)
            else:
                print('\n\nUSING GIVEN TARGET ENC\n\n')
                self.target_policy = target_policy
        
        self.num_update_loops_per_train_call = num_update_loops_per_train_call

        assert objective in ['mse', 'max_like']
        self.use_mse_objective = objective == 'mse'
        if self.use_mse_objective:
            self.mse_loss = nn.MSELoss()
            if ptu.gpu_enabled():
                self.mse_loss.cuda()
        self.max_KL_beta = max_KL_beta
        self.KL_ramp_up_start_iter = KL_ramp_up_start_iter
        self.KL_ramp_up_end_iter = KL_ramp_up_end_iter

        self.few_shot_version = few_shot_version
        self.max_context_size = max_context_size
        self.min_context_size = min_context_size
        assert num_context_trajs_for_training == max_context_size

        self.eval_deterministic = eval_deterministic
        self.query_det_expert = query_det_expert
    

    def get_exploration_policy(self, task_identifier):
        if self.wrap_absorbing: raise NotImplementedError('wrap absorbing')
        if self.few_shot_version:
            # no need for this if/else statement like this, make it cleaner
            this_context_size = np.random.randint(self.min_context_size, self.max_context_size+1)
            list_of_trajs = self.train_context_expert_replay_buffer.sample_trajs_from_task(
                task_identifier,
                this_context_size
            )
            mask = None
        else:
            list_of_trajs = self.train_context_expert_replay_buffer.sample_trajs_from_task(
                task_identifier,
                self.num_context_trajs_for_exploration,
            )
            mask = None

        if self.use_target_enc:
            enc_to_use = self.target_enc
        else:
            enc_to_use = self.encoder
        
        mode = enc_to_use.training
        enc_to_use.eval()
        post_dist = enc_to_use([list_of_trajs], mask)
        enc_to_use.train(mode)

        z = post_dist.sample()
        # z = post_dist.mean
        z = z.cpu().data.numpy()[0]
        if self.use_target_policy:
            return PostCondMLPPolicyWrapper(self.target_policy, z)
        else:
            return PostCondMLPPolicyWrapper(self.policy, z)
    

    def get_eval_policy(self, task_identifier, mode='meta_test'):
        if self.wrap_absorbing: raise NotImplementedError('wrap absorbing')
        if mode == 'meta_train':
            rb = self.train_context_expert_replay_buffer
        else:
            rb = self.test_context_expert_replay_buffer
        
        eval_context_size = np.random.randint(self.min_context_size, self.max_context_size+1)
        list_of_trajs = rb.sample_trajs_from_task(
            task_identifier,
            eval_context_size\
                if self.few_shot_version else self.num_context_trajs_for_eval,
        )
        # list_of_trajs = rb.sample_trajs_from_task(
        #     task_identifier,
        #     3 if self.few_shot_version else self.num_context_trajs_for_eval,
        # )
        
        if self.use_target_enc:
            enc_to_use = self.target_enc
        else:
            enc_to_use = self.encoder
        
        mode = enc_to_use.training
        enc_to_use.eval()
        post_dist = enc_to_use([list_of_trajs])
        enc_to_use.train(mode)

        z = post_dist.sample()
        # z = post_dist.mean
        z = z.cpu().data.numpy()[0]
        if self.use_target_policy:
            return PostCondMLPPolicyWrapper(self.target_policy, z, deterministic=self.eval_deterministic)
        else:
            return PostCondMLPPolicyWrapper(self.policy, z, deterministic=self.eval_deterministic)
    

    def _get_training_batch(self, epoch):
        if self.few_shot_version:
            context_batch, task_identifiers_list = self.train_context_expert_replay_buffer.sample_trajs(
                self.max_context_size,
                num_tasks=self.num_tasks_used_per_update,
                keys=['observations', 'actions', 'next_observations']
                # keys=['observations', 'actions']
            )
            mask = ptu.Variable(torch.zeros(self.num_tasks_used_per_update, self.max_context_size, 1))
            this_context_sizes = np.random.randint(self.min_context_size, self.max_context_size+1, size=self.num_tasks_used_per_update)
            for i, c_size in enumerate(this_context_sizes):
                mask[i,:c_size,:] = 1.0
        else:
            context_batch, task_identifiers_list = self.train_context_expert_replay_buffer.sample_trajs(
                self.num_context_trajs_for_training,
                num_tasks=self.num_tasks_used_per_update,
                keys=['observations', 'actions', 'next_observations']
                # keys=['observations', 'actions']
            )
            mask = None

        # get the test batch for the tasks from policy buffer
        if epoch == 0:
            # print('USING ONLY EXPERT DATA')
            policy_batch, _ = self.train_test_expert_replay_buffer.sample_random_batch(
                self.policy_optim_batch_size_per_task,
                task_identifiers_list=task_identifiers_list
            )
        else:
            # print('USING EXPERT AND POLICY DATA')
            policy_batch, _ = self.replay_buffer.sample_random_batch(
                self.policy_optim_batch_size_per_task,
                task_identifiers_list=task_identifiers_list
            )
            
        policy_obs = np.concatenate([d['observations'] for d in policy_batch], axis=0) # (N_tasks * batch_size) x Dim
        policy_acts = np.concatenate([d['actions'] for d in policy_batch], axis=0) # (N_tasks * batch_size) x Dim
        policy_batch = dict(
            observations=policy_obs,
            actions=policy_acts,
        )
        return context_batch, mask, policy_batch


    def _do_training(self, epoch):
        for t in range(self.num_update_loops_per_train_call):
            self._do_training_step(epoch, t)
        
    
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


    def _do_training_step(self, epoch, loop_iter):
        '''
            Train the discriminator
        '''
        self.encoder_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()

        # prep the batches
        # OLD VERSION -----------------------------------------------------------------------------
        # context_batch, context_pred_batch, test_pred_batch, mask = self._get_training_batch()
        
        # post_dist = self.encoder(context_batch, mask)
        # z = post_dist.sample() # N_tasks x Dim
        # # z = post_dist.mean

        # # convert it to a pytorch tensor
        # # note that our objective says we should maximize likelihood of
        # # BOTH the context_batch and the test_batch
        # obs_batch = np.concatenate((context_pred_batch['observations'], test_pred_batch['observations']), axis=0)
        # obs_batch = Variable(ptu.from_numpy(obs_batch), requires_grad=False)

        # acts_batch = np.concatenate((context_pred_batch['actions'], test_pred_batch['actions']), axis=0)
        # acts_batch = Variable(ptu.from_numpy(acts_batch), requires_grad=False)

        # # make z's for expert samples
        # context_pred_z = z.repeat(1, self.num_context_trajs_for_training * self.train_samples_per_traj).view(
        #     -1,
        #     z.size(1)
        # )
        # test_pred_z = z.repeat(1, self.num_test_trajs_for_training * self.train_samples_per_traj).view(
        #     -1,
        #     z.size(1)
        # )
        # z_batch = torch.cat([context_pred_z, test_pred_z], dim=0)
        # NEW VERSION (this is more fair to this model) -------------------------------------------
        context_batch, mask, pred_batch = self._get_training_batch(epoch)

        post_dist = self.encoder(context_batch, mask)
        z = post_dist.sample() # N_tasks x Dim
        # z = post_dist.mean

        obs_batch = Variable(ptu.from_numpy(pred_batch['observations']), requires_grad=False)
        acts_batch = Variable(ptu.from_numpy(pred_batch['actions']), requires_grad=False)
        z_batch = z.repeat(1, self.policy_optim_batch_size_per_task).view(
            -1,
            z.size(1)
        )

        input_batch = torch.cat([obs_batch, z_batch], dim=-1)
        
        if self.use_mse_objective:
            pred_acts = self.policy(input_batch)[1]
            recon_loss = self.mse_loss(pred_acts, acts_batch)
        else:
            recon_loss = -1.0 * self.policy.get_log_prob(input_batch, acts_batch).mean()
        
        # add KL loss term
        cur_KL_beta = linear_schedule(
            self._n_train_steps_total*self.num_update_loops_per_train_call + loop_iter - self.KL_ramp_up_start_iter,
            0.0,
            self.max_KL_beta,
            self.KL_ramp_up_end_iter - self.KL_ramp_up_start_iter
        )
        KL_loss = self._compute_KL_loss(post_dist)
        if cur_KL_beta == 0.0: KL_loss = KL_loss.detach()

        loss = recon_loss + cur_KL_beta * KL_loss
        loss.backward()

        self.policy_optimizer.step()
        self.encoder_optimizer.step()

        if self.use_target_policy:
            ptu.soft_update_from_to(self.policy, self.target_policy, self.soft_target_policy_tau)
        if self.use_target_enc:
            ptu.soft_update_from_to(self.encoder, self.target_enc, self.soft_target_enc_tau)

        """
        Save some statistics for eval
        """
        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
            if self.use_target_policy:
                enc_to_use = self.target_enc if self.use_target_enc else self.encoder
                pol_to_use = self.target_policy

                if self.use_mse_objective:
                    pred_acts = pol_to_use(input_batch)[1]
                    target_loss = self.mse_loss(pred_acts, acts_batch)
                    self.eval_statistics['Target MSE Loss'] = np.mean(ptu.get_numpy(target_loss))
                else:
                    target_loss = -1.0*pol_to_use.get_log_prob(input_batch, acts_batch).mean()
                    self.eval_statistics['Target Neg Log Like'] = np.mean(ptu.get_numpy(target_loss))
            else:
                if self.use_mse_objective:
                    self.eval_statistics['Target MSE Loss'] = np.mean(ptu.get_numpy(recon_loss))
                else:
                    self.eval_statistics['Target Neg Log Like'] = np.mean(ptu.get_numpy(recon_loss))
            self.eval_statistics['Target KL'] = np.mean(ptu.get_numpy(KL_loss))
            self.eval_statistics['Cur KL Beta'] = cur_KL_beta
            self.eval_statistics['Max KL Beta'] = self.max_KL_beta

            self.eval_statistics['Avg Post Mean Abs'] = np.mean(np.abs(ptu.get_numpy(post_dist.mean)))
            self.eval_statistics['Avg Post Cov Abs'] = np.mean(np.abs(ptu.get_numpy(post_dist.cov)))

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
        networks_list = [self.encoder, self.policy]
        if self.use_target_enc: networks_list += [self.target_enc]
        if self.use_target_policy: networks_list += [self.target_policy]
        return networks_list

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(encoder=self.encoder)
        snapshot.update(policy=self.policy)
        if self.use_target_enc: snapshot.update(target_enc=self.target_enc)
        if self.use_target_policy: snapshot.update(target_enc=self.target_policy)
        return snapshot

    def _handle_step(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminal,
        task_identifier,
        agent_info,
        env_info,
    ):
        # task_expert = self.expert_policy.get_exploration_policy(
        #     self.env.task_id_to_obs_task_params(task_identifier)
        # )
        # for ant linear classification case
        meta_env = self.training_env._wrapped_env
        task_expert = self.expert_policy.get_exploration_policy(meta_env.targets[meta_env.true_label])
        task_expert.deterministic = self.query_det_expert

        # if isinstance(self.obs_space, Dict):
        #     if self.get_full_obs_dict:
        #         agent_obs = observation
        #     else:
        #         if self.policy_uses_pixels:
        #             agent_obs = observation['pixels']
        #         else:
        #             agent_obs = observation['obs']
        # else:
        #     agent_obs = observation
        # agent_obs = agent_obs*self.env.obs_std[0] + self.env.obs_mean[0]
        # for ant lin class
        agent_obs = np.concatenate([
            meta_env.sim.data.qpos.flat,
            meta_env.sim.data.qvel.flat,
            meta_env.get_body_com("torso").flat
        ])

        action, _ = task_expert.get_action(agent_obs)
        super()._handle_step(
            observation,
            action,
            reward,
            next_observation,
            terminal,
            task_identifier,
            agent_info,
            env_info,
        )
