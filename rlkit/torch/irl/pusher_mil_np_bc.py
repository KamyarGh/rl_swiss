from collections import OrderedDict
import os.path as osp
from copy import deepcopy

import numpy as np
from scipy.misc import imsave

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

from rlkit.torch.sac.policies import PostCondPuhserPolicyWrapper, CondBaselineContextualPolicy
from rlkit.torch.sac.policies import YetAnotherPostCondPuhserPolicyWrapper
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


class PusherSpecificNeuralProcessBC(TorchMetaIRLAlgorithm):
    def __init__(
        self,
        policy,

        encoder,

        # ds means dataset
        train_ds,
        test_ds,

        easy_context=False,
        last_image_is_context=False,
        using_all_context=False,
        context_video_subsample_factor=4,

        num_tasks_used_per_update=5,
        num_context_trajs_for_training=3,
        num_test_trajs_for_training=5,
        train_samples_per_traj=8,

        num_tasks_per_eval=10,
        num_diff_context_per_eval_task=2,
        num_eval_trajs_per_post_sample=2,

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

        objective='mse', # or could be 'max_like'
        mse_loss_multiplier=50.0, # effectively choosing the scale
        max_KL_beta = 1.0,
        KL_ramp_up_start_iter=0,
        KL_ramp_up_end_iter=100,

        plotter=None,
        render_eval_paths=False,
        eval_deterministic=False,

        log_dir='',
        **kwargs
    ):
        if kwargs['policy_uses_pixels']: raise NotImplementedError('policy uses pixels')
        if kwargs['wrap_absorbing']: raise NotImplementedError('wrap absorbing')
        assert num_context_trajs_for_training == 1
        assert not use_target_policy
        assert not use_target_enc
        # assert objective == 'mse', 'NaN problems when we used MLE'
        # assert mse_loss_multiplier == 50.0, "What they used in their work"

        super().__init__(
            env=None,
            train_context_expert_replay_buffer=None,
            train_test_expert_replay_buffer=None,
            test_context_expert_replay_buffer=None,
            test_test_expert_replay_buffer=None,
            **kwargs
        )

        self.train_ds, self.test_ds = train_ds, test_ds
        self.context_video_subsample_factor = context_video_subsample_factor
        self.context_subsample_inds = np.arange(99, -1, -1*self.context_video_subsample_factor)[::-1]
        assert sum([easy_context, last_image_is_context, using_all_context]) <= 1
        self.easy_context = easy_context
        self.last_image_is_context = last_image_is_context
        self.using_all_context = using_all_context
        self.log_dir = log_dir

        self.policy = policy
        self.encoder = encoder
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

        
        self.num_tasks_per_eval = num_tasks_per_eval
        self.num_diff_context_per_eval_task = num_diff_context_per_eval_task
        self.num_eval_trajs_per_post_sample = num_eval_trajs_per_post_sample

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
        self.mse_loss_multiplier = mse_loss_multiplier
        if self.use_mse_objective:
            self.mse_loss = nn.MSELoss()
            if ptu.gpu_enabled():
                self.mse_loss.cuda()
        self.max_KL_beta = max_KL_beta
        self.KL_ramp_up_start_iter = KL_ramp_up_start_iter
        self.KL_ramp_up_end_iter = KL_ramp_up_end_iter

        self.eval_deterministic = eval_deterministic
        self.debug_idx = 0
    

    def get_exploration_policy(self, task_identifier):
        enc_to_use = self.encoder
        mode = enc_to_use.training
        enc_to_use.eval()

        self.policy.eval()

        # sample a trajectory from a task
        # TODO: this is inefficient because I am sampling a whole
        # batch of trajectories from a task and only using one of them
        # batch = self.train_ds[task_identifier]
        # vid = batch['videos'][0:1]

        batch = self.train_ds.get_single_traj(task_identifier)
        if self.easy_context:
            vid = batch['videos'][0][:,-1,-45:,30:75][None]
            vid = Variable(ptu.from_numpy(vid), requires_grad=False)
            vid.mul_(1.0/255.0)
        elif self.last_image_is_context:
            vid = batch['videos'][0][:,-1][None]
            vid = Variable(ptu.from_numpy(vid), requires_grad=False)
            vid.mul_(1.0/255.0)
        elif self.using_all_context:
            vid = batch['videos'][0][:,self.context_subsample_inds]
            vid = vid[None]
            vid = vid.transpose((0, 2, 1, 3, 4))
            vid = Variable(ptu.from_numpy(vid), requires_grad=False)
            vid.mul_(1.0/255.0)

            context_Xs = batch['states'][0][self.context_subsample_inds]
            context_Xs = Variable(ptu.from_numpy(context_Xs), requires_grad=False)

            context_Us = batch['actions'][0][self.context_subsample_inds]
            context_Us = Variable(ptu.from_numpy(context_Us), requires_grad=False)

            all_context = {
                'image': vid,
                'X': context_Xs,
                'U': context_Us,
            }
        else:
            vid = batch['videos'][0][:,self.context_subsample_inds]
            vid = vid[None]
            # vid = np.array(batch['videos'])
            # print('----')
            # print(vid.shape)
            vid = Variable(ptu.from_numpy(vid), requires_grad=False)
            vid.mul_(1.0/255.0)

        # from scipy.misc import imsave
        # print(vid.shape)
        # imsave('plots/junk_vis/crop_test_%d.png' % self.debug_idx, vid[:,-1,-45:,30:75].transpose(1,2,0))
        # self.debug_idx += 1
        # 1/0


        # for debugging to see things match up
        # from scipy.misc import imsave
        # imsave('plots/junk_vis/get_exp.png', ptu.get_numpy(vid[0,:,5]).transpose(1,2,0))

        if self.last_image_is_context:
            enc_to_use.train(mode)
            return CondBaselineContextualPolicy(self.policy, vid)
        elif self.using_all_context:
            film_feats, extra_latents = enc_to_use(all_context)
            return YetAnotherPostCondPuhserPolicyWrapper(self.policy, film_feats, extra_latents, deterministic=False)
        else:
            z = enc_to_use(vid)
            z = z.cpu().data.numpy()[0]
            enc_to_use.train(mode)
            return PostCondPuhserPolicyWrapper(self.policy, z, deterministic=False)

    

    def get_eval_policy(self, task_identifier, mode='meta_test', return_context=False):
        enc_to_use = self.encoder
        is_training_mode = enc_to_use.training
        enc_to_use.eval()

        self.policy.eval()

        # sample a trajectory from a task
        # TODO: this is inefficient because I am sampling a whole
        # batch of trajectories from a task and only using one of them
        # batch = self.train_ds[task_identifier]
        # vid = batch['videos'][0:1]
        if mode == 'meta_test':
            batch = self.test_ds.get_single_traj(task_identifier)
        else:
            batch = self.train_ds.get_single_traj(task_identifier)
        
        if self.easy_context:
            vid = batch['videos'][0][:,-1,-45:,30:75][None]
            vid = Variable(ptu.from_numpy(vid), requires_grad=False)
            vid.mul_(1.0/255.0)
        if self.using_all_context:
            vid = batch['videos'][0][:,self.context_subsample_inds]
            vid = vid[None]
            vid = vid.transpose((0, 2, 1, 3, 4))
            vid = Variable(ptu.from_numpy(vid), requires_grad=False)
            vid.mul_(1.0/255.0)

            context_Xs = batch['states'][0][self.context_subsample_inds]
            context_Xs = Variable(ptu.from_numpy(context_Xs), requires_grad=False)

            context_Us = batch['actions'][0][self.context_subsample_inds]
            context_Us = Variable(ptu.from_numpy(context_Us), requires_grad=False)

            all_context = {
                'image': vid,
                'X': context_Xs,
                'U': context_Us,
            }
        elif self.last_image_is_context:
            vid = batch['videos'][0][:,-1][None]
            vid = Variable(ptu.from_numpy(vid), requires_grad=False)
            vid.mul_(1.0/255.0)
        else:
            vid = batch['videos'][0][:,self.context_subsample_inds]
            vid = vid[None]
            # vid = np.array(batch['videos'])
            # print('----')
            # print(vid.shape)
            vid = Variable(ptu.from_numpy(vid), requires_grad=False)
            vid.mul_(1.0/255.0)

        # from scipy.misc import imsave
        # imsave('plots/junk_vis/val_check_get_eval.png', ptu.get_numpy(vid[0][:,0,:,:]).transpose(1,2,0))

        # if self.last_image_is_context:
        #     z = vid
        # else:
        #     z = enc_to_use(vid)
        # enc_to_use.train(is_training_mode)

        # z = z.cpu().data.numpy()[0]
        # if return_context:
        #     return PostCondPuhserPolicyWrapper(self.policy, z), vid
        # return PostCondPuhserPolicyWrapper(self.policy, z)

        if self.last_image_is_context:
            enc_to_use.train(mode)
            if return_context:
                return CondBaselineContextualPolicy(self.policy, vid), vid
            return CondBaselineContextualPolicy(self.policy, vid)
        elif self.using_all_context:
            film_feats, extra_latents = enc_to_use(all_context)
            if return_context:
                return YetAnotherPostCondPuhserPolicyWrapper(self.policy, film_feats, extra_latents, deterministic=False), all_context
            return YetAnotherPostCondPuhserPolicyWrapper(self.policy, film_feats, extra_latents, deterministic=False)
        else:
            z = enc_to_use(vid)
            enc_to_use.train(is_training_mode)
            z = z.cpu().data.numpy()[0]
            if mode == 'meta_test':
                det_eval = True
            else:
                det_eval = False
            if return_context:
                return PostCondPuhserPolicyWrapper(self.policy, z), vid
                return PostCondPuhserPolicyWrapper(self.policy, z, deterministic=det_eval), vid
            return PostCondPuhserPolicyWrapper(self.policy, z, deterministic=det_eval)


    def _get_training_batch(self):
        tasks = self.train_task_params_sampler.sample_unique(self.num_tasks_used_per_update)

        # print('\n\n')
        # print(tasks)

        task_inds = list(map(lambda t: int(t[-1].split('_')[-1]), tasks))
        task_batches = [self.train_ds[task] for task in task_inds]

        # print(task_inds)
        # print('\n\n')
        # we are just using single videos for context for now
        # subsample the videos by desired amount
        if self.easy_context:
            contexts = np.array([b['videos'][0][:,-1,-45:,30:75] for b in task_batches])
            contexts = Variable(ptu.from_numpy(contexts), requires_grad=False)
            # remember to divide the images by 255.0
            contexts.mul_(1.0/255.0)
        elif self.last_image_is_context:
            assert False, 'Check reshaping!'
            contexts = np.array([b['videos'][0][:,-1] for b in task_batches])
            contexts = Variable(ptu.from_numpy(contexts), requires_grad=False)
            # remember to divide the images by 255.0
            contexts.mul_(1.0/255.0)
        elif self.using_all_context:
            # N_tasks x 3 x len x h x w
            context_imgs = np.array([b['videos'][0][:,self.context_subsample_inds] for b in task_batches])
            context_imgs = context_imgs.transpose((0, 2, 1, 3, 4))
            context_imgs = Variable(ptu.from_numpy(context_imgs), requires_grad=False)
            # remember to divide the images by 255.0
            context_imgs.mul_(1.0/255.0)
            context_Xs = np.array([b['states'][0][self.context_subsample_inds] for b in task_batches])
            context_Xs = Variable(ptu.from_numpy(context_Xs), requires_grad=False)
            context_Us = np.array([b['actions'][0][self.context_subsample_inds] for b in task_batches])
            context_Us = Variable(ptu.from_numpy(context_Us), requires_grad=False)
            contexts = {
                'image': context_imgs,
                'X': context_Xs,
                'U': context_Us
            }
        else:
            assert False, 'Check reshaping!'
            contexts = np.array([b['videos'][0][:,self.context_subsample_inds] for b in task_batches])
            contexts = Variable(ptu.from_numpy(contexts), requires_grad=False)
            # remember to divide the images by 255.0
            contexts.mul_(1.0/255.0)

        # build the prediction batches
        # determine the indices you want to take per traj per task for prediction
        inds = [
            [
                np.random.choice(100, size=self.train_samples_per_traj, replace=False)
                for _ in range(len(batch['videos']))
            ] for batch in task_batches
        ]

        pred_imgs = []
        pred_Xs = []
        true_Us = []
        for i in range(len(task_batches)):
            batch = task_batches[i]
            batch_inds = inds[i]
            imgs = []
            Xs = []
            Us = []
            for j in range(len(batch['videos'])):
                imgs.append(batch['videos'][j][:, batch_inds[j]])
                Xs.append(batch['states'][j][batch_inds[j]])
                Us.append(batch['actions'][j][batch_inds[j]])
            pred_imgs.append(imgs)
            pred_Xs.append(Xs)
            true_Us.append(Us)
        pred_imgs = np.array(pred_imgs)
        pred_Xs = np.array(pred_Xs)
        true_Us = np.array(true_Us)

        pred_imgs = pred_imgs.transpose((0, 1, 3, 2, 4, 5))
        pred_imgs = Variable(ptu.from_numpy(pred_imgs), requires_grad=False)
        pred_imgs.mul_(1.0/255.0)

        pred_Xs = Variable(ptu.from_numpy(pred_Xs), requires_grad=False)
        true_Us = Variable(ptu.from_numpy(true_Us), requires_grad=False)

        # the Xs will be normalized when they are loaded

        return contexts, pred_imgs, pred_Xs, true_Us


    def _do_training(self, epoch):
        for t in range(self.num_update_loops_per_train_call):
            self._do_training_step(epoch, t)


    def _do_training_step(self, epoch, loop_iter):
        '''
            Train the discriminator
        '''
        self.policy.train()
        self.encoder.train()

        self.encoder_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()

        context, pred_imgs, pred_Xs, true_Us = self._get_training_batch()

        if self.last_image_is_context:
            z = context
        elif self.using_all_context:
            film_feats, extra_latent = self.encoder(context)
        else:
            z = self.encoder(context)
        # repeat the z
        total_per_task = (self.num_context_trajs_for_training + self.num_test_trajs_for_training) * self.train_samples_per_traj
        # repeated_z = z.repeat(1, total_per_task, 1, 1, 1).view(
        #     z.size(0)*total_per_task,
        #     z.size(1),
        #     z.size(2),
        #     z.size(3),
        #     z.size(4)
        # )
        if self.last_image_is_context:
            assert False, 'Check it!'
            repeated_z = z.repeat(1, total_per_task, 1, 1)
            repeated_z = repeated_z.view(z.size(0)*total_per_task, z.size(1), z.size(2), z.size(3))
        elif self.using_all_context:
            repeated_film_feats = [
                ff.repeat(1, total_per_task).view(ff.size(0)*total_per_task, ff.size(1))
                for ff in film_feats
            ]
            repeated_extra_latent = extra_latent.repeat(1, total_per_task).view(extra_latent.size(0)*total_per_task, extra_latent.size(1))
        else:
            repeated_z = z.repeat(1, total_per_task)
            repeated_z = repeated_z.view(z.size(0)*total_per_task, z.size(1))

        # reshape the pred batches
        # pred_imgs: N_tasks x N_trajs x N_samples x C x H x W
        N_tasks, N_trajs_per_task, N_samples_per_traj, C, H, W = pred_imgs.size(0), pred_imgs.size(1), pred_imgs.size(2), pred_imgs.size(3), pred_imgs.size(4), pred_imgs.size(5)
        pred_imgs = pred_imgs.view(
            N_tasks * N_trajs_per_task * N_samples_per_traj,
            C,
            H,
            W
        )
        pred_Xs = pred_Xs.view(
            N_tasks * N_trajs_per_task * N_samples_per_traj,
            -1
        )
        true_Us = true_Us.view(
            N_tasks * N_trajs_per_task * N_samples_per_traj,
            -1
        )
        # print(pred_imgs.size())
        # print(pred_Xs.size())
        # print(true_Us.size())
        
        if self.use_mse_objective:
            if self.using_all_context: assert False

            # print('\n\n\n\nUsing MSE objective')
            pred_Us = self.policy(
                {
                    'image': pred_imgs,
                    'z': repeated_z,
                    'X': pred_Xs,
                }
            )[1]
            loss = self.mse_loss(pred_Us, true_Us) * self.mse_loss_multiplier
        else:
            if self.using_all_context:
                inuput_dict = {
                    'image': pred_imgs,
                    'X': pred_Xs,
                    'film_feats': repeated_film_feats,
                    'extra_latents': repeated_extra_latent
                }
            else:
                inuput_dict = {
                    'image': pred_imgs,
                    'z': repeated_z,
                    'X': pred_Xs,
                }
            loss = -1.0 * self.policy.get_log_prob(
                inuput_dict,
                true_Us
            ).mean()
        
        loss.backward()

        # print(self.encoder.conv_part[0].bias.grad)
        # print(self.encoder.fc_part[0].bias.grad)
        # print(self.policy.fcs[0].bias.grad)

        # print(self.policy.conv_part[0].bias.grad)
        # 1/0

        # print(self.encoder)
        # print(self.encoder.convs_list[0].bias)
        # print(self.encoder.convs_list[0].bias.grad)

        # print(self.policy)
        # print(self.policy.image_processor.initial_conv[0].bias.grad)

        self.policy_optimizer.step()
        self.encoder_optimizer.step()

        """
        Save some statistics for eval
        """
        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
            if self.use_mse_objective:
                self.eval_statistics['Target MSE Loss'] = np.mean(ptu.get_numpy(loss)) / self.mse_loss_multiplier
                self.eval_statistics['MSE Loss Multiplier'] = self.mse_loss_multiplier
            else:
                self.eval_statistics['Target Neg Log Like'] = np.mean(ptu.get_numpy(loss))


    def evaluate(self, epoch):
        super().evaluate(epoch)


    def obtain_eval_samples(self, epoch, mode='meta_train'):
        self.training_mode(False)
        self.policy.eval()

        
        if mode == 'meta_train':
            params_samples = self.train_task_params_sampler.sample_unique(self.num_tasks_per_eval)
        else:
            params_samples = self.test_task_params_sampler.sample_unique(self.num_tasks_per_eval)
        all_eval_tasks_paths = []
        eval_task_num = -1
        for task_params, obs_task_params in params_samples:
            eval_task_num += 1
            saved_task_gif = False
            cur_eval_task_paths = []
            if mode == 'meta_train':
                self.env = self.training_env_getter(obs_task_params)
            else:
                self.env = self.test_env_getter(obs_task_params)
            self.env.reset()
            task_identifier = self.env.task_identifier

            for _ in range(self.num_diff_context_per_eval_task):
                eval_policy, context = self.get_eval_policy(task_identifier, mode=mode, return_context=True)

                for _ in range(self.num_eval_trajs_per_post_sample):
                    cur_eval_path_builder = PathBuilder()
                    observation = self.env.reset()
                    # from scipy.misc import imsave
                    # imsave('plots/junk_vis/val_check_obtain_eval.png', observation['image'].transpose(1,2,0))
                    # if mode == 'meta_test':
                    #     1/0
                    terminal = False

                    while (not terminal) and len(cur_eval_path_builder) < self.max_path_length:
                        agent_obs = observation
                        action, agent_info = self._get_action_and_info(agent_obs)

                        # print(self.env)
                        # print(action)
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
                        if not saved_task_gif:
                            saved_task_gif = True
                            if eval_task_num < 2:
                                path = cur_eval_task_paths[-1]
                                gif_frames = [d["image"]for d in path["observations"]]
                                for frame_num, frame in enumerate(gif_frames):
                                    if frame_num % 4 == 3:
                                        imsave(osp.join(self.log_dir, mode+'task_%d_frame_%d.png'%(eval_task_num, frame_num)), frame.transpose(1,2,0))
                                # print(gif_frames)
                                # for img in gif_frames:
                                #     print(np.max(img), np.min(img))
                                # write_gif(gif_frames, osp.join(self.log_dir, mode+'_%d.gif'%eval_task_num) , fps=20)
                                if self.easy_context or self.last_image_is_context:
                                    context_img = ptu.get_numpy(context)[0].transpose(1,2,0)
                                    imsave(osp.join(self.log_dir, mode+'task_%d_context_%d.png'%(eval_task_num, eval_task_num)), context_img)
                                if self.using_all_context:
                                    context_img = ptu.get_numpy(context['image'][0,-1]).transpose(1,2,0)
                                    imsave(osp.join(self.log_dir, mode+'task_%d_context_%d.png'%(eval_task_num, eval_task_num)), context_img)
                                print('Saved the gifs')
            all_eval_tasks_paths.extend(cur_eval_task_paths)
        
        # flatten the list of lists
        self.policy.train()
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
