'''
For now implement the online version, then we'll see if we can be more efficient
'''
from collections import OrderedDict, defaultdict

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
from rlkit.torch.distributions import ReparamMultivariateNormalDiag
from rlkit.torch.sac.policies import PostCondMLPPolicyWrapper
from rlkit.data_management.path_builder import PathBuilder
from gym.spaces import Dict


class FetchTuner():
    def __init__(
        self,
        alg,

        z_dim,
        
        num_updates,
        num_z_samples_per_update,
        num_trajs_per_z_sample,
        freq_eval,

        mu_lr,
        log_sig_lr,
        reward_scale,
        use_nat_grad,
    ):
        self.alg = alg
        self.env = alg.env
        # self.main_policy = alg.main_policy
        self.main_policy = alg.policy
        self.main_policy.eval()

        self.obs_task_params = []
        self.task_params = []
        for task_params, obs_task_params in alg.test_task_params_sampler:
            self.obs_task_params.append(obs_task_params)
            self.task_params.append(task_params)
        
        # cut them for now for DEBUGGING
        self.obs_task_params = self.obs_task_params[:2]
        self.task_params = self.task_params[:2]
        
        # initialize the mu and log sigmas
        # self.mus = Variable(ptu.from_numpy(np.zeros((len(self.obs_task_params), z_dim))), requires_grad=True)
        # self.log_sigmas = Variable(ptu.from_numpy(np.zeros((len(self.obs_task_params), z_dim))), requires_grad=True)
        self.np_mus = np.zeros((len(self.obs_task_params), z_dim))
        self.np_log_sigmas = np.zeros((len(self.obs_task_params), z_dim))

        self.num_updates = num_updates
        self.num_z_samples_per_update = num_z_samples_per_update
        self.num_trajs_per_z_sample = num_trajs_per_z_sample
        self.freq_eval = freq_eval

        self.mu_lr = mu_lr
        self.log_sig_lr = log_sig_lr
        self.reward_scale = reward_scale
        self.use_nat_grad = use_nat_grad


    def gen_rollout(self, obs_task_params, task_params, z, num_rollouts):
        # set up the post cond policy
        z = z.cpu().data.numpy()
        post_cond_policy = PostCondMLPPolicyWrapper(self.main_policy, z)
        post_cond_policy.policy.eval()

        # generate some rollouts
        successes = []
        for roll_num in range(num_rollouts):
            observation = self.env.reset(task_params=task_params, obs_task_params=obs_task_params)
            terminal = False
            timestep = 0
            cur_success = False

            while (not terminal) and timestep < self.alg.max_path_length:
                agent_obs = observation['obs']
                action, agent_info = post_cond_policy.get_action(agent_obs)
                
                next_ob, raw_reward, terminal, env_info = (self.env.step(action))
                if env_info['is_success']: cur_success = True
                if self.alg.no_terminal: terminal = False
                observation = next_ob
                timestep += 1
            
            successes.append(float(cur_success))
        return successes


    def train(self):
        for t in range(self.num_updates):
            self.mus = Variable(ptu.from_numpy(self.np_mus), requires_grad=True)
            self.log_sigmas = Variable(ptu.from_numpy(self.np_log_sigmas), requires_grad=True)

            # generate rollouts for each task
            normal = ReparamMultivariateNormalDiag(self.mus, self.log_sigmas)
            all_samples = []
            all_log_probs = []
            all_rewards = []
            for sample_num in range(self.num_z_samples_per_update):
                sample = normal.sample()
                all_samples.append(sample)
                all_log_probs.append(normal.log_prob(sample))

                # evaluate each of the z's for each task
                rewards = []
                for i in range(len(self.obs_task_params)):
                    successes = self.gen_rollout(
                        self.obs_task_params[i],
                        self.task_params[i],
                        sample[i],
                        self.num_trajs_per_z_sample
                    )
                    rewards.append(np.mean(successes))
                all_rewards.append(rewards)
            
            all_log_probs = torch.cat(all_log_probs, dim=-1) # num_tasks x num_samples

            np_all_rewards = np.array(all_rewards).T # num_tasks x num_samples
            all_rewards = Variable(ptu.from_numpy(np_all_rewards))
            all_rewards = all_rewards - torch.mean(all_rewards, dim=-1, keepdim=True)
            all_rewards = self.reward_scale * all_rewards

            # compute gradients wrt mus and sigmas
            pg_loss = -1.0 * torch.sum(
                torch.mean(all_log_probs * all_rewards, dim=-1)
            )
            grads = autograd.grad(
                outputs=pg_loss,
                inputs=[self.mus, self.log_sigmas],
                only_inputs=True
            )

            # update the mus and sigmas
            if self.use_nat_grad:
                print('Nat Grad')
                mu_grad = grads[0] * (torch.exp(2 * self.log_sigmas)).detach()
                log_sig_grad = 0.5 * grads[1]
            else:
                print('Normal Grad')
                mu_grad = grads[0]
                log_sig_grad = grads[1]
            self.mus = self.mus - self.mu_lr * mu_grad
            self.log_sigmas = self.log_sigmas - self.log_sig_lr * log_sig_grad
            self.np_mus = ptu.get_numpy(self.mus)
            self.np_log_sigmas = ptu.get_numpy(self.log_sigmas)

            # logging
            np_all_rewards = np.mean(np_all_rewards, axis=-1)
            print('\n-----------------------------------------------')
            # print('Avg Reward: {}'.format(np.mean(np_all_rewards)))
            # print('Std Reward: {}'.format(np.std(np_all_rewards)))
            # print('Max Reward: {}'.format(np.max(np_all_rewards)))
            # print('Min Reward: {}'.format(np.min(np_all_rewards)))
            print(np_all_rewards)
            print(self.np_mus)
            print(np.exp(2*self.np_log_sigmas))
            # print(grads[0])
            # print(grads[1])
