from collections import OrderedDict

import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

from rlkit.core import logger
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


class UpperBound():
    def __init__(
            self,
            env,

            train_context_expert_replay_buffer,
            train_test_expert_replay_buffer,
            test_context_expert_replay_buffer,
            test_test_expert_replay_buffer,

            mlp,
            encoder,

            num_tasks_used_per_update=5,
            num_context_trajs_for_training=3,
            num_test_trajs_for_training=5,

            training_regression=True,
            few_shot_version=False,
            min_context_size=1,
            max_context_size=5,
            classification_batch_size_per_task=32,

            num_tasks_per_eval=10,

            encoder_lr=1e-3,
            encoder_optimizer_class=optim.Adam,

            mlp_lr=1e-3,
            mlp_optimizer_class=optim.Adam,

            num_update_loops_per_train_call=1000,
            num_epochs=10000,

            **kwargs
    ):
        self.env = env

        self.train_context_expert_replay_buffer = train_context_expert_replay_buffer
        self.train_test_expert_replay_buffer = train_test_expert_replay_buffer
        self.test_context_expert_replay_buffer = test_context_expert_replay_buffer
        self.test_test_expert_replay_buffer = test_test_expert_replay_buffer

        self.mlp = mlp
        self.encoder = encoder

        self.num_tasks_used_per_update = num_tasks_used_per_update
        self.num_context_trajs_for_training = num_context_trajs_for_training
        self.num_test_trajs_for_training = num_test_trajs_for_training

        self.training_regression = training_regression
        self.few_shot_version = few_shot_version
        self.min_context_size = min_context_size
        self.max_context_size = max_context_size
        self.classification_batch_size_per_task = classification_batch_size_per_task

        self.num_tasks_per_eval = num_tasks_per_eval

        self.encoder_optimizer = encoder_optimizer_class(
            self.encoder.parameters(),
            lr=encoder_lr,
            betas=(0.9, 0.999)
        )
        self.mlp_optimizer = mlp_optimizer_class(
            self.mlp.parameters(),
            lr=mlp_lr,
            betas=(0.9, 0.999)
        )

        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.num_update_loops_per_train_call = num_update_loops_per_train_call
        self.num_epochs = num_epochs


    def _get_training_batch(self):
        keys_to_get = ['observations', 'actions', 'next_observations']
        # if self.transfer_version and 'next_observations' not in keys_to_get:
        #     keys_to_get.append('next_observations')
        
        if self.few_shot_version:
            context_batch, task_identifiers_list = self.train_context_expert_replay_buffer.sample_trajs(
                self.max_context_size,
                num_tasks=self.num_tasks_used_per_update,
                keys=keys_to_get
            )
            mask = ptu.Variable(torch.zeros(self.num_tasks_used_per_update, self.max_context_size, 1))
            this_context_sizes = np.random.randint(self.min_context_size, self.max_context_size+1, size=self.num_tasks_used_per_update)
            for i, c_size in enumerate(this_context_sizes):
                mask[i,:c_size,:] = 1.0
        else:
            context_batch, task_identifiers_list = self.train_context_expert_replay_buffer.sample_trajs(
                self.num_context_trajs_for_training,
                num_tasks=self.num_tasks_used_per_update,
                keys=keys_to_get
            )
            mask = None
        
        obs_task_params = np.array(list(map(lambda tid: self.env.task_id_to_obs_task_params(tid), task_identifiers_list)))
        task_params_size = obs_task_params.shape[-1]

        # now need to sample points for classification
        classification_inputs = []
        classification_labels = []
        for task in obs_task_params:
            for _ in range(self.classification_batch_size_per_task):
                good = self.env._sample_color_within_radius(task, self.env.same_color_radius)
                bad = self.env._sample_color_with_min_dist(task, self.env.same_color_radius)
                if np.random.uniform() > 0.5:
                    classification_inputs.append(np.concatenate((good, bad)))
                    classification_labels.append([0])
                else:
                    classification_inputs.append(np.concatenate((bad, good)))
                    classification_labels.append([1])
        classification_inputs = Variable(ptu.from_numpy(np.array(classification_inputs)))
        classification_labels = Variable(ptu.from_numpy(np.array(classification_labels)))

        return context_batch, mask, obs_task_params, classification_inputs, classification_labels
    

    def _get_eval_batch(self, num_tasks, num_trajs_per_task):
        keys_to_get = ['observations', 'actions', 'next_observations']
        # context_batch, task_identifiers_list = self.train_context_expert_replay_buffer.sample_trajs(
        context_batch, task_identifiers_list = self.test_context_expert_replay_buffer.sample_trajs(
            num_trajs_per_task,
            num_tasks=num_tasks,
            keys=keys_to_get
        )
        mask = None
    
        obs_task_params = np.array(list(map(lambda tid: self.env.task_id_to_obs_task_params(tid), task_identifiers_list)))
        task_params_size = obs_task_params.shape[-1]

        # now need to sample points for classification
        classification_inputs = []
        classification_labels = []
        for task in obs_task_params:
            for _ in range(self.classification_batch_size_per_task):
                good = self.env._sample_color_within_radius(task, self.env.same_color_radius)
                bad = self.env._sample_color_with_min_dist(task, self.env.same_color_radius)
                if np.random.uniform() > 0.5:
                    classification_inputs.append(np.concatenate((good, bad)))
                    classification_labels.append([0])
                else:
                    classification_inputs.append(np.concatenate((bad, good)))
                    classification_labels.append([1])
        classification_inputs = Variable(ptu.from_numpy(np.array(classification_inputs)))
        classification_labels = Variable(ptu.from_numpy(np.array(classification_labels)))

        return context_batch, mask, obs_task_params, classification_inputs, classification_labels
    

    def train(self):
        for e in range(self.num_epochs):
            self._do_training(e, self.num_update_loops_per_train_call)
            self.evaluate()


    def _do_training(self, epoch, num_updates):
        '''
            Train the discriminator
        '''
        self.mlp.train()
        self.encoder.train()
        for _ in range(num_updates):
            self.encoder_optimizer.zero_grad()
            self.mlp_optimizer.zero_grad()

            # prep the batches
            context_batch, mask, obs_task_params, classification_inputs, classification_labels = self._get_training_batch()

            post_dist = self.encoder(context_batch, mask)
            z = post_dist.sample() # N_tasks x Dim
            # z = post_dist.mean

            obs_task_params = Variable(ptu.from_numpy(obs_task_params))


            if self.training_regression:
                preds = self.mlp(z)
                loss = self.mse(preds, obs_task_params)
            else:
                repeated_z = z.repeat(1, self.classification_batch_size_per_task).view(-1, z.size(1))
                mlp_input = torch.cat([classification_inputs, repeated_z], dim=-1)
                preds = self.mlp(mlp_input)
                loss = self.bce(preds, classification_labels)
            loss.backward()

            self.mlp_optimizer.step()
            self.encoder_optimizer.step()


    def evaluate(self):
        eval_statistics = OrderedDict()
        self.mlp.eval()
        self.encoder.eval()
        # for i in range(self.min_context_size, self.max_context_size+1):
        for i in range(1, 12):
            # prep the batches
            context_batch, mask, obs_task_params, classification_inputs, classification_labels = self._get_eval_batch(self.num_tasks_per_eval, i)
            # print(len(context_batch))
            # print(len(context_batch[0]))

            post_dist = self.encoder(context_batch, mask)
            z = post_dist.sample() # N_tasks x Dim
            # z = post_dist.mean

            obs_task_params = Variable(ptu.from_numpy(obs_task_params))
            # print(obs_task_params)

            if self.training_regression:
                preds = self.mlp(z)
                loss = self.mse(preds, obs_task_params)
                eval_statistics['Loss for %d' % i] = np.mean(ptu.get_numpy(loss))
            else:
                repeated_z = z.repeat(1, self.classification_batch_size_per_task).view(-1, z.size(1))
                mlp_input = torch.cat([classification_inputs, repeated_z], dim=-1)
                preds = self.mlp(mlp_input)
                # loss = self.bce(preds, classification_labels)
                class_preds = (preds > 0).type(preds.data.type())
                accuracy = (class_preds == classification_labels).type(torch.FloatTensor).mean()
                eval_statistics['Acc for %d' % i] = np.mean(ptu.get_numpy(accuracy))

        for key, value in eval_statistics.items():
            logger.record_tabular(key, value)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    
    def cuda(self):
        self.encoder.cuda()
        self.mlp.cuda()
    

    def cpu(self):
        self.encoder.cpu()
        self.mlp.cpu()


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
