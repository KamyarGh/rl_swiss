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

from rlkit.torch.core import PyTorchModule

from rlkit.torch.sac.policies import PostCondMLPPolicyWrapper
from rlkit.data_management.path_builder import PathBuilder
from gym.spaces import Dict

from rlkit.torch.irl.encoders.aggregators import sum_aggregator
from rlkit.torch.distributions import ReparamMultivariateNormalDiag

INPUT_DIM = 4

# USE_FIXED_TRAIN_SET = True
# TRAIN_SET_SIZE = 128
# FIXED_TRAIN_SET = np.random.normal(size=(num, INPUT_DIM))

# 5 works very well
# for evaluation need to for each task episode subtract performance with least context
# from subsequent context ammounts and report the mean and std of those deltas instead
# check on finite tasks

class R2ZMap(PyTorchModule):
    def __init__(
        self,
        r_dim,
        z_dim,
        hid_dim,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(r_dim, hid_dim),
            # nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            # nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, z_dim),
        )

    def forward(self, r):
        trunk_output = self.trunk(r)
        return trunk_output


class Encoder(PyTorchModule):
    def __init__(self, z_dim):
        self.save_init_params(locals())
        super().__init__()

        HID_DIM = 128
        self.encoder_mlp = nn.Sequential(
            nn.Linear(2*INPUT_DIM, HID_DIM),
            # nn.BatchNorm1d(HID_DIM),
            nn.ReLU(),
            nn.Linear(HID_DIM, HID_DIM),
            # nn.BatchNorm1d(HID_DIM),
            nn.ReLU(),
            nn.Linear(HID_DIM, HID_DIM)
        )
        self.agg = sum_aggregator
        self.r2z_map = R2ZMap(HID_DIM, z_dim, HID_DIM)


    def forward(self, context, mask):
        N_tasks, N_max_cont, N_dim = context.size(0), context.size(1), context.size(2)
        context = context.view(-1, N_dim)
        embedded_context = self.encoder_mlp(context)

        # embedded_context = self.encoder_mlp(context)
        embed_dim = embedded_context.size(1)
        embedded_context = embedded_context.view(N_tasks, N_max_cont, embed_dim)
        
        agg = self.agg(embedded_context, mask)
        return self.r2z_map(agg)


class Classifier(PyTorchModule):
    def __init__(self, z_dim):
        self.save_init_params(locals())
        super().__init__()
        
        self.z_dim = z_dim

        HID_DIM = 128
        self.mlp = nn.Sequential(
            nn.Linear(2*INPUT_DIM + z_dim, HID_DIM),
            # nn.BatchNorm1d(HID_DIM),
            nn.ReLU(),
            nn.Linear(HID_DIM, HID_DIM),
            # nn.BatchNorm1d(HID_DIM),
            nn.ReLU(),
            nn.Linear(HID_DIM, 1)
        )


    def forward(self, input_batch):
        return self.mlp(input_batch)


class FetchLinClassTaskDesign():
    def __init__(
            self,
            num_tasks_used_per_update=16,
            min_context_size=1,
            max_context_size=8,
            classification_batch_size_per_task=64,

            encoder_lr=1e-3,
            encoder_optimizer_class=optim.Adam,

            mlp_lr=1e-3,
            mlp_optimizer_class=optim.Adam,

            num_update_loops_per_train_call=1000,
            num_epochs=10000,

            z_dim=16,

            **kwargs
    ):
        # self.mlp = mlp
        self.mlp = Classifier(z_dim)
        self.encoder = Encoder(z_dim)

        self.num_tasks_used_per_update = num_tasks_used_per_update
        self.min_context_size = min_context_size
        self.max_context_size = max_context_size
        self.classification_batch_size_per_task = classification_batch_size_per_task

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

        self.bce = nn.BCEWithLogitsLoss()
        self.num_update_loops_per_train_call = num_update_loops_per_train_call
        self.num_epochs = num_epochs


    def _sample_tasks(self, num):
        tasks = np.random.normal(size=(num, INPUT_DIM))
        tasks /= np.linalg.norm(tasks, axis=-1, keepdims=True)
        return tasks
    

    def _sample_points_for_task(self, task, batch_size):
        positives = []
        negatives = []
        while len(positives) < batch_size:
            x = 2*np.random.uniform(size=INPUT_DIM) - 1
            if np.dot(task, x) > 0:
                positives.append(x)
        
        while len(negatives) < batch_size:
            x = 2*np.random.uniform(size=INPUT_DIM) - 1
            if np.dot(task, x) < 0:
                negatives.append(x)

        return np.array(positives), np.array(negatives)


    def _get_batch(self, csize=None):
        tasks = self._sample_tasks(self.num_tasks_used_per_update)

        input_batch = []
        labels = []
        for task in tasks:
            positives, negatives = self._sample_points_for_task(task, self.classification_batch_size_per_task)
            for i in range(self.classification_batch_size_per_task):
                pos = positives[i]
                neg = negatives[i]
                if np.random.uniform() > 0.5:
                    input_batch.append(np.concatenate((pos, neg)))
                    labels.append([0.0])
                else:
                    input_batch.append(np.concatenate((neg, pos)))
                    labels.append([1.0])
        input_batch = Variable(ptu.from_numpy(np.array(input_batch)))
        labels = Variable(ptu.from_numpy(np.array(labels)))
        
        context = []
        mask = Variable(ptu.from_numpy(np.zeros((self.num_tasks_used_per_update, self.max_context_size, 1))))
        for task_idx, task in enumerate(tasks):
            task_context = []
            positives, negatives = self._sample_points_for_task(task, self.max_context_size)
            for i in range(self.max_context_size):
                pos = positives[i]
                neg = negatives[i]
                task_context.append(np.concatenate((pos, neg)))
            context.append(task_context)

            if csize is not None:
                con_size = csize
            else:
                con_size = np.random.randint(self.min_context_size, self.max_context_size+1)
            mask[task_idx,:con_size,:] = 1.0
        context = Variable(ptu.from_numpy(np.array(context)))

        return context, mask, input_batch, labels
    
    def _get_training_batch(self):
        return self._get_batch()

    def _get_eval_batch(self, eval_context_size):
        return self._get_batch(csize=eval_context_size)


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
            context, mask, input_batch, labels = self._get_training_batch()
            z = self.encoder(context, mask)

            repeated_z = z.repeat(1, self.classification_batch_size_per_task).view(-1, z.size(1))
            mlp_input = torch.cat([input_batch, repeated_z], dim=-1)
            preds = self.mlp(mlp_input)
            loss = self.bce(preds, labels)
            loss.backward()

            self.mlp_optimizer.step()
            self.encoder_optimizer.step()


    def evaluate(self):
        eval_statistics = OrderedDict()
        self.mlp.eval()
        self.encoder.eval()
        # for i in range(self.min_context_size, self.max_context_size):
        MAX_NUM = 9
        context, mask, input_batch, labels = self._get_batch()
        for i in range(1, MAX_NUM):
            # prep the batches
            # context, mask, input_batch, labels = self._get_training_batch()
            # context, mask, input_batch, labels = self._get_eval_batch(i)

            mask = Variable(ptu.from_numpy(np.zeros((self.num_tasks_used_per_update, self.max_context_size, 1))))
            mask[:,:i,:] = 1.0
            z = self.encoder(context, mask)

            repeated_z = z.repeat(1, self.classification_batch_size_per_task).view(-1, z.size(1))
            mlp_input = torch.cat([input_batch, repeated_z], dim=-1)
            preds = self.mlp(mlp_input)
            class_preds = (preds > 0).type(preds.data.type())
            accuracy = (class_preds == labels).type(torch.FloatTensor).mean()
            eval_statistics['Acc for %d' % i] = np.mean(ptu.get_numpy(accuracy))

        for key, value in eval_statistics.items():
            logger.record_tabular(key, value)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
        # print(np.mean(list(eval_statistics.values())))
        print('INPUT_DIM: %s' % INPUT_DIM)
    
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
