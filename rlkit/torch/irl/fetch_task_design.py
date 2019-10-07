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

OUTER_RADIUS = 2.0
TASK_RADIUS = 2.0
SAME_COLOUR_RADIUS = 1.0


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


class R2ZMap(PyTorchModule):
    def __init__(
        self,
        r_dim,
        z_dim,
        hid_dim,
        # this makes it be closer to deterministic, makes it easier to train
        # before we turn on the KL regularization
        LOG_STD_SUBTRACT_VALUE=2.0
    ):
        self.save_init_params(locals())
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(r_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
        )
        self.mean_fc = nn.Linear(hid_dim, z_dim)
        self.log_sig_fc = nn.Linear(hid_dim, z_dim)

        self.LOG_STD_SUBTRACT_VALUE = LOG_STD_SUBTRACT_VALUE
        print('LOG STD SUBTRACT VALUE IS FOR APPROX POSTERIOR IS %f' % LOG_STD_SUBTRACT_VALUE)


    def forward(self, r):
        trunk_output = self.trunk(r)
        mean = self.mean_fc(trunk_output)
        log_sig = self.log_sig_fc(trunk_output) - self.LOG_STD_SUBTRACT_VALUE
        return mean, log_sig


class Encoder(PyTorchModule):
    def __init__(self, z_dim):
        self.save_init_params(locals())
        super().__init__()

        HID_DIM = 64
        self.encoder_mlp = nn.Sequential(
            nn.Linear(6, HID_DIM),
            nn.BatchNorm1d(HID_DIM),
            nn.ReLU(),
            nn.Linear(HID_DIM, HID_DIM),
            nn.BatchNorm1d(HID_DIM),
            nn.ReLU(),
            nn.Linear(HID_DIM, HID_DIM)
        )
        self.agg = sum_aggregator
        self.r2z_map = R2ZMap(HID_DIM, z_dim, HID_DIM)


    def forward(self, context, mask):
        N_tasks, N_max_cont, N_dim = context.size(0), context.size(1), context.size(2)
        context = context.view(-1, N_dim)
        embedded_context = self.encoder_mlp(context)
        embed_dim = embedded_context.size(1)
        embedded_context = embedded_context.view(N_tasks, N_max_cont, embed_dim)
        
        agg = self.agg(embedded_context, mask)
        post_mean, post_log_sig = self.r2z_map(agg)
        return ReparamMultivariateNormalDiag(post_mean, post_log_sig)


class FetchTaskDesign():
    def __init__(
            self,
            mlp,

            num_tasks_used_per_update=5,
            min_context_size=1,
            max_context_size=5,
            classification_batch_size_per_task=32,

            encoder_lr=1e-3,
            encoder_optimizer_class=optim.Adam,

            mlp_lr=1e-3,
            mlp_optimizer_class=optim.Adam,

            num_update_loops_per_train_call=1000,
            num_epochs=10000,

            z_dim=16,

            **kwargs
    ):
        self.mlp = mlp
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


    def _sample_color_within_radius(self, center, radius):
        new_color = self._uniform_sample_from_sphere(radius) + center
        while np.linalg.norm(new_color) > OUTER_RADIUS:
            new_color = self._uniform_sample_from_sphere(radius) + center
        return new_color

    
    def _uniform_sample_from_sphere(self, radius):
        x = np.random.normal(size=3)
        x /= np.linalg.norm(x, axis=-1)
        r = radius
        u = np.random.uniform()
        sampled_color = r * (u**(1.0/3.0)) * x
        return sampled_color
    

    def _sample_color_with_min_dist(self, color, min_dist):
        new_color = self._uniform_sample_from_sphere(OUTER_RADIUS)
        while np.linalg.norm(new_color - color, axis=-1) < min_dist:
            new_color = self._uniform_sample_from_sphere(OUTER_RADIUS)
        return new_color


    def _get_training_batch(self):
        task_colors = []
        for _ in range(self.num_tasks_used_per_update):
            task_colors.append(self._uniform_sample_from_sphere(TASK_RADIUS))
        task_colors = np.array(task_colors)

        input_batch = []
        labels = []
        for task in task_colors:
            for _ in range(self.classification_batch_size_per_task):
                good = self._sample_color_within_radius(task, SAME_COLOUR_RADIUS)
                bad = self._sample_color_with_min_dist(task, 0.0) # HERE
                if np.random.uniform() > 0.5:
                    input_batch.append(np.concatenate((good, bad)))
                    labels.append([1.0])
                else:
                    input_batch.append(np.concatenate((bad, good)))
                    labels.append([0.0])
        input_batch = Variable(ptu.from_numpy(np.array(input_batch)))
        labels = Variable(ptu.from_numpy(np.array(labels)))
        
        context = []
        mask = Variable(ptu.from_numpy(np.zeros((self.num_tasks_used_per_update, self.max_context_size, 1))))
        for task_num, task in enumerate(task_colors):
            task_context = []
            for _ in range(self.max_context_size):
                good = self._sample_color_within_radius(task, SAME_COLOUR_RADIUS)
                bad = self._sample_color_with_min_dist(task, 0.0) # HERE
                # always the same order because it's the context
                task_context.append(np.concatenate((good, bad)))
            context.append(task_context)

            con_size = np.random.randint(self.min_context_size, self.max_context_size+1)
            mask[task_num,:con_size,:] = 1.0
        context = Variable(ptu.from_numpy(np.array(context)))

        return context, mask, input_batch, labels
    

    def _get_eval_batch(self):
        task_colors = []
        for _ in range(self.num_tasks_used_per_update):
            task_colors.append(self._uniform_sample_from_sphere(TASK_RADIUS))
        task_colors = np.array(task_colors)
        
        # task_colors = np.zeros((self.num_tasks_used_per_update, 3)) # THIS
        # task_colors[:,0] = -1.0 # THIS
        
        input_batch = []
        labels = []
        for task in task_colors:
            for _ in range(self.classification_batch_size_per_task):
                good = self._sample_color_within_radius(task, SAME_COLOUR_RADIUS)
                bad = self._sample_color_with_min_dist(task, SAME_COLOUR_RADIUS)
                if np.random.uniform() > 0.5:
                    input_batch.append(np.concatenate((good, bad)))
                    labels.append([1.0])
                else:
                    input_batch.append(np.concatenate((bad, good)))
                    labels.append([0.0])
        input_batch = Variable(ptu.from_numpy(np.array(input_batch)))
        labels = Variable(ptu.from_numpy(np.array(labels)))
        
        context = []
        mask = Variable(ptu.from_numpy(np.zeros((self.num_tasks_used_per_update, self.max_context_size, 1))))
        for task_num, task in enumerate(task_colors):
            task_context = []
            for _ in range(self.max_context_size):
                good = self._sample_color_within_radius(task, SAME_COLOUR_RADIUS)

                # good = np.zeros(3) # THIS

                bad = self._sample_color_with_min_dist(task, 0.0) # HERE

                # bad = np.array([2.0, 0.0, 0.0]) # THIS

                # always the same order because it's the context
                task_context.append(np.concatenate((good, bad)))
            context.append(task_context)

            con_size = np.random.randint(self.min_context_size, self.max_context_size+1)
            mask[task_num,:con_size,:] = 1.0
        context = Variable(ptu.from_numpy(np.array(context)))

        return context, mask, input_batch, labels


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
            post_dist = self.encoder(context, mask)

            # z = post_dist.sample() # N_tasks x Dim
            z = post_dist.mean

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
        for i in range(1, 12):
            # prep the batches
            # context, mask, input_batch, labels = self._get_training_batch()
            context, mask, input_batch, labels = self._get_eval_batch()
            post_dist = self.encoder(context, mask)

            # z = post_dist.sample() # N_tasks x Dim
            z = post_dist.mean

            repeated_z = z.repeat(1, self.classification_batch_size_per_task).view(-1, z.size(1))
            mlp_input = torch.cat([input_batch, repeated_z], dim=-1)
            preds = self.mlp(mlp_input)
            class_preds = (preds > 0).type(preds.data.type())
            accuracy = (class_preds == labels).type(torch.FloatTensor).mean()
            eval_statistics['Acc for %d' % i] = np.mean(ptu.get_numpy(accuracy))

        # for key, value in eval_statistics.items():
        #     logger.record_tabular(key, value)
        # logger.dump_tabular(with_prefix=False, with_timestamp=False)
        print(np.mean(list(eval_statistics.values())))

    
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
