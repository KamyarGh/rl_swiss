import os.path as osp
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd
from torch.optim import Adam

import numpy as np
from numpy import array
from numpy.random import choice, randint

from generic_map import GenericMap
from base_map import BaseMap
from neural_process import NeuralProcessV1
from aggregators import sum_aggregator, mean_aggregator, tanh_sum_aggregator

from tasks.sinusoidal import SinusoidalTask
from tasks.linear import LinearTask

from rlkit.core.vistools import save_plot, plot_returns_on_same_plot
from neural_processes.distributions import sample_diag_gaussians, local_repeat

# -----------------------------------------------------------------------------
N_tasks = 100
base_map_lr = encoder_lr = r_to_z_map_lr = 1e-3
max_iters = 10001
num_tasks_per_batch = 64
replace = False

data_sampling_mode = 'constant'
num_per_task_high = 10

# -----------------------------------------------------------------------------
slopes = np.linspace(-1, 1, N_tasks)
all_tasks = [LinearTask(slope) for slope in slopes]
def generate_data_batch(tasks_batch, num_samples_per_task, max_num):
    # Very inefficient will need to fix this
    X = torch.zeros(len(tasks_batch), max_num, 1)
    Y = torch.zeros(len(tasks_batch), max_num, 1)
    for i, (task, num_samples) in enumerate(zip(tasks_batch, num_samples_per_task)):
        num = int(num_samples)
        x, y = task.sample(num)
        if num==max_num:
            X[i,:] = x
            Y[i,:] = y
        else:
            X[i,:num] = x
            Y[i,:num] = y

    return Variable(X), Variable(Y)

# -----------------------------------------------------------------------------
enc_dim = 40
encoder = nn.Sequential(
    nn.Linear(2, enc_dim),
    nn.BatchNorm1d(enc_dim),
    nn.ReLU(),
    nn.Linear(enc_dim, enc_dim),
    nn.BatchNorm1d(enc_dim),
    nn.ReLU(),
    nn.Linear(enc_dim, enc_dim)
)
class R2Z(nn.Module):
    def __init__(self):
        super(R2Z, self).__init__()
        dim = 40
        self.hidden = nn.Sequential(
            nn.Linear(enc_dim, dim),
            nn.ReLU(),
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(dim, 1)
        self.log_cov_layer = nn.Linear(dim, 1)
    
    def forward(self, r):
        hid_out = self.hidden(r)
        return self.mean_layer(hid_out), self.log_cov_layer(hid_out)
r_to_z_map = R2Z()

# class Z2W(nn.Module):
#     def __init__(self):
#         super(Z2W, self).__init__()
#         self.z_l1 = nn.Linear(40,10)
#         self.z_l2 = nn.Linear(40,100)
#         self.z_l3 = nn.Linear(40,10)
    
#     def forward(self, z):
#         return self.z_l1(z), self.z_l2(z), self.z_l3(z)
# z2w = Z2W()


# class BaseMap(nn.Module):
#     def __init__(self):
#         super(BaseMap, self).__init__()
#         dim = 200
#         self.hidden = nn.Sequential(
#             nn.Linear(41, dim),
#             nn.BatchNorm1d(dim),
#             nn.ReLU(),
#             nn.Linear(dim, dim),
#             nn.BatchNorm1d(dim),
#             nn.ReLU(),
#             nn.Linear(dim, dim),
#             nn.BatchNorm1d(dim),
#             nn.ReLU(),
#             nn.Linear(dim, dim),
#             nn.BatchNorm1d(dim),
#             nn.ReLU(),
#             nn.Linear(dim, 1)
#         )
    
#     def forward(self, z, x):
#         return self.hidden(torch.cat([z,x],1))
# base_map = BaseMap()

encoder_optim = Adam(encoder.parameters(), lr=encoder_lr)
r_to_z_map_optim = Adam(r_to_z_map.parameters(), lr=r_to_z_map_lr)
# z2w_optim = Adam(z2w.parameters(), lr=base_map_lr)
# base_map_optim = Adam(base_map.parameters(), lr=base_map_lr)
# -----------------------------------------------------------------------------
test_elbos = defaultdict(list)
test_log_likelihoods = defaultdict(list)
for iter_num in range(max_iters):
    task_batch_idxs = choice(len(all_tasks), size=num_tasks_per_batch, replace=replace)
    num_samples_per_task = array([num_per_task_high for _ in range(num_tasks_per_batch)])
    max_num = num_per_task_high
    
    X, Y = generate_data_batch([all_tasks[i] for i in task_batch_idxs], num_samples_per_task, max_num)
    N_tasks, N_samples, X_dim = X.size(0), X.size(1), X.size(2)
    Y_dim = Y.size(2)
    X = X.view(N_tasks*N_samples, X_dim)
    Y = Y.view(N_tasks*N_samples, Y_dim)

    encoder_optim.zero_grad()
    r_to_z_map_optim.zero_grad()
    # z2w_optim.zero_grad()
    # base_map_optim.zero_grad()

    r = encoder(torch.cat([X,Y], 1))
    r_dim = r.size(-1)
    r = r.view(N_tasks, N_samples, r_dim)
    r = torch.mean(r, 1)
    # r = torch.sum(r, 1)
    mean, log_cov = r_to_z_map(r)
    cov = torch.exp(log_cov)

    z = Variable(torch.randn(mean.size())) * cov + mean
    z = local_repeat(z, N_samples)

    # Y_pred = base_map(z, X)

    Y_pred = X * z

    # w1, w2, w3 = z2w(z)
    # w1 = w1.view(-1, 1, w1.size(1))
    # w3 = w3.view(-1, w1.size(-1), 1)
    # Y_pred = torch.matmul(X.view(-1,1,1), w1)
    # Y_pred = torch.matmul(Y_pred, w3)
    # Y_pred = Y_pred.view(-1,1)

    KL = -0.5 * torch.sum(
        1.0 + log_cov - mean**2 - cov
    )

    cond_log_likelihood = -0.5 * torch.sum((Y_pred - Y)**2)

    neg_elbo = -1.0 * (cond_log_likelihood - KL) / float(N_tasks)
    neg_elbo.backward()

    # base_map_optim.step()
    # z2w_optim.step()
    r_to_z_map_optim.step()
    encoder_optim.step()

    if iter_num % 100 == 0:
        print('\nIter %d' % iter_num)
        print('LL: %.4f' % cond_log_likelihood)
        print('KL: %.4f' % KL)
        print('ELBO: %.4f' % (-1.0*neg_elbo))
        print('MSE: %.4f' % (cond_log_likelihood * -2 / (N_tasks * N_samples)))
