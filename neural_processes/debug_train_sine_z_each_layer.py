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

from rlkit.core.vistools import save_plot, plot_returns_on_same_plot, plot_multiple_plots
from neural_processes.distributions import sample_diag_gaussians, local_repeat

from numpy.random import choice

# -----------------------------------------------------------------------------
N_tasks = 100
N_val = 2
sample_one_z_per_sample = True
base_map_lr = encoder_lr = r_to_z_map_lr = 1e-3
max_iters = 50001
num_tasks_per_batch = 64
replace = True

data_sampling_mode = 'constant'
num_per_task_high = 10

# -----------------------------------------------------------------------------
all_tasks = [SinusoidalTask() for _ in range(N_tasks)]
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
    nn.Linear(enc_dim, enc_dim),
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
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(dim, 40)
        self.log_cov_layer = nn.Linear(dim, 40)
    
    def forward(self, r):
        hid_out = self.hidden(r)
        return self.mean_layer(hid_out), self.log_cov_layer(hid_out)
r_to_z_map = R2Z()

class BaseMap(nn.Module):
    def __init__(self):
        super(BaseMap, self).__init__()
        dim = 160
        mod_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(41, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                )
            ]
        )
        for _ in range(3):
            mod_list.extend(
                [
                    nn.Sequential(
                        nn.Linear(dim+40, dim),
                        nn.BatchNorm1d(dim),
                        nn.ReLU(),
                    )
                ]
            )
        self.mod_list = mod_list
        self.last = nn.Linear(dim, 1)
    
    def forward(self, z, x):
        out = x
        for mod in self.mod_list:
            out = mod(torch.cat([z,out], 1))
        out = self.last(out)
        return out

base_map = BaseMap()

encoder_optim = Adam(encoder.parameters(), lr=encoder_lr)
r_to_z_map_optim = Adam(r_to_z_map.parameters(), lr=r_to_z_map_lr)
base_map_optim = Adam(base_map.parameters(), lr=base_map_lr)
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
    base_map_optim.zero_grad()

    r = encoder(torch.cat([X,Y], 1))

    r_dim = r.size(-1)
    r = r.view(N_tasks, N_samples, r_dim)
    # r = torch.mean(r, 1)
    r = torch.sum(r, 1)
    mean, log_cov = r_to_z_map(r)
    cov = torch.exp(log_cov)

    if not sample_one_z_per_sample:
        z = Variable(torch.randn(mean.size())) * cov + mean
        z = local_repeat(z, N_samples)
    else:
        rep_mean = local_repeat(mean, N_samples)
        rep_cov = local_repeat(cov, N_samples)
        z = Variable(torch.randn(rep_mean.size())) * rep_cov + rep_mean

    Y_pred = base_map(z, X)

    KL = -0.5 * torch.sum(
        1.0 + log_cov - mean**2 - cov
    )

    cond_log_likelihood = -0.5 * torch.sum((Y_pred - Y)**2)

    neg_elbo = -1.0 * (cond_log_likelihood - KL) / float(N_tasks)
    neg_elbo.backward()

    base_map_optim.step()
    r_to_z_map_optim.step()
    encoder_optim.step()

    if iter_num % 100 == 0:
        print('\nIter %d' % iter_num)
        print('LL: %.4f' % cond_log_likelihood)
        print('KL: %.4f' % KL)
        print('ELBO: %.4f' % (-1.0*neg_elbo))
        print('MSE: %.4f' % (cond_log_likelihood * -2 / (N_tasks * N_samples)))

    if iter_num % 1000 == 0:
        task_batch_idxs = choice(len(all_tasks), size=N_val, replace=False)
        encoder.eval()
        r_to_z_map.eval()
        base_map.eval()

        X, Y = generate_data_batch([all_tasks[i] for i in task_batch_idxs], num_samples_per_task, max_num)
        N_tasks, N_samples, X_dim = X.size(0), X.size(1), X.size(2)
        Y_dim = Y.size(2)
        X = X.view(N_tasks*N_samples, X_dim)
        Y = Y.view(N_tasks*N_samples, Y_dim)

        r = encoder(torch.cat([X,Y], 1))
        
        r_dim = r.size(-1)
        r = r.view(N_tasks, N_samples, r_dim)
        r = torch.mean(r, 1)
        # r = torch.sum(r, 1)
        mean, log_cov = r_to_z_map(r)
        cov = torch.exp(log_cov)

        X_test = Variable(torch.linspace(-5, 5, 100))
        X_test = X_test.repeat(N_val).view(-1,1)
        z = mean # at test time we take the mean
        z = local_repeat(z, 100)
        Y_pred = base_map(z, X_test)

        plots_to_plot = []
        plot_names = []
        for i, idx in enumerate(task_batch_idxs):
            y_true = all_tasks[idx].A * np.sin(np.linspace(-5,5,100) - all_tasks[idx].phase)
            plots_to_plot.append(
                [
                    np.linspace(-5,5,100),
                    y_true
                ]
            )
            plot_names.append('true %d' % i)

            plots_to_plot.append(
                [
                    np.linspace(-5,5,100),
                    Y_pred[i*100:(i+1)*100].view(-1).data.numpy()
                ]
            )
            plot_names.append('pred %d' % i)

        plot_multiple_plots(
            plots_to_plot,
            plot_names,
            'debug iter %d' % iter_num,
            'junk_vis/debug_iter_%d.png' % iter_num
        )

        encoder.train()
        r_to_z_map.train()
        base_map.train()
