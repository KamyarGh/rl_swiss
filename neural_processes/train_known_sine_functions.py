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
N_val = 3
sample_one_z_per_sample = True
base_map_lr = encoder_lr = r_to_z_map_lr = 1e-3
max_iters = 10001
num_tasks_per_batch = 64
replace = True

data_sampling_mode = 'constant'
num_per_task_high = 10

# -----------------------------------------------------------------------------
all_tasks = [SinusoidalTask() for _ in range(N_tasks)]
def generate_data_batch(tasks_batch, num_samples_per_task, max_num):
    # Very inefficient will need to fix this
    X = torch.zeros(len(tasks_batch), max_num, 3)
    Y = torch.zeros(len(tasks_batch), max_num, 1)
    for i, (task, num_samples) in enumerate(zip(tasks_batch, num_samples_per_task)):
        num = int(num_samples)
        x, y = task.sample(num)
        A, phase = torch.ones(x.size()) * task.A, torch.ones(x.size()) * task.phase
        inputs = torch.cat([x, A, phase], -1)

        if num==max_num:
            X[i,:] = inputs
            Y[i,:] = y
        else:
            X[i,:num] = inputs
            Y[i,:num] = y

    return Variable(X), Variable(Y)

# -----------------------------------------------------------------------------
dim = 100
model = nn.Sequential(
    nn.Linear(3, dim, bias=True),
    nn.BatchNorm1d(dim),
    nn.ReLU(),
    nn.Linear(dim, dim, bias=True),
    nn.BatchNorm1d(dim),
    nn.ReLU(),
    nn.Linear(dim, dim, bias=True),
    nn.BatchNorm1d(dim),
    nn.ReLU(),
    nn.Linear(dim, dim, bias=False),
    nn.BatchNorm1d(dim),
    nn.ReLU(),
    nn.Linear(dim, 1)
)
model_optim = Adam(model.parameters(), lr=1e-3)
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

    model_optim.zero_grad()
    Y_pred = model(X)

    cond_log_likelihood = -0.5 * torch.sum((Y_pred - Y)**2)
    loss = -1.0 * cond_log_likelihood / float(N_tasks)
    loss.backward()

    model_optim.step()

    if iter_num % 100 == 0:
        print('\nIter %d' % iter_num)
        print('LL: %.4f' % cond_log_likelihood)
        print('MSE: %.4f' % (cond_log_likelihood * -2 / (N_tasks * N_samples)))

    if iter_num % 1000 == 0:
        task_batch_idxs = choice(len(all_tasks), size=N_val, replace=False)
        val_tasks = [all_tasks[i] for i in task_batch_idxs]
        model.eval()

        X_test = torch.linspace(-5, 5, 100)
        X_test = X_test.view(-1, 1)
        X_test = [
            torch.cat(
                [
                    X_test,
                    torch.ones(X_test.size())*task.A,
                    torch.ones(X_test.size())*task.phase
                ],
                -1
            ) for task in val_tasks
        ]
        X_test = torch.cat(X_test, 0).view(-1,3)
        X_test = Variable(X_test)

        Y_pred = model(X_test)

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

        model.train()
