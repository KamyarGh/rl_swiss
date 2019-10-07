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

from neural_processes.generic_map import GenericMap
from neural_processes.base_map import BaseMap
from neural_processes.neural_process import NeuralProcessV1
from neural_processes.aggregators import sum_aggregator, mean_aggregator, tanh_sum_aggregator

from neural_processes.tasks.sinusoidal import SinusoidalTask

from rlkit.core.vistools import save_plot, plot_returns_on_same_plot, plot_multiple_plots
from rlkit.launchers.launcher_util import create_log_dir


def get_agg(agg_name):
    if agg_name == 'sum_aggregator':
        return sum_aggregator
    elif agg_name == 'mean_aggregator':
        return mean_aggregator
    elif agg_name == 'tanh_sum_aggregator':
        return tanh_sum_aggregator
    raise Exception


def exp_fn(all_args):
    exp_specs, exp_id = all_args[0], all_args[1]
    # -----------------------------------------------------------------------------
    exp_dir = create_log_dir(
        exp_specs['exp_name'], exp_id=exp_id,
        base_log_dir=exp_specs['exp_dirs']
    )

    use_bn = exp_specs['use_bn']
    N_tasks = int(float(exp_specs['N_tasks']))
    r_dim = int(float(exp_specs['r_dim']))
    z_dim = int(float(exp_specs['z_dim']))
    base_map_lr = float(exp_specs['lr'])
    encoder_lr = float(exp_specs['lr'])
    r_to_z_map_lr = float(exp_specs['lr'])
    max_iters = int(float(exp_specs['max_iters']))
    num_tasks_per_batch = int(float(exp_specs['num_tasks_per_batch']))

    data_sampling_mode = exp_specs['data_sampling_mode']
    num_per_task_low = int(float(exp_specs['num_per_task_low']))
    num_per_task_high = int(float(exp_specs['num_per_task_high']))

    aggregator = get_agg(exp_specs['aggregator'])

    freq_val = int(float(exp_specs['freq_val']))

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

    def generate_mask(num_tasks_per_batch, max_num):
        mask = torch.ones(num_tasks_per_batch.shape[0], max_num, 1)
        for i, num in enumerate(num_tasks_per_batch):
            if num == max_num: continue
            mask[i,num:] = 0.0
        return Variable(mask)

    # -----------------------------------------------------------------------------
    encoder = GenericMap(
        [1,1], [r_dim], siamese_input=False,
        num_hidden_layers=exp_specs['num_enc_hid_layers'], hidden_dim=exp_specs['enc_hid_dim'],
        siamese_output=False, act='relu',
        deterministic=True,
        use_bn=use_bn
    )
    encoder_optim = Adam(encoder.parameters(), lr=encoder_lr)

    base_map = BaseMap(
        z_dim, [1], [1], siamese_input=False,
        num_hidden_layers=exp_specs['num_base_map_hid_layers'], hidden_dim=exp_specs['base_map_hid_dim'],
        siamese_output=False, act='relu',
        deterministic=True,
        use_bn=use_bn
    )
    base_map_optim = Adam(base_map.parameters(), lr=base_map_lr)

    r_to_z_map = GenericMap(
        [r_dim], [z_dim], siamese_input=False,
        num_hidden_layers=exp_specs['num_r_to_z_map_hid_layers'], hidden_dim=exp_specs['r_to_z_map_hid_dim'],
        siamese_output=False, act='relu',
        deterministic=False,
        use_bn=use_bn
    )
    r_to_z_map_optim = Adam(r_to_z_map.parameters(), lr=r_to_z_map_lr)

    neural_process = NeuralProcessV1(
        encoder,
        encoder_optim,
        aggregator,
        r_to_z_map,
        r_to_z_map_optim,
        base_map,
        base_map_optim,
        use_nat_grad=False
    )

    # -----------------------------------------------------------------------------
    test_elbos = defaultdict(list)
    test_log_likelihoods = defaultdict(list)
    for iter_num in range(max_iters):
        task_batch_idxs = choice(len(all_tasks), size=num_tasks_per_batch, replace=exp_specs['choice_replace'])
        if data_sampling_mode == 'random':
            num_samples_per_task = randint(
                num_per_task_low,
                high=num_per_task_high,
                size=(num_tasks_per_batch)
            )
            max_num = num_per_task_high
        else:
            max_num = num_per_task_high
            num_samples_per_task = array([max_num for _ in range(num_tasks_per_batch)])
        
        X, Y = generate_data_batch([all_tasks[i] for i in task_batch_idxs], num_samples_per_task, max_num)
        mask = generate_mask(num_samples_per_task, max_num)
        batch = {
            'input_batch_list': [X],
            'output_batch_list': [Y],
            'mask': mask
        }
        neural_process.train_step(batch)

        if iter_num % freq_val == 0:
            print('-'*80)
            print('Iter %d' % iter_num)
            neural_process.set_mode('eval')

            # get test samples
            NUM_TEST_SAMPLES = 30
            val_tasks = [SinusoidalTask() for _ in range(num_tasks_per_batch)]
            X_test = Variable(torch.arange(-5,5,0.1)).view(-1,1)
            Y_test = [
                task.A * torch.sin(X_test - task.phase)
                for task in val_tasks
            ]
            Y_test = torch.stack(Y_test)
            X_test = X_test.unsqueeze(0).expand(len(val_tasks), -1, -1).contiguous()
            mask_test = generate_mask(array([NUM_TEST_SAMPLES for _ in range(len(val_tasks))]), X_test.size(1))
            batch_test = {
                'input_batch_list': [X_test],
                'output_batch_list': [Y_test],
                'mask': mask_test
            }

            max_num_context = 21
            num_samples_per_task = array([max_num_context for _ in range(num_tasks_per_batch)])
            X_context, Y_context = generate_data_batch(val_tasks, num_samples_per_task, max_num_context)
            mask_context = generate_mask(num_samples_per_task, max_num_context)

            for num_context in range(1,max_num_context+1,4):
                print('-'*5)
                X, Y, mask = X_context[:,:num_context,:], Y_context[:,:num_context,:], mask_context[:,:num_context,:]
                batch_context = {
                    'input_batch_list': [X],
                    'output_batch_list': [Y],
                    'mask': mask
                }
                posts = neural_process.infer_posterior_params(batch_context)

                elbo = neural_process.compute_ELBO(posts, batch_test, mode='eval')
                test_log_likelihood = neural_process.compute_cond_log_likelihood(posts, batch_test, mode='eval')
                test_elbos[num_context].append(elbo.data[0])
                test_log_likelihoods[num_context].append(test_log_likelihood.data[0])
                mse = test_log_likelihood * (-2) / torch.sum(mask)

                print('%d Context:' % num_context)
                print('ELBO: %.4f' % elbo)
                print('Test Log Like: %.4f' % test_log_likelihood)
                print('Test MSE: %.4f' % mse)

            neural_process.set_mode('train')


    neural_process.set_mode('eval')
    X_test = X_test[0].unsqueeze(0)
    Y_test = Y_test[0].unsqueeze(0)
    n_samples = 5

    loss_plots = []
    loss_plot_names = []
    for num_context in sorted(test_elbos.keys()):
        loss_plots += [np.array(test_elbos[num_context]), np.array(test_log_likelihoods[num_context])]
        loss_plot_names += ['%d context elbo'%num_context, '%d context log_likelihood'%num_context]

        X, Y, mask = X_context[0,:num_context,:], Y_context[0,:num_context,:], mask_context[0,:num_context,:]
        X = X.unsqueeze(0)
        Y = Y.unsqueeze(0)
        mask = mask.unsqueeze(0)
        batch_context = {
            'input_batch_list': [X],
            'output_batch_list': [Y],
            'mask': mask
        }
        posts = neural_process.infer_posterior_params(batch_context)
        Y_preds = neural_process.sample_outputs(posts, [X_test], n_samples)

        plots_to_plot = [[X_test.view(-1).data.numpy(), Y_test.view(-1).data.numpy()]]
        plot_names = ['true fn']
        for i in range(n_samples):
            plots_to_plot.append(
                [
                    X_test.view(-1).data.numpy(),
                    Y_preds[0,i].view(-1).data.numpy()
                ]
            )
            plot_names.append('sample %d' % i)
        
        plot_multiple_plots(
            plots_to_plot,
            plot_names,
            '%d context points'%num_context,
            osp.join(exp_dir, '%d_context_samples.png' % num_context)
        )

    plot_returns_on_same_plot(
        loss_plots,
        loss_plot_names,
        'Test ELBO and log likelihood during training',
        osp.join(exp_dir, 'losses.png')
    )

    return 1
