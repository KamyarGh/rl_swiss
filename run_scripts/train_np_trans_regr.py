# Imports ---------------------------------------------------------------------
# Python
import argparse
import joblib
import yaml
import os.path as osp
from collections import defaultdict

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd
from torch.optim import Adam

# NumPy
import numpy as np
from numpy import array
from numpy.random import choice, randint

# Model Building
from neural_processes.generic_map import GenericMap
from neural_processes.base_map import BaseMap
from neural_processes.neural_process import NeuralProcessV1
from neural_processes.aggregators import sum_aggregator, mean_aggregator, tanh_sum_aggregator
from neural_processes.npv2 import NeuralProcessV2
from neural_processes.npv3 import NeuralProcessV3

# Data
from rlkit.data_management.prep_trans_for_neural_process import prep_trans
from rlkit.data_management.neural_process_data_sampler import NPTransDataSampler

# Logging
from rlkit.core import logger
from rlkit.launchers.launcher_util import setup_logger, set_seed

def experiment(exp_specs):
    # Set up logging ----------------------------------------------------------
    exp_id = exp_specs['exp_id']
    exp_prefix = exp_specs['exp_name']
    seed = exp_specs['seed']
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    # Prep the data -----------------------------------------------------------
    train_list, val_list = prep_trans(exp_specs['data_prep_specs'])
    train_sampler = NPTransDataSampler(train_list)
    val_sampler = NPTransDataSampler(val_list)

    obs_dim = train_list[0]['_observations'].shape[-1]
    act_dim = train_list[0]['_actions'].shape[-1]

    r_dim = exp_specs['r_dim']
    z_dim = exp_specs['z_dim']
    use_bn = exp_specs['use_bn']

    encoder_lr = float(exp_specs['encoder_lr'])
    base_map_lr = float(exp_specs['base_map_lr'])
    r_to_z_map_lr = float(exp_specs['r_to_z_map_lr'])
    # Model Definition --------------------------------------------------------
    encoder = GenericMap(
        [obs_dim, act_dim, obs_dim, 1], [r_dim], siamese_input=False,
        num_hidden_layers=exp_specs['num_encoder_hidden_layers'], hidden_dim=128,
        siamese_output=False, act='relu',
        deterministic=True,
        use_bn=use_bn
    )
    encoder_optim = Adam(encoder.parameters(), lr=encoder_lr)

    base_map = BaseMap(
        z_dim, [obs_dim, act_dim], [obs_dim, 1], siamese_input=False,
        num_hidden_layers=exp_specs['num_base_map_hidden_layers'], hidden_dim=128,
        siamese_output=False, act='relu',
        deterministic=True,
        use_bn=use_bn
    )
    base_map_optim = Adam(base_map.parameters(), lr=base_map_lr)

    r_to_z_map = GenericMap(
        [r_dim], [z_dim], siamese_input=False,
        num_hidden_layers=exp_specs['num_r_to_z_map_hidden_layers'], hidden_dim=40,
        siamese_output=False, act='relu',
        deterministic=False,
        use_bn=use_bn
    )
    r_to_z_map_optim = Adam(r_to_z_map.parameters(), lr=r_to_z_map_lr)

    if exp_specs['neural_process_version'] == 1:
        raise Exception()
        model_class = NeuralProcessV1
    elif exp_specs['neural_process_version'] == 2:
        raise Exception()
        model_class = NeuralProcessV2
    elif exp_specs['neural_process_version'] == 3:
        model_class = NeuralProcessV3    
        neural_process = model_class(
            encoder,
            encoder_optim,
            exp_specs['aggregator_mode'],
            r_to_z_map,
            r_to_z_map_optim,
            base_map,
            base_map_optim,
            exp_specs['r_dim'],
            exp_specs['z_dim'],
            use_nat_grad=False
        )
    else:
        raise Exception()

    # -----------------------------------------------------------------------------
    for iter_num in range(exp_specs['max_iters']):
        if exp_specs['neural_process_version'] == 1:
            X_context, Y_context, mask_context, X_test, Y_test, mask_test = train_sampler.sample_batch(
                exp_specs['train_batch_size'], exp_specs['context_size_range'], test_is_context=True
            )
        elif exp_specs['neural_process_version'] in [2, 3]:
            X_context, Y_context, mask_context, X_test, Y_test, mask_test = train_sampler.sample_batch(
                exp_specs['train_batch_size'], exp_specs['context_size_range'],
                test_is_context=False, test_size=exp_specs['train_test_size']
            )

        batch = {
            'input_batch_list': [X_context],
            'output_batch_list': [Y_context],
            'mask': mask_context
        }
        batch_test = {
            'input_batch_list': [X_test],
            'output_batch_list': [Y_test],
            'mask': mask_test
        }
        
        if exp_specs['neural_process_version'] == 1:
            neural_process.train_step(batch)
        elif exp_specs['neural_process_version'] in [2, 3]:
            neural_process.train_step(batch, batch_test)
        else:
            raise Exception()

        # assert False, 'make sure logging is working in progress.csv'
        if iter_num % exp_specs['freq_val'] == 0:
            logger.record_tabular('Iter', iter_num)
            print('-'*80)
            print('Iter %d' % iter_num)
            neural_process.set_mode('eval')

            # DEBUGGING
            # print('\n'*10)
            # post_state = neural_process.reset_posterior_state()
            # print(post_state)
            # for i in range(5):
            #     post_state = neural_process.update_posterior_state(post_state, np.ones(obs_dim), np.ones(act_dim), np.ones(1), np.ones(obs_dim))
            #     print(post_state)
            # print(neural_process.get_posterior_params(post_state))
            # print('\n'*10)



            for val_context_size in exp_specs['val_context_sizes']:
                X_context, Y_context, mask_context, X_test, Y_test, mask_test = val_sampler.sample_batch(
                    exp_specs['val_batch_size'], [val_context_size, val_context_size+1],
                    test_is_context=False, test_size=exp_specs['val_test_size']
                )            

                batch_context = {
                    'input_batch_list': [X_context],
                    'output_batch_list': [Y_context],
                    'mask': mask_context
                }
                batch_test = {
                    'input_batch_list': [X_test],
                    'output_batch_list': [Y_test],
                    'mask': mask_test 
                }
                batch_union = {
                    'input_batch_list': [
                        torch.cat([c, t], 1)
                        for c, t in zip(batch_context['input_batch_list'], batch_test['input_batch_list'])
                    ],
                    'output_batch_list': [
                        torch.cat([c, t], 1)
                        for c, t in zip(batch_context['output_batch_list'], batch_test['output_batch_list'])
                    ],
                    'mask': torch.cat([batch_context['mask'], batch_test['mask']], 1)
                }

                if exp_specs['neural_process_version'] == 1:
                    posts = neural_process.infer_posterior_params(batch_context)
                    elbo = neural_process.compute_ELBO(posts, batch_context, mode='eval')
                    context_log_likelihood = neural_process.compute_cond_log_likelihood(posts, batch_context, mode='eval')
                    test_log_likelihood = neural_process.compute_cond_log_likelihood(posts, batch_test, mode='eval')
                    context_mse = -2.0 * context_log_likelihood / torch.sum(mask_context)
                    test_mse = -2.0 * test_log_likelihood / torch.sum(mask_test)
                elif exp_specs['neural_process_version'] == 2:
                    context_posts = neural_process.infer_posterior_params(batch_context)
                    union_posts = neural_process.infer_posterior_params(batch_union)
                    elbo = neural_process.compute_ELBO(context_posts, union_posts, batch_test, mode='eval')

                    # now for computing MSE we don't infer the posterior using the union of context and test
                    # to show what loss would look like at test time
                    context_log_likelihood = neural_process.compute_cond_log_likelihood(context_posts, batch_context, mode='eval')
                    test_log_likelihood = neural_process.compute_cond_log_likelihood(context_posts, batch_test, mode='eval')
                    train_time_log_likelihood = neural_process.compute_cond_log_likelihood(union_posts, batch_test, mode='eval')
                    context_mse = -2.0 * context_log_likelihood / torch.sum(mask_context)
                    test_mse = -2.0 * test_log_likelihood / torch.sum(mask_test)
                    train_time_test_mse = -2.0 * train_time_log_likelihood / torch.sum(batch_union['mask'])

                    logger.record_tabular('con_size_%d_Train_Time_Test_Log_Like'%val_context_size, train_time_log_likelihood.data[0])
                    logger.record_tabular('con_size_%d_Train_Time_Test_MSE'%val_context_size, train_time_test_mse.data[0])
                elif exp_specs['neural_process_version'] == 3:
                    posts = neural_process.infer_posterior_params(batch_context)
                    elbo = neural_process.compute_ELBO(posts, batch_test, mode='eval')
                    context_log_likelihood = neural_process.compute_cond_log_likelihood(posts, batch_context, mode='eval')
                    test_log_likelihood = neural_process.compute_cond_log_likelihood(posts, batch_test, mode='eval')
                    context_mse = -2.0 * context_log_likelihood / torch.sum(mask_context)
                    test_mse = -2.0 * test_log_likelihood / torch.sum(mask_test)

                logger.record_tabular('con_size_%d_ELBO'%val_context_size, elbo.data[0])
                logger.record_tabular('con_size_%d_Context_Log_Like'%val_context_size, context_log_likelihood.data[0])
                logger.record_tabular('con_size_%d_Test_Log_Like'%val_context_size, test_log_likelihood.data[0])
                logger.record_tabular('con_size_%d_Context_MSE'%val_context_size, context_mse.data[0])
                logger.record_tabular('con_size_%d_Test_MSE'%val_context_size, test_mse.data[0])

            logger.dump_tabular(with_prefix=False, with_timestamp=False)
            neural_process.set_mode('train')
        
        if iter_num % exp_specs['freq_save'] == 0:
            dict_to_save = {'neural_process': neural_process}
            logger.save_itr_params(iter_num, dict_to_save)


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)
    
    experiment(exp_specs)
