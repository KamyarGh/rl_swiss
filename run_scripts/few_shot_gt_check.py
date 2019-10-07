import numpy as np
import torch

from gym.spaces import Dict

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam

from rllab.misc.instrument import VariantGenerator
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.networks import Mlp

from rlkit.envs import get_meta_env, get_meta_env_params_iters

from rlkit.torch.irl.np_bc import NeuralProcessBC
from rlkit.torch.irl.encoders.trivial_encoder import TrivialTrajEncoder, TrivialR2ZMap, TrivialNPEncoder

import yaml
import argparse
import importlib
import psutil
import os
from os import path
import argparse
import joblib
from time import sleep

EXPERT_LISTING_YAML_PATH = '/h/kamyar/oorl_rlkit/rlkit/torch/irl/experts.yaml'


class Classifier(nn.Module):
    def __init__(self, enc_hidden_sizes):
        super(Classifier, self).__init__()
        self.enc = Mlp(
            enc_hidden_sizes,
            1,
            9,
            hidden_activation=torch.nn.functional.relu,
            # batch_norm=True
            # layer_norm=True
        )
    
    def forward(self, context_batch, query_batch, label=None):
        # print(context_batch.size())
        # print(query_batch.size())

        norm0 = torch.norm(context_batch - query_batch[:,:3], dim=-1, keepdim=True)
        norm1 = torch.norm(context_batch - query_batch[:,3:6], dim=-1, keepdim=True)
        cond0 = 1.0*(norm0 < 0.3).type(torch.FloatTensor)
        cond1 = 1.0*(norm1 < 0.3).type(torch.FloatTensor)
        # print('----------')
        # print(torch.sum((cond0 + cond1)==1))
        # print(cond0[37:71])
        # if label is not None:
        #     print(torch.sum((cond0 == label).type(torch.FloatTensor)))
            # print(label[37:71])

        logits = self.enc(torch.cat([context_batch, query_batch], -1))
        return logits
        # return cond0 - 0.5


def get_batch(context_buffer, test_buffer, num_tasks, num_context_trajs, num_test_trajs):
    # build the context batch
    context_batch, task_identifiers_list = context_buffer.sample_trajs(
        num_context_trajs,
        num_tasks=num_tasks
    )

    task_ids = np.array([list(it) for it in task_identifiers_list])
    context_task_ids = np.tile(task_ids, (1, num_context_trajs))
    context_task_ids = np.reshape(context_task_ids, (-1, num_context_trajs, 3))
    test_task_ids = np.tile(task_ids, (1, num_test_trajs))
    test_task_ids = np.reshape(test_task_ids, (-1, num_test_trajs, 3))
    all_task_ids = np.tile(task_ids, (1, num_context_trajs+num_test_trajs))
    all_task_ids = np.reshape(all_task_ids, (-1, 3))
    # c_max = np.array([[[1.27175999, 1.26395128, 1.21729739]]])
    # c_min = np.array([[[-1.23604834e+00, -1.27612583e+00, -1.23701436e+00]]])
    # SCALE = 0.99

    obs = np.array(
        [
            [traj['observations'][-1] for traj in task_trajs]
            for task_trajs in context_batch
        ]
    )

    obj_0_color = obs[..., 12:15]
    obj_1_color = obs[..., 15:18]
    unsorted_context_colors = obs[..., 12:18]
    # obj_0_rel_goal = np.reshape(obj_0_rel_goal, (variant['num_tasks_per_batch']*variant['num_context_trajs'], 3))
    # obj_1_rel_goal = np.reshape(obj_1_rel_goal, (variant['num_tasks_per_batch']*variant['num_context_trajs'], 3))

    context_labels = np.linalg.norm(obj_0_color - context_task_ids, axis=-1) < 0.3
    context_labels = context_labels.astype(np.float32)
    # correct_obj_color = np.where(context_labels, obj_0_color, obj_1_color)
    # incorrect_obj_color = np.where(1 - context_labels, obj_0_color, obj_1_color)
    # context_input_batch = np.concatenate((correct_obj_color, incorrect_obj_color), 2)

    # unnorm_correct = c_min + (correct_obj_color + SCALE)*(c_max - c_min) / (2*SCALE)
    # unnorm_incorrect = c_min + (incorrect_obj_color + SCALE)*(c_max - c_min) / (2*SCALE)
    # print(task_ids.shape)
    # print(correct_obj_color.shape)
    # print(np.linalg.norm(task_ids - correct_obj_color, axis=-1))
    # print(np.linalg.norm(task_ids - incorrect_obj_color, axis=-1))

    # build the test batch
    test_batch, _ = test_buffer.sample_trajs(
        num_test_trajs,
        task_identifiers=task_identifiers_list
    )
    obs = np.array(
        [
            [traj['observations'][-1] for traj in task_trajs]
            for task_trajs in test_batch
        ]
    )
    unsorted_test_colors = obs[..., 12:18]
    # obj_0_rel_goal = obs[..., :3]
    # obj_1_rel_goal = obs[..., 3:6]
    obj_0_color = obs[..., 12:15]
    # obj_1_color = obs[..., 15:18]
    # obj_0_dist = np.linalg.norm(obj_0_rel_goal, axis=-1, keepdims=True)
    # obj_1_dist = np.linalg.norm(obj_1_rel_goal, axis=-1, keepdims=True)
    test_labels = np.linalg.norm(obj_0_color - test_task_ids, axis=-1) < 0.3
    test_labels = test_labels.astype(np.float32)
    # test_labels = (obj_0_dist < obj_1_dist).astype(np.float32)
    # correct_test_color = np.where(test_labels, obj_0_color, obj_1_color)
    # print(np.linalg.norm(task_ids - correct_test_color, axis=-1))

    # convert to pytorch
    # context_labels = np.reshape(context_labels.copy(), (num_tasks*num_context_trajs, 1))
    # test_labels = np.reshape(test_labels.copy(), (num_tasks*num_test_trajs, 1))

    query_batch = np.concatenate((unsorted_context_colors, unsorted_test_colors), 1)
    # print(query_batch.shape)
    labels = np.concatenate((context_labels, test_labels), 1)
    labels = np.reshape(labels, (-1,1))
    # print(labels.shape)

    # context_input_batch = Variable(ptu.from_numpy(context_input_batch), requires_grad=False)
    query_batch = Variable(ptu.from_numpy(query_batch), requires_grad=False)
    labels = Variable(ptu.from_numpy(labels), requires_grad=False)

    # print('------')
    # # print(context_input_batch[0])
    # tv = Variable(ptu.from_numpy(np.concatenate((task_ids, task_ids), -1)))[0,:,:3]
    # tv = torch.cat([tv, tv], 0)
    # print(tv[0])
    # print(query_batch[0])
    # print(torch.norm(query_batch[0,:,:3] - tv, dim=-1))
    # print(torch.norm(query_batch[0,:,3:] - tv, dim=-1))
    # print(labels[:query_batch.size(1)])

    # return context_input_batch, query_batch, labels
    # print(context_input_batch.shape)
    # print(np.concatenate((task_ids, task_ids), -1).shape)
    
    query_batch = query_batch.view(-1, 6)
    return Variable(ptu.from_numpy(all_task_ids), requires_grad=False), query_batch, labels


def experiment(variant):
    with open(EXPERT_LISTING_YAML_PATH, 'r') as f:
        listings = yaml.load(f.read())
    expert_dir = listings[variant['expert_name']]['exp_dir']
    specific_run = listings[variant['expert_name']]['seed_runs'][variant['expert_seed_run_idx']]
    file_to_load = path.join(expert_dir, specific_run, 'extra_data.pkl')
    extra_data = joblib.load(file_to_load)

    # this script is for the non-meta-learning GAIL
    train_context_buffer, train_test_buffer = extra_data['meta_train']['context'], extra_data['meta_train']['test']
    test_context_buffer, test_test_buffer = extra_data['meta_test']['context'], extra_data['meta_test']['test']

    net_size = variant['enc_net_size']
    num_layers = variant['enc_num_layers']
    enc_hidden_sizes = [net_size] * num_layers

    model = Classifier(enc_hidden_sizes)
    print(model)
    model_optim = Adam(
        model.parameters(),
        lr=variant['lr'],
    )
    bce_with_logits_loss = nn.BCEWithLogitsLoss()

    if ptu.gpu_enabled():
        model.cuda()
        bce_with_logits_loss.cuda()
    
    for iter_num in range(variant['max_train_iters']):
        context_input_batch, query_batch, labels = get_batch(
            train_context_buffer,
            train_test_buffer,
            variant['num_tasks_per_batch'],
            variant['num_context_trajs'],
            variant['num_test_trajs']
        )

        # model(context_input_batch, query_batch, label=labels)
        model_optim.zero_grad()
        logits = model(context_input_batch, query_batch)
        # print(labels[:10])
        loss = bce_with_logits_loss(logits, labels)
        loss.backward()
        model_optim.step()

        # print(model.enc.fc0.bias.grad)

        # IMPLEMENT A PROPER EVALUATION TO SEE IF IT'S ACTUALLY IMPORTANT TO HAVE TASKS PER BATCH 1

        if iter_num % variant['freq_val'] == 0:
            model.eval()
            print('\nIter %d -----------------------' % iter_num)

            # Evaluate meta-train
            context_input_batch, query_batch, labels = get_batch(
                train_context_buffer,
                train_test_buffer,
                50,
                variant['eval_num_context_trajs'],
                variant['eval_num_test_trajs']
            )
            logits = model(context_input_batch, query_batch, label=labels)
            loss = bce_with_logits_loss(logits, labels)
            preds = (logits > 0).type(torch.FloatTensor)
            num_context_points = 50*variant['eval_num_context_trajs']
            context_accuracy = (preds[:num_context_points] == labels[:num_context_points]).type(torch.FloatTensor).mean()
            test_accuracy = (preds[num_context_points:] == labels[num_context_points:]).type(torch.FloatTensor).mean()
            
            print('Meta-Train Loss: %.4f' % loss)
            print('Meta-Train Acc Ctxt: %.4f' % context_accuracy)
            print('Meta-Train Acc Test: %.4f' % test_accuracy)

            # Evaluate meta-test
            print('')
            context_input_batch, query_batch, labels = get_batch(
                test_context_buffer,
                test_test_buffer,
                50,
                variant['eval_num_context_trajs'],
                variant['eval_num_test_trajs']
            )
            logits = model(context_input_batch, query_batch, label=labels)
            loss = bce_with_logits_loss(logits, labels)
            preds = (logits > 0).type(torch.FloatTensor)
            num_context_points = 50*variant['eval_num_context_trajs']
            context_accuracy = (preds[:num_context_points] == labels[:num_context_points]).type(torch.FloatTensor).mean()
            test_accuracy = (preds[num_context_points:] == labels[num_context_points:]).type(torch.FloatTensor).mean()
            
            print('Meta-Test Loss: %.4f' % loss)
            print('Meta-Test Acc Ctxt: %.4f' % context_accuracy)
            print('Meta-Test Acc Test: %.4f' % test_accuracy)
            model.train()

    return 1


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)
    
    exp_id = exp_specs['exp_id']
    exp_prefix = exp_specs['exp_name']
    seed = exp_specs['seed']
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)
    if exp_specs['use_gpu']: ptu.set_gpu_mode(True)

    experiment(exp_specs)
