'''
    the r_tensor is N_tasks x N_samples x r_dim
    the mask is N_tasks x N_samples x 1
'''
import torch
from torch.nn.functional import tanh

def sum_aggregator(r_tensor, mask):
    return torch.sum(r_tensor*mask, 1)


def mean_aggregator(r_tensor, mask):
    num_r_per_task = torch.sum(mask, 1)
    return torch.sum(r_tensor*mask, 1) / num_r_per_task


def tanh_sum_aggregator(r_tensor, mask):
    return tanh(torch.sum(r_tensor*mask, 1))
