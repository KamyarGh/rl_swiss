import math

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def make_upconv_net(in_channels, in_h, upconv_specs):
    upconv_list = nn.ModuleList()
    kernel_sizes = upconv_specs['kernel_sizes']
    num_channels = upconv_specs['num_channels']
    strides = upconv_specs['strides']
    paddings = upconv_specs['paddings']
    output_paddings = upconv_specs['output_paddings']
    use_bn = upconv_specs['use_bn']
    in_ch = in_channels
    for k, ch, s, p, op in zip(kernel_sizes, num_channels, strides, paddings, output_paddings):
        seq = [nn.ConvTranspose2d(in_ch, ch, k, stride=s, padding=p, output_padding=op, bias=not use_bn)]
        if use_bn: seq.append(nn.BatchNorm2d(ch))
        seq.append(nn.ReLU())

        # seq.append(nn.Conv2d(ch, ch, 3, stride=1, padding=1, bias=not use_bn))
        # if use_bn: seq.append(nn.BatchNorm2d(ch))
        # seq.append(nn.ReLU())

        upconv_list.extend(seq)

        in_ch = ch
        in_h = (in_h - 1)*s - 2*p + k + op
        print('--> %dx%dx%d' % (in_ch, in_h, in_h))
    upconv_seq = nn.Sequential(*upconv_list)
    return upconv_seq, in_ch, in_h


def make_conv_net(in_channels, in_h, conv_specs):
    kernel_sizes = conv_specs['kernel_sizes']
    num_channels = conv_specs['num_channels']
    strides = conv_specs['strides']
    paddings = conv_specs['paddings']
    use_bn = conv_specs['use_bn']

    in_ch = in_channels
    conv_list = nn.ModuleList()
    for k, ch, s, p in zip(kernel_sizes, num_channels, strides, paddings):
        seq = [nn.Conv2d(in_ch, ch, k, stride=s, padding=p, bias=not use_bn)]
        if use_bn: seq.append(nn.BatchNorm2d(ch))
        seq.append(nn.ReLU())
        conv_list.extend(seq)

        in_ch = ch
        in_h = int(math.floor(
            1 + (in_h + 2*p - k)/s
        ))
        print('--> %dx%dx%d' % (in_ch, in_h, in_h))
    conv_seq = nn.Sequential(*conv_list)

    return conv_seq, in_ch, in_h


def make_fc_net(in_size, fc_specs):
    fc_hidden_sizes = fc_specs['hidden_sizes']
    use_bn = fc_specs['use_bn']

    fc_list = nn.ModuleList()
    for out_size in fc_hidden_sizes:
        seq = [nn.Linear(in_size, out_size, bias=not use_bn)]
        if use_bn: seq.append(nn.BatchNorm1d(out_size))
        seq.append(nn.ReLU())
        fc_list.extend(seq)
        in_size = out_size
    fc_seq = nn.Sequential(*fc_list)

    return fc_seq, out_size
