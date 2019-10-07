import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from rlkit.torch.core import PyTorchModule

class Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        hid_dim = 32
        self.model = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, obs_batch, act_batch):
        input_batch = torch.cat([obs_batch, act_batch], dim=1)
        return self.model(input_batch)
