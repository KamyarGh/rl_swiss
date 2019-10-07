import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from rlkit.torch.core import PyTorchModule

class Model(nn.Module):
    def __init__(self, input_dim, hid_dim=128, num_layers=2):
        super().__init__()

        mod_list = nn.ModuleList(
            [
                nn.Linear(input_dim, hid_dim),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            ]
        )
        for _ in range(num_layers-1):
            mod_list.extend(
                [
                    nn.Linear(hid_dim, hid_dim),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                ]
            )
        mod_list.append(nn.Linear(hid_dim, 1))
        self.model = nn.Sequential(*mod_list)

    def forward(self, obs_batch, act_batch):
        input_batch = torch.cat([obs_batch, act_batch], dim=1)
        return self.model(input_batch)
