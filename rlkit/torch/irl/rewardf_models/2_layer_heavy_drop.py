import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        hid_dim = 128
        self.model = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, input_batch):
        return self.model(input_batch)
