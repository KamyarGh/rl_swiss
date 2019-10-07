from neural_processes.generic_map import make_mlp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import ReLU

mlp = make_mlp(3, 10, 2, ReLU)
# mlp = nn.Sequential(
#     nn.Linear(3,5)
# )

x = Variable(torch.rand(6,3))
y = mlp(x)
print(y)
