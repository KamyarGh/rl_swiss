import torch
import numpy as np

from rlkit.torch.irl.encoders.trivial_encoder import TrivialTrajEncoder, TrivialR2ZMap, TrivialNPEncoder

N_TASKS = 5
N_TRAJS_PER_TASK = 10
TRAJ_LEN = 73
OBS_DIM = 4
ACT_DIM = 2
Z_DIM = 67

context = [
    [
        {
            'observations': np.random.rand(TRAJ_LEN, OBS_DIM),
            'actions': np.random.rand(TRAJ_LEN, ACT_DIM)
        } for j in range(N_TRAJS_PER_TASK)
    ] for i in range(N_TASKS)
]

traj_enc = TrivialTrajEncoder(
    dict(
        hidden_sizes=[17,19],
        output_size=23,
        input_size=OBS_DIM + ACT_DIM
    ),
    dict(
        hidden_sizes=[31,37],
        output_size=29,
        input_size=23 * TRAJ_LEN
    )
)

r2z_map = TrivialR2ZMap(
    dict(
        hidden_sizes=[43,47],
        output_size=41,
        input_size=29
    ),
    dict(
        hidden_sizes=[53,59],
        output_size=Z_DIM,
        input_size=41
    )
)

np_enc = TrivialNPEncoder(
    'tanh_sum',
    traj_enc,
    r2z_map   
)

post_dist = np_enc(context)
print(post_dist.mean.size())
print(post_dist.cov.size())
sample = post_dist.sample()
print(sample.size())
