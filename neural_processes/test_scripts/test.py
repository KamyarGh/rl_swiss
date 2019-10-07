import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam

from neural_processes.generic_map import make_mlp, GenericMap
from neural_processes.neural_process import compute_diag_log_prob

use_gpu = False

def test_make_mlp(use_bn=False):
    # simple input-output
    true_fn = nn.Sequential(nn.Linear(5,10), nn.ReLU(), nn.Linear(10,10))
    mlp = make_mlp(5, 20, 6, nn.ReLU, use_bn=use_bn, output_dim=10)
    if use_gpu:
        true_fn.cuda()
        mlp.cuda()
    optimizer = Adam(mlp.parameters(), lr=1e-3)

    for i in range(10000):
        optimizer.zero_grad()

        X = Variable(torch.rand(64,5)*10 - 5)
        if use_gpu: X = X.cuda()
        Y = true_fn(X).detach()

        loss = torch.sum((Y - mlp(X))**2)
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(loss)
    
    print(mlp)


def test_generic_map(deterministic, use_bn):
    true_fn = nn.Sequential(
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 20)
    )
    gen_map = GenericMap(
        [5, 15], [10,3,7], siamese_input=True,
        num_siamese_input_layers=1, siamese_input_layer_dim=10,
        num_hidden_layers=1, hidden_dim=10,
        siamese_output=True, siamese_output_layer_dim=10,
        act='relu', use_bn=use_bn, deterministic=deterministic
    )
    if use_gpu:
        true_fn.cuda()
        gen_map.cuda()
    
    optimizer = Adam(gen_map.parameters(), lr=1e-3)
    
    mask = [Variable(torch.ones(64, 10)), Variable(torch.ones(64, 3)), Variable(torch.ones(64, 7))]
    if use_gpu: mask = list(map(lambda m: m.cuda(), mask))
    for i in range(10000):
        optimizer.zero_grad()

        X1 = Variable(torch.rand(64,5)*10 - 5)
        X2 = Variable(torch.rand(64,15)*6 - 3)
        if use_gpu:
            X1 = X1.cuda()
            X2 = X2.cuda()
        X = torch.cat([X1,X2], 1)
        Y = true_fn(X).detach()
        Y_splits = [Y[:,:10], Y[:,:3], Y[:,:7]]
        Y_preds = gen_map([X1,X2])

        if deterministic:
            loss = sum(map(lambda z: torch.sum((z[0] - z[1])**2), zip(Y_splits, Y_preds)))
        else:
            loss = sum(
                map(
                    lambda z: compute_diag_log_prob(
                        z[1][0], z[1][1], z[0], z[2], 1
                    ),
                    zip(Y_splits, Y_preds, mask)
                )
            )

        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(loss)


if __name__ == '__main__':
    use_gpu = True
    # test_make_mlp(use_bn=False)
    # test_make_mlp(use_bn=True)
    test_generic_map(True, True)
