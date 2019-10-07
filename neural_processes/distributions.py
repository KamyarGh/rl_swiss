import torch
from torch.autograd import Variable

def local_repeat(tensor, n):
    tensor = tensor.unsqueeze(1)
    tensor = tensor.contiguous()
    tensor = tensor.expand(-1,n,-1)
    tensor = tensor.contiguous()
    tensor = tensor.view(-1, int(tensor.size(-1)))
    return tensor
    # return tensor.unsqueeze(1).contiguous().expand(-1,n,-1).view(-1, tensor.size(1))

def sample_diag_gaussians(means, diags, n_samples):
    batch_size, dim = means.size(0), means.size(1)
    if n_samples > 1:
        means = local_repeat(means, n_samples)
        diags = local_repeat(diags, n_samples)
    
    eps = Variable(torch.randn(means.size()))
    samples = eps*diags + means

    # return samples.view(batch_size,n_samples,dim)
    return samples

if __name__ == '__main__':
    rand_t = torch.rand(3,10)
    print(rand_t)
    rep_rand_t = local_repeat(rand_t, 4)
    print(rep_rand_t)

    means = torch.randn(3,5)
    diags = torch.rand(3,5) * 0.02
    samples = sample_diag_gaussians(means, diags, 2)

    print(means)
    print(diags)
    print(samples)
