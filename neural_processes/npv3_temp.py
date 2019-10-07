import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd

import numpy as np
from numpy import pi
from numpy import log as np_log

from neural_processes.base_neural_process import BaseNeuralProcess
from neural_processes.distributions import sample_diag_gaussians, local_repeat
from neural_processes.aggregators import sum_aggregator, mean_aggregator, tanh_sum_aggregator

log_2pi = np_log(2*pi)


def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def compute_spherical_log_prob(preds, true_outputs, mask, n_samples):
    '''
        Compute log prob assuming spherical Gaussian with some mean
        Up to additive constant
    '''
    log_prob = -0.5 * torch.sum(mask*(preds - true_outputs)**2)
    if n_samples > 1: log_prob /= float(n_samples)
    return log_prob


def compute_diag_log_prob(preds_mean, preds_log_cov, true_outputs, mask, n_samples):
    '''
        Compute log prob assuming diagonal Gaussian with some mean and log cov
    '''
    assert False, 'The tests so far have shown this to be unstable'
    preds_cov = torch.exp(preds_log_cov)

    log_prob = -0.5 * torch.sum(
        mask*(preds_mean - true_outputs)**2 / preds_cov
    )

    log_det_temp = mask*(torch.sum(preds_log_cov, 1) + log_2pi)
    log_prob += -0.5*torch.sum(log_det_temp)

    if n_samples > 1: log_prob /= float(n_samples)
    return log_prob


class NeuralProcessV3(BaseNeuralProcess):
    '''
        Neural process
        This is the version of Neural Processes from equation (7)
        of the Neural Processes paper

        Right now not dealing with taking multiple samples from z to
        do Monte-Carlo estimate. Hopefully won't become necessary.

        Prior is assumed to be a spherical Gaussian
    '''
    def __init__(
        self,
        encoder,
        encoder_optim,
        aggregator_mode,
        r_to_z_map,
        r_to_z_map_optim,
        base_map,
        base_map_optim,
        r_dim,
        z_dim,
        use_nat_grad=True, # whether to use natural gradient for posterior parameter updates
    ):
        self.base_map = base_map
        self.base_map_optim = base_map_optim
        self.encoder = encoder
        self.encoder_optim = encoder_optim
        self.r_to_z_map = r_to_z_map
        self.r_to_z_map_optim = r_to_z_map_optim
        self.use_nat_grad = use_nat_grad
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.set_mode('train')

        self.aggregator = {
            'sum_aggregator': sum_aggregator,
            'mean_aggregator': mean_aggregator,
            'tanh_sum_aggregator': tanh_sum_aggregator
        }[aggregator_mode]
        self.aggregator_mode = aggregator_mode
    

    def set_mode(self, mode):
        assert mode in ['train', 'eval']
        mode = True if mode=='train' else False
        self.base_map.train(mode)
        self.encoder.train(mode)
        self.r_to_z_map.train(mode)

        self.training = mode


    def reset_posterior_state(self):
        if self.aggregator_mode in ['sum_aggregator', 'tanh_sum_aggregator']:
            return {'pre_r': Variable(torch.zeros(1, self.r_dim))}
        else:
            return {'pre_r': Variable(torch.zeros(1, self.r_dim)), 'N': 0}
    

    def update_posterior_state(self, posterior_state, obs, act, rew, next_obs):
        input_list = [obs, act, rew, next_obs]
        input_list = list(map(
            lambda x: Variable(torch.FloatTensor(x[np.newaxis])),
            input_list
        ))
        new_r = self.encoder(input_list)[0]

        if self.aggregator_mode in ['sum_aggregator', 'tanh_sum_aggregator']:
            r = posterior_state['pre_r'] + new_r
            return {'r': r}
        else:
            r = posterior_state['pre_r'] + new_r
            N = posterior_state['N']
            return {'r': r, 'N': N+1}
    

    def get_posterior_params(self, posterior_state):
        if self.aggregator_mode == 'sum_aggregator'
            pre_r = posterior_state['pre_r']
            post = self.r_to_z_map([pre_r])
        elif self.aggregator_mode == 'tanh_sum_aggregator':
            pre_r = posterior_state['pre_r']
            post = self.r_to_z_map([F.tanh(pre_r)])
        elif self.aggregator_mode == 'mean_aggregator':
            pre_r = posterior_state['pre_r']
            N = posterior_state['N']
            post = self.r_to_z_map([pre_r / N])
        z_mean, z_log_cov = torch.unsqueeze(post[0]), torch.unsqueeze(post[1])
        z_cov = torch.exp(z_log_cov)
        return z_mean.data.numpy(), z_cov.data.numpy()


    def infer_posterior_params(self, batch):
        '''
            batch should be dictionary of arrays of size N_tasks x N_samples x dim
            batch = {
                'input_batch_list': ...,
                'output_batch_list': ...,
                'mask': ..., # used to condition on varying number of points
            }
        '''
        input_list = batch['input_batch_list']
        output_list = batch['output_batch_list']
        N_tasks, N_samples = input_list[0].size(0), input_list[0].size(1)
        reshaped_input_list = [inp.contiguous().view(-1, inp.size(2)) for inp in input_list]
        reshaped_output_list = [out.contiguous().view(-1, out.size(2)) for out in output_list]        

        r = self.encoder(reshaped_input_list + reshaped_output_list)[0]
        r = r.view(N_tasks, N_samples, -1)
        r_agg = self.aggregator(r, batch['mask'])
        
        # since r_to_z_map is a generic map, it will output
        # [[mean, log_cov]] because that is the interface
        # of the generic map
        post = self.r_to_z_map([r_agg])
        return post[0]
    

    def sample_outputs(self, posteriors, input_batch_list, n_samples):
        z_means = posteriors[0]
        z_log_covs = posteriors[1]
        z_covs = torch.exp(z_log_covs)
        z_samples = sample_diag_gaussians(z_means, z_covs, n_samples)
        z_samples = local_repeat(z_samples, input_batch_list[0].size(1))

        num_tasks, num_per_task = input_batch_list[0].size(0), input_batch_list[0].size(1)
        input_batch_list = [inp.contiguous().view(-1,inp.size(2)) for inp in input_batch_list]
        input_batch_list = [local_repeat(inp, n_samples) for inp in input_batch_list]

        if (not self.base_map.siamese_output) and self.base_map.deterministic:
            outputs = self.base_map(z_samples, input_batch_list)[0]
            outputs = outputs.view(num_tasks, n_samples, num_per_task, outputs.size(-1))
        else:
            raise NotImplementedError

        return outputs


    def train_step(self, context_batch, test_batch):
        self.encoder_optim.zero_grad()
        self.r_to_z_map.zero_grad()
        self.base_map_optim.zero_grad()

        posteriors = self.infer_posterior_params(context_batch)
        neg_elbo = -1.0*self.compute_ELBO(posteriors, test_batch)

        if self.use_nat_grad:
            raise NotImplementedError
            # some code from encoder-free NP
            mean_grad, log_cov_grad = z_means.grad, log_cov_grad.grad
            if self.use_nat_grad:
                mean_grad.mul_(torch.exp(z_log_covs))
                log_cov_grad.mul_(2.0)
            z_means.sub_(mean_grad * mean_lr)
            z_log_covs.sub_(log_cov_grad * log_cov_lr)
        else:
            neg_elbo.backward()
            self.base_map_optim.step()
            self.r_to_z_map_optim.step()
            self.encoder_optim.step()


    def compute_ELBO(self, posteriors, batch, mode='train'):
        '''
            n_samples is the number of samples used for
            Monte Carlo estimate of the ELBO
        '''
        cond_log_likelihood = self.compute_cond_log_likelihood(posteriors, batch, mode)
        KL = self.compute_ELBO_KL(posteriors)
        
        elbo = cond_log_likelihood - KL
        # idk whether to put this or not
        elbo = elbo / float(batch['input_batch_list'][0].size(0))
        return elbo
    

    def compute_cond_log_likelihood(self, posteriors, batch, mode='train'):
        '''
            Computer E[log p(y|x,z)] up to constant additional factors

            arrays are N_tasks x N_samples x dim
        '''
        # not dealing with more than 1 case right now
        n_samples = 1

        input_batch_list, output_batch_list = batch['input_batch_list'], batch['output_batch_list']
        mask = batch['mask'].view(-1, 1)

        z_means = posteriors[0]
        z_log_covs = posteriors[1]
        z_covs = torch.exp(z_log_covs)
    
        if mode == 'eval':
            z_samples = z_means
        else:
            z_samples = sample_diag_gaussians(z_means, z_covs, n_samples)
        z_samples = local_repeat(z_samples, input_batch_list[0].size(1))
        input_batch_list = [inp.view(-1,inp.size(2)) for inp in input_batch_list]
        output_batch_list = [out.view(-1,out.size(2)) for out in output_batch_list]

        preds = self.base_map(z_samples, input_batch_list)

        if self.base_map.siamese_output:
            log_prob = 0.0
            if self.base_map.deterministic:
                for pred, output in zip(preds, output_batch_list):
                    log_prob += compute_spherical_log_prob(pred, output, mask, n_samples)
            else:
                for pred, output in zip(preds, output_batch_list):
                    log_prob += compute_diag_log_prob(
                        pred[0], pred[1], output, mask, n_samples
                    )
        else:
            if self.base_map.deterministic:
                log_prob = compute_spherical_log_prob(
                    preds[0], output_batch_list[0], mask, n_samples
                )
            else:
                preds_mean, preds_log_cov = preds[0][0], preds[0][1]
                log_prob = compute_diag_log_prob(
                    preds_mean, preds_log_cov, output_batch_list[0], mask, n_samples
                )

        return log_prob


    def compute_ELBO_KL(self, posteriors):
        '''
            We always deal with spherical Gaussian prior
        '''
        z_means = posteriors[0]
        z_log_covs = posteriors[1]
        z_covs = torch.exp(z_log_covs)
        KL = -0.5 * torch.sum(
            1.0 + z_log_covs - z_means**2 - z_covs
        )
        return KL
