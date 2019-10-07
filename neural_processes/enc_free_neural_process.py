import torch
import torch.nn as nn
import torch.nn.function as F
from torch.autograd import Variable
from torch import autograd

from numpy import pi
from numpy import log as np_log

from base_neural_process import BaseNeuralProcess
from distributions import sample_diag_gaussians, local_repeat

log_2pi = np_log(2*pi)

def compute_spherical_log_prob(preds, true_outputs, n_samples)
    '''
        Compute log prob assuming spherical Gaussian with some mean
        Up to additive constant
    '''
    return -0.5 * torch.sum((preds - true_outputs)**2) / float(n_samples)


def compute_diag_log_prob(preds_mean, preds_log_cov, true_outputs, n_samples):
    '''
        Compute log prob assuming diagonal Gaussian with some mean and log cov
    '''
    preds_cov = torch.exp(preds_log_cov)

    log_prob = -0.5 * torch.sum(
        (preds_mean - repeated_true_outputs)**2 / preds_cov
    )

    log_det = torch.logsumexp(torch.sum(preds_log_cov, 1))
    log_det += log_2pi
    log_det *= -0.5
    log_prob += log_det

    log_prob /= float(n_samples)
    return log_prob


class EncFreeNeuralProcess(BaseNeuralProcess):
    '''
        Neural process without an encoder
        Always using a spherical Gaussian prior
    '''
    def __init__(
        self,
        base_map,
        base_map_optim,
        use_nat_grad=True, # whether to use natural gradient for posterior parameter updates
    ):
        assert False, 'You need to rethink how to do proper variational inference with this model'
        assert False, 'Also, you need to add and account for n_points_per_task'
        self.base_map = base_map
        self.base_map_optim = base_map_optim
        self.use_nat_grad = use_nat_grad


    def update_posteriors(self, post_init, batch, mean_lr, log_cov_lr, n_samples=1):
        z_means = post_init['means']
        z_log_covs = post_init['log_covs']

        neg_elbo = -1.0*self.compute_ELBO(post_init, batch, n_samples)
        post_grads = autograd.grad([neg_elbo],[z_means, z_log_covs])
        mean_grad, log_cov_grad = post_grads[0], post_grads[1]

        if self.use_nat_grad:
            mean_grad.mul_(torch.exp(z_log_covs))
            log_cov_grad.mul_(2.0)
        z_means.sub_(mean_grad * mean_lr)
        z_log_covs.sub_(log_cov_grad * log_cov_lr)
    

    def train_step(self, post_init, batch, mean_lr, log_cov_lr, n_samples=1):
        z_means = post_init['means']
        z_log_covs = post_init['log_covs']

        z_means.grad.zero_()
        z_log_covs.grad.zero_()
        self.base_map_optim.zero_grad()

        neg_elbo = -1.0*self.compute_ELBO(post_init, batch, n_samples)
        neg_elbo.backward()
        self.base_map_optim.step()

        mean_grad, log_cov_grad = z_means.grad, log_cov_grad.grad
        if self.use_nat_grad:
            mean_grad.mul_(torch.exp(z_log_covs))
            log_cov_grad.mul_(2.0)
        z_means.sub_(mean_grad * mean_lr)
        z_log_covs.sub_(log_cov_grad * log_cov_lr)


    def compute_ELBO(self, posteriors, test_set, n_samples=1):
        '''
            n_samples is the number of samples used for
            Monte Carlo estimate of the ELBO
        '''
        cond_log_likelihood = self.compute_cond_log_likelihood(posteriors, test_set, n_samples)
        KL = self.compute_ELBO_KL(posteriors)
        return cond_log_likelihood - KL
    

    def compute_cond_log_likelihood(self, posteriors, test_set, n_samples=1):
        '''
            Computer E[log p(y|x,z)] up to constant additional factors

            In the encoder-free version you don't have any encoders
            and as a result no context set
        '''
        test_inputs, test_outputs = test_set['inputs'], test_set['outputs']

        z_means = posteriors['means']
        z_log_covs = posteriors['log_covs']
        z_covs = torch.exp(z_log_covs)

        z_samples = sample_diag_gaussians(z_means, z_covs, n_samples)
        repeated_inputs = [
            local_repeat(test_inputs, n_samples) for inp in input_batch_list
        ]
        repeated_true_outputs = [
            local_repeat(test_outputs, n_samples) for out in output_batch_list
        ]

        preds = self.base_map(z_samples, repeated_inputs)

        if self.base_map.siamese_outputs:
            log_prob = 0.0
            if self.base_map.deterministic:
                for pred, output in zip(preds, repeated_true_outputs):
                    log_prob += compute_spherical_log_prob(pred, output, n_samples)
            else:
                for pred, output in zip(preds, repeated_true_outputs):
                    log_prob += compute_diag_log_prob(
                        pred[0], pred[1], output, n_samples
                    )
        else:
            if self.base_map.deterministic:
                log_prob = compute_spherical_log_prob(
                    preds[0], repeated_true_outputs[0], n_samples
                )
            else:
                preds_mean, preds_log_cov = preds[0][0], preds[0][1]
                log_prob = compute_diag_log_prob(
                    preds_mean, preds_log_cov, repeated_true_outputs[0], n_samples
                )

        return log_prob


    def compute_ELBO_KL(self, posteriors):
        '''
            We always deal with spherical Gaussian prior
        '''
        z_means = posteriors['means']
        z_log_covs = posteriors['log_covs']
        z_covs = torch.exp(z_log_covs)
        KL = 0.5 * torch.sum(
            1.0 + z_log_covs - z_means**2 - z_covs
        )
        return KL

