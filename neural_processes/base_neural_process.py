import abc

class BaseNeuralProcess(object, metaclass=abc.ABCMeta):
    '''
    General interface for a neural process
    '''
    @abc.abstractmethod
    def train_step(self, batch):
        pass
    

    @abc.abstractmethod
    def infer_posterior_params(self, batch):
        pass

    
    @abc.abstractmethod
    def compute_ELBO(self, posteriors, batch):
        pass

    @abc.abstractmethod
    def compute_cond_log_likelihood(self, posteriors, batch):
        pass
