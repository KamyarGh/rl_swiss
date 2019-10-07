import torch
from numpy.random import uniform
from numpy import pi

class SinusoidalTask():
    # These are the parameters from MAML
    AMPLITUDE_RANGE = [0.1, 5.0]
    PHASE_RANGE = [0.0, pi]
    X_RANGE = [-5.0, 5.0]

    # AMPLITUDE_RANGE = [1.0, 1.5]
    # PHASE_RANGE = [0.0, 0.5]
    # X_RANGE = [-5.0, 5.0]

    # AMPLITUDE_RANGE = [1.0, 1.0]
    # PHASE_RANGE = [0.0, np.pi]
    # X_RANGE = [-5.0, 5.0]

    # AMPLITUDE_RANGE = [1.0, 5.0]
    # PHASE_RANGE = [0.0, 0.0]
    # X_RANGE = [-5.0, 5.0]

    def __init__(self, A=None, phase=None):
        self.A = A if A is not None else \
            uniform(low=self.AMPLITUDE_RANGE[0], high=self.AMPLITUDE_RANGE[1])
        self.phase = phase if phase is not None else \
            uniform(low=self.PHASE_RANGE[0], high=self.PHASE_RANGE[1])
    
    def sample(self, N=1):
        X = torch.rand(N,1)
        X.mul_(self.X_RANGE[1] - self.X_RANGE[0])
        X.add_(self.X_RANGE[0])
        Y = self.A * torch.sin(X - self.phase)
        return X, Y
