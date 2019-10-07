import torch
from numpy.random import uniform
from numpy import pi

class LinearTask():
    # These are the parameters from MAML
    SLOPE_RANGE = [-1.0, 1.0]
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

    def __init__(self, slope=None):
        self.slope = slope if slope is not None else \
            uniform(low=self.SLOPE_RANGE[0], high=self.SLOPE_RANGE[1])
    
    def sample(self, N=1):
        X = torch.rand(N,1)
        X.mul_(self.X_RANGE[1] - self.X_RANGE[0])
        X.add_(self.X_RANGE[0])
        Y = self.slope * X
        return X, Y
