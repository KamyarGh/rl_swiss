import abc
import numpy as np

from rlkit.core.meta_base_algorithm import MetaBaseAlgorithm


class TorchMetaBaseAlgorithm(MetaBaseAlgorithm, metaclass=abc.ABCMeta):
    """
    A generic torch meta learning algorithm. Can be inherited to implement
    meta-learning (MAML, PEARL, etc.) or meta-imitation-learning (Meta-BC,
    Meta-DAgger, Meta-IRL/SMILe)
    """
    @property
    @abc.abstractmethod
    def networks(self):
        """
        Used in many settings such as moving to devices
        """
        pass

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device):
        for net in self.networks:
            net.to(device)
