import abc

class MetaTaskParamsSampler(metaclass=abc.ABCMeta):
    def __init__(self):
        pass
    

    @abc.abstractmethod
    def __iter__(self):
        '''
            returns an iterator that iterates and returns all possible
            task_params and obs_task_params

            forcing the existence of an iterator will lead to less bugs
            in the long run. it is used for example when generating meta-train
            and meta-test expert trajectories.
        '''
        pass
    

    @abc.abstractmethod
    def sample(self):
        pass


    @abc.abstractmethod
    def sample_unique(self, num):
        pass
