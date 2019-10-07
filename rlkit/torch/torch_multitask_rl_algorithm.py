import numpy as np
from collections import OrderedDict

from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.torch_meta_base_algorithm import TorchMetaBaseAlgorithm

from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.policies import PostCondMLPPolicyWrapper


class TorchMultiTaskRLAlgorithm(TorchMetaBaseAlgorithm):
    """
    For training a multi-task RL algorithm where the RL algorithm observes
    the ground-truth task parameters. Can be used for example to train
    expert policies for a meta-imitation-learning.

    The meta-environments used for training need to implement the
    following functions:
    - task_params_to_obs_task_params
    - task_params_to_task_identifier
    """
    def __init__(
        self, trainer, tasks_per_batch, batch_size_per_task,
        num_train_steps_per_train_call, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.trainer = trainer
        self.num_tasks_per_batch = num_tasks_per_batch
        self.batch_size_per_task = batch_size_per_task
        self.num_train_steps_per_train_call = num_train_steps_per_train_call


    def get_batch(self):
        sampled_task_params = self.train_task_params_sampler.sample_unique(self.num_tasks_per_batch)
        obs_task_params_list = list(map(
            lambda tp: self.env_factory.task_params_to_obs_task_params(tp),
            sampled_task_params
        ))
        task_ids = list(map(
            lambda tp: self.env_factory.task_params_to_task_identifiers(tp),
            sampled_task_params
        ))
        batches_list = self.replay_buffer.sample_random_batch(
            task_ids,
            self.num_samples_per_task_per_batch
        )

        # concatenate the observed task parameters
        for batch, obs_task_params in zip(batches_list, obs_task_params_list):
            obs_task_params = np.repeat(
                obs_task_params[None,:],
                self.num_samples_per_task_per_batch,
                axis=0
            )
            batch['observations'] = np.hstack((batch['observations'], obs_task_params))
            batch['next_observations'] = np.hstack((batch['next_observations'], obs_task_params))
        
        # concatenate all the task batches into one big bag
        return_batch = {}
        for k in batches_list[0]:
            return_batch[k] = np.vstack(b[k] for b in batches_list)
        
        # convert to torch tensors and put on correct device
        return np_to_pytorch_batch(return_batch)


    def get_exploration_policy(self, task_params):
        obs_task_params = self.env_factory.task_params_to_obs_task_params(task_params)
        return PostCondMLPPolicyWrapper(self.policy, obs_task_params)
    

    def get_eval_policy(self, task_params):
        obs_task_params = self.env_factory.task_params_to_obs_task_params(task_params)
        pol = PostCondMLPPolicyWrapper(self.policy, obs_task_params)
        if self.eval_deterministic:
            pol = MakeDeterministic(pol)
        return pol


    @property
    def networks(self):
        return self.trainer.networks


    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)


    def _do_training(self, epoch):
        for _ in range(self.num_train_steps_per_train_call):
            self.trainer.train_step(self.get_batch())


    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(epoch=epoch)
        data_to_save.update(self.trainer.get_snapshot())
        return data_to_save


    def evaluate(self, epoch):
        self.eval_statistics = self.trainer.get_eval_statistics()
        super().evaluate(epoch)


    def _end_epoch(self):
        self.trainer.end_epoch()
        super()._end_epoch()
