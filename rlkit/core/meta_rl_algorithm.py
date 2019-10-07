import abc
import pickle
import time
from copy import deepcopy

import gtimer as gt
import numpy as np

from rlkit.core import logger
from rlkit.data_management.env_replay_buffer import MetaEnvReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.policies.base import ExplorationPolicy
from rlkit.envs.wrapped_absorbing_env import WrappedAbsorbingEnv

from gym.spaces import Dict


class MetaRLAlgorithm(metaclass=abc.ABCMeta):
    '''
        While True:
            generate trajectories for a batch of different task settings
            update the models
    '''
    def __init__(
            self,
            env,
            train_task_params_sampler,
            test_task_params_sampler,
            training_env=None,
            num_epochs=100,
            num_rollouts_per_epoch=10,
            num_rollouts_between_updates=10,
            num_initial_rollouts_for_all_train_tasks=0,
            min_rollouts_before_training=10,
            max_path_length=1000,
            discount=0.99,
            replay_buffer_size_per_task=20000,
            reward_scale=1.0,
            render=False,
            save_replay_buffer=False,
            save_algorithm=False,
            save_environment=False,
            replay_buffer=None,
            policy_uses_pixels=False,
            policy_uses_task_params=False,
            wrap_absorbing=False,
            freq_saving=1,
            do_not_train=False,
            do_not_eval=False,
            # some environment like halfcheetah_v2 have a timelimit that defines the terminal
            # this is used as a minor hack to turn off time limits
            no_terminal=False,
            save_best=False,
            save_best_after_epoch=0,
            **kwargs
        ):
        self.training_env = training_env or pickle.loads(pickle.dumps(env))
        # self.training_env = training_env or deepcopy(env)
        self.num_epochs = num_epochs
        self.num_rollouts_per_epoch = num_rollouts_per_epoch
        self.num_rollouts_between_updates = num_rollouts_between_updates
        self.num_initial_rollouts_for_all_train_tasks = num_initial_rollouts_for_all_train_tasks
        self.min_rollouts_before_training = min_rollouts_before_training
        self.max_path_length = max_path_length
        self.discount = discount
        self.replay_buffer_size_per_task = replay_buffer_size_per_task
        self.reward_scale = reward_scale
        self.render = render
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment
        self.policy_uses_pixels = policy_uses_pixels
        self.policy_uses_task_params = policy_uses_task_params

        self.action_space = env.action_space
        self.obs_space = env.observation_space
        self.env = env
        if replay_buffer is None:
            replay_buffer = MetaEnvReplayBuffer(
                self.replay_buffer_size_per_task,
                self.training_env,
                policy_uses_pixels=self.policy_uses_pixels,
            )
        self.replay_buffer = replay_buffer

        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        self._current_path_builder = PathBuilder()
        self._exploration_paths = []
        self.wrap_absorbing = wrap_absorbing
        if self.wrap_absorbing:
            assert isinstance(env, WrappedAbsorbingEnv), 'Env is not wrapped!'
        self.freq_saving = freq_saving
        self.no_terminal = no_terminal

        self.train_task_params_sampler = train_task_params_sampler
        self.test_task_params_sampler = test_task_params_sampler
        self.do_not_train = do_not_train
        self.do_not_eval = do_not_eval
        self.best_meta_test = np.float('-inf')
        self.save_best = save_best
        self.save_best_after_epoch = save_best_after_epoch


    def train(self, start_epoch=0):
        self.pretrain()
        if start_epoch == 0:
            params = self.get_epoch_snapshot(-1)
            logger.save_itr_params(-1, params)
        self.training_mode(False)
        # self._n_env_steps_total = start_epoch * self.num_env_steps_per_epoch
        gt.reset()
        gt.set_def_unique(False)
        self.train_online(start_epoch=start_epoch)


    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        if self.num_initial_rollouts_for_all_train_tasks > 0:
            self.generate_rollouts_for_all_train_tasks(
                self.num_initial_rollouts_for_all_train_tasks
            )
            print('\nGenerated Initial Task Rollouts\n')
            gt.stamp('initial_task_rollouts')
    

    def generate_rollouts_for_all_train_tasks(self, num_rollouts_per_task):
        '''
        This is a simple work-around for a problem that arises when sampling
        batches for NP-AIRL because you need to be able to sample a minimum
        number of trajectories per train task.
        I will try to replace this with a better fix later.
        '''
        i = 0
        for task_params, obs_task_params in self.train_task_params_sampler:
            print('rollouts for task %d' % i)
            # print('new task rollout')
            for _ in range(num_rollouts_per_task):
                self.generate_exploration_rollout(
                    task_params=task_params, obs_task_params=obs_task_params
                )
            i += 1
        # exploration paths maintains the exploration paths in one epoch
        # so that we can analyze certain properties of the trajs if we
        # wanted. we don't want these trajs to count towards that really.
        self._exploration_paths = []


    def generate_exploration_rollout(self, task_params=None, obs_task_params=None):
        observation, task_identifier = self._start_new_rollout(
            task_params=task_params, obs_task_params=obs_task_params
        )
        
        # _current_path_builder is initialized to a new one everytime
        # you call handle rollout ending
        # When you start a new rollout, self.exploration_policy
        # is set to the one for the current task
        terminal = False
        while (not terminal) and len(self._current_path_builder) < self.max_path_length:
            if isinstance(self.obs_space, Dict):
                if self.policy_uses_pixels:
                    agent_obs = observation['pixels']
                else:
                    agent_obs = observation['obs']
            else:
                agent_obs = observation
            action, agent_info = self._get_action_and_info(agent_obs)
            if self.render:
                self.training_env.render()
            
            next_ob, raw_reward, terminal, env_info = (self.training_env.step(action))
            raw_reward *= self.reward_scale
            if self.no_terminal:
                terminal = False
            
            self._n_env_steps_total += 1
            reward = raw_reward
            terminal = np.array([terminal])
            reward = np.array([reward])
            self._handle_step(
                observation,
                action,
                reward,
                next_ob,
                np.array([False]) if self.wrap_absorbing else terminal,
                task_identifier,
                agent_info=agent_info,
                env_info=env_info,
            )
            observation = next_ob

        if terminal and self.wrap_absorbing:
            raise NotImplementedError("I think they used 0 actions for this")
            # next_ob is the absorbing state
            # for now just using the action from the previous timesteps
            # as well as agent info and env info
            self._handle_step(
                next_ob,
                action,
                # the reward doesn't matter cause it will be
                # overwritten by the model that defines the reward
                # e.g. the discriminator in GAIL
                reward,
                next_ob,
                terminal,
                task_identifier,
                agent_info=agent_info,
                env_info=env_info
            )
        
        self._handle_rollout_ending(task_identifier)


    def train_online(self, start_epoch=0):
        # No need for training mode to be True when generating trajectories
        # training mode is automatically set to True
        # in _try_to_train and before exiting
        # it that function it reverts it to False
        self.training_mode(False)
        self._current_path_builder = PathBuilder()
        self._n_rollouts_total = 0

        for epoch in gt.timed_for(
            range(start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            self._start_epoch(epoch)
            # print('epoch')
            for _ in range(self.num_rollouts_per_epoch):
                # print('rollout')
                task_params, obs_task_params = self.train_task_params_sampler.sample()
                self.generate_exploration_rollout(task_params=task_params, obs_task_params=obs_task_params)

                # print(self._n_rollouts_total)
                if self._n_rollouts_total % self.num_rollouts_between_updates == 0:
                    gt.stamp('sample')
                    # print('train')
                    if not self.do_not_train: self._try_to_train(epoch)
                    gt.stamp('train')

            if not self.do_not_eval:
                self._try_to_eval(epoch)
                gt.stamp('eval')

            self._end_epoch()


    def _try_to_train(self, epoch):
        if self._can_train():
            self.training_mode(True)
            self._do_training(epoch)
            self._n_train_steps_total += 1
            self.training_mode(False)


    def _try_to_eval(self, epoch):
        if epoch % self.freq_saving == 0:
            logger.save_extra_data(self.get_extra_data_to_save(epoch))
        if self._can_evaluate():
            self.evaluate(epoch)

            if epoch % self.freq_saving == 0:
                params = self.get_epoch_snapshot(epoch)
                logger.save_itr_params(epoch, params)
            table_keys = logger.get_table_key_set()

            # logger.record_tabular(
            #     "Number of train steps total",
            #     self._n_policy_train_steps_total,
            # )
            logger.record_tabular(
                "Number of env steps total",
                self._n_env_steps_total,
            )
            logger.record_tabular(
                "Number of rollouts total",
                self._n_rollouts_total,
            )

            times_itrs = gt.get_times().stamps.itrs
            train_time = times_itrs['train'][-1]
            sample_time = times_itrs['sample'][-1]
            eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
            epoch_time = train_time + sample_time + eval_time
            total_time = gt.get_times().total

            logger.record_tabular('Train Time (s)', train_time)
            logger.record_tabular('(Previous) Eval Time (s)', eval_time)
            logger.record_tabular('Sample Time (s)', sample_time)
            logger.record_tabular('Epoch Time (s)', epoch_time)
            logger.record_tabular('Total Train Time (s)', total_time)

            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")


    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        return (
            len(self._exploration_paths) > 0
            and self._n_rollouts_total >= self.min_rollouts_before_training
        )


    def _can_train(self):
        return self._n_rollouts_total >= self.min_rollouts_before_training


    def _get_action_and_info(self, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        self.exploration_policy.set_num_steps_total(self._n_env_steps_total)
        return self.exploration_policy.get_action(
            observation,
        )


    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)


    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()


    def _start_new_rollout(self, task_params=None, obs_task_params=None):
        if task_params is None:
            task_params, obs_task_params = self.train_task_params_sampler.sample()
        observation = self.training_env.reset(task_params=task_params, obs_task_params=obs_task_params)
        task_id = self.training_env.task_identifier

        self.exploration_policy = self.get_exploration_policy(obs_task_params)
        self.exploration_policy.reset()
        return observation, task_id


    def _handle_path(self, path, task_identifier):
        """
        Naive implementation: just loop through each transition.
        :param path:
        :return:
        """
        for (
            ob,
            action,
            reward,
            next_ob,
            terminal,
            agent_info,
            env_info
        ) in zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"],
        ):
            self._handle_step(
                ob,
                action,
                reward,
                next_ob,
                terminal,
                task_identifier,
                agent_info=agent_info,
                env_info=env_info,
            )
        self._handle_rollout_ending(task_identifier)


    def _handle_step(
            self,
            observation,
            action,
            reward,
            next_observation,
            terminal,
            task_identifier,
            agent_info,
            env_info,
    ):
        """
        Implement anything that needs to happen after every step
        :return:
        """
        self._current_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
            task_identifiers=task_identifier
        )
        self.replay_buffer.add_sample(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            task_identifier=task_identifier,
            agent_info=agent_info,
            env_info=env_info,
        )


    def _handle_rollout_ending(self, task_identifier):
        """
        Implement anything that needs to happen after every rollout.
        """
        self.replay_buffer.terminate_episode(task_identifier)
        self._n_rollouts_total += 1
        if len(self._current_path_builder) > 0:
            self._exploration_paths.append(
                self._current_path_builder.get_all_stacked()
            )
            self._current_path_builder = PathBuilder()


    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save


    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save
    

    @abc.abstractmethod
    def get_exploration_policy(self, obs_task_params):
        '''
            Since for each task a meta-irl algorithm needs to somehow
            use some expert demonstrations, this is a convenience method
            to get a version of the policy that is handling this stuff internally.

            Example:
            In the neural process meta-irl method, for a given task we need to,
            take some demonstrations, infer the posterior, sample from the posterior,
            then conidtion the policy by concatenating the sample to any observations
            that are passed to the policy. So internally, in np_bc and np_airl, when
            we call get_exploration_policy we set the latent sample for a
            PostCondReparamTanhMultivariateGaussianPolicy and return that. From then on,
            whenever we call get_action on the policy, it internally concatenates the
            latent to the observation passed to it.
        '''
        pass
    

    @abc.abstractmethod
    def get_eval_policy(self, obs_task_params):
        '''
            Since for each task a meta-irl algorithm needs to somehow
            use some expert demonstrations, this is a convenience method
            to get a version of the policy that is handling this stuff internally.

            Example:
            In the neural process meta-irl method, for a given task we need to,
            take some demonstrations, infer the posterior, sample from the posterior,
            then conidtion the policy by concatenating the sample to any observations
            that are passed to the policy. So internally, in np_bc and np_airl, when
            we call get_exploration_policy we set the latent sample for a
            PostCondReparamTanhMultivariateGaussianPolicy and return that. From then on,
            whenever we call get_action on the policy, it internally concatenates the
            latent to the observation passed to it.
        '''
        pass
    

    @abc.abstractmethod
    def obtain_eval_samples(self, epoch):
        pass


    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass


    @abc.abstractmethod
    def cuda(self):
        """
        Turn cuda on.
        :return:
        """
        pass
    

    @abc.abstractmethod
    def cpu(self):
        """
        Turn cuda off.
        :return:
        """
        pass


    @abc.abstractmethod
    def evaluate(self, epoch):
        """
        Evaluate the policy, e.g. save/print progress.
        :param epoch:
        :return:
        """
        pass


    @abc.abstractmethod
    def _do_training(self):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass
