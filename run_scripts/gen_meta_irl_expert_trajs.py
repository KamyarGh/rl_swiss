import numpy as np
from random import randint
from gym.envs.mujoco import HalfCheetahEnv, InvertedPendulumEnv, ReacherEnv
from gym.spaces import Dict

from copy import deepcopy

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.gen_exp_traj_algorithm import ExpertTrajGeneratorAlgorithm

from rlkit.core import logger
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.sac.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.sac.policies import MakeDeterministic

from rlkit.envs import get_meta_env, get_meta_env_params_iters
from rlkit.scripted_experts import get_scripted_policy
from rlkit.data_management.env_replay_buffer import MetaEnvReplayBuffer
from rlkit.data_management.path_builder import PathBuilder

from rlkit.envs.ant_linear_classification import AntLinearClassifierEnv
from rlkit.envs.walker_random_dynamics import Walker2DRandomDynamicsEnv

import yaml
import argparse
import importlib
import psutil
import os
from os import path
import argparse
import joblib
import json


def fill_buffer(
    buffer,
    meta_env,
    expert,
    expert_policy_specs,
    task_params_sampler,
    num_rollouts_per_task,
    max_path_length,
    no_terminal=False,
    policy_is_scripted=False,
    render=False,
    check_for_success=False,
    wrap_absorbing=False,
    subsample_factor=1,
    deterministic=True
):
    expert_uses_pixels = expert_policy_specs['policy_uses_pixels']
    expert_uses_task_params = expert_policy_specs['policy_uses_task_params']
    # hack
    if 'concat_task_params_to_policy_obs' in expert_policy_specs:
        concat_task_params_to_policy_obs = expert_policy_specs['concat_task_params_to_policy_obs']
    else:
        concat_task_params_to_policy_obs = False

    # this is something for debugging few shot fetch demos
    # first_complete_list = []

    for task_params, obs_task_params in task_params_sampler:
        # print('Doing Task {}...'.format(task_params))
        
        debug_stats = []
        meta_env.reset(task_params=task_params, obs_task_params=obs_task_params)
        task_id = meta_env.task_identifier

        num_rollouts_completed = 0
        while num_rollouts_completed < num_rollouts_per_task:
            cur_rollout_rewards = 0
            print('\tRollout %d...' % num_rollouts_completed)
            cur_path_builder = PathBuilder()

            observation = meta_env.reset(task_params=task_params, obs_task_params=obs_task_params)
            if policy_is_scripted:
                policy = expert
                policy.reset(meta_env)
            else:
                if isinstance(meta_env, AntLinearClassifierEnv):
                    policy = expert.get_exploration_policy(meta_env.targets[meta_env.true_label])
                    # print(meta_env.true_label)
                    if deterministic: policy.deterministic = True
                elif isinstance(meta_env, Walker2DRandomDynamicsEnv):
                    # print('WalkerEnv')
                    policy = expert.get_exploration_policy(obs_task_params)
                    if deterministic:
                        # print('deterministic')
                        policy = MakeDeterministic(policy)
                else:
                    policy = expert.get_exploration_policy(obs_task_params)
                    if deterministic: policy.deterministic = True
            terminal = False

            subsample_mod = randint(0, subsample_factor-1)
            step_num = 0

            rollout_debug = []
            while (not terminal) and step_num < max_path_length:
                if render: meta_env.render()
                if isinstance(meta_env.observation_space, Dict):
                    if expert_uses_pixels:
                        agent_obs = observation['pixels']
                    else:
                        agent_obs = observation['obs']
                        if isinstance(meta_env, AntLinearClassifierEnv):
                            if meta_env.use_relative_pos:
                                agent_obs = np.concatenate([agent_obs[:-12], meta_env.get_body_com("torso").flat]).copy()
                            else:
                                agent_obs = agent_obs[:-12]
                else:
                    agent_obs = observation
                if expert_uses_task_params:
                    if concat_task_params_to_policy_obs:
                        agent_obs = np.concatenate((agent_obs, obs_task_params), -1)
                    # else:
                        # agent_obs = {'obs': agent_obs, 'obs_task_params': obs_task_params}

                if policy_is_scripted:
                    action, agent_info = policy.get_action(agent_obs, meta_env, len(cur_path_builder))
                else:
                    action, agent_info = policy.get_action(agent_obs)

                next_ob, raw_reward, terminal, env_info = (meta_env.step(action))
                # raw_reward = -1.0 * env_info['run_cost']
                # raw_reward = env_info['vel']
                cur_rollout_rewards += raw_reward
                # if step_num < 200: cur_rollout_rewards += raw_reward

                # rollout_debug.append(env_info['l2_dist'])

                if no_terminal: terminal = False
                if wrap_absorbing:
                    terminal_array = np.array([False])
                else:
                    terminal_array = np.array([terminal])
                
                reward = raw_reward
                reward = np.array([reward])

                if step_num % subsample_factor == subsample_mod:
                    cur_path_builder.add_all(
                        observations=observation,
                        actions=action,
                        rewards=reward,
                        next_observations=next_ob,
                        terminals=terminal_array,
                        absorbing=np.array([0.0, 0.0]),
                        agent_infos=agent_info,
                        env_infos=env_info
                    )
                observation = next_ob
                step_num += 1

            if terminal and wrap_absorbing:
                '''
                If we wrap absorbing states, two additional
                transitions must be added: (s_T, s_abs) and
                (s_abs, s_abs). In Disc Actor Critic paper
                they make s_abs be a vector of 0s with last
                dim set to 1. Here we are going to add the following:
                ([next_ob,0], random_action, [next_ob, 1]) and
                ([next_ob,1], random_action, [next_ob, 1])
                This way we can handle varying types of terminal states.
                '''
                # next_ob is the absorbing state
                # for now just sampling random action
                cur_path_builder.add_all(
                    observations=next_ob,
                    actions=action,
                    # the reward doesn't matter
                    rewards=0.0,
                    next_observations=next_ob,
                    terminals=np.array([False]),
                    absorbing=np.array([0.0, 1.0]),
                    agent_infos=agent_info,
                    env_infos=env_info
                )
                cur_path_builder.add_all(
                    observations=next_ob,
                    actions=action,
                    # the reward doesn't matter
                    rewards=0.0,
                    next_observations=next_ob,
                    terminals=np.array([False]),
                    absorbing=np.array([1.0, 1.0]),
                    agent_infos=agent_info,
                    env_infos=env_info
                )
            
            # if necessary check if it was successful
            if check_for_success:
                was_successful = np.sum([e_info['is_success'] for e_info in cur_path_builder['env_infos']]) > 0
                if was_successful:
                    print('\t\tSuccessful')
                else:
                    print('\t\tNot Successful')
            if (check_for_success and was_successful) or (not check_for_success):
                for timestep in range(len(cur_path_builder)):
                    buffer.add_sample(
                        cur_path_builder['observations'][timestep],
                        cur_path_builder['actions'][timestep],
                        cur_path_builder['rewards'][timestep],
                        cur_path_builder['terminals'][timestep],
                        cur_path_builder['next_observations'][timestep],
                        task_id,
                        agent_info=cur_path_builder['agent_infos'][timestep],
                        env_info=cur_path_builder['env_infos'][timestep],
                        absorbing=cur_path_builder['absorbing'][timestep]
                    )
                buffer.terminate_episode(task_id)                
                num_rollouts_completed += 1
                print('\t\tReturn: %.2f' % (cur_rollout_rewards))
                debug_stats.append(cur_rollout_rewards)

                # print('Min L2: %.3f' % np.min(rollout_debug))
            
            # print(policy.first_time_all_complete)
            # first_complete_list.append(expert_policy.first_time_all_complete)
    # print(np.histogram(first_complete_list, bins=100))
        print('%.1f +/- %.1f' % (np.mean(debug_stats), np.std(debug_stats)))
        print('\n\n')


def experiment(specs):
    # this is just bad nomenclature: specific_exp_dir is the dir where you will find
    # the specific experiment run (with a particular seed etc.) of the expert policy
    # to use for generating trajectories
    if not specs['use_scripted_policy']:
        policy_is_scripted = False
        expert = joblib.load(path.join(specs['expert_dir'], 'extra_data.pkl'))['algorithm']
        # max_path_length = expert.max_path_length
        max_path_length = specs['max_path_length']
        if max_path_length != expert.max_path_length:
            print('\n\nUsing max_path_length {}! Expert\'s was {}!'.format(max_path_length, expert.max_path_length))
        attrs = [
            'max_path_length', 'policy_uses_pixels',
            'policy_uses_task_params',
            'no_terminal'
        ]
        expert_policy_specs = {att: getattr(expert, att) for att in attrs}
        expert_policy_specs['wrap_absorbing'] = specs['wrap_absorbing']
        no_terminal = specs['no_terminal']
    else:
        policy_is_scripted = True
        max_path_length = specs['max_path_length']
        wrap_absorbing = specs['wrap_absorbing']
        expert_policy_specs = {
            'policy_uses_pixels': specs['policy_uses_pixels'],
            'policy_uses_task_params': specs['policy_uses_task_params'],
            'concat_task_params_to_policy_obs': specs['concat_task_params_to_policy_obs']
        }
        no_terminal = specs['no_terminal']
        expert = get_scripted_policy(specs['scripted_policy_name'])


    # set up the envs
    env_specs = specs['env_specs']
    meta_train_env, meta_test_env = get_meta_env(env_specs)

    # get the task param iterators for the meta envs
    meta_train_params_sampler, meta_test_params_sampler = get_meta_env_params_iters(env_specs)

    # make the replay buffers
    if specs['wrap_absorbing']:
        _max_buffer_size = (max_path_length+2) * specs['num_rollouts_per_task']        
    else:
        _max_buffer_size = max_path_length * specs['num_rollouts_per_task']
    _max_buffer_size = int(np.ceil(_max_buffer_size / float(specs['subsample_factor']))) + 10
    # + 10 is just in case somewhere someone uses ._size of replay buffers incorrectly
    
    buffer_constructor = lambda env_for_buffer: MetaEnvReplayBuffer(
        _max_buffer_size,
        env_for_buffer,
        policy_uses_pixels=specs['student_policy_uses_pixels'],
        # we don't want the student policy to be looking at true task parameters
        policy_uses_task_params=False,
        concat_task_params_to_policy_obs=False
    )

    train_context_buffer = buffer_constructor(meta_train_env)
    test_context_buffer = buffer_constructor(meta_test_env)

    render = specs['render']
    check_for_success = specs['check_for_success']
    # fill the train buffers
    fill_buffer(
        train_context_buffer, meta_train_env,
        expert,
        expert_policy_specs,
        meta_train_params_sampler, specs['num_rollouts_per_task'], max_path_length,
        no_terminal=no_terminal, wrap_absorbing=specs['wrap_absorbing'],
        policy_is_scripted=policy_is_scripted, render=render,
        check_for_success=check_for_success,
        subsample_factor=specs['subsample_factor'],
        deterministic=specs['get_deterministic_expert_demos']
    )
    train_test_buffer = deepcopy(train_context_buffer)

    # fill the test buffers
    fill_buffer(
        test_context_buffer, meta_train_env,
        expert,
        expert_policy_specs,
        meta_test_params_sampler, specs['num_rollouts_per_task'], max_path_length,
        no_terminal=no_terminal, wrap_absorbing=specs['wrap_absorbing'],
        policy_is_scripted=policy_is_scripted, render=render,
        check_for_success=check_for_success,
        subsample_factor=specs['subsample_factor'],
        deterministic=specs['get_deterministic_expert_demos']
    )
    test_test_buffer = deepcopy(test_context_buffer)

    # save the replay buffers
    d = {
        'meta_train': {
            'context': train_context_buffer,
            'test': train_test_buffer
        },
        'meta_test': {
            'context': test_context_buffer,
            'test': test_test_buffer
        }
    }
    logger.save_extra_data(d)

    return 1


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)
    
    exp_id = exp_specs['exp_id']
    exp_prefix = exp_specs['exp_name']
    seed = exp_specs['seed']
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    experiment(exp_specs)
