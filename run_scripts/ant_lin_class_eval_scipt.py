'''
Given only a single context trajectory, infer the posterior and evaluate
for each k, if we take k samples from the posterior, what is percentage of
tasks that are solved by taking the OR over all k for the same test envs.
'''
import yaml
import argparse
import os
from os import path as osp
import argparse
import joblib
from time import sleep

import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.sac.policies import PostCondMLPPolicyWrapper

from rlkit.data_management.path_builder import PathBuilder

from rlkit.envs.wrappers import ScaledMetaEnv
from rlkit.envs.ant_linear_classification import AntLinearClassifierEnv
from rlkit.envs.ant_linear_classification import _ExpertTestParamsSampler as EvalParamsSampler
# from rlkit.envs.ant_linear_classification import _ExpertTrainParamsSampler as EvalParamsSampler

from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.policies import PostCondMLPPolicyWrapper

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print('Train Params Sampler')
EVAL_SEED = 89205

def rollout_path(env, task_params, obs_task_params, post_cond_policy, max_path_length):
    cur_eval_path_builder = PathBuilder()
    within_correct = False
    within_incorrect = False
    
    # reset the env using the params
    observation = env.reset(task_params=task_params, obs_task_params=obs_task_params)
    terminal = False
    task_identifier = env.task_identifier

    while (not terminal) and len(cur_eval_path_builder) < max_path_length:
        agent_obs = observation['obs']
        action, agent_info = post_cond_policy.get_action(agent_obs)
        
        next_ob, raw_reward, terminal, env_info = (env.step(action))
        terminal = False

        # print(env_info['l2_dist'])
        # print('{}: {}'.format(agent_obs[-3:], env_info['l2_dist']))
        # print(agent_obs)
        # print(env_info['l2_dist'])
        
        reward = raw_reward
        terminal = np.array([terminal])
        reward = np.array([reward])
        cur_eval_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_ob,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
            task_identifiers=task_identifier
        )
        observation = next_ob

        if env_info['within_radius_of_correct']:
            within_correct = True
        if env_info['within_radius_of_incorrect']:
            within_incorrect = True

    return within_correct, within_incorrect


def gather_eval_data(
        policy,
        encoder,
        env,
        expert_buffer_for_eval_tasks=None,
        num_diff_context_per_task=8,
        context_size_min=1,
        context_size_max=12,
        num_rollouts_per_context=20,
        deterministic=True,
        params_sampler=None,
    ):
    policy.eval()
    encoder.eval()

    all_success_transitions = []
    all_no_op_transitions = []

    task_num = 0
    for task_params, obs_task_params in params_sampler:
        print('\n\tEvaluating task {}...'.format(task_num))
        task_num += 1
        env.reset(task_params=task_params, obs_task_params=obs_task_params)
        task_id = env.task_identifier

        for _ in range(num_diff_context_per_task):
            print('new context transition')

            transition_success_rate = []
            transition_no_op_rate = []
            list_of_trajs = expert_buffer_for_eval_tasks.sample_trajs_from_task(
                task_id,
                context_size_max
            )
            for i in range(context_size_min, context_size_max+1):
                print('next size')
                correct = []
                incorrect = []
                no_op = []

                new_list_of_trajs = list_of_trajs[:i]
                print(len(new_list_of_trajs))
                post_dist = encoder([new_list_of_trajs])
                z = post_dist.mean
                z = z.cpu().data.numpy()[0]

                post_cond_policy = PostCondMLPPolicyWrapper(policy, z)
                post_cond_policy.policy.eval()
                post_cond_policy.deterministic = deterministic
            
                for _ in range(num_rollouts_per_context):
                    max_path_length = 50
                    within_correct, within_incorrect = rollout_path(
                        env,
                        task_params,
                        obs_task_params,
                        post_cond_policy,
                        max_path_length
                    )
                    correct.append(within_correct)
                    incorrect.append(within_incorrect)
                    no_op.append(not (within_correct or within_incorrect))
                
                transition_success_rate.append(np.mean(correct))
                transition_no_op_rate.append(np.mean(no_op))
                # task_rets.append(np.sum(stacked_path['rewards']))
            all_success_transitions.append(transition_success_rate)
            all_no_op_transitions.append(transition_no_op_rate)

            print(transition_success_rate)
            print(transition_no_op_rate)
        
        if task_num == 32: break


        # print('Returns: %.1f +/- %.1f' % (np.mean(task_rets), np.std(task_rets)))
        # all_statistics[task_id] = task_rets
    
    return {
        'all_success_transitions': all_success_transitions,
        'all_no_op_transitions': all_no_op_transitions,
    }


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)
    exp_path = exp_specs['exp_path']

    if exp_specs['use_gpu']:
        print('\n\nUSING GPU\n\n')
        ptu.set_gpu_mode(True)

    # seed
    # set_seed(EVAL_SEED)

    d = joblib.load('/scratch/hdd001/home/kamyar/expert_demos/norm_rel_pos_ant_lin_class_64_tasks_16_demos_each_sub_1/extra_data.pkl')
    expert_buffer = d['meta_test']['context']
    # expert_buffer = d['meta_train']['context']

    env = ScaledMetaEnv(
        AntLinearClassifierEnv(use_relative_pos=True),
        obs_mean=d['obs_mean'],
        obs_std=d['obs_std'],
        acts_mean=d['acts_mean'],
        acts_std=d['acts_std'],
    )

    # do eval
    all_stats = []
    all_paths = []
    if exp_specs['sub_exp_mode']:
        # all_paths = [osp.join(exp_path, 'params.pkl')]
        all_paths = [osp.join(exp_path, 'best_meta_test.pkl')]
    else:
        for sub_exp in os.listdir(exp_path):
            if os.path.isdir(osp.join(exp_path, sub_exp)):
                assert False
                # all_paths.append(osp.join(exp_path, sub_exp, 'params.pkl'))
    for p in all_paths:
        try:
            d = joblib.load(p)
            alg = d['algorithm']
            # policy = alg.main_policy # for np_airl
            policy = alg.policy # for np_bc
            encoder = alg.encoder
            # policy = d['policy']
            # encoder = d['encoder']
            print('\nLOADED CHECKPOINT\n')
        except Exception as e:
            print('Failed on {}'.format(p))
            raise e
        
        if exp_specs['use_gpu']:
            policy.cuda()
            encoder.cuda()
        else:
            policy.cpu()
            encoder.cpu()

        print('\n\nEVALUATING SUB EXPERIMENT %d...' % len(all_stats))
        

        sub_exp_stats = gather_eval_data(
            policy,
            encoder,
            env,
            expert_buffer_for_eval_tasks=expert_buffer,
            num_diff_context_per_task=4,
            context_size_min=1,
            context_size_max=12,
            num_rollouts_per_context=20,
            deterministic=True,
            params_sampler=EvalParamsSampler(),
        )
        all_stats.append(sub_exp_stats)

    # save all of the results
    save_name = 'all_eval_stats.pkl'
    joblib.dump(
        {'all_eval_stats': all_stats},
        osp.join(exp_path, save_name),
        compress=3
    )
