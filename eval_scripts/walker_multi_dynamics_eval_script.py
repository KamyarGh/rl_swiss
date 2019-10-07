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

# from rlkit.envs.walker_random_dynamics import _MetaExpertEvalParamsSampler as EvalParamsSampler
from rlkit.envs.walker_random_dynamics import _MetaExpertTrainParamsSampler as EvalParamsSampler
from rlkit.envs.walker_random_dynamics import Walker2DRandomDynamicsEnv
from rlkit.torch.sac.policies import MakeDeterministic

'''
Things I need:
- (done) being able to set seeds for the replay buffers
- an expert dataset generated using the task identities I am using for evaluation
'''

EVAL_SEED = 89205

def rollout_path(env, task_params, obs_task_params, post_cond_policy, max_path_length):
    cur_eval_path_builder = PathBuilder()
    
    # reset the env using the params
    observation = env.reset(task_params=task_params, obs_task_params=obs_task_params)
    terminal = False
    task_identifier = env.task_identifier

    while (not terminal) and len(cur_eval_path_builder) < max_path_length:
        agent_obs = observation['obs']
        action, agent_info = post_cond_policy.get_action(agent_obs)
        
        next_ob, raw_reward, terminal, env_info = (env.step(action))
        terminal = False
        
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

    return cur_eval_path_builder.get_all_stacked()


def gather_eval_data(
        alg,
        num_rollouts_per_context=8,
        deterministic=True,
        num_diff_context=1,
        eval_params_sampler=None,
        expert_buffer_for_eval_tasks=None,
        evaluating_expert=False,
        eval_deterministic=True,
        eval_no_task_info=False
    ):
    context_sizes = [1]
    if not evaluating_expert:
        alg.encoder.eval()

    all_statistics = {}
    task_num = 0

    # env = alg.env
    env = Walker2DRandomDynamicsEnv()

    _means = []
    _stds = []

    for task_params, obs_task_params in eval_params_sampler:
        env.reset(task_params=task_params, obs_task_params=obs_task_params)
        task_rets = []
        print('\tEvaluating task {}...'.format(obs_task_params))
        print(task_params)
        task_num += 1
        task_id = env.task_identifier

        for context_size in context_sizes:
            _cont_size_dict = {}

            for c_idx in range(num_diff_context):
                if not evaluating_expert:
                    if eval_no_task_info:
                        print('Evaluting with no task information!')
                        new_task_params = {}
                        for k in task_params:
                            new_task_params[k] = np.ones(task_params[k].shape)
                        raise NotImplementedError()
                    else:
                        list_of_trajs = alg.expert_buffer_for_eval_tasks.sample_trajs_from_task(
                            task_id,
                            context_size
                        )
                    alg.encoder.eval()
                    post_dist = alg.encoder([list_of_trajs])
                    z = post_dist.sample()
                    z = z.cpu().data.numpy()[0]
                    # post_cond_policy = PostCondMLPPolicyWrapper(alg.main_policy, z)
                    post_cond_policy = PostCondMLPPolicyWrapper(alg.main_policy, z)
                    post_cond_policy.policy.eval()
                else:
                    # if eval_no_task_info:
                    #     print('Evaluting with no task information!')
                    #     post_cond_policy = alg.get_eval_policy(0.0*np.ones(obs_task_params.shape))
                    # else:
                    #     post_cond_policy = alg.get_eval_policy(np.ones(obs_task_params))

                    # For evaluating a standard walker expert
                    # post_cond_policy = alg.policy
                    # post_cond_policy = alg.eval_policy
                    post_cond_policy = MakeDeterministic(alg.policy)
                
                post_cond_policy.deterministic = eval_deterministic
                context_returns = []
                for _ in range(num_rollouts_per_context):
                    stacked_path = rollout_path(
                        env,
                        task_params,
                        obs_task_params,
                        post_cond_policy,
                        alg.max_path_length
                    )
                    context_returns.append(np.sum(stacked_path['rewards']))
                task_rets.extend(context_returns)

        all_statistics[task_id] = task_rets
        print('\nReturns: %.4f +/- %.4f' % (np.mean(task_rets), np.std(task_rets)))
        _means.append(np.mean(task_rets))
        _stds.append(np.std(task_rets))
    for i in range(len(_means)):
        print('%.4f +/- %.4f' % (_means[i], _stds[i]))
    return all_statistics


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)
    exp_path = exp_specs['exp_path']

    print('\n\nUSING GPU\n\n')
    ptu.set_gpu_mode(True)

    # seed
    set_seed(EVAL_SEED)

    # do eval
    all_stats = []
    all_paths = []
    fname = 'best_meta_test.pkl' if exp_specs['use_best'] else 'extra_data.pkl'
    if exp_specs['sub_exp_mode']:
        all_paths = [osp.join(exp_path, fname)]
    else:
        for sub_exp in os.listdir(exp_path):
            if os.path.isdir(osp.join(exp_path, sub_exp)):
                all_paths.append(osp.join(exp_path, sub_exp, fname))
    
    for p in all_paths:
        try:
            # alg = joblib.load(osp.join(exp_path, sub_exp, 'best_meta_test.pkl'))['algorithm']
            alg = joblib.load(p)['algorithm']
            print('\nLOADED ALGORITHM\n')
        except Exception as e:
            print('Failed on {}/{}'.format(exp_path, sub_exp))
            raise e
        
        if exp_specs['use_gpu']:
            alg.cuda()
        else:
            alg.cpu()

        print('\n\nEVALUATING SUB EXPERIMENT %d...' % len(all_stats))
        
        eval_params_sampler = EvalParamsSampler()
        if exp_specs['evaluating_expert']:
            expert_buffer_for_eval_tasks = None
        else:
            raise NotImplementedError()
        
        sub_exp_stats = gather_eval_data(
            alg,
            num_rollouts_per_context=exp_specs['num_rollouts_per_context'],
            num_diff_context=exp_specs['num_diff_context'],
            eval_params_sampler=eval_params_sampler,
            expert_buffer_for_eval_tasks=expert_buffer_for_eval_tasks,
            evaluating_expert=exp_specs['evaluating_expert'],
            eval_deterministic=exp_specs['eval_deterministic'],
            eval_no_task_info=exp_specs['eval_no_task_info']
        )
        all_stats.append(sub_exp_stats)

    # save all of the results
    save_name = 'all_eval_stats.pkl'
    joblib.dump(
        {'all_eval_stats': all_stats},
        osp.join(exp_path, save_name),
        compress=3
    )
