import yaml
import argparse
import os
from os import path as osp
import argparse
import joblib
from time import sleep

import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.few_shot_fetch_env import _BaseParamsSampler
from rlkit.envs.few_shot_fetch_env import StatsFor50Tasks25EachScaled0p9LinearBasicFewShotFetchEnv as EvalEnv
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.sac.policies import PostCondMLPPolicyWrapper

from rlkit.data_management.path_builder import PathBuilder

'''
Things I need:
- (done) being able to set seeds for the replay buffers
- an expert dataset generated using the task identities I am using for evaluation
'''

MAX_PATH_LENGTH = 65
EXPERT_BUFFER_PATH = '/scratch/gobi2/kamyar/oorl_rlkit/expert_demos/few_shot_fetch_eval_expert_trajs/extra_data.pkl'

ENV_EVAL_SEED = 89205


def rollout_path(env, task_params, obs_task_params, post_cond_policy):
    cur_eval_path_builder = PathBuilder()
    
    # reset the env using the params
    observation = env.reset(task_params=task_params, obs_task_params=obs_task_params)
    terminal = False
    task_identifier = env.task_identifier

    while (not terminal) and len(cur_eval_path_builder) < MAX_PATH_LENGTH:
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


def gather_eval_data(policy, np_encoder, expert_buffer_for_eval_tasks):
    # return all the metrics we would need for evaluating the models
    # for each trajectory we need to know 1) was it successful 2) was it a good reach
    # policy.cuda()
    # np_encoder.cuda()

    policy.eval()
    np_encoder.eval()

    params_sampler = _BaseParamsSampler(random=52269, num_colors=16)
    env = EvalEnv()

    all_statistics = {}
    task_num = 0

    algorithm_all_percent_good_reach = []
    algorithm_all_percent_solved = []
    for task_params, obs_task_params in params_sampler:
        print('\tEvaluating task %d...' % task_num)
        task_num += 1
        env.reset(task_params=task_params, obs_task_params=obs_task_params)
        task_id = env.task_identifier

        for context_num in range(4):
            print('\t\tTry with new context number %d...' % context_num)
            # get a context
            list_of_trajs = expert_buffer_for_eval_tasks.sample_trajs_from_task(
                task_id,
                1
            )
            post_dist = np_encoder([list_of_trajs])
            all_good_reach_for_context = [0 for _ in range(20)]
            all_solved_for_context = [0 for _ in range(20)]
            for post_sample_num in range(4):
                z = post_dist.sample()
                z = z.cpu().data.numpy()[0]
                post_cond_policy = PostCondMLPPolicyWrapper(policy, z)
                # reset the env seed
                env.seed(seed=ENV_EVAL_SEED)
                for t in range(20):
                    stacked_path = rollout_path(
                        env,
                        task_params,
                        obs_task_params,
                        post_cond_policy
                    )
                    # print(stacked_path['observations'][0])
                    stats = env.log_statistics([stacked_path])
                    if stats['Percent_Good_Reach'] > 0: all_good_reach_for_context[t] = 1.0
                    if stats['Percent_Solved'] > 0: all_solved_for_context[t] = 1.0
                    # paths_for_context_size.append(stacked_path)
            algorithm_all_percent_good_reach.append(np.mean(all_good_reach_for_context))
            algorithm_all_percent_solved.append(np.mean(all_solved_for_context))
    return {
        'algorithm_all_percent_good_reach': algorithm_all_percent_good_reach,
        'algorithm_all_percent_solved': algorithm_all_percent_solved
    }


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='path to an experiment dir, stats will be computed for all subdir experiments')
    args = parser.parse_args()
    exp_path = args.path
    
    print('\n\nUSING GPU\n\n')
    ptu.set_gpu_mode(True)

    # load the expert replay buffer
    expert_buffer = joblib.load(EXPERT_BUFFER_PATH)['meta_train']['context']

    # for each subdir experiment evaluate it
    all_stats = []
    for subdir in os.listdir(args.path):
        try:
            alg = joblib.load(osp.join(args.path, subdir, 'best_meta_test.pkl'))['algorithm']
            print('\nLOADED ALGORITHM\n')
            alg.cuda()
            alg.main_policy.preprocess_model.cuda()
        except:
            continue

        print('\n\nEVALUATING SUB EXPERIMENT %d...' % len(all_stats))
        
        sub_exp_stats = gather_eval_data(
            # alg.policy,
            alg.main_policy,
            alg.encoder,
            expert_buffer
        )
        all_stats.append(sub_exp_stats)

    # save all of the results
    joblib.dump({'all_few_shot_eval_stats': all_stats}, osp.join(args.path, 'all_few_shot_eval_stats.pkl'), compress=3)
