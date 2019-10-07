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

EVAL_SEED = 89205
NUM_EPS_PER_POST = 20

NUM_EVAL_TASKS = 16


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


def gather_eval_data(policy, np_encoder, expert_buffer_for_eval_tasks, max_k=8, sample_from_prior=False):
    # return all the metrics we would need for evaluating the models
    # for each trajectory we need to know 1) was it successful 2) was it a good reach
    # policy.cuda()
    # np_encoder.cuda()

    policy.eval()
    np_encoder.eval()

    params_sampler = _BaseParamsSampler(random=52269, num_colors=NUM_EVAL_TASKS)
    env = EvalEnv()

    all_statistics = {}
    task_num = 0

    '''
    algorithm = [tasks]
    task = [contexts]
    context = [post_samples]
    # post samples run on the same set of trajs
    post_samples = [trajs]
    trajs \in {0,1}
    '''
    algorithm_good_reach = []
    algorithm_solved = []
    for task_params, obs_task_params in params_sampler:
        print('\tEvaluating task %d...' % task_num)
        task_num += 1
        env.reset(task_params=task_params, obs_task_params=obs_task_params)
        task_id = env.task_identifier

        task_good_reach = []
        task_solved = []

        for context_num in range(4):
            print('\t\tTry with new context, number %d...' % context_num)
            # get a single trajectory context
            list_of_trajs = expert_buffer_for_eval_tasks.sample_trajs_from_task(
                task_id,
                1
            )
            post_dist = np_encoder([list_of_trajs])
            context_good_reach = []
            context_solved = []

            # evaluate all posterior sample trajs with same initial state
            env_seed = np.random.randint(0, high=10000)

            for post_sample_num in range(1,max_k+1):
                z = post_dist.sample()
                z = z.cpu().data.numpy()[0]
                if sample_from_prior:
                    z = np.random.normal(size=z.shape)
                post_cond_policy = PostCondMLPPolicyWrapper(policy, z)
                # reset the env seed
                env.seed(seed=env_seed)

                post_good_reach = []
                post_solved = []
                for t in range(20):
                    stacked_path = rollout_path(
                        env,
                        task_params,
                        obs_task_params,
                        post_cond_policy
                    )
                    # print(stacked_path['observations'][0])
                    stats = env.log_statistics([stacked_path])
                    if stats['Percent_Good_Reach'] > 0:
                        post_good_reach.append(1.0)
                    else:
                        post_good_reach.append(0.0)
                    
                    if stats['Percent_Solved'] > 0:
                        post_solved.append(1.0)
                    else:
                        post_solved.append(0.0)
                    # paths_for_context_size.append(stacked_path)
                
                context_good_reach.append(post_good_reach)
                context_solved.append(post_solved)
            
            task_good_reach.append(context_good_reach)
            task_solved.append(context_solved)
        
        algorithm_good_reach.append(task_good_reach)
        algorithm_solved.append(task_solved)
    return {
        'algorithm_good_reach': algorithm_good_reach,
        'algorithm_solved': algorithm_solved
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
    sub_exp = exp_specs['sub_exp']
    sample_from_prior = exp_specs['sample_from_prior']

    print('\n\nUSING GPU\n\n')
    ptu.set_gpu_mode(True)

    # seed
    set_seed(EVAL_SEED)

    # load the expert replay buffer
    expert_buffer = joblib.load(EXPERT_BUFFER_PATH)['meta_train']['context']

    # do eval
    all_stats = []
    try:
        alg = joblib.load(osp.join(exp_path, sub_exp, 'best_meta_test.pkl'))['algorithm']
        print('\nLOADED ALGORITHM\n')
        if exp_specs['evaluating_np_airl']:
            alg.cuda()
            alg.main_policy.preprocess_model.cuda()
        else:
            alg.cuda()
    except Exception as e:
        print('Failed on {}/{}'.format(exp_path, sub_exp))
        raise e

    print('\n\nEVALUATING SUB EXPERIMENT %d...' % len(all_stats))
    
    sub_exp_stats = gather_eval_data(
        alg.main_policy if exp_specs['evaluating_np_airl'] else alg.policy,
        alg.encoder,
        expert_buffer,
        sample_from_prior=sample_from_prior
    )
    all_stats.append(sub_exp_stats)

    # save all of the results
    save_name = 'all_recall_at_k_stats.pkl'
    if sample_from_prior: save_name = 'prior_sampled_' + save_name
    joblib.dump(
        {'all_recall_at_k_stats': all_stats},
        osp.join(exp_path, sub_exp, save_name),
        compress=3
    )
