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
from rlkit.torch.sac.policies import PostCondMLPPolicyWrapper

from rlkit.envs.ant_rand_direc_2d import AntRandDirec2DEnv
# from rlkit.envs.ant_rand_direc_2d import _Expert180DegreesParamsSampler as EvalParamsSampler
from rlkit.envs.ant_rand_direc_2d import _DebugParamsSamplerV1 as EvalParamsSampler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


'''
Things I need:
- (done) being able to set seeds for the replay buffers
- an expert dataset generated using the task identities I am using for evaluation
'''

EVAL_SEED = 89205

def rollout_path(env, task_params, obs_task_params, post_cond_policy, max_path_length, eval_expert, render):
    cur_eval_path_builder = PathBuilder()
    
    # reset the env using the params
    observation = env.reset(task_params=task_params, obs_task_params=obs_task_params)
    terminal = False
    task_identifier = env.task_identifier
    this_roll_debug = 0.0

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
        this_roll_debug += env_info['reward_forward']

        if render: env.render()

    print(this_roll_debug / 100.0)
    return cur_eval_path_builder.get_all_stacked()


def gather_eval_data(
    alg,
    sample_from_prior=False,
    num_rollouts_per_task=8,
    context_sizes=[4],
    deterministic=True,
    eval_expert=False,
    just_loading_policy=False,
    render=False
    ):
    if not eval_expert: alg.encoder.eval()

    all_statistics = {}
    task_num = 0

    params_sampler = EvalParamsSampler()
    if not just_loading_policy:
        env = alg.env
    else:
        env = AntRandDirec2DEnv()

    for task_params, obs_task_params in params_sampler:
        _task_dict = {}
        # print('\tEvaluating task %.4f...' % obs_task_params)
        print('\n\tEvaluating task {}'.format(obs_task_params))
        task_num += 1
        env.reset(task_params=task_params, obs_task_params=obs_task_params)
        task_id = env.task_identifier

        for context_size in context_sizes:
            _cont_size_dict = {}
            print('\t\tTry with context size: %d...' % context_size)

            # evaluate all posterior sample trajs with same initial state
            env_seed = np.random.randint(0, high=10000)

            if sample_from_prior: raise NotImplementedError
            # z = post_dist.sample()
            # z = z.cpu().data.numpy()[0]
            # if sample_from_prior:
            #     z = np.random.normal(size=z.shape)
            if eval_expert:
                if just_loading_policy:
                    post_cond_policy = PostCondMLPPolicyWrapper(alg, obs_task_params)
                else:
                    post_cond_policy = alg.get_eval_policy(obs_task_params)
            else:
                post_cond_policy = alg.get_eval_policy(task_id, mode='meta_test')
            post_cond_policy.policy.eval()
            post_cond_policy.deterministic = deterministic

            # reset the env seed
            env.seed(seed=env_seed)
            _rets = []
            _min_dists = []
            _last_100 = []
            for _ in range(num_rollouts_per_task):
                if just_loading_policy:
                    # max_path_length = 200
                    # max_path_length = 300
                    max_path_length = 100
                else:
                    alg.max_path_length
                stacked_path = rollout_path(
                    env,
                    task_params,
                    obs_task_params,
                    post_cond_policy,
                    max_path_length,
                    eval_expert,
                    render
                )
                obs = np.array([d['obs'] for d in stacked_path['observations']])

        all_statistics[task_id] = _task_dict
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
    sample_from_prior = exp_specs['sample_from_prior']

    if exp_specs['use_gpu']:
        print('\n\nUSING GPU\n\n')
        ptu.set_gpu_mode(True)

    # seed
    set_seed(EVAL_SEED)

    # do eval
    all_stats = []
    all_paths = []
    if exp_specs['sub_exp_mode']:
        all_paths = [osp.join(exp_path, 'extra_data.pkl')]
    else:
        for sub_exp in os.listdir(exp_path):
            if os.path.isdir(osp.join(exp_path, sub_exp)):
                all_paths.append(osp.join(exp_path, sub_exp, 'extra_data.pkl'))
    for p in all_paths:
        try:
            # alg = joblib.load(osp.join(exp_path, sub_exp, 'best_meta_test.pkl'))['algorithm']
            if exp_specs['just_loading_policy']:
                assert exp_specs['eval_expert']
                alg = joblib.load(p)['policy']
            else:
                alg = joblib.load(p)['algorithm']
            print('\nLOADED ALGORITHM\n')
        except Exception as e:
            print('Failed on {}'.format(p))
            raise e
        
        if exp_specs['use_gpu']:
            alg.cuda()
        else:
            alg.cpu()

        print('\n\nEVALUATING SUB EXPERIMENT %d...' % len(all_stats))
        
        sub_exp_stats = gather_eval_data(
            alg,
            sample_from_prior=sample_from_prior,
            context_sizes=exp_specs['context_sizes'],
            num_rollouts_per_task=exp_specs['num_rollouts_per_task'],
            deterministic=exp_specs['eval_deterministic'],
            eval_expert=exp_specs['eval_expert'], # this means we are evaluating the expert meta-rl policy,
            just_loading_policy=exp_specs['just_loading_policy'],
            render=exp_specs['render']
        )
        all_stats.append(sub_exp_stats)

    # save all of the results
    save_name = 'all_eval_stats.pkl'
    if sample_from_prior: save_name = 'prior_sampled_' + save_name
    joblib.dump(
        {'all_eval_stats': all_stats},
        osp.join(exp_path, save_name),
        compress=3
    )
