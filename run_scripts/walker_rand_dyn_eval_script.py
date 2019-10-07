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
from rlkit.envs.walker_random_dynamics import Walker2DRandomDynamicsEnv
from rlkit.envs.walker_random_dynamics import _MetaExpertTrainParamsSampler as TrainParamsSampler
from rlkit.envs.walker_random_dynamics import _MetaExpertTestParamsSampler as TestParamsSampler

from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.policies import PostCondMLPPolicyWrapper

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.misc import imsave

EVAL_SEED = 89205

def rollout_path(env, task_params, obs_task_params, post_cond_policy, max_path_length, task_idx):
    cur_eval_path_builder = PathBuilder()
    
    # reset the env using the params
    observation = env.reset(task_params=task_params, obs_task_params=obs_task_params)
    terminal = False
    task_identifier = env.task_identifier

    while (not terminal) and len(cur_eval_path_builder) < max_path_length:
        agent_obs = observation['obs']
        action, agent_info = post_cond_policy.get_action(agent_obs)
        
        next_ob, raw_reward, terminal, env_info = (env.step(action))
        # img = env.render(mode='rgb_array', width=200, height=200)
        if len(cur_eval_path_builder) % 10 == 0:
            # img = env.render(mode='rgb_array')

            env._wrapped_env._get_viewer('rgb_array').render(200, 200, camera_id=0)
            # window size used for old mujoco-py:
            data = env._wrapped_env._get_viewer('rgb_array').read_pixels(200, 200, depth=False)
            # original image is upside-down, so flip it
            img = data[::-1, :, :]
            imsave('plots/walker_irl_frames/walker_task_%02d_step_%03d.png' % (task_idx, len(cur_eval_path_builder)), img)
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

    return cur_eval_path_builder.get_all_stacked()


def gather_eval_data(
    policy,
    encoder,
    env,
    num_diff_context=4,
    num_rollouts_per_context=4,
    deterministic=True,
    expert_buffer_for_eval_tasks=None,
    params_sampler=None,
    eval_non_meta_policy=False
    ):
    policy.eval()
    if not eval_non_meta_policy:
        encoder.eval()

    all_statistics = {}
    task_num = 0

    for task_params, obs_task_params in params_sampler:
        task_rets = []
        # print('\tEvaluating task %.4f...' % obs_task_params)
        # print('\n\tEvaluating task {}...'.format(obs_task_params))
        print('\n\tEvaluating task {}...'.format(task_num))
        task_num += 1
        env.reset(task_params=task_params, obs_task_params=obs_task_params)
        task_id = env.task_identifier

        for _ in range(num_diff_context):
            if not eval_non_meta_policy:
                list_of_trajs = expert_buffer_for_eval_tasks.sample_trajs_from_task(
                    task_id,
                    1
                )
                post_dist = encoder([list_of_trajs])
                z = post_dist.mean
                z = z.cpu().data.numpy()[0]

                post_cond_policy = PostCondMLPPolicyWrapper(policy, z)
                post_cond_policy.policy.eval()
                post_cond_policy.deterministic = deterministic
            else:
                if deterministic:
                    print('DETERMINISTIC')
                    post_cond_policy = MakeDeterministic(policy)
                else:
                    post_cond_policy = policy
            
            for _ in range(num_rollouts_per_context):
                max_path_length = 1000

                stacked_path = rollout_path(
                    env,
                    task_params,
                    obs_task_params,
                    post_cond_policy,
                    max_path_length,
                    task_num
                )
                task_rets.append(np.sum(stacked_path['rewards']))

        print('Returns: %.1f +/- %.1f' % (np.mean(task_rets), np.std(task_rets)))
        all_statistics[task_id] = task_rets
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

    if exp_specs['use_gpu']:
        print('\n\nUSING GPU\n\n')
        ptu.set_gpu_mode(True)

    # seed
    set_seed(EVAL_SEED)

    assert exp_specs['eval_mode'] in ['train', 'test']
    expert_buffer = joblib.load(exp_specs['eval_buffer_path'])['meta_train']['context']
    train_buffer_data = joblib.load(exp_specs['train_buffer_path'])
    if exp_specs['eval_unscaled_env']:
        print('EVAL UNSCALED')
        env = Walker2DRandomDynamicsEnv()
    else:
        env = ScaledMetaEnv(
            Walker2DRandomDynamicsEnv(),
            obs_mean=train_buffer_data['obs_mean'],
            obs_std=train_buffer_data['obs_std'],
            acts_mean=train_buffer_data['acts_mean'],
            acts_std=train_buffer_data['acts_std'],
        )

    # do eval
    all_stats = []
    all_paths = []
    if exp_specs['sub_exp_mode']:
        all_paths = [osp.join(exp_path, 'params.pkl')]
        # all_paths = [osp.join(exp_path, 'best_meta_test.pkl')]
    else:
        for sub_exp in os.listdir(exp_path):
            if os.path.isdir(osp.join(exp_path, sub_exp)):
                all_paths.append(osp.join(exp_path, sub_exp, 'params.pkl'))
    for p in all_paths:
        try:
            d = joblib.load(p)
            if exp_specs['eval_non_meta_policy']:
                print('NON META')
                policy = d['policy']
                encoder = None
            else:
                # policy = d['training_policy']
                policy = d['policy']
                encoder = d['encoder']
            print('\nLOADED CHECKPOINT\n')
        except Exception as e:
            print('Failed on {}'.format(p))
            raise e
        
        if exp_specs['use_gpu']:
            policy.cuda()
            if not exp_specs['eval_non_meta_policy']:
                encoder.cuda()
        else:
            policy.cpu()
            if not exp_specs['eval_non_meta_policy']:
                encoder.cpu()

        print('\n\nEVALUATING SUB EXPERIMENT %d...' % len(all_stats))
        

        sub_exp_stats = gather_eval_data(
            policy,
            encoder,
            env,
            num_diff_context=exp_specs['num_diff_context'],
            num_rollouts_per_context=exp_specs['num_rollouts_per_context'],
            deterministic=exp_specs['eval_deterministic'],
            expert_buffer_for_eval_tasks=expert_buffer,
            params_sampler=TrainParamsSampler() if exp_specs['eval_mode'] == 'train' else TestParamsSampler(),
            eval_non_meta_policy=exp_specs['eval_non_meta_policy']
        )
        all_stats.append(sub_exp_stats)

    # save all of the results
    save_name = 'all_eval_stats.pkl'
    joblib.dump(
        {'all_eval_stats': all_stats},
        osp.join(exp_path, save_name),
        compress=3
    )
