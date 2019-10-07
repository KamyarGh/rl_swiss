import torch
import numpy as np
import joblib

from rlkit.envs import get_meta_env, get_meta_env_params_iters

# d_path = '/ais/gobi6/kamyar/oorl_rlkit/output/test-gen-meta-irl-trajs/test_gen_meta_irl_trajs_2018_11_21_01_21_33_0000--s-0/extra_data.pkl'
d_path = '/ais/gobi6/kamyar/oorl_rlkit/output/test-pixel-traj-gen/test_pixel_traj_gen_2018_11_29_00_11_20_0000--s-0/extra_data.pkl'
d = joblib.load(d_path)

# env_specs = {
#     'base_env_name': 'meta_simple_meta_reacher',
#     'normalized': False
# }
env_specs = {
    'base_env_name': 'meta_simple_meta_reacher',
    'normalized': False,
    'need_pixels': True,
    'render_kwargs': {
      'height': 64,
      'width': 64,
      'camera_id': 0
    }
}

meta_train_env, meta_test_env = get_meta_env(env_specs)

meta_train_params_sampler, meta_test_params_sampler = get_meta_env_params_iters(env_specs)
buffer = d['meta_train']['context']

buffer.policy_uses_pixels = True

task_params, obs_task_params = meta_train_params_sampler.sample()
meta_train_env.reset(task_params=task_params, obs_task_params=obs_task_params)
task_id = meta_train_env.task_identifier

# print(buffer.num_steps_can_sample())

# print(buffer.task_replay_buffers.keys())

# trajs, sample_params = buffer.sample_trajs(2, num_tasks=2)
# print(sample_params)
# print(buffer.policy_uses_pixels)
# print(trajs[0][0]['observations'].keys())
# print(trajs[0][0]['observations']['obs'].shape)
# print(trajs[0][0]['observations']['pixels'].shape)

# print(task_id)
# trajs, sample_params = buffer.sample_trajs(2, task_identifiers=[task_id, task_id])
# print(sample_params)
# print(len(trajs))
# print(len(trajs[0]))
# print(trajs[0][1])

# print(task_id)
# trajs = buffer.sample_trajs_from_task(task_id, 4)
# print(len(trajs))
# # print(trajs[1])
# for t in trajs:
#     print(t['observations'].shape)

# samples, params = buffer.sample_random_batch(4, num_task_params=3)
# print(len(samples))
# for d in samples:
#     for k, v in d.items():
#         print(k, v.shape)

# print(task_id)
# samples, params = buffer.sample_random_batch(4, task_identifiers_list=[task_id, task_id])
# print(params)
# print(len(samples))
# for d in samples:
#     for k, v in d.items():
#         print(k, v.shape)

# print(buffer.task_replay_buffers[task_id]._cur_start)
# print(buffer.task_replay_buffers[task_id]._traj_endpoints)
