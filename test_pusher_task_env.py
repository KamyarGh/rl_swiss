from rlkit.envs.state_matching_pusher_env import StateMatchingPusherEnv

from rlkit.torch.sac.policies import MakeDeterministic

from rlkit.samplers.in_place import InPlacePathSampler


import numpy as np

import joblib

env = StateMatchingPusherEnv()
print(env.action_space)

# env.reset()
# env.reset()
# env.reset()
# env.reset()
# env.reset()


# for _ in range(100):
#     env.render()
#     action = np.zeros(4)
#     action[0] = 1
#     # action[1] = 1
#     env.step(action)

# p = joblib.load('pusher_policy.pkl')['policy']
# p = joblib.load('best_pusher_policy.pkl')['policy']
p = joblib.load('new_best_pusher.pkl')['policy']
p = MakeDeterministic(p)

# while True:
#     env.render()

while True:
    obs = env.reset()
    for i in range(100):
        env.render()

        action = env.action_space.sample()
        # action, *_ = p.get_action(obs)
        # print(action)
        # action = np.zeros(7)

        obs, *_ = env.step(action)
        # print(i)
        # print(np.linalg.norm(obs[:2] - obs[3:5]))
        # print(obs[:2], obs[3:5])
        # print(obs[0] - obs[3], obs[1] - obs[4])
        # print(obs)

    # print('\n\n\n\n')


# eval_sampler = InPlacePathSampler(
#     env=env,
#     policy=p,
#     max_samples=100*200,
#     max_path_length=100,
#     policy_uses_pixels=False,
#     policy_uses_task_params=False,
#     concat_task_params_to_policy_obs=False
# )
# test_paths = eval_sampler.obtain_samples()

# print(env.log_new_ant_multi_statistics(test_paths, 0, 'plots/junk_vis'))
