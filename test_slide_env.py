from rlkit.envs.state_matching_slide_env import StateMatchingFetchSlideEnv
from rlkit.torch.sac.policies import MakeDeterministic

import numpy as np
import joblib


env = StateMatchingFetchSlideEnv()
print(env.action_space)

env.reset()

# for _ in range(100):
#     env.render()
#     action = np.zeros(4)
#     action[0] = 1
#     # action[1] = 1
#     env.step(action)

p = joblib.load('slide_policy.pkl')['policy']
# p = MakeDeterministic(p)


# while True:
#     env.render()

while True:
    obs = env.reset()
    for i in range(200):
        env.render()
        # action = np.zeros(2)
        # action[1] = 0.3
        # action[0] = 0.3

        # action = env.action_space.sample()
        
        action, *_ = p.get_action(obs)

        obs, rew, done, info = env.step(action)
        # print('----')
        # print(obs[:2])
        # print(obs[2:4])

        print(info['obj_pos'][2])
        if done: break
