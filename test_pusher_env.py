from rlkit.envs.state_matching_pusher_env_no_obj import StateMatchingPusherNoObjEnv
import numpy as np

import joblib

from rlkit.torch.sac.policies import MakeDeterministic

env = StateMatchingPusherNoObjEnv()
print(env.action_space)

env.reset()

# for _ in range(100):
#     env.render()
#     action = np.zeros(4)
#     action[0] = 1
#     # action[1] = 1
#     env.step(action)

# p = joblib.load('pusher_policy.pkl')['policy']
# p = joblib.load('sin_trace_pusher.pkl')['policy']
# p = MakeDeterministic(p)

while True:
    obs = env.reset()
    for i in range(500):
        env.render()

        action = env.action_space.sample()
        # action, *_ = p.get_action(obs)
        # print(action)

        obs, *_ = env.step(action)
