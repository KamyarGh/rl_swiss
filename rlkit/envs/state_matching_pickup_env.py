import os
import numpy as np

from gym import utils
from rlkit.envs.state_matching_fetch_env import FetchSMMEnv


MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "assets", 'fetch/pick_and_place.xml')

# Derived from the fetch pickup env, used to be called fetch pickup env
class FetchPushSMMEnv(FetchSMMEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', obs_with_time=True, episode_len=200):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            # 'robot0:slide1': 0.28,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        FetchSMMEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.1, target_in_the_air=False, target_offset=0.0,
            obj_range=0.1, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type,
            obs_with_time=obs_with_time, episode_len=episode_len)
        utils.EzPickle.__init__(self)

    # def step(self, action):
    #     obs, reward, done, info = super().step(action)

    #     # x = info['obj_pos'][0]
    #     # y = info['obj_pos'][1]

    #     # if (x < 0.98697072 - 0.3) or (x > 0.98697072 + 0.225) or (y < 0.74914774 - 0.5) or (y > 0.74914774 + 0.45):
    #     #     done = True

    #     return obs, reward, done, info
