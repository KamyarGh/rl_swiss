import os
import numpy as np

from gym import utils
from rlkit.envs.state_matching_fetch_env import StateMatchingFetchEnv


# Ensure we get the path separator correct on windows
# MODEL_XML_PATH = 'rlkit/envs/assets/fetch/slide.xml'
MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "assets", 'fetch/slide.xml')


# ORIGINAL
# class FetchSlideEnv(fetch_env.FetchEnv, utils.EzPickle):
#     def __init__(self, reward_type='sparse'):
#         initial_qpos = {
#             'robot0:slide0': 0.05,
#             'robot0:slide1': 0.48,
#             'robot0:slide2': 0.0,
#             'object0:joint': [1.7, 1.1, 0.4, 1., 0., 0., 0.],
#         }
#         fetch_env.FetchEnv.__init__(
#             self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
#             gripper_extra_height=-0.02, target_in_the_air=False, target_offset=np.array([0.4, 0.0, 0.0]),
#             obj_range=0.1, target_range=0.3, distance_threshold=0.05,
#             initial_qpos=initial_qpos, reward_type=reward_type)
#         utils.EzPickle.__init__(self)


class StateMatchingFetchSlideEnv(StateMatchingFetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.05,
            'robot0:slide1': 0.38,
            'robot0:slide2': 0.0,
            'object0:joint': [1.7, 1.1, 0.4, 1., 0., 0., 0.],
        }
        StateMatchingFetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.09, target_in_the_air=False, target_offset=np.array([0.4, 0.0, 0.0]),
            obj_range=0.0, target_range=0.3, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        x = info['obj_pos'][0]
        y = info['obj_pos'][1]

        if (x < 0.98697072 - 0.3) or (x > 0.98697072 + 0.225) or (y < 0.74914774 - 0.5) or (y > 0.74914774 + 0.45):
            done = True

        return obs, reward, done, info
