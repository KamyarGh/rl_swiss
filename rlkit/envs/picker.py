'''
Taken and modified from divide and conquer paper
Main difference is reward-shaping
'''

import numpy as np
import os.path as osp

from rlkit.envs.mujoco_env import MujocoEnv
from rlkit.core.serializable import Serializable
from rlkit.core import logger

MIN_OBJ_Z = 0.08

class PickerEnv(MujocoEnv, Serializable):
    """
    Picking a block, where the block position is randomized over a square region
    
    goal_args is of form ('noisy', center_of_box, half-width of box)
    
    """
    def __init__(self, goal_args=('noisy', (.6,.2), .1), frame_skip=5, *args, **kwargs):
        self.init_serialization(locals())
        self.goal_args = goal_args

        # just to make stuff work -------
        self.cur_dist = 0
        self.prev_dist = 0
        self.init_obj_z = 0
        # -------------------------------
        
        super().__init__(
            'picker.xml',
            frame_skip=frame_skip,
            automatically_set_obs_and_action_space=True
        )
        

    def get_current_obs(self):
        finger_com = self.get_body_com("jaco_link_finger_1") + self.get_body_com("jaco_link_finger_2") + self.get_body_com("jaco_link_finger_3")
        finger_com = finger_com / 3.

        return np.concatenate([
            self.sim.data.qpos.flat[:],
            self.sim.data.qvel.flat[:],
            finger_com,
        ]).reshape(-1)


    def step(self,action):
        self.sim.data.ctrl[:] = action
        
        reward = 0
        timesInHand = 0

        for _ in range(self.frame_skip):
            self.sim.step()

            # update dists for reward shaping
            self.obj_pos, dist = self._get_obj_pos_and_dist()
            self.prev_dist = self.cur_dist
            self.cur_dist = dist

            if self.is_in_hand() > 0:
                self.numClose += 1
                timesInHand += 1

            # print('dist %.4f' % dist)

            step_reward = self.reward()
            reward += step_reward

        # done = reward == 0 and self.numClose > 0 # Stop it if the block is flinged
        done = False

        ob = self.get_current_obs()

        return ob, float(reward), done, {'timeInHand': timesInHand, 'FingerToObj': self.cur_dist, 'ObjZ': self.obj_pos[2]}


    def _get_obj_pos_and_dist(self):
        obj_position = self.get_body_com("object")
        # print(obj_position)

        finger_com = self.get_body_com("jaco_link_finger_1") + self.get_body_com("jaco_link_finger_2") + self.get_body_com("jaco_link_finger_3")
        finger_com = finger_com / 3.
        
        vec_1 = obj_position - finger_com
        dist_1 = np.linalg.norm(vec_1)

        return obj_position, dist_1


    def is_in_hand(self):
        if self.cur_dist < .1 and self.obj_pos[2] > MIN_OBJ_Z:
            return 1
        return -1


    def reward(self):
        # print('--')
        # print(self.obj_pos[2] - self.init_obj_z)
        # print(self.cur_dist)
        return self.obj_pos[2] - self.init_obj_z - self.cur_dist/10.
        # ---------------------
        # shaping = self.prev_dist - self.cur_dist
        # print('shaping %.4f' % shaping)
        # if self.obj_pos[2] < 0.08:
        #     return shaping
        # elif self.cur_dist < .1 and self.obj_pos[2] > MIN_OBJ_Z:
        #     # print('----')
        #     # print('reward %.4f' % obj_pos)
        #     # print('----')
        #     return self.obj_pos[2] + shaping
        # else:
        #     return shaping


    def sample_position(self,goal_type,center=(0.6,0.2),noise=0):
        if goal_type == 'fixed':
            return [center[0],center[1],.03]
        elif goal_type == 'noisy':
            x,y = center
            return [x+(np.random.rand()-0.5)*2*noise,y+(np.random.rand()-0.5)*2*noise,.03]
        else:
            raise NotImplementedError()


    def retrieve_centers(self,full_states):
        return full_states[:,9:12]


    def propose_original(self):
        return self.sample_position(*self.goal_args)


    def reset(self):
        # print('--------------------------------------------')
        qpos = self.init_qpos.copy().reshape(-1)
        qvel = self.init_qvel.copy().reshape(-1) + np.random.uniform(low=-0.005,
                high=0.005, size=self.init_qvel.size)

        qpos[1] = -1

        self.position = self.propose_original() # Proposal

        qpos[9:12] = self.position
        qvel[9:12] = 0

        # print(qpos[9:12])

        self.set_state(qpos.reshape(-1), qvel)

        self.numClose = 0

        self.obj_pos, dist = self._get_obj_pos_and_dist()
        self.cur_dist = dist
        self.prev_dist = dist
        self.init_obj_z = self.obj_pos[2]

        return self.get_current_obs()


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0
        self.viewer.cam.azimuth = +0.0
        self.viewer.cam.elevation = -40


    def log_diagnostics(self, paths, prefix=''):

        # print(paths[0])
        # print(paths[0]['env_infos'])
        # print(len(paths[0]['env_infos']))

        # for path in paths:
            # print(path['env_infos'])

        max_obj_z = np.array(
            [np.max([s['ObjZ'] for s in path['env_infos']]) for path in paths]
        )
        arg_max_obj_z = [np.argmax([s['ObjZ'] for s in path['env_infos']]) for path in paths]
        dist_at_max_obj_z = []
        for i, path in enumerate(paths):
            dist_at_max_obj_z.append(path['env_infos'][arg_max_obj_z[i]]['FingerToObj'])
        dist_at_max_obj_z = np.array(dist_at_max_obj_z)

        min_f2o = np.array(
            [np.min([s['FingerToObj'] for s in path['env_infos']]) for path in paths]
        )
        timeOffGround = np.array(
            [np.sum(s['timeInHand'] for s in path['env_infos']) for path in paths]
        )

        # print(timeOffGround)
        timeInAir = timeOffGround[timeOffGround.nonzero()]

        if len(timeInAir) == 0:
            timeInAir = [0]

        avgPct = lambda x: round(np.mean(x) * 100, 2)

        logger.record_tabular(prefix+'PctPicked', avgPct(timeOffGround > .3))
        logger.record_tabular(prefix+'PctReceivedReward', avgPct(timeOffGround > 0))
        
        logger.record_tabular(prefix+'AverageTimeInAir',np.mean(timeOffGround))
        logger.record_tabular(prefix+'MaxTimeInAir',np.max(timeOffGround ))

        logger.record_tabular(prefix+'AvgObjZMax',np.mean(max_obj_z))
        logger.record_tabular(prefix+'AvgDistAtObjZMax',np.mean(dist_at_max_obj_z))
        logger.record_tabular(prefix+'MaxObjZMax',np.max(max_obj_z))
        logger.record_tabular(prefix+'DistAtMaxObjZMax',dist_at_max_obj_z[np.argmax(max_obj_z)])

        logger.record_tabular(prefix+'AvgF2OMin',np.mean(min_f2o))
        logger.record_tabular(prefix+'MinF2OMin',np.min(min_f2o))
