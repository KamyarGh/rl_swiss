from dnc.envs.base  import KMeansEnv
import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger

import os.path as osp

raise NotImplementedError('This is taken from DNC repo and needs to be made to work with this repo')

class LobberEnv(KMeansEnv, Serializable):
    
    FILE = osp.join(osp.abspath(osp.dirname(__file__)), 'assets/lob.xml')

    def __init__(self, box_center=(0,0), box_noise=0.4, frame_skip=5, *args, **kwargs):

        self.box_center = box_center
        self.box_noise = box_noise

        super(LobberEnv, self).__init__(frame_skip=frame_skip, *args, **kwargs)
        Serializable.__init__(self, box_center, box_noise, frame_skip, *args, **kwargs)
    
    def get_current_obs(self):
        finger_com = self.get_body_com("jaco_link_finger_1") + self.get_body_com("jaco_link_finger_2") + self.get_body_com("jaco_link_finger_3")
        finger_com = finger_com / 3.
        
        return np.concatenate([
            self.model.data.qpos.flat[:],
            self.model.data.qvel.flat[:],
            finger_com,
            self.relativeBoxPosition,
        ]).reshape(-1)

    def step(self,action):
        
        self.model.data.ctrl = action

        # Taking Steps in the Environment
        
        reward = 0
        for _ in range(self.frame_skip):
            self.model.step()
            step_reward = self.timestep_reward()
            reward += step_reward

        # Reached the End of Trajectory        
        
        done = False
        onGround = self.touching_group("geom_object", ["ground", "goal_wall1", "goal_wall2", "goal_wall3", "goal_wall4"])
        if onGround and self.numClose > 10:
            reward += self.final_reward()
            done = True
 
        ob = self.get_current_obs()
        new_com = self.model.data.com_subtree[0]
        self.dcom = new_com - self.current_com
        self.current_com = new_com

        # Recording Metrics

        obj_position = self.get_body_com("object")
        goal_position = self.get_body_com("goal")
        distance = np.linalg.norm((goal_position - obj_position)[:2])
        normalizedDistance = distance / self.init_block_goal_dist
        
        return Step(ob, float(reward), done, distance=distance, norm_distance=normalizedDistance)

    @overrides
    def reset(self):
        self.numClose = 0

        qpos = self.init_qpos.copy().reshape(-1)
        qvel = self.init_qvel.copy().reshape(-1) + np.random.uniform(low=-0.005,
                high=0.005, size=self.model.nv)

        qpos[1] = -1
        qpos[9:12] = np.array((0.6, 0.2,0.03))
        qvel[9:12] = 0

        self.relativeBoxPosition = self.propose() # Proposal
        qpos[-2:] += self.relativeBoxPosition
                
        self.set_state(qpos.reshape(-1), qvel)

        # Save initial distance between object and goal
        obj_position = self.get_body_com("object")
        goal_position = self.get_body_com("goal")
        self.init_block_goal_dist = np.linalg.norm(obj_position - goal_position)

        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_obs()

    def timestep_reward(self):
        obj_position = self.get_body_com("object")

        if obj_position[2] < 0.08:
            return 0

        finger_com = self.get_body_com("jaco_link_finger_1") + self.get_body_com("jaco_link_finger_2") + self.get_body_com("jaco_link_finger_3")
        finger_com = finger_com / 3.
        
        vec_1 = obj_position - finger_com
        dist_1 = np.linalg.norm(vec_1)

        if dist_1 < .1 and obj_position[0] > .2:
            self.numClose += 1
            return obj_position[2]
        else:
            return 0

    def final_reward(self):
        obj_position = self.get_body_com("object")
        goal_position = self.get_body_com("goal")

        vec_2 = obj_position - goal_position
        dist_2 = np.linalg.norm(vec_2[:2])
        normalized_dist_2 = dist_2 / self.init_block_goal_dist
        clipped_dist_2 = min(1.0, normalized_dist_2)

        if dist_2 < .1:
            return 40
            
        reward = 1 - clipped_dist_2

        return 40 * reward

    def retrieve_centers(self,full_states):
        return full_states[:,16:18]-self.init_qpos.copy().reshape(-1)[-2:]

    def propose_original(self):
        return np.array(self.box_center) + 2*(np.random.random(2)-0.5)*self.box_noise

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0
        self.viewer.cam.azimuth = +60.0
        self.viewer.cam.elevation = -30
        
    @overrides
    def log_diagnostics(self, paths, prefix=''):

        progs = np.array([
            path['env_infos']['norm_distance'][-1] for path in paths
        ])

        inGoal = np.array([
            path['env_infos']['distance'][-1] < .1 for path in paths
        ]) 
        
        avgPct = lambda x: round(np.mean(x)*100,3)

        logger.record_tabular(prefix+'PctInGoal', avgPct(inGoal))
        logger.record_tabular(prefix+'AverageFinalDistance', np.mean(progs))
        logger.record_tabular(prefix+'MinFinalDistance', np.min(progs ))