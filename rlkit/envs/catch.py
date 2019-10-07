import numpy as np
from dnc.envs.base  import KMeansEnv

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.misc import logger
from rllab.misc.overrides import overrides

import os.path as osp

raise NotImplementedError('This is taken from DNC repo and needs to be made to work with this repo')

class CatchEnv(KMeansEnv, Serializable):
    
    FILE = osp.join(osp.abspath(osp.dirname(__file__)), 'assets/catch.xml')

    def __init__(
            self, start_pos=(.1,1.7), start_noise=0.2, frame_skip=2,
            *args, **kwargs):

        self.start_pos = start_pos
        self.start_noise = start_noise
        
        super(CatchEnv, self).__init__(frame_skip=frame_skip, *args, **kwargs)   
        Serializable.__init__(self, start_pos, start_noise, frame_skip, *args, **kwargs)
    
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:],
            self.model.data.qvel.flat[:],
        ]).reshape(-1)
    
    def step(self, action):
        self.forward_dynamics(action)
        
        ball_position = self.get_body_com("object")
        
        mitt_position = self.get_body_com("jaco_link_finger_1") + self.get_body_com("jaco_link_finger_2") + self.get_body_com("jaco_link_finger_3")
        mitt_position = mitt_position/3.
        difference = np.linalg.norm((ball_position - mitt_position))
        
        reward = 1 if (difference < .15 and ball_position[2] > .05) else 0
        
        done = False
        return Step(self.get_current_obs(), float(reward), done, distance=difference, in_hand=(reward == 1))
    
    @overrides
    def reset(self):
        
        qpos = self.init_qpos.copy().reshape(-1)
        qvel = self.init_qvel.copy().reshape(-1)

        qpos[6] += 1
        qpos[10] += 1.5
        
        qvel[:3] = [-7, 0, 0]
        
        self.current_start = self.propose() # Proposal
        qpos[1:3] = self.current_start
                
        self.set_state(qpos.reshape(-1), qvel)

        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        self.last_location = self.get_body_com("object")

        return self.get_current_obs()

    def retrieve_centers(self,full_states):
        return full_states[:, 1:3]

    def propose_original(self):
        return (np.random.rand(2) - 0.5) * 2 * self.start_noise + np.array(self.start_pos)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 3.5
        self.viewer.cam.azimuth = -180
        self.viewer.cam.elevation = -20
    
    @overrides
    def log_diagnostics(self, paths,prefix=''):
        progs = np.array([
            np.sum(path['env_infos']['in_hand'] > 0)
            for path in paths
        ]+[1])


        logger.record_tabular(prefix+'PercentWithReward', 100*np.mean(progs > 0))
        logger.record_tabular(prefix+'PercentOK', 100*np.mean(progs > 5))
        logger.record_tabular(prefix+'PercentGood', 100*np.mean(progs > 15))
        logger.record_tabular(prefix+'AverageTimeInHand', self.frame_skip*.01*np.mean(progs[progs.nonzero()]))

if __name__ == "__main__":
    env = CatchEnv()
    for itr in range(5):
        env.reset()
        for t in range(50):
            env.step(env.action_space.sample())
            env.render()