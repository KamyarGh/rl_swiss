import numpy as np
import os
from time import sleep
from collections import OrderedDict
from gym.envs.robotics import utils
from gym import utils as gym_utils
from rlkit.envs import few_shot_robot_env

from rlkit.envs.meta_task_params_sampler import MetaTaskParamsSampler


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


# These random seeds are literally me typing random numbers
# (I didn't tune my random seed LOL)
def get_task_params_iterator(train_env=True):
    # when you fix this make sure really all the colors are correct
    raise NotImplementedError()
    if train_env:
        return _BaseParamsSampler(random=2497)
    else:
        return _BaseParamsSampler(random=8384)

# this debug one uses only a few tasks so we can make sure things are actually working first
def get_debug_task_params_iterator(train_env=True):
    if train_env:
        return _FullySpecifiedParamsSampler(random=7342, num_colors=1, num_random_samples_per_color=1000, same_color_radius=0.3)
    else:
        return _FullySpecifiedParamsSampler(random=7342, num_colors=1, num_random_samples_per_color=1000, same_color_radius=0.3)
        # return _BaseParamsSampler(random=7342, num_colors=1)
        # return _FullySpecifiedParamsSampler(random=7342, num_colors=1, num_random_samples_per_color=500, same_color_radius=0.3)

# another debug one
def get_zero_task_params_iterator(train_env=True):
    # when you fix this make sure really all the colors are correct
    if train_env:
        return _ZeroParamsSampler(random=2497)
    else:
        return _ZeroParamsSampler(random=8384)


class _ZeroParamsSampler(MetaTaskParamsSampler):
    def __init__(self, random=None, num_colors=50):
        super().__init__()
        if not isinstance(random, np.random.RandomState):
          random = np.random.RandomState(random)
        self._random = random
    
    def sample(self):
        color = np.zeros(3)
        return {'goal_color_center': color}, color
    
    def sample_unique(self, num):
        # this means sample uniques tasks
        color = np.zeros(3)
        return [({'goal_color_center': color}, color)]

    def __iter__(self):
        self.itr_ptr = 0
        return self
    
    def __next__(self):
        if self.itr_ptr == 1: raise StopIteration
        color = np.zeros(3)
        self.itr_ptr += 1
        return {'goal_color_center': color}, color


# task_params = {
#     'goal_color_center': np.array...
#     'other_color_center': np.array... (optional)
#     'specific_color_from_goal_radius': np.array (optional)
#     'specific_color_from_other_radius': np.array (optional)
# }
class _FullySpecifiedParamsSampler(MetaTaskParamsSampler):
    def __init__(self, random=None, num_colors=50, num_random_samples_per_color=10, same_color_radius=0.3):
        super().__init__()
        if not isinstance(random, np.random.RandomState):
          random = np.random.RandomState(random)
        self._random = random

        # sample the goal color centers
        self.num_colors = num_colors
        self.num_random_samples_per_color = num_random_samples_per_color
        self.same_color_radius = same_color_radius

        self.goal_color_centers = self._random.uniform(-1.0, 1.0, size=(self.num_colors,3))
        self.specific_params = {tuple(gc): [] for gc in self.goal_color_centers}
        for goal_color in self.goal_color_centers:
            for _ in range(self.num_random_samples_per_color):
                goal_specific_color = self._sample_color_within_radius(goal_color, self.same_color_radius)
                other_center = self._sample_color_with_min_dist(goal_color, 2 * self.same_color_radius)
                other_specific_color = self._sample_color_within_radius(other_center, self.same_color_radius)
                self.specific_params[tuple(goal_color)].append({
                    'goal_color_center': goal_color,
                    'specific_color_from_goal_radius': goal_specific_color,
                    'specific_color_from_other_radius': other_specific_color
                })

    def _sample_specific_params(self, color):
        idx = self._random.randint(0, self.num_random_samples_per_color)
        spec_par = self.specific_params[tuple(color)][idx]
        return spec_par

    def sample(self):
        idx = self._random.randint(0, self.num_colors)
        color = self.goal_color_centers[idx]
        spec_par = self._sample_specific_params(color)
        return spec_par, color
    
    def sample_unique(self, num):
        # this means sample uniques tasks
        idxs = self._random.choice(self.num_colors, size=num, replace=False)
        return list(
            map(lambda color: (self._sample_specific_params(color), color), (self.goal_color_centers[idx] for idx in idxs))
        )

    def _sample_color_within_radius(self, center, radius):
        x = self._random.normal(size=3)
        x /= np.linalg.norm(x, axis=-1)
        r = radius
        u = self._random.uniform()
        return r * (u**(1.0/3.0)) * x + center
    
    def _sample_color_with_min_dist(self, color, min_dist):
        new_color = self._random.uniform(-1.0, 1.0, size=3)
        while np.linalg.norm(new_color - color, axis=-1) < min_dist:
            new_color = self._random.uniform(-1.0, 1.0, size=3)
        return new_color

    def __iter__(self):
        self.itr_ptr = 0
        self.sub_itr_ptr = 0
        return self
    
    def __next__(self):
        if self.itr_ptr == self.num_colors: raise StopIteration
        color = self.goal_color_centers[self.itr_ptr]
        self.itr_ptr += 1
        return self._sample_specific_params(color), color

        # if self.itr_ptr == self.num_colors: raise StopIteration
        # color = self.goal_color_centers[self.itr_ptr]
        # spec_par = self.specific_params[tuple(color)][self.sub_itr_ptr]
        # self.sub_itr_ptr += 1
        # if self.sub_itr_ptr % self.num_random_samples_per_color == 0:
        #     self.sub_itr_ptr = 0
        #     self.itr_ptr += 1
        # return spec_par, color


class _BaseParamsSampler(MetaTaskParamsSampler):
    def __init__(self, random=None, num_colors=50):
        super().__init__()
        if not isinstance(random, np.random.RandomState):
          random = np.random.RandomState(random)
        self._random = random

        # sample the goal color centers
        self.num_colors = num_colors
        self.goal_color_centers = self._random.uniform(-1.0, 1.0, size=(self.num_colors,3))

    def sample(self):
        idx = self._random.randint(0, self.num_colors)
        color = self.goal_color_centers[idx]
        return {'goal_color_center': color}, color
    
    def sample_unique(self, num):
        idxs = self._random.choice(self.num_colors, size=num, replace=False)
        return list(
            map(lambda color: ({'goal_color_center': color}, color), (self.goal_color_centers[idx] for idx in idxs))
        )

    def __iter__(self):
        self.itr_ptr = 0
        return self
    
    def __next__(self):
        if self.itr_ptr == self.num_colors: raise StopIteration
        color = self.goal_color_centers[self.itr_ptr]
        self.itr_ptr += 1
        return {'goal_color_center': color}, color


class FewShotReachEnv(few_shot_robot_env.FewShotRobotEnv):
    """Superclass for all Fetch environments.
    I think easy fetch env is the one where you only need to lift up
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type, goal_high_prob,
        min_goal_extra_height=0.0, max_goal_extra_height=0.45,
        min_dist_between_objs=0.1, same_color_radius=0.1
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            goal_high_prob ([0,1]): probability that the goal should be higher than the table
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.goal_high_prob = goal_high_prob
        self.min_goal_extra_height = min_goal_extra_height
        self.max_goal_extra_height = max_goal_extra_height
        self.min_dist_between_objs = min_dist_between_objs
        self.same_color_radius = same_color_radius

        few_shot_robot_env.FewShotRobotEnv.__init__(
            self, model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos
        )


    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, obs, goal, info):
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        d = np.linalg.norm(grip_pos - goal, axis=-1)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        object0_pos = self.sim.data.get_site_xpos('object0')
        object1_pos = self.sim.data.get_site_xpos('object1')
        # gripper state
        object0_rel_pos = object0_pos - grip_pos
        object1_rel_pos = object1_pos - grip_pos
        
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric
        
        obs = np.concatenate([
            self.goal - object0_pos, self.goal - object1_pos,
            object0_rel_pos, object1_rel_pos,
            self.object0_color, self.object1_color,
            gripper_state, gripper_vel
        ])

        return {
            'obs': obs.copy(),
            'obs_task_params': self.goal_color_center.copy()
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object0_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object1_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            while np.linalg.norm(object0_xpos - object1_xpos) < self.min_dist_between_objs:
                object0_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                object1_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)

            object0_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object0_qpos.shape == (7,)
            object0_qpos[:2] = object0_xpos

            object1_qpos = self.sim.data.get_joint_qpos('object1:joint')
            assert object1_qpos.shape == (7,)
            object1_qpos[:2] = object1_xpos

            self.sim.data.set_joint_qpos('object0:joint', object0_qpos)
            self.sim.data.set_joint_qpos('object1:joint', object1_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.correct_obj_idx == 0:
            target_pos = self.sim.data.get_site_xpos('object0')
        else:
            target_pos = self.sim.data.get_site_xpos('object1')
        goal = target_pos + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        if self.target_in_the_air and self.np_random.uniform() < self.goal_high_prob:
            goal[2] += self.np_random.uniform(self.min_goal_extra_height, self.max_goal_extra_height)
        return goal.copy()

    def _sample_color_within_radius(self, center, radius):
        x = np.random.normal(size=3)
        x /= np.linalg.norm(x, axis=-1)
        r = radius
        u = np.random.uniform()
        return r * (u**(1.0/3.0)) * x + center
    
    def _sample_color_with_min_dist(self, color, min_dist):
        new_color = np.random.uniform(-1.0, 1.0, size=3)
        while np.linalg.norm(new_color - color, axis=-1) < min_dist:
            new_color = np.random.uniform(-1.0, 1.0, size=3)
        return new_color

    def reset(self, task_params=None, obs_task_params=None):
        '''
        task_params = {
            'goal_color_center': np.array...
            'other_color_center': np.array... (optional)
            'specific_color_from_goal_radius': np.array (optional)
            'specific_color_from_other_radius': np.array (optional)
        }

        obs_task_params = np.array([r,g,b]) describing the goal_color_center
        '''
        if task_params is None:
            self.goal_color_center = np.random.uniform(-1.0, 1.0, size=3)
            self.goal_specific_color = self._sample_color_within_radius(self.goal_color_center, self.same_color_radius)
            other_center = self._sample_color_with_min_dist(self.goal_color_center, 2 * self.same_color_radius)
            self.other_specific_color = self._sample_color_within_radius(other_center, self.same_color_radius)
        else:
            # handle the goal color
            self.goal_color_center = task_params['goal_color_center']
            if 'specific_color_from_goal_radius' in task_params:
                self.goal_specific_color = task_params['specific_color_from_goal_radius']
                assert np.linalg.norm(self.goal_color_center - self.goal_specific_color, axis=-1) < self.same_color_radius
            else:
                self.goal_specific_color = self._sample_color_within_radius(self.goal_color_center, self.same_color_radius)
            
            # handle the other color
            if 'specific_color_from_other_radius' in task_params:
                self.other_specific_color = task_params['specific_color_from_other_radius']
            else:
                if 'other_color_center' in task_params:
                    self.other_specific_color = self._sample_color_within_radius(task_params['other_color_center'], self.same_color_radius)
                else:
                    other_center = self._sample_color_with_min_dist(self.goal_color_center, 2*self.same_color_radius)
                    self.other_specific_color = self._sample_color_within_radius(other_center, self.same_color_radius)
        
        self.correct_obj_idx = np.random.randint(0, 2)
        if self.correct_obj_idx == 0:
            self.object0_color = self.goal_specific_color
            self.object1_color = self.other_specific_color
        else:
            self.object0_color = self.other_specific_color
            self.object1_color = self.goal_specific_color

        obs = super().reset()       
        return obs
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        info['correct_obj_idx'] = self.correct_obj_idx
        return obs, reward, done, info
    
    @property
    def task_identifier(self):
        return tuple(self.goal_color_center)

    def _is_success(self, obs):
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        d = np.linalg.norm(grip_pos - self.goal, axis=-1)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]
    
    def log_statistics(self, test_paths):
        # compute proportion of episodes that were fully solved
        successes = []
        for path in test_paths:
            successes.append(np.sum([e_info['is_success'] for e_info in path['env_infos']]) > 0)
        percent_solved = np.sum(successes) / float(len(successes))

        # compute proportion of episodes that the arm reached for the right
        # object but was not necessarily able to pick it up
        # the way I am computing this is a proxy for this, but it will probably
        # be good enough
        all_reached_for_correct = []
        min_dist_to_cor = []
        min_cor_z = []
        for path in test_paths:
            cor_idx = path['env_infos'][0]['correct_obj_idx']
            incor_idx = 1-cor_idx
            
            cor_rel_pos = np.array([obs_dict[6+3*cor_idx:9+3*cor_idx] for obs_dict in path['observations']])
            incor_rel_pos = np.array([obs_dict[6+3*incor_idx:9+3*incor_idx] for obs_dict in path['observations']])
            cor_z = np.array([obs_dict[3*cor_idx+2] for obs_dict in path['observations']])

            # cor_rel_pos = np.array([obs_dict['obs'][6+3*cor_idx:9+3*cor_idx] for obs_dict in path['observations']])
            # incor_rel_pos = np.array([obs_dict['obs'][6+3*incor_idx:9+3*incor_idx] for obs_dict in path['observations']])            
            # cor_z = np.array([obs_dict['obs'][3*cor_idx+2] for obs_dict in path['observations']])

            # cor_min_norm = np.min(np.linalg.norm(cor_rel_pos, axis=-1))
            # incor_min_norm = np.min(np.linalg.norm(incor_rel_pos, axis=-1))
            # all_reached_for_correct.append(cor_min_norm < incor_min_norm)
            cor_sum_dist = np.sum(np.linalg.norm(cor_rel_pos[:,:2], axis=-1)[:-30])
            incor_sum_dist = np.sum(np.linalg.norm(incor_rel_pos[:,:2], axis=-1)[:-30])
            all_reached_for_correct.append(cor_sum_dist < incor_sum_dist)
            min_dist_to_cor.append(np.min(np.linalg.norm(cor_rel_pos, axis=-1)))
            min_cor_z.append(np.min(cor_z))
        percent_good_reach = np.sum(all_reached_for_correct) / float(len(all_reached_for_correct))

        return_dict = OrderedDict()
        return_dict['Percent_Good_Reach'] = percent_good_reach
        return_dict['Percent_Solved'] = percent_solved
        return_dict['Avg Min Dist to Cor'] = np.mean(min_dist_to_cor)
        return_dict['Std Min Dist to Cor'] = np.std(min_dist_to_cor)
        return_dict['Avg Min Cor Z'] = np.mean(min_cor_z)
        return_dict['Std Min Cor Z'] = np.std(min_cor_z)
        return return_dict


FEW_SHOT_ENV_XML_PATH = os.path.join(os.path.split(few_shot_robot_env.__file__)[0], 'assets', 'fetch', 'few_shot_pick_and_place.xml')
class BasicFewShotReachEnv(FewShotReachEnv, gym_utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }

        FewShotReachEnv.__init__(
            self, FEW_SHOT_ENV_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.0, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, goal_high_prob=1.0,
            min_goal_extra_height=0.05, max_goal_extra_height=0.05,
            min_dist_between_objs=0.1, same_color_radius=0.3
        )
        gym_utils.EzPickle.__init__(self)
        self._max_episode_steps = 30


class ScaledBasicFewShotReachEnv(BasicFewShotReachEnv):
    def __init__(self, reward_type='sparse'):
        assert False, 'Invalid Statistics'
        self.obs_max = np.array([0.19673975, 0.19944288, 0.20234512, 0.19673975, 0.19944288,
            0.20234512, 0.28635685, 0.29541265, 0.00469703, 0.28635685,
            0.29541265, 0.00469703, 1.3, 1.3, 1.3,
            1.3, 1.3, 1.3, 0.05095022, 0.05092848,
            0.01019219, 0.01034121])
        self.obs_min = np.array([-1.94986926e-01, -1.97374503e-01, -3.04622497e-03, -1.94986926e-01,
            -1.97374503e-01, -3.04622497e-03, -3.00136632e-01, -2.82639213e-01,
            -2.17494754e-01, -3.00136632e-01, -2.82639213e-01, -2.17494754e-01,
            -1.3, -1.3, -1.3, -1.3,
            -1.3, -1.3, 2.55108763e-06, -8.67902630e-08,
            -9.42624227e-03, -9.39642018e-03])
        self.acts_max = np.array([0.24999889, 0.2499995 , 0.2499997 , 0.01499927])
        self.acts_min = np.array([-0.24999355, -0.24999517, -0.24999965, -0.01499985])
        self.SCALE = 0.99    
        super(ScaledBasicFewShotReachEnv, self).__init__()


    def _normalize_obs(self, observation):
        observation = (observation - self.obs_min) / (self.obs_max - self.obs_min)
        observation *= 2 * self.SCALE
        observation -= self.SCALE
        return observation
    

    def _unnormalize_act(self, act):
        return self.acts_min + (act + self.SCALE)*(self.acts_max - self.acts_min) / (2 * self.SCALE)
    

    def reset(self, task_params=None, obs_task_params=None):
        obs = super().reset(task_params=task_params, obs_task_params=obs_task_params)
        obs['obs'] = self._normalize_obs(obs['obs'].copy())
        return obs


    def step(self, action):
        action = self._unnormalize_act(action.copy())
        obs, reward, done, info = super().step(action)
        obs['obs'] = self._normalize_obs(obs['obs'].copy())
        return obs, reward, done, info


class ZeroScaledFewShotReachEnv(ScaledBasicFewShotReachEnv):
    '''
    This is a debug env, do not use!
    '''
    def __init__(self):
        super().__init__()
    

    def reset(self):
        return super().reset(task_params={'goal_color_center': np.zeros(3)}, obs_task_params=np.zeros(3))


class ZeroUnscaledFewShotReachEnv(BasicFewShotReachEnv):
    '''
    This is a debug env, do not use!
    '''
    def __init__(self):
        super().__init__()
    

    def reset(self):
        return super().reset(task_params={'goal_color_center': np.zeros(3)}, obs_task_params=np.zeros(3))


class Scaled0p9BasicFewShotReachEnv(BasicFewShotReachEnv):
    def __init__(self, reward_type='sparse'):
        assert False, 'Invalid Statistics'
        self.SCALE = 0.90
        self.obs_max = np.array([0.19673975, 0.19944288, 0.20234512, 0.19673975, 0.19944288,
            0.20234512, 0.28635685, 0.29541265, 0.00469703, 0.28635685,
            0.29541265, 0.00469703, 1.3, 1.3, 1.3,
            1.3, 1.3, 1.3, 0.05095022, 0.05092848,
            0.01019219, 0.01034121])
        self.obs_min = np.array([-1.94986926e-01, -1.97374503e-01, -3.04622497e-03, -1.94986926e-01,
            -1.97374503e-01, -3.04622497e-03, -3.00136632e-01, -2.82639213e-01,
            -2.17494754e-01, -3.00136632e-01, -2.82639213e-01, -2.17494754e-01,
            -1.3, -1.3, -1.3, -1.3,
            -1.3, -1.3, 2.55108763e-06, -8.67902630e-08,
            -9.42624227e-03, -9.39642018e-03])
        self.acts_max = np.array([0.24999889, 0.2499995 , 0.2499997 , 0.01499927])
        self.acts_min = np.array([-0.24999355, -0.24999517, -0.24999965, -0.01499985])
        super(Scaled0p9BasicFewShotReachEnv, self).__init__()


    def _normalize_obs(self, observation):
        observation = (observation - self.obs_min) / (self.obs_max - self.obs_min)
        observation *= 2 * self.SCALE
        observation -= self.SCALE
        return observation
    

    def _unnormalize_act(self, act):
        return self.acts_min + (act + self.SCALE)*(self.acts_max - self.acts_min) / (2 * self.SCALE)
    

    def reset(self, task_params=None, obs_task_params=None):
        obs = super().reset(task_params=task_params, obs_task_params=obs_task_params)
        obs['obs'] = self._normalize_obs(obs['obs'].copy())
        return obs


    def step(self, action):
        action = self._unnormalize_act(action.copy())
        obs, reward, done, info = super().step(action)
        obs['obs'] = self._normalize_obs(obs['obs'].copy())
        return obs, reward, done, info


class ZeroScaled0p9FewShotReachEnv(Scaled0p9BasicFewShotReachEnv):
    '''
    This is a debug env, do not use!
    '''
    def __init__(self):
        super().__init__()
    

    def reset(self):
        return super().reset(task_params={'goal_color_center': np.zeros(3)}, obs_task_params=np.zeros(3))


class Scaled0p9LinearBasicFewShotReachEnv(BasicFewShotReachEnv):
    def __init__(self, reward_type='sparse'):
        self.SCALE = 0.90
        self.obs_max = np.array([ 2.96110769e-01,  2.89824382e-01,  5.00027539e-02,  2.96110769e-01,
            2.89824382e-01,  5.00027539e-02,  2.91497603e-01,  2.98248790e-01,
            -4.82176644e-02,  2.91497603e-01,  2.98248790e-01, -4.82176644e-02,
            1.3, 1.3, 1.3, 1.3,
            1.3, 1.3,  3.49485219e-04,  4.44059408e-04,
            2.06717618e-04,  2.27631923e-04])
        self.obs_min = np.array([-2.93748042e-01, -2.98049287e-01,  4.99170965e-02, -2.93748042e-01,
            -2.98049287e-01,  4.99170965e-02, -2.97426647e-01, -2.92919821e-01,
            -1.15507540e-01, -2.97426647e-01, -2.92919821e-01, -1.15507540e-01,
            -1.3, -1.3, -1.3, -1.3,
            -1.3, -1.3, -2.58833534e-05, -2.61153351e-05,
            -2.76764282e-04, -3.55273460e-04])
        self.acts_max = np.array([ 0.36886871,  0.36598542,  0.11753301, -0.005])
        self.acts_min = np.array([-0.26792828, -0.26727201, -0.27240187, -0.015])
        super(Scaled0p9LinearBasicFewShotReachEnv, self).__init__()


    def _normalize_obs(self, observation):
        observation = (observation - self.obs_min) / (self.obs_max - self.obs_min)
        observation *= 2 * self.SCALE
        observation -= self.SCALE
        return observation
    

    def _unnormalize_act(self, act):
        return self.acts_min + (act + self.SCALE)*(self.acts_max - self.acts_min) / (2 * self.SCALE)
    

    def reset(self, task_params=None, obs_task_params=None):
        obs = super().reset(task_params=task_params, obs_task_params=obs_task_params)
        obs['obs'] = self._normalize_obs(obs['obs'].copy())
        return obs


    def step(self, action):
        action = self._unnormalize_act(action.copy())
        obs, reward, done, info = super().step(action)
        obs['obs'] = self._normalize_obs(obs['obs'].copy())
        return obs, reward, done, info


class ZeroScaled0p9LinearFewShotReachEnv(Scaled0p9LinearBasicFewShotReachEnv):
    '''
    This is a debug env, do not use!
    '''
    def __init__(self):
        super().__init__()
    

    def reset(self):
        return super().reset(task_params={'goal_color_center': np.zeros(3)}, obs_task_params=np.zeros(3))



if __name__ == '__main__':
    # TESTING SCRIPT
    FEW_SHOT_ENV_XML_PATH = os.path.join(os.path.split(few_shot_robot_env.__file__)[0], 'assets', 'fetch', 'few_shot_pick_and_place.xml')
    initial_qpos = {
        'robot0:slide0': 0.405,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
        'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        'object1:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
    }
    env = FewShotReachEnv(
        FEW_SHOT_ENV_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
        gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
        obj_range=0.15, target_range=0.01, distance_threshold=0.05,
        initial_qpos=initial_qpos, reward_type='sparse', goal_high_prob=1.0,
        min_goal_extra_height=0.15, max_goal_extra_height=0.2,
        min_dist_between_objs=0.1, same_color_radius=0.3
    )

    # while True:
    #     print(env.reset())
    #     for i in range(100): env.render()
    #     # sleep(2)

    # test setting the colors
    for i in range(100):
        goal_color_center = np.random.uniform(-1.0, 1.0, size=3)
        obs = env.reset(task_params={'goal_color_center': goal_color_center})

        assert np.array_equal(goal_color_center, obs['obs_task_params'])
        correct = env.correct_obj_idx
        color_of_correct_obj = obs['obs'][12+3*correct:12+3*correct+3]
        assert np.linalg.norm(goal_color_center - color_of_correct_obj) < env.same_color_radius
        not_correct = 1-correct
        color_of_not_correct_obj = obs['obs'][12+3*not_correct:12+3*not_correct+3]
        assert np.linalg.norm(goal_color_center - color_of_not_correct_obj) > env.same_color_radius
        assert np.array_equal(env.task_identifier, goal_color_center)
        
        for j in range(5):
            obs = env.step(np.random.uniform(size=4))[0]
            assert np.array_equal(color_of_correct_obj, obs['obs'][12+3*correct:12+3*correct+3])
            assert np.array_equal(color_of_not_correct_obj, obs['obs'][12+3*not_correct:12+3*not_correct+3])
            assert np.array_equal(goal_color_center, obs['obs_task_params'])
            assert np.array_equal(env.task_identifier, goal_color_center)

    # compute average distance of goal center to distractor color
    d = []
    d_correct = []
    for i in range(1000):
        obs = env.reset()
        goal_color_center = env.task_identifier
        correct = env.correct_obj_idx
        color_of_correct_obj = obs['obs'][12+3*correct:12+3*correct+3]
        not_correct = 1-env.correct_obj_idx
        color_of_not_correct_obj = obs['obs'][12+3*not_correct:12+3*not_correct+3]
        d.append(np.linalg.norm(goal_color_center - color_of_not_correct_obj))
        d_correct.append(np.linalg.norm(color_of_correct_obj - color_of_not_correct_obj))
    # as a sanity check these two prints should be almost equal
    print('%.4f +/- %.4f' % (np.mean(d), np.std(d)))
    print('%.4f +/- %.4f' % (np.mean(d_correct), np.std(d_correct)))

    print(np.min(d_correct))
    print(np.max(d_correct))

    # ugly, yes
    import matplotlib.pyplot as plt
    def plot_histogram(flat_array, num_bins, title, save_path):
        fig, ax = plt.subplots(1)
        ax.set_title(title)
        plt.hist(flat_array, bins=num_bins)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    plot_histogram(d_correct, 100, 'distance of goal specific color from distractor', 'd_correct.png')


    # print some sample colors
    goal_color_center = np.random.uniform(-1.0, 1.0, size=3)
    print(goal_color_center)
    print('\n')
    for i in range(10):
        c = env._sample_color_within_radius(goal_color_center, env.same_color_radius)
        print(c)
    

    # visualize how far things are from each other
    from mpl_toolkits.mplot3d import Axes3D
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = 1.0 * np.outer(np.cos(u), np.sin(v))
    y = 1.0 * np.outer(np.sin(u), np.sin(v))
    z = 1.0 * np.outer(np.ones(np.size(u)), np.cos(v))

    def _sample_color_within_radius(center, radius):
        x = np.random.normal(size=3)
        x /= np.linalg.norm(x, axis=-1)
        r = radius
        u = np.random.uniform()
        return r * (u**(1.0/3.0)) * x + center
    
    def _sample_color_with_min_dist(color, min_dist):
        new_color = np.random.uniform(-1.0, 1.0, size=3)
        while np.linalg.norm(new_color - color, axis=-1) < min_dist:
            new_color = np.random.uniform(-1.0, 1.0, size=3)
        return new_color

    def plot_sphere(ax, center, radius, color, alpha):
        s_x = radius * x + center[0]
        s_y = radius * y + center[1]
        s_z = radius * z + center[2]
        # ax[0].plot(s_x, s_y, color=color)
        # ax[1].plot(s_x, s_z, color=color)
        # ax[2].plot(s_y, s_z, color=color)
        ax.plot_surface(s_x, s_y, s_z,  rstride=4, cstride=4, color=color, linewidth=0, alpha=alpha)


    print('\n')
    N = 3
    for i in range(20):
        fig = plt.figure(figsize=(20,20), dpi=60)
        ax = fig.add_subplot(111, projection='3d')
        # ax = []
        # for i in range(3):
        #     cur_ax = fig.add_subplot(1, 3, i+1)
        #     cur_ax.set_aspect('equal')
        #     cur_ax.set_xlim(-1.0, 1.0)
        #     cur_ax.set_ylim(-1.0, 1.0)
        #     ax.append(cur_ax)

        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-1.0, 1.0)

        goal_color_center = np.random.uniform(-1.0, 1.0, size=3)
        plot_sphere(ax, goal_color_center, env.same_color_radius, 'b', 0.5)

        goal_colors = [_sample_color_within_radius(goal_color_center, env.same_color_radius) for _ in range(N)]
        for specific_color in goal_colors:
            print(np.linalg.norm(specific_color - goal_color_center, axis=-1))
            plot_sphere(ax, specific_color, env.same_color_radius, 'yellow', 0.5)
        
        for i in range(20):
            other_center = _sample_color_with_min_dist(goal_color_center, 2 * env.same_color_radius)
            # print(goal_color_center)
            # print(other_center)
            # print(np.linalg.norm(other_center - goal_color_center))
            print(np.linalg.norm(other_center - goal_color_center, axis=-1))
            plot_sphere(ax, other_center, env.same_color_radius, 'green', 0.5)
        # other_colors = [_sample_color_within_radius(other_center, env.same_color_radius) for _ in range(N)]
        # for other_specific_color in other_colors:
        #     plot_sphere(ax, other_specific_color, env.same_color_radius, 'red', 0.5)

        plt.show()
