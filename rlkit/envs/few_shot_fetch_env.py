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
# (I didn't tune my random seed for demo generation LOL)
def get_some_task_params_iterator(train_env=True, num=50):
    # when you fix this make sure really all the colors are correct
    if train_env:
        return _BaseParamsSampler(random=2497, num_colors=num)
    else:
        return _BaseParamsSampler(random=8384, num_colors=num)


# These random seeds are literally me typing random numbers
# (I didn't tune my random seed for demo generation LOL)
def get_task_params_iterator(train_env=True):
    # when you fix this make sure really all the colors are correct
    if train_env:
        return _BaseParamsSampler(random=2497)
    else:
        return _BaseParamsSampler(random=8384)

# this debug one uses only a few tasks so we can make sure things are actually working first
def get_debug_task_params_iterator(train_env=True):
    if train_env:
        return _FullySpecifiedParamsSampler(random=7342, num_colors=1, num_random_samples_per_color=1000, same_color_radius=0.5)
    else:
        return _FullySpecifiedParamsSampler(random=7342, num_colors=1, num_random_samples_per_color=1000, same_color_radius=0.5)
        # return _BaseParamsSampler(random=7342, num_colors=1)
        # return _FullySpecifiedParamsSampler(random=7342, num_colors=1, num_random_samples_per_color=500, same_color_radius=0.5)

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
    def __init__(self, random=None, num_colors=50, num_random_samples_per_color=10, same_color_radius=0.5):
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
                other_center = self._sample_color_with_min_dist(goal_color, self.same_color_radius)
                # other_specific_color = self._sample_color_within_radius(other_center, self.same_color_radius)
                other_specific_color = other_center
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
        sampled_color = r * (u**(1.0/3.0)) * x + center
        return np.clip(sampled_color, -1.0, 1.0)
    
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
        # if self.itr_ptr == self.num_colors: raise StopIteration
        # color = self.goal_color_centers[self.itr_ptr]
        # self.itr_ptr += 1
        # return self._sample_specific_params(color), color

        if self.itr_ptr == self.num_colors: raise StopIteration
        color = self.goal_color_centers[self.itr_ptr]
        spec_par = self.specific_params[tuple(color)][self.sub_itr_ptr]
        self.sub_itr_ptr += 1
        if self.sub_itr_ptr % self.num_random_samples_per_color == 0:
            self.sub_itr_ptr = 0
            self.itr_ptr += 1
        return spec_par, color


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
        # dangerous
        self.itr_ptr = 0
        return self
    
    def __next__(self):
        if self.itr_ptr == self.num_colors: raise StopIteration
        color = self.goal_color_centers[self.itr_ptr]
        self.itr_ptr += 1
        return {'goal_color_center': color}, color


class FewShotFetchEnv(few_shot_robot_env.FewShotRobotEnv):
    """Superclass for all Fetch environments.
    I think easy fetch env is the one where you only need to lift up
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type, goal_high_prob,
        min_goal_extra_height=0.0, max_goal_extra_height=0.45,
        min_dist_between_objs=0.1, same_color_radius=0.5,
        terminate_on_success=False
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
            initial_qpos=initial_qpos, terminate_on_success=terminate_on_success
        )


    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, obs, goal, info):
        correct_obj_rel_to_goal = obs['obs'][3*self.correct_obj_idx:3*self.correct_obj_idx+3].copy()
        d = np.linalg.norm(correct_obj_rel_to_goal, axis=-1)
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
        goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        goal += self.target_offset
        goal[2] = self.height_offset
        if self.target_in_the_air and self.np_random.uniform() < self.goal_high_prob:
            goal[2] += self.np_random.uniform(self.min_goal_extra_height, self.max_goal_extra_height)
        return goal.copy()

    def _sample_color_within_radius(self, center, radius):
        x = self.np_random.normal(size=3)
        x /= np.linalg.norm(x, axis=-1)
        r = radius
        u = self.np_random.uniform()
        sampled_color = r * (u**(1.0/3.0)) * x + center
        return np.clip(sampled_color, -1.0, 1.0)
    
    def _sample_color_with_min_dist(self, color, min_dist):
        new_color = self.np_random.uniform(-1.0, 1.0, size=3)
        while np.linalg.norm(new_color - color, axis=-1) < min_dist:
            new_color = self.np_random.uniform(-1.0, 1.0, size=3)
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
            self.goal_color_center = self.np_random.uniform(-1.0, 1.0, size=3)
            self.goal_specific_color = self._sample_color_within_radius(self.goal_color_center, self.same_color_radius)
            other_center = self._sample_color_with_min_dist(self.goal_color_center, self.same_color_radius)
            self.other_specific_color = other_center
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
                    other_center = self._sample_color_with_min_dist(self.goal_color_center, self.same_color_radius)
                    self.other_specific_color = other_center
        
        self.correct_obj_idx = self.np_random.randint(0, 2)
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

        # log some info so we can track whether failure was due
        # to no-op or due to incorrect choice
        yes_op = False
        for idx in [0,1]:
            obj_rel_to_goal = obs['obs'][3*idx:3*idx+3].copy()
            d = np.linalg.norm(obj_rel_to_goal, axis=-1)
            yes_op |= d < self.distance_threshold
        info['yes_op'] = yes_op

        return obs, reward, done, info
    
    @property
    def task_identifier(self):
        return tuple(self.goal_color_center)
    
    def task_id_to_obs_task_params(self, task_id):
        return np.array(task_id)

    def _is_success(self, obs):
        correct_obj_rel_to_goal = obs['obs'][3*self.correct_obj_idx:3*self.correct_obj_idx+3].copy()
        d = np.linalg.norm(correct_obj_rel_to_goal, axis=-1)
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
        num_total_failures = 0
        num_failures_due_to_no_op = 0
        for path in test_paths:
            successes.append(np.sum([e_info['is_success'] for e_info in path['env_infos']]) > 0)
            if not successes[-1]:
                num_total_failures += 1
                if np.sum([e_info['yes_op'] for e_info in path['env_infos']]) == 0:
                    num_failures_due_to_no_op += 1
        percent_solved = np.sum(successes) / float(len(successes))
        if num_total_failures == 0:
            percent_no_op_fail = 0
        else:
            percent_no_op_fail = num_failures_due_to_no_op / float(num_total_failures)

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
            
            # cor_rel_pos = np.array([obs_dict[6+3*cor_idx:9+3*cor_idx] for obs_dict in path['observations']])
            # incor_rel_pos = np.array([obs_dict[6+3*incor_idx:9+3*incor_idx] for obs_dict in path['observations']])
            # cor_z = np.array([obs_dict[3*cor_idx+2] for obs_dict in path['observations']])

            cor_rel_pos = np.array([obs_dict['obs'][6+3*cor_idx:9+3*cor_idx] for obs_dict in path['observations']])
            incor_rel_pos = np.array([obs_dict['obs'][6+3*incor_idx:9+3*incor_idx] for obs_dict in path['observations']])            
            cor_z = np.array([obs_dict['obs'][3*cor_idx+2] for obs_dict in path['observations']])

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
        return_dict['Percent_NoOp_Fail'] = percent_no_op_fail
        return_dict['Avg Min Dist to Cor'] = np.mean(min_dist_to_cor)
        return_dict['Std Min Dist to Cor'] = np.std(min_dist_to_cor)
        return_dict['Avg Min Cor Z'] = np.mean(min_cor_z)
        return_dict['Std Min Cor Z'] = np.std(min_cor_z)
        return return_dict


FEW_SHOT_ENV_XML_PATH = os.path.join(os.path.split(few_shot_robot_env.__file__)[0], 'assets', 'fetch', 'few_shot_pick_and_place.xml')
class BasicFewShotFetchEnv(FewShotFetchEnv, gym_utils.EzPickle):
    def __init__(self, reward_type='sparse', terminate_on_success=False):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }

        FewShotFetchEnv.__init__(
            self, FEW_SHOT_ENV_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.05, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, goal_high_prob=1.0,
            min_goal_extra_height=0.15, max_goal_extra_height=0.2,
            min_dist_between_objs=0.1, same_color_radius=0.5,
            terminate_on_success=terminate_on_success
        )
        gym_utils.EzPickle.__init__(self)
        self._max_episode_steps = 65


class ScaledBasicFewShotFetchEnv(BasicFewShotFetchEnv):
    def __init__(self, reward_type='sparse'):
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
        super(ScaledBasicFewShotFetchEnv, self).__init__()


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


class ZeroScaledFewShotFetchEnv(ScaledBasicFewShotFetchEnv):
    '''
    This is a debug env, do not use!
    '''
    def __init__(self):
        super().__init__()
    

    def reset(self):
        return super().reset(task_params={'goal_color_center': np.zeros(3)}, obs_task_params=np.zeros(3))


class ZeroUnscaledFewShotFetchEnv(BasicFewShotFetchEnv):
    '''
    This is a debug env, do not use!
    '''
    def __init__(self):
        super().__init__()
    

    def reset(self):
        return super().reset(task_params={'goal_color_center': np.zeros(3)}, obs_task_params=np.zeros(3))


class Scaled0p9BasicFewShotFetchEnv(BasicFewShotFetchEnv):
    def __init__(self, reward_type='sparse'):
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
        super(Scaled0p9BasicFewShotFetchEnv, self).__init__()


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


class ZeroScaled0p9FewShotFetchEnv(Scaled0p9BasicFewShotFetchEnv):
    '''
    This is a debug env, do not use!
    '''
    def __init__(self):
        super().__init__()
    

    def reset(self):
        return super().reset(task_params={'goal_color_center': np.zeros(3)}, obs_task_params=np.zeros(3))


class Scaled0p9LinearBasicFewShotFetchEnv(BasicFewShotFetchEnv):
    def __init__(self, reward_type='sparse', obs_max=None, obs_min=None, acts_max=None, acts_min=None, terminate_on_success=False):
        self.SCALE = 0.90
        self.obs_max = obs_max
        self.obs_min = obs_min
        self.acts_max = acts_max
        self.acts_min = acts_min
        # self.obs_max = np.array([0.22051651, 0.22935722, 0.20480309, 0.22051651, 0.22935722,
        #     0.20480309, 0.30151219, 0.29303502, 0.00444365, 0.30151219,
        #     0.29303502, 0.00444365, 1.3, 1.3, 1.3,
        #     1.3, 1.3, 1.3, 0.05099135, 0.05091496,
        #     0.01034575, 0.0103919 ])
        # self.obs_min = np.array([-1.98124936e-01, -2.04234846e-01, -8.51241789e-03, -1.98124936e-01,
        #     -2.04234846e-01, -8.51241789e-03, -3.03874692e-01, -3.00712133e-01,
        #     -2.30561716e-01, -3.03874692e-01, -3.00712133e-01, -2.30561716e-01,
        #     -1.3, -1.3, -1.3, -1.3,
        #     -1.3, -1.3,  2.55108763e-06, -8.67902630e-08,
        #     -1.20198677e-02, -9.60486720e-03])
        # self.acts_max = np.array([0.3667496 , 0.3676551 , 0.37420813, 0.015])
        # self.acts_min = np.array([-0.27095875, -0.26862562, -0.27479879, -0.015])
        super(Scaled0p9LinearBasicFewShotFetchEnv, self).__init__(terminate_on_success=terminate_on_success)


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


class StatsFor50Tasks25EachScaled0p9LinearBasicFewShotFetchEnv(Scaled0p9LinearBasicFewShotFetchEnv):
    def __init__(self, terminate_on_success=False):
        # obs_max = np.array([0.20873973, 0.21238721, 0.20497428, 0.20873973, 0.21238721,
        #     0.20497428, 0.29729787, 0.29597882, 0.00660929, 0.29729787,
        #     0.29597882, 0.00660929, 1.0, 1.0, 1.0,
        #     1.0, 1.0, 1.0, 0.05099425, 0.05097209,
        #     0.01045247, 0.01020353])
        # obs_min = np.array([-2.07733303e-01, -2.22872196e-01, -6.20862381e-03, -2.07733303e-01,
        #     -2.22872196e-01, -6.20862381e-03, -3.02834854e-01, -3.18478521e-01,
        #     -2.35453885e-01, -3.02834854e-01, -3.18478521e-01, -2.35453885e-01,
        #     -1.0, -1.0, -1.0, -1.0,
        #     -1.0, -1.0,  2.55108763e-06, -8.67902630e-08,
        #     -1.12767104e-02, -1.15187468e-02])
        # acts_max = np.array([0.36385158, 0.36506858, 0.37287046, 0.015])
        # acts_min = np.array([-0.27378214, -0.27318582, -0.27457426, -0.015])
        
        # obs_max = np.array([0.19732151, 0.19501755, 0.2032467 , 0.19732151, 0.19501755,
        #     0.2032467 , 0.28952909, 0.27034638, 0.00461512, 0.28952909,
        #     0.27034638, 0.00461512, 1.        , 1.        , 1.        ,
        #     1.        , 1.        , 1.        , 0.05084346, 0.05089836,
        #     0.01020451, 0.01024073])
        # obs_min = np.array([-1.94163008e-01, -2.06672946e-01, -4.34817497e-03, -1.94163008e-01,
        #     -2.06672946e-01, -4.34817497e-03, -2.57836261e-01, -3.02357607e-01,
        #     -2.26000082e-01, -2.57836261e-01, -3.02357607e-01, -2.26000082e-01,
        #     -1., -1., -1., -1.,
        #     -1., -1.,  2.55108763e-06, -8.67902630e-08,
        #     -9.79891841e-03, -9.23147216e-03])
        # acts_max = np.array([0.36071754, 0.35800805, 0.37175567, 0.015])
        # acts_min = np.array([-0.26463221, -0.26663373, -0.27413371, -0.015])

        obs_max = np.array([0.20061923, 0.19781174, 0.20549539, 0.20061923, 0.19781174,
            0.20549539, 0.29141252, 0.28891717, 0.00129714, 0.29141252,
            0.28891717, 0.00129714, 1.0        , 1.0        , 1.0        ,
            1.0        , 1.0        , 1.0        , 0.05096386, 0.05090749,
            0.01046458, 0.01028522])
        obs_min = np.array([-1.83014661e-01, -2.07445100e-01, -4.79934195e-03, -1.83014661e-01,
            -2.07445100e-01, -4.79934195e-03, -2.89125464e-01, -2.96987424e-01,
            -2.30655094e-01, -2.89125464e-01, -2.96987424e-01, -2.30655094e-01,
            -1.0, -1.0, -1.0, -1.0,
            -1.0, -1.0,  2.55108763e-06, -8.67902630e-08,
            -1.11994283e-02, -9.10341004e-03])
        acts_max = np.array([0.36051396, 0.36032055, 0.37415428, 0.015])
        acts_min = np.array([-0.2696256 , -0.27399028, -0.27453274, -0.015])

        super().__init__(
            obs_max=obs_max,
            obs_min=obs_min,
            acts_max=acts_max,
            acts_min=acts_min,
            terminate_on_success=terminate_on_success
        )


class ZeroScaled0p9LinearFewShotFetchEnv(Scaled0p9LinearBasicFewShotFetchEnv):
    '''
    This is a debug env, do not use!
    '''
    def __init__(self):
        self.obs_max = np.array([0.22392513, 0.23576041, 0.2074778 , 0.22392513, 0.23576041,
            0.2074778 , 0.32363979, 0.32648092, 0.0049561 , 0.32363979,
            0.32648092, 0.0049561 , 1.3, 1.3, 1.3,
            1.3, 1.3, 1.3, 0.05101674, 0.05100188,
            0.01055062, 0.01049931])
        self.obs_min = np.array([-1.99576796e-01, -2.14964995e-01, -6.89937522e-03, -1.99576796e-01,
            -2.14964995e-01, -6.89937522e-03, -3.09594735e-01, -3.15771113e-01,
            -2.35295369e-01, -3.09594735e-01, -3.15771113e-01, -2.35295369e-01,
            -1.3, -1.3, -1.3, -1.3,
            -1.3, -1.3,  2.55108763e-06, -8.67902630e-08,
            -1.26888212e-02, -1.02645506e-02])
        self.acts_max = np.array([0.36875932, 0.36954177, 0.37465581, 0.015])
        self.acts_min = np.array([-0.27403988, -0.27340838, -0.2749071 , -0.015])
        super().__init__(obs_max=self.obs_max, obs_min=self.obs_min, acts_max=self.acts_max, acts_min=self.acts_min)
    

    def reset(self):
        return super().reset(task_params={'goal_color_center': np.zeros(3)}, obs_task_params=np.zeros(3))



class WrapAbsScaled0p9LinearBasicFewShotFetchEnv(BasicFewShotFetchEnv):
    def __init__(self, reward_type='sparse'):
        self.SCALE = 0.90
        self.obs_max = np.array([0.22691067, 0.24073516, 0.20616085, 0.22691067, 0.24073516,
            0.20616085, 0.30655007, 0.31246556, 0.00573548, 0.30655007,
            0.31246556, 0.00573548, 1.3, 1.3, 1.3,
            1.3, 1.3, 1.3, 0.05101679, 0.05100176,
            0.01049234, 0.01052882])
        self.obs_min = np.array([-2.07510251e-01, -2.21086958e-01, -3.47862349e-03, -2.07510251e-01,
            -2.21086958e-01, -3.47862349e-03, -3.12571681e-01, -3.14835529e-01,
            -2.17068484e-01, -3.12571681e-01, -3.14835529e-01, -2.17068484e-01,
            -1.3, -1.3, -1.3, -1.3,
            -1.3, -1.3,  0.00000000e+00, -8.67902630e-08,
            -1.23168940e-02, -1.09300949e-02])
        self.acts_max = np.array([0.36900074, 0.36956025, 0.37478169, 0.015])
        self.acts_min = np.array([-0.26874253, -0.27001242, -0.27486427, -0.015])

        super(WrapAbsScaled0p9LinearBasicFewShotFetchEnv, self).__init__(terminate_on_success=True)


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


class WrapAbsZeroScaled0p9LinearFewShotFetchEnv(WrapAbsScaled0p9LinearBasicFewShotFetchEnv):
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
    env = FewShotFetchEnv(
        FEW_SHOT_ENV_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
        gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
        obj_range=0.15, target_range=0.01, distance_threshold=0.05,
        initial_qpos=initial_qpos, reward_type='sparse', goal_high_prob=1.0,
        min_goal_extra_height=0.15, max_goal_extra_height=0.2,
        min_dist_between_objs=0.1, same_color_radius=0.5
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
        sampled_color = r * (u**(1.0/3.0)) * x + center
        return np.clip(sampled_color, -1.0, 1.0)
    
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
            other_center = _sample_color_with_min_dist(goal_color_center, env.same_color_radius)
            # print(goal_color_center)
            # print(other_center)
            # print(np.linalg.norm(other_center - goal_color_center))
            print(np.linalg.norm(other_center - goal_color_center, axis=-1))
            plot_sphere(ax, other_center, env.same_color_radius, 'green', 0.5)
        # other_colors = [_sample_color_within_radius(other_center, env.same_color_radius) for _ in range(N)]
        # for other_specific_color in other_colors:
        #     plot_sphere(ax, other_specific_color, env.same_color_radius, 'red', 0.5)

        plt.show()
