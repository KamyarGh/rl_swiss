import numpy as np
import os
from time import sleep
from collections import OrderedDict
from gym.envs.robotics import utils
from gym import utils as gym_utils
from rlkit.envs import few_shot_robot_env

from rlkit.envs.meta_task_params_sampler import MetaTaskParamsSampler


class _BaseParamsSampler(MetaTaskParamsSampler):
    def __init__(self, tasks, random=7823):
        super().__init__()
        if not isinstance(random, np.random.RandomState):
            random = np.random.RandomState(random)
        self._random = random
        self.tasks = tasks
        self._ptr = 0

    def sample(self):
        p = self.tasks[self._random.choice(self.tasks.shape[0])]
        return {'task_hyperplane': p}, p

    def sample_unique(self, num):
        idxs = self._random.choice(self.tasks.shape[0], size=num, replace=False)
        p_samples = self.tasks[idxs]
        return list(
            map(
                lambda p: ({'task_hyperplane': p}, p),
                p_samples
            )
        )

    def __iter__(self):
        # dangerous
        self._ptr = 0
        return self

    def __next__(self):
        if self._ptr == self.tasks.shape[0]:
            self._ptr = 0
            raise StopIteration
        p = self.tasks[self._ptr]
        self._ptr += 1
        return {'task_hyperplane': p}, p


class _ExpertTrainParamsSampler(_BaseParamsSampler):
    def __init__(self, random=77129, num_tasks=64):
        if not isinstance(random, np.random.RandomState):
            random = np.random.RandomState(random)
        tasks = random.normal(size=(num_tasks, 4))
        tasks /= np.linalg.norm(tasks, axis=-1, keepdims=True)
        super().__init__(tasks, random=random)


class _ExpertTestParamsSampler(_ExpertTrainParamsSampler):
    def __init__(self, random=66082, num_tasks=64):
        super().__init__(random=random, num_tasks=num_tasks)


class _ExpertEvalParamsSampler(_ExpertTrainParamsSampler):
    def __init__(self, random=10101, num_tasks=128):
        super().__init__(random=random, num_tasks=num_tasks)


class FetchLinearClassificationEnv(few_shot_robot_env.FewShotRobotEnv):
    def __init__(
        self,
        n_substeps=20,
        gripper_extra_height=0.2,
        block_gripper=False,
        has_object=True,
        target_in_the_air=True,
        target_offset=0.0,
        obj_range=0.15,
        target_range=0.05,
        distance_threshold=0.05,
        reward_type='sparse',
        goal_high_prob=1.0,
        min_goal_extra_height=0.15,
        max_goal_extra_height=0.2,
        min_dist_between_objs=0.1,
        same_color_radius=0.5,
        terminate_on_success=False,
        max_episode_steps=65
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
        self._max_episode_steps = max_episode_steps

        self.ACCEPT_RADIUS = 0.5
        self.true_label = 0
        self.task_hyperplane = np.ones(4) / 2.0
        self.first_sample = np.random.uniform(size=4)
        self.second_sample = np.random.uniform(size=4)
        self.targets = np.array(
            [
                [1.41, 1.41],
                [-1.41, 1.41]
            ]
        )
        
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }

        few_shot_robot_env.FewShotRobotEnv.__init__(
            self,
            model_path=os.path.join(os.path.split(few_shot_robot_env.__file__)[0], 'assets', 'fetch', 'few_shot_pick_and_place.xml'),
            n_substeps=n_substeps,
            n_actions=4,
            initial_qpos=initial_qpos,
            terminate_on_success=terminate_on_success
        )


    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, obs, goal, info):
        correct_obj_rel_to_goal = obs['obs'][6 + 3*self.true_label: 6 + 3*self.true_label + 3].copy()
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
            object0_rel_pos, object1_rel_pos,
            self.goal - object0_pos, self.goal - object1_pos,
            self.first_sample, self.second_sample,
            gripper_state, gripper_vel
        ])

        return {
            'obs': obs.copy(),
            'obs_task_params': self.task_hyperplane.copy()
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

    def sample_points_for_cur_task(self):
        x = 2*np.random.uniform(size=4) - 1
        while np.dot(self.task_hyperplane, x) < 0:
            x = 2*np.random.uniform(size=4) - 1
        # print(np.dot(self.task_hyperplane, x))
        pos_sample = x.copy()
        # print(pos_sample)
        
        x = 2*np.random.uniform(size=4) - 1
        while np.dot(self.task_hyperplane, x) > 0:
            x = 2*np.random.uniform(size=4) - 1
        # print(np.dot(self.task_hyperplane, x))
        neg_sample = x.copy()
        # print(neg_sample)

        if np.random.uniform() > 0.5:
            self.true_label = 1
            self.first_sample = pos_sample
            self.second_sample = neg_sample
        else:
            self.true_label = 0
            self.first_sample = neg_sample
            self.second_sample = pos_sample

    def reset(self, task_params=None, obs_task_params=None):
        if task_params is not None:
            self.task_hyperplane = task_params['task_hyperplane']
        # this should only be None at __init__
        self.sample_points_for_cur_task()
        obs = super().reset()
        return obs
    
    def step(self, action):
        obs_dict, reward, done, info = super().step(action)
        obs = obs_dict['obs']
        correct_to_target = obs[6 + 3*self.true_label: 6 + 3*self.true_label + 3]
        incorrect_to_target = obs[6 + 3*(1-self.true_label): 6 + 3*(1-self.true_label) + 3]

        correct_dist = np.linalg.norm(correct_to_target)
        incorrect_dist = np.linalg.norm(incorrect_to_target)

        correct_is_within_radius = correct_dist < self.ACCEPT_RADIUS
        incorrect_is_within_radius = incorrect_dist < self.ACCEPT_RADIUS

        return obs_dict, reward, done, dict(
            correct_dist=correct_dist,
            incorrect_dist=incorrect_dist,
            true_label=self.true_label,
            correct_is_within_radius=correct_is_within_radius,
            incorrect_is_within_radius=incorrect_is_within_radius,
            is_success=self._is_success(obs_dict)
        )
    
    @property
    def task_identifier(self):
        return tuple(self.task_hyperplane)
    
    def task_id_to_obs_task_params(self, task_id):
        return np.array(task_id)

    def _is_success(self, obs):
        correct_to_target = obs['obs'][6 + 3*self.true_label: 6 + 3*self.true_label + 3]
        d = np.linalg.norm(correct_to_target, axis=-1)
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
    
    def log_statistics(self, paths):
        success = [np.sum([d["correct_is_within_radius"] for d in path["env_infos"]]) > 0 for path in paths]
        went_to_incorrect = [np.sum([d["incorrect_is_within_radius"] for d in path["env_infos"]]) > 0 for path in paths]
        no_op = [np.sum([d["correct_is_within_radius"] or d["incorrect_is_within_radius"] for d in path["env_infos"]]) == 0 for path in paths]
        min_dist_to_correct = [np.min([d["correct_dist"] for d in path["env_infos"]]) for path in paths]
        min_dist_to_incorrect = [np.min([d["incorrect_dist"] for d in path["env_infos"]]) for path in paths]

        num_paths = float(len(paths))
        perc_success = np.sum(success) / num_paths
        perc_incorrect = np.sum(went_to_incorrect) / num_paths
        perc_no_op = np.sum(no_op) / num_paths

        return_dict = OrderedDict()
        return_dict['Perc Success'] = perc_success
        return_dict['Perc Incorrect'] = perc_incorrect
        return_dict['Perc No-Op'] = perc_no_op

        return_dict['AverageClosest Correct'] = np.mean(min_dist_to_correct)
        return_dict['MaxClosest Correct'] = np.max(min_dist_to_correct)
        return_dict['MinClosest Correct'] = np.min(min_dist_to_correct)
        return_dict['StdClosest Correct'] = np.std(min_dist_to_correct)

        return_dict['AverageClosest Incorrect'] = np.mean(min_dist_to_incorrect)
        return_dict['MaxClosest Incorrect'] = np.max(min_dist_to_incorrect)
        return_dict['MinClosest Incorrect'] = np.min(min_dist_to_incorrect)
        return_dict['StdClosest Incorrect'] = np.std(min_dist_to_incorrect)
        return return_dict
