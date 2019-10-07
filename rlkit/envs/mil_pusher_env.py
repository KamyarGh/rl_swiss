import os
import os.path as osp
import glob
import pickle
import random
from collections import OrderedDict

import numpy as np
import PIL
from PIL import Image

from gym import utils
# from gym.envs.mujoco import mujoco_env
from rlkit.envs.pusher_mujoco_env import PusherMujocoEnv
# from rlkit.envs.meta_task_params_sampler import MetaTaskParamsSampler

import mujoco_py

XML_PATH = '/scratch/ssd001/home/kamyar/gym/gym/envs/mujoco/assets/sim_push_xmls/'
SCALE_FILE_PATH = '/h/kamyar/mil/data/scale_and_bias_sim_push.pkl'
TRAIN_VAL_DEMO_DIR = '/scratch/ssd001/home/kamyar/mil/data/sim_push'
TEST_DEMO_DIR = '/scratch/ssd001/home/kamyar/mil/data/sim_push_test'


class ParamsSampler():
    def __init__(self, task_names, random=7823):
        super().__init__()
        if not isinstance(random, np.random.RandomState):
          random = np.random.RandomState(random)
        self._random = random
        self._ptr = 0

        self.task_names = task_names
    
    def sample(self):
        task = self.task_names[self._random.choice(len(self.task_names))]
        return {'task_name': task}, task

    def sample_unique(self, num):
        # print(self.task_names)
        task_name_samples = random.sample(self.task_names, num)
        return list(
            map(
                lambda t: ({'task_name': t}, t),
                task_name_samples
            )
        )
    
    def __iter__(self):
        # dangerous
        self._ptr = 0
        return self
    
    def next(self):
        if self._ptr == len(self.task_names):
            self._ptr = 0
            raise StopIteration
        task = self.task_names[self._ptr]
        self._ptr += 1
        return {'task_name': task}, task


class PusherEnvGetter():
    def __init__(
        self,
        all_xmls_dict,
        distractors=False,
        state_mean=None,
        state_std=None
    ):
        self.all_envs = {}
        print(all_xmls_dict)
        for k in all_xmls_dict:
            # try:
            self.all_envs[k] = PusherEnv(
                all_xmls_dict[k], distractors, task_id=k,
                state_mean=state_mean, state_std=state_std
            )
            # except:
                # print('\n\n{} FAILED {}\n\n'.format(k, all_xmls_dict[k]))
        self.distractors = distractors
        # assert False, 'Need to scale things'
    
    def get_env(self, task_id):
        # print(task_id)
        return self.all_envs[task_id]
    
    def __call__(self, task_id):
        return self.get_env(task_id)


class PusherEnv(PusherMujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file=None,
        distractors=False,
        task_id='',
        state_mean=None,
        state_std=None
    ):
        utils.EzPickle.__init__(self)
        # print('PusherEnv')
        # print(xml_file)
        if xml_file is None:
            assert False, 'No!'
            xml_file = 'pusher.xml'
        self.include_distractors = distractors
        self.task_id = task_id
        self.state_mean = state_mean
        self.state_std = state_std
        PusherMujocoEnv.__init__(self, xml_file, 5)

        self._get_viewer('rgb_array')
        # self.viewer.autoscale()

        assert state_mean is not None
        assert state_std is not None

        # print(self.observation_space)
        # print(self.action_space)
        # print(self.state_mean.shape)
        # print(self.reset_model().shape)

    @property
    def task_identifier(self):
        return self.task_id

    def step(self, a):
        # normalize actions
        if self.action_space is not None:
            # print(a)
            lb, ub = self.action_space.low, self.action_space.high
            a = lb + (a + 1.) * 0.5 * (ub - lb)
            a = np.clip(a, lb, ub)
            # print(a)
            # 1/0

        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        reward_near = - np.linalg.norm(vec_1)
        reward_dist = - np.linalg.norm(vec_2)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        self.do_simulation(a, self.frame_skip)
        # extra added to copy rllab forward_dynamics.
        # print(dir(self.sim))
        # print(help(self.sim.step))
        # self.model.forward()

        ob = self._get_obs()
        done = False

        env_info = dict(
            reward_dist=reward_dist,
            reward_ctrl=reward_ctrl,
            distractor_pos=self.get_body_com("distractor").copy(),
            object_pos=self.get_body_com("object").copy(),
            goal_pos=self.get_body_com("goal").copy(),
            obj2arm_dist=-reward_near,
            obj2goal_dist=-reward_dist
        )

        return ob, reward, done, env_info

    def eval_success(self, obj_pos, goal_pos):
        obj_pos = obj_pos[:,:2]
        goal_pos = goal_pos[:,:2]
        dists = np.sum((goal_pos - obj_pos)**2, 1) # distances at each timestep
        num_close = np.sum(dists < 0.017)
        return num_close >= 10, num_close

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

        # new viewpoint
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.lookat[0] = 0.4 # more positive moves the dot left
        self.viewer.cam.lookat[1] = -0.1 # more positive moves the dot down
        self.viewer.cam.lookat[2] = 0.0
        self.viewer.cam.distance = 0.75
        self.viewer.cam.elevation = -50
        self.viewer.cam.azimuth = -90

    def reset_model(self):
        qpos = self.init_qpos

        self.goal_pos = np.asarray([0, 0])
        while True:
            self.obj_pos = np.concatenate([
                    np.random.uniform(low=-0.3, high=0, size=1),
                    np.random.uniform(low=-0.2, high=0.2, size=1)])
            if np.linalg.norm(self.obj_pos - self.goal_pos) > 0.17:
                break

        if self.include_distractors:
            if self.obj_pos[1] < 0:
                y_range = [0.0, 0.2]
            else:
                y_range = [-0.2, 0.0]
            while True:
                self.distractor_pos = np.concatenate([
                        np.random.uniform(low=-0.3, high=0, size=1),
                        np.random.uniform(low=y_range[0], high=y_range[1], size=1)])
                if np.linalg.norm(self.distractor_pos - self.goal_pos) > 0.17 and np.linalg.norm(self.obj_pos - self.distractor_pos) > 0.1:
                    break
            qpos[-6:-4] = self.distractor_pos


        qpos[-4:-2] = self.obj_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + np.random.uniform(low=-0.005,
                high=0.005, size=(self.model.nv))

        #qvel[-4:] = 0
        #self.set_state(qpos, qvel)
        #return self._get_obs()

        # setattr(self.model.data, 'qpos', qpos)
        # setattr(self.model.data, 'qvel', qvel)
        # self.model.data.qvel = qvel
        # self.model._compute_subtree()
        # self.model.forward()
        # self.current_com = self.model.data.com_subtree[0]
        # self.dcom = np.zeros_like(self.current_com)

        # print('Resetting state')
        # print(qpos)
        # print(qvel)
        # print(self.obj_pos)
        # print(self.distractor_pos)
        # print(self.goal_pos)
        self.set_state(qpos, qvel)

        # # self.set_state(self.init_qpos, self.init_qvel)
        # d = self._get_obs()
        # image = d['image']
        # X = d['X']
        # from scipy.misc import imsave
        # imsave('plots/junk_vis/fuck_you.png', image.transpose((1,2,0)))
        # 1/0
        # return image, X

        # when you reset for some reason there's a bug that the first time you call
        # this the rendered image does not reflect the correct position for things
        self._get_obs()
        return self._get_obs()

    def get_current_image_obs(self):
        # print(self._get_viewer)
        # print(self._get_viewer())
        # self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
        # self.viewer = mujoco_py.MjViewer()
        # self.viewer.start()
        # self.viewer.set_model(self.model)
        # self.viewer_setup()
        # self._get_viewer()

        
        # image = self.viewer.get_image()
        # pil_image = Image.frombytes('RGB', (image[1], image[2]), image[0])
        # pil_image = pil_image.resize((125,125), Image.ANTIALIAS)
        # image = np.flipud(np.array(pil_image))

        # image = self.render(mode='rgb_array', width=125, height=125)
        # transpose to make it have correct ordering of dimensions for pytorch
        # image = image.transpose((2,0,1))
        # image = np.array(image).astype(np.float32)
        # image /= 255.0

        image = self.render(mode='rgb_array', width=500, height=500)
        image = Image.fromarray(image)
        image = image.resize((125,125), PIL.Image.LANCZOS)
        image = np.array(image)
        image = image.transpose((2,0,1))
        image = np.array(image).astype(np.float32)
        image /= 255.0

        X = np.concatenate([
            self.sim.data.qpos.flat[:7],
            self.sim.data.qvel.flat[:7],
            self.get_body_com('tips_arm'),
            self.get_body_com('goal'),
        ]).copy()
        X = (X - self.state_mean) / self.state_std

        return image, X

    def _get_obs(self):
        # WRONG
        # if self.include_distractors:
        #     X = np.concatenate([
        #         self.sim.data.qpos.flat[:7],
        #         self.sim.data.qvel.flat[:7],
        #         self.get_body_com("tips_arm"),
        #         self.get_body_com("distractor"),
        #         self.get_body_com("object"),
        #         self.get_body_com("goal"),
        #     ])
        # else:
        #     X = np.concatenate([
        #         self.model.data.qpos.flat[:7],
        #         self.model.data.qvel.flat[:7],
        #         self.get_body_com("tips_arm"),
        #         self.get_body_com("object"),
        #         self.get_body_com("goal"),
        #     ])

        # CORRECT
        # X = np.concatenate([
        #     self.sim.data.qpos.flat[:7],
        #     self.sim.data.qvel.flat[:7],
        #     self.get_body_com('tips_arm'),
        #     self.get_body_com('goal'),
        # ])
        # X = (X - self.state_mean) / self.state_std
        # return X

        image, X = self.get_current_image_obs()
        return {'image': image, 'X':X}
    

    def log_statistics(self, paths):
        obj_pos = [np.array([d["object_pos"] for d in path["env_infos"]]) for path in paths]
        dist_pos = [np.array([d["distractor_pos"] for d in path["env_infos"]]) for path in paths]
        goal_pos = [np.array([d["goal_pos"] for d in path["env_infos"]]) for path in paths]
        min_obj2arm_dist = [np.min([d["obj2arm_dist"] for d in path["env_infos"]]) for path in paths]
        min_obj2goal_dist = [np.min([d["obj2goal_dist"] for d in path["env_infos"]]) for path in paths]

        correct_obj_successes = []
        correct_obj_num_close = []
        for op, gp in zip(obj_pos, goal_pos):
            is_success, num_close = self.eval_success(op, gp)
            correct_obj_successes.append(is_success)
            correct_obj_num_close.append(num_close)
        
        incorrect_obj_successes = []
        incorrect_obj_num_close = []
        for dp, gp in zip(dist_pos, goal_pos):
            is_success, num_close = self.eval_success(dp, gp)
            incorrect_obj_successes.append(is_success)
            incorrect_obj_num_close.append(num_close)
        
        return_dict = OrderedDict()
        return_dict['Correct Obj Success Rate'] = np.mean(correct_obj_successes)
        return_dict['Incorrect Obj Success Rate'] = np.mean(incorrect_obj_successes)

        return_dict['Correct Obj Num Steps Close Mean'] = np.mean(correct_obj_num_close)
        return_dict['Correct Obj Num Steps Close Std'] = np.std(correct_obj_num_close)
        return_dict['Correct Obj Num Steps Close Max'] = np.max(correct_obj_num_close)
        return_dict['Correct Obj Num Steps Close Min'] = np.min(correct_obj_num_close)

        return_dict['Incorrect Obj Num Steps Close Mean'] = np.mean(incorrect_obj_num_close)
        return_dict['Incorrect Obj Num Steps Close Std'] = np.std(incorrect_obj_num_close)
        return_dict['Incorrect Obj Num Steps Close Max'] = np.max(incorrect_obj_num_close)
        return_dict['Incorrect Obj Num Steps Close Min'] = np.min(incorrect_obj_num_close)

        return_dict['Closest Obj2Arm Mean'] = np.mean(min_obj2arm_dist)
        return_dict['Closest Obj2Arm Std'] = np.std(min_obj2arm_dist)
        return_dict['Closest Obj2Arm Max'] = np.max(min_obj2arm_dist)
        return_dict['Closest Obj2Arm Min'] = np.min(min_obj2arm_dist)

        return_dict['Closest Obj2Goal Mean'] = np.mean(min_obj2goal_dist)
        return_dict['Closest Obj2Goal Std'] = np.std(min_obj2goal_dist)
        return_dict['Closest Obj2Goal Max'] = np.max(min_obj2goal_dist)
        return_dict['Closest Obj2Goal Min'] = np.min(min_obj2goal_dist)

        print(len(correct_obj_successes))
        print(len(incorrect_obj_successes))

        return return_dict


def build_pusher_getter(task_ids, distractors=True, mode='train', state_mean=None, state_std=None):
    assert mode in ['train', 'val', 'test']
    
    all_xmls_dict = {}
    for task_id in task_ids:
        if mode == 'test':
            demo_info = pickle.load(open(osp.join(TEST_DEMO_DIR, str(task_id)+'.pkl'), 'rb'))
        else:
            demo_info = pickle.load(open(osp.join(TRAIN_VAL_DEMO_DIR, 'demos_'+str(task_id)+'.pkl'), 'rb'))

        xml_filepath = demo_info['xml']
        # print('----> {}'.format(xml_filepath))
        # suffix = xml_filepath[xml_filepath.index('pusher'):]
        # prefix = XML_PATH + 'test2_ensure_woodtable_distractor_'
        # xml_filepath = str(prefix + suffix)

        file_name = osp.split(xml_filepath)[1]
        xml_filepath = osp.join(XML_PATH, file_name)

        all_xmls_dict[mode+'_'+str(task_id)] = xml_filepath
    
    env_getter = PusherEnvGetter(all_xmls_dict, distractors=distractors, state_mean=state_mean, state_std=state_std)
    return env_getter


if __name__ == '__main__':
    # train/val task ids are 0-768 ---------------
    # task_ids = list(range(769))
    import pickle
    from scipy.misc import imsave
    with open('/h/kamyar/mil/data/scale_and_bias_sim_push.pkl', 'rb') as f:
        d = pickle.load(f)
    task_ids = list(range(4))
    eg = build_pusher_getter(task_ids, distractors=True, mode='train', state_mean=d['mean'], state_std=d['std'])
    for i in range(4):
        env = eg.get_env('train_%d'%i)
        # print(env)
        # print(env.reset())
        print(env.task_identifier)
    
    env = eg.get_env('train_3')
    env.reset()

    print(env.action_space.high, env.action_space.low)

    # env._viewers = {}
    # viewer = env._get_viewer('human')
    # data, width, height = viewer.get_image()
    
    image = env.render(mode='rgb_array', width=125, height=125)
    imsave('plots/junk_vis/low_quality.png', image)

    image = env.render(mode='rgb_array', width=250, height=250)
    image = Image.fromarray(image)
    image = image.resize((125,125), PIL.Image.LANCZOS)
    image = np.array(image)
    imsave('plots/junk_vis/high_quality.png', image)

    image = env.render(mode='rgb_array', width=500, height=500)
    image = Image.fromarray(image)
    image = image.resize((125,125), PIL.Image.LANCZOS)
    image = np.array(image)
    imsave('plots/junk_vis/higher_quality.png', image)


    image, rest = env.get_current_image_obs()
    imsave('test_render.png', image.transpose(1,2,0))
    print(np.mean(image))
    for i in range(5):
        env.step(env.action_space.sample())
    image, rest = env.get_current_image_obs()
    imsave('test_render_0.png', image.transpose(1,2,0))



    # get the test task ids ----------------------
    # test_task_ids = []
    # for p in os.listdir(TEST_DEMO_DIR):
    #     if '.pkl' in p:
    #         test_task_ids.append(int(p.split('.')[0]))
    # test_task_ids = sorted(test_task_ids)
    # # print(len(test_task_ids))
    # build_pusher_getter(test_task_ids, distractors=True, mode='test')
