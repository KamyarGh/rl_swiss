"""
Modified from
https://github.com/tianheyu927/mil/blob/master/data_generator.py
Code for loading data and generating data batches during training
"""
from __future__ import division

import copy
import logging
import os
import glob
import tempfile
import pickle
from datetime import datetime
from collections import OrderedDict

import numpy as np
import random
from mil_utils import extract_demo_dict, Timer
from natsort import natsorted
from random import shuffle


class DataGenerator(object):
    def __init__(self, config={}):
        # Hyperparameters
        self.config = config
        self.update_batch_size = self.config['update_batch_size']
        self.test_batch_size = self.config['train_update_batch_size'] if self.config['train_update_batch_size'] != -1 else self.update_batch_size
        self.meta_batch_size = self.config['meta_batch_size']
        self.T = self.config['T']
        self.demo_gif_dir = self.config['demo_gif_dir']
        self.gif_prefix = self.config['gif_prefix']
        self.restore_iter = self.config['restore_iter']
        # Scale and bias for data normalization
        self.scale, self.bias = None, None

        demo_file = self.config['demo_file']
        demo_file = natsorted(glob.glob(demo_file + '/*pkl'))
        self.dataset_size = len(demo_file)
        if self.config['train'] and self.config['training_set_size'] != -1:
            tmp = demo_file[:self.config['training_set_size']]
            tmp.extend(demo_file[-self.config['val_set_size']:])
            demo_file = tmp
        self.extract_supervised_data(demo_file)
        if self.config['use_noisy_demos']:
            self.noisy_demo_gif_dir = self.config['noisy_demo_gif_dir']
            noisy_demo_file = self.config['noisy_demo_file']
            self.extract_supervised_data(noisy_demo_file, noisy=True)

    def extract_supervised_data(self, demo_file, noisy=False):
        """
            Load the states and actions of the demos into memory.
            Args:
                demo_file: list of demo files where each file contains expert's states and actions of one task.
        """
        demos = extract_demo_dict(demo_file)
        # We don't need the whole dataset of simulated pushing.
        if self.config['experiment'] == 'sim_push':
            for key in demos.keys():
                demos[key]['demoX'] = demos[key]['demoX'][6:-6, :, :].copy()
                demos[key]['demoU'] = demos[key]['demoU'][6:-6, :, :].copy()
        n_folders = len(demos.keys())
        N_demos = np.sum(demo['demoX'].shape[0] for i, demo in demos.items())
        self.state_idx = range(demos[0]['demoX'].shape[-1])
        self._dU = demos[0]['demoU'].shape[-1]
        print("Number of demos: %d" % N_demos)
        idx = np.arange(n_folders)
        if self.config['train']:
            n_val = self.config['val_set_size'] # number of demos for testing
            if not hasattr(self, 'train_idx'):
                if n_val != 0:
                    if not self.config['shuffle_val']:
                        self.val_idx = idx[-n_val:]
                        self.train_idx = idx[:-n_val]
                    else:
                        self.val_idx = np.sort(np.random.choice(idx, size=n_val, replace=False))
                        mask = np.array([(i in self.val_idx) for i in idx])
                        self.train_idx = np.sort(idx[~mask])
                else:
                    self.train_idx = idx
                    self.val_idx = []
            # Normalize the states if it's training.
            with Timer('Normalizing states'):
                if self.scale is None or self.bias is None:
                    states = np.vstack((demos[i]['demoX'] for i in self.train_idx)) # hardcoded here to solve the memory issue
                    states = states.reshape(-1, len(self.state_idx))
                    # 1e-3 to avoid infs if some state dimensions don't change in the
                    # first batch of samples
                    self.scale = np.diag(
                        1.0 / np.maximum(np.std(states, axis=0), 1e-3))
                    self.bias = - np.mean(
                        states.dot(self.scale), axis=0)
                    # Save the scale and bias.
                    with open('/h/kamyar/mil/data/scale_and_bias_%s.pkl' % self.config['experiment'], 'wb') as f:
                        pickle.dump({'scale': self.scale, 'bias': self.bias}, f)
                for key in demos.keys():
                    demos[key]['demoX'] = demos[key]['demoX'].reshape(-1, len(self.state_idx))
                    demos[key]['demoX'] = demos[key]['demoX'].dot(self.scale) + self.bias
                    demos[key]['demoX'] = demos[key]['demoX'].reshape(-1, self.T, len(self.state_idx))
        if not noisy:
            self.demos = demos
        else:
            self.noisy_demos = demos

    def generate_batches(self, noisy=False):
        with Timer('Generating batches for each iteration'):
            if self.config['training_set_size'] != -1:
                offset = self.dataset_size - self.config['training_set_size'] - self.config['val_set_size']
            else:
                offset = 0
            img_folders = natsorted(glob.glob(self.demo_gif_dir + self.gif_prefix + '_*'))
            train_img_folders = {i: img_folders[i] for i in self.train_idx}
            val_img_folders = {i: img_folders[i+offset] for i in self.val_idx}
            if noisy:
                noisy_img_folders = natsorted(glob.glob(self.noisy_demo_gif_dir + self.gif_prefix + '_*'))
                noisy_train_img_folders = {i: noisy_img_folders[i] for i in self.train_idx}
                noisy_val_img_folders = {i: noisy_img_folders[i] for i in self.val_idx}
            TEST_PRINT_INTERVAL = 500
            TOTAL_ITERS = self.config['metatrain_iterations']
            self.all_training_filenames = []
            self.all_val_filenames = []
            self.training_batch_idx = {i: OrderedDict() for i in range(TOTAL_ITERS)}
            self.val_batch_idx = {i: OrderedDict() for i in TEST_PRINT_INTERVAL*np.arange(1, int(TOTAL_ITERS/TEST_PRINT_INTERVAL))}
            if noisy:
                self.noisy_training_batch_idx = {i: OrderedDict() for i in range(TOTAL_ITERS)}
                self.noisy_val_batch_idx = {i: OrderedDict() for i in TEST_PRINT_INTERVAL*np.arange(1, TOTAL_ITERS/TEST_PRINT_INTERVAL)}
            for itr in range(TOTAL_ITERS):
                # print(self.train_idx)
                # print(self.meta_batch_size)
                # print(type(self.train_idx))
                # sampled_train_idx = random.sample(self.train_idx, self.meta_batch_size)
                sampled_train_idx = np.random.choice(self.train_idx, size=self.meta_batch_size, replace=False)
                print('itr {} of {}'.format(itr, TOTAL_ITERS))
                for idx in sampled_train_idx:
                    sampled_folder = train_img_folders[idx]
                    image_paths = natsorted(os.listdir(sampled_folder))
                    if self.config['experiment'] == 'sim_push':
                        image_paths = image_paths[6:-6]
                    try:
                        assert len(image_paths) == self.demos[idx]['demoX'].shape[0]
                    except AssertionError:
                        import pdb; pdb.set_trace()
                    if noisy:
                        noisy_sampled_folder = noisy_train_img_folders[idx]
                        noisy_image_paths = natsorted(os.listdir(noisy_sampled_folder))
                        assert len(noisy_image_paths) == self.noisy_demos[idx]['demoX'].shape[0]
                    if not noisy:
                        sampled_image_idx = np.random.choice(range(len(image_paths)), size=self.update_batch_size+self.test_batch_size, replace=False) # True
                        sampled_images = [os.path.join(sampled_folder, image_paths[i]) for i in sampled_image_idx]
                    else:
                        noisy_sampled_image_idx = np.random.choice(range(len(noisy_image_paths)), size=self.update_batch_size, replace=False) #True
                        sampled_image_idx = np.random.choice(range(len(image_paths)), size=self.test_batch_size, replace=False) #True
                        sampled_images = [os.path.join(noisy_sampled_folder, noisy_image_paths[i]) for i in noisy_sampled_image_idx]
                        sampled_images.extend([os.path.join(sampled_folder, image_paths[i]) for i in sampled_image_idx])
                    self.all_training_filenames.extend(sampled_images)
                    self.training_batch_idx[itr][idx] = sampled_image_idx
                    if noisy:
                        self.noisy_training_batch_idx[itr][idx] = noisy_sampled_image_idx
                if itr != 0 and itr % TEST_PRINT_INTERVAL == 0:
                    # sampled_val_idx = random.sample(self.val_idx, self.meta_batch_size)
                    sampled_val_idx = np.random.choice(self.val_idx, size=self.meta_batch_size, replace=False)
                    for idx in sampled_val_idx:
                        sampled_folder = val_img_folders[idx]
                        image_paths = natsorted(os.listdir(sampled_folder))
                        if self.config['experiment'] == 'sim_push':
                            image_paths = image_paths[6:-6]
                        assert len(image_paths) == self.demos[idx]['demoX'].shape[0]
                        if noisy:
                            noisy_sampled_folder = noisy_val_img_folders[idx]
                            noisy_image_paths = natsorted(os.listdir(noisy_sampled_folder))
                            assert len(noisy_image_paths) == self.noisy_demos[idx]['demoX'].shape[0]
                        if not noisy:
                            sampled_image_idx = np.random.choice(range(len(image_paths)), size=self.update_batch_size+self.test_batch_size, replace=False) # True
                            sampled_images = [os.path.join(sampled_folder, image_paths[i]) for i in sampled_image_idx]
                        else:
                            noisy_sampled_image_idx = np.random.choice(range(len(noisy_image_paths)), size=self.update_batch_size, replace=False) # True
                            sampled_image_idx = np.random.choice(range(len(image_paths)), size=self.test_batch_size, replace=False) # True
                            sampled_images = [os.path.join(noisy_sampled_folder, noisy_image_paths[i]) for i in noisy_sampled_image_idx]
                            sampled_images.extend([os.path.join(sampled_folder, image_paths[i]) for i in sampled_image_idx])
                        self.all_val_filenames.extend(sampled_images)
                        self.val_batch_idx[itr][idx] = sampled_image_idx
                        if noisy:
                            self.noisy_val_batch_idx[itr][idx] = noisy_sampled_image_idx

    def make_batch_tensor(self, network_config, restore_iter=0, train=True):
        TEST_INTERVAL = 500
        batch_image_size = (self.update_batch_size + self.test_batch_size) * self.meta_batch_size
        if train:
            all_filenames = self.all_training_filenames
            if restore_iter > 0:
                all_filenames = all_filenames[batch_image_size*(restore_iter+1):]
        else:
            all_filenames = self.all_val_filenames
            if restore_iter > 0:
                all_filenames = all_filenames[batch_image_size*(int(restore_iter/TEST_INTERVAL)+1):]
        
        im_height = network_self.config['image_height']
        im_width = network_self.config['image_width']
        num_channels = network_self.config['image_channels']
        # make queue for tensorflow to read from
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        print('Generating image processing ops')
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        image = tf.image.decode_gif(image_file)
        # should be T x C x W x H
        image.set_shape((self.T, im_height, im_width, num_channels))
        image = tf.cast(image, tf.float32)
        image /= 255.0
        if self.config['hsv']:
            eps_min, eps_max = 0.5, 1.5
            assert eps_max >= eps_min >= 0
            # convert to HSV only fine if input images in [0, 1]
            img_hsv = tf.image.rgb_to_hsv(image)
            img_h = img_hsv[..., 0]
            img_s = img_hsv[..., 1]
            img_v = img_hsv[..., 2]
            eps = tf.random_uniform([self.T, 1, 1], eps_min, eps_max)
            img_v = tf.clip_by_value(eps * img_v, 0., 1.)
            img_hsv = tf.stack([img_h, img_s, img_v], 3)
            image_rgb = tf.image.hsv_to_rgb(img_hsv)
            image = image_rgb
        image = tf.transpose(image, perm=[0, 3, 2, 1]) # transpose to mujoco setting for images
        image = tf.reshape(image, [self.T, -1])
        num_preprocess_threads = 1 # TODO - enable this to be set to >1
        min_queue_examples = 64 #128 #256
        print('Batching images')
        images = tf.train.batch(
                [image],
                batch_size = batch_image_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_image_size,
                )
        all_images = []
        for i in range(self.meta_batch_size):
            image = images[i*(self.update_batch_size+self.test_batch_size):(i+1)*(self.update_batch_size+self.test_batch_size)]
            image = tf.reshape(image, [(self.update_batch_size+self.test_batch_size)*self.T, -1])
            all_images.append(image)
        return tf.stack(all_images)
        
    def generate_data_batch(self, itr, train=True):
        if train:
            demos = {key: self.demos[key].copy() for key in self.train_idx}
            idxes = self.training_batch_idx[itr]
            if self.config['use_noisy_demos']:
                noisy_demos = {key: self.noisy_demos[key].copy() for key in self.train_idx}
                noisy_idxes = self.noisy_training_batch_idx[itr]
        else:
            demos = {key: self.demos[key].copy() for key in self.val_idx}
            idxes = self.val_batch_idx[itr]
            if self.config['use_noisy_demos']:
                noisy_demos = {key: self.noisy_demos[key].copy() for key in self.val_idx}
                noisy_idxes = self.noisy_val_batch_idx[itr]
        batch_size = self.meta_batch_size
        update_batch_size = self.update_batch_size
        test_batch_size = self.test_batch_size
        if not self.config['use_noisy_demos']:
            U = [demos[k]['demoU'][v].reshape((test_batch_size+update_batch_size)*self.T, -1) for k, v in idxes.items()]
            U = np.array(U)
            X = [demos[k]['demoX'][v].reshape((test_batch_size+update_batch_size)*self.T, -1) for k, v in idxes.items()]
            X = np.array(X)
        else:
            noisy_U = [noisy_demos[k]['demoU'][v].reshape(update_batch_size*self.T, -1) for k, v in noisy_idxes.items()]
            noisy_X = [noisy_demos[k]['demoX'][v].reshape(update_batch_size*self.T, -1) for k, v in noisy_idxes.items()]
            U = [demos[k]['demoU'][v].reshape(test_batch_size*self.T, -1) for k, v in idxes.items()]
            U = np.concatenate((np.array(noisy_U), np.array(U)), axis=1)
            X = [demos[k]['demoX'][v].reshape(test_batch_size*self.T, -1) for k, v in idxes.items()]
            X = np.concatenate((np.array(noisy_X), np.array(X)), axis=1)
        assert U.shape[2] == self._dU
        assert X.shape[2] == len(self.state_idx)
        return X, U


if __name__ == '__main__':
    # specs used for the video + state + action experiments
    specs = {
        'update_batch_size': 1, # number of examples used for inner loop gradient
        'train_update_batch_size': -1,
        'meta_batch_size': 15,
        'T': 100,
        'demo_gif_dir': '/h/kamyar/mil/data/sim_push/',
        'gif_prefix': 'object',
        'restore_iter': 0,
        'demo_file': '/h/kamyar/mil/data/sim_push',
        'train': True,
        'training_set_size': -1, # -1 means all data except validation set
        'val_set_size': 76,
        'use_noisy_demos': False,
        'noisy_demo_gif_dir': None,
        'noisy_demo_file': None,
        'experiment': 'sim_push',
        'shuffle_val': False,
        'metatrain_iterations': 30000,
        'hsv': False
    }

    dg = DataGenerator(config=specs)
    dg.generate_batches()
    dg.generate_data_batch(0)
