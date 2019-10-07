'''
pusher_mil_data_generatory.py from original work is too slow
This is my implementation using pytorch data loaders class
'''
import os
import os.path as osp
import pickle

import numpy as np
import imageio

import torch
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted
import glob

from rlkit.data_management.mil_utils import extract_demo_dict, Timer


def pusher_collate_fn(batch):
    new_dict = {}
    for k in batch[0]:
        new_dict[k] = np.array(
            [b[k] for b in batch]
        )
    return new_dict


class PusherDataset(Dataset):
    def __init__(
        self,
        demos,
        state_dim,
        act_dim,
        vid_folders,
        num_context_trajs,
        num_test_trajs,
        prefix,
        mode='video_and_state_and_action', # or 'video_and_state' or 'video_only',
    ):
        super().__init__()
        if mode != 'video_and_state_and_action': raise NotImplementedError()

        self.demos = demos
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.num_context_trajs = num_context_trajs
        self.num_test_trajs = num_test_trajs
        self.total_num_trajs_per_task = self.num_context_trajs + self.num_test_trajs
        self.prefix = prefix

        vid_paths_dict = {}
        for task in vid_folders:
            task_folder = vid_folders[task]
            task_vid_paths = natsorted(
                [
                    p for p in os.listdir(task_folder)
                    if '.npy' in p
                    # if '.gif' in p
                ]
            )
            # for 'sim_push' they aren't using all the demos
            task_vid_paths = task_vid_paths[6:-6]
            task_vid_paths = [
                osp.join(task_folder, p) for p in task_vid_paths
            ]
            try:
                assert len(task_vid_paths) == self.demos[task]['demoX'].shape[0]
            except AssertionError:
                import pdb; pdb.set_trace()
            vid_paths_dict[task] = task_vid_paths
        self.vid_paths_dict = vid_paths_dict
        self._size = len(list(vid_paths_dict.keys()))
    

    def __len__(self):
        return self._size

    
    def __getitem__(self, task):
        '''
        We use idx as meaning the task idx
        '''
        task = self.prefix + '_' + str(task)
        traj_idxs = np.random.choice(
            self.demos[task]['demoU'].shape[0],
            size=self.total_num_trajs_per_task,
            replace=False
        )

        # get the states
        U = [self.demos[task]['demoU'][v] for v in traj_idxs]
        U = np.array(U)
        X = [self.demos[task]['demoX'][v] for v in traj_idxs]
        X = np.array(X)
        assert U.shape[2] == self.act_dim
        assert X.shape[2] == self.state_dim

        # get the videos
        vids = []
        for idx in traj_idxs:
            # vid = imageio.mimread(self.vid_paths_dict[task][idx])

            # no need for transposing since I've already saved them
            # with correct ordering of dimensions
            vid = np.load(self.vid_paths_dict[task][idx])

            # we will do this on the GPU
            # .astype(np.float32)
            # vid /= 255.0
            vids.append(vid)
        # vids = np.array(vids)

        return {
            'videos': vids,
            'states': X,
            'actions': U
        }
    

    def get_single_traj(self, task):
        '''
        We use idx as meaning the task idx
        '''
        sample_idx = np.random.choice(self.demos[task]['demoU'].shape[0])

        # get the states
        U = [self.demos[task]['demoU'][sample_idx]]
        U = np.array(U)
        X = [self.demos[task]['demoX'][sample_idx]]
        X = np.array(X)
        assert U.shape[2] == self.act_dim
        assert X.shape[2] == self.state_dim

        # get the videos
        vids = [np.load(self.vid_paths_dict[task][sample_idx])]

        return {
            'videos': vids,
            'states': X,
            'actions': U
        }


def extract_supervised_data(
    demo_file,
    demo_gif_dir,
    gif_prefix,
    training_set_size,
    val_set_size,
    shuffle_val,
    T,
):
    """
        Taken and modified from the original work's data loader.
        This is fast so it's ok to use it.

        Load the states and actions of the demos into memory.
        Args:
            demo_file: list of demo files where each file contains expert's states and actions of one task.
    """
    demo_file = natsorted(glob.glob(demo_file + '/*pkl'))
    dataset_size = len(demo_file)
    if training_set_size != -1:
        tmp = demo_file[:training_set_size]
        tmp.extend(demo_file[-val_set_size:])
        demo_file = tmp

    demos = extract_demo_dict(demo_file)
    # print(demos.keys())
    # We don't need the whole dataset of simulated pushing.
    for key in demos.keys():
        demos[key]['demoX'] = demos[key]['demoX'][6:-6, :, :].copy()
        # original
        demos[key]['demoU'] = demos[key]['demoU'][6:-6, :, :].copy()
        # the env's weird normalization and clipping effectively clips
        # the range of actions to 1 and -1
        # demos[key]['demoU'] = np.clip(demos[key]['demoU'][6:-6, :, :].copy(), -1.0, 1.0)

    n_folders = len(demos.keys())
    N_demos = np.sum(demo['demoX'].shape[0] for i, demo in demos.items())
    state_idx = range(demos[0]['demoX'].shape[-1])
    _dU = demos[0]['demoU'].shape[-1]
    print("Number of demos: %d" % N_demos)
    idx = np.arange(n_folders)
    
    n_val = val_set_size # number of demos for testing
    if n_val != 0:
        if not shuffle_val:
            val_idx = idx[-n_val:]
            train_idx = idx[:-n_val]
        else:
            val_idx = np.sort(np.random.choice(idx, size=n_val, replace=False))
            mask = np.array([(i in val_idx) for i in idx])
            train_idx = np.sort(idx[~mask])
    else:
        assert False
        train_idx = idx
        val_idx = []
    # Normalize the states if it's training.
    with Timer('Normalizing states'):
        states = np.vstack((demos[i]['demoX'] for i in train_idx)) # hardcoded here to solve the memory issue
        states = states.reshape(-1, len(state_idx))

        # actions = np.vstack((demos[i]['demoU'] for i in train_idx)) # hardcoded here to solve the memory issue
        # actions = actions.reshape(-1, _dU)
        # print(np.mean(actions, axis=0))
        # print(np.std(actions, axis=0))
        # print(np.max(actions, axis=0))
        # print(np.min(actions, axis=0))
        # from rlkit.core.vistools import plot_histogram
        # for i in range(7):
        #     plot_histogram(actions[:,i], 100, 'pusher_action_dim_%d'%i, 'plots/junk_vis/pusher_action_dim_%d.png'%i)
        # 1/0

        # 1e-3 to avoid infs if some state dimensions don't change in the
        # first batch of samples
        # original --------
        # scale = np.diag(
        #     1.0 / np.maximum(np.std(states, axis=0), 1e-3))
        # bias = - np.mean(
        #     states.dot(scale), axis=0)
        # mine ------------
        # print(np.std(states, axis=0))
        # print(np.mean(states, axis=0))
        std = np.maximum(np.std(states, axis=0, keepdims=True), 1e-3)
        mean = np.mean(states, axis=0, keepdims=True)
        # print(np.max((states - mean) / std, axis=0))
        # print(np.min((states - mean) / std, axis=0))
        # 1/0
        # print(np.mean((states-mean)/std, axis=0))
        # print(np.std((states-mean)/std, axis=0))
        # print(mean)
        # print(std)
        # Save the scale and bias.
        with open('/h/kamyar/mil/data/scale_and_bias_%s.pkl' % 'sim_push', 'wb') as f:
            pickle.dump({'std': std, 'mean': mean}, f)
        
        for key in demos.keys():
            # original --------
            # demos[key]['demoX'] = demos[key]['demoX'].reshape(-1, len(state_idx))
            # demos[key]['demoX'] = demos[key]['demoX'].dot(scale) + bias
            # demos[key]['demoX'] = demos[key]['demoX'].reshape(-1, T, len(state_idx))
            # mine ------------
            # print(demos[key]['demoX'].shape)
            # print(len(state_idx))
            # prev = demos[key]['demoX'][0][:10]
            # print(prev)
            demos[key]['demoX'] = demos[key]['demoX'].reshape(-1, len(state_idx))
            # print(demos[key]['demoX'][:10])
            # print(demos[key]['demoX'].shape)
            demos[key]['demoX'] = (demos[key]['demoX'] - mean) / std
            # print(demos[key]['demoX'][:10])
            # print(demos[key]['demoX'].shape)
            demos[key]['demoX'] = demos[key]['demoX'].reshape(-1, T, len(state_idx))
            # print(demos[key]['demoX'].shape)
            # post_prev = demos[key]['demoX'][0]*std + mean
            # print(post_prev.shape)
            # print(post_prev == prev)
            # print(demos[key]['demoX'][0][:10])
            # print(prev.shape)

    train_demos = {'train_%d' % i: demos[i] for i in train_idx}
    val_demos = {'val_%d' % i: demos[i] for i in val_idx}

    if training_set_size != -1:
        offset = dataset_size - training_set_size - val_set_size
    else:
        offset = 0
    img_folders = natsorted(glob.glob(demo_gif_dir + gif_prefix + '_*'))
    train_img_folders = {'train_%d'%i: img_folders[i] for i in train_idx}
    val_img_folders = {'val_%d' % i: img_folders[i+offset] for i in val_idx}

    return train_demos, val_demos, train_img_folders, val_img_folders, train_idx, val_idx, state_idx, _dU, mean, std


def build_train_val_datasets(num_context_trajs, num_test_trajs):
    train_demos, val_demos, train_img_folders, val_img_folders, train_idx, val_idx, state_idx, _dU, mean, std = extract_supervised_data(
        '/scratch/ssd001/home/kamyar/mil/data/sim_push',
        '/scratch/ssd001/home/kamyar/mil/data/sim_push/',
        'object',
        -1,
        76,
        False,
        100,
    )

    train_ds = PusherDataset(
        train_demos,
        len(state_idx),
        _dU,
        train_img_folders,
        num_context_trajs,
        num_test_trajs,
        'train'
    )
    val_ds = PusherDataset(
        val_demos,
        len(state_idx),
        _dU,
        val_img_folders,
        num_context_trajs,
        num_test_trajs,
        'val'
    )
    
    return train_ds, val_ds, train_idx, val_idx, len(state_idx), _dU, mean, std



if __name__ == '__main__':
    train_demos, val_demos, train_img_folders, val_img_folders, train_idx, val_idx,state_idx, _dU = extract_supervised_data(
        '/scratch/ssd001/home/kamyar/mil/data/sim_push',
        '/scratch/ssd001/home/kamyar/mil/data/sim_push/',
        'object',
        -1,
        76,
        False,
        100,
    )
    # k = list(train_img_folders.keys())
    # print(k)
    # print(train_img_folders[k[0]])

    ds = PusherDataset(
        train_demos,
        len(state_idx),
        _dU,
        train_img_folders,
        1,
        4
    )

    print(train_idx)
    print(val_idx)

    # for k,v in ds[4].items():
    #     print(k, v.shape)

    # for i in range(len(ds)):
    #     ds[i]
    #     print(i)

    # dl = DataLoader(
    #     ds,
    #     batch_size=8,
    #     shuffle=True,
    #     num_workers=8,
    #     collate_fn=pusher_collate_fn,
    #     drop_last=True
    # )
    # for i, batch in enumerate(dl):
    #     print(i)

    # for i in range(100):
    #     print('now')
    #     tasks = np.random.choice(100, size=8)
    #     for task in tasks:
    #         ds[task]
